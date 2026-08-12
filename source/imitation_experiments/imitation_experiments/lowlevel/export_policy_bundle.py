"""Export a trained RLOpt tracker checkpoint into an Embodied-Control policy bundle.

The bundle is the only interface between this training repository and the
`embodied_control.lowlevel` runtime (see wiki/embodied-control-tracker-runtime.md).
It carries TorchScript modules rebuilt from raw checkpoint tensors, the ordered
observation contract, the action contract, and provenance hashes. Nothing in the
bundle imports RLOpt, torchrl, or Isaac Lab.

Run from the repository root in the isolated ONNX export Pixi environment:

    pixi run -e onnx-export python -m imitation_experiments.lowlevel.export_policy_bundle \
        --checkpoint logs/downloaded_checkpoints/bones129k_l2t_1b/model_step_1000341504.pt \
        --preset l2t_student_v2 --output logs/policy_bundles/l2t_student_1b

The observation contracts here are presets pinned to the task-layout snapshot
(`source/isaaclab_imitation/tests/g1_task_layout_default.json`) and are
cross-validated against the checkpoint's own shapes and normalize mask. The
env-config-derived exporter mode (running under the isaaclab env) is future
work; a preset/shape mismatch fails loudly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch import nn

from imitation_experiments.paths import REPO_ROOT

BUNDLE_API_VERSION = "ec.bundle/v1"
ONNX_OPSET = 18

# ---------------------------------------------------------------------------
# G1 constants (transcribed from isaaclab_imitation; sources noted per block).
# ---------------------------------------------------------------------------

# config/g1/common/constants.py:36-66 — breadth-first USD/PhysX articulation order.
G1_ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# assets/robots/unitree.py:220-229 (UNITREE_G1_29DOF_MIMIC_CFG.init_state).
_DEFAULT_JOINT_POS_PATTERNS = {
    "hip_pitch": -0.312,
    "knee": 0.669,
    "ankle_pitch": -0.363,
    "elbow": 0.6,
    "left_shoulder_roll": 0.2,
    "left_shoulder_pitch": 0.2,
    "right_shoulder_roll": -0.2,
    "right_shoulder_pitch": 0.2,
}

# assets/robots/unitree.py:199-215 + 235-356 + the SONIC hip-pitch override
# (:379-391). Values are the formulas, not rounded decimals.
_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
_SOFT_JOINT_LIMIT_FACTOR = 0.9


def _actuator(model: str, multiplier: float = 1.0) -> tuple[float, float]:
    armature = _ARMATURE[model] * multiplier
    stiffness = armature * _NATURAL_FREQ**2
    damping = 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
    return stiffness, damping


# joint substring -> (actuator model, multiplier, effort_limit), SONIC contract.
_SONIC_ACTUATOR_TABLE = [
    ("hip_pitch", "7520_22", 1.0, 139.0),
    ("hip_roll", "7520_22", 1.0, 139.0),
    ("hip_yaw", "7520_14", 1.0, 88.0),
    ("knee", "7520_22", 1.0, 139.0),
    ("waist_yaw", "7520_14", 1.0, 88.0),
    ("waist_roll", "5020", 2.0, 50.0),
    ("waist_pitch", "5020", 2.0, 50.0),
    ("ankle_pitch", "5020", 2.0, 50.0),
    ("ankle_roll", "5020", 2.0, 50.0),
    ("shoulder_pitch", "5020", 1.0, 25.0),
    ("shoulder_roll", "5020", 1.0, 25.0),
    ("shoulder_yaw", "5020", 1.0, 25.0),
    ("elbow", "5020", 1.0, 25.0),
    ("wrist_roll", "5020", 1.0, 25.0),
    ("wrist_pitch", "4010", 1.0, 5.0),
    ("wrist_yaw", "4010", 1.0, 5.0),
]


def _per_joint_actuation() -> dict[str, list[float]]:
    table: dict[str, list[float]] = {
        "default_joint_pos": [],
        "action_scale": [],
        "stiffness": [],
        "damping": [],
        "armature": [],
        "effort_limit": [],
    }
    for name in G1_ISAAC_JOINT_NAMES:
        default = 0.0
        for pattern, value in _DEFAULT_JOINT_POS_PATTERNS.items():
            if pattern in name:
                default = value
                break
        matches = [row for row in _SONIC_ACTUATOR_TABLE if row[0] in name]
        if len(matches) != 1:
            raise RuntimeError(f"actuator table match failed for {name}: {matches}")
        _, model, multiplier, effort = matches[0]
        stiffness, damping = _actuator(model, multiplier)
        table["default_joint_pos"].append(default)
        table["action_scale"].append(0.25 * effort / stiffness)
        table["stiffness"].append(stiffness)
        table["damping"].append(damping)
        table["armature"].append(_ARMATURE[model] * multiplier)
        table["effort_limit"].append(effort)
    return table


def _sdk_joint_names() -> list[str]:
    """Load the SDK order from the self-validating stdlib-only module."""
    module_path = (
        REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets/robots/"
        "unitree_joint_order.py"
    )
    spec = importlib.util.spec_from_file_location("_unitree_joint_order", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.UNITREE_G1_29DOF_SDK_JOINT_NAMES)


def _soft_joint_limits() -> tuple[list[float], list[float]]:
    """Read the G1 MJCF limits and apply the Isaac articulation soft factor."""
    model_path = (
        REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets/unitree/"
        "g1_description/g1_29dof_rev_1_0.xml"
    )
    root = ET.parse(model_path).getroot()
    hard_limits: dict[str, tuple[float, float]] = {}
    for element in root.iter("joint"):
        name = element.get("name")
        raw_range = element.get("range")
        if name is None or raw_range is None:
            continue
        values = [float(value) for value in raw_range.split()]
        if len(values) == 2:
            hard_limits[name] = (values[0], values[1])
    missing = [name for name in G1_ISAAC_JOINT_NAMES if name not in hard_limits]
    if missing:
        raise RuntimeError(f"G1 MJCF misses joint limits for: {missing}")
    lower: list[float] = []
    upper: list[float] = []
    for name in G1_ISAAC_JOINT_NAMES:
        hard_lower, hard_upper = hard_limits[name]
        midpoint = 0.5 * (hard_lower + hard_upper)
        half_range = 0.5 * (hard_upper - hard_lower) * _SOFT_JOINT_LIMIT_FACTOR
        lower.append(midpoint - half_range)
        upper.append(midpoint + half_range)
    return lower, upper


# ---------------------------------------------------------------------------
# Observation-contract presets (pinned against g1_task_layout_default.json).
# ---------------------------------------------------------------------------

_PROPRIO_TERMS = [
    ("projected_gravity", 3, True),
    ("base_ang_vel", 3, True),
    ("joint_pos_rel", 29, True),
    ("joint_vel_rel", 29, True),
    ("last_action", 29, True),
]


def _observation_term(
    name: str,
    width: int,
    normalize: bool,
    *,
    history_length: int = 1,
    history_stride: int = 1,
    history_order: str = "oldest_first",
    reset_fill: str = "repeat_first",
) -> dict:
    """Return one complete runtime observation term.

    History fields are explicit even when history is disabled. This keeps the
    model input contract visible in the bundle instead of relying on runtime
    defaults.
    """
    return {
        "name": name,
        "width": width,
        "normalize": normalize,
        "history_length": history_length,
        "history_stride": history_stride,
        "history_order": history_order,
        "reset_fill": reset_fill,
    }


@dataclasses.dataclass(frozen=True)
class Preset:
    name: str
    interface: str
    terms: list[tuple[str, int, bool]]
    z_dim: int | None = None
    phase_mode: str = "none"
    default_hold_steps: int = 1
    encoder_state_dim: int | None = None
    horizon_steps: int | None = None
    quantizer: str = "none"

    @property
    def total_width(self) -> int:
        return sum(width for _, width, _ in self.terms)

    @property
    def phase_dim(self) -> int:
        return 2 if self.phase_mode == "sin_cos" else 0


PRESETS = {
    "l2t_student_v2": Preset(
        name="l2t_student_v2",
        interface="latent",
        terms=[("latent_command", 258, False), *_PROPRIO_TERMS],
        z_dim=256,
        phase_mode="sin_cos",
        default_hold_steps=25,
        encoder_state_dim=38,
        horizon_steps=10,
    ),
    "latent_v2": Preset(
        name="latent_v2",
        interface="latent",
        terms=[("latent_command", 258, False), *_PROPRIO_TERMS],
        z_dim=256,
        phase_mode="sin_cos",
        default_hold_steps=25,
        encoder_state_dim=38,
        horizon_steps=10,
    ),
    # SONIC-style FSQ arm: the command is the quantized lattice vector (64 dims,
    # multiples of 1/16 in [-1, 1]) plus the sin/cos phase. A planner trained
    # against this bundle regresses the PRE-quantized bounded vector, SONIC
    # convention; the runtime snaps commands onto the lattice at consume time.
    "fsq64_v2": Preset(
        name="fsq64_v2",
        interface="latent",
        terms=[("latent_command", 66, False), *_PROPRIO_TERMS],
        z_dim=64,
        phase_mode="sin_cos",
        default_hold_steps=10,
        encoder_state_dim=38,
        horizon_steps=10,
        quantizer="fsq",
    ),
    "explicit_v2": Preset(
        name="explicit_v2",
        interface="explicit",
        terms=[
            ("expert_motion", 58, True),
            ("expert_anchor_pos_b", 3, True),
            ("expert_anchor_ori_b", 6, True),
            *_PROPRIO_TERMS,
        ],
    ),
    "vanilla_legacy": Preset(
        name="vanilla_legacy",
        interface="explicit",
        terms=[
            ("expert_motion", 58, True),
            ("expert_anchor_pos_b", 3, True),
            ("expert_anchor_ori_b", 6, True),
            *_PROPRIO_TERMS[1:],
        ],
    ),
}


# ---------------------------------------------------------------------------
# Standalone modules rebuilt from raw checkpoint tensors.
# ---------------------------------------------------------------------------

_ACTIVATIONS = {"silu": nn.SiLU, "elu": nn.ELU, "mish": nn.Mish}


def _mlp_from_flat_state(state: dict, prefix: str, activation: str) -> nn.Sequential:
    indices = sorted(
        int(key[len(prefix) :].split(".")[0])
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )
    layers: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight = state[f"{prefix}{index}.weight"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        layers.append(linear)
        if position < len(indices) - 1:
            layers.append(_ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


class StandaloneTracker(nn.Module):
    """Running-stat normalizer + MLP; deterministic action = raw output.

    Reproduces `RunningMeanStdCatInputs` (RLOpt ppo.py:90-180) in eval mode:
    `clamp((x - mean) / sqrt(var + eps), -clip, clip)`, with mask-False dims
    passed through raw, then the policy MLP. `IndependentNormal` makes the
    deterministic action the MLP output itself — no tanh, no rescale.
    """

    def __init__(
        self,
        mlp: nn.Sequential,
        mean: torch.Tensor,
        var: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1.0e-5,
        clip: float = 5.0,
    ):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("mean", mean.float())
        self.register_buffer("var", var.float())
        self.register_buffer("mask", mask.bool())
        self.epsilon = float(epsilon)
        self.clip = float(clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.mean) / torch.sqrt(self.var + self.epsilon)
        normalized = torch.clamp(normalized, -self.clip, self.clip)
        return self.mlp(torch.where(self.mask, normalized, obs))


class StandaloneSkillEncoder(nn.Module):
    """Deterministic DiffSR trunk: (Linear, LayerNorm?, Mish)* -> Linear.

    Rebuilt from `skill_encoder_state_dict` (`net.{i}.*` keys), taking the flat
    `[state ; window]` vector (hl_skill_encoder.py:240-289).
    """

    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        return self.net(flat)


class StandaloneFSQEncoder(nn.Module):
    """SONIC-FSQ trunk + lattice: `round(bound(net(flat))) / (L//2)`.

    Reproduces `SONICFSQSkillEncoder` at inference
    (hl_skill_encoder.py:35-83, 619-682): `bound(z) = tanh(z + shift) * half
    - offset` with `half = (L-1)/2`, `offset = 0.5` for even L,
    `shift = atanh(offset / half)`; the command is the rounded value divided
    by `L // 2`, so every entry is an exact lattice multiple in [-1, 1].
    """

    def __init__(
        self, net: nn.Sequential, levels: list[int], half_levels: torch.Tensor
    ):
        super().__init__()
        self.net = net
        levels_t = torch.tensor([int(level) for level in levels], dtype=torch.float32)
        bound_half = (levels_t - 1.0) * 0.5
        offset = (levels_t.long() % 2 == 0).float() * 0.5
        shift = torch.atanh(offset / bound_half.clamp(min=1.0))
        self.register_buffer("bound_half", bound_half)
        self.register_buffer("offset", offset)
        self.register_buffer("shift", shift)
        self.register_buffer("half_levels", half_levels.float())

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        z = self.net(flat)
        bounded = torch.tanh(z + self.shift) * self.bound_half - self.offset
        return torch.round(bounded) / self.half_levels


def _encoder_trunk_from_state(state: dict, activation: str) -> nn.Sequential:
    indices = sorted(
        {int(key.split(".")[1]) for key in state if key.startswith("net.")}
    )
    layers: list[nn.Module] = []
    renamed: dict[str, torch.Tensor] = {}
    previous = None
    for index in indices:
        if previous is not None and index > previous + 1:
            layers.append(_ACTIVATIONS[activation]())
        weight = state[f"net.{index}.weight"]
        layer: nn.Module
        if weight.ndim == 2:
            layer = nn.Linear(weight.shape[1], weight.shape[0])
        else:
            layer = nn.LayerNorm(weight.shape[0])
        renamed[f"{len(layers)}.weight"] = weight
        renamed[f"{len(layers)}.bias"] = state[f"net.{index}.bias"]
        layers.append(layer)
        previous = index
    net = nn.Sequential(*layers)
    net.load_state_dict(renamed, strict=True)
    return net


def _encoder_from_state(
    state: dict, activation: str = "mish", fsq_levels: list[int] | None = None
) -> nn.Module:
    trunk = _encoder_trunk_from_state(state, activation)
    if "_half_levels" in state:
        if fsq_levels is None:
            raise ValueError(
                "encoder state carries _half_levels (SONIC-FSQ) but the FSQ "
                "levels are unknown; pass the skill checkpoint so its config's "
                "sonic_fsq_levels can be read"
            )
        half = state["_half_levels"].float()
        expected = torch.tensor(
            [max(int(level) // 2, 1) for level in fsq_levels], dtype=torch.float32
        )
        if half.shape != expected.shape or not torch.equal(half, expected):
            raise ValueError(
                "_half_levels does not match the config's sonic_fsq_levels"
            )
        return StandaloneFSQEncoder(trunk, fsq_levels, half)
    if fsq_levels is not None:
        raise ValueError("fsq_levels given but the encoder state has no _half_levels")
    return StandaloneSkillEncoder(trunk)


# ---------------------------------------------------------------------------
# Checkpoint parsing and gates.
# ---------------------------------------------------------------------------

_POLICY_PREFIX = "module.0.module."


def _strip_policy_state(policy_state: dict) -> dict:
    stripped = {}
    for key, value in policy_state.items():
        if key.startswith(_POLICY_PREFIX):
            stripped[key[len(_POLICY_PREFIX) :]] = value
    if not stripped:
        raise ValueError(
            f"policy state dict has no {_POLICY_PREFIX!r} keys; got {sorted(policy_state)[:5]}"
        )
    return stripped


def _build_tracker(stripped: dict, activation: str) -> StandaloneTracker:
    mean = stripped["base.running_mean"]
    var = stripped["base.running_var"]
    mask = stripped.get("base.normalize_mask")
    if mask is None:
        mask = torch.ones_like(mean, dtype=torch.bool)
    mlp = _mlp_from_flat_state(stripped, "base.module.", activation)
    mlp_state = {
        key[len("base.module.") :]: value
        for key, value in stripped.items()
        if key.startswith("base.module.")
    }
    mlp.load_state_dict(mlp_state, strict=True)
    tracker = StandaloneTracker(mlp, mean, var, mask)
    tracker.eval()
    for parameter in tracker.parameters():
        parameter.requires_grad_(False)
    return tracker


def _rlopt_native_parity(
    stripped: dict, tracker: StandaloneTracker, rows: int = 512, atol: float = 1e-6
) -> float:
    """Rebuild the actor with RLOpt's own modules and compare loc outputs."""
    from rlopt.agent.ppo.ppo import RunningMeanStdCatInputs
    from rlopt.models.gaussian_policy import GaussianPolicyHead
    from torchrl.modules import MLP

    width = stripped["base.running_mean"].shape[0]
    action_dim = tracker.mlp[-1].out_features
    hidden = [m.out_features for m in tracker.mlp if isinstance(m, nn.Linear)][:-1]
    activation_cls = type(next(m for m in tracker.mlp if not isinstance(m, nn.Linear)))
    base = RunningMeanStdCatInputs(
        MLP(
            in_features=width,
            out_features=action_dim,
            num_cells=hidden,
            activation_class=activation_cls,
        ),
        feature_dim=width,
        normalize_mask=stripped.get("base.normalize_mask"),
    )
    head = GaussianPolicyHead(base=base, action_dim=action_dim)
    head.load_state_dict(stripped, strict=True)
    head.eval()
    generator = torch.Generator().manual_seed(1234)
    obs = torch.randn(rows, width, generator=generator)
    with torch.inference_mode():
        loc, _scale = head(obs)
        ours = tracker(obs)
    worst = float((loc - ours).abs().max())
    if worst > atol:
        raise ValueError(f"RLOpt-native parity failed: max abs err {worst} > {atol}")
    return worst


def _tensors_identical(left: dict, right: dict) -> None:
    if sorted(left) != sorted(right):
        raise ValueError(
            f"encoder key sets differ: {sorted(set(left) ^ set(right))[:6]}"
        )
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError(f"encoder tensor mismatch at {key}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Export.
# ---------------------------------------------------------------------------


def _export_onnx(
    module: nn.Module,
    sample: torch.Tensor,
    path: Path,
    *,
    input_name: str,
    output_name: str,
) -> None:
    """Write one static batch-one ONNX graph and validate its structure."""
    import onnx  # noqa: PLC0415

    torch.onnx.export(
        module,
        sample,
        path,
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=None,
        dynamo=False,
    )
    model = onnx.load(path)
    onnx.checker.check_model(model)


def _onnx_replay(
    path: Path,
    values: np.ndarray,
    *,
    input_name: str,
    output_name: str,
) -> np.ndarray:
    """Replay static batch-one rows through the CPU execution provider."""
    import onnxruntime as ort  # noqa: PLC0415

    session = ort.InferenceSession(
        str(path),
        sess_options=_ort_session_options(),
        providers=["CPUExecutionProvider"],
    )
    return np.concatenate(
        [
            session.run(
                [output_name],
                {input_name: np.ascontiguousarray(row[None], dtype=np.float32)},
            )[0]
            for row in values
        ],
        axis=0,
    )


def _ort_session_options():
    import onnxruntime as ort  # noqa: PLC0415

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def export_bundle(args: argparse.Namespace) -> Path:
    preset = PRESETS[args.preset]
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"policy bundle output already exists: {output}. "
            "Use a fresh path so a failed export cannot mix artifacts."
        )
    if args.trace_rows < 1:
        raise ValueError("trace_rows must be positive")
    if not math.isfinite(args.onnx_atol) or args.onnx_atol <= 0.0:
        raise ValueError("onnx_atol must be finite and positive")
    if args.hold_steps is not None and args.hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    if args.macro_frame_stride is not None and args.macro_frame_stride < 1:
        raise ValueError("macro_frame_stride must be positive")
    if args.macro_anchor_mode not in (None, "robot", "expert_heading"):
        raise ValueError("macro_anchor_mode must be 'robot' or 'expert_heading'")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    metadata = checkpoint.get("checkpoint_metadata") or {}
    role = args.role
    if role == "student":
        policy_state = checkpoint["policy_state_dict"]
        if metadata and metadata.get("primary_policy_role") not in (None, "student"):
            raise ValueError(f"checkpoint primary role is {metadata!r}, not student")
    else:
        if not args.allow_privileged:
            raise SystemExit(
                "teacher export refused: the teacher is a training ceiling, not a "
                "deployable policy. Pass --allow-privileged to export for diagnostics."
            )
        policy_state = checkpoint["teacher_policy_state_dict"]
    if "vec_norm_msg" in checkpoint:
        raise ValueError(
            "checkpoint contains vec_norm_msg; env-side normalization is unsupported"
        )

    stripped = {k: v.clone() for k, v in _strip_policy_state(policy_state).items()}
    tracker = _build_tracker(stripped, args.activation)

    in_features = tracker.mean.shape[0]
    interface = preset.interface if role == "student" else "privileged-teacher"
    if role == "student" and in_features != preset.total_width:
        raise ValueError(
            f"checkpoint expects {in_features} inputs; preset {preset.name} is "
            f"{preset.total_width}. Wrong preset or wrong checkpoint."
        )
    action_dim = tracker.mlp[-1].out_features
    if action_dim != 29:
        raise ValueError(f"action dim {action_dim} != 29")

    mask = tracker.mask
    cursor = 0
    for name, width, normalize in preset.terms:
        if role != "student":
            break
        span = mask[cursor : cursor + width]
        uniform = bool(span.all()) or bool((~span).all())
        if not uniform or normalize != bool(span.all()):
            raise ValueError(
                f"normalize mask disagrees with preset at term {name}: "
                f"preset={normalize}, span all={bool(span.all())} any={bool(span.any())}"
            )
        cursor += width

    encoder = None
    encoder_provenance: dict = {}
    skill_checkpoint_path: Path | None = None
    if preset.interface == "latent" and role == "student":
        if preset.z_dim is None:
            raise ValueError(f"latent preset {preset.name} must define z_dim")
        latent_term_width = preset.terms[0][1]
        if latent_term_width != preset.z_dim + preset.phase_dim:
            raise ValueError(
                f"preset latent term width {latent_term_width} != z_dim "
                f"{preset.z_dim} + phase_dim {preset.phase_dim}"
            )
        sampler_state = checkpoint.get("hl_skill_command_sampler_state_dict")
        if not sampler_state or "skill_encoder_state_dict" not in sampler_state:
            raise ValueError("latent preset needs hl_skill_command_sampler_state_dict")
        embedded = sampler_state["skill_encoder_state_dict"]
        if preset.quantizer == "fsq" and not args.skill_checkpoint:
            raise ValueError(
                "FSQ presets require --skill-checkpoint: the lattice levels and "
                "bound constants live only in the skill checkpoint's config"
            )
        fsq_levels = None
        if args.skill_checkpoint:
            skill_checkpoint_path = Path(args.skill_checkpoint).expanduser().resolve()
            if not skill_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"skill checkpoint not found: {skill_checkpoint_path}"
                )
            skill = torch.load(
                skill_checkpoint_path, map_location="cpu", weights_only=False
            )
            _tensors_identical(embedded, skill["skill_encoder_state_dict"])
            config = skill.get("config", {})
            # Config values are authoritative; CLI flags only fill fields a
            # pre-field skill checkpoint does not carry.
            encoder_provenance = {
                "macro_frame_stride": config.get(
                    "macro_frame_stride", args.macro_frame_stride
                ),
                "macro_anchor_mode": config.get(
                    "macro_anchor_mode", args.macro_anchor_mode
                ),
                "horizon_steps": config.get("horizon_steps"),
                "encoder_window_mode": config.get("encoder_window_mode"),
                "encoder_activation": config.get(
                    "encoder_activation", args.encoder_activation
                ),
                "encoder_layer_norm": config.get(
                    "encoder_layer_norm", args.encoder_layer_norm
                ),
            }
            if preset.quantizer == "fsq":
                fsq_levels = config.get("sonic_fsq_levels")
                if not fsq_levels:
                    raise ValueError("skill config has no sonic_fsq_levels")
                if len(fsq_levels) != preset.z_dim:
                    raise ValueError(
                        f"sonic_fsq_levels length {len(fsq_levels)} != z_dim "
                        f"{preset.z_dim}"
                    )
        else:
            encoder_provenance = {
                "macro_frame_stride": args.macro_frame_stride,
                "macro_anchor_mode": args.macro_anchor_mode,
                "horizon_steps": preset.horizon_steps,
                "encoder_window_mode": "intermediate",
                "encoder_activation": args.encoder_activation,
                "encoder_layer_norm": args.encoder_layer_norm,
            }
        for field_name in (
            "horizon_steps",
            "encoder_window_mode",
            "macro_frame_stride",
            "macro_anchor_mode",
            "encoder_activation",
            "encoder_layer_norm",
        ):
            if encoder_provenance.get(field_name) is None:
                raise ValueError(
                    f"encoder provenance field {field_name!r} is unknown; pass "
                    f"--{field_name.replace('_', '-')} or --skill-checkpoint. A "
                    "mismatch is width-invisible, so it must be recorded."
                )
        horizon_steps = int(encoder_provenance["horizon_steps"])
        if horizon_steps < 1:
            raise ValueError("encoder horizon_steps must be positive")
        window_mode = str(encoder_provenance["encoder_window_mode"])
        if window_mode not in {"full", "intermediate"}:
            raise ValueError("encoder_window_mode must be 'full' or 'intermediate'")
        if window_mode == "intermediate" and horizon_steps <= 1:
            raise ValueError(
                "encoder_window_mode='intermediate' requires horizon_steps > 1"
            )
        if int(encoder_provenance["macro_frame_stride"]) < 1:
            raise ValueError("macro_frame_stride must be positive")
        if encoder_provenance["macro_anchor_mode"] not in {
            "robot",
            "expert_heading",
        }:
            raise ValueError("macro_anchor_mode must be 'robot' or 'expert_heading'")
        encoder = _encoder_from_state(
            embedded,
            activation=encoder_provenance["encoder_activation"],
            fsq_levels=fsq_levels,
        )
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        actual_layer_norm = any(
            isinstance(layer, nn.LayerNorm) for layer in encoder.net
        )
        if actual_layer_norm != bool(encoder_provenance["encoder_layer_norm"]):
            raise ValueError(
                "encoder_layer_norm provenance disagrees with the encoder tensors"
            )
        encoder_in = encoder.net[0].in_features
        window_steps = horizon_steps - (1 if window_mode == "intermediate" else 0)
        if encoder_in != preset.encoder_state_dim * (window_steps + 1):
            raise ValueError(
                f"encoder input {encoder_in} != state_dim {preset.encoder_state_dim} "
                f"x (window_steps {window_steps} + 1)"
            )
        z_out = encoder.net[-1].out_features
        if z_out != preset.z_dim:
            raise ValueError(f"encoder z dim {z_out} != preset z_dim {preset.z_dim}")

    if not args.skip_rlopt_parity and role == "student":
        worst = _rlopt_native_parity(stripped, tracker)
        print(f"RLOpt-native parity: max abs err {worst:.3e} over 512 obs")

    output.mkdir(parents=True, exist_ok=False)

    scripted = torch.jit.script(tracker)
    generator = torch.Generator().manual_seed(args.trace_seed)
    obs = torch.randn(args.trace_rows, in_features, generator=generator)
    # The runtime infers batch-1, and fp32 matmul reduction order differs by
    # batch size, so the recorded trace must be produced batch-1 as well.
    with torch.inference_mode():
        eager_actions = torch.cat([tracker(row.unsqueeze(0)) for row in obs])
        scripted_actions = torch.cat([scripted(row.unsqueeze(0)) for row in obs])
    worst = float((eager_actions - scripted_actions).abs().max())
    if worst > 1e-6:
        raise ValueError(f"TorchScript parity failed: {worst}")
    scripted.save(str(output / "policy.pt"))
    _export_onnx(
        tracker,
        obs[0].unsqueeze(0),
        output / "policy.onnx",
        input_name="obs",
        output_name="action",
    )
    onnx_actions = _onnx_replay(
        output / "policy.onnx",
        obs.numpy(),
        input_name="obs",
        output_name="action",
    )
    onnx_worst = float(
        np.abs(onnx_actions - eager_actions.numpy().astype(np.float32)).max()
    )
    if onnx_worst > args.onnx_atol:
        raise ValueError(f"policy ONNX parity failed: {onnx_worst} > {args.onnx_atol}")
    trace_arrays = {
        "obs": obs.numpy().astype(np.float32),
        "action": eager_actions.numpy().astype(np.float32),
    }

    if encoder is not None:
        scripted_encoder = torch.jit.script(encoder)
        encoder_obs = torch.randn(64, encoder.net[0].in_features, generator=generator)
        with torch.inference_mode():
            eager_z = torch.cat([encoder(row.unsqueeze(0)) for row in encoder_obs])
            scripted_z = torch.cat(
                [scripted_encoder(row.unsqueeze(0)) for row in encoder_obs]
            )
        worst = float((eager_z - scripted_z).abs().max())
        if worst > 1e-6:
            raise ValueError(f"encoder TorchScript parity failed: {worst}")
        scripted_encoder.save(str(output / "encoder.pt"))
        _export_onnx(
            encoder,
            encoder_obs[0].unsqueeze(0),
            output / "encoder.onnx",
            input_name="macro_window",
            output_name="latent",
        )
        onnx_z = _onnx_replay(
            output / "encoder.onnx",
            encoder_obs.numpy(),
            input_name="macro_window",
            output_name="latent",
        )
        encoder_onnx_worst = float(
            np.abs(onnx_z - eager_z.numpy().astype(np.float32)).max()
        )
        if encoder_onnx_worst > args.onnx_atol:
            raise ValueError(
                f"encoder ONNX parity failed: {encoder_onnx_worst} > {args.onnx_atol}"
            )
        trace_arrays["encoder_in"] = encoder_obs.numpy().astype(np.float32)
        trace_arrays["encoder_out"] = eager_z.numpy().astype(np.float32)

    np.savez(output / "golden_trace.npz", **trace_arrays)
    np.savez(
        output / "norm_stats.npz",
        running_mean=tracker.mean.numpy(),
        running_var=tracker.var.numpy(),
        normalize_mask=tracker.mask.numpy(),
    )

    actuation = _per_joint_actuation()
    joint_limits_lower, joint_limits_upper = _soft_joint_limits()
    sdk_names = _sdk_joint_names()
    isaac_to_sdk = [sdk_names.index(name) for name in G1_ISAAC_JOINT_NAMES]
    obs_contract = {
        "terms": [
            _observation_term(name, width, normalize)
            for name, width, normalize in preset.terms
        ],
        "total_width": preset.total_width if role == "student" else in_features,
    }
    if role != "student":
        obs_contract["terms"] = [
            _observation_term("privileged_input", in_features, True)
        ]
    action_contract = {
        "width": 29,
        "isaac_joint_names": G1_ISAAC_JOINT_NAMES,
        "sdk_joint_names": sdk_names,
        "isaac_to_sdk": isaac_to_sdk,
        "default_joint_pos": actuation["default_joint_pos"],
        "default_joint_vel": [0.0] * 29,
        "action_scale": actuation["action_scale"],
        "stiffness": actuation["stiffness"],
        "damping": actuation["damping"],
        "armature": actuation["armature"],
        "effort_limit": actuation["effort_limit"],
        "joint_limits_lower": joint_limits_lower,
        "joint_limits_upper": joint_limits_upper,
        "last_action_is_raw": True,
    }
    command_contract: dict = {}
    if preset.interface == "latent" and role == "student":
        command_contract = {
            "z_dim": preset.z_dim,
            "phase_mode": preset.phase_mode,
            "phase_dim": 2 if preset.phase_mode == "sin_cos" else 0,
            "hold_steps": args.hold_steps or preset.default_hold_steps,
            "state_dim": preset.encoder_state_dim,
            "encoder_state_interface": {
                38: "root_qpos",
                67: "full_body",
            }.get(preset.encoder_state_dim),
            "window_steps": window_steps,
            "horizon_steps": encoder_provenance["horizon_steps"],
            "encoder_window_mode": encoder_provenance["encoder_window_mode"],
            "macro_frame_stride": encoder_provenance["macro_frame_stride"],
            "macro_anchor_mode": encoder_provenance["macro_anchor_mode"],
            "activation": encoder_provenance["encoder_activation"],
            "layer_norm": bool(encoder_provenance.get("encoder_layer_norm")),
            "quantizer": preset.quantizer,
        }
        if preset.quantizer == "fsq":
            if not isinstance(encoder, StandaloneFSQEncoder):
                raise ValueError("fsq preset produced a non-FSQ encoder")
            command_contract["fsq_half_levels"] = [
                float(v) for v in encoder.half_levels.tolist()
            ]
    elif preset.interface == "explicit":
        command_contract = {
            "components": [
                {"name": name, "width": width, "normalize": normalize}
                for name, width, normalize in preset.terms
                if name.startswith("expert_")
            ]
        }

    (output / "obs_contract.json").write_text(json.dumps(obs_contract, indent=2))
    (output / "action_contract.json").write_text(json.dumps(action_contract, indent=2))

    files = {
        name: sha256_file(output / name)
        for name in [
            "policy.pt",
            "policy.onnx",
            "obs_contract.json",
            "action_contract.json",
            "golden_trace.npz",
            "norm_stats.npz",
        ]
        + (["encoder.pt", "encoder.onnx"] if encoder is not None else [])
    }
    import onnx  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    manifest = {
        "api_version": BUNDLE_API_VERSION,
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_metadata": metadata,
            "primary_policy_role": role,
            "preset": preset.name,
            "activation": args.activation,
            "repo_commit": _git_commit(),
            "export_command": args.export_command,
            "tool_versions": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": ort.__version__,
            },
            "parity": "torchscript-eager"
            "+onnxruntime-cpu" + ("" if args.skip_rlopt_parity else "+rlopt-native"),
        },
        "interface": interface,
        "obs": obs_contract,
        "action": action_contract,
        "command": command_contract,
        "rates": {"control_hz": 50, "physics_dt": 0.005, "decimation": 4},
        "models": {
            "policy_onnx": {
                "format": "onnx",
                "path": "policy.onnx",
                "input_name": "obs",
                "output_name": "action",
                "input_shape": [1, in_features],
                "output_shape": [1, action_dim],
                "opset": ONNX_OPSET,
                "parity_atol": args.onnx_atol,
                "max_abs_error": onnx_worst,
            },
            "policy_torchscript": {
                "format": "torchscript",
                "path": "policy.pt",
                "input_name": "obs",
                "output_name": "action",
                "input_shape": [1, in_features],
                "output_shape": [1, action_dim],
                "parity_atol": 1e-6,
                "max_abs_error": float((eager_actions - scripted_actions).abs().max()),
            },
        },
        "files": files,
    }
    if skill_checkpoint_path is not None:
        manifest["source"].update(
            {
                "skill_checkpoint_path": str(skill_checkpoint_path),
                "skill_checkpoint_sha256": sha256_file(skill_checkpoint_path),
            }
        )
    if encoder is not None:
        manifest["command"]["encoder_sha256"] = files["encoder.pt"]
        manifest["models"].update(
            {
                "encoder_onnx": {
                    "format": "onnx",
                    "path": "encoder.onnx",
                    "input_name": "macro_window",
                    "output_name": "latent",
                    "input_shape": [1, encoder.net[0].in_features],
                    "output_shape": [1, preset.z_dim],
                    "opset": ONNX_OPSET,
                    "parity_atol": args.onnx_atol,
                    "max_abs_error": encoder_onnx_worst,
                },
                "encoder_torchscript": {
                    "format": "torchscript",
                    "path": "encoder.pt",
                    "input_name": "macro_window",
                    "output_name": "latent",
                    "input_shape": [1, encoder.net[0].in_features],
                    "output_shape": [1, preset.z_dim],
                    "parity_atol": 1e-6,
                    "max_abs_error": float((eager_z - scripted_z).abs().max()),
                },
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output


def verify_bundle_dir(root: Path, atol: float = 1e-5) -> dict:
    """RLOpt-free replay of the golden traces (numpy + torch only)."""
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = sha256_file(root / name)
        if actual != expected:
            raise ValueError(f"hash mismatch for {name}")
    trace = np.load(root / "golden_trace.npz")
    policy = torch.jit.load(str(root / "policy.pt"), map_location="cpu").eval()
    with torch.inference_mode():
        actions = torch.cat(
            [policy(torch.from_numpy(row).unsqueeze(0)) for row in trace["obs"]]
        ).numpy()
    worst = float(np.abs(actions - trace["action"]).max())
    if worst > atol:
        raise ValueError(f"policy golden trace mismatch: {worst}")
    report = {"policy_max_abs_err": worst, "rows": int(trace["obs"].shape[0])}
    onnx_actions = _onnx_replay(
        root / "policy.onnx",
        trace["obs"],
        input_name="obs",
        output_name="action",
    )
    onnx_worst = float(np.abs(onnx_actions - trace["action"]).max())
    policy_onnx_atol = float(
        manifest.get("models", {}).get("policy_onnx", {}).get("parity_atol", atol)
    )
    if onnx_worst > policy_onnx_atol:
        raise ValueError(f"policy ONNX golden trace mismatch: {onnx_worst}")
    report["policy_onnx_max_abs_err"] = onnx_worst
    if "encoder_in" in trace:
        encoder = torch.jit.load(str(root / "encoder.pt"), map_location="cpu").eval()
        with torch.inference_mode():
            z = torch.cat(
                [
                    encoder(torch.from_numpy(row).unsqueeze(0))
                    for row in trace["encoder_in"]
                ]
            ).numpy()
        enc_worst = float(np.abs(z - trace["encoder_out"]).max())
        if enc_worst > atol:
            raise ValueError(f"encoder golden trace mismatch: {enc_worst}")
        report["encoder_max_abs_err"] = enc_worst
        onnx_z = _onnx_replay(
            root / "encoder.onnx",
            trace["encoder_in"],
            input_name="macro_window",
            output_name="latent",
        )
        encoder_onnx_worst = float(np.abs(onnx_z - trace["encoder_out"]).max())
        encoder_onnx_atol = float(
            manifest.get("models", {}).get("encoder_onnx", {}).get("parity_atol", atol)
        )
        if encoder_onnx_worst > encoder_onnx_atol:
            raise ValueError(
                f"encoder ONNX golden trace mismatch: {encoder_onnx_worst}"
            )
        report["encoder_onnx_max_abs_err"] = encoder_onnx_worst
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=["student", "teacher"], default="student")
    parser.add_argument("--allow-privileged", action="store_true")
    parser.add_argument("--activation", choices=sorted(_ACTIVATIONS), default="silu")
    parser.add_argument(
        "--skill-checkpoint",
        default=None,
        help="skill checkpoint for the binding gate + encoder provenance",
    )
    parser.add_argument("--macro-frame-stride", type=int, default=None)
    parser.add_argument("--macro-anchor-mode", default=None)
    parser.add_argument(
        "--encoder-activation",
        choices=sorted(_ACTIVATIONS),
        default=None,
        help="encoder trunk activation when no --skill-checkpoint supplies it",
    )
    parser.add_argument("--encoder-layer-norm", action="store_true")
    parser.add_argument("--hold-steps", type=int, default=None)
    parser.add_argument("--trace-rows", type=int, default=512)
    parser.add_argument("--trace-seed", type=int, default=0)
    parser.add_argument("--onnx-atol", type=float, default=2e-5)
    parser.add_argument("--skip-rlopt-parity", action="store_true")
    parser.add_argument(
        "--verify", action="store_true", help="replay the golden traces after writing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(effective_argv)
    args.export_command = shlex.join([str(Path(__file__).resolve()), *effective_argv])
    output = export_bundle(args)
    print(f"bundle written: {output}")
    if args.verify:
        report = verify_bundle_dir(output)
        print(f"verify: {json.dumps(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
