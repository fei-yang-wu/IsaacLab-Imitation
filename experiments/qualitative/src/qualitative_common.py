#!/usr/bin/env python3
"""Shared pieces for the 129k skill-encoder qualitative analyses.

Deliberately Isaac-free: nothing here imports ``isaaclab`` or Omniverse, so the
code-space arithmetic and plotting the Isaac entrypoints depend on also run in
the default Pixi environment. The Isaac-side rollout scaffolding they share
lives in ``qualitative_rollout.py`` instead.

Three code spaces load here. Two are *discrete* and group-local -- an edit to
one group touches exactly the ``code_dim`` values of ``z`` that group owns and
nothing else, which is what makes the intervention experiments interpretable.
The third has no code at all. :attr:`EncoderBundle.is_discrete` is the test;
``groups``, ``categories``, and every ``category_*`` output column are defined
only when it is true.

``gumbel_multicat`` (64 groups x 128 categories -> 256 values)
    :class:`~rlopt.agent.hl_skill_encoder.GumbelMultiCategoricalSkillEncoder`.
    A plain per-group codebook lookup::

        z = codebook[arange(G), categories].reshape(B, G * code_dim)

    with ``code_to_latent = nn.Identity()`` and ``code_dim = z_dim // G``.

``sonic_fsq`` (64 coordinates x 32 levels -> 64 values)
    :class:`~rlopt.agent.hl_skill_encoder.SONICFSQSkillEncoder`. There is no
    codebook: coordinate ``g`` owns exactly one value of ``z``, and the level
    index is an integer position on a fixed lattice::

        z = (level - L // 2) / (L // 2)

    so ``code_dim = 1``, "group" reads as *coordinate* and "category" as
    *level*. Levels are ordered, unlike nominal category ids, so an edit of
    +-1 level is a small move and an edit across the lattice is a large one --
    the one interpretation difference between the two discrete spaces.

``deterministic`` (64 continuous values)
    :class:`~rlopt.agent.hl_skill_encoder.DeterministicSkillEncoder`. The trunk
    output *is* ``z``: no quantizer, no codebook, no lattice. The only thing
    bounding it is the ``--reg_coeff`` L2 penalty during pretraining, so unlike
    the other two ``z`` is unbounded and its per-dimension scale is a property
    of the trained encoder rather than of the code space. There is no discrete
    code to name, edit, or count, and this module refuses to invent one:
    :func:`code_to_z` and :func:`sample_base_code` raise, and
    :func:`encode_windows` returns no ``categories``.

:func:`code_to_z` implements both discrete spaces, and the test suites assert
each is bit-identical to its encoder's own deterministic path.

The published command is ``[z ; sin(2*pi*phase) ; cos(2*pi*phase)]``; see
:func:`append_sin_cos_phase`, mirrored from
``FrozenHighLevelSkillCommandSampler._append_command_phase``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

# --------------------------------------------------------------------------- #
# Protocol constants. These mirror this directory's train_lowlevel_fsq64.sh
# and memo.md; a mismatch against a checkpoint is asserted, never assumed.
# --------------------------------------------------------------------------- #

#: Latent modes with a group-local discrete code. Only these have "perturb one
#: group" semantics, so an intervention analysis pins itself to this tuple
#: rather than rendering a comparison that does not mean what it looks like.
DISCRETE_LATENT_MODES = ("gumbel_multicat", "sonic_fsq")

#: Every code space the shared loader understands. ``deterministic`` has no
#: discrete code at all, so an analysis that needs one must ask for
#: :data:`DISCRETE_LATENT_MODES` explicitly instead of taking the default.
SUPPORTED_LATENT_MODES = ("deterministic", *DISCRETE_LATENT_MODES)

TASK_NAME = "Isaac-Imitation-G1-v2"
AGENT_ENTRY_POINT = "rlopt_ipmd_tuned_cfg_entry_point"
PHYSICS = "newton_mjwarp"
PERSIST_ID = "bones_seed_sonic_full_129785@e714bbff"
ANCHOR_BODY = "pelvis"
EXPECTED_MOTIONS = 129785
EXPECTED_TRANSITIONS = 47491234

#: root+qpos macro state: 29 joint positions + root position + 6D root
#: orientation. The encoder input is ``MACRO_STATE_VALUES_PER_FRAME *
#: horizon_steps`` (380 at h10) and is asserted at load time.
MACRO_STATE_VALUES_PER_FRAME = 38
MACRO_STATE_TERMS = [
    "expert_motion_qpos",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
]

#: Order is column position in the reference arrays, not a set. Do not sort.
RUNTIME_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

DEFAULT_REFERENCE_ARRAYS_DIRNAME = "data/g1_bones_seed_sonic_129k_50hz_refarrays"
REFERENCE_ARRAYS_MANIFEST_NAME = "reference_arrays_manifest.json"


def repo_root() -> Path:
    """Locate the repository root without relying on a fixed parent depth."""
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "scripts" / "rlopt" / "train.py").is_file():
            return candidate
    msg = f"Could not locate the repository root above {Path(__file__).resolve()}."
    raise RuntimeError(msg)


def sha256(path: str | Path) -> str:
    """SHA-256 of a file, streamed so a 168 MB checkpoint is not held in RAM."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Hydra overrides shared by every entrypoint.
# --------------------------------------------------------------------------- #


def base_hydra_overrides(
    *,
    reference_arrays_dir: str | Path,
    device: str = "cuda:0",
    runtime_cache_device: str = "cpu",
    latent_command_dim: int = 258,
    z_dim: int = 256,
    horizon_steps: int = 10,
    reference_arrays_warm_workers: int = 8,
) -> list[str]:
    """The data + command-layout overrides every qualitative run needs.

    Mirrors the block in ``train_lowlevel_129k.sh``. Reference arrays only: no
    Zarr and no persisted replay is opened, which is what lets these run on a
    box that has the 49.4 GB artifact and nothing else.
    """
    body_names = ",".join(RUNTIME_BODY_NAMES)
    return [
        f"physics={PHYSICS}",
        f"env.expert_macro_state_terms=[{','.join(MACRO_STATE_TERMS)}]",
        "env.data.manifest=null",
        f"env.data.reference_arrays_dir={Path(reference_arrays_dir).resolve()}",
        f"env.data.reference_arrays_warm_workers={int(reference_arrays_warm_workers)}",
        f"env.data.runtime_cache_device={runtime_cache_device}",
        f"env.data.runtime_cache_body_names=[{body_names}]",
        f"env.data.macro_cache_device={device}",
        f"env.data.persist_id={PERSIST_ID}",
        f"env.command_interface.actor.dim={int(latent_command_dim)}",
        f"agent.ipmd.latent_dim={int(latent_command_dim)}",
        f"agent.ipmd.latent_learning.code_latent_dim={int(z_dim)}",
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
        f"agent.ipmd.latent_learning.code_period={int(horizon_steps)}",
        f"agent.ipmd.latent_steps_min={int(horizon_steps)}",
        f"agent.ipmd.latent_steps_max={int(horizon_steps)}",
    ]


def hl_skill_hydra_overrides(
    encoder_checkpoint: str | Path, *, horizon_steps: int = 10
) -> list[str]:
    """Drive the actor command from the frozen encoder, exactly as in training.

    With these the IPMD agent builds a ``FrozenHighLevelSkillCommandSampler``
    that re-encodes the live expert window every ``latent_steps`` control steps.
    That is the window-by-window behaviour task 1 wants, so it is reused rather
    than reimplemented.
    """
    return [
        "agent.ipmd.command_source=hl_skill",
        f"agent.ipmd.hl_skill_checkpoint_path={Path(encoder_checkpoint).resolve()}",
        f"agent.ipmd.hl_skill_horizon_steps={int(horizon_steps)}",
        "agent.ipmd.hl_skill_command_mode=z",
        "agent.ipmd.hl_skill_finetune_enabled=false",
    ]


# --------------------------------------------------------------------------- #
# Checkpoint discovery and binding.
# --------------------------------------------------------------------------- #

_MODEL_STEP_RE = re.compile(r"^model_step_(\d+)\.pt$")


def resolve_latest_policy_checkpoint(run_dir: str | Path) -> Path:
    """Newest ``model_step_<N>.pt`` under ``run_dir``, searched recursively.

    The low-level run is still training, so "the checkpoint" is whichever step
    is highest right now. Sorted numerically, not lexicographically --
    ``model_step_975175680.pt`` must not beat ``model_step_4325179392.pt``.
    """
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        msg = f"Policy run directory not found: {root}"
        raise FileNotFoundError(msg)
    best: tuple[int, Path] | None = None
    for candidate in root.rglob("model_step_*.pt"):
        match = _MODEL_STEP_RE.match(candidate.name)
        if match is None:
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, candidate)
    if best is None:
        msg = (
            f"No model_step_<N>.pt checkpoint under {root}. Pass an explicit "
            "--policy_checkpoint."
        )
        raise FileNotFoundError(msg)
    return best[1]


def policy_checkpoint_step(path: str | Path) -> int | None:
    """Frame count encoded in a ``model_step_<N>.pt`` name, if it has one."""
    match = _MODEL_STEP_RE.match(Path(path).name)
    return int(match.group(1)) if match is not None else None


def assert_encoder_binding(
    encoder_checkpoint: str | Path,
    policy_checkpoint: str | Path,
) -> dict[str, Any]:
    """Fail unless the tracker embeds tensor-identical encoder weights.

    Reuses the audit implementation so this gate cannot drift from the one the
    paper submission gates apply.
    """
    from imitation_experiments.audit.validate_latent_skill_checkpoint_binding import (
        validate_binding,
    )

    record = validate_binding(
        Path(policy_checkpoint).expanduser(),
        Path(encoder_checkpoint).expanduser(),
    )
    if not record["passed"]:
        msg = (
            "Encoder/tracker binding FAILED -- the selected encoder is not the "
            "one this tracker was trained against.\n"
            f"  encoder : {record['skill_checkpoint']}\n"
            f"  tracker : {record['low_level_checkpoint']}\n"
            f"  missing={record['missing_keys'][:5]} "
            f"unexpected={record['unexpected_keys'][:5]} "
            f"mismatched={record['mismatched_keys'][:5]}"
        )
        raise RuntimeError(msg)
    return record


# --------------------------------------------------------------------------- #
# Encoder loading and encoding.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EncoderBundle:
    """A loaded encoder plus everything derived from its checkpoint."""

    encoder: Any
    config: Any
    state_dim: int
    window_steps: int
    #: Discrete groups (FSQ: coordinates). For ``deterministic`` this is
    #: ``z_dim`` with ``code_dim == 1``, which keeps :func:`group_z_slice`
    #: meaningful as "one latent value", but there is no code behind it.
    groups: int
    #: Codes per group. ``0`` for ``deterministic``: a continuous latent has no
    #: alphabet. Read it only behind :attr:`is_discrete`.
    categories: int
    code_dim: int
    z_dim: int
    horizon_steps: int
    device: torch.device
    checkpoint_path: Path
    #: ``deterministic``, ``gumbel_multicat``, or ``sonic_fsq``. Defaulted so
    #: the older multicat-only construction sites keep working unchanged.
    latent_mode: str = "gumbel_multicat"
    #: Per-coordinate FSQ level counts, ``None`` for any other code space.
    levels: tuple[int, ...] | None = None

    @property
    def latent_command_dim(self) -> int:
        return self.z_dim + 2

    @property
    def is_fsq(self) -> bool:
        return self.latent_mode == "sonic_fsq"

    @property
    def is_discrete(self) -> bool:
        """Whether this code space has a discrete code to read or edit.

        False for ``deterministic``. Everything named ``category``,
        ``level``, or ``group`` is undefined when this is False, and the
        entrypoints omit those output columns rather than writing a
        placeholder that a later reader would take at face value.
        """
        return self.latent_mode in DISCRETE_LATENT_MODES

    @property
    def group_noun(self) -> str:
        """What one "group" is called in this code space."""
        if not self.is_discrete:
            return "dimension"
        return "coordinate" if self.is_fsq else "group"

    @property
    def category_noun(self) -> str:
        """What one "category" is called in this code space."""
        if not self.is_discrete:
            msg = (
                f"{self.latent_mode!r} is a continuous latent with no code "
                "alphabet, so 'category' names nothing. Gate on is_discrete."
            )
            raise ValueError(msg)
        return "level" if self.is_fsq else "category"


def load_skill_encoder(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    require_latent_mode: str | Sequence[str] | None = SUPPORTED_LATENT_MODES,
) -> EncoderBundle:
    """Load the frozen skill encoder standalone, without building an env.

    The state dimension is recovered from the trunk's first Linear exactly the
    way ``FrozenHighLevelSkillCommandSampler`` does it, so this and the live
    sampler agree on the encoder's shape by construction.
    """
    from rlopt.agent import HighLevelSkillDiffSRConfig, build_skill_encoder
    from rlopt.agent.hl_skill_diffsr import (
        FrozenHighLevelSkillCommandSampler,
        _encoder_window_steps,
    )

    device = torch.device(device)
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        msg = f"Skill encoder checkpoint not found: {path}"
        raise FileNotFoundError(msg)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])
    latent_mode = str(config.latent_mode)
    if require_latent_mode is not None:
        allowed = (
            (require_latent_mode,)
            if isinstance(require_latent_mode, str)
            else tuple(require_latent_mode)
        )
        if latent_mode not in allowed:
            msg = (
                f"{path} is a {latent_mode!r} encoder; this analysis requires one "
                f"of {allowed}. Group interventions are only defined for a code "
                "space where each group owns a disjoint slice of z."
            )
            raise ValueError(msg)

    state_dict = checkpoint["skill_encoder_state_dict"]
    window_steps = _encoder_window_steps(config)
    state_dim = FrozenHighLevelSkillCommandSampler._state_dim_from_encoder_state(
        state_dict, window_steps=window_steps
    )
    # Exactly the call FrozenHighLevelSkillCommandSampler makes, activation and
    # layer_norm included. Omitting those two silently rebuilds a DIFFERENT
    # network: a SiLU/no-LayerNorm encoder like the scaled FSQ64 one fails on
    # shape, but an activation-only difference loads cleanly and then computes
    # the wrong latent.
    encoder = build_skill_encoder(
        state_dim=state_dim,
        window_steps=window_steps,
        z_dim=config.z_dim,
        hidden_dims=config.encoder_hidden_dims,
        spec=config.latent_spec(),
        activation=config.encoder_activation,
        layer_norm=config.encoder_layer_norm,
    ).to(device)
    # ``strict=False`` tolerates the persisted-but-optional ``tau`` buffer that
    # some encoder revisions carry; anything else missing is a real mismatch.
    result = encoder.load_state_dict(state_dict, strict=False)
    missing = [key for key in result.missing_keys if key.split(".")[-1] != "tau"]
    if missing or result.unexpected_keys:
        msg = (
            f"Skill encoder state mismatch for {path}: missing={missing}, "
            f"unexpected={list(result.unexpected_keys)}."
        )
        raise RuntimeError(msg)
    encoder.eval()
    encoder.requires_grad_(False)

    levels: tuple[int, ...] | None = None
    if latent_mode == "deterministic":
        # The trunk output IS z. Assert the absence of the other two spaces'
        # machinery: a checkpoint whose config says "deterministic" while the
        # module carries a quantizer would otherwise load cleanly here and then
        # compute a latent this branch does not describe.
        for attribute in ("fsq", "codebook"):
            if getattr(encoder, attribute, None) is not None:
                msg = (
                    f"{path} declares latent_mode='deterministic' but its "
                    f"encoder carries a {attribute!r}. The checkpoint's config "
                    "and its weights disagree; refusing rather than guessing."
                )
                raise TypeError(msg)
        # groups = z_dim with code_dim = 1 keeps group_z_slice meaningful as
        # "one latent value". categories = 0 is the no-alphabet sentinel, and
        # is_discrete is what every reader gates on.
        groups, categories, code_dim = int(config.z_dim), 0, 1
    elif latent_mode == "sonic_fsq":
        # No codebook: the lattice IS the code space. One coordinate owns one
        # value of z, and the published command is the quantizer output, so
        # `code_to_latent` must be the identity for a level edit to be local.
        quantizer = getattr(encoder, "fsq", None)
        if quantizer is None or not hasattr(quantizer, "_levels"):
            msg = f"Expected an FSQ quantizer on the encoder; got {type(encoder).__name__}."
            raise TypeError(msg)
        levels = tuple(int(value) for value in quantizer._levels.tolist())
        if len(set(levels)) != 1:
            msg = (
                "This analysis assumes one shared level count across FSQ "
                f"coordinates, but {path} has {sorted(set(levels))}. A per-group "
                "sweep would compare unequal alphabets."
            )
            raise ValueError(msg)
        if not isinstance(encoder.code_to_latent, torch.nn.Identity):
            msg = (
                "sonic_fsq must publish the quantizer output directly, but "
                f"code_to_latent is {type(encoder.code_to_latent).__name__}. A "
                "learned projection makes a per-coordinate edit non-local."
            )
            raise TypeError(msg)
        groups, categories, code_dim = len(levels), levels[0], 1
        if int(config.z_dim) != groups:
            msg = (
                f"sonic_fsq publishes one value per coordinate, so z_dim must be "
                f"{groups}; got {int(config.z_dim)}."
            )
            raise ValueError(msg)
    else:
        codebook = getattr(encoder, "codebook", None)
        if not isinstance(codebook, torch.Tensor) or codebook.ndim != 3:
            msg = (
                "Expected a per-group codebook [groups, categories, code_dim] on the "
                f"encoder; got {type(codebook).__name__}."
            )
            raise TypeError(msg)
        groups, categories, code_dim = (int(dim) for dim in codebook.shape)

    expected_input = MACRO_STATE_VALUES_PER_FRAME * int(config.horizon_steps)
    actual_input = state_dim * (window_steps + 1)
    if actual_input != expected_input:
        msg = (
            "Encoder input width does not match the root_qpos macro state: "
            f"{actual_input} = {state_dim} x {window_steps + 1}, expected "
            f"{expected_input} = {MACRO_STATE_VALUES_PER_FRAME} x "
            f"{config.horizon_steps}. A pre-v2 670-wide encoder is refused here."
        )
        raise ValueError(msg)

    return EncoderBundle(
        encoder=encoder,
        config=config,
        state_dim=state_dim,
        window_steps=window_steps,
        groups=groups,
        categories=categories,
        code_dim=code_dim,
        z_dim=int(config.z_dim),
        horizon_steps=int(config.horizon_steps),
        device=device,
        checkpoint_path=path,
        latent_mode=latent_mode,
        levels=levels,
    )


def load_multicat_encoder(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    require_latent_mode: str | Sequence[str] | None = "gumbel_multicat",
) -> EncoderBundle:
    """:func:`load_skill_encoder` pinned to ``gumbel_multicat``.

    Kept so a caller that means "the multicat encoder specifically" still says
    so at the call site.
    """
    return load_skill_encoder(
        checkpoint_path, device, require_latent_mode=require_latent_mode
    )


def encoder_input_window(config: Any, future_window: torch.Tensor) -> torch.Tensor:
    """Slice ``future_window`` the way the encoder was trained to see it."""
    from rlopt.agent.hl_skill_diffsr import _encoder_input_window

    return _encoder_input_window(config, future_window)


@torch.no_grad()
def encode_windows(
    bundle: EncoderBundle,
    state: torch.Tensor,
    future_window: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Deterministically encode a batch of macro windows.

    Returns the latent plus the per-group diagnostics the codebook plots need.
    ``categories`` is the discrete code the tracker's frozen sampler commits to
    at inference: the per-group argmax for ``gumbel_multicat``, the per-
    coordinate level index for ``sonic_fsq``. See
    :func:`code_diagnostics_meaning` -- the three diagnostic keys carry the same
    names in both discrete code spaces but are computed differently, and every
    entrypoint records which is which in ``provenance.json``.

    **A ``deterministic`` bundle returns no ``categories`` key at all**, along
    with none of the posterior diagnostics. Callers must use ``.get()`` and omit
    the corresponding output column rather than storing a placeholder. Read
    ``bundle.is_discrete`` to know which shape to expect before calling.
    """
    if not bundle.is_discrete:
        return _encode_windows_continuous(bundle, state, future_window)
    if bundle.is_fsq:
        return _encode_windows_fsq(bundle, state, future_window)
    window = encoder_input_window(bundle.config, future_window)
    logits = bundle.encoder._pre_quantize(bundle.encoder._raw(state, window))
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    categories = logits.argmax(dim=-1)

    top2 = probs.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    entropy = -(probs * log_probs).sum(dim=-1)
    probability = probs.gather(-1, categories.unsqueeze(-1)).squeeze(-1)

    z = code_to_z(bundle, categories)
    return {
        "z": z,
        "logits": logits,
        "categories": categories,
        "probability": probability,
        "margin": margin,
        "entropy": entropy,
    }


@torch.no_grad()
def _encode_windows_continuous(
    bundle: EncoderBundle,
    state: torch.Tensor,
    future_window: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """``encode_windows`` for ``deterministic``: a latent, and nothing discrete.

    The latent comes from the encoder's own ``forward``, which is
    ``encode(..., deterministic=True)[0]`` -- the exact call
    ``FrozenHighLevelSkillCommandSampler`` makes -- so this cannot drift from
    what the tracker is published. There is no code, no posterior, and no
    lattice, so ``categories``, ``probability``, ``margin``, and ``entropy`` are
    absent rather than filled with a stand-in.

    The two scale summaries are here because a deterministic ``z`` is unbounded:
    only the ``--reg_coeff`` L2 penalty limits it, so its spread is a property
    of the trained encoder. Downstream analyses that measure distances in this
    space (clustering above all) record these so a scale-dominated result is
    visible as such instead of being read as structure.
    """
    window = encoder_input_window(bundle.config, future_window)
    z = bundle.encoder(state, window)
    return {
        "z": z,
        # Per-row, not per-batch: the caller aggregates across the whole run.
        "z_abs_max": z.abs().amax(dim=-1),
        "z_norm": z.norm(dim=-1),
    }


@torch.no_grad()
def _encode_windows_fsq(
    bundle: EncoderBundle,
    state: torch.Tensor,
    future_window: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """``encode_windows`` for ``sonic_fsq``: lattice codes, not a posterior.

    The latent is produced by the encoder's own ``_latent`` body -- trunk,
    quantizer, ``code_to_latent`` -- so this cannot drift from what the frozen
    sampler publishes. There is no distribution over levels, so the three
    diagnostic keys are recomputed from the quantization residual
    ``bounded - round(bounded)``, which is the FSQ analogue of "how close was
    this to being assigned somewhere else": 0 means the trunk landed exactly on
    a lattice point, +-0.5 means it sat on a decision boundary and an
    arbitrarily small change of input would flip the level.
    """
    encoder = bundle.encoder
    window = encoder_input_window(bundle.config, future_window)
    raw = encoder._raw(state, window)
    z_e = encoder._pre_quantize(raw)
    z_q, _flat_or_per_dim, _, _ = encoder._quantize(z_e, deterministic=True, step=None)
    z = encoder.code_to_latent(z_q).reshape(*raw.shape[:-1], bundle.z_dim)

    # Per-coordinate level indices, recomputed exactly as FSQQuantizer.forward
    # does. Its returned `code` cannot be used: it is a FLAT lattice index when
    # the level product fits int64 and per-dimension indices only when it does
    # not. SONIC's 64 x 32 is 2^320 and always returns per-dimension, but a
    # smaller lattice (a unit-test encoder, a narrower ablation) returns one
    # scalar per row, which is not what a per-coordinate analysis can use.
    bounded = encoder.fsq._bound(z_e)
    rounded = torch.round(bounded)
    residual = (bounded - rounded).abs()
    levels = encoder.fsq._levels.to(device=z_e.device)
    code = torch.minimum((rounded.long() + levels // 2).clamp(min=0), levels - 1)
    saturated = (code <= 0) | (code >= levels - 1)

    return {
        "z": z,
        "logits": z_e,
        "categories": code.to(torch.long),
        # 1 exactly on a lattice point, 0 on a decision boundary.
        "probability": (1.0 - 2.0 * residual).clamp_min(0.0),
        # Distance to the decision boundary, in lattice units.
        "margin": (0.5 - residual).clamp_min(0.0),
        # Normalized residual: high means the level is about to flip.
        "entropy": (2.0 * residual).clamp(0.0, 1.0),
        # tanh saturation: the trunk is pinned at an end of the lattice.
        "saturated": saturated,
    }


def latent_scale_summary(latent: torch.Tensor) -> dict[str, float]:
    """Per-dimension spread of a latent matrix ``[rows, z_dim]``.

    Reported because a ``deterministic`` ``z`` is unbounded: only the pretrain
    L2 penalty limits it, so its per-dimension scale is a property of the
    trained encoder rather than of the code space. Any analysis that measures
    distance in this space -- clustering above all -- inherits that scale, and
    a result that merely followed the widest few dimensions must be visible as
    such rather than read as structure. An FSQ latent lies in ``[-1, 1]`` by
    construction and its numbers are here for comparison.

    ``effective_rank`` is ``exp(H)`` of the singular-value distribution: the
    number of directions the latent actually uses, at most ``z_dim``.
    """
    values = torch.as_tensor(latent, dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        msg = f"latent must be [rows, z_dim] with at least 2 rows, got {tuple(values.shape)}."
        raise ValueError(msg)
    std = values.std(dim=0, unbiased=False)
    median = std.median().clamp(min=1e-12)
    centered = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    spectrum = singular / singular.sum().clamp(min=1e-12)
    entropy = -(spectrum * spectrum.clamp(min=1e-12).log()).sum()
    return {
        "z_std_mean": float(std.mean()),
        "z_std_min": float(std.min()),
        "z_std_max": float(std.max()),
        "z_std_ratio_max_over_median": float(std.max() / median),
        "z_abs_max": float(values.abs().max()),
        "effective_rank": float(entropy.exp()),
        "z_dim": int(values.shape[1]),
    }


def code_diagnostics_meaning(bundle: EncoderBundle) -> dict[str, str]:
    """What the shared diagnostic keys mean for this bundle's code space.

    The keys are shared so one plotting and storage path serves every encoder;
    their definitions are not. Entrypoints write this into ``provenance.json``
    so a stored ``entropy`` column is never read as the wrong quantity, and so
    a run over a continuous latent is never mistaken for a run over a code.
    """
    if not bundle.is_discrete:
        return {
            "code_space": (
                f"deterministic continuous {bundle.z_dim}-D latent: no "
                "quantizer and no codebook, bounded only by the pretrain L2 "
                "penalty, so z is unbounded and its per-dimension scale is a "
                "property of the trained encoder"
            ),
            "categories": "n/a -- a continuous latent has no code alphabet",
            "probability": "n/a -- no posterior over codes",
            "margin": "n/a -- no posterior over codes",
            "entropy": "n/a -- no posterior over codes",
            "ordered_categories": "n/a -- no categories",
            "z_abs_max": "per-row max |z|; unbounded by construction",
            "z_norm": "per-row L2 norm of z",
        }
    if bundle.is_fsq:
        return {
            "code_space": "sonic_fsq lattice: 64 coordinates x 32 ordered levels",
            "categories": "per-coordinate level index in [0, levels)",
            "probability": "1 - 2*|bounded - round(bounded)|; 1 on a lattice point",
            "margin": "0.5 - |bounded - round(bounded)|; distance to the level boundary",
            "entropy": "2*|bounded - round(bounded)|; normalized quantization residual",
            "saturated": "level pinned at 0 or levels-1 (tanh saturation)",
            "ordered_categories": "yes -- a +-1 level edit is a small move",
        }
    return {
        "code_space": "gumbel_multicat product codebook: groups x categories",
        "categories": "per-group argmax category id",
        "probability": "softmax probability of the selected category",
        "margin": "top-1 minus top-2 softmax probability",
        "entropy": "softmax entropy over the group's categories (nats)",
        "ordered_categories": "no -- category ids are nominal and group-local",
    }


def _require_discrete(bundle: EncoderBundle, what: str) -> None:
    """Refuse an operation that only means something over a discrete code."""
    if bundle.is_discrete:
        return
    msg = (
        f"{what} needs a discrete code, but this encoder is "
        f"{bundle.latent_mode!r} -- a continuous latent with no code alphabet. "
        f"Gate on bundle.is_discrete, or load with "
        f"require_latent_mode=DISCRETE_LATENT_MODES so the refusal happens at "
        f"load time instead of here."
    )
    raise ValueError(msg)


def code_to_z(bundle: EncoderBundle, categories: torch.Tensor) -> torch.Tensor:
    """Decode discrete codes ``[B, G]`` to latents ``[B, z_dim]``.

    This is the encoder's own deterministic branch written out: per-group
    codebook lookup, then flatten. ``code_to_latent`` is ``nn.Identity`` for a
    multi-categorical encoder, so there is nothing else in the path.
    """
    _require_discrete(bundle, "code_to_z")
    if categories.ndim != 2 or int(categories.shape[1]) != bundle.groups:
        msg = (
            f"categories must have shape [B, {bundle.groups}], got "
            f"{tuple(categories.shape)}."
        )
        raise ValueError(msg)
    if bool(((categories < 0) | (categories >= bundle.categories)).any()):
        msg = f"categories must all lie in [0, {bundle.categories})."
        raise ValueError(msg)
    if bundle.is_fsq:
        # SONICFSQSkillEncoder publishes ``round(bound(z_e)) / (L // 2)`` and
        # indexes levels as ``round + L // 2``, so inverting the index is the
        # whole decode. No learned parameters are involved.
        half = bundle.encoder._half_levels.detach()
        offset = (bundle.encoder.fsq._levels.detach() // 2).to(half.dtype)
        index = categories.to(device=half.device, dtype=half.dtype)
        return (index - offset) / half
    codebook = bundle.encoder.codebook.detach()
    index = categories.to(device=codebook.device, dtype=torch.long)
    group_index = torch.arange(bundle.groups, device=codebook.device)
    z_q = codebook[group_index, index]  # [B, G, code_dim]
    return z_q.reshape(int(index.shape[0]), bundle.z_dim)


def group_z_slice(bundle: EncoderBundle, group: int) -> slice:
    """The contiguous span of ``z`` owned by one group."""
    if not 0 <= int(group) < bundle.groups:
        msg = f"group must lie in [0, {bundle.groups}), got {group}."
        raise ValueError(msg)
    start = int(group) * bundle.code_dim
    return slice(start, start + bundle.code_dim)


def append_sin_cos_phase(z: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """``[z ; sin(2*pi*phase) ; cos(2*pi*phase)]``.

    Mirrors ``FrozenHighLevelSkillCommandSampler._append_command_phase``.
    Dropping the phase is catastrophic for this tracker (episode length 21
    against 144 on the 2026-08-02 screen), so it is never optional here.
    """
    angle = phase.to(device=z.device, dtype=z.dtype).reshape(-1) * (2.0 * math.pi)
    if int(angle.numel()) != int(z.shape[0]):
        msg = (
            f"phase must have one entry per row of z: {angle.numel()} != {z.shape[0]}."
        )
        raise ValueError(msg)
    features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    return torch.cat((z, features), dim=-1)


# --------------------------------------------------------------------------- #
# Code sampling and perturbation.
# --------------------------------------------------------------------------- #


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def sample_base_code(bundle: EncoderBundle, generator: torch.Generator) -> torch.Tensor:
    """One uniformly random product code ``[G]``."""
    _require_discrete(bundle, "sample_base_code")
    return torch.randint(
        0,
        bundle.categories,
        (bundle.groups,),
        generator=generator,
        device=generator.device,
        dtype=torch.long,
    )


def sample_random_codes(
    bundle: EncoderBundle,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """``count`` independent uniform product codes ``[count, G]``.

    The batched form of :func:`sample_base_code`, drawn from the same uniform
    prior: one category per group, independently per row. For ``sonic_fsq``
    that is one of the 32 lattice levels for each of the 64 coordinates, which
    is the prior the composability analysis samples.

    Rows are independent, so ``count`` robots each get their own code rather
    than sharing one draw.
    """
    _require_discrete(bundle, "sample_random_codes")
    count = int(count)
    if count < 1:
        msg = f"count must be >= 1, got {count}."
        raise ValueError(msg)
    return torch.randint(
        0,
        bundle.categories,
        (count, bundle.groups),
        generator=generator,
        device=generator.device,
        dtype=torch.long,
    )


def lattice_distance(codes: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-transition distance between consecutive codes of one sequence.

    ``codes`` is ``[T, G]``. Returns ``mean``, ``max``, and ``changed``, each
    ``[T - 1]``, measuring ``|code_t - code_{t-1}|`` over the groups.

    For an FSQ lattice the level ids are ORDERED, so "how far the code moved"
    is the meaningful quantity and a plain change count is not: a +-1 level
    edit is a small move and an edit across the lattice is a large one.
    ``changed`` (the Hamming fraction) is returned beside them because it is
    the only reading that survives for a nominal alphabet, where the level
    distances mean nothing.
    """
    if codes.ndim != 2:
        msg = f"codes must have shape [T, G], got {tuple(codes.shape)}."
        raise ValueError(msg)
    if int(codes.shape[0]) < 2:
        empty = torch.zeros(0, dtype=torch.float32)
        return {"mean": empty, "max": empty.clone(), "changed": empty.clone()}
    delta = (codes[1:].to(torch.float32) - codes[:-1].to(torch.float32)).abs()
    return {
        "mean": delta.mean(dim=-1),
        "max": delta.max(dim=-1).values,
        "changed": (delta > 0).to(torch.float32).mean(dim=-1),
    }


def distinct_categories(
    bundle: EncoderBundle,
    baseline: int,
    count: int,
    generator: torch.Generator,
) -> list[int]:
    """``count`` distinct category ids, the baseline first.

    Keeping the baseline as variant 0 makes every video self-contained: robot 0
    is the unperturbed code and the other 31 are the comparison.
    """
    count = int(count)
    if not 1 <= count <= bundle.categories:
        msg = f"count must lie in [1, {bundle.categories}], got {count}."
        raise ValueError(msg)
    pool = [c for c in range(bundle.categories) if c != int(baseline)]
    order = torch.randperm(len(pool), generator=generator, device=generator.device)
    chosen = [int(baseline)] + [pool[int(i)] for i in order[: count - 1]]
    return chosen


def perturb_one_group(
    bundle: EncoderBundle,
    base_code: torch.Tensor,
    group: int,
    categories: Sequence[int],
) -> torch.Tensor:
    """``[len(categories), G]`` codes that differ from the base only in ``group``."""
    if base_code.ndim != 1 or int(base_code.numel()) != bundle.groups:
        msg = f"base_code must have shape [{bundle.groups}], got {tuple(base_code.shape)}."
        raise ValueError(msg)
    if not 0 <= int(group) < bundle.groups:
        msg = f"group must lie in [0, {bundle.groups}), got {group}."
        raise ValueError(msg)
    codes = base_code.reshape(1, -1).repeat(len(categories), 1).clone()
    for row, category in enumerate(categories):
        if not 0 <= int(category) < bundle.categories:
            msg = f"category must lie in [0, {bundle.categories}), got {category}."
            raise ValueError(msg)
        codes[row, int(group)] = int(category)
    return codes


def perturb_distinct_groups(
    bundle: EncoderBundle,
    base_code: torch.Tensor,
    *,
    variants: int,
    generator: torch.Generator,
    groups_per_robot: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """A different set of groups per variant: ``(codes [V, G], groups [V, N], disjoint)``.

    Variant ``i`` perturbs the ``groups_per_robot`` groups in ``groups[i]``,
    each to a category drawn uniformly from the other ``categories - 1``. Every
    variant therefore differs from the base in exactly
    ``groups_per_robot * code_dim`` latent values, but in a different slice of
    ``z``, so one grid answers "what does each group (or group set) do?".

    Group sets are **disjoint** across robots when ``variants * groups_per_robot
    <= groups``: a single permutation of the codebook is dealt out in
    consecutive chunks, which is the natural generalization of one distinct
    group per robot. When the product exceeds the group count disjointness is
    impossible, so each robot samples its own subset independently and the
    returned flag is ``False`` -- sets then overlap and two robots can share a
    perturbed group.

    Unlike the other two modes there is no unperturbed robot: every slot carries
    an edit. The base code is recorded in provenance, and
    :func:`perturb_one_group` keeps variant 0 as the baseline when that
    comparison is wanted.
    """
    if base_code.ndim != 1 or int(base_code.numel()) != bundle.groups:
        msg = f"base_code must have shape [{bundle.groups}], got {tuple(base_code.shape)}."
        raise ValueError(msg)
    variants = int(variants)
    groups_per_robot = int(groups_per_robot)
    if variants < 1:
        msg = f"variants must be >= 1, got {variants}."
        raise ValueError(msg)
    if not 1 <= groups_per_robot <= bundle.groups:
        msg = (
            f"groups_per_robot must lie in [1, {bundle.groups}], got "
            f"{groups_per_robot}."
        )
        raise ValueError(msg)

    disjoint = variants * groups_per_robot <= bundle.groups
    if disjoint:
        # One permutation dealt out in consecutive chunks: no group is touched
        # by two robots, so a difference between robots is a difference of
        # groups rather than a shared group appearing twice.
        order = torch.randperm(
            bundle.groups, generator=generator, device=generator.device
        )
        groups = order[: variants * groups_per_robot].reshape(
            variants, groups_per_robot
        )
    else:
        groups = torch.stack(
            [
                torch.randperm(
                    bundle.groups, generator=generator, device=generator.device
                )[:groups_per_robot]
                for _ in range(variants)
            ]
        )

    codes = base_code.reshape(1, -1).repeat(variants, 1).clone()
    for row in range(variants):
        for group in groups[row].tolist():
            baseline = int(base_code[int(group)])
            # Offset by 1..C-1 so the category always actually changes.
            offset = int(
                torch.randint(
                    1,
                    bundle.categories,
                    (1,),
                    generator=generator,
                    device=generator.device,
                )
            )
            codes[row, int(group)] = (baseline + offset) % bundle.categories
    return codes, groups.cpu(), disjoint


def perturb_shared_groups(
    bundle: EncoderBundle,
    base_code: torch.Tensor,
    *,
    variants: int,
    num_groups: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unified group intervention: ``(codes [V, G], groups [N])``.

    One set of ``num_groups`` groups is chosen and shared by every robot.
    Variant 0 keeps the base code untouched as the visual baseline; variants
    1..V-1 resample the category of each chosen group. For any single group the
    categories are drawn *without replacement* across robots, so no two robots
    duplicate each other on that group.

    ``num_groups`` is the only knob, and its endpoints recover the earlier
    hand-written analyses:

    * ``num_groups=1`` is the one-group category sweep -- one group, a distinct
      category per robot -- i.e. :func:`perturb_one_group`.
    * ``num_groups = groups // 2`` resamples half the code per robot, the
      half-group analysis.

    Sharing the group set across robots is what makes a sweep over
    ``num_groups`` interpretable: between two runs the only thing that changes
    is how *many* groups moved, not also *which* ones.
    """
    if base_code.ndim != 1 or int(base_code.numel()) != bundle.groups:
        msg = f"base_code must have shape [{bundle.groups}], got {tuple(base_code.shape)}."
        raise ValueError(msg)
    variants = int(variants)
    num_groups = int(num_groups)
    if variants < 2:
        msg = f"variants must be >= 2 (variant 0 is the baseline), got {variants}."
        raise ValueError(msg)
    if not 1 <= num_groups <= bundle.groups:
        msg = f"num_groups must lie in [1, {bundle.groups}], got {num_groups}."
        raise ValueError(msg)
    if variants - 1 > bundle.categories - 1:
        msg = (
            f"{variants - 1} perturbed robots need that many distinct categories "
            f"per group, but only {bundle.categories - 1} non-base categories exist."
        )
        raise ValueError(msg)

    order = torch.randperm(bundle.groups, generator=generator, device=generator.device)
    groups = order[:num_groups].clone()

    codes = base_code.reshape(1, -1).repeat(variants, 1).clone()
    for group in groups.tolist():
        group = int(group)
        baseline = int(base_code[group])
        pool = [c for c in range(bundle.categories) if c != baseline]
        pick = torch.randperm(len(pool), generator=generator, device=generator.device)
        for row in range(1, variants):
            codes[row, group] = pool[int(pick[row - 1])]
    return codes, groups.cpu()


def perturb_group_subset(
    bundle: EncoderBundle,
    base_code: torch.Tensor,
    *,
    variants: int,
    groups_per_variant: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-code perturbations: ``(codes [V, G], selected_mask [V, G])``.

    Each variant independently samples ``groups_per_variant`` groups **and** a
    fresh category for every selected group, so both the "which groups" and the
    "which categories" axes vary across the grid. Variant 0 is left as the
    unperturbed base code for reference.
    """
    if base_code.ndim != 1 or int(base_code.numel()) != bundle.groups:
        msg = f"base_code must have shape [{bundle.groups}], got {tuple(base_code.shape)}."
        raise ValueError(msg)
    variants = int(variants)
    groups_per_variant = int(groups_per_variant)
    if not 1 <= groups_per_variant <= bundle.groups:
        msg = (
            f"groups_per_variant must lie in [1, {bundle.groups}], got "
            f"{groups_per_variant}."
        )
        raise ValueError(msg)

    codes = base_code.reshape(1, -1).repeat(variants, 1).clone()
    mask = torch.zeros(variants, bundle.groups, dtype=torch.bool)
    for row in range(1, variants):
        order = torch.randperm(
            bundle.groups, generator=generator, device=generator.device
        )
        selected = order[:groups_per_variant]
        mask[row, selected.cpu()] = True
        # Resample until the category actually differs, so a "perturbed" group
        # is never silently identical to the base.
        for group in selected.tolist():
            baseline = int(base_code[group])
            offset = int(
                torch.randint(
                    1,
                    bundle.categories,
                    (1,),
                    generator=generator,
                    device=generator.device,
                )
            )
            codes[row, group] = (baseline + offset) % bundle.categories
    return codes, mask


# --------------------------------------------------------------------------- #
# Motion selection, straight from the reference-arrays manifest.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MotionEntry:
    rank: int
    dataset: str
    motion: str
    trajectory: str
    length: int


class MotionCatalog:
    """The 129,785 trajectories, read from the arrays' own manifest.

    No Isaac needed: the manifest carries ``ordered_traj_list`` (rank ->
    dataset/motion/trajectory) and ``start_index``/``end_index`` (rank ->
    length), which is everything motion selection needs.
    """

    def __init__(self, entries: list[MotionEntry], manifest_path: Path) -> None:
        self.entries = entries
        self.manifest_path = manifest_path
        self._by_motion: dict[str, list[int]] = {}
        for entry in entries:
            self._by_motion.setdefault(entry.motion, []).append(entry.rank)

    @classmethod
    def from_reference_arrays(cls, reference_arrays_dir: str | Path) -> MotionCatalog:
        manifest_path = (
            Path(reference_arrays_dir).expanduser().resolve()
            / REFERENCE_ARRAYS_MANIFEST_NAME
        )
        if not manifest_path.is_file():
            msg = (
                f"Reference arrays manifest not found: {manifest_path}. Fetch or "
                "build the arrays first (see this campaign's README.md)."
            )
            raise FileNotFoundError(msg)
        manifest = json.loads(manifest_path.read_text())
        traj_info = manifest["traj_info"]
        ordered = traj_info["ordered_traj_list"]
        starts = traj_info["start_index"]
        ends = traj_info["end_index"]
        if not len(ordered) == len(starts) == len(ends):
            msg = (
                "Malformed reference arrays manifest: ordered_traj_list, "
                "start_index, and end_index have different lengths "
                f"({len(ordered)}, {len(starts)}, {len(ends)})."
            )
            raise ValueError(msg)
        entries = [
            MotionEntry(
                rank=rank,
                dataset=str(row[0]),
                motion=str(row[1]),
                trajectory=str(row[2]),
                length=int(ends[rank]) - int(starts[rank]),
            )
            for rank, row in enumerate(ordered)
        ]
        return cls(entries, manifest_path)

    def __len__(self) -> int:
        return len(self.entries)

    def by_rank(self, rank: int) -> MotionEntry:
        if not 0 <= int(rank) < len(self.entries):
            msg = f"rank must lie in [0, {len(self.entries)}), got {rank}."
            raise ValueError(msg)
        return self.entries[int(rank)]

    def rank_for_motion(self, motion: str) -> int:
        ranks = self._by_motion.get(str(motion))
        if not ranks:
            msg = f"No trajectory named {motion!r} in {self.manifest_path}."
            raise KeyError(msg)
        if len(ranks) > 1:
            msg = (
                f"Motion {motion!r} is ambiguous ({len(ranks)} trajectories: "
                f"{ranks[:5]}...). Select it by rank instead."
            )
            raise KeyError(msg)
        return ranks[0]

    def select(
        self,
        *,
        count: int,
        seed: int,
        min_length: int,
        ranks: Sequence[int] | None = None,
        motions: Sequence[str] | None = None,
    ) -> list[MotionEntry]:
        """Explicit ranks/motions if given, else a seeded draw over the catalog.

        Trajectories shorter than ``min_length`` are refused (explicit) or
        excluded (drawn) -- a motion with no complete encoder window would
        silently produce an empty plot.
        """
        if ranks is not None and motions is not None:
            msg = "Pass ranks or motions, not both."
            raise ValueError(msg)

        if ranks is not None:
            selected = [self.by_rank(rank) for rank in ranks]
        elif motions is not None:
            selected = [self.by_rank(self.rank_for_motion(m)) for m in motions]
        else:
            eligible = [e.rank for e in self.entries if e.length >= int(min_length)]
            if len(eligible) < int(count):
                msg = (
                    f"Only {len(eligible)} trajectories are at least "
                    f"{min_length} frames long; cannot draw {count}."
                )
                raise ValueError(msg)
            generator = make_generator(seed)
            order = torch.randperm(len(eligible), generator=generator)
            selected = [self.by_rank(eligible[int(i)]) for i in order[: int(count)]]
            return selected

        short = [e for e in selected if e.length < int(min_length)]
        if short:
            names = ", ".join(f"{e.motion} ({e.length} frames)" for e in short[:5])
            msg = (
                f"Explicitly selected trajectories are shorter than {min_length} "
                f"frames and have no complete encoder window: {names}"
            )
            raise ValueError(msg)
        return selected


def plan_window_starts(length: int, horizon: int, count: int) -> list[int]:
    """Evenly spaced window start frames for one motion.

    A window at start ``t`` reads frames ``[t, t + horizon]``, so the last
    usable start is ``length - 1 - horizon``; a motion with no complete window
    yields an empty list rather than an error, so the caller can skip it.

    Evenly spaced over the whole trajectory rather than the first ``count``
    windows: most motions open on a neutral standing pose, so taking the first
    ``count`` would sample that pose over and over instead of the motion.

    Short motions produce fewer than ``count`` distinct starts. Duplicates are
    dropped rather than returned twice -- a repeated row would silently weight
    that motion higher in any downstream statistic.
    """
    if int(count) <= 0:
        raise ValueError("count must be > 0.")
    last_start = int(length) - 1 - int(horizon)
    if last_start < 0:
        return []
    if int(count) == 1:
        return [0]
    step = last_start / (int(count) - 1)
    return sorted({int(round(index * step)) for index in range(int(count))})


# --------------------------------------------------------------------------- #
# Output directories and provenance.
# --------------------------------------------------------------------------- #


def plan_code_schedule(
    *,
    warmup_seconds: float,
    step_dt: float,
    phase_period: int,
    segment_steps: int,
    num_segments: int,
) -> dict[str, Any]:
    """The step schedule for a run that holds one code per segment.

    The warmup is rounded DOWN to whole command windows so the first code
    switch lands on a window boundary, and ``segment_steps`` must itself be a
    whole multiple of ``phase_period`` so every later switch does too. A switch
    part-way through a window would leave one window carrying two codes, which
    the tracker never saw in training, so this raises rather than rounding the
    caller's request into something else.
    """
    step_dt = float(step_dt)
    if step_dt <= 0.0:
        msg = f"step_dt must be positive, got {step_dt}."
        raise ValueError(msg)
    phase_period = int(phase_period)
    if phase_period < 1:
        msg = f"phase_period must be >= 1, got {phase_period}."
        raise ValueError(msg)
    segment_steps = int(segment_steps)
    num_segments = int(num_segments)
    if segment_steps < 1 or num_segments < 1:
        msg = (
            f"segment_steps and num_segments must both be >= 1, got "
            f"{segment_steps} and {num_segments}."
        )
        raise ValueError(msg)
    if segment_steps % phase_period != 0:
        msg = (
            f"segment_steps must be a whole multiple of the encoder's "
            f"horizon_steps ({phase_period}); {segment_steps} is not. A switch "
            "part-way through a command window would mix two codes in one "
            "window, which the tracker never saw in training."
        )
        raise ValueError(msg)
    requested = float(warmup_seconds)
    if requested < 0.0:
        msg = f"warmup_seconds must be >= 0, got {requested}."
        raise ValueError(msg)
    warmup_steps = int(round(requested / step_dt))
    warmup_steps = (warmup_steps // phase_period) * phase_period
    switch_steps = [warmup_steps + i * segment_steps for i in range(num_segments)]
    total_steps = warmup_steps + num_segments * segment_steps
    return {
        "warmup_steps": warmup_steps,
        "warmup_seconds_actual": warmup_steps * step_dt,
        "warmup_seconds_requested": requested,
        "switch_steps": switch_steps,
        "segment_steps": segment_steps,
        "num_segments": num_segments,
        "total_steps": total_steps,
        "total_seconds": total_steps * step_dt,
        "phase_period": phase_period,
        "step_dt": step_dt,
    }


def prepare_output_dir(path: str | Path, *, overwrite: bool) -> Path:
    """Refuse to silently write over a previous mode's artifacts."""
    directory = Path(path).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        msg = (
            f"Output directory is not empty: {directory}. Re-run with "
            "OVERWRITE=1 (or --overwrite) to intentionally replace it."
        )
        raise FileExistsError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_provenance(directory: str | Path, **fields: Any) -> Path:
    """Write ``provenance.json`` with SHA-bound inputs and the exact protocol."""
    path = Path(directory).expanduser().resolve() / "provenance.json"
    payload = {key: fields[key] for key in sorted(fields)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def announce_video(path: str | Path) -> None:
    """Print a retained video's absolute path.

    AGENTS.md requires this: the Codex app does not pass video files through
    remote SSH targets, so the path must be visible for direct access.
    """
    print(f"[VIDEO] {Path(path).expanduser().resolve()}")


# --------------------------------------------------------------------------- #
# Plots.
# --------------------------------------------------------------------------- #
# One shared figure style so every plot in the suite ships paper-ready:
# named colormaps instead of per-plot literals, one font-size scale, one
# recessive grid, and one savefig contract (print-DPI PNG plus a vector PDF
# sibling for direct \includegraphics use).

CODE_CMAP = "turbo"  # code ids: nominal categories or ordered lattice levels
OUTCOME_CMAP = "viridis"  # continuous outcomes (timeline strips)
CLUSTER_CMAP = "tab20"  # cluster identities (scatter panels)
SAVEFIG_DPI = 300
TITLE_FONTSIZE = 10
LABEL_FONTSIZE = 9
TICK_FONTSIZE = 8
LEGEND_FONTSIZE = 8
ANNOTATION_FONTSIZE = 7
GRID_KWARGS = {"alpha": 0.3, "linewidth": 0.6}


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def despine(*axes) -> None:
    """Recessive axes: drop the top/right spines, thin what remains."""
    for axis in axes:
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(labelsize=TICK_FONTSIZE, width=0.8)


def save_figure(figure, output_path: str | Path) -> Path:
    """Write ``output_path`` at print DPI plus a vector ``.pdf`` sibling."""
    path = Path(output_path).expanduser().resolve()
    figure.savefig(path, dpi=SAVEFIG_DPI, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


def adjacent_hamming(categories: torch.Tensor) -> torch.Tensor:
    """Fraction of groups whose category changes between adjacent windows."""
    if categories.ndim != 2 or int(categories.shape[0]) < 2:
        return torch.zeros(0)
    changed = (categories[1:] != categories[:-1]).to(torch.float32)
    return changed.mean(dim=-1)


def plot_codebook_selection(
    *,
    categories: torch.Tensor,
    local_steps: Sequence[int],
    motion: str,
    num_categories: int,
    output_path: str | Path,
    group_noun: str = "group",
    category_noun: str = "category",
    ordered_categories: bool = False,
) -> Path:
    """Group x window category heatmap, plus change-rate and reuse panels.

    ``group_noun``/``category_noun`` rename the axes for a code space that is
    not a product codebook -- ``sonic_fsq`` passes "coordinate" and "level".
    ``ordered_categories`` says the ids are positions on a lattice rather than
    nominal labels, which is stated on the colorbar so a reader does not infer
    an ordering that is not there (or miss one that is).
    """
    plt = _matplotlib()
    codes = categories.detach().cpu()
    num_windows, num_groups = codes.shape

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(max(8.0, 0.45 * num_windows + 4.0), 12.0),
        gridspec_kw={"height_ratios": [4.0, 1.2, 1.2]},
    )

    image = axes[0].imshow(
        codes.T.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=CODE_CMAP,
        vmin=0,
        vmax=num_categories - 1,
    )
    axes[0].set_title(
        f"{motion}\ncode selection: {num_groups} {group_noun}s x {num_windows} windows",
        fontsize=TITLE_FONTSIZE,
    )
    axes[0].set_xlabel("window (reference frame)", fontsize=LABEL_FONTSIZE)
    axes[0].set_ylabel(group_noun, fontsize=LABEL_FONTSIZE)
    # Cap the tick density: every window gets a cell, but a long motion cannot
    # afford a frame number under each one without the labels colliding.
    tick_step = max(1, (num_windows + 24) // 25)
    ticks = list(range(0, num_windows, tick_step))
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(
        [str(int(local_steps[t])) for t in ticks],
        rotation=90,
        fontsize=TICK_FONTSIZE,
    )
    axes[0].tick_params(labelsize=TICK_FONTSIZE, width=0.8)
    colorbar = figure.colorbar(
        image,
        ax=axes[0],
        fraction=0.04,
        pad=0.02,
        label=(
            f"{category_noun} index (ordered lattice position)"
            if ordered_categories
            else f"{category_noun} id (nominal, per {group_noun})"
        ),
    )
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    hamming = adjacent_hamming(codes)
    axes[1].plot(
        range(1, num_windows), hamming.numpy(), marker="o", markersize=3, linewidth=1.4
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("adjacent\nHamming", fontsize=LABEL_FONTSIZE)
    axes[1].set_xlabel("window index", fontsize=LABEL_FONTSIZE)
    axes[1].grid(True, **GRID_KWARGS)
    if int(hamming.numel()) > 0:
        axes[1].axhline(
            float(hamming.mean()),
            color="crimson",
            linestyle="--",
            linewidth=1.0,
            label=f"mean {float(hamming.mean()):.3f}",
        )
        axes[1].legend(loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=False)

    distinct = torch.tensor(
        [int(codes[:, g].unique().numel()) for g in range(num_groups)],
        dtype=torch.float32,
    )
    axes[2].bar(range(num_groups), distinct.numpy(), width=0.9)
    axes[2].set_ylabel(f"distinct\n{category_noun}s", fontsize=LABEL_FONTSIZE)
    axes[2].set_xlabel(group_noun, fontsize=LABEL_FONTSIZE)
    axes[2].set_xlim(-0.5, num_groups - 0.5)
    axes[2].grid(True, axis="y", **GRID_KWARGS)

    despine(axes[1], axes[2])
    figure.tight_layout()
    path = save_figure(figure, output_path)
    plt.close(figure)
    return path


def plot_code_timeline(
    *,
    values: torch.Tensor,
    annotations: torch.Tensor,
    switch_steps: Sequence[int],
    output_path: str | Path,
    value_label: str,
    annotation_label: str,
    title: str = "",
    fell: torch.Tensor | None = None,
) -> Path:
    """Robot x segment strip: colour is what the robot did, text is the command.

    Both arrays are ``[robots, segments]``. ``values`` drives the colour and is
    meant to be an OUTCOME (root displacement over the segment, say);
    ``annotations`` is printed in each cell and is meant to be the command that
    produced it (how far the code moved from the previous one).

    Colour deliberately does not carry the code distance. Two independent
    uniform draws over ``L`` levels differ by ``(L^2 - 1) / (3L)`` on average --
    10.66 levels at ``L = 32`` -- with very little spread, so a strip coloured
    by code distance is the same colour everywhere and says nothing. Printing it
    keeps the number available and makes that flatness visible; the colour is
    then free to answer whether different codes produced different behaviour.

    ``fell``, if given, is a boolean ``[robots, segments]`` marking the segments
    a robot entered from a fall reset, hatched so a large outcome beside a reset
    is not read as the code alone.
    """
    plt = _matplotlib()
    outcome = values.detach().cpu().to(torch.float32).numpy()
    labels = annotations.detach().cpu().to(torch.float32).numpy()
    num_robots, num_segments = outcome.shape
    if labels.shape != outcome.shape:
        msg = (
            f"values and annotations must have the same shape: "
            f"{outcome.shape} != {labels.shape}."
        )
        raise ValueError(msg)
    if len(switch_steps) != num_segments:
        msg = (
            f"switch_steps must have one entry per segment: "
            f"{len(switch_steps)} != {num_segments}."
        )
        raise ValueError(msg)

    figure, axis = plt.subplots(
        figsize=(max(6.0, 0.9 * num_segments + 3.0), max(3.0, 0.5 * num_robots + 2.0))
    )
    image = axis.imshow(
        outcome, aspect="auto", interpolation="nearest", cmap=OUTCOME_CMAP
    )
    axis.set_xticks(range(num_segments))
    axis.set_xticklabels(
        [f"{i}\n@{int(s)}" for i, s in enumerate(switch_steps)], fontsize=TICK_FONTSIZE
    )
    axis.set_yticks(range(num_robots))
    axis.set_yticklabels(
        [f"robot {i}" for i in range(num_robots)], fontsize=TICK_FONTSIZE
    )
    axis.set_title(title or f"{value_label} per segment", fontsize=TITLE_FONTSIZE)
    colorbar = figure.colorbar(
        image, ax=axis, fraction=0.04, pad=0.02, label=value_label
    )
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    cmap = image.get_cmap()
    for robot in range(num_robots):
        for segment in range(num_segments):
            # Segment 0 has no predecessor, so its annotation is not a move.
            text = "first" if segment == 0 else f"{labels[robot, segment]:.1f}"
            # Ink follows the cell: white text disappears on the bright end of
            # the colormap, so pick black or white from the cell's luminance.
            r, g, b, _ = cmap(image.norm(outcome[robot, segment]))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            axis.text(
                segment,
                robot,
                text,
                ha="center",
                va="center",
                fontsize=ANNOTATION_FONTSIZE,
                color="black" if luminance > 0.5 else "white",
            )
    if fell is not None:
        marks = fell.detach().cpu().numpy()
        for robot in range(num_robots):
            for segment in range(num_segments):
                if bool(marks[robot, segment]):
                    axis.add_patch(
                        plt.Rectangle(
                            (segment - 0.5, robot - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            hatch="///",
                            edgecolor="crimson",
                            linewidth=1.2,
                        )
                    )

    axis.set_xlabel(
        f"segment (index and switch step)\ncell text: {annotation_label}"
        "\nhatched: entered from a fall reset",
        fontsize=TICK_FONTSIZE,
    )
    figure.tight_layout()
    path = save_figure(figure, output_path)
    plt.close(figure)
    return path


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """``[..., 6]`` -> ``[..., 3, 3]``, inverting ``quat_to_rot6d_flat``.

    That function stores the first two COLUMNS of the rotation matrix,
    ``matrix[..., :2].reshape(-1)``, so the six values are
    ``[r00, r01, r10, r11, r20, r21]`` and the third column is their cross
    product. The columns come from a real rotation matrix, so they are already
    orthonormal; that is asserted rather than repaired, because quietly
    re-orthonormalizing would hide a convention mistake.
    """
    if int(rot6d.shape[-1]) != 6:
        msg = f"rot6d must have 6 trailing values, got {tuple(rot6d.shape)}."
        raise ValueError(msg)
    columns = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    first, second = columns[..., 0], columns[..., 1]
    norms = torch.stack((first.norm(dim=-1), second.norm(dim=-1)), dim=-1)
    if float((norms - 1.0).abs().max()) > 1.0e-3:
        msg = (
            "rot6d columns are not unit length "
            f"(max deviation {float((norms - 1.0).abs().max()):.3e}); the value "
            "is not a rotation in the expected layout."
        )
        raise ValueError(msg)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """``[..., 3, 3]`` -> ``[..., 6]``, matching ``quat_to_rot6d_flat``."""
    if tuple(matrix.shape[-2:]) != (3, 3):
        msg = f"matrix must be [..., 3, 3], got {tuple(matrix.shape)}."
        raise ValueError(msg)
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def reanchor_window_to_first_frame(
    window: torch.Tensor,
    *,
    pos_slice: tuple[int, int],
    ori_slice: tuple[int, int],
) -> torch.Tensor:
    """Re-express a macro window's anchor terms relative to its OWN first frame.

    ``window`` is ``[batch, frames, values_per_frame]`` as the environment
    builds it: the anchor terms arrive in the live robot's frame, so the same
    motion encodes differently depending on where the robot happens to stand.
    This removes that dependence exactly.

    Writing the robot pose as ``(t_r, R_r)`` and the expert anchors as
    ``(p_k, R_k)`` in world, the environment hands over
    ``p_b[k] = R_r^T (p_k - t_r)`` and ``R_b[k] = R_r^T R_k``. Anchoring on
    frame 0 gives::

        p'[k] = R_b[0]^T (p_b[k] - p_b[0]) = R_0^T (p_k - p_0)
        R'[k] = R_b[0]^T R_b[k]           = R_0^T R_k

    -- the robot transform cancels on both lines. What remains is the window in
    the motion's own frame, which is the "expert" context DiffSR pretraining
    samples (its ``center_index`` is the window's first frame). Joint values are
    untouched: they never depended on the robot.
    """
    if window.ndim != 3:
        msg = f"window must be [batch, frames, values], got {tuple(window.shape)}."
        raise ValueError(msg)
    pos_start, pos_end = int(pos_slice[0]), int(pos_slice[1])
    ori_start, ori_end = int(ori_slice[0]), int(ori_slice[1])
    if pos_end - pos_start != 3:
        msg = f"pos_slice must span 3 values, got {pos_end - pos_start}."
        raise ValueError(msg)
    if ori_end - ori_start != 6:
        msg = f"ori_slice must span 6 values, got {ori_end - ori_start}."
        raise ValueError(msg)

    result = window.clone()
    positions = window[..., pos_start:pos_end]
    rotations = rot6d_to_matrix(window[..., ori_start:ori_end])

    first_rotation = rotations[:, 0:1]  # [batch, 1, 3, 3]
    first_position = positions[:, 0:1]  # [batch, 1, 3]
    inverse = first_rotation.transpose(-1, -2)

    relative_position = torch.matmul(
        inverse, (positions - first_position).unsqueeze(-1)
    ).squeeze(-1)
    relative_rotation = torch.matmul(inverse, rotations)

    result[..., pos_start:pos_end] = relative_position
    result[..., ori_start:ori_end] = matrix_to_rot6d(relative_rotation)
    return result


@dataclass(frozen=True)
class CodePanel:
    """A rendered code heatmap plus the pixel geometry a cursor needs.

    ``image`` is the panel as RGB pixels. ``window_x`` is the pixel column of
    each window's centre, so a cursor can be placed on the exact code it is
    pointing at rather than on a proportion of the figure -- matplotlib's
    margins, tick labels, and colorbar all sit inside ``image`` and would
    otherwise shift the mapping.
    """

    image: Any  # np.ndarray (H, W, 3) uint8
    window_x: Any  # np.ndarray (windows,) float, pixel centre of each column
    plot_top: int
    plot_bottom: int

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def windows(self) -> int:
        return int(self.window_x.shape[0])


def render_code_panel(
    *,
    categories: torch.Tensor,
    local_steps: Sequence[int],
    motion: str,
    num_categories: int,
    height_px: int,
    group_noun: str = "group",
    category_noun: str = "category",
    ordered_categories: bool = False,
    dpi: int = 100,
    max_xticks: int = 24,
    boundary_windows: Sequence[int] = (),
) -> CodePanel:
    """Render the group x window code grid once, for use behind a moving cursor.

    Same data as :func:`plot_codebook_selection`, without the change-rate and
    reuse panels: this one has to sit beside a video frame at its height, so it
    is the heatmap alone.

    ``boundary_windows`` draws a hard vertical rule immediately BEFORE each
    listed window. It marks a discontinuity that the code grid alone would not
    explain -- a mid-rollout reference switch, where the columns to the right
    come from a different motion.
    """
    import numpy as np

    plt = _matplotlib()
    codes = categories.detach().cpu()
    num_windows, num_groups = codes.shape
    if num_windows < 1:
        msg = "render_code_panel needs at least one window."
        raise ValueError(msg)

    height_in = max(1.0, float(height_px) / float(dpi))
    figure, axis = plt.subplots(figsize=(1.6 * height_in, height_in), dpi=dpi)
    image = axis.imshow(
        codes.T.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=CODE_CMAP,
        vmin=0,
        vmax=num_categories - 1,
    )
    axis.set_title(f"{motion}\n{num_groups} {group_noun}s x {num_windows} windows")
    axis.set_xlabel("window (reference frame)")
    axis.set_ylabel(group_noun)
    # One tick per window is unreadable past a few dozen; thin them evenly and
    # keep the first and last so the span stays legible.
    stride = max(1, int(np.ceil(num_windows / max(1, int(max_xticks)))))
    ticks = list(range(0, num_windows, stride))
    if ticks[-1] != num_windows - 1:
        ticks.append(num_windows - 1)
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [str(int(local_steps[t])) for t in ticks], rotation=90, fontsize=6
    )
    figure.colorbar(
        image,
        ax=axis,
        label=(
            f"{category_noun} index (ordered lattice position)"
            if ordered_categories
            else f"{category_noun} id (nominal, per {group_noun})"
        ),
    )
    for boundary in boundary_windows:
        boundary = int(boundary)
        if not 1 <= boundary < num_windows:
            msg = (
                f"boundary_windows entries must lie in [1, {num_windows}), got "
                f"{boundary}. A boundary before the first column marks nothing."
            )
            raise ValueError(msg)
        # Drawn in data coordinates, so it lands exactly on the column edge.
        axis.axvline(
            boundary - 0.5, color="black", linewidth=2.5, linestyle="-", zorder=5
        )

    figure.tight_layout()
    figure.canvas.draw()

    buffer = np.asarray(figure.canvas.buffer_rgba())
    panel = buffer[..., :3].copy()
    height = int(panel.shape[0])

    # Data -> display coordinates, then flip: matplotlib's display origin is
    # bottom-left and an image array's is top-left.
    corners = axis.transData.transform([[float(w), 0.0] for w in range(num_windows)])
    window_x = corners[:, 0].astype(float)
    vertical = axis.transData.transform([[0.0, -0.5], [0.0, float(num_groups) - 0.5]])
    top = int(round(height - float(vertical[:, 1].max())))
    bottom = int(round(height - float(vertical[:, 1].min())))
    plt.close(figure)

    return CodePanel(
        image=panel,
        window_x=window_x,
        plot_top=max(0, min(top, height - 1)),
        plot_bottom=max(1, min(bottom, height)),
    )


def cursor_x_for_steps(
    *,
    window_x: Any,
    renewal_steps: Sequence[int],
    steps: Any,
) -> Any:
    """Pixel x of the cursor at each control step, interpolated between windows.

    ``renewal_steps[k]`` is the control step at which window ``k``'s code was
    published, so the cursor sits exactly on column ``k`` at that step and
    slides smoothly toward column ``k+1`` in between. Steps before the first
    renewal or after the last clamp to the end columns rather than running off
    the panel.
    """
    import numpy as np

    x = np.asarray(window_x, dtype=float)
    published = np.asarray(list(renewal_steps), dtype=float)
    if x.shape[0] != published.shape[0]:
        msg = (
            f"window_x has {x.shape[0]} entries but renewal_steps has "
            f"{published.shape[0]}; every window must have a publication step."
        )
        raise ValueError(msg)
    if x.shape[0] == 0:
        msg = "cursor_x_for_steps needs at least one window."
        raise ValueError(msg)
    if bool(np.any(np.diff(published) <= 0)):
        msg = f"renewal_steps must be strictly increasing, got {list(renewal_steps)}."
        raise ValueError(msg)
    return np.interp(np.asarray(steps, dtype=float), published, x)


def draw_cursor(
    panel: CodePanel,
    x: float,
    *,
    width: int = 3,
    color: tuple[int, int, int] = (255, 255, 255),
    edge: tuple[int, int, int] = (0, 0, 0),
) -> Any:
    """A copy of the panel with a vertical playhead drawn at pixel ``x``.

    The line is drawn with a dark edge because it crosses a turbo colormap,
    where a plain white line disappears over the light-yellow band.
    """
    import numpy as np

    frame = np.array(panel.image, copy=True)
    half = max(1, int(width)) // 2
    centre = int(round(float(x)))
    top, bottom = panel.plot_top, panel.plot_bottom
    for offset in range(-half - 1, half + 2):
        column = centre + offset
        if not 0 <= column < panel.width:
            continue
        is_edge = abs(offset) > half
        frame[top:bottom, column] = edge if is_edge else color
    return frame


def compose_side_by_side(
    left: Any,
    right: Any,
    *,
    background: tuple[int, int, int] = (16, 16, 16),
    gap: int = 8,
) -> Any:
    """Place two RGB images side by side, top-aligned on a common height.

    Neither image is rescaled: a resampled video frame and a resampled plot both
    lose detail, and the panel is already rendered at the frame's height. The
    shorter one is padded instead.
    """
    import numpy as np

    left_rgb = np.asarray(left)[..., :3]
    right_rgb = np.asarray(right)[..., :3]
    height = max(int(left_rgb.shape[0]), int(right_rgb.shape[0]))
    width = int(left_rgb.shape[1]) + int(gap) + int(right_rgb.shape[1])
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(background, dtype=np.uint8)
    canvas[: left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
    start = int(left_rgb.shape[1]) + int(gap)
    canvas[: right_rgb.shape[0], start : start + right_rgb.shape[1]] = right_rgb
    return canvas


def annotate_image(
    image: Any,
    text: str,
    *,
    xy: tuple[int, int] = (8, 8),
    color: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] = (0, 0, 0),
) -> Any:
    """Draw a short caption with a solid backing box, in place on a copy.

    Uses PIL's built-in bitmap font on purpose: it needs no font file, so the
    composite renders identically on a workstation and inside a container.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    picture = Image.fromarray(np.asarray(image)[..., :3].astype("uint8"))
    draw = ImageDraw.Draw(picture)
    left, top = int(xy[0]), int(xy[1])
    box = draw.textbbox((left, top), text)
    draw.rectangle(
        (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=tuple(background)
    )
    draw.text((left, top), text, fill=tuple(color))
    return np.asarray(picture)


def pad_to_macro_block(image: Any, *, block: int = 16) -> Any:
    """Pad an RGB image up to a multiple of ``block`` in both dimensions.

    H.264 wants macro-block-aligned dimensions. Padding keeps every source
    pixel; letting the encoder resize instead would rescale the plot and the
    video frame by a fraction of a percent and soften both.
    """
    import numpy as np

    picture = np.asarray(image)[..., :3]
    height, width = int(picture.shape[0]), int(picture.shape[1])
    block = max(1, int(block))
    padded_height = ((height + block - 1) // block) * block
    padded_width = ((width + block - 1) // block) * block
    if padded_height == height and padded_width == width:
        return picture
    canvas = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    canvas[:height, :width] = picture
    return canvas


def compose_filmstrip(
    frames: Sequence[Any],
    *,
    tile_px: int = 300,
    gap_px: int = 2,
    background: int = 255,
) -> Any:
    """Square-crop each frame, shrink to ``tile_px``, lay them out left to right.

    Each frame is center-cropped to its shorter side before resizing, so a
    16:9 viewport render becomes a square tile without distorting the robot;
    the camera should already have the robot at frame center. A ``gap_px``
    spacer separates tiles so adjacent poses read as distinct frames.
    """
    import numpy as np
    from PIL import Image

    if not frames:
        msg = "compose_filmstrip needs at least one frame."
        raise ValueError(msg)
    tile_px = int(tile_px)
    if tile_px < 16:
        msg = f"tile_px must be at least 16, got {tile_px}."
        raise ValueError(msg)
    tiles = []
    for frame in frames:
        picture = np.asarray(frame)[..., :3].astype(np.uint8)
        height, width = int(picture.shape[0]), int(picture.shape[1])
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        square = picture[top : top + side, left : left + side]
        resized = Image.fromarray(square).resize(
            (tile_px, tile_px), Image.Resampling.LANCZOS
        )
        tiles.append(np.asarray(resized))
    gap = max(0, int(gap_px))
    strip_width = len(tiles) * tile_px + (len(tiles) - 1) * gap
    strip = np.full((tile_px, strip_width, 3), int(background), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        x = index * (tile_px + gap)
        strip[:, x : x + tile_px] = tile
    return strip


def save_image(image: Any, output_path: str | Path) -> Path:
    """Write an RGB uint8 array to ``output_path`` (format from the suffix)."""
    import numpy as np
    from PIL import Image

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image)[..., :3].astype(np.uint8)).save(path)
    return path


def plot_effect_scatter(
    *,
    labels: Sequence[str],
    values: torch.Tensor,
    value_names: Sequence[str],
    title: str,
    output_path: str | Path,
) -> Path:
    """2-D PCA of per-variant effect vectors, annotated with the variant label."""
    plt = _matplotlib()
    matrix = values.detach().cpu().to(torch.float64)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False).clamp_min(1.0e-12)
    centered = centered / scale
    if int(centered.shape[0]) >= 2 and int(centered.shape[1]) >= 2:
        _, _, v = torch.linalg.svd(centered, full_matrices=False)
        projected = centered @ v[:2].T
    else:
        projected = torch.zeros(int(centered.shape[0]), 2, dtype=torch.float64)

    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    axis.scatter(
        projected[:, 0].numpy(),
        projected[:, 1].numpy(),
        s=64,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    for index, label in enumerate(labels):
        axis.annotate(
            label,
            (float(projected[index, 0]), float(projected[index, 1])),
            fontsize=ANNOTATION_FONTSIZE,
            color="0.25",
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_title(
        f"{title}\nPCA of standardized [{', '.join(value_names)}]",
        fontsize=TITLE_FONTSIZE,
    )
    axis.set_xlabel("PC1", fontsize=LABEL_FONTSIZE)
    axis.set_ylabel("PC2", fontsize=LABEL_FONTSIZE)
    # Equal aspect keeps PCA distances honest: a stretched axis would overstate
    # separation along it.
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, **GRID_KWARGS)
    despine(axis)
    figure.tight_layout()
    path = save_figure(figure, output_path)
    plt.close(figure)
    return path
