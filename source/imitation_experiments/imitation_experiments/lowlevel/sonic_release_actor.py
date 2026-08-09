"""Load NVIDIA's released SONIC G1 tracker (``sonic_release/last.pt``).

The released checkpoint is a PPO training snapshot from ``gear_sonic``. Its
``policy_state_dict`` holds the whole action transform module (ATM): three
motion encoders, an FSQ bottleneck with no parameters, and two decoders. Only
the ``g1`` encoder and the ``g1_dyn`` decoder are needed to track a reference
motion.

Shapes, read from the checkpoint rather than assumed:

* ``encoders.g1``      640 -> [2048, 1024, 512, 512] -> 64
* FSQ                  64 values seen as ``max_num_tokens`` x ``num_fsq_levels``
                       = 2 x 32, 32 levels per coordinate
* ``decoders.g1_dyn``  994 -> [2048, 2048, 1024, 1024, 512, 512] -> 29

``994 = 64 token + 930 proprioception`` and ``640 = 10 frames x 64``. All
hidden activations are SiLU and there is no observation normalizer.

Two loading obstacles are handled here:

1. The checkpoint pickles HuggingFace ``trl``/``accelerate`` objects that this
   workspace does not install. They carry no tensors, so unresolvable classes
   are replaced by inert stubs.
2. The published module names are ``gear_sonic``-internal. Only the tensors are
   used; no upstream code is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
import types
from typing import Any, Mapping

import torch
from torch import Tensor, nn


ENCODER_PREFIX = "actor_module.encoders.g1.module."
DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."
KIN_DECODER_PREFIX = "actor_module.decoders.g1_kin.module."

# gear_sonic ``sonic_release`` contract. Verified against the checkpoint by
# :func:`load_sonic_release_actor`, which refuses a mismatch.
NUM_FSQ_LEVELS = 32
MAX_NUM_TOKENS = 2
FSQ_LEVEL = 32
ENCODER_FRAMES = 10
ENCODER_FRAME_WIDTH = 64
PROPRIOCEPTION_DIM = 930
ACTION_DIM = 29


class _Stub:
    """Placeholder for a pickled class this workspace cannot import."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except Exception:  # noqa: BLE001 - any import failure means "stub it"
            return type(name, (_Stub,), {})


def _stub_pickle_module() -> types.ModuleType:
    shim = types.ModuleType("sonic_release_stub_pickle")
    shim.Unpickler = _StubUnpickler  # type: ignore[attr-defined]
    for attribute in (
        "load",
        "loads",
        "dump",
        "dumps",
        "UnpicklingError",
        "HIGHEST_PROTOCOL",
        "DEFAULT_PROTOCOL",
    ):
        setattr(shim, attribute, getattr(pickle, attribute))
    return shim


def load_sonic_release_state_dict(checkpoint: str | Path) -> dict[str, Tensor]:
    """Return the released actor tensors, ignoring the training-only objects."""
    payload = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=False,
        pickle_module=_stub_pickle_module(),
    )
    state = payload.get("policy_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"{checkpoint} has no policy_state_dict mapping.")
    return {key: value for key, value in state.items() if torch.is_tensor(value)}


class FSQ(nn.Module):
    """Finite scalar quantization, ``vector_quantize_pytorch.FSQ`` convention.

    The released config sets ``num_fsq_levels: 32`` and ``fsq_level_list: 32``,
    so every coordinate has 32 levels. ``eps`` reproduces the upstream bound;
    it keeps ``tanh`` strictly inside the outermost level instead of only
    reaching it in the limit.
    """

    def __init__(self, levels: int = FSQ_LEVEL, eps: float = 1e-3) -> None:
        super().__init__()
        if int(levels) < 2:
            raise ValueError(f"FSQ levels must be >= 2, got {levels}.")
        self.levels = int(levels)
        self.eps = float(eps)
        self.half_l = (self.levels - 1) * (1.0 + self.eps) / 2.0
        self.offset = 0.5 if self.levels % 2 == 0 else 0.0
        self.half_width = self.levels // 2
        shift = torch.atanh(torch.tensor(self.offset / self.half_l))
        self.register_buffer("_shift", shift, persistent=False)

    def bound(self, z: Tensor) -> Tensor:
        return (z + self._shift.to(z.dtype)).tanh() * self.half_l - self.offset

    def forward(self, z: Tensor) -> Tensor:
        """Return the normalized lattice value the decoder consumes."""
        bounded = self.bound(z)
        quantized = bounded + (bounded.round() - bounded).detach()
        return quantized / self.half_width

    def snap(self, codes: Tensor) -> Tensor:
        """Project already-normalized values back onto the lattice."""
        limit = self.half_width
        levels = (codes * limit).round().clamp(-limit, limit - 1)
        return levels / limit


def _mlp_from_state_dict(
    state: Mapping[str, Tensor], prefix: str, *, activation: type[nn.Module] = nn.SiLU
) -> nn.Sequential:
    """Rebuild ``nn.Sequential(Linear, act, Linear, act, ..., Linear)``.

    ``gear_sonic`` stores the layers as ``module.<even index>``, so the odd
    indices are exactly the activations that carry no parameters.
    """
    indices = sorted(
        {
            int(key[len(prefix) :].split(".", 1)[0])
            for key in state
            if key.startswith(prefix) and key.endswith(".weight")
        }
    )
    if not indices:
        raise ValueError(f"No linear layers found under {prefix!r}.")
    layers: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight = state[f"{prefix}{index}.weight"]
        bias = state[f"{prefix}{index}.bias"]
        linear = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        layers.append(linear)
        if position < len(indices) - 1:
            layers.append(activation())
    return nn.Sequential(*layers)


def pack_encoder_window(
    joint_pos: Tensor, joint_vel: Tensor, anchor_ori: Tensor
) -> Tensor:
    """Lay out the reference window the way the released weights expect.

    Inputs are ``[B, 10, 29]``, ``[B, 10, 29]``, and ``[B, 10, 6]``: reference
    joint positions, reference joint velocities, and the 6D anchor orientation,
    sampled at ``dt_future_ref_frames=0.1`` (stride 5 at 50 fps, a 0.9 s span).

    The layout is **not** ten frames of ``[qpos, qvel, ori]``. SONIC's flat
    ``command_multi_future`` is term-major ``[qpos(290) | qvel(290)]`` and gets
    reshaped to ``(10, 58)`` before the 6-wide orientation is appended, so each
    64-wide block holds *two consecutive frames of one term*::

        block 0..4 : [qpos[2b], qpos[2b+1], anchor_ori[b]]
        block 5..9 : [qvel[2b], qvel[2b+1], anchor_ori[5 + b]]

    Feeding a tidy per-frame ``[qpos, qvel, ori]`` layout instead produces
    plausible but wrong tokens. This order was recovered by probing the
    released ``model_encoder.onnx`` slot by slot and reproduces it exactly
    (bitwise, on random inputs).
    """
    for name, tensor, width in (
        ("joint_pos", joint_pos, 29),
        ("joint_vel", joint_vel, 29),
        ("anchor_ori", anchor_ori, 6),
    ):
        if tensor.dim() != 3 or tuple(tensor.shape[1:]) != (ENCODER_FRAMES, width):
            raise ValueError(
                f"{name} must be [B, {ENCODER_FRAMES}, {width}], "
                f"got {tuple(tensor.shape)}."
            )
    blocks: list[Tensor] = []
    for block in range(ENCODER_FRAMES // 2):
        blocks += [
            joint_pos[:, 2 * block],
            joint_pos[:, 2 * block + 1],
            anchor_ori[:, block],
        ]
    for block in range(ENCODER_FRAMES // 2):
        blocks += [
            joint_vel[:, 2 * block],
            joint_vel[:, 2 * block + 1],
            anchor_ori[:, ENCODER_FRAMES // 2 + block],
        ]
    return torch.cat(blocks, dim=-1)


PROPRIOCEPTION_FRAMES = 10
# Term-major order and per-term width of SONIC's 930 policy observation, read
# from the released PolicyCfg declaration. Isaac Lab concatenates by class
# field order, not by the YAML key order.
PROPRIOCEPTION_TERMS: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
    ("last_action", 29),
    ("gravity_dir", 3),
)


def assemble_proprioception(
    gravity_dir: Tensor,
    base_ang_vel: Tensor,
    joint_pos_rel: Tensor,
    joint_vel_rel: Tensor,
    last_action: Tensor,
) -> Tensor:
    """Assemble SONIC's 930 proprioception from its five history terms.

    Each input is ``[B, 10, width]`` with frames ordered **oldest first, newest
    last** — the order IsaacLab's ``CircularBuffer.buffer`` yields (verified in
    the installed IsaacLab: "most recent entry at the end").

    The layout is **term-major**: each term's full 10-frame history is
    contiguous, and the five terms concatenate in the order above::

        [base_ang_vel(30) | joint_pos(290) | joint_vel(290) | last_action(290) | gravity(30)]

    This is *not* the ``planner_state`` ``10 x 93`` frame-major layout. Both are
    930 wide; feeding one where the other is expected is a silent, plausible
    error, so this assembler is explicit about which one SONIC's decoder wants.

    ``gravity_dir`` is gravity in the pelvis-anchor frame (SONIC's
    ``gravity_dir``), not base-frame ``projected_gravity``; for the G1 the
    pelvis is the base link, but any anchor heading offset must be reproduced
    when this is wired into the env.
    """
    terms = {
        "gravity_dir": gravity_dir,
        "base_ang_vel": base_ang_vel,
        "joint_pos_rel": joint_pos_rel,
        "joint_vel_rel": joint_vel_rel,
        "last_action": last_action,
    }
    flattened: list[Tensor] = []
    for name, width in PROPRIOCEPTION_TERMS:
        tensor = terms[name]
        if tensor.dim() != 3 or tuple(tensor.shape[1:]) != (
            PROPRIOCEPTION_FRAMES,
            width,
        ):
            raise ValueError(
                f"{name} must be [B, {PROPRIOCEPTION_FRAMES}, {width}], "
                f"got {tuple(tensor.shape)}."
            )
        flattened.append(tensor.reshape(tensor.shape[0], -1))
    return torch.cat(flattened, dim=-1)


@dataclass(frozen=True)
class SonicReleaseSpec:
    """Shapes read out of the checkpoint, for provenance records."""

    encoder_input_dim: int
    token_dim: int
    decoder_input_dim: int
    action_dim: int
    proprioception_dim: int
    encoder_frames: int
    encoder_frame_width: int

    def to_dict(self) -> dict[str, int]:
        return {
            "encoder_input_dim": self.encoder_input_dim,
            "token_dim": self.token_dim,
            "decoder_input_dim": self.decoder_input_dim,
            "action_dim": self.action_dim,
            "proprioception_dim": self.proprioception_dim,
            "encoder_frames": self.encoder_frames,
            "encoder_frame_width": self.encoder_frame_width,
        }


class SonicReleaseActor(nn.Module):
    """SONIC's ``g1`` encoder, FSQ bottleneck, and ``g1_dyn`` decoder.

    The forward pass is deterministic: the released ``std`` is a fixed
    exploration scale used during PPO, and evaluation uses the mean action.
    """

    def __init__(self, state: Mapping[str, Tensor]) -> None:
        super().__init__()
        self.encoder = _mlp_from_state_dict(state, ENCODER_PREFIX)
        self.decoder = _mlp_from_state_dict(state, DECODER_PREFIX)
        self.quantizer = FSQ(FSQ_LEVEL)

        encoder_input_dim = int(self.encoder[0].in_features)
        token_dim = int(self.encoder[-1].out_features)
        decoder_input_dim = int(self.decoder[0].in_features)
        action_dim = int(self.decoder[-1].out_features)
        if token_dim != NUM_FSQ_LEVELS * MAX_NUM_TOKENS:
            raise ValueError(
                "Released SONIC token width must be "
                f"{NUM_FSQ_LEVELS * MAX_NUM_TOKENS}, got {token_dim}."
            )
        if decoder_input_dim - token_dim != PROPRIOCEPTION_DIM:
            raise ValueError(
                "Decoder input must be token + 930 proprioception, got "
                f"{decoder_input_dim} - {token_dim}."
            )
        if action_dim != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM} actions, got {action_dim}.")
        if encoder_input_dim % ENCODER_FRAMES:
            raise ValueError(
                f"Encoder input {encoder_input_dim} is not {ENCODER_FRAMES} frames."
            )
        self.spec = SonicReleaseSpec(
            encoder_input_dim=encoder_input_dim,
            token_dim=token_dim,
            decoder_input_dim=decoder_input_dim,
            action_dim=action_dim,
            proprioception_dim=decoder_input_dim - token_dim,
            encoder_frames=ENCODER_FRAMES,
            encoder_frame_width=encoder_input_dim // ENCODER_FRAMES,
        )
        action_std = state.get("std")
        if torch.is_tensor(action_std):
            self.register_buffer("action_std", action_std.clone(), persistent=False)
        self.eval()

    def encode(self, window: Tensor) -> Tensor:
        """Reference window ``[B, 640]`` (or ``[B, 10, 64]``) -> token ``[B, 64]``."""
        if window.dim() == 3:
            window = window.reshape(window.shape[0], -1)
        if window.shape[-1] != self.spec.encoder_input_dim:
            raise ValueError(
                f"Encoder window must be {self.spec.encoder_input_dim} wide, "
                f"got {window.shape[-1]}."
            )
        latent = self.encoder(window)
        tokens = latent.reshape(*latent.shape[:-1], MAX_NUM_TOKENS, NUM_FSQ_LEVELS)
        return self.quantizer(tokens).reshape(*latent.shape[:-1], self.spec.token_dim)

    def encode_pre_quantization(self, window: Tensor) -> Tensor:
        """The continuous latent before FSQ, for a pre-quantization study."""
        if window.dim() == 3:
            window = window.reshape(window.shape[0], -1)
        return self.encoder(window)

    def decode(self, token: Tensor, proprioception: Tensor) -> Tensor:
        """Token ``[B, 64]`` plus proprioception ``[B, 930]`` -> action ``[B, 29]``."""
        if proprioception.shape[-1] != self.spec.proprioception_dim:
            raise ValueError(
                f"Proprioception must be {self.spec.proprioception_dim} wide, "
                f"got {proprioception.shape[-1]}."
            )
        return self.decoder(torch.cat([token, proprioception], dim=-1))

    def forward(self, window: Tensor, proprioception: Tensor) -> Tensor:
        return self.decode(self.encode(window), proprioception)


def load_sonic_release_actor(checkpoint: str | Path) -> SonicReleaseActor:
    """Build the deterministic released tracker from ``last.pt``."""
    return SonicReleaseActor(load_sonic_release_state_dict(checkpoint))
