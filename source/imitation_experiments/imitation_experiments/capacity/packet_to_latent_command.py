#!/usr/bin/env python3
"""Drive the latent tracker from an explicit-packet planner through the frozen encoder.

Why
---
Every interface comparison so far used a *different* low-level tracker per
interface (oracles 23.6-30.6 mm), which is why oracle-normalization was needed --
and normalization structurally flatters interfaces with weak trackers. This
module removes that confound by fixing the tracker to the latent one and putting
the frozen skill encoder in front of it, so only the planner's output space
differs:

    explicit:  planner -> 670 packet -> [frozen encoder] -> z -> latent tracker
    latent:    planner ------------------------------------> z -> latent tracker

The comparison then needs no normalization at all: same tracker, same decoder,
same oracle ceiling.

The layout contract (easy to get wrong, silently)
-------------------------------------------------
`HighLevelSkillEncoder` concatenates ``[state ; flat_window]``, i.e. its input
width is ``state_dim * (window_steps + 1)`` = 67 * (9 + 1) = **670** -- exactly
the full-body packet width, and exactly the same content: current frame plus
nine future frames. So:

    frame 0      -> the encoder's ``state`` argument
    frames 1..9  -> the encoder's ``future_window`` argument

Two further traps:

* The planner target is stored **term-major**
  ``[motion 10x58 | anchor_pos 10x3 | anchor_ori 10x6]`` while the encoder wants
  **frame-interleaved** ``[motion 58, pos 3, ori 6] x 10``. Feeding term-major
  data straight in encodes garbage without erroring.
* The encoder takes **raw** features. ``feature_normalization_state_dict``
  belongs to ``diffsr.obs_norm`` and is only loaded when ``command_mode != "z"``
  or online finetuning is on (``hl_skill_diffsr.py:649-656``); this protocol sets
  neither. Normalizing first measures a network on inputs it never sees.

Installation
------------
`FrozenHighLevelSkillCommandSampler.sample_for_step` dispatches through
``self._encode_current_macro_batch``, so replacing that bound method is
sufficient to swap the command source. That keeps the publish schedule, phase
handling, tracker and metrics byte-identical to the latent oracle path, and
requires no change inside RLOpt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

# Historical full-body defaults remain public for existing diagnostics.
TERM_WIDTHS: tuple[tuple[str, int], ...] = (
    ("expert_motion", 58),
    ("expert_anchor_pos_b", 3),
    ("expert_anchor_ori_b", 6),
)
FRAME_WIDTH = sum(width for _, width in TERM_WIDTHS)
PACKET_FRAMES = 10


@dataclass(frozen=True)
class PacketLayout:
    """Term-major packet layout accepted by one frozen skill encoder."""

    term_widths: tuple[tuple[str, int], ...]
    packet_frames: int = PACKET_FRAMES

    def __post_init__(self) -> None:
        if int(self.packet_frames) <= 0:
            raise ValueError("packet_frames must be positive.")
        if not self.term_widths:
            raise ValueError("term_widths must not be empty.")
        if any(int(width) <= 0 for _, width in self.term_widths):
            raise ValueError(
                f"Every per-frame term width must be positive: {self.term_widths}."
            )

    @property
    def frame_width(self) -> int:
        return sum(int(width) for _, width in self.term_widths)

    @property
    def packet_width(self) -> int:
        return int(self.packet_frames) * self.frame_width

    @classmethod
    def from_target_spec(cls, spec: Any, *, packet_frames: int) -> "PacketLayout":
        """Derive per-frame widths from an ``InterfaceTargetSpec`` packet."""
        frames = int(packet_frames)
        names = tuple(str(name) for name in spec.term_names)
        packet_widths = tuple(int(width) for width in spec.term_widths)
        if len(names) != len(packet_widths):
            raise ValueError("Target spec has different term-name and width counts.")
        invalid = [
            (name, width)
            for name, width in zip(names, packet_widths)
            if width % frames != 0
        ]
        if invalid:
            raise ValueError(
                f"Target widths are not divisible by packet_frames={frames}: {invalid}."
            )
        return cls(
            term_widths=tuple(
                (name, width // frames) for name, width in zip(names, packet_widths)
            ),
            packet_frames=frames,
        )


DEFAULT_PACKET_LAYOUT = PacketLayout(TERM_WIDTHS, PACKET_FRAMES)


def term_major_to_frames(
    packet: torch.Tensor, layout: PacketLayout = DEFAULT_PACKET_LAYOUT
) -> torch.Tensor:
    """Convert a term-major packet to frame-interleaved encoder input."""
    if packet.ndim != 2 or int(packet.shape[-1]) != layout.packet_width:
        raise ValueError(
            f"Expected a rank-2 packet of width {layout.packet_width}, got "
            f"{tuple(packet.shape)}."
        )
    batch = int(packet.shape[0])
    blocks: list[torch.Tensor] = []
    cursor = 0
    for _, width in layout.term_widths:
        span = layout.packet_frames * width
        blocks.append(
            packet[:, cursor : cursor + span].reshape(
                batch, layout.packet_frames, width
            )
        )
        cursor += span
    return torch.cat(blocks, dim=-1)


def frames_to_term_major(
    frames: torch.Tensor, layout: PacketLayout = DEFAULT_PACKET_LAYOUT
) -> torch.Tensor:
    """Inverse of :func:`term_major_to_frames`. Used by the round-trip gate."""
    expected = (layout.packet_frames, layout.frame_width)
    if frames.ndim != 3 or tuple(frames.shape[1:]) != expected:
        raise ValueError(
            f"Expected [B, {expected[0]}, {expected[1]}], got {tuple(frames.shape)}."
        )
    batch = int(frames.shape[0])
    blocks: list[torch.Tensor] = []
    cursor = 0
    for _, width in layout.term_widths:
        blocks.append(frames[:, :, cursor : cursor + width].reshape(batch, -1))
        cursor += width
    return torch.cat(blocks, dim=-1)


def _quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = quat.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    columns = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    first = torch.nn.functional.normalize(columns[..., :, 0], dim=-1)
    second = columns[..., :, 1]
    second = second - first * (first * second).sum(dim=-1, keepdim=True)
    second = torch.nn.functional.normalize(second, dim=-1)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def first_packet_window(
    packet: torch.Tensor,
    *,
    prediction_layout: PacketLayout,
    execution_frames: int,
) -> tuple[torch.Tensor, PacketLayout]:
    """Execute the first sub-window and discard the unused VLA prediction."""
    execution_layout = PacketLayout(
        prediction_layout.term_widths, packet_frames=int(execution_frames)
    )
    if prediction_layout.packet_frames < execution_layout.packet_frames:
        raise ValueError(
            f"Prediction H{prediction_layout.packet_frames} is shorter than "
            f"execution H{execution_layout.packet_frames}."
        )
    frames = term_major_to_frames(packet, prediction_layout)
    return (
        frames_to_term_major(
            frames[:, : execution_layout.packet_frames].contiguous(),
            execution_layout,
        ),
        execution_layout,
    )


class OverlappingPacketEnsembler:
    """ACT-style ensemble over aligned receding-horizon explicit packets.

    Packets are published every ``execution_frames``. At a renewal, the next
    execution window is covered by the current packet at slots 0:K, the prior
    packet at K:2K, and so on. Root terms are transformed from each packet's
    publication-anchor frame into the current publication-anchor frame before
    averaging. History is cleared on every asynchronous episode discontinuity.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        prediction_layout: PacketLayout,
        execution_frames: int,
        decay: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if execution_frames <= 0:
            raise ValueError("execution_frames must be positive.")
        if prediction_layout.packet_frames % int(execution_frames) != 0:
            raise ValueError(
                "Temporal ensemble requires prediction_frames to be an exact "
                "multiple of execution_frames."
            )
        if decay < 0:
            raise ValueError("Temporal ensemble decay must be non-negative.")
        self.prediction_layout = prediction_layout
        self.execution_layout = PacketLayout(
            prediction_layout.term_widths, packet_frames=int(execution_frames)
        )
        self.overlap = prediction_layout.packet_frames // int(execution_frames)
        self.decay = float(decay)
        self.device = torch.device(device)
        self.dtype = dtype
        self._frames = torch.zeros(
            int(num_envs),
            self.overlap,
            prediction_layout.packet_frames,
            prediction_layout.frame_width,
            device=self.device,
            dtype=dtype,
        )
        self._anchor_pos = torch.zeros(
            int(num_envs), self.overlap, 3, device=self.device, dtype=dtype
        )
        self._anchor_quat = torch.zeros(
            int(num_envs), self.overlap, 4, device=self.device, dtype=dtype
        )
        self._anchor_quat[..., 3] = 1.0
        self._valid = torch.zeros(
            int(num_envs), self.overlap, device=self.device, dtype=torch.bool
        )
        self._last_episode_step = torch.full(
            (int(num_envs),), -1, device=self.device, dtype=torch.long
        )
        self.publications = 0
        self.history_resets = 0
        self.candidate_histogram = [0 for _ in range(self.overlap + 1)]

        cursor = 0
        self._term_slices: dict[str, slice] = {}
        for name, width in prediction_layout.term_widths:
            self._term_slices[name] = slice(cursor, cursor + int(width))
            cursor += int(width)

    def _reexpress(
        self,
        frames: torch.Tensor,
        *,
        publication_pos: torch.Tensor,
        publication_quat: torch.Tensor,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
    ) -> torch.Tensor:
        result = frames.clone()
        current_rot = _quat_xyzw_to_matrix(current_quat)
        publication_rot = _quat_xyzw_to_matrix(publication_quat)
        delta_rot = current_rot.transpose(-1, -2) @ publication_rot
        delta_pos = torch.einsum(
            "bij,bj->bi",
            current_rot.transpose(-1, -2),
            publication_pos - current_pos,
        )
        for name, width in self.prediction_layout.term_widths:
            term_slice = self._term_slices[name]
            values = frames[..., term_slice]
            if name.endswith("_pos_b"):
                if int(width) % 3 != 0:
                    raise ValueError(f"Position term {name!r} is not 3-vector packed.")
                vectors = values.reshape(values.shape[0], values.shape[1], -1, 3)
                rotated = torch.einsum("bij,btkj->btki", delta_rot, vectors)
                result[..., term_slice] = (
                    rotated + delta_pos[:, None, None, :]
                ).reshape_as(values)
            elif name.endswith("_ori_b"):
                if int(width) % 6 != 0:
                    raise ValueError(f"Orientation term {name!r} is not rot6d packed.")
                rotations = _rot6d_to_matrix(
                    values.reshape(values.shape[0], values.shape[1], -1, 6)
                )
                transformed = torch.einsum("bij,btkjl->btkil", delta_rot, rotations)
                result[..., term_slice] = _matrix_to_rot6d(transformed).reshape_as(
                    values
                )
        return result

    def update(
        self,
        *,
        env_ids: torch.Tensor,
        packet: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_quat: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> torch.Tensor:
        env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        packet = packet.to(device=self.device, dtype=self.dtype)
        anchor_pos = anchor_pos.to(device=self.device, dtype=self.dtype)
        anchor_quat = anchor_quat.to(device=self.device, dtype=self.dtype)
        episode_steps = episode_steps.to(device=self.device, dtype=torch.long)
        frames = term_major_to_frames(packet, self.prediction_layout)
        previous_steps = self._last_episode_step.index_select(0, env_ids)
        discontinuity = (episode_steps == 0) | (
            previous_steps + self.execution_layout.packet_frames != episode_steps
        )
        if bool(discontinuity.any()):
            reset_ids = env_ids[discontinuity]
            self._valid.index_fill_(0, reset_ids, False)
            self.history_resets += int(reset_ids.numel())

        if self.overlap > 1:
            self._frames[env_ids, 1:] = self._frames[env_ids, :-1].clone()
            self._anchor_pos[env_ids, 1:] = self._anchor_pos[env_ids, :-1].clone()
            self._anchor_quat[env_ids, 1:] = self._anchor_quat[env_ids, :-1].clone()
            self._valid[env_ids, 1:] = self._valid[env_ids, :-1].clone()
        self._frames[env_ids, 0] = frames
        self._anchor_pos[env_ids, 0] = anchor_pos
        self._anchor_quat[env_ids, 0] = anchor_quat
        self._valid[env_ids, 0] = True
        self._last_episode_step.index_copy_(0, env_ids, episode_steps)

        candidate_frames: list[torch.Tensor] = []
        candidate_valid: list[torch.Tensor] = []
        for age in range(self.overlap):
            start = age * self.execution_layout.packet_frames
            stop = start + self.execution_layout.packet_frames
            raw = self._frames[env_ids, age, start:stop]
            candidate_frames.append(
                self._reexpress(
                    raw,
                    publication_pos=self._anchor_pos[env_ids, age],
                    publication_quat=self._anchor_quat[env_ids, age],
                    current_pos=anchor_pos,
                    current_quat=anchor_quat,
                )
            )
            candidate_valid.append(self._valid[env_ids, age])
        candidates = torch.stack(candidate_frames, dim=1)
        valid = torch.stack(candidate_valid, dim=1)
        ages = torch.arange(self.overlap, device=self.device, dtype=self.dtype).reshape(
            1, -1
        )
        weights = torch.exp(-self.decay * ages) * valid.to(dtype=self.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        blended = (candidates * weights[:, :, None, None]).sum(dim=1)

        # Average orientations as rotations, then project the weighted matrix
        # back onto SO(3). This avoids invalid raw rot6d columns.
        for name, width in self.prediction_layout.term_widths:
            if not name.endswith("_ori_b"):
                continue
            term_slice = self._term_slices[name]
            rotations = _rot6d_to_matrix(
                candidates[..., term_slice].reshape(
                    candidates.shape[0],
                    candidates.shape[1],
                    candidates.shape[2],
                    -1,
                    6,
                )
            )
            matrix_mean = (rotations * weights[:, :, None, None, None, None]).sum(dim=1)
            u, _, vh = torch.linalg.svd(matrix_mean)
            projected = u @ vh
            negative = torch.linalg.det(projected) < 0
            if bool(negative.any()):
                u = u.clone()
                u[negative, :, -1] *= -1
                projected = u @ vh
            blended[..., term_slice] = _matrix_to_rot6d(projected).reshape(
                blended.shape[0], blended.shape[1], int(width)
            )

        counts = valid.sum(dim=1)
        for count in range(1, self.overlap + 1):
            self.candidate_histogram[count] += int((counts == count).sum().item())
        self.publications += int(env_ids.numel())
        return frames_to_term_major(blended, self.execution_layout)

    def stats(self) -> dict[str, Any]:
        return {
            "temporal_ensemble_mode": "exponential",
            "temporal_ensemble_decay": self.decay,
            "temporal_ensemble_overlap": self.overlap,
            "temporal_ensemble_publications": self.publications,
            "temporal_ensemble_history_resets": self.history_resets,
            "temporal_ensemble_candidate_histogram": {
                str(index): count
                for index, count in enumerate(self.candidate_histogram)
                if index > 0
            },
        }


def split_packet_for_encoder(
    packet: torch.Tensor, layout: PacketLayout = DEFAULT_PACKET_LAYOUT
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a packet into the encoder's current frame and future window."""

    frames = term_major_to_frames(packet, layout)
    return frames[:, 0, :].contiguous(), frames[:, 1:, :].contiguous()


def verify_frame_layout(
    feature_slices: Any, layout: PacketLayout = DEFAULT_PACKET_LAYOUT
) -> None:
    """Gate: selected terms must match the environment's per-frame layout.

    Checked against ``ImitationRLEnv._expert_macro_feature_slices``, which the
    environment fills in when it builds the macro sequence. That is an
    *external* reference, which matters: a round-trip check through
    :func:`frames_to_term_major` / :func:`term_major_to_frames` can never fail,
    because those two are inverses by construction whatever the term order is.
    Only a comparison against what the environment actually produced can catch a
    wrong assumption about where `motion`, `anchor_pos` and `anchor_ori` sit
    inside each 67-wide frame.
    """
    if not isinstance(feature_slices, dict) or not feature_slices:
        raise RuntimeError(
            "Environment did not expose _expert_macro_feature_slices, so the "
            "assumed per-frame term layout cannot be verified against it. "
            "Refusing to encode a packet whose layout is unchecked."
        )
    cursor = 0
    for name, width in layout.term_widths:
        if name not in feature_slices:
            raise RuntimeError(
                f"Environment frame layout has no term {name!r}; got "
                f"{sorted(feature_slices)}. PacketLayout is stale."
            )
        span = feature_slices[name]
        # `getattr(span, "start", span[0])` would raise on a slice: Python
        # evaluates the default eagerly even when the attribute exists.
        if isinstance(span, slice):
            start, stop = int(span.start), int(span.stop)
        else:
            start, stop = int(span[0]), int(span[1])
        if start != cursor or stop - start != width:
            raise RuntimeError(
                f"Term {name!r} occupies [{start}, {stop}) in the environment's "
                f"frame but PacketLayout assumes [{cursor}, {cursor + width}). "
                "The encoder would receive permuted features without erroring."
            )
        cursor = stop
    if cursor != layout.frame_width:
        raise RuntimeError(
            f"Environment frame width is {cursor}, layout sums to {layout.frame_width}."
        )


def build_noise_reference(
    encoder: Any,
    packets: torch.Tensor,
    *,
    packet_layout: PacketLayout = DEFAULT_PACKET_LAYOUT,
) -> dict[str, torch.Tensor]:
    """Per-dimension stds of the packet and of z, for calibrated BB3 noise.

    Both are measured on the SAME set of oracle packets so that a given alpha
    means the same relative perturbation on either side of the encoder. Without
    a shared reference the two curves would be plotted in incomparable units and
    any crossing between them would be an artifact of the scaling.
    """
    packets = packets.float()
    state, window = split_packet_for_encoder(packets, packet_layout)
    with torch.no_grad():
        z = encoder(
            state.to(next(encoder.parameters()).device),
            window.to(next(encoder.parameters()).device),
        )
    return {
        "packet_std": packets.std(dim=0, unbiased=False).clamp_min(1e-8).cpu(),
        "z_std": z.std(dim=0, unbiased=False).clamp_min(1e-8).cpu(),
    }


def install_packet_encoder_command_source(
    sampler: Any,
    *,
    planner: torch.nn.Module,
    causal_state_provider: Callable[[torch.Tensor], torch.Tensor],
    env: Any,
    packet_layout: PacketLayout = DEFAULT_PACKET_LAYOUT,
    flow_num_inference_steps: int = 16,
    flow_inference_noise_std: float = 0.0,
    packet_source: str = "planner",
    packet_noise_alpha: float = 0.0,
    z_noise_alpha: float = 0.0,
    noise_reference: dict[str, torch.Tensor] | None = None,
    noise_seed: int = 0,
    verify_layout: bool = True,
    temporal_ensemble_mode: str = "none",
    temporal_ensemble_decay: float = 0.5,
) -> Callable[[], dict[str, Any]]:
    """Route the sampler's command through ``planner -> packet -> encoder -> z``.

    ``sampler`` must be a ``FrozenHighLevelSkillCommandSampler`` (the oracle
    path), because that is the one that actually holds the frozen skill encoder.
    Returns a callable yielding provenance/diagnostic counters for the summary.
    """
    encoder = getattr(sampler, "skill_encoder", None)
    if encoder is None:
        raise ValueError(
            "The active command sampler has no skill_encoder. BB1 requires the "
            "oracle sampler (agent.ipmd.command_source=hl_skill with "
            "--skill_checkpoint), not a frozen commander."
        )
    if int(getattr(encoder, "state_dim", -1)) != packet_layout.frame_width:
        raise ValueError(
            f"Packet frame width {packet_layout.frame_width} does not match the "
            f"frozen encoder state_dim={getattr(encoder, 'state_dim', None)}."
        )
    execution_frames = int(getattr(encoder, "window_steps", -1)) + 1
    if execution_frames <= 0:
        raise ValueError("Frozen encoder has no valid state-plus-window horizon.")
    if packet_layout.packet_frames < execution_frames:
        raise ValueError(
            f"Planner packet H{packet_layout.packet_frames} is shorter than the "
            f"frozen encoder's H{execution_frames} input."
        )
    temporal_ensemble_mode = str(temporal_ensemble_mode)
    if temporal_ensemble_mode not in {"none", "exponential"}:
        raise ValueError("temporal_ensemble_mode must be 'none' or 'exponential'.")
    if temporal_ensemble_mode == "exponential" and (
        packet_layout.packet_frames == execution_frames
    ):
        raise ValueError("Temporal ensemble needs a prediction horizon with overlap.")
    if packet_layout.packet_frames > execution_frames and (
        packet_noise_alpha > 0.0 or z_noise_alpha > 0.0
    ):
        raise ValueError("Long-horizon packet execution is not a BB3 noise protocol.")
    execution_layout = PacketLayout(
        packet_layout.term_widths, packet_frames=execution_frames
    )
    original = sampler._encode_current_macro_batch
    # CPU generator so the injected noise is reproducible independently of GPU
    # kernel scheduling -- the rollout itself is already non-deterministic, and
    # a non-reproducible perturbation on top would make the curve unreadable.
    generator = torch.Generator().manual_seed(int(noise_seed))
    stats = {
        "publishes": 0,
        "layout_verified": False,
        "packet_frames": execution_layout.packet_frames,
        "planner_prediction_frames": packet_layout.packet_frames,
        "packet_frame_width": packet_layout.frame_width,
        "packet_width": execution_layout.packet_width,
        "planner_prediction_width": packet_layout.packet_width,
        "packet_noise_alpha": float(packet_noise_alpha),
        "z_noise_alpha": float(z_noise_alpha),
        "temporal_ensemble_mode": temporal_ensemble_mode,
        "temporal_ensemble_decay": float(temporal_ensemble_decay),
        "expert_pin_latent_value_count": 0,
        "expert_pin_latent_squared_error_sum": 0.0,
        "expert_pin_latent_max_abs": 0.0,
    }
    if packet_noise_alpha > 0.0 and z_noise_alpha > 0.0:
        raise ValueError(
            "Inject noise on one side of the encoder at a time; setting both "
            "confounds the two curves BB3 exists to separate."
        )
    planner.eval()
    planner.requires_grad_(False)
    ensembler: OverlappingPacketEnsembler | None = None
    if temporal_ensemble_mode == "exponential":
        ensembler = OverlappingPacketEnsembler(
            num_envs=int(getattr(env, "num_envs")),
            prediction_layout=packet_layout,
            execution_frames=execution_frames,
            decay=float(temporal_ensemble_decay),
            device=next(planner.parameters()).device,
            dtype=next(planner.parameters()).dtype,
        )

    @torch.no_grad()
    def _encode_from_predicted_packet(env_ids: torch.Tensor):
        # Keep the expert-derived tensors: `state` feeds _command_code_from_state_z
        # for non-z command modes, and the rest populate the finetune cache. Only
        # z is replaced, so everything else about the publish is unchanged.
        _, state, future_window, target, initial_z = original(env_ids)
        if verify_layout and not stats["layout_verified"]:
            verify_frame_layout(
                getattr(env, "_expert_macro_feature_slices", None), execution_layout
            )
            stats["layout_verified"] = True
        if packet_source == "expert":
            if packet_layout.packet_frames != execution_frames:
                raise ValueError(
                    "The expert pin test only supports the encoder's native horizon."
                )
            # Pin test. Pack the environment's OWN expert window into a
            # term-major packet exactly as the full-body interface would, then
            # push it back through the same split/encode path the planner packet
            # takes. A correct implementation must reproduce the latent oracle
            # exactly, because the encoder receives numerically identical input
            # to the oracle path. Any deviation localizes the bug to this
            # module rather than to the planner or the interface.
            packet = frames_to_term_major(
                torch.cat(
                    [
                        state.unsqueeze(1),
                        future_window[:, : packet_layout.packet_frames - 1],
                    ],
                    dim=1,
                ),
                execution_layout,
            )
        else:
            causal_state = causal_state_provider(env_ids)
            packet = planner(
                causal_state.to(device=state.device, dtype=state.dtype),
                num_inference_steps=int(flow_num_inference_steps),
                inference_noise_std=float(flow_inference_noise_std),
            )
            if packet_layout.packet_frames > execution_frames:
                if ensembler is None:
                    packet, _ = first_packet_window(
                        packet,
                        prediction_layout=packet_layout,
                        execution_frames=execution_frames,
                    )
                else:
                    anchor_body_name = str(
                        getattr(env, "_expert_anchor_body_name", "pelvis")
                    )
                    all_anchor_pos, all_anchor_quat = (
                        env._get_robot_anchor_state_w_fast(anchor_body_name)
                    )
                    env_ids_device = env_ids.to(
                        device=all_anchor_pos.device, dtype=torch.long
                    )
                    episode_steps = getattr(env, "episode_length_buf").index_select(
                        0, env_ids_device
                    )
                    packet = ensembler.update(
                        env_ids=env_ids,
                        packet=packet,
                        anchor_pos=all_anchor_pos.index_select(0, env_ids_device),
                        anchor_quat=all_anchor_quat.index_select(0, env_ids_device),
                        episode_steps=episode_steps,
                    )
        # BB3: inject calibrated noise on ONE side of the encoder at a time.
        # Both alphas are in per-dimension std units of the clean quantity, so
        # `packet_noise_alpha=a` and `z_noise_alpha=a` are the same relative
        # perturbation applied before vs after compression. Driven from the
        # expert packet (packet_source="expert"), this isolates the interface's
        # error tolerance from planner quality entirely: alpha=0 is the oracle.
        if packet_noise_alpha > 0.0:
            if noise_reference is None or "packet_std" not in noise_reference:
                raise RuntimeError(
                    "packet_noise_alpha needs a per-dimension packet std; pass "
                    "noise_reference={'packet_std': ...}."
                )
            std = noise_reference["packet_std"].to(packet.device, packet.dtype)
            packet = packet + torch.randn(
                packet.shape, generator=generator, device="cpu"
            ).to(packet.device, packet.dtype) * std * float(packet_noise_alpha)
        packet_state, packet_window = split_packet_for_encoder(packet, execution_layout)
        # Compare against the ENCODER's window, not the environment's. The env
        # returns horizon_steps=10 future frames (t+1..t+10), while the encoder
        # consumes state + 9 of them (t..t+9) -- `_encoder_input_window` drops
        # the last frame under encoder_window_mode='intermediate'. 67*(9+1)=670,
        # which is exactly the full-body packet: current plus nine future.
        expected_window_steps = int(encoder.window_steps)
        if int(packet_window.shape[1]) != expected_window_steps:
            raise RuntimeError(
                "Packet window does not match the encoder's expected window: "
                f"{tuple(packet_window.shape)} has {int(packet_window.shape[1])} "
                f"frames, encoder wants {expected_window_steps}."
            )
        z = encoder(packet_state, packet_window)
        if packet_source == "expert":
            # Compare inside the publication call, against the oracle encoder
            # output computed from the exact same state/window by `original`.
            # A rollout-level "published z vs current target" metric is not a
            # pin: between 5 Hz renewals the held command intentionally differs
            # from the reference window recomputed at every 50 Hz step.
            pin_error = z - initial_z
            stats["expert_pin_latent_value_count"] += int(pin_error.numel())
            stats["expert_pin_latent_squared_error_sum"] += float(
                pin_error.double().square().sum().item()
            )
            stats["expert_pin_latent_max_abs"] = max(
                float(stats["expert_pin_latent_max_abs"]),
                float(pin_error.abs().max().item()),
            )
        if z_noise_alpha > 0.0:
            if noise_reference is None or "z_std" not in noise_reference:
                raise RuntimeError(
                    "z_noise_alpha needs a per-dimension z std; pass "
                    "noise_reference={'z_std': ...}."
                )
            zstd = noise_reference["z_std"].to(z.device, z.dtype)
            z = z + torch.randn(z.shape, generator=generator, device="cpu").to(
                z.device, z.dtype
            ) * zstd * float(z_noise_alpha)
        stats["publishes"] += int(env_ids.numel())
        return z, state, future_window, target, initial_z

    sampler._encode_current_macro_batch = _encode_from_predicted_packet

    def _stats() -> dict[str, Any]:
        result = dict(stats)
        count = int(result["expert_pin_latent_value_count"])
        result["expert_pin_latent_mse"] = (
            float(result["expert_pin_latent_squared_error_sum"]) / count
            if count > 0
            else None
        )
        if ensembler is not None:
            result.update(ensembler.stats())
        return result

    return _stats
