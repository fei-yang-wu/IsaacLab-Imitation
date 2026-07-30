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

from typing import Any, Callable

import torch

# Per-frame term widths, in the order the environment concatenates them.
TERM_WIDTHS: tuple[tuple[str, int], ...] = (
    ("expert_motion", 58),
    ("expert_anchor_pos_b", 3),
    ("expert_anchor_ori_b", 6),
)
FRAME_WIDTH = sum(width for _, width in TERM_WIDTHS)
PACKET_FRAMES = 10


def term_major_to_frames(packet: torch.Tensor) -> torch.Tensor:
    """``[B, 670]`` term-major -> ``[B, 10, 67]`` frame-interleaved."""
    expected = PACKET_FRAMES * FRAME_WIDTH
    if packet.ndim != 2 or int(packet.shape[-1]) != expected:
        raise ValueError(
            f"Expected a rank-2 packet of width {expected}, got {tuple(packet.shape)}."
        )
    batch = int(packet.shape[0])
    blocks: list[torch.Tensor] = []
    cursor = 0
    for _, width in TERM_WIDTHS:
        span = PACKET_FRAMES * width
        blocks.append(
            packet[:, cursor : cursor + span].reshape(batch, PACKET_FRAMES, width)
        )
        cursor += span
    return torch.cat(blocks, dim=-1)


def frames_to_term_major(frames: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`term_major_to_frames`. Used by the round-trip gate."""
    if frames.ndim != 3 or tuple(frames.shape[1:]) != (PACKET_FRAMES, FRAME_WIDTH):
        raise ValueError(
            f"Expected [B, {PACKET_FRAMES}, {FRAME_WIDTH}], got {tuple(frames.shape)}."
        )
    batch = int(frames.shape[0])
    blocks: list[torch.Tensor] = []
    cursor = 0
    for _, width in TERM_WIDTHS:
        blocks.append(frames[:, :, cursor : cursor + width].reshape(batch, -1))
        cursor += width
    return torch.cat(blocks, dim=-1)


def split_packet_for_encoder(
    packet: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, 670]`` term-major -> ``(state [B, 67], future_window [B, 9, 67])``."""
    frames = term_major_to_frames(packet)
    return frames[:, 0, :].contiguous(), frames[:, 1:, :].contiguous()


def verify_frame_layout(feature_slices: Any) -> None:
    """Gate: TERM_WIDTHS must match the environment's own per-frame term layout.

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
    for name, width in TERM_WIDTHS:
        if name not in feature_slices:
            raise RuntimeError(
                f"Environment frame layout has no term {name!r}; got "
                f"{sorted(feature_slices)}. TERM_WIDTHS is stale."
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
                f"frame but TERM_WIDTHS assumes [{cursor}, {cursor + width}). "
                "The encoder would receive permuted features without erroring."
            )
        cursor = stop
    if cursor != FRAME_WIDTH:
        raise RuntimeError(
            f"Environment frame width is {cursor}, TERM_WIDTHS sums to {FRAME_WIDTH}."
        )


def build_noise_reference(
    encoder: Any, packets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-dimension stds of the packet and of z, for calibrated BB3 noise.

    Both are measured on the SAME set of oracle packets so that a given alpha
    means the same relative perturbation on either side of the encoder. Without
    a shared reference the two curves would be plotted in incomparable units and
    any crossing between them would be an artifact of the scaling.
    """
    packets = packets.float()
    state, window = split_packet_for_encoder(packets)
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
    flow_num_inference_steps: int = 16,
    flow_inference_noise_std: float = 0.0,
    packet_source: str = "planner",
    packet_noise_alpha: float = 0.0,
    z_noise_alpha: float = 0.0,
    noise_reference: dict[str, torch.Tensor] | None = None,
    noise_seed: int = 0,
    verify_layout: bool = True,
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
    original = sampler._encode_current_macro_batch
    # CPU generator so the injected noise is reproducible independently of GPU
    # kernel scheduling -- the rollout itself is already non-deterministic, and
    # a non-reproducible perturbation on top would make the curve unreadable.
    generator = torch.Generator().manual_seed(int(noise_seed))
    stats = {
        "publishes": 0,
        "layout_verified": False,
        "packet_noise_alpha": float(packet_noise_alpha),
        "z_noise_alpha": float(z_noise_alpha),
    }
    if packet_noise_alpha > 0.0 and z_noise_alpha > 0.0:
        raise ValueError(
            "Inject noise on one side of the encoder at a time; setting both "
            "confounds the two curves BB3 exists to separate."
        )
    planner.eval()
    planner.requires_grad_(False)

    @torch.no_grad()
    def _encode_from_predicted_packet(env_ids: torch.Tensor):
        # Keep the expert-derived tensors: `state` feeds _command_code_from_state_z
        # for non-z command modes, and the rest populate the finetune cache. Only
        # z is replaced, so everything else about the publish is unchanged.
        _, state, future_window, target, initial_z = original(env_ids)
        if verify_layout and not stats["layout_verified"]:
            verify_frame_layout(getattr(env, "_expert_macro_feature_slices", None))
            stats["layout_verified"] = True
        if packet_source == "expert":
            # Pin test. Pack the environment's OWN expert window into a
            # term-major packet exactly as the full-body interface would, then
            # push it back through the same split/encode path the planner packet
            # takes. A correct implementation must reproduce the latent oracle
            # exactly, because the encoder receives numerically identical input
            # to the oracle path. Any deviation localizes the bug to this
            # module rather than to the planner or the interface.
            packet = frames_to_term_major(
                torch.cat(
                    [state.unsqueeze(1), future_window[:, : PACKET_FRAMES - 1]], dim=1
                )
            )
        else:
            causal_state = causal_state_provider(env_ids)
            packet = planner(
                causal_state.to(device=state.device, dtype=state.dtype),
                num_inference_steps=int(flow_num_inference_steps),
                inference_noise_std=float(flow_inference_noise_std),
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
        packet_state, packet_window = split_packet_for_encoder(packet)
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
    return lambda: dict(stats)
