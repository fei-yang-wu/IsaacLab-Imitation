#!/usr/bin/env python3
"""Does the skill encoder contract command error? Measured without simulation.

The paper's claim is that a latent command tolerates planner error better than
the raw packet it was compressed from. One candidate mechanism is that the
encoder *projects noise onto the skill manifold*: a perturbation of the raw
670-value packet produces a smaller relative perturbation in z than it does in
the packet itself.

That is testable with no rollout at all. Encode the clean packet, encode the
packet plus calibrated noise, and compare the two relative errors:

    input error  = RMSE(dx) / std(x)   per dimension  (== alpha by construction)
    output error = RMSE(dz) / std(z)   per dimension
    contraction  = output / input      (< 1 means the encoder suppresses noise)

Both are expressed in the same normalized-RMSE units the planner reports as
`planner_target_rmse`, so the numbers are directly comparable to the measured
planner error of each capacity tier.

Layout note (this is easy to get wrong): the full-body planner target is stored
**term-major** as [motion 10x58 | anchor_pos 10x3 | anchor_ori 10x6], while the
encoder consumes **frame-interleaved** [motion 58, pos 3, ori 6] x 10 -- see
`ImitationRLEnv._expert_macro_state_sequence_from_terms`, which reshapes each
term to [B, T, D] and concatenates on the last axis. Feeding term-major data
straight in silently encodes garbage.

Usage
-----
    pixi run python .../measure_encoder_noise_contraction.py \\
        --encoder logs/downloaded_checkpoints/lafan1_latent_deterministic_5b_seed0/skill_encoder/latest.pt \\
        --packets logs/interface_baselines/lafan1_interface_capacity/oracle_baselines/full_body_trajectory/oracle_demonstrations/rollout_training_samples/sample_step_000000.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

WINDOW_STEPS = 10
# Term widths per frame, in the order the encoder expects them.
TERM_WIDTHS = (
    ("expert_motion", 58),
    ("expert_anchor_pos_b", 3),
    ("expert_anchor_ori_b", 6),
)
FRAME_WIDTH = sum(width for _, width in TERM_WIDTHS)


def term_major_to_frame_interleaved(packet: torch.Tensor) -> torch.Tensor:
    """[B, 670] term-major -> [B, 670] frame-interleaved, the encoder's layout."""
    expected = WINDOW_STEPS * FRAME_WIDTH
    if packet.shape[-1] != expected:
        raise ValueError(f"Expected packet width {expected}, got {packet.shape[-1]}")
    batch = packet.shape[0]
    blocks: list[torch.Tensor] = []
    cursor = 0
    for _, width in TERM_WIDTHS:
        span = WINDOW_STEPS * width
        blocks.append(
            packet[:, cursor : cursor + span].reshape(batch, WINDOW_STEPS, width)
        )
        cursor += span
    return torch.cat(blocks, dim=-1).reshape(batch, expected)


class SkillEncoder(torch.nn.Module):
    """Deterministic HighLevelSkillEncoder trunk: (Linear, LayerNorm, Mish) x3 -> Linear."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        dims = [
            (state["net.0.weight"].shape[1], state["net.0.weight"].shape[0]),
            (state["net.3.weight"].shape[1], state["net.3.weight"].shape[0]),
            (state["net.6.weight"].shape[1], state["net.6.weight"].shape[0]),
            (state["net.9.weight"].shape[1], state["net.9.weight"].shape[0]),
        ]
        layers: list[torch.nn.Module] = []
        for index, (fan_in, fan_out) in enumerate(dims[:-1]):
            layers += [
                torch.nn.Linear(fan_in, fan_out),
                torch.nn.LayerNorm(fan_out),
                torch.nn.Mish(),
            ]
            del index
        layers.append(torch.nn.Linear(*dims[-1]))
        self.net = torch.nn.Sequential(*layers)
        self.load_state_dict(state, strict=True)
        self.eval()
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
        help="Injected noise, in per-dimension std units of the packet. These are "
        "the same units as the planner's reported normalized RMSE.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    blob = torch.load(args.encoder, map_location="cpu", weights_only=False)
    encoder = SkillEncoder(blob["skill_encoder_state_dict"])
    # NO input normalization. `feature_normalization_state_dict` belongs to
    # `diffsr.obs_norm`, not to the skill encoder: hl_skill_diffsr.py:649-656
    # loads it into the DiffSR module, and only when command_mode != "z" or
    # online finetuning is enabled -- neither of which this protocol sets.
    # `_encode_current_macro_batch` calls `skill_encoder(state, future_window)`
    # on raw environment features. An earlier version of this script normalized
    # the packet first and therefore measured a network being fed inputs it
    # never sees in deployment; those contraction numbers were wrong.

    samples = torch.load(args.packets, map_location="cpu", weights_only=False)
    key = "causal_target" if "causal_target" in samples else "demonstration_target"
    packets = samples[key].float()
    print(f"[INFO] {key}: {tuple(packets.shape)}")

    def encode(raw: torch.Tensor) -> torch.Tensor:
        # Frame-interleaved order IS the encoder's input layout: it concatenates
        # [state ; flat_window], i.e. frame 0 followed by frames 1..9. Raw, not
        # normalized -- see the note above.
        return encoder(term_major_to_frame_interleaved(raw))

    z_clean = encode(packets)
    packet_std = packets.std(dim=0, unbiased=False).clamp_min(1e-8)
    z_std = z_clean.std(dim=0, unbiased=False).clamp_min(1e-8)

    generator = torch.Generator().manual_seed(int(args.seed))
    rows = []
    print(
        f"\n{'alpha (in)':>11}{'measured in':>13}{'out (norm)':>12}"
        f"{'contraction':>13}{'cos(z,z~)':>11}"
    )
    for alpha in args.alphas:
        noise = (
            torch.randn(packets.shape, generator=generator) * packet_std * float(alpha)
        )
        z_noisy = encode(packets + noise)
        # Both errors as per-dimension normalized RMSE, the planner's own units.
        in_err = float(((noise / packet_std) ** 2).mean().sqrt())
        out_err = float((((z_noisy - z_clean) / z_std) ** 2).mean().sqrt())
        cos = float(F.cosine_similarity(z_noisy, z_clean, dim=-1).mean())
        rows.append(
            {
                "alpha": alpha,
                "input_normalized_rmse": in_err,
                "latent_normalized_rmse": out_err,
                "contraction_ratio": out_err / in_err if in_err else float("nan"),
                "latent_cosine": cos,
            }
        )
        print(
            f"{alpha:>11.2f}{in_err:>13.3f}{out_err:>12.3f}"
            f"{out_err / in_err:>13.3f}{cos:>11.4f}"
        )

    print(
        "\ncontraction < 1 means the encoder suppresses packet noise: the same\n"
        "relative perturbation produces a smaller relative change in z. > 1 means\n"
        "it amplifies, which would refute the projection mechanism."
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "encoder": str(args.encoder.resolve()),
                    "packets": str(args.packets.resolve()),
                    "sample_count": int(packets.shape[0]),
                    "seed": int(args.seed),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Wrote {args.output}")


if __name__ == "__main__":
    main()
