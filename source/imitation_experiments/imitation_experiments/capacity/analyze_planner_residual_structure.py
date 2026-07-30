#!/usr/bin/env python3
"""Are planner errors worse than isotropic noise of the same size? Why?

Why this exists
---------------
BB3 measured the closed-loop cost of *injected* command noise through the shared
latent tracker and found the packet side and the z side identical (1.01x, 0.99x,
0.93x at alpha 0.10/0.25/0.50), and cheap: alpha=0.50 costs only 35.3 mm against
a 30.4 mm oracle. So the interface does not tolerate error better on one side of
the encoder than the other, and moderate command error is nearly free.

Yet a real full-body planner whose measured command error is ~0.556 in those same
normalized units tracks at 103-137 mm -- 3-4x worse than injected noise of
comparable magnitude. Something about *planner* error is far more damaging than
white noise of the same size.

The obvious candidate is structure. Isotropic noise is independent across
dimensions and across the 10 frames of the packet; a planner's residual need not
be. A residual that is coherent within a frame (a consistent pose offset) or
persistent across the window (a drift) moves the commanded trajectory somewhere
physically different, whereas the same energy spread as white noise mostly
averages out inside the tracker.

This script quantifies that difference with no simulation at all:

    magnitude    residual RMSE / target std, per dimension -- the "alpha" the
                 planner is actually operating at, directly comparable to BB3's
                 injected alphas.

    temporal     correlation between residuals at frame t and frame t+1 within
                 one packet. Isotropic noise gives ~0; a drifting planner gives
                 a large positive value.

    spatial      mean off-diagonal correlation between residual dimensions
                 within a frame. Isotropic noise gives ~0; a coherent pose
                 offset gives a large positive value.

    spectrum     fraction of residual energy in the top principal component.
                 Isotropic noise over D dims gives ~1/D; a low-rank residual
                 concentrates.

Each is reported against a matched isotropic control drawn to the same
per-dimension magnitude, so the comparison is like-for-like rather than against
a theoretical expectation.

Usage
-----
    pixi run python .../analyze_planner_residual_structure.py \\
        --planner logs/.../full_body_trajectory/planner_pretrain/checkpoints/latest.pt \\
        --samples logs/.../oracle_demonstrations/rollout_training_samples/sample_step_000000.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from imitation_experiments.planner.interface_planner_common import load_planner_checkpoint
from imitation_experiments.capacity.packet_to_latent_command import (
    FRAME_WIDTH,
    PACKET_FRAMES,
    term_major_to_frames,
)


def _structure(residual_frames: torch.Tensor, label: str) -> dict[str, float]:
    """Structure statistics for a [N, frames, width] residual."""
    n, frames, width = residual_frames.shape
    flat = residual_frames.reshape(n, -1)

    # Temporal: correlate frame t against frame t+1, pooled over dims and rows.
    if frames < 2:
        temporal = float("nan")
    else:
        a = residual_frames[:, :-1, :].reshape(-1)
        b = residual_frames[:, 1:, :].reshape(-1)
        temporal = float(
            torch.corrcoef(torch.stack([a, b]))[0, 1] if a.numel() > 1 else float("nan")
        )

    # Spatial: mean off-diagonal |corr| between dimensions within a frame.
    per_frame = residual_frames.reshape(-1, width)
    centered = per_frame - per_frame.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1e-8)
    corr = (centered / std).T @ (centered / std) / centered.shape[0]
    off = corr - torch.diag(torch.diag(corr))
    spatial = float(off.abs().sum() / (width * (width - 1)))

    # Spectrum: energy fraction in the leading principal component.
    centered_flat = flat - flat.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centered_flat.float())
    energy = (sv**2).sum().clamp_min(1e-12)
    top1 = float((sv[0] ** 2 / energy).item())

    return {
        "label": label,
        "temporal_corr_t_t1": temporal,
        "spatial_mean_abs_corr": spatial,
        "top_pc_energy_fraction": top1,
        "isotropic_reference_top_pc": 1.0 / float(flat.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--state_key", type=str, default="planner_state")
    parser.add_argument("--flow_num_inference_steps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    planner, spec, _ = load_planner_checkpoint(args.planner, map_location=args.device)
    planner = planner.to(args.device).eval()
    planner.requires_grad_(False)

    data = torch.load(args.samples, map_location="cpu", weights_only=False)
    target_key = "causal_target" if "causal_target" in data else "demonstration_target"
    state = data[args.state_key].float().to(args.device)
    target = data[target_key].float().to(args.device)
    # The 670-wide full-body packet has explicit 10-frame structure, so the
    # temporal statistic is meaningful. A latent target (256-d z) is a single
    # vector per publish -- there are no frames inside it, so "correlation
    # between frame t and t+1" is undefined and reporting it as 0 would look
    # like an absence of structure rather than an absence of the axis. Fall back
    # to a single pseudo-frame; magnitude and top-PC energy stay comparable.
    framed = int(target.shape[-1]) == PACKET_FRAMES * FRAME_WIDTH
    if not framed:
        print(
            f"[NOTE] Target width {target.shape[-1]} for interface "
            f"{spec.interface!r} has no packet frame structure; the temporal "
            "statistic is not applicable and is reported as nan."
        )

    with torch.no_grad():
        prediction = planner(
            state,
            num_inference_steps=int(args.flow_num_inference_steps),
            inference_noise_std=0.0,
        )
    residual = prediction - target

    target_std = target.std(dim=0, unbiased=False).clamp_min(1e-8)
    alpha = float(((residual / target_std) ** 2).mean().sqrt())

    # Matched isotropic control: same per-dimension magnitude, no structure.
    generator = torch.Generator().manual_seed(int(args.seed))
    control = (
        torch.randn(residual.shape, generator=generator).to(args.device)
        * target_std
        * alpha
    )

    def _frames(x: torch.Tensor) -> torch.Tensor:
        return term_major_to_frames(x) if framed else x.unsqueeze(1)

    rows = [
        _structure(_frames(residual), "planner residual"),
        _structure(_frames(control), "isotropic control"),
    ]

    print(f"interface            : {spec.interface}  ({spec.target_dim})")
    print(f"rows                 : {tuple(target.shape)}")
    print(f"residual magnitude   : alpha = {alpha:.3f}  (BB3 injected 0.10/0.25/0.50)")
    print()
    hdr = f"{'':20}{'temporal r':>13}{'spatial |r|':>13}{'top-PC energy':>15}"
    print(hdr)
    for r in rows:
        print(
            f"{r['label']:20}{r['temporal_corr_t_t1']:>13.3f}"
            f"{r['spatial_mean_abs_corr']:>13.3f}{r['top_pc_energy_fraction']:>15.4f}"
        )
    print(f"\nisotropic top-PC expectation ~ 1/D = {rows[0]['isotropic_reference_top_pc']:.4f}")
    print(
        "\nHigher temporal/spatial correlation or concentrated top-PC energy means\n"
        "the planner's error is coherent rather than white -- which is the\n"
        "candidate explanation for why planner error at a given alpha costs far\n"
        "more tracking accuracy than BB3's injected noise at the same alpha."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "planner": str(args.planner.resolve()),
                    "samples": str(args.samples.resolve()),
                    "interface": str(spec.interface),
                    "residual_alpha": alpha,
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
