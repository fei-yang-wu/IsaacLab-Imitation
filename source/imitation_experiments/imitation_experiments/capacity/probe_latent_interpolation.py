#!/usr/bin/env python3
"""Measure what a linearly interpolated skill latent means to the grounding.

The linear-closure question: for two latents ``z1``, ``z2`` produced by the
encoder from real motion chunks, does ``z_a = a*z1 + (1-a)*z2`` behave like a
skill? This probe answers the part that lives entirely in the frozen encoder
plus its DiffSR heads, before any tracker exists. Three measurements against
one ``hl_skill_diffsr`` checkpoint:

1. Score affinity: the relative gap between ``phi(s, z_a)`` and
   ``a*phi(s, z1) + (1-a)*phi(s, z2)``. Under
   ``diffsr_phi_parameterization='affine'`` this is zero to floating-point
   precision by construction; under 'concat' or 'bilinear' it is the measured
   nonlinearity of the current model, and the number a closure design has to
   remove.
2. Interpolant geometry: whether ``z_a`` looks like a latent the encoder could
   have emitted -- its norm against the real-z norm distribution, and its
   distance to the nearest real z against the real-z nearest-neighbour
   baseline. A chord that leaves the data manifold is a chord the decoder and
   the policy were never trained on.
3. Denoising transfer: the endpoint head's noise-prediction error for
   ``z_a`` evaluated against each endpoint's OWN true future, at fixed noise
   draws so the alpha sweep is comparable. A lawful blend trades the two
   errors off smoothly; a mixed latent that is meaningless to the grounding
   shows a bump at the interior instead.

Pairs are drawn between DIFFERENT motions, so an interpolation is a genuine
skill blend rather than two nearby frames of one clip.

Windows are rebuilt with the exact expert-sampler math by reusing
``probe_skill_window_usage``: slot-0 heading anchor (yaw-only rotation, xy-only
origin) and the interleaved ``quat_to_rot6d_flat`` 6-D layout.

Run from the repository root:

    pixi run python -m imitation_experiments.capacity.probe_latent_interpolation \\
        --skill_checkpoint <path/to/latest.pt> \\
        --reference_arrays_dir <path/to/ref_arrays> \\
        --output_dir <fresh dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from rlopt.agent.hl_skill_diffsr import (
    HighLevelSkillDiffSRConfig,
    _build_diffsr,
)
from rlopt.agent.ipmd.module import DiffSRBilinear

from imitation_experiments.capacity.probe_skill_window_usage import (
    _sha256,
    encode_windows,
    heading_anchored_expert_windows,
    load_encoder_bundle,
)
from imitation_experiments.evaluation.analyze_cross_motion_latent_structure import (
    _open_array,
)
from imitation_experiments.evaluation.analyze_reference_latent_scale import (
    _publication_plan,
    _select_motion_ranks,
)

DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_endpoint_head(
    path: Path, config: HighLevelSkillDiffSRConfig, state_dim: int, device: torch.device
) -> DiffSRBilinear:
    """Rebuild the endpoint DiffSR head, which scores p(s[t+H] | s_t, z)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "diffsr_state_dict" not in blob:
        raise ValueError(f"{path} carries no diffsr_state_dict.")
    head = _build_diffsr(config, state_dim, torch.device("cpu"))
    if not isinstance(head, DiffSRBilinear):
        raise TypeError(f"Expected a DiffSR head, got {type(head).__name__}.")
    head.load_state_dict(blob["diffsr_state_dict"], strict=True)
    head.to(device)
    head.eval()
    return head


def pair_across_motions(
    trajectory_ranks: np.ndarray, *, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs whose two windows come from different motions."""
    rng = np.random.default_rng(seed)
    total = int(trajectory_ranks.shape[0])
    if total < 2:
        raise ValueError("Need at least two windows to form a pair.")
    left = rng.integers(0, total, size=count * 4)
    right = rng.integers(0, total, size=count * 4)
    keep = trajectory_ranks[left] != trajectory_ranks[right]
    left, right = left[keep][:count], right[keep][:count]
    if left.shape[0] < count:
        raise ValueError(
            f"Only formed {left.shape[0]} cross-motion pairs out of {count}; "
            "increase --motion_count."
        )
    return left, right


def score_affinity(
    head: DiffSRBilinear,
    state: Tensor,
    z_left: Tensor,
    z_right: Tensor,
    alphas: tuple[float, ...],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Relative gap between phi at the mixed latent and the mixed phi."""
    results: dict[str, dict[str, float]] = {}
    state = state.to(device)
    z_left, z_right = z_left.to(device), z_right.to(device)
    with torch.no_grad():
        phi_left = head.forward_phi(state, z_left)
        phi_right = head.forward_phi(state, z_right)
        for alpha in alphas:
            mixed_z = alpha * z_left + (1.0 - alpha) * z_right
            phi_at_mix = head.forward_phi(state, mixed_z)
            mix_of_phi = alpha * phi_left + (1.0 - alpha) * phi_right
            gap = torch.linalg.vector_norm(phi_at_mix - mix_of_phi, dim=-1)
            scale = torch.linalg.vector_norm(mix_of_phi, dim=-1).clamp_min(1e-12)
            relative = gap / scale
            results[f"{alpha:g}"] = {
                "relative_gap_mean": float(relative.mean()),
                "relative_gap_max": float(relative.max()),
                "absolute_gap_mean": float(gap.mean()),
                "phi_norm_mean": float(scale.mean()),
            }
    return results


def interpolant_geometry(
    z_all: Tensor,
    z_left: Tensor,
    z_right: Tensor,
    alphas: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    """Norm and nearest-real-neighbour distance of the mixed latent.

    The real-z nearest-neighbour distance (excluding self) is the baseline: a
    mixed latent that sits no further from the manifold than a real latent
    does is, by this measure, in-distribution.
    """
    real_norm = torch.linalg.vector_norm(z_all, dim=-1)
    # Excluding self: the diagonal of the all-pairs distance matrix is zero.
    real_dist = torch.cdist(z_all, z_all)
    real_dist.fill_diagonal_(float("inf"))
    real_nn = real_dist.min(dim=1).values
    baseline = float(real_nn.mean())

    results: dict[str, dict[str, float]] = {
        "_baseline": {
            "real_z_norm_mean": float(real_norm.mean()),
            "real_z_norm_std": float(real_norm.std()),
            "real_z_nearest_neighbor_mean": baseline,
        }
    }
    for alpha in alphas:
        mixed = alpha * z_left + (1.0 - alpha) * z_right
        norm = torch.linalg.vector_norm(mixed, dim=-1)
        nn_dist = torch.cdist(mixed, z_all).min(dim=1).values
        results[f"{alpha:g}"] = {
            "norm_mean": float(norm.mean()),
            "norm_ratio_to_real": float(norm.mean() / real_norm.mean()),
            "nearest_real_distance_mean": float(nn_dist.mean()),
            "nearest_real_distance_ratio": float(nn_dist.mean() / max(baseline, 1e-12)),
        }
    return results


def denoising_transfer(
    head: DiffSRBilinear,
    state: Tensor,
    endpoint_left: Tensor,
    endpoint_right: Tensor,
    z_left: Tensor,
    z_right: Tensor,
    alphas: tuple[float, ...],
    *,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Endpoint-head noise-prediction error under the mixed latent.

    Noise draws and diffusion times are fixed across the sweep so the alpha
    axis is the only thing that moves.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = state.to(device)
    z_left, z_right = z_left.to(device), z_right.to(device)
    targets = {
        "left": endpoint_left.to(device),
        "right": endpoint_right.to(device),
    }
    batch = state.shape[0]
    noise_index = torch.randint(
        0, int(head.num_noises), (batch,), generator=generator
    ).to(device)
    eps = torch.randn(
        (batch, int(head.next_obs_dim)), generator=generator, dtype=torch.float32
    ).to(device)

    # The VP schedule is a registered buffer; index it directly so the forward
    # noising matches `DiffSRBilinear.add_noise` without its random draw.
    alphabars = head.get_buffer("alphabars")[noise_index]

    results: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        noisy: dict[str, Tensor] = {}
        for side, target in targets.items():
            x0 = head.obs_norm.normalize(target)
            noisy[side] = alphabars.sqrt() * x0 + (1 - alphabars).sqrt() * eps
        for alpha in alphas:
            mixed_z = alpha * z_left + (1.0 - alpha) * z_right
            row: dict[str, float] = {}
            for side in targets:
                eps_pred = head.forward_eps(
                    s=state,
                    a=mixed_z,
                    sp=noisy[side],
                    t=noise_index.unsqueeze(-1).to(eps.dtype),
                )
                row[f"eps_mse_{side}_target"] = float(
                    (eps_pred - eps).pow(2).sum(-1).mean()
                )
            results[f"{alpha:g}"] = row
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--reference_arrays_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--motion_count", type=int, default=400)
    parser.add_argument("--windows_per_motion", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="Interpolation weights on z1; 1.0 is z1 and 0.0 is z2.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if str(args.device).strip().lower() == "auto"
        else args.device
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    checkpoint_path = args.skill_checkpoint.expanduser().resolve()
    encoder, config = load_encoder_bundle(checkpoint_path)
    encoder = encoder.to(device)
    horizon = int(config.horizon_steps)

    root = args.reference_arrays_dir.expanduser().resolve()
    sidecar_path = root / "reference_arrays_manifest.json"
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    selection = _select_motion_ranks(
        metadata, motion_count=args.motion_count, seed=args.seed
    )
    plan = _publication_plan(selection, windows_per_motion=args.windows_per_motion)
    frames_per_window = horizon + 1

    qpos = _open_array(root, metadata, "qpos")
    anchor_pos = _open_array(root, metadata, "anchor_pos_w")
    anchor_quat = _open_array(root, metadata, "anchor_quat_w")
    index = (
        np.asarray(
            [int(row["reference_start"]) + int(row["reference_step"]) for row in plan],
            dtype=np.int64,
        )[:, None]
        + np.arange(frames_per_window, dtype=np.int64)[None, :]
    )
    windows = heading_anchored_expert_windows(
        torch.from_numpy(np.asarray(qpos[index, 7:], dtype=np.float64)),
        torch.from_numpy(np.asarray(anchor_pos[index], dtype=np.float64)),
        torch.from_numpy(np.asarray(anchor_quat[index], dtype=np.float64)),
    )
    trajectory_ranks = np.asarray([row["trajectory_rank"] for row in plan])

    z_all = encode_windows(
        encoder, config, windows, device=device, batch_size=args.batch_size
    )
    state_dim = int(windows.shape[-1])
    head = load_endpoint_head(checkpoint_path, config, state_dim, device)

    left_idx, right_idx = pair_across_motions(
        trajectory_ranks, count=args.pairs, seed=args.seed
    )
    state = windows[left_idx, 0]
    # Each side's own endpoint: the frame the encoder never sees.
    endpoint_left = windows[left_idx, horizon]
    endpoint_right = windows[right_idx, horizon]
    z_left, z_right = z_all[left_idx], z_all[right_idx]
    alphas = tuple(float(a) for a in args.alphas)

    print(
        f"[probe] {windows.shape[0]} windows, {len(left_idx)} cross-motion pairs, "
        f"z_dim {z_all.shape[1]}, phi parameterization "
        f"{config.diffsr_phi_parameterization!r}"
    )

    analysis = {
        "schema": "probe_latent_interpolation_v1",
        "protocol": {
            "skill_checkpoint": str(checkpoint_path),
            "skill_checkpoint_sha256": _sha256(checkpoint_path),
            "reference_arrays_dir": str(root),
            "reference_arrays_sidecar_sha256": _sha256(sidecar_path),
            "diffsr_phi_parameterization": str(config.diffsr_phi_parameterization),
            "encoder_window_mode": str(config.encoder_window_mode),
            "horizon_steps": horizon,
            "windows": int(windows.shape[0]),
            "pairs": int(left_idx.shape[0]),
            "pairing": "cross-motion (the two windows never share a clip)",
            "alphas": list(alphas),
            "interpolation": "straight-line in z; z is not norm-constrained",
            "seed": int(args.seed),
        },
        "score_affinity": score_affinity(
            head, state, z_left, z_right, alphas, device=device
        ),
        "interpolant_geometry": interpolant_geometry(z_all, z_left, z_right, alphas),
        "denoising_transfer": denoising_transfer(
            head,
            state,
            endpoint_left,
            endpoint_right,
            z_left,
            z_right,
            alphas,
            device=device,
            seed=args.seed,
        ),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("score affinity (relative gap between phi at the mix and the mixed phi):")
    for alpha, values in analysis["score_affinity"].items():
        print(
            f"  alpha {alpha:>5s}  mean {values['relative_gap_mean']:.3e}  "
            f"max {values['relative_gap_max']:.3e}"
        )
    print("interpolant geometry (ratios against the real-z baseline):")
    for alpha, values in analysis["interpolant_geometry"].items():
        if alpha == "_baseline":
            continue
        print(
            f"  alpha {alpha:>5s}  norm x{values['norm_ratio_to_real']:.3f}  "
            f"nearest-real x{values['nearest_real_distance_ratio']:.3f}"
        )
    print("denoising transfer (endpoint-head eps MSE, fixed noise):")
    for alpha, values in analysis["denoising_transfer"].items():
        print(
            f"  alpha {alpha:>5s}  left target {values['eps_mse_left_target']:.4f}  "
            f"right target {values['eps_mse_right_target']:.4f}"
        )
    print(f"[probe] wrote {output_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
