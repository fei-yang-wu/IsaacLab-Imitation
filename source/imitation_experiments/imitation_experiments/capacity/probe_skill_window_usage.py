#!/usr/bin/env python3
"""Measure how much of the encoder window the skill code actually uses.

The question under test: does the pretrained skill encoder summarize the
intermediate frames of its future window, or does it collapse to a function of
the last visible frames (the endpoint-collapse hypothesis)? Four offline
analyses run against one frozen ``hl_skill_diffsr`` checkpoint:

1. Frame-sufficiency regression: predict z from small frame subsets
   (last visible frame, last two, state+last, the unseen endpoint). High
   held-out R2 from the last frames alone supports collapse.
2. Per-offset probes: predict each window frame from z, from the visible
   boundary pair (state, last visible frame), and from boundary+z. The
   increment of boundary+z over boundary alone is the linear information z
   carries about mid-window frames beyond what smooth motion already implies.
3. Sensitivity: normalized latent response (||dz|| per unit input RMSE) to
   on-manifold mid-frame replacement versus last-frame replacement.
4. Integrated gradients: per-slot attribution of ||z(x) - z(baseline)||^2
   against a batch-permuted counterfactual-window baseline.

Windows are rebuilt with the exact expert-sampler math: slot-0 heading anchor
(yaw-only rotation, xy-only origin), ``subtract_frame_transforms`` pose
composition, and the interleaved ``quat_to_rot6d_flat`` 6-D layout. The
encoder-visible slice comes from the checkpoint's own ``encoder_window_mode``
via the RLOpt helpers, so the probe input matches pretraining bit-for-bit in
convention.

Run from the repository root:

    pixi run python -m imitation_experiments.capacity.probe_skill_window_usage \
        --skill_checkpoint <path/to/latest.pt> \
        --reference_arrays_dir <path/to/ref_arrays> \
        --output_dir <fresh dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from rlopt.agent.hl_skill_diffsr import (
    HighLevelSkillDiffSRConfig,
    _encoder_input_window,
    _encoder_window_steps,
)
from rlopt.agent.hl_skill_encoder import build_skill_encoder

from imitation_experiments.evaluation.analyze_cross_motion_latent_structure import (
    _open_array,
)
from imitation_experiments.evaluation.analyze_reference_latent_scale import (
    _publication_plan,
    _select_motion_ranks,
)

QPOS_WIDTH = 29
FRAME_WIDTH = 38  # qpos(29) + anchor_pos_b(3) + anchor_ori_b rot6d(6)


# -- expert window construction (sampler-faithful math) ----------------------


def _quat_conjugate_xyzw(quat: Tensor) -> Tensor:
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def _quat_mul_xyzw(a: Tensor, b: Tensor) -> Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _quat_rotate_xyzw(quat: Tensor, vec: Tensor) -> Tensor:
    xyz = quat[..., :3]
    w = quat[..., 3:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def _matrix_from_quat_xyzw(quat: Tensor) -> Tensor:
    x, y, z, w = quat.unbind(-1)
    row0 = torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1
    )
    row1 = torch.stack(
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1
    )
    row2 = torch.stack(
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


def _rot6d_flat_interleaved(quat: Tensor) -> Tensor:
    """The data plane's ``quat_to_rot6d_flat``: R[..., :2] flattened row-major."""
    matrix = _matrix_from_quat_xyzw(quat)
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def heading_anchored_expert_windows(
    qpos: Tensor, anchor_pos_w: Tensor, anchor_quat_xyzw: Tensor
) -> Tensor:
    """Build [N, T, 38] expert frames anchored at slot-0's heading frame.

    Replicates the expert sampler: the anchor frame is slot-0's heading twist
    (yaw only, x-y-zeroed quaternion renormalized) with an xy-only origin, and
    every slot's pose is expressed in it with ``subtract_frame_transforms``
    semantics. Absolute height and roll/pitch versus gravity survive.
    """
    if qpos.ndim != 3 or qpos.shape[-1] != QPOS_WIDTH:
        raise ValueError(
            f"Expected joint positions [N,T,{QPOS_WIDTH}], got {qpos.shape}."
        )
    heading = anchor_quat_xyzw[:, 0].clone()
    heading[..., 0] = 0.0
    heading[..., 1] = 0.0
    heading = heading / torch.linalg.vector_norm(
        heading, dim=-1, keepdim=True
    ).clamp_min(1e-9)
    origin = anchor_pos_w[:, 0].clone()
    origin[..., 2] = 0.0
    inv_heading = _quat_conjugate_xyzw(heading)[:, None, :]
    pos_b = _quat_rotate_xyzw(inv_heading, anchor_pos_w - origin[:, None, :])
    quat_b = _quat_mul_xyzw(inv_heading, anchor_quat_xyzw)
    return torch.cat([qpos, pos_b, _rot6d_flat_interleaved(quat_b)], dim=-1).to(
        torch.float32
    )


# -- checkpoint restore ------------------------------------------------------


def load_encoder_bundle(
    path: Path,
) -> tuple[torch.nn.Module, HighLevelSkillDiffSRConfig]:
    """Rebuild the skill encoder exactly as the trainer constructed it."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in blob or "skill_encoder_state_dict" not in blob:
        raise ValueError(f"{path} is not an hl_skill_diffsr checkpoint.")
    config = HighLevelSkillDiffSRConfig.from_dict(blob["config"])
    if str(config.latent_mode) != "deterministic":
        raise ValueError(
            "This probe supports latent_mode='deterministic' only, got "
            f"{config.latent_mode!r}."
        )
    state_dict = blob["skill_encoder_state_dict"]
    first_weight = next(
        value for key, value in state_dict.items() if key.endswith("0.weight")
    )
    window_steps = _encoder_window_steps(config)
    state_dim, remainder = divmod(int(first_weight.shape[1]), window_steps + 1)
    if remainder != 0:
        raise ValueError(
            f"Encoder input width {first_weight.shape[1]} is not divisible by "
            f"window_steps+1 = {window_steps + 1}."
        )
    encoder = build_skill_encoder(
        state_dim=state_dim,
        window_steps=window_steps,
        z_dim=config.z_dim,
        hidden_dims=config.encoder_hidden_dims,
        spec=config.latent_spec(),
        activation=config.encoder_activation,
        layer_norm=config.encoder_layer_norm,
    )
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    return encoder, config


# -- probe machinery ---------------------------------------------------------


def _standardize(train: Tensor, *others: Tensor) -> tuple[Tensor, ...]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-8)
    return tuple((t - mean) / std for t in (train, *others))


def ridge_r2(
    features_train: Tensor,
    targets_train: Tensor,
    features_test: Tensor,
    targets_test: Tensor,
    *,
    lambdas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
) -> float:
    """Held-out variance-weighted R2 of a closed-form ridge fit.

    The regularizer is selected on a 90/10 split of the training rows; the
    reported number never sees the test rows during fitting.
    """
    x_train, x_test = _standardize(features_train, features_test)
    ones = torch.ones(x_train.shape[0], 1, dtype=x_train.dtype)
    x_train = torch.cat([x_train, ones], dim=1)
    x_test = torch.cat(
        [x_test, torch.ones(x_test.shape[0], 1, dtype=x_test.dtype)], dim=1
    )
    n_fit = int(x_train.shape[0] * 0.9)
    xtx_full = x_train.T @ x_train
    xty_full = x_train.T @ targets_train
    xtx_fit = x_train[:n_fit].T @ x_train[:n_fit]
    xty_fit = x_train[:n_fit].T @ targets_train[:n_fit]
    eye = torch.eye(x_train.shape[1], dtype=x_train.dtype)
    best_lambda, best_val = None, float("inf")
    for lam in lambdas:
        weights = torch.linalg.solve(xtx_fit + lam * n_fit * eye, xty_fit)
        val_err = float(
            (x_train[n_fit:] @ weights - targets_train[n_fit:]).pow(2).sum()
        )
        if val_err < best_val:
            best_val, best_lambda = val_err, lam
    weights = torch.linalg.solve(
        xtx_full + float(best_lambda) * x_train.shape[0] * eye, xty_full
    )
    prediction = x_test @ weights
    residual = (prediction - targets_test).pow(2).sum()
    baseline = (targets_test - targets_train.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(1.0 - residual / baseline.clamp_min(1e-12))


def mlp_probe_r2(
    features_train: Tensor,
    targets_train: Tensor,
    features_test: Tensor,
    targets_test: Tensor,
    *,
    device: torch.device,
    hidden: int = 256,
    epochs: int = 300,
    seed: int = 0,
) -> float:
    """Held-out R2 of a small nonlinear probe (2x hidden SiLU MLP)."""
    x_train, x_test = _standardize(features_train, features_test)
    y_train, y_test = _standardize(targets_train, targets_test)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden, y_train.shape[1]),
    ).to(device)
    x_train_d, y_train_d = x_train.to(device), y_train.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x_train_d), y_train_d)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x_test.to(device)).cpu()
    residual = (prediction - y_test).pow(2).sum()
    baseline = y_test.pow(2).sum()
    return float(1.0 - residual / baseline.clamp_min(1e-12))


def group_split(
    trajectory_ranks: np.ndarray, *, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row masks for a by-motion train/test split (no window leaks a motion)."""
    unique = np.unique(trajectory_ranks)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_ranks = set(unique[:n_test].tolist())
    test_mask = np.isin(trajectory_ranks, list(test_ranks))
    return ~test_mask, test_mask


# -- analyses ----------------------------------------------------------------


def encode_windows(
    encoder: torch.nn.Module,
    config: HighLevelSkillDiffSRConfig,
    windows: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """z for [N, horizon+1, 38] windows (slot 0 is the state)."""
    horizon = int(config.horizon_steps)
    state = windows[:, 0]
    visible = _encoder_input_window(config, windows[:, 1 : horizon + 1])
    outputs: list[Tensor] = []
    for start in range(0, windows.shape[0], batch_size):
        chunk_state = state[start : start + batch_size].to(device)
        chunk_window = visible[start : start + batch_size].to(device)
        with torch.no_grad():
            z, _, _ = encoder.encode(chunk_state, chunk_window, deterministic=True)
        outputs.append(z.cpu())
    return torch.cat(outputs)


def frame_sufficiency(
    windows: Tensor,
    z: Tensor,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    visible_steps: int,
    device: torch.device,
    mlp_epochs: int,
) -> dict[str, dict[str, float]]:
    """R2 of z from frame subsets. Slot indexing: 0=state, v=last visible."""
    last = visible_steps  # window slot index of the last visible frame
    feature_sets: dict[str, list[int]] = {
        "last1": [last],
        "last2": [last - 1, last],
        "state_last": [0, last],
        "endpoint_unseen": [last + 1],
        "visible_all": list(range(0, last + 1)),
    }
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    results: dict[str, dict[str, float]] = {}
    for name, slots in feature_sets.items():
        if max(slots) >= windows.shape[1]:
            continue
        features = windows[:, slots, :].reshape(windows.shape[0], -1).double()
        r2_linear = ridge_r2(
            features[train_idx],
            z[train_idx].double(),
            features[test_idx],
            z[test_idx].double(),
        )
        r2_mlp = mlp_probe_r2(
            features[train_idx].float(),
            z[train_idx],
            features[test_idx].float(),
            z[test_idx],
            device=device,
            epochs=mlp_epochs,
        )
        results[name] = {"ridge_r2": r2_linear, "mlp_r2": r2_mlp}
    return results


def per_offset_probes(
    windows: Tensor,
    z: Tensor,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    visible_steps: int,
) -> list[dict[str, float]]:
    """Per window slot: R2 from z, from the visible boundary, and combined."""
    last = visible_steps
    boundary = windows[:, [0, last], :].reshape(windows.shape[0], -1).double()
    z_double = z.double()
    combined = torch.cat([boundary, z_double], dim=1)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    rows: list[dict[str, float]] = []
    for slot in range(1, windows.shape[1]):
        target = windows[:, slot, :].double()
        row = {
            "slot": float(slot),
            "visible": float(slot <= last),
            "r2_from_z": ridge_r2(
                z_double[train_idx],
                target[train_idx],
                z_double[test_idx],
                target[test_idx],
            ),
            "r2_from_boundary": ridge_r2(
                boundary[train_idx],
                target[train_idx],
                boundary[test_idx],
                target[test_idx],
            ),
            "r2_from_boundary_plus_z": ridge_r2(
                combined[train_idx],
                target[train_idx],
                combined[test_idx],
                target[test_idx],
            ),
        }
        row["z_increment_over_boundary"] = (
            row["r2_from_boundary_plus_z"] - row["r2_from_boundary"]
        )
        rows.append(row)
    return rows


def _perturbed_windows(
    windows: Tensor, *, visible_steps: int, seed: int
) -> dict[str, Tensor]:
    """On-manifold and interpolation perturbations of the visible window."""
    last = visible_steps
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(windows.shape[0], generator=generator)
    mid = slice(1, last)  # visible mid slots, boundary excluded

    mid_swap = windows.clone()
    mid_swap[:, mid] = windows[permutation][:, mid]

    last_swap = windows.clone()
    last_swap[:, last] = windows[permutation][:, last]

    mid_interp = windows.clone()
    fractions = torch.arange(1, last, dtype=windows.dtype) / float(last)
    mid_interp[:, mid] = (
        windows[:, 0:1] * (1.0 - fractions)[None, :, None]
        + windows[:, last : last + 1] * fractions[None, :, None]
    )

    variants = {"mid_swap": mid_swap, "last_swap": last_swap, "mid_interp": mid_interp}

    # Noise on the last visible frame, RMSE-matched to mid_swap's input change,
    # so "same input magnitude, different location" is directly comparable.
    reference_rmse = (mid_swap - windows).pow(2).mean().sqrt()
    scale = (
        reference_rmse
        * (float(windows.shape[1] * windows.shape[2]) / float(windows.shape[2])) ** 0.5
    )
    noise = torch.randn(
        windows.shape[0], windows.shape[2], generator=generator, dtype=windows.dtype
    )
    last_noise = windows.clone()
    last_noise[:, last] = windows[:, last] + noise * scale
    variants["last_noise_matched"] = last_noise
    return variants


def sensitivity(
    encoder: torch.nn.Module,
    config: HighLevelSkillDiffSRConfig,
    windows: Tensor,
    z: Tensor,
    *,
    visible_steps: int,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for name, variant in _perturbed_windows(
        windows, visible_steps=visible_steps, seed=seed
    ).items():
        z_variant = encode_windows(
            encoder, config, variant, device=device, batch_size=batch_size
        )
        input_rmse = (variant - windows).pow(2).mean(dim=(1, 2)).sqrt()
        dz = torch.linalg.vector_norm(z_variant - z, dim=1)
        z_norm = torch.linalg.vector_norm(z, dim=1)
        results[name] = {
            "input_rmse": float(input_rmse.mean()),
            "dz_norm_mean": float(dz.mean()),
            "dz_over_z_mean": float((dz / z_norm.clamp_min(1e-9)).mean()),
            "dz_per_input_rmse": float((dz / input_rmse.clamp_min(1e-9)).mean()),
        }
    return results


def integrated_gradients_per_slot(
    encoder: torch.nn.Module,
    config: HighLevelSkillDiffSRConfig,
    windows: Tensor,
    *,
    visible_steps: int,
    device: torch.device,
    steps: int,
    max_windows: int,
    seed: int,
) -> dict[str, Any]:
    """|IG| of ||z(x)-z(b)||^2 per input slot; b = a batch-permuted window.

    The baseline is another window from the batch (a full counterfactual), so
    x - b is nonzero at every slot and the attribution can land anywhere. A
    boundary-preserving baseline would force zero attribution onto the
    boundary slots by construction.
    """
    horizon = int(config.horizon_steps)
    subset = windows[:max_windows]
    generator = torch.Generator().manual_seed(seed)
    baseline = subset[torch.randperm(subset.shape[0], generator=generator)]

    def _encoder_inputs(batch: Tensor) -> tuple[Tensor, Tensor]:
        return batch[:, 0], _encoder_input_window(config, batch[:, 1 : horizon + 1])

    encoder = encoder.to(device)
    x_state, x_window = (t.to(device) for t in _encoder_inputs(subset))
    b_state, b_window = (t.to(device) for t in _encoder_inputs(baseline))
    with torch.no_grad():
        z_baseline, _, _ = encoder.encode(b_state, b_window, deterministic=True)

    state_grad = torch.zeros_like(x_state)
    window_grad = torch.zeros_like(x_window)
    alphas = (torch.arange(steps, dtype=windows.dtype, device=device) + 0.5) / steps
    for alpha in alphas:
        interp_state = (b_state + alpha * (x_state - b_state)).requires_grad_(True)
        interp_window = (b_window + alpha * (x_window - b_window)).requires_grad_(True)
        z_alpha, _, _ = encoder.encode(interp_state, interp_window, deterministic=True)
        objective = (z_alpha - z_baseline).pow(2).sum()
        grad_state, grad_window = torch.autograd.grad(
            objective, (interp_state, interp_window)
        )
        state_grad += grad_state
        window_grad += grad_window
    ig_state = (x_state - b_state) * state_grad / steps
    ig_window = (x_window - b_window) * window_grad / steps

    per_slot = [float(ig_state.abs().sum())] + [
        float(ig_window[:, slot].abs().sum()) for slot in range(ig_window.shape[1])
    ]
    total = sum(per_slot)
    with torch.no_grad():
        z_full, _, _ = encoder.encode(x_state, x_window, deterministic=True)
        completeness_target = float((z_full - z_baseline).pow(2).sum())
    completeness_sum = float(ig_state.sum() + ig_window.sum())
    return {
        "baseline": "batch-permuted counterfactual window",
        "windows_used": int(subset.shape[0]),
        "steps": int(steps),
        # Slot 0 is the state frame; slots 1..W are the visible future frames
        # in window order (slot W = last visible frame).
        "abs_attribution_per_slot": per_slot,
        "share_per_slot": [value / max(total, 1e-12) for value in per_slot],
        "completeness_sum_ig": completeness_sum,
        "completeness_target": completeness_target,
    }


# -- entry point -------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--reference_arrays_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--motion_count", type=int, default=500)
    parser.add_argument("--windows_per_motion", type=int, default=5)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--mlp_epochs", type=int, default=300)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--ig_max_windows", type=int, default=512)
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
    visible_steps = _encoder_window_steps(config)
    if visible_steps >= horizon:
        raise ValueError(
            "encoder_window_mode='full' leaves no hidden endpoint; this probe "
            "targets endpoint-hiding modes ('intermediate', 'suffix<N>')."
        )

    root = args.reference_arrays_dir.expanduser().resolve()
    sidecar_path = root / "reference_arrays_manifest.json"
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    selection = _select_motion_ranks(
        metadata, motion_count=args.motion_count, seed=args.seed
    )
    plan = _publication_plan(selection, windows_per_motion=args.windows_per_motion)
    # Slot 0 is the state; slots 1..horizon are the future window. The last
    # future slot is the endpoint the encoder never sees.
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
    train_mask, test_mask = group_split(
        trajectory_ranks, test_fraction=args.test_fraction, seed=args.seed
    )

    z = encode_windows(
        encoder, config, windows, device=device, batch_size=args.batch_size
    )
    print(
        f"[probe] {windows.shape[0]} windows, horizon {horizon}, visible steps "
        f"{visible_steps}, z_dim {z.shape[1]}, mode {config.encoder_window_mode}"
    )

    sufficiency = frame_sufficiency(
        windows,
        z,
        train_mask,
        test_mask,
        visible_steps=visible_steps,
        device=device,
        mlp_epochs=args.mlp_epochs,
    )
    offsets = per_offset_probes(
        windows, z, train_mask, test_mask, visible_steps=visible_steps
    )
    sens = sensitivity(
        encoder,
        config,
        windows,
        z,
        visible_steps=visible_steps,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    attribution = integrated_gradients_per_slot(
        encoder,
        config,
        windows,
        visible_steps=visible_steps,
        device=device,
        steps=args.ig_steps,
        max_windows=args.ig_max_windows,
        seed=args.seed,
    )

    analysis = {
        "schema": "probe_skill_window_usage_v1",
        "protocol": {
            "skill_checkpoint": str(checkpoint_path),
            "skill_checkpoint_sha256": _sha256(checkpoint_path),
            "reference_arrays_dir": str(root),
            "reference_arrays_sidecar_sha256": _sha256(sidecar_path),
            "encoder_window_mode": str(config.encoder_window_mode),
            "horizon_steps": horizon,
            "visible_steps": visible_steps,
            "windows": int(windows.shape[0]),
            "train_windows": int(train_mask.sum()),
            "test_windows": int(test_mask.sum()),
            "split": "by motion (trajectory rank), no window leakage",
            "seed": int(args.seed),
            "frame_layout": (
                "38-wide [qpos(29), anchor_pos_b(3), rot6d interleaved(6)], "
                "slot-0 heading anchor (yaw-only, xy origin), sampler-faithful"
            ),
        },
        "frame_sufficiency_r2": sufficiency,
        "per_offset_probes": offsets,
        "sensitivity": sens,
        "integrated_gradients": attribution,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("frame-sufficiency R2 (held-out, by-motion split):")
    for name, values in sufficiency.items():
        print(
            f"  {name:16s} ridge {values['ridge_r2']:+.4f}   "
            f"mlp {values['mlp_r2']:+.4f}"
        )
    print("per-offset probes (slot, from z, from boundary, increment):")
    for row in offsets:
        print(
            f"  slot {int(row['slot']):2d} "
            f"{'vis' if row['visible'] else 'END'}  "
            f"z {row['r2_from_z']:+.4f}  boundary {row['r2_from_boundary']:+.4f}  "
            f"increment {row['z_increment_over_boundary']:+.4f}"
        )
    print("sensitivity (dz per unit input RMSE):")
    for name, values in sens.items():
        print(
            f"  {name:18s} input_rmse {values['input_rmse']:.4f}  "
            f"dz {values['dz_norm_mean']:.4f}  "
            f"dz/rmse {values['dz_per_input_rmse']:.2f}"
        )
    shares = attribution["share_per_slot"]
    print("integrated-gradients share per slot (0=state, last=visible boundary):")
    print("  " + "  ".join(f"{share:.3f}" for share in shares))
    print(f"[PASS] analysis written to {output_dir}")


if __name__ == "__main__":
    main()
