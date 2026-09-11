#!/usr/bin/env python3
"""Scale the cross-motion latent-locality test to reference-only motion windows.

This analysis is deliberately separate from rollout analysis.  It constructs
the exact 10-frame, 380-value ``root_qpos`` expert packet, encodes it with the
frozen skill encoder, and asks whether latent-nearest windows from *other*
motions are also kinematically close.  One clip is sampled per normalized
action family so repeated actors/takes cannot make the result trivially easy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import torch

from rlopt.agent.hl_skill_diffsr import (
    HighLevelSkillDiffSRConfig,
    _encoder_window_steps,
)
from rlopt.agent.hl_skill_encoder import build_skill_encoder
from imitation_experiments.evaluation.analyze_cross_motion_latent_structure import (
    AggregatedPublications,
    _open_array,
    _pca_features,
    _plot_summary,
    analyze_clustering,
    analyze_cross_motion_retrieval,
    load_reference_kinematics,
)


ENCODER_WINDOW_STEPS = 10
ROOT_QPOS_FRAME_WIDTH = 38


def normalized_action_family(motion_name: str) -> str:
    """Remove mirror, take, and actor suffixes from a BONES-SEED clip name."""
    name = re.sub(r"_M$", "", str(motion_name))
    return re.sub(r"_\d{3}_A\d+$", "", name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_arrays_dir", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--motion_count", type=int, default=500)
    parser.add_argument("--windows_per_motion", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--max_kmeans_clusters", type=int, default=12)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1500)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized XYZW quaternions to rotation matrices."""
    values = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    values = values / np.maximum(norm, 1.0e-12)
    x, y, z, w = np.moveaxis(values, -1, 0)
    matrix = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def root_qpos_expert_windows(
    joint_pos: np.ndarray,
    anchor_pos_w: np.ndarray,
    anchor_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Build combo H10 packets in the expert heading-anchor frame.

    Combo's ``robot_heading`` contract removes the anchor's XY origin and yaw
    while preserving absolute height and roll/pitch relative to gravity.  The
    older scale analysis used the full anchor pose, which silently removed
    those quantities and therefore did not feed the combo encoder's input
    distribution.
    """
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    anchor_pos_w = np.asarray(anchor_pos_w, dtype=np.float64)
    anchor_quat_xyzw = np.asarray(anchor_quat_xyzw, dtype=np.float64)
    if joint_pos.ndim != 3 or joint_pos.shape[1:] != (ENCODER_WINDOW_STEPS, 29):
        raise ValueError(f"Expected joint positions [N,10,29], got {joint_pos.shape}.")
    if anchor_pos_w.shape != (joint_pos.shape[0], ENCODER_WINDOW_STEPS, 3):
        raise ValueError("Anchor positions are not aligned [N,10,3].")
    if anchor_quat_xyzw.shape != (joint_pos.shape[0], ENCODER_WINDOW_STEPS, 4):
        raise ValueError("Anchor quaternions are not aligned [N,10,4].")

    rotation = _xyzw_to_matrix(anchor_quat_xyzw)
    # Build a yaw-only frame from the first anchor quaternion.  The origin is
    # [x, y, 0], so the relative position retains absolute height.
    x, y, z, w = np.moveaxis(anchor_quat_xyzw[:, 0], -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    heading = np.zeros_like(rotation[:, 0])
    heading[:, 0, 0] = cosine
    heading[:, 0, 1] = -sine
    heading[:, 1, 0] = sine
    heading[:, 1, 1] = cosine
    heading[:, 2, 2] = 1.0
    center_inverse = np.swapaxes(heading, -1, -2)
    origin = anchor_pos_w[:, :1].copy()
    origin[..., 2] = 0.0
    relative_pos = np.einsum(
        "nij,ntj->nti", center_inverse, anchor_pos_w - origin
    )
    relative_rotation = np.einsum("nij,ntjk->ntik", center_inverse, rotation)
    # Isaac Lab's quat_to_rot6d_flat takes the first two matrix rows.
    relative_rot6d = relative_rotation[..., :2, :].reshape(
        joint_pos.shape[0], ENCODER_WINDOW_STEPS, 6
    )
    frames = np.concatenate((joint_pos, relative_pos, relative_rot6d), axis=-1)
    return frames.astype(np.float32)


def _select_motion_ranks(
    metadata: dict[str, Any], *, motion_count: int, seed: int
) -> list[dict[str, Any]]:
    trajectory_info = metadata["traj_info"]
    ordered = trajectory_info["ordered_traj_list"]
    starts = np.asarray(trajectory_info["start_index"], dtype=np.int64)
    ends = np.asarray(trajectory_info["end_index"], dtype=np.int64)
    families: dict[str, list[int]] = {}
    for rank, entry in enumerate(ordered):
        name = str(entry[1])
        length = int(ends[rank] - starts[rank])
        if name.endswith("_M") or length < 30:
            continue
        families.setdefault(normalized_action_family(name), []).append(rank)
    if motion_count > len(families):
        raise ValueError(
            f"Requested {motion_count} motions but only {len(families)} eligible "
            "non-mirrored action families exist."
        )
    rng = np.random.default_rng(seed)
    family_names = np.asarray(sorted(families), dtype=object)
    selected_families = sorted(
        str(value)
        for value in rng.choice(family_names, size=motion_count, replace=False)
    )
    result: list[dict[str, Any]] = []
    for family in selected_families:
        candidates = families[family]
        rank = int(candidates[int(rng.integers(len(candidates)))])
        result.append(
            {
                "action_family": family,
                "trajectory_rank": rank,
                "motion_name": str(ordered[rank][1]),
                "reference_start": int(starts[rank]),
                "reference_length": int(ends[rank] - starts[rank]),
                "eligible_family_members": len(candidates),
            }
        )
    return result


def _publication_plan(
    selection: Sequence[dict[str, Any]], *, windows_per_motion: int
) -> list[dict[str, Any]]:
    if windows_per_motion < 1:
        raise ValueError("--windows_per_motion must be positive.")
    rows: list[dict[str, Any]] = []
    for motion in selection:
        last_start = int(motion["reference_length"]) - 30
        steps = np.unique(
            np.rint(np.linspace(0, last_start, windows_per_motion)).astype(np.int64)
        )
        if steps.size != windows_per_motion:
            raise ValueError(
                f"{motion['motion_name']} cannot supply {windows_per_motion} unique windows."
            )
        for step in steps:
            rows.append({**motion, "reference_step": int(step)})
    return rows


def _load_encoder(path: Path) -> torch.nn.Module:
    """Restore the checkpoint's configured encoder architecture exactly."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in blob or "skill_encoder_state_dict" not in blob:
        raise ValueError(f"{path} is not an hl_skill_diffsr checkpoint.")
    config = HighLevelSkillDiffSRConfig.from_dict(blob["config"])
    state = blob["skill_encoder_state_dict"]
    first_weight = next(value for key, value in state.items() if key.endswith("0.weight"))
    window_steps = _encoder_window_steps(config)
    state_dim, remainder = divmod(int(first_weight.shape[1]), window_steps + 1)
    if remainder:
        raise ValueError(f"Encoder input width {first_weight.shape[1]} is incompatible with {window_steps} visible steps.")
    encoder = build_skill_encoder(
        state_dim=state_dim,
        window_steps=window_steps,
        z_dim=config.z_dim,
        hidden_dims=config.encoder_hidden_dims,
        spec=config.latent_spec(),
        activation=config.encoder_activation,
        layer_norm=config.encoder_layer_norm,
    )
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    if int(config.horizon_steps) != ENCODER_WINDOW_STEPS:
        raise ValueError(
            f"Expected horizon_steps={ENCODER_WINDOW_STEPS}, got {config.horizon_steps}."
        )
    if str(config.macro_anchor_mode) != "robot_heading":
        raise ValueError(
            "Combo semantic analysis requires macro_anchor_mode='robot_heading', "
            f"got {config.macro_anchor_mode!r}."
        )
    return encoder


def _encode(
    encoder: torch.nn.Module,
    frames: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    encoder = encoder.to(device)
    flat = torch.from_numpy(frames.reshape(frames.shape[0], -1))
    outputs: list[torch.Tensor] = []
    for start in range(0, int(flat.shape[0]), batch_size):
        with torch.inference_mode():
            batch = flat[start : start + batch_size].to(device)
            state = batch[:, :ROOT_QPOS_FRAME_WIDTH]
            future = batch[:, ROOT_QPOS_FRAME_WIDTH:].reshape(
                batch.shape[0], ENCODER_WINDOW_STEPS - 1, ROOT_QPOS_FRAME_WIDTH
            )
            outputs.append(encoder(state, future).cpu())
    return torch.cat(outputs).numpy().astype(np.float32, copy=False)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    if args.motion_count < 2:
        raise ValueError("--motion_count must be at least two.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive.")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap_samples must be at least 100.")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    root = args.reference_arrays_dir.expanduser().resolve()
    sidecar_path = root / "reference_arrays_manifest.json"
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    selection = _select_motion_ranks(
        metadata, motion_count=args.motion_count, seed=args.seed
    )
    plan = _publication_plan(selection, windows_per_motion=args.windows_per_motion)

    qpos = _open_array(root, metadata, "qpos")
    anchor_pos = _open_array(root, metadata, "anchor_pos_w")
    anchor_quat = _open_array(root, metadata, "anchor_quat_w")
    index = (
        np.asarray(
            [int(row["reference_start"]) + int(row["reference_step"]) for row in plan],
            dtype=np.int64,
        )[:, None]
        + np.arange(ENCODER_WINDOW_STEPS, dtype=np.int64)[None, :]
    )
    frames = root_qpos_expert_windows(
        np.asarray(qpos[index, 7:]),
        np.asarray(anchor_pos[index]),
        np.asarray(anchor_quat[index]),
    )
    skill_checkpoint = args.skill_checkpoint.expanduser().resolve()
    latent = _encode(
        _load_encoder(skill_checkpoint),
        frames,
        batch_size=args.batch_size,
        device=_device(args.device),
    )

    publications = AggregatedPublications(
        latent_mean=latent,
        latent_rms_spread=np.zeros(len(plan), dtype=np.float32),
        motion_names=np.asarray([row["motion_name"] for row in plan], dtype=object),
        reference_steps=np.asarray([row["reference_step"] for row in plan]),
        reference_lengths=np.asarray([row["reference_length"] for row in plan]),
        replicate_counts=np.ones(len(plan), dtype=np.int64),
        activities=None,
        phase_labels=None,
        axes={},
    )
    kinematics = load_reference_kinematics(root, publications)
    latent_features, latent_pca = _pca_features(latent)
    raw_features, raw_pca = _pca_features(frames.reshape(frames.shape[0], -1))
    kinematic_features, kinematic_pca = _pca_features(kinematics.descriptor)
    retrieval, neighbor_rows = analyze_cross_motion_retrieval(
        publications,
        latent_features,
        kinematic_features,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    raw_retrieval, raw_neighbor_rows = analyze_cross_motion_retrieval(
        publications,
        raw_features,
        kinematic_features,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    clustering, assignments = analyze_clustering(
        publications,
        latent_features,
        seed=args.seed,
        max_clusters=args.max_kmeans_clusters,
    )
    raw_clustering, raw_assignments = analyze_clustering(
        publications,
        raw_features,
        seed=args.seed,
        max_clusters=args.max_kmeans_clusters,
    )
    plot_clustering = dict(clustering)
    plot_clustering["_hdbscan_labels"] = assignments["hdbscan"].tolist()
    tsne, tsne_trust = _plot_summary(
        output_dir / "reference_scale_summary.png",
        publications,
        latent_features,
        retrieval,
        plot_clustering,
        seed=args.seed,
        perplexity=args.tsne_perplexity,
        iterations=args.tsne_iterations,
    )
    raw_plot_clustering = dict(raw_clustering)
    raw_plot_clustering["_hdbscan_labels"] = raw_assignments["hdbscan"].tolist()
    raw_tsne, raw_tsne_trust = _plot_summary(
        output_dir / "reference_scale_raw_summary.png",
        publications,
        raw_features,
        raw_retrieval,
        raw_plot_clustering,
        seed=args.seed,
        perplexity=args.tsne_perplexity,
        iterations=args.tsne_iterations,
    )

    selection_payload = {
        "schema": "reference_latent_scale_selection_v1",
        "seed": int(args.seed),
        "selection_rule": (
            "uniform random normalized action families; one random non-mirrored "
            "actor/take per family; clips shorter than 30 frames excluded"
        ),
        "motions": selection,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publication_rows: list[dict[str, Any]] = []
    for index_row, row in enumerate(plan):
        publication_rows.append(
            {
                "publication_index": index_row,
                "motion_name": row["motion_name"],
                "action_family": row["action_family"],
                "trajectory_rank": row["trajectory_rank"],
                "reference_step": row["reference_step"],
                "reference_length": row["reference_length"],
                "tsne_1": float(tsne[index_row, 0]),
                "tsne_2": float(tsne[index_row, 1]),
                "raw_tsne_1": float(raw_tsne[index_row, 0]),
                "raw_tsne_2": float(raw_tsne[index_row, 1]),
                **{
                    name: float(kinematics.summary[index_row, summary_index])
                    for summary_index, name in enumerate(kinematics.summary_names)
                },
            }
        )
    _write_csv(output_dir / "publications.csv", publication_rows)
    _write_csv(output_dir / "cross_motion_neighbors.csv", neighbor_rows)
    _write_csv(output_dir / "raw_cross_motion_neighbors.csv", raw_neighbor_rows)
    _write_csv(output_dir / "kmeans_sweep.csv", clustering["kmeans"])
    _write_csv(output_dir / "raw_kmeans_sweep.csv", raw_clustering["kmeans"])
    np.savez_compressed(
        output_dir / "canonical_latents.npz",
        latent=latent,
        raw_window=frames.reshape(frames.shape[0], -1),
        motion_name=publications.motion_names,
        reference_step=publications.reference_steps,
    )

    analysis = {
        "schema": "reference_latent_scale_v1",
        "protocol": {
            "reference_arrays_dir": str(root),
            "reference_arrays_sidecar_sha256": _sha256(sidecar_path),
            "skill_checkpoint": str(skill_checkpoint),
            "skill_checkpoint_sha256": _sha256(skill_checkpoint),
            "encoder_input": (
                "10 frame-interleaved [29 joint qpos, 3 anchor position, 6 anchor "
                "rotation] expert windows, expressed in the first-frame pelvis frame"
            ),
            "scope": (
                "reference-only intrinsic encoder geometry; rollout-conditioned "
                "geometry is measured separately by analyze_cross_motion_latent_structure"
            ),
            "retrieval_exclusion": "all publications from the query motion",
            "seed": int(args.seed),
        },
        "rows": {
            "motions": int(args.motion_count),
            "normalized_action_families": int(args.motion_count),
            "windows_per_motion": int(args.windows_per_motion),
            "publications": len(plan),
        },
        "latent_pca": latent_pca,
        "raw_pca": raw_pca,
        "kinematic_pca": kinematic_pca,
        "retrieval": {"latent_vs_kinematic": retrieval, "raw_vs_kinematic": raw_retrieval},
        "clustering": {"latent": clustering, "raw": raw_clustering},
        "tsne": {
            "role": "visualization only; no retrieval or clustering uses t-SNE",
            "latent_trustworthiness_k10": tsne_trust,
            "raw_trustworthiness_k10": raw_tsne_trust,
        },
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[PASS] {len(plan)} windows from {len(selection)} distinct BONES-SEED "
        f"action families -> {output_dir}"
    )


if __name__ == "__main__":
    main()
