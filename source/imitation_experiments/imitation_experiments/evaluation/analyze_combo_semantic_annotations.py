#!/usr/bin/env python3
"""Compare combo latent and raw windows on frozen BONES temporal labels.

This is a reference-only probe. It samples complete native H10 windows from
the existing 30-motion human annotation set, encodes them with combo's frozen
64-D encoder, and applies identical PCA, t-SNE, clustering, retrieval, and
grouped linear-probe procedures to the raw 380-D packet and latent space.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_mutual_info_score,
    balanced_accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from imitation_experiments.evaluation.analyze_reference_latent_scale import (
    ENCODER_WINDOW_STEPS,
    ROOT_QPOS_FRAME_WIDTH,
    _encode,
    _load_encoder,
    _open_array,
    _pca_features,
    _sha256,
    root_qpos_expert_windows,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_arrays_dir", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--phase_annotations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_windows_per_phase", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--tsne_iterations", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sample_rows(
    sidecar: dict[str, Any],
    annotations: dict[str, Any],
    *,
    max_windows: int,
    axis_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    trajectory = sidecar["traj_info"]
    starts = np.asarray(trajectory["start_index"], dtype=np.int64)
    ends = np.asarray(trajectory["end_index"], dtype=np.int64)
    by_rank = {
        int(rank): (int(start), int(end), str(entry[1]))
        for rank, (start, end, entry) in enumerate(
            zip(starts, ends, trajectory["ordered_traj_list"], strict=True)
        )
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for motion in annotations["motions"]:
        rank = int(motion["trajectory_rank"])
        if rank not in by_rank:
            raise ValueError(f"Annotation rank {rank} is absent from the cache.")
        start, end, name = by_rank[rank]
        phases = motion["phases"]
        for phase_index, phase in enumerate(phases):
            lo = max(0, int(phase["start_step"]))
            hi = min(int(phase["end_step"]), end - start)
            candidates = np.arange(lo, max(lo, hi - ENCODER_WINDOW_STEPS + 1), 10)
            if candidates.size > max_windows:
                candidates = np.sort(rng.choice(candidates, max_windows, replace=False))
            for step in candidates.tolist():
                row = {
                        "trajectory_rank": rank,
                        "motion_name": name,
                        "reference_step": int(step),
                        "phase_index": phase_index,
                        "activity": str(phase["activity"]),
                        "phase_label": str(phase["label"]),
                    }
                semantics = phase.get("semantics", {})
                for axis in axis_names:
                    row[f"axis_{axis}"] = int(bool(semantics.get(axis, False)))
                rows.append(row)
    if not rows:
        raise ValueError("No annotated windows were sampled.")
    return rows


def _group_probe(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, seed: int
) -> dict[str, float]:
    classes = np.unique(labels)
    rows: list[dict[str, float]] = []
    for offset in range(5):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=seed + offset
        )
        train, test = next(splitter.split(features, labels, groups))
        if not np.all(np.isin(classes, labels[train])) or not np.all(
            np.isin(classes, labels[test])
        ):
            continue
        scaler = StandardScaler().fit(features[train])
        x_train = scaler.transform(features[train])
        x_test = scaler.transform(features[test])
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed + offset
        )
        model.fit(x_train, labels[train])
        prediction = model.predict(x_test)
        rows.append(
            {
                "train_windows": float(train.size),
                "test_windows": float(test.size),
                "train_groups": float(np.unique(groups[train]).size),
                "test_groups": float(np.unique(groups[test]).size),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[test], prediction)
                ),
                "macro_f1": float(
                    f1_score(labels[test], prediction, labels=classes, average="macro")
                ),
            }
        )
    if not rows:
        raise ValueError("No grouped split contained every semantic class in train/test.")
    return {
        "valid_splits": int(len(rows)),
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        },
    }


def _retrieval(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    distances[groups[:, None] == groups[None, :]] = np.inf
    k = min(10, features.shape[0] - 1)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    agreement = np.mean(labels[neighbors] == labels[:, None], axis=1)
    random = np.asarray(
        [np.mean(labels[groups != group] == labels[i]) for i, group in enumerate(groups)]
    )
    return {
        "k": int(k),
        "agreement": float(np.mean(agreement)),
        "matched_random_agreement": float(np.mean(random)),
        "agreement_improvement": float(np.mean(agreement - random)),
    }


def _cluster(features: np.ndarray, labels: np.ndarray, *, seed: int) -> dict[str, Any]:
    k = max(2, np.unique(labels).size)
    assignments = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(features)
    return {
        "requested_clusters": int(k),
        "semantic_ami": float(adjusted_mutual_info_score(labels, assignments)),
        "silhouette": float(silhouette_score(features, assignments)),
        "seed_stability": float(
            adjusted_mutual_info_score(
                assignments,
                KMeans(n_clusters=k, n_init=20, random_state=seed + 1).fit_predict(features),
            )
        ),
    }


def _temporal_smoothness(features: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compare adjacent, offset-five, within-phase, and activity-boundary steps."""
    adjacent: list[float] = []
    offset_five: list[float] = []
    within_phase: list[float] = []
    boundary: list[float] = []
    groups = sorted({str(row["trajectory_rank"]) for row in rows})
    for group in groups:
        indices = [i for i, row in enumerate(rows) if str(row["trajectory_rank"]) == group]
        indices.sort(key=lambda i: int(rows[i]["reference_step"]))
        if len(indices) < 6:
            continue
        adjacent.extend(
            np.linalg.norm(features[indices[1:]] - features[indices[:-1]], axis=1).tolist()
        )
        offset_five.extend(
            np.linalg.norm(features[indices[5:]] - features[indices[:-5]], axis=1).tolist()
        )
        for left, right in zip(indices[:-1], indices[1:], strict=True):
            distance = float(np.linalg.norm(features[right] - features[left]))
            if rows[left]["activity"] == rows[right]["activity"]:
                within_phase.append(distance)
            else:
                boundary.append(distance)
    if not adjacent or not offset_five:
        raise ValueError("Temporal smoothness requires at least six windows per motion.")
    return {
        "adjacent_count": int(len(adjacent)),
        "offset_five_count": int(len(offset_five)),
        "adjacent_mean": float(np.mean(adjacent)),
        "offset_five_mean": float(np.mean(offset_five)),
        "adjacent_to_offset_five_ratio": float(np.mean(adjacent) / np.mean(offset_five)),
        "within_activity_count": int(len(within_phase)),
        "boundary_count": int(len(boundary)),
        "within_activity_mean": float(np.mean(within_phase)),
        "boundary_mean": float(np.mean(boundary)),
        "boundary_to_within_ratio": float(np.mean(boundary) / np.mean(within_phase)),
    }


def main() -> None:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    output.mkdir(parents=True)
    root = args.reference_arrays_dir.expanduser().resolve()
    sidecar_path = root / "reference_arrays_manifest.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    annotations = json.loads(args.phase_annotations.read_text(encoding="utf-8"))
    axis_names = tuple(str(axis) for axis in annotations.get("semantic_axes", {}))
    rows = _sample_rows(
        sidecar,
        annotations,
        max_windows=args.max_windows_per_phase,
        axis_names=axis_names,
    )
    index = np.asarray(
        [
            int(sidecar["traj_info"]["start_index"][row["trajectory_rank"]])
            + int(row["reference_step"])
            for row in rows
        ],
        dtype=np.int64,
    )[:, None] + np.arange(ENCODER_WINDOW_STEPS)[None, :]
    qpos = _open_array(root, sidecar, "qpos")
    anchor_pos = _open_array(root, sidecar, "anchor_pos_w")
    anchor_quat = _open_array(root, sidecar, "anchor_quat_w")
    frames = root_qpos_expert_windows(
        np.asarray(qpos[index, 7:]), np.asarray(anchor_pos[index]), np.asarray(anchor_quat[index])
    )
    latent = _encode(
        _load_encoder(args.skill_checkpoint),
        frames,
        batch_size=args.batch_size,
        device=_device(args.device),
    )
    torch.manual_seed(args.seed)
    random_encoder = copy.deepcopy(_load_encoder(args.skill_checkpoint))
    for module in random_encoder.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
            module.reset_parameters()
    random_latent = _encode(
        random_encoder,
        frames,
        batch_size=args.batch_size,
        device=_device(args.device),
    )
    raw_pca_features, raw_pca = _pca_features(frames.reshape(len(rows), -1))
    latent_pca_features, latent_pca = _pca_features(latent)
    random_pca_features, random_pca = _pca_features(random_latent)
    labels = np.asarray([row["activity"] for row in rows], dtype=object)
    groups = np.asarray([row["trajectory_rank"] for row in rows], dtype=np.int64)
    axes = {
        axis: np.asarray([int(row[f"axis_{axis}"]) for row in rows], dtype=np.int8)
        for axis in axis_names
    }

    rng = np.random.default_rng(args.seed)
    display_indices = rng.choice(len(rows), min(len(rows), 4000), replace=False)
    display: dict[str, dict[str, float]] = {}
    for name, features in (("raw", raw_pca_features), ("latent", latent_pca_features)):
        tsne = TSNE(
            n_components=2,
            init="pca",
            perplexity=min(30.0, max(5.0, (len(display_indices) - 1) / 3.0)),
            max_iter=args.tsne_iterations,
            random_state=args.seed,
        ).fit_transform(features[display_indices])
        display[name] = {
            "trustworthiness_k10": float(
                trustworthiness(features[display_indices], tsne, n_neighbors=10)
            )
        }
        display[name]["tsne_rows"] = int(display_indices.size)
        np.save(output / f"{name}_tsne.npy", tsne)

    analysis = {
        "schema": "combo_semantic_annotation_analysis_v1",
        "protocol": {
            "reference_arrays_dir": str(root),
            "reference_arrays_sidecar_sha256": _sha256(sidecar_path),
            "skill_checkpoint": str(args.skill_checkpoint.resolve()),
            "skill_checkpoint_sha256": _sha256(args.skill_checkpoint.resolve()),
            "phase_annotations": str(args.phase_annotations.resolve()),
            "sampling": "up to max_windows_per_phase windows, stride 10, complete H10 window",
            "split": "grouped by trajectory rank",
            "raw_space": "standardized exact 380-D combo input followed by PCA-50",
            "latent_space": "standardized 64-D combo z followed by PCA-50",
        },
        "rows": {
            "windows": len(rows),
            "motions": int(np.unique(groups).size),
            "activities": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        },
        "pca": {"raw": raw_pca, "latent": latent_pca, "random": random_pca},
        "retrieval_k10": {
            "raw": _retrieval(raw_pca_features, labels, groups),
            "latent": _retrieval(latent_pca_features, labels, groups),
            "random": _retrieval(random_pca_features, labels, groups),
        },
        "grouped_linear_probe": {
            "raw": _group_probe(raw_pca_features, labels, groups, seed=args.seed),
            "latent": _group_probe(latent_pca_features, labels, groups, seed=args.seed),
            "random": _group_probe(random_pca_features, labels, groups, seed=args.seed),
        },
        "semantic_axis_probes": {},
        "semantic_axis_retrieval_k10": {},
        "clustering": {
            "raw": _cluster(raw_pca_features, labels, seed=args.seed),
            "latent": _cluster(latent_pca_features, labels, seed=args.seed),
            "random": _cluster(random_pca_features, labels, seed=args.seed),
        },
        "temporal_smoothness": {
            "raw": _temporal_smoothness(raw_pca_features, rows),
            "latent": _temporal_smoothness(latent_pca_features, rows),
            "random": _temporal_smoothness(random_pca_features, rows),
        },
        "tsne": display,
    }
    for axis, target in axes.items():
        if np.unique(target).size < 2:
            continue
        axis_result: dict[str, Any] = {"positive_windows": int(target.sum())}
        for name, features in (
            ("raw", raw_pca_features),
            ("latent", latent_pca_features),
            ("random", random_pca_features),
        ):
            try:
                axis_result.setdefault("grouped_linear_probe", {})[name] = _group_probe(
                    features, target, groups, seed=args.seed
                )
            except ValueError as error:
                axis_result.setdefault("grouped_linear_probe", {})[name] = {
                    "valid_splits": 0,
                    "error": str(error),
                }
            axis_result.setdefault("retrieval_k10", {})[name] = _retrieval(
                features, target, groups
            )
        analysis["semantic_axis_probes"][axis] = axis_result["grouped_linear_probe"]
        analysis["semantic_axis_retrieval_k10"][axis] = {
            "positive_windows": axis_result["positive_windows"],
            **axis_result["retrieval_k10"],
        }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / "features.npz",
        raw=raw_pca_features,
        latent=latent_pca_features,
        random=random_pca_features,
        labels=labels,
        groups=groups,
    )
    with (output / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, name in zip(axes, ("raw", "latent"), strict=True):
        points = np.load(output / f"{name}_tsne.npy")
        for label in np.unique(labels[display_indices]):
            mask = labels[display_indices] == label
            axis.scatter(points[mask, 0], points[mask, 1], s=7, alpha=0.55, label=str(label))
        axis.set_title(f"{name} t-SNE")
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
    axes[1].legend(fontsize=7, loc="best")
    figure.savefig(output / "semantic_tsne.png", dpi=160)
    plt.close(figure)
    print(f"[PASS] {len(rows)} annotated windows from {len(np.unique(groups))} motions -> {output}")


if __name__ == "__main__":
    main()
