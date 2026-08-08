#!/usr/bin/env python3
"""Analyze clustering and temporal structure in collected latent commands.

The analysis is deliberately offline: it reads planner sample shards and does
not launch Isaac, run a policy, or infer new commands.  A balanced subset is
used for PCA/t-SNE and probes so long motions do not dominate the result.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEMANTIC_AXES = (
    "locomoting",
    "manipulating",
    "object_loaded",
    "torso_lowered",
    "turning",
)


@dataclass(frozen=True)
class CollectedLatents:
    """Compact row-aligned fields extracted from large planner sample shards."""

    current: np.ndarray
    future: np.ndarray
    language: np.ndarray
    motion_names: np.ndarray
    env_ids: np.ndarray
    planner_steps: np.ndarray
    reference_steps: np.ndarray
    reference_lengths: np.ndarray
    shard_paths: tuple[Path, ...]

    @property
    def phase(self) -> np.ndarray:
        denominator = np.maximum(self.reference_lengths.astype(np.float64) - 1.0, 1.0)
        return self.reference_steps.astype(np.float64) / denominator


@dataclass(frozen=True)
class SemanticPhaseAssignments:
    """Row-aligned manual semantic annotations for collected latent commands."""

    labels: np.ndarray
    activities: np.ndarray
    phase_indices: np.ndarray
    axes: dict[str, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--phase_annotations",
        type=Path,
        help=(
            "Optional semantic-phase JSON. Boundaries are exact, zero-based, "
            "end-exclusive reference control-step indices."
        ),
    )
    parser.add_argument("--max_points_per_motion", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne_perplexity", type=float, default=40.0)
    parser.add_argument("--tsne_iterations", type=int, default=1500)
    return parser.parse_args()


def _as_numpy(value: torch.Tensor, *, dtype: np.dtype[Any]) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def load_collected_latents(samples_dir: Path) -> CollectedLatents:
    """Stream only the fields needed for analysis from planner sample shards."""
    paths = tuple(sorted(samples_dir.expanduser().resolve().glob("sample_step_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No sample_step_*.pt shards under {samples_dir}.")

    current_parts: list[np.ndarray] = []
    future_parts: list[np.ndarray] = []
    language_parts: list[np.ndarray] = []
    motion_names: list[str] = []
    env_parts: list[np.ndarray] = []
    planner_step_parts: list[np.ndarray] = []
    reference_step_parts: list[np.ndarray] = []
    reference_length_parts: list[np.ndarray] = []
    latent_width: int | None = None
    language_width: int | None = None

    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        current = sample.get("source_h1_latent_target")
        future = sample.get("z_target")
        language = sample.get("language_embedding")
        names = sample.get("motion_name")
        if not isinstance(current, torch.Tensor) or current.ndim != 2:
            raise ValueError(f"{path} has no [N,D] source_h1_latent_target.")
        rows, width = map(int, current.shape)
        if latent_width is None:
            latent_width = width
        if width != latent_width:
            raise ValueError(f"{path} latent width {width} != {latent_width}.")
        if not isinstance(future, torch.Tensor) or future.shape != (rows, 3 * width):
            raise ValueError(f"{path} has no matching three-token z_target.")
        if not isinstance(language, torch.Tensor) or language.ndim != 2:
            raise ValueError(f"{path} has no [N,L] language_embedding.")
        if int(language.shape[0]) != rows:
            raise ValueError(f"{path} language rows are not aligned.")
        if language_width is None:
            language_width = int(language.shape[1])
        if int(language.shape[1]) != language_width:
            raise ValueError(f"{path} language width changed across shards.")
        if not isinstance(names, list) or len(names) != rows:
            raise ValueError(f"{path} motion_name is not row aligned.")

        required_vectors: dict[str, list[np.ndarray]] = {
            "env_id": env_parts,
            "planner_step": planner_step_parts,
            "reference_local_step": reference_step_parts,
            "reference_trajectory_length": reference_length_parts,
        }
        for key, destination in required_vectors.items():
            value = sample.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows,):
                raise ValueError(f"{path} has no row-aligned {key}.")
            destination.append(_as_numpy(value, dtype=np.int64).copy())

        current_parts.append(_as_numpy(current, dtype=np.float32).copy())
        future_parts.append(
            _as_numpy(future, dtype=np.float32).reshape(rows, 3, width).copy()
        )
        language_parts.append(_as_numpy(language, dtype=np.float32).copy())
        motion_names.extend(str(name) for name in names)
        del sample

    return CollectedLatents(
        current=np.concatenate(current_parts),
        future=np.concatenate(future_parts),
        language=np.concatenate(language_parts),
        motion_names=np.asarray(motion_names, dtype=object),
        env_ids=np.concatenate(env_parts),
        planner_steps=np.concatenate(planner_step_parts),
        reference_steps=np.concatenate(reference_step_parts),
        reference_lengths=np.concatenate(reference_length_parts),
        shard_paths=paths,
    )


def _balanced_indices(
    labels: np.ndarray, *, max_points_per_motion: int, seed: int
) -> np.ndarray:
    if max_points_per_motion <= 0:
        raise ValueError("max_points_per_motion must be positive.")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        count = min(int(max_points_per_motion), int(indices.size))
        selected.append(rng.choice(indices, size=count, replace=False))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def _motion_metadata(selection: Path | None) -> dict[str, dict[str, str]]:
    if selection is None:
        return {}
    payload = json.loads(selection.expanduser().resolve().read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    for row in payload.get("motions", []):
        name = str(row.get("motion_name", ""))
        if name:
            result[name] = {
                "category": str(row.get("category", "unknown")),
                "language_goal": str(row.get("language_goal", name)),
                "task_context": str(row.get("task_context", "")),
            }
    return result


def load_semantic_phase_annotations(path: Path) -> dict[str, Any]:
    """Load and validate a complete, gap-free semantic phase annotation file."""
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "semantic_phase_annotations_v1":
        raise ValueError(f"{resolved} is not semantic_phase_annotations_v1.")
    motions = payload.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError(f"{resolved} has no annotated motions.")
    declared_axes = payload.get("semantic_axes", {})
    if declared_axes and not isinstance(declared_axes, dict):
        raise ValueError(f"{resolved} semantic_axes must be an object.")
    semantic_axes = tuple(dict.fromkeys((*SEMANTIC_AXES, *declared_axes)))

    seen: set[str] = set()
    for motion in motions:
        name = str(motion.get("motion_name", ""))
        if not name or name in seen:
            raise ValueError(
                f"{resolved} has a missing or duplicate motion name: {name!r}."
            )
        seen.add(name)
        reference_frames = int(motion.get("reference_frames", 0))
        phases = motion.get("phases")
        if reference_frames <= 0 or not isinstance(phases, list) or not phases:
            raise ValueError(f"{name} has no positive reference length or phases.")
        expected_start = 0
        for index, phase in enumerate(phases):
            start = int(phase.get("start_step", -1))
            end = int(phase.get("end_step", -1))
            if start != expected_start or end <= start or end > reference_frames:
                raise ValueError(
                    f"{name} phase {index} must be contiguous and end-exclusive: "
                    f"expected start {expected_start}, got [{start}, {end})."
                )
            if not str(phase.get("label", "")) or not str(phase.get("activity", "")):
                raise ValueError(f"{name} phase {index} lacks label or activity.")
            semantics = phase.get("semantics")
            if not isinstance(semantics, dict):
                raise ValueError(f"{name} phase {index} lacks semantics.")
            for axis in semantic_axes:
                if not isinstance(semantics.get(axis), bool):
                    raise ValueError(
                        f"{name} phase {index} semantic axis {axis!r} must be boolean."
                    )
            expected_start = end
        if expected_start != reference_frames:
            raise ValueError(
                f"{name} phases end at {expected_start}, not {reference_frames}."
            )
    return payload


def assign_semantic_phases(
    data: CollectedLatents, annotations: dict[str, Any]
) -> SemanticPhaseAssignments:
    """Map every collected row to the phase covering its reference step."""
    by_motion = {
        str(motion["motion_name"]): motion for motion in annotations["motions"]
    }
    rows = int(data.current.shape[0])
    labels = np.empty(rows, dtype=object)
    activities = np.empty(rows, dtype=object)
    phase_indices = np.empty(rows, dtype=np.int64)
    declared_axes = annotations.get("semantic_axes", {})
    semantic_axes = tuple(dict.fromkeys((*SEMANTIC_AXES, *declared_axes)))
    axes = {axis: np.empty(rows, dtype=np.int8) for axis in semantic_axes}

    for name in _ordered_motion_names(data.motion_names):
        if name not in by_motion:
            raise ValueError(
                f"No semantic phase annotations for collected motion {name!r}."
            )
        motion = by_motion[name]
        reference_frames = int(motion["reference_frames"])
        row_indices = np.flatnonzero(data.motion_names == name)
        lengths = np.unique(data.reference_lengths[row_indices])
        if lengths.tolist() != [reference_frames]:
            raise ValueError(
                f"{name} collected reference lengths {lengths.tolist()} do not match "
                f"annotation length {reference_frames}."
            )
        steps = data.reference_steps[row_indices]
        if np.any(steps < 0) or np.any(steps >= reference_frames):
            raise ValueError(f"{name} contains reference steps outside its annotation.")
        phases = motion["phases"]
        ends = np.asarray([int(phase["end_step"]) for phase in phases])
        local_phase = np.searchsorted(ends, steps, side="right")
        for phase_index, phase in enumerate(phases):
            selected = row_indices[local_phase == phase_index]
            labels[selected] = str(phase["label"])
            activities[selected] = str(phase["activity"])
            phase_indices[selected] = phase_index
            for axis in semantic_axes:
                axes[axis][selected] = int(phase["semantics"][axis])
    return SemanticPhaseAssignments(
        labels=labels,
        activities=activities,
        phase_indices=phase_indices,
        axes=axes,
    )


def _ordered_motion_names(names: Sequence[object]) -> list[str]:
    return list(dict.fromkeys(str(name) for name in names))


def _cross_validated_probes(
    latent: np.ndarray,
    labels: np.ndarray,
    phase: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    unique_group_counts = [
        np.unique(groups[labels == label]).size for label in np.unique(labels)
    ]
    folds = min(5, min(unique_group_counts))
    if folds < 2:
        return {
            "folds": 0,
            "motion_linear_probe_accuracy_mean": None,
            "motion_linear_probe_accuracy_std": None,
            "phase_ridge_r2_mean": None,
            "phase_ridge_r2_std": None,
        }
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=int(seed)
    )
    classification_scores: list[float] = []
    phase_scores: list[float] = []
    for train, test in splitter.split(latent, labels, groups):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
        )
        classifier.fit(latent[train], labels[train])
        classification_scores.append(
            accuracy_score(labels[test], classifier.predict(latent[test]))
        )
        phase_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        phase_model.fit(latent[train], phase[train])
        phase_scores.append(r2_score(phase[test], phase_model.predict(latent[test])))
    return {
        "folds": int(folds),
        "motion_linear_probe_accuracy_mean": float(np.mean(classification_scores)),
        "motion_linear_probe_accuracy_std": float(np.std(classification_scores)),
        "phase_ridge_r2_mean": float(np.mean(phase_scores)),
        "phase_ridge_r2_std": float(np.std(phase_scores)),
    }


def _leave_one_motion_out_semantic_probes(
    latent: np.ndarray,
    assignments: SemanticPhaseAssignments,
    motion_labels: np.ndarray,
    motion_names: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    """Test whether binary semantics transfer to an entirely unseen motion."""
    result: dict[str, Any] = {}
    for axis in SEMANTIC_AXES:
        target = assignments.axes[axis].astype(np.int64)
        per_motion: dict[str, float] = {}
        for motion_label, motion_name in enumerate(motion_names):
            test = motion_labels == motion_label
            train = ~test
            if np.unique(target[test]).size < 2 or np.unique(target[train]).size < 2:
                continue
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
            classifier.fit(latent[train], target[train])
            predicted = classifier.predict(latent[test])
            per_motion[motion_name] = float(
                balanced_accuracy_score(target[test], predicted)
            )
        scores = list(per_motion.values())
        result[axis] = {
            "definition": (
                "balanced accuracy after training on nine motions and testing on "
                "one held-out motion; only motions containing both axis values count"
            ),
            "evaluated_motion_count": len(scores),
            "balanced_accuracy_mean": float(np.mean(scores)) if scores else None,
            "balanced_accuracy_std": float(np.std(scores)) if scores else None,
            "per_held_out_motion": per_motion,
        }
    return result


def _distance_structure(
    latent_scaled: np.ndarray,
    labels: np.ndarray,
    env_ids: np.ndarray,
    planner_steps: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    order = np.lexsort((planner_steps, env_ids, labels))
    same_track = (
        (labels[order][1:] == labels[order][:-1])
        & (env_ids[order][1:] == env_ids[order][:-1])
        & (planner_steps[order][1:] == planner_steps[order][:-1] + 1)
    )
    left = order[:-1][same_track]
    right = order[1:][same_track]
    temporal = np.sqrt(
        np.mean((latent_scaled[left] - latent_scaled[right]) ** 2, axis=1)
    )

    rng = np.random.default_rng(seed)
    label_values = np.unique(labels)
    by_label = {label: np.flatnonzero(labels == label) for label in label_values}
    pair_count = min(20000, max(1000, int(labels.size)))
    within_left = rng.integers(0, labels.size, size=pair_count)
    within_right = np.asarray(
        [rng.choice(by_label[labels[index]]) for index in within_left], dtype=np.int64
    )
    between_left = rng.integers(0, labels.size, size=pair_count)
    between_right: list[int] = []
    for index in between_left:
        other_labels = label_values[label_values != labels[index]]
        between_right.append(int(rng.choice(by_label[rng.choice(other_labels)])))
    between_right_array = np.asarray(between_right, dtype=np.int64)

    within = np.sqrt(
        np.mean((latent_scaled[within_left] - latent_scaled[within_right]) ** 2, axis=1)
    )
    between = np.sqrt(
        np.mean(
            (latent_scaled[between_left] - latent_scaled[between_right_array]) ** 2,
            axis=1,
        )
    )
    return {
        "consecutive_pair_count": int(temporal.size),
        "consecutive_normalized_rms_mean": float(np.mean(temporal)),
        "consecutive_normalized_rms_median": float(np.median(temporal)),
        "random_within_motion_normalized_rms_mean": float(np.mean(within)),
        "random_within_motion_normalized_rms_median": float(np.median(within)),
        "random_between_motion_normalized_rms_mean": float(np.mean(between)),
        "random_between_motion_normalized_rms_median": float(np.median(between)),
        "between_to_within_mean_ratio": float(np.mean(between) / np.mean(within)),
        "within_to_consecutive_mean_ratio": float(np.mean(within) / np.mean(temporal)),
    }


def analyze_latents(
    data: CollectedLatents,
    *,
    semantic_assignments: SemanticPhaseAssignments | None = None,
    max_points_per_motion: int,
    seed: int,
    tsne_perplexity: float,
    tsne_iterations: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return metrics and balanced projection rows for collected commands."""
    if tsne_iterations < 250:
        raise ValueError("tsne_iterations must be at least 250.")
    names = _ordered_motion_names(data.motion_names)
    name_to_label = {name: index for index, name in enumerate(names)}
    labels = np.asarray([name_to_label[str(name)] for name in data.motion_names])
    selected = _balanced_indices(
        labels, max_points_per_motion=max_points_per_motion, seed=seed
    )
    sample = data.current[selected]
    sample_labels = labels[selected]
    sample_phase = data.phase[selected]
    group_ids = labels[selected].astype(np.int64) * 1_000_000 + data.env_ids[selected]

    scaler = StandardScaler().fit(sample)
    sample_scaled = scaler.transform(sample)
    components = min(50, sample_scaled.shape[0] - 1, sample_scaled.shape[1])
    pca = PCA(n_components=components, random_state=seed).fit(sample_scaled)
    sample_pca = pca.transform(sample_scaled)
    perplexity = min(float(tsne_perplexity), max(5.0, (sample.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=int(tsne_iterations),
        random_state=int(seed),
    ).fit_transform(sample_pca)

    neighbor_count = min(11, sample.shape[0])
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(sample_pca)
    neighbor_indices = neighbors.kneighbors(return_distance=False)
    neighbor_indices = (
        neighbor_indices[:, 1:] if neighbor_count > 1 else neighbor_indices
    )
    neighbor_purity = float(
        np.mean(sample_labels[neighbor_indices] == sample_labels[:, None])
    )
    silhouette_rows = min(3000, sample.shape[0])
    silhouette = float(
        silhouette_score(
            sample_pca,
            sample_labels,
            sample_size=silhouette_rows,
            random_state=seed,
        )
    )

    centered = sample_scaled - sample_scaled.mean(axis=0, keepdims=True)
    total_ss = float(np.square(centered).sum())
    between_ss = 0.0
    centroids: list[np.ndarray] = []
    language_centroids: list[np.ndarray] = []
    for label in range(len(names)):
        mask = sample_labels == label
        centroid = sample_scaled[mask].mean(axis=0)
        centroids.append(centroid)
        between_ss += float(mask.sum()) * float(np.square(centroid).sum())
        language_centroids.append(data.language[selected[mask]].mean(axis=0))
    centroid_array = np.stack(centroids)
    language_array = np.stack(language_centroids)
    latent_distance = squareform(pdist(centroid_array, metric="cosine"))
    language_distance = squareform(pdist(language_array, metric="cosine"))
    upper = np.triu_indices(len(names), k=1)
    alignment = spearmanr(latent_distance[upper], language_distance[upper])

    full_scaled = scaler.transform(data.current)
    centroid_prediction = np.argmin(
        np.mean((full_scaled[:, None, :] - centroid_array[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    horizon_rows: list[dict[str, float | int]] = []
    for token in range(3):
        token_scaled = scaler.transform(data.future[:, token])
        token_prediction = np.argmin(
            np.mean(
                (token_scaled[:, None, :] - centroid_array[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        displacement = np.sqrt(np.mean((token_scaled - full_scaled) ** 2, axis=1))
        horizon_rows.append(
            {
                "control_step_offset": int(token * 10),
                "nearest_motion_centroid_accuracy": float(
                    accuracy_score(labels, token_prediction)
                ),
                "normalized_rms_from_current_mean": float(np.mean(displacement)),
                "normalized_rms_from_current_median": float(np.median(displacement)),
            }
        )

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dimensions_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    if cumulative[-1] < 0.90:
        dimensions_90 = int(components + 1)
    metrics: dict[str, Any] = {
        "rows": int(data.current.shape[0]),
        "latent_width": int(data.current.shape[1]),
        "motion_count": len(names),
        "balanced_projection_rows": int(selected.size),
        "balanced_rows_per_motion": {
            names[label]: int(np.sum(sample_labels == label))
            for label in range(len(names))
        },
        "pca_explained_variance_first_2": float(
            cumulative[min(1, len(cumulative) - 1)]
        ),
        "pca_dimensions_for_90_percent": dimensions_90,
        "motion_variance_fraction": float(between_ss / total_ss),
        "motion_silhouette": silhouette,
        "ten_neighbor_motion_purity": neighbor_purity,
        "nearest_motion_centroid_accuracy": float(
            accuracy_score(labels, centroid_prediction)
        ),
        "latent_language_pairwise_distance_spearman": float(alignment.statistic),
        "latent_language_pairwise_distance_pvalue": float(alignment.pvalue),
        "probes": _cross_validated_probes(
            sample,
            sample_labels,
            sample_phase,
            group_ids,
            seed=seed,
        ),
        "distance_structure": _distance_structure(
            full_scaled,
            labels,
            data.env_ids,
            data.planner_steps,
            seed=seed,
        ),
        "future_token_structure": horizon_rows,
        "latent_centroid_cosine_distance": latent_distance.tolist(),
        "language_centroid_cosine_distance": language_distance.tolist(),
    }
    if semantic_assignments is not None:
        selected_semantics = SemanticPhaseAssignments(
            labels=semantic_assignments.labels[selected],
            activities=semantic_assignments.activities[selected],
            phase_indices=semantic_assignments.phase_indices[selected],
            axes={
                axis: semantic_assignments.axes[axis][selected]
                for axis in SEMANTIC_AXES
            },
        )
        activity_labels = selected_semantics.activities
        activity_purity = float(
            np.mean(activity_labels[neighbor_indices] == activity_labels[:, None])
        )
        unique_activities, activity_counts = np.unique(
            activity_labels, return_counts=True
        )
        activity_silhouette: float | None = None
        if unique_activities.size > 1 and np.min(activity_counts) >= 2:
            activity_silhouette = float(
                silhouette_score(
                    sample_pca,
                    activity_labels,
                    sample_size=min(3000, sample.shape[0]),
                    random_state=seed,
                )
            )
        metrics["semantic_phase_structure"] = {
            "activity_counts": {
                str(activity): int(count)
                for activity, count in zip(
                    unique_activities, activity_counts, strict=True
                )
            },
            "ten_neighbor_activity_purity": activity_purity,
            "activity_silhouette": activity_silhouette,
            "semantic_axis_positive_fraction": {
                axis: float(np.mean(selected_semantics.axes[axis]))
                for axis in SEMANTIC_AXES
            },
            "leave_one_motion_out_probes": _leave_one_motion_out_semantic_probes(
                sample,
                selected_semantics,
                sample_labels,
                names,
                seed=seed,
            ),
        }
    projections = {
        "selected": selected,
        "labels": sample_labels,
        "phase": sample_phase,
        "pca": sample_pca[:, :2],
        "tsne": tsne,
    }
    if semantic_assignments is not None:
        projections.update(
            {
                "semantic_phase_label": semantic_assignments.labels[selected],
                "semantic_activity": semantic_assignments.activities[selected],
                "semantic_phase_index": semantic_assignments.phase_indices[selected],
                **{
                    f"semantic_{axis}": semantic_assignments.axes[axis][selected]
                    for axis in SEMANTIC_AXES
                },
            }
        )
    return metrics, projections


def _plot_analysis(
    output_path: Path,
    *,
    metrics: dict[str, Any],
    projections: dict[str, np.ndarray],
    motion_names: Sequence[str],
) -> None:
    labels = projections["labels"]
    pca = projections["pca"]
    tsne = projections["tsne"]
    phase = projections["phase"]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(motion_names)))
    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    for label, name in enumerate(motion_names):
        mask = labels == label
        short = name.replace("_", " ")[:42]
        axes[0, 0].scatter(
            pca[mask, 0],
            pca[mask, 1],
            s=7,
            alpha=0.45,
            color=colors[label],
            label=short,
        )
        axes[0, 1].scatter(
            tsne[mask, 0], tsne[mask, 1], s=7, alpha=0.45, color=colors[label]
        )
    axes[0, 0].set_title("PCA: motion identity")
    axes[0, 0].set_xlabel("PC 1")
    axes[0, 0].set_ylabel("PC 2")
    axes[0, 0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    axes[0, 1].set_title("t-SNE: motion identity")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")
    if "semantic_activity" in projections:
        activities = projections["semantic_activity"]
        activity_names = list(dict.fromkeys(str(value) for value in activities))
        activity_colors = plt.get_cmap("Set2")(
            np.linspace(0.0, 1.0, len(activity_names))
        )
        for index, activity in enumerate(activity_names):
            mask = activities == activity
            axes[1, 0].scatter(
                tsne[mask, 0],
                tsne[mask, 1],
                color=activity_colors[index],
                s=7,
                alpha=0.55,
                label=activity.replace("_", " "),
            )
        axes[1, 0].set_title("Same t-SNE: annotated semantic activity")
        axes[1, 0].legend(loc="best", fontsize=8)
    else:
        phase_plot = axes[1, 0].scatter(
            tsne[:, 0], tsne[:, 1], c=phase, cmap="viridis", s=7, alpha=0.55
        )
        axes[1, 0].set_title("Same t-SNE: normalized motion phase")
        figure.colorbar(phase_plot, ax=axes[1, 0], label="motion progress (0 to 1)")
    axes[1, 0].set_xlabel("t-SNE 1")
    axes[1, 0].set_ylabel("t-SNE 2")
    distance = np.asarray(metrics["latent_centroid_cosine_distance"])
    heatmap = axes[1, 1].imshow(distance, cmap="magma", interpolation="nearest")
    axes[1, 1].set_title("Motion-centroid cosine distance")
    axes[1, 1].set_xticks(
        range(len(motion_names)), labels=range(1, len(motion_names) + 1)
    )
    axes[1, 1].set_yticks(
        range(len(motion_names)), labels=range(1, len(motion_names) + 1)
    )
    axes[1, 1].set_xlabel("motion index")
    axes[1, 1].set_ylabel("motion index")
    figure.colorbar(heatmap, ax=axes[1, 1], label="cosine distance")
    figure.suptitle("Collected 256-D latent-command structure", fontsize=16)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _write_embedding_csv(
    path: Path,
    *,
    data: CollectedLatents,
    projections: dict[str, np.ndarray],
    motion_names: Sequence[str],
    metadata: dict[str, dict[str, str]],
) -> None:
    selected = projections["selected"]
    labels = projections["labels"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "motion_index",
                "motion_name",
                "category",
                "language_goal",
                "env_id",
                "planner_step",
                "reference_step",
                "phase",
                "semantic_phase_index",
                "semantic_phase_label",
                "semantic_activity",
                *SEMANTIC_AXES,
                "pca_1",
                "pca_2",
                "tsne_1",
                "tsne_2",
            ]
        )
        for row, source_index in enumerate(selected):
            name = motion_names[int(labels[row])]
            details = metadata.get(name, {})
            writer.writerow(
                [
                    int(labels[row]) + 1,
                    name,
                    details.get("category", "unknown"),
                    details.get("language_goal", name),
                    int(data.env_ids[source_index]),
                    int(data.planner_steps[source_index]),
                    int(data.reference_steps[source_index]),
                    f"{projections['phase'][row]:.6f}",
                    (
                        int(projections["semantic_phase_index"][row])
                        if "semantic_phase_index" in projections
                        else ""
                    ),
                    (
                        str(projections["semantic_phase_label"][row])
                        if "semantic_phase_label" in projections
                        else ""
                    ),
                    (
                        str(projections["semantic_activity"][row])
                        if "semantic_activity" in projections
                        else ""
                    ),
                    *[
                        (
                            int(projections[f"semantic_{axis}"][row])
                            if f"semantic_{axis}" in projections
                            else ""
                        )
                        for axis in SEMANTIC_AXES
                    ],
                    f"{projections['pca'][row, 0]:.6f}",
                    f"{projections['pca'][row, 1]:.6f}",
                    f"{projections['tsne'][row, 0]:.6f}",
                    f"{projections['tsne'][row, 1]:.6f}",
                ]
            )


def main() -> None:
    args = _parse_args()
    samples_dir = args.samples_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_collected_latents(samples_dir)
    motion_names = _ordered_motion_names(data.motion_names)
    metadata = _motion_metadata(args.selection)
    phase_annotations = (
        load_semantic_phase_annotations(args.phase_annotations)
        if args.phase_annotations
        else None
    )
    semantic_assignments = (
        assign_semantic_phases(data, phase_annotations)
        if phase_annotations is not None
        else None
    )
    metrics, projections = analyze_latents(
        data,
        semantic_assignments=semantic_assignments,
        max_points_per_motion=int(args.max_points_per_motion),
        seed=int(args.seed),
        tsne_perplexity=float(args.tsne_perplexity),
        tsne_iterations=int(args.tsne_iterations),
    )
    manifest_path = samples_dir / "materialization_manifest.json"
    payload = {
        "schema": "collected_latent_space_analysis_v2",
        "protocol": {
            "source": "collected planner samples only",
            "samples_dir": str(samples_dir),
            "sample_shards": [str(path) for path in data.shard_paths],
            "materialization_manifest": (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else None
            ),
            "selection": str(args.selection.expanduser().resolve())
            if args.selection
            else None,
            "phase_annotations": (
                str(args.phase_annotations.expanduser().resolve())
                if args.phase_annotations
                else None
            ),
            "semantic_phase_definition": (
                phase_annotations.get("phase_definition")
                if phase_annotations is not None
                else None
            ),
            "seed": int(args.seed),
            "max_points_per_motion": int(args.max_points_per_motion),
            "tsne_perplexity": float(args.tsne_perplexity),
            "tsne_iterations": int(args.tsne_iterations),
            "preprocessing": "balanced by motion; feature standardization; PCA-50 before t-SNE",
        },
        "motion_order": motion_names,
        "motion_metadata": metadata,
        "metrics": metrics,
        "interpretation_limits": [
            "t-SNE preserves local neighborhoods, not global distance or cluster size.",
            "Motion labels and language goals are confounded: there is one reference motion per goal.",
            "Semantic phase boundaries are manual observational labels, not learned change points.",
            "Leave-one-motion-out semantic probes test transfer to unseen motions, but only for held-out motions containing both values of an axis.",
            "The data can demonstrate clustering, temporal continuity, and cross-modal alignment, but not causal compositional generalization.",
        ],
    }
    json_path = output_dir / "analysis.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "embedding.csv"
    _write_embedding_csv(
        csv_path,
        data=data,
        projections=projections,
        motion_names=motion_names,
        metadata=metadata,
    )
    plot_path = output_dir / "latent_space.png"
    _plot_analysis(
        plot_path,
        metrics=metrics,
        projections=projections,
        motion_names=motion_names,
    )
    print(f"[PASS] {json_path}")
    print(f"[PASS] {csv_path}")
    print(f"[PASS] {plot_path}")


if __name__ == "__main__":
    main()
