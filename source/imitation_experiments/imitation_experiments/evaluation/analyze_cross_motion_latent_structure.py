#!/usr/bin/env python3
"""Test whether latent-local motion windows are similar across source motions.

Unlike a conventional t-SNE inspection, this analysis excludes the query's
source motion from every retrieval result.  Repeated randomized rollouts of the
same motion/reference publication are aggregated before fitting distances or
clusters, so they contribute an estimate and a spread rather than pretending
to be independent motion examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from imitation_experiments.evaluation.analyze_collected_latent_space import (
    CollectedLatents,
    SemanticPhaseAssignments,
    assign_semantic_phases,
    load_semantic_phase_annotations,
)


DEFAULT_NEIGHBOR_COUNTS = (1, 5, 10, 20)
KINEMATIC_WINDOW_STEPS = 30
KINEMATIC_TEMPORAL_SAMPLES = 8


@dataclass(frozen=True)
class AggregatedPublications:
    """One row per unique motion/reference publication, averaged over rollouts."""

    latent_mean: np.ndarray
    latent_rms_spread: np.ndarray
    motion_names: np.ndarray
    reference_steps: np.ndarray
    reference_lengths: np.ndarray
    replicate_counts: np.ndarray
    activities: np.ndarray | None
    phase_labels: np.ndarray | None
    axes: dict[str, np.ndarray]


@dataclass(frozen=True)
class ReferenceKinematics:
    """Motion-intrinsic window descriptors and human-readable summary values."""

    descriptor: np.ndarray
    summary_names: tuple[str, ...]
    summary: np.ndarray
    trajectory_ranks: np.ndarray


def select_publications(
    publications: AggregatedPublications, selected: np.ndarray
) -> AggregatedPublications:
    """Return a row subset while preserving every optional semantic field."""
    indices = np.asarray(selected)
    return AggregatedPublications(
        latent_mean=publications.latent_mean[indices],
        latent_rms_spread=publications.latent_rms_spread[indices],
        motion_names=publications.motion_names[indices],
        reference_steps=publications.reference_steps[indices],
        reference_lengths=publications.reference_lengths[indices],
        replicate_counts=publications.replicate_counts[indices],
        activities=publications.activities[indices]
        if publications.activities is not None
        else None,
        phase_labels=publications.phase_labels[indices]
        if publications.phase_labels is not None
        else None,
        axes={axis: values[indices] for axis, values in publications.axes.items()},
    )


def select_collected_rows(
    data: CollectedLatents, selected: np.ndarray
) -> CollectedLatents:
    """Return a row subset of the compact collected-latent table."""
    indices = np.asarray(selected)
    return CollectedLatents(
        current=data.current[indices],
        future=data.future[indices],
        language=data.language[indices],
        motion_names=data.motion_names[indices],
        env_ids=data.env_ids[indices],
        planner_steps=data.planner_steps[indices],
        reference_steps=data.reference_steps[indices],
        reference_lengths=data.reference_lengths[indices],
        shard_paths=data.shard_paths,
    )


def load_current_collected_latents(samples_dir: Path) -> CollectedLatents:
    """Load current oracle latents from raw collection or H3 materialization."""
    paths = tuple(sorted(samples_dir.expanduser().resolve().glob("sample_step_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No sample_step_*.pt shards under {samples_dir}.")
    current_parts: list[np.ndarray] = []
    language_parts: list[np.ndarray] = []
    motion_names: list[str] = []
    vector_parts: dict[str, list[np.ndarray]] = {
        "env_id": [],
        "planner_step": [],
        "reference_local_step": [],
        "reference_trajectory_length": [],
    }
    latent_width: int | None = None
    language_width: int | None = None
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        current = sample.get("source_h1_latent_target")
        if current is None:
            current = sample.get("latent_skill_target")
        language = sample.get("language_embedding")
        names = sample.get("motion_name")
        if not isinstance(current, torch.Tensor) or current.ndim != 2:
            raise ValueError(f"{path} has no [N,D] current oracle latent target.")
        rows, width = map(int, current.shape)
        if latent_width is None:
            latent_width = width
        if width != latent_width:
            raise ValueError(f"{path} latent width {width} != {latent_width}.")
        if not isinstance(language, torch.Tensor) or language.ndim != 2:
            raise ValueError(f"{path} has no [N,L] language embedding.")
        if int(language.shape[0]) != rows:
            raise ValueError(f"{path} language rows are not aligned.")
        if language_width is None:
            language_width = int(language.shape[1])
        if int(language.shape[1]) != language_width:
            raise ValueError(f"{path} language width changed across shards.")
        if not isinstance(names, list) or len(names) != rows:
            raise ValueError(f"{path} motion_name is not row aligned.")
        for key, parts in vector_parts.items():
            value = sample.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows,):
                raise ValueError(f"{path} has no row-aligned {key}.")
            parts.append(value.detach().cpu().numpy().astype(np.int64, copy=True))
        current_parts.append(
            current.detach().cpu().numpy().astype(np.float32, copy=True)
        )
        language_parts.append(
            language.detach().cpu().numpy().astype(np.float32, copy=True)
        )
        motion_names.extend(str(name) for name in names)
        del sample
    current = np.concatenate(current_parts)
    return CollectedLatents(
        current=current,
        future=np.zeros((current.shape[0], 0, current.shape[1]), dtype=np.float32),
        language=np.concatenate(language_parts),
        motion_names=np.asarray(motion_names, dtype=object),
        env_ids=np.concatenate(vector_parts["env_id"]),
        planner_steps=np.concatenate(vector_parts["planner_step"]),
        reference_steps=np.concatenate(vector_parts["reference_local_step"]),
        reference_lengths=np.concatenate(vector_parts["reference_trajectory_length"]),
        shard_paths=paths,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--reference_arrays_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--phase_annotations", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1500)
    parser.add_argument("--max_kmeans_clusters", type=int, default=12)
    parser.add_argument(
        "--exclude_motion_names",
        nargs="*",
        default=(),
        help="Optional source motions excluded before aggregation and retrieval.",
    )
    return parser.parse_args()


def _ordered_unique(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def aggregate_publications(
    data: CollectedLatents,
    assignments: SemanticPhaseAssignments | None = None,
) -> AggregatedPublications:
    """Average randomized replicas sharing a motion and reference step."""
    if assignments is not None and assignments.labels.shape[0] != data.current.shape[0]:
        raise ValueError("Semantic assignments are not row aligned with latent data.")

    keys: dict[tuple[str, int], list[int]] = {}
    for index, (name, step) in enumerate(
        zip(data.motion_names, data.reference_steps, strict=True)
    ):
        keys.setdefault((str(name), int(step)), []).append(index)

    ordered_keys = sorted(
        keys, key=lambda key: (_ordered_unique(data.motion_names).index(key[0]), key[1])
    )
    latent_mean: list[np.ndarray] = []
    latent_spread: list[float] = []
    motion_names: list[str] = []
    reference_steps: list[int] = []
    reference_lengths: list[int] = []
    replicate_counts: list[int] = []
    activities: list[str] = []
    phase_labels: list[str] = []
    assignment_axis_names = tuple(assignments.axes) if assignments is not None else ()
    semantic_axes: dict[str, list[int]] = {axis: [] for axis in assignment_axis_names}

    for name, step in ordered_keys:
        indices = np.asarray(keys[(name, step)], dtype=np.int64)
        values = data.current[indices].astype(np.float64)
        mean = values.mean(axis=0)
        latent_mean.append(mean.astype(np.float32))
        latent_spread.append(float(np.sqrt(np.mean(np.square(values - mean)))))
        motion_names.append(name)
        reference_steps.append(step)
        lengths = np.unique(data.reference_lengths[indices])
        if lengths.size != 1:
            raise ValueError(f"{name}@{step} has inconsistent reference lengths.")
        reference_lengths.append(int(lengths[0]))
        replicate_counts.append(int(indices.size))
        if assignments is not None:
            group_activities = np.unique(assignments.activities[indices])
            group_labels = np.unique(assignments.labels[indices])
            if group_activities.size != 1 or group_labels.size != 1:
                raise ValueError(f"{name}@{step} crosses semantic annotations.")
            activities.append(str(group_activities[0]))
            phase_labels.append(str(group_labels[0]))
            for axis in assignment_axis_names:
                values_for_axis = np.unique(assignments.axes[axis][indices])
                if values_for_axis.size != 1:
                    raise ValueError(f"{name}@{step} changes semantic axis {axis}.")
                semantic_axes[axis].append(int(values_for_axis[0]))

    return AggregatedPublications(
        latent_mean=np.stack(latent_mean),
        latent_rms_spread=np.asarray(latent_spread, dtype=np.float32),
        motion_names=np.asarray(motion_names, dtype=object),
        reference_steps=np.asarray(reference_steps, dtype=np.int64),
        reference_lengths=np.asarray(reference_lengths, dtype=np.int64),
        replicate_counts=np.asarray(replicate_counts, dtype=np.int64),
        activities=np.asarray(activities, dtype=object)
        if assignments is not None
        else None,
        phase_labels=np.asarray(phase_labels, dtype=object)
        if assignments is not None
        else None,
        axes={
            axis: np.asarray(values, dtype=np.int8)
            for axis, values in semantic_axes.items()
        }
        if assignments is not None
        else {},
    )


def _open_array(root: Path, metadata: dict[str, Any], name: str) -> np.memmap:
    spec = metadata["key"]["arrays"].get(name)
    if not isinstance(spec, dict):
        raise ValueError(f"Reference arrays do not contain {name!r}.")
    return np.memmap(
        root / f"{name}.memmap",
        mode="r",
        dtype=np.dtype(spec["dtype"]),
        shape=tuple(int(value) for value in spec["shape"]),
    )


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate_xy(values: np.ndarray, yaw: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    x = result[..., 0].copy()
    y = result[..., 1].copy()
    result[..., 0] = cosine * x - sine * y
    result[..., 1] = sine * x + cosine * y
    return result


def load_reference_kinematics(
    reference_arrays_dir: Path,
    publications: AggregatedPublications,
    *,
    window_steps: int = KINEMATIC_WINDOW_STEPS,
    temporal_samples: int = KINEMATIC_TEMPORAL_SAMPLES,
) -> ReferenceKinematics:
    """Build translation/yaw-normalized reference-window descriptors."""
    root = reference_arrays_dir.expanduser().resolve()
    sidecar_path = root / "reference_arrays_manifest.json"
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    trajectory_info = metadata["traj_info"]
    ordered = trajectory_info["ordered_traj_list"]
    starts = np.asarray(trajectory_info["start_index"], dtype=np.int64)
    ends = np.asarray(trajectory_info["end_index"], dtype=np.int64)
    exact_ranks: dict[str, list[int]] = {}
    for rank, entry in enumerate(ordered):
        exact_ranks.setdefault(str(entry[1]), []).append(rank)

    body_names = [str(name) for name in metadata["key"]["body_names"]]
    required_bodies = (
        "pelvis",
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    missing = [name for name in required_bodies if name not in body_names]
    if missing:
        raise ValueError(f"Reference arrays lack required bodies: {missing}.")
    body_index = {name: body_names.index(name) for name in required_bodies}

    qpos = _open_array(root, metadata, "qpos")
    qvel = _open_array(root, metadata, "qvel")
    body_pos = _open_array(root, metadata, "body_pos_w")
    body_lin_vel = _open_array(root, metadata, "body_lin_vel_w")
    body_ang_vel = _open_array(root, metadata, "body_ang_vel_w")
    sample_steps = np.rint(np.linspace(0, window_steps - 1, temporal_samples)).astype(
        np.int64
    )

    descriptors: list[np.ndarray] = []
    summaries: list[list[float]] = []
    ranks: list[int] = []
    for name, local_step, expected_length in zip(
        publications.motion_names,
        publications.reference_steps,
        publications.reference_lengths,
        strict=True,
    ):
        candidates = exact_ranks.get(str(name), [])
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one exact reference-array rank for {name!r}, got {candidates}."
            )
        rank = int(candidates[0])
        length = int(ends[rank] - starts[rank])
        if length != int(expected_length):
            raise ValueError(
                f"{name} reference-array length {length} != collected {expected_length}."
            )
        if int(local_step) < 0 or int(local_step) + window_steps > length:
            raise ValueError(
                f"{name}@{local_step} cannot provide a {window_steps}-step window."
            )
        indices = int(starts[rank] + local_step) + np.arange(window_steps)
        window_qpos = np.asarray(qpos[indices], dtype=np.float64)
        window_qvel = np.asarray(qvel[indices], dtype=np.float64)
        window_body_pos = np.asarray(body_pos[indices], dtype=np.float64)
        window_body_lin = np.asarray(body_lin_vel[indices], dtype=np.float64)
        window_body_ang = np.asarray(body_ang_vel[indices], dtype=np.float64)

        yaw = _yaw_from_wxyz(window_qpos[0, 3:7])
        pelvis = body_index["pelvis"]
        pelvis_path = _rotate_xy(
            window_body_pos - window_body_pos[0:1, pelvis : pelvis + 1], -yaw
        )[:, pelvis]
        body_relative = _rotate_xy(
            window_body_pos - window_body_pos[:, pelvis : pelvis + 1], -yaw
        )
        body_linear = _rotate_xy(window_body_lin, -yaw)
        body_angular = _rotate_xy(window_body_ang, -yaw)
        selected = sample_steps
        descriptor = np.concatenate(
            (
                window_qpos[selected, 7:].reshape(-1),
                window_qvel[selected, 6:].reshape(-1),
                pelvis_path[selected].reshape(-1),
                body_relative[selected].reshape(-1),
                body_linear[selected].reshape(-1),
                body_angular[selected].reshape(-1),
            )
        )
        descriptors.append(descriptor.astype(np.float32))

        torso_vector = (
            window_body_pos[:, body_index["torso_link"]] - window_body_pos[:, pelvis]
        )
        torso_norm = np.maximum(np.linalg.norm(torso_vector, axis=1), 1.0e-8)
        torso_tilt = np.arccos(np.clip(torso_vector[:, 2] / torso_norm, -1.0, 1.0))
        planar_speed = np.linalg.norm(window_body_lin[:, pelvis, :2], axis=1)
        joint_speed = np.sqrt(np.mean(np.square(window_qvel[:, 6:]), axis=1))
        summaries.append(
            [
                float(planar_speed.mean()),
                float(np.abs(window_body_ang[:, pelvis, 2]).mean()),
                float(window_body_pos[:, pelvis, 2].mean()),
                float(torso_tilt.mean()),
                float(joint_speed.mean()),
                float(
                    np.linalg.norm(
                        window_body_lin[:, body_index["left_wrist_yaw_link"]], axis=1
                    ).mean()
                ),
                float(
                    np.linalg.norm(
                        window_body_lin[:, body_index["right_wrist_yaw_link"]], axis=1
                    ).mean()
                ),
                float(
                    0.5
                    * (
                        np.linalg.norm(
                            window_body_lin[:, body_index["left_ankle_roll_link"]],
                            axis=1,
                        ).mean()
                        + np.linalg.norm(
                            window_body_lin[:, body_index["right_ankle_roll_link"]],
                            axis=1,
                        ).mean()
                    )
                ),
            ]
        )
        ranks.append(rank)

    return ReferenceKinematics(
        descriptor=np.stack(descriptors),
        summary_names=(
            "planar_speed_m_s",
            "absolute_yaw_rate_rad_s",
            "pelvis_height_m",
            "torso_tilt_rad",
            "joint_speed_rms_rad_s",
            "left_wrist_speed_m_s",
            "right_wrist_speed_m_s",
            "mean_ankle_speed_m_s",
        ),
        summary=np.asarray(summaries, dtype=np.float32),
        trajectory_ranks=np.asarray(ranks, dtype=np.int64),
    )


def _pca_features(
    values: np.ndarray, *, max_components: int = 50
) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = StandardScaler().fit_transform(values)
    components = min(max_components, scaled.shape[0] - 1, scaled.shape[1])
    if components < 1:
        raise ValueError("At least two samples are required for PCA.")
    pca = PCA(n_components=components, random_state=0).fit(scaled)
    transformed = pca.transform(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    dimensions_90 = int(np.searchsorted(cumulative, 0.9) + 1)
    if cumulative[-1] < 0.9:
        dimensions_90 = components + 1
    return transformed, {
        "components": int(components),
        "dimensions_for_90_percent": int(dimensions_90),
        "explained_variance_first_2": float(cumulative[min(1, components - 1)]),
    }


def _cross_motion_neighbor_indices(
    distances: np.ndarray, motion_names: np.ndarray, max_neighbors: int
) -> np.ndarray:
    masked = np.asarray(distances, dtype=np.float64).copy()
    masked[motion_names[:, None] == motion_names[None, :]] = np.inf
    available = int(np.min(np.sum(np.isfinite(masked), axis=1)))
    if available < 1:
        raise ValueError("Cross-motion retrieval requires at least two motions.")
    count = min(int(max_neighbors), available)
    partition = np.argpartition(masked, kth=count - 1, axis=1)[:, :count]
    selected_distances = np.take_along_axis(masked, partition, axis=1)
    order = np.argsort(selected_distances, axis=1)
    return np.take_along_axis(partition, order, axis=1)


def _bootstrap_motion_mean_ci(
    values: np.ndarray,
    motion_names: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    names = np.asarray(_ordered_unique(motion_names), dtype=object)
    per_motion = np.asarray(
        [float(np.mean(values[motion_names == name])) for name in names],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_motion, size=(samples, per_motion.size), replace=True).mean(
        axis=1
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def analyze_cross_motion_retrieval(
    publications: AggregatedPublications,
    latent_features: np.ndarray,
    kinematic_features: np.ndarray,
    *,
    neighbor_counts: Sequence[int] = DEFAULT_NEIGHBOR_COUNTS,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure semantic and kinematic agreement after excluding source motion."""
    latent_distance = pairwise_distances(latent_features, metric="euclidean")
    kinematic_distance = pairwise_distances(kinematic_features, metric="euclidean")
    max_neighbors = max(int(value) for value in neighbor_counts)
    neighbor_indices = _cross_motion_neighbor_indices(
        latent_distance, publications.motion_names, max_neighbors
    )
    same_motion = (
        publications.motion_names[:, None] == publications.motion_names[None, :]
    )
    cross_motion_mask = ~same_motion
    np.fill_diagonal(cross_motion_mask, False)

    rows: list[dict[str, Any]] = []
    for query in range(neighbor_indices.shape[0]):
        for rank, neighbor in enumerate(neighbor_indices[query], start=1):
            row: dict[str, Any] = {
                "query_index": query,
                "query_motion": str(publications.motion_names[query]),
                "query_reference_step": int(publications.reference_steps[query]),
                "neighbor_rank": rank,
                "neighbor_index": int(neighbor),
                "neighbor_motion": str(publications.motion_names[neighbor]),
                "neighbor_reference_step": int(publications.reference_steps[neighbor]),
                "latent_distance": float(latent_distance[query, neighbor]),
                "kinematic_distance": float(kinematic_distance[query, neighbor]),
            }
            if publications.activities is not None:
                row["query_activity"] = str(publications.activities[query])
                row["neighbor_activity"] = str(publications.activities[neighbor])
                row["activity_match"] = int(
                    publications.activities[query] == publications.activities[neighbor]
                )
                for axis in publications.axes:
                    row[f"{axis}_query"] = int(publications.axes[axis][query])
                    row[f"{axis}_neighbor"] = int(publications.axes[axis][neighbor])
            rows.append(row)

    metrics: dict[str, Any] = {
        "definition": (
            "For each unique motion/reference publication, retrieve latent-nearest "
            "publications after excluding every publication from the query motion."
        ),
        "neighbor_counts": [int(value) for value in neighbor_counts],
        "kinematic": {},
    }
    cross_pair_latent = latent_distance[cross_motion_mask]
    cross_pair_kinematic = kinematic_distance[cross_motion_mask]
    alignment = spearmanr(cross_pair_latent, cross_pair_kinematic)
    metrics["latent_kinematic_cross_motion_spearman"] = {
        "rho": float(alignment.statistic),
        "pvalue": float(alignment.pvalue),
        "directed_pair_count": int(cross_pair_latent.size),
    }

    for count in neighbor_counts:
        count = min(int(count), neighbor_indices.shape[1])
        selected = neighbor_indices[:, :count]
        retrieved_distance = np.take_along_axis(
            kinematic_distance, selected, axis=1
        ).mean(axis=1)
        random_distance = np.asarray(
            [
                float(np.mean(kinematic_distance[index, cross_motion_mask[index]]))
                for index in range(len(selected))
            ]
        )
        ratio = retrieved_distance / np.maximum(random_distance, 1.0e-12)
        low, high = _bootstrap_motion_mean_ci(
            ratio,
            publications.motion_names,
            seed=seed + count,
            samples=bootstrap_samples,
        )
        kinematic_neighbors = _cross_motion_neighbor_indices(
            kinematic_distance, publications.motion_names, count
        )[:, :count]
        overlap = np.asarray(
            [
                len(
                    set(selected[index].tolist())
                    & set(kinematic_neighbors[index].tolist())
                )
                / count
                for index in range(selected.shape[0])
            ]
        )
        expected_overlap = np.asarray(
            [
                count / float(np.sum(cross_motion_mask[index]))
                for index in range(selected.shape[0])
            ]
        )
        metrics["kinematic"][str(count)] = {
            "retrieved_to_random_distance_ratio_mean": float(np.mean(ratio)),
            "retrieved_to_random_distance_ratio_motion_bootstrap_95ci": [low, high],
            "kinematic_neighbor_recall": float(np.mean(overlap)),
            "random_expected_neighbor_recall": float(np.mean(expected_overlap)),
            "neighbor_recall_lift": float(np.mean(overlap) / np.mean(expected_overlap)),
        }

    if publications.activities is not None:
        semantic: dict[str, Any] = {}
        targets: dict[str, np.ndarray] = {"activity": publications.activities}
        targets.update(publications.axes)
        for target_name, target in targets.items():
            target_metrics: dict[str, Any] = {}
            for count in neighbor_counts:
                count = min(int(count), neighbor_indices.shape[1])
                selected = neighbor_indices[:, :count]
                agreement = np.mean(target[selected] == target[:, None], axis=1)
                random_agreement = np.asarray(
                    [
                        float(
                            np.mean(target[cross_motion_mask[index]] == target[index])
                        )
                        for index in range(target.size)
                    ]
                )
                difference = agreement - random_agreement
                low, high = _bootstrap_motion_mean_ci(
                    difference,
                    publications.motion_names,
                    seed=seed + 100 + count,
                    samples=bootstrap_samples,
                )
                row: dict[str, Any] = {
                    "agreement": float(np.mean(agreement)),
                    "matched_random_agreement": float(np.mean(random_agreement)),
                    "agreement_improvement": float(np.mean(difference)),
                    "agreement_improvement_motion_bootstrap_95ci": [low, high],
                }
                if target_name != "activity":
                    positive = target.astype(bool)
                    if np.any(positive):
                        positive_precision = np.mean(target[selected[positive]], axis=1)
                        random_positive = np.asarray(
                            [
                                float(np.mean(target[cross_motion_mask[index]]))
                                for index in np.flatnonzero(positive)
                            ]
                        )
                        row.update(
                            {
                                "positive_query_count": int(np.sum(positive)),
                                "positive_query_precision": float(
                                    np.mean(positive_precision)
                                ),
                                "positive_query_matched_random_rate": float(
                                    np.mean(random_positive)
                                ),
                                "positive_query_precision_improvement": float(
                                    np.mean(positive_precision - random_positive)
                                ),
                            }
                        )
                target_metrics[str(count)] = row
            semantic[target_name] = target_metrics
        metrics["semantic"] = semantic
    return metrics, rows


def _cluster_label_metrics(
    cluster_labels: np.ndarray,
    publications: AggregatedPublications,
    latent_features: np.ndarray,
) -> dict[str, Any]:
    retained = cluster_labels >= 0
    labels = cluster_labels[retained]
    result: dict[str, Any] = {
        "cluster_count": int(np.unique(labels).size),
        "noise_fraction": float(1.0 - np.mean(retained)),
    }
    if np.unique(labels).size < 2:
        result["silhouette"] = None
    else:
        result["silhouette"] = float(
            silhouette_score(latent_features[retained], labels)
        )
    if not np.any(retained):
        return result
    motions = publications.motion_names[retained]
    result["motion_adjusted_mutual_information"] = float(
        adjusted_mutual_info_score(motions, labels)
    )
    cluster_motion_counts = [
        np.unique(motions[labels == label]).size for label in np.unique(labels)
    ]
    result["clusters_with_multiple_motions_fraction"] = float(
        np.mean(np.asarray(cluster_motion_counts) >= 2)
    )
    result["mean_motions_per_cluster"] = float(np.mean(cluster_motion_counts))
    if publications.activities is not None:
        activities = publications.activities[retained]
        result["activity_adjusted_mutual_information"] = float(
            adjusted_mutual_info_score(activities, labels)
        )
        result["semantic_axis_adjusted_mutual_information"] = {
            axis: float(
                adjusted_mutual_info_score(publications.axes[axis][retained], labels)
            )
            for axis in publications.axes
        }
    return result


def analyze_clustering(
    publications: AggregatedPublications,
    latent_features: np.ndarray,
    *,
    seed: int,
    max_clusters: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run clustering in latent PCA space and quantify semantics versus identity."""
    upper = min(int(max_clusters), max(2, latent_features.shape[0] - 1))
    kmeans_rows: list[dict[str, Any]] = []
    assignments: dict[str, np.ndarray] = {}
    for clusters in range(2, upper + 1):
        base = KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(
            latent_features
        )
        stability_scores: list[float] = []
        for offset in range(1, 6):
            alternative = KMeans(
                n_clusters=clusters, n_init=10, random_state=seed + offset
            ).fit_predict(latent_features)
            stability_scores.append(adjusted_rand_score(base, alternative))
        row = {
            "requested_clusters": clusters,
            "assignment_stability_adjusted_rand_mean": float(np.mean(stability_scores)),
            **_cluster_label_metrics(base, publications, latent_features),
        }
        kmeans_rows.append(row)
        assignments[f"kmeans_{clusters}"] = base

    min_cluster_size = max(5, min(30, latent_features.shape[0] // 40))
    hdbscan_labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(3, min_cluster_size // 2),
        copy=True,
    ).fit_predict(latent_features)
    assignments["hdbscan"] = hdbscan_labels
    hdbscan = {
        "definition": (
            "HDBSCAN identifies dense latent regions without choosing a cluster "
            "count and labels insufficiently dense publications as noise."
        ),
        "min_cluster_size": min_cluster_size,
        **_cluster_label_metrics(hdbscan_labels, publications, latent_features),
    }
    return {
        "space": "standardized latent followed by PCA-50; never t-SNE coordinates",
        "kmeans": kmeans_rows,
        "hdbscan": hdbscan,
    }, assignments


def _selection_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("motions", payload.get("candidates", []))
    return {
        str(row["motion_name"]): {
            "category": str(row.get("category", "unknown")),
            "language_goal": str(row.get("language_goal", row["motion_name"])),
        }
        for row in rows
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    path: Path,
    publications: AggregatedPublications,
    latent_features: np.ndarray,
    retrieval: dict[str, Any],
    clustering: dict[str, Any],
    *,
    seed: int,
    perplexity: float,
    iterations: int,
) -> tuple[np.ndarray, float]:
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (latent_features.shape[0] - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        max_iter=iterations,
        random_state=seed,
    ).fit_transform(latent_features)
    trust = float(
        trustworthiness(
            latent_features,
            tsne,
            n_neighbors=min(10, max(1, latent_features.shape[0] // 3 - 1)),
        )
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    motion_names = _ordered_unique(publications.motion_names)
    motion_labels = np.asarray(
        [motion_names.index(str(name)) for name in publications.motion_names]
    )
    axes[0, 0].scatter(
        tsne[:, 0], tsne[:, 1], c=motion_labels, s=16, alpha=0.75, cmap="tab20"
    )
    axes[0, 0].set_title(
        f"Unique publications: motion identity (trustworthiness={trust:.3f})"
    )
    axes[0, 0].set_xlabel("t-SNE 1")
    axes[0, 0].set_ylabel("t-SNE 2")

    if publications.activities is not None:
        activities = _ordered_unique(publications.activities)
        activity_labels = np.asarray(
            [activities.index(str(value)) for value in publications.activities]
        )
        scatter = axes[0, 1].scatter(
            tsne[:, 0], tsne[:, 1], c=activity_labels, s=16, alpha=0.75, cmap="tab10"
        )
        handles, _ = scatter.legend_elements()
        axes[0, 1].legend(handles, activities, fontsize=8, loc="best")
        axes[0, 1].set_title("Same t-SNE: semantic activity")
    else:
        hdbscan = np.asarray(clustering["_hdbscan_labels"])
        axes[0, 1].scatter(
            tsne[:, 0], tsne[:, 1], c=hdbscan, s=16, alpha=0.75, cmap="tab20"
        )
        axes[0, 1].set_title("Same t-SNE: HDBSCAN assignment")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")

    counts = [str(value) for value in retrieval["neighbor_counts"]]
    ratios = [
        retrieval["kinematic"][count]["retrieved_to_random_distance_ratio_mean"]
        for count in counts
    ]
    low = [
        retrieval["kinematic"][count][
            "retrieved_to_random_distance_ratio_motion_bootstrap_95ci"
        ][0]
        for count in counts
    ]
    high = [
        retrieval["kinematic"][count][
            "retrieved_to_random_distance_ratio_motion_bootstrap_95ci"
        ][1]
        for count in counts
    ]
    x = np.arange(len(counts))
    axes[1, 0].errorbar(
        x,
        ratios,
        yerr=[
            np.asarray(ratios) - np.asarray(low),
            np.asarray(high) - np.asarray(ratios),
        ],
        marker="o",
        capsize=4,
    )
    axes[1, 0].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1, 0].set_xticks(x, counts)
    axes[1, 0].set_xlabel("Cross-motion latent neighbors k")
    axes[1, 0].set_ylabel("Kinematic distance / random distance")
    axes[1, 0].set_title("Do latent neighbors move alike? (lower is better)")

    kmeans = clustering["kmeans"]
    cluster_counts = [int(row["requested_clusters"]) for row in kmeans]
    motion_ami = [float(row["motion_adjusted_mutual_information"]) for row in kmeans]
    axes[1, 1].plot(cluster_counts, motion_ami, marker="o", label="motion identity")
    if publications.activities is not None:
        activity_ami = [
            float(row["activity_adjusted_mutual_information"]) for row in kmeans
        ]
        axes[1, 1].plot(
            cluster_counts, activity_ami, marker="o", label="semantic activity"
        )
    axes[1, 1].set_xlabel("K-means cluster count")
    axes[1, 1].set_ylabel("Adjusted mutual information")
    axes[1, 1].set_title("What do unsupervised clusters align with?")
    axes[1, 1].legend()
    figure.suptitle("Cross-motion latent-command structure", fontsize=16)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return tsne, trust


def main() -> None:
    args = _parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap_samples must be at least 100.")
    if args.tsne_iterations < 250:
        raise ValueError("--tsne_iterations must be at least 250.")
    data = load_current_collected_latents(args.samples_dir)
    excluded_motion_names = tuple(str(name) for name in args.exclude_motion_names)
    if excluded_motion_names:
        present = set(str(name) for name in data.motion_names)
        unknown = sorted(set(excluded_motion_names) - present)
        if unknown:
            raise ValueError(f"Cannot exclude absent motion names: {unknown}.")
        data = select_collected_rows(
            data,
            ~np.isin(
                data.motion_names, np.asarray(excluded_motion_names, dtype=object)
            ),
        )
    assignments = None
    if args.phase_annotations is not None:
        assignments = assign_semantic_phases(
            data, load_semantic_phase_annotations(args.phase_annotations)
        )
    all_publications = aggregate_publications(data, assignments)
    full_window = (
        all_publications.reference_steps + KINEMATIC_WINDOW_STEPS
        <= all_publications.reference_lengths
    )
    publications = select_publications(all_publications, full_window)
    kinematics = load_reference_kinematics(args.reference_arrays_dir, publications)
    latent_features, latent_pca = _pca_features(publications.latent_mean)
    kinematic_features, kinematic_pca = _pca_features(kinematics.descriptor)
    retrieval, retrieval_rows = analyze_cross_motion_retrieval(
        publications,
        latent_features,
        kinematic_features,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    clustering, cluster_assignments = analyze_clustering(
        publications,
        latent_features,
        seed=args.seed,
        max_clusters=args.max_kmeans_clusters,
    )
    clustering_for_plot = dict(clustering)
    clustering_for_plot["_hdbscan_labels"] = cluster_assignments["hdbscan"].tolist()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tsne, tsne_trust = _plot_summary(
        output_dir / "cross_motion_summary.png",
        publications,
        latent_features,
        retrieval,
        clustering_for_plot,
        seed=args.seed,
        perplexity=args.tsne_perplexity,
        iterations=args.tsne_iterations,
    )
    metadata = _selection_metadata(args.selection)
    publication_rows: list[dict[str, Any]] = []
    for index in range(publications.latent_mean.shape[0]):
        name = str(publications.motion_names[index])
        row: dict[str, Any] = {
            "publication_index": index,
            "motion_name": name,
            "reference_step": int(publications.reference_steps[index]),
            "reference_length": int(publications.reference_lengths[index]),
            "trajectory_rank": int(kinematics.trajectory_ranks[index]),
            "replicate_count": int(publications.replicate_counts[index]),
            "latent_rms_spread": float(publications.latent_rms_spread[index]),
            "tsne_1": float(tsne[index, 0]),
            "tsne_2": float(tsne[index, 1]),
            "category": metadata.get(name, {}).get("category", "unknown"),
        }
        if publications.activities is not None:
            row["semantic_activity"] = str(publications.activities[index])
            row["semantic_phase_label"] = str(publications.phase_labels[index])
            for axis in publications.axes:
                row[axis] = int(publications.axes[axis][index])
        for summary_index, summary_name in enumerate(kinematics.summary_names):
            row[summary_name] = float(kinematics.summary[index, summary_index])
        publication_rows.append(row)
    _write_csv(output_dir / "unique_publications.csv", publication_rows)
    _write_csv(output_dir / "cross_motion_neighbors.csv", retrieval_rows)
    _write_csv(output_dir / "kmeans_sweep.csv", clustering["kmeans"])

    analysis = {
        "schema": "cross_motion_latent_structure_v1",
        "protocol": {
            "samples_dir": str(args.samples_dir.expanduser().resolve()),
            "reference_arrays_dir": str(
                args.reference_arrays_dir.expanduser().resolve()
            ),
            "phase_annotations": str(args.phase_annotations.expanduser().resolve())
            if args.phase_annotations is not None
            else None,
            "seed": int(args.seed),
            "excluded_motion_names": list(excluded_motion_names),
            "replicate_aggregation": (
                "mean latent per exact motion/reference step; RMS latent spread retained"
            ),
            "retrieval_exclusion": "all publications from the query motion",
            "kinematic_descriptor": (
                "30-step reference window; root translation and initial yaw removed; "
                "eight samples of joint pose/velocity, pelvis path, and 14-body "
                "relative pose and linear/angular velocity"
            ),
        },
        "rows": {
            "collected": int(data.current.shape[0]),
            "unique_publications": int(publications.latent_mean.shape[0]),
            "unique_publications_before_full_window_filter": int(
                all_publications.latent_mean.shape[0]
            ),
            "dropped_tail_publications_without_30_step_kinematic_window": int(
                np.sum(~full_window)
            ),
            "motions": len(_ordered_unique(publications.motion_names)),
            "replicate_count_values": sorted(
                int(value) for value in np.unique(publications.replicate_counts)
            ),
            "latent_rms_spread_mean": float(np.mean(publications.latent_rms_spread)),
        },
        "latent_pca": latent_pca,
        "kinematic_pca": kinematic_pca,
        "retrieval": retrieval,
        "clustering": clustering,
        "tsne": {
            "role": "visualization only; no clustering or retrieval uses t-SNE coordinates",
            "trustworthiness_k10": tsne_trust,
        },
        "interpretation": {
            "support": (
                "A distance ratio below one with its motion-bootstrap confidence interval "
                "below one means cross-motion latent neighbors are kinematically closer "
                "than random cross-motion publications."
            ),
            "cluster_support": (
                "A useful semantic cluster contains multiple motion identities and aligns "
                "more strongly with semantic activity/axes than with motion identity."
            ),
        },
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[PASS] {data.current.shape[0]} collected rows -> "
        f"{publications.latent_mean.shape[0]} unique publications across "
        f"{analysis['rows']['motions']} motions."
    )
    print(f"[PASS] {output_dir / 'analysis.json'}")
    print(f"[PASS] {output_dir / 'cross_motion_summary.png'}")


if __name__ == "__main__":
    main()
