#!/usr/bin/env python3
"""Test whether semantic motion phases occupy and revisit shared latent regions.

The quantitative tests run in standardized latent PCA space. t-SNE and PCA-2
are exported only as display coordinates for a global map and time-ordered
trajectory overlays. Repeated randomized rollouts are first averaged into one
unique ``(motion, reference step)`` publication.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from imitation_experiments.evaluation.analyze_collected_latent_space import (
    assign_semantic_phases,
    load_semantic_phase_annotations,
)
from imitation_experiments.evaluation.analyze_cross_motion_latent_structure import (
    AggregatedPublications,
    _bootstrap_motion_mean_ci,
    _cross_motion_neighbor_indices,
    _ordered_unique,
    _write_csv,
    aggregate_publications,
    load_current_collected_latents,
    select_collected_rows,
)


DEFAULT_NEIGHBOR_COUNTS = (1, 3, 5, 10)


@dataclass(frozen=True)
class PhaseUnits:
    """One phase centroid per annotated motion segment."""

    motion_names: np.ndarray
    phase_indices: np.ndarray
    phase_labels: np.ndarray
    activities: np.ndarray
    regions: np.ndarray
    start_steps: np.ndarray
    end_steps: np.ndarray
    publication_counts: np.ndarray
    centroid_features: np.ndarray
    pca2: np.ndarray
    tsne2: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--phase_annotations", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iterations", type=int, default=1500)
    parser.add_argument("--exclude_motion_names", nargs="*", default=())
    parser.add_argument("--trajectory_names", nargs="*", default=())
    parser.add_argument(
        "--inline_html",
        type=Path,
        help="Optional second standalone HTML path for the Codex inline visual.",
    )
    return parser.parse_args()


def load_semantic_region_taxonomy(path: Path) -> dict[str, Any]:
    """Load and validate the ordered shared-region rule table."""
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "semantic_region_taxonomy_v1":
        raise ValueError(f"{resolved} is not semantic_region_taxonomy_v1.")
    regions = payload.get("regions")
    if not isinstance(regions, list) or len(regions) < 2:
        raise ValueError(f"{resolved} needs at least two ordered region rules.")
    names = [str(row.get("name", "")) for row in regions]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{resolved} has missing or duplicate region names.")
    defaults = [index for index, row in enumerate(regions) if row.get("default")]
    if defaults != [len(regions) - 1]:
        raise ValueError("Exactly the final semantic-region rule must be default.")
    for row in regions:
        if not str(row.get("display_name", "")) or not str(row.get("description", "")):
            raise ValueError(f"Semantic region {row.get('name')!r} lacks prose.")
        color = str(row.get("color", ""))
        if len(color) != 7 or not color.startswith("#"):
            raise ValueError(f"Semantic region {row['name']!r} lacks a hex color.")
    return payload


def assign_semantic_region(
    activity: str, axes: Mapping[str, int | bool], taxonomy: Mapping[str, Any]
) -> str:
    """Return the first ordered region rule matching one annotated phase."""
    for rule in taxonomy["regions"]:
        if rule.get("default"):
            return str(rule["name"])
        all_true = tuple(str(value) for value in rule.get("all_true", ()))
        any_true = tuple(str(value) for value in rule.get("any_true", ()))
        activity_in = tuple(str(value) for value in rule.get("activity_in", ()))
        if any(not bool(axes.get(axis, False)) for axis in all_true):
            continue
        if any_true and not any(bool(axes.get(axis, False)) for axis in any_true):
            continue
        if activity_in and activity not in activity_in:
            continue
        return str(rule["name"])
    raise ValueError("Semantic-region taxonomy has no matching/default rule.")


def assign_publication_regions(
    publications: AggregatedPublications, taxonomy: Mapping[str, Any]
) -> np.ndarray:
    """Assign the shared vocabulary to each unique latent publication."""
    if publications.activities is None:
        raise ValueError("Semantic activities are required for region assignment.")
    return np.asarray(
        [
            assign_semantic_region(
                str(publications.activities[index]),
                {
                    axis: int(values[index])
                    for axis, values in publications.axes.items()
                },
                taxonomy,
            )
            for index in range(publications.latent_mean.shape[0])
        ],
        dtype=object,
    )


def publication_phase_indices(
    publications: AggregatedPublications, annotations: Mapping[str, Any]
) -> np.ndarray:
    """Recover phase indices after exact-publication aggregation."""
    by_motion = {str(row["motion_name"]): row for row in annotations["motions"]}
    result = np.empty(publications.latent_mean.shape[0], dtype=np.int64)
    for name in _ordered_unique(publications.motion_names):
        if name not in by_motion:
            raise ValueError(f"No semantic phases for publication motion {name!r}.")
        selected = np.flatnonzero(publications.motion_names == name)
        phases = by_motion[name]["phases"]
        ends = np.asarray([int(phase["end_step"]) for phase in phases])
        result[selected] = np.searchsorted(
            ends, publications.reference_steps[selected], side="right"
        )
    return result


def fit_latent_space(
    latent: np.ndarray,
    *,
    seed: int,
    perplexity: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the metric PCA space plus PCA-2 and display-only t-SNE."""
    scaled = StandardScaler().fit_transform(latent)
    components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    if components < 2:
        raise ValueError("At least three publications are required.")
    pca = PCA(n_components=components, random_state=seed).fit(scaled)
    features = pca.transform(scaled)
    tsne2 = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (features.shape[0] - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        max_iter=iterations,
        random_state=seed,
    ).fit_transform(features)
    trust = float(
        trustworthiness(
            features,
            tsne2,
            n_neighbors=min(10, max(1, features.shape[0] // 3 - 1)),
        )
    )
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return (
        features,
        features[:, :2],
        tsne2,
        {
            "metric_space": "standardized 256-D latent followed by PCA-50",
            "components": int(components),
            "explained_variance_first_2": float(cumulative[1]),
            "dimensions_for_90_percent": int(np.searchsorted(cumulative, 0.9) + 1)
            if cumulative[-1] >= 0.9
            else int(components + 1),
            "tsne_trustworthiness_k10": trust,
            "projection_warning": (
                "PCA-2 and t-SNE are display coordinates only. Every distance, "
                "classifier, cluster, and return score uses the metric PCA space."
            ),
        },
    )


def build_phase_units(
    publications: AggregatedPublications,
    phase_indices: np.ndarray,
    regions: np.ndarray,
    latent_features: np.ndarray,
    pca2: np.ndarray,
    tsne2: np.ndarray,
    annotations: Mapping[str, Any],
) -> PhaseUnits:
    """Average ordered publication features inside each annotated phase."""
    by_motion = {str(row["motion_name"]): row for row in annotations["motions"]}
    values: dict[str, list[Any]] = {
        "motion_names": [],
        "phase_indices": [],
        "phase_labels": [],
        "activities": [],
        "regions": [],
        "start_steps": [],
        "end_steps": [],
        "publication_counts": [],
        "centroid_features": [],
        "pca2": [],
        "tsne2": [],
    }
    for name in _ordered_unique(publications.motion_names):
        motion = by_motion[name]
        for phase_index, phase in enumerate(motion["phases"]):
            selected = np.flatnonzero(
                (publications.motion_names == name) & (phase_indices == phase_index)
            )
            if selected.size == 0:
                continue
            phase_regions = np.unique(regions[selected])
            if phase_regions.size != 1:
                raise ValueError(f"{name} phase {phase_index} spans shared regions.")
            values["motion_names"].append(name)
            values["phase_indices"].append(phase_index)
            values["phase_labels"].append(str(phase["label"]))
            values["activities"].append(str(phase["activity"]))
            values["regions"].append(str(phase_regions[0]))
            values["start_steps"].append(int(phase["start_step"]))
            values["end_steps"].append(int(phase["end_step"]))
            values["publication_counts"].append(int(selected.size))
            values["centroid_features"].append(latent_features[selected].mean(axis=0))
            values["pca2"].append(pca2[selected].mean(axis=0))
            values["tsne2"].append(tsne2[selected].mean(axis=0))
    return PhaseUnits(
        motion_names=np.asarray(values["motion_names"], dtype=object),
        phase_indices=np.asarray(values["phase_indices"], dtype=np.int64),
        phase_labels=np.asarray(values["phase_labels"], dtype=object),
        activities=np.asarray(values["activities"], dtype=object),
        regions=np.asarray(values["regions"], dtype=object),
        start_steps=np.asarray(values["start_steps"], dtype=np.int64),
        end_steps=np.asarray(values["end_steps"], dtype=np.int64),
        publication_counts=np.asarray(values["publication_counts"], dtype=np.int64),
        centroid_features=np.stack(values["centroid_features"]),
        pca2=np.stack(values["pca2"]),
        tsne2=np.stack(values["tsne2"]),
    )


def analyze_semantic_neighbors(
    features: np.ndarray,
    motion_names: np.ndarray,
    regions: np.ndarray,
    *,
    neighbor_counts: Sequence[int] = DEFAULT_NEIGHBOR_COUNTS,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Measure shared-region purity after excluding the complete query motion."""
    distances = pairwise_distances(features)
    neighbors = _cross_motion_neighbor_indices(
        distances, motion_names, max(int(value) for value in neighbor_counts)
    )
    same_motion = motion_names[:, None] == motion_names[None, :]
    eligible = ~same_motion
    np.fill_diagonal(eligible, False)
    result: dict[str, Any] = {
        "definition": (
            "For each query, find latent-nearest samples after excluding its entire "
            "source motion, then compare semantic-region agreement with the exact "
            "eligible cross-motion label frequency for that query."
        ),
        "by_k": {},
    }
    for count_value in neighbor_counts:
        count = min(int(count_value), neighbors.shape[1])
        selected = neighbors[:, :count]
        agreement = np.mean(regions[selected] == regions[:, None], axis=1)
        baseline = np.asarray(
            [
                float(np.mean(regions[eligible[index]] == regions[index]))
                for index in range(regions.size)
            ]
        )
        improvement = agreement - baseline
        low, high = _bootstrap_motion_mean_ci(
            improvement,
            motion_names,
            seed=seed + count,
            samples=bootstrap_samples,
        )
        result["by_k"][str(count)] = {
            "agreement": float(np.mean(agreement)),
            "matched_random_agreement": float(np.mean(baseline)),
            "agreement_improvement": float(np.mean(improvement)),
            "agreement_improvement_motion_bootstrap_95ci": [low, high],
            "by_region": {
                region: {
                    "query_count": int(np.sum(regions == region)),
                    "agreement": float(np.mean(agreement[regions == region])),
                    "matched_random_agreement": float(
                        np.mean(baseline[regions == region])
                    ),
                    "agreement_improvement": float(
                        np.mean(improvement[regions == region])
                    ),
                }
                for region in _ordered_unique(regions)
            },
        }
    return result


def analyze_region_separation(
    phase_units: PhaseUnits, *, seed: int, bootstrap_samples: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Compare each phase to its nearest same-region and other-region phases."""
    distance = pairwise_distances(phase_units.centroid_features)
    ratios = np.full(phase_units.regions.size, np.nan, dtype=np.float64)
    for index in range(phase_units.regions.size):
        cross_motion = phase_units.motion_names != phase_units.motion_names[index]
        same_region = cross_motion & (phase_units.regions == phase_units.regions[index])
        other_region = cross_motion & (
            phase_units.regions != phase_units.regions[index]
        )
        if np.any(same_region) and np.any(other_region):
            ratios[index] = float(
                np.min(distance[index, same_region])
                / max(np.min(distance[index, other_region]), 1.0e-12)
            )
    valid = np.isfinite(ratios)
    low, high = _bootstrap_motion_mean_ci(
        ratios[valid],
        phase_units.motion_names[valid],
        seed=seed,
        samples=bootstrap_samples,
    )
    return {
        "definition": (
            "Nearest same-region phase distance divided by nearest different-region "
            "phase distance, with all phases from the query motion excluded. Below "
            "1 means the nearest cross-motion phase has the expected semantic region."
        ),
        "eligible_phases": int(np.sum(valid)),
        "mean_ratio": float(np.mean(ratios[valid])),
        "median_ratio": float(np.median(ratios[valid])),
        "fraction_below_one": float(np.mean(ratios[valid] < 1.0)),
        "mean_ratio_motion_bootstrap_95ci": [low, high],
        "by_region": {
            region: {
                "phase_count": int(np.sum(valid & (phase_units.regions == region))),
                "mean_ratio": float(
                    np.mean(ratios[valid & (phase_units.regions == region)])
                ),
                "fraction_below_one": float(
                    np.mean(ratios[valid & (phase_units.regions == region)] < 1.0)
                ),
            }
            for region in _ordered_unique(phase_units.regions[valid])
        },
    }, ratios


def analyze_leave_one_motion_out(
    phase_units: PhaseUnits,
) -> tuple[dict[str, Any], np.ndarray]:
    """Classify each phase using region centroids fitted on other motions only."""
    predictions = np.empty_like(phase_units.regions)
    per_motion: list[dict[str, Any]] = []
    for name in _ordered_unique(phase_units.motion_names):
        test = phase_units.motion_names == name
        train = ~test
        train_regions = _ordered_unique(phase_units.regions[train])
        centroids = np.stack(
            [
                phase_units.centroid_features[
                    train & (phase_units.regions == region)
                ].mean(axis=0)
                for region in train_regions
            ]
        )
        nearest = np.argmin(
            pairwise_distances(phase_units.centroid_features[test], centroids), axis=1
        )
        predictions[test] = np.asarray(train_regions, dtype=object)[nearest]
        per_motion.append(
            {
                "motion_name": name,
                "phase_count": int(np.sum(test)),
                "accuracy": float(
                    np.mean(predictions[test] == phase_units.regions[test])
                ),
            }
        )
    labels = _ordered_unique(phase_units.regions)
    recall = {
        label: float(np.mean(predictions[phase_units.regions == label] == label))
        for label in labels
    }
    majority = max(labels, key=lambda label: int(np.sum(phase_units.regions == label)))
    return {
        "definition": (
            "Nearest semantic-region centroid classification. For a held-out motion, "
            "every class centroid is estimated only from phases belonging to other "
            "motions; the detailed source phrase is never used as an input feature."
        ),
        "accuracy": float(np.mean(predictions == phase_units.regions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(phase_units.regions, predictions)
        ),
        "balanced_chance_accuracy": float(1.0 / len(labels)),
        "majority_class": majority,
        "majority_accuracy": float(np.mean(phase_units.regions == majority)),
        "recall_by_region": recall,
        "per_motion": per_motion,
    }, predictions


def _cluster_metrics(labels: np.ndarray, phase_units: PhaseUnits) -> dict[str, Any]:
    retained = labels >= 0
    clusters = np.unique(labels[retained])
    result: dict[str, Any] = {
        "cluster_count": int(clusters.size),
        "noise_fraction": float(1.0 - np.mean(retained)),
    }
    if clusters.size < 2:
        result.update({"silhouette": None, "semantic_region_ami": None})
        return result
    result["silhouette"] = float(
        silhouette_score(phase_units.centroid_features[retained], labels[retained])
    )
    result["semantic_region_ami"] = float(
        adjusted_mutual_info_score(phase_units.regions[retained], labels[retained])
    )
    result["motion_identity_ami"] = float(
        adjusted_mutual_info_score(phase_units.motion_names[retained], labels[retained])
    )
    majority = 0
    mixed = 0
    for cluster in clusters:
        selected = retained & (labels == cluster)
        counts = [
            int(np.sum(phase_units.regions[selected] == region))
            for region in np.unique(phase_units.regions[selected])
        ]
        majority += max(counts)
        mixed += int(np.unique(phase_units.motion_names[selected]).size >= 2)
    result["semantic_region_purity"] = float(majority / np.sum(retained))
    result["clusters_with_multiple_motions_fraction"] = float(mixed / clusters.size)
    return result


def _cluster_composition(
    labels: np.ndarray, phase_units: PhaseUnits
) -> list[dict[str, Any]]:
    """Return the semantic and source-motion makeup of every retained cluster."""
    rows: list[dict[str, Any]] = []
    for cluster in np.unique(labels[labels >= 0]):
        selected = labels == cluster
        region_counts = {
            region: int(np.sum(phase_units.regions[selected] == region))
            for region in _ordered_unique(phase_units.regions[selected])
        }
        dominant = max(region_counts, key=region_counts.get)
        rows.append(
            {
                "cluster": int(cluster),
                "phase_count": int(np.sum(selected)),
                "motion_count": int(np.unique(phase_units.motion_names[selected]).size),
                "dominant_region": dominant,
                "dominant_region_fraction": float(
                    region_counts[dominant] / np.sum(selected)
                ),
                "region_counts": region_counts,
            }
        )
    return rows


def analyze_phase_clustering(
    phase_units: PhaseUnits, *, seed: int
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Cluster phase centroids without labels and score semantic agreement."""
    region_count = int(np.unique(phase_units.regions).size)
    fixed = KMeans(n_clusters=region_count, n_init=50, random_state=seed).fit_predict(
        phase_units.centroid_features
    )
    stability = []
    for offset in range(1, 11):
        alternative = KMeans(
            n_clusters=region_count, n_init=20, random_state=seed + offset
        ).fit_predict(phase_units.centroid_features)
        stability.append(adjusted_rand_score(fixed, alternative))
    sweep: list[dict[str, Any]] = []
    for clusters in range(2, min(12, phase_units.regions.size - 1) + 1):
        labels = KMeans(n_clusters=clusters, n_init=30, random_state=seed).fit_predict(
            phase_units.centroid_features
        )
        sweep.append(
            {"requested_clusters": clusters, **_cluster_metrics(labels, phase_units)}
        )
    minimum = max(3, min(6, phase_units.regions.size // 12))
    density = HDBSCAN(
        min_cluster_size=minimum,
        min_samples=max(2, minimum // 2),
        copy=True,
    ).fit_predict(phase_units.centroid_features)
    fixed_metrics = _cluster_metrics(fixed, phase_units)
    return (
        {
            "space": "phase centroids in standardized latent PCA-50; never 2-D projections",
            "semantic_region_count": region_count,
            "kmeans_at_semantic_region_count": {
                **fixed_metrics,
                "assignment_stability_adjusted_rand_mean": float(np.mean(stability)),
                "cluster_composition": _cluster_composition(fixed, phase_units),
            },
            "hdbscan": {
                "min_cluster_size": minimum,
                **_cluster_metrics(density, phase_units),
            },
        },
        fixed,
        sweep,
    )


def analyze_return_excursions(
    phase_units: PhaseUnits,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Find A→B→A semantic sequences and test whether the latent path returns."""
    rows: list[dict[str, Any]] = []
    for name in _ordered_unique(phase_units.motion_names):
        indices = np.flatnonzero(phase_units.motion_names == name)
        indices = indices[np.argsort(phase_units.phase_indices[indices])]
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 2 :]:
                between = indices[
                    (
                        phase_units.phase_indices[indices]
                        > phase_units.phase_indices[left]
                    )
                    & (
                        phase_units.phase_indices[indices]
                        < phase_units.phase_indices[right]
                    )
                ]
                if (
                    phase_units.regions[left] != phase_units.regions[right]
                    or between.size == 0
                    or np.all(phase_units.regions[between] == phase_units.regions[left])
                ):
                    continue
                middle = phase_units.centroid_features[between].mean(axis=0)
                return_distance = float(
                    np.linalg.norm(
                        phase_units.centroid_features[left]
                        - phase_units.centroid_features[right]
                    )
                )
                outward = float(
                    np.linalg.norm(phase_units.centroid_features[left] - middle)
                )
                inward = float(
                    np.linalg.norm(middle - phase_units.centroid_features[right])
                )
                ratio = return_distance / max(0.5 * (outward + inward), 1.0e-12)
                rows.append(
                    {
                        "motion_name": name,
                        "return_region": str(phase_units.regions[left]),
                        "start_phase_index": int(phase_units.phase_indices[left]),
                        "start_phase_label": str(phase_units.phase_labels[left]),
                        "intervening_phase_indices": ";".join(
                            str(int(value))
                            for value in phase_units.phase_indices[between]
                        ),
                        "intervening_regions": ";".join(
                            str(value) for value in phase_units.regions[between]
                        ),
                        "end_phase_index": int(phase_units.phase_indices[right]),
                        "end_phase_label": str(phase_units.phase_labels[right]),
                        "return_distance": return_distance,
                        "mean_excursion_leg_distance": 0.5 * (outward + inward),
                        "return_ratio": ratio,
                        "returned_closer_than_excursion": int(ratio < 1.0),
                    }
                )
    ratios = np.asarray([float(row["return_ratio"]) for row in rows])
    return {
        "definition": (
            "For a non-adjacent A→other→A semantic sequence, the return ratio is "
            "the distance between the two A phase centroids divided by the mean "
            "outward/inward distance through the intervening phase centroid. Below "
            "1 means the trajectory returns closer to its earlier semantic region "
            "than the size of its latent excursion."
        ),
        "sequence_count": len(rows),
        "motion_count": len({str(row["motion_name"]) for row in rows}),
        "mean_return_ratio": float(np.mean(ratios)) if rows else None,
        "median_return_ratio": float(np.median(ratios)) if rows else None,
        "fraction_below_one": float(np.mean(ratios < 1.0)) if rows else None,
    }, rows


def analyze_transition_consistency(
    phase_units: PhaseUnits,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare latent direction for the same semantic phase transition."""
    transitions: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    vector_norms: list[float] = []
    for name in _ordered_unique(phase_units.motion_names):
        indices = np.flatnonzero(phase_units.motion_names == name)
        indices = indices[np.argsort(phase_units.phase_indices[indices])]
        for source, target in zip(indices[:-1], indices[1:], strict=True):
            vector = (
                phase_units.centroid_features[target]
                - phase_units.centroid_features[source]
            )
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-12:
                continue
            vectors.append(vector / norm)
            vector_norms.append(norm)
            transitions.append(
                {
                    "motion_name": name,
                    "source_phase_index": int(phase_units.phase_indices[source]),
                    "target_phase_index": int(phase_units.phase_indices[target]),
                    "source_region": str(phase_units.regions[source]),
                    "target_region": str(phase_units.regions[target]),
                    "transition_type": f"{phase_units.regions[source]} -> {phase_units.regions[target]}",
                }
            )
    if not transitions:
        return {"eligible_pair_count": 0}, []
    vector_array = np.stack(vectors)
    cosine = vector_array @ vector_array.T
    same_values: list[float] = []
    baseline_values: list[float] = []
    type_counts: dict[str, set[str]] = {}
    for row in transitions:
        type_counts.setdefault(str(row["transition_type"]), set()).add(
            str(row["motion_name"])
        )
    eligible_types = {key for key, motions in type_counts.items() if len(motions) >= 2}
    for left in range(len(transitions)):
        for right in range(left + 1, len(transitions)):
            if transitions[left]["motion_name"] == transitions[right]["motion_name"]:
                continue
            same_type = (
                transitions[left]["transition_type"]
                == transitions[right]["transition_type"]
            )
            if same_type and transitions[left]["transition_type"] in eligible_types:
                same_values.append(float(cosine[left, right]))
            elif transitions[left]["transition_type"] in eligible_types:
                baseline_values.append(float(cosine[left, right]))
    for index, row in enumerate(transitions):
        row["direction_norm"] = vector_norms[index]
    same_mean = float(np.mean(same_values)) if same_values else None
    baseline_mean = float(np.mean(baseline_values)) if baseline_values else None
    return {
        "definition": (
            "Cosine similarity between phase-centroid direction vectors for the same "
            "source-region→target-region transition in different motions. The baseline "
            "uses different transition types from different motions. Higher is more "
            "directionally repeatable; this tests path geometry, not region locality."
        ),
        "eligible_transition_types": sorted(eligible_types),
        "same_type_cross_motion_pair_count": len(same_values),
        "different_type_cross_motion_pair_count": len(baseline_values),
        "same_type_cosine_mean": same_mean,
        "different_type_cosine_mean": baseline_mean,
        "cosine_improvement": same_mean - baseline_mean
        if same_mean is not None and baseline_mean is not None
        else None,
    }, transitions


def _selection_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return {
        str(row["motion_name"]): {
            "language_goal": str(row.get("language_goal", row["motion_name"])),
            "category": str(row.get("category", "unknown")),
        }
        for row in payload.get("motions", [])
    }


def _phase_annotation_lookup(
    annotations: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(motion["motion_name"]), index): phase
        for motion in annotations["motions"]
        for index, phase in enumerate(motion["phases"])
    }


def _plot_path(
    axis: plt.Axes,
    name: str,
    publications: AggregatedPublications,
    regions: np.ndarray,
    projection: np.ndarray,
    phase_units: PhaseUnits,
    colors: Mapping[str, str],
    display_names: Mapping[str, str],
) -> None:
    other = publications.motion_names != name
    axis.scatter(
        projection[other, 0],
        projection[other, 1],
        s=8,
        c="#C8CDD4",
        alpha=0.16,
        linewidths=0,
    )
    selected = np.flatnonzero(publications.motion_names == name)
    selected = selected[np.argsort(publications.reference_steps[selected])]
    axis.plot(
        projection[selected, 0],
        projection[selected, 1],
        color="#243043",
        linewidth=1.4,
        alpha=0.65,
        zorder=2,
    )
    for region in _ordered_unique(regions[selected]):
        local = selected[regions[selected] == region]
        axis.scatter(
            projection[local, 0],
            projection[local, 1],
            s=24,
            c=colors[region],
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    phases = np.flatnonzero(phase_units.motion_names == name)
    phases = phases[np.argsort(phase_units.phase_indices[phases])]
    phase_projection = phase_units.tsne2[phases]
    for left, right in zip(phase_projection[:-1], phase_projection[1:], strict=True):
        axis.annotate(
            "",
            xy=right,
            xytext=left,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#111827",
                "lw": 1.8,
                "mutation_scale": 10,
            },
        )
    for ordinal, phase in enumerate(phases, start=1):
        point = phase_units.tsne2[phase]
        axis.text(
            point[0],
            point[1],
            str(ordinal),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            bbox={
                "boxstyle": "circle,pad=0.25",
                "facecolor": colors[str(phase_units.regions[phase])],
                "edgecolor": "#111827",
                "linewidth": 0.7,
            },
            zorder=5,
        )
    sequence = " → ".join(
        display_names[str(phase_units.regions[index])] for index in phases
    )
    axis.set_title(f"{name}\n{sequence}", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_semantic_map(
    path: Path,
    publications: AggregatedPublications,
    regions: np.ndarray,
    tsne2: np.ndarray,
    phase_units: PhaseUnits,
    taxonomy: Mapping[str, Any],
    trajectory_names: Sequence[str],
) -> None:
    """Render a global semantic map plus fixed trajectory overlays."""
    colors = {str(row["name"]): str(row["color"]) for row in taxonomy["regions"]}
    display = {
        str(row["name"]): str(row["display_name"]) for row in taxonomy["regions"]
    }
    names = [
        name for name in trajectory_names if name in set(publications.motion_names)
    ]
    if not names:
        names = [
            name
            for name in _ordered_unique(publications.motion_names)
            if np.unique(phase_units.regions[phase_units.motion_names == name]).size
            >= 2
        ][:3]
    names = names[:3]
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    global_axis = axes[0, 0]
    for region in _ordered_unique(regions):
        selected = regions == region
        global_axis.scatter(
            tsne2[selected, 0],
            tsne2[selected, 1],
            s=13,
            alpha=0.58,
            c=colors[region],
            label=display[region],
            linewidths=0,
        )
    global_axis.set_title("Global latent map: shared semantic regions")
    global_axis.set_xlabel("t-SNE 1 (display only)")
    global_axis.set_ylabel("t-SNE 2 (display only)")
    global_axis.legend(fontsize=7, ncol=2, loc="best")
    for axis, name in zip(axes.flat[1:], names, strict=False):
        _plot_path(
            axis, name, publications, regions, tsne2, phase_units, colors, display
        )
    for axis in axes.flat[1 + len(names) :]:
        axis.axis("off")
    figure.suptitle(
        "Semantic latent regions and time-ordered trajectory traversal", fontsize=16
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _short_motion_name(name: str) -> str:
    """Make a compact legend label while preserving the distinctive action text."""
    without_take = re.sub(r"_(?:R_)?\d+_A\d+$", "", name)
    return without_take.replace("_", " ")


def plot_motion_identity_map(
    path: Path,
    publications: AggregatedPublications,
    tsne2: np.ndarray,
) -> None:
    """Render t-SNE colored and directly labeled by source motion identity."""
    motion_names = _ordered_unique(publications.motion_names)
    colors = plt.get_cmap("tab10").colors[:6]
    markers = ("o", "s", "D", "^", "P")
    figure, axis = plt.subplots(figsize=(17, 11))
    handles = []
    labels = []
    for index, name in enumerate(motion_names):
        selected = publications.motion_names == name
        color = colors[index % len(colors)]
        marker = markers[(index // len(colors)) % len(markers)]
        handle = axis.scatter(
            tsne2[selected, 0],
            tsne2[selected, 1],
            s=22,
            alpha=0.72,
            c=[color],
            marker=marker,
            linewidths=0,
        )
        code = f"M{index + 1:02d}"
        axis.text(
            float(np.median(tsne2[selected, 0])),
            float(np.median(tsne2[selected, 1])),
            code,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.88,
            },
            zorder=5,
        )
        handles.append(handle)
        labels.append(f"{code}  {_short_motion_name(name)}")
    axis.set_title("Global latent map by source motion")
    axis.set_xlabel("t-SNE 1 (display only)")
    axis.set_ylabel("t-SNE 2 (display only)")
    axis.legend(
        handles,
        labels,
        title="Motion identity",
        fontsize=7,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    figure.subplots_adjust(right=0.67)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_interactive_html(
    path: Path,
    publication_rows: Sequence[dict[str, Any]],
    phase_rows: Sequence[dict[str, Any]],
    taxonomy: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    data = {
        "publications": publication_rows,
        "phases": phase_rows,
        "regions": taxonomy["regions"],
        "metrics": {
            "neighbor": analysis["publication_neighbor_semantics"]["by_k"]["10"],
            "heldout": analysis["leave_one_motion_out_phase_classification"],
            "return": analysis["return_excursions"],
            "pca2": analysis["latent_space"]["explained_variance_first_2"],
            "trust": analysis["latent_space"]["tsne_trustworthiness_k10"],
        },
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic latent traversal</title>
<style>
:root{color-scheme:light dark;--bg:#f4f1e8;--ink:#172033;--muted:#667085;--line:#c8c4ba;--panel:#fffdf7;--accent:#c4512d}
@media(prefers-color-scheme:dark){:root{--bg:#111722;--ink:#eef2f7;--muted:#aeb8c8;--line:#3a4557;--panel:#18202d;--accent:#ff916d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:24px}h1{font-family:Georgia,serif;font-size:clamp(25px,4vw,44px);font-weight:500;line-height:1.05;margin:0 0 8px}.dek{max-width:850px;color:var(--muted);line-height:1.5;margin:0 0 20px}.metrics{display:flex;gap:22px;flex-wrap:wrap;border-block:1px solid var(--line);padding:12px 0;margin-bottom:14px}.metric b{display:block;font:600 22px Georgia,serif}.metric span{font-size:12px;color:var(--muted)}.controls{display:flex;gap:14px;align-items:end;flex-wrap:wrap;margin:12px 0}.control{display:grid;gap:4px;min-width:180px}.control:first-child{flex:1}.control label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--ink)}.chart{background:var(--panel);border:1px solid var(--line);min-height:520px;position:relative}.chart svg{display:block;width:100%;height:auto}.tooltip{position:absolute;pointer-events:none;display:none;max-width:290px;background:var(--ink);color:var(--bg);padding:7px 9px;border-radius:4px;font-size:11px;line-height:1.35;z-index:3}.sequence{margin:12px 0 4px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}.phase{display:inline-flex;align-items:center;gap:5px;font-size:12px}.dot{width:9px;height:9px;border-radius:50%}.arrow{color:var(--muted)}.legend{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 18px;font-size:11px;color:var(--muted)}.legend.motion{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.legend span{display:inline-flex;gap:5px;align-items:center}.notes{display:grid;grid-template-columns:1fr 1fr;gap:22px;border-top:1px solid var(--line);padding-top:16px}.notes h2{font:500 18px Georgia,serif;margin:0 0 6px}.notes p{font-size:12px;line-height:1.5;color:var(--muted);margin:0}.definitions{margin-top:14px;font-size:11px;color:var(--muted);line-height:1.5}@media(max-width:650px){main{padding:16px}.chart{min-height:350px}.notes{grid-template-columns:1fr}.metrics{gap:14px}.metric b{font-size:18px}}
</style></head><body><main>
<h1>Does a motion trajectory traverse semantic latent regions?</h1>
<p class="dek">Colored points are unique latent publications, averaged over repeated randomized rollouts. Choose a motion to trace it from frame 0 through its annotated phases. The path is descriptive; every reported test is computed in PCA-50, not in this 2-D display.</p>
<section class="metrics" id="metrics"></section>
<section class="controls"><div class="control"><label for="motion">Trajectory</label><select id="motion"></select></div><div class="control"><label for="projection">Projection</label><select id="projection"><option value="tsne">t-SNE · local view</option><option value="pca">PCA · global linear view</option></select></div><div class="control"><label for="color-mode">Background legend</label><select id="color-mode"><option value="motion">Source motions</option><option value="semantic">Semantic regions</option></select></div></section>
<div class="chart"><svg id="map" viewBox="0 0 1100 620" role="img" aria-label="Semantic latent map with selected trajectory"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="currentColor"/></marker></defs></svg><div class="tooltip" id="tip"></div></div>
<div class="sequence" id="sequence"></div><div class="legend" id="legend"></div>
<section class="notes"><div><h2>How to read the map</h2><p>Background color is the shared semantic region. The selected path is ordered by reference frame; numbered circles are phase centroids and arrows connect consecutive phases. Returning near an earlier same-colored region after visiting another color is the compositional pattern under test.</p></div><div><h2>What counts as evidence</h2><p>Cross-motion neighbor purity excludes the complete query motion. Held-out classification estimates each semantic centroid from other motions only. A return ratio below 1 means repeated A phases are closer to each other than the average A→B and B→A excursion legs.</p></div></section>
<div class="definitions" id="definitions"></div>
</main><script>
const D=__DATA__;const svg=document.getElementById('map'),motion=document.getElementById('motion'),projection=document.getElementById('projection'),colorMode=document.getElementById('color-mode'),tip=document.getElementById('tip'),legend=document.getElementById('legend');
const region=Object.fromEntries(D.regions.map(x=>[x.name,x])),motionNames=[...new Set(D.publications.map(x=>x.motion_name))],motionStyles=Object.fromEntries(motionNames.map((name,index)=>[name,{code:`M${String(index+1).padStart(2,'0')}`,color:`hsl(${Math.round(index*360/motionNames.length)} 62% 52%)`}])) ,names=[...new Set(D.phases.map(x=>x.motion_name))].filter(n=>new Set(D.phases.filter(x=>x.motion_name===n).map(x=>x.semantic_region)).size>1);
const preferred=['Neutral_stoop_down_001_A057','drinking_standing_mug_R_001_A282','cellphone_typing_sequence_one_hand_idle_R_001_A423','inside_door_handle_right_side_open_walk_turn_close_R_001_A514'];names.sort((a,b)=>(preferred.indexOf(a)<0?99:preferred.indexOf(a))-(preferred.indexOf(b)<0?99:preferred.indexOf(b))||a.localeCompare(b));
motion.innerHTML=names.map(n=>`<option>${n}</option>`).join('');
document.getElementById('definitions').innerHTML='<b>Shared-region vocabulary.</b> '+D.regions.map(r=>`<b>${r.display_name}:</b> ${r.description}`).join(' · ');
const pct=x=>(100*x).toFixed(1)+'%';document.getElementById('metrics').innerHTML=`<div class="metric"><b>${pct(D.metrics.neighbor.agreement)}</b><span>cross-motion k=10 semantic agreement</span></div><div class="metric"><b>+${pct(D.metrics.neighbor.agreement_improvement)}</b><span>above matched random</span></div><div class="metric"><b>${pct(D.metrics.heldout.balanced_accuracy)}</b><span>held-out motion balanced accuracy</span></div><div class="metric"><b>${D.metrics.return.fraction_below_one===null?'n/a':pct(D.metrics.return.fraction_below_one)}</b><span>A→other→A returns below ratio 1</span></div>`;
function el(name,attrs={}){const n=document.createElementNS('http://www.w3.org/2000/svg',name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n}
function shortName(name){return name.replace(/_(?:R_)?\d+_A\d+$/,'').replaceAll('_',' ')}
function updateLegend(){const byMotion=colorMode.value==='motion';legend.classList.toggle('motion',byMotion);legend.innerHTML=byMotion?motionNames.map(name=>`<span><i class="dot" style="background:${motionStyles[name].color}"></i>${motionStyles[name].code} ${shortName(name)}</span>`).join(''):D.regions.map(r=>`<span><i class="dot" style="background:${r.color}"></i>${r.display_name}</span>`).join('')}
function render(){svg.querySelectorAll(':scope > :not(defs)').forEach(n=>n.remove());const key=projection.value==='tsne'?['tsne_1','tsne_2']:['pca_1','pca_2'];const xs=D.publications.map(d=>d[key[0]]),ys=D.publications.map(d=>d[key[1]]);const pad=.055,loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys);const sx=x=>55+(x-loX)/(hiX-loX||1)*990,sy=y=>575-(y-loY)/(hiY-loY||1)*520;const chosen=motion.value;
 D.publications.forEach(d=>{const c=el('circle',{cx:sx(d[key[0]]),cy:sy(d[key[1]]),r:d.motion_name===chosen?4.4:2.25,fill:colorMode.value==='motion'?motionStyles[d.motion_name].color:region[d.semantic_region].color,opacity:d.motion_name===chosen?.94:.22,stroke:d.motion_name===chosen?'var(--panel)':'none','stroke-width':.8});c.addEventListener('pointerenter',e=>{tip.style.display='block';tip.innerHTML=`<b>${motionStyles[d.motion_name].code} ${shortName(d.motion_name)}</b><br>${region[d.semantic_region].display_name}<br>frame ${d.reference_step} · phase ${d.phase_index+1}: ${d.phase_label}`});c.addEventListener('pointermove',e=>{tip.style.left=(e.offsetX+12)+'px';tip.style.top=(e.offsetY+12)+'px'});c.addEventListener('pointerleave',()=>tip.style.display='none');svg.appendChild(c)});
 if(colorMode.value==='motion'){motionNames.forEach(name=>{const points=D.publications.filter(d=>d.motion_name===name),px=points.map(d=>d[key[0]]).sort((a,b)=>a-b),py=points.map(d=>d[key[1]]).sort((a,b)=>a-b),middle=Math.floor(points.length/2),label=el('text',{x:sx(px[middle]),y:sy(py[middle]),'text-anchor':'middle',fill:'var(--ink)',stroke:'var(--panel)','stroke-width':3,'paint-order':'stroke','font-size':12});label.textContent=motionStyles[name].code;svg.appendChild(label)})}
 const path=D.publications.filter(d=>d.motion_name===chosen).sort((a,b)=>a.reference_step-b.reference_step);for(let i=0;i<path.length-1;i++){const a=path[i],b=path[i+1];svg.appendChild(el('line',{x1:sx(a[key[0]]),y1:sy(a[key[1]]),x2:sx(b[key[0]]),y2:sy(b[key[1]]),stroke:'currentColor','stroke-width':1.35,opacity:.46}))}
 const phases=D.phases.filter(d=>d.motion_name===chosen).sort((a,b)=>a.phase_index-b.phase_index);for(let i=0;i<phases.length-1;i++){const a=phases[i],b=phases[i+1],ax=sx(a[key[0]]),ay=sy(a[key[1]]),bx=sx(b[key[0]]),by=sy(b[key[1]]),dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy)||1;svg.appendChild(el('line',{x1:ax+dx/len*13,y1:ay+dy/len*13,x2:bx-dx/len*15,y2:by-dy/len*15,stroke:'currentColor','stroke-width':2.3,'marker-end':'url(#arrow)',opacity:.9}))}
 phases.forEach((d,i)=>{const g=el('g'),c=el('circle',{cx:sx(d[key[0]]),cy:sy(d[key[1]]),r:12,fill:region[d.semantic_region].color,stroke:'currentColor','stroke-width':1.5}),t=el('text',{x:sx(d[key[0]]),y:sy(d[key[1]])+4,'text-anchor':'middle',fill:'#fff','font-size':11,'font-weight':700});t.textContent=i+1;g.append(c,t);svg.appendChild(g)});
 document.getElementById('sequence').innerHTML=phases.map((d,i)=>`${i?'<span class="arrow">→</span>':''}<span class="phase"><i class="dot" style="background:${region[d.semantic_region].color}"></i><b>${i+1}</b> ${region[d.semantic_region].display_name}</span>`).join('');
}
motion.addEventListener('change',render);projection.addEventListener('change',render);colorMode.addEventListener('change',()=>{updateLegend();render()});updateLegend();render();
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__DATA__", payload), encoding="utf-8")


def _write_inline_fragment(
    path: Path,
    publication_rows: Sequence[dict[str, Any]],
    phase_rows: Sequence[dict[str, Any]],
    taxonomy: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    """Write the compact, theme-aware Codex inline version of the map."""
    data = {
        "p": [
            [
                row["motion_name"],
                row["reference_step"],
                row["phase_index"],
                row["phase_label"],
                row["semantic_region"],
                round(float(row["pca_1"]), 3),
                round(float(row["pca_2"]), 3),
                round(float(row["tsne_1"]), 3),
                round(float(row["tsne_2"]), 3),
            ]
            for row in publication_rows
        ],
        "h": [
            [
                row["motion_name"],
                row["phase_index"],
                row["phase_label"],
                row["semantic_region"],
                round(float(row["pca_1"]), 3),
                round(float(row["pca_2"]), 3),
                round(float(row["tsne_1"]), 3),
                round(float(row["tsne_2"]), 3),
            ]
            for row in phase_rows
        ],
        "r": [
            [str(row["name"]), str(row["display_name"])] for row in taxonomy["regions"]
        ],
        "m": {
            "agreement": analysis["publication_neighbor_semantics"]["by_k"]["10"][
                "agreement"
            ],
            "random": analysis["publication_neighbor_semantics"]["by_k"]["10"][
                "matched_random_agreement"
            ],
            "heldout": analysis["leave_one_motion_out_phase_classification"][
                "balanced_accuracy"
            ],
            "returned": analysis["return_excursions"]["fraction_below_one"],
            "return_count": analysis["return_excursions"]["sequence_count"],
        },
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = r"""<div id="semantic-latent-traversal">
<style>
#semantic-latent-traversal{width:100%;color:var(--foreground)}
#semantic-latent-traversal .sl-title{margin:0 0 .5rem}
#semantic-latent-traversal .sl-metrics{margin:.35rem 0 .65rem;color:var(--muted-foreground)}
#semantic-latent-traversal .sl-chart{width:100%;position:relative}
#semantic-latent-traversal .sl-chart svg{display:block;width:100%;height:auto}
#semantic-latent-traversal .sl-frame{fill:none;stroke:var(--border)}
#semantic-latent-traversal .sl-path{stroke:var(--foreground);fill:none;stroke-width:1.5;opacity:.58}
#semantic-latent-traversal .sl-arrow{stroke:var(--foreground);fill:none;stroke-width:2}
#semantic-latent-traversal .sl-axis{fill:var(--foreground);font-size:12px}
#semantic-latent-traversal .sl-sequence,#semantic-latent-traversal .sl-legend{display:flex;gap:.55rem;align-items:center;flex-wrap:wrap;margin:.55rem 0}
#semantic-latent-traversal .sl-mode-note{margin:.15rem 0 .5rem;color:var(--muted-foreground)}
#semantic-latent-traversal .sl-legend{color:var(--muted-foreground)}
#semantic-latent-traversal .sl-legend.sl-motion-legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(14rem,1fr));align-items:start}
#semantic-latent-traversal .sl-item{display:inline-flex;gap:.3rem;align-items:center}
#semantic-latent-traversal .sl-glyph{width:.8rem;height:.8rem;display:inline-block;flex:0 0 auto;overflow:visible}
#semantic-latent-traversal .sl-legend-title{flex-basis:100%;color:var(--foreground);font-weight:600}
#semantic-latent-traversal .sl-motion-label{fill:var(--foreground);stroke:var(--background);stroke-width:3px;paint-order:stroke;stroke-linejoin:round;font-size:11px;font-weight:500}
#semantic-latent-traversal circle,#semantic-latent-traversal rect,#semantic-latent-traversal path,#semantic-latent-traversal line{transition:cx .2s,cy .2s,x1 .2s,x2 .2s,y1 .2s,y2 .2s}
@media(prefers-reduced-motion:reduce){#semantic-latent-traversal circle,#semantic-latent-traversal rect,#semantic-latent-traversal path,#semantic-latent-traversal line{transition:none}}
</style>
<h2 class="sl-title">Phase-level semantic latent traversal</h2>
<div class="sl-metrics" id="slt-metrics"></div>
<div class="viz-controls">
  <label class="form-label" for="slt-motion">Trajectory<select class="form-select" id="slt-motion"></select></label>
  <label class="form-label" for="slt-projection">Projection<select class="form-select" id="slt-projection"><option value="tsne">t-SNE · local display</option><option value="pca">PCA · linear display</option></select></label>
  <label class="form-label" for="slt-color">Color points by<select class="form-select" id="slt-color"><option value="semantic">Semantic phase class</option><option value="motion">Source trajectory / motion</option></select></label>
</div>
<div class="sl-mode-note text-small" id="slt-mode-note"></div>
<div class="sl-chart"><svg id="slt-map" role="img" aria-label="Unique latent publications with a time-ordered selected trajectory"><defs><marker id="slt-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="currentColor"></path></marker></defs></svg></div>
<div class="sl-sequence" id="slt-sequence"></div>
<div class="sl-legend text-small" id="slt-legend"></div>
<script>
(()=>{const root=document.getElementById('semantic-latent-traversal'),D=__DATA__,svg=root.querySelector('#slt-map'),motion=root.querySelector('#slt-motion'),projection=root.querySelector('#slt-projection'),colorMode=root.querySelector('#slt-color'),legend=root.querySelector('#slt-legend');
const tokens=['var(--viz-series-1)','var(--viz-series-2)','var(--viz-series-3)','var(--viz-series-4)','var(--viz-series-5)','var(--viz-series-6)'];
const shapes=['circle','square','diamond','triangle','cross'],styleAt=i=>({color:tokens[i%tokens.length],shape:shapes[i%shapes.length]});
const regions=Object.fromEntries(D.r.map((d,i)=>[d[0],{label:d[1],...styleAt(i)}]));
const motionNames=[...new Set(D.p.map(d=>d[0]))],motionStyles=Object.fromEntries(motionNames.map((n,i)=>[n,{code:`M${String(i+1).padStart(2,'0')}`,...styleAt(i)}]));
const names=[...new Set(D.h.map(d=>d[0]))].filter(n=>new Set(D.h.filter(d=>d[0]===n).map(d=>d[3])).size>1),preferred=['Neutral_stoop_down_001_A057','drinking_standing_mug_R_001_A282','cellphone_typing_sequence_one_hand_idle_R_001_A423','inside_door_handle_right_side_open_walk_turn_close_R_001_A514'];
names.sort((a,b)=>(preferred.indexOf(a)<0?99:preferred.indexOf(a))-(preferred.indexOf(b)<0?99:preferred.indexOf(b))||a.localeCompare(b));motion.innerHTML=names.map(n=>`<option>${n}</option>`).join('');
const pct=x=>(100*x).toFixed(1)+'%',returns=D.m.returned===null?'n/a':`${Math.round(D.m.returned*D.m.return_count)}/${D.m.return_count}`;
root.querySelector('#slt-metrics').textContent=`Cross-motion k=10 agreement ${pct(D.m.agreement)} (matched random ${pct(D.m.random)}) · held-out balanced accuracy ${pct(D.m.heldout)} · A→other→A returns ${returns}`;
function E(name,attrs={}){const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const[k,v]of Object.entries(attrs))node.setAttribute(k,v);return node}
function shortName(name){return name.replace(/_(?:R_)?\d+_A\d+$/,'').replaceAll('_',' ')}
function glyph(style){const common=`fill="${style.color}" stroke="${style.color}"`;if(style.shape==='circle')return `<svg class="sl-glyph" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5" ${common}/></svg>`;if(style.shape==='square')return `<svg class="sl-glyph" viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="3" width="10" height="10" ${common}/></svg>`;if(style.shape==='diamond')return `<svg class="sl-glyph" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2L14 8L8 14L2 8Z" ${common}/></svg>`;if(style.shape==='triangle')return `<svg class="sl-glyph" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2L14 13H2Z" ${common}/></svg>`;return `<svg class="sl-glyph" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3L13 13M13 3L3 13" fill="none" stroke="${style.color}" stroke-width="3" stroke-linecap="round"/></svg>`}
function mark(x,y,r,style,attrs={}){if(style.shape==='circle')return E('circle',{cx:x,cy:y,r,fill:style.color,...attrs});if(style.shape==='square')return E('rect',{x:x-r,y:y-r,width:2*r,height:2*r,fill:style.color,...attrs});if(style.shape==='diamond')return E('path',{d:`M${x} ${y-r}L${x+r} ${y}L${x} ${y+r}L${x-r} ${y}Z`,fill:style.color,...attrs});if(style.shape==='triangle')return E('path',{d:`M${x} ${y-r}L${x+r} ${y+r}L${x-r} ${y+r}Z`,fill:style.color,...attrs});return E('path',{d:`M${x-r} ${y-r}L${x+r} ${y+r}M${x+r} ${y-r}L${x-r} ${y+r}`,fill:'none',stroke:style.color,'stroke-width':Math.max(1.25,r*.55),'stroke-linecap':'round',...attrs})}
function updateLegend(){const byMotion=colorMode.value==='motion',note=root.querySelector('#slt-mode-note');legend.classList.toggle('sl-motion-legend',byMotion);note.textContent=byMotion?'One color-and-shape identity is assigned to the whole source trajectory.':'Each point is styled by its current semantic phase class; one trajectory can change styles across phases.';legend.innerHTML=byMotion?`<span class="sl-legend-title">Source trajectories — one identity per whole trajectory</span>${motionNames.map(n=>`<span class="sl-item">${glyph(motionStyles[n])}${motionStyles[n].code} ${shortName(n)}</span>`).join('')}`:`<span class="sl-legend-title">Semantic phase classes — multiple classes may occur in one trajectory</span>${D.r.map(d=>`<span class="sl-item">${glyph(regions[d[0]])}${d[1]}</span>`).join('')}`}
function draw(){svg.querySelectorAll(':scope > :not(defs)').forEach(n=>n.remove());const width=Math.max(320,Math.round(root.getBoundingClientRect().width||736)),height=Math.max(300,Math.min(560,Math.round(width*.62))),margin={l:58,r:18,t:16,b:46},key=projection.value==='tsne'?[7,8]:[5,6],xs=D.p.map(d=>d[key[0]]),ys=D.p.map(d=>d[key[1]]),loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys),sx=x=>margin.l+(x-loX)/(hiX-loX||1)*(width-margin.l-margin.r),sy=y=>height-margin.b-(y-loY)/(hiY-loY||1)*(height-margin.t-margin.b),chosen=motion.value;
svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.appendChild(E('rect',{class:'sl-frame','data-chart-frame':'',x:margin.l,y:margin.t,width:width-margin.l-margin.r,height:height-margin.t-margin.b}));
D.p.forEach(d=>{const selected=d[0]===chosen,style=colorMode.value==='motion'?motionStyles[d[0]]:regions[d[4]],x=sx(d[key[0]]),y=sy(d[key[1]]);if(selected)svg.appendChild(E('circle',{cx:x,cy:y,r:5.1,fill:'var(--background)',opacity:.9}));const c=mark(x,y,selected?3.8:2.1,style,{opacity:selected?.96:.27});if(selected)c.setAttribute('data-tooltip',`${motionStyles[d[0]].code} ${shortName(d[0])} · ${regions[d[4]].label} · frame ${d[1]}`);svg.appendChild(c)});
if(colorMode.value==='motion'){motionNames.forEach(n=>{const points=D.p.filter(d=>d[0]===n),px=points.map(d=>d[key[0]]).sort((a,b)=>a-b),py=points.map(d=>d[key[1]]).sort((a,b)=>a-b),middle=Math.floor(points.length/2),label=E('text',{class:'sl-motion-label',x:sx(px[middle]),y:sy(py[middle]),'text-anchor':'middle','data-tooltip':n});label.textContent=motionStyles[n].code;svg.appendChild(label)})}
const path=D.p.filter(d=>d[0]===chosen).sort((a,b)=>a[1]-b[1]);for(let i=0;i<path.length-1;i++){svg.appendChild(E('line',{class:'sl-path',x1:sx(path[i][key[0]]),y1:sy(path[i][key[1]]),x2:sx(path[i+1][key[0]]),y2:sy(path[i+1][key[1]])}))}
const phases=D.h.filter(d=>d[0]===chosen).sort((a,b)=>a[1]-b[1]);for(let i=0;i<phases.length-1;i++){const a=phases[i],b=phases[i+1],ax=sx(a[key[0]-1]),ay=sy(a[key[1]-1]),bx=sx(b[key[0]-1]),by=sy(b[key[1]-1]),dx=bx-ax,dy=by-ay,len=Math.hypot(dx,dy)||1;svg.appendChild(E('line',{class:'sl-arrow',x1:ax+dx/len*13,y1:ay+dy/len*13,x2:bx-dx/len*15,y2:by-dy/len*15,'marker-end':'url(#slt-arrowhead)'}))}
phases.forEach((d,i)=>{const x=sx(d[key[0]-1]),y=sy(d[key[1]-1]),g=E('g',{'data-tooltip':`${i+1}. ${regions[d[3]].label} · ${d[2]}`}),outer=mark(x,y,14,regions[d[3]]),inner=E('circle',{cx:x,cy:y,r:8.5,fill:'var(--background)',stroke:'var(--foreground)','stroke-width':.7}),t=E('text',{x,y:y+4,'text-anchor':'middle',fill:'var(--foreground)','font-size':12,'font-weight':700});t.textContent=i+1;g.append(outer,inner,t);svg.appendChild(g)});
const xTitle=E('text',{class:'sl-axis','data-axis':'x',x:(margin.l+width-margin.r)/2,y:height-10,'text-anchor':'middle'});xTitle.textContent=projection.value==='tsne'?'t-SNE 1 (display only)':'PCA 1 (display only)';svg.appendChild(xTitle);const yTitle=E('text',{class:'sl-axis','data-axis':'y',x:15,y:height/2,'text-anchor':'middle',transform:`rotate(-90 15 ${height/2})`});yTitle.textContent=projection.value==='tsne'?'t-SNE 2 (display only)':'PCA 2 (display only)';svg.appendChild(yTitle);
root.querySelector('#slt-sequence').innerHTML=`<span class="sl-legend-title">Selected trajectory phase order</span>${phases.map((d,i)=>`${i?'→':''}<span class="sl-item">${glyph(regions[d[3]])}<b>${i+1}</b> ${regions[d[3]].label}</span>`).join(' ')}`}
let observedWidth=0;const resizeObserver=new ResizeObserver(entries=>{const nextWidth=Math.round(entries[0].contentRect.width);if(nextWidth!==observedWidth){observedWidth=nextWidth;draw()}});
motion.addEventListener('change',draw);projection.addEventListener('change',draw);colorMode.addEventListener('change',()=>{updateLegend();draw()});updateLegend();observedWidth=Math.round(root.getBoundingClientRect().width);draw();resizeObserver.observe(root)})();
</script>
</div>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__DATA__", payload), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap_samples must be at least 100.")
    if args.tsne_iterations < 250:
        raise ValueError("--tsne_iterations must be at least 250.")
    annotations = load_semantic_phase_annotations(args.phase_annotations)
    taxonomy = load_semantic_region_taxonomy(args.taxonomy)
    data = load_current_collected_latents(args.samples_dir)
    excluded = tuple(str(name) for name in args.exclude_motion_names)
    if excluded:
        present = set(str(name) for name in data.motion_names)
        unknown = sorted(set(excluded) - present)
        if unknown:
            raise ValueError(f"Cannot exclude absent motion names: {unknown}.")
        data = select_collected_rows(
            data, ~np.isin(data.motion_names, np.asarray(excluded, dtype=object))
        )
    assignments = assign_semantic_phases(data, annotations)
    publications = aggregate_publications(data, assignments)
    phase_indices = publication_phase_indices(publications, annotations)
    regions = assign_publication_regions(publications, taxonomy)
    latent_features, pca2, tsne2, latent_space = fit_latent_space(
        publications.latent_mean,
        seed=args.seed,
        perplexity=args.tsne_perplexity,
        iterations=args.tsne_iterations,
    )
    phase_units = build_phase_units(
        publications,
        phase_indices,
        regions,
        latent_features,
        pca2,
        tsne2,
        annotations,
    )
    publication_neighbors = analyze_semantic_neighbors(
        latent_features,
        publications.motion_names,
        regions,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    phase_neighbors = analyze_semantic_neighbors(
        phase_units.centroid_features,
        phase_units.motion_names,
        phase_units.regions,
        neighbor_counts=(1, 3, 5),
        seed=args.seed + 20,
        bootstrap_samples=args.bootstrap_samples,
    )
    separation, separation_ratios = analyze_region_separation(
        phase_units, seed=args.seed + 40, bootstrap_samples=args.bootstrap_samples
    )
    heldout, heldout_predictions = analyze_leave_one_motion_out(phase_units)
    clustering, cluster_labels, cluster_sweep = analyze_phase_clustering(
        phase_units, seed=args.seed
    )
    returns, return_rows = analyze_return_excursions(phase_units)
    transitions, transition_rows = analyze_transition_consistency(phase_units)
    counts = {
        str(row["name"]): {
            "phases": int(np.sum(phase_units.regions == row["name"])),
            "publications": int(np.sum(regions == row["name"])),
            "motions": int(
                np.unique(
                    phase_units.motion_names[phase_units.regions == row["name"]]
                ).size
            ),
        }
        for row in taxonomy["regions"]
    }
    analysis: dict[str, Any] = {
        "schema": "semantic_latent_trajectory_analysis_v1",
        "protocol": {
            "samples_dir": str(args.samples_dir.expanduser().resolve()),
            "phase_annotations": str(args.phase_annotations.expanduser().resolve()),
            "taxonomy": str(args.taxonomy.expanduser().resolve()),
            "excluded_motion_names": list(excluded),
            "replicate_aggregation": "mean latent per exact motion/reference step",
            "quantitative_space": "standardized latent PCA-50",
            "display_spaces": ["PCA-2", "t-SNE-2"],
        },
        "rows": {
            "collected": int(data.current.shape[0]),
            "unique_publications": int(publications.latent_mean.shape[0]),
            "annotated_phases_with_publications": int(phase_units.regions.size),
            "motions": len(_ordered_unique(publications.motion_names)),
        },
        "semantic_region_counts": counts,
        "latent_space": latent_space,
        "publication_neighbor_semantics": publication_neighbors,
        "phase_centroid_neighbor_semantics": phase_neighbors,
        "cross_motion_phase_region_separation": separation,
        "leave_one_motion_out_phase_classification": heldout,
        "unsupervised_phase_clustering": clustering,
        "return_excursions": returns,
        "transition_direction_consistency": transitions,
        "interpretation": {
            "locality_support": (
                "Neighbor-improvement confidence intervals above zero, separation "
                "ratios below one, and held-out balanced accuracy above the class-prior "
                "baseline support reusable cross-motion semantic locality."
            ),
            "compositionality_support": (
                "A trajectory that crosses differently labeled regions and has an "
                "A→other→A return ratio below one demonstrates traversal and return "
                "on this representation; it does not by itself prove arithmetic latent "
                "composition or causal skill control."
            ),
        },
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = _selection_metadata(args.selection)
    publication_rows: list[dict[str, Any]] = []
    for index in range(regions.size):
        name = str(publications.motion_names[index])
        publication_rows.append(
            {
                "publication_index": index,
                "motion_name": name,
                "reference_step": int(publications.reference_steps[index]),
                "reference_length": int(publications.reference_lengths[index]),
                "phase_index": int(phase_indices[index]),
                "phase_label": str(publications.phase_labels[index]),
                "semantic_activity": str(publications.activities[index]),
                "semantic_region": str(regions[index]),
                "language_goal": metadata.get(name, {}).get("language_goal", name),
                "pca_1": float(pca2[index, 0]),
                "pca_2": float(pca2[index, 1]),
                "tsne_1": float(tsne2[index, 0]),
                "tsne_2": float(tsne2[index, 1]),
                "replicate_count": int(publications.replicate_counts[index]),
                "latent_rms_spread": float(publications.latent_rms_spread[index]),
            }
        )
    lookup = _phase_annotation_lookup(annotations)
    phase_rows: list[dict[str, Any]] = []
    for index in range(phase_units.regions.size):
        key = (
            str(phase_units.motion_names[index]),
            int(phase_units.phase_indices[index]),
        )
        annotation = lookup[key]
        phase_rows.append(
            {
                "phase_unit_index": index,
                "motion_name": key[0],
                "phase_index": key[1],
                "phase_label": str(phase_units.phase_labels[index]),
                "source_description": str(annotation["source_description"]),
                "semantic_activity": str(phase_units.activities[index]),
                "semantic_region": str(phase_units.regions[index]),
                "start_step": int(phase_units.start_steps[index]),
                "end_step": int(phase_units.end_steps[index]),
                "publication_count": int(phase_units.publication_counts[index]),
                "nearest_same_to_other_region_ratio": float(separation_ratios[index])
                if np.isfinite(separation_ratios[index])
                else "",
                "heldout_predicted_region": str(heldout_predictions[index]),
                "heldout_correct": int(
                    heldout_predictions[index] == phase_units.regions[index]
                ),
                "kmeans_cluster": int(cluster_labels[index]),
                "pca_1": float(phase_units.pca2[index, 0]),
                "pca_2": float(phase_units.pca2[index, 1]),
                "tsne_1": float(phase_units.tsne2[index, 0]),
                "tsne_2": float(phase_units.tsne2[index, 1]),
            }
        )
    _write_csv(output / "publications.csv", publication_rows)
    _write_csv(output / "phases.csv", phase_rows)
    _write_csv(output / "return_excursions.csv", return_rows)
    _write_csv(output / "transitions.csv", transition_rows)
    _write_csv(output / "kmeans_sweep.csv", cluster_sweep)
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trajectory_names = tuple(str(value) for value in args.trajectory_names)
    plot_semantic_map(
        output / "semantic_trajectory_map.png",
        publications,
        regions,
        tsne2,
        phase_units,
        taxonomy,
        trajectory_names,
    )
    plot_motion_identity_map(output / "latent_map_by_motion.png", publications, tsne2)
    _write_interactive_html(
        output / "semantic_trajectory_map.html",
        publication_rows,
        phase_rows,
        taxonomy,
        analysis,
    )
    if args.inline_html is not None:
        _write_inline_fragment(
            args.inline_html.expanduser().resolve(),
            publication_rows,
            phase_rows,
            taxonomy,
            analysis,
        )
    print(
        f"[PASS] {analysis['rows']['unique_publications']} publications, {analysis['rows']['annotated_phases_with_publications']} phases, {analysis['rows']['motions']} motions."
    )
    print(f"[PASS] {output / 'analysis.json'}")
    print(f"[PASS] {output / 'semantic_trajectory_map.png'}")
    print(f"[PASS] {output / 'latent_map_by_motion.png'}")
    print(f"[PASS] {output / 'semantic_trajectory_map.html'}")


if __name__ == "__main__":
    main()
