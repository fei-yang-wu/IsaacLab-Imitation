#!/usr/bin/env python3
"""Probe combo latent semantics on the full BONES temporal sidecar.

The temporal sidecar supplies event text and start/end times, but no fixed
semantic ontology. This analysis therefore records both the original text and
small, explicit keyword-derived event/attribute labels. Those labels are
diagnostics for scaling the semantic probe; they are not treated as human
annotations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score

from imitation_experiments.evaluation.analyze_reference_latent_scale import (
    ENCODER_WINDOW_STEPS,
    _device,
    _encode,
    _load_encoder,
    _open_array,
    _pca_features,
    _sha256,
    normalized_action_family,
    root_qpos_expert_windows,
)


FPS = 50.0
AXES = (
    "locomoting",
    "manipulating",
    "jumping",
    "turning",
    "stationary",
    "backward",
    "sideways",
    "slow_locomotion",
    "torso_lowered",
    "object_loaded",
    "active_right_hand",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_arrays_dir", type=Path, required=True)
    parser.add_argument("--language_sidecar", type=Path, required=True)
    parser.add_argument("--temporal_sidecar", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_families", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--tsne_rows", type=int, default=5000)
    parser.add_argument("--tsne_iterations", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _sha256_jsonl(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_labels(text: str, category: str, goal: str) -> tuple[str, dict[str, int]]:
    value = f"{text} {category} {goal}".lower()
    rules = {
        "locomoting": r"\b(walk|walking|jog|jogging|run|running|step|stepping|stride|sprint|locomot|pace)\b",
        "manipulating": r"\b(hold|holds|carry|carr(y|ies)|lift|lifting|grab|grasp|pick|place|push|pull|throw|catch|open|close|drink|eat|kick|reach)\b",
        "jumping": r"\b(jump|jumps|jumping|hop|hops|hopping|leap|leaps)\b",
        "turning": r"\b(turn|turns|turning|rotate|rotates|rotation|spin|spins|twirl)\b",
        "stationary": r"\b(idle|standing|stand|stands|still|stationary|stop|stops|sitting|sit)\b",
        "backward": r"\b(backward|backwards|reverse|reversing)\b",
        "sideways": r"\b(sideways|side-to-side|lateral|to the side|toward the side)\b",
        "slow_locomotion": r"\b(slow|slowly|gentle|gently)\b",
        "torso_lowered": r"\b(bend|bends|bending|stoop|stoops|crouch|crouches|squat|squats|kneel|kneels|lean|leans|lower|lowers|hunch)\b",
        "object_loaded": r"\b(object|box|crate|ball|mug|cup|phone|bottle|chair|basket|bag|weight)\b",
        "active_right_hand": r"\bright (hand|arm|wrist)\b",
    }
    axes = {name: int(bool(re.search(pattern, value))) for name, pattern in rules.items()}
    if axes["jumping"]:
        event_class = "jumping"
    elif axes["turning"]:
        event_class = "turning"
    elif axes["manipulating"]:
        event_class = "manipulating"
    elif axes["locomoting"]:
        event_class = "locomoting"
    elif axes["stationary"] or axes["torso_lowered"]:
        event_class = "posture_or_stationary"
    elif re.search(r"\b(wave|waving|clap|clapping|point|pointing|gesture|greet|salute|dance|dancer)\b", value):
        event_class = "gesture"
    else:
        event_class = "other"
    axes["event_class"] = event_class  # type: ignore[assignment]
    return event_class, axes


def _event_window(start: int, end: int, event: dict[str, Any]) -> int | None:
    event_start = max(start, int(np.floor(float(event.get("start_time", 0.0)) * FPS)))
    event_end = min(end, int(np.ceil(float(event.get("end_time", 0.0)) * FPS)))
    latest = end - ENCODER_WINDOW_STEPS
    if latest < start or event_start > latest:
        return None
    lo = max(start, event_start)
    hi = min(latest, event_end - ENCODER_WINDOW_STEPS)
    if hi < lo:
        # Very short events are represented by the nearest complete native H10
        # window, while retaining an explicit short-event flag in the row.
        return max(start, min(latest, event_start))
    return int((lo + hi) // 2)


def _select_rows(
    metadata: dict[str, Any],
    motions: list[dict[str, Any]],
    temporal: dict[str, dict[str, Any]],
    *,
    max_families: int,
    seed: int,
) -> list[dict[str, Any]]:
    info = metadata["traj_info"]
    starts = np.asarray(info["start_index"], dtype=np.int64)
    ends = np.asarray(info["end_index"], dtype=np.int64)
    by_name = {
        str(entry[1]): (rank, int(starts[rank]), int(ends[rank]))
        for rank, entry in enumerate(info["ordered_traj_list"])
    }
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for motion in motions:
        source_name = str(motion["bones_seed_filename"])
        lookup = source_name.replace("__", "_")
        if lookup not in by_name or source_name not in temporal:
            continue
        rank, start, end = by_name[lookup]
        if end - start < ENCODER_WINDOW_STEPS:
            continue
        family = normalized_action_family(lookup)
        events = temporal[source_name].get("events", [])
        valid = [(event, _event_window(start, end, event)) for event in events]
        valid = [(event, step) for event, step in valid if step is not None]
        if not valid:
            continue
        rng = np.random.default_rng(seed + rank)
        event, step = valid[int(rng.integers(len(valid)))]
        text = str(event.get("description", ""))
        event_class, axes = _text_labels(text, str(motion.get("category", "")), str(motion.get("language_goal", "")))
        candidates[family].append({
            "trajectory_rank": rank,
            "motion_name": lookup,
            "reference_step": int(step - start),
            "category": str(motion.get("category", "")),
            "language_goal": str(motion.get("language_goal", "")),
            "event_description": text,
            "event_class": event_class,
            "short_event": int(float(event.get("end_time", 0.0)) - float(event.get("start_time", 0.0)) < 0.18),
            "family": family,
            **{f"axis_{axis}": int(axes[axis]) for axis in AXES},
        })
    families_by_category: dict[str, list[str]] = defaultdict(list)
    for family, rows in candidates.items():
        families_by_category[rows[0]["category"]].append(family)
    rng = np.random.default_rng(seed)
    categories = sorted(families_by_category)
    chosen: list[str] = []
    quota = max(1, max_families // max(1, len(categories)))
    for category in categories:
        values = np.asarray(sorted(families_by_category[category]), dtype=object)
        take = min(quota, values.size)
        chosen.extend(str(x) for x in rng.choice(values, take, replace=False))
    remaining = sorted(set(candidates) - set(chosen))
    if len(chosen) < max_families and remaining:
        take = min(max_families - len(chosen), len(remaining))
        chosen.extend(str(x) for x in rng.choice(np.asarray(remaining, dtype=object), take, replace=False))
    return [candidates[family][0] for family in chosen[:max_families]]


def _group_probe(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, float]:
    classes = np.unique(labels)
    scores = []
    for offset in range(3):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + offset)
        train, test = next(splitter.split(features, labels, groups))
        if not np.all(np.isin(classes, labels[train])) or not np.all(np.isin(classes, labels[test])):
            continue
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + offset)
        model.fit(scaler.transform(features[train]), labels[train])
        prediction = model.predict(scaler.transform(features[test]))
        scores.append((balanced_accuracy_score(labels[test], prediction), f1_score(labels[test], prediction, average="macro", labels=classes)))
    if not scores:
        return {"valid_splits": 0}
    return {"valid_splits": len(scores), "balanced_accuracy": float(np.mean([x[0] for x in scores])), "macro_f1": float(np.mean([x[1] for x in scores]))}


def _retrieval(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int = 10) -> dict[str, float]:
    neighbors = NearestNeighbors(n_neighbors=min(len(features), k + 20), metric="euclidean").fit(features)
    _, indices = neighbors.kneighbors(features)
    selected = []
    for row, group in zip(indices, groups, strict=True):
        row = row[(groups[row] != group)]
        selected.append(row[:k])
    selected = np.asarray(selected)
    agreement = np.mean(labels[selected] == labels[:, None], axis=1)
    baseline = np.asarray([np.mean(labels[groups != group] == labels[i]) for i, group in enumerate(groups)])
    return {"k": int(k), "agreement": float(np.mean(agreement)), "matched_random_agreement": float(np.mean(baseline)), "agreement_improvement": float(np.mean(agreement - baseline))}


def _run_tsne(features: np.ndarray, labels: np.ndarray, output: Path, seed: int, iterations: int) -> dict[str, float]:
    points = TSNE(n_components=2, init="pca", perplexity=min(30.0, max(5.0, (len(features) - 1) / 3.0)), max_iter=iterations, random_state=seed).fit_transform(features)
    np.save(output.with_suffix(".npy"), points)
    return {"rows": int(len(features)), "trustworthiness_k10": float(trustworthiness(features, points, n_neighbors=10))}


def main() -> None:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output directory: {output}")
    output.mkdir(parents=True)
    root = args.reference_arrays_dir.expanduser().resolve()
    metadata_path = root / "reference_arrays_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    motions = json.loads(args.language_sidecar.read_text())["motions"]
    temporal = {str(row["filename"]): row for row in (json.loads(line) for line in args.temporal_sidecar.open())}
    rows = _select_rows(metadata, motions, temporal, max_families=args.max_families, seed=args.seed)
    if len(rows) < 1000:
        raise ValueError(f"Only {len(rows)} eligible temporal rows were selected")
    starts = np.asarray(metadata["traj_info"]["start_index"], dtype=np.int64)
    index = np.asarray([starts[row["trajectory_rank"]] + row["reference_step"] for row in rows], dtype=np.int64)[:, None] + np.arange(ENCODER_WINDOW_STEPS)[None, :]
    qpos = _open_array(root, metadata, "qpos")
    anchor_pos = _open_array(root, metadata, "anchor_pos_w")
    anchor_quat = _open_array(root, metadata, "anchor_quat_w")
    frames = root_qpos_expert_windows(np.asarray(qpos[index, 7:]), np.asarray(anchor_pos[index]), np.asarray(anchor_quat[index]))
    latent = _encode(_load_encoder(args.skill_checkpoint), frames, batch_size=args.batch_size, device=_device(args.device))
    torch.manual_seed(args.seed)
    random_encoder = copy.deepcopy(_load_encoder(args.skill_checkpoint))
    for module in random_encoder.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
            module.reset_parameters()
    random_latent = _encode(random_encoder, frames, batch_size=args.batch_size, device=_device(args.device))
    raw_features, raw_pca = _pca_features(frames.reshape(len(rows), -1))
    latent_features, latent_pca = _pca_features(latent)
    random_features, random_pca = _pca_features(random_latent)
    groups = np.asarray([row["family"] for row in rows], dtype=object)
    category = np.asarray([row["category"] for row in rows], dtype=object)
    category_counts = {str(key): int(value) for key, value in zip(*np.unique(category, return_counts=True))}
    # A few source categories have fewer than 20 selected families. Merge only
    # those rare labels for the grouped probe so every split can contain each
    # class; retain the unmerged labels for retrieval and the coverage audit.
    category_probe_labels = np.asarray(
        [value if category_counts[str(value)] >= 20 else "__rare_category__" for value in category],
        dtype=object,
    )
    event_class = np.asarray([row["event_class"] for row in rows], dtype=object)
    rng = np.random.default_rng(args.seed)
    display_indices = rng.choice(len(rows), min(args.tsne_rows, len(rows)), replace=False)
    display = {}
    for name, features in (("raw", raw_features), ("latent", latent_features), ("random", random_features)):
        display[name] = _run_tsne(features[display_indices], category[display_indices], output / f"{name}_tsne", args.seed, args.tsne_iterations)
    analysis: dict[str, Any] = {
        "schema": "combo_temporal_sidecar_analysis_v1",
        "protocol": {"reference_arrays_dir": str(root), "reference_arrays_manifest_sha256": _sha256(metadata_path), "language_sidecar": str(args.language_sidecar.resolve()), "temporal_sidecar": str(args.temporal_sidecar.resolve()), "temporal_sidecar_sha256": _sha256_jsonl(args.temporal_sidecar), "skill_checkpoint": str(args.skill_checkpoint.resolve()), "skill_checkpoint_sha256": _sha256(args.skill_checkpoint.resolve()), "sampling": "one event-centered native H10 window per selected normalized action family", "labeling": "keyword-derived event classes and axes; original event text retained", "split": "grouped by normalized action family", "raw_space": "standardized exact 380-D combo input followed by PCA-50", "latent_space": "standardized 64-D combo z followed by PCA-50"},
        "rows": {"windows": len(rows), "families": int(np.unique(groups).size), "categories": category_counts, "category_probe_labels": {str(k): int(v) for k, v in zip(*np.unique(category_probe_labels, return_counts=True))}, "event_classes": {str(k): int(v) for k, v in zip(*np.unique(event_class, return_counts=True))}, "short_event_count": int(sum(row["short_event"] for row in rows))},
        "label_rules": {"axes": list(AXES), "source": "regular expressions over event description, clip category, and language goal"},
        "pca": {"raw": raw_pca, "latent": latent_pca, "random": random_pca},
        "category_probe": {"raw": _group_probe(raw_features, category_probe_labels, groups, args.seed), "latent": _group_probe(latent_features, category_probe_labels, groups, args.seed), "random": _group_probe(random_features, category_probe_labels, groups, args.seed)},
        "event_class_probe": {"raw": _group_probe(raw_features, event_class, groups, args.seed), "latent": _group_probe(latent_features, event_class, groups, args.seed), "random": _group_probe(random_features, event_class, groups, args.seed)},
        "category_retrieval_k10": {"raw": _retrieval(raw_features, category, groups), "latent": _retrieval(latent_features, category, groups), "random": _retrieval(random_features, category, groups)},
        "event_class_retrieval_k10": {"raw": _retrieval(raw_features, event_class, groups), "latent": _retrieval(latent_features, event_class, groups), "random": _retrieval(random_features, event_class, groups)},
        "semantic_axis_probes": {},
        "semantic_axis_retrieval_k10": {},
        "tsne": display,
    }
    for axis in AXES:
        labels = np.asarray([row[f"axis_{axis}"] for row in rows], dtype=np.int8)
        if np.unique(labels).size < 2 or labels.sum() < 20:
            continue
        analysis["semantic_axis_probes"][axis] = {"positive_windows": int(labels.sum()), "raw": _group_probe(raw_features, labels, groups, args.seed), "latent": _group_probe(latent_features, labels, groups, args.seed), "random": _group_probe(random_features, labels, groups, args.seed)}
        analysis["semantic_axis_retrieval_k10"][axis] = {"positive_windows": int(labels.sum()), "raw": _retrieval(raw_features, labels, groups), "latent": _retrieval(latent_features, labels, groups), "random": _retrieval(random_features, labels, groups)}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(output / "features.npz", raw=raw_features, latent=latent_features, random=random_features, category=category, event_class=event_class, groups=groups)
    with (output / "windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for axis, name in zip(axes, ("raw", "latent", "random"), strict=True):
        points = np.load(output / f"{name}_tsne.npy")
        for label in np.unique(category[display_indices]):
            mask = category[display_indices] == label
            axis.scatter(points[mask, 0], points[mask, 1], s=5, alpha=0.45, label=str(label))
        axis.set_title(f"{name} t-SNE"); axis.set_xlabel("t-SNE 1"); axis.set_ylabel("t-SNE 2")
    axes[-1].legend(fontsize=6, loc="best", ncol=2)
    figure.savefig(output / "category_tsne.png", dpi=160); plt.close(figure)
    print(f"[PASS] {len(rows)} temporal windows from {len(np.unique(groups))} families -> {output}")


if __name__ == "__main__":
    main()
