#!/usr/bin/env python3
"""Probe broad clip-level BONES categories on an expanded combo sample.

The language sidecar supplies one category per source motion, not per temporal
window. This script treats the category as a weak clip-level label and reports
it separately from temporal semantic annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_analysis_dir", type=Path, required=True)
    parser.add_argument("--language_sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _pca(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = StandardScaler().fit_transform(values)
    components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=components, random_state=0)
    result = pca.fit_transform(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return result, {
        "components": int(components),
        "dimensions_for_90_percent": int(np.searchsorted(cumulative, 0.9) + 1)
        if cumulative[-1] >= 0.9
        else int(components + 1),
    }


def _probe(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, Any]:
    classes = np.unique(labels)
    rows: list[dict[str, float]] = []
    for offset in range(5):
        train, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + offset).split(
                features, labels, groups
            )
        )
        if not np.all(np.isin(classes, labels[train])) or not np.all(np.isin(classes, labels[test])):
            continue
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + offset)
        model.fit(scaler.transform(features[train]), labels[train])
        prediction = model.predict(scaler.transform(features[test]))
        rows.append(
            {
                "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
                "macro_f1": float(f1_score(labels[test], prediction, labels=classes, average="macro")),
                "train_groups": float(np.unique(groups[train]).size),
                "test_groups": float(np.unique(groups[test]).size),
            }
        )
    if not rows:
        raise ValueError("No grouped split contained every category in train and test.")
    return {"valid_splits": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]}}


def _retrieval(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    distances[groups[:, None] == groups[None, :]] = np.inf
    k = min(10, features.shape[0] - 1)
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    agreement = np.mean(labels[neighbors] == labels[:, None], axis=1)
    random = np.asarray([np.mean(labels[groups != group] == labels[i]) for i, group in enumerate(groups)])
    return {
        "k": int(k),
        "agreement": float(np.mean(agreement)),
        "matched_random_agreement": float(np.mean(random)),
        "agreement_improvement": float(np.mean(agreement - random)),
    }


def main() -> None:
    args = _args()
    analysis_dir = args.reference_analysis_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    features = np.load(analysis_dir / "canonical_latents.npz", allow_pickle=True)
    names = np.asarray(features["motion_name"]).astype(str)
    groups = names.copy()
    sidecar = json.loads(args.language_sidecar.expanduser().resolve().read_text())
    categories = {str(row["name"]): str(row["category"]) for row in sidecar["motions"]}
    missing = sorted(set(names) - set(categories))
    if missing:
        raise ValueError(f"Missing {len(missing)} sampled names in language sidecar.")
    labels = np.asarray([categories[name] for name in names], dtype=object)
    raw, raw_pca = _pca(np.asarray(features["raw_window"], dtype=np.float32))
    latent, latent_pca = _pca(np.asarray(features["latent"], dtype=np.float32))
    result = {
        "schema": "combo_clip_category_probe_v1",
        "protocol": {
            "reference_analysis_dir": str(analysis_dir),
            "language_sidecar": str(args.language_sidecar.expanduser().resolve()),
            "label_scope": "clip-level category repeated across windows; weak semantic label",
            "split": "grouped by source motion",
        },
        "rows": {
            "windows": int(len(names)),
            "motions": int(np.unique(groups).size),
            "categories": {str(key): int(value) for key, value in zip(*np.unique(labels, return_counts=True))},
        },
        "pca": {"raw": raw_pca, "latent": latent_pca},
        "grouped_linear_probe": {
            "raw": _probe(raw, labels, groups, args.seed),
            "latent": _probe(latent, labels, groups, args.seed),
        },
        "retrieval_k10": {
            "raw": _retrieval(raw, labels, groups),
            "latent": _retrieval(latent, labels, groups),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[PASS] {len(names)} windows, {len(np.unique(groups))} motions -> {output}")


if __name__ == "__main__":
    main()
