#!/usr/bin/env python3
"""Offline diagnostics for SkillCommander language embedding tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from language_prompts import humanize_motion_name


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _load_table(path: Path) -> dict[str, Any]:
    table = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    if not isinstance(table, dict):
        raise ValueError(f"Embedding table must be a dict: {path}")
    names = table.get("names")
    embeddings = table.get("embeddings")
    if not isinstance(names, list) or not names:
        raise ValueError(f"Embedding table has no non-empty names list: {path}")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"Embedding table has no [N, D] embeddings tensor: {path}")
    if int(embeddings.shape[0]) != len(names):
        raise ValueError(
            f"names/embeddings row mismatch in {path}: "
            f"{len(names)} names vs {tuple(embeddings.shape)} embeddings."
        )
    return table


def _table_fields(table: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    names = [str(name) for name in table["names"]]
    prompts_raw = table.get("prompt_texts", table.get("phrases", names))
    categories_raw = table.get("categories")
    if not isinstance(prompts_raw, list) or len(prompts_raw) != len(names):
        prompts = list(names)
    else:
        prompts = [str(prompt) for prompt in prompts_raw]
    if not isinstance(categories_raw, list) or len(categories_raw) != len(names):
        categories = [humanize_motion_name(name) for name in names]
    else:
        categories = [str(category) for category in categories_raw]
    return names, prompts, categories


def normalized_embeddings(table: dict[str, Any]) -> torch.Tensor:
    """Return unit-normalized embedding rows from a validated table."""
    embeddings = table["embeddings"].to(dtype=torch.float32)
    return F.normalize(embeddings, dim=-1, eps=1.0e-12)


def cosine_matrix(table: dict[str, Any]) -> torch.Tensor:
    embeddings = normalized_embeddings(table)
    return embeddings @ embeddings.T


def _cosine_stats(cosine: torch.Tensor) -> dict[str, float | None]:
    n = int(cosine.shape[0])
    if n <= 1:
        return {
            "offdiag_min": None,
            "offdiag_max": None,
            "offdiag_mean": None,
            "offdiag_std": None,
            "offdiag_p05": None,
            "offdiag_p50": None,
            "offdiag_p95": None,
        }
    mask = ~torch.eye(n, dtype=torch.bool)
    values = cosine[mask].to(dtype=torch.float32)
    return {
        "offdiag_min": float(values.min().item()),
        "offdiag_max": float(values.max().item()),
        "offdiag_mean": float(values.mean().item()),
        "offdiag_std": float(values.std(unbiased=False).item()),
        "offdiag_p05": float(torch.quantile(values, 0.05).item()),
        "offdiag_p50": float(torch.quantile(values, 0.50).item()),
        "offdiag_p95": float(torch.quantile(values, 0.95).item()),
    }


def _nearest_neighbors(
    names: list[str],
    prompts: list[str],
    categories: list[str],
    cosine: torch.Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    n = len(names)
    rows: list[dict[str, Any]] = []
    if n <= 1:
        return [
            {
                "name": names[0],
                "category": categories[0],
                "prompt": prompts[0],
                "neighbors": [],
            }
        ]
    k = min(max(int(top_k), 1), n - 1)
    scores = cosine.clone()
    scores.fill_diagonal_(-float("inf"))
    values, indices = torch.topk(scores, k=k, dim=-1)
    for row, name in enumerate(names):
        neighbors = []
        for value, index in zip(values[row].tolist(), indices[row].tolist()):
            neighbors.append(
                {
                    "name": names[int(index)],
                    "category": categories[int(index)],
                    "prompt": prompts[int(index)],
                    "cosine": float(value),
                }
            )
        rows.append(
            {
                "name": name,
                "category": categories[row],
                "prompt": prompts[row],
                "neighbors": neighbors,
            }
        )
    return rows


def _category_nn_accuracy(
    categories: list[str],
    cosine: torch.Tensor,
) -> float | None:
    counts = Counter(categories)
    eligible = [idx for idx, category in enumerate(categories) if counts[category] > 1]
    if not eligible:
        return None
    scores = cosine.clone()
    scores.fill_diagonal_(-float("inf"))
    nearest = torch.argmax(scores, dim=-1).tolist()
    correct = sum(
        1 for idx in eligible if categories[int(nearest[idx])] == categories[idx]
    )
    return float(correct / len(eligible))


def _intra_inter_category_cosine(
    categories: list[str],
    cosine: torch.Tensor,
) -> dict[str, float | None]:
    same: list[float] = []
    different: list[float] = []
    n = len(categories)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(cosine[i, j].item())
            if categories[i] == categories[j]:
                same.append(value)
            else:
                different.append(value)
    return {
        "intra_category_mean": _mean_or_none(same),
        "inter_category_mean": _mean_or_none(different),
        "intra_minus_inter_mean": (
            None
            if not same or not different
            else float(_mean_or_none(same) - _mean_or_none(different))
        ),
    }


def _paraphrase_stability(
    prompts: list[str],
    categories: list[str],
    cosine: torch.Tensor,
) -> dict[str, float | int | None]:
    same_prompt: list[float] = []
    same_category: list[float] = []
    n = len(prompts)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(cosine[i, j].item())
            if prompts[i] == prompts[j]:
                same_prompt.append(value)
            if categories[i] == categories[j]:
                same_category.append(value)
    category_prompt_counts: dict[str, set[str]] = defaultdict(set)
    for prompt, category in zip(prompts, categories):
        category_prompt_counts[category].add(prompt)
    multi_prompt_categories = sum(
        1 for values in category_prompt_counts.values() if len(values) > 1
    )
    return {
        "same_prompt_pair_count": len(same_prompt),
        "same_prompt_cosine_mean": _mean_or_none(same_prompt),
        "same_prompt_cosine_min": min(same_prompt) if same_prompt else None,
        "same_category_pair_count": len(same_category),
        "same_category_cosine_mean": _mean_or_none(same_category),
        "same_category_cosine_min": min(same_category) if same_category else None,
        "categories_with_multiple_prompts": multi_prompt_categories,
    }


def _category_centroids(
    embeddings: torch.Tensor,
    categories: list[str],
) -> dict[str, torch.Tensor]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        buckets[category].append(index)
    centroids: dict[str, torch.Tensor] = {}
    for category, indices in buckets.items():
        centroid = embeddings[indices].mean(dim=0, keepdim=True)
        centroids[category] = F.normalize(centroid, dim=-1, eps=1.0e-12).squeeze(0)
    return centroids


def _centroid_cosine(
    centroids: dict[str, torch.Tensor],
    lhs: str,
    rhs: str,
) -> float | None:
    if lhs not in centroids or rhs not in centroids:
        return None
    return float(torch.dot(centroids[lhs], centroids[rhs]).item())


def _expected_ordering_checks(
    embeddings: torch.Tensor,
    categories: list[str],
) -> dict[str, Any]:
    centroids = _category_centroids(embeddings, categories)
    pair_scores = {
        "walk_run": _centroid_cosine(centroids, "walk", "run"),
        "walk_sprint": _centroid_cosine(centroids, "walk", "sprint"),
        "walk_fight": _centroid_cosine(centroids, "walk", "fight"),
        "walk_fall_and_get_up": _centroid_cosine(centroids, "walk", "fall and get up"),
        "run_sprint": _centroid_cosine(centroids, "run", "sprint"),
        "run_fight": _centroid_cosine(centroids, "run", "fight"),
        "run_fall_and_get_up": _centroid_cosine(centroids, "run", "fall and get up"),
    }
    checks: dict[str, Any] = {"pair_scores": pair_scores}

    walk_run = pair_scores["walk_run"]
    walk_sprint = pair_scores["walk_sprint"]
    walk_fight = pair_scores["walk_fight"]
    walk_fall = pair_scores["walk_fall_and_get_up"]
    checks["walk_near_locomotion"] = (
        None
        if None in (walk_run, walk_sprint, walk_fight, walk_fall)
        else bool(min(walk_run, walk_sprint) > max(walk_fight, walk_fall))
    )

    run_sprint = pair_scores["run_sprint"]
    run_fight = pair_scores["run_fight"]
    run_fall = pair_scores["run_fall_and_get_up"]
    checks["run_near_sprint"] = (
        None
        if None in (run_sprint, run_fight, run_fall)
        else bool(run_sprint > max(run_fight, run_fall))
    )

    fall_targets = [
        category
        for category in ("walk", "run", "sprint", "fight", "fight and sports")
        if category in centroids and "fall and get up" in centroids
    ]
    fall_scores = {
        category: _centroid_cosine(centroids, "fall and get up", category)
        for category in fall_targets
    }
    max_fall_score = max(fall_scores.values()) if fall_scores else None
    checks["fall_separated_at_0_75"] = (
        None if max_fall_score is None else bool(max_fall_score < 0.75)
    )
    checks["fall_pair_scores"] = fall_scores
    return checks


def summarize_table(
    table: dict[str, Any],
    *,
    label: str,
    top_k: int = 5,
) -> dict[str, Any]:
    names, prompts, categories = _table_fields(table)
    embeddings = normalized_embeddings(table)
    cosine = embeddings @ embeddings.T
    return {
        "label": label,
        "num_rows": len(names),
        "embedding_dim": int(embeddings.shape[-1]),
        "backend": table.get("backend"),
        "model": table.get("embedding_model", table.get("model")),
        "prompt_tier": table.get(
            "prompt_tier",
            "category" if not table.get("raw_names") else "raw_name",
        ),
        "manifest": table.get("manifest"),
        "manifest_sha256": table.get("manifest_sha256"),
        "num_categories": len(set(categories)),
        "category_counts": dict(sorted(Counter(categories).items())),
        "cosine_stats": _cosine_stats(cosine),
        "nearest_neighbors": _nearest_neighbors(
            names,
            prompts,
            categories,
            cosine,
            top_k=top_k,
        ),
        "category_nn_accuracy": _category_nn_accuracy(categories, cosine),
        "category_cosine": _intra_inter_category_cosine(categories, cosine),
        "paraphrase_stability": _paraphrase_stability(
            prompts,
            categories,
            cosine,
        ),
        "expected_ordering_checks": _expected_ordering_checks(
            embeddings,
            categories,
        ),
    }


def summarize_table_path(path: Path, *, top_k: int = 5) -> dict[str, Any]:
    return summarize_table(_load_table(path), label=str(path), top_k=top_k)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_summary(summaries: list[dict[str, Any]]) -> str:
    lines = [
        (
            "| table | rows | dim | tier | backend | model | categories | "
            "nn acc | intra | inter |"
        ),
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        category_cosine = summary["category_cosine"]
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(str(summary["label"])).name,
                    str(summary["num_rows"]),
                    str(summary["embedding_dim"]),
                    str(summary.get("prompt_tier") or ""),
                    str(summary.get("backend") or ""),
                    str(summary.get("model") or ""),
                    str(summary["num_categories"]),
                    _fmt(summary.get("category_nn_accuracy")),
                    _fmt(category_cosine.get("intra_category_mean")),
                    _fmt(category_cosine.get("inter_category_mean")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one or more offline SkillCommander embedding tables."
    )
    parser.add_argument("tables", nargs="+", help="Embedding table .pt paths.")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional JSON summary output path.",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="Optional Markdown summary output path.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Nearest neighbors to record per motion.",
    )
    args = parser.parse_args()

    summaries = [
        summarize_table_path(Path(path), top_k=int(args.top_k)) for path in args.tables
    ]
    payload = {"tables": summaries}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
        print(f"[audit-language] wrote JSON: {output_json}")
    else:
        print(text)
    if args.output_md is not None:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown_summary(summaries), encoding="utf-8")
        print(f"[audit-language] wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
