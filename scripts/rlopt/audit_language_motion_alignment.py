#!/usr/bin/env python3
"""Compare language embedding geometry with saved skill/motion geometry.

This script is intentionally offline-only. It never launches Isaac or computes
oracle z targets itself; provide a saved centroid/sample file when motion-space
alignment should be measured.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from audit_language_embeddings import (
    _load_table,
    _table_fields,
    normalized_embeddings,
    summarize_table,
)


def _as_name_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return None


def _first_tensor(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> torch.Tensor | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _align_named_matrix(
    matrix: torch.Tensor,
    row_names: list[str] | None,
    target_names: list[str],
    *,
    source: Path,
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a rank-2 z centroid matrix in {source}.")
    matrix = matrix.to(dtype=torch.float32)
    if row_names is None:
        if int(matrix.shape[0]) != len(target_names):
            raise ValueError(
                f"{source} has {matrix.shape[0]} rows but the embedding table "
                f"has {len(target_names)} names and no row names were provided."
            )
        return matrix
    if len(row_names) != int(matrix.shape[0]):
        raise ValueError(
            f"{source} row_names length {len(row_names)} does not match matrix "
            f"rows {matrix.shape[0]}."
        )
    name_to_row = {name: index for index, name in enumerate(row_names)}
    missing = [name for name in target_names if name not in name_to_row]
    if missing:
        raise ValueError(f"{source} is missing z centroids for names: {missing}")
    rows = [matrix[name_to_row[name]] for name in target_names]
    return torch.stack(rows, dim=0)


def _centroids_from_ranked_samples(
    z: torch.Tensor,
    traj_rank: torch.Tensor,
    target_names: list[str],
    *,
    source: Path,
) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"{source} z samples must have shape [N, D].")
    ranks = traj_rank.reshape(-1).to(dtype=torch.long)
    if int(ranks.numel()) != int(z.shape[0]):
        raise ValueError(f"{source} traj_rank length does not match z sample rows.")
    centroids: list[torch.Tensor] = []
    for rank, name in enumerate(target_names):
        mask = ranks == rank
        if not bool(mask.any().item()):
            raise ValueError(f"{source} has no z samples for rank {rank} ({name}).")
        centroids.append(z[mask].to(dtype=torch.float32).mean(dim=0))
    return torch.stack(centroids, dim=0)


def _centroids_from_named_samples(
    z: torch.Tensor,
    sample_names: list[str],
    target_names: list[str],
    *,
    source: Path,
) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"{source} z samples must have shape [N, D].")
    if len(sample_names) != int(z.shape[0]):
        raise ValueError(f"{source} sample name count does not match z sample rows.")
    buckets: dict[str, list[int]] = {name: [] for name in target_names}
    for index, name in enumerate(sample_names):
        if name in buckets:
            buckets[name].append(index)
    missing = [name for name, indices in buckets.items() if not indices]
    if missing:
        raise ValueError(f"{source} has no named z samples for: {missing}")
    return torch.stack(
        [
            z[indices].to(dtype=torch.float32).mean(dim=0)
            for indices in buckets.values()
        ],
        dim=0,
    )


def load_z_centroids(path: Path, target_names: list[str]) -> torch.Tensor:
    """Load and align saved z centroids or samples to embedding-table names."""
    source = path.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return _align_named_matrix(payload, None, target_names, source=source)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported z payload type in {source}: {type(payload)!r}")

    matrix = _first_tensor(
        payload,
        ("z_centroids", "centroids", "skill_centroids", "z_by_motion"),
    )
    row_names = (
        _as_name_list(payload.get("names"))
        or _as_name_list(payload.get("motion_names"))
        or _as_name_list(payload.get("row_names"))
    )
    if matrix is not None:
        return _align_named_matrix(matrix, row_names, target_names, source=source)

    z_samples = _first_tensor(payload, ("z", "z_samples", "skill_z"))
    if z_samples is None:
        raise ValueError(f"{source} must contain z_centroids/centroids or z/z_samples.")
    ranks = payload.get("traj_rank", payload.get("traj_ranks"))
    if isinstance(ranks, torch.Tensor):
        return _centroids_from_ranked_samples(
            z_samples,
            ranks,
            target_names,
            source=source,
        )
    sample_names = _as_name_list(payload.get("names")) or _as_name_list(
        payload.get("motion_names")
    )
    if sample_names is not None:
        return _centroids_from_named_samples(
            z_samples,
            sample_names,
            target_names,
            source=source,
        )
    raise ValueError(
        f"{source} z samples require either traj_rank/traj_ranks or per-row names."
    )


def _rank_1d(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(int(values.numel()), dtype=torch.float32)
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if int(x.numel()) < 2:
        return None
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if float(denom.item()) <= 1.0e-12:
        return None
    return float((x * y).sum().div(denom).item())


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    return _pearson(_rank_1d(x), _rank_1d(y))


def _topk_overlap(
    lhs_scores: torch.Tensor,
    rhs_scores: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, float | None]:
    n = int(lhs_scores.shape[0])
    if n <= 1:
        return {"top1_agreement": None, "topk_overlap": None}
    lhs = lhs_scores.clone()
    rhs = rhs_scores.clone()
    lhs.fill_diagonal_(-float("inf"))
    rhs.fill_diagonal_(-float("inf"))
    top1 = (
        (torch.argmax(lhs, dim=-1) == torch.argmax(rhs, dim=-1))
        .to(dtype=torch.float32)
        .mean()
    )
    k = min(max(int(top_k), 1), n - 1)
    lhs_topk = torch.topk(lhs, k=k, dim=-1).indices.tolist()
    rhs_topk = torch.topk(rhs, k=k, dim=-1).indices.tolist()
    overlaps = []
    for lhs_row, rhs_row in zip(lhs_topk, rhs_topk):
        overlaps.append(len(set(lhs_row).intersection(rhs_row)) / float(k))
    return {
        "top1_agreement": float(top1.item()),
        "topk_overlap": float(sum(overlaps) / len(overlaps)),
    }


def alignment_summary(
    table: dict[str, Any],
    z_centroids: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, Any]:
    names, _, _ = _table_fields(table)
    lang = normalized_embeddings(table)
    z = F.normalize(z_centroids.to(dtype=torch.float32), dim=-1, eps=1.0e-12)
    if int(z.shape[0]) != len(names):
        raise ValueError(
            f"z centroid rows {z.shape[0]} do not match embedding rows {len(names)}."
        )
    lang_cos = lang @ lang.T
    z_cos = z @ z.T
    mask = torch.triu(torch.ones(len(names), len(names), dtype=torch.bool), diagonal=1)
    lang_dist = 1.0 - lang_cos[mask]
    z_dist = 1.0 - z_cos[mask]
    overlap = _topk_overlap(lang_cos, z_cos, top_k=top_k)
    return {
        "num_rows": len(names),
        "z_dim": int(z.shape[-1]),
        "pairwise_spearman_distance": _spearman(lang_dist, z_dist),
        "pairwise_pearson_distance": _pearson(lang_dist, z_dist),
        **overlap,
    }


def _markdown(payload: dict[str, Any]) -> str:
    language = payload["language"]
    alignment = payload.get("motion_alignment")
    lines = [
        "| table | rows | dim | tier | z dim | spearman | top1 | topk |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.append(
        "| "
        + " | ".join(
            [
                Path(str(language["label"])).name,
                str(language["num_rows"]),
                str(language["embedding_dim"]),
                str(language.get("prompt_tier") or ""),
                "" if alignment is None else str(alignment["z_dim"]),
                (
                    ""
                    if alignment is None
                    else _fmt(alignment["pairwise_spearman_distance"])
                ),
                "" if alignment is None else _fmt(alignment["top1_agreement"]),
                "" if alignment is None else _fmt(alignment["topk_overlap"]),
            ]
        )
        + " |"
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit language embedding alignment with saved z geometry."
    )
    parser.add_argument("--embeddings", required=True, help="Embedding table .pt path.")
    parser.add_argument(
        "--z_centroids",
        type=str,
        default=None,
        help=(
            "Optional .pt file containing z_centroids/centroids or z samples. "
            "If omitted, only language-space diagnostics are emitted."
        ),
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    table_path = Path(args.embeddings).expanduser().resolve()
    table = _load_table(table_path)
    names, _, _ = _table_fields(table)
    payload: dict[str, Any] = {
        "language": summarize_table(table, label=str(table_path), top_k=args.top_k),
        "motion_alignment": None,
    }
    if args.z_centroids is None:
        payload["note"] = (
            "No --z_centroids file was provided; skipped motion-space alignment."
        )
    else:
        z_centroids = load_z_centroids(Path(args.z_centroids), names)
        payload["z_centroids"] = str(Path(args.z_centroids).expanduser().resolve())
        payload["motion_alignment"] = alignment_summary(
            table,
            z_centroids,
            top_k=int(args.top_k),
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
        print(f"[audit-language-motion] wrote JSON: {output_json}")
    else:
        print(text)
    if args.output_md is not None:
        output_md = Path(args.output_md).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_markdown(payload), encoding="utf-8")
        print(f"[audit-language-motion] wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
