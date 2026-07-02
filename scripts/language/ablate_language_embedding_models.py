#!/usr/bin/env python3
"""Ablate text embedding models for language-goal separability.

This script is intentionally offline with respect to Isaac. It reads the same
LaFAN1-style manifest and prompt tiers used by SkillCommander, embeds the
motion-goal texts with several model backends, then reuses the existing
language audit metrics plus extra goal-collision diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
RLOPT_SCRIPTS = REPO_ROOT / "scripts" / "rlopt"
if str(RLOPT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RLOPT_SCRIPTS))

from audit_language_embeddings import summarize_table  # noqa: E402
from build_language_goal_embeddings import load_motion_names  # noqa: E402
from language_prompts import (  # noqa: E402
    PROMPT_TIERS,
    humanize_motion_name,
    load_prompt_overrides,
    normalize_prompt_tier,
    resolve_prompt_for_motion,
)

DEFAULT_LANGUAGE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "lafan1"
    / "language"
    / "g1_lafan1_manifest.with_codex_storyboard_language_v1.json"
)
DEFAULT_BASE_MANIFEST = (
    REPO_ROOT / "data" / "lafan1" / "manifests" / "g1_lafan1_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "language_embedding_ablation"
DEFAULT_DUMMY_DIM = 384


@dataclass(frozen=True)
class ModelSpec:
    """Embedding model selection for one ablation row."""

    backend: str
    model: str
    alias: str
    embed_dim: int = DEFAULT_DUMMY_DIM


MODEL_SETS: dict[str, list[str]] = {
    "smoke": [
        "dummy:dummy-384",
        "sentence-transformer:all-MiniLM-L6-v2",
    ],
    "recommended": [
        "dummy:dummy-384",
        "sentence-transformer:all-MiniLM-L6-v2",
        "sentence-transformer:Qwen/Qwen3-Embedding-0.6B",
        "sentence-transformer:Qwen/Qwen3-Embedding-4B",
        "sentence-transformer:Qwen/Qwen3-Embedding-8B",
    ],
    "ollama": [
        "ollama:embeddinggemma",
        "ollama:qwen3-embedding",
        "ollama:all-minilm",
        "ollama:nomic-embed-text",
        "ollama:mxbai-embed-large",
    ],
}
MODEL_SETS["all"] = list(
    dict.fromkeys(MODEL_SETS["recommended"] + MODEL_SETS["ollama"])
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    slug = []
    for char in str(value):
        if char.isalnum():
            slug.append(char.lower())
        elif char in {".", "-", "_"}:
            slug.append(char)
        else:
            slug.append("_")
    text = "".join(slug).strip("._-")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "model"


def _dummy_embedding(phrase: str, dim: int, seed: int) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:{phrase}".encode("utf-8")).digest()
    generator = torch.Generator()
    generator.manual_seed(int.from_bytes(digest[:8], "little"))
    return torch.randn(dim, generator=generator, dtype=torch.float32)


def _parse_spec(text: str, *, dummy_dim: int) -> ModelSpec:
    if ":" not in text:
        raise ValueError(
            f"Model spec {text!r} must look like 'backend:model', "
            "for example 'sentence-transformer:all-MiniLM-L6-v2'."
        )
    backend, model = text.split(":", 1)
    backend = backend.strip()
    model = model.strip()
    if backend not in {"dummy", "sentence-transformer", "ollama"}:
        raise ValueError(
            f"Unknown backend {backend!r}; expected dummy, sentence-transformer, "
            "or ollama."
        )
    if not model:
        raise ValueError(f"Model spec {text!r} has an empty model name.")
    if backend == "dummy":
        dim = dummy_dim
        if model.startswith("dummy-"):
            suffix = model.removeprefix("dummy-")
            if suffix.isdigit():
                dim = int(suffix)
        return ModelSpec(
            backend=backend,
            model=model,
            alias=f"dummy_{dim}",
            embed_dim=dim,
        )
    prefix = "st" if backend == "sentence-transformer" else "ollama"
    return ModelSpec(
        backend=backend,
        model=model,
        alias=f"{prefix}_{_slug(model)}",
        embed_dim=dummy_dim,
    )


def _resolve_specs(
    model_set: str, extra_specs: list[str], dummy_dim: int
) -> list[ModelSpec]:
    raw_specs = list(MODEL_SETS[model_set])
    raw_specs.extend(extra_specs)
    specs: list[ModelSpec] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_specs:
        spec = _parse_spec(raw, dummy_dim=dummy_dim)
        key = (spec.backend, spec.model)
        if key not in seen:
            seen.add(key)
            specs.append(spec)
    return specs


def _resolve_manifest(path: str | None) -> Path:
    if path is not None:
        manifest = Path(path).expanduser().resolve()
    elif DEFAULT_LANGUAGE_MANIFEST.is_file():
        manifest = DEFAULT_LANGUAGE_MANIFEST
    else:
        manifest = DEFAULT_BASE_MANIFEST
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    return manifest


def _resolve_prompt_texts(
    *,
    manifest_path: Path,
    prompt_tier: str,
    prompt_json: str | None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    names = load_motion_names(manifest_path)
    categories = [humanize_motion_name(name) for name in names]
    manifest_overrides = load_prompt_overrides(manifest_path, prompt_tier)
    explicit_overrides = load_prompt_overrides(prompt_json, prompt_tier)
    prompt_overrides = dict(manifest_overrides)
    prompt_overrides.update(explicit_overrides)
    prompts = [
        resolve_prompt_for_motion(name, prompt_tier, prompt_overrides) for name in names
    ]
    metadata = {
        "manifest_prompt_count": len(manifest_overrides),
        "prompt_json_prompt_count": len(explicit_overrides),
        "unique_prompt_count": len(set(prompts)),
    }
    return names, categories, prompts, metadata


def _embed_dummy(phrases: list[str], *, embed_dim: int, seed: int) -> torch.Tensor:
    return torch.stack(
        [_dummy_embedding(phrase, embed_dim, seed) for phrase in phrases]
    )


def _embed_sentence_transformer(
    phrases: list[str],
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
    trust_remote_code: bool,
) -> torch.Tensor:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sentence-transformer backend requires sentence-transformers in "
            "the active Pixi environment."
        ) from exc

    model_kwargs: dict[str, Any] = {}
    if device:
        model_kwargs["device"] = device
    try:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
    except TypeError:
        model = SentenceTransformer(model_name, **model_kwargs)

    vectors = model.encode(
        phrases,
        batch_size=max(int(batch_size), 1),
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=len(phrases) > 8,
    )
    return torch.as_tensor(vectors, dtype=torch.float32)


def _chunked(values: list[str], size: int) -> list[list[str]]:
    size = max(int(size), 1)
    return [values[index : index + size] for index in range(0, len(values), size)]


def _ollama_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _embed_ollama(
    phrases: list[str],
    *,
    model_name: str,
    host: str,
    batch_size: int,
    timeout_s: float,
    dimensions: int | None,
) -> torch.Tensor:
    endpoint = host.rstrip("/") + "/api/embed"
    vectors: list[list[float]] = []
    for chunk in _chunked(phrases, batch_size):
        payload: dict[str, Any] = {
            "model": model_name,
            "input": chunk,
            "truncate": True,
        }
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=_ollama_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama {model_name!r} failed with HTTP {exc.code}: {body}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {endpoint}. Start ollama serve, "
                "or use a non-Ollama model set."
            ) from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(chunk):
            raise RuntimeError(
                f"Ollama response for {model_name!r} did not contain "
                f"{len(chunk)} embeddings."
            )
        vectors.extend(embeddings)
    return torch.as_tensor(vectors, dtype=torch.float32)


def _maybe_pull_ollama(spec: ModelSpec) -> None:
    if spec.backend != "ollama":
        return
    subprocess.run(["ollama", "pull", spec.model], check=True)


def _embed_unique_prompts(
    unique_prompts: list[str],
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
) -> torch.Tensor:
    if spec.backend == "dummy":
        return _embed_dummy(
            unique_prompts,
            embed_dim=spec.embed_dim,
            seed=int(args.seed),
        )
    if spec.backend == "sentence-transformer":
        return _embed_sentence_transformer(
            unique_prompts,
            model_name=spec.model,
            batch_size=int(args.batch_size),
            device=args.device,
            trust_remote_code=bool(args.trust_remote_code),
        )
    if spec.backend == "ollama":
        if bool(args.pull_ollama):
            _maybe_pull_ollama(spec)
        return _embed_ollama(
            unique_prompts,
            model_name=spec.model,
            host=str(args.ollama_host),
            batch_size=int(args.ollama_batch_size),
            timeout_s=float(args.ollama_timeout_s),
            dimensions=args.ollama_dimensions,
        )
    raise ValueError(f"Unsupported backend: {spec.backend}")


def _embedding_inputs_for_spec(
    prompts: list[str],
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    model_lower = spec.model.lower()
    if (
        not bool(args.disable_model_prompt_adapters)
        and spec.backend == "sentence-transformer"
        and "e5" in model_lower
    ):
        prefix = str(args.e5_prefix)
        return (
            [
                prompt
                if prompt.startswith(("query: ", "passage: "))
                else f"{prefix}{prompt}"
                for prompt in prompts
            ],
            {"embedding_input_adapter": "e5", "embedding_input_prefix": prefix},
        )
    return prompts, {"embedding_input_adapter": "none", "embedding_input_prefix": ""}


def _build_table(
    *,
    names: list[str],
    categories: list[str],
    prompts: list[str],
    manifest_path: Path,
    prompt_tier: str,
    prompt_json: str | None,
    prompt_metadata: dict[str, Any],
    spec: ModelSpec,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float]:
    embedding_inputs, adapter_metadata = _embedding_inputs_for_spec(
        prompts,
        spec=spec,
        args=args,
    )
    unique_prompts = list(dict.fromkeys(embedding_inputs))
    prompt_to_row = {phrase: row for row, phrase in enumerate(unique_prompts)}
    start = time.monotonic()
    prompt_matrix = _embed_unique_prompts(unique_prompts, spec=spec, args=args)
    elapsed_s = time.monotonic() - start
    prompt_matrix = F.normalize(
        prompt_matrix.to(dtype=torch.float32), dim=-1, eps=1e-12
    )
    embeddings = torch.stack(
        [prompt_matrix[prompt_to_row[p]] for p in embedding_inputs]
    )
    table = {
        "names": names,
        "phrases": prompts,
        "prompt_texts": prompts,
        "embedding_input_texts": embedding_inputs,
        "prompt_tier": prompt_tier,
        "categories": categories,
        "name_to_index": {name: index for index, name in enumerate(names)},
        "embeddings": embeddings.contiguous(),
        "embed_dim": int(embeddings.shape[-1]),
        "backend": spec.backend,
        "model": spec.model,
        "embedding_model": spec.model,
        "raw_names": prompt_tier == "raw_name",
        "normalized": True,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "prompt_json": (
            str(Path(prompt_json).expanduser().resolve())
            if prompt_json is not None
            else None
        ),
        "language_ablation": {
            "alias": spec.alias,
            "embed_elapsed_s": elapsed_s,
            **prompt_metadata,
            **adapter_metadata,
        },
    }
    return table, elapsed_s


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def _goal_collision_metrics(
    table: dict[str, Any],
    *,
    top_k: int,
    near_duplicate_cosine: float,
) -> dict[str, Any]:
    names = [str(name) for name in table["names"]]
    categories = [str(category) for category in table["categories"]]
    prompts = [str(prompt) for prompt in table["prompt_texts"]]
    embeddings = F.normalize(table["embeddings"].to(dtype=torch.float32), dim=-1)
    cosine = embeddings @ embeddings.T
    n = len(names)
    scores = cosine.clone()
    scores.fill_diagonal_(-float("inf"))
    nearest = torch.argmax(scores, dim=-1).tolist()

    wrong_top1 = []
    same_best_values: list[float] = []
    diff_best_values: list[float] = []
    positive_gaps: list[float] = []
    same_pair_values: list[float] = []
    diff_pair_values: list[float] = []
    cross_near_duplicates = []
    duplicate_prompt_pairs = []

    for i in range(n):
        nn = int(nearest[i])
        if categories[nn] != categories[i]:
            wrong_top1.append(
                {
                    "name": names[i],
                    "category": categories[i],
                    "neighbor": names[nn],
                    "neighbor_category": categories[nn],
                    "cosine": float(cosine[i, nn].item()),
                }
            )

        same = [
            float(cosine[i, j].item())
            for j in range(n)
            if i != j and categories[i] == categories[j]
        ]
        different = [
            float(cosine[i, j].item())
            for j in range(n)
            if categories[i] != categories[j]
        ]
        if same:
            same_best_values.append(max(same))
        if different:
            diff_best_values.append(max(different))
        if same and different:
            positive_gaps.append(max(same) - max(different))

    for i in range(n):
        for j in range(i + 1, n):
            value = float(cosine[i, j].item())
            if prompts[i] == prompts[j]:
                duplicate_prompt_pairs.append([names[i], names[j]])
            if categories[i] == categories[j]:
                same_pair_values.append(value)
            else:
                diff_pair_values.append(value)
                if value >= near_duplicate_cosine:
                    cross_near_duplicates.append(
                        {
                            "lhs": names[i],
                            "lhs_category": categories[i],
                            "rhs": names[j],
                            "rhs_category": categories[j],
                            "cosine": value,
                        }
                    )

    top_k = max(int(top_k), 1)
    top_k_wrong_counts = []
    if n > 1:
        k = min(top_k, n - 1)
        _, top_indices = torch.topk(scores, k=k, dim=-1)
        for i in range(n):
            wrong = sum(
                1
                for index in top_indices[i].tolist()
                if categories[int(index)] != categories[i]
            )
            top_k_wrong_counts.append(wrong)

    positive_count = sum(1 for gap in positive_gaps if gap > 0.0)
    return {
        "wrong_top1_count": len(wrong_top1),
        "wrong_top1": wrong_top1,
        "best_same_gt_best_different_rate": (
            None if not positive_gaps else positive_count / len(positive_gaps)
        ),
        "best_same_minus_best_different_mean": (
            None if not positive_gaps else sum(positive_gaps) / len(positive_gaps)
        ),
        "best_same_cosine_mean": (
            None
            if not same_best_values
            else sum(same_best_values) / len(same_best_values)
        ),
        "best_different_cosine_mean": (
            None
            if not diff_best_values
            else sum(diff_best_values) / len(diff_best_values)
        ),
        "inter_category_p95": _quantile(diff_pair_values, 0.95),
        "same_category_p05": _quantile(same_pair_values, 0.05),
        "cross_category_near_duplicate_count": len(cross_near_duplicates),
        "cross_category_near_duplicates": cross_near_duplicates,
        "duplicate_prompt_pair_count": len(duplicate_prompt_pairs),
        "duplicate_prompt_pairs": duplicate_prompt_pairs,
        "top_k": top_k,
        "top_k_wrong_mean": (
            None
            if not top_k_wrong_counts
            else sum(top_k_wrong_counts) / len(top_k_wrong_counts)
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _row_from_success(
    *,
    prompt_tier: str,
    spec: ModelSpec,
    table_path: Path,
    audit_json: Path,
    audit_md: Path,
    summary: dict[str, Any],
    goal_metrics: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    category_cosine = summary["category_cosine"]
    checks = summary["expected_ordering_checks"]
    nn_acc = summary.get("category_nn_accuracy")
    gap_rate = goal_metrics.get("best_same_gt_best_different_rate")
    margin = category_cosine.get("intra_minus_inter_mean")
    score_terms = [value for value in (nn_acc, gap_rate, margin) if value is not None]
    score = sum(float(value) for value in score_terms) / len(score_terms)
    return {
        "status": "ok",
        "score": score,
        "prompt_tier": prompt_tier,
        "backend": spec.backend,
        "model": spec.model,
        "alias": spec.alias,
        "rows": summary["num_rows"],
        "dim": summary["embedding_dim"],
        "nn_category_accuracy": nn_acc,
        "intra_category_mean": category_cosine.get("intra_category_mean"),
        "inter_category_mean": category_cosine.get("inter_category_mean"),
        "intra_minus_inter_mean": margin,
        "best_same_gt_best_different_rate": gap_rate,
        "best_same_minus_best_different_mean": goal_metrics.get(
            "best_same_minus_best_different_mean"
        ),
        "wrong_top1_count": goal_metrics.get("wrong_top1_count"),
        "cross_category_near_duplicate_count": goal_metrics.get(
            "cross_category_near_duplicate_count"
        ),
        "inter_category_p95": goal_metrics.get("inter_category_p95"),
        "same_category_p05": goal_metrics.get("same_category_p05"),
        "top_k_wrong_mean": goal_metrics.get("top_k_wrong_mean"),
        "run_near_sprint": checks.get("run_near_sprint"),
        "walk_near_locomotion": checks.get("walk_near_locomotion"),
        "fall_separated_at_0_75": checks.get("fall_separated_at_0_75"),
        "embed_elapsed_s": elapsed_s,
        "table": str(table_path),
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
        "error": "",
    }


def _row_from_failure(
    *,
    prompt_tier: str,
    spec: ModelSpec,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "score": "",
        "prompt_tier": prompt_tier,
        "backend": spec.backend,
        "model": spec.model,
        "alias": spec.alias,
        "rows": "",
        "dim": "",
        "nn_category_accuracy": "",
        "intra_category_mean": "",
        "inter_category_mean": "",
        "intra_minus_inter_mean": "",
        "best_same_gt_best_different_rate": "",
        "best_same_minus_best_different_mean": "",
        "wrong_top1_count": "",
        "cross_category_near_duplicate_count": "",
        "inter_category_p95": "",
        "same_category_p05": "",
        "top_k_wrong_mean": "",
        "run_near_sprint": "",
        "walk_near_locomotion": "",
        "fall_separated_at_0_75": "",
        "embed_elapsed_s": "",
        "table": "",
        "audit_json": "",
        "audit_md": "",
        "error": str(error),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "score",
        "prompt_tier",
        "backend",
        "model",
        "alias",
        "rows",
        "dim",
        "nn_category_accuracy",
        "intra_category_mean",
        "inter_category_mean",
        "intra_minus_inter_mean",
        "best_same_gt_best_different_rate",
        "best_same_minus_best_different_mean",
        "wrong_top1_count",
        "cross_category_near_duplicate_count",
        "inter_category_p95",
        "same_category_p05",
        "top_k_wrong_mean",
        "run_near_sprint",
        "walk_near_locomotion",
        "fall_separated_at_0_75",
        "embed_elapsed_s",
        "table",
        "audit_json",
        "audit_md",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    lines = [
        "# Language Embedding Model Ablation",
        "",
        f"- manifest: `{metadata['manifest']}`",
        f"- prompt tiers: `{', '.join(metadata['prompt_tiers'])}`",
        f"- models requested: `{metadata['model_count']}`",
        f"- near-duplicate threshold: `{metadata['near_duplicate_cosine']}`",
        "",
        (
            "| rank | tier | backend | model | nn acc | intra | inter | "
            "margin | gap rate | wrong top1 | xcat near dup | run/sprint | "
            "fall sep |"
        ),
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(ok_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["prompt_tier"]),
                    str(row["backend"]),
                    str(row["model"]),
                    _fmt(row["nn_category_accuracy"]),
                    _fmt(row["intra_category_mean"]),
                    _fmt(row["inter_category_mean"]),
                    _fmt(row["intra_minus_inter_mean"]),
                    _fmt(row["best_same_gt_best_different_rate"]),
                    _fmt(row["wrong_top1_count"]),
                    _fmt(row["cross_category_near_duplicate_count"]),
                    str(row["run_near_sprint"]),
                    str(row["fall_separated_at_0_75"]),
                ]
            )
            + " |"
        )
    if failed_rows:
        lines.extend(["", "## Failed Or Skipped", ""])
        lines.append("| tier | backend | model | error |")
        lines.append("| --- | --- | --- | --- |")
        for row in failed_rows:
            error = str(row["error"]).replace("\n", " ")
            lines.append(
                f"| {row['prompt_tier']} | {row['backend']} | "
                f"{row['model']} | {error} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sort_rows(rows: list[dict[str, Any]], *, rank_by: str) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, float, float, float]:
        if row["status"] != "ok":
            return (1, -1.0, -1.0, -1.0)
        if rank_by == "cosine":
            return (
                0,
                -float(row["intra_minus_inter_mean"] or 0.0),
                float(row["inter_category_mean"] or 0.0),
                -float(row["best_same_minus_best_different_mean"] or 0.0),
            )
        if rank_by == "balanced":
            return (
                0,
                -float(row["score"] or 0.0),
                -float(row["intra_minus_inter_mean"] or 0.0),
                -float(row["nn_category_accuracy"] or 0.0),
            )
        return (
            0,
            -float(row["nn_category_accuracy"] or 0.0),
            -float(row["best_same_gt_best_different_rate"] or 0.0),
            -float(row["intra_minus_inter_mean"] or 0.0),
        )

    return sorted(rows, key=key)


def _run_one(
    *,
    output_dir: Path,
    manifest_path: Path,
    prompt_tier: str,
    prompt_json: str | None,
    spec: ModelSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    names, categories, prompts, prompt_metadata = _resolve_prompt_texts(
        manifest_path=manifest_path,
        prompt_tier=prompt_tier,
        prompt_json=prompt_json,
    )
    table_name = f"lafan1_{prompt_tier}_{spec.alias}.pt"
    table_path = output_dir / "tables" / table_name
    audit_json = output_dir / "audits" / table_name.replace(".pt", ".audit.json")
    audit_md = output_dir / "audits" / table_name.replace(".pt", ".audit.md")

    if bool(args.skip_existing) and table_path.is_file() and audit_json.is_file():
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        summary = payload["tables"][0]
        goal_metrics = payload["goal_metrics"]
        elapsed_s = float(payload.get("embed_elapsed_s", 0.0))
    else:
        table, elapsed_s = _build_table(
            names=names,
            categories=categories,
            prompts=prompts,
            manifest_path=manifest_path,
            prompt_tier=prompt_tier,
            prompt_json=prompt_json,
            prompt_metadata=prompt_metadata,
            spec=spec,
            args=args,
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(table, table_path)
        summary = summarize_table(table, label=str(table_path), top_k=int(args.top_k))
        goal_metrics = _goal_collision_metrics(
            table,
            top_k=int(args.top_k),
            near_duplicate_cosine=float(args.near_duplicate_cosine),
        )
        _write_json(
            audit_json,
            {
                "tables": [summary],
                "goal_metrics": goal_metrics,
                "embed_elapsed_s": elapsed_s,
            },
        )
        audit_md.parent.mkdir(parents=True, exist_ok=True)
        audit_md.write_text(
            "| table | rows | dim | tier | backend | model | nn acc | intra | inter | margin | gap rate | wrong top1 |\n"
            "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            f"| {table_path.name} | {summary['num_rows']} | "
            f"{summary['embedding_dim']} | {prompt_tier} | {spec.backend} | "
            f"{spec.model} | {_fmt(summary.get('category_nn_accuracy'))} | "
            f"{_fmt(summary['category_cosine'].get('intra_category_mean'))} | "
            f"{_fmt(summary['category_cosine'].get('inter_category_mean'))} | "
            f"{_fmt(summary['category_cosine'].get('intra_minus_inter_mean'))} | "
            f"{_fmt(goal_metrics.get('best_same_gt_best_different_rate'))} | "
            f"{_fmt(goal_metrics.get('wrong_top1_count'))} |\n",
            encoding="utf-8",
        )

    return _row_from_success(
        prompt_tier=prompt_tier,
        spec=spec,
        table_path=table_path,
        audit_json=audit_json,
        audit_md=audit_md,
        summary=summary,
        goal_metrics=goal_metrics,
        elapsed_s=elapsed_s,
    )


def _parse_prompt_tiers(values: list[str]) -> list[str]:
    tiers: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                tiers.append(normalize_prompt_tier(part))
    return list(dict.fromkeys(tiers))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit multiple language-goal embedding models for "
            "goal separability."
        )
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Manifest with motion names and optional language fields. Defaults "
            "to the captioned LaFAN1 language manifest when present."
        ),
    )
    parser.add_argument(
        "--prompt_tiers",
        nargs="+",
        default=["attribute_text"],
        help=f"Prompt tier(s) to evaluate. Choices: {', '.join(PROMPT_TIERS)}.",
    )
    parser.add_argument(
        "--prompt_json",
        default=None,
        help="Optional prompt override JSON, same format as build_language_goal_embeddings.py.",
    )
    parser.add_argument(
        "--model_set",
        default="smoke",
        choices=sorted(MODEL_SETS),
        help=(
            "Registered model set. Use 'smoke' for dummy+MiniLM, "
            "'recommended' for local sentence-transformer ablations, "
            "'ollama' for Ollama embeddings, or 'all' for both."
        ),
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help=(
            "Additional backend:model spec. Examples: "
            "sentence-transformer:BAAI/bge-large-en-v1.5, "
            "ollama:nomic-embed-text."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to outputs/language_embedding_ablation/<timestamp>.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--near_duplicate_cosine",
        type=float,
        default=0.95,
        help="Flag cross-category pairs above this cosine as goal collisions.",
    )
    parser.add_argument(
        "--rank_by",
        choices=("cosine", "accuracy", "balanced"),
        default="cosine",
        help=(
            "How to rank successful rows in the summary. 'cosine' prioritizes "
            "intra-minus-inter margin and low inter-category cosine."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device for sentence-transformers.",
    )
    parser.add_argument(
        "--e5_prefix",
        default="passage: ",
        help=(
            "Prefix automatically added to E5-family sentence-transformer inputs "
            "unless --disable_model_prompt_adapters is set."
        ),
    )
    parser.add_argument(
        "--disable_model_prompt_adapters",
        action="store_true",
        help="Disable small model-family input adapters such as the E5 prefix.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to sentence-transformer model loading.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dummy_dim", type=int, default=DEFAULT_DUMMY_DIM)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing per-model tables/audits in the output directory.",
    )
    parser.add_argument(
        "--no_skip_unavailable",
        action="store_true",
        help="Fail immediately when a model cannot be loaded or embedded.",
    )
    parser.add_argument(
        "--allow_zero_success",
        action="store_true",
        help="Exit 0 even if every requested model fails or is unavailable.",
    )
    parser.add_argument(
        "--ollama_host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host for /api/embed.",
    )
    parser.add_argument("--ollama_batch_size", type=int, default=32)
    parser.add_argument("--ollama_timeout_s", type=float, default=300.0)
    parser.add_argument(
        "--ollama_dimensions",
        type=int,
        default=None,
        help="Optional Ollama /api/embed dimensions field.",
    )
    parser.add_argument(
        "--pull_ollama",
        action="store_true",
        help="Run 'ollama pull <model>' before each Ollama model.",
    )
    args = parser.parse_args()

    manifest_path = _resolve_manifest(args.manifest)
    prompt_tiers = _parse_prompt_tiers(args.prompt_tiers)
    specs = _resolve_specs(args.model_set, args.spec, int(args.dummy_dim))
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / _now_stamp()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[language-ablation] manifest: {manifest_path}")
    print(f"[language-ablation] output:   {output_dir}")
    print(f"[language-ablation] tiers:    {', '.join(prompt_tiers)}")
    print(
        "[language-ablation] models:   "
        + ", ".join(f"{spec.backend}:{spec.model}" for spec in specs)
    )

    rows: list[dict[str, Any]] = []
    for prompt_tier in prompt_tiers:
        for spec in specs:
            print(
                f"[language-ablation] running {prompt_tier} {spec.backend}:{spec.model}"
            )
            try:
                row = _run_one(
                    output_dir=output_dir,
                    manifest_path=manifest_path,
                    prompt_tier=prompt_tier,
                    prompt_json=args.prompt_json,
                    spec=spec,
                    args=args,
                )
            except Exception as exc:
                if bool(args.no_skip_unavailable):
                    raise
                row = _row_from_failure(
                    prompt_tier=prompt_tier,
                    spec=spec,
                    error=exc,
                )
                print(f"[language-ablation] skipped {spec.backend}:{spec.model}: {exc}")
            rows.append(row)

    rows = _sort_rows(rows, rank_by=str(args.rank_by))
    metadata = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "prompt_tiers": prompt_tiers,
        "model_set": args.model_set,
        "model_count": len(specs),
        "near_duplicate_cosine": float(args.near_duplicate_cosine),
        "rank_by": str(args.rank_by),
        "output_dir": str(output_dir),
    }
    report = {"metadata": metadata, "rows": rows}
    _write_json(output_dir / "language_embedding_ablation_summary.json", report)
    _write_csv(output_dir / "language_embedding_ablation_summary.csv", rows)
    _write_markdown(
        output_dir / "language_embedding_ablation_summary.md",
        rows,
        metadata,
    )
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(
        f"[language-ablation] complete: {ok_count}/{len(rows)} succeeded. "
        f"Summary: {output_dir / 'language_embedding_ablation_summary.md'}"
    )
    if ok_count == 0 and not bool(args.allow_zero_success):
        raise SystemExit(
            "No embedding models completed successfully. See the summary report "
            "for skipped models and errors."
        )


if __name__ == "__main__":
    main()
