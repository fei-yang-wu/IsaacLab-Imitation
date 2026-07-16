#!/usr/bin/env python3
"""Compare two BONES-SEED manifests and verify final Phase-5 provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--language-embeddings", type=Path, required=True)
    parser.add_argument("--preparation-record", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entries(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _json(manifest_path)
    entries = manifest.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no LAFAN1-style entries: {manifest_path}")
    result: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        raw_path = Path(str(entry.get("path", ""))).expanduser()
        path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        path = path.resolve()
        if not name or not path.is_file():
            raise FileNotFoundError(f"Invalid manifest entry {name!r}: {path}")
        result.append({"name": name, "path": path, "sha256": _sha256(path)})
    names = [entry["name"] for entry in result]
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest contains duplicate names: {manifest_path}")
    return result


def _aggregate_hash(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["name"])):
        digest.update(str(entry["name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    reference_manifest = args.reference_manifest.expanduser().resolve()
    candidate_manifest = args.candidate_manifest.expanduser().resolve()
    language_path = args.language_embeddings.expanduser().resolve()
    preparation_path = args.preparation_record.expanduser().resolve()
    output = args.output_json.expanduser().resolve()
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    reference = _entries(reference_manifest)
    candidate = _entries(candidate_manifest)
    reference_by_name = {str(entry["name"]): entry for entry in reference}
    candidate_by_name = {str(entry["name"]): entry for entry in candidate}
    reference_names = [str(entry["name"]) for entry in reference]
    candidate_names = [str(entry["name"]) for entry in candidate]
    require(
        set(reference_by_name) == set(candidate_by_name),
        "reference and candidate motion sets differ",
    )
    require(reference_names == candidate_names, "manifest motion order differs")
    differing_names = [
        name
        for name in sorted(set(reference_by_name) & set(candidate_by_name))
        if reference_by_name[name]["sha256"] != candidate_by_name[name]["sha256"]
    ]
    require(not differing_names, "one or more NPZ files differ")

    language = torch.load(language_path, map_location="cpu", weights_only=False)
    language_names = [str(value) for value in language.get("names", [])]
    embeddings = language.get("embeddings")
    require(language_names == candidate_names, "language table order differs")
    require(
        isinstance(embeddings, torch.Tensor)
        and embeddings.ndim == 2
        and int(embeddings.shape[0]) == len(candidate_names),
        "language embedding shape does not match the candidate manifest",
    )

    preparation = _json(preparation_path)
    candidate_manifest_sha = _sha256(candidate_manifest)
    require(preparation.get("status") == "complete", "preparation is not complete")
    require(
        preparation.get("artifacts", {}).get("manifest_sha256")
        == candidate_manifest_sha,
        "preparation record does not match the candidate manifest",
    )
    prepared_files = preparation.get("artifacts", {}).get("npz_files", [])
    prepared_hashes = {
        str(item.get("name")): str(item.get("npz_sha256"))
        for item in prepared_files
        if isinstance(item, dict)
    }
    require(
        prepared_hashes
        == {name: str(item["sha256"]) for name, item in candidate_by_name.items()},
        "preparation record does not cover the candidate NPZ hashes",
    )

    report = {
        "passed": not errors,
        "errors": errors,
        "reference_manifest": {
            "path": str(reference_manifest),
            "sha256": _sha256(reference_manifest),
        },
        "candidate_manifest": {
            "path": str(candidate_manifest),
            "sha256": candidate_manifest_sha,
        },
        "preparation_record": {
            "path": str(preparation_path),
            "sha256": _sha256(preparation_path),
            "status": preparation.get("status"),
        },
        "language_embeddings": {
            "path": str(language_path),
            "sha256": _sha256(language_path),
            "shape": list(embeddings.shape)
            if isinstance(embeddings, torch.Tensor)
            else None,
        },
        "motion_count": len(candidate),
        "exact_name_order_match": reference_names == candidate_names,
        "byte_identical_npz_count": len(candidate) - len(differing_names),
        "differing_motion_names": differing_names,
        "reference_npz_aggregate_sha256": _aggregate_hash(reference),
        "candidate_npz_aggregate_sha256": _aggregate_hash(candidate),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(f"[FAIL] {error}" for error in errors))
    print(f"[PASS] BONES-SEED export comparison: {output}")


if __name__ == "__main__":
    main()
