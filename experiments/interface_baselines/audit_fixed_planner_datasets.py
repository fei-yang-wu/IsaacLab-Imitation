#!/usr/bin/env python3
"""Audit fixed planner finetuning sample directories.

Use this before training planner variants to freeze the exact achieved-state
dataset. Run once per representation with all planner variants that should share
the same samples and pass ``--require_same_hash``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SAMPLE_GLOBS = ("sample_step_*.pt", "sample_chunk_*.pt")
LATENT_KEYS = ("planner_state", "expert_planner_state", "lang", "z_target", "traj_rank")
INTERFACE_KEYS = ("planner_state", "expert_planner_state", "target", "traj_rank")
_UNSET = object()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash and validate fixed planner finetuning sample directories."
    )
    parser.add_argument("samples_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--require_same_hash",
        action="store_true",
        default=False,
        help="Fail unless every provided directory has the same dataset hash.",
    )
    parser.add_argument(
        "--write_manifest",
        action="store_true",
        default=False,
        help="Write dataset_manifest.json inside each sample directory.",
    )
    parser.add_argument(
        "--manifest_name",
        default="dataset_manifest.json",
        help="Manifest filename used with --write_manifest.",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path for the combined audit JSON.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_paths(samples_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in SAMPLE_GLOBS:
        paths.extend(samples_dir.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No sample_step_*.pt or sample_chunk_*.pt files found in {samples_dir}."
        )
    return paths


def _metadata_signature(metadata: Any) -> str | None:
    if metadata is None:
        return None
    return json.dumps(metadata, sort_keys=True, default=str)


def _tensor_shape(value: Any, *, key: str, path: Path) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} key {key!r} must be a tensor.")
    return tuple(int(dim) for dim in value.shape)


def _row_count(sample: dict[str, Any], *, key: str, path: Path) -> int:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} key {key!r} must be a tensor.")
    if value.ndim == 0:
        raise ValueError(f"{path} key {key!r} must have a batch dimension.")
    return int(value.reshape(-1).shape[0]) if key == "traj_rank" else int(value.shape[0])


def _sample_kind(sample: dict[str, Any], *, path: Path) -> tuple[str, tuple[str, ...]]:
    if all(key in sample for key in LATENT_KEYS):
        return "latent_skill", LATENT_KEYS
    if all(key in sample for key in INTERFACE_KEYS):
        metadata = sample.get("metadata")
        interface = "interface"
        if isinstance(metadata, dict) and metadata.get("interface"):
            interface = str(metadata["interface"])
        return interface, INTERFACE_KEYS
    missing_latent = [key for key in LATENT_KEYS if key not in sample]
    missing_interface = [key for key in INTERFACE_KEYS if key not in sample]
    raise KeyError(
        f"{path} is not a recognized latent or interface sample. "
        f"Missing latent keys={missing_latent}; missing interface keys={missing_interface}."
    )


def _audit_dir(samples_dir: Path) -> dict[str, Any]:
    samples_dir = samples_dir.expanduser().resolve()
    paths = _sample_paths(samples_dir)
    file_rows: list[dict[str, Any]] = []
    kind: str | None = None
    key_shapes: dict[str, tuple[int, ...]] = {}
    metadata_signature: str | None | object = _UNSET
    sample_rows = 0
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"{path} must contain a dict sample payload.")
        sample_kind, required_keys = _sample_kind(sample, path=path)
        if kind is None:
            kind = sample_kind
        elif sample_kind != kind:
            raise ValueError(
                f"{samples_dir} mixes sample kinds {kind!r} and {sample_kind!r}."
            )
        row_count = _row_count(sample, key=required_keys[0], path=path)
        for key in required_keys:
            if _row_count(sample, key=key, path=path) != row_count:
                raise ValueError(f"{path} key {key!r} row count mismatch.")
            shape = _tensor_shape(sample[key], key=key, path=path)
            if key not in key_shapes:
                key_shapes[key] = shape
            elif len(shape) != len(key_shapes[key]) or shape[1:] != key_shapes[key][1:]:
                raise ValueError(
                    f"{path} key {key!r} trailing shape {shape[1:]} does not match "
                    f"{key_shapes[key][1:]}."
                )
        current_metadata = _metadata_signature(sample.get("metadata"))
        if metadata_signature is _UNSET:
            metadata_signature = current_metadata
        elif current_metadata != metadata_signature:
            raise ValueError(f"{samples_dir} contains mixed sample metadata.")
        sample_rows += row_count
        file_rows.append(
            {
                "file": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "rows": int(row_count),
            }
        )
    dataset_hash = hashlib.sha256(
        json.dumps(file_rows, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "samples_dir": str(samples_dir),
        "kind": kind,
        "dataset_hash": dataset_hash,
        "sample_file_count": len(file_rows),
        "sample_row_count": int(sample_rows),
        "tensor_shapes": {key: list(shape) for key, shape in sorted(key_shapes.items())},
        "metadata_present": metadata_signature not in (None, _UNSET),
        "files": file_rows,
    }


def main() -> None:
    args = _parse_args()
    summaries = [_audit_dir(path) for path in args.samples_dirs]
    if args.require_same_hash:
        hashes = {summary["dataset_hash"] for summary in summaries}
        if len(hashes) != 1:
            raise SystemExit(
                "Dataset hashes differ: "
                + json.dumps(
                    {
                        summary["samples_dir"]: summary["dataset_hash"]
                        for summary in summaries
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    if args.write_manifest:
        for summary in summaries:
            manifest_path = Path(summary["samples_dir"]) / str(args.manifest_name)
            manifest_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    payload = {"datasets": summaries}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.json_out.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
