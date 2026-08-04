#!/usr/bin/env python3
"""Write an N-motion subset manifest without changing the source manifest.

The subset keeps each selected entry byte-identical apart from resolving its
NPZ path absolute, so the subset can live outside the source tree. Selection
provenance (source manifest path and SHA-256, requested names, command) is
recorded under ``metadata.selection``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--motion_names",
        nargs="+",
        required=True,
        help="Ordered motion names to keep; each must match exactly one entry.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        default=False,
        help="Replace an existing output manifest instead of refusing.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_against(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def write_motion_subset_manifest(
    manifest: Path,
    *,
    motion_names: list[str],
    output: Path,
    allow_overwrite: bool = False,
) -> None:
    manifest = manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == manifest:
        raise ValueError("Refusing to overwrite the source manifest in place.")
    if output.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Output manifest already exists; pass --allow_overwrite: {output}"
        )
    if len(set(motion_names)) != len(motion_names):
        raise ValueError("--motion_names must not contain duplicates.")

    payload: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest has no dataset.trajectories.lafan1_csv entries.")
    by_name: dict[str, Any] = {}
    for entry in entries:
        name = str(entry.get("name"))
        if name in by_name:
            raise ValueError(f"Manifest contains duplicate motion name: {name!r}")
        by_name[name] = entry
    missing = [name for name in motion_names if name not in by_name]
    if missing:
        raise ValueError(f"Requested motions are absent from the manifest: {missing}")

    selected: list[dict[str, Any]] = []
    for name in motion_names:
        entry = copy.deepcopy(by_name[name])
        source_path = _resolve_against(manifest.parent, str(entry["path"]))
        if not source_path.is_file():
            raise FileNotFoundError(f"Motion NPZ not found for {name!r}: {source_path}")
        entry["path"] = str(source_path)
        selected.append(entry)
    payload["dataset"]["trajectories"]["lafan1_csv"] = selected

    selection_record = {
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "motion_names": list(motion_names),
        "command": " ".join(sys.argv),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        preparation = metadata.get("preparation_record")
        if preparation:
            metadata["preparation_record"] = str(
                _resolve_against(manifest.parent, str(preparation))
            )
        language_path = metadata.get("language_annotations_path")
        if language_path:
            # The Phase-5 preflight requires the language sidecar to cover
            # exactly the manifest motions, so write a matching subset sidecar
            # beside the subset manifest.
            source_sidecar = _resolve_against(manifest.parent, str(language_path))
            sidecar_payload: dict[str, Any] = json.loads(
                source_sidecar.read_text(encoding="utf-8")
            )
            sidecar_motions = sidecar_payload.get("motions")
            if not isinstance(sidecar_motions, list):
                raise ValueError(
                    f"Language sidecar has no motions list: {source_sidecar}"
                )
            sidecar_by_name = {
                str(entry.get("name")): entry for entry in sidecar_motions
            }
            sidecar_missing = [
                name for name in motion_names if name not in sidecar_by_name
            ]
            if sidecar_missing:
                raise ValueError(
                    "Language sidecar is missing requested motions: "
                    f"{sidecar_missing}"
                )
            sidecar_payload["motions"] = [
                copy.deepcopy(sidecar_by_name[name]) for name in motion_names
            ]
            sidecar_payload["manifest"] = str(output)
            sidecar_payload["selection"] = {
                **selection_record,
                "source_language_sidecar": str(source_sidecar),
                "source_language_sidecar_sha256": _sha256(source_sidecar),
            }
            subset_sidecar = output.with_name(f"{output.stem}_language.json")
            subset_sidecar.write_text(
                json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8"
            )
            metadata["language_annotations_path"] = str(subset_sidecar)
        if "num_motions" in metadata:
            metadata["num_motions"] = len(selected)
        if "paths_are_relative_to_manifest" in metadata:
            metadata["paths_are_relative_to_manifest"] = False
        metadata["selection"] = selection_record

    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    write_motion_subset_manifest(
        args.manifest,
        motion_names=[str(name) for name in args.motion_names],
        output=args.output,
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(f"[INFO] Wrote {len(args.motion_names)}-motion subset manifest: {args.output}")


if __name__ == "__main__":
    main()
