#!/usr/bin/env python3
"""Write a one-motion manifest without changing the source manifest."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--motion_name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_single_motion_manifest(
    manifest: Path,
    *,
    motion_name: str,
    output: Path,
) -> None:
    manifest = manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    payload: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest has no dataset.trajectories.lafan1_csv entries.")
    matches = [entry for entry in entries if str(entry.get("name")) == motion_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one motion named {motion_name!r}, found {len(matches)}."
        )

    selected = copy.deepcopy(matches[0])
    source_path = Path(str(selected["path"])).expanduser()
    if not source_path.is_absolute():
        source_path = manifest.parent / source_path
    selected["path"] = str(source_path.resolve())
    payload["dataset"]["trajectories"]["lafan1_csv"] = [selected]

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        language_path = metadata.get("language_annotations_path")
        if language_path:
            resolved_language = Path(str(language_path)).expanduser()
            if not resolved_language.is_absolute():
                resolved_language = manifest.parent / resolved_language
            metadata["language_annotations_path"] = str(resolved_language.resolve())
        metadata["selection"] = {
            "source_manifest": str(manifest),
            "motion_name": motion_name,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    write_single_motion_manifest(
        args.manifest,
        motion_name=str(args.motion_name),
        output=args.output,
    )
    print(f"[INFO] Wrote one-motion manifest: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
