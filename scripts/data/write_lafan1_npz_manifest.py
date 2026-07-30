#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a LAFAN1 manifest JSON from an existing folder of NPZ motions.

Examples:
    pixi run python scripts/data/write_lafan1_npz_manifest.py \
      --npz_dir data/lafan1/npz/g1 \
      --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json

    pixi run python scripts/data/write_lafan1_npz_manifest.py \
      --npz_dir data/lafan1/npz/g1 \
      --manifest_path data/lafan1/manifests/g1_debug_manifest.json \
      --select dance1_subject1 dance1_subject2 walk1_subject1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def _sanitize_motion_name(path_without_suffix: Path) -> str:
    name = path_without_suffix.as_posix()
    name = re.sub(r"[^A-Za-z0-9_\-/]+", "_", name)
    name = name.replace("/", "__").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "motion"


def _discover_npz_files(npz_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(path.resolve() for path in npz_dir.glob(pattern) if path.is_file())


def _read_npz_fps(npz_path: Path) -> float | None:
    try:
        with np.load(npz_path) as npz_data:
            if "fps" not in npz_data.files:
                return None
            return float(np.asarray(npz_data["fps"]).reshape(-1)[0])
    except Exception:
        return None


def _resolve_selected_files(all_files: list[Path], selections: list[str]) -> list[Path]:
    if len(selections) == 0:
        return all_files

    basename_map: dict[str, list[Path]] = {}
    stem_map: dict[str, list[Path]] = {}
    for path in all_files:
        basename_map.setdefault(path.name, []).append(path)
        stem_map.setdefault(path.stem, []).append(path)

    selected: list[Path] = []
    seen: set[Path] = set()
    for raw_query in selections:
        query = raw_query.strip()
        if not query:
            continue

        resolved: Path | None = None
        path_query = Path(query)
        if path_query.suffix.lower() == ".npz" and path_query.is_absolute():
            candidate = path_query.resolve()
            if candidate in all_files:
                resolved = candidate

        if resolved is None:
            basename_matches = basename_map.get(Path(query).name, [])
            stem_matches = stem_map.get(Path(query).stem, [])
            combined_matches = list(dict.fromkeys(basename_matches + stem_matches))
            if len(combined_matches) == 1:
                resolved = combined_matches[0]
            elif len(combined_matches) > 1:
                matches = ", ".join(str(path) for path in combined_matches)
                raise ValueError(
                    f"Selection '{query}' is ambiguous. Matching files: {matches}"
                )

        if resolved is None:
            raise FileNotFoundError(
                f"Could not match selection '{query}' to an NPZ file."
            )
        if resolved not in seen:
            seen.add(resolved)
            selected.append(resolved)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a manifest JSON for existing LAFAN1-style NPZ motions."
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        required=True,
        help="Input folder containing NPZ motion files.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan npz_dir.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lafan1",
        help="Dataset name written to the manifest.",
    )
    parser.add_argument(
        "--motion_name_prefix",
        type=str,
        default="",
        help="Optional prefix for generated motion names in the manifest.",
    )
    parser.add_argument(
        "--fallback_input_fps",
        type=float,
        default=50.0,
        help="Fallback FPS when an NPZ file does not include fps metadata.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=None,
        help=(
            "Optional subset of NPZs to include, matched by basename, stem, or absolute path. "
            "The listed order is preserved in the manifest."
        ),
    )
    parser.add_argument(
        "--path_mode",
        type=str,
        choices=("relative", "absolute"),
        default="relative",
        help="Write manifest motion paths relative to the manifest file or as absolute paths.",
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    if not npz_dir.is_dir():
        raise NotADirectoryError(f"npz_dir does not exist: {npz_dir}")

    all_npz_files = _discover_npz_files(npz_dir=npz_dir, recursive=bool(args.recursive))
    if len(all_npz_files) == 0:
        raise RuntimeError(f"No NPZ files found under: {npz_dir}")

    selected_queries = list(args.select or [])
    npz_files = _resolve_selected_files(
        all_files=all_npz_files, selections=selected_queries
    )
    if len(npz_files) == 0:
        raise RuntimeError("No NPZ files selected for the manifest.")

    manifest_entries: list[dict[str, object]] = []
    manifest_fps_values: list[float] = []
    for npz_file in npz_files:
        relative_no_suffix = npz_file.relative_to(npz_dir).with_suffix("")
        motion_name = _sanitize_motion_name(relative_no_suffix)
        if args.motion_name_prefix:
            motion_name = f"{args.motion_name_prefix}{motion_name}"

        fps = _read_npz_fps(npz_file)
        entry_fps = float(
            fps if fps is not None and fps > 0.0 else args.fallback_input_fps
        )
        manifest_fps_values.append(entry_fps)
        entry_path = (
            os.path.relpath(npz_file, manifest_path.parent)
            if args.path_mode == "relative"
            else str(npz_file)
        )
        manifest_entries.append(
            {
                "name": motion_name,
                "path": entry_path,
                "input_fps": entry_fps,
            }
        )

    inferred_control_freq = None
    if manifest_fps_values and all(
        abs(value - manifest_fps_values[0]) <= 1.0e-6 for value in manifest_fps_values
    ):
        inferred_control_freq = manifest_fps_values[0]

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset": {
            "trajectories": {
                "lafan1_csv": manifest_entries,
            }
        },
        "metadata": {
            "npz_dir": str(npz_dir),
            "num_motions": len(manifest_entries),
            "recursive": bool(args.recursive),
            "fallback_input_fps": float(args.fallback_input_fps),
            "selected_queries": selected_queries,
            "generated_from_existing_npz": True,
            "path_mode": args.path_mode,
            "fps_values": manifest_fps_values,
            "control_freq": inferred_control_freq,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Motion count: {len(manifest_entries)}")
    print(
        "\n"
        "# Paste into ImitationG1LafanTrackEnvCfg\n"
        f'lafan1_manifest_path = "{manifest_path}"\n'
    )


if __name__ == "__main__":
    main()
