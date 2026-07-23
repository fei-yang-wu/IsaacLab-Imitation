#!/usr/bin/env python3
"""Select the full SONIC-filtered BONES-SEED G1 motion set from the metadata CSV.

This applies the *exact* public SONIC release keyword-exclusion filter (the same
40-keyword list vendored in ``filter_bones_seed_sonic_exclusions.py``) to every
G1 motion row in ``seed_metadata_v004.csv`` and writes a deterministic selection
JSON that downstream conversion consumes.

Unlike ``select_bones_seed_100.py`` (curated quotas, non-mirror, no dancing),
this keeps *everything* the SONIC filter keeps: mirrors and non-neutral motions
included. On ``seed_metadata_v004.csv`` (142,220 rows) it keeps 129,785 motions.

The SONIC filter is purely name-based; no metadata column is consulted for
inclusion. Verified against
``NVlabs/GR00T-WholeBodyControl gear_sonic/data_process/filter_and_copy_bones_data.py``
(default ``--filter-keywords``) on 2026-07-21.

Run from the repo root:

    pixi run python scripts/select_bones_seed_sonic.py \
        --metadata_csv data/bones_seed/raw/metadata/seed_metadata_v004.csv \
        --output ~/Storage/bones_seed_full/selection/g1_bones_seed_sonic_selection.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_CSV = (
    REPO_ROOT / "data" / "bones_seed" / "raw" / "metadata" / "seed_metadata_v004.csv"
)


def _load_sonic_keywords() -> list[str]:
    """Load the SONIC exclusion keyword list from the vendored filter module."""
    module_path = REPO_ROOT / "scripts" / "filter_bones_seed_sonic_exclusions.py"
    spec = importlib.util.spec_from_file_location(
        "_bones_seed_sonic_exclusions", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SONIC filter module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.SONIC_RELEASE_FILTER_KEYWORDS)


def _matched_keywords(name: str, keywords: list[str]) -> list[str]:
    name_lower = name.lower()
    return [kw for kw in keywords if kw.lower() in name_lower]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--report_only",
        action="store_true",
        default=False,
        help="Print counts and the selection hash without writing the JSON.",
    )
    args = parser.parse_args()
    if not args.report_only and args.output is None:
        parser.error("--output is required unless --report_only is set.")

    metadata_csv = args.metadata_csv.expanduser().resolve()
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    keywords = _load_sonic_keywords()
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    motions: list[dict[str, Any]] = []
    excluded = 0
    missing_path = 0
    for row in rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        if _matched_keywords(filename, keywords):
            excluded += 1
            continue
        g1_path = (row.get("move_g1_path") or "").strip()
        if not g1_path:
            missing_path += 1
            continue
        duration = row.get("move_duration_frames")
        motions.append(
            {
                "filename": filename,
                "g1_path": g1_path,
                "category": (row.get("category") or "").strip() or None,
                "is_mirror": (row.get("is_mirror") or "").strip().lower() == "true",
                "is_neutral": (row.get("is_neutral") or "").strip() in ("1", "1.0"),
                "move_duration_frames": int(duration)
                if duration and str(duration).strip().isdigit()
                else None,
                "language_goal": (row.get("content_short_description") or "").strip()
                or None,
            }
        )

    # Deterministic order: by g1_path so the selection is stable across runs.
    motions.sort(key=lambda m: m["g1_path"])

    payload = {
        "dataset_name": "bones_seed_sonic_full",
        "source": "bones-studio/seed",
        "selection": "sonic_keyword_exclusion",
        "sonic_filter_source": (
            "NVlabs/GR00T-WholeBodyControl gear_sonic/data_process/"
            "filter_and_copy_bones_data.py default --filter-keywords"
        ),
        "sonic_filter_keywords": keywords,
        "metadata_csv": str(metadata_csv),
        "num_total_rows": len(rows),
        "num_excluded": excluded,
        "num_missing_g1_path": missing_path,
        "num_selected": len(motions),
        "motions": motions,
    }

    selection_bytes = json.dumps(payload["motions"], sort_keys=True).encode("utf-8")
    selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    payload["selection_sha256"] = selection_hash

    print(f"[INFO] Metadata rows:     {len(rows)}")
    print(f"[INFO] SONIC-excluded:    {excluded}")
    print(f"[INFO] Missing g1_path:   {missing_path}")
    print(f"[INFO] Selected motions:  {len(motions)}")
    print(f"[INFO] Selection SHA-256: {selection_hash}")

    if args.report_only:
        return

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote selection: {output}")


if __name__ == "__main__":
    sys.exit(main())
