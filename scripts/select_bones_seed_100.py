#!/usr/bin/env python3
"""Select a diverse, locomotion+manipulation-weighted 100-motion BONES-SEED subset.

Deterministic selection from ``seed_metadata_v004.csv``:

* exclude the ``Dancing`` category (user asked for "less dancing"),
* keep only canonical (non-mirror) takes and dedup to one clip per ``content_name``,
* apply per-category quotas weighted toward locomotion + object manipulation/interaction,
* force-include the 8 demo motions (they count against their category quota).

Outputs:

1. a shortlist JSON consumable by ``scripts/prepare_bones_seed_subset.py``
   (list of ``{filename, overview_description, category, content_name}``), and
2. a provenance manifest JSON recording where every clip came from
   (archive path, package, category, actor, take, duration, flags).

Run from the repo root (default environment is fine; no Isaac needed):

    pixi run python scripts/select_bones_seed_100.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = REPO_ROOT / "data" / "bones_seed" / "raw" / "metadata" / "seed_metadata_v004.csv"
DEFAULT_SHORTLIST = REPO_ROOT / "data" / "bones_seed" / "curated" / "bones_seed_100_shortlist.timeline.json"
DEFAULT_PROVENANCE = REPO_ROOT / "data" / "bones_seed" / "curated" / "bones_seed_100_provenance.json"

# The 8 demo motions to subsume (exact source filenames in the metadata CSV).
DEMO8_FILENAMES = (
    "Neutral_stoop_down_001__A057",
    "big_heavy_one_hand_front_high_to_front_low_R_001__A524",
    "big_heavy_one_hand_front_low_to_front_high_R_001__A524",
    "big_light_two_hands_pick_up_front_medium_R_001__A509",
    "drinking_standing_mug_R_001__A282",
    "inside_door_handle_left_side_open_walk_close_behind_R_001__A513",
    "inside_door_handle_right_side_open_walk_turn_close_R_001__A514",
    "read_book_both_hands_sitting_R_001__A456",
)

# Locomotion + manipulation weighted quotas (sum = 100).
CATEGORY_QUOTAS: dict[str, int] = {
    "Object Manipulation": 16,
    "Object Interaction": 14,
    "Basic Locomotion Neutral": 12,
    "Basic Locomotion Styles": 10,
    "Advanced Locomotion": 10,
    "Gestures": 10,
    "Unusual Locomotion": 5,
    "Sports": 4,
    "Household": 4,
    "Consuming": 3,
    "Communication": 3,
    "Baseline": 3,
    "Complex Actions": 2,
    "Stunts": 2,
    "Environments": 1,
    "Other": 1,
}
EXCLUDE_CATEGORIES = {"Dancing"}
# Aerial / prop-climbing motions spawn the robot far off the ground in a propless
# env (e.g. ladder starts ~2.5 m up) and are not trainable here.
EXCLUDE_NAME_SUBSTRINGS = ("ladder", "climb")
TARGET_TOTAL = 100

# Keep clips in a trainable duration band (source frames at 120 fps).
MIN_SOURCE_FRAMES = 150
MAX_SOURCE_FRAMES = 2600


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "0.0", "", "no"}


def _overview(row: dict[str, str]) -> str:
    for key in ("content_short_description", "content_natural_desc_1", "content_technical_description"):
        text = (row.get(key) or "").strip()
        if text:
            return text
    return row.get("content_name", "")


def _select(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    by_filename = {r["filename"]: r for r in rows}

    # Force-included demo rows first.
    selected: dict[str, dict[str, str]] = {}
    used_contents: set[str] = set()
    for fn in DEMO8_FILENAMES:
        row = by_filename.get(fn)
        if row is None:
            raise SystemExit(f"[ERROR] demo motion not found in metadata: {fn}")
        selected[fn] = row
        used_contents.add(row.get("content_name", fn))

    # Build the candidate pool: canonical (non-mirror) takes, dedup by content_name,
    # excluded categories dropped, duration band enforced. Deterministic ordering.
    pool_by_category: dict[str, list[dict[str, str]]] = {}
    seen_content: set[str] = set(used_contents)
    for row in sorted(rows, key=lambda r: (r.get("category", ""), r.get("content_name", ""), r["filename"])):
        category = row.get("category", "")
        content = row.get("content_name", "")
        if category in EXCLUDE_CATEGORIES:
            continue
        lname = (row.get("content_name", "") + " " + row["filename"]).lower()
        if any(sub in lname for sub in EXCLUDE_NAME_SUBSTRINGS):
            continue
        if not _is_false(row.get("is_mirror")):
            continue
        if content in seen_content:
            continue
        frames = _as_int(row.get("move_duration_frames"))
        if frames < MIN_SOURCE_FRAMES or frames > MAX_SOURCE_FRAMES:
            continue
        seen_content.add(content)
        pool_by_category.setdefault(category, []).append(row)

    # Count demo motions already occupying each category quota.
    demo_per_category: dict[str, int] = {}
    for row in selected.values():
        demo_per_category[row.get("category", "")] = demo_per_category.get(row.get("category", ""), 0) + 1

    # Fill category quotas (spread across content_names by even striding for diversity).
    shortfall = 0
    for category, quota in CATEGORY_QUOTAS.items():
        remaining = max(quota - demo_per_category.get(category, 0), 0)
        candidates = pool_by_category.get(category, [])
        if remaining <= 0 or not candidates:
            shortfall += remaining
            continue
        if remaining >= len(candidates):
            picked = candidates
            shortfall += remaining - len(candidates)
        else:
            stride = len(candidates) / remaining
            picked = [candidates[int(i * stride)] for i in range(remaining)]
        for row in picked:
            selected[row["filename"]] = row

    # Redistribute any shortfall into the biggest weighted pools (locomotion + manipulation).
    if len(selected) < TARGET_TOTAL:
        backfill_order = [
            "Object Manipulation", "Object Interaction", "Basic Locomotion Neutral",
            "Advanced Locomotion", "Basic Locomotion Styles", "Gestures",
        ]
        for category in backfill_order:
            if len(selected) >= TARGET_TOTAL:
                break
            for row in pool_by_category.get(category, []):
                if row["filename"] in selected:
                    continue
                selected[row["filename"]] = row
                if len(selected) >= TARGET_TOTAL:
                    break

    stats = {
        "total_selected": len(selected),
        "target": TARGET_TOTAL,
        "by_category": {},
        "demo_included": sorted(DEMO8_FILENAMES),
    }
    for row in selected.values():
        cat = row.get("category", "")
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

    ordered = sorted(selected.values(), key=lambda r: (r.get("category", ""), r["filename"]))
    return ordered, stats


def _write_shortlist(ordered: list[dict[str, str]], path: Path) -> None:
    payload = [
        {
            "filename": row["filename"],
            "overview_description": _overview(row),
            "category": row.get("category", ""),
            "content_name": row.get("content_name", ""),
        }
        for row in ordered
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_provenance(ordered: list[dict[str, str]], path: Path, stats: dict[str, Any]) -> None:
    motions = [
        {
            "filename": row["filename"],
            "move_name": row.get("move_name"),
            "content_name": row.get("content_name"),
            "category": row.get("category"),
            "package": row.get("package"),
            "archive_csv_path": row.get("move_g1_path"),
            "take_name": row.get("take_name"),
            "take_actor": row.get("take_actor"),
            "take_date": row.get("take_date"),
            "move_duration_frames": _as_int(row.get("move_duration_frames")),
            "is_neutral": row.get("is_neutral"),
            "is_mirror": row.get("is_mirror"),
            "is_demo8": row["filename"] in set(DEMO8_FILENAMES),
        }
        for row in ordered
    ]
    payload = {
        "dataset_name": "bones_seed_100",
        "source_repo": "bones-studio/seed",
        "source_metadata": "seed_metadata_v004.csv",
        "selection": {
            "strategy": "locomotion+manipulation weighted, non-mirror, dedup by content_name, exclude Dancing, force-include demo8",
            "category_quotas": CATEGORY_QUOTAS,
            "excluded_categories": sorted(EXCLUDE_CATEGORIES),
            "duration_band_source_frames": [MIN_SOURCE_FRAMES, MAX_SOURCE_FRAMES],
        },
        "stats": stats,
        "motions": motions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()

    if not args.metadata.is_file():
        raise SystemExit(f"[ERROR] metadata CSV not found: {args.metadata}")

    csv.field_size_limit(1 << 24)
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("filename")]

    ordered, stats = _select(rows)
    _write_shortlist(ordered, args.shortlist)
    _write_provenance(ordered, args.provenance, stats)

    print(f"[INFO] selected {stats['total_selected']} motions (target {TARGET_TOTAL})")
    print("[INFO] by category:")
    for cat, n in sorted(stats["by_category"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {n:3d}  {cat}")
    print(f"[INFO] shortlist:   {args.shortlist}")
    print(f"[INFO] provenance:  {args.provenance}")


if __name__ == "__main__":
    sys.exit(main())
