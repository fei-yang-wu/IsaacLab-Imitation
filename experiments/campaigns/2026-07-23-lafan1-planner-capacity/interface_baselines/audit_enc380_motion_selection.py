#!/usr/bin/env python3
"""Bind the one-motion enc380 diagnostic to corrected LAFAN1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sys

CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))
from enc380_capacity_grid import MOTIONS  # noqa: E402

POSITION_ONE_BASED = 29
EXPECTED_MANIFEST_MOTIONS = 40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(manifest_path: Path) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectories = payload.get("dataset", {}).get("trajectories", {})
    if not isinstance(trajectories, dict) or len(trajectories) != 1:
        raise ValueError("Expected exactly one ordered trajectory group in manifest.")
    entries = next(iter(trajectories.values()))
    if not isinstance(entries, list) or len(entries) != EXPECTED_MANIFEST_MOTIONS:
        raise ValueError(
            f"Expected {EXPECTED_MANIFEST_MOTIONS} manifest motions, got "
            f"{len(entries) if isinstance(entries, list) else 'non-list'}."
        )
    ordered_names = [str(entry["name"]) for entry in entries]
    selected = (ordered_names[POSITION_ONE_BASED - 1],)
    if selected != MOTIONS:
        raise ValueError(
            "Frozen enc380 motion does not match corrected-manifest position "
            f"{POSITION_ONE_BASED}: expected {MOTIONS}, observed {selected}."
        )
    return {
        "format": "enc380_motion_selection_certificate",
        "version": 2,
        "passed": True,
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "manifest_motion_count": len(entries),
        "selection_rule": (
            "user-requested walk1_subject1 continuity diagnostic, matching the "
            "previous one-motion planner study"
        ),
        "positions_one_based": [POSITION_ONE_BASED],
        "motions": list(selected),
        "performance_data_used": True,
        "paper_representative_motion_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest)
    output = args.output_json.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing certificate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[PASS] one-motion diagnostic selection -> {output}")


if __name__ == "__main__":
    main()
