"""Freeze a selected motion set, canonical language, and preparation provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from imitation_experiments.data.write_motion_subset_manifest import (
    write_motion_subset_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hold_screen(
    selection_payload: dict[str, Any], selected_rows: list[Any]
) -> None:
    """Enforce optional activity limits declared by a selected-motion set."""
    hold_screen = selection_payload.get("hold_screen")
    if hold_screen is None:
        return
    if not isinstance(hold_screen, dict):
        raise ValueError("hold_screen must be an object when provided.")
    try:
        max_fraction = float(hold_screen["max_hold_fraction"])
        max_longest_s = float(hold_screen["max_longest_hold_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "hold_screen requires numeric max_hold_fraction and max_longest_hold_s."
        ) from exc
    if not 0.0 <= max_fraction <= 1.0 or max_longest_s < 0.0:
        raise ValueError("hold_screen limits are outside their valid ranges.")
    for row in selected_rows:
        if not isinstance(row, dict):
            raise ValueError("Every selected motion must be an object.")
        name = str(row.get("motion_name", "<unnamed>"))
        try:
            hold_fraction = float(row["hold_fraction"])
            longest_hold_s = float(row["longest_hold_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Activity-screened motion {name!r} requires numeric "
                "hold_fraction and longest_hold_s."
            ) from exc
        if not 0.0 <= hold_fraction <= 1.0 or longest_hold_s < 0.0:
            raise ValueError(f"Motion {name!r} has invalid hold metrics.")
        if hold_fraction > max_fraction or longest_hold_s > max_longest_s:
            raise ValueError(
                f"Motion {name!r} fails the hold screen: "
                f"hold_fraction={hold_fraction:.6g} (max {max_fraction:.6g}), "
                f"longest_hold_s={longest_hold_s:.6g} "
                f"(max {max_longest_s:.6g})."
            )


def prepare_language_motion_selection(
    *,
    source_manifest: Path,
    selection: Path,
    output_manifest: Path,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Write a subset manifest and replace its sidecar text with frozen goals."""
    source_manifest = source_manifest.expanduser().resolve()
    selection = selection.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    for source in (source_manifest, selection):
        if not source.is_file():
            raise FileNotFoundError(source)

    selection_payload = json.loads(selection.read_text(encoding="utf-8"))
    selected_rows = selection_payload.get("motions")
    if not isinstance(selected_rows, list) or not selected_rows:
        raise ValueError("Selection JSON must contain a non-empty motions list.")
    _validate_hold_screen(selection_payload, selected_rows)
    motion_names: list[str] = []
    goals: dict[str, str] = {}
    selection_by_name: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        if not isinstance(row, dict):
            raise ValueError("Every selected motion must be an object.")
        name = str(row.get("motion_name", "")).strip()
        goal = str(row.get("language_goal", "")).strip()
        if not name or not goal:
            raise ValueError(
                "Every selected motion needs motion_name and language_goal."
            )
        if name in goals:
            raise ValueError(f"Duplicate selected motion: {name!r}.")
        motion_names.append(name)
        goals[name] = goal
        selection_by_name[name] = dict(row)

    write_motion_subset_manifest(
        source_manifest,
        motion_names=motion_names,
        output=output_manifest,
        allow_overwrite=allow_overwrite,
    )
    manifest_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    metadata = manifest_payload.setdefault("metadata", {})
    raw_sidecar = metadata.get("language_annotations_path")
    if not raw_sidecar:
        raise ValueError("Source manifest does not declare a language sidecar.")
    sidecar = Path(str(raw_sidecar)).expanduser()
    if not sidecar.is_absolute():
        sidecar = output_manifest.parent / sidecar
    sidecar = sidecar.resolve()
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_rows = sidecar_payload.get("motions")
    if not isinstance(sidecar_rows, list):
        raise ValueError("Subset language sidecar has no motions list.")
    if [str(row.get("name")) for row in sidecar_rows] != motion_names:
        raise ValueError("Subset language sidecar order differs from the selection.")
    for row in sidecar_rows:
        name = str(row["name"])
        row["language_goal"] = goals[name]
        row["language_goal_source"] = "frozen_canonical_instruction"
        row["selection_metrics"] = {
            key: value
            for key, value in selection_by_name[name].items()
            if key not in {"motion_name", "language_goal"}
        }
    sidecar_payload["canonical_language_selection"] = {
        "selection": str(selection),
        "selection_sha256": _sha256(selection),
        "instruction_count": len(motion_names),
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")

    preparation_record = output_manifest.with_name(
        f"{output_manifest.stem}_preparation.json"
    )
    metadata["language_annotations_path"] = str(sidecar)
    metadata["preparation_record"] = str(preparation_record)
    metadata["canonical_language_selection"] = {
        "selection": str(selection),
        "selection_sha256": _sha256(selection),
        "motion_names": motion_names,
    }
    output_manifest.write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )
    record = {
        "schema": "bones_language_motion_selection_preparation_v1",
        "command": " ".join(sys.argv),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "selection": str(selection),
        "selection_sha256": _sha256(selection),
        "motion_names": motion_names,
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": _sha256(output_manifest),
        "language_sidecar": str(sidecar),
        "language_sidecar_sha256": _sha256(sidecar),
    }
    preparation_record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()
    record = prepare_language_motion_selection(
        source_manifest=args.source_manifest,
        selection=args.selection,
        output_manifest=args.output_manifest,
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(f"[PASS] Prepared {len(record['motion_names'])} language motions.")
    print(f"[INFO] Manifest: {record['output_manifest']}")
    print(f"[INFO] Sidecar:  {record['language_sidecar']}")


if __name__ == "__main__":
    main()
