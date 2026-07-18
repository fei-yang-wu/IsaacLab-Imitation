#!/usr/bin/env python3
"""Check that G1 LAFAN1 root and rigid-body positions share one frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    return parser.parse_args()


def _manifest_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest has no dataset.trajectories.lafan1_csv entries.")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Manifest trajectory entries must be JSON objects.")
    return entries


def main() -> None:
    args = _parse_args()
    manifest = args.manifest.expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in _manifest_entries(payload):
        name = str(entry.get("name", ""))
        source = Path(str(entry.get("path", ""))).expanduser()
        if not source.is_absolute():
            source = (manifest.parent / source).resolve()
        if not source.is_file():
            failures.append(f"{name}: missing source {source}")
            continue
        with np.load(source) as data:
            if "body_pos_w" not in data.files:
                failures.append(f"{name}: missing body_pos_w")
                continue
            body_pos = np.asarray(data["body_pos_w"])
            if "root_pos" not in data.files:
                failures.append(
                    f"{name}: missing independent root_pos; the loader can use "
                    "body_pos_w[:, 0] as root, but that cannot reveal a shared "
                    "Isaac scene-grid offset"
                )
                continue
            root_pos = np.asarray(data["root_pos"])
        if (
            root_pos.ndim != 2
            or root_pos.shape[-1] != 3
            or body_pos.ndim != 3
            or body_pos.shape[0] != root_pos.shape[0]
        ):
            failures.append(
                f"{name}: incompatible shapes root={root_pos.shape}, body={body_pos.shape}"
            )
            continue
        offsets = body_pos[:, 0, :] - root_pos
        max_abs_offset = float(np.max(np.abs(offsets)))
        max_drift = float(np.max(np.abs(offsets - np.median(offsets, axis=0))))
        record = {
            "name": name,
            "source": str(source),
            "root_source": "root_pos",
            "first_offset_xyz": offsets[0].astype(float).tolist(),
            "max_abs_offset": max_abs_offset,
            "max_offset_drift": max_drift,
            "passed": max_abs_offset <= float(args.tolerance),
        }
        records.append(record)
        if not record["passed"]:
            failures.append(
                f"{name}: body/root offset {max_abs_offset:.6g} exceeds "
                f"{float(args.tolerance):.6g}"
            )

    report = {
        "passed": not failures,
        "manifest": str(manifest),
        "tolerance": float(args.tolerance),
        "motion_count": len(records),
        "failures": failures,
        "motions": records,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(
            "G1 LAFAN1 body-frame audit failed:\n- " + "\n- ".join(failures[:10])
        )
    print(f"[PASS] G1 LAFAN1 body frames: {report_path}")


if __name__ == "__main__":
    main()
