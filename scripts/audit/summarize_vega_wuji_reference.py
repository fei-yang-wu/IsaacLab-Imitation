#!/usr/bin/env python3
"""Print the acceptance evidence stored in a Vega-Wuji Reference NPZ.

One screen of numbers per Reference: the four per-hand finger gates, the
wrist-IK residuals, the hand morphology standoff, and every geometry
clearance group with its worst signed distance. Use it to compare candidate
retargets without re-reading the raw metadata tree.

Run from the repository root:

    pixi run -e isaaclab python \
        scripts/audit/summarize_vega_wuji_reference.py --reference <npz> ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from iltools.core import load_dexterous_reference_npz


def _clearance_rows(node: Any, prefix: str = "") -> list[tuple[str, float, Any]]:
    rows: list[tuple[str, float, Any]] = []
    if not isinstance(node, Mapping):
        return rows
    for key, value in node.items():
        if not isinstance(value, Mapping):
            continue
        distance = value.get("minimum_signed_distance_m")
        if distance is None:
            rows.extend(_clearance_rows(value, f"{prefix}{key}."))
        else:
            rows.append((f"{prefix}{key}", float(distance), value.get("qualified")))
    return rows


def summarize(path: Path) -> None:
    reference = load_dexterous_reference_npz(path)
    metadata = reference.metadata
    print(f"\n=== {path.name} ===")
    print(
        f"sequence: {reference.sequence_id}\n"
        f"frames: {reference.frame_count} @ {reference.fps:g} Hz "
        f"({(reference.frame_count - 1) / reference.fps:.2f} s)   "
        f"runtime_qualified: {metadata.get('isaac_runtime_qualified')}"
    )

    finger_mapping = metadata.get("finger_mapping", {})
    acceptance = finger_mapping.get("final_acceptance")
    if isinstance(acceptance, Mapping):
        print("\nfinger gates (wrist<=2.5mm tip_p95<=5mm jump<=35deg hyper=0%):")
        for side in ("left", "right"):
            record = acceptance.get("sides", {}).get(side)
            if not isinstance(record, Mapping):
                continue
            gates = record["gates"]
            values = gates["values"]
            verdict = "PASS" if gates["pass"] else "fail " + ",".join(gates["failures"])
            print(
                f"  {side:5s} wrist {values['wrist_mean_mm']:6.2f}mm  "
                f"tip_p95 {values['tip_p95_mm']:6.2f}mm  "
                f"jump {values['jump_max_deg']:6.2f}deg  "
                f"hyper {values['hyperextension_pct']:4.1f}%   {verdict}"
            )

    standoff = finger_mapping.get("hand_morphology_standoff")
    if isinstance(standoff, Mapping):
        sides = standoff.get("sides", {})
        print(
            "\nhand morphology standoff (m): "
            + "  ".join(
                f"{side} {record['standoff_m']:.4f}" for side, record in sides.items()
            )
        )

    wrist = metadata.get("wrist_ik")
    if isinstance(wrist, Mapping):
        p95 = wrist["per_side_p95_position_error_m"]
        maximum = wrist["per_side_max_position_error_m"]
        print(
            "\nwrist IK: "
            + "  ".join(
                f"{side} p95 {p95[side] * 1e3:.2f}mm/max {maximum[side] * 1e3:.2f}mm"
                for side in ("left", "right")
            )
            + f"   mean orientation {wrist['mean_orientation_error_rad'] * 57.2958:.2f} deg"
        )

    audit = metadata.get("geometry_clearance_audit", {})
    print(f"\ngeometry qualified: {audit.get('qualified')}")
    for name, distance, qualified in sorted(
        _clearance_rows(audit), key=lambda row: row[1]
    ):
        if "dense" in name:
            continue
        print(f"  {name:56s} min {distance * 1e3:8.2f} mm  qualified={qualified}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for path in args.reference:
        summarize(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
