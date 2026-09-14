#!/usr/bin/env python3
"""Compare two Sharpa dual-hand references frame by frame.

The tool is the retarget parity gate. It loads two ILTools references (a
manifest or one NPZ each), pairs their motions, and reports the largest
per-frame difference in wrist position, wrist rotation, finger joints, object
pose, and contact activity. Pass thresholds to turn the report into a check
that exits with a non-zero code.

Example:

    python scripts/audit/compare_sharpa_references.py \\
        --reference /path/source_ik/manifest.json \\
        --reference /path/iltools_pink/manifest.json \\
        --max-wrist-position-m 0.001 --max-wrist-rotation-rad 0.01 \\
        --max-joint-rad 0.001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from iltools.core import (
    DexterousReference,
    load_dexterous_reference_manifest,
    load_dexterous_reference_npz,
)


def quaternion_angle_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the rotation angle between unit ``wxyz`` quaternion arrays."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"max": 0.0, "mean": 0.0}
    return {"max": float(np.max(values)), "mean": float(np.mean(values))}


def _contact_agreement(
    first: DexterousReference, second: DexterousReference
) -> dict[str, Any] | None:
    if first.contacts is None or second.contacts is None:
        return None
    active_a = np.asarray(first.contacts.active, dtype=bool)
    active_b = np.asarray(second.contacts.active, dtype=bool)
    if active_a.shape != active_b.shape:
        return {"shape_a": list(active_a.shape), "shape_b": list(active_b.shape)}
    return {
        "agreement_fraction": float(np.mean(active_a == active_b)),
        "active_slots_a": int(active_a.sum()),
        "active_slots_b": int(active_b.sum()),
    }


def compare_references(
    first: DexterousReference, second: DexterousReference
) -> dict[str, Any]:
    """Return per-motion difference statistics for two aligned references."""

    if first.frame_count != second.frame_count:
        raise ValueError(
            f"Frame counts differ: {first.frame_count} vs {second.frame_count} "
            f"for {first.sequence_id!r} and {second.sequence_id!r}."
        )
    if tuple(first.joint_names) != tuple(second.joint_names):
        raise ValueError("Joint name order differs between the references.")
    report: dict[str, Any] = {
        "sequence_id_a": first.sequence_id,
        "sequence_id_b": second.sequence_id,
        "frames": int(first.frame_count),
        "fps_a": float(first.fps),
        "fps_b": float(second.fps),
    }
    for side in ("left", "right"):
        pose_a = np.asarray(getattr(first, f"{side}_wrist_pose_w"), dtype=np.float64)
        pose_b = np.asarray(getattr(second, f"{side}_wrist_pose_w"), dtype=np.float64)
        report[f"{side}_wrist_position_m"] = _stats(
            np.linalg.norm(pose_a[:, :3] - pose_b[:, :3], axis=-1)
        )
        report[f"{side}_wrist_rotation_rad"] = _stats(
            quaternion_angle_rad(pose_a[:, 3:7], pose_b[:, 3:7])
        )
    joint_error = np.abs(
        np.asarray(first.qpos, dtype=np.float64)
        - np.asarray(second.qpos, dtype=np.float64)
    )
    report["finger_joint_rad"] = _stats(joint_error)
    worst = int(np.argmax(joint_error.max(axis=0)))
    report["finger_joint_worst"] = {
        "joint": str(first.joint_names[worst]),
        "max_rad": float(joint_error[:, worst].max()),
    }
    object_a = np.asarray(first.object_poses_w, dtype=np.float64)
    object_b = np.asarray(second.object_poses_w, dtype=np.float64)
    if object_a.shape != object_b.shape:
        raise ValueError("Object pose arrays differ in shape.")
    report["object_position_m"] = _stats(
        np.linalg.norm(object_a[..., :3] - object_b[..., :3], axis=-1)
    )
    report["object_rotation_rad"] = _stats(
        quaternion_angle_rad(object_a[..., 3:7], object_b[..., 3:7])
    )
    report["contacts"] = _contact_agreement(first, second)
    report["metadata_a"] = dict(first.metadata)
    report["metadata_b"] = dict(second.metadata)
    return report


def load_references(path: str | Path) -> tuple[DexterousReference, ...]:
    """Load one NPZ or every motion of a manifest."""

    source = Path(path).expanduser().resolve()
    if source.suffix == ".npz":
        return (load_dexterous_reference_npz(source),)
    return load_dexterous_reference_manifest(source).load_references(verify_hashes=True)


def check_thresholds(
    report: dict[str, Any], thresholds: dict[str, float | None]
) -> list[str]:
    """Return one message per violated threshold."""

    failures: list[str] = []
    checks = (
        ("max_wrist_position_m", ("left_wrist_position_m", "right_wrist_position_m")),
        (
            "max_wrist_rotation_rad",
            ("left_wrist_rotation_rad", "right_wrist_rotation_rad"),
        ),
        ("max_joint_rad", ("finger_joint_rad",)),
        ("max_object_position_m", ("object_position_m",)),
        ("max_object_rotation_rad", ("object_rotation_rad",)),
    )
    for name, keys in checks:
        limit = thresholds.get(name)
        if limit is None:
            continue
        for key in keys:
            value = report[key]["max"]
            if value > limit:
                failures.append(f"{key} max {value:.6g} exceeds {limit:.6g}")
    limit = thresholds.get("min_contact_agreement")
    contacts = report.get("contacts")
    if limit is not None and contacts and "agreement_fraction" in contacts:
        if contacts["agreement_fraction"] < limit:
            failures.append(
                f"contact agreement {contacts['agreement_fraction']:.4f} is below "
                f"{limit:.4f}"
            )
    return failures


def build_report(
    path_a: str | Path, path_b: str | Path, thresholds: dict[str, float | None]
) -> dict[str, Any]:
    """Compare every paired motion and collect threshold failures."""

    references_a = load_references(path_a)
    references_b = load_references(path_b)
    if len(references_a) != len(references_b):
        raise ValueError(
            f"Motion counts differ: {len(references_a)} vs {len(references_b)}."
        )
    motions = []
    failures: list[str] = []
    for first, second in zip(references_a, references_b, strict=True):
        motion = compare_references(first, second)
        motion["failures"] = check_thresholds(motion, thresholds)
        failures.extend(f"{first.sequence_id}: {item}" for item in motion["failures"])
        motions.append(motion)
    return {
        "reference_a": str(Path(path_a).expanduser().resolve()),
        "reference_b": str(Path(path_b).expanduser().resolve()),
        "thresholds": {k: v for k, v in thresholds.items() if v is not None},
        "motion_count": len(motions),
        "motions": motions,
        "failures": failures,
        "passed": not failures,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        action="append",
        type=Path,
        required=True,
        help="Manifest JSON or reference NPZ. Pass exactly twice.",
    )
    parser.add_argument("--max-wrist-position-m", type=float, default=None)
    parser.add_argument("--max-wrist-rotation-rad", type=float, default=None)
    parser.add_argument("--max-joint-rad", type=float, default=None)
    parser.add_argument("--max-object-position-m", type=float, default=None)
    parser.add_argument("--max-object-rotation-rad", type=float, default=None)
    parser.add_argument("--min-contact-agreement", type=float, default=None)
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    args = parser.parse_args(argv)
    if len(args.reference) != 2:
        parser.error("--reference must be given exactly twice.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = {
        "max_wrist_position_m": args.max_wrist_position_m,
        "max_wrist_rotation_rad": args.max_wrist_rotation_rad,
        "max_joint_rad": args.max_joint_rad,
        "max_object_position_m": args.max_object_position_m,
        "max_object_rotation_rad": args.max_object_rotation_rad,
        "min_contact_agreement": args.min_contact_agreement,
    }
    report = build_report(args.reference[0], args.reference[1], thresholds)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    for motion in report["motions"]:
        print(
            f"{motion['sequence_id_a']} vs {motion['sequence_id_b']}: "
            f"{motion['frames']} frames, "
            f"wrist L/R max {motion['left_wrist_position_m']['max']:.4g}/"
            f"{motion['right_wrist_position_m']['max']:.4g} m, "
            f"rot L/R max {motion['left_wrist_rotation_rad']['max']:.4g}/"
            f"{motion['right_wrist_rotation_rad']['max']:.4g} rad, "
            f"joint max {motion['finger_joint_rad']['max']:.4g} rad "
            f"({motion['finger_joint_worst']['joint']}), "
            f"object max {motion['object_position_m']['max']:.4g} m"
        )
    for failure in report["failures"]:
        print(f"FAIL {failure}")
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
