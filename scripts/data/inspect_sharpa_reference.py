#!/usr/bin/env python3
"""Validate and summarize an ILTools Sharpa reference manifest.

The report is intentionally small enough to use as a conversion gate: it
loads every hash-bound NPZ, validates the declared rigid scene assets, and
prints trajectory/contact/object statistics without launching Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iltools.core import (
    DexterousReference,
    load_dexterous_reference_manifest,
    sha256_file,
)


def _distance(first: np.ndarray, last: np.ndarray) -> float:
    if len(first) < 2:
        return 0.0
    return float(np.linalg.norm(np.asarray(last)[..., :3] - np.asarray(first)[..., :3]))


def summarize_reference(reference: DexterousReference) -> dict[str, Any]:
    """Return JSON-compatible diagnostics for one validated reference."""

    reference.verify_scene_assets(require_hashes=True)
    contacts = reference.contacts
    active_contacts = 0
    contact_frames = 0
    if contacts is not None:
        active = np.asarray(contacts.active, dtype=bool)
        active_contacts = int(active.sum())
        contact_frames = int(active.any(axis=tuple(range(1, active.ndim))).sum())
    object_pose = np.asarray(reference.object_poses_w)
    return {
        "name": reference.sequence_id,
        "frames": reference.frame_count,
        "duration_s": float((reference.frame_count - 1) / reference.fps),
        "fps": float(reference.fps),
        "robot_layout": reference.robot_layout,
        "joint_count": len(reference.joint_names),
        "left_joint_count": len(reference.left_joint_names),
        "right_joint_count": len(reference.right_joint_names),
        "wrist_translation": {
            "left_m": _distance(
                reference.left_wrist_pose_w[0], reference.left_wrist_pose_w[-1]
            ),
            "right_m": _distance(
                reference.right_wrist_pose_w[0], reference.right_wrist_pose_w[-1]
            ),
        },
        "object_names": list(reference.object_names),
        "object_translation_m": [
            _distance(object_pose[0, index], object_pose[-1, index])
            for index in range(object_pose.shape[1])
        ],
        "active_contact_slots": active_contacts,
        "contact_frames": contact_frames,
        "object_assets": [
            {
                "path": path,
                "exists": Path(path).is_file(),
                "sha256": digest,
                "hash_matches": bool(
                    Path(path).is_file() and sha256_file(path) == digest
                ),
            }
            for path, digest in zip(
                reference.object_asset_paths,
                reference.object_asset_sha256,
                strict=True,
            )
        ],
        "metadata": dict(reference.metadata),
    }


def retarget_warnings(metadata: dict[str, Any]) -> list[str]:
    """Explain why a reference's Sharpa solution is not a trusted IK result."""

    warnings: list[str] = []
    kind = metadata.get("retarget_solution_kind")
    source = metadata.get("retarget_source", "unknown")
    if kind == "placeholder":
        warnings.append(
            "The Sharpa solution is a placeholder (zero task error after one "
            "iteration on every frame). It is a schema fixture, not an IK result."
        )
    elif kind == "unknown":
        warnings.append(
            f"The reference carries no IK convergence record (retarget_source="
            f"{source}); the Sharpa solution cannot be qualified."
        )
    return warnings


def build_report(manifest_path: str | Path) -> dict[str, Any]:
    """Load, hash-verify, and summarize a Sharpa manifest."""

    manifest = load_dexterous_reference_manifest(manifest_path)
    references = manifest.load_references(verify_hashes=True)
    motions = [summarize_reference(reference) for reference in references]
    return {
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "schema": manifest.schema,
        "dataset_name": manifest.dataset_name,
        "robot_name": manifest.robot_name,
        "reference_fps": manifest.reference_fps,
        "motion_count": len(motions),
        "total_frames": sum(item["frames"] for item in motions),
        "manifest_metadata": dict(manifest.metadata),
        "motions": motions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON only."
    )
    args = parser.parse_args()
    report = build_report(args.manifest)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return
    print(
        f"{report['dataset_name']}: {report['motion_count']} motion(s), "
        f"{report['total_frames']} frames at {report['reference_fps']:.3g} Hz"
    )
    for motion in report["motions"]:
        print(
            f"- {motion['name']}: {motion['frames']} frames, "
            f"{motion['duration_s']:.2f} s, objects={motion['object_names']}, "
            f"contact_frames={motion['contact_frames']}"
        )
        for warning in retarget_warnings(motion["metadata"]):
            print(f"  WARNING: {warning}")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
