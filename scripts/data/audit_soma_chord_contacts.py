#!/usr/bin/env python3
"""Audit SOMA-X reconstruction and CHORD contacts in a motion Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iltools.retarget import (  # noqa: E402
    extract_chord_contact_targets,
    infer_soma_frame_bridge,
    reconstruct_soma_motion,
    sample_object_surface,
)
from scripts.data.convert_soma_g1_parquet_to_vega_wuji import (  # noqa: E402
    DEFAULT_SOURCE,
    PINNED_END_FRAME_EXCLUSIVE,
    PINNED_PAYLOAD_SHA256,
    PINNED_START_FRAME,
    SOMA_JOINT_NAMES,
    _read_parquet_row,
    _resolve_relative_asset,
    _safe_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the SOMA mesh from the pinned global-joint payload and "
            "apply CHORD's 4,096-point / 1 cm hand-object contact rule."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--start-frame", type=int, default=PINNED_START_FRAME)
    parser.add_argument(
        "--end-frame-exclusive", type=int, default=PINNED_END_FRAME_EXCLUSIVE
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--surface-points", type=int, default=4096)
    parser.add_argument("--surface-seed", type=int, default=0)
    parser.add_argument("--contact-threshold", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional NPZ path for the dense SOMA-link contact targets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    row = _read_parquet_row(source)
    payload_bytes = row["source_payload"]
    if not isinstance(payload_bytes, bytes):
        raise ValueError("source_payload must be bytes.")
    payload = _safe_payload(payload_bytes, expected_sha256=PINNED_PAYLOAD_SHA256)
    positions = np.asarray(payload["soma_joints"], dtype=np.float64)
    rotations = np.asarray(payload["soma_joints_wxyz"], dtype=np.float64)
    start = int(args.start_frame)
    stop = int(args.end_frame_exclusive)
    if not 0 <= start < stop <= len(positions):
        raise ValueError(
            f"Frame crop [{start}, {stop}) is outside [0, {len(positions)})."
        )
    crop = slice(start, stop)

    reconstruction = reconstruct_soma_motion(
        global_joint_positions=positions[crop],
        global_joint_wxyz=rotations[crop],
        identity_coeffs=np.asarray(payload["soma_identity_coeffs"]),
        scale_params=np.asarray(payload["soma_scale_params"]),
        joint_names=SOMA_JOINT_NAMES,
        device=str(args.device),
        chunk_size=int(args.chunk_size),
    )
    bridge = infer_soma_frame_bridge(
        soma_root_positions=positions[crop, 0],
        soma_root_wxyz=rotations[crop, 0],
        target_root_positions=np.asarray(payload["nvhuman_root_translation"])[crop],
        target_root_wxyz=np.asarray(payload["nvhuman_root_wxyz"])[crop],
    )
    joints_world = bridge.apply(reconstruction.joints)
    vertices_world = bridge.apply(reconstruction.vertices)

    mesh_values = tuple(str(value) for value in row["object_mesh_paths"])
    if len(mesh_values) != 1:
        raise ValueError("The SOMA/CHORD audit currently requires one rigid object.")
    mesh_path = _resolve_relative_asset(source, mesh_values[0])
    object_points, object_normals = sample_object_surface(
        mesh_path,
        count=int(args.surface_points),
        seed=int(args.surface_seed),
    )
    object_positions = np.asarray(row["object_body_position"], dtype=np.float64)[
        crop, 0
    ]
    object_wxyz = np.asarray(row["object_body_wxyz"], dtype=np.float64)[crop, 0]
    object_poses = np.concatenate((object_positions, object_wxyz), axis=-1)
    source_active = np.asarray(row["hand_contact_active"], dtype=bool)[:, crop].T
    contacts = extract_chord_contact_targets(
        joint_names=SOMA_JOINT_NAMES,
        joints_world=joints_world,
        vertices_world=vertices_world,
        faces=reconstruction.faces,
        hand_vertex_indices=reconstruction.hand_vertex_indices,
        object_surface_points=object_points,
        object_surface_normals=object_normals,
        object_poses_wxyz=object_poses,
        source_active=source_active,
        threshold_m=float(args.contact_threshold),
    )

    report = {
        "source": str(source),
        "frame_crop": [start, stop],
        "reconstruction": reconstruction.report.as_dict(),
        "frame_bridge": {
            "rotation": bridge.rotation.tolist(),
            "rotation_relation": bridge.rotation_relation,
            "max_rotation_deviation_rad": bridge.max_rotation_deviation_rad,
            "translation_min": bridge.translations.min(axis=0).tolist(),
            "translation_max": bridge.translations.max(axis=0).tolist(),
        },
        "chord": contacts.report.as_dict(),
        "contact_threshold_m": float(args.contact_threshold),
        "object_surface_points": int(args.surface_points),
        "object_surface_seed": int(args.surface_seed),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            sides=np.asarray(contacts.sides),
            link_names=np.asarray(contacts.link_names),
            active=contacts.active,
            hand_positions=contacts.hand_positions,
            hand_normals=contacts.hand_normals,
            object_positions=contacts.object_positions,
            object_normals=contacts.object_normals,
            object_part_ids=contacts.object_part_ids,
            minimum_distances=contacts.minimum_distances,
            report_json=np.asarray(json.dumps(report, sort_keys=True)),
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
