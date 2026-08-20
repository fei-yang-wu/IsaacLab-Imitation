#!/usr/bin/env python3
"""Create a strict ILTools Manifest from full dexterous Reference NPZs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from iltools.core import (
    DexterousReference,
    create_dexterous_reference_manifest,
    load_dexterous_reference_manifest,
    load_dexterous_reference_npz,
    verify_training_qualification,
)

from validate_vega_wuji_asset import WRIST_FRAME_CONTRACT, validate_vega_wuji_asset


DEFAULT_MODEL = Path(
    "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
    "/vega_u_wuji_v2_beta1_with_mount.xml"
)
DEFAULT_DATASET_NAME = "Vega-Wuji dexterous Reference motions"
EXPECTED_ROBOT_NAME = "vega_wuji"
EXPECTED_REFERENCE_FPS = 20.0
# Isaac Lab applies this speed ceiling to every Vega/Wuji actuator group.  The
# MJCF format does not provide a standard joint velocity-limit field, so this
# task-level ceiling is the fail-closed bound when the Reference is admitted.
MAX_ABS_REFERENCE_JOINT_VELOCITY = 100.0
JOINT_POSITION_LIMIT_TOLERANCE = 1.0e-5
QVEL_CONSISTENCY_ABS_TOLERANCE = 1.0e-3
QVEL_CONSISTENCY_REL_TOLERANCE = 1.0e-2
EXPECTED_WRIST_FRAME_NAMES = {
    side: str(contract["reference_site"])
    for side, contract in WRIST_FRAME_CONTRACT.items()
}
REQUIRED_FINGERTIP_FRAMES = {
    "right": (
        "r_thumb_distal",
        "r_index_finger_distal",
        "r_middle_finger_distal",
        "r_ring_finger_distal",
        "r_pinky_distal",
    ),
    "left": (
        "l_thumb_distal",
        "l_index_finger_distal",
        "l_middle_finger_distal",
        "l_ring_finger_distal",
        "l_pinky_distal",
    ),
}
_SHARED_EXACT_FIELDS = (
    "object_names",
    "object_asset_paths",
    "object_asset_sha256",
    "left_wrist_frame_name",
    "right_wrist_frame_name",
    "left_hand_frame_names",
    "right_hand_frame_names",
    "support_surface_names",
    "support_surface_asset_paths",
    "support_surface_asset_sha256",
    "collision_asset_dependencies",
)
_SHARED_ARRAY_FIELDS = (
    "fixed_root_pose_w",
    "object_scales",
    "object_radii",
    "support_surface_scales",
    "support_surface_poses_w",
)
_SHARED_PHYSICS_ARRAY_FIELDS = (
    "object_mass_kg",
    "object_center_of_mass_m",
    "object_diagonal_inertia_kg_m2",
    "object_static_friction",
    "object_dynamic_friction",
    "object_restitution",
    "support_static_friction",
    "support_dynamic_friction",
    "support_restitution",
)


def _named_contact_links(
    reference: DexterousReference,
    side: str,
) -> tuple[str, ...]:
    """Return the named contact-link row used by the live environment."""

    contacts = reference.contacts
    if contacts is None:
        raise ValueError(
            f"Reference {reference.sequence_id!r} must contain reference contacts."
        )
    try:
        side_index = contacts.hand_sides.index(side)
    except ValueError as exc:
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no {side} contact group."
        ) from exc
    names = tuple(str(name) for name in contacts.link_names[side_index] if str(name))
    if not names:
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no named {side} contact links."
        )
    return names


def _validate_shared_layout(
    reference: DexterousReference,
    first: DexterousReference,
    *,
    first_contact_links: Mapping[str, tuple[str, ...]],
) -> None:
    """Require the scene and contact layout used by one vectorized task."""

    for field_name in _SHARED_EXACT_FIELDS:
        if getattr(reference, field_name) != getattr(first, field_name):
            raise ValueError(
                f"Reference {reference.sequence_id!r} changes {field_name}. "
                "All motions in one task must use one scene contract."
            )
    for field_name in _SHARED_ARRAY_FIELDS:
        if not np.allclose(
            getattr(reference, field_name),
            getattr(first, field_name),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                f"Reference {reference.sequence_id!r} changes {field_name}."
            )
    if (reference.scene_physics is None) != (first.scene_physics is None):
        raise ValueError(f"Reference {reference.sequence_id!r} changes scene_physics.")
    if reference.scene_physics is not None and first.scene_physics is not None:
        for field_name in _SHARED_PHYSICS_ARRAY_FIELDS:
            if not np.allclose(
                getattr(reference.scene_physics, field_name),
                getattr(first.scene_physics, field_name),
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError(
                    f"Reference {reference.sequence_id!r} changes scene physics "
                    f"{field_name}."
                )
    for side in ("right", "left"):
        if _named_contact_links(reference, side) != first_contact_links[side]:
            raise ValueError(
                f"Reference {reference.sequence_id!r} changes {side} contact links."
            )


def _validate_robot_trajectory(
    reference: DexterousReference,
    *,
    joint_names: tuple[str, ...],
    joint_position_limited: np.ndarray,
    joint_position_limits: np.ndarray,
) -> None:
    """Reject robot samples that cannot be one physical 20 Hz trajectory."""

    qpos = np.asarray(reference.qpos, dtype=np.float64)
    qvel = np.asarray(reference.qvel, dtype=np.float64)
    expected_shape = (reference.frame_count, len(joint_names))
    if qpos.shape != expected_shape:
        raise ValueError(
            f"Reference {reference.sequence_id!r} qpos has shape {qpos.shape}, "
            f"expected {expected_shape}."
        )
    if qvel.shape != expected_shape:
        raise ValueError(
            f"Reference {reference.sequence_id!r} qvel has shape {qvel.shape}, "
            f"expected {expected_shape}."
        )
    if not np.isfinite(qpos).all():
        raise ValueError(f"Reference {reference.sequence_id!r} qpos is not finite.")
    if not np.isfinite(qvel).all():
        raise ValueError(f"Reference {reference.sequence_id!r} qvel is not finite.")

    lower = joint_position_limits[:, 0]
    upper = joint_position_limits[:, 1]
    outside = joint_position_limited[None, :] & (
        (qpos < lower[None, :] - JOINT_POSITION_LIMIT_TOLERANCE)
        | (qpos > upper[None, :] + JOINT_POSITION_LIMIT_TOLERANCE)
    )
    if np.any(outside):
        frame_index, joint_index = np.argwhere(outside)[0]
        raise ValueError(
            f"Reference {reference.sequence_id!r} qpos is outside the Vega-Wuji "
            f"joint position limits at frame {frame_index}, joint "
            f"{joint_names[joint_index]!r}: {qpos[frame_index, joint_index]:.9g} "
            f"not in [{lower[joint_index]:.9g}, {upper[joint_index]:.9g}] "
            f"within tolerance {JOINT_POSITION_LIMIT_TOLERANCE:.1e}."
        )

    implausible_velocity = np.abs(qvel) > (
        MAX_ABS_REFERENCE_JOINT_VELOCITY + QVEL_CONSISTENCY_ABS_TOLERANCE
    )
    if np.any(implausible_velocity):
        frame_index, joint_index = np.argwhere(implausible_velocity)[0]
        raise ValueError(
            f"Reference {reference.sequence_id!r} qvel is implausible at frame "
            f"{frame_index}, joint {joint_names[joint_index]!r}: "
            f"{qvel[frame_index, joint_index]:.9g} exceeds the Vega-Wuji "
            f"{MAX_ABS_REFERENCE_JOINT_VELOCITY:.9g} unit/s runtime limit."
        )

    # ILTools defines qvel[k] for k >= 1 as the backward difference from
    # qpos[k - 1] to qpos[k].  qvel[0] is an endpoint estimate and therefore
    # has no independent position interval to validate here.
    expected_qvel = np.diff(qpos, axis=0) * reference.fps
    actual_qvel = qvel[1:]
    inconsistent = ~np.isclose(
        actual_qvel,
        expected_qvel,
        rtol=QVEL_CONSISTENCY_REL_TOLERANCE,
        atol=QVEL_CONSISTENCY_ABS_TOLERANCE,
    )
    if np.any(inconsistent):
        interval_index, joint_index = np.argwhere(inconsistent)[0]
        frame_index = int(interval_index) + 1
        raise ValueError(
            f"Reference {reference.sequence_id!r} qvel is inconsistent with "
            f"qpos and {reference.fps:.9g} Hz at frame {frame_index}, joint "
            f"{joint_names[joint_index]!r}: stored "
            f"{actual_qvel[interval_index, joint_index]:.9g}, finite difference "
            f"{expected_qvel[interval_index, joint_index]:.9g}."
        )


def validate_task_references(
    references: Sequence[DexterousReference],
    *,
    actuator_joint_names: Sequence[str],
    actuator_joint_position_limited: Sequence[bool],
    actuator_joint_position_limits: Sequence[Sequence[float]],
    robot_body_names: Sequence[str],
    require_training_qualification: bool = True,
) -> None:
    """Validate the complete task contract before the Manifest is written."""

    if not references:
        raise ValueError("At least one reference path is required.")
    target_joint_names = tuple(str(name) for name in actuator_joint_names)
    position_limited = np.asarray(actuator_joint_position_limited, dtype=np.bool_)
    position_limits = np.asarray(actuator_joint_position_limits, dtype=np.float64)
    joint_count = len(target_joint_names)
    if position_limited.shape != (joint_count,):
        raise ValueError(
            "Vega-Wuji actuator joint limit flags do not match the actuator order."
        )
    if position_limits.shape != (joint_count, 2):
        raise ValueError(
            "Vega-Wuji actuator joint position limits do not match the actuator order."
        )
    if not np.isfinite(position_limits[position_limited]).all():
        raise ValueError("Vega-Wuji limited joint position ranges must be finite.")
    if np.any(position_limited & (position_limits[:, 0] >= position_limits[:, 1])):
        raise ValueError("Vega-Wuji limited joint position ranges must be ordered.")
    model_body_names = frozenset(str(name) for name in robot_body_names)
    missing_fingertip_bodies = [
        name
        for side in ("right", "left")
        for name in REQUIRED_FINGERTIP_FRAMES[side]
        if name not in model_body_names
    ]
    if missing_fingertip_bodies:
        raise ValueError(
            "Vega-Wuji MJCF is missing required fingertip bodies: "
            f"{missing_fingertip_bodies}."
        )
    first = references[0]
    first_contact_links = {
        side: _named_contact_links(first, side) for side in ("right", "left")
    }

    for reference in references:
        if reference.robot_name != EXPECTED_ROBOT_NAME:
            raise ValueError(
                f"Reference {reference.sequence_id!r} targets "
                f"{reference.robot_name!r}, expected {EXPECTED_ROBOT_NAME!r}."
            )
        if not math.isclose(
            reference.fps,
            EXPECTED_REFERENCE_FPS,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"Reference {reference.sequence_id!r} uses {reference.fps} Hz, "
                f"expected {EXPECTED_REFERENCE_FPS} Hz."
            )
        if reference.joint_names != target_joint_names:
            raise ValueError(
                f"Reference {reference.sequence_id!r} joint_names do not match "
                "the Vega-Wuji MJCF actuator joint order."
            )
        _validate_robot_trajectory(
            reference,
            joint_names=target_joint_names,
            joint_position_limited=position_limited,
            joint_position_limits=position_limits,
        )
        for side, expected_frame_name in EXPECTED_WRIST_FRAME_NAMES.items():
            frame_name = str(getattr(reference, f"{side}_wrist_frame_name"))
            if frame_name != expected_frame_name:
                raise ValueError(
                    f"Reference {reference.sequence_id!r} {side} wrist frame is "
                    f"{frame_name!r}, expected {expected_frame_name!r}."
                )
        if not reference.object_names:
            raise ValueError(
                f"Reference {reference.sequence_id!r} must contain at least one "
                "rigid object."
            )
        if any(not path for path in reference.object_asset_paths):
            raise ValueError(
                f"Reference {reference.sequence_id!r} has an object without an "
                "asset path."
            )
        if np.any(np.asarray(reference.object_radii) <= 0.0):
            raise ValueError(
                f"Reference {reference.sequence_id!r} has a non-positive object radius."
            )
        if any(not path for path in reference.support_surface_asset_paths):
            raise ValueError(
                f"Reference {reference.sequence_id!r} has a support surface "
                "without an asset path."
            )
        for side, required_names in REQUIRED_FINGERTIP_FRAMES.items():
            frame_names = tuple(getattr(reference, f"{side}_hand_frame_names"))
            missing = [name for name in required_names if name not in frame_names]
            if missing:
                raise ValueError(
                    f"Reference {reference.sequence_id!r} is missing {side} hand "
                    f"frames: {missing}."
                )
            contact_links = _named_contact_links(reference, side)
            missing_contact_bodies = [
                name for name in contact_links if name not in model_body_names
            ]
            if missing_contact_bodies:
                raise ValueError(
                    f"Reference {reference.sequence_id!r} {side} contact links "
                    "are not Vega-Wuji MJCF bodies: "
                    f"{missing_contact_bodies}."
                )
        _validate_shared_layout(
            reference,
            first,
            first_contact_links=first_contact_links,
        )
        if require_training_qualification:
            verify_training_qualification(reference)


def build_manifest(
    references: Sequence[str | Path],
    *,
    output_path: str | Path,
    model_path: str | Path | None = DEFAULT_MODEL,
    dataset_name: str = DEFAULT_DATASET_NAME,
    metadata: Mapping[str, Any] | None = None,
    allow_unqualified_inspection: bool = False,
) -> dict[str, Any]:
    """Create and reload one canonical ILTools dexterous Reference Manifest.

    Training qualification is mandatory by default. The inspection escape is
    explicit and marks the output Manifest as unsuitable for training.
    """

    if model_path is None:
        raise ValueError("A Vega-Wuji JSON Manifest must declare its robot model.")
    reference_paths = tuple(references)
    references_loaded = tuple(
        load_dexterous_reference_npz(reference_path)
        for reference_path in reference_paths
    )
    model_record = validate_vega_wuji_asset(model_path)
    if not model_record["valid"]:
        issues = "; ".join(str(issue) for issue in model_record["issues"])
        raise ValueError(f"Invalid Vega-Wuji robot-only MJCF: {issues}")
    validate_task_references(
        references_loaded,
        actuator_joint_names=model_record["actuator_joint_names"],
        actuator_joint_position_limited=model_record["actuator_joint_position_limited"],
        actuator_joint_position_limits=model_record["actuator_joint_position_limits"],
        robot_body_names=model_record["body_names"],
        require_training_qualification=not allow_unqualified_inspection,
    )
    manifest_metadata = dict(metadata or {})
    manifest_metadata["training_qualified"] = not allow_unqualified_inspection
    manifest_metadata["inspection_only"] = allow_unqualified_inspection
    manifest_path = create_dexterous_reference_manifest(
        reference_paths,
        output_path,
        dataset_name=dataset_name,
        model_path=model_path,
        metadata=manifest_metadata,
    )
    return load_dexterous_reference_manifest(manifest_path).to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        action="append",
        required=True,
        help="Full ILTools dexterous Reference NPZ; repeat for each motion.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--allow-unqualified-inspection",
        action="store_true",
        help=(
            "Write an inspection-only Manifest without the typed training gate. "
            "The output is marked as not training-qualified."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_manifest(
        args.npz,
        output_path=args.output,
        model_path=args.model,
        dataset_name=args.dataset_name,
        allow_unqualified_inspection=args.allow_unqualified_inspection,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
