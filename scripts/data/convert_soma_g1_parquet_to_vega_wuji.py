#!/usr/bin/env python3
"""Convert the pinned SOMA corn-can handover to a Vega-Wuji Reference.

The upstream Parquet contains real G1 wrist and object poses. Its G1 finger
columns are zero, but its pinned source payload contains the full 77-joint
SOMA hand skeleton. This converter derives bounded Wuji finger hinges from
that skeleton and solves both Vega wrists with ILTools. A documented
contact-phase workspace projection keeps active wrist/object geometry while
parking the inactive fixed-base arm at its neutral palm pose.

The selected source has only a per-hand binary contact signal. It has no
contact link names, points, normals, or object part identifiers. Conversion
therefore fails by default. ``--inspection-only`` writes an
explicitly non-runtime-qualified Reference for inspection without inventing
contact geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
from itertools import combinations
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import qmc

from iltools.core import (
    ContactSequence,
    DexterousReference,
    ScenePhysics,
    Trajectory,
    build_urdf_collision_asset_dependencies,
    save_dexterous_reference_npz,
)
from iltools.retarget import (
    CollisionProjectionConfig,
    ContactConstraintProjectionConfig,
    ContactTargetIKConfig,
    KeypointJointSpec,
    KeypointRetargeter,
    MujocoCollisionProjector,
    MujocoContactConstraintProjector,
    MujocoContactTargetRetargeter,
    MujocoDualHandRetargeter,
    MujocoKeypointProjectionConfig,
    MujocoKeypointProjector,
    MujocoRadialEscapeProjector,
    MujocoSceneCollisionClosureProjector,
    MujocoSelfCollisionClosureProjector,
    RadialEscapeProjectionConfig,
    SceneCollisionClosureProjectionConfig,
    SelfCollisionProjectionConfig,
    dexterous_reference_from_trajectory,
    discrete_time_stretch_indices,
    extract_chord_contact_targets,
    infer_soma_frame_bridge,
    reconstruct_soma_identity_rest_pose,
    reconstruct_soma_motion,
    sample_object_surface,
)
from iltools.retarget.wuji_finger import (
    FingertipAvoidanceConfig,
    FingertipIkConfig,
    WUJI_ANATOMICAL_LIMITS_DEG,
    apply_anatomical_finger_limits,
    audit_finger_joints,
    evaluate_finger_gates,
    fit_palm_yaw_from_finger_directions,
    fit_source_palm_frame_from_knuckles,
    measure_fingertip_residuals,
    solve_fingertip_trajectory,
    wuji_finger_directions_palm_frame,
    wuji_knuckle_positions_palm_frame,
)
from iltools.retarget.wuji_finger import _axis_rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO_ROOT
    / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
    / "vega_u_wuji_v2_beta1_with_mount.xml"
)
SOMA_DATA_ROOT = Path(
    "/home/fwu91/Documents/DexManip/video_to_data/robotic_grounding/source/"
    "robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body"
)


@dataclass(frozen=True)
class SequencePin:
    """One pinned SOMA source sequence with its verified asset contract.

    The support cylinder values are the explicit ``Cylinder`` prim attributes
    of the pinned support USDA (height/radius/translate), copied verbatim.
    """

    sequence_id: str
    source_path: Path
    support_path: Path
    parquet_sha256: str
    payload_sha256: str
    support_sha256: str
    object_asset_sha256: Mapping[str, str]
    upstream_commit: str
    start_frame: int
    end_frame_exclusive: int
    support_center_source: tuple[float, float, float]
    support_height_source: float
    support_radius_source: float
    object_anchor_target: tuple[float, float, float]
    runtime_object_name: str


PINNED_SEQUENCES: dict[str, SequencePin] = {
    "corn_can_handover": SequencePin(
        sequence_id="2026-03-23_18-10-01_corn_can_right_left_handover_01",
        source_path=(
            SOMA_DATA_ROOT
            / "soma/sequence_id=2026-03-23_18-10-01_corn_can_right_left_handover_01"
            / "robot_name=g1/data.parquet"
        ),
        support_path=(
            SOMA_DATA_ROOT
            / "reconstructed_stage"
            / "2026-03-23_18-10-01_corn_can_right_left_handover_01_support.usda"
        ),
        parquet_sha256=(
            "1679e920f20cc741ffccaf8164a960bf46e1f124d670d7bddb2416ca6ea8b082"
        ),
        payload_sha256=(
            "5acf75dee07040e3bd826d50d0843c84185f56352eb24afb8b1b6b5bc311a229"
        ),
        support_sha256=(
            "c25529df26f73f7ba2ec205fdb9d54716d5591f2079c7901d1e1763d94e41631"
        ),
        object_asset_sha256={
            "textured_mesh.urdf": (
                "3ca858449cf9ee7c1d19e4e17d49558ebd9d170186a6ec226e1961c6c5250975"
            ),
            "textured_mesh.obj": (
                "d0668549cbcf73a9c7935b6690485a6a51e475fcbf2e68d73108be0eb5b532ed"
            ),
        },
        upstream_commit="afa9dffda748e301ddf79b7002ce4cc5cb552adb",
        start_frame=522,
        end_frame_exclusive=840,
        support_center_source=(
            -1.3406889001888564,
            0.3631554151795554,
            0.8561500119144527,
        ),
        support_height_source=0.01,
        support_radius_source=0.4337941740262468,
        object_anchor_target=(0.8, 0.15, 1.14),
        runtime_object_name="corn_can",
    ),
    # Bimanual snack-box grasp+lift. The crop [400, 480) at 200 Hz holds the
    # pre-grasp reach, both contacts closing (from ~f440), and the lift
    # (+0.13 m); it ends before the carrier starts walking (max object XY
    # travel 0.287 m inside the crop). Chosen 2026-08-24 from the source
    # renders in logs/reference_replay/source_motion_candidates/.
    "snack_box_pick": SequencePin(
        sequence_id="2026-03-06_10-24-18_snack_box_pick_and_place_01",
        source_path=(
            SOMA_DATA_ROOT
            / "soma/sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01"
            / "robot_name=g1/data.parquet"
        ),
        support_path=(
            SOMA_DATA_ROOT
            / "reconstructed_stage"
            / "2026-03-06_10-24-18_snack_box_pick_and_place_01_support.usda"
        ),
        parquet_sha256=(
            "04f55c50a6dfc04826b73b22cc42cde90d9209e003d242b3d300b0a631c9671d"
        ),
        payload_sha256=(
            "9dcd982dd13361ab927a017739da420f0b8651f658329fed8f0063234f7f3ae4"
        ),
        support_sha256=(
            "1a31d46961bbc4d502fafa09f397f153dc3ad07362850e890e25e960cfd29678"
        ),
        object_asset_sha256={
            "textured_mesh.urdf": (
                "3ca858449cf9ee7c1d19e4e17d49558ebd9d170186a6ec226e1961c6c5250975"
            ),
            "textured_mesh.obj": (
                "380248270cb7e23aed66c33e74d100e1e4dd5e329ff3d8d63ac85753c859066a"
            ),
        },
        upstream_commit="afa9dffda748e301ddf79b7002ce4cc5cb552adb",
        start_frame=400,
        end_frame_exclusive=480,
        support_center_source=(
            1.8040543894793986,
            0.05697129174318872,
            0.7503690559680148,
        ),
        support_height_source=0.01,
        support_radius_source=0.22521347938612124,
        # Box mid-height at the neutral palm height (1.078 m) and its side
        # faces (+/-0.24 m) at the neutral palm spacing (+/-0.229 m).
        object_anchor_target=(0.65, 0.0, 1.05),
        runtime_object_name="snack_box",
    ),
}
DEFAULT_SEQUENCE_KEY = "corn_can_handover"
_CORN_CAN_PIN = PINNED_SEQUENCES[DEFAULT_SEQUENCE_KEY]

# Backwards-compatible aliases for the historical single-sequence pin names.
DEFAULT_SOURCE = _CORN_CAN_PIN.source_path
DEFAULT_SUPPORT = _CORN_CAN_PIN.support_path
PINNED_PARQUET_SHA256 = _CORN_CAN_PIN.parquet_sha256
PINNED_PAYLOAD_SHA256 = _CORN_CAN_PIN.payload_sha256
PINNED_OBJECT_ASSET_SHA256 = _CORN_CAN_PIN.object_asset_sha256
PINNED_SUPPORT_SHA256 = _CORN_CAN_PIN.support_sha256
PINNED_UPSTREAM_COMMIT = _CORN_CAN_PIN.upstream_commit
PINNED_SEQUENCE_ID = _CORN_CAN_PIN.sequence_id
PINNED_START_FRAME = _CORN_CAN_PIN.start_frame
PINNED_END_FRAME_EXCLUSIVE = _CORN_CAN_PIN.end_frame_exclusive
TARGET_FPS = 20.0
DEFAULT_FILTER_CUTOFF_HZ = 6.0
DEFAULT_FILTER_ORDER = 4
DEFAULT_MOTION_TIME_SCALE = 1.0
DEFAULT_OBJECT_ANCHOR_TARGET = np.asarray((0.8, 0.15, 1.14), dtype=np.float64)
DEFAULT_GLOBAL_MOTION_SCALE = 0.40
DEFAULT_LOCAL_GEOMETRY_SCALE = 0.75
DEFAULT_CONTACT_DILATION_FRAMES = 41
DEFAULT_CONTACT_BLEND_SIGMA_FRAMES = 15.0
DEFAULT_IK_ORIENTATION_WEIGHT = 0.03
DEFAULT_IK_DAMPING = 0.30
DEFAULT_SOMA_TO_WUJI_HAND_SCALE = 1.0
DEFAULT_FINGER_POSTURE_PRIOR = 2.0e-3
DEFAULT_PALM_OBJECT_STANDOFF_M = 0.28
DEFAULT_PALM_SUPPORT_STANDOFF_M = 0.20
DEFAULT_GEOMETRY_CLEARANCE_M = 0.005
DEFAULT_DENSE_GEOMETRY_AUDIT_FPS = 1000.0
BENT_ELBOW_INACTIVE_REST_QPOS = {
    "L_arm_j1": 2.06,
    "L_arm_j2": -0.10,
    "L_arm_j3": -0.28,
    "L_arm_j4": -1.81,
    "L_arm_j5": -0.06,
    "L_arm_j6": -0.90,
    "L_arm_j7": 0.08,
    "R_arm_j1": -2.06,
    "R_arm_j2": 0.10,
    "R_arm_j3": 0.28,
    "R_arm_j4": -1.81,
    "R_arm_j5": 0.06,
    "R_arm_j6": 0.90,
    "R_arm_j7": -0.08,
}
PINNED_SUPPORT_CENTER_SOURCE = np.asarray(
    _CORN_CAN_PIN.support_center_source, dtype=np.float64
)
PINNED_SUPPORT_HEIGHT_SOURCE = _CORN_CAN_PIN.support_height_source
PINNED_SUPPORT_RADIUS_SOURCE = _CORN_CAN_PIN.support_radius_source
DEFAULT_FIXED_ROOT_POSE_W = np.asarray(
    (0.0, 0.0, 0.19, 1.0, 0.0, 0.0, 0.0), dtype=np.float64
)

SOMA_JOINT_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumbEnd",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandIndexEnd",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandMiddleEnd",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandRingEnd",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandPinkyEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumbEnd",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandIndexEnd",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandRingEnd",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandPinkyEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
)
SOMA_JOINT_NAMES_SHA256 = hashlib.sha256(
    json.dumps(SOMA_JOINT_NAMES, separators=(",", ":")).encode("utf-8")
).hexdigest()
EXPECTED_SOMA_JOINT_NAMES_SHA256 = (
    "30b0331242f84f9cc5b84219ebda363948cb9fe11567fa2b65e903011dd1a415"
)
if SOMA_JOINT_NAMES_SHA256 != EXPECTED_SOMA_JOINT_NAMES_SHA256:
    raise RuntimeError("The pinned SOMA joint-name contract changed.")

FINGERTIP_NAMES = {
    "left": (
        "l_thumb_distal",
        "l_index_finger_distal",
        "l_middle_finger_distal",
        "l_ring_finger_distal",
        "l_pinky_distal",
    ),
    "right": (
        "r_thumb_distal",
        "r_index_finger_distal",
        "r_middle_finger_distal",
        "r_ring_finger_distal",
        "r_pinky_distal",
    ),
}
WUJI_FINGERTIP_SITE_NAMES = {
    side: (
        f"{prefix}_thumb_tip",
        f"{prefix}_index_finger_tip",
        f"{prefix}_middle_finger_tip",
        f"{prefix}_ring_finger_tip",
        f"{prefix}_pinky_tip",
    )
    for side, prefix in (("left", "l"), ("right", "r"))
}
SOMA_HAND_KEYPOINT_NAMES = {
    side: (
        f"{prefix}Hand",
        f"{prefix}HandThumbEnd",
        f"{prefix}HandIndexEnd",
        f"{prefix}HandMiddleEnd",
        f"{prefix}HandRingEnd",
        f"{prefix}HandPinkyEnd",
    )
    for side, prefix in (("left", "Left"), ("right", "Right"))
}
SOMA_HAND_KNUCKLE_NAMES = {
    side: tuple(
        # SOMA's non-thumb chain is Hand -> *1 (metacarpal/CMC base) ->
        # *2 (MCP knuckle) -> *3 (PIP) -> *4 (DIP) -> *End.  The Wuji palm
        # fit uses identity-zero MCP geometry, not posed MCPs or metacarpal
        # bases. The posed *2 points are not rigid because *1 articulates.
        f"{prefix}Hand{finger}2"
        for finger in ("Index", "Middle", "Ring", "Pinky")
    )
    for side, prefix in (("left", "Left"), ("right", "Right"))
}
SOMA_HAND_PIP_NAMES = {
    # The PIP joint of each non-thumb finger. MCP (*2) to PIP (*3) is the
    # proximal phalanx, whose direction defines the finger fan used by the
    # palm-yaw finger-direction fit.
    side: tuple(
        f"{prefix}Hand{finger}3" for finger in ("Index", "Middle", "Ring", "Pinky")
    )
    for side, prefix in (("left", "Left"), ("right", "Right"))
}
SOMA_CHORD_LINK_KEYS = (
    "palm",
    "thumb1",
    "thumb2",
    "thumb3",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "ring1",
    "ring2",
    "ring3",
    "pinky1",
    "pinky2",
    "pinky3",
)
WUJI_CHORD_CONTACT_BODIES = {
    side: (
        f"{prefix}_mount",
        f"{prefix}_thumb_proximal_abd",
        f"{prefix}_thumb_middle",
        f"{prefix}_thumb_distal",
        f"{prefix}_index_finger_proximal_abd",
        f"{prefix}_index_finger_middle",
        f"{prefix}_index_finger_distal",
        f"{prefix}_middle_finger_proximal_abd",
        f"{prefix}_middle_finger_middle",
        f"{prefix}_middle_finger_distal",
        f"{prefix}_ring_finger_proximal_abd",
        f"{prefix}_ring_finger_middle",
        f"{prefix}_ring_finger_distal",
        f"{prefix}_pinky_proximal_abd",
        f"{prefix}_pinky_middle",
        f"{prefix}_pinky_distal",
    )
    for side, prefix in (("left", "l"), ("right", "r"))
}
REQUIRED_SOURCE_FIELDS = (
    "schema_version",
    "motion_kind",
    "source_dataset",
    "fps",
    "coord_frame",
    "ee_link_names",
    "ee_pose_w",
    "hand_sides",
    "object_name",
    "safe_object_name",
    "object_body_names",
    "object_urdf_paths",
    "object_mesh_paths",
    "object_mesh_radius",
    "object_body_position",
    "object_body_wxyz",
    "hand_contact_link_names",
    "hand_link_contact_positions",
    "hand_link_contact_normals",
    "hand_object_contact_positions",
    "hand_object_contact_normals",
    "hand_object_contact_part_ids",
    "hand_contact_active",
    "source_kind",
    "source_payload",
)


@dataclass(frozen=True, slots=True)
class ContactPhaseProjection:
    """Fixed-yaw projection with separate global and local spatial scales."""

    rotation: Rotation
    source_object_anchor: np.ndarray
    target_object_anchor: np.ndarray
    global_motion_scale: float
    local_geometry_scale: float
    target_object_anchor_trajectory: np.ndarray | None = None
    # Support cylinder of the pinned sequence, in the source frame. Defaults
    # keep historical constructions (tests, corn-can) unchanged.
    support_center_source: tuple[float, float, float] = (
        _CORN_CAN_PIN.support_center_source
    )
    support_height_source: float = _CORN_CAN_PIN.support_height_source

    def __post_init__(self) -> None:
        source = np.asarray(self.source_object_anchor, dtype=np.float64)
        target = np.asarray(self.target_object_anchor, dtype=np.float64)
        if source.shape != (3,) or target.shape != (3,):
            raise ValueError("Object anchors must have shape (3,).")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("Object anchors must be finite.")
        if (
            not np.isfinite((self.global_motion_scale, self.local_geometry_scale)).all()
            or self.global_motion_scale <= 0.0
            or self.local_geometry_scale <= 0.0
        ):
            raise ValueError("Projection scales must be finite and positive.")
        if self.target_object_anchor_trajectory is not None:
            trajectory = np.asarray(
                self.target_object_anchor_trajectory, dtype=np.float64
            )
            if trajectory.ndim != 2 or trajectory.shape[1:] != (3,):
                raise ValueError(
                    "target_object_anchor_trajectory must have shape [T, 3]."
                )
            if not np.isfinite(trajectory).all():
                raise ValueError("target_object_anchor_trajectory must be finite.")

    def _target_anchors(self, frame_count: int) -> np.ndarray:
        if self.target_object_anchor_trajectory is None:
            return np.broadcast_to(
                np.asarray(self.target_object_anchor, dtype=np.float64),
                (frame_count, 3),
            )
        trajectory = np.asarray(self.target_object_anchor_trajectory, dtype=np.float64)
        if trajectory.shape[0] != frame_count:
            raise ValueError(
                "target_object_anchor_trajectory length does not match the "
                f"trajectory: {trajectory.shape[0]} != {frame_count}."
            )
        return trajectory

    def object_positions(self, values: np.ndarray) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("Object positions must have shape [T, objects, 3].")
        flat = (points - self.source_object_anchor).reshape(-1, 3)
        rotated = self.rotation.apply(flat).reshape(points.shape)
        return self._target_anchors(len(points))[:, None, :] + (
            self.global_motion_scale * rotated
        )

    def active_wrist_positions(
        self, wrist_positions: np.ndarray, object_positions: np.ndarray
    ) -> np.ndarray:
        wrists = np.asarray(wrist_positions, dtype=np.float64)
        objects = np.asarray(object_positions, dtype=np.float64)
        if wrists.shape[:-2] != objects.shape[:-2] or wrists.shape[-2:] != (2, 3):
            raise ValueError("Wrist positions must have shape [T, 2, 3].")
        if objects.shape[-2:] != (1, 3):
            raise ValueError("Object positions must have shape [T, 1, 3].")
        object_target = self.object_positions(objects)
        relative = (wrists - objects).reshape(-1, 3)
        local_target = self.local_geometry_scale * self.rotation.apply(relative)
        return object_target + local_target.reshape(wrists.shape)

    def orientations(self, wxyz: np.ndarray) -> np.ndarray:
        quaternions = np.asarray(wxyz, dtype=np.float64)
        source = _rotation_from_wxyz(quaternions.reshape(-1, 4))
        transformed = _rotation_to_wxyz(self.rotation * source)
        return transformed.reshape(quaternions.shape)

    def support_asset_pose_wxyz(self) -> np.ndarray:
        """Pose a source-world USD so its local geometry uses the local scale."""

        translation = self.target_object_anchor - self.local_geometry_scale * (
            self.rotation.apply(self.source_object_anchor)
        )
        return np.concatenate((translation, _rotation_to_wxyz(self.rotation)[0]))

    def support_center_target(self) -> np.ndarray:
        return self.target_object_anchor + self.local_geometry_scale * (
            self.rotation.apply(
                np.asarray(self.support_center_source, dtype=np.float64)
                - self.source_object_anchor
            )
        )

    def support_top_z(self) -> float:
        return float(
            self.support_center_target()[2]
            + 0.5 * self.local_geometry_scale * self.support_height_source
        )


def _phase_object_anchor_trajectory(
    base_anchor: np.ndarray,
    phase_anchors: np.ndarray,
    blend_weights: np.ndarray,
) -> np.ndarray:
    """Blend one target object anchor per hand contact phase.

    The object remains one rigid body. The phase targets only change its
    *placement* over the demonstration, and the same blended placement is
    used for the object and both active wrists. When no hand is active, the
    trajectory returns to the explicit base anchor. This keeps the old fixed
    anchor path available and makes the phase-aware path auditable as a
    deterministic transform rather than a second IK target.
    """

    base = _as_finite_array(base_anchor, name="base object anchor", shape=(3,))
    targets = _as_finite_array(phase_anchors, name="phase object anchors", shape=(2, 3))
    weights = np.asarray(blend_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != 2:
        raise ValueError("Phase blend weights must have shape [2, T].")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Phase blend weights must be finite and non-negative.")
    weights = np.clip(weights, 0.0, 1.0)
    total = np.sum(weights, axis=0)
    weighted = np.einsum("st,sd->td", weights, targets)
    active_target = np.divide(
        weighted,
        np.maximum(total, 1.0e-12)[:, None],
        out=np.broadcast_to(base, (len(total), 3)).copy(),
        where=total[:, None] > 1.0e-12,
    )
    # The maximum side weight is the amount by which a phase target is
    # trusted. With both hands active, their targets are averaged; with no
    # hand active, the explicit base anchor is retained.
    alpha = np.max(weights, axis=0)
    return (1.0 - alpha[:, None]) * base[None, :] + alpha[:, None] * active_target


def _infer_phase_object_anchors(
    wrist_positions: np.ndarray,
    object_positions: np.ndarray,
    source_active: np.ndarray,
    *,
    rotation: Rotation,
    local_geometry_scale: float,
    neutral_positions: np.ndarray,
    base_anchor: np.ndarray,
    max_shift_m: float = 0.20,
) -> np.ndarray:
    """Place each contact phase near the corresponding neutral palm.

    This is the reachability-first surrogate. It does not claim that the
    neutral palm is the full reachable workspace; it removes the known global
    placement error before the expensive arm IK and collision audits. Only X
    and Y are adjusted. Z stays at the explicit anchor because the can/support
    vertical contract is independent of arm reach.
    """

    wrists = _as_finite_array(
        wrist_positions,
        name="phase wrist positions",
        shape=(len(wrist_positions), 2, 3),
    )
    objects = _as_finite_array(
        object_positions,
        name="phase object positions",
        shape=(len(wrists), 1, 3),
    )
    active = np.asarray(source_active, dtype=bool)
    if active.shape != (2, len(wrists)):
        raise ValueError("source_active must have shape [2, T].")
    neutral = _as_finite_array(
        neutral_positions, name="phase neutral palm positions", shape=(2, 3)
    )
    base = _as_finite_array(base_anchor, name="phase base anchor", shape=(3,))
    if not np.isfinite(local_geometry_scale) or local_geometry_scale <= 0.0:
        raise ValueError("local_geometry_scale must be finite and positive.")
    if not np.isfinite(max_shift_m) or max_shift_m <= 0.0:
        raise ValueError("max_shift_m must be finite and positive.")

    relative = wrists - objects
    local_offsets = local_geometry_scale * rotation.apply(relative.reshape(-1, 3))
    local_offsets = local_offsets.reshape(relative.shape)
    result = np.repeat(base[None, :], 2, axis=0)
    for side in range(2):
        mask = active[side]
        if not np.any(mask):
            continue
        # Median is stable against the short handover transition and against
        # the filtered endpoint overshoot in the source trajectory.
        candidate = neutral[side] - np.median(local_offsets[mask, side], axis=0)
        shift = candidate - base
        shift[2] = 0.0
        shift[:2] = np.clip(shift[:2], -max_shift_m, max_shift_m)
        result[side] = base + shift
    return result


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Load payloads that contain only primitive Python containers."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"source_payload requests forbidden global {module}.{name}"
        )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_float32_array_sha256(value: np.ndarray) -> str:
    """Hash an array exactly as the Reference NPZ stores numeric trajectories."""

    array = np.ascontiguousarray(value, dtype="<f4")
    shape = ",".join(str(size) for size in array.shape).encode("ascii")
    return _sha256_bytes(b"float32:" + shape + b"\0" + array.tobytes())


def _safe_payload(payload: bytes, *, expected_sha256: str) -> dict[str, Any]:
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError(
            "source_payload SHA-256 mismatch; refusing to unpickle: "
            f"expected {expected_sha256}, got {actual}."
        )
    result = _NoGlobalsUnpickler(io.BytesIO(payload)).load()
    if not isinstance(result, dict):
        raise ValueError("source_payload must contain one dictionary.")
    required = {
        "soma_identity_coeffs",
        "soma_scale_params",
        "soma_joints",
        "soma_joints_wxyz",
        "nvhuman_head_translation",
        "nvhuman_head_wxyz",
        "nvhuman_root_translation",
        "nvhuman_root_wxyz",
    }
    if set(result) != required:
        raise ValueError(
            "source_payload keys differ from the pinned SOMA contract: "
            f"expected {sorted(required)}, got {sorted(result)}."
        )
    return result


def _read_parquet_row(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet conversion requires the Isaac Lab Pixi environment. Run "
            "with `pixi run -e isaaclab`."
        ) from exc

    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"Expected one nested motion row, found {table.num_rows}.")
    row = {name: table[name][0].as_py() for name in table.column_names}
    for parent in path.parents:
        match = re.fullmatch(r"(sequence_id|robot_name)=(.+)", parent.name)
        if match and match.group(1) not in row:
            row[match.group(1)] = match.group(2)
    missing = [name for name in REQUIRED_SOURCE_FIELDS if name not in row]
    if missing:
        raise ValueError(f"Source Parquet is missing required fields: {missing}.")
    return row


def _as_finite_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None:
        if result.ndim != len(shape) or any(
            expected is not None and result.shape[index] != expected
            for index, expected in enumerate(shape)
        ):
            raise ValueError(f"{name} must have shape {shape}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _validate_source(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    expected_sequence_id: str = PINNED_SEQUENCE_ID,
) -> dict[str, np.ndarray]:
    expected_scalars = {
        "schema_version": "motion_v1",
        "motion_kind": "single_robot",
        "source_dataset": "soma",
        "source_kind": "soma",
        "coord_frame": "robot_base_z_up",
        "sequence_id": expected_sequence_id,
        "robot_name": "g1",
    }
    for name, expected in expected_scalars.items():
        if row.get(name) != expected:
            raise ValueError(
                f"Pinned source requires {name}={expected!r}, got {row.get(name)!r}."
            )
    source_fps = float(row["fps"])
    if not math.isclose(source_fps, 200.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"Pinned source requires 200 Hz, got {source_fps}.")

    ee = _as_finite_array(row["ee_pose_w"], name="ee_pose_w")
    if ee.ndim != 3 or ee.shape[1:] != (2, 7):
        raise ValueError(f"ee_pose_w must have shape [T, 2, 7], got {ee.shape}.")
    frame_count = len(ee)
    if tuple(str(value) for value in row["ee_link_names"]) != (
        "left_hand_palm_link",
        "right_hand_palm_link",
    ):
        raise ValueError("Pinned source end-effector names changed.")
    if tuple(str(value) for value in row["hand_sides"]) != ("left", "right"):
        raise ValueError("Pinned source hand order changed.")
    if tuple(str(value) for value in row["object_body_names"]) != ("object",):
        raise ValueError("Pinned source must contain one rigid object body.")

    arrays = {
        "ee_pose_w": ee,
        "robot_root_position": _as_finite_array(
            row["robot_root_position"],
            name="robot_root_position",
            shape=(frame_count, 3),
        ),
        "robot_root_wxyz": _unit_wxyz(
            _as_finite_array(
                row["robot_root_wxyz"],
                name="robot_root_wxyz",
                shape=(frame_count, 4),
            ),
            name="robot_root_wxyz",
        ),
        "object_body_position": _as_finite_array(
            row["object_body_position"],
            name="object_body_position",
            shape=(frame_count, 1, 3),
        ),
        "object_body_wxyz": _unit_wxyz(
            _as_finite_array(
                row["object_body_wxyz"],
                name="object_body_wxyz",
                shape=(frame_count, 1, 4),
            ),
            name="object_body_wxyz",
        ),
        "hand_contact_active": _as_finite_array(
            row["hand_contact_active"],
            name="hand_contact_active",
            shape=(2, frame_count),
        ),
        "soma_joints": _as_finite_array(
            payload["soma_joints"],
            name="source_payload.soma_joints",
            shape=(frame_count, len(SOMA_JOINT_NAMES), 3),
        ),
        "soma_joints_wxyz": _unit_wxyz(
            _as_finite_array(
                payload["soma_joints_wxyz"],
                name="source_payload.soma_joints_wxyz",
                shape=(frame_count, len(SOMA_JOINT_NAMES), 4),
            ),
            name="source_payload.soma_joints_wxyz",
        ),
    }
    if np.any(
        (arrays["hand_contact_active"] < 0.0) | (arrays["hand_contact_active"] > 1.0)
    ):
        raise ValueError("hand_contact_active must stay within [0, 1].")
    return arrays


def _unit_wxyz(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-8):
        raise ValueError(f"{name} contains a zero quaternion.")
    if not np.allclose(norm, 1.0, rtol=0.0, atol=1.0e-3):
        raise ValueError(f"{name} must contain unit WXYZ quaternions.")
    return result / norm


def _normalize_wxyz(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    if not np.isfinite(result).all() or np.any(norm <= 1.0e-8):
        raise ValueError(f"{name} contains an invalid quaternion.")
    return result / norm


def _continuous_wxyz(value: np.ndarray) -> np.ndarray:
    result = _unit_wxyz(value, name="quaternion series").copy()
    flat = result.reshape(result.shape[0], -1, 4)
    for stream in range(flat.shape[1]):
        for frame in range(1, len(flat)):
            if np.dot(flat[frame - 1, stream], flat[frame, stream]) < 0.0:
                flat[frame, stream] *= -1.0
    return result


def _rotation_from_wxyz(value: np.ndarray) -> Rotation:
    quaternions = _unit_wxyz(value, name="WXYZ quaternion")
    return Rotation.from_quat(quaternions[..., (1, 2, 3, 0)])


def _rotation_to_wxyz(value: Rotation) -> np.ndarray:
    xyzw = np.atleast_2d(value.as_quat())
    return xyzw[:, (3, 0, 1, 2)]


def _lowpass(
    values: np.ndarray,
    *,
    fps: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if not 0.0 < cutoff_hz < 0.5 * fps:
        raise ValueError("filter cutoff must be between zero and source Nyquist.")
    if order < 1:
        raise ValueError("filter order must be positive.")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fps, output="sos")
    try:
        return sosfiltfilt(sos, data, axis=0)
    except ValueError as exc:
        raise ValueError(
            "Source segment is too short for zero-phase filtering."
        ) from exc


def _lowpass_wxyz(
    values: np.ndarray,
    *,
    fps: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    continuous = _continuous_wxyz(values)
    filtered = _lowpass(
        continuous,
        fps=fps,
        cutoff_hz=cutoff_hz,
        order=order,
    )
    return _normalize_wxyz(filtered, name="filtered quaternion series")


def _target_times(
    *,
    start_frame: int,
    end_frame_exclusive: int,
    source_fps: float,
    motion_time_scale: float = DEFAULT_MOTION_TIME_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    if start_frame < 0 or end_frame_exclusive <= start_frame + 1:
        raise ValueError("Source crop must contain at least two frames.")
    if not np.isfinite(motion_time_scale) or motion_time_scale <= 0.0:
        raise ValueError("motion_time_scale must be finite and positive.")
    source_times = np.arange(start_frame, end_frame_exclusive) / source_fps
    duration = source_times[-1] - source_times[0]
    count = int(np.floor(duration * TARGET_FPS * motion_time_scale + 1.0e-9)) + 1
    # Reference samples remain at TARGET_FPS.  Sampling source time more
    # densely stretches the motion without changing the runtime clock or
    # inventing interpolated poses after retargeting.
    target_times = source_times[0] + np.arange(count) / (TARGET_FPS * motion_time_scale)
    return source_times, target_times


def _source_frame_change_bound(
    *,
    source_fps: float,
    target_fps: float,
    motion_time_scale: float,
    target_frame_change_rad: float,
) -> float:
    """Convert an emitted-frame change limit to the source sample rate."""

    values = np.asarray(
        (source_fps, target_fps, motion_time_scale, target_frame_change_rad),
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Frame rates, time scale, and change limit must be positive.")
    return float(target_frame_change_rad * target_fps * motion_time_scale / source_fps)


def _resample_linear(
    values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    flat = data.reshape(len(data), -1)
    result = np.column_stack(
        [
            np.interp(target_times, source_times, flat[:, index])
            for index in range(flat.shape[1])
        ]
    )
    return result.reshape((len(target_times), *data.shape[1:]))


def _resample_wxyz(
    values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    data = _continuous_wxyz(values)
    flat = data.reshape(len(data), -1, 4)
    result = np.empty((len(target_times), flat.shape[1], 4), dtype=np.float64)
    for stream in range(flat.shape[1]):
        rotations = _rotation_from_wxyz(flat[:, stream])
        result[:, stream] = _rotation_to_wxyz(
            Slerp(source_times, rotations)(target_times)
        )
    return result.reshape((len(target_times), *data.shape[1:]))


def _resample_binary(
    values: np.ndarray,
    *,
    source_fps: float,
    target_times: np.ndarray,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    source_indices = np.rint(target_times * source_fps).astype(np.int64)
    source_indices = np.clip(source_indices, 0, data.shape[-1] - 1)
    return data[..., source_indices] >= 0.5


def _densify_scene_trajectory(
    qpos: np.ndarray,
    object_poses: np.ndarray,
    *,
    source_fps: float,
    audit_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Densify robot joints and rigid-object poses for inter-sample audit."""

    joints = _as_finite_array(qpos, name="dense audit qpos")
    poses = _as_finite_array(object_poses, name="dense audit object poses")
    if joints.ndim != 2 or len(joints) < 2:
        raise ValueError("Dense audit qpos must have shape [T, joints].")
    if poses.shape != (len(joints), 1, 7):
        raise ValueError("Dense audit object poses must have shape [T, 1, 7].")
    if (
        not np.isfinite((source_fps, audit_fps)).all()
        or source_fps <= 0.0
        or audit_fps <= 0.0
    ):
        raise ValueError("Dense audit rates must be finite and positive.")
    ratio = int(round(audit_fps / source_fps))
    if ratio < 1 or not math.isclose(
        audit_fps, ratio * source_fps, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("Dense audit FPS must be an integer source-FPS multiple.")
    source_times = np.arange(len(joints), dtype=np.float64) / source_fps
    dense_count = (len(joints) - 1) * ratio + 1
    dense_times = np.arange(dense_count, dtype=np.float64) / audit_fps
    dense_qpos = _resample_linear(joints, source_times, dense_times)
    dense_positions = _resample_linear(poses[..., :3], source_times, dense_times)
    dense_wxyz = _resample_wxyz(poses[..., 3:7], source_times, dense_times)
    return dense_qpos, np.concatenate((dense_positions, dense_wxyz), axis=-1)


def _model_joint_contract(
    model: mujoco.MjModel,
) -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    names: list[str] = []
    limits: dict[str, tuple[float, float]] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None or name in limits:
            raise ValueError("Vega-Wuji actuator joint names must be unique.")
        names.append(name)
        limits[name] = tuple(float(value) for value in model.jnt_range[joint_id])
    if len(names) != 59:
        raise ValueError(f"Expected 59 Vega-Wuji actuators, found {len(names)}.")
    return tuple(names), limits


def _finger_specs(
    side: str, limits: Mapping[str, tuple[float, float]]
) -> tuple[KeypointJointSpec, ...]:
    source_side = "Left" if side == "left" else "Right"
    target_side = "l" if side == "left" else "r"

    def spec(
        target_suffix: str, parent: str, joint: str, child: str
    ) -> KeypointJointSpec:
        target = f"{target_side}_{target_suffix}"
        lower, upper = limits[target]
        return KeypointJointSpec(
            target_name=target,
            parent=parent,
            joint=joint,
            child=child,
            lower=lower,
            upper=upper,
            scale=-1.0,
            offset=float(np.pi),
        )

    hand = f"{source_side}Hand"
    thumb = f"{hand}Thumb"
    result = [
        spec("thumb_cmc_flex", hand, f"{thumb}1", f"{thumb}2"),
        spec("thumb_mcp", f"{thumb}1", f"{thumb}2", f"{thumb}3"),
        spec("thumb_ip", f"{thumb}2", f"{thumb}3", f"{thumb}End"),
    ]
    for source_finger, target_finger in (
        ("Index", "index_finger"),
        ("Middle", "middle_finger"),
        ("Ring", "ring_finger"),
        ("Pinky", "pinky"),
    ):
        prefix = f"{hand}{source_finger}"
        result.extend(
            (
                spec(
                    f"{target_finger}_mcp_flex",
                    f"{prefix}1",
                    f"{prefix}2",
                    f"{prefix}3",
                ),
                spec(
                    f"{target_finger}_pip",
                    f"{prefix}2",
                    f"{prefix}3",
                    f"{prefix}4",
                ),
                spec(
                    f"{target_finger}_dip",
                    f"{prefix}3",
                    f"{prefix}4",
                    f"{prefix}End",
                ),
            )
        )
    return tuple(result)


def _retarget_fingers(
    soma_joints: np.ndarray,
    *,
    source_fps: float,
    target_joint_names: Sequence[str],
    limits: Mapping[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        trajectory = Trajectory(
            observations={"keypoints": soma_joints},
            infos={"coordinate_frame": "source_soma"},
            dt=1.0 / source_fps,
        )
        retargeted = KeypointRetargeter(
            SOMA_JOINT_NAMES, _finger_specs(side, limits)
        ).retarget(trajectory)
        mapped_names = tuple(
            retargeted.infos["retarget"]["target_joint_names"]  # type: ignore[index]
        )
        mapped = {
            name: retargeted.observations["qpos"][:, index]
            for index, name in enumerate(mapped_names)
        }
        prefix = "l_" if side == "left" else "r_"
        names = tuple(name for name in target_joint_names if name.startswith(prefix))
        fingers = np.zeros((len(soma_joints), len(names)), dtype=np.float64)
        for index, name in enumerate(names):
            if name in mapped:
                fingers[:, index] = mapped[name]
        result[side] = fingers
    return result


def _yaw_only_inverse(root_wxyz: np.ndarray) -> Rotation:
    root = _rotation_from_wxyz(np.asarray(root_wxyz)[None, :])
    yaw = float(root.as_euler("xyz")[0, 2])
    return Rotation.from_euler("z", -yaw)


def _neutral_wrist_poses(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result: dict[str, np.ndarray] = {}
    for side, site_name in (("left", "left_palm"), ("right", "right_palm")):
        site_id = model.site(site_name).id
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        result[side] = np.concatenate((data.site_xpos[site_id], quaternion))
    return result


def _bent_elbow_inactive_rest(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a collision-remote, bent-elbow arm seed and its palm poses."""

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for name, value in BENT_ELBOW_INACTIVE_REST_QPOS.items():
        joint = model.joint(name)
        data.qpos[joint.qposadr] = value
    mujoco.mj_forward(model, data)
    target_qpos = np.asarray(
        [
            data.qpos[int(model.jnt_qposadr[model.joint(name).id])]
            for name in joint_names
        ],
        dtype=np.float64,
    )
    poses: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        site_id = model.site(f"{side}_palm").id
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        poses[side] = np.concatenate((data.site_xpos[site_id], quaternion))
    return target_qpos, poses


def _neutral_fingertip_vectors(
    model: mujoco.MjModel,
) -> dict[str, np.ndarray]:
    """Return neutral fingertip vectors expressed in each palm-site frame."""

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        palm_id = model.site(f"{side}_palm").id
        palm_rotation = data.site_xmat[palm_id].reshape(3, 3)
        vectors_world = np.asarray(
            [
                data.site_xpos[model.site(site_name).id] - data.site_xpos[palm_id]
                for site_name in WUJI_FINGERTIP_SITE_NAMES[side]
            ],
            dtype=np.float64,
        )
        result[side] = vectors_world @ palm_rotation
    return result


def _calibrate_soma_wrist_orientations(
    source_wxyz: np.ndarray,
    *,
    alignment_rotation: Rotation,
    palm_frame_calibration: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Apply the geometry-only SOMA-to-Wuji palm-frame corrections.

    The correction follows the same composition as the successful MANO
    recipe: ``R_task = R_world_alignment @ R_source @ R_source_to_wuji``.
    Unlike the legacy anchor-frame calibration, it does not make one motion
    frame equal the robot neutral pose.
    """

    quaternions = _unit_wxyz(source_wxyz, name="SOMA wrist orientations")
    if quaternions.ndim != 3 or quaternions.shape[1:] != (2, 4):
        raise ValueError("SOMA wrist orientations must have shape [T, 2, 4].")
    output = np.empty_like(quaternions)
    offsets: dict[str, list[float]] = {}
    for side_index, side in enumerate(("left", "right")):
        correction_matrix = _as_finite_array(
            palm_frame_calibration[side]["source_to_robot_rotation"],
            name=f"{side} source-to-Wuji palm rotation",
            shape=(3, 3),
        )
        if not np.allclose(
            correction_matrix.T @ correction_matrix, np.eye(3), atol=1.0e-8
        ) or not np.isclose(np.linalg.det(correction_matrix), 1.0, atol=1.0e-8):
            raise ValueError(f"{side} source-to-Wuji palm rotation is not proper.")
        correction = Rotation.from_matrix(correction_matrix)
        source = _rotation_from_wxyz(quaternions[:, side_index])
        calibrated = alignment_rotation * source * correction
        output[:, side_index] = _rotation_to_wxyz(calibrated)
        offsets[side] = _rotation_to_wxyz(correction)[0].tolist()
    return output, offsets


def _contact_phase_weights(
    active: np.ndarray,
    *,
    dilation_frames: int,
    gaussian_sigma_frames: float,
) -> np.ndarray:
    """Return smooth per-hand activity weights without changing side order."""

    values = np.asarray(active, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("Contact activity must have shape [2, T].")
    if dilation_frames < 1 or dilation_frames % 2 != 1:
        raise ValueError("contact dilation must be a positive odd frame count.")
    if not np.isfinite(gaussian_sigma_frames) or gaussian_sigma_frames <= 0.0:
        raise ValueError("contact blend sigma must be finite and positive.")
    result = np.empty_like(values)
    structure = np.ones(dilation_frames, dtype=np.bool_)
    for side in range(2):
        dilated = binary_dilation(values[side] >= 0.5, structure=structure)
        result[side] = gaussian_filter1d(
            dilated.astype(np.float64),
            sigma=gaussian_sigma_frames,
            mode="nearest",
        )
    return np.clip(result, 0.0, 1.0)


def _project_palm_positions_nonpenetrating(
    active_positions: np.ndarray,
    object_positions: np.ndarray,
    neutral_positions: np.ndarray,
    blend_weights: np.ndarray,
    *,
    support_top_z: float,
    palm_object_standoff_m: float,
    palm_support_standoff_m: float,
    raise_rest_above_support: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project palm origins to conservative non-contact scene standoffs."""

    active = np.asarray(active_positions, dtype=np.float64).copy()
    objects = np.asarray(object_positions, dtype=np.float64)
    neutral = np.asarray(neutral_positions, dtype=np.float64)
    weights = np.asarray(blend_weights, dtype=np.float64)
    if active.ndim != 3 or active.shape[1:] != (2, 3):
        raise ValueError("Active palm positions must have shape [T, 2, 3].")
    if objects.shape != (len(active), 1, 3):
        raise ValueError("Object positions must have shape [T, 1, 3].")
    if neutral.shape != (2, 3) or weights.shape != (2, len(active)):
        raise ValueError("Neutral palms or blend weights have invalid shapes.")
    for name, value in (
        ("palm_object_standoff_m", palm_object_standoff_m),
        ("palm_support_standoff_m", palm_support_standoff_m),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    relative = active - objects
    distance = np.linalg.norm(relative, axis=-1, keepdims=True)
    if np.any(distance <= 1.0e-8):
        raise ValueError("A source palm coincides with the object origin.")
    scale = np.maximum(1.0, palm_object_standoff_m / distance)
    active = objects + relative * scale
    palm_floor_z = float(support_top_z + palm_support_standoff_m)
    active[..., 2] = np.maximum(active[..., 2], palm_floor_z)

    rest = neutral.copy()
    if raise_rest_above_support:
        rest[:, 2] = np.maximum(rest[:, 2], palm_floor_z)
    projected = rest[None, ...] + weights.T[..., None] * (active - rest[None, ...])
    return projected, active, rest


def _blend_wrist_orientations(
    calibrated_wxyz: np.ndarray,
    neutral_poses: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Blend neutral to calibrated orientations along the shortest rotation."""

    values = np.asarray(calibrated_wxyz, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 4):
        raise ValueError("Calibrated wrist orientations must have shape [T, 2, 4].")
    if np.asarray(weights).shape != (2, len(values)):
        raise ValueError("Wrist blend weights must have shape [2, T].")
    output = np.empty_like(values)
    for index, side in enumerate(("left", "right")):
        neutral = _rotation_from_wxyz(neutral_poses[side][None, 3:7])
        target = _rotation_from_wxyz(values[:, index])
        relative = neutral.inv() * target
        blended = neutral * Rotation.from_rotvec(
            relative.as_rotvec() * np.asarray(weights[index])[:, None]
        )
        output[:, index] = _rotation_to_wxyz(blended)
    return output


def _mesh_vertices(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            values = line.split()
            if len(values) != 4:
                raise ValueError(f"Malformed OBJ vertex in {path}.")
            vertices.append((float(values[1]), float(values[2]), float(values[3])))
    if not vertices:
        raise ValueError(f"Object OBJ has no vertices: {path}")
    result = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Object OBJ contains non-finite vertices: {path}")
    return result


def _anchor_vertical_clearance(
    *,
    object_mesh: Path,
    source_object_wxyz: np.ndarray,
    projection: ContactPhaseProjection,
) -> dict[str, float | list[float]]:
    """Measure can-bottom clearance above the uniformly scaled support top."""

    object_rotation = projection.rotation * _rotation_from_wxyz(
        np.asarray(source_object_wxyz, dtype=np.float64)[None, :]
    )
    rotated_vertices = object_rotation.apply(_mesh_vertices(object_mesh))
    can_bottom_z = float(
        projection.target_object_anchor[2]
        + projection.local_geometry_scale * np.min(rotated_vertices[:, 2])
    )
    support_center = projection.support_center_target()
    support_top_z = projection.support_top_z()
    return {
        "support_center_target_m": support_center.tolist(),
        "support_top_z_m": support_top_z,
        "can_bottom_z_m": can_bottom_z,
        "vertical_clearance_m": can_bottom_z - support_top_z,
    }


def _audit_can_support_vertical_clearance(
    object_poses: np.ndarray,
    *,
    object_mesh: Path,
    object_scale: float,
    support_top_z: float,
    # Every other penetration gate in this converter allows 1 mm (robot
    # self-collision, contact recovery, the declared
    # CollisionClearanceQualification). This audit previously allowed 1 um,
    # which is a numerical epsilon rather than a physical criterion: the
    # object mesh is reconstructed from video, then scaled and re-anchored, so
    # sub-millimetre rest-height error is inherent to the source and is not a
    # visible object sinking into the table.
    penetration_tolerance_m: float = 1.0e-3,
    support_center_xy: Sequence[float] | None = None,
    support_radius_m: float | None = None,
) -> dict[str, Any]:
    """Prove the object stays above the support at every sample.

    The support is a finite cylinder. Measuring every mesh vertex against an
    infinite plane therefore reports a penetration whenever an object wider
    than its support tilts, even though the low vertex is overhanging the
    table edge in free space. Supplying the support disc restricts the test to
    vertices actually over the support, which is the physical question. The
    plane-wide number is still reported so the stricter reading stays visible.
    """

    poses = _as_finite_array(object_poses, name="can-support audit object poses")
    if poses.ndim != 3 or poses.shape[1:] != (1, 7):
        raise ValueError("Can-support audit poses must have shape [T, 1, 7].")
    if not np.isfinite((object_scale, support_top_z, penetration_tolerance_m)).all():
        raise ValueError("Can-support audit parameters must be finite.")
    if object_scale <= 0.0 or penetration_tolerance_m < 0.0:
        raise ValueError("Can scale must be positive and tolerance non-negative.")
    use_disc = support_center_xy is not None and support_radius_m is not None
    if use_disc:
        center = _as_finite_array(
            np.asarray(support_center_xy, dtype=np.float64)[:2],
            name="support center",
            shape=(2,),
        )
        radius = float(support_radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Support radius must be finite and positive.")
    vertices = np.unique(_mesh_vertices(object_mesh), axis=0) * object_scale
    rotations = _rotation_from_wxyz(poses[:, 0, 3:7])
    clearances = np.empty(len(poses), dtype=np.float64)
    plane_clearances = np.empty(len(poses), dtype=np.float64)
    overhanging_frames: list[int] = []
    for frame in range(len(poses)):
        world = rotations[frame].apply(vertices) + poses[frame, 0, :3]
        plane_clearances[frame] = float(np.min(world[:, 2])) - support_top_z
        if not use_disc:
            clearances[frame] = plane_clearances[frame]
            continue
        over_support = np.linalg.norm(world[:, :2] - center, axis=-1) <= radius
        if not bool(over_support.any()):
            # Nothing is above the table at this sample, so the table cannot
            # be penetrated. Record it rather than inventing a clearance.
            overhanging_frames.append(frame)
            clearances[frame] = np.inf
            continue
        clearances[frame] = float(np.min(world[over_support, 2])) - support_top_z
    violation_frames = np.flatnonzero(clearances < -penetration_tolerance_m).tolist()
    finite = np.isfinite(clearances)
    minimum_frame = (
        int(np.argmin(np.where(finite, clearances, np.inf))) if finite.any() else 0
    )
    return {
        "qualified": not violation_frames,
        "frame_count": len(poses),
        "minimum_vertical_clearance_m": (
            float(clearances[minimum_frame]) if finite.any() else None
        ),
        "minimum_frame": minimum_frame,
        "penetration_tolerance_m": penetration_tolerance_m,
        "violation_frame_count": len(violation_frames),
        "violation_frames": violation_frames,
        "unique_mesh_vertex_count": len(vertices),
        "support_disc_restricted": bool(use_disc),
        "support_radius_m": float(radius) if use_disc else None,
        "frames_entirely_beyond_support_edge": overhanging_frames,
        "minimum_vertical_clearance_to_infinite_plane_m": float(plane_clearances.min()),
        "method": (
            "scaled object OBJ vertices above the support top; restricted to "
            "vertices inside the finite support disc when its geometry is "
            "supplied, because a vertex past the table edge overhangs free "
            "space. The infinite-plane minimum is reported alongside."
            if use_disc
            else "all scaled object OBJ vertices above the analytic support "
            "top plane; this is conservative for the finite cylinder"
        ),
    }


def _forward_kinematics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    addresses = np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in joint_names],
        dtype=np.int32,
    )
    wrists = {
        side: np.empty((len(qpos), 7), dtype=np.float64) for side in ("left", "right")
    }
    fingertips = {
        side: np.empty((len(qpos), len(FINGERTIP_NAMES[side]), 7), dtype=np.float64)
        for side in ("left", "right")
    }
    data = mujoco.MjData(model)
    for frame, values in enumerate(qpos):
        mujoco.mj_resetData(model, data)
        data.qpos[addresses] = values
        mujoco.mj_forward(model, data)
        for side, site_name in (("left", "left_palm"), ("right", "right_palm")):
            site_id = model.site(site_name).id
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
            wrists[side][frame] = np.concatenate((data.site_xpos[site_id], quaternion))
            for index, body_name in enumerate(FINGERTIP_NAMES[side]):
                body_id = model.body(body_name).id
                fingertips[side][frame, index] = np.concatenate(
                    (data.xpos[body_id], data.xquat[body_id])
                )
    return wrists, fingertips


def _fingertip_contact_geom_names(
    model: mujoco.MjModel,
) -> dict[str, tuple[str, ...]]:
    """Resolve one named collision geom for each Wuji fingertip body."""

    result: dict[str, tuple[str, ...]] = {}
    for side in ("left", "right"):
        names: list[str] = []
        for body_name in FINGERTIP_NAMES[side]:
            body_id = model.body(body_name).id
            candidates = [
                model.geom(geom_id).name
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
                and model.geom(geom_id).name
                and (
                    "tip_pad_proxy" in model.geom(geom_id).name
                    or int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                )
            ]
            if not candidates:
                raise ValueError(
                    f"Fingertip body {body_name!r} has no named collision geom."
                )
            candidates.sort(key=lambda name: ("tip_pad_proxy" not in name, name))
            names.append(str(candidates[0]))
        result[side] = tuple(names)
    return result


def _contact_geom_names_for_bodies(
    model: mujoco.MjModel,
    body_names: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str | int, ...]]:
    """Resolve one collidable geom per semantic hand-contact body."""

    result: dict[str, tuple[str | int, ...]] = {}
    for side, names in body_names.items():
        geoms: list[str | int] = []
        for body_name in names:
            body_id = int(model.body(str(body_name)).id)
            candidates = [
                model.geom(geom_id).name
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
                and model.geom(geom_id).name
                and (
                    int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                    or "tip_pad_proxy" in model.geom(geom_id).name
                )
            ]
            if candidates:
                candidates.sort(key=lambda name: ("tip_pad_proxy" not in name, name))
                geoms.append(str(candidates[0]))
                continue
            fallback_ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
                and (
                    int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                )
            ]
            if not fallback_ids:
                raise ValueError(f"Contact body {body_name!r} has no named geom.")
            geoms.append(int(fallback_ids[0]))
        result[str(side)] = tuple(geoms)
    return result


def _extract_soma_chord_source_contacts(
    *,
    arrays: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    crop: slice,
    object_mesh: Path,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct the source hand mesh and apply CHORD's exact 1 cm rule."""

    reconstruction = reconstruct_soma_motion(
        global_joint_positions=arrays["soma_joints"][crop],
        global_joint_wxyz=arrays["soma_joints_wxyz"][crop],
        identity_coeffs=np.asarray(payload["soma_identity_coeffs"]),
        scale_params=np.asarray(payload["soma_scale_params"]),
        joint_names=SOMA_JOINT_NAMES,
        device=device,
    )
    bridge = infer_soma_frame_bridge(
        soma_root_positions=arrays["soma_joints"][crop, 0],
        soma_root_wxyz=arrays["soma_joints_wxyz"][crop, 0],
        target_root_positions=np.asarray(
            payload["nvhuman_root_translation"], dtype=np.float64
        )[crop],
        target_root_wxyz=np.asarray(payload["nvhuman_root_wxyz"], dtype=np.float64)[
            crop
        ],
    )
    surface_points, surface_normals = sample_object_surface(
        object_mesh, count=4096, seed=0
    )
    source_object_poses = np.concatenate(
        (
            arrays["object_body_position"][crop, 0],
            arrays["object_body_wxyz"][crop, 0],
        ),
        axis=-1,
    )
    contacts = extract_chord_contact_targets(
        joint_names=SOMA_JOINT_NAMES,
        joints_world=bridge.apply(reconstruction.joints),
        vertices_world=bridge.apply(reconstruction.vertices),
        faces=reconstruction.faces,
        hand_vertex_indices=reconstruction.hand_vertex_indices,
        object_surface_points=surface_points,
        object_surface_normals=surface_normals,
        object_poses_wxyz=source_object_poses,
        source_active=arrays["hand_contact_active"][:, crop].T,
        threshold_m=0.01,
    )
    expected_names = tuple(
        tuple(f"{side}_{key}" for key in SOMA_CHORD_LINK_KEYS)
        for side in ("left", "right")
    )
    if contacts.link_names != expected_names:
        raise ValueError(
            "SOMA/CHORD link ordering changed; refusing to map contacts to Wuji."
        )
    report = {
        "method": "SOMA-X mesh reconstruction plus CHORD 4096-point 1cm rule",
        "reconstruction": reconstruction.report.as_dict(),
        "frame_bridge": {
            "rotation": bridge.rotation.tolist(),
            "rotation_relation": bridge.rotation_relation,
            "max_rotation_deviation_rad": bridge.max_rotation_deviation_rad,
            "translation_min_m": bridge.translations.min(axis=0).tolist(),
            "translation_max_m": bridge.translations.max(axis=0).tolist(),
        },
        "contacts": contacts.report.as_dict(),
        "surface_point_count": 4096,
        "surface_seed": 0,
        "threshold_m": 0.01,
    }
    return contacts, report


def _soma_hand_poses_in_target_world(
    *,
    arrays: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
    crop: slice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Recover the anatomical SOMA hand roots in the object world frame.

    CHORD retargets the MANO/SOMA wrist itself.  The Parquet ``ee_pose_w``
    belongs to the already-retargeted source robot and is offset from that
    anatomical wrist by 5--6 cm on this sequence.  Using it for a second
    embodiment silently breaks the hand/object transform that produced the
    measured mesh contacts.
    """

    bridge = infer_soma_frame_bridge(
        soma_root_positions=arrays["soma_joints"][crop, 0],
        soma_root_wxyz=arrays["soma_joints_wxyz"][crop, 0],
        target_root_positions=np.asarray(
            payload["nvhuman_root_translation"], dtype=np.float64
        )[crop],
        target_root_wxyz=np.asarray(payload["nvhuman_root_wxyz"], dtype=np.float64)[
            crop
        ],
    )
    hand_indices = np.asarray(
        [
            SOMA_JOINT_NAMES.index(names[0])
            for names in SOMA_HAND_KEYPOINT_NAMES.values()
        ],
        dtype=np.int32,
    )
    positions = bridge.apply(arrays["soma_joints"][crop][:, hand_indices])
    keypoint_indices = np.asarray(
        [
            SOMA_JOINT_NAMES.index(name)
            for names in SOMA_HAND_KEYPOINT_NAMES.values()
            for name in names
        ],
        dtype=np.int32,
    )
    keypoints = bridge.apply(arrays["soma_joints"][crop][:, keypoint_indices]).reshape(
        len(positions), 2, 6, 3
    )
    knuckle_indices = np.asarray(
        [
            SOMA_JOINT_NAMES.index(name)
            for names in SOMA_HAND_KNUCKLE_NAMES.values()
            for name in names
        ],
        dtype=np.int32,
    )
    knuckles = bridge.apply(arrays["soma_joints"][crop][:, knuckle_indices]).reshape(
        len(positions), 2, 4, 3
    )
    pip_indices = np.asarray(
        [
            SOMA_JOINT_NAMES.index(name)
            for names in SOMA_HAND_PIP_NAMES.values()
            for name in names
        ],
        dtype=np.int32,
    )
    pips = bridge.apply(arrays["soma_joints"][crop][:, pip_indices]).reshape(
        len(positions), 2, 4, 3
    )
    source = _rotation_from_wxyz(
        arrays["soma_joints_wxyz"][crop][:, hand_indices].reshape(-1, 4)
    )
    basis = Rotation.from_matrix(bridge.rotation)
    if bridge.rotation_relation == "left_multiply":
        rotations = basis * source
    elif bridge.rotation_relation == "basis_similarity":
        rotations = basis * source * basis.inv()
    else:
        raise ValueError(
            f"Unsupported SOMA bridge relation: {bridge.rotation_relation!r}."
        )
    return (
        positions,
        _rotation_to_wxyz(rotations).reshape(len(positions), 2, 4),
        keypoints,
        knuckles,
        pips,
        bridge,
    )


def _soma_identity_rest_knuckles_in_calibration_frames(
    *,
    payload: Mapping[str, Any],
    bridge: Any,
    device: str,
) -> np.ndarray:
    """Return identity-zero MCPs in the converter's bridged hand frames."""

    rest = reconstruct_soma_identity_rest_pose(
        identity_coeffs=np.asarray(payload["soma_identity_coeffs"]),
        scale_params=np.asarray(payload["soma_scale_params"]),
        joint_names=SOMA_JOINT_NAMES,
        # The full/default SOMA rig is required; low LOD is diagnostic only.
        device=device,
    )
    result = np.empty((2, 4, 3), dtype=np.float64)
    for side_index, side in enumerate(("left", "right")):
        source_prefix = "Left" if side == "left" else "Right"
        hand_index = rest.joint_names.index(f"{source_prefix}Hand")
        knuckle_indices = [
            rest.joint_names.index(name) for name in SOMA_HAND_KNUCKLE_NAMES[side]
        ]
        hand_rotation = rest.transforms[hand_index + 1, :3, :3]
        local = (rest.joints[knuckle_indices] - rest.joints[hand_index]) @ hand_rotation
        if bridge.rotation_relation == "basis_similarity":
            # The saved target-root quaternion is expressed by B R B^T while
            # points use B. Consequently the hand-local coordinates recovered
            # by that bridged quaternion are B times SOMA's native local frame.
            local = local @ np.asarray(bridge.rotation, dtype=np.float64).T
        elif bridge.rotation_relation != "left_multiply":
            raise ValueError(
                f"Unsupported SOMA bridge relation: {bridge.rotation_relation!r}."
            )
        result[side_index] = local
    return result


def _fit_soma_wuji_palm_frames(
    model: mujoco.MjModel,
    *,
    source_hand_positions: np.ndarray,
    source_hand_wxyz: np.ndarray,
    source_knuckles: np.ndarray,
    identity_rest_knuckles_local: np.ndarray,
    source_pips: np.ndarray | None = None,
    align_finger_directions: bool = False,
) -> dict[str, dict[str, Any]]:
    """Measure each SOMA wrist-to-Wuji palm correction from zero-pose MCPs.

    With ``align_finger_directions`` the knuckle-position fit (the notes'
    definition A) is composed with the palm yaw that centres the MCP abduction
    demand (definition B). Definition A fixes where the fingers start;
    definition B fixes which way they point, and only the second controls how
    much abduction the retarget must spend against the Wuji's +/-40 deg
    mechanical limit.
    """

    positions = _as_finite_array(source_hand_positions, name="SOMA hand positions")
    quaternions = _unit_wxyz(source_hand_wxyz, name="SOMA hand orientations")
    knuckles = _as_finite_array(source_knuckles, name="SOMA hand knuckles")
    if positions.ndim != 3 or positions.shape[1:] != (2, 3):
        raise ValueError("SOMA hand positions must have shape [T, 2, 3].")
    if quaternions.shape != (len(positions), 2, 4):
        raise ValueError("SOMA hand orientations must have shape [T, 2, 4].")
    if knuckles.shape != (len(positions), 2, 4, 3):
        raise ValueError("SOMA hand knuckles must have shape [T, 2, 4, 3].")
    rest_knuckles = _as_finite_array(
        identity_rest_knuckles_local,
        name="identity-rest SOMA hand knuckles",
        shape=(2, 4, 3),
    )
    pips: np.ndarray | None = None
    if align_finger_directions:
        if source_pips is None:
            raise ValueError("align_finger_directions requires the SOMA PIP keypoints.")
        pips = _as_finite_array(source_pips, name="SOMA hand PIPs")
        if pips.shape != knuckles.shape:
            raise ValueError("SOMA hand PIPs must have shape [T, 2, 4, 3].")

    result: dict[str, dict[str, Any]] = {}
    for side_index, side in enumerate(("left", "right")):
        source_rotation = _rotation_from_wxyz(quaternions[:, side_index]).as_matrix()
        local_knuckles = np.einsum(
            "tji,tkj->tki",
            source_rotation,
            knuckles[:, side_index] - positions[:, side_index, None],
        )
        result[side] = fit_source_palm_frame_from_knuckles(
            rest_knuckles[side_index, None],
            wuji_knuckle_positions_palm_frame(model, side),
        )
        result[side].update(
            {
                "calibration_source": (
                    "identity-specific full-resolution SOMA zero-pose MCP geometry"
                ),
                "posed_mcp_local_coordinate_std_mm": float(
                    local_knuckles.std(axis=0).max() * 1e3
                ),
                "posed_mcp_pairwise_distance_range_mm": float(
                    max(
                        np.ptp(
                            np.linalg.norm(
                                local_knuckles[:, first] - local_knuckles[:, second],
                                axis=-1,
                            )
                        )
                        for first in range(4)
                        for second in range(first + 1, 4)
                    )
                    * 1e3
                ),
            }
        )
        if pips is None:
            continue
        # Definition B: express the posed proximal phalanges in the palm frame
        # the knuckle fit just produced, then rotate that frame about its own
        # normal until the mapped fan sits on the robot's zero-pose fan.
        correction = np.asarray(
            result[side]["source_to_robot_rotation"], dtype=np.float64
        )
        segments = pips[:, side_index] - knuckles[:, side_index]
        local_segments = np.einsum("tji,tkj->tki", source_rotation, segments)
        lengths = np.linalg.norm(local_segments, axis=-1, keepdims=True)
        if float(lengths.min()) <= 1.0e-9:
            raise ValueError(f"{side} SOMA proximal phalanx has zero length.")
        directions = (local_segments / lengths) @ correction
        yaw_fit = fit_palm_yaw_from_finger_directions(
            directions,
            wuji_finger_directions_palm_frame(model, side),
            palm_normal_axis=int(result[side]["palm_normal_axis"]),
        )
        aligned = correction @ _axis_rotation(
            int(result[side]["palm_normal_axis"]), float(np.radians(yaw_fit["yaw_deg"]))
        )
        result[side]["source_to_robot_rotation"] = aligned.tolist()
        result[side]["knuckle_position_only_rotation"] = correction.tolist()
        result[side]["finger_direction_yaw"] = yaw_fit
        result[side]["calibration_source"] = (
            "identity-specific full-resolution SOMA zero-pose MCP geometry "
            "plus posed finger-direction palm yaw"
        )
    return result


def _map_soma_fingertips_to_wuji_targets(
    *,
    source_keypoints: np.ndarray,
    source_hand_wxyz: np.ndarray,
    wrist_positions: np.ndarray,
    wrist_wxyz: np.ndarray,
    palm_frame_calibration: Mapping[str, Mapping[str, Any]],
    neutral_fingertip_vectors: Mapping[str, np.ndarray],
    blend_weights: np.ndarray,
    hand_scale: float = DEFAULT_SOMA_TO_WUJI_HAND_SCALE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map SOMA wrist-to-tip vectors into calibrated Wuji palm frames.

    This mirrors CHORD's multi-keypoint hand fit: the anatomical wrist fixes
    the palm frame while five fingertip targets determine the articulated hand
    pose.  A single source-to-robot scale preserves the relative hand shape and
    motion-dependent shortening from finger flexion.  Outside contact phases,
    targets blend to the robot's neutral fingertip layout together with the
    palm target.
    """

    points = _as_finite_array(source_keypoints, name="SOMA hand keypoints")
    source_quaternions = _unit_wxyz(
        source_hand_wxyz, name="SOMA hand keypoint orientations"
    )
    palms = _as_finite_array(wrist_positions, name="Wuji palm target positions")
    palm_quaternions = _unit_wxyz(wrist_wxyz, name="Wuji palm target orientations")
    weights = _as_finite_array(blend_weights, name="hand keypoint blend weights")
    if points.ndim != 4 or points.shape[1:] != (2, 6, 3):
        raise ValueError("SOMA hand keypoints must have shape [T, 2, 6, 3].")
    if source_quaternions.shape != (len(points), 2, 4):
        raise ValueError("SOMA hand orientations must have shape [T, 2, 4].")
    if palms.shape != (len(points), 2, 3):
        raise ValueError("Wuji palm positions must have shape [T, 2, 3].")
    if palm_quaternions.shape != (len(points), 2, 4):
        raise ValueError("Wuji palm orientations must have shape [T, 2, 4].")
    if weights.shape != (2, len(points)) or np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError(
            "Hand keypoint blend weights must have shape [2, T] in [0, 1]."
        )

    output = np.empty((len(points), 2, 5, 3), dtype=np.float64)
    report: dict[str, Any] = {
        "method": (
            "bridged SOMA anatomical wrist plus five fingertips; "
            f"{hand_scale:g} source-to-robot scale; calibrated palm-frame blend"
        ),
        "hand_scale": float(hand_scale),
        "sides": {},
    }
    for side_index, side in enumerate(("left", "right")):
        vectors = points[:, side_index, 1:] - points[:, side_index, None, 0]
        source_hand = _rotation_from_wxyz(source_quaternions[:, side_index])
        source_local = np.einsum("tji,tkj->tki", source_hand.as_matrix(), vectors)
        calibration = palm_frame_calibration[side]
        correction = _as_finite_array(
            calibration["source_to_robot_rotation"],
            name=f"{side} source-to-Wuji palm rotation",
            shape=(3, 3),
        )
        if not np.allclose(
            correction.T @ correction, np.eye(3), atol=1.0e-8
        ) or not np.isclose(np.linalg.det(correction), 1.0, atol=1.0e-8):
            raise ValueError(f"{side} source-to-Wuji palm rotation is not proper.")
        robot_local = np.einsum("tki,ij->tkj", source_local, correction)
        normal_axis = int(calibration["palm_normal_axis"])
        normal_offset = float(calibration["palm_normal_offset_m"])
        neutral = _as_finite_array(
            neutral_fingertip_vectors[side],
            name=f"{side} neutral fingertip vectors",
            shape=(5, 3),
        )
        neutral_lengths = np.linalg.norm(neutral, axis=-1)
        morphology_scale = np.full(5, hand_scale)
        source_targets_local = robot_local * hand_scale
        # Relocate the SOMA and Wuji wrist origins after scaling the anatomical
        # wrist-to-tip vector. This order is exact for the adopted scale 1.0 and
        # remains algebraically correct for diagnostic non-unit scales.
        source_targets_local[..., normal_axis] -= normal_offset
        source_extended_lengths = np.percentile(
            np.linalg.norm(source_targets_local, axis=-1), 95.0, axis=0
        )
        if np.any(source_extended_lengths <= 1.0e-8) or np.any(
            neutral_lengths <= 1.0e-8
        ):
            raise ValueError(f"{side} hand has a zero wrist-to-tip length.")
        weight = weights[side_index, :, None, None]
        targets_local = neutral[None, :, :] + weight * (
            source_targets_local - neutral[None, :, :]
        )
        palm_rotation = _rotation_from_wxyz(palm_quaternions[:, side_index])
        targets_world = np.einsum(
            "tij,tkj->tki", palm_rotation.as_matrix(), targets_local
        )
        output[:, side_index] = palms[:, side_index, None, :] + targets_world
        report["sides"][side] = {
            "source_p95_wrist_to_tip_length_m": source_extended_lengths.tolist(),
            "target_neutral_wrist_to_tip_length_m": neutral_lengths.tolist(),
            "morphology_scale": morphology_scale.tolist(),
            "wrist_to_tip_excess_m": (
                neutral_lengths - source_extended_lengths
            ).tolist(),
            "palm_frame_calibration": dict(calibration),
        }
    return output, report


def _hand_morphology_standoff_m(
    target_report: Mapping[str, Any],
    *,
    override_m: float | None = None,
) -> dict[str, float]:
    """Distance each Wuji palm must retreat to reach the human's fingertips.

    The Wuji hand is longer from wrist to fingertip than a human hand. Pinning
    the robot palm at the human's anatomical wrist therefore asks longer robot
    fingers to reach nearer targets: the fingers cannot extend that far back,
    so the solver curls them and drives the hand through the object surface.

    Retreating the palm along its own approach axis by the measured wrist-to-
    tip excess keeps the fingertip targets exactly where the human's
    fingertips were, on the object.

    Measured negative on the snack-box clip (2026-08-24) and therefore off by
    default: moving the palm without moving its own targets only makes the
    targets unreachable (tip p95 20 -> 45 mm on its own; 18 -> 105 mm when
    stacked on a radial standoff). ``palm_object_standoff_m`` is the correct
    hand-size correction because it translates the palm and its fingertip
    targets together, radially away from the object surface. Kept as a
    documented diagnostic so the comparison can be reproduced.
    """

    result: dict[str, float] = {}
    for side, record in target_report["sides"].items():
        if override_m is not None:
            result[side] = float(override_m)
            continue
        source = np.asarray(
            record["source_p95_wrist_to_tip_length_m"], dtype=np.float64
        )
        neutral = np.asarray(
            record["target_neutral_wrist_to_tip_length_m"], dtype=np.float64
        )
        # Non-thumb fingers define the grasp approach; the thumb opposes it.
        result[side] = float(max(0.0, float(np.mean(neutral[1:] - source[1:]))))
    return result


def _retreat_wrist_targets_for_hand_morphology(
    wrist_positions: np.ndarray,
    wrist_wxyz: np.ndarray,
    *,
    neutral_fingertip_vectors: Mapping[str, np.ndarray],
    standoff_m: Mapping[str, float],
    blend_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Move palm targets back along their own approach axis; tips stay put."""

    palms = _as_finite_array(wrist_positions, name="Wuji palm target positions")
    quaternions = _unit_wxyz(wrist_wxyz, name="Wuji palm target orientations")
    weights = _as_finite_array(blend_weights, name="hand keypoint blend weights")
    if palms.ndim != 3 or palms.shape[1:] != (2, 3):
        raise ValueError("Wrist positions must have shape [T, 2, 3].")
    if quaternions.shape != (len(palms), 2, 4):
        raise ValueError("Wrist orientations must have shape [T, 2, 4].")
    if weights.shape != (2, len(palms)):
        raise ValueError("Blend weights must have shape [2, T].")

    output = palms.copy()
    report: dict[str, Any] = {
        "method": (
            "retreat the palm target along its neutral finger-approach axis by "
            "the measured wrist-to-tip excess; fingertip targets unchanged"
        ),
        "sides": {},
    }
    for side_index, side in enumerate(("left", "right")):
        neutral = _as_finite_array(
            neutral_fingertip_vectors[side],
            name=f"{side} neutral fingertip vectors",
            shape=(5, 3),
        )
        axis = np.mean(neutral[1:], axis=0)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-8:
            raise ValueError(f"{side} neutral finger axis is degenerate.")
        axis = axis / norm
        distance = float(standoff_m[side])
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError(f"{side} morphology standoff must be non-negative.")
        rotation = _rotation_from_wxyz(quaternions[:, side_index]).as_matrix()
        # Blend with contact activity so an inactive hand keeps its authored
        # rest pose exactly.
        shift = (
            weights[side_index, :, None]
            * distance
            * np.einsum("tij,j->ti", rotation, axis)
        )
        output[:, side_index] -= shift
        report["sides"][side] = {
            "standoff_m": distance,
            "palm_frame_approach_axis": axis.tolist(),
            "max_applied_shift_m": float(np.linalg.norm(shift, axis=-1).max()),
        }
    return output, report


#: CHORD link indices of the distal phalanges, in fingertip site order
#: (thumb, index, middle, ring, pinky) within ``SOMA_CHORD_LINK_KEYS``.
CHORD_DISTAL_LINK_INDICES = tuple(
    SOMA_CHORD_LINK_KEYS.index(key)
    for key in ("thumb3", "index3", "middle3", "ring3", "pinky3")
)


def _wuji_fingertip_collision_radius(model: mujoco.MjModel, side: str) -> float:
    """Radius of the sphere that actually touches the object at a fingertip.

    The IK aims a fingertip *site*, which is a point, while contact happens on
    the surrounding collision sphere. Aiming the point at a surface contact
    location therefore buries the sphere by one radius.
    """

    prefix = "l" if side == "left" else "r"
    bodies = (
        f"{prefix}_thumb_distal",
        f"{prefix}_index_finger_distal",
        f"{prefix}_middle_finger_distal",
        f"{prefix}_ring_finger_distal",
        f"{prefix}_pinky_distal",
    )
    radii: list[float] = []
    for name in bodies:
        body_id = int(model.body(name).id)
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) != body_id:
                continue
            if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
                continue
            radii.append(float(model.geom_size[geom, 0]))
            break
    if len(radii) != len(bodies):
        raise ValueError(f"{side} hand has no fingertip collision sphere per finger.")
    return float(np.mean(radii))


def _object_anchored_fingertip_targets(
    wrist_anchored_targets: np.ndarray,
    *,
    chord_targets: np.ndarray,
    chord_active: np.ndarray,
    blend_frames: int,
    chord_normals: np.ndarray | None = None,
    tip_radius_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retarget fingertips to where the human touched the object.

    The Wuji hand is larger than the source hand, so it cannot put its
    fingertips where the human's were *and* grip a true-size object: those are
    different points. Only one of them is a physical requirement. The contact
    location on the object is what the manipulation actually depends on, so
    during contact the fingertip target becomes the measured CHORD contact
    point carried on the object, and the robot is free to reach it with
    whatever hand pose its own proportions require.

    Away from contact there is no object-anchored point to aim at, so the
    target stays the wrist-anchored one and the two are cross-faded over
    ``blend_frames`` so a finger never jumps when contact starts or stops.
    """

    wrist_anchored = _as_finite_array(
        wrist_anchored_targets, name="wrist-anchored fingertip targets"
    )
    targets = _as_finite_array(chord_targets, name="CHORD targets")
    active = np.asarray(chord_active, dtype=bool)
    if wrist_anchored.ndim != 4 or wrist_anchored.shape[1:] != (2, 5, 3):
        raise ValueError("Fingertip targets must have shape [T, 2, 5, 3].")
    if targets.shape[:2] != wrist_anchored.shape[:2] or targets.shape[-1] != 3:
        raise ValueError("CHORD targets must have shape [T, 2, links, 3].")
    if active.shape != targets.shape[:-1]:
        raise ValueError("CHORD activity does not align with its targets.")
    if blend_frames < 1 or blend_frames % 2 != 1:
        raise ValueError("blend_frames must be a positive odd frame count.")

    distal = np.asarray(CHORD_DISTAL_LINK_INDICES, dtype=np.int32)
    object_anchored = targets[:, :, distal]
    contact = active[:, :, distal]
    if tip_radius_m > 0.0:
        if chord_normals is None:
            raise ValueError("A fingertip radius offset needs the CHORD normals.")
        normals = _as_finite_array(chord_normals, name="CHORD normals")
        if normals.shape != targets.shape:
            raise ValueError("CHORD normals must match the CHORD targets.")
        outward = normals[:, :, distal]
        lengths = np.linalg.norm(outward, axis=-1, keepdims=True)
        # Inactive slots carry a zero normal; leave those untouched, they are
        # masked out below anyway.
        unit = np.divide(
            outward, lengths, out=np.zeros_like(outward), where=lengths > 1e-9
        )
        # Stand the fingertip off the surface by its own collision radius so
        # the sphere touches the object instead of sinking one radius in.
        object_anchored = object_anchored + tip_radius_m * unit
    # CHORD leaves an inactive slot at the world origin rather than at a
    # meaningful point, so the raw target array must never be averaged over
    # time. Smooth the DISPLACEMENT from the wrist-anchored target instead: it
    # is exactly zero wherever there is no contact, so an inactive slot
    # contributes nothing and cannot drag a fingertip toward the origin.
    offsets = np.where(contact[..., None], object_anchored - wrist_anchored, 0.0)
    window = np.ones(blend_frames, dtype=np.float64) / float(blend_frames)
    smoothed = np.empty_like(offsets)
    for side in range(offsets.shape[1]):
        for finger in range(offsets.shape[2]):
            for axis in range(3):
                padded = np.pad(
                    offsets[:, side, finger, axis], blend_frames, mode="edge"
                )
                smoothed[:, side, finger, axis] = np.convolve(
                    padded, window, mode="same"
                )[blend_frames:-blend_frames]
    blended = wrist_anchored + smoothed
    displacement = np.linalg.norm(smoothed, axis=-1)
    contact = contact.astype(np.float64)
    return blended, {
        "method": (
            "fingertip targets follow the measured CHORD contact point on the "
            "object during contact and the wrist-anchored human fingertip "
            "elsewhere, cross-faded so the path stays continuous"
        ),
        "blend_frames": int(blend_frames),
        "fingertip_collision_radius_offset_m": float(tip_radius_m),
        "chord_distal_link_indices": [int(index) for index in distal],
        "contact_frame_fraction_per_side": [
            float(contact[:, side].any(axis=-1).mean())
            for side in range(contact.shape[1])
        ],
        "max_target_displacement_m": float(displacement.max()),
        "mean_target_displacement_m": float(displacement.mean()),
    }


def _map_chord_targets_to_object_trajectory(
    contacts: Any,
    *,
    source_object_poses: np.ndarray,
    target_object_poses: np.ndarray,
    geometry_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve source contact coordinates in the retargeted object frame."""

    source = _as_finite_array(source_object_poses, name="source CHORD object poses")
    target = _as_finite_array(target_object_poses, name="target CHORD object poses")
    if source.shape != target.shape or source.ndim != 2 or source.shape[1:] != (7,):
        raise ValueError("Source/target CHORD object poses must share shape [T, 7].")
    if not np.isfinite(geometry_scale) or geometry_scale <= 0.0:
        raise ValueError("CHORD geometry_scale must be finite and positive.")
    active = np.asarray(contacts.active, dtype=bool)
    source_points = np.asarray(contacts.object_positions, dtype=np.float64)
    source_normals = np.asarray(contacts.object_normals, dtype=np.float64)
    if (
        source_points.shape != (*active.shape, 3)
        or source_normals.shape != source_points.shape
    ):
        raise ValueError("CHORD geometry does not align with its active mask.")
    source_rotation = _rotation_from_wxyz(source[:, 3:7]).as_matrix()
    target_rotation = _rotation_from_wxyz(target[:, 3:7]).as_matrix()
    source_delta = source_points - source[:, None, None, :3]
    local_points = np.einsum("tskj,tji->tski", source_delta, source_rotation)
    local_normals = np.einsum("tskj,tji->tski", source_normals, source_rotation)
    mapped_points = target[:, None, None, :3] + geometry_scale * np.einsum(
        "tij,tskj->tski", target_rotation, local_points
    )
    mapped_normals = np.einsum("tij,tskj->tski", target_rotation, local_normals)
    mapped_points[~active] = 0.0
    mapped_normals[~active] = 0.0
    return active, mapped_points, mapped_normals


def _resample_chord_targets_nearest(
    active: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    *,
    source_fps: float,
    target_times: np.ndarray,
    source_start_time: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.clip(
        np.rint(
            (np.asarray(target_times) - float(source_start_time)) * float(source_fps)
        ).astype(np.int64),
        0,
        len(active) - 1,
    )
    return active[indices], positions[indices], normals[indices]


def _audit_robot_self_penetration(
    auditor: "_SceneCollisionAuditor",
    qpos: np.ndarray,
    *,
    penetration_tolerance_m: float = 0.001,
) -> dict[str, Any]:
    """Audit authored robot self-contact pairs along the emitted trajectory."""

    model = auditor.model
    data = auditor.data
    robot_geoms = frozenset(auditor._robot_geom_ids)
    violations: list[dict[str, Any]] = []
    minimum = 0.0
    for frame, values in enumerate(np.asarray(qpos, dtype=np.float64)):
        mujoco.mj_resetData(model, data)
        data.qpos[auditor._qpos_addresses] = values
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in robot_geoms or geom2 not in robot_geoms:
                continue
            distance = float(contact.dist)
            minimum = min(minimum, distance)
            if distance >= -penetration_tolerance_m - 1.0e-9:
                continue
            violations.append(
                {
                    "frame": frame,
                    "geom1": auditor._geom_label(geom1),
                    "geom2": auditor._geom_label(geom2),
                    "signed_distance_m": distance,
                }
            )
    worst = min(violations, key=lambda item: item["signed_distance_m"], default=None)
    return {
        "qualified": not violations,
        "frame_count": len(qpos),
        "penetration_tolerance_m": penetration_tolerance_m,
        "minimum_signed_distance_m": minimum,
        "violation_count": len(violations),
        "violation_frame_count": len({item["frame"] for item in violations}),
        "worst_violation": worst,
        "method": "MuJoCo authored robot-robot contact replay",
    }


def _audit_semantic_robot_can_clearance(
    auditor: "_SceneCollisionAuditor",
    qpos: np.ndarray,
    object_poses: np.ndarray,
    *,
    maximum_intended_penetration_m: float = 0.001,
    forbidden_clearance_m: float = 0.0,
) -> dict[str, Any]:
    """Gate intended hand contact separately from forbidden can collision.

    Finger and palm bodies may contact the manipulated object, but no authored
    robot geometry may penetrate it by more than the contact-recovery limit.
    Arm, wrist, and mount-adapter geometry are forbidden from penetrating at
    all.  This typed audit replaces the legacy all-robot 5 mm can clearance,
    which is incompatible with manipulation by construction.
    """

    values = _as_finite_array(qpos, name="semantic can audit qpos")
    poses = _as_finite_array(object_poses, name="semantic can audit object poses")
    if values.ndim != 2 or values.shape[1] != len(auditor._joint_names):
        raise ValueError("Semantic can audit qpos does not align with joint names.")
    if poses.shape != (len(values), 1, 7):
        raise ValueError("Semantic can audit object poses must have shape [T, 1, 7].")
    _unit_wxyz(poses[..., 3:7], name="semantic can audit object poses")
    for name, value in (
        ("maximum_intended_penetration_m", maximum_intended_penetration_m),
        ("forbidden_clearance_m", forbidden_clearance_m),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    can_geom_id = auditor._scene_geom_ids["can"]
    records = {
        kind: {
            "minimum_signed_distance_m": float("inf"),
            "violation_frames": set(),
            "violating_robot_geometries": set(),
            "violating_pair_count": 0,
            "worst": None,
        }
        for kind in ("intended_hand", "forbidden_robot")
    }
    segment = np.zeros(6, dtype=np.float64)
    query_cutoff = max(0.005, float(forbidden_clearance_m) + 0.001)
    for frame, frame_qpos in enumerate(values):
        mujoco.mj_resetData(auditor.model, auditor.data)
        auditor.data.qpos[auditor._qpos_addresses] = frame_qpos
        auditor.data.mocap_pos[auditor._can_mocap_id] = poses[frame, 0, :3]
        auditor.data.mocap_quat[auditor._can_mocap_id] = poses[frame, 0, 3:7]
        mujoco.mj_forward(auditor.model, auditor.data)
        for robot_geom_id in auditor._robot_geom_ids:
            body_name = auditor.model.body(
                int(auditor.model.geom_bodyid[robot_geom_id])
            ).name
            intended = body_name.startswith(("l_", "r_")) and body_name not in (
                "l_wrist",
                "r_wrist",
            )
            kind = "intended_hand" if intended else "forbidden_robot"
            threshold = (
                -float(maximum_intended_penetration_m)
                if intended
                else float(forbidden_clearance_m)
            )
            distance = float(
                mujoco.mj_geomDistance(
                    auditor.model,
                    auditor.data,
                    int(robot_geom_id),
                    int(can_geom_id),
                    query_cutoff,
                    segment,
                )
            )
            record = records[kind]
            record["minimum_signed_distance_m"] = min(
                record["minimum_signed_distance_m"], distance
            )
            if distance + 1.0e-9 >= threshold:
                continue
            label = auditor._geom_label(int(robot_geom_id))
            record["violation_frames"].add(frame)
            record["violating_robot_geometries"].add(label)
            record["violating_pair_count"] += 1
            worst = record["worst"]
            if worst is None or distance < worst["signed_distance_m"]:
                record["worst"] = {
                    "frame": frame,
                    "robot_geometry": label,
                    "signed_distance_m": distance,
                }

    groups: dict[str, Any] = {}
    qualified = True
    for kind, record in records.items():
        frames = sorted(record["violation_frames"])
        qualified = qualified and not frames
        groups[kind] = {
            "qualified": not frames,
            "minimum_signed_distance_m": (
                None
                if not np.isfinite(record["minimum_signed_distance_m"])
                else float(record["minimum_signed_distance_m"])
            ),
            "violation_frame_count": len(frames),
            "violation_frames": frames,
            "violating_robot_geometries": sorted(record["violating_robot_geometries"]),
            "violating_pair_count": int(record["violating_pair_count"]),
            "worst_violation": record["worst"],
        }
    return {
        "qualified": bool(qualified),
        "frame_count": len(values),
        "maximum_intended_penetration_m": float(maximum_intended_penetration_m),
        "forbidden_clearance_m": float(forbidden_clearance_m),
        "groups": groups,
        "method": (
            "typed MuJoCo signed-distance replay: palm/finger contact is intended; "
            "arm, wrist, and mount-adapter penetration is forbidden"
        ),
    }


def _robot_self_violation_pairs_by_frame(
    auditor: _SceneCollisionAuditor,
    qpos: np.ndarray,
    *,
    penetration_tolerance_m: float,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Collect exact authored self-contact pairs that violate a tolerance."""

    model = auditor.model
    data = auditor.data
    robot_geoms = frozenset(auditor._robot_geom_ids)
    output: list[tuple[tuple[int, int], ...]] = []
    for values in np.asarray(qpos, dtype=np.float64):
        pairs: set[tuple[int, int]] = set()
        mujoco.mj_resetData(model, data)
        data.qpos[auditor._qpos_addresses] = values
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 in robot_geoms
                and geom2 in robot_geoms
                and float(contact.dist) < -penetration_tolerance_m - 1.0e-9
            ):
                pairs.add(tuple(sorted((geom1, geom2))))
        output.append(tuple(sorted(pairs)))
    return tuple(output)


def _contact_guided_finger_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    *,
    joint_names: Sequence[str],
    object_poses: np.ndarray,
    source_active: np.ndarray,
    max_target_distance_m: float = 0.30,
    iterations: int = 30,
    damping: float = 0.02,
    max_step: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Close active fingertips toward measured object witness points.

    This is a geometry-seeking stage, not a contact certificate. It uses the
    current fingertip/object witness segment as a target and solves only the
    corresponding finger joints. The later settle and recovery stages remain
    authoritative: a target that cannot be reached stays uncontacted.
    """

    values = _as_finite_array(qpos, name="contact-guided finger qpos")
    poses = _as_finite_array(object_poses, name="contact-guided object poses")
    active = np.asarray(source_active, dtype=bool)
    if values.ndim != 2 or poses.shape != (len(values), 1, 7):
        raise ValueError("Finger IK qpos or object poses have an invalid shape.")
    if active.shape != (len(values), 2):
        raise ValueError("Finger IK activity must have shape [T, 2].")
    if (
        iterations < 1
        or not np.isfinite((max_target_distance_m, damping, max_step)).all()
    ):
        raise ValueError("Finger IK settings must be finite and positive.")
    if max_target_distance_m <= 0.0 or damping <= 0.0 or max_step <= 0.0:
        raise ValueError("Finger IK settings must be positive.")

    tip_geom_names = _fingertip_contact_geom_names(model)
    object_geom_id = int(model.geom("audit_can_geom").id)
    object_body_id = int(model.body("audit_can_body").id)
    object_mocap_id = int(model.body_mocapid[object_body_id])
    finger_joint_indices = {
        "left": np.asarray(
            [i for i, name in enumerate(joint_names) if name.startswith("l_")],
            dtype=np.int32,
        ),
        "right": np.asarray(
            [i for i, name in enumerate(joint_names) if name.startswith("r_")],
            dtype=np.int32,
        ),
    }
    qpos_addresses = np.asarray(
        [int(model.jnt_qposadr[model.joint(name).id]) for name in joint_names],
        dtype=np.int32,
    )
    dof_addresses = np.asarray(
        [int(model.jnt_dofadr[model.joint(name).id]) for name in joint_names],
        dtype=np.int32,
    )
    lower = np.asarray(
        [float(model.jnt_range[model.joint(name).id, 0]) for name in joint_names],
        dtype=np.float64,
    )
    upper = np.asarray(
        [float(model.jnt_range[model.joint(name).id, 1]) for name in joint_names],
        dtype=np.float64,
    )
    output = values.copy()
    report: dict[str, Any] = {
        "frames": len(values),
        "active_frames": int(np.count_nonzero(active)),
        "target_fingertips": 0,
        "solved_fingertips": 0,
        "skipped_fingertips": 0,
        "mean_joint_shift_rad": 0.0,
        "max_joint_shift_rad": 0.0,
    }
    fromto = np.zeros(6, dtype=np.float64)
    shifts: list[float] = []
    for frame in range(len(output)):
        data.qpos[qpos_addresses] = output[frame]
        data.mocap_pos[object_mocap_id] = poses[frame, 0, :3]
        data.mocap_quat[object_mocap_id] = poses[frame, 0, 3:7]
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(("left", "right")):
            if not active[frame, side_index]:
                continue
            for geom_name in tip_geom_names[side]:
                report["target_fingertips"] += 1
                geom_id = int(model.geom(geom_name).id)
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        geom_id,
                        object_geom_id,
                        max_target_distance_m,
                        fromto,
                    )
                )
                if distance >= max_target_distance_m - 1.0e-9:
                    report["skipped_fingertips"] += 1
                    continue
                body_id = int(model.geom_bodyid[geom_id])
                body_position = data.xpos[body_id].copy()
                target_position = body_position + fromto[3:] - fromto[:3]
                finger_indices = finger_joint_indices[side]
                finger_dofs = dof_addresses[finger_indices]
                before = output[frame, finger_indices].copy()
                for _ in range(int(iterations)):
                    mujoco.mj_forward(model, data)
                    error = target_position - data.xpos[body_id]
                    if float(np.linalg.norm(error)) <= 1.0e-4:
                        break
                    jacobian = np.zeros((3, model.nv), dtype=np.float64)
                    rotation_jacobian = np.zeros((3, model.nv), dtype=np.float64)
                    mujoco.mj_jacBody(model, data, jacobian, rotation_jacobian, body_id)
                    jacobian = jacobian[:, finger_dofs]
                    normal = jacobian.T @ jacobian
                    normal += damping**2 * np.eye(len(finger_dofs))
                    delta = np.linalg.solve(normal, jacobian.T @ error)
                    norm = float(np.linalg.norm(delta))
                    if norm > max_step:
                        delta *= max_step / norm
                    output[frame, finger_indices] = np.clip(
                        output[frame, finger_indices] + delta,
                        lower[finger_indices],
                        upper[finger_indices],
                    )
                    data.qpos[qpos_addresses] = output[frame]
                shift = np.abs(output[frame, finger_indices] - before)
                shifts.append(float(np.max(shift)))
                report["solved_fingertips"] += 1
    if shifts:
        report["mean_joint_shift_rad"] = float(np.mean(shifts))
        report["max_joint_shift_rad"] = float(np.max(shifts))
    return output, report


def _orientation_error(
    target_wxyz: np.ndarray, achieved_wxyz: np.ndarray
) -> np.ndarray:
    target = _rotation_from_wxyz(target_wxyz)
    achieved = _rotation_from_wxyz(achieved_wxyz)
    return (target * achieved.inv()).magnitude()


def _resolve_relative_asset(path: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        result = raw.resolve()
        if not result.is_file():
            raise FileNotFoundError(f"Source asset is missing: {result}")
        return result
    for parent in path.parents:
        candidate = (parent / raw).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve source asset {value!r} from {path}.")


def _usd_safe_identifier(value: str, *, fallback: str = "object") -> str:
    """Return a stable Python/USD identifier from an upstream display name."""

    result = re.sub(r"[^A-Za-z0-9_]", "_", str(value))
    result = re.sub(r"_+", "_", result).strip("_")
    if not result:
        result = fallback
    if not re.match(r"[A-Za-z_]", result[0]):
        result = f"_{result}"
    return result


def _asset_dependency_hashes(
    object_urdf: Path,
    object_collision_mesh: Path,
    *,
    expected_sha256: Mapping[str, str] = PINNED_OBJECT_ASSET_SHA256,
) -> dict[str, str]:
    # Only the wrapper and collision geometry affect physics. Material and
    # texture files are intentionally outside the training asset contract.
    dependencies = (object_urdf, object_collision_mesh)
    for path in dependencies:
        if not path.is_file():
            raise FileNotFoundError(f"Object asset dependency is missing: {path}")
    result = {str(path.resolve()): _sha256_file(path) for path in dependencies}
    for path in dependencies:
        expected = expected_sha256.get(path.name)
        actual = result[str(path.resolve())]
        if expected is None or actual != expected:
            raise ValueError(
                f"Pinned object asset SHA-256 mismatch for {path}: "
                f"expected {expected}, got {actual}."
            )
    return result


class _SceneCollisionAuditor:
    """Measure robot clearance to the exact target can and support geometry."""

    def __init__(
        self,
        *,
        robot_model_path: Path,
        object_mesh_path: Path,
        object_scale: float,
        support_center: np.ndarray,
        support_radius: float,
        support_height: float,
        joint_names: Sequence[str],
        required_clearance_m: float,
        include_visual_geometry: bool = False,
        gate_scenes: Sequence[str] | None = None,
    ) -> None:
        if not np.isfinite(required_clearance_m) or required_clearance_m <= 0.0:
            raise ValueError("Geometry clearance must be finite and positive.")
        for name, value in (
            ("object_scale", object_scale),
            ("support_radius", support_radius),
            ("support_height", support_height),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        center = _as_finite_array(support_center, name="support_center", shape=(3,))
        scale_text = " ".join([f"{object_scale:.17g}"] * 3)
        center_text = " ".join(f"{value:.17g}" for value in center)
        xml = (
            '<mujoco model="vega_wuji_scene_clearance_audit">'
            f"<include file={quoteattr(str(robot_model_path))}/>"
            "<asset>"
            f'<mesh name="audit_can_mesh" file={quoteattr(str(object_mesh_path))} '
            f'scale="{scale_text}"/>'
            "</asset>"
            "<worldbody>"
            '<body name="audit_can_body" mocap="true">'
            '<geom name="audit_can_geom" type="mesh" mesh="audit_can_mesh" '
            'contype="1" conaffinity="1"/>'
            "</body>"
            '<geom name="audit_support_geom" type="cylinder" '
            f'size="{support_radius:.17g} {0.5 * support_height:.17g}" '
            f'pos="{center_text}" contype="1" conaffinity="1"/>'
            "</worldbody>"
            "</mujoco>"
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.required_clearance_m = float(required_clearance_m)
        # Query only the required threshold.  MuJoCo 3.8 can return a false
        # zero for separated mesh pairs when asked to resolve a much larger
        # distance.  A threshold-capped query is fail-closed: safe pairs
        # return the cutoff and anything smaller is a violation.
        self._distance_limit_m = self.required_clearance_m
        self._joint_names = tuple(joint_names)
        self._qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in self._joint_names
            ],
            dtype=np.int32,
        )
        can_body_id = self.model.body("audit_can_body").id
        self._can_mocap_id = int(self.model.body_mocapid[can_body_id])
        self._scene_geom_ids = {
            "can": self.model.geom("audit_can_geom").id,
            "support": self.model.geom("audit_support_geom").id,
        }
        # Which scenes make the audit fail. Contact-seeking retargeting keeps
        # the support fail-closed while letting the hand approach the can,
        # because a clearance gate on the can forbids the very thing the task
        # needs. Every scene is still measured and reported either way.
        if gate_scenes is None:
            self._gate_scenes = tuple(self._scene_geom_ids)
        else:
            unknown = set(gate_scenes) - set(self._scene_geom_ids)
            if unknown:
                raise ValueError(f"Unknown audit gate scenes: {sorted(unknown)}.")
            self._gate_scenes = tuple(gate_scenes)
        scene_ids = frozenset(self._scene_geom_ids.values())
        self._include_visual_geometry = bool(include_visual_geometry)
        self._robot_geom_ids = tuple(
            geom_id
            for geom_id in range(self.model.ngeom)
            if geom_id not in scene_ids
            and (
                self._include_visual_geometry
                or int(self.model.geom_contype[geom_id]) != 0
                or int(self.model.geom_conaffinity[geom_id]) != 0
                or int(self.model.geom_group[geom_id]) == 4
            )
        )
        if not self._robot_geom_ids:
            raise ValueError("Vega-Wuji audit model has no collision geometry.")

    def _geom_label(self, geom_id: int) -> str:
        geom_name = self.model.geom(geom_id).name
        body_id = int(self.model.geom_bodyid[geom_id])
        body_name = self.model.body(body_id).name or f"body_{body_id}"
        return geom_name or f"{body_name}:geom_{geom_id}"

    def audit(self, qpos: np.ndarray, object_poses: np.ndarray) -> dict[str, Any]:
        values = _as_finite_array(qpos, name="collision audit qpos")
        poses = _as_finite_array(object_poses, name="collision audit object poses")
        if values.ndim != 2 or values.shape[1] != len(self._joint_names):
            raise ValueError("Collision audit qpos does not align with joint names.")
        if poses.shape != (len(values), 1, 7):
            raise ValueError("Collision audit object poses must have shape [T, 1, 7].")
        _unit_wxyz(poses[..., 3:7], name="collision audit object poses")

        records: dict[str, dict[str, Any]] = {
            scene: {
                "minimum_threshold_query_return_m": self._distance_limit_m,
                "violation_frames": set(),
                "violating_robot_geometries": set(),
                "violating_pair_count": 0,
                "worst": None,
            }
            for scene in self._scene_geom_ids
        }
        closest_points = np.empty(6, dtype=np.float64)
        for frame, frame_qpos in enumerate(values):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self._qpos_addresses] = frame_qpos
            self.data.mocap_pos[self._can_mocap_id] = poses[frame, 0, :3]
            self.data.mocap_quat[self._can_mocap_id] = poses[frame, 0, 3:7]
            mujoco.mj_forward(self.model, self.data)
            for scene, scene_geom_id in self._scene_geom_ids.items():
                record = records[scene]
                for robot_geom_id in self._robot_geom_ids:
                    distance = float(
                        mujoco.mj_geomDistance(
                            self.model,
                            self.data,
                            robot_geom_id,
                            scene_geom_id,
                            self._distance_limit_m,
                            closest_points,
                        )
                    )
                    if distance < record["minimum_threshold_query_return_m"]:
                        record["minimum_threshold_query_return_m"] = distance
                    if distance + 1.0e-9 >= self.required_clearance_m:
                        continue
                    label = self._geom_label(robot_geom_id)
                    record["violation_frames"].add(frame)
                    record["violating_robot_geometries"].add(label)
                    record["violating_pair_count"] += 1
                    worst = record["worst"]
                    if worst is None or distance < worst["threshold_query_return_m"]:
                        record["worst"] = {
                            "frame": frame,
                            "robot_geometry": label,
                            "threshold_query_return_m": distance,
                        }

        output_scenes: dict[str, Any] = {}
        qualified = True
        for scene, record in records.items():
            frames = sorted(record["violation_frames"])
            geometries = sorted(record["violating_robot_geometries"])
            if scene in self._gate_scenes:
                qualified = qualified and not frames
            output_scenes[scene] = {
                "minimum_clearance_lower_bound_m": (
                    self.required_clearance_m if not frames else None
                ),
                "minimum_threshold_query_return_m": float(
                    record["minimum_threshold_query_return_m"]
                ),
                "violation_frame_count": len(frames),
                "violation_frames": frames,
                "violating_robot_geometry_count": len(geometries),
                "violating_robot_geometries": geometries,
                "violating_pair_count": int(record["violating_pair_count"]),
                "worst_violation": record["worst"],
            }
        return {
            "qualified": qualified,
            "gate_scenes": list(self._gate_scenes),
            "required_clearance_m": self.required_clearance_m,
            "frame_count": len(values),
            "robot_collision_geometry_count": len(self._robot_geom_ids),
            "robot_geometry_filter": (
                "all rendered and collision geoms"
                if self._include_visual_geometry
                else (
                    "authored-collidable Wuji geoms plus Vega group-4 collision "
                    "meshes; visual/group-1 geoms excluded"
                )
            ),
            "method": (
                "fail-closed MuJoCo mj_geomDistance threshold queries against "
                "the scaled corn-can convex mesh and analytic support cylinder; "
                "returns at the cutoff are censored lower bounds, not measured "
                "clearances"
            ),
            "scenes": output_scenes,
        }


def _collision_safe_finger_projection(
    qpos: np.ndarray,
    *,
    joint_names: Sequence[str],
    object_poses: np.ndarray,
    auditor: _SceneCollisionAuditor,
    bisection_steps: int = 14,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    """Line-search one uniform closure scale per hand and fail closed."""

    if bisection_steps < 1:
        raise ValueError("Finger closure bisection steps must be positive.")
    target = np.asarray(qpos, dtype=np.float64)
    result = target.copy()
    side_indices = {
        side: np.asarray(
            [
                index
                for index, name in enumerate(joint_names)
                if name.startswith("l_" if side == "left" else "r_")
            ],
            dtype=np.int32,
        )
        for side in ("left", "right")
    }
    for indices in side_indices.values():
        result[:, indices] = 0.0
    open_audit = auditor.audit(result, object_poses)
    if not open_audit["qualified"]:
        raise ValueError(
            "Projected wrists collide with the can or support even with both "
            f"hands open: {open_audit['scenes']}."
        )

    scales: dict[str, float] = {}
    for side in ("left", "right"):
        indices = side_indices[side]

        def candidate(scale: float) -> np.ndarray:
            trial = result.copy()
            trial[:, indices] = scale * target[:, indices]
            return trial

        full = candidate(1.0)
        if auditor.audit(full, object_poses)["qualified"]:
            result = full
            scales[side] = 1.0
            continue
        lower = 0.0
        upper = 1.0
        for _ in range(bisection_steps):
            middle = 0.5 * (lower + upper)
            if auditor.audit(candidate(middle), object_poses)["qualified"]:
                lower = middle
            else:
                upper = middle
        scale = max(0.0, lower - 1.0e-3)
        result = candidate(scale)
        scales[side] = scale

    final_audit = auditor.audit(result, object_poses)
    if not final_audit["qualified"]:
        raise ValueError(
            "Finger closure projection did not produce a collision-qualified "
            f"trajectory: {final_audit['scenes']}."
        )
    final_audit["open_baseline"] = open_audit
    return result, scales, final_audit


def _project_intended_hand_penetration_to_tucked_pose(
    qpos: np.ndarray,
    *,
    joint_names: Sequence[str],
    joint_limits: Mapping[str, tuple[float, float]],
    object_poses: np.ndarray,
    auditor: _SceneCollisionAuditor,
    maximum_penetration_m: float = 0.001,
    grid_steps: int = 16,
    bisection_steps: int = 12,
    sobol_samples: int = 512,
    safe_endpoint_limit: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the nearest collision-safe finger pose with both wrists fixed.

    CHORD/keypoint projection can leave a finger link inside the object even
    though the fixed wrist admits a valid hand pose.  A single interpolation
    toward joint upper limits is not a reliable escape direction for an
    articulated hand: it can move a middle link further into the object or
    introduce finger-finger collision.  This projection therefore searches
    deterministic safe endpoints from nearby trajectory poses, simple tuck
    poses, and a seeded Sobol sequence over the side-local finger limits.  It
    then line-searches from the authored pose toward the best endpoints and
    applies the minimum-norm feasible correction.  No arm joint is variable.
    """

    target = _as_finite_array(qpos, name="finger tuck qpos")
    poses = _as_finite_array(object_poses, name="finger tuck object poses")
    if target.ndim != 2 or target.shape[1] != len(joint_names):
        raise ValueError("Finger tuck qpos must align with joint_names.")
    if poses.shape != (len(target), 1, 7):
        raise ValueError("Finger tuck object poses must have shape [T, 1, 7].")
    if not np.isfinite(maximum_penetration_m) or maximum_penetration_m < 0.0:
        raise ValueError("maximum_penetration_m must be finite and non-negative.")
    if grid_steps < 2 or bisection_steps < 1:
        raise ValueError("Finger tuck search iteration counts are too small.")
    if sobol_samples < 2 or sobol_samples & (sobol_samples - 1):
        raise ValueError("sobol_samples must be a power of two and at least 2.")
    if safe_endpoint_limit < 1:
        raise ValueError("safe_endpoint_limit must be positive.")

    names = tuple(str(name) for name in joint_names)
    qpos_addresses = auditor._qpos_addresses
    can_geom_id = auditor._scene_geom_ids["can"]
    digits = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
    side_digit_indices = {
        side: {
            digit: np.asarray(
                [
                    index
                    for index, name in enumerate(names)
                    if name.startswith("l_" if side == "left" else "r_")
                    and digit in name
                    and not name.endswith("_abd")
                ],
                dtype=np.int32,
            )
            for digit in digits
        }
        for side in ("left", "right")
    }
    side_geom_ids: dict[str, tuple[int, ...]] = {}
    for side, body_prefix in (("left", "l_"), ("right", "r_")):
        side_geom_ids[side] = tuple(
            int(geom_id)
            for geom_id in auditor._robot_geom_ids
            if (
                auditor.model.body(
                    int(auditor.model.geom_bodyid[int(geom_id)])
                ).name.startswith(body_prefix)
                and auditor.model.body(
                    int(auditor.model.geom_bodyid[int(geom_id)])
                ).name
                != f"{body_prefix}wrist"
            )
        )
        if (
            any(not len(indices) for indices in side_digit_indices[side].values())
            or not side_geom_ids[side]
        ):
            raise ValueError(f"No articulated {side} hand flexion joints/geoms found.")

    lower = np.asarray([joint_limits[name][0] for name in names], dtype=np.float64)
    upper = np.asarray([joint_limits[name][1] for name in names], dtype=np.float64)
    result = target.copy()
    segment = np.zeros(6, dtype=np.float64)
    query_cutoff = 0.005

    robot_geom_ids = frozenset(auditor._robot_geom_ids)
    support_geom_id = auditor._scene_geom_ids["support"]

    def set_state(frame: int, values: np.ndarray) -> None:
        mujoco.mj_resetData(auditor.model, auditor.data)
        auditor.data.qpos[qpos_addresses] = values
        auditor.data.mocap_pos[auditor._can_mocap_id] = poses[frame, 0, :3]
        auditor.data.mocap_quat[auditor._can_mocap_id] = poses[frame, 0, 3:7]
        mujoco.mj_forward(auditor.model, auditor.data)

    def minimum_distance(frame: int, side: str, values: np.ndarray) -> float:
        set_state(frame, values)
        return min(
            float(
                mujoco.mj_geomDistance(
                    auditor.model,
                    auditor.data,
                    geom_id,
                    int(can_geom_id),
                    query_cutoff,
                    segment,
                )
            )
            for geom_id in side_geom_ids[side]
        )

    def collision_safe(frame: int, side: str, values: np.ndarray) -> bool:
        set_state(frame, values)
        if any(
            float(
                mujoco.mj_geomDistance(
                    auditor.model,
                    auditor.data,
                    geom_id,
                    int(can_geom_id),
                    query_cutoff,
                    segment,
                )
            )
            < threshold - 1.0e-9
            for geom_id in side_geom_ids[side]
        ):
            return False
        if any(
            float(
                mujoco.mj_geomDistance(
                    auditor.model,
                    auditor.data,
                    geom_id,
                    int(support_geom_id),
                    auditor.required_clearance_m,
                    segment,
                )
            )
            < auditor.required_clearance_m - 1.0e-9
            for geom_id in side_geom_ids[side]
        ):
            return False
        return not any(
            int(contact.geom1) in robot_geom_ids
            and int(contact.geom2) in robot_geom_ids
            and float(contact.dist) < -0.001 - 1.0e-9
            for contact in auditor.data.contact[: auditor.data.ncon]
        )

    threshold = -float(maximum_penetration_m)
    report_sides: dict[str, Any] = {}
    for side in ("left", "right"):
        prefix = "l_" if side == "left" else "r_"
        side_indices = np.asarray(
            [index for index, name in enumerate(names) if name.startswith(prefix)],
            dtype=np.int32,
        )
        if not len(side_indices):
            raise ValueError(f"No {side} finger joints found.")
        if not (
            np.isfinite(lower[side_indices]).all()
            and np.isfinite(upper[side_indices]).all()
        ):
            raise ValueError(f"{side.capitalize()} finger limits must be finite.")
        sobol_unit = qmc.Sobol(
            d=len(side_indices),
            scramble=True,
            seed=20260820 + (0 if side == "left" else 1),
        ).random_base2(int(math.log2(sobol_samples)))
        sobol_values = qmc.scale(
            sobol_unit,
            lower[side_indices],
            upper[side_indices],
        )

        corrected_frames: list[int] = []
        scales: dict[int, float] = {}
        selected_digits: dict[int, list[str]] = {}
        selected_sources: dict[int, str] = {}
        correction_norms: dict[int, float] = {}
        proposal_counts: dict[int, int] = {}
        for frame in range(len(result)):
            frame_seed = result[frame].copy()
            if minimum_distance(frame, side, frame_seed) >= threshold - 1.0e-9:
                continue

            proposals: list[tuple[str, tuple[str, ...], np.ndarray]] = []
            for source_frame in sorted(
                range(len(result)), key=lambda index: (abs(index - frame), index)
            ):
                if source_frame == frame:
                    continue
                trial = frame_seed.copy()
                trial[side_indices] = result[source_frame, side_indices]
                proposals.append((f"trajectory_frame_{source_frame}", (), trial))

            for subset_size in range(1, len(digits) + 1):
                for subset in combinations(digits, subset_size):
                    indices = np.concatenate(
                        [side_digit_indices[side][digit] for digit in subset]
                    )
                    trial = frame_seed.copy()
                    trial[indices] = upper[indices]
                    proposals.append(("digit_upper_tuck", subset, trial))

            for label, values in (
                ("joint_lower", lower[side_indices]),
                ("joint_midpoint", 0.5 * (lower[side_indices] + upper[side_indices])),
                (
                    "joint_zero",
                    np.clip(
                        np.zeros(len(side_indices), dtype=np.float64),
                        lower[side_indices],
                        upper[side_indices],
                    ),
                ),
                ("joint_upper", upper[side_indices]),
            ):
                trial = frame_seed.copy()
                trial[side_indices] = values
                proposals.append((label, (), trial))
            for sample_index, values in enumerate(sobol_values):
                trial = frame_seed.copy()
                trial[side_indices] = values
                proposals.append((f"sobol_{sample_index}", (), trial))

            proposal_counts[frame] = len(proposals)
            safe_endpoints = [
                (float(np.linalg.norm(trial - frame_seed)), label, subset, trial)
                for label, subset, trial in proposals
                if collision_safe(frame, side, trial)
            ]
            if not safe_endpoints:
                raise ValueError(
                    f"No deterministic {side} finger sample satisfies can, "
                    f"support, and self-collision constraints at frame {frame}; "
                    "the wrist placement is not finger-feasible."
                )

            candidates: list[tuple[float, str, tuple[str, ...], float, np.ndarray]] = []
            for _, label, subset, endpoint in sorted(
                safe_endpoints, key=lambda item: item[0]
            )[:safe_endpoint_limit]:
                unsafe_scale = 0.0
                safe_scale: float | None = None
                for scale in np.linspace(0.0, 1.0, grid_steps + 1)[1:]:
                    trial = frame_seed + float(scale) * (endpoint - frame_seed)
                    if collision_safe(frame, side, trial):
                        safe_scale = float(scale)
                        break
                    unsafe_scale = float(scale)
                if safe_scale is None:
                    continue
                for _ in range(bisection_steps):
                    scale = 0.5 * (unsafe_scale + safe_scale)
                    trial = frame_seed + scale * (endpoint - frame_seed)
                    if collision_safe(frame, side, trial):
                        safe_scale = scale
                    else:
                        unsafe_scale = scale
                trial = frame_seed + safe_scale * (endpoint - frame_seed)
                if collision_safe(frame, side, trial):
                    candidates.append(
                        (
                            float(np.linalg.norm(trial - frame_seed)),
                            label,
                            subset,
                            float(safe_scale),
                            trial,
                        )
                    )
            if not candidates:
                raise ValueError(
                    f"Safe {side} finger endpoints exist at frame {frame}, but "
                    "no collision-safe interpolation from the authored pose was "
                    "found."
                )
            correction_norm, label, subset, final_scale, trial = min(
                candidates, key=lambda item: item[0]
            )
            result[frame] = trial
            corrected_frames.append(frame)
            scales[frame] = float(final_scale)
            selected_digits[frame] = list(subset)
            selected_sources[frame] = label
            correction_norms[frame] = float(correction_norm)
        report_sides[side] = {
            "corrected_frame_count": len(corrected_frames),
            "corrected_frames": corrected_frames,
            "minimum_projection_scale": (
                None if not scales else float(min(scales.values()))
            ),
            "maximum_projection_scale": (
                None if not scales else float(max(scales.values()))
            ),
            "maximum_correction_norm_rad": (
                None if not correction_norms else float(max(correction_norms.values()))
            ),
            "per_frame_projection_scale": {str(k): v for k, v in scales.items()},
            "per_frame_endpoint_source": {
                str(k): v for k, v in selected_sources.items()
            },
            "per_frame_endpoint_digits": {
                str(k): v for k, v in selected_digits.items()
            },
            "per_frame_correction_norm_rad": {
                str(k): v for k, v in correction_norms.items()
            },
            "per_frame_proposal_count": {str(k): v for k, v in proposal_counts.items()},
        }

    final_audit = _audit_semantic_robot_can_clearance(
        auditor,
        result,
        poses,
        maximum_intended_penetration_m=maximum_penetration_m,
    )
    if not final_audit["qualified"]:
        raise ValueError(
            "Final finger tuck did not satisfy the typed robot-can audit: "
            f"{final_audit['groups']}."
        )
    final_self_audit = _audit_robot_self_penetration(
        auditor, result, penetration_tolerance_m=0.001
    )
    if not final_self_audit["qualified"]:
        raise ValueError(
            f"Final digit tuck reintroduced robot self-collision: {final_self_audit}."
        )
    return result, {
        "method": (
            "per-frame deterministic multistart safe-endpoint search followed "
            "by minimum-norm interpolation; can, support, and self-collision "
            "constrained; all wrist-IK joints fixed"
        ),
        "maximum_penetration_m": float(maximum_penetration_m),
        "sobol_samples_per_side": int(sobol_samples),
        "safe_endpoint_line_search_limit": int(safe_endpoint_limit),
        "sides": report_sides,
        "final_semantic_can_audit": final_audit,
        "final_self_collision_audit": final_self_audit,
    }


def _has_exact_contact_geometry(row: Mapping[str, Any]) -> bool:
    def has_leaf(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return any(has_leaf(item) for item in value)
        return True

    fields = (
        "hand_contact_link_names",
        "hand_link_contact_positions",
        "hand_link_contact_normals",
        "hand_object_contact_positions",
        "hand_object_contact_normals",
        "hand_object_contact_part_ids",
    )
    present = [has_leaf(row[name]) for name in fields]
    if any(present) and not all(present):
        raise ValueError("Source contact geometry is incomplete.")
    return all(present)


def _inspection_contacts(frame_count: int) -> ContactSequence:
    """Return a named, inactive layout that permits visual replay only."""

    link_names = np.asarray(
        (FINGERTIP_NAMES["left"], FINGERTIP_NAMES["right"]), dtype=np.str_
    )
    vector_shape = (frame_count, 2, 5, 3)
    index_shape = (frame_count, 2, 5)
    return ContactSequence(
        hand_sides=("left", "right"),
        link_names=link_names,
        link_positions_w=np.zeros(vector_shape, dtype=np.float32),
        link_normals_w=np.zeros(vector_shape, dtype=np.float32),
        object_positions_w=np.zeros(vector_shape, dtype=np.float32),
        object_normals_w=np.zeros(vector_shape, dtype=np.float32),
        object_indices=np.full(index_shape, -1, dtype=np.int32),
        active=np.zeros(index_shape, dtype=np.bool_),
    )


def _contact_recovery_gate(
    report: Mapping[str, Any] | None,
    *,
    minimum_per_side_rate: float = 0.50,
) -> tuple[bool, dict[str, float]]:
    """Require the recovery threshold independently on every active hand."""

    if (
        report is None
        or report.get("status") == "blocked"
        or not np.isfinite(minimum_per_side_rate)
        or not 0.0 <= minimum_per_side_rate <= 1.0
    ):
        return False, {}
    recovered = report.get("per_side_recovered")
    unreached = report.get("per_side_unreached")
    if not isinstance(recovered, Mapping) or not isinstance(unreached, Mapping):
        return False, {}
    rates: dict[str, float] = {}
    for side in ("left", "right"):
        recovered_count = int(recovered.get(side, 0))
        unreached_count = int(unreached.get(side, 0))
        total = recovered_count + unreached_count
        if total <= 0:
            continue
        rates[side] = recovered_count / total
    return bool(rates) and all(
        rate >= minimum_per_side_rate for rate in rates.values()
    ), rates


def _verify_locked_joint_trajectory(
    before: np.ndarray,
    after: np.ndarray,
    *,
    joint_names: Sequence[str],
    locked_joint_names: Sequence[str],
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Fail closed if a hand-refinement stage changes wrist-IK joints."""

    source = np.asarray(before, dtype=np.float64)
    candidate = np.asarray(after, dtype=np.float64)
    names = tuple(str(name) for name in joint_names)
    locked = tuple(str(name) for name in locked_joint_names)
    if source.shape != candidate.shape or source.ndim != 2:
        raise ValueError("Locked-joint trajectories must share shape [T, J].")
    if source.shape[1] != len(names):
        raise ValueError("Locked-joint trajectories must align with joint_names.")
    if not locked or len(set(locked)) != len(locked):
        raise ValueError("locked_joint_names must be non-empty and unique.")
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = tuple(name for name in locked if name not in name_to_index)
    if missing:
        raise ValueError(f"Locked joints are absent from joint_names: {missing}.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative.")

    indices = np.asarray([name_to_index[name] for name in locked], dtype=np.int32)
    absolute_drift = np.abs(candidate[:, indices] - source[:, indices])
    max_drift = float(np.max(absolute_drift))
    changed = np.argwhere(absolute_drift > float(atol))
    if len(changed):
        frame, locked_index = changed[0]
        joint_name = locked[int(locked_index)]
        raise RuntimeError(
            "A post-wrist hand refinement changed a locked IK joint: "
            f"frame={int(frame)}, joint={joint_name!r}, "
            f"drift={absolute_drift[frame, locked_index]:.6e}."
        )
    return {
        "verified": True,
        "checked_frame_count": int(len(source)),
        "locked_joint_names": list(locked),
        "absolute_tolerance_rad_or_m": float(atol),
        "maximum_absolute_drift_rad_or_m": max_drift,
    }


def convert_soma_g1_parquet(
    source_path: str | Path | None = None,
    *,
    pin: SequencePin = _CORN_CAN_PIN,
    model_path: str | Path = DEFAULT_MODEL,
    support_path: str | Path | None = None,
    start_frame: int | None = None,
    end_frame_exclusive: int | None = None,
    anchor_frame: int | None = None,
    object_anchor_target: np.ndarray | None = None,
    global_motion_scale: float = DEFAULT_GLOBAL_MOTION_SCALE,
    local_geometry_scale: float = DEFAULT_LOCAL_GEOMETRY_SCALE,
    contact_dilation_frames: int = DEFAULT_CONTACT_DILATION_FRAMES,
    contact_blend_sigma_frames: float = DEFAULT_CONTACT_BLEND_SIGMA_FRAMES,
    palm_object_standoff_m: float = DEFAULT_PALM_OBJECT_STANDOFF_M,
    palm_support_standoff_m: float = DEFAULT_PALM_SUPPORT_STANDOFF_M,
    geometry_clearance_m: float = DEFAULT_GEOMETRY_CLEARANCE_M,
    dense_geometry_audit_fps: float = DEFAULT_DENSE_GEOMETRY_AUDIT_FPS,
    fixed_root_pose_w: np.ndarray = DEFAULT_FIXED_ROOT_POSE_W,
    filter_cutoff_hz: float = DEFAULT_FILTER_CUTOFF_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
    motion_time_scale: float = DEFAULT_MOTION_TIME_SCALE,
    qualified_state_hold_frames: int = 1,
    contact_seeking: bool = False,
    soma_chord_contacts: bool = False,
    soma_device: str = "cuda",
    contact_target_ik: bool = False,
    contact_guided_finger_ik: bool = False,
    contact_settle_steps: int = 60,
    phase_aware_object_placement: bool = False,
    phase_anchor_max_shift_m: float = 0.20,
    phase_object_anchor_targets: np.ndarray | None = None,
    max_wrist_position_error: float = 0.05,
    max_wrist_orientation_error: float = math.pi,
    wrist_orientation_weight: float = DEFAULT_IK_ORIENTATION_WEIGHT,
    arm_previous_posture_weight: float = 0.0,
    arm_neutral_posture_weight: float = 0.0,
    position_priority_wrist_ik: bool = False,
    position_priority_slack_m: float = 0.005,
    bent_elbow_inactive_rest: bool = False,
    split_wrist_hand_ik: bool = True,
    anatomical_finger_limits: bool = True,
    soma_hand_scale: float = DEFAULT_SOMA_TO_WUJI_HAND_SCALE,
    object_anchored_fingertips: bool = True,
    # Measured negative on the snack-box clip: standing the fingertip site off
    # by its 9 mm collision radius rotates the finger and drives the middle
    # phalanges into the object, so whole-hand penetration got worse
    # (-10.25 -> -15.73 mm) even though the tip itself sat correctly.
    fingertip_contact_standoff_m: float = 0.0,
    reconcile_fingertips_after_contact: bool = True,
    reconciliation_clearance_m: float | None = 0.0005,
    mcp_abduction_limit_deg: tuple[float, float] | None = None,
    shared_positioning_axes: bool = True,
    align_finger_directions: bool = True,
    hand_morphology_standoff: bool = False,
    hand_morphology_standoff_m: float | None = None,
    finger_posture_prior: float = DEFAULT_FINGER_POSTURE_PRIOR,
    require_finger_gates: bool = False,
    inspection_only: bool = False,
    expected_parquet_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
) -> DexterousReference:
    """Build one validated Reference without writing it to disk."""

    # Sequence-scoped values default to the selected pin.
    if source_path is None:
        source_path = pin.source_path
    if support_path is None:
        support_path = pin.support_path
    start_frame = pin.start_frame if start_frame is None else int(start_frame)
    end_frame_exclusive = (
        pin.end_frame_exclusive
        if end_frame_exclusive is None
        else int(end_frame_exclusive)
    )
    if object_anchor_target is None:
        object_anchor_target = np.asarray(pin.object_anchor_target, dtype=np.float64)
    if expected_parquet_sha256 is None:
        expected_parquet_sha256 = pin.parquet_sha256
    if expected_payload_sha256 is None:
        expected_payload_sha256 = pin.payload_sha256

    source = Path(source_path).expanduser().resolve()
    model_file = Path(model_path).expanduser().resolve()
    support = Path(support_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source Parquet is missing: {source}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Vega-Wuji MJCF is missing: {model_file}")
    if not support.is_file():
        raise FileNotFoundError(f"Support USD is missing: {support}")
    actual_parquet_hash = _sha256_file(source)
    if actual_parquet_hash != expected_parquet_sha256:
        raise ValueError(
            "Source Parquet SHA-256 mismatch: "
            f"expected {expected_parquet_sha256}, got {actual_parquet_hash}."
        )

    row = _read_parquet_row(source)
    payload_bytes = row["source_payload"]
    if not isinstance(payload_bytes, bytes):
        raise ValueError("source_payload must be bytes.")
    payload = _safe_payload(payload_bytes, expected_sha256=expected_payload_sha256)
    arrays = _validate_source(row, payload, expected_sequence_id=pin.sequence_id)
    frame_count = len(arrays["ee_pose_w"])
    if not 0 <= start_frame < end_frame_exclusive <= frame_count:
        raise ValueError(
            f"Crop [{start_frame}, {end_frame_exclusive}) is outside [0, {frame_count})."
        )
    anchor = start_frame if anchor_frame is None else int(anchor_frame)
    if not start_frame <= anchor < end_frame_exclusive:
        raise ValueError("anchor_frame must be inside the source crop.")
    for name, value in (
        ("global_motion_scale", global_motion_scale),
        ("local_geometry_scale", local_geometry_scale),
        ("phase_anchor_max_shift_m", phase_anchor_max_shift_m),
        ("motion_time_scale", motion_time_scale),
        ("wrist_orientation_weight", wrist_orientation_weight),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in (
        ("arm_previous_posture_weight", arm_previous_posture_weight),
        ("arm_neutral_posture_weight", arm_neutral_posture_weight),
        ("position_priority_slack_m", position_priority_slack_m),
        ("finger_posture_prior", finger_posture_prior),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if not np.isfinite(soma_hand_scale) or soma_hand_scale <= 0.0:
        raise ValueError("soma_hand_scale must be finite and positive.")
    if finger_posture_prior > 0.0 and not split_wrist_hand_ik:
        raise ValueError(
            "finger_posture_prior requires split_wrist_hand_ik: the tips-only "
            "finger IK must not move arm joints."
        )
    if contact_guided_finger_ik and not contact_seeking:
        raise ValueError("contact_guided_finger_ik requires contact_seeking.")
    if not inspection_only:
        adopted_recipe_errors: list[str] = []
        if not split_wrist_hand_ik:
            adopted_recipe_errors.append("split_wrist_hand_ik must be enabled")
        if not anatomical_finger_limits:
            adopted_recipe_errors.append("anatomical_finger_limits must be enabled")
        if not np.isclose(soma_hand_scale, 1.0, rtol=0.0, atol=1.0e-12):
            adopted_recipe_errors.append("soma_hand_scale must equal 1.0")
        if not np.isclose(
            finger_posture_prior,
            DEFAULT_FINGER_POSTURE_PRIOR,
            rtol=0.0,
            atol=1.0e-12,
        ):
            adopted_recipe_errors.append(
                f"finger_posture_prior must equal {DEFAULT_FINGER_POSTURE_PRIOR:g}"
            )
        if not np.isclose(
            wrist_orientation_weight,
            DEFAULT_IK_ORIENTATION_WEIGHT,
            rtol=0.0,
            atol=1.0e-12,
        ):
            adopted_recipe_errors.append(
                f"wrist_orientation_weight must equal {DEFAULT_IK_ORIENTATION_WEIGHT:g}"
            )
        if adopted_recipe_errors:
            raise ValueError(
                "Training-candidate conversion requires the adopted Wuji recipe: "
                + "; ".join(adopted_recipe_errors)
                + "."
            )
    if (
        int(qualified_state_hold_frames) != qualified_state_hold_frames
        or int(qualified_state_hold_frames) < 1
    ):
        raise ValueError("qualified_state_hold_frames must be a positive integer.")
    object_anchor = _as_finite_array(
        object_anchor_target, name="object_anchor_target", shape=(3,)
    )
    explicit_phase_anchors = None
    if phase_object_anchor_targets is not None:
        explicit_phase_anchors = _as_finite_array(
            phase_object_anchor_targets,
            name="phase_object_anchor_targets",
            shape=(2, 3),
        )
        if not phase_aware_object_placement:
            raise ValueError(
                "phase_object_anchor_targets requires phase_aware_object_placement."
            )
    fixed_root = _unit_wxyz(
        _as_finite_array(fixed_root_pose_w, name="fixed_root_pose_w", shape=(7,))[3:7],
        name="fixed_root_pose_w",
    )
    fixed_root = np.concatenate(
        (np.asarray(fixed_root_pose_w, dtype=np.float64)[:3], fixed_root)
    )

    if soma_chord_contacts and not contact_seeking:
        raise ValueError("soma_chord_contacts requires contact_seeking.")
    if contact_target_ik and not soma_chord_contacts:
        raise ValueError("contact_target_ik requires soma_chord_contacts.")

    exact_contacts = _has_exact_contact_geometry(row)
    if exact_contacts:
        raise NotImplementedError(
            "This pinned converter has no verified G1-contact-link to Wuji-link "
            "mapping. Add and test that mapping before converting exact contacts."
        )
    if not inspection_only and not soma_chord_contacts:
        raise ValueError(
            "The pinned SOMA Parquet has binary hand_contact_active only. Exact "
            "contact link names, points, normals, and object part IDs are absent. "
            "Refusing to invent a ContactSequence. Enable --soma-chord-contacts "
            "to reconstruct measured mesh contacts, or pass --inspection-only "
            "for a non-runtime-qualified inspection Reference."
        )

    source_fps = float(row["fps"])
    source_times, target_times = _target_times(
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        source_fps=source_fps,
        motion_time_scale=float(motion_time_scale),
    )
    crop = slice(start_frame, end_frame_exclusive)
    model = mujoco.MjModel.from_xml_path(str(model_file))
    # Tighten the finger limits inside the model BEFORE any stage reads them:
    # the joint contract, hinge mapping bounds, every keypoint/contact
    # projector, and the final clip all inherit the anatomical envelope, so a
    # tips-only solve cannot pick a Z-folded (hyperextended) finger.
    anatomical_finger_limit_rows: list[dict[str, Any]] = []
    if anatomical_finger_limits:
        finger_limits_deg = dict(WUJI_ANATOMICAL_LIMITS_DEG)
        if mcp_abduction_limit_deg is not None:
            lower, upper = (float(value) for value in mcp_abduction_limit_deg)
            if not np.isfinite((lower, upper)).all() or lower >= upper:
                raise ValueError(
                    "mcp_abduction_limit_deg must be a finite (lower, upper) "
                    "pair with lower < upper."
                )
            finger_limits_deg["mcp_abd"] = (lower, upper)
        anatomical_finger_limit_rows = apply_anatomical_finger_limits(
            model, limits_deg=finger_limits_deg
        )
    joint_names, limits = _model_joint_contract(model)
    neutral_poses = _neutral_wrist_poses(model)
    initial_retarget_qpos: np.ndarray | None = None
    inactive_rest_poses = neutral_poses
    if bent_elbow_inactive_rest:
        initial_retarget_qpos, inactive_rest_poses = _bent_elbow_inactive_rest(
            model, joint_names
        )
    neutral_fingertip_vectors = _neutral_fingertip_vectors(model)

    object_urdf_values = tuple(str(value) for value in row["object_urdf_paths"])
    if len(object_urdf_values) != 1:
        raise ValueError("Pinned source must declare one rigid-object URDF.")
    object_mesh_values = tuple(str(value) for value in row["object_mesh_paths"])
    if len(object_mesh_values) != 1:
        raise ValueError("Pinned source must declare one rigid-object mesh.")
    object_urdf = _resolve_relative_asset(source, object_urdf_values[0])
    object_mesh = _resolve_relative_asset(source, object_mesh_values[0])
    if object_mesh != (object_urdf.parent / "textured_mesh.obj").resolve():
        raise ValueError("Pinned object mesh path does not match the URDF dependency.")
    asset_hashes = _asset_dependency_hashes(
        object_urdf, object_mesh, expected_sha256=pin.object_asset_sha256
    )
    collision_asset_dependencies = build_urdf_collision_asset_dependencies(
        object_urdf,
        asset_role="object",
        asset_index=0,
    )
    support_hash = _sha256_file(support)
    if support_hash != pin.support_sha256:
        raise ValueError(
            "Pinned support SHA-256 mismatch: "
            f"expected {pin.support_sha256}, got {support_hash}."
        )

    chord_source_contacts: Any | None = None
    chord_source_report: dict[str, Any] | None = None
    if soma_chord_contacts:
        chord_source_contacts, chord_source_report = (
            _extract_soma_chord_source_contacts(
                arrays=arrays,
                payload=payload,
                crop=crop,
                object_mesh=object_mesh,
                device=str(soma_device),
            )
        )

    finger_source = _retarget_fingers(
        arrays["soma_joints"],
        source_fps=source_fps,
        target_joint_names=joint_names,
        limits=limits,
    )
    filtered_fingers = {
        side: _lowpass(
            value,
            fps=source_fps,
            cutoff_hz=filter_cutoff_hz,
            order=filter_order,
        )
        for side, value in finger_source.items()
    }

    (
        soma_hand_position,
        soma_hand_wxyz,
        soma_hand_keypoints,
        soma_hand_knuckles,
        soma_hand_pips,
        soma_frame_bridge,
    ) = _soma_hand_poses_in_target_world(
        arrays=arrays,
        payload=payload,
        crop=crop,
    )
    identity_rest_knuckles_local = _soma_identity_rest_knuckles_in_calibration_frames(
        payload=payload,
        bridge=soma_frame_bridge,
        device=str(soma_device),
    )
    source_hand_keypoints_crop = _lowpass(
        soma_hand_keypoints,
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
    source_hand_wxyz_crop = _lowpass_wxyz(
        soma_hand_wxyz,
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
    palm_frame_calibration = _fit_soma_wuji_palm_frames(
        model,
        source_hand_positions=soma_hand_position,
        source_hand_wxyz=soma_hand_wxyz,
        source_knuckles=soma_hand_knuckles,
        identity_rest_knuckles_local=identity_rest_knuckles_local,
        source_pips=soma_hand_pips,
        align_finger_directions=bool(align_finger_directions),
    )
    source_wrist_position_crop = _lowpass(
        soma_hand_position,
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
    source_wrist_wxyz_crop = source_hand_wxyz_crop
    wrist_pose_source = "bridged_soma_anatomical_hand_root"
    filtered_object_position = _lowpass(
        arrays["object_body_position"],
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
    filtered_object_wxyz = _lowpass_wxyz(
        arrays["object_body_wxyz"],
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
    source_object_anchor = arrays["object_body_position"][anchor, 0].copy()
    filtered_object_position += (
        source_object_anchor - filtered_object_position[anchor, 0]
    )
    filtered_object_rotation = _rotation_from_wxyz(filtered_object_wxyz.reshape(-1, 4))
    raw_anchor_rotation = _rotation_from_wxyz(
        arrays["object_body_wxyz"][anchor, 0][None, :]
    )
    filtered_anchor_rotation = _rotation_from_wxyz(
        filtered_object_wxyz[anchor, 0][None, :]
    )
    object_orientation_anchor_offset = (
        filtered_anchor_rotation.inv() * raw_anchor_rotation
    )
    filtered_object_wxyz = _rotation_to_wxyz(
        filtered_object_rotation * object_orientation_anchor_offset
    ).reshape(filtered_object_wxyz.shape)

    projection_rotation = _yaw_only_inverse(arrays["robot_root_wxyz"][anchor])
    contact_blend_200hz = _contact_phase_weights(
        arrays["hand_contact_active"][:, crop],
        dilation_frames=contact_dilation_frames,
        gaussian_sigma_frames=contact_blend_sigma_frames,
    )
    neutral_positions = np.stack(
        [neutral_poses[side][:3] for side in ("left", "right")], axis=0
    )
    inactive_rest_positions = np.stack(
        [inactive_rest_poses[side][:3] for side in ("left", "right")], axis=0
    )
    phase_anchor_targets = np.repeat(object_anchor[None, :], 2, axis=0)
    target_anchor_trajectory = None
    if phase_aware_object_placement:
        phase_anchor_targets = (
            explicit_phase_anchors
            if explicit_phase_anchors is not None
            else _infer_phase_object_anchors(
                source_wrist_position_crop,
                filtered_object_position[crop],
                arrays["hand_contact_active"][:, crop],
                rotation=projection_rotation,
                local_geometry_scale=float(local_geometry_scale),
                neutral_positions=neutral_positions,
                base_anchor=object_anchor,
                max_shift_m=float(phase_anchor_max_shift_m),
            )
        )
        target_anchor_trajectory = _phase_object_anchor_trajectory(
            object_anchor,
            phase_anchor_targets,
            contact_blend_200hz,
        )
    projection = ContactPhaseProjection(
        rotation=projection_rotation,
        source_object_anchor=source_object_anchor,
        target_object_anchor=object_anchor,
        global_motion_scale=float(global_motion_scale),
        local_geometry_scale=float(local_geometry_scale),
        target_object_anchor_trajectory=target_anchor_trajectory,
        support_center_source=pin.support_center_source,
        support_height_source=pin.support_height_source,
    )
    object_position_200hz = projection.object_positions(filtered_object_position[crop])
    calibrated_wrist_wxyz, wrist_offsets = _calibrate_soma_wrist_orientations(
        source_wrist_wxyz_crop,
        alignment_rotation=projection.rotation,
        palm_frame_calibration=palm_frame_calibration,
    )
    active_wrist_position_200hz = projection.active_wrist_positions(
        source_wrist_position_crop, filtered_object_position[crop]
    )
    calibrated_wrist_rotations = (
        _rotation_from_wxyz(calibrated_wrist_wxyz.reshape(-1, 4))
        .as_matrix()
        .reshape(len(calibrated_wrist_wxyz), 2, 3, 3)
    )
    for side_index, side in enumerate(("left", "right")):
        local_offset = np.zeros(3, dtype=np.float64)
        local_offset[int(palm_frame_calibration[side]["palm_normal_axis"])] = float(
            palm_frame_calibration[side]["palm_normal_offset_m"]
        )
        active_wrist_position_200hz[:, side_index] += np.einsum(
            "tij,j->ti", calibrated_wrist_rotations[:, side_index], local_offset
        )
    wrist_position_200hz, projected_active_wrist_200hz, safe_rest_positions = (
        _project_palm_positions_nonpenetrating(
            active_wrist_position_200hz,
            object_position_200hz,
            inactive_rest_positions,
            contact_blend_200hz,
            support_top_z=projection.support_top_z(),
            palm_object_standoff_m=palm_object_standoff_m,
            palm_support_standoff_m=palm_support_standoff_m,
            raise_rest_above_support=not bent_elbow_inactive_rest,
        )
    )
    wrist_wxyz_200hz = _blend_wrist_orientations(
        calibrated_wrist_wxyz, inactive_rest_poses, contact_blend_200hz
    )
    object_wxyz_200hz = projection.orientations(filtered_object_wxyz[crop])

    soma_fingertip_targets_200hz: np.ndarray | None = None
    soma_fingertip_targets_20hz: np.ndarray | None = None
    soma_hand_target_report: dict[str, Any] | None = None
    hand_morphology_report: dict[str, Any] | None = None
    object_anchored_report: dict[str, Any] | None = None
    if source_hand_keypoints_crop is not None:
        soma_fingertip_targets_200hz, soma_hand_target_report = (
            _map_soma_fingertips_to_wuji_targets(
                source_keypoints=source_hand_keypoints_crop,
                source_hand_wxyz=source_hand_wxyz_crop,
                wrist_positions=wrist_position_200hz,
                wrist_wxyz=wrist_wxyz_200hz,
                palm_frame_calibration=palm_frame_calibration,
                neutral_fingertip_vectors=neutral_fingertip_vectors,
                blend_weights=contact_blend_200hz,
                hand_scale=soma_hand_scale,
            )
        )
        soma_fingertip_targets_20hz = _resample_linear(
            soma_fingertip_targets_200hz, source_times, target_times
        )
        # The fingertip targets above are final. Only the palm target moves,
        # so the Wuji hand's longer fingers span from a feasible palm pose to
        # the same contact points instead of curling through the object.
        if hand_morphology_standoff:
            hand_standoff_m = _hand_morphology_standoff_m(
                soma_hand_target_report,
                override_m=hand_morphology_standoff_m,
            )
            wrist_position_200hz, hand_morphology_report = (
                _retreat_wrist_targets_for_hand_morphology(
                    wrist_position_200hz,
                    wrist_wxyz_200hz,
                    neutral_fingertip_vectors=neutral_fingertip_vectors,
                    standoff_m=hand_standoff_m,
                    blend_weights=contact_blend_200hz,
                )
            )

    object_position_20hz = _resample_linear(
        object_position_200hz, source_times, target_times
    )
    object_wxyz_20hz = _resample_wxyz(object_wxyz_200hz, source_times, target_times)
    contact_active_20hz = _resample_binary(
        arrays["hand_contact_active"],
        source_fps=source_fps,
        target_times=target_times,
    )
    target_wrist_position_20hz = _resample_linear(
        wrist_position_200hz, source_times, target_times
    )
    target_wrist_wxyz_20hz = _resample_wxyz(
        wrist_wxyz_200hz, source_times, target_times
    )
    contact_blend_20hz = _resample_linear(
        contact_blend_200hz.T, source_times, target_times
    ).T

    object_pose_200hz = np.concatenate(
        (object_position_200hz, object_wxyz_200hz), axis=-1
    )
    chord_active_200hz: np.ndarray | None = None
    chord_targets_200hz: np.ndarray | None = None
    chord_normals_200hz: np.ndarray | None = None
    chord_active_20hz: np.ndarray | None = None
    chord_targets_20hz: np.ndarray | None = None
    chord_normals_20hz: np.ndarray | None = None
    if chord_source_contacts is not None:
        source_object_pose_crop = np.concatenate(
            (
                arrays["object_body_position"][crop, 0],
                arrays["object_body_wxyz"][crop, 0],
            ),
            axis=-1,
        )
        (
            chord_active_200hz,
            chord_targets_200hz,
            chord_normals_200hz,
        ) = _map_chord_targets_to_object_trajectory(
            chord_source_contacts,
            source_object_poses=source_object_pose_crop,
            target_object_poses=object_pose_200hz[:, 0],
            geometry_scale=float(local_geometry_scale),
        )
        (
            chord_active_20hz,
            chord_targets_20hz,
            chord_normals_20hz,
        ) = _resample_chord_targets_nearest(
            chord_active_200hz,
            chord_targets_200hz,
            chord_normals_200hz,
            source_fps=source_fps,
            target_times=target_times,
            source_start_time=float(source_times[0]),
        )
        if object_anchored_fingertips and soma_fingertip_targets_200hz is not None:
            soma_fingertip_targets_200hz, object_anchored_report = (
                _object_anchored_fingertip_targets(
                    soma_fingertip_targets_200hz,
                    chord_targets=chord_targets_200hz,
                    chord_active=chord_active_200hz,
                    blend_frames=contact_dilation_frames,
                    chord_normals=chord_normals_200hz,
                    tip_radius_m=float(fingertip_contact_standoff_m),
                )
            )
            # Re-derive the emitted-rate targets so the acceptance gate scores
            # against the same points the solver was given.
            soma_fingertip_targets_20hz = _resample_linear(
                soma_fingertip_targets_200hz, source_times, target_times
            )
    source_trajectory_200hz = Trajectory(
        observations={
            "left_wrist_pose_w": np.concatenate(
                (wrist_position_200hz[:, 0], wrist_wxyz_200hz[:, 0]), axis=-1
            ),
            "right_wrist_pose_w": np.concatenate(
                (wrist_position_200hz[:, 1], wrist_wxyz_200hz[:, 1]), axis=-1
            ),
            "left_finger_qpos": (
                filtered_fingers["left"][crop] * contact_blend_200hz[0, :, None]
            ),
            "right_finger_qpos": (
                filtered_fingers["right"][crop] * contact_blend_200hz[1, :, None]
            ),
            "object_poses_w": object_pose_200hz,
        },
        infos={"coordinate_frame": "robot"},
        dt=1.0 / source_fps,
    )
    # Lift and torso_flip are shared positioning axes for both arms. Locking
    # them costs the dual-wrist solve two degrees of freedom, and the palm
    # orientation it then cannot reach is paid for by the fingers: measured on
    # the snack-box clip, locking them leaves 23.8 deg of mean palm
    # orientation error and 12-14 mm of fingertip residual with every MCP
    # abduction joint pinned at its mechanical limit, while authorizing them
    # gives 1.15 deg and 0.36 mm with the same object trajectory to the bit.
    # They stay authorizable on their own so the legacy fixed-base transform
    # remains reproducible.
    shared_positioning_joints = (
        {"Lift", "torso_flip"}
        if (shared_positioning_axes or phase_aware_object_placement or contact_seeking)
        else set()
    )
    arm_joint_names = tuple(
        name
        for name in joint_names
        if name in shared_positioning_joints
        or name.startswith("L_arm_")
        or name.startswith("R_arm_")
    )
    left_finger_names = tuple(name for name in joint_names if name.startswith("l_"))
    right_finger_names = tuple(name for name in joint_names if name.startswith("r_"))
    retargeted_200hz = MujocoDualHandRetargeter(
        model,
        target_joint_names=joint_names,
        arm_joint_names=arm_joint_names,
        left_finger_joint_names=left_finger_names,
        right_finger_joint_names=right_finger_names,
        left_wrist_site="left_palm",
        right_wrist_site="right_palm",
        iterations=200,
        damping=DEFAULT_IK_DAMPING,
        max_step=0.08,
        tolerance=1.0e-5,
        position_weight=1.0,
        # Pink FrameTask costs are amplitudes and are squared in its QP.
        # MujocoDualHandRetargeter accepts the already-quadratic objective
        # coefficient, so preserve the colleague's 0.03 cost as 0.03**2.
        orientation_weight=float(wrist_orientation_weight) ** 2,
        previous_posture_weight=float(arm_previous_posture_weight),
        neutral_posture_weight=float(arm_neutral_posture_weight),
        position_priority=bool(position_priority_wrist_ik),
        position_priority_slack=float(position_priority_slack_m),
        initial_qpos=initial_retarget_qpos,
    ).retarget(source_trajectory_200hz)
    qpos_200hz_raw = np.asarray(retargeted_200hz.observations["qpos"], dtype=np.float64)
    # The source wrist/object trajectories were already zero-phase filtered
    # before sequential IK. Filtering the solved joint trajectory a second
    # time invalidates the task-space solution and creates centimetre-scale
    # wrist residuals at contact-phase transitions. Preserve the warm-started
    # IK output and sample it at the exact 20 Hz source timestamps instead.
    qpos_200hz = qpos_200hz_raw.copy()
    lower = np.asarray([limits[name][0] for name in joint_names], dtype=np.float64)
    upper = np.asarray([limits[name][1] for name in joint_names], dtype=np.float64)
    qpos_200hz = np.clip(qpos_200hz, lower, upper)
    soma_hand_keypoint_projection_report: dict[str, Any] | None = None
    if soma_fingertip_targets_20hz is not None:
        soma_hand_keypoint_projection_report = {
            "target_construction": soma_hand_target_report,
            "sides": {},
        }
    if split_wrist_hand_ik and soma_fingertip_targets_200hz is not None:
        # Match the colleague implementation's sequential warm start at the
        # source trajectory rate. Solving only the emitted 20 Hz samples makes
        # ten-frame target jumps look discontinuous and can force the bounded
        # IK onto a high-residual branch.
        for side_index, side in enumerate(("left", "right")):
            finger_prefix = "l_" if side == "left" else "r_"
            variable_joint_names = tuple(
                name for name in joint_names if name.startswith(finger_prefix)
            )
            target_site_names = WUJI_FINGERTIP_SITE_NAMES[side]
            target_positions = soma_fingertip_targets_200hz[:, side_index]
            source_frame_change_rad = _source_frame_change_bound(
                source_fps=source_fps,
                target_fps=TARGET_FPS,
                motion_time_scale=motion_time_scale,
                target_frame_change_rad=float(np.radians(34.99)),
            )
            finger_config = FingertipIkConfig(
                iterations=100,
                damping=0.03,
                max_step_rad=0.15,
                # The audit applies the 35 degree limit between emitted 20 Hz
                # Reference frames. Ten 200 Hz changes can otherwise add up
                # to a much larger emitted-frame jump. This source-rate bound
                # makes linear sampling preserve the emitted-frame limit.
                max_frame_change_rad=source_frame_change_rad,
                tolerance_m=1.0e-4,
                posture_cost=float(finger_posture_prior),
            )
            qpos_200hz, report = solve_fingertip_trajectory(
                model,
                trajectory_joint_names=joint_names,
                qpos=qpos_200hz,
                variable_joint_names=variable_joint_names,
                target_site_names=target_site_names,
                target_positions=target_positions,
                config=finger_config,
            )
            side_report = {
                **report.as_dict(),
                "method": (
                    "source-rate sequential tips-only bounded DLS with "
                    "slight-curl prior"
                ),
                "solve_rate_hz": float(source_fps),
                "source_frame_change_limit_deg": float(
                    np.degrees(source_frame_change_rad)
                ),
                "emitted_frame_change_limit_deg": 34.99,
                "variable_joint_names": list(variable_joint_names),
                "target_site_names": list(target_site_names),
                "tip_costs": list(finger_config.tip_weights),
                "posture_cost": finger_config.posture_cost,
                "source_to_robot_scale": float(soma_hand_scale),
            }
            assert soma_hand_keypoint_projection_report is not None
            soma_hand_keypoint_projection_report["sides"][side] = side_report

    qpos = _resample_linear(qpos_200hz, source_times, target_times)
    wrist_ik_qpos = qpos.copy()
    finger_joint_names = (*left_finger_names, *right_finger_names)
    if soma_fingertip_targets_20hz is not None and not split_wrist_hand_ik:
        assert soma_hand_keypoint_projection_report is not None
        for side_index, side in enumerate(("left", "right")):
            finger_prefix = "l_" if side == "left" else "r_"
            arm_prefix = "L_arm_" if side == "left" else "R_arm_"
            variable_joint_names = tuple(
                name
                for name in joint_names
                if name.startswith((arm_prefix, finger_prefix))
            )
            target_site_names = (
                f"{side}_palm",
                *WUJI_FINGERTIP_SITE_NAMES[side],
            )
            target_positions = np.concatenate(
                (
                    target_wrist_position_20hz[:, side_index, None, :],
                    soma_fingertip_targets_20hz[:, side_index],
                ),
                axis=1,
            )
            task_weights = (25.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            projector = MujocoKeypointProjector(
                model,
                trajectory_joint_names=joint_names,
                variable_joint_names=variable_joint_names,
                target_site_names=target_site_names,
                task_weights=task_weights,
                config=MujocoKeypointProjectionConfig(
                    iterations=240,
                    damping=0.03,
                    regularization=0.001,
                    max_step=0.08,
                    tolerance_m=0.002,
                    line_search_steps=8,
                ),
            )
            qpos, report = projector.project(
                qpos,
                target_positions=target_positions,
            )
            side_report = {
                **report.as_dict(),
                "method": "legacy_joint_arm_finger_keypoint_projection",
                "variable_joint_names": list(variable_joint_names),
                "target_site_names": list(target_site_names),
            }
            soma_hand_keypoint_projection_report["sides"][side] = side_report
    object_pose_model = np.concatenate(
        (object_position_20hz, object_wxyz_20hz), axis=-1
    )
    collision_auditor = _SceneCollisionAuditor(
        robot_model_path=model_file,
        object_mesh_path=object_mesh,
        object_scale=local_geometry_scale,
        support_center=projection.support_center_target(),
        support_radius=pin.support_radius_source * local_geometry_scale,
        support_height=pin.support_height_source * local_geometry_scale,
        joint_names=joint_names,
        required_clearance_m=geometry_clearance_m,
        gate_scenes=() if contact_seeking else None,
    )
    rendered_geometry_auditor = _SceneCollisionAuditor(
        robot_model_path=model_file,
        object_mesh_path=object_mesh,
        object_scale=local_geometry_scale,
        support_center=projection.support_center_target(),
        support_radius=pin.support_radius_source * local_geometry_scale,
        support_height=pin.support_height_source * local_geometry_scale,
        joint_names=joint_names,
        required_clearance_m=geometry_clearance_m,
        include_visual_geometry=True,
        gate_scenes=() if contact_seeking else None,
    )
    contact_target_ik_report: dict[str, Any] | None = None
    if contact_target_ik:
        assert chord_active_200hz is not None
        assert chord_targets_200hz is not None
        assert chord_active_20hz is not None
        assert chord_targets_20hz is not None
        contact_geoms = _contact_geom_names_for_bodies(
            collision_auditor.model, WUJI_CHORD_CONTACT_BODIES
        )
        contact_solver = MujocoContactTargetRetargeter(
            collision_auditor.model,
            trajectory_joint_names=joint_names,
            variable_joint_names={
                "left": tuple(name for name in joint_names if name.startswith("l_")),
                "right": tuple(name for name in joint_names if name.startswith("r_")),
            },
            contact_body_names=WUJI_CHORD_CONTACT_BODIES,
            contact_geom_names=contact_geoms,
            object_geom_name="audit_can_geom",
            object_mocap_body_name="audit_can_body",
            config=ContactTargetIKConfig(
                iterations=80,
                damping=0.03,
                regularization=0.005,
                max_step_rad=0.08,
                tolerance_m=0.001,
            ),
        )
        qpos, report = contact_solver.retarget(
            qpos,
            object_poses_wxyz=object_pose_model[:, 0],
            target_positions=chord_targets_20hz,
            active=chord_active_20hz,
        )
        contact_target_ik_report = report.as_dict()
    finger_ik_report: dict[str, Any] | None = None
    if contact_seeking and contact_guided_finger_ik:
        emitted_active_20hz = contact_active_20hz.T
        qpos, finger_ik_report = _contact_guided_finger_ik(
            collision_auditor.model,
            collision_auditor.data,
            qpos,
            joint_names=joint_names,
            object_poses=object_pose_model,
            source_active=emitted_active_20hz,
        )
    # Physical correction acts only on the emitted Reference samples.  The
    # 200 Hz retarget remains useful diagnostic evidence, but independently
    # correcting it cannot change what the runtime emits and previously made
    # qualification both expensive and conceptually ambiguous.
    combined_qpos = qpos.copy()
    combined_object_poses = object_pose_model
    try:
        combined_qpos, finger_closure_scales, projection_search_audit = (
            _collision_safe_finger_projection(
                combined_qpos,
                joint_names=joint_names,
                object_poses=combined_object_poses,
                auditor=collision_auditor,
            )
        )
    except ValueError as exc:
        if not inspection_only:
            raise
        # An inspection conversion must retain the exact failed trajectory so
        # the later final-FK and geometry audits can diagnose it. Never apply a
        # partial correction or let this branch qualify for runtime use.
        combined_qpos = qpos.copy()
        finger_closure_scales = {"left": 1.0, "right": 1.0}
        projection_search_audit = {
            "qualified": False,
            "status": "blocked",
            "reason": str(exc),
            "inspection_trajectory_retained_unmodified": True,
        }
    settling_report: dict[str, Any] | None = None
    support_projection_report: dict[str, Any] | None = None
    self_collision_projection_report: dict[str, Any] | None = None
    contact_constraint_projection_report: dict[str, Any] | None = None
    contact_scene_projection_report: dict[str, Any] | None = None
    strict_final_finger_tuck_report: dict[str, Any] | None = None
    if contact_seeking:
        # DexMachina's functional retargeting: replay the retarget as soft
        # position targets while the scene is held fixed, so contact pushes the
        # hand onto the surface instead of a standoff pushing it away.
        from iltools.retarget.contact_settling import (
            ContactSettlingConfig,
            settle_contact_trajectory,
        )

        settle_model = collision_auditor.model
        settle_data = collision_auditor.data
        # Only genuinely named geoms; _geom_label synthesises a label for
        # unnamed ones, which is not resolvable. This list only drives the
        # penetration report. MuJoCo resolves every contact during the step
        # regardless of what is listed here.
        robot_geom_names = [
            name
            for name in (
                settle_model.geom(geom_id).name
                for geom_id in collision_auditor._robot_geom_ids
            )
            if name
        ]
        if split_wrist_hand_ik:
            settling_report = {
                "status": "skipped",
                "reason": (
                    "Split wrist/hand IK locks every arm and shared positioning "
                    "joint after wrist IK; physics settling would move those joints."
                ),
            }
        elif chord_active_20hz is None:
            settled_qpos, report = settle_contact_trajectory(
                settle_model,
                settle_data,
                qpos=combined_qpos,
                joint_names=joint_names,
                object_geom_names=["audit_can_geom"],
                robot_geom_names=[
                    name for name in robot_geom_names if not name.endswith(":geom_-1")
                ],
                object_mocap_poses=combined_object_poses,
                object_mocap_body_names=["audit_can_body"],
                config=ContactSettlingConfig(settle_steps=int(contact_settle_steps)),
            )
            combined_qpos = settled_qpos
            settling_report = report.as_dict()
        else:
            settling_report = {
                "status": "skipped",
                "reason": (
                    "SOMA/CHORD uses explicit semantic contact, radial escape, "
                    "and typed collision projections; independent per-frame "
                    "physics settling introduces discontinuous IK branches."
                ),
            }

        # The can is an intentional contact target; the support is not.  Soft
        # replay resolves most table contact, then this constrained projection
        # removes any residual support penetration while preserving both palm
        # positions and remaining close to the settled pose.
        support_projector = MujocoCollisionProjector(
            collision_auditor.model,
            trajectory_joint_names=joint_names,
            variable_joint_names=(
                finger_joint_names if split_wrist_hand_ik else joint_names
            ),
            robot_geom_names=collision_auditor._robot_geom_ids,
            scene_geom_names=("audit_support_geom",),
            preserve_site_names=("left_palm", "right_palm"),
            config=CollisionProjectionConfig(
                # Keep a 0.1 mm numerical margin because the independent
                # audit gates the exact requested threshold fail-closed.
                clearance_m=float(geometry_clearance_m) + 0.0001,
                iterations=40,
                damping=0.02,
                regularization=0.01,
                collision_weight=10.0,
                preserve_site_weight=1.0,
                max_step=0.08,
                tolerance_m=1.0e-6,
            ),
        )
        combined_qpos, report = support_projector.project(combined_qpos)
        support_projection_report = report.as_dict()

        closure_joint_names = tuple(
            name for name in joint_names if name.startswith(("l_", "r_"))
        )
        self_collision_projector = MujocoSelfCollisionClosureProjector(
            collision_auditor.model,
            trajectory_joint_names=joint_names,
            closure_joint_names=closure_joint_names,
            robot_geom_names=collision_auditor._robot_geom_ids,
            config=SelfCollisionProjectionConfig(
                penetration_tolerance_m=0.001,
                search_iterations=20,
            ),
        )
        combined_qpos, _, report = self_collision_projector.project(combined_qpos)
        self_collision_projection_report = report.as_dict()

        residual_pairs_by_frame = _robot_self_violation_pairs_by_frame(
            collision_auditor,
            combined_qpos,
            penetration_tolerance_m=0.001,
        )
        residual_pairs = tuple(
            sorted(
                {
                    pair
                    for frame_pairs in residual_pairs_by_frame
                    for pair in frame_pairs
                }
            )
        )
        if residual_pairs:
            exact_self_projector = MujocoCollisionProjector(
                collision_auditor.model,
                trajectory_joint_names=joint_names,
                variable_joint_names=(
                    finger_joint_names if split_wrist_hand_ik else joint_names
                ),
                forbidden_geom_pairs=residual_pairs,
                preserve_site_names=("left_palm", "right_palm"),
                config=CollisionProjectionConfig(
                    clearance_m=0.0005,
                    iterations=30,
                    damping=0.02,
                    regularization=0.01,
                    collision_weight=10.0,
                    preserve_site_weight=0.2,
                    max_step=0.05,
                    tolerance_m=1.0e-5,
                ),
            )
            combined_qpos, report = exact_self_projector.project(
                combined_qpos,
                active_geom_pairs_by_frame=residual_pairs_by_frame,
            )
            self_collision_projection_report["exact_residual_projection"] = (
                report.as_dict()
            )
            combined_qpos, _, report = self_collision_projector.project(combined_qpos)
            self_collision_projection_report["post_exact_closure_projection"] = (
                report.as_dict()
            )
        combined_qpos, report = support_projector.project(combined_qpos)
        support_projection_report["post_self_collision"] = report.as_dict()

        if chord_active_200hz is not None and chord_active_20hz is not None:
            # CHORD's dense source graph is frequently over-constrained for a
            # different robot morphology.  Preserve its semantic lower bound:
            # one source-active mapped link per hand must reach a shallow
            # contact band.  Scene projection is side-local; a left collision
            # must never be "fixed" by moving the right arm.
            combined_chord_active = chord_active_20hz
            contact_geoms = _contact_geom_names_for_bodies(
                collision_auditor.model, WUJI_CHORD_CONTACT_BODIES
            )
            side_variables = {
                side: (
                    tuple(
                        name
                        for name in joint_names
                        if name.startswith("l_" if side == "left" else "r_")
                    )
                    if split_wrist_hand_ik
                    else tuple(
                        name
                        for name in joint_names
                        if name.startswith(
                            (
                                "L_arm_" if side == "left" else "R_arm_",
                                "l_" if side == "left" else "r_",
                            )
                        )
                    )
                )
                for side in ("left", "right")
            }
            contact_variables = {
                side: tuple(
                    name
                    for name in joint_names
                    if name.startswith("l_" if side == "left" else "r_")
                )
                for side in ("left", "right")
            }
            contact_projector = MujocoContactConstraintProjector(
                collision_auditor.model,
                trajectory_joint_names=joint_names,
                variable_joint_names=contact_variables,
                contact_geom_names=contact_geoms,
                object_geom_name="audit_can_geom",
                object_mocap_body_name="audit_can_body",
                config=ContactConstraintProjectionConfig(
                    target_distance_m=0.0005,
                    contact_tolerance_m=0.002,
                    max_penetration_m=0.001,
                    iterations=160,
                    damping=0.01,
                    max_step=0.03,
                    distance_tolerance_m=0.00015,
                    query_cutoff_m=0.15,
                    line_search_steps=8,
                ),
            )

            side_scene_projectors: dict[str, tuple[Any, ...]] = {}
            for side, body_prefix, arm_prefix in (
                ("left", "l_", "L_arm_"),
                ("right", "r_", "R_arm_"),
            ):
                side_geoms: list[int] = []
                finger_geoms: list[int] = []
                rigid_geoms: list[int] = []
                for geom_id in collision_auditor._robot_geom_ids:
                    body_name = collision_auditor.model.body(
                        int(collision_auditor.model.geom_bodyid[geom_id])
                    ).name
                    if not body_name.startswith((body_prefix, arm_prefix)):
                        continue
                    side_geoms.append(int(geom_id))
                    is_finger = body_name.startswith(body_prefix) and body_name not in (
                        f"{body_prefix}mount",
                        f"{body_prefix}wrist",
                    )
                    (finger_geoms if is_finger else rigid_geoms).append(int(geom_id))
                arm_variables = tuple(
                    name for name in joint_names if name.startswith(arm_prefix)
                )
                scene_variables = side_variables[side]
                if split_wrist_hand_ik:
                    radial_projector = None
                    rigid_projector = None
                    hand_robot_geoms = finger_geoms
                    support_robot_geoms = finger_geoms
                else:
                    radial_projector = MujocoRadialEscapeProjector(
                        collision_auditor.model,
                        trajectory_joint_names=joint_names,
                        variable_joint_names=arm_variables,
                        robot_geom_names=finger_geoms,
                        scene_geom_name="audit_can_geom",
                        target_site_name=f"{side}_palm",
                        scene_mocap_body_name="audit_can_body",
                        config=RadialEscapeProjectionConfig(
                            clearance_m=-0.0005,
                            tolerance_m=1.0e-5,
                            retreat_step_m=0.005,
                            maximum_retreat_m=0.10,
                            ik_iterations=100,
                            damping=0.01,
                            max_step=0.05,
                            position_tolerance_m=1.0e-5,
                            query_margin_m=0.15,
                        ),
                    )
                    rigid_projector = MujocoCollisionProjector(
                        collision_auditor.model,
                        trajectory_joint_names=joint_names,
                        variable_joint_names=arm_variables,
                        robot_geom_names=rigid_geoms,
                        scene_geom_names=("audit_can_geom",),
                        scene_mocap_body_names=("audit_can_body",),
                        config=CollisionProjectionConfig(
                            clearance_m=0.0002,
                            iterations=240,
                            damping=0.01,
                            regularization=0.0001,
                            collision_weight=50.0,
                            preserve_site_weight=0.001,
                            max_step=0.04,
                            tolerance_m=1.0e-5,
                            query_margin_m=0.15,
                        ),
                    )
                    hand_robot_geoms = finger_geoms
                    support_robot_geoms = side_geoms
                side_scene_projectors[side] = (
                    MujocoSceneCollisionClosureProjector(
                        collision_auditor.model,
                        trajectory_joint_names=joint_names,
                        closure_joint_names=contact_variables[side],
                        robot_geom_names=finger_geoms,
                        scene_geom_name="audit_can_geom",
                        scene_mocap_body_name="audit_can_body",
                        config=SceneCollisionClosureProjectionConfig(
                            clearance_m=-0.0005,
                            tolerance_m=1.0e-5,
                            search_iterations=24,
                            query_margin_m=0.15,
                        ),
                    ),
                    radial_projector,
                    rigid_projector,
                    MujocoCollisionProjector(
                        collision_auditor.model,
                        trajectory_joint_names=joint_names,
                        variable_joint_names=scene_variables,
                        robot_geom_names=hand_robot_geoms,
                        scene_geom_names=("audit_can_geom",),
                        scene_mocap_body_names=("audit_can_body",),
                        config=CollisionProjectionConfig(
                            clearance_m=-0.0005,
                            iterations=240,
                            damping=0.01,
                            regularization=0.0001,
                            collision_weight=50.0,
                            preserve_site_weight=0.001,
                            max_step=0.04,
                            tolerance_m=1.0e-5,
                            query_margin_m=0.15,
                        ),
                    ),
                    MujocoCollisionProjector(
                        collision_auditor.model,
                        trajectory_joint_names=joint_names,
                        variable_joint_names=scene_variables,
                        robot_geom_names=support_robot_geoms,
                        scene_geom_names=("audit_support_geom",),
                        config=CollisionProjectionConfig(
                            clearance_m=float(geometry_clearance_m) + 0.0001,
                            iterations=240,
                            damping=0.01,
                            regularization=0.0001,
                            collision_weight=50.0,
                            preserve_site_weight=0.001,
                            max_step=0.04,
                            tolerance_m=1.0e-5,
                            query_margin_m=0.03,
                        ),
                    ),
                )

            scene_cycle_reports: list[dict[str, Any]] = []

            def project_side_scene(values: np.ndarray) -> np.ndarray:
                cycle: dict[str, Any] = {}
                projected = values
                for side in ("left", "right"):
                    (
                        closure_projector,
                        radial_projector,
                        rigid_projector,
                        hand_projector,
                        table_projector,
                    ) = side_scene_projectors[side]
                    projected, _, closure_report = closure_projector.project(
                        projected, scene_mocap_poses=combined_object_poses
                    )
                    if radial_projector is None:
                        radial_report_data = {
                            "status": "skipped",
                            "reason": "arm joints are locked after wrist IK",
                        }
                    else:
                        projected, radial_report = radial_projector.project(
                            projected, scene_mocap_poses=combined_object_poses
                        )
                        radial_report_data = radial_report.as_dict()
                    if rigid_projector is None:
                        rigid_report_data = {
                            "status": "audit_only",
                            "reason": "rigid arm/wrist geometry cannot move after wrist IK",
                        }
                    else:
                        projected, rigid_report = rigid_projector.project(
                            projected, scene_mocap_poses=combined_object_poses
                        )
                        rigid_report_data = rigid_report.as_dict()
                    projected, hand_report = hand_projector.project(
                        projected, scene_mocap_poses=combined_object_poses
                    )
                    projected, table_report = table_projector.project(projected)
                    cycle[side] = {
                        "finger_can_closure": closure_report.as_dict(),
                        "radial_hand_can_escape": radial_report_data,
                        "rigid_can": rigid_report_data,
                        "hand_can": hand_report.as_dict(),
                        "support": table_report.as_dict(),
                    }
                scene_cycle_reports.append(cycle)
                return projected

            combined_qpos = project_side_scene(combined_qpos)
            attraction_reports: list[dict[str, Any]] = []
            exact_self_reports: list[dict[str, Any]] = []
            closure_self_reports: list[dict[str, Any]] = []
            exact_self_variables = tuple(
                name
                for name in joint_names
                if (
                    name.startswith(("l_", "r_"))
                    if split_wrist_hand_ik
                    else name.startswith(("L_arm_", "R_arm_", "l_", "r_"))
                )
            )
            # Alternating projections converge on the intersection of the
            # semantic-contact, scene-clearance, and self-collision sets.
            for _ in range(2):
                combined_qpos, attraction_report = contact_projector.project(
                    combined_qpos,
                    object_poses_wxyz=combined_object_poses[:, 0],
                    active=combined_chord_active,
                )
                attraction_reports.append(attraction_report.as_dict())
                combined_qpos = project_side_scene(combined_qpos)
                frame_pairs = _robot_self_violation_pairs_by_frame(
                    collision_auditor,
                    combined_qpos,
                    penetration_tolerance_m=0.001,
                )
                exact_pairs = tuple(
                    sorted({pair for pairs in frame_pairs for pair in pairs})
                )
                if exact_pairs:
                    exact_projector = MujocoCollisionProjector(
                        collision_auditor.model,
                        trajectory_joint_names=joint_names,
                        variable_joint_names=exact_self_variables,
                        forbidden_geom_pairs=exact_pairs,
                        preserve_site_names=("left_palm", "right_palm"),
                        config=CollisionProjectionConfig(
                            clearance_m=0.0005,
                            iterations=100,
                            damping=0.01,
                            regularization=0.0001,
                            collision_weight=30.0,
                            preserve_site_weight=0.01,
                            max_step=0.04,
                            tolerance_m=1.0e-5,
                            query_margin_m=0.03,
                        ),
                    )
                    combined_qpos, exact_report = exact_projector.project(
                        combined_qpos,
                        active_geom_pairs_by_frame=frame_pairs,
                    )
                    exact_self_reports.append(exact_report.as_dict())
                    combined_qpos, _, closure_report = self_collision_projector.project(
                        combined_qpos
                    )
                    closure_self_reports.append(closure_report.as_dict())
                    combined_qpos = project_side_scene(combined_qpos)
            # Each alternating cycle ends with a side-local scene projection,
            # which can reintroduce a two-hand collision during the handover.
            # Finish with finger-only self closure: it preserves both palms
            # exactly and only opens the minimum amount needed for the robot
            # self-contact tolerance.  The independent typed can audit below
            # remains fail-closed if that final opening harms scene clearance.
            combined_qpos, _, final_closure_report = self_collision_projector.project(
                combined_qpos
            )
            closure_self_reports.append(
                {
                    "stage": "final_after_scene_projection",
                    **final_closure_report.as_dict(),
                }
            )
            final_frame_pairs = _robot_self_violation_pairs_by_frame(
                collision_auditor,
                combined_qpos,
                penetration_tolerance_m=0.001,
            )
            final_exact_pairs = tuple(
                sorted({pair for pairs in final_frame_pairs for pair in pairs})
            )
            if final_exact_pairs:
                final_exact_projector = MujocoCollisionProjector(
                    collision_auditor.model,
                    trajectory_joint_names=joint_names,
                    variable_joint_names=exact_self_variables,
                    forbidden_geom_pairs=final_exact_pairs,
                    preserve_site_names=("left_palm", "right_palm"),
                    config=CollisionProjectionConfig(
                        clearance_m=0.0005,
                        iterations=240,
                        damping=0.01,
                        regularization=0.0001,
                        collision_weight=100.0,
                        preserve_site_weight=0.1,
                        max_step=0.03,
                        tolerance_m=1.0e-5,
                        query_margin_m=0.03,
                    ),
                )
                combined_qpos, final_exact_report = final_exact_projector.project(
                    combined_qpos,
                    active_geom_pairs_by_frame=final_frame_pairs,
                )
                exact_self_reports.append(
                    {
                        "stage": "final_after_finger_self_closure",
                        **final_exact_report.as_dict(),
                    }
                )
            contact_constraint_projection_report = {
                "attraction_cycles": attraction_reports,
                "exact_self_collision_cycles": exact_self_reports,
                "closure_self_collision_cycles": closure_self_reports,
            }
            contact_scene_projection_report = {
                "method": (
                    "finger-only side-local alternating can/support projection; "
                    "wrist-IK joints locked"
                    if split_wrist_hand_ik
                    else "side-local alternating can/support projection"
                ),
                "cycles": scene_cycle_reports,
            }

    if split_wrist_hand_ik and chord_active_20hz is not None:
        combined_qpos, strict_final_finger_tuck_report = (
            _project_intended_hand_penetration_to_tucked_pose(
                combined_qpos,
                joint_names=joint_names,
                joint_limits=limits,
                object_poses=combined_object_poses,
                auditor=collision_auditor,
                maximum_penetration_m=0.001,
            )
        )
    wrist_hand_partition_report: dict[str, Any] | None = None
    if split_wrist_hand_ik:
        wrist_hand_partition_report = _verify_locked_joint_trajectory(
            wrist_ik_qpos,
            combined_qpos,
            joint_names=joint_names,
            locked_joint_names=arm_joint_names,
        )
    qpos = combined_qpos
    # The contact and collision repair stages shape the whole hand against
    # object-surface targets, which bulk-shifts the fingertips the tips-only
    # solve had already placed on their contact points: measured on the
    # snack-box clip, 0.1-4.2 mm at the IK stage became 9-25 mm here, uniform
    # across all five fingers. This pass puts them back with the same bounded,
    # limit-aware solver, seeded from the repaired pose so it stays in that
    # basin. It runs BEFORE every geometry audit, so nothing it changes
    # escapes re-verification. It is deliberately placed AFTER the penetration
    # tuck: running it before instead leaves the hand deep enough in the object
    # that the tuck's multistart search finds no feasible left-finger sample at
    # all. Both orderings were measured; see the wiki.
    fingertip_reconciliation_report: dict[str, Any] | None = None
    if reconcile_fingertips_after_contact and soma_fingertip_targets_20hz is not None:
        fingertip_reconciliation_report = {
            "method": (
                "bounded tips-only re-solve toward the fingertip targets after "
                "contact and collision repair, seeded from the repaired pose; "
                "re-audited by every geometry gate below"
            ),
            "sides": {},
        }
        for side_index, side in enumerate(("left", "right")):
            prefix = "l_" if side == "left" else "r_"
            side_joint_names = tuple(
                name for name in joint_names if name.startswith(prefix)
            )
            # Fingers of one hand foul each other once the tips are pulled
            # back onto their contact points, and jammed fingers cannot
            # reach. Those pairs do not exist in the tucked pose, so they can
            # only be discovered from a first solve: round one finds them,
            # round two carries them in the same task as the object clearance.
            extra_self_pairs: list[tuple[int, int]] = []
            before = measure_fingertip_residuals(
                model,
                trajectory_joint_names=joint_names,
                qpos=qpos,
                target_site_names=WUJI_FINGERTIP_SITE_NAMES[side],
                target_positions=soma_fingertip_targets_20hz[:, side_index],
            ).as_dict()
            # Solve against the audit model, which carries the object, so the
            # clearance task can see it. The robot joints and fingertip sites
            # are identical in both models.
            avoidance = None
            if reconciliation_clearance_m is not None:
                hand_geoms = _contact_geom_names_for_bodies(
                    collision_auditor.model, WUJI_CHORD_CONTACT_BODIES
                )[side]
                pairs = [(geom, "audit_can_geom") for geom in hand_geoms]
                pairs.extend(extra_self_pairs)
                avoidance = FingertipAvoidanceConfig(
                    geom_pairs=tuple(pairs),
                    clearance_m=float(reconciliation_clearance_m),
                    weight=25.0,
                    query_cutoff_m=0.05,
                    mocap_body_name="audit_can_body",
                )
            qpos, solve_report = solve_fingertip_trajectory(
                collision_auditor.model if avoidance is not None else model,
                trajectory_joint_names=joint_names,
                qpos=qpos,
                variable_joint_names=side_joint_names,
                target_site_names=WUJI_FINGERTIP_SITE_NAMES[side],
                target_positions=soma_fingertip_targets_20hz[:, side_index],
                config=FingertipIkConfig(
                    iterations=100,
                    damping=0.03,
                    max_step_rad=0.15,
                    max_frame_change_rad=float(np.radians(34.99)),
                    tolerance_m=1.0e-4,
                    posture_cost=float(finger_posture_prior),
                ),
                avoidance=avoidance,
                scene_object_poses=(
                    object_pose_model[:, 0] if avoidance is not None else None
                ),
            )
            if avoidance is not None:
                # The audited geoms are unnamed, so pair them by id and select
                # this hand by the body each geom belongs to.
                side_bodies = {
                    int(collision_auditor.model.body(name).id)
                    for name in WUJI_CHORD_CONTACT_BODIES[side]
                }
                discovered: set[tuple[int, int]] = set()
                for frame_pairs in _robot_self_violation_pairs_by_frame(
                    collision_auditor, qpos, penetration_tolerance_m=0.001
                ):
                    for first, second in frame_pairs:
                        bodies = [
                            int(collision_auditor.model.geom_bodyid[geom])
                            for geom in (first, second)
                        ]
                        if all(body in side_bodies for body in bodies):
                            discovered.add(
                                (int(min(first, second)), int(max(first, second)))
                            )
                if discovered:
                    extra_self_pairs = sorted(discovered)
                    qpos, solve_report = solve_fingertip_trajectory(
                        collision_auditor.model,
                        trajectory_joint_names=joint_names,
                        qpos=qpos,
                        variable_joint_names=side_joint_names,
                        target_site_names=WUJI_FINGERTIP_SITE_NAMES[side],
                        target_positions=soma_fingertip_targets_20hz[:, side_index],
                        config=FingertipIkConfig(
                            iterations=100,
                            damping=0.03,
                            max_step_rad=0.15,
                            max_frame_change_rad=float(np.radians(34.99)),
                            tolerance_m=1.0e-4,
                            posture_cost=float(finger_posture_prior),
                        ),
                        avoidance=FingertipAvoidanceConfig(
                            geom_pairs=tuple(
                                [(geom, "audit_can_geom") for geom in hand_geoms]
                                + extra_self_pairs
                            ),
                            clearance_m=float(reconciliation_clearance_m),
                            weight=25.0,
                            query_cutoff_m=0.05,
                            mocap_body_name="audit_can_body",
                        ),
                        scene_object_poses=object_pose_model[:, 0],
                    )
            fingertip_reconciliation_report["sides"][side] = {
                "self_contact_pairs_resolved": len(extra_self_pairs),
                "tip_p95_mm_before": float(before["tip_p95_mm"]),
                "tip_p95_mm_after": float(solve_report.as_dict()["tip_p95_mm"]),
                "object_clearance_task_m": (
                    None
                    if reconciliation_clearance_m is None
                    else float(reconciliation_clearance_m)
                ),
            }

    qualified_state_hold_frames = int(qualified_state_hold_frames)
    emitted_frame_count_before_hold = len(qpos)
    if qualified_state_hold_frames > 1:
        # Every post-projection state has already passed the contact/collision
        # intersection.  A zero-order hold slows command transitions without
        # inventing the penetrative intermediate poses produced by smoothing.
        hold_indices = discrete_time_stretch_indices(
            len(qpos),
            hold_frames=qualified_state_hold_frames,
        )
        qpos = qpos[hold_indices]
        object_pose_model = object_pose_model[hold_indices]
        target_wrist_position_20hz = target_wrist_position_20hz[hold_indices]
        target_wrist_wxyz_20hz = target_wrist_wxyz_20hz[hold_indices]
        contact_active_20hz = contact_active_20hz[:, hold_indices]
        contact_blend_20hz = contact_blend_20hz[:, hold_indices]
        if soma_fingertip_targets_20hz is not None:
            soma_fingertip_targets_20hz = soma_fingertip_targets_20hz[hold_indices]
        if chord_active_20hz is not None:
            chord_active_20hz = chord_active_20hz[hold_indices]
        if chord_targets_20hz is not None:
            chord_targets_20hz = chord_targets_20hz[hold_indices]
        if chord_normals_20hz is not None:
            chord_normals_20hz = chord_normals_20hz[hold_indices]
    source_dense_qpos, source_dense_object_poses = _densify_scene_trajectory(
        qpos_200hz,
        object_pose_200hz,
        source_fps=source_fps,
        audit_fps=dense_geometry_audit_fps,
    )
    emitted_dense_qpos, emitted_dense_object_poses = _densify_scene_trajectory(
        qpos,
        object_pose_model,
        source_fps=TARGET_FPS,
        audit_fps=dense_geometry_audit_fps,
    )
    collision_audit_200hz = collision_auditor.audit(qpos_200hz, object_pose_200hz)
    collision_audit_20hz = collision_auditor.audit(qpos, object_pose_model)
    collision_audit_source_dense = collision_auditor.audit(
        source_dense_qpos, source_dense_object_poses
    )
    collision_audit_emitted_dense = collision_auditor.audit(
        emitted_dense_qpos, emitted_dense_object_poses
    )
    rendered_audit_20hz = rendered_geometry_auditor.audit(qpos, object_pose_model)
    rendered_audit_source_dense = rendered_geometry_auditor.audit(
        source_dense_qpos, source_dense_object_poses
    )
    rendered_audit_emitted_dense = rendered_geometry_auditor.audit(
        emitted_dense_qpos, emitted_dense_object_poses
    )
    can_support_source_dense = _audit_can_support_vertical_clearance(
        source_dense_object_poses,
        object_mesh=object_mesh,
        object_scale=local_geometry_scale,
        support_top_z=projection.support_top_z(),
        support_center_xy=projection.support_center_target()[:2],
        support_radius_m=pin.support_radius_source * local_geometry_scale,
    )
    can_support_emitted_dense = _audit_can_support_vertical_clearance(
        emitted_dense_object_poses,
        object_mesh=object_mesh,
        object_scale=local_geometry_scale,
        support_top_z=projection.support_top_z(),
        support_center_xy=projection.support_center_target()[:2],
        support_radius_m=pin.support_radius_source * local_geometry_scale,
    )
    can_support_20hz = _audit_can_support_vertical_clearance(
        object_pose_model,
        object_mesh=object_mesh,
        object_scale=local_geometry_scale,
        support_top_z=projection.support_top_z(),
        support_center_xy=projection.support_center_target()[:2],
        support_radius_m=pin.support_radius_source * local_geometry_scale,
    )
    robot_self_collision_20hz = _audit_robot_self_penetration(
        collision_auditor, qpos, penetration_tolerance_m=0.001
    )
    semantic_can_20hz = _audit_semantic_robot_can_clearance(
        collision_auditor,
        qpos,
        object_pose_model,
        maximum_intended_penetration_m=0.001,
    )
    semantic_can_emitted_dense = _audit_semantic_robot_can_clearance(
        collision_auditor,
        emitted_dense_qpos,
        emitted_dense_object_poses,
        maximum_intended_penetration_m=0.001,
    )
    support_clearance_qualified = bool(
        collision_audit_20hz["scenes"]["support"]["violation_frame_count"] == 0
    )
    collision_audit = {
        "qualified": bool(
            semantic_can_20hz["qualified"]
            and can_support_20hz["qualified"]
            and robot_self_collision_20hz["qualified"]
            and support_clearance_qualified
        ),
        "qualification_scope": (
            "authored 20 Hz Reference states; runtime consumes one Reference "
            "frame per control transition without linear joint interpolation"
        ),
        "required_clearance_m": geometry_clearance_m,
        "source_200hz": collision_audit_200hz,
        "emitted_20hz": collision_audit_20hz,
        "source_path_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": source_fps,
            "interpolation": "linear joint/translation and quaternion Slerp",
            "qualification_role": "diagnostic_only_not_runtime_interpolation",
            **collision_audit_source_dense,
        },
        "emitted_path_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": TARGET_FPS,
            "interpolation": "linear joint/translation and quaternion Slerp",
            "qualification_role": "diagnostic_only_not_runtime_interpolation",
            **collision_audit_emitted_dense,
        },
        "rendered_geometry_20hz": rendered_audit_20hz,
        "source_path_rendered_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": source_fps,
            "qualification_role": "diagnostic_only_not_runtime_interpolation",
            **rendered_audit_source_dense,
        },
        "emitted_path_rendered_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": TARGET_FPS,
            "qualification_role": "diagnostic_only_not_runtime_interpolation",
            **rendered_audit_emitted_dense,
        },
        "authored_20hz_can_support": can_support_20hz,
        "source_path_can_support_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": source_fps,
            "qualification_role": "diagnostic_only_not_runtime_interpolation",
            **can_support_source_dense,
        },
        "emitted_path_can_support_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": TARGET_FPS,
            "qualification_role": "diagnostic_only_not_runtime_interpolation",
            **can_support_emitted_dense,
        },
        "robot_self_collision_20hz": robot_self_collision_20hz,
        "semantic_can_20hz": semantic_can_20hz,
        "semantic_can_emitted_path_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": TARGET_FPS,
            "qualification_role": "diagnostic_only_not_runtime_interpolation",
            **semantic_can_emitted_dense,
        },
        "support_clearance_qualified": support_clearance_qualified,
        "contact_settling": settling_report,
        "support_collision_projection": support_projection_report,
        "self_collision_projection": self_collision_projection_report,
        "contact_constraint_projection": contact_constraint_projection_report,
        "contact_scene_projection": contact_scene_projection_report,
        "strict_final_finger_tuck": strict_final_finger_tuck_report,
        "finger_projection_search": projection_search_audit,
        "contact_guided_finger_ik": finger_ik_report,
        "soma_chord_contact_target_ik": contact_target_ik_report,
        "soma_chord_hand_keypoint_projection": (soma_hand_keypoint_projection_report),
    }
    if not collision_audit["qualified"] and not inspection_only:
        raise ValueError(
            f"Collision audit failed after finger projection: {collision_audit}."
        )
    target_wrists = {
        "left": np.concatenate(
            (target_wrist_position_20hz[:, 0], target_wrist_wxyz_20hz[:, 0]),
            axis=-1,
        ),
        "right": np.concatenate(
            (target_wrist_position_20hz[:, 1], target_wrist_wxyz_20hz[:, 1]),
            axis=-1,
        ),
    }
    retargeted = Trajectory(
        observations={
            "qpos": qpos,
            "left_wrist_pose_w": target_wrists["left"],
            "right_wrist_pose_w": target_wrists["right"],
            "object_poses_w": object_pose_model,
        },
        infos=retargeted_200hz.infos,
        dt=1.0 / TARGET_FPS,
    )
    achieved_wrists, fingertips = _forward_kinematics(model, qpos, joint_names)
    position_errors = np.column_stack(
        [
            np.linalg.norm(
                achieved_wrists[side][:, :3] - target_wrists[side][:, :3], axis=-1
            )
            for side in ("left", "right")
        ]
    )
    orientation_errors = np.column_stack(
        [
            _orientation_error(
                target_wrists[side][:, 3:7], achieved_wrists[side][:, 3:7]
            )
            for side in ("left", "right")
        ]
    )
    position_error_p95 = np.percentile(position_errors, 95.0, axis=0)
    if (
        float(np.max(position_error_p95)) > max_wrist_position_error
        and not inspection_only
    ):
        raise ValueError(
            "Vega wrist IK per-side p95 position error exceeds the limit: "
            f"{np.max(position_error_p95):.6f} m > "
            f"{max_wrist_position_error:.6f} m."
        )
    if float(np.max(orientation_errors)) > max_wrist_orientation_error:
        raise ValueError(
            "Vega wrist IK orientation error exceeds the limit: "
            f"{np.max(orientation_errors):.6f} rad > "
            f"{max_wrist_orientation_error:.6f} rad."
        )
    if soma_fingertip_targets_20hz is None:
        raise RuntimeError("The adopted Wuji recipe did not produce fingertip targets.")
    finger_acceptance: dict[str, Any] = {
        "method": (
            "final FK after every finger/contact/collision projection; raw gates "
            "from wuji_retargeting_notes"
        ),
        "required_for_training": True,
        "anatomical_limits_applied_before_ik": bool(anatomical_finger_limits),
        "anatomical_limit_rows": anatomical_finger_limit_rows,
        "source_to_robot_scale": float(soma_hand_scale),
        "posture_cost": float(finger_posture_prior),
        "sides": {},
    }
    for side_index, side in enumerate(("left", "right")):
        prefix = "l_" if side == "left" else "r_"
        columns = [
            index for index, name in enumerate(joint_names) if name.startswith(prefix)
        ]
        side_joint_names = tuple(joint_names[index] for index in columns)
        side_limits = {name: limits[name] for name in side_joint_names}
        joint_audit = audit_finger_joints(
            qpos[:, columns], side_joint_names, side_limits
        )
        residual_report = measure_fingertip_residuals(
            model,
            trajectory_joint_names=joint_names,
            qpos=qpos,
            target_site_names=WUJI_FINGERTIP_SITE_NAMES[side],
            target_positions=soma_fingertip_targets_20hz[:, side_index],
        )
        residuals = residual_report.as_dict()
        gates = evaluate_finger_gates(
            audit=joint_audit,
            tip_p95_mm=float(residuals["tip_p95_mm"]),
            wrist_mean_mm=float(np.mean(position_errors[:, side_index]) * 1e3),
        )
        finger_acceptance["sides"][side] = {
            "joint_audit": joint_audit,
            "fingertip_residuals": residuals,
            "wrist_mean_mm": float(np.mean(position_errors[:, side_index]) * 1e3),
            "gates": gates,
        }
    finger_acceptance["qualified"] = all(
        bool(side_report["gates"]["pass"])
        for side_report in finger_acceptance["sides"].values()
    )
    finger_acceptance["evidence_binding"] = {
        "schema": "isaaclab_imitation_wuji_final_fk_evidence/v1",
        "qpos_float32_sha256": _canonical_float32_array_sha256(qpos),
        "fingertip_targets_float32_sha256": _canonical_float32_array_sha256(
            soma_fingertip_targets_20hz
        ),
        "model_sha256": _sha256_file(model_file),
        "joint_names_sha256": _sha256_bytes(
            json.dumps(list(joint_names), separators=(",", ":")).encode("utf-8")
        ),
    }
    if (require_finger_gates or not inspection_only) and not finger_acceptance[
        "qualified"
    ]:
        failures = {
            side: report["gates"]
            for side, report in finger_acceptance["sides"].items()
            if not report["gates"]["pass"]
        }
        raise ValueError(f"Final Wuji finger acceptance gates failed: {failures}.")
    retargeted.observations["left_wrist_pose_w"] = achieved_wrists["left"]
    retargeted.observations["right_wrist_pose_w"] = achieved_wrists["right"]
    retargeted.observations["object_poses_w"] = object_pose_model

    contact_sequence = _inspection_contacts(len(qpos))
    contact_recovery_report: dict[str, Any] | None = None
    if contact_seeking:
        from iltools.retarget.contact_recovery import (
            ContactRecoveryConfig,
            recover_contact_sequence,
        )

        try:
            if chord_active_20hz is not None:
                recovery_geom_names = _contact_geom_names_for_bodies(
                    collision_auditor.model, WUJI_CHORD_CONTACT_BODIES
                )
                recovery_link_names = WUJI_CHORD_CONTACT_BODIES
                recovery_side_active = np.any(chord_active_20hz, axis=-1)
            else:
                recovery_geom_names = _fingertip_contact_geom_names(
                    collision_auditor.model
                )
                recovery_link_names = FINGERTIP_NAMES
                recovery_side_active = contact_active_20hz.T
            contact_sequence, recovery = recover_contact_sequence(
                collision_auditor.model,
                collision_auditor.data,
                qpos=qpos,
                qpos_addresses=collision_auditor._qpos_addresses,
                object_geom_names=["audit_can_geom"],
                fingertip_geom_names=recovery_geom_names,
                contact_link_names=recovery_link_names,
                source_active=recovery_side_active,
                object_mocap_poses=object_pose_model,
                object_mocap_body_names=["audit_can_body"],
                config=ContactRecoveryConfig(max_penetration_m=0.001),
            )
            contact_recovery_report = recovery.as_dict()
        except ValueError as exc:
            # Contact recovery is deliberately fail-closed. Keep the
            # inspection Reference usable for diagnosis, but never retain a
            # partial sequence after a penetration or witness failure.
            contact_recovery_report = {
                "status": "blocked",
                "reason": str(exc),
            }

    runtime_requested = not inspection_only
    contact_recovery_qualified, per_side_recovery_rates = _contact_recovery_gate(
        contact_recovery_report
    )
    contact_recovery_qualified = bool(
        contact_recovery_qualified and np.any(contact_sequence.active)
    )
    if contact_recovery_report is not None:
        contact_recovery_report = {
            **contact_recovery_report,
            "minimum_per_side_recovery_rate": 0.50,
            "per_side_recovery_rate": per_side_recovery_rates,
            "per_side_gate_pass": contact_recovery_qualified,
        }
    if runtime_requested and not contact_recovery_qualified:
        raise ValueError(
            "SOMA/CHORD retargeting did not recover measured robot contact on "
            "at least 50% of source-active frames for every required hand: "
            f"{contact_recovery_report}."
        )

    object_radius_source = _as_finite_array(
        row["object_mesh_radius"], name="object_mesh_radius", shape=(1,)
    )
    sanitized_source_object_name = _usd_safe_identifier(
        str(row["safe_object_name"]), fallback=pin.runtime_object_name
    )
    object_name = pin.runtime_object_name
    clearance = _anchor_vertical_clearance(
        object_mesh=object_mesh,
        source_object_wxyz=filtered_object_wxyz[anchor, 0],
        projection=projection,
    )
    if not 0.0 <= float(clearance["vertical_clearance_m"]) <= 0.01:
        raise ValueError(
            "Mapped corn-can/support clearance is outside [0, 0.01] m: "
            f"{clearance['vertical_clearance_m']}."
        )
    saturation = np.isclose(qpos, lower, rtol=0.0, atol=1.0e-5) | np.isclose(
        qpos, upper, rtol=0.0, atol=1.0e-5
    )
    saturated_joint_frames = {
        name: int(np.count_nonzero(saturation[:, index]))
        for index, name in enumerate(joint_names)
        if np.any(saturation[:, index])
    }
    target_last_source_frame = float(target_times[-1] * source_fps)
    offline_retarget_qualified = bool(
        contact_recovery_qualified
        and collision_audit["qualified"]
        and finger_acceptance["qualified"]
    )
    # Offline MuJoCo fitting is not Isaac runtime evidence.  This converter
    # emits an inspection candidate only.  A separate, hash-bound promotion
    # step must attach TrainingQualification after full-horizon Newton
    # state-lock and zero-assistance dynamics replays both pass.
    runtime_qualified = False
    metadata = {
        "runtime_qualified": runtime_qualified,
        "isaac_runtime_qualified": runtime_qualified,
        "inspection_only": True,
        "offline_retarget_qualified": offline_retarget_qualified,
        "inspection_scene_geometry_qualified": bool(collision_audit["qualified"]),
        "inspection_geometry_qualification_scope": (
            "Retarget gate covers authored 20 Hz states: typed intended-hand "
            "versus forbidden-robot can contact, robot-support, can-support, "
            "and robot self-collision. Dense linear/Slerp paths, the 200 Hz "
            "source retarget, and rendered geoms are diagnostic only because "
            "runtime advances one Reference frame per control transition. "
            "Newton state-lock and unassisted replay are separate runtime gates."
        ),
        "qualification_blocker": (
            "Offline retarget gates have not all passed."
            if not offline_retarget_qualified
            else (
                "Offline retarget gates passed, but TrainingQualification requires "
                "hash-bound full-horizon Newton state-lock and zero-assistance "
                "dynamics replay records from a separate promotion step."
            )
        ),
        "source": {
            "kind": "soma_motion_v1",
            "upstream_commit": pin.upstream_commit,
            "parquet_path": str(source),
            "parquet_sha256": actual_parquet_hash,
            "payload_sha256": _sha256_bytes(payload_bytes),
            "soma_joint_names_sha256": SOMA_JOINT_NAMES_SHA256,
            "fps": source_fps,
            "coord_frame": str(row["coord_frame"]),
            "requested_frame_range": [start_frame, end_frame_exclusive],
            "anchor_frame": anchor,
            "last_sampled_source_frame": target_last_source_frame,
            "dropped_crop_tail_seconds": (
                (end_frame_exclusive - 1) / source_fps - target_times[-1]
            ),
            "motion_time_scale": float(motion_time_scale),
            "output_duration_seconds": (len(qpos) - 1) / TARGET_FPS,
            **(
                {
                    "projected_frame_count_before_qualified_state_hold": (
                        emitted_frame_count_before_hold
                    ),
                    "qualified_state_hold_frames": qualified_state_hold_frames,
                }
                if qualified_state_hold_frames > 1
                else {}
            ),
        },
        "spatial_mapping": {
            "method": "contact_phase_fixed_base_workspace_projection",
            "wrist_pose_source": wrist_pose_source,
            "source_object_anchor": projection.source_object_anchor.tolist(),
            "target_object_anchor": projection.target_object_anchor.tolist(),
            "phase_aware_object_placement": bool(phase_aware_object_placement),
            "phase_anchor_max_shift_m": float(phase_anchor_max_shift_m),
            "phase_object_anchor_targets": phase_anchor_targets.tolist(),
            "phase_object_anchor_trajectory_range_robot_m": (
                None
                if target_anchor_trajectory is None
                else {
                    "minimum": np.min(target_anchor_trajectory, axis=0).tolist(),
                    "maximum": np.max(target_anchor_trajectory, axis=0).tolist(),
                }
            ),
            "global_object_motion_scale": projection.global_motion_scale,
            "local_hand_object_geometry_scale": projection.local_geometry_scale,
            "asset_and_support_geometry_scale": projection.local_geometry_scale,
            "rotation_wxyz": _rotation_to_wxyz(projection.rotation)[0].tolist(),
            "wrist_frame_offsets_wxyz": wrist_offsets,
            "inactive_wrist_target": (
                "bent-elbow palm pose outside and below the finite support"
                if bent_elbow_inactive_rest
                else "Vega neutral palm XY/orientation with support-safe raised Z"
            ),
            "bent_elbow_inactive_rest_enabled": bool(bent_elbow_inactive_rest),
            "inactive_safe_rest_positions_robot_m": safe_rest_positions.tolist(),
            "palm_object_standoff_m": palm_object_standoff_m,
            "palm_support_standoff_m": palm_support_standoff_m,
            "projected_active_wrist_position_range_robot_m": {
                "minimum": np.min(projected_active_wrist_200hz, axis=(0, 1)).tolist(),
                "maximum": np.max(projected_active_wrist_200hz, axis=(0, 1)).tolist(),
            },
            "contact_mask_dilation_frames_at_200hz": contact_dilation_frames,
            "contact_mask_gaussian_sigma_frames_at_200hz": (contact_blend_sigma_frames),
            "support_asset_pose_robot": (projection.support_asset_pose_wxyz().tolist()),
            "anchor_vertical_clearance": clearance,
        },
        "temporal_filter": {
            "method": (
                "zero_phase_butterworth source targets; 200hz sequential IK; "
                "no post-IK joint filter; exact linear_or_slerp sample at 20hz"
            ),
            "order": filter_order,
            "cutoff_hz": filter_cutoff_hz,
            "target_fps": TARGET_FPS,
            "motion_time_scale": float(motion_time_scale),
        },
        "finger_mapping": {
            "method": (
                "SOMA chain-relative hinge seed plus geometry-calibrated, "
                "tips-only Wuji IK with anatomical limits and slight-curl prior"
                if soma_hand_keypoint_projection_report is not None
                else "SOMA chain-relative hinge angles through ILTools KeypointRetargeter"
            ),
            "recipe_source": "wuji_retargeting_notes successful v1 map",
            "anatomical_limits_applied_before_ik": bool(anatomical_finger_limits),
            "anatomical_limit_rows": anatomical_finger_limit_rows,
            "source_to_robot_scale": float(soma_hand_scale),
            "posture_cost": float(finger_posture_prior),
            "palm_frame_calibration": palm_frame_calibration,
            "object_anchored_fingertips_enabled": bool(object_anchored_fingertips),
            "fingertip_reconciliation": fingertip_reconciliation_report,
            "object_anchored_fingertips": object_anchored_report,
            "hand_morphology_standoff_enabled": bool(hand_morphology_standoff),
            "hand_morphology_standoff": hand_morphology_report,
            "mapped_flexion_joints_per_hand": 15,
            "unobserved_abduction_joints_per_hand": 5,
            "unobserved_abduction_value_rad": 0.0,
            "inactive_hand_policy": "smoothly gate finger targets to open zero pose",
            "collision_safe_uniform_closure_scale": finger_closure_scales,
            "contact_guided_finger_ik_enabled": bool(contact_guided_finger_ik),
            "contact_guided_finger_ik": finger_ik_report,
            "soma_chord_contact_target_ik_enabled": bool(contact_target_ik),
            "soma_chord_contact_target_ik": contact_target_ik_report,
            "soma_chord_hand_keypoint_projection": (
                soma_hand_keypoint_projection_report
            ),
            "strict_final_finger_tuck": strict_final_finger_tuck_report,
            "final_acceptance": finger_acceptance,
        },
        "contacts": {
            "status": (
                "measured_lower_bound"
                if np.any(contact_sequence.active)
                else "binary_activity_only_no_geometry"
            ),
            "contact_geometry": (
                "mujoco_fingertip_witnesses"
                if np.any(contact_sequence.active)
                else "unavailable"
            ),
            "contact_recovery": contact_recovery_report,
            "soma_chord_enabled": bool(soma_chord_contacts),
            "soma_chord_source": chord_source_report,
            "soma_chord_robot_link_names": (
                None
                if chord_source_contacts is None
                else {
                    side: list(WUJI_CHORD_CONTACT_BODIES[side])
                    for side in ("left", "right")
                }
            ),
            "inspection_contact_sequence": (
                "five named fingertip slots per hand; all inactive with zero "
                "geometry; never use as supervision"
            ),
            "hand_sides": ["left", "right"],
            "source_binary_active_crop": arrays["hand_contact_active"][:, crop]
            .astype(np.int8)
            .tolist(),
            "active_20hz": contact_active_20hz.astype(np.int8).tolist(),
            "workspace_blend_weight_20hz": contact_blend_20hz.tolist(),
        },
        "wrist_ik": {
            "solver_partition": (
                "sequential dual-wrist SE(3) arm IK followed by arm-locked "
                "finger/keypoint/contact refinement"
                if split_wrist_hand_ik
                else "joint arm/finger keypoint and collision refinement"
            ),
            "split_wrist_hand_ik_enabled": bool(split_wrist_hand_ik),
            "shared_positioning_axes_enabled": bool(shared_positioning_joints),
            "shared_positioning_axis_names": sorted(shared_positioning_joints),
            "position_priority_enabled": bool(position_priority_wrist_ik),
            "position_priority_slack_m": float(position_priority_slack_m),
            "bent_elbow_initial_seed_enabled": bool(bent_elbow_inactive_rest),
            "post_wrist_joint_lock": wrist_hand_partition_report,
            "max_position_error_m": float(np.max(position_errors)),
            "mean_position_error_m": float(np.mean(position_errors)),
            "per_side_mean_position_error_m": {
                side: float(np.mean(position_errors[:, index]))
                for index, side in enumerate(("left", "right"))
            },
            "per_side_p95_position_error_m": {
                side: float(position_error_p95[index])
                for index, side in enumerate(("left", "right"))
            },
            "per_side_max_position_error_m": {
                side: float(np.max(position_errors[:, index]))
                for index, side in enumerate(("left", "right"))
            },
            "max_orientation_error_rad": float(np.max(orientation_errors)),
            "mean_orientation_error_rad": float(np.mean(orientation_errors)),
            "position_error_gate_percentile": 95.0,
            "position_error_limit_m": float(max_wrist_position_error),
            "orientation_error_limit_rad": float(max_wrist_orientation_error),
            "position_weight": 1.0,
            "orientation_weight": float(wrist_orientation_weight),
            "weight_semantics": (
                "position_weight/orientation_weight are Pink-compatible cost "
                "amplitudes; MuJoCo solver coefficients are their squares"
            ),
            "solver_position_objective_weight": 1.0,
            "solver_orientation_objective_weight": float(wrist_orientation_weight) ** 2,
            "previous_posture_weight": float(arm_previous_posture_weight),
            "neutral_posture_weight": float(arm_neutral_posture_weight),
            "damping": DEFAULT_IK_DAMPING,
            "saturated_joint_frame_fraction": float(np.mean(saturation)),
            "saturated_joint_frames": saturated_joint_frames,
        },
        "geometry_clearance_audit": collision_audit,
        "assets": {
            "source_object_name": str(row["safe_object_name"]),
            "sanitized_source_object_name": sanitized_source_object_name,
            "runtime_object_name": object_name,
            "object_source_radius_m": float(object_radius_source[0]),
            "object_target_radius_m": float(
                object_radius_source[0] * local_geometry_scale
            ),
            "dependency_sha256": asset_hashes,
            "support_sha256": support_hash,
        },
    }
    scene_physics = ScenePhysics(
        # These values match the authored object URDF inertial and make the
        # simulator material contract explicit instead of accepting backend
        # defaults. They are deliberately serialized even for inspection.
        object_mass_kg=np.asarray([1.0], dtype=np.float64),
        object_center_of_mass_m=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        object_diagonal_inertia_kg_m2=np.asarray(
            [[0.01, 0.01, 0.01]], dtype=np.float64
        ),
        object_static_friction=np.asarray([0.8], dtype=np.float64),
        object_dynamic_friction=np.asarray([0.6], dtype=np.float64),
        object_restitution=np.asarray([0.0], dtype=np.float64),
        support_static_friction=np.asarray([0.9], dtype=np.float64),
        support_dynamic_friction=np.asarray([0.7], dtype=np.float64),
        support_restitution=np.asarray([0.0], dtype=np.float64),
    )
    training_qualification = None

    reference = dexterous_reference_from_trajectory(
        retargeted,
        sequence_id=(f"{pin.sequence_id}_frames_{start_frame}_{end_frame_exclusive}"),
        robot_name="vega_wuji",
        fps=TARGET_FPS,
        fixed_root_pose_w=fixed_root,
        object_names=(object_name,),
        object_asset_paths=(str(object_urdf),),
        object_asset_sha256=(_sha256_file(object_urdf),),
        object_scales=np.full((1, 3), local_geometry_scale, dtype=np.float64),
        object_radii=object_radius_source * local_geometry_scale,
        left_hand_frame_names=FINGERTIP_NAMES["left"],
        left_hand_frame_poses_w=fingertips["left"],
        right_hand_frame_names=FINGERTIP_NAMES["right"],
        right_hand_frame_poses_w=fingertips["right"],
        support_surface_names=(f"{pin.runtime_object_name}_support",),
        support_surface_asset_paths=(str(support),),
        support_surface_asset_sha256=(support_hash,),
        collision_asset_dependencies=collision_asset_dependencies,
        support_surface_scales=np.full((1, 3), local_geometry_scale, dtype=np.float64),
        support_surface_poses_w=projection.support_asset_pose_wxyz()[None, :],
        scene_physics=scene_physics,
        contacts=contact_sequence,
        training_qualification=training_qualification,
        metadata=metadata,
    )
    reference.verify_scene_assets(require_hashes=True)
    return reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        choices=sorted(PINNED_SEQUENCES),
        default=DEFAULT_SEQUENCE_KEY,
        help=(
            "Pinned SOMA source sequence. It selects the source Parquet, "
            "support USD, asset hashes, crop, support cylinder, and object "
            "name. --source/--support/--start-frame/--end-frame-exclusive/"
            "--object-anchor-target override the selected entry."
        ),
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--support", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame-exclusive", type=int, default=None)
    parser.add_argument("--anchor-frame", type=int)
    parser.add_argument(
        "--object-anchor-target",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--global-motion-scale", type=float, default=DEFAULT_GLOBAL_MOTION_SCALE
    )
    parser.add_argument(
        "--local-geometry-scale", type=float, default=DEFAULT_LOCAL_GEOMETRY_SCALE
    )
    parser.add_argument(
        "--contact-dilation-frames",
        type=int,
        default=DEFAULT_CONTACT_DILATION_FRAMES,
    )
    parser.add_argument(
        "--contact-blend-sigma-frames",
        type=float,
        default=DEFAULT_CONTACT_BLEND_SIGMA_FRAMES,
    )
    parser.add_argument(
        "--palm-object-standoff-m",
        type=float,
        default=DEFAULT_PALM_OBJECT_STANDOFF_M,
    )
    parser.add_argument(
        "--palm-support-standoff-m",
        type=float,
        default=DEFAULT_PALM_SUPPORT_STANDOFF_M,
    )
    parser.add_argument(
        "--geometry-clearance-m",
        type=float,
        default=DEFAULT_GEOMETRY_CLEARANCE_M,
    )
    parser.add_argument(
        "--dense-geometry-audit-fps",
        type=float,
        default=DEFAULT_DENSE_GEOMETRY_AUDIT_FPS,
    )
    parser.add_argument(
        "--filter-cutoff-hz", type=float, default=DEFAULT_FILTER_CUTOFF_HZ
    )
    parser.add_argument("--filter-order", type=int, default=DEFAULT_FILTER_ORDER)
    parser.add_argument(
        "--motion-time-scale",
        type=float,
        default=DEFAULT_MOTION_TIME_SCALE,
        help=(
            "Stretch source motion duration while retaining the required 20 Hz "
            "Reference clock (2.0 emits the motion at half source speed)."
        ),
    )
    parser.add_argument(
        "--qualified-state-hold-frames",
        type=int,
        default=1,
        help=(
            "Repeat each collision-qualified 20 Hz state for this many control "
            "intervals. This zero-order hold improves actuator settling without "
            "interpolating through unaudited geometry."
        ),
    )
    parser.add_argument(
        "--contact-seeking",
        action="store_true",
        default=False,
        help=(
            "Let the hands reach the object. The support surface stays "
            "fail-closed, but the can is measured and reported instead of "
            "gated, because a clearance gate on the can forbids contact. Pair "
            "with a small --palm-object-standoff-m."
        ),
    )
    parser.add_argument(
        "--contact-settle-steps",
        type=int,
        default=60,
        help=(
            "Physics steps for each frame of the contact-seeking settle. "
            "Measured on the corn-can sequence: penetration converges by 60 "
            "steps and deeper settling changes nothing."
        ),
    )
    parser.add_argument(
        "--soma-chord-contacts",
        action="store_true",
        help=(
            "Reconstruct the 18,056-vertex SOMA-X mesh and extract the same "
            "deterministic 4,096-point, 1 cm contact targets used by CHORD. "
            "Requires --contact-seeking."
        ),
    )
    parser.add_argument(
        "--soma-device",
        default="cuda",
        help="Torch device for SOMA-X mesh reconstruction (default: cuda).",
    )
    parser.add_argument(
        "--contact-target-ik",
        action="store_true",
        help=(
            "Optimize all 16 semantic Wuji hand links against reconstructed "
            "SOMA/CHORD object-surface targets. Requires --soma-chord-contacts."
        ),
    )
    parser.add_argument(
        "--contact-guided-finger-ik",
        action="store_true",
        help=(
            "Run the experimental fingertip witness IK stage before settling. "
            "It is opt-in because contact recovery and collision audits remain "
            "authoritative."
        ),
    )
    parser.add_argument(
        "--phase-aware-object-placement",
        action="store_true",
        help=(
            "Infer one reachable object placement per hand contact phase and "
            "smoothly blend those placements through the handover. The "
            "explicit --object-anchor-target remains the no-contact base."
        ),
    )
    parser.add_argument(
        "--phase-anchor-max-shift-m",
        type=float,
        default=0.20,
        help=(
            "Maximum X/Y shift from --object-anchor-target for the inferred "
            "phase placements."
        ),
    )
    parser.add_argument(
        "--phase-object-anchor-targets",
        type=float,
        nargs=6,
        metavar=("LX", "LY", "LZ", "RX", "RY", "RZ"),
        help=(
            "Explicit left-phase and right-phase object anchors. Requires "
            "--phase-aware-object-placement; otherwise the anchors are "
            "inferred from the source wrists."
        ),
    )
    parser.add_argument("--max-wrist-position-error", type=float, default=0.05)
    parser.add_argument("--max-wrist-orientation-error-deg", type=float, default=180.0)
    parser.add_argument(
        "--wrist-orientation-weight",
        type=float,
        default=DEFAULT_IK_ORIENTATION_WEIGHT,
        help=(
            "Relative orientation cost in the initial dual-wrist SE(3) arm IK. "
            "The historical near-zero default remains available for comparison."
        ),
    )
    parser.add_argument(
        "--split-wrist-hand-ik",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Solve arm/shared joints from the two wrist poses once, then lock "
            "them while SOMA keypoints, CHORD contacts, and collision cleanup "
            "change Wuji finger joints only (default: enabled)."
        ),
    )
    parser.add_argument(
        "--anatomical-finger-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Tighten Wuji PIP/DIP/thumb limits inside the MuJoCo model before "
            "finger IK (default: enabled; required for training promotion)."
        ),
    )
    parser.add_argument(
        "--fingertip-contact-standoff-m",
        type=float,
        default=0.0,
        help=(
            "Stand each object-anchored fingertip target off the surface "
            "along its contact normal, e.g. by the 9 mm fingertip collision "
            "radius. Measured negative on the snack-box clip: it rotates the "
            "finger and drives the middle phalanges in, so whole-hand "
            "penetration worsened from -10.25 to -15.73 mm (default: 0)."
        ),
    )
    parser.add_argument(
        "--reconciliation-clearance-m",
        type=float,
        default=0.0005,
        help=(
            "Object clearance the fingertip reconciliation must respect, as a "
            "task inside the solve rather than a repair afterwards. Pass a "
            "negative value to permit that much intended interpenetration, or "
            "'none' semantics via --no-reconcile-fingertips-after-contact to "
            "skip the pass entirely (default: 0.0005)."
        ),
    )
    parser.add_argument(
        "--reconcile-fingertips-after-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After contact and collision repair, re-solve the fingertips back "
            "onto their targets with the same bounded solver. The repair "
            "stages shape the whole hand and bulk-shift the fingertips off "
            "the contact points the tips-only solve had reached. This runs "
            "before every geometry audit, so its result is re-verified "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--object-anchored-fingertips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During contact, aim each fingertip at the measured CHORD contact "
            "point carried on the object instead of at the human fingertip "
            "position. The Wuji hand is larger than the source hand, so on a "
            "true-size object those are different points and only the contact "
            "location is a physical requirement (default: enabled). Requires "
            "--soma-chord-contacts."
        ),
    )
    parser.add_argument(
        "--mcp-abduction-limit-deg",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOWER", "UPPER"),
        help=(
            "Override the non-thumb MCP abduction envelope. The Wuji "
            "+/-40 deg range is a mechanical limit; adjacent fingers collide "
            "with each other before reaching it, so a tighter anatomical "
            "envelope can remove ring/pinky self-penetration."
        ),
    )
    parser.add_argument(
        "--soma-hand-scale",
        type=float,
        default=DEFAULT_SOMA_TO_WUJI_HAND_SCALE,
        help=(
            "Source wrist-to-tip scale. The adopted contact-preserving recipe "
            "uses exactly 1.0."
        ),
    )
    parser.add_argument(
        "--shared-positioning-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Let the dual-wrist solve use the shared Lift and torso_flip "
            "axes. Locking them costs two degrees of freedom and the fingers "
            "pay for the palm orientation the arms cannot reach: measured on "
            "the snack-box clip, locked gives 23.8 deg palm orientation error "
            "and 12-14 mm fingertip p95, authorized gives 1.15 deg and 0.36 "
            "mm on a bit-identical object trajectory. Disable to reproduce "
            "the legacy fixed-base transform (default: enabled)."
        ),
    )
    parser.add_argument(
        "--align-finger-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compose the knuckle-position palm fit with the palm yaw that "
            "centres the MCP abduction demand (the notes' definition B). "
            "Without it the mapped finger fan sits rotated against the Wuji "
            "zero-pose fan and pins every abduction joint at its +/-40 deg "
            "mechanical limit. Measured on the snack-box clip: fingertip p95 "
            "20.9/17.9 -> 14.3/12.2 mm (default: enabled)."
        ),
    )
    parser.add_argument(
        "--hand-morphology-standoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic. Retreat each palm target along its own approach axis "
            "by the measured Wuji-minus-human wrist-to-tip excess. Measured "
            "negative on the snack-box clip: it moves the palm without moving "
            "its own fingertip targets, so reach gets worse (tip p95 20 -> 45 "
            "mm alone, 18 -> 105 mm with a radial standoff). Use "
            "--palm-object-standoff-m instead, which translates the palm and "
            "its targets together (default: disabled)."
        ),
    )
    parser.add_argument(
        "--hand-morphology-standoff-m",
        type=float,
        default=None,
        help=(
            "Override the measured per-side morphology standoff with one "
            "explicit distance in meters."
        ),
    )
    parser.add_argument(
        "--finger-posture-prior",
        type=float,
        default=DEFAULT_FINGER_POSTURE_PRIOR,
        help=(
            "Slight-curl posture-task cost for tips-only finger IK "
            f"(default: {DEFAULT_FINGER_POSTURE_PRIOR:g})."
        ),
    )
    parser.add_argument(
        "--require-finger-gates",
        action="store_true",
        help=(
            "Fail conversion unless both hands pass the complete final-FK "
            "hyperextension, wrist, tip, and temporal-jump gates."
        ),
    )
    parser.add_argument(
        "--arm-previous-posture-weight",
        type=float,
        default=0.0,
        help=(
            "Sequential wrist IK cost that keeps each arm solution near the "
            "previous frame and discourages redundant elbow-branch changes."
        ),
    )
    parser.add_argument(
        "--arm-neutral-posture-weight",
        type=float,
        default=0.0,
        help=(
            "Wrist IK cost that selects arm solutions near the model's neutral "
            "pose when multiple elbow configurations reach the same wrists."
        ),
    )
    parser.add_argument(
        "--position-priority-wrist-ik",
        action="store_true",
        help=(
            "Solve dual-wrist position as the primary task, orientation only "
            "in its exact Jacobian null space, and arm posture as a tertiary "
            "task. This prevents orientation tracking from pulling a wrist "
            "away from its source-motion position."
        ),
    )
    parser.add_argument(
        "--position-priority-slack-m",
        type=float,
        default=0.005,
        help=(
            "Maximum per-wrist RMS position slack offered to lower-priority "
            "orientation steps (default: 0.005 m)."
        ),
    )
    parser.add_argument(
        "--bent-elbow-inactive-rest",
        action="store_true",
        help=(
            "Park an inactive hand beside the torso, outside and below the "
            "finite support, and seed wrist IK from the same bent-elbow pose."
        ),
    )
    parser.add_argument(
        "--inspection-only",
        action="store_true",
        help=(
            "Write an inspection-only Reference with named but inactive contact "
            "slots. The output is explicitly not runtime-qualified."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reference = convert_soma_g1_parquet(
        args.source,
        pin=PINNED_SEQUENCES[args.sequence],
        model_path=args.model,
        support_path=args.support,
        start_frame=args.start_frame,
        end_frame_exclusive=args.end_frame_exclusive,
        anchor_frame=args.anchor_frame,
        object_anchor_target=(
            None
            if args.object_anchor_target is None
            else np.asarray(args.object_anchor_target, dtype=np.float64)
        ),
        global_motion_scale=args.global_motion_scale,
        local_geometry_scale=args.local_geometry_scale,
        contact_dilation_frames=args.contact_dilation_frames,
        contact_blend_sigma_frames=args.contact_blend_sigma_frames,
        contact_seeking=args.contact_seeking,
        soma_chord_contacts=args.soma_chord_contacts,
        soma_device=args.soma_device,
        contact_target_ik=args.contact_target_ik,
        contact_guided_finger_ik=args.contact_guided_finger_ik,
        contact_settle_steps=args.contact_settle_steps,
        phase_aware_object_placement=args.phase_aware_object_placement,
        phase_anchor_max_shift_m=args.phase_anchor_max_shift_m,
        phase_object_anchor_targets=(
            None
            if args.phase_object_anchor_targets is None
            else np.asarray(args.phase_object_anchor_targets, dtype=np.float64).reshape(
                2, 3
            )
        ),
        palm_object_standoff_m=args.palm_object_standoff_m,
        palm_support_standoff_m=args.palm_support_standoff_m,
        geometry_clearance_m=args.geometry_clearance_m,
        dense_geometry_audit_fps=args.dense_geometry_audit_fps,
        filter_cutoff_hz=args.filter_cutoff_hz,
        filter_order=args.filter_order,
        motion_time_scale=args.motion_time_scale,
        qualified_state_hold_frames=args.qualified_state_hold_frames,
        max_wrist_position_error=args.max_wrist_position_error,
        max_wrist_orientation_error=math.radians(args.max_wrist_orientation_error_deg),
        wrist_orientation_weight=args.wrist_orientation_weight,
        arm_previous_posture_weight=args.arm_previous_posture_weight,
        arm_neutral_posture_weight=args.arm_neutral_posture_weight,
        position_priority_wrist_ik=args.position_priority_wrist_ik,
        position_priority_slack_m=args.position_priority_slack_m,
        bent_elbow_inactive_rest=args.bent_elbow_inactive_rest,
        split_wrist_hand_ik=args.split_wrist_hand_ik,
        anatomical_finger_limits=args.anatomical_finger_limits,
        soma_hand_scale=args.soma_hand_scale,
        object_anchored_fingertips=args.object_anchored_fingertips,
        fingertip_contact_standoff_m=args.fingertip_contact_standoff_m,
        reconcile_fingertips_after_contact=args.reconcile_fingertips_after_contact,
        reconciliation_clearance_m=args.reconciliation_clearance_m,
        mcp_abduction_limit_deg=(
            None
            if args.mcp_abduction_limit_deg is None
            else tuple(args.mcp_abduction_limit_deg)
        ),
        shared_positioning_axes=args.shared_positioning_axes,
        align_finger_directions=args.align_finger_directions,
        hand_morphology_standoff=args.hand_morphology_standoff,
        hand_morphology_standoff_m=args.hand_morphology_standoff_m,
        finger_posture_prior=args.finger_posture_prior,
        require_finger_gates=args.require_finger_gates,
        inspection_only=args.inspection_only,
    )
    output = save_dexterous_reference_npz(reference, args.output)
    summary = {
        "output": str(output),
        "frames": reference.frame_count,
        "fps": reference.fps,
        "duration_s": (reference.frame_count - 1) / reference.fps,
        "isaac_runtime_qualified": reference.metadata["isaac_runtime_qualified"],
        "qualification_blocker": reference.metadata["qualification_blocker"],
        "wrist_ik": reference.metadata["wrist_ik"],
        "geometry_clearance_audit": reference.metadata["geometry_clearance_audit"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
