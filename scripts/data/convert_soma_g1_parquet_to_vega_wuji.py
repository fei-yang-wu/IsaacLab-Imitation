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

from iltools.core import (
    ContactSequence,
    DexterousReference,
    Trajectory,
    build_urdf_collision_asset_dependencies,
    save_dexterous_reference_npz,
)
from iltools.retarget import (
    KeypointJointSpec,
    KeypointRetargeter,
    MujocoDualHandRetargeter,
    dexterous_reference_from_trajectory,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO_ROOT
    / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
    / "vega_u_wuji_v2_beta1_with_mount.xml"
)
DEFAULT_SOURCE = Path(
    "/home/fwu91/Documents/DexManip/video_to_data/robotic_grounding/source/"
    "robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/"
    "soma/sequence_id=2026-03-23_18-10-01_corn_can_right_left_handover_01/"
    "robot_name=g1/data.parquet"
)
DEFAULT_SUPPORT = Path(
    "/home/fwu91/Documents/DexManip/video_to_data/robotic_grounding/source/"
    "robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/"
    "reconstructed_stage/"
    "2026-03-23_18-10-01_corn_can_right_left_handover_01_support.usda"
)
PINNED_PARQUET_SHA256 = (
    "1679e920f20cc741ffccaf8164a960bf46e1f124d670d7bddb2416ca6ea8b082"
)
PINNED_PAYLOAD_SHA256 = (
    "5acf75dee07040e3bd826d50d0843c84185f56352eb24afb8b1b6b5bc311a229"
)
PINNED_OBJECT_ASSET_SHA256 = {
    "textured_mesh.urdf": "3ca858449cf9ee7c1d19e4e17d49558ebd9d170186a6ec226e1961c6c5250975",
    "textured_mesh.obj": "d0668549cbcf73a9c7935b6690485a6a51e475fcbf2e68d73108be0eb5b532ed",
}
PINNED_SUPPORT_SHA256 = (
    "c25529df26f73f7ba2ec205fdb9d54716d5591f2079c7901d1e1763d94e41631"
)
PINNED_UPSTREAM_COMMIT = "afa9dffda748e301ddf79b7002ce4cc5cb552adb"
PINNED_SEQUENCE_ID = "2026-03-23_18-10-01_corn_can_right_left_handover_01"
PINNED_START_FRAME = 522
PINNED_END_FRAME_EXCLUSIVE = 840
TARGET_FPS = 20.0
DEFAULT_FILTER_CUTOFF_HZ = 6.0
DEFAULT_FILTER_ORDER = 4
DEFAULT_OBJECT_ANCHOR_TARGET = np.asarray((0.8, 0.15, 1.14), dtype=np.float64)
DEFAULT_GLOBAL_MOTION_SCALE = 0.40
DEFAULT_LOCAL_GEOMETRY_SCALE = 0.75
DEFAULT_CONTACT_DILATION_FRAMES = 41
DEFAULT_CONTACT_BLEND_SIGMA_FRAMES = 15.0
DEFAULT_IK_ORIENTATION_WEIGHT = 1.0e-6
DEFAULT_IK_DAMPING = 0.30
DEFAULT_PALM_OBJECT_STANDOFF_M = 0.28
DEFAULT_PALM_SUPPORT_STANDOFF_M = 0.20
DEFAULT_GEOMETRY_CLEARANCE_M = 0.005
DEFAULT_DENSE_GEOMETRY_AUDIT_FPS = 1000.0
PINNED_SUPPORT_CENTER_SOURCE = np.asarray(
    (-1.3406889001888564, 0.3631554151795554, 0.8561500119144527),
    dtype=np.float64,
)
PINNED_SUPPORT_HEIGHT_SOURCE = 0.01
PINNED_SUPPORT_RADIUS_SOURCE = 0.4337941740262468
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

    def object_positions(self, values: np.ndarray) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64)
        flat = (points - self.source_object_anchor).reshape(-1, 3)
        transformed = self.target_object_anchor + self.global_motion_scale * (
            self.rotation.apply(flat)
        )
        return transformed.reshape(points.shape)

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
                PINNED_SUPPORT_CENTER_SOURCE - self.source_object_anchor
            )
        )

    def support_top_z(self) -> float:
        return float(
            self.support_center_target()[2]
            + 0.5 * self.local_geometry_scale * PINNED_SUPPORT_HEIGHT_SOURCE
        )


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
    row: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    expected_scalars = {
        "schema_version": "motion_v1",
        "motion_kind": "single_robot",
        "source_dataset": "soma",
        "source_kind": "soma",
        "coord_frame": "robot_base_z_up",
        "sequence_id": PINNED_SEQUENCE_ID,
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
    *, start_frame: int, end_frame_exclusive: int, source_fps: float
) -> tuple[np.ndarray, np.ndarray]:
    if start_frame < 0 or end_frame_exclusive <= start_frame + 1:
        raise ValueError("Source crop must contain at least two frames.")
    source_times = np.arange(start_frame, end_frame_exclusive) / source_fps
    duration = source_times[-1] - source_times[0]
    count = int(np.floor(duration * TARGET_FPS + 1.0e-9)) + 1
    target_times = source_times[0] + np.arange(count) / TARGET_FPS
    return source_times, target_times


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
                spec(f"{target_finger}_mcp_flex", hand, f"{prefix}1", f"{prefix}2"),
                spec(f"{target_finger}_pip", f"{prefix}1", f"{prefix}2", f"{prefix}3"),
                spec(f"{target_finger}_dip", f"{prefix}2", f"{prefix}3", f"{prefix}4"),
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


def _calibrate_wrist_orientations(
    source_wxyz: np.ndarray,
    *,
    alignment_rotation: Rotation,
    anchor_local_index: int,
    neutral_poses: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    output = np.empty_like(source_wxyz)
    offsets: dict[str, list[float]] = {}
    for index, side in enumerate(("left", "right")):
        aligned = alignment_rotation * _rotation_from_wxyz(source_wxyz[:, index])
        neutral = _rotation_from_wxyz(neutral_poses[side][None, 3:7])
        local_offset = aligned[anchor_local_index].inv() * neutral
        calibrated = aligned * local_offset
        output[:, index] = _rotation_to_wxyz(calibrated)
        offsets[side] = _rotation_to_wxyz(local_offset)[0].tolist()
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
    support_center = projection.target_object_anchor + (
        projection.local_geometry_scale
        * projection.rotation.apply(
            PINNED_SUPPORT_CENTER_SOURCE - projection.source_object_anchor
        )
    )
    support_top_z = float(
        support_center[2]
        + 0.5 * projection.local_geometry_scale * PINNED_SUPPORT_HEIGHT_SOURCE
    )
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
    penetration_tolerance_m: float = 1.0e-6,
) -> dict[str, Any]:
    """Prove the can stays above the support top plane at every sample."""

    poses = _as_finite_array(object_poses, name="can-support audit object poses")
    if poses.ndim != 3 or poses.shape[1:] != (1, 7):
        raise ValueError("Can-support audit poses must have shape [T, 1, 7].")
    if not np.isfinite((object_scale, support_top_z, penetration_tolerance_m)).all():
        raise ValueError("Can-support audit parameters must be finite.")
    if object_scale <= 0.0 or penetration_tolerance_m < 0.0:
        raise ValueError("Can scale must be positive and tolerance non-negative.")
    vertices = np.unique(_mesh_vertices(object_mesh), axis=0) * object_scale
    rotations = _rotation_from_wxyz(poses[:, 0, 3:7])
    clearances = np.empty(len(poses), dtype=np.float64)
    for frame in range(len(poses)):
        minimum_z = float(
            poses[frame, 0, 2] + np.min(rotations[frame].apply(vertices)[:, 2])
        )
        clearances[frame] = minimum_z - support_top_z
    violation_frames = np.flatnonzero(clearances < -penetration_tolerance_m).tolist()
    minimum_frame = int(np.argmin(clearances))
    return {
        "qualified": not violation_frames,
        "frame_count": len(poses),
        "minimum_vertical_clearance_m": float(clearances[minimum_frame]),
        "minimum_frame": minimum_frame,
        "penetration_tolerance_m": penetration_tolerance_m,
        "violation_frame_count": len(violation_frames),
        "violation_frames": violation_frames,
        "unique_mesh_vertex_count": len(vertices),
        "method": (
            "all scaled corn-can OBJ vertices above the analytic support top "
            "plane; this is conservative for the finite cylinder"
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
) -> dict[str, str]:
    # Only the wrapper and collision geometry affect physics. Material and
    # texture files are intentionally outside the training asset contract.
    dependencies = (object_urdf, object_collision_mesh)
    for path in dependencies:
        if not path.is_file():
            raise FileNotFoundError(f"Object asset dependency is missing: {path}")
    result = {str(path.resolve()): _sha256_file(path) for path in dependencies}
    for path in dependencies:
        expected = PINNED_OBJECT_ASSET_SHA256.get(path.name)
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


def convert_soma_g1_parquet(
    source_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    support_path: str | Path = DEFAULT_SUPPORT,
    start_frame: int = PINNED_START_FRAME,
    end_frame_exclusive: int = PINNED_END_FRAME_EXCLUSIVE,
    anchor_frame: int | None = None,
    object_anchor_target: np.ndarray = DEFAULT_OBJECT_ANCHOR_TARGET,
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
    contact_seeking: bool = False,
    contact_settle_steps: int = 60,
    max_wrist_position_error: float = 0.05,
    max_wrist_orientation_error: float = math.pi,
    inspection_only: bool = False,
    expected_parquet_sha256: str = PINNED_PARQUET_SHA256,
    expected_payload_sha256: str = PINNED_PAYLOAD_SHA256,
) -> DexterousReference:
    """Build one validated Reference without writing it to disk."""

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
    arrays = _validate_source(row, payload)
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
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    object_anchor = _as_finite_array(
        object_anchor_target, name="object_anchor_target", shape=(3,)
    )
    fixed_root = _unit_wxyz(
        _as_finite_array(fixed_root_pose_w, name="fixed_root_pose_w", shape=(7,))[3:7],
        name="fixed_root_pose_w",
    )
    fixed_root = np.concatenate(
        (np.asarray(fixed_root_pose_w, dtype=np.float64)[:3], fixed_root)
    )

    exact_contacts = _has_exact_contact_geometry(row)
    if exact_contacts:
        raise NotImplementedError(
            "This pinned converter has no verified G1-contact-link to Wuji-link "
            "mapping. Add and test that mapping before converting exact contacts."
        )
    if not inspection_only:
        raise ValueError(
            "The pinned SOMA Parquet has binary hand_contact_active only. Exact "
            "contact link names, points, normals, and object part IDs are absent. "
            "Refusing to invent a ContactSequence. Pass "
            "--inspection-only only for a non-runtime-qualified "
            "inspection Reference."
        )

    source_fps = float(row["fps"])
    source_times, target_times = _target_times(
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        source_fps=source_fps,
    )
    crop = slice(start_frame, end_frame_exclusive)
    anchor_local = anchor - start_frame

    model = mujoco.MjModel.from_xml_path(str(model_file))
    joint_names, limits = _model_joint_contract(model)
    neutral_poses = _neutral_wrist_poses(model)

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
    asset_hashes = _asset_dependency_hashes(object_urdf, object_mesh)
    collision_asset_dependencies = build_urdf_collision_asset_dependencies(
        object_urdf,
        asset_role="object",
        asset_index=0,
    )
    support_hash = _sha256_file(support)
    if support_hash != PINNED_SUPPORT_SHA256:
        raise ValueError(
            "Pinned support SHA-256 mismatch: "
            f"expected {PINNED_SUPPORT_SHA256}, got {support_hash}."
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

    filtered_ee_position = _lowpass(
        arrays["ee_pose_w"][..., :3],
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
    filtered_ee_wxyz = _lowpass_wxyz(
        arrays["ee_pose_w"][..., 3:7],
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
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

    projection = ContactPhaseProjection(
        rotation=_yaw_only_inverse(arrays["robot_root_wxyz"][anchor]),
        source_object_anchor=source_object_anchor,
        target_object_anchor=object_anchor,
        global_motion_scale=float(global_motion_scale),
        local_geometry_scale=float(local_geometry_scale),
    )
    object_position_200hz = projection.object_positions(filtered_object_position[crop])
    active_wrist_position_200hz = projection.active_wrist_positions(
        filtered_ee_position[crop], filtered_object_position[crop]
    )
    contact_blend_200hz = _contact_phase_weights(
        arrays["hand_contact_active"][:, crop],
        dilation_frames=contact_dilation_frames,
        gaussian_sigma_frames=contact_blend_sigma_frames,
    )
    neutral_positions = np.stack(
        [neutral_poses[side][:3] for side in ("left", "right")], axis=0
    )
    wrist_position_200hz, projected_active_wrist_200hz, safe_rest_positions = (
        _project_palm_positions_nonpenetrating(
            active_wrist_position_200hz,
            object_position_200hz,
            neutral_positions,
            contact_blend_200hz,
            support_top_z=projection.support_top_z(),
            palm_object_standoff_m=palm_object_standoff_m,
            palm_support_standoff_m=palm_support_standoff_m,
        )
    )
    calibrated_wrist_wxyz, wrist_offsets = _calibrate_wrist_orientations(
        filtered_ee_wxyz[crop],
        alignment_rotation=projection.rotation,
        anchor_local_index=anchor_local,
        neutral_poses=neutral_poses,
    )
    wrist_wxyz_200hz = _blend_wrist_orientations(
        calibrated_wrist_wxyz, neutral_poses, contact_blend_200hz
    )
    object_wxyz_200hz = projection.orientations(filtered_object_wxyz[crop])

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
    arm_joint_names = tuple(
        name
        for name in joint_names
        if name.startswith("L_arm_") or name.startswith("R_arm_")
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
        orientation_weight=DEFAULT_IK_ORIENTATION_WEIGHT,
    ).retarget(source_trajectory_200hz)
    qpos_200hz_raw = np.asarray(retargeted_200hz.observations["qpos"], dtype=np.float64)
    qpos_200hz = _lowpass(
        qpos_200hz_raw,
        fps=source_fps,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
    )
    lower = np.asarray([limits[name][0] for name in joint_names], dtype=np.float64)
    upper = np.asarray([limits[name][1] for name in joint_names], dtype=np.float64)
    qpos_200hz = np.clip(qpos_200hz, lower, upper)
    qpos = _resample_linear(qpos_200hz, source_times, target_times)
    object_pose_model = np.concatenate(
        (object_position_20hz, object_wxyz_20hz), axis=-1
    )
    collision_auditor = _SceneCollisionAuditor(
        robot_model_path=model_file,
        object_mesh_path=object_mesh,
        object_scale=local_geometry_scale,
        support_center=projection.support_center_target(),
        support_radius=PINNED_SUPPORT_RADIUS_SOURCE * local_geometry_scale,
        support_height=PINNED_SUPPORT_HEIGHT_SOURCE * local_geometry_scale,
        joint_names=joint_names,
        required_clearance_m=geometry_clearance_m,
        gate_scenes=() if contact_seeking else None,
    )
    rendered_geometry_auditor = _SceneCollisionAuditor(
        robot_model_path=model_file,
        object_mesh_path=object_mesh,
        object_scale=local_geometry_scale,
        support_center=projection.support_center_target(),
        support_radius=PINNED_SUPPORT_RADIUS_SOURCE * local_geometry_scale,
        support_height=PINNED_SUPPORT_HEIGHT_SOURCE * local_geometry_scale,
        joint_names=joint_names,
        required_clearance_m=geometry_clearance_m,
        include_visual_geometry=True,
        gate_scenes=() if contact_seeking else None,
    )
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
    combined_qpos = np.concatenate(
        (qpos_200hz, qpos, source_dense_qpos, emitted_dense_qpos), axis=0
    )
    combined_object_poses = np.concatenate(
        (
            object_pose_200hz,
            object_pose_model,
            source_dense_object_poses,
            emitted_dense_object_poses,
        ),
        axis=0,
    )
    combined_qpos, finger_closure_scales, projection_search_audit = (
        _collision_safe_finger_projection(
            combined_qpos,
            joint_names=joint_names,
            object_poses=combined_object_poses,
            auditor=collision_auditor,
        )
    )
    settling_report: dict[str, Any] | None = None
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

    qpos_200hz = combined_qpos[: len(qpos_200hz)]
    emitted_start = len(qpos_200hz)
    source_dense_start = emitted_start + len(qpos)
    emitted_dense_start = source_dense_start + len(source_dense_qpos)
    qpos = combined_qpos[emitted_start:source_dense_start]
    source_dense_qpos = combined_qpos[source_dense_start:emitted_dense_start]
    emitted_dense_qpos = combined_qpos[emitted_dense_start:]
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
    )
    can_support_emitted_dense = _audit_can_support_vertical_clearance(
        emitted_dense_object_poses,
        object_mesh=object_mesh,
        object_scale=local_geometry_scale,
        support_top_z=projection.support_top_z(),
    )
    collision_audit = {
        "qualified": bool(
            collision_audit_200hz["qualified"]
            and collision_audit_20hz["qualified"]
            and collision_audit_source_dense["qualified"]
            and collision_audit_emitted_dense["qualified"]
            and rendered_audit_20hz["qualified"]
            and rendered_audit_source_dense["qualified"]
            and rendered_audit_emitted_dense["qualified"]
            and can_support_source_dense["qualified"]
            and can_support_emitted_dense["qualified"]
        ),
        "required_clearance_m": geometry_clearance_m,
        "source_200hz": collision_audit_200hz,
        "emitted_20hz": collision_audit_20hz,
        "source_path_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": source_fps,
            "interpolation": "linear joint/translation and quaternion Slerp",
            **collision_audit_source_dense,
        },
        "emitted_path_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": TARGET_FPS,
            "interpolation": "linear joint/translation and quaternion Slerp",
            **collision_audit_emitted_dense,
        },
        "rendered_geometry_20hz": rendered_audit_20hz,
        "source_path_rendered_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": source_fps,
            **rendered_audit_source_dense,
        },
        "emitted_path_rendered_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": TARGET_FPS,
            **rendered_audit_emitted_dense,
        },
        "source_path_can_support_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": source_fps,
            **can_support_source_dense,
        },
        "emitted_path_can_support_dense_1khz": {
            "fps": dense_geometry_audit_fps,
            "source_fps": TARGET_FPS,
            **can_support_emitted_dense,
        },
        "contact_settling": settling_report,
        "finger_projection_search": projection_search_audit,
    }
    if not collision_audit["qualified"]:
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
    if float(np.max(position_errors)) > max_wrist_position_error:
        raise ValueError(
            "Vega wrist IK position error exceeds the limit: "
            f"{np.max(position_errors):.6f} m > {max_wrist_position_error:.6f} m."
        )
    if float(np.max(orientation_errors)) > max_wrist_orientation_error:
        raise ValueError(
            "Vega wrist IK orientation error exceeds the limit: "
            f"{np.max(orientation_errors):.6f} rad > "
            f"{max_wrist_orientation_error:.6f} rad."
        )
    retargeted.observations["left_wrist_pose_w"] = achieved_wrists["left"]
    retargeted.observations["right_wrist_pose_w"] = achieved_wrists["right"]
    retargeted.observations["object_poses_w"] = object_pose_model

    object_radius_source = _as_finite_array(
        row["object_mesh_radius"], name="object_mesh_radius", shape=(1,)
    )
    sanitized_source_object_name = _usd_safe_identifier(
        str(row["safe_object_name"]), fallback="corn_can"
    )
    object_name = "corn_can"
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
    metadata = {
        "runtime_qualified": False,
        "isaac_runtime_qualified": False,
        "inspection_only": True,
        "inspection_scene_geometry_qualified": True,
        "inspection_geometry_qualification_scope": (
            "1 kHz sampled MuJoCo robot-vs-can/support and can-vs-support along "
            "both the 200 Hz source and emitted 20 Hz interpolation paths; "
            "includes all rendered robot geoms but excludes robot self-collision, "
            "Newton backend variance, and continuous swept-volume certification"
        ),
        "qualification_blocker": (
            "Source has per-hand binary contact activity but no exact contact "
            "link names, points, normals, or object part IDs."
        ),
        "source": {
            "kind": "soma_motion_v1",
            "upstream_commit": PINNED_UPSTREAM_COMMIT,
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
        },
        "spatial_mapping": {
            "method": "contact_phase_fixed_base_workspace_projection",
            "source_object_anchor": projection.source_object_anchor.tolist(),
            "target_object_anchor": projection.target_object_anchor.tolist(),
            "global_object_motion_scale": projection.global_motion_scale,
            "local_hand_object_geometry_scale": projection.local_geometry_scale,
            "asset_and_support_geometry_scale": projection.local_geometry_scale,
            "rotation_wxyz": _rotation_to_wxyz(projection.rotation)[0].tolist(),
            "wrist_frame_offsets_wxyz": wrist_offsets,
            "inactive_wrist_target": (
                "Vega neutral palm XY/orientation with support-safe raised Z"
            ),
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
                "zero_phase_butterworth; 200hz sequential IK; zero_phase joint "
                "filter; linear_or_slerp_resample_to_20hz"
            ),
            "order": filter_order,
            "cutoff_hz": filter_cutoff_hz,
            "target_fps": TARGET_FPS,
        },
        "finger_mapping": {
            "method": "SOMA chain-relative hinge angles through ILTools KeypointRetargeter",
            "mapped_flexion_joints_per_hand": 15,
            "unobserved_abduction_joints_per_hand": 5,
            "unobserved_abduction_value_rad": 0.0,
            "inactive_hand_policy": "smoothly gate finger targets to open zero pose",
            "collision_safe_uniform_closure_scale": finger_closure_scales,
        },
        "contacts": {
            "status": "binary_activity_only_no_geometry",
            "contact_geometry": "unavailable",
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
            "max_position_error_m": float(np.max(position_errors)),
            "mean_position_error_m": float(np.mean(position_errors)),
            "per_side_mean_position_error_m": {
                side: float(np.mean(position_errors[:, index]))
                for index, side in enumerate(("left", "right"))
            },
            "per_side_p95_position_error_m": {
                side: float(np.percentile(position_errors[:, index], 95.0))
                for index, side in enumerate(("left", "right"))
            },
            "per_side_max_position_error_m": {
                side: float(np.max(position_errors[:, index]))
                for index, side in enumerate(("left", "right"))
            },
            "max_orientation_error_rad": float(np.max(orientation_errors)),
            "mean_orientation_error_rad": float(np.mean(orientation_errors)),
            "position_error_limit_m": float(max_wrist_position_error),
            "orientation_error_limit_rad": float(max_wrist_orientation_error),
            "position_weight": 1.0,
            "orientation_weight": DEFAULT_IK_ORIENTATION_WEIGHT,
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
    reference = dexterous_reference_from_trajectory(
        retargeted,
        sequence_id=f"{PINNED_SEQUENCE_ID}_frames_{start_frame}_{end_frame_exclusive}",
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
        support_surface_names=("corn_can_support",),
        support_surface_asset_paths=(str(support),),
        support_surface_asset_sha256=(support_hash,),
        collision_asset_dependencies=collision_asset_dependencies,
        support_surface_scales=np.full((1, 3), local_geometry_scale, dtype=np.float64),
        support_surface_poses_w=projection.support_asset_pose_wxyz()[None, :],
        contacts=_inspection_contacts(len(qpos)),
        metadata=metadata,
    )
    reference.verify_scene_assets(require_hashes=True)
    return reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=PINNED_START_FRAME)
    parser.add_argument(
        "--end-frame-exclusive", type=int, default=PINNED_END_FRAME_EXCLUSIVE
    )
    parser.add_argument("--anchor-frame", type=int)
    parser.add_argument(
        "--object-anchor-target",
        type=float,
        nargs=3,
        default=tuple(DEFAULT_OBJECT_ANCHOR_TARGET),
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
    parser.add_argument("--max-wrist-position-error", type=float, default=0.05)
    parser.add_argument("--max-wrist-orientation-error-deg", type=float, default=180.0)
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
        model_path=args.model,
        support_path=args.support,
        start_frame=args.start_frame,
        end_frame_exclusive=args.end_frame_exclusive,
        anchor_frame=args.anchor_frame,
        object_anchor_target=np.asarray(args.object_anchor_target, dtype=np.float64),
        global_motion_scale=args.global_motion_scale,
        local_geometry_scale=args.local_geometry_scale,
        contact_dilation_frames=args.contact_dilation_frames,
        contact_blend_sigma_frames=args.contact_blend_sigma_frames,
        contact_seeking=args.contact_seeking,
        contact_settle_steps=args.contact_settle_steps,
        palm_object_standoff_m=args.palm_object_standoff_m,
        palm_support_standoff_m=args.palm_support_standoff_m,
        geometry_clearance_m=args.geometry_clearance_m,
        dense_geometry_audit_fps=args.dense_geometry_audit_fps,
        filter_cutoff_hz=args.filter_cutoff_hz,
        filter_order=args.filter_order,
        max_wrist_position_error=args.max_wrist_position_error,
        max_wrist_orientation_error=math.radians(args.max_wrist_orientation_error_deg),
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
