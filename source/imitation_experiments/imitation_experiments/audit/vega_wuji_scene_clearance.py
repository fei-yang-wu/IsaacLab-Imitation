"""Audit Vega/Wuji Reference clearance with the source MuJoCo geometry.

This tool is an inspection aid. It does not qualify a Reference. MuJoCo uses
one convex hull for a mesh collision query. Newton can use a different convex
decomposition for the same mesh.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import quoteattr

import numpy as np

from imitation_experiments.paths import REPO_ROOT


_AUDIT_CAN_BODY = "audit_corn_can"
_AUDIT_CAN_GEOM = "audit_corn_can_collision"
_AUDIT_CAN_MESH = "audit_corn_can_mesh"
_AUDIT_SUPPORT_GEOM = "audit_support_cylinder"


@dataclass(frozen=True)
class Cylinder:
    """A cylinder pose and size in world coordinates."""

    center: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    radius: float
    half_height: float


@dataclass(frozen=True)
class PairClearance:
    """The minimum signed clearance for one frame and one geom set."""

    frame: int
    clearance_m: float
    robot_body: str | None
    robot_geom: str | None


def quat_wxyz_to_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Convert a unit WXYZ quaternion to a 3-by-3 rotation matrix."""

    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}.")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion norm must be finite and positive.")
    w, x, y, z = quat / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def quat_wxyz_multiply(
    left_wxyz: Sequence[float], right_wxyz: Sequence[float]
) -> np.ndarray:
    """Multiply two WXYZ quaternions."""

    left = np.asarray(left_wxyz, dtype=np.float64)
    right = np.asarray(right_wxyz, dtype=np.float64)
    if left.shape != (4,) or right.shape != (4,):
        raise ValueError("Both quaternions must have shape (4,).")
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def relative_pose_wxyz(
    parent_pose_wxyz: Sequence[float], child_pose_wxyz: Sequence[float]
) -> np.ndarray:
    """Express a world child pose in the parent pose frame."""

    parent = np.asarray(parent_pose_wxyz, dtype=np.float64)
    child = np.asarray(child_pose_wxyz, dtype=np.float64)
    if parent.shape != (7,) or child.shape != (7,):
        raise ValueError("Both poses must have shape (7,).")
    parent_quat = parent[3:] / np.linalg.norm(parent[3:])
    child_quat = child[3:] / np.linalg.norm(child[3:])
    position = quat_wxyz_to_matrix(parent_quat).T @ (child[:3] - parent[:3])
    orientation = quat_wxyz_multiply(
        (parent_quat[0], -parent_quat[1], -parent_quat[2], -parent_quat[3]),
        child_quat,
    )
    orientation /= np.linalg.norm(orientation)
    return np.concatenate((position, orientation))


def compose_scaled_translation(
    root_pose_wxyz: Sequence[float],
    scale_xyz: Sequence[float],
    local_translation: Sequence[float],
) -> np.ndarray:
    """Map a scaled local translation through a WXYZ root pose."""

    root_pose = np.asarray(root_pose_wxyz, dtype=np.float64)
    scale = np.asarray(scale_xyz, dtype=np.float64)
    local = np.asarray(local_translation, dtype=np.float64)
    if root_pose.shape != (7,) or scale.shape != (3,) or local.shape != (3,):
        raise ValueError("Expected pose (7,), scale (3,), and translation (3,).")
    return root_pose[:3] + quat_wxyz_to_matrix(root_pose[3:]) @ (scale * local)


def _match_usda_float(text: str, field: str) -> float:
    match = re.search(rf"\bdouble\s+{re.escape(field)}\s*=\s*([^\s]+)", text)
    if match is None:
        raise ValueError(f"USD file does not contain one double {field} field.")
    return float(match.group(1))


def _match_usda_translate(text: str) -> np.ndarray:
    match = re.search(r"\bdouble3\s+xformOp:translate\s*=\s*\(([^)]+)\)", text)
    if match is None:
        raise ValueError("USD file does not contain one xformOp:translate field.")
    values = np.fromstring(match.group(1), sep=",", dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("USD xformOp:translate must contain three values.")
    return values


def load_support_cylinder(
    asset_path: Path,
    root_pose_wxyz: Sequence[float],
    scale_xyz: Sequence[float],
) -> Cylinder:
    """Load the one Z-axis cylinder used by the current support USD."""

    text = asset_path.read_text(encoding="utf-8")
    cylinder_count = len(re.findall(r'\bdef\s+Cylinder\s+"', text))
    if cylinder_count != 1:
        raise ValueError(f"Expected one USD Cylinder, found {cylinder_count}.")
    if not re.search(r'\buniform\s+token\s+axis\s*=\s*"Z"', text):
        raise ValueError("The support audit requires a Z-axis USD cylinder.")
    scale = np.asarray(scale_xyz, dtype=np.float64)
    if scale.shape != (3,) or not np.allclose(scale, scale[0], rtol=0.0, atol=1.0e-9):
        raise ValueError("The support audit requires a uniform cylinder scale.")
    pose = np.asarray(root_pose_wxyz, dtype=np.float64)
    center = compose_scaled_translation(pose, scale, _match_usda_translate(text))
    return Cylinder(
        center=tuple(float(value) for value in center),
        quat_wxyz=tuple(float(value) for value in pose[3:]),
        radius=float(scale[0] * _match_usda_float(text, "radius")),
        half_height=float(0.5 * scale[0] * _match_usda_float(text, "height")),
    )


def resolve_urdf_collision_mesh(urdf_path: Path) -> Path:
    """Resolve the one collision mesh in an object URDF."""

    root = ET.parse(urdf_path).getroot()
    meshes = root.findall("./link/collision/geometry/mesh")
    if len(meshes) != 1:
        raise ValueError(f"Expected one URDF collision mesh, found {len(meshes)}.")
    filename = meshes[0].get("filename")
    if not filename:
        raise ValueError("URDF collision mesh does not have a filename.")
    mesh_path = Path(filename)
    if not mesh_path.is_absolute():
        mesh_path = urdf_path.parent / mesh_path
    return mesh_path.resolve()


def load_obj_vertices(obj_path: Path) -> np.ndarray:
    """Load only geometric vertices from an OBJ file."""

    vertices: list[tuple[float, float, float]] = []
    with obj_path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            values = line.split()
            if len(values) < 4:
                raise ValueError(f"Invalid OBJ vertex line: {line.rstrip()}")
            vertices.append((float(values[1]), float(values[2]), float(values[3])))
    if not vertices:
        raise ValueError(f"OBJ file has no vertices: {obj_path}")
    return np.asarray(vertices, dtype=np.float64)


def transform_points_wxyz(points: np.ndarray, pose_wxyz: Sequence[float]) -> np.ndarray:
    """Transform local points to world coordinates with one WXYZ pose."""

    pose = np.asarray(pose_wxyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or pose.shape != (7,):
        raise ValueError("Expected points (N, 3) and pose (7,).")
    return points @ quat_wxyz_to_matrix(pose[3:]).T + pose[:3]


def cylinder_local_points(points_w: np.ndarray, cylinder: Cylinder) -> np.ndarray:
    """Transform world points to the support cylinder frame."""

    rotation = quat_wxyz_to_matrix(cylinder.quat_wxyz)
    return (points_w - np.asarray(cylinder.center, dtype=np.float64)) @ rotation


def mesh_above_cylinder_clearance(
    points_w: np.ndarray, cylinder: Cylinder
) -> tuple[float, float]:
    """Return vertical clearance and the largest radial vertex position.

    This result is exact only when all mesh vertices are above the top cap and
    inside its radial disk. The caller must check the returned radial value.
    """

    local = cylinder_local_points(points_w, cylinder)
    max_radius = float(np.max(np.linalg.norm(local[:, :2], axis=1)))
    clearance = float(np.min(local[:, 2]) - cylinder.half_height)
    return clearance, max_radius


def _name(mujoco: Any, model: Any, object_type: Any, object_id: int) -> str:
    value = mujoco.mj_id2name(model, object_type, object_id)
    return "" if value is None else str(value)


def _geom_label(mujoco: Any, model: Any, geom_id: int) -> tuple[str, str]:
    body_id = int(model.geom_bodyid[geom_id])
    body_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    geom_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if geom_name:
        return body_name, geom_name
    mesh_id = int(model.geom_dataid[geom_id])
    if (
        int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        and mesh_id >= 0
    ):
        mesh_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        return body_name, f"mesh:{mesh_name}"
    type_name = (
        mujoco.mjtGeom(int(model.geom_type[geom_id]))
        .name.removeprefix("mjGEOM_")
        .lower()
    )
    return body_name, f"{type_name}:geom_{geom_id}"


def _minimum_pair(
    mujoco: Any,
    model: Any,
    data: Any,
    frame: int,
    robot_geom_ids: Sequence[int],
    scene_geom_id: int,
    distance_limit_m: float,
) -> PairClearance:
    best_distance = distance_limit_m
    best_geom_id = -1
    closest_points = np.empty(6, dtype=np.float64)
    for geom_id in robot_geom_ids:
        distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                int(geom_id),
                int(scene_geom_id),
                distance_limit_m,
                closest_points,
            )
        )
        if distance < best_distance:
            best_distance = distance
            best_geom_id = int(geom_id)
    if best_geom_id < 0:
        return PairClearance(frame, distance_limit_m, None, None)
    body_name, geom_name = _geom_label(mujoco, model, best_geom_id)
    return PairClearance(frame, best_distance, body_name, geom_name)


def _threshold_summary(
    rows: Sequence[PairClearance], distance_limit_m: float
) -> dict[str, Any]:
    deepest = min(rows, key=lambda row: row.clearance_m)
    violations = [row for row in rows if row.clearance_m + 1.0e-9 < distance_limit_m]
    penetrating = [row for row in rows if row.clearance_m < 0.0]
    passed = not violations
    return {
        "distance_cutoff_m": distance_limit_m,
        "passed": passed,
        "minimum_clearance_lower_bound_m": distance_limit_m if passed else None,
        "minimum_query_value_m": deepest.clearance_m,
        "minimum_query_below_cutoff": (
            asdict(deepest) if deepest.clearance_m < distance_limit_m else None
        ),
        "clearance_violation_frame_count": len(violations),
        "clearance_violations": [asdict(row) for row in violations],
        "penetrating_frame_count": len(penetrating),
        "penetrations": [asdict(row) for row in penetrating],
    }


def _exact_summary(rows: Sequence[PairClearance]) -> dict[str, Any]:
    deepest = min(rows, key=lambda row: row.clearance_m)
    penetrating = [row for row in rows if row.clearance_m < 0.0]
    return {
        "minimum": asdict(deepest),
        "penetrating_frame_count": len(penetrating),
        "penetrations": [asdict(row) for row in penetrating],
    }


def _reference_scalar_text(reference: Any, key: str) -> str:
    value = reference[key]
    if value.shape != ():
        raise ValueError(f"Reference field {key} must be a scalar.")
    return str(value.item())


def _xml_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _compile_xml_include_model(
    mujoco: Any, robot_mjcf_path: Path, extra_sections: str
) -> Any:
    """Compile a robot plus extra XML without an MjSpec round trip.

    MuJoCo 3.8 can return a false zero from ``mj_geomDistance`` for a disabled
    mesh geom after an ``MjSpec.from_file`` model is extended and compiled.
    The XML include path keeps the mesh collision data valid.
    """

    robot_attribute = quoteattr(str(robot_mjcf_path.resolve()))
    xml = (
        '<mujoco model="vega_wuji_scene_clearance_audit">'
        f"<include file={robot_attribute}/>"
        f"{extra_sections}"
        "</mujoco>"
    )
    return mujoco.MjModel.from_xml_string(xml)


def _compile_audit_model(
    mujoco: Any,
    robot_mjcf_path: Path,
    object_obj: Path,
    object_scale: Sequence[float],
    support: Cylinder,
) -> Any:
    object_attribute = quoteattr(str(object_obj.resolve()))
    extra_sections = f"""
<asset>
  <mesh name="{_AUDIT_CAN_MESH}" file={object_attribute}
        scale="{_xml_vector(object_scale)}"/>
</asset>
<worldbody>
  <geom name="{_AUDIT_SUPPORT_GEOM}" type="cylinder"
        pos="{_xml_vector(support.center)}"
        size="{support.radius:.17g} {support.half_height:.17g}"
        contype="1" conaffinity="1" density="0"/>
  <body name="{_AUDIT_CAN_BODY}" mocap="true">
    <geom name="{_AUDIT_CAN_GEOM}" type="mesh" mesh="{_AUDIT_CAN_MESH}"
          contype="1" conaffinity="1"/>
  </body>
</worldbody>
"""
    return _compile_xml_include_model(mujoco, robot_mjcf_path, extra_sections)


def audit_reference(reference_path: Path, robot_mjcf_path: Path) -> dict[str, Any]:
    """Run the full offline robot, can, and support clearance audit."""

    import mujoco

    with np.load(reference_path, allow_pickle=False) as reference:
        qpos = np.asarray(reference["qpos"], dtype=np.float64)
        joint_names = tuple(str(value) for value in reference["joint_names"].tolist())
        root_pose = np.asarray(reference["fixed_root_pose_w"], dtype=np.float64)
        object_names = tuple(str(value) for value in reference["object_names"].tolist())
        object_poses = np.asarray(reference["object_poses_w"], dtype=np.float64)
        object_scales = np.asarray(reference["object_scales"], dtype=np.float64)
        object_paths = tuple(
            str(value) for value in reference["object_asset_paths"].tolist()
        )
        support_names = tuple(
            str(value) for value in reference["support_surface_names"].tolist()
        )
        support_paths = tuple(
            str(value) for value in reference["support_surface_asset_paths"].tolist()
        )
        support_scales = np.asarray(
            reference["support_surface_scales"], dtype=np.float64
        )
        support_poses = np.asarray(
            reference["support_surface_poses_w"], dtype=np.float64
        )
        fps = float(reference["fps"].item())
        sequence_id = _reference_scalar_text(reference, "sequence_id")
        metadata = json.loads(_reference_scalar_text(reference, "metadata_json"))

    if object_names != ("corn_can",) or len(object_paths) != 1:
        raise ValueError(f"Expected only object ('corn_can',), got {object_names}.")
    if len(support_names) != 1 or len(support_paths) != 1:
        raise ValueError("Expected exactly one support surface.")
    if qpos.ndim != 2 or qpos.shape[1] != len(joint_names):
        raise ValueError("Reference qpos shape does not match joint_names.")
    if object_poses.shape != (qpos.shape[0], 1, 7):
        raise ValueError("Reference object_poses_w shape does not match qpos frames.")
    try:
        geometry_audit = metadata["geometry_clearance_audit"]
        required_clearance_m = float(geometry_audit["required_clearance_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Reference metadata must declare geometry_clearance_audit."
        ) from error
    if not np.isfinite(required_clearance_m) or required_clearance_m <= 0.0:
        raise ValueError("Declared required clearance must be finite and positive.")

    object_urdf = Path(object_paths[0]).resolve()
    object_obj = resolve_urdf_collision_mesh(object_urdf)
    support = load_support_cylinder(
        Path(support_paths[0]).resolve(), support_poses[0], support_scales[0]
    )
    object_poses_model = np.stack(
        [relative_pose_wxyz(root_pose, pose[0]) for pose in object_poses]
    )
    support_pose_model = relative_pose_wxyz(
        root_pose, (*support.center, *support.quat_wxyz)
    )
    support_axis_model = quat_wxyz_to_matrix(support_pose_model[3:]) @ np.asarray(
        [0.0, 0.0, 1.0]
    )
    if not np.allclose(support_axis_model, (0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-8):
        raise ValueError(
            "The support cylinder axis is not Z after conversion to the robot frame."
        )
    support_model = Cylinder(
        center=tuple(float(value) for value in support_pose_model[:3]),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        radius=support.radius,
        half_height=support.half_height,
    )

    model = _compile_audit_model(
        mujoco, robot_mjcf_path, object_obj, object_scales[0], support_model
    )
    data = mujoco.MjData(model)

    can_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _AUDIT_CAN_GEOM)
    support_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, _AUDIT_SUPPORT_GEOM
    )
    can_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _AUDIT_CAN_BODY)
    can_mocap_id = int(model.body_mocapid[can_body_id])
    if can_mocap_id < 0:
        raise ValueError("Audit can body is not a mocap body.")
    joint_addresses: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Robot MJCF does not have Reference joint {joint_name}.")
        if int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(f"Reference joint {joint_name} is not scalar.")
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))

    robot_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id not in (can_geom_id, support_geom_id)
    ]
    left_hand_active: list[int] = []
    right_hand_active: list[int] = []
    intended_robot: list[int] = []
    for geom_id in robot_geom_ids:
        body_id = int(model.geom_bodyid[geom_id])
        body_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        is_active = bool(
            int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
        )
        if body_name.startswith("l_") and is_active:
            left_hand_active.append(geom_id)
        if body_name.startswith("r_") and is_active:
            right_hand_active.append(geom_id)
        if is_active or int(model.geom_group[geom_id]) == 4:
            intended_robot.append(geom_id)

    categories: dict[str, list[PairClearance]] = {
        "left_hand_can": [],
        "right_hand_can": [],
        "left_hand_support": [],
        "right_hand_support": [],
        "intended_robot_can": [],
        "intended_robot_support": [],
    }
    can_support_rows: list[PairClearance] = []
    analytic_rows: list[PairClearance] = []
    analytic_max_radial: list[float] = []
    scaled_vertices = load_obj_vertices(object_obj) * object_scales[0]
    closest_points = np.empty(6, dtype=np.float64)

    for frame in range(qpos.shape[0]):
        mujoco.mj_resetData(model, data)
        data.qpos[joint_addresses] = qpos[frame]
        data.mocap_pos[can_mocap_id] = object_poses_model[frame, :3]
        data.mocap_quat[can_mocap_id] = object_poses_model[frame, 3:7]
        mujoco.mj_forward(model, data)
        categories["left_hand_can"].append(
            _minimum_pair(
                mujoco,
                model,
                data,
                frame,
                left_hand_active,
                can_geom_id,
                required_clearance_m,
            )
        )
        categories["right_hand_can"].append(
            _minimum_pair(
                mujoco,
                model,
                data,
                frame,
                right_hand_active,
                can_geom_id,
                required_clearance_m,
            )
        )
        categories["left_hand_support"].append(
            _minimum_pair(
                mujoco,
                model,
                data,
                frame,
                left_hand_active,
                support_geom_id,
                required_clearance_m,
            )
        )
        categories["right_hand_support"].append(
            _minimum_pair(
                mujoco,
                model,
                data,
                frame,
                right_hand_active,
                support_geom_id,
                required_clearance_m,
            )
        )
        categories["intended_robot_can"].append(
            _minimum_pair(
                mujoco,
                model,
                data,
                frame,
                intended_robot,
                can_geom_id,
                required_clearance_m,
            )
        )
        categories["intended_robot_support"].append(
            _minimum_pair(
                mujoco,
                model,
                data,
                frame,
                intended_robot,
                support_geom_id,
                required_clearance_m,
            )
        )
        can_support_distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                can_geom_id,
                support_geom_id,
                required_clearance_m,
                closest_points,
            )
        )
        can_support_rows.append(PairClearance(frame, can_support_distance, None, None))
        points_w = transform_points_wxyz(scaled_vertices, object_poses[frame, 0])
        analytic_clearance, max_radial = mesh_above_cylinder_clearance(
            points_w, support
        )
        analytic_rows.append(PairClearance(frame, analytic_clearance, None, None))
        analytic_max_radial.append(max_radial)

    category_summaries = {
        name: _threshold_summary(rows, required_clearance_m)
        for name, rows in categories.items()
    }
    robot_scene_gate_passed = bool(
        category_summaries["intended_robot_can"]["passed"]
        and category_summaries["intended_robot_support"]["passed"]
    )
    return {
        "audit_kind": "offline_inspection_not_qualification",
        "reference_path": str(reference_path.resolve()),
        "robot_mjcf_path": str(robot_mjcf_path.resolve()),
        "sequence_id": sequence_id,
        "fps": fps,
        "frame_count": int(qpos.shape[0]),
        "object_urdf_path": str(object_urdf),
        "object_obj_path": str(object_obj),
        "support": asdict(support),
        "required_clearance_m": required_clearance_m,
        "robot_scene_threshold_gate_passed": robot_scene_gate_passed,
        "geom_sets": {
            "left_authored_active": len(left_hand_active),
            "right_authored_active": len(right_hand_active),
            "intended_robot": len(intended_robot),
            "note": (
                "Authored-active selects nonzero MuJoCo collision masks on Wuji. "
                "Intended-robot also adds Vega group-4 collision geoms, whose "
                "source MuJoCo masks are zero."
            ),
        },
        "clearance_method": (
            "Fail-closed MuJoCo mj_geomDistance query capped at the Reference "
            "required clearance. A value at the cutoff is only a certified lower "
            "bound. Mesh queries use MuJoCo convex hulls and can differ from "
            "Newton convex decomposition."
        ),
        "categories": category_summaries,
        "reference_embedded_emitted_geometry_audit": geometry_audit.get("emitted_20hz"),
        "can_support": {
            "mujoco_convex_hull": _exact_summary(can_support_rows),
            "obj_vertex_vertical": _exact_summary(analytic_rows),
            "all_vertices_inside_support_disk": bool(
                max(analytic_max_radial) <= support.radius
            ),
            "maximum_vertex_radius_in_support_frame_m": max(analytic_max_radial),
            "support_radius_m": support.radius,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference", type=Path, help="Current-schema Vega/Wuji Reference NPZ."
    )
    parser.add_argument(
        "--robot-mjcf",
        type=Path,
        default=(
            REPO_ROOT
            / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
            / "vega_u_wuji_v2_beta1_with_mount.xml"
        ),
        help="Integrated Vega/Wuji MuJoCo model.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line audit."""

    args = _build_parser().parse_args(argv)
    result = audit_reference(args.reference, args.robot_mjcf)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
