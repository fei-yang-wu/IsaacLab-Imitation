#!/usr/bin/env python3
"""Render a Vega-Wuji DexterousReference NPZ without qualification gating.

This renderer is intentionally kinematic: it is for comparing retargeted
posture candidates, including inspection-only candidates that Isaac's runtime
replay correctly refuses to load.  It renders the exact NPZ qpos and object
poses in the same MuJoCo scene used by the converter's geometry audit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from xml.sax.saxutils import quoteattr

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The converter imports MuJoCo, so choose the headless backend before importing
# any of its scene-building helpers.
os.environ.setdefault("MUJOCO_GL", "egl")

from scripts.data.convert_soma_g1_parquet_to_vega_wuji import (  # noqa: E402
    DEFAULT_GEOMETRY_CLEARANCE_M,
    DEFAULT_MODEL,
    PINNED_SUPPORT_HEIGHT_SOURCE,
    PINNED_SUPPORT_RADIUS_SOURCE,
    _SceneCollisionAuditor,
)


def _rotation_from_wxyz(values: np.ndarray) -> Rotation:
    quaternions = np.asarray(values, dtype=np.float64)
    return Rotation.from_quat(quaternions[..., (1, 2, 3, 0)])


def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return xyzw[..., (3, 0, 1, 2)]


def _poses_from_world_to_robot(
    poses_w: np.ndarray, root_pose_w: np.ndarray
) -> np.ndarray:
    poses = np.asarray(poses_w, dtype=np.float64).copy()
    root = np.asarray(root_pose_w, dtype=np.float64)
    root_rotation = _rotation_from_wxyz(root[3:7])
    poses[..., :3] = (
        root_rotation.inv()
        .apply((poses[..., :3] - root[:3]).reshape(-1, 3))
        .reshape(poses[..., :3].shape)
    )
    world_rotation = _rotation_from_wxyz(poses[..., 3:7].reshape(-1, 4))
    poses[..., 3:7] = _rotation_to_wxyz(root_rotation.inv() * world_rotation).reshape(
        poses[..., 3:7].shape
    )
    return poses


def _object_mesh_path(
    object_asset_path: Path, dependencies_json: str, *, object_index: int = 0
) -> Path:
    dependencies = json.loads(dependencies_json)
    for dependency in dependencies:
        if (
            dependency.get("asset_role") == "object"
            and int(dependency.get("asset_index", -1)) == object_index
        ):
            path = (object_asset_path.parent / str(dependency["uri"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Object collision mesh is missing: {path}")
            return path
    raise ValueError("Reference has no collision mesh dependency for object zero.")


def _presentation_scene_xml(
    *,
    robot_model_path: Path,
    object_mesh_path: Path,
    object_scale: float,
    support_center: np.ndarray,
    support_radius: float,
    support_height: float,
    floor_z: float,
) -> str:
    """Scene for looking at, not for auditing.

    The audit scene carries no lights or materials, which is correct for
    signed-distance queries and useless for judging a grasp: everything renders
    the same flat grey and the far hand disappears into the object. This builds
    the same bodies with a three-point light rig, shadows, a ground plane and
    distinguishable materials. It is a sibling of the audit scene, never a
    replacement, and nothing measured is read from it.
    """

    scale_text = " ".join([f"{object_scale:.17g}"] * 3)
    center_text = " ".join(f"{value:.17g}" for value in support_center)
    return (
        '<mujoco model="vega_wuji_presentation">'
        f"<include file={quoteattr(str(robot_model_path))}/>"
        "<visual>"
        # A headlight alone flattens the scene; keep it low and let the rig work.
        '<headlight ambient="0.38 0.39 0.43" diffuse="0.28 0.28 0.29" '
        'specular="0.05 0.05 0.05"/>'
        '<quality shadowsize="8192" offsamples="8"/>'
        '<map znear="0.02" zfar="40" shadowclip="6" shadowscale="1.4"/>'
        "</visual>"
        "<asset>"
        '<texture name="pres_sky" type="skybox" builtin="gradient" '
        'rgb1="0.26 0.30 0.37" rgb2="0.04 0.05 0.07" width="512" height="512"/>'
        '<texture name="pres_grid" type="2d" builtin="checker" '
        'rgb1="0.20 0.21 0.24" rgb2="0.15 0.16 0.19" width="512" height="512"/>'
        '<material name="pres_floor" texture="pres_grid" texrepeat="10 10" '
        'reflectance="0.10" specular="0.15" shininess="0.2"/>'
        '<material name="pres_box" rgba="0.74 0.57 0.35 1" specular="0.25" '
        'shininess="0.30"/>'
        '<material name="pres_table" rgba="0.26 0.28 0.33 1" specular="0.35" '
        'shininess="0.45" reflectance="0.06"/>'
        f'<mesh name="pres_can_mesh" file={quoteattr(str(object_mesh_path))} '
        f'scale="{scale_text}"/>'
        "</asset>"
        "<worldbody>"
        # Key light from the camera's upper left, the only shadow caster.
        '<light name="pres_key" directional="true" castshadow="true" '
        'pos="1.6 1.3 3.0" dir="-0.45 -0.38 -1" diffuse="0.72 0.70 0.67" '
        'specular="0.30 0.30 0.30"/>'
        # Cool fill from the opposite side lifts the shadowed hand.
        '<light name="pres_fill" directional="true" castshadow="false" '
        'pos="-1.8 -1.2 2.2" dir="0.5 0.34 -1" diffuse="0.26 0.28 0.34"/>'
        # Rim light separates the arms from the background.
        '<light name="pres_rim" directional="true" castshadow="false" '
        'pos="0.1 -2.4 1.8" dir="0 1 -0.35" diffuse="0.22 0.22 0.26"/>'
        '<geom name="pres_floor_geom" type="plane" size="8 8 0.1" '
        'material="pres_floor" contype="0" conaffinity="0" '
        f'pos="0 0 {floor_z:.17g}"/>'
        '<body name="audit_can_body" mocap="true">'
        '<geom name="audit_can_geom" type="mesh" mesh="pres_can_mesh" '
        'material="pres_box" contype="0" conaffinity="0"/>'
        "</body>"
        '<geom name="audit_support_geom" type="cylinder" material="pres_table" '
        f'size="{support_radius:.17g} {0.5 * support_height:.17g}" '
        f'pos="{center_text}" contype="0" conaffinity="0"/>'
        "</worldbody>"
        "</mujoco>"
    )


def _slerp_wxyz(first: np.ndarray, second: np.ndarray, blend: float) -> np.ndarray:
    """Shortest-arc interpolation between two wxyz quaternions."""

    start = _rotation_from_wxyz(np.asarray(first)[None, :])
    end = _rotation_from_wxyz(np.asarray(second)[None, :])
    relative = (start.inv() * end).as_rotvec()[0]
    return _rotation_to_wxyz(start * Rotation.from_rotvec(blend * relative))[0]


def _resample_timeline(
    qpos: np.ndarray,
    object_poses: np.ndarray,
    *,
    sample_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the reference onto arbitrary frame times.

    Repeating frames to stretch a clip looks stuttery at the exact moment a
    reviewer is trying to judge finger placement, so joints are interpolated
    linearly and object orientation along the shortest arc.
    """

    last = len(qpos) - 1
    times = np.clip(np.asarray(sample_times, dtype=np.float64), 0.0, float(last))
    lower = np.floor(times).astype(int)
    upper = np.minimum(lower + 1, last)
    blend = (times - lower)[:, None]
    out_qpos = qpos[lower] * (1.0 - blend) + qpos[upper] * blend
    out_poses = np.empty((len(times), *object_poses.shape[1:]), dtype=np.float64)
    for index, (low, high, ratio) in enumerate(zip(lower, upper, times - lower)):
        out_poses[index, :, :3] = (
            object_poses[low, :, :3] * (1.0 - ratio) + object_poses[high, :, :3] * ratio
        )
        for body in range(object_poses.shape[1]):
            out_poses[index, body, 3:7] = _slerp_wxyz(
                object_poses[low, body, 3:7],
                object_poses[high, body, 3:7],
                float(ratio),
            )
    return out_qpos, out_poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=3.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.55, 0.0, 0.85),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help=(
            "Total video length. The reference is interpolated onto the new "
            "timeline, so a longer value is smooth slow motion rather than "
            "repeated frames (default: the reference's own duration)."
        ),
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Video frame rate (default: the reference rate).",
    )
    parser.add_argument(
        "--orbit-deg",
        type=float,
        default=0.0,
        help=(
            "Sweep the camera this many degrees of azimuth across the clip. "
            "A slow orbit is the only way to see the far hand, which the "
            "object hides from any fixed side view."
        ),
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=0.0,
        help="Freeze the first and last frame for this long at each end.",
    )
    parser.add_argument(
        "--plain-scene",
        action="store_true",
        help="Render the unlit audit scene instead of the presentation scene.",
    )
    args = parser.parse_args()

    import imageio.v2 as imageio
    import mujoco

    reference_path = args.reference.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    with np.load(reference_path, allow_pickle=False) as data:
        joint_names = tuple(str(name) for name in data["joint_names"].tolist())
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        object_poses_w = np.asarray(data["object_poses_w"], dtype=np.float64)
        fixed_root_pose_w = np.asarray(data["fixed_root_pose_w"], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).item())
        object_asset_path = Path(str(data["object_asset_paths"][0])).resolve()
        object_scale = float(np.asarray(data["object_scales"])[0, 0])
        dependencies_json = str(
            np.asarray(data["collision_asset_dependencies_json"]).item()
        )
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))

    support_center = np.asarray(
        metadata["spatial_mapping"]["anchor_vertical_clearance"][
            "support_center_target_m"
        ],
        dtype=np.float64,
    )
    object_mesh = _object_mesh_path(object_asset_path, dependencies_json)
    scene = _SceneCollisionAuditor(
        robot_model_path=args.model.expanduser().resolve(),
        object_mesh_path=object_mesh,
        object_scale=object_scale,
        support_center=support_center,
        support_radius=PINNED_SUPPORT_RADIUS_SOURCE * object_scale,
        support_height=PINNED_SUPPORT_HEIGHT_SOURCE * object_scale,
        joint_names=joint_names,
        required_clearance_m=DEFAULT_GEOMETRY_CLEARANCE_M,
        include_visual_geometry=True,
        gate_scenes=(),
    )
    object_poses = _poses_from_world_to_robot(object_poses_w, fixed_root_pose_w)

    if args.width < 1 or args.height < 1 or args.stride < 1:
        raise ValueError("width, height, and stride must be positive.")
    render_model = scene.model
    render_data = scene.data
    qpos_addresses = scene._qpos_addresses
    can_mocap_id = scene._can_mocap_id
    if not args.plain_scene:
        # Place the ground just under the robot's lowest geom at rest so the
        # floor grounds the scene without clipping the base.
        mujoco.mj_resetData(scene.model, scene.data)
        scene.data.qpos[scene._qpos_addresses] = qpos[0]
        mujoco.mj_forward(scene.model, scene.data)
        floor_z = float(np.min(scene.data.geom_xpos[:, 2])) - 0.06
        render_model = mujoco.MjModel.from_xml_string(
            _presentation_scene_xml(
                robot_model_path=args.model.expanduser().resolve(),
                object_mesh_path=object_mesh,
                object_scale=object_scale,
                support_center=support_center,
                support_radius=PINNED_SUPPORT_RADIUS_SOURCE * object_scale,
                support_height=PINNED_SUPPORT_HEIGHT_SOURCE * object_scale,
                floor_z=floor_z,
            )
        )
        render_data = mujoco.MjData(render_model)
        qpos_addresses = np.asarray(
            [
                int(render_model.jnt_qposadr[render_model.joint(name).id])
                for name in joint_names
            ],
            dtype=np.int32,
        )
        can_mocap_id = int(
            render_model.body_mocapid[render_model.body("audit_can_body").id]
        )
    render_model.vis.global_.offwidth = max(
        args.width, render_model.vis.global_.offwidth
    )
    render_model.vis.global_.offheight = max(
        args.height, render_model.vis.global_.offheight
    )

    # Build the output timeline before rendering so the clip length, the frame
    # rate and the orbit are all defined against the same sample list.
    native_fps = float(fps) / float(args.stride)
    output_fps = float(args.output_fps) if args.output_fps else native_fps
    if output_fps <= 0.0:
        raise ValueError("output-fps must be positive.")
    source_times = np.arange(0, len(qpos), args.stride, dtype=np.float64)
    if args.duration_s is not None:
        if args.duration_s <= 0.0:
            raise ValueError("duration-s must be positive.")
        motion_frames = max(2, int(round(args.duration_s * output_fps)))
        source_times = np.linspace(0.0, float(len(qpos) - 1), motion_frames)
    qpos_seq, object_seq = _resample_timeline(
        qpos, object_poses, sample_times=source_times
    )
    hold = max(0, int(round(float(args.hold_s) * output_fps)))
    if hold:
        qpos_seq = np.concatenate(
            [
                np.repeat(qpos_seq[:1], hold, axis=0),
                qpos_seq,
                np.repeat(qpos_seq[-1:], hold, axis=0),
            ]
        )
        object_seq = np.concatenate(
            [
                np.repeat(object_seq[:1], hold, axis=0),
                object_seq,
                np.repeat(object_seq[-1:], hold, axis=0),
            ]
        )
    azimuths = np.full(len(qpos_seq), float(args.azimuth))
    if args.orbit_deg:
        azimuths = float(args.azimuth) + np.linspace(
            0.0, float(args.orbit_deg), len(qpos_seq)
        )

    renderer = mujoco.Renderer(render_model, width=args.width, height=args.height)
    camera = mujoco.MjvCamera()
    camera.distance = float(args.distance)
    camera.azimuth = float(args.azimuth)
    camera.elevation = float(args.elevation)
    camera.lookat[:] = np.asarray(args.lookat, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=max(1, int(round(output_fps))),
        codec="libx264",
        macro_block_size=None,
        quality=9,
    )
    try:
        for frame in range(len(qpos_seq)):
            mujoco.mj_resetData(render_model, render_data)
            render_data.qpos[qpos_addresses] = qpos_seq[frame]
            render_data.mocap_pos[can_mocap_id] = object_seq[frame, 0, :3]
            render_data.mocap_quat[can_mocap_id] = object_seq[frame, 0, 3:7]
            mujoco.mj_forward(render_model, render_data)
            camera.azimuth = float(azimuths[frame])
            renderer.update_scene(render_data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    print(
        f"Rendered {len(qpos_seq)} frames "
        f"({len(qpos_seq) / output_fps:.2f} s @ {output_fps:g} fps) to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
