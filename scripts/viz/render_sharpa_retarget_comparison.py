#!/usr/bin/env python3
"""Render source MANO motion next to its Sharpa retarget for visual inspection.

Left panel: the source MANO joints (spheres and bones) and the object.
Right panel: the retargeted Sharpa hands from the ILTools reference, driven
through the retarget MJCFs, and the same object. Both panels use one camera
and are synchronized by source time, so a frame on the left and its partner
on the right show the same instant.

Run through the ``isaaclab`` environment with EGL rendering:

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \\
    pixi run -e isaaclab python scripts/viz/render_sharpa_retarget_comparison.py \\
        --loaded-parquet /path/to/<ds>_loaded \\
        --manifest data/dexmanip/sharpa/<set>/manifest.json \\
        --output logs/rsl_rl/sharpa_v2d/retarget_videos/<set>.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = (
    REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets/sharpa_wave"
)
DEFAULT_MJCF = {
    side: ASSET_ROOT / "xmls" / "sharpawave" / f"{side}_sharpawave.xml"
    for side in ("left", "right")
}

# MANO joint order used by the video-to-data loader (21 joints).
MANO_BONES = [(0, 1), (1, 2), (2, 3), (3, 4)]
MANO_BONES += [(0, 5), (5, 6), (6, 7), (7, 8)]
MANO_BONES += [(0, 9), (9, 10), (10, 11), (11, 12)]
MANO_BONES += [(0, 13), (13, 14), (14, 15), (15, 16)]
MANO_BONES += [(0, 17), (17, 18), (18, 19), (19, 20)]
HAND_COLOR = {"right": (0.85, 0.25, 0.2, 1.0), "left": (0.2, 0.45, 0.9, 1.0)}
OBJECT_COLOR = (0.35, 0.65, 0.35, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loaded-parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-mjcf", type=Path, default=DEFAULT_MJCF["left"])
    parser.add_argument("--right-mjcf", type=Path, default=DEFAULT_MJCF["right"])
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=None,
        help="Object mesh (OBJ/STL, meters). Defaults to a cube of --object-half-size.",
    )
    parser.add_argument("--object-half-size", type=float, default=0.04)
    parser.add_argument("--rigid-object-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--distance", type=float, default=0.9)
    parser.add_argument("--azimuth", type=float, default=150.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument(
        "--ground-z",
        type=float,
        default=None,
        help="Ground-plane height. Defaults to 0.3 m below the lowest object pose.",
    )
    parser.add_argument(
        "--track-object",
        action="store_true",
        help="Keep the camera centered on the object instead of on its mean pose.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--contact-sheet-frames",
        type=int,
        default=6,
        help="Also write a PNG with this many evenly spaced frame pairs.",
    )
    return parser.parse_args()


def _wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()


def _resample_source(
    row: dict[str, Any], times: np.ndarray, object_index: int
) -> dict[str, np.ndarray]:
    """Linearly interpolate the source MANO joints and object pose at ``times``."""

    from scipy.spatial.transform import Rotation, Slerp

    frames = len(row["mano_right_joints"])
    source_times = np.arange(frames) / float(row["fps"])
    times = np.clip(times, source_times[0], source_times[-1])

    def linear(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(frames, -1)
        out = np.stack(
            [np.interp(times, source_times, flat[:, i]) for i in range(flat.shape[1])],
            axis=1,
        )
        return out.reshape((len(times),) + values.shape[1:])

    result = {}
    for side in ("left", "right"):
        result[f"{side}_joints"] = linear(np.asarray(row[f"mano_{side}_joints"]))
    position = np.asarray(row["object_body_position"])[:, object_index]
    wxyz = np.asarray(row["object_body_wxyz"])[:, object_index]
    result["object_position"] = linear(position)
    rotations = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
    result["object_matrix"] = Slerp(source_times, rotations)(times).as_matrix()
    return result


def _build_scene(
    args: argparse.Namespace, with_hands: bool, with_skeleton: bool
) -> tuple[Any, dict[str, Any]]:
    import mujoco

    spec = mujoco.MjSpec()
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.visual.global_.offwidth = args.width
    spec.visual.global_.offheight = args.height
    spec.worldbody.add_light(pos=[0.0, -1.0, 2.0], dir=[0.0, 0.4, -1.0])
    spec.worldbody.add_light(pos=[1.0, 1.0, 2.0], dir=[-0.4, -0.4, -1.0])
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.01],
        pos=[0.0, 0.0, float(args.ground_z or 0.0)],
        rgba=[0.9, 0.9, 0.9, 1],
    )
    handles: dict[str, Any] = {}

    if args.object_mesh is not None:
        spec.add_mesh(name="object_mesh", file=str(args.object_mesh.resolve()))
    object_body = spec.worldbody.add_body(name="object")
    object_body.add_freejoint(name="object_free")
    if args.object_mesh is not None:
        object_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname="object_mesh", rgba=OBJECT_COLOR
        )
    else:
        object_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[args.object_half_size] * 3,
            rgba=OBJECT_COLOR,
        )

    if with_hands:
        for side, path in (("left", args.left_mjcf), ("right", args.right_mjcf)):
            child = mujoco.MjSpec.from_file(str(path.resolve()))
            root = spec.worldbody.add_body(name=f"{side}_root")
            root.add_freejoint(name=f"{side}_free")
            frame = root.add_frame()
            # Prefix every attached name: both MJCFs share mesh names.
            frame.attach_body(child.worldbody.first_body(), f"{side}_", "")

    if with_skeleton:
        for side in ("left", "right"):
            for index in range(21):
                body = spec.worldbody.add_body(name=f"{side}_j{index}", mocap=True)
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.007, 0, 0],
                    rgba=HAND_COLOR[side],
                )
            for bone, (_a, _b) in enumerate(MANO_BONES):
                body = spec.worldbody.add_body(name=f"{side}_b{bone}", mocap=True)
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[0.004, 0.01, 0],
                    rgba=HAND_COLOR[side],
                )

    model = spec.compile()
    data = mujoco.MjData(model)
    handles["model"] = model
    handles["data"] = data
    return model, handles


def _set_object(
    model: Any, data: Any, position: np.ndarray, matrix: np.ndarray
) -> None:
    from scipy.spatial.transform import Rotation

    address = model.joint("object_free").qposadr[0]
    data.qpos[address : address + 3] = position
    quat = Rotation.from_matrix(matrix).as_quat()  # xyzw
    data.qpos[address + 3 : address + 7] = quat[[3, 0, 1, 2]]


def _set_skeleton(model: Any, data: Any, side: str, joints: np.ndarray) -> None:
    from scipy.spatial.transform import Rotation

    for index in range(21):
        body_id = model.body(f"{side}_j{index}").mocapid[0]
        data.mocap_pos[body_id] = joints[index]
    for bone, (a, b) in enumerate(MANO_BONES):
        start, end = joints[a], joints[b]
        body_id = model.body(f"{side}_b{bone}").mocapid[0]
        center = 0.5 * (start + end)
        direction = end - start
        length = float(np.linalg.norm(direction))
        data.mocap_pos[body_id] = center
        if length < 1.0e-6:
            continue
        z_axis = direction / length
        reference = np.array([0.0, 0.0, 1.0])
        axis = np.cross(reference, z_axis)
        sin = np.linalg.norm(axis)
        cos = float(np.dot(reference, z_axis))
        if sin < 1.0e-8:
            rotation = (
                Rotation.identity() if cos > 0 else Rotation.from_rotvec([np.pi, 0, 0])
            )
        else:
            rotation = Rotation.from_rotvec(axis / sin * np.arctan2(sin, cos))
        quat = rotation.as_quat()
        data.mocap_quat[body_id] = quat[[3, 0, 1, 2]]
        geom_id = model.body(f"{side}_b{bone}").geomadr[0]
        model.geom_size[geom_id, 1] = 0.5 * length


def _set_hand(
    model: Any,
    data: Any,
    side: str,
    wrist_pose: np.ndarray,
    joint_names: tuple[str, ...],
    qpos: np.ndarray,
) -> None:
    address = model.joint(f"{side}_free").qposadr[0]
    data.qpos[address : address + 7] = wrist_pose
    for name, value in zip(joint_names, qpos, strict=True):
        data.qpos[model.joint(f"{side}_{name}").qposadr[0]] = value


def _pose_only_forward(model: Any, data: Any) -> None:
    """Update body, geom, and site poses without running the dynamics pipeline.

    The full forward pass also builds and factorizes the constraint system. A
    retargeted hand rests inside the object's collision geometry, which makes
    that factorization rank deficient and aborts. Rendering needs poses only.
    """

    import mujoco

    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def _camera(args: argparse.Namespace, lookat: np.ndarray) -> Any:
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    return camera


def _label(image: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - optional
        return image
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 22], fill=(30, 30, 30))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return np.asarray(pil)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco
    from iltools.core import load_dexterous_reference_manifest
    from iltools.datasets import ManoSharpaLoader

    row = ManoSharpaLoader(args.loaded_parquet).rows()[0]
    manifest = load_dexterous_reference_manifest(args.manifest)
    reference = manifest.load_references(verify_hashes=True)[args.motion_index]
    resample = dict(reference.metadata.get("resample", {}))
    source_frames = int(resample.get("source_frames", len(row["mano_right_joints"])))
    source_fps = float(resample.get("source_fps", row["fps"]))
    duration = (source_frames - 1) / source_fps
    times = np.linspace(0.0, duration, reference.frame_count)
    source = _resample_source(row, times, args.rigid_object_index)

    object_positions = np.asarray(reference.object_poses_w)[:, 0, :3]
    if args.ground_z is None:
        args.ground_z = float(object_positions[:, 2].min() - 0.3)

    left_model, left = _build_scene(args, with_hands=False, with_skeleton=True)
    right_model, right = _build_scene(args, with_hands=True, with_skeleton=False)
    left_renderer = mujoco.Renderer(left_model, height=args.height, width=args.width)
    right_renderer = mujoco.Renderer(right_model, height=args.height, width=args.width)

    camera = _camera(args, object_positions.mean(axis=0))
    left_names = tuple(reference.left_joint_names)
    right_names = tuple(reference.right_joint_names)
    qpos = np.asarray(reference.qpos)
    left_count = len(left_names)

    frames_out = []
    start = max(0, int(args.start_frame))
    end = reference.frame_count if args.end_frame is None else int(args.end_frame)
    end = min(end, reference.frame_count)
    selected = list(range(start, end, args.stride))
    if not selected:
        raise SystemExit("The selected frame range is empty.")
    sheet_indices = set(
        np.asarray(selected)[
            np.linspace(0, len(selected) - 1, args.contact_sheet_frames).astype(int)
        ].tolist()
    )
    sheet = []
    for frame in selected:
        if args.track_object:
            camera.lookat[:] = object_positions[frame]
        _set_object(
            left_model,
            left["data"],
            source["object_position"][frame],
            source["object_matrix"][frame],
        )
        for side in ("left", "right"):
            _set_skeleton(
                left_model, left["data"], side, source[f"{side}_joints"][frame]
            )
        _pose_only_forward(left_model, left["data"])
        left_renderer.update_scene(left["data"], camera)
        left_image = _label(
            left_renderer.render().copy(),
            f"source MANO + object  t={times[frame]:.2f}s  frame {frame + 1}/{reference.frame_count}",
        )

        object_pose = np.asarray(reference.object_poses_w)[frame, 0]
        _set_object(
            right_model,
            right["data"],
            object_pose[:3],
            _wxyz_to_matrix(object_pose[3:7]),
        )
        _set_hand(
            right_model,
            right["data"],
            "left",
            np.asarray(reference.left_wrist_pose_w)[frame],
            left_names,
            qpos[frame, :left_count],
        )
        _set_hand(
            right_model,
            right["data"],
            "right",
            np.asarray(reference.right_wrist_pose_w)[frame],
            right_names,
            qpos[frame, left_count:],
        )
        _pose_only_forward(right_model, right["data"])
        right_renderer.update_scene(right["data"], camera)
        right_image = _label(
            right_renderer.render().copy(),
            f"Sharpa retarget ({manifest.dataset_name})  frame {frame + 1}/{reference.frame_count}",
        )
        composite = np.concatenate((left_image, right_image), axis=1)
        frames_out.append(composite)
        if frame in sheet_indices:
            sheet.append(composite)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(reference.fps / args.stride)))
    imageio.mimwrite(args.output, frames_out, fps=fps)
    print(f"wrote {args.output} ({len(frames_out)} frames at {fps} fps)")
    if sheet:
        sheet_path = args.output.with_suffix(".frames.png")
        imageio.imwrite(sheet_path, np.concatenate(sheet, axis=0))
        print(f"wrote {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
