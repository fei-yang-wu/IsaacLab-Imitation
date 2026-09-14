#!/usr/bin/env python3
"""Render detached Wuji hands at pre-arm-IK wrist and finger targets.

The input is a diagnostic NPZ containing robot-frame WXYZ wrist poses,
side-local Wuji finger qpos, and object poses.  Rendering the hands without
arms distinguishes a bad wrist/hand target from a bad arm IK solution.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.convert_soma_g1_parquet_to_vega_wuji import DEFAULT_MODEL  # noqa: E402


def _rotation_from_wxyz(values: np.ndarray) -> Rotation:
    quaternions = np.asarray(values, dtype=np.float64)
    return Rotation.from_quat(quaternions[..., (1, 2, 3, 0)])


def _sample_linear(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    lower = np.floor(indices).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = (indices - lower).reshape((len(indices),) + (1,) * (source.ndim - 1))
    return (1.0 - alpha) * source[lower] + alpha * source[upper]


def _sample_poses_wxyz(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 7:
        raise ValueError("Pose targets must have shape [T, 7].")
    output = np.empty((len(indices), 7), dtype=np.float64)
    output[:, :3] = _sample_linear(source[:, :3], indices)
    rotations = Slerp(
        np.arange(len(source), dtype=np.float64),
        _rotation_from_wxyz(source[:, 3:7]),
    )(indices)
    output[:, 3:7] = rotations.as_quat()[..., (3, 0, 1, 2)]
    return output


def _sample_object_poses(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (1, 7):
        raise ValueError("Object poses must have shape [T, 1, 7].")
    return _sample_poses_wxyz(source[:, 0], indices)[:, None]


def _hand_topology(
    model: mujoco.MjModel, side: str
) -> tuple[tuple[int, ...], tuple[tuple[int | None, int], ...], tuple[int, ...]]:
    prefix = "l_" if side == "left" else "r_"
    body_ids = tuple(
        body_id
        for body_id in range(model.nbody)
        if model.body(body_id).name.startswith(prefix)
        and model.body(body_id).name not in {f"{prefix}mount"}
    )
    body_set = frozenset(body_ids)
    edges = tuple(
        (
            (
                int(model.body_parentid[body_id])
                if int(model.body_parentid[body_id]) in body_set
                else None
            ),
            body_id,
        )
        for body_id in body_ids
    )
    tip_site_ids = tuple(
        model.site(name).id
        for name in (
            f"{prefix}thumb_tip",
            f"{prefix}index_finger_tip",
            f"{prefix}middle_finger_tip",
            f"{prefix}ring_finger_tip",
            f"{prefix}pinky_tip",
        )
    )
    return body_ids, edges, tip_site_ids


def _local_hand_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
    finger_joint_names: tuple[str, ...],
    finger_qpos: np.ndarray,
    body_ids: tuple[int, ...],
    tip_site_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_resetData(model, data)
    for name, value in zip(finger_joint_names, finger_qpos, strict=True):
        address = int(model.jnt_qposadr[model.joint(name).id])
        data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)
    palm_site_id = model.site(f"{side}_palm").id
    palm_position = data.site_xpos[palm_site_id]
    palm_rotation = data.site_xmat[palm_site_id].reshape(3, 3)
    body_world = np.asarray([data.xpos[body_id] for body_id in body_ids])
    tip_world = np.asarray([data.site_xpos[site_id] for site_id in tip_site_ids])
    # MuJoCo rotation matrices map local column vectors into world.  Points are
    # represented as row vectors here, hence world-to-local multiplies by R.
    body_local = (body_world - palm_position) @ palm_rotation
    tip_local = (tip_world - palm_position) @ palm_rotation
    return body_local, tip_local


def _cylinder_segments(
    pose: np.ndarray, *, radius: float, half_height: float, count: int = 32
) -> tuple[np.ndarray, ...]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    circles = []
    rotation = _rotation_from_wxyz(pose[3:7]).as_matrix()
    for z in (-half_height, half_height):
        local = np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.full(count, z))
        )
        circles.append(local @ rotation.T + pose[:3])
    segments: list[np.ndarray] = [
        np.vstack((circles[0], circles[0][:1])),
        np.vstack((circles[1], circles[1][:1])),
    ]
    for index in range(0, count, 4):
        segments.append(np.vstack((circles[0][index], circles[1][index])))
    return tuple(segments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-fps", type=float, default=200.0)
    parser.add_argument("--output-fps", type=float, default=20.0)
    parser.add_argument("--playback-time-scale", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=-55.0)
    parser.add_argument("--elevation", type=float, default=22.0)
    parser.add_argument("--object-radius", type=float, default=0.0669213831)
    parser.add_argument("--object-half-height", type=float, default=0.0555)
    args = parser.parse_args()

    for name, value in (
        ("source_fps", args.source_fps),
        ("output_fps", args.output_fps),
        ("playback_time_scale", args.playback_time_scale),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.width < 1 or args.height < 1:
        raise ValueError("Render dimensions must be positive.")

    with np.load(args.targets.expanduser().resolve(), allow_pickle=False) as targets:
        joint_names = tuple(str(name) for name in targets["joint_names"])
        source_count = len(targets["left_wrist_pose_w"])
        duration = (source_count - 1) / args.source_fps
        output_count = (
            int(math.floor(duration * args.playback_time_scale * args.output_fps)) + 1
        )
        source_indices = (
            np.arange(output_count, dtype=np.float64)
            * args.source_fps
            / (args.output_fps * args.playback_time_scale)
        )
        source_indices = np.minimum(source_indices, source_count - 1.0)
        wrist_poses = {
            side: _sample_poses_wxyz(
                np.asarray(targets[f"{side}_wrist_pose_w"]), source_indices
            )
            for side in ("left", "right")
        }
        finger_qpos = {
            side: _sample_linear(
                np.asarray(targets[f"{side}_finger_qpos"]), source_indices
            )
            for side in ("left", "right")
        }
        object_poses = _sample_object_poses(
            np.asarray(targets["object_poses_w"]), source_indices
        )

    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser().resolve()))
    data = mujoco.MjData(model)
    side_joint_names = {
        side: tuple(
            name
            for name in joint_names
            if name.startswith("l_" if side == "left" else "r_")
        )
        for side in ("left", "right")
    }
    topology = {side: _hand_topology(model, side) for side in ("left", "right")}

    hand_points: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    hand_tips: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for frame in range(output_count):
        for side in ("left", "right"):
            body_ids, _, tip_site_ids = topology[side]
            body_local, tip_local = _local_hand_points(
                model,
                data,
                side=side,
                finger_joint_names=side_joint_names[side],
                finger_qpos=finger_qpos[side][frame],
                body_ids=body_ids,
                tip_site_ids=tip_site_ids,
            )
            pose = wrist_poses[side][frame]
            rotation = _rotation_from_wxyz(pose[3:7]).as_matrix()
            hand_points[side].append(body_local @ rotation.T + pose[:3])
            hand_tips[side].append(tip_local @ rotation.T + pose[:3])
    hand_points_array = {
        side: np.asarray(values) for side, values in hand_points.items()
    }
    hand_tips_array = {side: np.asarray(values) for side, values in hand_tips.items()}

    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    figure = plt.figure(figsize=(args.width / dpi, args.height / dpi), dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor("#10141c")
    axis.set_facecolor("#10141c")
    axis.view_init(elev=args.elevation, azim=args.azimuth)
    all_points = np.concatenate(
        (
            hand_points_array["left"].reshape(-1, 3),
            hand_points_array["right"].reshape(-1, 3),
            object_poses[:, 0, :3],
        )
    )
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    span = np.maximum(np.ptp(all_points, axis=0), (0.65, 0.65, 0.45))
    radius = 0.58 * float(np.max(span))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - 0.65 * radius, center[2] + 0.65 * radius)
    axis.set_box_aspect((1.0, 1.0, 0.65))
    axis.set_xlabel("robot x [m]", color="#b7c2d0")
    axis.set_ylabel("robot y [m]", color="#b7c2d0")
    axis.set_zlabel("robot z [m]", color="#b7c2d0")
    axis.tick_params(colors="#7f8c9e", labelsize=8)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.08, 0.11, 0.16, 1.0))
        pane.set_edgecolor((0.25, 0.31, 0.40, 1.0))

    colors = {"left": "#39a8ff", "right": "#ff9d3a"}
    point_artists = {
        side: axis.scatter([], [], [], s=18, color=colors[side], depthshade=False)
        for side in ("left", "right")
    }
    tip_artists = {
        side: axis.scatter([], [], [], s=45, color="#59ff8a", depthshade=False)
        for side in ("left", "right")
    }
    edge_artists: dict[str, list[object]] = {"left": [], "right": []}
    for side in ("left", "right"):
        _, edges, _ = topology[side]
        edge_artists[side] = [
            axis.plot([], [], [], color=colors[side], linewidth=2.0)[0] for _ in edges
        ]
    axis_artists = {
        side: [
            axis.plot([], [], [], color=color, linewidth=3.0)[0]
            for color in ("#ff4050", "#55e36a", "#4f83ff")
        ]
        for side in ("left", "right")
    }
    object_artists = [
        axis.plot([], [], [], color="#ef3340", linewidth=2.5)[0] for _ in range(10)
    ]
    status = figure.text(0.02, 0.95, "", color="white", fontsize=13, va="top")
    figure.text(
        0.02,
        0.04,
        "detached pre-IK targets | blue: left | orange: right | RGB: palm axes | green: fingertips",
        color="#b7c2d0",
        fontsize=10,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output),
        fps=max(1, int(round(args.output_fps))),
        codec="libx264",
        macro_block_size=None,
        quality=8,
    )
    try:
        for frame in range(output_count):
            for side in ("left", "right"):
                points = hand_points_array[side][frame]
                tips = hand_tips_array[side][frame]
                point_artists[side]._offsets3d = (
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                )
                tip_artists[side]._offsets3d = (
                    tips[:, 0],
                    tips[:, 1],
                    tips[:, 2],
                )
                body_ids, edges, _ = topology[side]
                body_index = {body_id: index for index, body_id in enumerate(body_ids)}
                palm = wrist_poses[side][frame, :3]
                for (parent, child), artist in zip(
                    edges, edge_artists[side], strict=True
                ):
                    child_point = points[body_index[child]]
                    parent_point = (
                        palm if parent is None else points[body_index[parent]]
                    )
                    segment = np.vstack((parent_point, child_point))
                    artist.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
                rotation = _rotation_from_wxyz(
                    wrist_poses[side][frame, 3:7]
                ).as_matrix()
                for axis_index, artist in enumerate(axis_artists[side]):
                    segment = np.vstack((palm, palm + 0.08 * rotation[:, axis_index]))
                    artist.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
            segments = _cylinder_segments(
                object_poses[frame, 0],
                radius=args.object_radius,
                half_height=args.object_half_height,
            )
            for artist, segment in zip(object_artists, segments, strict=True):
                artist.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
            status.set_text(
                f"pre-IK floating Wuji hand targets | output frame {frame}/{output_count - 1} | "
                f"source frame {source_indices[frame]:.1f}"
            )
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[..., :3])
    finally:
        writer.close()
        plt.close(figure)
    print(f"Rendered {output_count} floating-target frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
