#!/usr/bin/env python3
"""Render the original SOMA human, object, and contact phases from Parquet.

This is a source-side diagnostic: it reconstructs the SOMA-X body from the
embedded payload and applies only the dataset's SOMA-to-object-world bridge.
It does not use Vega-Wuji placement, IK, collision projection, or joint data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from iltools.retarget import infer_soma_frame_bridge, reconstruct_soma_motion


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Decode the primitive-container SOMA payload without loading classes."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"source_payload requests forbidden global {module}.{name}"
        )


def _load_row(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Rendering SOMA Parquet requires pyarrow.") from exc

    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"Expected one nested motion row, found {table.num_rows}.")
    return {name: table[name][0].as_py() for name in table.column_names}


def _load_payload(value: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(value, bytes):
        raise TypeError("source_payload must be bytes.")
    digest = hashlib.sha256(value).hexdigest()
    payload = _NoGlobalsUnpickler(io.BytesIO(value)).load()
    if not isinstance(payload, dict):
        raise TypeError("source_payload must decode to one dictionary.")
    required = {
        "soma_identity_coeffs",
        "soma_scale_params",
        "soma_joints",
        "soma_joints_wxyz",
        "nvhuman_root_translation",
        "nvhuman_root_wxyz",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"source_payload is missing fields: {missing}.")
    return payload, digest


def _linear_sample(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    lo = np.floor(indices).astype(np.int64)
    hi = np.minimum(lo + 1, len(source) - 1)
    alpha_shape = (len(indices),) + (1,) * (source.ndim - 1)
    alpha = (indices - lo).reshape(alpha_shape)
    return (1.0 - alpha) * source[lo] + alpha * source[hi]


def _object_rotations(wxyz: np.ndarray, indices: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    values = np.asarray(wxyz, dtype=np.float64)
    source_times = np.arange(len(values), dtype=np.float64)
    xyzw = values[:, (1, 2, 3, 0)]
    return Slerp(source_times, Rotation.from_quat(xyzw))(indices).as_matrix()


def _cylinder_segments(
    center: np.ndarray,
    rotation: np.ndarray,
    *,
    radius: float,
    half_height: float,
    count: int = 24,
) -> list[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    circles = []
    for z in (-half_height, half_height):
        local = np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.full(count, z))
        )
        circles.append(local @ rotation.T + center)
    segments = [np.vstack((circle, circle[:1])) for circle in circles]
    for index in range(0, count, 3):
        segments.append(np.vstack((circles[0][index], circles[1][index])))
    return segments


def _render(
    *,
    output: Path,
    joints: np.ndarray,
    body_points: np.ndarray,
    object_positions: np.ndarray,
    object_rotations: np.ndarray,
    contact_active: np.ndarray,
    joint_names: tuple[str, ...],
    parent_indices: np.ndarray,
    fps: float,
    source_indices: np.ndarray,
    source_fps: float,
    width: int,
    height: int,
    azimuth: float,
    elevation: float,
    object_radius: float,
    object_half_height: float,
    title_text: str,
) -> None:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor("#10141c")
    axis.set_facecolor("#10141c")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_proj_type("persp", focal_length=0.9)

    all_points = np.concatenate((joints.reshape(-1, 3), object_positions), axis=0)
    xy_center = 0.5 * (all_points[:, :2].min(axis=0) + all_points[:, :2].max(axis=0))
    xy_span = max(float(np.ptp(all_points[:, 0])), float(np.ptp(all_points[:, 1])))
    xy_radius = max(0.75, 0.65 * xy_span)
    z_min = min(0.0, float(all_points[:, 2].min()) - 0.05)
    z_max = float(all_points[:, 2].max()) + 0.12
    axis.set_xlim(xy_center[0] - xy_radius, xy_center[0] + xy_radius)
    axis.set_ylim(xy_center[1] - xy_radius, xy_center[1] + xy_radius)
    axis.set_zlim(z_min, z_max)
    axis.set_box_aspect((2.0 * xy_radius, 2.0 * xy_radius, z_max - z_min))
    axis.set_xlabel("world x [m]", color="#b7c2d0")
    axis.set_ylabel("world y [m]", color="#b7c2d0")
    axis.set_zlabel("world z [m]", color="#b7c2d0")
    axis.tick_params(colors="#7f8c9e", labelsize=8)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.08, 0.11, 0.16, 1.0))
        pane.set_edgecolor((0.25, 0.31, 0.40, 1.0))
    axis.grid(True, color="#273244", alpha=0.45)

    name_to_index = {name: index for index, name in enumerate(joint_names)}
    left_ids = np.asarray(
        [index for index, name in enumerate(joint_names) if name.startswith("Left")]
    )
    right_ids = np.asarray(
        [index for index, name in enumerate(joint_names) if name.startswith("Right")]
    )
    left_hand_id = name_to_index["LeftHand"]
    right_hand_id = name_to_index["RightHand"]

    body_artist = axis.scatter(
        [], [], [], s=1.0, color="#c8d0db", alpha=0.22, depthshade=True
    )
    joint_artist = axis.scatter(
        [], [], [], s=7.0, color="#f0f3f7", alpha=0.95, depthshade=False
    )
    left_artist = axis.scatter(
        [], [], [], s=10.0, color="#39a8ff", alpha=0.95, depthshade=False
    )
    right_artist = axis.scatter(
        [], [], [], s=10.0, color="#ff9d3a", alpha=0.95, depthshade=False
    )
    left_contact_artist = axis.scatter(
        [], [], [], s=110.0, facecolors="none", edgecolors="#59ff8a", linewidths=2.5
    )
    right_contact_artist = axis.scatter(
        [], [], [], s=110.0, facecolors="none", edgecolors="#59ff8a", linewidths=2.5
    )

    bone_artists = []
    for child, parent in enumerate(parent_indices):
        if parent < 0:
            continue
        child_name = joint_names[child]
        color = (
            "#39a8ff"
            if child_name.startswith("Left")
            else "#ff9d3a"
            if child_name.startswith("Right")
            else "#e8edf4"
        )
        (artist,) = axis.plot([], [], [], color=color, linewidth=2.0, alpha=0.95)
        bone_artists.append((child, int(parent), artist))

    object_artists = [
        axis.plot([], [], [], color="#ef3340", linewidth=2.5)[0] for _ in range(10)
    ]
    left_trace = axis.plot(
        joints[:, left_hand_id, 0],
        joints[:, left_hand_id, 1],
        joints[:, left_hand_id, 2],
        color="#39a8ff",
        linewidth=1.0,
        alpha=0.25,
    )[0]
    right_trace = axis.plot(
        joints[:, right_hand_id, 0],
        joints[:, right_hand_id, 1],
        joints[:, right_hand_id, 2],
        color="#ff9d3a",
        linewidth=1.0,
        alpha=0.25,
    )[0]
    object_trace = axis.plot(
        object_positions[:, 0],
        object_positions[:, 1],
        object_positions[:, 2],
        color="#ef3340",
        linewidth=1.5,
        alpha=0.35,
    )[0]
    del left_trace, right_trace, object_trace

    title = figure.text(
        0.02,
        0.965,
        title_text,
        color="white",
        fontsize=14,
        va="top",
    )
    status = figure.text(0.02, 0.925, "", color="#c9d3df", fontsize=10, va="top")
    legend = figure.text(
        0.02,
        0.035,
        "blue: left hand/arm   orange: right hand/arm   red: object   green ring: source contact active",
        color="#aeb9c7",
        fontsize=9,
    )
    del title, legend

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output),
        fps=max(1, int(round(fps))),
        codec="libx264",
        macro_block_size=None,
        quality=8,
    )
    try:
        for frame in range(len(joints)):
            posed = joints[frame]
            cloud = body_points[frame]
            body_artist._offsets3d = (cloud[:, 0], cloud[:, 1], cloud[:, 2])
            joint_artist._offsets3d = (posed[:, 0], posed[:, 1], posed[:, 2])
            left_artist._offsets3d = (
                posed[left_ids, 0],
                posed[left_ids, 1],
                posed[left_ids, 2],
            )
            right_artist._offsets3d = (
                posed[right_ids, 0],
                posed[right_ids, 1],
                posed[right_ids, 2],
            )
            for child, parent, artist in bone_artists:
                segment = posed[[parent, child]]
                artist.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])

            left_hand = posed[left_hand_id]
            right_hand = posed[right_hand_id]
            left_contact_artist._offsets3d = (
                [left_hand[0]],
                [left_hand[1]],
                [left_hand[2]],
            )
            right_contact_artist._offsets3d = (
                [right_hand[0]],
                [right_hand[1]],
                [right_hand[2]],
            )
            left_contact_artist.set_visible(bool(contact_active[frame, 0]))
            right_contact_artist.set_visible(bool(contact_active[frame, 1]))

            segments = _cylinder_segments(
                object_positions[frame],
                object_rotations[frame],
                radius=object_radius,
                half_height=object_half_height,
            )
            for artist, segment in zip(object_artists, segments, strict=True):
                artist.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])

            active_labels = [
                side
                for side, active in zip(("left", "right"), contact_active[frame])
                if active
            ]
            source_time = source_indices[frame] / source_fps
            status.set_text(
                f"source t={source_time:5.2f} s   frame={source_indices[frame]:6.1f}   "
                f"active contact: {', '.join(active_labels) if active_labels else 'none'}"
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.append_data(rgba[..., :3])
    finally:
        writer.close()
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame-exclusive", type=int, default=None)
    parser.add_argument("--output-fps", type=float, default=20.0)
    parser.add_argument("--playback-time-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--body-point-stride", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=-55.0)
    parser.add_argument("--elevation", type=float, default=12.0)
    parser.add_argument(
        "--title",
        default=None,
        help=(
            "Figure title. Default: 'Pre-retarget SOMA-X source: "
            "<sequence_id>' from the Parquet row."
        ),
    )
    args = parser.parse_args()

    source = args.parquet.expanduser().resolve()
    output = args.output.expanduser().resolve()
    row = _load_row(source)
    payload, payload_sha256 = _load_payload(row.get("source_payload"))
    source_fps = float(row["fps"])
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("Source fps must be finite and positive.")
    if not math.isfinite(args.output_fps) or args.output_fps <= 0.0:
        raise ValueError("Output fps must be finite and positive.")
    if not math.isfinite(args.playback_time_scale) or args.playback_time_scale <= 0.0:
        raise ValueError("playback-time-scale must be finite and positive.")
    if args.body_point_stride < 1:
        raise ValueError("body-point-stride must be positive.")

    source_joints = np.asarray(payload["soma_joints"], dtype=np.float64)
    source_wxyz = np.asarray(payload["soma_joints_wxyz"], dtype=np.float64)
    total = len(source_joints)
    stop = total if args.end_frame_exclusive is None else args.end_frame_exclusive
    if not 0 <= args.start_frame < stop <= total:
        raise ValueError(f"Crop [{args.start_frame}, {stop}) is outside [0, {total}).")
    crop = slice(args.start_frame, stop)

    from soma import SOMALayer

    layer = SOMALayer(
        device=args.device,
        identity_model_type="mhr",
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    joint_names = tuple(str(name) for name in layer.public_joint_names[1:])
    if source_joints.shape[1] != len(joint_names):
        raise ValueError(
            f"Payload has {source_joints.shape[1]} joints; SOMA-X has {len(joint_names)}."
        )
    reconstruction = reconstruct_soma_motion(
        global_joint_positions=source_joints[crop],
        global_joint_wxyz=source_wxyz[crop],
        identity_coeffs=np.asarray(payload["soma_identity_coeffs"]),
        scale_params=np.asarray(payload["soma_scale_params"]),
        joint_names=joint_names,
        layer=layer,
        device=args.device,
    )
    bridge = infer_soma_frame_bridge(
        soma_root_positions=source_joints[crop, 0],
        soma_root_wxyz=source_wxyz[crop, 0],
        target_root_positions=np.asarray(
            payload["nvhuman_root_translation"], dtype=np.float64
        )[crop],
        target_root_wxyz=np.asarray(payload["nvhuman_root_wxyz"], dtype=np.float64)[
            crop
        ],
    )

    source_count = stop - args.start_frame
    source_duration = (source_count - 1) / source_fps
    output_count = (
        int(math.floor(source_duration * args.playback_time_scale * args.output_fps))
        + 1
    )
    source_indices = (
        np.arange(output_count, dtype=np.float64)
        * source_fps
        / (args.output_fps * args.playback_time_scale)
    )
    source_indices = np.minimum(source_indices, source_count - 1.0)

    rotation = bridge.rotation
    joints_world = reconstruction.joints @ rotation.T + bridge.translations[:, None, :]
    sampled_vertices = reconstruction.vertices[:, :: args.body_point_stride]
    body_world = sampled_vertices @ rotation.T + bridge.translations[:, None, :]
    joints = _linear_sample(joints_world, source_indices)
    body_points = _linear_sample(body_world, source_indices)

    object_positions_source = np.asarray(row["object_body_position"], dtype=np.float64)[
        crop, 0
    ]
    object_wxyz_source = np.asarray(row["object_body_wxyz"], dtype=np.float64)[crop, 0]
    object_positions = _linear_sample(object_positions_source, source_indices)
    object_rotations = _object_rotations(object_wxyz_source, source_indices)
    contact_source = np.asarray(row["hand_contact_active"], dtype=np.float64)[:, crop].T
    contact_active = (
        contact_source[
            np.clip(np.rint(source_indices).astype(np.int64), 0, source_count - 1)
        ]
        > 0.5
    )

    parent_ids = layer.output_joint_parent_ids.detach().cpu().numpy()[1:]
    parent_indices = np.where(parent_ids == 0, -1, parent_ids - 1)
    object_radius = float(np.asarray(row.get("object_mesh_radius", [0.04]))[0])
    object_half_height = 0.0555
    mesh_paths = row.get("object_mesh_paths") or []
    if mesh_paths:
        try:
            import trimesh

            mesh = trimesh.load(
                Path(mesh_paths[0]).expanduser(), force="mesh", process=False
            )
            object_radius = 0.25 * float(mesh.extents[0] + mesh.extents[1])
            object_half_height = 0.5 * float(mesh.extents[2])
        except Exception:  # noqa: BLE001 - cylinder dimensions have a safe fallback
            pass

    _render(
        output=output,
        joints=joints,
        body_points=body_points,
        object_positions=object_positions,
        object_rotations=object_rotations,
        contact_active=contact_active,
        joint_names=joint_names,
        parent_indices=parent_indices,
        fps=args.output_fps,
        source_indices=source_indices + args.start_frame,
        source_fps=source_fps,
        width=args.width,
        height=args.height,
        azimuth=args.azimuth,
        elevation=args.elevation,
        object_radius=object_radius,
        object_half_height=object_half_height,
        title_text=(
            args.title
            if args.title is not None
            else f"Pre-retarget SOMA-X source: {row.get('sequence_id', source.name)}"
        ),
    )
    print(f"[SOMA] payload SHA-256: {payload_sha256}")
    print(f"[SOMA] reconstruction: {reconstruction.report.as_dict()}")
    print(
        f"[SOMA] wrote {output} ({output_count} frames @ {args.output_fps:g} fps, "
        f"{args.playback_time_scale:g}x slower than source)"
    )


if __name__ == "__main__":
    main()
