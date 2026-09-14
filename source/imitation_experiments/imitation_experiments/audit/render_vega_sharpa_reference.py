"""Render a mounted Reference or recorded Newton states for visual inspection."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from iltools.core import load_dexterous_reference_set
from imitation_experiments.data.vega_sharpa_scene import build_audit_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rollout",
        type=Path,
        help="Recorded Newton states from evaluate_vega_sharpa.py.",
    )
    parser.add_argument(
        "--trajectory-key", help="Trajectory key recorded in the evaluation JSON."
    )
    parser.add_argument(
        "--reference-overlay",
        action="store_true",
        help="Show the Reference box and final goal beside the recorded box motion.",
    )
    parser.add_argument("--label", default="Recorded policy rollout")
    parser.add_argument("--goal-position-tolerance-m", type=float, default=0.05)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    reference = load_dexterous_reference_set(args.reference)[0]
    qpos = reference.qpos
    object_poses = reference.object_poses_w[:, 0]
    reference_frames = np.arange(reference.frame_count)
    if args.rollout:
        if not args.trajectory_key:
            parser.error("--rollout requires --trajectory-key.")
        with np.load(args.rollout, allow_pickle=False) as rollout:
            qpos = rollout[f"{args.trajectory_key}__qpos"]
            object_poses = rollout[f"{args.trajectory_key}__object_pose_w"]
            reference_frames = rollout[f"{args.trajectory_key}__reference_frame"]
        if (
            qpos.ndim != 2
            or qpos.shape[1] != len(reference.joint_names)
            or object_poses.shape != (len(qpos), 7)
        ):
            raise ValueError("Recorded rollout dimensions differ from the Reference.")
        if not np.isfinite(qpos).all() or not np.isfinite(object_poses).all():
            raise ValueError("Cannot render non-finite rollout states.")
    if args.reference_overlay and not args.rollout:
        parser.error("--reference-overlay requires a recorded --rollout.")
    model = build_audit_model(
        Path(reference.metadata["mounted_robot_asset"]["model_path"]),
        reference.fixed_root_pose_w,
        reference.metadata["mounted_scene"],
    )
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.headlight.ambient[:] = 0.4
    data = mujoco.MjData(model)
    object_mocap = model.body_mocapid[model.body("object").id]
    object_geoms = np.flatnonzero(model.geom_bodyid == model.body("object").id)
    model.geom_rgba[object_geoms] = [0.95, 0.42, 0.08, 1.0]
    model.geom_matid[object_geoms] = -1
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.15, 0.8]
    camera.distance = 2.8
    camera.azimuth = -55
    camera.elevation = -12
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    if args.reference_overlay:
        from PIL import Image, ImageDraw, ImageFont
        from imitation_experiments.evaluation.object_task_metrics import box_pose_errors

        other_groups = set(
            model.geom_group[
                np.setdiff1d(np.arange(model.ngeom), object_geoms)
            ].tolist()
        )
        overlay_group = next(
            group for group in range(5, -1, -1) if group not in other_groups
        )
        model.geom_group[object_geoms] = overlay_group
        option.geomgroup[overlay_group] = 1
        ghost_option = mujoco.MjvOption()
        ghost_option.geomgroup[:] = 0
        ghost_option.geomgroup[overlay_group] = 1
        ghost_option.sitegroup[:] = 0
        ghost_data = mujoco.MjData(model)
        perturb = mujoco.MjvPerturb()
        com = reference.scene_physics.object_center_of_mass_m[0]
        path_errors = box_pose_errors(
            object_poses, reference.object_poses_w[reference_frames, 0], com
        )
        goal_errors = box_pose_errors(
            object_poses, reference.object_poses_w[-1, 0], com
        )
        goal_center = goal_errors["reference_center_position_w"][0]
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        with imageio.get_writer(
            args.output, fps=reference.fps, codec="libx264", quality=8
        ) as writer:
            for frame in range(len(qpos)):
                data.qpos[:] = qpos[frame]
                data.mocap_pos[object_mocap] = object_poses[frame, :3]
                data.mocap_quat[object_mocap] = object_poses[frame, 3:]
                mujoco.mj_forward(model, data)
                model.geom_rgba[object_geoms] = [0.95, 0.42, 0.08, 1.0]
                renderer.update_scene(data, camera=camera, scene_option=option)
                if args.reference_overlay:
                    ref_frame = int(reference_frames[frame])
                    ghost_data.qpos[:] = reference.qpos[ref_frame]
                    ghost_data.mocap_pos[object_mocap] = reference.object_poses_w[
                        ref_frame, 0, :3
                    ]
                    ghost_data.mocap_quat[object_mocap] = reference.object_poses_w[
                        ref_frame, 0, 3:
                    ]
                    mujoco.mj_kinematics(model, ghost_data)
                    model.geom_rgba[object_geoms] = [0.0, 0.75, 1.0, 0.28]
                    mujoco.mjv_addGeoms(
                        model,
                        ghost_data,
                        ghost_option,
                        perturb,
                        mujoco.mjtCatBit.mjCAT_ALL,
                        renderer.scene,
                    )
                    goal_marker = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        goal_marker,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([args.goal_position_tolerance_m, 0, 0]),
                        goal_center,
                        np.eye(3).reshape(-1),
                        np.array([0.1, 1.0, 0.25, 0.18]),
                    )
                    renderer.scene.ngeom += 1
                ground = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    ground,
                    mujoco.mjtGeom.mjGEOM_PLANE,
                    np.array([10.0, 10.0, 0.1]),
                    np.zeros(3),
                    np.eye(3).reshape(-1),
                    np.array([0.2, 0.23, 0.25, 1.0]),
                )
                renderer.scene.ngeom += 1
                pixels = renderer.render()
                if args.reference_overlay:
                    annotated = Image.fromarray(pixels)
                    draw = ImageDraw.Draw(annotated)
                    draw.rectangle((0, 0, 1280, 82), fill=(15, 20, 26))
                    draw.text(
                        (18, 8),
                        f"{args.label} | t={(frame + 1) / reference.fps:.2f} s",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (18, 34),
                        "Orange: actual box   Cyan: Reference box   Green: final center goal",
                        font=font,
                        fill="white",
                    )
                    draw.text(
                        (18, 58),
                        f"Box-center error: {100 * path_errors['center_position_error_m'][frame]:.1f} cm   Final-goal error: {100 * goal_errors['center_position_error_m'][frame]:.1f} cm",
                        font=font,
                        fill="white",
                    )
                    pixels = np.asarray(annotated)
                writer.append_data(pixels)
                if frame == 0:
                    imageio.imwrite(args.output.with_suffix(".png"), pixels)
    kind = (
        "Recorded Newton policy/zero-residual trajectory"
        if args.rollout
        else "Kinematic reference"
    )
    print(f"{kind} video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
