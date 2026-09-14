"""Mechanical checks and a short rollout for the mounted Isaac task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Imitation-Vega-Sharpa-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unassisted", action="store_true")
    args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + hydra_args
    args.headless = True

    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_imitation.tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation, resolve_task_config
    from isaaclab_imitation.tasks.manager_based.dexmanip.sharpa_command import _torch
    from iltools.core import load_dexterous_reference_set, sha256_file

    cfg, _ = resolve_task_config(args.task, "rlopt_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.commands.sharpa_reference.always_reset_to_first_frame = True
    if args.unassisted:
        cfg.curriculum.fixed_timestep = None
        cfg.commands.sharpa_reference.initial_virtual_object_control_curriculum_scale = 0.0
    reference = load_dexterous_reference_set(cfg.reference.manifest)[0]
    result = {
        "qualification": "local mechanical and integration checks; not policy convergence",
        "task": args.task,
        "rollout_assistance": "disabled" if args.unassisted else "source curriculum",
        "num_envs": args.num_envs,
        "reference_sha256": sha256_file(reference.source_path),
        "robot_asset": reference.metadata["mounted_robot_asset"],
        "checks": {},
        "failures": [],
    }

    def check(name, passed, **evidence):
        result["checks"][name] = {"passed": bool(passed), **evidence}
        if not passed:
            result["failures"].append(name)
        print(name, result["checks"][name], flush=True)

    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            env.reset()
            command = env.command_manager.get_term("sharpa_reference")
            if type(command.robot).__module__.startswith("isaaclab_newton"):
                from isaaclab_newton.physics import NewtonManager

                model = NewtonManager.get_model()
                check(
                    "independent_physics_worlds",
                    model.world_count == env.num_envs,
                    world_count=model.world_count,
                    environment_count=env.num_envs,
                )
                from newton import ShapeFlags
                import warp as wp

                labels = list(model.body_label)
                robot_ids = [i for i, label in enumerate(labels) if "/robot/" in label]
                object_ids = [
                    i for i, label in enumerate(labels) if label.endswith("/object")
                ]
                gravity = wp.to_torch(model.mujoco.gravcomp)
                check(
                    "robot_gravity_compensation",
                    bool(robot_ids and object_ids)
                    and bool((gravity[robot_ids] == 1).all())
                    and bool((gravity[object_ids] == 0).all()),
                    robot_body_count=len(robot_ids),
                    compensated_robot_bodies=int((gravity[robot_ids] == 1).sum()),
                )
                shape_bodies = wp.to_torch(model.shape_body)
                colliding = (
                    wp.to_torch(model.shape_flags) & int(ShapeFlags.COLLIDE_SHAPES)
                ) != 0
                required = {
                    f"{side}_arm_l{i}" for side in ("L", "R") for i in range(1, 9)
                }
                arm_bodies = [
                    i
                    for i, label in enumerate(labels)
                    if label.rsplit("/", 1)[-1] in required
                ]
                missing = [
                    labels[i]
                    for i in arm_bodies
                    if not bool(((shape_bodies == i) & colliding).any())
                ]
                check(
                    "backbone_collision_shapes",
                    len(arm_bodies) == 16 * env.num_envs and not missing,
                    arm_body_count=len(arm_bodies),
                    missing_collision_bodies=missing,
                )
                # This checks the imported model, not just authored USD data.
                # The torso is welded through arm_center to the first shoulder
                # joint; their housing overlap must remain structurally filtered.
                shape_body_ids = model.shape_body.numpy()
                excluded = model.shape_collision_filter_pairs
                missing_shoulder_filters = []
                for body in arm_bodies:
                    if labels[body].rsplit("/", 1)[-1] not in ("L_arm_l1", "R_arm_l1"):
                        continue
                    torso = labels.index(labels[body].split("/arm_center/")[0])
                    for a in np.flatnonzero(shape_body_ids == body):
                        for b in np.flatnonzero(shape_body_ids == torso):
                            if (
                                colliding[a]
                                and colliding[b]
                                and (min(a, b), max(a, b)) not in excluded
                            ):
                                missing_shoulder_filters.append([int(a), int(b)])
                check(
                    "structural_shoulder_collision_exclusions",
                    not missing_shoulder_filters,
                    missing_pair_count=len(missing_shoulder_filters),
                )
                from scipy.spatial.transform import Rotation

                body_poses = NewtonManager.get_state_0().body_q.numpy()
                shape_poses = model.shape_transform.numpy()
                shape_scales = model.shape_scale.numpy()
                base_bodies = [
                    i
                    for i, label in enumerate(labels)
                    if label.endswith("/robot/Geometry/base_link")
                ]
                base_bottoms = []
                for body in base_bodies:
                    bottoms = []
                    for shape in np.flatnonzero(shape_body_ids == body):
                        if not colliding[shape]:
                            continue
                        vertices = (
                            np.asarray(model.shape_source[shape].vertices)
                            * shape_scales[shape]
                        )
                        local = (
                            Rotation.from_quat(shape_poses[shape, 3:]).apply(vertices)
                            + shape_poses[shape, :3]
                        )
                        world = (
                            Rotation.from_quat(body_poses[body, 3:]).apply(local)
                            + body_poses[body, :3]
                        )
                        bottoms.append(float(world[:, 2].min()))
                    if bottoms:
                        base_bottoms.append(min(bottoms))
                check(
                    "base_ground_alignment",
                    len(base_bottoms) == env.num_envs
                    and max(abs(z) for z in base_bottoms) < 0.001,
                    measured_base_count=len(base_bottoms),
                    minimum_base_z_m=min(base_bottoms) if base_bottoms else None,
                    maximum_base_z_m=max(base_bottoms) if base_bottoms else None,
                )
            robot, obj = command.robot, command.object
            support = env.scene["stand"]
            support_initial = (
                _torch(support.data.root_link_pos_w).clone()
                if hasattr(support, "data")
                else None
            )
            dt = env.physics_dt
            zeros = torch.zeros((env.num_envs, 1, 3), device=env.device)
            action = torch.zeros(env.action_space.shape, device=env.device)

            def pose_object(pose_wxyz):
                pose = torch.as_tensor(
                    pose_wxyz, dtype=torch.float32, device=env.device
                ).repeat(env.num_envs, 1)
                pose[:, :3] += env.scene.env_origins
                obj.write_root_link_pose_to_sim(
                    torch.cat((pose[:, :3], pose[:, 4:7], pose[:, 3:4]), -1)
                )
                obj.write_root_com_velocity_to_sim(
                    torch.zeros((env.num_envs, 6), device=env.device)
                )
                env.sim.forward()
                env.scene.update(0.0)

            def raw_step(count, force=None, torque=None):
                for _ in range(count):
                    obj.set_external_force_and_torque(
                        forces=zeros if force is None else force,
                        torques=zeros if torque is None else torque,
                        is_global=False,
                    )
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(dt)
                # The production command suppresses contacts until a real
                # post-reset step. These direct mechanical steps bypass its
                # control-step callback, so mark that a step has occurred.
                command.steps_since_last_reset.clamp_min_(1)
                command._live_contact_cache_step = -1
                command._support_cache_step = -1

            def contact_force_pairs():
                if not type(robot).__module__.startswith("isaaclab_newton"):
                    return []
                contacts = NewtonManager.get_contacts()
                count = int(wp.to_torch(contacts.rigid_contact_count)[0])
                a = wp.to_torch(contacts.rigid_contact_shape0)[:count].cpu().tolist()
                b = wp.to_torch(contacts.rigid_contact_shape1)[:count].cpu().tolist()
                forces = (
                    wp.to_torch(contacts.force)[:count, :3].norm(dim=-1).cpu().tolist()
                )
                bodies = model.shape_body.numpy()
                pairs = {}
                for first, second, force in zip(a, b, forces, strict=True):

                    def name(shape):
                        body = int(bodies[shape])
                        return (
                            model.body_label[body]
                            if body >= 0
                            else model.shape_label[shape]
                        ).rsplit("/", 1)[-1]

                    pair = tuple(sorted((name(first), name(second))))
                    pairs[pair] = max(pairs.get(pair, 0.0), force)
                return [
                    [*pair, force]
                    for pair, force in sorted(pairs.items(), key=lambda item: -item[1])[
                        :12
                    ]
                ]

            frame = int(command.timestep_counter[0])
            wrist_errors = {}
            for side in ("left", "right"):
                actual = getattr(command, f"{side}_hand_wrist_position_e")
                expected = torch.tensor(
                    getattr(reference, f"{side}_wrist_pose_w")[frame, :3],
                    device=env.device,
                )
                wrist_errors[side] = float((actual - expected).norm(dim=-1).max())
            check(
                "mounted_reset",
                max(wrist_errors.values()) < 0.001 and robot.num_joints == 58,
                joint_count=robot.num_joints,
                maximum_wrist_position_error_m=wrist_errors,
            )
            frame_errors = {}
            for side in ("left", "right"):
                names = getattr(reference, f"{side}_hand_frame_names")
                ids, _ = robot.find_bodies(list(names), preserve_order=True)
                actual = (
                    _torch(robot.data.body_link_pos_w)[:, ids]
                    - env.scene.env_origins[:, None]
                )
                expected = torch.tensor(
                    getattr(reference, f"{side}_hand_frame_poses_w")[frame, :, :3],
                    device=env.device,
                )
                frame_errors[side] = float((actual - expected).norm(dim=-1).max())
            check(
                "hand_forward_kinematics",
                max(frame_errors.values()) < 0.001,
                maximum_body_position_error_m=frame_errors,
            )

            # Place a solid convex object piece through each palm independently.
            # This deliberate mechanical overlap establishes a known contact;
            # it is not a training reset or an assessment of grasp quality.
            import trimesh

            part_center = trimesh.load(
                reference.metadata["mounted_scene"]["object_parts"][0], force="mesh"
            ).center_mass
            contact = {}
            minimum_contact = {}
            for side in ("left", "right"):
                env.reset()
                palm_id = getattr(command, f"{side}_wrist_body_id")
                palm = (
                    (
                        _torch(robot.data.body_com_pos_w)[0, palm_id]
                        - env.scene.env_origins[0]
                    )
                    .cpu()
                    .numpy()
                )
                pose = np.r_[palm - part_center, [1, 0, 0, 0]]
                pose_object(pose)
                peak = torch.zeros(env.num_envs, device=env.device)
                for _ in range(5):
                    raw_step(1)
                    forces = (
                        getattr(command, f"{side}_hand_object_contact_forces_w")
                        .norm(dim=-1)
                        .reshape(env.num_envs, -1)
                        .max(-1)
                        .values
                    )
                    peak = torch.maximum(peak, forces)
                contact[side] = float(peak.max())
                minimum_contact[side] = float(peak.min())
            check(
                "known_hand_object_contacts",
                all(v > 0.1 for v in minimum_contact.values()),
                maximum_force_n=contact,
                minimum_across_environments_n=minimum_contact,
            )

            pose_object([2, 0, 2, 1, 0, 0, 0])
            raw_step(5)
            if support_initial is not None:
                drift = float(
                    (_torch(support.data.root_link_pos_w) - support_initial)
                    .norm(dim=-1)
                    .max()
                )
                check("fixed_support_pose", drift < 1e-5, maximum_drift_m=drift)
            separated = {
                side: float(
                    getattr(command, f"{side}_hand_object_contact_forces_w")
                    .norm(dim=-1)
                    .max()
                )
                for side in ("left", "right")
            }
            check(
                "separated_hand_object_contacts",
                max(separated.values()) < 1e-4,
                maximum_force_n=separated,
            )

            env.reset()
            pose_object([2, 0, 2, 1, 0, 0, 0])
            target = _torch(robot.data.joint_pos).clone()
            limits = _torch(robot.data.soft_joint_pos_limits)
            arm_ids = command.arm_joint_ids
            direction = torch.where(
                target[:, arm_ids] < limits[:, arm_ids].mean(-1), 1.0, -1.0
            )
            perturbed = target.clone()
            perturbed[:, arm_ids] = torch.clamp(
                target[:, arm_ids] + direction * 0.05,
                limits[:, arm_ids, 0],
                limits[:, arm_ids, 1],
            )
            robot.write_joint_state_to_sim(perturbed, torch.zeros_like(perturbed))
            robot.set_joint_position_target(target)
            robot.set_joint_velocity_target(torch.zeros_like(target))
            initial_error = float(
                (perturbed[:, arm_ids] - target[:, arm_ids]).abs().mean()
            )
            peak_speed, peak_effort_ratio = 0.0, 0.0
            pd_steps = round(1.0 / dt)
            for _ in range(pd_steps):
                raw_step(1)
                peak_speed = max(
                    peak_speed,
                    float(_torch(robot.data.joint_vel)[:, arm_ids].abs().max()),
                )
                ratio = _torch(robot.data.applied_torque).abs() / _torch(
                    robot.data.joint_effort_limits
                ).clamp_min(1e-6)
                peak_effort_ratio = max(peak_effort_ratio, float(ratio.max()))
            final_error = float(
                (_torch(robot.data.joint_pos)[:, arm_ids] - target[:, arm_ids])
                .abs()
                .mean()
            )
            check(
                "free_space_arm_pd",
                final_error < initial_error * 0.25
                and peak_speed < 2.45
                and peak_effort_ratio <= 1.01,
                settle_time_s=pd_steps * dt,
                initial_mean_error_rad=initial_error,
                final_mean_error_rad=final_error,
                maximum_arm_speed_rad_s=peak_speed,
                maximum_effort_limit_ratio=peak_effort_ratio,
                final_contact_pairs=contact_force_pairs(),
            )

            # A 1 N body-X force must rotate with the body. This also checks
            # the pinned Newton wrench-frame correction on the task object.
            force = zeros.clone()
            force[..., 0] = 1
            mass = float(_torch(obj.data.default_mass)[0, 0])
            velocities = []
            errors = []
            orientations = []
            for quat, direction in (
                ([1, 0, 0, 0], [1, 0]),
                ([2**-0.5, 0, 0, 2**-0.5], [0, 1]),
            ):
                pose_object([2, 0, 2, *quat])
                orientations.append(_torch(obj.data.root_link_quat_w)[0].cpu().tolist())
                raw_step(1, force)
                velocity = _torch(obj.data.root_com_vel_w)[:, :2].clone()
                expected = torch.tensor(direction, device=env.device) * dt / mass
                errors.append(float((velocity - expected).norm(dim=-1).max()))
                velocities.append(velocity[0].cpu().tolist())
            check(
                "rotated_body_wrench",
                max(errors) < 0.003,
                mass_kg=mass,
                measured_xy_velocity_m_s=velocities,
                velocity_error_m_s=errors,
                actual_start_orientation_xyzw=orientations,
            )

            torque = zeros.clone()
            torque[..., 0] = 0.001
            torque_errors, angular_velocities = [], []
            inertia_x = float(
                reference.scene_physics.object_diagonal_inertia_kg_m2[0, 0]
            )
            for quat, direction in (
                ([1, 0, 0, 0], [1, 0, 0]),
                ([2**-0.5, 0, 0, 2**-0.5], [0, 1, 0]),
            ):
                pose_object([2, 0, 2, *quat])
                raw_step(1, torque=torque)
                angular = _torch(obj.data.root_com_vel_w)[:, 3:].clone()
                expected = (
                    torch.tensor(direction, device=env.device) * 0.001 * dt / inertia_x
                )
                torque_errors.append(float((angular - expected).norm(dim=-1).max()))
                angular_velocities.append(angular[0].cpu().tolist())
            check(
                "rotated_body_torque",
                max(torque_errors) < 1e-5,
                angular_velocity_error_rad_s=torque_errors,
                measured_angular_velocity_rad_s=angular_velocities,
            )

            # Clear the hands from the stand with a declared ready posture.
            ready = torch.zeros((env.num_envs, robot.num_joints), device=env.device)
            for name in ("L_arm_j4", "R_arm_j4"):
                ready[:, robot.joint_names.index(name)] = -1.2
            robot.write_joint_state_to_sim(ready, torch.zeros_like(ready))
            robot.set_joint_position_target(ready)
            robot.set_joint_velocity_target(torch.zeros_like(ready))
            stand_pose = np.asarray(
                reference.metadata["mounted_scene"]["source_initial_object_pose_w"]
            ).copy()
            stand_pose[2] += 0.01
            pose_object(stand_pose)
            raw_step(100)
            rest = (
                (_torch(obj.data.root_link_pos_w) - env.scene.env_origins).cpu().numpy()
            )
            rest_error = float(np.linalg.norm(rest - stand_pose[:3], axis=-1).max())
            check(
                "object_supported_by_stand",
                rest_error < 0.05,
                maximum_position_change_m=rest_error,
                first_environment_position_m=rest[0].tolist(),
            )
            away = stand_pose.copy()
            away[0] += 1
            pose_object(away)
            raw_step(30)
            fall = float(
                (
                    away[2]
                    - (
                        _torch(obj.data.root_link_pos_w)[:, 2]
                        - env.scene.env_origins[:, 2]
                    )
                ).min()
            )
            check("unsupported_object_falls", fall > 0.25, minimum_fall_distance_m=fall)

            env.reset()
            finite, termination_count = True, 0
            longest_episode = 0
            contact_peak = 0.0
            for _ in range(args.steps):
                if args.unassisted:
                    command.virtual_object_controller_scale_factor_per_env.zero_()
                observations, reward, terminated, _, _ = env.step(action)
                observation_finite = all(
                    bool(torch.isfinite(value).all()) for value in observations.values()
                )
                finite &= bool(
                    observation_finite
                    and torch.isfinite(reward).all()
                    and torch.isfinite(_torch(robot.data.joint_pos)).all()
                    and torch.isfinite(_torch(robot.data.joint_vel)).all()
                )
                termination_count += int(terminated.sum())
                longest_episode = max(
                    longest_episode, int(env.episode_length_buf.max())
                )
                for side in ("left", "right"):
                    contact_peak = max(
                        contact_peak,
                        float(
                            getattr(command, f"{side}_hand_object_contact_forces_w")
                            .norm(dim=-1)
                            .max()
                        ),
                    )
            check(
                "finite_rollout",
                finite and longest_episode > 1,
                steps=args.steps,
                termination_count=termination_count,
                longest_episode_control_steps=longest_episode,
                maximum_hand_object_force_n=contact_peak,
            )
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
