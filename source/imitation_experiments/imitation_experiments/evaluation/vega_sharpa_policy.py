"""Evaluate object goals for mounted PPO and zero residuals on identical starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from imitation_experiments.evaluation.object_task_metrics import object_endpoint_metrics


class EpisodeMeasurements:
    """Accumulate only each world's first episode, including its terminal step."""

    def __init__(self, num_envs: int):
        self.counts = np.zeros(num_envs, dtype=np.int64)
        self.sums: dict[str, np.ndarray] = {}
        self.maxima: dict[str, np.ndarray] = {}

    def add(self, values: dict[str, np.ndarray], active: np.ndarray):
        active = np.asarray(active, dtype=bool)
        for name, raw in values.items():
            value = np.asarray(raw, dtype=np.float64)
            if value.shape != self.counts.shape or not np.isfinite(value[active]).all():
                raise ValueError(f"Invalid active-episode metric: {name}")
            self.sums.setdefault(name, np.zeros_like(value))[active] += value[active]
            maximum = self.maxima.setdefault(name, np.full_like(value, -np.inf))
            maximum[active] = np.maximum(maximum[active], value[active])
        self.counts[active] += 1

    def summary(self) -> dict:
        if (self.counts == 0).any():
            raise ValueError(
                "Every evaluation world must have at least one measured step."
            )
        return {
            name: {
                "mean_over_episode_means": float((total / self.counts).mean()),
                "maximum": float(self.maxima[name].max()),
                "per_episode_mean": (total / self.counts).tolist(),
            }
            for name, total in self.sums.items()
        }


def fixed_start_frames(reset_frames, num_envs: int, mode: str) -> np.ndarray:
    allowed = np.asarray(reset_frames, dtype=np.int64)
    if allowed.ndim != 1 or len(allowed) == 0 or num_envs < 1:
        raise ValueError(
            "Evaluation needs a nonempty reset pool and positive world count."
        )
    if mode == "first":
        return np.full(num_envs, allowed[0], dtype=np.int64)
    if mode != "grid":
        raise ValueError(f"Unknown start mode: {mode}")
    return allowed[np.linspace(0, len(allowed) - 1, num_envs).round().astype(int)]


def main():
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Imitation-Vega-Sharpa-v0")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--zero-residual", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assistance", type=float, nargs="+", default=[0.75, 0.0])
    parser.add_argument(
        "--starts", choices=["first", "grid"], nargs="+", default=["first", "grid"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnose-collector", action="store_true")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Record observations and control-step force-balance traces for world zero.",
    )
    parser.add_argument(
        "--protocol", choices=["tracking", "object-task"], default="object-task"
    )
    parser.add_argument("--goal-position-tolerance-m", type=float, default=0.05)
    parser.add_argument("--goal-orientation-tolerance-rad", type=float, default=0.35)
    args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + hydra_args
    args.headless = True
    if any(not 0 <= scale <= 1 for scale in args.assistance):
        parser.error("Assistance scales must lie in [0, 1].")
    if not args.checkpoints and not args.zero_residual:
        parser.error("Select a checkpoint or the zero-residual baseline.")

    import gymnasium as gym
    import torch
    from tensordict import TensorDict
    from torchrl.envs import TransformedEnv, Compose
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    from rlopt.agent import PPO
    from iltools.core import load_dexterous_reference_set, sha256_file
    from isaaclab_tasks.utils import launch_simulation, resolve_task_config
    import isaaclab_imitation.tasks  # noqa: F401
    from isaaclab_imitation.envs.rlopt import IsaacLabWrapper
    from isaaclab_imitation.tasks.manager_based.dexmanip.sharpa_command import _torch
    from isaaclab_imitation.tasks.manager_based.dexmanip.mdp import quat_error_magnitude
    from isaaclab_imitation.tasks.manager_based.dexmanip.mdp import quat_rotate
    from imitation_experiments.lowlevel.low_level_tracker import (
        load_frozen_low_level_tracker,
    )

    cfg, agent_cfg = resolve_task_config(args.task, "rlopt_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.curriculum.fixed_timestep = None
    if args.protocol == "object-task":
        cfg.terminations.hand_wrist_away_from_trajectory = None
        cfg.terminations.object_away_from_trajectory = None
    cfg.commands.sharpa_reference.object_goal_position_tolerance_m = (
        args.goal_position_tolerance_m
    )
    cfg.commands.sharpa_reference.object_goal_orientation_tolerance_rad = (
        args.goal_orientation_tolerance_rad
    )
    cfg.observations.policy.enable_corruption = False
    # Keep the same reward weights across checkpoints and assistance levels.
    cfg.rewards.object_keypoints_tracking_exp.weight = 0.1
    cfg.rewards.hand_keypoints_tracking_exp.weight = 0.25
    cfg.rewards.hand_joint_pos_tracking_exp.weight = 0.25
    reference = load_dexterous_reference_set(cfg.reference.manifest)[0]
    goal_pose = reference.object_poses_w[-1, 0]
    center_of_mass_b = reference.scene_physics.object_center_of_mass_m[0]
    report = {
        "qualification": "one-seed fixed-start policy evaluation on the cropped/slowed local Reference",
        "complete": False,
        "expected_result_count": (len(args.checkpoints) + int(args.zero_residual))
        * len(args.assistance)
        * len(args.starts),
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "reference_manifest_sha256": sha256_file(cfg.reference.manifest),
        "reference_sha256": sha256_file(reference.source_path),
        "reference_frame_count": reference.frame_count,
        "protocol": {
            "name": args.protocol,
            "primary_metric": "final box-center position success rate at the Reference end"
            if args.protocol == "object-task"
            else "tracking-termination-free Reference completion",
            "position_tolerance_m": args.goal_position_tolerance_m,
            "orientation_tolerance_rad_secondary": args.goal_orientation_tolerance_rad,
            "intermediate_tracking_terminations_enabled": args.protocol == "tracking",
            "observation_corruption": False,
            "diagnostic_traces": args.diagnostics,
            "assistance": "fixed at the declared scale before every physics control step, including reset hold",
            "measurement": "after physics and before reset/reference advance; each world's first episode only",
            "wrist_error_target": "Reference wrist pose relative to the Reference object, transformed by the current live object pose; this is not fixed-world arm FK error",
            "completion": "Reached Reference end; endpoint object success is a separate metric"
            if args.protocol == "object-task"
            else "Reference timeout with no simultaneous wrist/object tracking termination",
            "object_termination_position_m": 0.2,
            "object_termination_orientation_rad": 0.7,
            "wrist_termination_position_m": 0.2,
            "first_start": "first frame of the local crop, not the original full capture",
            "grid_start": "equally spaced entries of the same audited reset pool for every policy",
            "reward_weights": {
                name: getattr(cfg.rewards, name).weight
                for name in vars(cfg.rewards)
                if hasattr(getattr(cfg.rewards, name), "weight")
            },
        },
        "results": [],
    }
    trajectories = {}
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            env.reset()
            command = env.command_manager.get_term("sharpa_reference")
            virtual = env.action_manager.get_term("virtual_object")
            virtual_forces_this_step = []
            if args.diagnostics:
                from isaaclab_newton.physics import NewtonManager

                diagnostic_model = NewtonManager.get_model()
                diagnostic_shape_bodies = diagnostic_model.shape_body.numpy()
                diagnostic_body_labels = list(diagnostic_model.body_label)
                diagnostic_object_id = next(
                    i
                    for i, label in enumerate(diagnostic_body_labels)
                    if "/env_0/" in label and label.endswith("/object")
                )
                original_apply = virtual.apply_actions

                def record_virtual_force():
                    original_apply()
                    virtual_forces_this_step.append(
                        quat_rotate(
                            command.object_orientation_e[:, 0],
                            virtual.raw_actions[:, :3],
                        )
                        .detach()
                        .clone()
                    )

                virtual.apply_actions = record_virtual_force
                report["diagnostics"] = {
                    "contact_body_labels": [*diagnostic_body_labels, "world/static"],
                    "mass_kg": float(command.object_mass[0]),
                    "gravity_w": list(cfg.sim.gravity),
                    "control_dt_s": float(env.step_dt),
                    "linear_stiffness": virtual.cfg.tracking_controller_linear_stiffness,
                    "linear_damping": virtual.cfg.tracking_controller_linear_damping,
                    "virtual_force_w": "mean over physics substeps, includes assistance gravity compensation",
                    "hand_contact_force_w": "instantaneous final-substep sum on the object",
                    "policy_observation": "before the control step; positions and velocities are after physics",
                }
            agent = None
            if args.checkpoints or args.diagnose_collector:
                agent_cfg.seed = args.seed
                agent_cfg.env.num_envs = args.num_envs
                agent_cfg.env.env_name = args.task
                agent_cfg.collector.frames_per_batch = 24 * args.num_envs
                agent_cfg.collector.total_frames = (3 if args.diagnose_collector else 1) * 24 * args.num_envs
                agent_cfg.loss.mini_batch_size = max(1, 6 * args.num_envs)
                agent_cfg.replay_buffer.size = 24 * args.num_envs
                agent_cfg.logger.backend = ""
                agent_cfg.logger.log_dir = str(args.output.parent)
                agent = PPO(
                    env=TransformedEnv(IsaacLabWrapper(env), Compose()),
                    config=agent_cfg,
                )
            if args.diagnose_collector:
                from imitation_experiments.audit.vega_sharpa_collector import check_collector

                torch.set_float32_matmul_precision("high")
                return check_collector(agent, args.num_envs, args.output)
            candidates = ([None] if args.zero_residual else []) + args.checkpoints
            original_compute = env.reward_manager.compute
            for checkpoint_path in candidates:
                frozen = None
                label = "zero_residual"
                if checkpoint_path is not None:
                    checkpoint_path = checkpoint_path.expanduser().resolve()
                    payload = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                    state = payload.get("isaaclab_training_state", {})
                    if (
                        state.get("reference_manifest_sha256")
                        != report["reference_manifest_sha256"]
                    ):
                        raise ValueError(
                            "Evaluation checkpoint and Reference fingerprints differ."
                        )
                    frozen = load_frozen_low_level_tracker(
                        agent, checkpoint_path, expected_input_keys=["policy"]
                    )
                    label = checkpoint_path.stem
                for scale in args.assistance:
                    for mode in args.starts:
                        starts = fixed_start_frames(
                            command._reset_frames.cpu().numpy(), env.num_envs, mode
                        )
                        start_tensor = torch.as_tensor(starts, device=env.device)
                        command._sample_start_frames = lambda lengths: start_tensor[
                            : len(lengths)
                        ]
                        command.virtual_object_controller_curriculum_scale.fill_(scale)
                        observations, _ = env.reset(seed=args.seed)
                        active = np.ones(env.num_envs, dtype=bool)
                        completed = np.zeros(env.num_envs, dtype=bool)
                        terminated_by = [[] for _ in range(env.num_envs)]
                        final_frames = starts.copy()
                        final_object_poses = np.zeros(
                            (env.num_envs, 7), dtype=np.float64
                        )
                        final_object_speeds = np.zeros(env.num_envs, dtype=np.float64)
                        terminal_logged_success = None
                        terminal_logged_position_error = None
                        measured = EpisodeMeasurements(env.num_envs)
                        captured = {}
                        frames = {
                            key: []
                            for key in ("qpos", "object_pose_w", "reference_frame")
                        }
                        if args.diagnostics:
                            frames.update(
                                {
                                    key: []
                                    for key in (
                                        "policy_observation",
                                        "virtual_force_w",
                                        "hand_contact_force_w",
                                        "com_velocity_before_w",
                                        "com_velocity_after_w",
                                        "root_error_w",
                                        "raw_object_contact_by_body_w",
                                    )
                                }
                            )
                        current_action = torch.zeros(
                            env.action_space.shape, device=env.device
                        )

                        def capture_before_reset(dt):
                            reward = original_compute(dt)
                            values = {
                                "reward": reward,
                                "object_position_error_m": (
                                    command.object_position_e
                                    - command.object_body_position_command_e
                                )
                                .norm(dim=-1)
                                .max(dim=-1)
                                .values,
                                "object_orientation_error_rad": quat_error_magnitude(
                                    command.object_orientation_e,
                                    command.object_body_wxyz_command_e,
                                )
                                .max(dim=-1)
                                .values,
                                "left_wrist_position_error_m": (
                                    command.left_hand_wrist_position_e
                                    - command.left_hand_wrist_pose_command_e[:, :3]
                                ).norm(dim=-1),
                                "right_wrist_position_error_m": (
                                    command.right_hand_wrist_position_e
                                    - command.right_hand_wrist_pose_command_e[:, :3]
                                ).norm(dim=-1),
                                "arm_joint_error_rad": (
                                    command.arm_position_command
                                    - _torch(command.robot.data.joint_pos)[
                                        :, command.arm_joint_ids
                                    ]
                                )
                                .abs()
                                .mean(dim=-1),
                                "action_abs_mean": current_action.abs().mean(dim=-1),
                                "hand_object_force_n": torch.maximum(
                                    command.left_hand_object_contact_forces_w.norm(
                                        dim=-1
                                    )
                                    .flatten(start_dim=1)
                                    .max(dim=-1)
                                    .values,
                                    command.right_hand_object_contact_forces_w.norm(
                                        dim=-1
                                    )
                                    .flatten(start_dim=1)
                                    .max(dim=-1)
                                    .values,
                                ),
                            }
                            captured["values"] = {
                                k: _torch(v).detach().cpu().numpy().copy()
                                for k, v in values.items()
                            }
                            captured["terminations"] = {
                                name: _torch(env.termination_manager.get_term(name))
                                .cpu()
                                .numpy()
                                .copy()
                                for name in env.termination_manager.active_terms
                            }
                            captured["reference_frame"] = (
                                command.timestep_counter.cpu().numpy().copy()
                            )
                            captured["object_pose_w"] = (
                                torch.cat(
                                    (
                                        command.object_position_e[:, 0],
                                        command.object_orientation_e[:, 0],
                                    ),
                                    dim=-1,
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .copy()
                            )
                            captured["object_speed_m_s"] = (
                                _torch(command.object.data.root_com_vel_w)[:, :3]
                                .norm(dim=-1)
                                .cpu()
                                .numpy()
                                .copy()
                            )
                            if active[0]:
                                if args.diagnostics:
                                    contacts = NewtonManager.get_contacts()
                                    count = int(contacts.rigid_contact_count.numpy()[0])
                                    shapes_a = contacts.rigid_contact_shape0.numpy()[
                                        :count
                                    ]
                                    shapes_b = contacts.rigid_contact_shape1.numpy()[
                                        :count
                                    ]
                                    forces = contacts.force.numpy()[:count, :3]
                                    by_body = np.zeros(
                                        (len(diagnostic_body_labels) + 1, 3),
                                        dtype=np.float32,
                                    )
                                    for shape_a, shape_b, force in zip(
                                        shapes_a, shapes_b, forces, strict=True
                                    ):
                                        if shape_a < 0 or shape_b < 0:
                                            continue
                                        body_a = int(diagnostic_shape_bodies[shape_a])
                                        body_b = int(diagnostic_shape_bodies[shape_b])
                                        if body_a == diagnostic_object_id:
                                            by_body[body_b] += force
                                        elif body_b == diagnostic_object_id:
                                            by_body[body_a] -= force
                                    frames["raw_object_contact_by_body_w"].append(
                                        by_body
                                    )
                                    diagnostics = {
                                        "policy_observation": _torch(
                                            observations["policy"]
                                        )[0],
                                        "virtual_force_w": torch.stack(
                                            virtual_forces_this_step
                                        ).mean(0)[0],
                                        "hand_contact_force_w": (
                                            command.left_hand_object_contact_forces_w
                                            + command.right_hand_object_contact_forces_w
                                        ).sum(dim=(1, 2))[0],
                                        "com_velocity_before_w": velocity_before[0],
                                        "com_velocity_after_w": _torch(
                                            command.object.data.root_com_vel_w
                                        )[0, :3],
                                        "root_error_w": command.object_position_e[0, 0]
                                        - command.object_body_position_command_e[0, 0],
                                    }
                                    for name, value in diagnostics.items():
                                        frames[name].append(
                                            value.detach().cpu().numpy().copy()
                                        )
                                frames["qpos"].append(
                                    _torch(command.robot.data.joint_pos)[
                                        0, command.reference_joint_ids
                                    ]
                                    .cpu()
                                    .numpy()
                                    .copy()
                                )
                                frames["object_pose_w"].append(
                                    torch.cat(
                                        (
                                            command.object_position_e[0, 0],
                                            command.object_orientation_e[0, 0],
                                        )
                                    )
                                    .cpu()
                                    .numpy()
                                    .copy()
                                )
                                frames["reference_frame"].append(
                                    int(command.timestep_counter[0])
                                )
                            return reward

                        env.reward_manager.compute = capture_before_reset
                        before = (
                            {k: v.clone() for k, v in agent.policy.state_dict().items()}
                            if frozen
                            else {}
                        )
                        with (
                            torch.no_grad(),
                            set_exploration_type(ExplorationType.DETERMINISTIC),
                        ):
                            for _ in range(
                                reference.frame_count
                                + command.cfg.virtual_object_control_decay_steps
                                + 2
                            ):
                                command.virtual_object_controller_curriculum_scale.fill_(
                                    scale
                                )
                                command.virtual_object_controller_scale_factor_per_env.fill_(
                                    scale
                                )
                                if frozen:
                                    td = TensorDict(
                                        {"policy": _torch(observations["policy"])},
                                        batch_size=[env.num_envs],
                                        device=env.device,
                                    )
                                    current_action = frozen.policy(td)["action"]
                                if args.diagnostics:
                                    virtual_forces_this_step.clear()
                                    velocity_before = _torch(
                                        command.object.data.root_com_vel_w
                                    )[:, :3].clone()
                                observations, reward, terminated, truncated, _ = (
                                    env.step(current_action)
                                )
                                if not torch.isfinite(
                                    _torch(observations["policy"])
                                ).all():
                                    raise ValueError(
                                        "Non-finite policy observations during evaluation."
                                    )
                                measured.add(captured["values"], active)
                                done = (
                                    (_torch(terminated) | _torch(truncated))
                                    .cpu()
                                    .numpy()
                                )
                                newly_done = done & active
                                if (
                                    args.protocol == "object-task"
                                    and mode == "first"
                                    and newly_done.all()
                                ):
                                    terminal_logged_success = float(
                                        env.extras["log"][
                                            "Metrics/sharpa_reference/object_task_success"
                                        ]
                                    )
                                    terminal_logged_position_error = float(
                                        env.extras["log"][
                                            "Metrics/sharpa_reference/final_box_position_error_m"
                                        ]
                                    )
                                final_frames[active] = captured["reference_frame"][
                                    active
                                ]
                                final_object_poses[active] = captured["object_pose_w"][
                                    active
                                ]
                                final_object_speeds[active] = captured[
                                    "object_speed_m_s"
                                ][active]
                                reasons = captured["terminations"]
                                for index in np.flatnonzero(newly_done):
                                    terminated_by[index] = [
                                        name
                                        for name, value in reasons.items()
                                        if value[index]
                                    ]
                                failed = np.zeros(env.num_envs, dtype=bool)
                                for name, value in reasons.items():
                                    if name != "time_out":
                                        failed |= value
                                completed[newly_done] = (
                                    reasons["time_out"][newly_done]
                                    & ~failed[newly_done]
                                )
                                active[newly_done] = False
                                if not active.any():
                                    break
                        env.reward_manager.compute = original_compute
                        if frozen and any(
                            not torch.equal(before[k], v)
                            for k, v in agent.policy.state_dict().items()
                        ):
                            raise ValueError(
                                "Frozen policy/normalizer changed during evaluation."
                            )
                        key = f"{label}__{mode}__assistance_{scale:g}"
                        endpoint = object_endpoint_metrics(
                            final_object_poses,
                            goal_pose,
                            center_of_mass_b,
                            final_frames >= reference.frame_count - 1,
                            position_tolerance_m=args.goal_position_tolerance_m,
                            orientation_tolerance_rad=args.goal_orientation_tolerance_rad,
                        )
                        if args.protocol == "object-task" and mode == "first":
                            if terminal_logged_success is None or not np.isclose(
                                terminal_logged_success,
                                endpoint["position_success_rate"],
                            ):
                                raise ValueError(
                                    "Terminal object-success logging differs from the pre-reset box state."
                                )
                            if terminal_logged_position_error is None or not np.isclose(
                                terminal_logged_position_error,
                                endpoint["position_error_m"]["mean"],
                                atol=1e-5,
                            ):
                                raise ValueError(
                                    "Terminal box-position logging differs from the pre-reset box state."
                                )
                        for field, data in frames.items():
                            trajectories[f"{key}__{field}"] = np.asarray(data)
                        result = {
                            "policy": label,
                            "checkpoint": frozen.provenance if frozen else None,
                            "assistance_scale": scale,
                            "start_mode": mode,
                            "start_frames": starts.tolist(),
                            "all_episodes_finished": not bool(active.any()),
                            "reference_completion_fraction": float(completed.mean()),
                            "completed": completed.tolist(),
                            "episode_control_steps": measured.counts.tolist(),
                            "terminal_reference_frames": final_frames.tolist(),
                            "remaining_reference_fraction_completed": (
                                (final_frames - starts)
                                / np.maximum(1, reference.frame_count - 1 - starts)
                            ).tolist(),
                            "termination_reasons": terminated_by,
                            "metrics": measured.summary(),
                            "trajectory_key": key,
                            "object_endpoint": endpoint,
                            "terminal_logged_object_success_rate": terminal_logged_success,
                            "terminal_logged_box_position_error_m": terminal_logged_position_error,
                            "final_box_speed_m_s": final_object_speeds.tolist(),
                            "primary_success_rate": endpoint["position_success_rate"]
                            if args.protocol == "object-task"
                            else float(completed.mean()),
                            "policy_and_normalizer_unchanged": True,
                        }
                        report["results"].append(result)
                        report["complete"] = len(report["results"]) == report[
                            "expected_result_count"
                        ] and all(
                            row["all_episodes_finished"] for row in report["results"]
                        )
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        np.savez_compressed(
                            args.output.with_suffix(".npz"), **trajectories
                        )
                        print(
                            json.dumps(
                                {
                                    "policy": label,
                                    "assistance": scale,
                                    "starts": mode,
                                    "completion": result[
                                        "reference_completion_fraction"
                                    ],
                                    "mean_steps": float(measured.counts.mean()),
                                    "object_success_rate": endpoint[
                                        "position_success_rate"
                                    ],
                                    "final_box_position_error_m": endpoint[
                                        "position_error_m"
                                    ]["mean"],
                                    "object_error_m": result["metrics"][
                                        "object_position_error_m"
                                    ]["mean_over_episode_means"],
                                }
                            ),
                            flush=True,
                        )
                        if active.any():
                            raise ValueError(
                                "The evaluation cap was reached before every first episode ended."
                            )
        finally:
            env.close()
    print(f"Policy evaluation: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
