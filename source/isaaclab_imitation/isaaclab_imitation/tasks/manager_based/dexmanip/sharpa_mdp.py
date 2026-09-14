# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""MDP terms for the PhysX Sharpa floating-hand baseline."""

from __future__ import annotations

from typing import Any

import torch

from . import mdp as dex_mdp


def _command(env: Any, name: str = "sharpa_reference") -> Any:
    return env.command_manager.get_term(name)


def wrist_position_e(env: Any, command_name: str = "sharpa_reference") -> torch.Tensor:
    command = _command(env, command_name)
    return torch.cat(
        (command.right_hand_wrist_position_e, command.left_hand_wrist_position_e),
        dim=-1,
    )


def wrist_orientation_e(
    env: Any, command_name: str = "sharpa_reference"
) -> torch.Tensor:
    command = _command(env, command_name)
    return torch.cat(
        (command.right_hand_wrist_wxyz_e, command.left_hand_wrist_wxyz_e), dim=-1
    )


def wrist_velocity_b(env: Any, command_name: str = "sharpa_reference") -> torch.Tensor:
    command = _command(env, command_name)
    return torch.cat(
        (command.right_hand_wrist_velocity_b, command.left_hand_wrist_velocity_b),
        dim=-1,
    )


def finger_joint_pos(env: Any, command_name: str = "sharpa_reference") -> torch.Tensor:
    command = _command(env, command_name)
    values = []
    for robot, ids, position in (
        (
            command.right_robot,
            command.right_finger_joint_ids,
            command.right_hand_finger_joint_pos,
        ),
        (
            command.left_robot,
            command.left_finger_joint_ids,
            command.left_hand_finger_joint_pos,
        ),
    ):
        # Positions follow the reference joint order (finger_joint_ids); the
        # limits must be gathered in the same order, not the Isaac order.
        limits = getattr(
            robot.data.joint_pos_limits, "torch", robot.data.joint_pos_limits
        )[:, ids]
        values.append(
            2.0
            * (position - limits[..., 0])
            / (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
            - 1.0
        )
    return torch.cat(values, dim=-1)


def finger_joint_vel(env: Any, command_name: str = "sharpa_reference") -> torch.Tensor:
    command = _command(env, command_name)
    return torch.cat(
        (command.right_hand_finger_joint_vel, command.left_hand_finger_joint_vel),
        dim=-1,
    )


def object_pose_e(env: Any, command_name: str = "sharpa_reference") -> torch.Tensor:
    command = _command(env, command_name)
    return torch.cat(
        (command.object_position_e[:, 0], command.object_orientation_e[:, 0]), dim=-1
    )


def generated_command(env: Any, command_name: str = "sharpa_reference") -> torch.Tensor:
    return _command(env, command_name).command


def processed_actions(env: Any) -> torch.Tensor:
    return torch.cat(
        (
            env.action_manager.get_term("right_hand").processed_actions,
            env.action_manager.get_term("left_hand").processed_actions,
        ),
        dim=-1,
    )


def object_keypoints_tracking_exp(
    env: Any, command_name: str = "sharpa_reference", var: float = 0.1
) -> torch.Tensor:
    command = _command(env, command_name)
    directions = (
        torch.tensor(
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
            dtype=torch.float32,
            device=env.device,
        )
        * 0.05
    )
    current = command.object_position_e[:, 0, None] + dex_mdp.quat_rotate(
        command.object_orientation_e[:, 0, None].expand(-1, 6, -1),
        directions[None].expand(env.num_envs, -1, -1),
    )
    target = command.object_body_position_command_e[:, 0, None] + dex_mdp.quat_rotate(
        command.object_body_wxyz_command_e[:, 0, None].expand(-1, 6, -1),
        directions[None].expand(env.num_envs, -1, -1),
    )
    return torch.exp(-torch.square(current - target).sum(dim=-1) / var).mean(dim=-1)


def object_pose_tracking_exp(
    env: Any,
    command_name: str = "sharpa_reference",
    position_var: float = 0.1,
    orientation_var: float = 0.5,
) -> torch.Tensor:
    """Track the live object pose, including orientation, to the reference."""

    if position_var <= 0.0 or orientation_var <= 0.0:
        raise ValueError("position_var and orientation_var must be positive.")
    command = _command(env, command_name)
    position_error = torch.square(
        command.object_position_e[:, 0] - command.object_body_position_command_e[:, 0]
    ).sum(dim=-1)
    orientation_error = dex_mdp.quat_error_magnitude(
        command.object_orientation_e[:, 0],
        command.object_body_wxyz_command_e[:, 0],
    )
    return torch.exp(
        -position_error / position_var
        - torch.square(orientation_error) / orientation_var
    )


def object_goal_tracking_exp(
    env: Any,
    command_name: str = "sharpa_reference",
    position_var: float = 0.1,
    orientation_var: float = 0.5,
) -> torch.Tensor:
    """Track the final object pose, providing an explicit task objective."""

    if position_var <= 0.0 or orientation_var <= 0.0:
        raise ValueError("position_var and orientation_var must be positive.")
    command = _command(env, command_name)
    position_error = torch.square(
        command.object_position_e[:, 0] - command.object_goal_pose_e[:, :3]
    ).sum(dim=-1)
    orientation_error = dex_mdp.quat_error_magnitude(
        command.object_orientation_e[:, 0], command.object_goal_pose_e[:, 3:7]
    )
    return torch.exp(
        -position_error / position_var
        - torch.square(orientation_error) / orientation_var
    )


def object_lift_progress(
    env: Any,
    command_name: str = "sharpa_reference",
    target_height: float = 0.1,
) -> torch.Tensor:
    """Return normalized upward progress from the episode's sampled pose."""

    if target_height <= 0.0:
        raise ValueError("target_height must be positive.")
    command = _command(env, command_name)
    return (command.object_lift_height / target_height).clamp(0.0, 1.0)


def object_task_success(
    env: Any,
    command_name: str = "sharpa_reference",
    position_threshold: float = 0.05,
    orientation_threshold: float = 0.35,
    min_lift: float = 0.05,
) -> torch.Tensor:
    """Return a binary final-goal success signal for episode diagnostics/reward."""

    if position_threshold <= 0.0 or orientation_threshold <= 0.0 or min_lift < 0.0:
        raise ValueError(
            "Task success thresholds must be positive (min_lift may be zero)."
        )
    command = _command(env, command_name)
    position_error = torch.linalg.vector_norm(
        command.object_position_e[:, 0] - command.object_goal_pose_e[:, :3], dim=-1
    )
    orientation_error = dex_mdp.quat_error_magnitude(
        command.object_orientation_e[:, 0], command.object_goal_pose_e[:, 3:7]
    )
    return (
        (position_error <= position_threshold)
        & (orientation_error <= orientation_threshold)
        & (command.object_lift_height >= min_lift)
    ).float()


def hand_keypoints_tracking_exp(
    env: Any, command_name: str = "sharpa_reference", var: float = 0.1
) -> torch.Tensor:
    command = _command(env, command_name)
    errors = torch.cat(
        (
            torch.square(
                command.right_hand_wrist_position_e
                - command.right_hand_wrist_pose_command_e[:, :3]
            ).sum(dim=-1, keepdim=True),
            torch.square(
                command.left_hand_wrist_position_e
                - command.left_hand_wrist_pose_command_e[:, :3]
            ).sum(dim=-1, keepdim=True),
            torch.square(
                command.right_hand_fingertip_position_e
                - command.right_hand_fingertip_position_command_e
            ).sum(dim=-1),
            torch.square(
                command.left_hand_fingertip_position_e
                - command.left_hand_fingertip_position_command_e
            ).sum(dim=-1),
        ),
        dim=-1,
    )
    return torch.exp(-errors / var).mean(dim=-1)


def hand_joint_pos_tracking_exp(
    env: Any, command_name: str = "sharpa_reference", var: float = 1.0
) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.cat(
        (
            command.right_hand_finger_joint_pos_command
            - command.right_hand_finger_joint_pos,
            command.left_hand_finger_joint_pos_command
            - command.left_hand_finger_joint_pos,
        ),
        dim=-1,
    )
    return torch.exp(-torch.square(error).sum(dim=-1) / var)


def action_l1(env: Any) -> torch.Tensor:
    return torch.abs(env.action_manager.action).mean(dim=-1)


def _live_contact(env: Any, side: str, threshold: float) -> torch.Tensor:
    sensor = env.scene.sensors.get(f"{side}_hand_object_contacts")
    if sensor is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    forces = getattr(sensor.data.net_forces_w, "torch", sensor.data.net_forces_w)
    # ContactSensor keeps history and one entry per sensed body.  Reduce all
    # non-environment axes so the MDP terms always receive one boolean per env.
    force_norm = torch.linalg.vector_norm(forces, dim=-1)
    return force_norm.reshape(force_norm.shape[0], -1).amax(dim=-1) > threshold


def live_contact(env: Any, side: str, threshold: float = 0.1) -> torch.Tensor:
    """Return one observed-contact boolean per environment for diagnostics."""

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    return _live_contact(env, side, threshold)


def contact_wrench_support_reward(
    env: Any, command_name: str = "sharpa_reference", threshold: float = 0.1
) -> torch.Tensor:
    command = _command(env, command_name)
    scores = []
    for side in ("left", "right"):
        scores.append(
            (command.contact_active(side) & _live_contact(env, side, threshold)).float()
        )
    return torch.stack(scores).mean(dim=0)


def missed_contact_penalty(
    env: Any, command_name: str = "sharpa_reference", threshold: float = 0.1
) -> torch.Tensor:
    command = _command(env, command_name)
    values = [
        command.contact_active(side) & ~_live_contact(env, side, threshold)
        for side in ("left", "right")
    ]
    return torch.stack(values).float().mean(dim=0)


def unintended_contact_penalty(
    env: Any, command_name: str = "sharpa_reference", threshold: float = 0.1
) -> torch.Tensor:
    command = _command(env, command_name)
    values = [
        ~command.contact_active(side) & _live_contact(env, side, threshold)
        for side in ("left", "right")
    ]
    return torch.stack(values).float().mean(dim=0)


def reference_finished(
    env: Any, command_name: str = "sharpa_reference"
) -> torch.Tensor:
    command = _command(env, command_name)
    return command.timestep_counter >= command._lengths[command.motion_index] - 1


def wrist_too_far(
    env: Any, command_name: str = "sharpa_reference", threshold: float = 0.2
) -> torch.Tensor:
    command = _command(env, command_name)
    right = torch.linalg.vector_norm(
        command.right_hand_wrist_position_e
        - command.right_hand_wrist_pose_command_e[:, :3],
        dim=-1,
    )
    left = torch.linalg.vector_norm(
        command.left_hand_wrist_position_e
        - command.left_hand_wrist_pose_command_e[:, :3],
        dim=-1,
    )
    return (right > threshold) | (left > threshold)


def object_too_far(
    env: Any, command_name: str = "sharpa_reference", threshold: float = 0.2
) -> torch.Tensor:
    command = _command(env, command_name)
    return (
        torch.linalg.vector_norm(
            command.object_position_e[:, 0]
            - command.object_body_position_command_e[:, 0],
            dim=-1,
        )
        > threshold
    )


__all__ = [name for name in globals() if not name.startswith("_")]
