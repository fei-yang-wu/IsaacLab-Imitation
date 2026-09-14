"""Mounted Vega U / Sharpa commands on the shared hand-object reward contract."""

from __future__ import annotations

import torch
from iltools.core import load_dexterous_reference_set
from isaaclab.utils.configclass import configclass

from isaaclab_imitation.contracts.object_task_rewards import box_center_tracking_reward

from . import mdp
from .sharpa_command import _torch, xyzw_to_wxyz
from .sharpa_source_command import (
    SharpaSourceReferenceCommand,
    SharpaSourceReferenceCommandCfg,
)


@configclass
class VegaSharpaReferenceCommandCfg(SharpaSourceReferenceCommandCfg):
    robot_layout: str = "fixed_base"
    right_robot_name: str = "robot"
    left_robot_name: str = "robot"
    wrist_body_expr: str = "{side}_hand_C_MC"
    fingertip_body_expr: str = "{side}_.*_DP$"
    reset_finger_openness: float = 0.0
    legacy_reset_hold: bool = True
    object_goal_position_tolerance_m: float = 0.05
    object_goal_orientation_tolerance_rad: float = 0.35

    def __post_init__(self) -> None:
        self.class_type = VegaSharpaReferenceCommand


class VegaSharpaReferenceCommand(SharpaSourceReferenceCommand):
    """Use measured object/contact targets and one fixed-base articulation."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        references = load_dexterous_reference_set(cfg.reference_manifest)
        if len(references) != 1:
            raise ValueError(
                "The mounted local-training task currently requires one Reference."
            )
        reference = references[0]
        if reference.robot_name != "vega_u_sharpa":
            raise ValueError("Expected the Vega U / Sharpa mounted model.")
        self.robot = self.right_robot
        self.reference_joint_ids, self.reference_joint_names = self._resolve_joints(
            self.robot, reference.joint_names
        )
        if len(self.reference_joint_ids) != self.robot.num_joints:
            raise ValueError("The Reference must cover every articulation joint.")
        self._mounted_qpos = _torch(
            torch.as_tensor(reference.qpos, device=self.device)
        )[None]
        self._mounted_qvel = _torch(
            torch.as_tensor(reference.qvel, device=self.device)
        )[None]
        self.arm_joint_names = [
            f"{prefix}_arm_j{i}" for prefix in ("L", "R") for i in range(1, 8)
        ]
        self.arm_joint_ids, _ = self._resolve_joints(
            self.robot, tuple(self.arm_joint_names)
        )
        self._arm_reference_indices = [
            reference.joint_names.index(n) for n in self.arm_joint_names
        ]
        frames = reference.metadata.get("mounted_retarget", {}).get("reset_frames", [])
        if not frames:
            raise ValueError("Mounted Reference has no collision-checked reset_frames.")
        self._reset_frames = torch.tensor(frames, dtype=torch.long, device=self.device)
        if (
            self._reset_frames.min() < 0
            or self._reset_frames.max() >= reference.frame_count
        ):
            raise ValueError("Mounted reset frames are outside the Reference.")
        self.metrics["arm_joint_error"] = torch.zeros(self.num_envs, device=self.device)
        for name in (
            "final_box_position_error_m",
            "final_box_orientation_error_rad",
            "object_task_success",
            "object_pose_success",
            "object_task_reached_end",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def final_box_center_target_e(self):
        goal = self.object_goal_pose_e
        return goal[:, :3] + mdp.quat_rotate(
            goal[:, 3:], self._object_com_pose_b[:, :3]
        )

    def _update_object_task_metrics(self):
        center = _torch(self.object.data.root_com_pos_w) - self._env.scene.env_origins
        position_error = (center - self.final_box_center_target_e).norm(dim=-1)
        orientation_error = mdp.quat_error_magnitude(
            self.object_orientation_e[:, 0], self.object_goal_pose_e[:, 3:]
        )
        reached_end = self.timestep_counter >= self.retargeted_horizon - 1
        position_success = reached_end & (
            position_error <= self.cfg.object_goal_position_tolerance_m
        )
        self.metrics["final_box_position_error_m"] = position_error
        self.metrics["final_box_orientation_error_rad"] = orientation_error
        self.metrics["object_task_reached_end"] = reached_end.float()
        self.metrics["object_task_success"] = position_success.float()
        self.metrics["object_pose_success"] = (
            position_success
            & (orientation_error <= self.cfg.object_goal_orientation_tolerance_rad)
        ).float()
        # The floating-hand metric also required a lift from the reset pose;
        # that is not the endpoint objective for this lowering/rotation crop.
        self.metrics["task_success"] = self.metrics["object_task_success"].clone()

    def reset(self, env_ids=None):
        # CommandManager logs these values before _resample writes new poses.
        # Refresh here to include the actual terminal physics transition.
        self._update_object_task_metrics()
        return super().reset(env_ids)

    @property
    def arm_position_command(self):
        return self._select(self._mounted_qpos)[:, self._arm_reference_indices]

    @property
    def arm_velocity_command(self):
        velocity = self._select(self._mounted_qvel)[:, self._arm_reference_indices]
        return (
            velocity
            * (
                self.steps_since_last_reset
                >= self.cfg.virtual_object_control_decay_steps
            )[:, None]
        )

    def _sample_start_frames(self, lengths):
        if self.cfg.always_reset_to_first_frame:
            return self._reset_frames[0].expand_as(lengths)
        choices = torch.randint(
            len(self._reset_frames), lengths.shape, device=self.device
        )
        return self._reset_frames[choices]

    def _reset_robot_state(self, ids):
        positions = self._select(self._mounted_qpos)[ids]
        self.robot.write_joint_state_to_sim(
            positions,
            torch.zeros_like(positions),
            joint_ids=self.reference_joint_ids,
            env_ids=ids,
        )
        self.robot.set_joint_position_target(
            positions, joint_ids=self.reference_joint_ids, env_ids=ids
        )
        self.robot.set_joint_velocity_target(
            torch.zeros_like(positions), joint_ids=self.reference_joint_ids, env_ids=ids
        )

    def _wrist_position(self, side):
        body_id = getattr(self, f"{side}_wrist_body_id")
        return (
            _torch(self.robot.data.body_link_pos_w)[:, body_id]
            - self._env.scene.env_origins
        )

    def _wrist_quaternion(self, side):
        body_id = getattr(self, f"{side}_wrist_body_id")
        quat = xyzw_to_wxyz(_torch(self.robot.data.body_link_quat_w)[:, body_id])
        return torch.where(quat[:, :1] < 0, -quat, quat)

    def _wrist_velocity(self, side):
        body_id = getattr(self, f"{side}_wrist_body_id")
        velocity = _torch(self.robot.data.body_link_vel_w)[:, body_id]
        inverse = mdp.quat_inverse(self._wrist_quaternion(side))
        return torch.cat(
            (
                mdp.quat_rotate(inverse, velocity[:, :3]),
                mdp.quat_rotate(inverse, velocity[:, 3:]),
            ),
            dim=-1,
        )

    @property
    def left_hand_wrist_position_e(self):
        return self._wrist_position("left")

    @property
    def right_hand_wrist_position_e(self):
        return self._wrist_position("right")

    @property
    def left_hand_wrist_wxyz_e(self):
        return self._wrist_quaternion("left")

    @property
    def right_hand_wrist_wxyz_e(self):
        return self._wrist_quaternion("right")

    @property
    def left_hand_wrist_velocity_b(self):
        return self._wrist_velocity("left")

    @property
    def right_hand_wrist_velocity_b(self):
        return self._wrist_velocity("right")

    def _update_metrics(self):
        super()._update_metrics()
        self._update_object_task_metrics()
        self.metrics["arm_joint_error"] = (
            (
                self.arm_position_command
                - _torch(self.robot.data.joint_pos)[:, self.arm_joint_ids]
            )
            .abs()
            .mean(-1)
        )


def arm_state(env, command_name="sharpa_reference"):
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        (
            _torch(command.robot.data.joint_pos)[:, command.arm_joint_ids],
            _torch(command.robot.data.joint_vel)[:, command.arm_joint_ids],
            command.arm_position_command,
            command.arm_velocity_command,
        ),
        dim=-1,
    )


def box_center_position_tracking_exp(
    env, command_name="sharpa_reference", position_scale_m=0.05
):
    """Track the current Reference box COM at the endpoint metric's scale."""
    command = env.command_manager.get_term(command_name)
    actual = _torch(command.object.data.root_com_pos_w) - env.scene.env_origins
    target = command.object_body_position_command_e[:, 0] + mdp.quat_rotate(
        command.object_body_wxyz_command_e[:, 0], command._object_com_pose_b[:, :3]
    )
    return box_center_tracking_reward(actual, target, position_scale_m)
