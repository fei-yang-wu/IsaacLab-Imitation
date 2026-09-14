# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ILTools reference command for the dual free-floating Sharpa task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from iltools.core import load_dexterous_reference_set
from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from . import sharpa_mdp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _torch(value: Any) -> torch.Tensor:
    return getattr(value, "torch", value)


def wxyz_to_xyzw(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., 1:4], value[..., 0:1]), dim=-1)


def xyzw_to_wxyz(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., 3:4], value[..., 0:3]), dim=-1)


def reference_advance_mask(
    timestep: torch.Tensor,
    horizon: torch.Tensor,
    steps_since_reset: torch.Tensor,
    *,
    legacy_reset_hold: bool,
    hold_steps: int,
) -> torch.Tensor:
    """Select environments whose reference advances on this control step."""

    has_next_frame = timestep < horizon - 1
    if legacy_reset_hold:
        # The released command advances once steps_since_last_reset reaches
        # the decay-step count (``>=`` in hand_object_commands._update_command).
        return has_next_frame & (steps_since_reset >= hold_steps)
    return has_next_frame


@configclass
class SharpaReferenceCommandCfg(CommandTermCfg):
    """Configuration for one single-object Sharpa reference set."""

    class_type: type = None  # type: ignore[assignment]
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
    reference_manifest: str = ""
    robot_layout: str = "dual_floating_hand"
    right_robot_name: str = "right_robot"
    left_robot_name: str = "left_robot"
    object_name: str = "object"
    wrist_body_expr: str = ".*_hand_C_MC"
    fingertip_body_expr: str = ".*_DP$"
    randomize_reference: bool = True
    always_reset_to_first_frame: bool = False
    reset_to_first_frame_prob: float = 0.1
    reset_finger_openness: float = 0.7
    initial_virtual_object_control_curriculum_scale: float = 1.0
    virtual_object_control_decay_steps: int = 20
    legacy_reset_hold: bool = False

    def __post_init__(self) -> None:
        self.class_type = SharpaReferenceCommand


class SharpaReferenceCommand(CommandTerm):
    """Publish object-centric wrist, fingertip, finger, and object targets."""

    cfg: SharpaReferenceCommandCfg

    def __init__(self, cfg: SharpaReferenceCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if not cfg.reference_manifest:
            raise ValueError(
                "Sharpa training requires env.commands.sharpa_reference."
                "reference_manifest."
            )
        references = load_dexterous_reference_set(cfg.reference_manifest)
        if not references:
            raise ValueError("The Sharpa reference set is empty.")
        for reference in references:
            if reference.robot_layout != cfg.robot_layout:
                raise ValueError(f"Sharpa requires robot_layout={cfg.robot_layout!r}.")
            if len(reference.object_names) != 1:
                raise ValueError("Sharpa v1 supports exactly one rigid object.")
            if reference.metadata.get("object_kind", "rigid") != "rigid":
                raise ValueError("Sharpa v1 does not support articulated objects.")
        first = references[0]
        for reference in references[1:]:
            if (
                reference.joint_names != first.joint_names
                or reference.left_hand_frame_names != first.left_hand_frame_names
                or reference.right_hand_frame_names != first.right_hand_frame_names
            ):
                raise ValueError("All Sharpa references must share one robot layout.")

        self.right_robot: Articulation = env.scene[cfg.right_robot_name]
        self.left_robot: Articulation = env.scene[cfg.left_robot_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.objects = [self.object]
        self.object_idx = 0
        self.num_bodies = 1
        self.right_wrist_body_id = self._resolve_wrist(self.right_robot, "right")
        self.left_wrist_body_id = self._resolve_wrist(self.left_robot, "left")
        self.right_finger_joint_ids, self.right_finger_joint_names = (
            self._resolve_joints(self.right_robot, first.right_joint_names)
        )
        self.left_finger_joint_ids, self.left_finger_joint_names = self._resolve_joints(
            self.left_robot, first.left_joint_names
        )
        self.right_fingertip_body_ids, right_tips = self._resolve_tips(self.right_robot, "right")
        self.left_fingertip_body_ids, left_tips = self._resolve_tips(self.left_robot, "left")
        self._right_reference_tip_ids = [
            first.right_hand_frame_names.index(name) for name in right_tips
        ]
        self._left_reference_tip_ids = [
            first.left_hand_frame_names.index(name) for name in left_tips
        ]

        self._lengths = torch.as_tensor(
            [reference.frame_count for reference in references],
            device=self.device,
            dtype=torch.long,
        )
        max_length = int(self._lengths.max().item())
        motion_count = len(references)

        def pack(getter: Any, shape: tuple[int, ...]) -> torch.Tensor:
            out = torch.zeros((motion_count, max_length, *shape), device=self.device)
            for index, reference in enumerate(references):
                value = torch.as_tensor(
                    getter(reference), device=self.device, dtype=torch.float32
                )
                out[index, : len(value)] = value
                out[index, len(value) :] = value[-1]
            return out

        self._left_qpos = pack(
            lambda item: item.qpos[:, [item.joint_names.index(n) for n in item.left_joint_names]],
            (len(first.left_joint_names),),
        )
        self._right_qpos = pack(
            lambda item: item.qpos[:, [item.joint_names.index(n) for n in item.right_joint_names]],
            (len(first.right_joint_names),),
        )
        self._left_wrist = pack(lambda item: item.left_wrist_pose_w, (7,))
        self._right_wrist = pack(lambda item: item.right_wrist_pose_w, (7,))
        self._object_pose = pack(lambda item: item.object_poses_w[:, 0], (7,))
        self._object_mass = torch.as_tensor(
            [
                float(
                    reference.scene_physics.object_mass_kg[0]
                    if reference.scene_physics is not None
                    else 1.0
                )
                for reference in references
            ],
            device=self.device,
            dtype=torch.float32,
        )[:, None]
        self._object_initial_pose = torch.stack(
            [
                torch.as_tensor(
                    reference.object_poses_w[0, 0],
                    device=self.device,
                    dtype=torch.float32,
                )
                for reference in references
            ]
        )
        self._object_goal_pose = torch.stack(
            [
                torch.as_tensor(
                    reference.object_poses_w[-1, 0],
                    device=self.device,
                    dtype=torch.float32,
                )
                for reference in references
            ]
        )
        self._left_tips = pack(
            lambda item: item.left_hand_frame_poses_w[:, self._left_reference_tip_ids],
            (len(self._left_reference_tip_ids), 7),
        )
        self._right_tips = pack(
            lambda item: item.right_hand_frame_poses_w[
                :, self._right_reference_tip_ids
            ],
            (len(self._right_reference_tip_ids), 7),
        )
        self._contact_active: dict[str, torch.Tensor] = {}
        for side in ("left", "right"):
            values = torch.zeros(
                (motion_count, max_length), device=self.device, dtype=torch.bool
            )
            for index, reference in enumerate(references):
                if reference.contacts is None:
                    continue
                slots = [
                    slot
                    for slot, hand_side in enumerate(reference.contacts.hand_sides)
                    if hand_side == side
                ]
                if slots:
                    active = torch.as_tensor(
                        reference.contacts.active[:, slots].any(axis=(1, 2)),
                        device=self.device,
                    )
                    values[index, : len(active)] = active
                    values[index, len(active) :] = active[-1]
            self._contact_active[side] = values

        schedule = (
            ResetSchedule.RANDOM
            if cfg.randomize_reference
            else ResetSchedule.SEQUENTIAL
        )
        self.trajectory_manager = ParallelTrajectoryManager.from_lengths(
            self._lengths,
            num_envs=self.num_envs,
            reset_schedule=schedule,
            device=self.device,
        )
        self.motion_index = self.trajectory_manager.env_traj_rank
        self.timestep_counter = self.trajectory_manager.env_step
        self.tracking_lengths = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.steps_since_last_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_initial_object_position = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.virtual_object_controller_scale_factor = torch.tensor(
            float(cfg.initial_virtual_object_control_curriculum_scale),
            device=self.device,
        )
        self.virtual_object_controller_curriculum_scale = (
            self.virtual_object_controller_scale_factor
        )
        self.virtual_object_controller_scale_factor_per_env = torch.ones(
            self.num_envs, 1, device=self.device
        )
        self.metrics = {
            "right_wrist_position_error": torch.zeros(
                self.num_envs, device=self.device
            ),
            "left_wrist_position_error": torch.zeros(self.num_envs, device=self.device),
            "object_position_error": torch.zeros(self.num_envs, device=self.device),
            "virtual_object_controller_scale": torch.zeros(
                self.num_envs, device=self.device
            ),
        }

    def _resolve_wrist(self, robot: Articulation, side: str = "") -> int:
        ids, names = robot.find_bodies(self.cfg.wrist_body_expr.format(side=side))
        if len(ids) != 1:
            raise ValueError(
                f"Sharpa wrist expression resolved {names}; expected one body."
            )
        return int(ids[0])

    @staticmethod
    def _resolve_joints(
        robot: Articulation, names: tuple[str, ...]
    ) -> tuple[list[int], list[str]]:
        ids, resolved = robot.find_joints(list(names), preserve_order=True)
        if tuple(resolved) != tuple(names):
            raise ValueError(f"Sharpa joint order mismatch: {resolved} versus {names}.")
        return list(ids), list(resolved)

    def _resolve_tips(self, robot: Articulation, side: str = "") -> tuple[list[int], list[str]]:
        ids, names = robot.find_bodies(self.cfg.fingertip_body_expr.format(side=side))
        if len(ids) != 5:
            raise ValueError(
                f"Sharpa fingertip expression resolved {names}; expected five bodies."
            )
        return list(ids), list(names)

    def _select(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.motion_index, self.timestep_counter]

    @property
    def object_position_e(self) -> torch.Tensor:
        return (_torch(self.object.data.root_link_pos_w) - self._env.scene.env_origins)[
            :, None, :
        ]

    @property
    def object_orientation_e(self) -> torch.Tensor:
        return xyzw_to_wxyz(_torch(self.object.data.root_link_quat_w))[:, None, :]

    @property
    def object_orientation_xyzw_e(self) -> torch.Tensor:
        """Live object orientation in the quaternion order used by Isaac Lab math."""
        return _torch(self.object.data.root_link_quat_w)[:, None, :]

    @property
    def object_body_position_command_e(self) -> torch.Tensor:
        return self._select(self._object_pose)[:, None, :3]

    @property
    def object_body_wxyz_command_e(self) -> torch.Tensor:
        return self._select(self._object_pose)[:, None, 3:7]

    @property
    def object_body_xyzw_command_e(self) -> torch.Tensor:
        return wxyz_to_xyzw(self.object_body_wxyz_command_e)

    @property
    def object_initial_pose_e(self) -> torch.Tensor:
        """Initial pose of the selected reference trajectory in env coordinates."""

        return self._object_initial_pose[self.motion_index]

    @property
    def object_goal_pose_e(self) -> torch.Tensor:
        """Final object pose of the selected reference in env coordinates."""

        return self._object_goal_pose[self.motion_index]

    @property
    def object_mass(self) -> torch.Tensor:
        """Mass of the selected rigid object, from the validated reference metadata."""

        return self._object_mass[self.motion_index]

    @property
    def object_lift_height(self) -> torch.Tensor:
        """Live object height above the pose sampled at episode reset."""

        return (
            self.object_position_e[:, 0, 2]
            - self._episode_initial_object_position[:, 2]
        )

    def _wrist_command(self, side: str) -> torch.Tensor:
        reference_wrist = self._select(
            self._left_wrist if side == "left" else self._right_wrist
        )
        reference_object = self._select(self._object_pose)
        current_object = torch.cat(
            (self.object_position_e[:, 0], self.object_orientation_e[:, 0]), dim=-1
        )
        position, orientation = mdp.retarget_pose_to_live_object(
            reference_object[:, :3],
            reference_object[:, 3:7],
            current_object[:, :3],
            current_object[:, 3:7],
            reference_wrist[:, :3],
            reference_wrist[:, 3:7],
        )
        return torch.cat((position, orientation), dim=-1)

    @property
    def left_hand_wrist_pose_command_e(self) -> torch.Tensor:
        return self._wrist_command("left")

    @property
    def right_hand_wrist_pose_command_e(self) -> torch.Tensor:
        return self._wrist_command("right")

    def wrist_pose_command_xyzw(self, side: str) -> torch.Tensor:
        """Return an object-centric wrist command for Isaac Lab controllers."""
        command = self._wrist_command(side)
        return torch.cat((command[:, :3], wxyz_to_xyzw(command[:, 3:7])), dim=-1)

    @property
    def left_hand_finger_joint_pos_command(self) -> torch.Tensor:
        return self._select(self._left_qpos)

    @property
    def right_hand_finger_joint_pos_command(self) -> torch.Tensor:
        return self._select(self._right_qpos)

    def _root_position(self, robot: Articulation) -> torch.Tensor:
        return _torch(robot.data.root_link_pos_w) - self._env.scene.env_origins

    def _root_wxyz(self, robot: Articulation) -> torch.Tensor:
        return xyzw_to_wxyz(_torch(robot.data.root_link_quat_w))

    @property
    def left_hand_wrist_position_e(self) -> torch.Tensor:
        return self._root_position(self.left_robot)

    @property
    def right_hand_wrist_position_e(self) -> torch.Tensor:
        return self._root_position(self.right_robot)

    @property
    def left_hand_wrist_wxyz_e(self) -> torch.Tensor:
        return self._root_wxyz(self.left_robot)

    @property
    def right_hand_wrist_wxyz_e(self) -> torch.Tensor:
        return self._root_wxyz(self.right_robot)

    @property
    def left_hand_wrist_velocity_b(self) -> torch.Tensor:
        velocity = _torch(self.left_robot.data.root_com_vel_w)
        inverse = mdp.quat_inverse(self.left_hand_wrist_wxyz_e)
        return torch.cat(
            (
                mdp.quat_rotate(inverse, velocity[:, :3]),
                mdp.quat_rotate(inverse, velocity[:, 3:6]),
            ),
            dim=-1,
        )

    @property
    def right_hand_wrist_velocity_b(self) -> torch.Tensor:
        velocity = _torch(self.right_robot.data.root_com_vel_w)
        inverse = mdp.quat_inverse(self.right_hand_wrist_wxyz_e)
        return torch.cat(
            (
                mdp.quat_rotate(inverse, velocity[:, :3]),
                mdp.quat_rotate(inverse, velocity[:, 3:6]),
            ),
            dim=-1,
        )

    @property
    def left_hand_finger_joint_pos(self) -> torch.Tensor:
        return _torch(self.left_robot.data.joint_pos)[:, self.left_finger_joint_ids]

    @property
    def right_hand_finger_joint_pos(self) -> torch.Tensor:
        return _torch(self.right_robot.data.joint_pos)[:, self.right_finger_joint_ids]

    @property
    def left_hand_finger_joint_vel(self) -> torch.Tensor:
        return _torch(self.left_robot.data.joint_vel)[:, self.left_finger_joint_ids]

    @property
    def right_hand_finger_joint_vel(self) -> torch.Tensor:
        return _torch(self.right_robot.data.joint_vel)[:, self.right_finger_joint_ids]

    def _fingertip_positions(self, robot: Articulation, ids: list[int]) -> torch.Tensor:
        return (
            _torch(robot.data.body_link_pos_w)[:, ids]
            - self._env.scene.env_origins[:, None]
        )

    @property
    def left_hand_fingertip_position_e(self) -> torch.Tensor:
        return self._fingertip_positions(self.left_robot, self.left_fingertip_body_ids)

    @property
    def right_hand_fingertip_position_e(self) -> torch.Tensor:
        return self._fingertip_positions(
            self.right_robot, self.right_fingertip_body_ids
        )

    def _tip_command(self, side: str) -> torch.Tensor:
        tips = self._select(self._left_tips if side == "left" else self._right_tips)
        reference_object = self._select(self._object_pose)
        current_object = torch.cat(
            (self.object_position_e[:, 0], self.object_orientation_e[:, 0]), dim=-1
        )
        position, _ = mdp.retarget_pose_to_live_object(
            reference_object[:, None, :3],
            reference_object[:, None, 3:7],
            current_object[:, None, :3],
            current_object[:, None, 3:7],
            tips[..., :3],
            tips[..., 3:7],
        )
        return position

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        return self._tip_command("left")

    @property
    def right_hand_fingertip_position_command_e(self) -> torch.Tensor:
        return self._tip_command("right")

    def contact_active(self, side: str) -> torch.Tensor:
        """Return whether the demonstration requests any contact for one hand."""

        return self._select(self._contact_active[side])

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            (
                self.right_hand_wrist_pose_command_e[:, :3]
                - self.right_hand_wrist_position_e,
                self.left_hand_wrist_pose_command_e[:, :3]
                - self.left_hand_wrist_position_e,
                self.right_hand_finger_joint_pos_command
                - self.right_hand_finger_joint_pos,
                self.left_hand_finger_joint_pos_command
                - self.left_hand_finger_joint_pos,
                (self.object_body_position_command_e - self.object_position_e).flatten(
                    1
                ),
            ),
            dim=-1,
        )

    def _update_metrics(self) -> None:
        self.metrics["right_wrist_position_error"] = torch.linalg.vector_norm(
            self.right_hand_wrist_pose_command_e[:, :3]
            - self.right_hand_wrist_position_e,
            dim=-1,
        )
        self.metrics["left_wrist_position_error"] = torch.linalg.vector_norm(
            self.left_hand_wrist_pose_command_e[:, :3]
            - self.left_hand_wrist_position_e,
            dim=-1,
        )
        self.metrics["object_position_error"] = torch.linalg.vector_norm(
            self.object_body_position_command_e[:, 0] - self.object_position_e[:, 0],
            dim=-1,
        )
        self.metrics["object_goal_position_error"] = torch.linalg.vector_norm(
            self.object_goal_pose_e[:, :3] - self.object_position_e[:, 0],
            dim=-1,
        )
        self.metrics["object_goal_orientation_error"] = mdp.quat_error_magnitude(
            self.object_orientation_e[:, 0], self.object_goal_pose_e[:, 3:7]
        )
        self.metrics["object_lift_height"] = self.object_lift_height
        self.metrics["task_success"] = (
            (self.metrics["object_goal_position_error"] <= 0.05)
            & (self.metrics["object_goal_orientation_error"] <= 0.35)
            & (self.object_lift_height >= 0.05)
        ).float()
        expected_contacts = torch.stack(
            [self.contact_active(side) for side in ("left", "right")], dim=-1
        )
        observed_contacts = torch.stack(
            [sharpa_mdp.live_contact(self._env, side) for side in ("left", "right")],
            dim=-1,
        )
        self.metrics["contact_expected"] = expected_contacts.float().mean(dim=-1)
        self.metrics["contact_observed"] = observed_contacts.float().mean(dim=-1)
        self.metrics["contact_match"] = (
            (expected_contacts == observed_contacts).float().mean(dim=-1)
        )
        self.metrics["virtual_object_controller_scale"] = (
            self.virtual_object_controller_scale_factor_per_env[:, 0]
        )

    def _sample_start_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        """Sample one start frame per reset environment (task-variant rule)."""

        if self.cfg.always_reset_to_first_frame:
            return torch.zeros_like(lengths)
        start = (
            (torch.rand(len(lengths), device=self.device) * lengths.float())
            .long()
            .clamp_max(lengths - 1)
        )
        first = torch.rand(len(lengths), device=self.device) < float(
            self.cfg.reset_to_first_frame_prob
        )
        start[first] = 0
        return start

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(ids) == 0:
            return
        if self.cfg.randomize_reference and len(self._lengths) > 1:
            ranks = torch.randint(len(self._lengths), (len(ids),), device=self.device)
        else:
            ranks = torch.zeros(len(ids), dtype=torch.long, device=self.device)
        lengths = self._lengths[ranks]
        start = self._sample_start_frames(lengths)
        self.trajectory_manager.set_env_cursor(
            env_ids=ids,
            ranks=ranks,
            steps=start,
        )
        self.tracking_lengths[ids] = (lengths - start).clamp_min(1)
        self.steps_since_last_reset[ids] = 0
        self.virtual_object_controller_scale_factor_per_env[ids] = 1.0

        motion = self.motion_index[ids]
        frame = self.timestep_counter[ids]
        origins = self._env.scene.env_origins[ids]
        object_pose = self._object_pose[motion, frame].clone()
        object_pose[:, :3] += origins
        self._episode_initial_object_position[ids] = object_pose[:, :3] - origins
        self.object.write_root_link_pose_to_sim(
            torch.cat((object_pose[:, :3], wxyz_to_xyzw(object_pose[:, 3:7])), dim=-1),
            env_ids=ids,
        )
        zero_root_velocity = torch.zeros(len(ids), 6, device=self.device)
        self.object.write_root_com_velocity_to_sim(zero_root_velocity, env_ids=ids)
        self._reset_robot_state(ids)

    def _reset_robot_state(self, ids: torch.Tensor) -> None:
        """Reset independent hand roots; mounted variants reset one articulation."""
        motion = self.motion_index[ids]
        frame = self.timestep_counter[ids]
        origins = self._env.scene.env_origins[ids]
        zero_root_velocity = torch.zeros(len(ids), 6, device=self.device)
        openness = torch.rand(len(ids), 1, device=self.device) * float(
            self.cfg.reset_finger_openness
        )
        for side, robot, qpos, wrist in (
            ("left", self.left_robot, self._left_qpos, self._left_wrist),
            ("right", self.right_robot, self._right_qpos, self._right_wrist),
        ):
            pose = wrist[motion, frame].clone()
            pose[:, :3] += origins
            robot.write_root_link_pose_to_sim(
                torch.cat((pose[:, :3], wxyz_to_xyzw(pose[:, 3:7])), dim=-1),
                env_ids=ids,
            )
            robot.write_root_com_velocity_to_sim(zero_root_velocity, env_ids=ids)
            joints = qpos[motion, frame] * openness
            robot.write_joint_state_to_sim(
                joints, torch.zeros_like(joints), env_ids=ids
            )

    def _update_command(self) -> None:
        self.steps_since_last_reset += 1
        decay_steps = max(1, int(self.cfg.virtual_object_control_decay_steps))
        decay = (1.0 - self.steps_since_last_reset.float() / decay_steps).clamp(
            0.0, 1.0
        )
        curriculum = self.virtual_object_controller_scale_factor.expand_as(decay)
        self.virtual_object_controller_scale_factor_per_env[:, 0] = torch.maximum(
            decay, curriculum
        )
        horizon = self._lengths[self.motion_index]
        advance = reference_advance_mask(
            self.timestep_counter,
            horizon,
            self.steps_since_last_reset,
            legacy_reset_hold=self.cfg.legacy_reset_hold,
            hold_steps=decay_steps,
        )
        self.trajectory_manager.advance_cursors(
            torch.nonzero(advance, as_tuple=False).flatten()
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        del debug_vis

    def _debug_vis_callback(self, event: Any) -> None:
        del event


SharpaReferenceCommandCfg.class_type = SharpaReferenceCommand

__all__ = [
    "SharpaReferenceCommand",
    "SharpaReferenceCommandCfg",
    "reference_advance_mask",
]
