# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Motion-backed reference command term for the fixed-base Vega-Wuji robot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .newton_contacts import NewtonContactPairAdapter

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def wxyz_to_xyzw_tensor(value: torch.Tensor) -> torch.Tensor:
    """Convert WXYZ quaternions to Isaac Lab 3.0 XYZW."""

    return torch.cat((value[..., 1:4], value[..., 0:1]), dim=-1)


def xyzw_to_wxyz_tensor(value: torch.Tensor) -> torch.Tensor:
    """Convert Isaac Lab 3.0 XYZW quaternions to WXYZ."""

    return torch.cat((value[..., 3:4], value[..., 0:3]), dim=-1)


def dominant_contact_object_indices(
    object_indices: torch.Tensor,
    active: torch.Tensor,
    lengths: torch.Tensor,
    *,
    num_objects: int,
) -> torch.Tensor:
    """Select each hand's dominant object and backfill no-contact frames."""

    if object_indices.shape != active.shape or object_indices.ndim != 3:
        raise ValueError("object_indices and active must have shape [M, T, C].")
    if lengths.shape != (object_indices.shape[0],):
        raise ValueError("lengths must have shape [M].")
    if num_objects <= 0:
        raise ValueError("num_objects must be positive.")
    result = torch.zeros(
        object_indices.shape[:2], dtype=torch.long, device=object_indices.device
    )
    for motion_index in range(object_indices.shape[0]):
        length = int(lengths[motion_index].item())
        if not 1 <= length <= object_indices.shape[1]:
            raise ValueError("Each Reference length must be within the packed horizon.")
        next_object = 0
        for frame_index in range(length - 1, -1, -1):
            frame_active = active[motion_index, frame_index]
            if torch.any(frame_active):
                values = object_indices[motion_index, frame_index, frame_active]
                if torch.any(values < 0) or torch.any(values >= num_objects):
                    raise ValueError("Active contacts must identify a declared object.")
                next_object = int(
                    torch.bincount(values, minlength=num_objects).argmax().item()
                )
            result[motion_index, frame_index] = next_object
        result[motion_index, length:] = result[motion_index, length - 1]
    return result


def match_newton_counterpart_columns(
    object_body_labels: Sequence[str],
    counterpart_body_labels: Sequence[str],
) -> tuple[int, ...]:
    """Map each scene object's full Newton body label to one force column."""

    columns: list[int] = []
    for label in object_body_labels:
        matches = [
            index
            for index, counterpart in enumerate(counterpart_body_labels)
            if counterpart == label
        ]
        if len(matches) != 1:
            raise ValueError(
                "Each Reference object must match exactly one Newton contact "
                f"counterpart; {label!r} matched {len(matches)}."
            )
        columns.append(matches[0])
    if len(set(columns)) != len(columns):
        raise ValueError("Reference objects must use distinct Newton force columns.")
    return tuple(columns)


class VegaWujiReferenceCommand(CommandTerm):
    """Publish object-relative hand and object targets from ILTools References."""

    cfg: VegaWujiReferenceCommandCfg

    def __init__(
        self, cfg: VegaWujiReferenceCommandCfg, env: ManagerBasedRLEnv
    ) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.objects: list[RigidObject] = [
            env.scene[name] for name in env._reference_object_names
        ]
        self.object_names = tuple(env._reference_object_names)
        self.num_bodies = len(self.objects)
        self._reference_qpos = env._reference_qpos
        self._reference_qvel = env._reference_qvel
        self._reference_lengths = env._reference_lengths
        self._reference_wrist_pose_wxyz = env._reference_wrist_pose_wxyz
        self._reference_object_pose_wxyz = env._reference_object_pose_wxyz
        self._reference_object_twist_w = env._reference_object_twist_w
        self._reference_hand_frame_pose_wxyz = env._reference_hand_frame_pose_wxyz
        self._reference_contacts = env._reference_contacts
        self._reference_contact_link_names = env._reference_contact_link_names
        self.object_radii = env._reference_object_radii
        self._reference_hand_object_indices = {
            side: dominant_contact_object_indices(
                self._reference_contacts[side]["object_indices"],
                self._reference_contacts[side]["active"],
                self._reference_lengths,
                num_objects=self.num_bodies,
            )
            for side in ("right", "left")
        }
        self.refresh_object_com_poses()

        self._motion_count = int(self._reference_qpos.shape[0])
        if not 0 <= int(cfg.fixed_motion_index) < self._motion_count:
            raise ValueError(
                f"fixed_motion_index must be within [0, {self._motion_count}), "
                f"got {cfg.fixed_motion_index}."
            )
        self.timestep_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.tracking_lengths = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.steps_since_last_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.virtual_object_controller_curriculum_scale = torch.tensor(
            float(cfg.initial_virtual_object_control_curriculum_scale),
            dtype=torch.float32,
            device=self.device,
        )
        self.virtual_object_controller_scale = torch.ones(
            self.num_envs, 1, dtype=torch.float32, device=self.device
        )

        self._wrist_body_ids: dict[str, int] = {}
        self._finger_joint_ids: dict[str, list[int]] = {}
        self._fingertip_body_ids: dict[str, list[int]] = {}
        for side in ("right", "left"):
            wrist_ids, wrist_names = self.robot.find_bodies(
                [getattr(cfg, f"{side}_wrist_body_name")], preserve_order=True
            )
            finger_ids, finger_names = self.robot.find_joints(
                [getattr(cfg, f"{side}_finger_joint_expr")], preserve_order=True
            )
            fingertip_ids, fingertip_names = self.robot.find_bodies(
                list(getattr(cfg, f"{side}_fingertip_body_names")),
                preserve_order=True,
            )
            if len(wrist_ids) != 1:
                raise ValueError(
                    f"{side} wrist body must resolve once; got {wrist_names}."
                )
            if len(finger_ids) != 20:
                raise ValueError(
                    f"{side} finger expression must resolve 20 joints; got "
                    f"{finger_names}."
                )
            if len(fingertip_ids) != 5:
                raise ValueError(
                    f"{side} fingertip names must resolve 5 bodies; got "
                    f"{fingertip_names}."
                )
            self._wrist_body_ids[side] = int(wrist_ids[0])
            self._finger_joint_ids[side] = list(finger_ids)
            self._fingertip_body_ids[side] = list(fingertip_ids)

        self._contact_body_ids: dict[str, list[int]] = {}
        self._contact_sensor_indices: dict[str, list[int]] = {}
        self._contact_object_indices: dict[str, tuple[int, ...]] = {}
        self._contact_sensors: dict[str, Any | None] = {}
        self._contact_adapters: dict[str, NewtonContactPairAdapter | None] = {}
        for side in ("right", "left"):
            link_names = self._reference_contact_link_names[side]
            if link_names:
                body_ids, resolved = self.robot.find_bodies(
                    list(link_names), preserve_order=True
                )
                if tuple(resolved) != tuple(link_names):
                    raise ValueError(
                        f"{side} Reference contact links differ from robot bodies: "
                        f"{link_names} versus {resolved}."
                    )
                self._contact_body_ids[side] = list(body_ids)
            else:
                self._contact_body_ids[side] = []
            sensor_name = f"{side}_hand_object_contacts"
            sensor = env.scene.sensors.get(sensor_name)
            self._contact_sensors[side] = sensor
            if sensor is None:
                self._contact_sensor_indices[side] = []
                self._contact_object_indices[side] = ()
                self._contact_adapters[side] = None
            else:
                sensor_names = tuple(str(name) for name in sensor.body_names)
                missing = [name for name in link_names if name not in sensor_names]
                if missing:
                    raise ValueError(
                        f"Newton {side} contact sensor is missing bodies: {missing}."
                    )
                self._contact_sensor_indices[side] = [
                    sensor_names.index(name) for name in link_names
                ]
                contact_view = sensor.contact_view
                if contact_view.counterpart_type != "body":
                    raise ValueError(
                        "The Newton contact filter must resolve body counterparts."
                    )
                counterpart_rows = contact_view.counterpart_indices
                if not counterpart_rows:
                    raise ValueError(
                        "The Newton contact filter matched no counterparts."
                    )
                model_labels = tuple(
                    str(item) for item in contact_view._model.body_label
                )
                counterpart_labels = tuple(
                    model_labels[index] for index in counterpart_rows[0]
                )
                object_labels: list[str] = []
                for object_name, item in zip(
                    self.object_names, self.objects, strict=True
                ):
                    labels = tuple(str(label) for label in item.root_view.link_labels)
                    if len(labels) != 1:
                        raise ValueError(
                            f"Reference object {object_name!r} must resolve one Newton "
                            f"rigid body; got {labels}."
                        )
                    object_labels.append(labels[0])
                self._contact_object_indices[side] = match_newton_counterpart_columns(
                    object_labels,
                    counterpart_labels,
                )
                self._contact_adapters[side] = (
                    NewtonContactPairAdapter.from_contact_sensor(
                        sensor,
                        link_body_names=link_names,
                        object_body_names=self.object_names,
                        force_threshold=float(cfg.contact_force_threshold),
                    )
                    if link_names
                    else None
                )

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(cfg.wrench_basis_seed))
        self.wrench_space_bases = torch.stack(
            [
                mdp.sample_wrench_space_basis_scaled(
                    int(cfg.num_wrench_basis),
                    1.0,
                    self.device,
                    generator=generator,
                )
                for _radius in self.object_radii.tolist()
            ],
            dim=0,
        )
        self._support_cache_step = -1
        self._support_cache_frames: torch.Tensor | None = None
        self._support_cache_motions: torch.Tensor | None = None
        self._live_contact_cache_step = -1
        self._live_contact_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        support_shape = (
            self.num_envs,
            self.num_bodies,
            int(cfg.num_wrench_basis),
        )
        self._command_supports = {
            side: torch.zeros(support_shape, device=self.device)
            for side in ("right", "left")
        }
        self._current_supports = {
            side: torch.zeros(support_shape, device=self.device)
            for side in ("right", "left")
        }

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

    @property
    def retargeted_horizon(self) -> torch.Tensor:
        """Per-environment end frame for the selected Reference."""

        return self._reference_lengths[self.motion_index]

    @property
    def reference_joint_pos(self) -> torch.Tensor:
        return self._reference_qpos[self.motion_index, self.timestep_counter]

    @property
    def reference_joint_vel(self) -> torch.Tensor:
        return self._reference_qvel[self.motion_index, self.timestep_counter]

    @property
    def action_target_frame(self) -> torch.Tensor:
        """Reference frame targeted by the next policy transition.

        Isaac Lab computes rewards and terminations before advancing managed
        commands, then advances the command before emitting the next
        observation.  Consequently ``timestep_counter`` in an observation is
        already the target for the action selected from that observation.  An
        additional ``+1`` here would skip a Reference frame.
        """

        return self.timestep_counter

    @property
    def current_robot_state(self) -> torch.Tensor:
        """Live robot state as ``[joint_pos, joint_vel]`` in runtime order."""

        return torch.cat(
            (self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch),
            dim=-1,
        )

    def refresh_object_com_poses(self) -> None:
        """Refresh cached link-to-COM poses after runtime inertia writes."""

        object_com_positions_b = torch.stack(
            [item.data.body_com_pos_b.torch[:, 0] for item in self.objects], dim=1
        )
        identity_wxyz = torch.zeros(
            (*object_com_positions_b.shape[:-1], 4),
            dtype=object_com_positions_b.dtype,
            device=object_com_positions_b.device,
        )
        identity_wxyz[..., 0] = 1.0
        self._object_com_poses_b_wxyz = torch.cat(
            (object_com_positions_b, identity_wxyz),
            dim=-1,
        )

    @property
    def current_object_state(self) -> torch.Tensor:
        """Live per-object XYZ+WXYZ pose and world linear/angular twist.

        Positions are environment-local while their axes, orientations, and
        twists remain world-aligned.  The result interleaves each object's
        13-value state before flattening across objects.
        """

        pose = torch.cat((self.object_position_e, self.object_orientation_e), dim=-1)
        return torch.cat((pose, self.object_twist_w), dim=-1).reshape(self.num_envs, -1)

    @property
    def next_desired_robot_state(self) -> torch.Tensor:
        """Action-aligned retargeted state as ``[joint_pos, joint_vel]``.

        The name describes transition semantics: this is the state the policy
        should produce after its next action.  Its frame is exactly
        :attr:`action_target_frame`.
        """

        return torch.cat((self.reference_joint_pos, self.reference_joint_vel), dim=-1)

    @property
    def next_desired_object_state(self) -> torch.Tensor:
        """Action-aligned per-object XYZ+WXYZ pose and world-frame twist."""

        pose = self._reference_object_pose_wxyz[
            self.motion_index, self.action_target_frame
        ]
        twist = self._reference_object_twist_w[
            self.motion_index, self.action_target_frame
        ]
        return torch.cat((pose, twist), dim=-1).reshape(self.num_envs, -1)

    def _reference_wrist_pose(self, side: str) -> torch.Tensor:
        return self._reference_wrist_pose_wxyz[side][
            self.motion_index, self.timestep_counter
        ]

    @property
    def object_body_position_command_e(self) -> torch.Tensor:
        return self._reference_object_pose_wxyz[
            self.motion_index, self.timestep_counter, :, :3
        ]

    @property
    def object_body_wxyz_command_e(self) -> torch.Tensor:
        return self._reference_object_pose_wxyz[
            self.motion_index, self.timestep_counter, :, 3:7
        ]

    @property
    def object_body_twist_command_w(self) -> torch.Tensor:
        """Return object target twists as world linear XYZ then angular XYZ."""

        return self._reference_object_twist_w[self.motion_index, self.timestep_counter]

    @property
    def object_twist_w(self) -> torch.Tensor:
        """Return live root-link twists as world linear then angular velocity."""

        return torch.stack(
            [item.data.root_link_vel_w.torch for item in self.objects], dim=1
        )

    @property
    def object_body_xyzw_command_e(self) -> torch.Tensor:
        return wxyz_to_xyzw_tensor(self.object_body_wxyz_command_e)

    def object_target_pose(self, asset_name: str) -> torch.Tensor:
        """Return one object target in environment-local XYZ+XYZW form."""

        try:
            object_index = self.object_names.index(asset_name)
        except ValueError as exc:
            raise KeyError(f"Unknown Reference object: {asset_name!r}.") from exc
        return torch.cat(
            (
                self.object_body_position_command_e[:, object_index],
                self.object_body_xyzw_command_e[:, object_index],
            ),
            dim=-1,
        )

    @property
    def object_position_e(self) -> torch.Tensor:
        return torch.stack(
            [
                item.data.root_link_pos_w.torch - self._env.scene.env_origins
                for item in self.objects
            ],
            dim=1,
        )

    @property
    def object_position_w(self) -> torch.Tensor:
        return torch.stack(
            [item.data.root_link_pos_w.torch for item in self.objects], dim=1
        )

    @property
    def object_com_position_and_wxyz_w(self) -> torch.Tensor:
        """Return live object COM poses as XYZ plus WXYZ."""

        poses = torch.stack(
            [item.data.root_com_pose_w.torch for item in self.objects], dim=1
        )
        return torch.cat((poses[..., :3], xyzw_to_wxyz_tensor(poses[..., 3:7])), dim=-1)

    @property
    def object_orientation_e(self) -> torch.Tensor:
        return torch.stack(
            [
                xyzw_to_wxyz_tensor(item.data.root_link_quat_w.torch)
                for item in self.objects
            ],
            dim=1,
        )

    def _reference_contact_object_index(self, side: str) -> torch.Tensor:
        return self._reference_hand_object_indices[side][
            self.motion_index, self.timestep_counter
        ]

    def _wrist_pose_command_wxyz(self, side: str) -> torch.Tensor:
        reference_wrist = self._reference_wrist_pose(side)
        object_index = self._reference_contact_object_index(side)
        env_ids = torch.arange(self.num_envs, device=self.device)
        reference_object = self._reference_object_pose_wxyz[
            self.motion_index, self.timestep_counter, object_index
        ]
        live_object_position = self.object_position_e[env_ids, object_index]
        live_object_wxyz = self.object_orientation_e[env_ids, object_index]
        position, orientation = mdp.retarget_pose_to_live_object(
            reference_object[:, :3],
            reference_object[:, 3:7],
            live_object_position,
            live_object_wxyz,
            reference_wrist[:, :3],
            reference_wrist[:, 3:7],
        )
        assert orientation is not None
        return torch.cat((position, orientation), dim=-1)

    def wrist_pose_command(self, side: str) -> torch.Tensor:
        """Return an Isaac XYZW wrist target for the action adapter."""

        pose = self._wrist_pose_command_wxyz(side)
        return torch.cat((pose[:, :3], wxyz_to_xyzw_tensor(pose[:, 3:7])), dim=-1)

    @property
    def right_hand_wrist_pose_command_e(self) -> torch.Tensor:
        return self._wrist_pose_command_wxyz("right")

    @property
    def left_hand_wrist_pose_command_e(self) -> torch.Tensor:
        return self._wrist_pose_command_wxyz("left")

    def finger_joint_command(self, side: str) -> torch.Tensor:
        return self.reference_joint_pos[:, self._finger_joint_ids[side]]

    @property
    def right_hand_finger_joint_pos_command(self) -> torch.Tensor:
        return self.finger_joint_command("right")

    @property
    def left_hand_finger_joint_pos_command(self) -> torch.Tensor:
        return self.finger_joint_command("left")

    def _wrist_position_w(self, side: str) -> torch.Tensor:
        return self.robot.data.body_link_pos_w.torch[:, self._wrist_body_ids[side]]

    def _wrist_position_e(self, side: str) -> torch.Tensor:
        return self._wrist_position_w(side) - self._env.scene.env_origins

    def _wrist_wxyz(self, side: str) -> torch.Tensor:
        return xyzw_to_wxyz_tensor(
            self.robot.data.body_link_quat_w.torch[:, self._wrist_body_ids[side]]
        )

    def _wrist_velocity_b(self, side: str) -> torch.Tensor:
        velocity_w = self.robot.data.body_link_vel_w.torch[
            :, self._wrist_body_ids[side]
        ]
        inverse = mdp.quat_inverse(self._wrist_wxyz(side))
        return torch.cat(
            (
                mdp.quat_rotate(inverse, velocity_w[:, :3]),
                mdp.quat_rotate(inverse, velocity_w[:, 3:6]),
            ),
            dim=-1,
        )

    @property
    def right_hand_wrist_position_w(self) -> torch.Tensor:
        return self._wrist_position_w("right")

    @property
    def left_hand_wrist_position_w(self) -> torch.Tensor:
        return self._wrist_position_w("left")

    @property
    def right_hand_wrist_position_e(self) -> torch.Tensor:
        return self._wrist_position_e("right")

    @property
    def left_hand_wrist_position_e(self) -> torch.Tensor:
        return self._wrist_position_e("left")

    @property
    def right_hand_wrist_wxyz_e(self) -> torch.Tensor:
        return self._wrist_wxyz("right")

    @property
    def left_hand_wrist_wxyz_e(self) -> torch.Tensor:
        return self._wrist_wxyz("left")

    @property
    def right_hand_wrist_velocity_b(self) -> torch.Tensor:
        return self._wrist_velocity_b("right")

    @property
    def left_hand_wrist_velocity_b(self) -> torch.Tensor:
        return self._wrist_velocity_b("left")

    @property
    def right_hand_finger_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch[:, self._finger_joint_ids["right"]]

    @property
    def left_hand_finger_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.torch[:, self._finger_joint_ids["left"]]

    @property
    def right_hand_finger_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch[:, self._finger_joint_ids["right"]]

    @property
    def left_hand_finger_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.torch[:, self._finger_joint_ids["left"]]

    def _finger_joint_pos_scaled(self, side: str) -> torch.Tensor:
        joint_ids = self._finger_joint_ids[side]
        position = self.robot.data.joint_pos.torch[:, joint_ids]
        limits = self.robot.data.joint_pos_limits.torch[:, joint_ids]
        return (
            2.0
            * (position - limits[..., 0])
            / (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
            - 1.0
        )

    @property
    def right_hand_finger_joint_pos_scaled(self) -> torch.Tensor:
        return self._finger_joint_pos_scaled("right")

    @property
    def left_hand_finger_joint_pos_scaled(self) -> torch.Tensor:
        return self._finger_joint_pos_scaled("left")

    def _fingertip_position_e(self, side: str) -> torch.Tensor:
        return self.robot.data.body_link_pos_w.torch[
            :, self._fingertip_body_ids[side]
        ] - self._env.scene.env_origins.unsqueeze(1)

    @property
    def right_hand_fingertip_position_e(self) -> torch.Tensor:
        return self._fingertip_position_e("right")

    @property
    def left_hand_fingertip_position_e(self) -> torch.Tensor:
        return self._fingertip_position_e("left")

    def _fingertip_position_command_e(self, side: str) -> torch.Tensor:
        reference_frames = self._reference_hand_frame_pose_wxyz[side][
            self.motion_index, self.timestep_counter
        ]
        object_index = self._reference_contact_object_index(side)
        env_ids = torch.arange(self.num_envs, device=self.device)
        reference_object = self._reference_object_pose_wxyz[
            self.motion_index, self.timestep_counter, object_index
        ]
        live_object_position = self.object_position_e[env_ids, object_index]
        live_object_wxyz = self.object_orientation_e[env_ids, object_index]
        position, _ = mdp.retarget_pose_to_live_object(
            reference_object[:, None, :3],
            reference_object[:, None, 3:7],
            live_object_position[:, None, :],
            live_object_wxyz[:, None, :],
            reference_frames[:, :, :3],
        )
        return position

    @property
    def right_hand_fingertip_position_command_e(self) -> torch.Tensor:
        return self._fingertip_position_command_e("right")

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        return self._fingertip_position_command_e("left")

    def _reference_contacts_grouped(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._reference_contacts[side]
        positions = data["object_positions_w"][self.motion_index, self.timestep_counter]
        normals = data["link_normals_w"][self.motion_index, self.timestep_counter]
        indices = data["object_indices"][self.motion_index, self.timestep_counter]
        active = data["active"][self.motion_index, self.timestep_counter]
        slots = positions.shape[1]
        grouped_positions = torch.zeros(
            self.num_envs,
            self.num_bodies,
            slots,
            3,
            device=self.device,
        )
        grouped_normals = torch.zeros_like(grouped_positions)
        grouped_active = torch.zeros(
            self.num_envs,
            self.num_bodies,
            slots,
            dtype=torch.bool,
            device=self.device,
        )
        safe_indices = indices.clamp(0, self.num_bodies - 1)
        for body_index in range(self.num_bodies):
            mask = active & (safe_indices == body_index)
            grouped_positions[:, body_index] = positions * mask.unsqueeze(-1)
            grouped_normals[:, body_index] = normals * mask.unsqueeze(-1)
            grouped_active[:, body_index] = mask
        return grouped_positions, grouped_normals, grouped_active

    def _live_contacts_grouped(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._refresh_live_contact_cache()
        return self._live_contact_cache[side]

    def _empty_live_contacts(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = len(self._contact_body_ids[side])
        positions = torch.zeros(
            self.num_envs, self.num_bodies, slots, 3, device=self.device
        )
        forces = torch.zeros_like(positions)
        active = torch.zeros(
            self.num_envs,
            self.num_bodies,
            slots,
            dtype=torch.bool,
            device=self.device,
        )
        return positions, forces, active

    def _refresh_live_contact_cache(self) -> None:
        """Aggregate exact Newton contact points and forces once per step."""

        current_step = int(self._env.common_step_counter)
        if self._live_contact_cache_step == current_step:
            return
        valid_envs = self.steps_since_last_reset > 0
        for side in ("right", "left"):
            adapter = self._contact_adapters[side]
            if adapter is None:
                self._live_contact_cache[side] = self._empty_live_contacts(side)
                continue
            pairs = adapter.refresh(valid_env_mask=valid_envs)
            self._live_contact_cache[side] = (
                pairs.positions_w,
                pairs.forces_on_object_w,
                pairs.active,
            )
        self._live_contact_cache_step = current_step

    def _refresh_support_cache(self) -> None:
        current_step = int(self._env.common_step_counter)
        if (
            self._support_cache_step == current_step
            and self._support_cache_frames is not None
            and self._support_cache_motions is not None
            and torch.equal(self._support_cache_frames, self.timestep_counter)
            and torch.equal(self._support_cache_motions, self.motion_index)
        ):
            return
        object_com_pose = self.object_com_position_and_wxyz_w
        reference_object = self._reference_object_pose_wxyz[
            self.motion_index, self.timestep_counter
        ]
        reference_com_position, reference_com_wxyz = mdp.compose_pose(
            reference_object[..., :3],
            reference_object[..., 3:7],
            self._object_com_poses_b_wxyz[..., :3],
            self._object_com_poses_b_wxyz[..., 3:7],
        )
        assert reference_com_wxyz is not None
        for side in ("right", "left"):
            ref_position, ref_normal, ref_active = self._reference_contacts_grouped(
                side
            )
            self._command_supports[side] = mdp.compute_contact_wrench_supports(
                ref_position,
                ref_normal,
                reference_com_position,
                reference_com_wxyz,
                self.wrench_space_bases,
                self.object_radii,
                contact_active=ref_active,
                num_friction_cone_edges=int(self.cfg.num_friction_cone_edges),
                friction_coefficient=float(self.cfg.friction_coefficient),
            )
            live_position, live_force, live_active = self._live_contacts_grouped(side)
            self._current_supports[side] = mdp.compute_contact_wrench_supports(
                live_position,
                live_force,
                object_com_pose[..., :3],
                object_com_pose[..., 3:7],
                self.wrench_space_bases,
                self.object_radii,
                contact_active=live_active,
                num_friction_cone_edges=int(self.cfg.num_friction_cone_edges),
                friction_coefficient=float(self.cfg.friction_coefficient),
            )
        self._support_cache_step = current_step
        self._support_cache_frames = self.timestep_counter.clone()
        self._support_cache_motions = self.motion_index.clone()

    @property
    def right_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._command_supports["right"]

    @property
    def left_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._command_supports["left"]

    @property
    def right_hand_contact_wrench_supports(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._current_supports["right"]

    @property
    def left_hand_contact_wrench_supports(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._current_supports["left"]

    @property
    def right_hand_object_contact_positions_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("right")[0]

    @property
    def left_hand_object_contact_positions_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("left")[0]

    @property
    def right_hand_object_contact_forces_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("right")[1]

    @property
    def left_hand_object_contact_forces_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("left")[1]

    @property
    def right_hand_object_contact_active(self) -> torch.Tensor:
        return self._live_contacts_grouped("right")[2]

    @property
    def left_hand_object_contact_active(self) -> torch.Tensor:
        return self._live_contacts_grouped("left")[2]

    @property
    def command(self) -> torch.Tensor:
        right_delta = mdp.pose_delta_command(
            self.right_hand_wrist_position_e,
            self.right_hand_wrist_wxyz_e,
            self.right_hand_wrist_pose_command_e[:, :3],
            self.right_hand_wrist_pose_command_e[:, 3:7],
        )
        left_delta = mdp.pose_delta_command(
            self.left_hand_wrist_position_e,
            self.left_hand_wrist_wxyz_e,
            self.left_hand_wrist_pose_command_e[:, :3],
            self.left_hand_wrist_pose_command_e[:, 3:7],
        )
        object_delta = mdp.pose_delta_command(
            self.object_position_e,
            self.object_orientation_e,
            self.object_body_position_command_e,
            self.object_body_wxyz_command_e,
        )
        return torch.cat(
            (
                right_delta,
                left_delta,
                self.right_hand_finger_joint_pos_command
                - self.right_hand_finger_joint_pos,
                self.left_hand_finger_joint_pos_command
                - self.left_hand_finger_joint_pos,
                object_delta.reshape(self.num_envs, -1),
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
            self.object_body_position_command_e - self.object_position_e,
            dim=-1,
        ).mean(dim=-1)
        self.metrics["virtual_object_controller_scale"] = (
            self.virtual_object_controller_scale.squeeze(-1)
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if bool(self.cfg.randomize_reference) and self._motion_count > 1:
            self.motion_index[ids] = torch.randint(
                self._motion_count,
                (len(ids),),
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.motion_index[ids] = int(self.cfg.fixed_motion_index)
        lengths = self._reference_lengths[self.motion_index[ids]]
        start = mdp.sample_reference_start_frames(
            lengths,
            always_reset_to_first_frame=bool(self.cfg.always_reset_to_first_frame),
            reset_to_first_frame_prob=float(self.cfg.reset_to_first_frame_prob),
            virtual_object_control_scale=(
                self.virtual_object_controller_curriculum_scale
            ),
        )
        self.timestep_counter[ids] = start
        self.tracking_lengths[ids] = (lengths - start).clamp_min(1)
        self.steps_since_last_reset[ids] = 0
        self.virtual_object_controller_scale[ids] = (
            mdp.virtual_object_controller_scale_schedule(
                self.steps_since_last_reset[ids],
                warmup_steps=int(self.cfg.warmup_steps),
                curriculum_scale=self.virtual_object_controller_curriculum_scale,
                force_unassisted=bool(self.cfg.force_unassisted_object_control),
            )
        )
        self._support_cache_step = -1
        self._live_contact_cache_step = -1

    def _update_command(self) -> None:
        self.steps_since_last_reset += 1
        self.virtual_object_controller_scale[:] = (
            mdp.virtual_object_controller_scale_schedule(
                self.steps_since_last_reset,
                warmup_steps=int(self.cfg.warmup_steps),
                curriculum_scale=self.virtual_object_controller_curriculum_scale,
                force_unassisted=bool(self.cfg.force_unassisted_object_control),
            )
        )
        # The Reference frame advances from the first step. The warmup only
        # ramps the virtual-object-controller scale. Holding the frame through
        # the warmup froze the whole episode, because an episode ends well
        # before ``warmup_steps``, so the task never tracked a trajectory.
        horizon = self._reference_lengths[self.motion_index]
        advance = self.timestep_counter < horizon - 1
        self.timestep_counter[advance] += 1
        self._support_cache_step = -1
        self._live_contact_cache_step = -1


@configclass
class VegaWujiReferenceCommandCfg(CommandTermCfg):
    """Configuration for the Vega-Wuji motion command."""

    class_type: type[CommandTerm] = VegaWujiReferenceCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
    asset_name: str = "robot"
    randomize_reference: bool = True
    fixed_motion_index: int = 0
    always_reset_to_first_frame: bool = False
    reset_to_first_frame_prob: float = 0.1
    randomize_reset_finger_openness: bool = False
    reset_finger_openness: float = 0.7
    warmup_steps: int = 20
    initial_virtual_object_control_curriculum_scale: float = 1.0
    force_unassisted_object_control: bool = False
    # Runtime bodies are identity-equivalent to the named Reference sites.
    right_wrist_body_name: str = "r_mount"
    left_wrist_body_name: str = "l_mount"
    right_wrist_reference_frame_name: str = "right_palm"
    left_wrist_reference_frame_name: str = "left_palm"
    right_finger_joint_expr: str = "r_.*"
    left_finger_joint_expr: str = "l_.*"
    right_fingertip_body_names: tuple[str, ...] = (
        "r_thumb_distal",
        "r_index_finger_distal",
        "r_middle_finger_distal",
        "r_ring_finger_distal",
        "r_pinky_distal",
    )
    left_fingertip_body_names: tuple[str, ...] = (
        "l_thumb_distal",
        "l_index_finger_distal",
        "l_middle_finger_distal",
        "l_ring_finger_distal",
        "l_pinky_distal",
    )
    num_wrench_basis: int = 512
    wrench_basis_seed: int = 17
    num_friction_cone_edges: int = 8
    friction_coefficient: float = 0.1
    contact_force_threshold: float = 0.1


__all__ = [
    "dominant_contact_object_indices",
    "match_newton_counterpart_columns",
    "VegaWujiReferenceCommand",
    "VegaWujiReferenceCommandCfg",
    "wxyz_to_xyzw_tensor",
    "xyzw_to_wxyz_tensor",
]
