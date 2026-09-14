# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Newton-compatible Reference-residual actions for the Vega-Wuji assembly.

The default adapter gives the policy one residual for each of the 59 live
actuators.  The older wrist-pose adapter remains available for checkpoint and
configuration compatibility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


SIDE_ORDER = ("right", "left")
"""Policy action order.  This matches the released recipe's task config."""

VEGA_WUJI_JOINT_COUNT = 59
"""Number of actuators in the validated Vega plus two Wuji-hand asset."""

WUJI_ANATOMICAL_LIMITS_DEG: tuple[tuple[str, float | None, float | None], ...] = (
    ("mcp_abd", -40.0, 40.0),
    ("_pip", 0.0, 100.0),
    ("_dip", 0.0, 80.0),
    ("thumb_mcp", 0.0, None),
    ("thumb_ip", 0.0, 100.0),
)
"""Runtime finger envelope from the adopted Wuji retargeting recipe."""


def wuji_anatomical_joint_limits(
    joint_names: Sequence[str], soft_joint_pos_limits: torch.Tensor
) -> torch.Tensor:
    """Intersect live soft limits with the adopted Wuji finger envelope."""

    names = tuple(str(name) for name in joint_names)
    if soft_joint_pos_limits.ndim < 2 or soft_joint_pos_limits.shape[-2:] != (
        len(names),
        2,
    ):
        raise ValueError("soft_joint_pos_limits must end with shape [J, 2].")
    result = soft_joint_pos_limits.clone()
    for joint_index, name in enumerate(names):
        match = next(
            (row for row in WUJI_ANATOMICAL_LIMITS_DEG if row[0] in name), None
        )
        if match is None:
            continue
        _, lower_deg, upper_deg = match
        if lower_deg is not None:
            lower = result.new_tensor(math.radians(lower_deg))
            result[..., joint_index, 0] = torch.maximum(
                result[..., joint_index, 0], lower
            )
        if upper_deg is not None:
            upper = result.new_tensor(math.radians(upper_deg))
            result[..., joint_index, 1] = torch.minimum(
                result[..., joint_index, 1], upper
            )
    if bool(torch.any(result[..., 0] > result[..., 1]).item()):
        raise ValueError("Wuji anatomical limits do not intersect live soft limits.")
    return result


def validate_reference_joint_order(
    reference_joint_names: Sequence[str],
    live_joint_names: Sequence[str],
    *,
    expected_joint_count: int = VEGA_WUJI_JOINT_COUNT,
) -> tuple[str, ...]:
    """Require one exact Reference column for each live actuator, in live order."""

    reference = tuple(str(name) for name in reference_joint_names)
    live = tuple(str(name) for name in live_joint_names)
    if len(live) != expected_joint_count:
        raise ValueError(
            f"Vega-Wuji must expose exactly {expected_joint_count} live joints; "
            f"got {len(live)}."
        )
    if len(set(live)) != len(live):
        raise ValueError("Vega-Wuji live joint names must be unique.")
    if len(reference) != expected_joint_count:
        raise ValueError(
            "The Vega-Wuji Reference must contain exactly "
            f"{expected_joint_count} joint columns; got {len(reference)}."
        )
    if len(set(reference)) != len(reference):
        raise ValueError("Vega-Wuji Reference joint names must be unique.")
    if reference != live:
        raise ValueError(
            "Vega-Wuji Reference joints must be in live actuator order; "
            f"Reference={reference}; live={live}."
        )
    return live


def reference_residual_joint_target(
    raw_action: torch.Tensor,
    previous_residual: torch.Tensor,
    reference_joint_pos: torch.Tensor,
    scale: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    *,
    ema_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the filtered residual and the soft-limit-clipped joint target.

    The result implements ``q_target = clamp(q_ref +
    EMA(scale * tanh(raw_action)), soft_limits)``.  ``ema_factor`` is the
    weight of the previous residual.
    """

    if raw_action.shape != previous_residual.shape:
        raise ValueError("raw_action and previous_residual must have equal shapes.")
    if raw_action.shape != reference_joint_pos.shape:
        raise ValueError("raw_action and reference_joint_pos must have equal shapes.")
    if scale.shape not in (raw_action.shape, raw_action.shape[-1:]):
        raise ValueError("scale must have shape [J] or the full action shape [N, J].")
    if soft_joint_pos_limits.shape != (*raw_action.shape, 2):
        raise ValueError("soft_joint_pos_limits must have shape [N, J, 2].")
    if not 0.0 <= ema_factor <= 1.0:
        raise ValueError("ema_factor must be within [0, 1].")

    scaled_residual = torch.tanh(raw_action) * scale
    filtered_residual = (
        ema_factor * previous_residual + (1.0 - ema_factor) * scaled_residual
    )
    unclipped_target = reference_joint_pos + filtered_residual
    target = torch.maximum(
        torch.minimum(unclipped_target, soft_joint_pos_limits[..., 1]),
        soft_joint_pos_limits[..., 0],
    )
    return filtered_residual, target


def damped_least_squares_delta(
    jacobian: torch.Tensor,
    error: torch.Tensor,
    *,
    damping: float,
    posture_error: torch.Tensor | None = None,
    posture_gain: float = 0.0,
) -> torch.Tensor:
    """Solve a batched damped least-squares task with an optional null target.

    Args:
        jacobian: Task Jacobian with shape ``[N, task_dim, dof]``.
        error: Task-space error with shape ``[N, task_dim]``.
        damping: Positive Levenberg damping value.
        posture_error: Optional joint-space target error with shape ``[N, dof]``.
        posture_gain: Gain for the null-space posture term.
    """

    if jacobian.ndim != 3 or error.shape != jacobian.shape[:2]:
        raise ValueError(
            "jacobian must have shape [N, task_dim, dof] and error must have "
            f"shape [N, task_dim]; got {jacobian.shape} and {error.shape}."
        )
    if damping <= 0.0:
        raise ValueError("damping must be positive.")
    if posture_error is not None and posture_error.shape != (
        jacobian.shape[0],
        jacobian.shape[2],
    ):
        raise ValueError("posture_error must match the Jacobian batch and DoF axes.")

    jt = jacobian.transpose(-1, -2)
    identity_task = torch.eye(
        jacobian.shape[1], dtype=jacobian.dtype, device=jacobian.device
    ).expand(jacobian.shape[0], -1, -1)
    task_matrix = jacobian @ jt + damping**2 * identity_task
    solved_error = torch.linalg.solve(task_matrix, error.unsqueeze(-1))
    pseudo_inverse = jt @ torch.linalg.solve(task_matrix, identity_task)
    delta = (jt @ solved_error).squeeze(-1)

    if posture_error is not None and posture_gain != 0.0:
        identity_joint = torch.eye(
            jacobian.shape[2], dtype=jacobian.dtype, device=jacobian.device
        ).expand(jacobian.shape[0], -1, -1)
        null_projector = identity_joint - pseudo_inverse @ jacobian
        delta = delta + posture_gain * (
            null_projector @ posture_error.unsqueeze(-1)
        ).squeeze(-1)
    return delta


def clip_vector_norm(value: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Limit the last-axis norm without changing vectors below the limit."""

    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp_min(norm, 1.0e-9), max=1.0)
    return value * scale


class VegaWujiReferenceResidualAction(ActionTerm):
    """Apply one bounded Reference-relative position target per live actuator."""

    cfg: VegaWujiReferenceResidualActionCfg

    def __init__(
        self,
        cfg: VegaWujiReferenceResidualActionCfg,
        env: ManagerBasedEnv,
    ) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.command = env.command_manager.get_term(cfg.command_name)
        self._contract_env = env
        self._joint_names = tuple(str(name) for name in self.robot.joint_names)
        if len(self._joint_names) != VEGA_WUJI_JOINT_COUNT:
            raise ValueError(
                f"Vega-Wuji must expose exactly {VEGA_WUJI_JOINT_COUNT} live "
                f"joints; got {len(self._joint_names)}."
            )
        if len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError("Vega-Wuji live joint names must be unique.")

        lift_ids, lift_names = self.robot.find_joints(
            [cfg.lift_joint_name], preserve_order=True
        )
        if len(lift_ids) != 1 or tuple(lift_names) != (cfg.lift_joint_name,):
            raise ValueError(
                f"Vega-Wuji lift joint must resolve once as {cfg.lift_joint_name!r}; "
                f"got {lift_names}."
            )
        self._scale = torch.full(
            (self.action_dim,),
            float(cfg.revolute_joint_scale),
            dtype=self.robot.data.joint_pos.torch.dtype,
            device=self.device,
        )
        self._scale[int(lift_ids[0])] = float(cfg.lift_joint_scale)

        self._side_joint_ids: dict[str, list[int]] = {}
        for side in SIDE_ORDER:
            arm_ids, arm_names = self.robot.find_joints(
                [getattr(cfg, f"{side}_arm_joint_expr")], preserve_order=True
            )
            finger_ids, finger_names = self.robot.find_joints(
                [getattr(cfg, f"{side}_finger_joint_expr")], preserve_order=True
            )
            if len(arm_ids) != 7:
                raise ValueError(
                    f"{side} Vega arm must resolve to 7 joints; got {arm_names}."
                )
            if len(finger_ids) != 20:
                raise ValueError(
                    f"{side} Wuji hand must resolve to 20 joints; got {finger_names}."
                )
            ids = [int(index) for index in arm_ids]
            ids.extend(int(index) for index in finger_ids)
            if len(set(ids)) != 27:
                raise ValueError(f"{side} arm and finger joint sets overlap.")
            self._side_joint_ids[side] = ids

        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            dtype=self.robot.data.joint_pos.torch.dtype,
            device=self.device,
        )
        self._filtered_residual = torch.zeros_like(self._raw_actions)
        self._processed_actions = self.robot.data.joint_pos.torch.clone()
        self._effective_joint_pos_limits = wuji_anatomical_joint_limits(
            self._joint_names, self.robot.data.soft_joint_pos_limits.torch
        )

    @property
    def action_dim(self) -> int:
        return VEGA_WUJI_JOINT_COUNT

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Return all 59 joint-position targets in live actuator order."""

        return self._processed_actions

    def processed_action(self, side: str) -> torch.Tensor:
        """Return 7 arm then 20 finger targets for one side.

        This keeps the old 27-value per-side observation width.  Unlike the
        legacy adapter, the first seven values are arm joint targets, not a
        wrist pose.
        """

        try:
            joint_ids = self._side_joint_ids[side]
        except KeyError as exc:
            raise KeyError(f"Unknown Vega-Wuji hand side: {side!r}.") from exc
        return self._processed_actions[:, joint_ids]

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "VegaWujiReferenceJointResidual"
        self._IO_descriptor.scale = self._scale
        return self._IO_descriptor

    def _validate_reference_contract(self) -> None:
        reference_joint_names = getattr(
            self._contract_env, "_reference_joint_names", ()
        )
        validate_reference_joint_order(
            reference_joint_names,
            self._joint_names,
        )
        if self.command.reference_joint_pos.shape != self._raw_actions.shape:
            raise ValueError(
                "Vega-Wuji Reference joint positions must have shape "
                f"{tuple(self._raw_actions.shape)}; got "
                f"{tuple(self.command.reference_joint_pos.shape)}."
            )

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                "Vega-Wuji joint residual action must have shape "
                f"{tuple(self._raw_actions.shape)}; got {tuple(actions.shape)}."
            )
        self._validate_reference_contract()
        self._raw_actions[:] = actions
        filtered_residual, target = reference_residual_joint_target(
            actions,
            self._filtered_residual,
            self.command.reference_joint_pos,
            self._scale,
            self._effective_joint_pos_limits,
            ema_factor=float(self.cfg.ema_factor),
        )
        self._filtered_residual[:] = filtered_residual
        self._processed_actions[:] = target

    def apply_actions(self) -> None:
        self.robot.set_joint_position_target_index(target=self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._filtered_residual[env_ids] = 0.0
        self._processed_actions[env_ids] = self.robot.data.joint_pos.torch[env_ids]


class VegaWujiWristPoseResidualAction(ActionTerm):
    """Track two wrist poses and forty finger joints with one articulation."""

    cfg: VegaWujiWristPoseResidualActionCfg

    def __init__(
        self, cfg: VegaWujiWristPoseResidualActionCfg, env: ManagerBasedEnv
    ) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.command = env.command_manager.get_term(cfg.command_name)

        self._arm_ids: dict[str, list[int]] = {}
        self._finger_ids: dict[str, list[int]] = {}
        self._wrist_body_ids: dict[str, int] = {}
        for side in SIDE_ORDER:
            arm_expr = getattr(cfg, f"{side}_arm_joint_expr")
            finger_expr = getattr(cfg, f"{side}_finger_joint_expr")
            wrist_name = getattr(cfg, f"{side}_wrist_body_name")
            arm_ids, arm_names = self.robot.find_joints([arm_expr], preserve_order=True)
            finger_ids, finger_names = self.robot.find_joints(
                [finger_expr], preserve_order=True
            )
            body_ids, body_names = self.robot.find_bodies(
                [wrist_name], preserve_order=True
            )
            if len(arm_ids) != 7:
                raise ValueError(
                    f"{side} Vega arm must resolve to 7 joints; got {arm_names}."
                )
            if len(finger_ids) != 20:
                raise ValueError(
                    f"{side} Wuji hand must resolve to 20 joints; got {finger_names}."
                )
            if len(body_ids) != 1:
                raise ValueError(
                    f"{side} wrist body must resolve once; got {body_names}."
                )
            self._arm_ids[side] = list(arm_ids)
            self._finger_ids[side] = list(finger_ids)
            self._wrist_body_ids[side] = int(body_ids[0])

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._filtered_residual = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros(self.num_envs, 54, device=self.device)
        self._previous_residual = torch.zeros_like(self._raw_actions)
        self._joint_targets = self.robot.data.default_joint_pos.torch.clone()
        self._effective_joint_pos_limits = wuji_anatomical_joint_limits(
            self.robot.joint_names, self.robot.data.soft_joint_pos_limits.torch
        )
        self._wrist_target_pose: dict[str, torch.Tensor] = {
            side: torch.zeros(self.num_envs, 7, device=self.device)
            for side in SIDE_ORDER
        }
        for pose in self._wrist_target_pose.values():
            pose[:, 6] = 1.0  # Isaac Lab 3.0 uses xyzw.
        for side_index, _side in enumerate(SIDE_ORDER):
            self._processed_actions[:, side_index * 27 + 3] = 1.0

        scale = torch.empty_like(self._raw_actions)
        clip = torch.empty_like(self._raw_actions)
        for side_index, _side in enumerate(SIDE_ORDER):
            start = side_index * 26
            scale[:, start : start + 3] = cfg.wrist_position_scale
            scale[:, start + 3 : start + 6] = cfg.wrist_orientation_scale
            scale[:, start + 6 : start + 26] = cfg.finger_joint_scale
            clip[:, start : start + 3] = cfg.wrist_position_clip
            clip[:, start + 3 : start + 6] = cfg.wrist_orientation_clip
            clip[:, start + 6 : start + 26] = cfg.finger_joint_clip
        self._scale = scale
        self._clip = clip

    @property
    def action_dim(self) -> int:
        return 52

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Return right-then-left target poses and finger positions in WXYZ."""

        return self._processed_actions

    def processed_action(self, side: str) -> torch.Tensor:
        """Return one 27-value wrist-pose and finger target."""

        try:
            side_index = SIDE_ORDER.index(side)
        except ValueError as exc:
            raise KeyError(f"Unknown Vega-Wuji hand side: {side!r}.") from exc
        start = side_index * 27
        return self._processed_actions[:, start : start + 27]

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "VegaWujiWristPoseResidual"
        self._IO_descriptor.scale = self._scale[0]
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Vega-Wuji wrist-pose action must have shape {self._raw_actions.shape}; "
                f"got {actions.shape}."
            )
        self._raw_actions[:] = actions
        # Keep the released EMA convention: ema * previous +
        # (1 - ema) * new residual.
        residual = (
            self.cfg.ema_factor * self._previous_residual
            + (1.0 - self.cfg.ema_factor) * actions * self._scale
        )
        residual = torch.clamp(residual, min=-self._clip, max=self._clip)
        self._previous_residual[:] = residual
        self._filtered_residual[:] = residual

        for side_index, side in enumerate(SIDE_ORDER):
            start = side_index * 26
            processed_start = side_index * 27
            reference_pose = self.command.wrist_pose_command(side)
            target = self._wrist_target_pose[side]
            target[:, :3] = reference_pose[:, :3] + residual[:, start : start + 3]
            rotation_residual = quat_from_euler_xyz(
                residual[:, start + 3],
                residual[:, start + 4],
                residual[:, start + 5],
            )
            target[:, 3:7] = quat_mul(reference_pose[:, 3:7], rotation_residual)
            self._processed_actions[:, processed_start : processed_start + 3] = target[
                :, :3
            ]
            self._processed_actions[:, processed_start + 3 : processed_start + 7] = (
                torch.cat((target[:, 6:7], target[:, 3:6]), dim=-1)
            )
            self._processed_actions[:, processed_start + 7 : processed_start + 27] = (
                self.command.finger_joint_command(side)
                + residual[:, start + 6 : start + 26]
            )

    def _arm_target(self, side: str) -> torch.Tensor:
        body_id = self._wrist_body_ids[side]
        jacobian_body_id = body_id - 1 if self.robot.is_fixed_base else body_id
        all_jacobians = self.robot.data.body_link_jacobian_w.torch
        arm_ids = self._arm_ids[side]
        jacobian = all_jacobians[:, jacobian_body_id, :, arm_ids]

        current_position = (
            self.robot.data.body_link_pos_w.torch[:, body_id]
            - self._env.scene.env_origins
        )
        current_quat = self.robot.data.body_link_quat_w.torch[:, body_id]
        target = self._wrist_target_pose[side]
        position_error = target[:, :3] - current_position
        orientation_error = axis_angle_from_quat(
            quat_mul(target[:, 3:7], quat_inv(current_quat))
        )
        task_error = torch.cat((position_error, orientation_error), dim=-1)

        current_joint = self.robot.data.joint_pos.torch[:, arm_ids]
        reference_joint = self.command.reference_joint_pos[:, arm_ids]
        delta = damped_least_squares_delta(
            jacobian,
            task_error,
            damping=float(self.cfg.dls_damping),
            posture_error=reference_joint - current_joint,
            posture_gain=float(self.cfg.null_posture_gain),
        )
        delta = clip_vector_norm(delta, float(self.cfg.max_arm_joint_step))
        return current_joint + delta

    def apply_actions(self) -> None:
        self._joint_targets[:] = self.command.reference_joint_pos
        for side_index, side in enumerate(SIDE_ORDER):
            arm_ids = self._arm_ids[side]
            finger_ids = self._finger_ids[side]
            processed_start = side_index * 27
            self._joint_targets[:, arm_ids] = self._arm_target(side)
            self._joint_targets[:, finger_ids] = self._processed_actions[
                :, processed_start + 7 : processed_start + 27
            ]

        limits = self._effective_joint_pos_limits
        self._joint_targets[:] = torch.maximum(
            torch.minimum(self._joint_targets, limits[..., 1]), limits[..., 0]
        )
        self.robot.set_joint_position_target_index(target=self._joint_targets)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        for side_index, _side in enumerate(SIDE_ORDER):
            self._processed_actions[env_ids, side_index * 27 + 3] = 1.0
        self._filtered_residual[env_ids] = 0.0
        self._previous_residual[env_ids] = 0.0
        self._joint_targets[env_ids] = self.robot.data.joint_pos.torch[env_ids]


@configclass
class VegaWujiReferenceResidualActionCfg(ActionTermCfg):
    """Configuration for the 59-value actuator-order Reference residual."""

    class_type: type[ActionTerm] = VegaWujiReferenceResidualAction
    asset_name: str = "robot"
    command_name: str = "motion"
    lift_joint_name: str = "Lift"
    right_arm_joint_expr: str = "R_arm_j[1-7]"
    left_arm_joint_expr: str = "L_arm_j[1-7]"
    right_finger_joint_expr: str = "r_.*"
    left_finger_joint_expr: str = "l_.*"
    lift_joint_scale: float = 0.05
    revolute_joint_scale: float = 0.15
    ema_factor: float = 0.3


@configclass
class VegaWujiWristPoseResidualActionCfg(ActionTermCfg):
    """Configuration for the 52-value wrist-pose residual action."""

    class_type: type[ActionTerm] = VegaWujiWristPoseResidualAction
    asset_name: str = "robot"
    command_name: str = "motion"
    # The vendored right_palm and left_palm sites are identity frames on these
    # bodies. The task validates that MJCF invariant before Isaac starts.
    right_wrist_body_name: str = "r_mount"
    left_wrist_body_name: str = "l_mount"
    right_arm_joint_expr: str = "R_arm_j[1-7]"
    left_arm_joint_expr: str = "L_arm_j[1-7]"
    right_finger_joint_expr: str = "r_.*"
    left_finger_joint_expr: str = "l_.*"
    wrist_position_scale: float = 0.05
    wrist_orientation_scale: float = 0.15
    finger_joint_scale: float = 0.15
    wrist_position_clip: float = 0.2
    wrist_orientation_clip: float = 1.0
    finger_joint_clip: float = 1.0
    ema_factor: float = 0.3
    dls_damping: float = 0.05
    null_posture_gain: float = 0.1
    max_arm_joint_step: float = 0.12


__all__ = [
    "SIDE_ORDER",
    "VEGA_WUJI_JOINT_COUNT",
    "WUJI_ANATOMICAL_LIMITS_DEG",
    "VegaWujiWristPoseResidualAction",
    "VegaWujiWristPoseResidualActionCfg",
    "VegaWujiReferenceResidualAction",
    "VegaWujiReferenceResidualActionCfg",
    "clip_vector_norm",
    "damped_least_squares_delta",
    "reference_residual_joint_target",
    "validate_reference_joint_order",
    "wuji_anatomical_joint_limits",
]
