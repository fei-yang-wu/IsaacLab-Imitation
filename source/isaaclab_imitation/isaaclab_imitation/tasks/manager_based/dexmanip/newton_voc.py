# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Newton-compatible virtual rigid-object controller.

The action term applies a pose-PD wrench through Isaac Lab's common
``RigidObject`` data and wrench-composer APIs. It does not access a PhysX view
or any backend-private data. All live and command quaternions use XYZW order.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from .voc_math import compute_pose_pd_wrench_xyzw

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


logger = logging.getLogger(__name__)


def _torch(value: Any) -> torch.Tensor:
    """Return a Torch view from a common Isaac Lab tensor or ProxyArray."""

    tensor = getattr(value, "torch", value)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected a Torch-compatible Isaac Lab tensor, got {type(value)!r}."
        )
    return tensor


def _gravity_magnitude(env: ManagerBasedEnv, override: float | None) -> float | None:
    """Read the public simulation gravity magnitude, or use an override."""

    if override is not None:
        if not math.isfinite(override) or override < 0.0:
            raise ValueError("gravity_magnitude must be finite and non-negative.")
        return float(override)

    sim_cfg = getattr(getattr(env, "cfg", None), "sim", None)
    gravity = getattr(sim_cfg, "gravity", None)
    if gravity is None:
        return None
    gravity_tensor = torch.as_tensor(gravity, dtype=torch.float64)
    if gravity_tensor.shape != (3,) or not torch.isfinite(gravity_tensor).all():
        return None
    return float(torch.linalg.vector_norm(gravity_tensor).item())


class NewtonVirtualRigidObjectControl(ActionTerm):
    """Track one rigid-object reference pose with a virtual body-frame wrench."""

    cfg: NewtonVirtualRigidObjectControlCfg

    def __init__(
        self, cfg: NewtonVirtualRigidObjectControlCfg, env: ManagerBasedEnv
    ) -> None:
        super().__init__(cfg, env)
        self.object: RigidObject = env.scene[cfg.asset_name]
        if self.object.num_instances != self.num_envs:
            raise ValueError(
                f"Rigid object {cfg.asset_name!r} has {self.object.num_instances} "
                f"instances, expected {self.num_envs}."
            )
        if self.object.num_bodies != 1:
            raise ValueError(
                f"Virtual rigid-object control needs one body; {cfg.asset_name!r} "
                f"has {self.object.num_bodies}."
            )

        self.command = env.command_manager.get_term(cfg.command_name)
        object_names = getattr(self.command, "object_names", None)
        if object_names is not None and cfg.asset_name not in object_names:
            raise ValueError(
                f"Command term {cfg.command_name!r} does not contain object "
                f"{cfg.asset_name!r}; available objects are {tuple(object_names)}."
            )
        if not callable(getattr(self.command, "object_target_pose", None)):
            raise TypeError(
                f"Command term {cfg.command_name!r} must define "
                "object_target_pose(asset_name)."
            )
        if not hasattr(self.command, "virtual_object_controller_scale"):
            raise TypeError(
                f"Command term {cfg.command_name!r} must define the per-environment "
                "virtual_object_controller_scale tensor."
            )

        self._env_origins = _torch(env.scene.env_origins)
        self._gravity_magnitude = (
            _gravity_magnitude(env, cfg.gravity_magnitude)
            if cfg.compensate_gravity
            else None
        )
        if cfg.compensate_gravity and self._gravity_magnitude is None:
            logger.warning(
                "Gravity compensation is disabled for %s because the simulation "
                "gravity vector is not available.",
                cfg.asset_name,
            )

        self._raw_actions = torch.zeros(
            self.num_envs, 0, device=self.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._wrench_b = torch.zeros(
            self.num_envs, 6, device=self.device, dtype=torch.float32
        )

    @property
    def action_dim(self) -> int:
        """The object controller does not consume policy actions."""

        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def wrench_b(self) -> torch.Tensor:
        """Last body-frame force and torque, for diagnostics."""

        return self._wrench_b

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (0,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "NewtonVirtualRigidObjectControl"
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        """Validate the empty action slice allocated by the action manager."""

        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Virtual object action must have shape {tuple(self._raw_actions.shape)}; "
                f"got {tuple(actions.shape)}."
            )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear diagnostic and pending controller wrenches."""

        selected: Sequence[int] | torch.Tensor | slice
        selected = slice(None) if env_ids is None else env_ids
        self._wrench_b[selected] = 0.0
        self.object.instantaneous_wrench_composer.reset(env_ids=selected)

    def _gravity_inputs(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None]:
        if self._gravity_magnitude is None:
            return None, None, None
        data = self.object.data
        try:
            mass = _torch(data.body_mass)
            projected_gravity = _torch(data.projected_gravity_b)
        except (AttributeError, TypeError):
            logger.warning(
                "Gravity compensation is disabled for %s because common rigid-object "
                "mass or projected-gravity data is not available.",
                self.cfg.asset_name,
            )
            self._gravity_magnitude = None
            return None, None, None
        return mass, projected_gravity, self._gravity_magnitude

    def apply_actions(self) -> None:
        """Compute and apply the current virtual object-control wrench."""

        current_pose_w = _torch(self.object.data.root_link_pose_w)
        current_com_velocity_w = _torch(self.object.data.root_com_vel_w)
        target_pose_e = self.command.object_target_pose(self.cfg.asset_name)
        if not isinstance(target_pose_e, torch.Tensor):
            raise TypeError("object_target_pose must return a Torch tensor.")
        if target_pose_e.shape != (self.num_envs, 7):
            raise ValueError(
                "object_target_pose must return environment-local XYZ+XYZW with "
                f"shape {(self.num_envs, 7)}; got {tuple(target_pose_e.shape)}."
            )
        target_position_w = target_pose_e[:, :3] + self._env_origins

        mass, projected_gravity, gravity_magnitude = self._gravity_inputs()
        force_b, torque_b = compute_pose_pd_wrench_xyzw(
            current_position_w=current_pose_w[:, :3],
            current_orientation_xyzw_w=current_pose_w[:, 3:7],
            current_com_velocity_w=current_com_velocity_w,
            target_position_w=target_position_w,
            target_orientation_xyzw_w=target_pose_e[:, 3:7],
            controller_scale=self.command.virtual_object_controller_scale,
            linear_stiffness=float(self.cfg.linear_stiffness),
            linear_damping=float(self.cfg.linear_damping),
            angular_stiffness=float(self.cfg.angular_stiffness),
            angular_damping=float(self.cfg.angular_damping),
            max_force=float(self.cfg.max_force),
            max_torque=float(self.cfg.max_torque),
            body_mass=mass,
            projected_gravity_b=projected_gravity,
            gravity_magnitude=gravity_magnitude,
        )
        self._wrench_b[:, :3] = force_b
        self._wrench_b[:, 3:] = torque_b

        # The instantaneous composer is a public RigidObject API. Isaac Lab
        # writes it before each Newton physics step and then clears it.
        self.object.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=force_b.unsqueeze(1).contiguous(),
            torques=torque_b.unsqueeze(1).contiguous(),
            is_global=False,
        )


@configclass
class NewtonVirtualRigidObjectControlCfg(ActionTermCfg):
    """Configuration for one Newton virtual rigid-object controller."""

    class_type: type[ActionTerm] = NewtonVirtualRigidObjectControl
    asset_name: str = MISSING
    command_name: str = "motion"
    linear_stiffness: float = 50.0
    linear_damping: float = 10.0
    # The angular gains act on an object inertia three orders of magnitude
    # below its mass, so the rotational loop reaches its discrete stability
    # limit long before the linear one. A tracked scene object of about
    # 0.00125 kg m^2 gives k dt^2 / I = 20 at the 20 Hz control rate, well past
    # the limit of 4: the object tumbles, and because the controller wrench is
    # body-frame, a tumbling object misdirects its own gravity compensation and
    # is then flung. Measured on the corn-can Reference, 10.0/0.1 holds an
    # episode for 3.82 steps and 1.0/0.05 holds it for 6.09.
    # The angular gains act on an object inertia three orders of magnitude
    # below its mass, so the rotational loop reaches its discrete stability
    # limit long before the linear one. A tracked scene object of about
    # 0.00125 kg m^2 gives k dt^2 / I = 20 at the 20 Hz control rate, well past
    # the limit of 4: the object tumbles, and because the controller wrench is
    # body-frame, a tumbling object misdirects its own gravity compensation and
    # is then flung out of the scene. Measured on the corn-can Reference,
    # 10.0/0.1 holds an episode for 3.75 steps and 1.0/0.05 holds it for 5.99.
    angular_stiffness: float = 1.0
    angular_damping: float = 0.05
    max_force: float = 60.0
    max_torque: float = 60.0
    compensate_gravity: bool = True
    gravity_magnitude: float | None = None


__all__ = [
    "NewtonVirtualRigidObjectControl",
    "NewtonVirtualRigidObjectControlCfg",
]
