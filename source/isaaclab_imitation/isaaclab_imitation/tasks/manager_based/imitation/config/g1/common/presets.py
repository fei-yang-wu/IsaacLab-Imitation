# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-backend, robot, and contact-sensor presets for the G1 tasks.

Each preset resolves per launch-time ``physics=...`` selection (default is
PhysX); see Isaac Lab's ``PresetCfg``.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from isaaclab_imitation.assets.robots.unitree import (
    UNITREE_G1_29DOF_MIMIC_CFG,
    UNITREE_G1_29DOF_SONIC_CFG,
    unitree_g1_29dof_usd_articulation_cfg,
)


@configclass
class G1ImitationPhysicsCfg(PresetCfg):
    """Physics backend presets; select at launch with ``physics=physx`` or
    ``physics=newton_mjwarp`` (default is PhysX).

    Newton solver values mirror IsaacLab's official G1 flat-locomotion preset.
    """

    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=95,
            nconmax=10,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        num_substeps=1,
        debug_mode=False,
    )


@configclass
class G1ImitationRobotCfg(PresetCfg):
    """One preconverted G1 USD contract shared by both physics backends."""

    default = unitree_g1_29dof_usd_articulation_cfg(UNITREE_G1_29DOF_MIMIC_CFG)
    physx = unitree_g1_29dof_usd_articulation_cfg(UNITREE_G1_29DOF_MIMIC_CFG)
    newton_mjwarp = unitree_g1_29dof_usd_articulation_cfg(UNITREE_G1_29DOF_MIMIC_CFG)


@configclass
class G1SonicRobotCfg(PresetCfg):
    """SONIC actuators on one preconverted G1 USD for both backends."""

    default = unitree_g1_29dof_usd_articulation_cfg(UNITREE_G1_29DOF_SONIC_CFG)
    physx = unitree_g1_29dof_usd_articulation_cfg(UNITREE_G1_29DOF_SONIC_CFG)
    newton_mjwarp = unitree_g1_29dof_usd_articulation_cfg(UNITREE_G1_29DOF_SONIC_CFG)


@configclass
class G1ImitationContactSensorCfg(PresetCfg):
    """Contact sensor presets matching the active physics backend."""

    default = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    physx = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    newton_mjwarp = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


def _set_contact_sensor_update_period(contact_cfg, update_period: float) -> None:
    """Set update_period on a contact sensor cfg or on every preset variant."""
    if isinstance(contact_cfg, G1ImitationContactSensorCfg):
        for variant in (
            contact_cfg.default,
            contact_cfg.physx,
            contact_cfg.newton_mjwarp,
        ):
            variant.update_period = update_period
    else:
        contact_cfg.update_period = update_period
