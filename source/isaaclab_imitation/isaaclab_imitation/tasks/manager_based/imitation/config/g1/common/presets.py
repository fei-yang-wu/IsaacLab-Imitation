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
    # Aligned to stock CPU MuJoCo, 2026-08-03. Newton MJWarp *is* MuJoCo Warp,
    # so it can be made to agree with stock MuJoCo -- and in free flight it
    # already does, to <= 1.7e-6 rad over 13 control steps. Add ground contact
    # and it leaves both MuJoCo and PhysX, so the remaining differences are all
    # contact-side.
    #
    # A field-by-field diff of `mjw_model.opt` against a CPU MuJoCo model
    # (`scripts/audit/dump_mjwarp_model_contract.py` prints the whole option
    # table) found the solver core already identical -- cone, integrator, solver,
    # iterations, ls_iterations, impratio, gravity, timestep -- and exactly three
    # differences, closed here:
    #
    # 1. `use_mujoco_contacts`: False routed Newton's own collision pipeline into
    #    MJWarp instead of letting MuJoCo generate the contacts. Verified to
    #    reach the solver -- it flips `mjw_model.opt.run_collision_detection`.
    # 2. `nconmax=10` (which MJWarp raises to 18) drops contacts on a 30-body
    #    self-colliding humanoid; it overflowed in measured rollouts.
    # 3. `tolerance` defaulted to 1e-6 against stock MuJoCo's 1e-8.
    #
    # One residual is NOT reachable from config: MJWarp sets `disableflags` bit
    # 19, `mjDSBL_MULTICCD`, which stock MuJoCo leaves clear, and
    # `MJWarpSolverCfg` exposes no `disableflags` field. Fewer contact points per
    # convex pair is invisible in flight and decisive on the floor, so this is
    # "as close as the config allows", not parity.
    #
    # Do not add a *new* physics preset name for variants of this. Preset
    # alternatives are matched by name across groups, and
    # `G1ImitationContactSensorCfg` has no alternative for an unknown name, so it
    # silently falls back to `default` -- a PhysX contact sensor on a Newton
    # backend, which dies in `create_rigid_body_view` on a None handle.
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=288,
            nconmax=200,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            use_mujoco_contacts=True,
            tolerance=1.0e-8,
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
