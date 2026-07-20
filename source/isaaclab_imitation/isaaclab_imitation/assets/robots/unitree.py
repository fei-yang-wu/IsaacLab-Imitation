# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree robot configurations vendored for IsaacLab-Imitation.

The G1 URDF and referenced meshes are packaged under
``isaaclab_imitation/assets/unitree/g1_description`` so G1 imitation tasks do not
need ``unitree_rl_lab`` at import or runtime.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.schemas import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg, UsdFileCfg
from isaaclab.utils.configclass import configclass

from .unitree_joint_order import (
    UNITREE_G1_29DOF_JOINT_ORDER_SOURCE,  # noqa: F401
    UNITREE_G1_29DOF_SDK_JOINT_NAMES,
    UNITREE_G1_29DOF_URDF_FILE,
    UNITREE_G1_29DOF_URDF_REVOLUTE_JOINT_NAMES,  # noqa: F401
    UNITREE_G1_29DOF_USD_FILE,
    UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES,  # noqa: F401
    UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES,  # noqa: F401
)


UNITREE_G1_29DOF_URDF_PATH = str(UNITREE_G1_29DOF_URDF_FILE)
UNITREE_G1_29DOF_USD_PATH = str(UNITREE_G1_29DOF_USD_FILE)


def resolve_unitree_g1_29dof_usd_path() -> str:
    """Resolve the G1 USD used for simulation.

    The repo packages the official Unitree ``g1_29dof_rev_1_0`` USD tree
    (git-lfs, from ``unitreerobotics/unitree_model`` commit ``b6a8942b``);
    it is the only supported training asset. The matched 2026-07-20 50M
    comparisons retired the alternatives: the official asset reached ep_len
    63.5 versus 58.9 for Isaac Sim's bundled mesh-less test fixture and 8.0
    for the old modified URDF-conversion tree.
    ``ISAACLAB_IMITATION_UNITREE_USD_PATH`` remains as an explicit escape
    hatch for asset experiments only.
    """
    explicit_path = os.environ.get("ISAACLAB_IMITATION_UNITREE_USD_PATH")
    if explicit_path:
        return str(Path(explicit_path).expanduser())
    if not UNITREE_G1_29DOF_USD_FILE.is_file():
        raise FileNotFoundError(
            "The packaged official Unitree G1 USD is missing: "
            f"{UNITREE_G1_29DOF_USD_FILE}. Run `git lfs pull` to materialize "
            "the asset files."
        )
    return UNITREE_G1_29DOF_USD_PATH


def unitree_g1_29dof_usd_articulation_cfg(
    cfg: UnitreeArticulationCfg,
) -> UnitreeArticulationCfg:
    """Clone a G1 articulation spawning the packaged official USD."""
    urdf_spawn = cfg.spawn
    return cfg.replace(
        spawn=UsdFileCfg(
            usd_path=resolve_unitree_g1_29dof_usd_path(),
            activate_contact_sensors=urdf_spawn.activate_contact_sensors,
            articulation_props=urdf_spawn.articulation_props,
            rigid_props=urdf_spawn.rigid_props,
        )
    )


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] | None = None
    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUrdfFileCfg(UrdfFileCfg):
    """Common URDF import settings for Unitree robots."""

    usd_dir: str | None = None
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(asset_path=UNITREE_G1_29DOF_URDF_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=list(UNITREE_G1_29DOF_SDK_JOINT_NAMES),
)


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

UNITREE_G1_29DOF_MIMIC_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(asset_path=UNITREE_G1_29DOF_URDF_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
    joint_sdk_names=UNITREE_G1_29DOF_CFG.joint_sdk_names.copy(),
)


def _action_scale_from_actuators(cfg: ArticulationCfg) -> dict[str, float]:
    action_scale: dict[str, float] = {}
    for actuator in cfg.actuators.values():
        effort_limit = actuator.effort_limit_sim
        stiffness = actuator.stiffness
        names = actuator.joint_names_expr
        if not isinstance(effort_limit, dict):
            effort_limit = {name: effort_limit for name in names}
        if not isinstance(stiffness, dict):
            stiffness = {name: stiffness for name in names}
        for name in names:
            if name in effort_limit and name in stiffness and stiffness[name]:
                action_scale[name] = 0.25 * effort_limit[name] / stiffness[name]
    return action_scale


UNITREE_G1_29DOF_MIMIC_ACTION_SCALE = _action_scale_from_actuators(
    UNITREE_G1_29DOF_MIMIC_CFG
)

# SONIC's released G1 controller uses the larger 7520-22 actuator for hip
# pitch. Keep this opt-in so existing IsaacLab-Imitation checkpoints retain
# their original actuator contract.
UNITREE_G1_29DOF_SONIC_CFG = copy.deepcopy(UNITREE_G1_29DOF_MIMIC_CFG)
_sonic_leg_actuator = UNITREE_G1_29DOF_SONIC_CFG.actuators["legs"]
_sonic_leg_actuator.effort_limit_sim[".*_hip_pitch_joint"] = 139.0
_sonic_leg_actuator.velocity_limit_sim[".*_hip_pitch_joint"] = 20.0
_sonic_leg_actuator.stiffness[".*_hip_pitch_joint"] = STIFFNESS_7520_22
_sonic_leg_actuator.damping[".*_hip_pitch_joint"] = DAMPING_7520_22
_sonic_leg_actuator.armature[".*_hip_pitch_joint"] = ARMATURE_7520_22
UNITREE_G1_29DOF_SONIC_ACTION_SCALE = _action_scale_from_actuators(
    UNITREE_G1_29DOF_SONIC_CFG
)
