# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
from collections.abc import Mapping
from pathlib import Path

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_imitation.assets.robots.unitree import (
    UNITREE_G1_29DOF_SDK_JOINT_NAMES,
    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    UNITREE_G1_29DOF_MIMIC_CFG,
    UNITREE_G1_29DOF_SONIC_ACTION_SCALE,
    UNITREE_G1_29DOF_SONIC_CFG,
    unitree_g1_29dof_usd_articulation_cfg,
)

from ... import mdp
from ...imitation_env_cfg import ImitationLearningEnvCfg
from ...lafan1_manifest import (
    build_lafan1_loader_kwargs,
    dataset_path_from_entries,
    infer_npz_manifest_control_freq,
    load_lafan1_manifest,
    load_lafan1_manifest_loader_options,
)


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

G1_29DOF_JOINT_NAMES: list[str] = list(UNITREE_G1_29DOF_SDK_JOINT_NAMES)

# IsaacLab G1 articulation (USD) joint order, i.e. the order of
# ``robot.joint_names`` / ``robot.data.joint_pos.torch`` at runtime. This is a
# breadth-first (level-order) traversal and is NOT the Unitree SDK/URDF order.
# The env applies the reference directly to the articulation, so this is the
# ground-truth ``target_joint_names``. Verified against a live articulation via
# ``robot.joint_names``; guarded at runtime in the env.
G1_29DOF_ISAACLAB_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Canonical body order of the recorded NPZ reference datasets. The NPZ body
# arrays carry no body-name metadata; they were recorded from the PhysX
# articulation, whose breadth-first (level-order) body enumeration is captured
# here. Do NOT derive this from the live robot at runtime: the Newton backend
# enumerates bodies depth-first per limb, which silently permutes the mapping.
G1_29DOF_DATASET_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# Body tracking set aligned with the original Unitree G1 mimic tracking config.
G1_TRACKED_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_EE_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

G1_OBS_ANCHOR_BODY_NAME = "torso_link"


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


def _g1_tracked_body_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        body_names=G1_TRACKED_BODY_NAMES,
        preserve_order=True,
    )


def _g1_tracked_body_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": _g1_tracked_body_asset_cfg(),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
    }


def _g1_expert_motion_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_JOINT_NAMES,
        )
    }


def _g1_expert_anchor_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
    }


def _g1_expert_window_motion_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_JOINT_NAMES,
        ),
        "past_steps": 0,
        "future_steps": 0,
    }


def _g1_expert_window_anchor_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
        "past_steps": 0,
        "future_steps": 0,
    }


def _g1_expert_window_ee_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "reference_body_names": tuple(G1_EE_BODY_NAMES),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
        "past_steps": 0,
        "future_steps": 0,
    }


def _g1_canonical_joint_obs_params() -> dict[str, object]:
    """Return the backend-independent policy joint ordering."""
    return {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
            preserve_order=True,
        )
    }


@configclass
class G1ActionsCfg:
    """Action settings for 29-DoF mimic G1."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
        preserve_order=True,
        scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class G1SonicActionsCfg(G1ActionsCfg):
    """Action scale induced by SONIC's released actuator configuration."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
        preserve_order=True,
        scale=UNITREE_G1_29DOF_SONIC_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class G1ObservationCfg:
    """Observation settings aligned with the 29-DoF tracking environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        expert_motion = ObsTerm(
            func=mdp.policy_expert_motion_command,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.policy_expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.policy_expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params=_g1_canonical_joint_obs_params(),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params=_g1_canonical_joint_obs_params(),
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        expert_motion = ObsTerm(
            func=mdp.expert_motion_command,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params=_g1_tracked_body_obs_params(),
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params=_g1_tracked_body_obs_params(),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, params=_g1_canonical_joint_obs_params()
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, params=_g1_canonical_joint_obs_params()
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class ExpertStateCfg(ObsGroup):
        """Single-frame expert observations exposed through the observation manager."""

        joint_pos = ObsTerm(
            func=mdp.expert_joint_pos,
            params=_g1_expert_motion_obs_params(),
        )
        joint_vel = ObsTerm(
            func=mdp.expert_joint_vel,
            params=_g1_expert_motion_obs_params(),
        )
        root_pos = ObsTerm(func=mdp.expert_root_pos)
        root_quat = ObsTerm(func=mdp.expert_root_quat)
        root_lin_vel = ObsTerm(func=mdp.expert_root_lin_vel)
        root_ang_vel = ObsTerm(func=mdp.expert_root_ang_vel)
        expert_motion = ObsTerm(
            func=mdp.expert_motion_command,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class ExpertWindowCfg(ObsGroup):
        """Temporal expert observations exposed through the observation manager."""

        expert_motion = ObsTerm(
            func=mdp.expert_window_motion,
            params=_g1_expert_window_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_window_anchor_pos_b,
            params=_g1_expert_window_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_window_anchor_ori_b,
            params=_g1_expert_window_anchor_obs_params(),
        )
        expert_ee_pos_b = ObsTerm(
            func=mdp.expert_window_ee_pos_b,
            params=_g1_expert_window_ee_obs_params(),
        )
        expert_ee_ori_b = ObsTerm(
            func=mdp.expert_window_ee_ori_b,
            params=_g1_expert_window_ee_obs_params(),
        )

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class RewardInputCfg(ObsGroup):
        """Inputs consumed by discriminator / reward estimator networks.

        On rollout, terms are computed from the robot's actual state; on the
        expert minibatch the env's expert-observation mapper returns the
        idealized-expert counterpart (reference motion, zero anchor error).
        """

        expert_motion = ObsTerm(
            func=mdp.robot_motion,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class PolicySupervisionCfg(ObsGroup):
        """Training-only labels excluded from every actor and planner input."""

        expert_action = ObsTerm(func=mdp.reconstructed_reference_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    expert_state: ExpertStateCfg = ExpertStateCfg()
    expert_window: ExpertWindowCfg = ExpertWindowCfg()
    reward_input: RewardInputCfg = RewardInputCfg()
    policy_supervision: PolicySupervisionCfg = PolicySupervisionCfg()


@configclass
class G1EventCfg:
    """Event settings aligned with the 29-DoF tracking environment."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    reset_reference_state = EventTerm(
        func=mdp.reset_root_and_joints_to_reference_with_randomization,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": VELOCITY_RANGE,
            "joint_position_range": (-0.1, 0.1),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class G1SonicEventCfg(G1EventCfg):
    """Domain randomization used by the public SONIC release recipe."""

    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=r".*wrist_yaw.*|torso_link"
            ),
            "mass_distribution_params": (0.8, 2.5),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class G1RewardsCfg:
    """Reward terms aligned to the 29-DoF tracking environment."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.reference_global_anchor_position_error_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "std": 0.3,
        },
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.reference_global_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "std": 0.4,
        },
    )
    motion_body_pos = RewTerm(
        func=mdp.reference_relative_body_position_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.3,
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.reference_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.4,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.reference_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 1.0,
        },
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.reference_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 3.14,
        },
    )

    # -- metrics (weight=0.0: logged to Episode_Reward/mpjpe_m each episode,
    # averaged across envs, but does not affect the return)
    mpjpe_m = RewTerm(
        func=mdp.mpjpe_relative_body_pos_m,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    (
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    )
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class G1SonicRewardsCfg(G1RewardsCfg):
    """Additional reward terms and contact exclusions from SONIC release."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.reference_global_anchor_position_error_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "pelvis",
            "std": 0.3,
        },
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.reference_global_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "pelvis",
            "std": 0.4,
        },
    )
    motion_body_pos = RewTerm(
        func=mdp.reference_relative_body_position_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_TRACKED_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "pelvis",
            "std": 0.3,
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.reference_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_TRACKED_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "pelvis",
            "std": 0.4,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    (
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                        r"(?!left_elbow_link$)(?!right_elbow_link$).+$"
                    )
                ],
            ),
            "threshold": 1.0,
        },
    )
    tracking_reward_points = RewTerm(
        func=mdp.reference_local_reward_point_position_error_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "torso_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                preserve_order=True,
            ),
            "reference_body_names": [
                "torso_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "body_offsets": ((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "anchor_body_name": "pelvis",
            "std": 0.1,
        },
    )
    anti_shake_ang_vel = RewTerm(
        func=mdp.body_angular_velocity_excess_l2,
        weight=-5.0e-3,
        params={
            # The bundled 29-DoF asset has no separate head rigid body; its
            # fixed torso is the corresponding angular-velocity proxy.
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                    "torso_link",
                ],
                preserve_order=True,
            ),
            "threshold": 1.5,
        },
    )
    feet_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[r".*ankle.*"]),
        },
    )


@configclass
class G1TerminationsCfg:
    """Termination terms aligned to the 29-DoF tracking environment."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reference_finished = DoneTerm(func=mdp.reference_trajectory_finished)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.25,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.8,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_reference_body_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_EE_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_EE_BODY_NAMES,
            "threshold": 0.25,
        },
    )
    # body too low
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.4,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )


@configclass
class G1SonicTerminationsCfg(G1TerminationsCfg):
    """Strict adaptive release termination protocol from SONIC."""

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_adaptive,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "pelvis",
            "threshold": 0.15,
            "down_threshold": 0.75,
            "root_height_threshold": 0.5,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori_full,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "pelvis",
            "threshold": 0.2,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_reference_body_pos_z_adaptive,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_EE_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_EE_BODY_NAMES,
            "threshold": 0.15,
            "down_threshold": 0.75,
            "root_height_threshold": 0.5,
        },
    )
    foot_pos_xyz = DoneTerm(
        func=mdp.bad_reference_body_pos_relative,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "reference_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
            "anchor_body_name": "pelvis",
            "threshold": 0.2,
        },
    )
    base_too_low = None


def _sonic_threshold_anneal_params(
    term_name: str,
    start_value: float,
    end_value: float,
) -> dict[str, object]:
    return {
        "term_name": term_name,
        "start_value": start_value,
        "end_value": end_value,
        "start_frames": 50_000_000,
        "end_frames": 500_000_000,
    }


@configclass
class G1SonicTerminationCurriculumCfg:
    """Anneal termination thresholds from SONIC base/eval values to strict.

    The release trains strict-from-scratch at 64+ GPU scale; locally that
    spends most of the early budget on ~5-step episodes. Starting at the
    release's own base/eval thresholds and reaching the strict release values
    by 500M frames recovers fast early learning while keeping every frame
    after the anneal - and the final policy's protocol - strictly SONIC.
    Override the shared window via
    ``env.curriculum.<term>.params.{start_frames,end_frames}``.
    """

    anchor_pos_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("anchor_pos", 0.25, 0.15),
    )
    anchor_ori_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("anchor_ori", 1.0, 0.2),
    )
    ee_body_pos_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("ee_body_pos", 0.25, 0.15),
    )
    foot_pos_xyz_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("foot_pos_xyz", 0.3, 0.2),
    )


@configclass
class ImitationG1BaseTrackingEnvCfg(ImitationLearningEnvCfg):
    """Shared 29-DoF G1 tracking config aligned with Unitree mimic tracking settings."""

    actions = G1ActionsCfg()
    observations = G1ObservationCfg()
    rewards = G1RewardsCfg()  # type: ignore
    terminations = G1TerminationsCfg()  # type: ignore
    events = G1EventCfg()

    device: str = "cuda"
    replay_reference: bool = False
    replay_only: bool = False
    reference_start_frame: int = 0
    enable_latent_command: bool = False
    latent_command_dim: int = 64
    latent_patch_past_steps: int = 0
    latent_patch_future_steps: int = 0
    # Anchor used when constructing expert batches and high-level macro states.
    # Keep the historical torso convention by default; SONIC overrides this to
    # pelvis so offline skill pretraining and live policy commands agree.
    expert_anchor_body_name: str = "torso_link"
    # Hold command-window observations for N control steps between renewals
    # (VLA-style chunk consumption): the window is snapshotted every N steps in
    # the renewal-time anchor frame and consumed as a time-shifted view with
    # tail padding. 0 keeps the per-step sliding-window behavior. Requires
    # latent_patch_past_steps == 0 when enabled.
    command_hold_steps: int = 0
    random_reset_step_min: int = 0
    random_reset_step_max: int = 0
    random_reset_full_trajectory: bool = False
    adaptive_failure_reset_bin_size: int = 50
    adaptive_failure_reset_sequence_length_agnostic: bool = True
    adaptive_failure_reset_init_num_failures: float = 1.0
    adaptive_failure_reset_uniform_ratio: float = 0.1
    adaptive_failure_reset_pre_failure_window: int = 200
    adaptive_failure_reset_failure_rate_max_over_mean: float = 50.0

    _debug_rewards: bool = False

    # Offscreen-video camera. Default: a static elevated bird view over the
    # env grid near the origin (set below via cfg.viewer), which shows a
    # couple dozen robots for generic motion-quality checks. The follow
    # camera remains available (video_follow_robot=True) for close-ups of a
    # single environment, e.g. with full-trajectory random starts where
    # robots wander far from their origins.
    video_follow_robot: bool = False
    video_follow_env_index: int = 0
    video_follow_eye_offset: tuple[float, float, float] = (3.5, 3.5, 2.0)
    video_follow_lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Master switch for all expensive visualizers/marker debug rendering.
    # Keep disabled by default for training/runtime performance.
    enable_visualizers: bool = False
    visualize_reference_arrows: bool = True
    print_reference_velocity: bool = False
    print_reference_velocity_every: int = 50

    # `target_joint_names` MUST be the robot articulation (USD) order because the
    # reference is written directly onto robot.data.joint_pos.torch. `reference_joint_names`
    # is the default order assumed for reference data; it is overridden at runtime
    # by the dataset's own `joint_names` when present (self-describing data), and the
    # reference->target remap converts to articulation order.
    reference_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    target_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    # Body order of the recorded NPZ body arrays (PhysX enumeration). Used by
    # the env instead of the live robot's body order, which is backend-specific.
    reference_body_names: list[str] = G1_29DOF_DATASET_BODY_NAMES.copy()
    command_ee_body_names: list[str] = G1_EE_BODY_NAMES.copy()
    command_observation_source: str = "reference"
    # The chunk adapter redirects only the three policy command tensors while
    # preserving the vanilla actor keys and 67-D command contract.
    policy_command_mode: str = "reference"

    def _sync_expert_window_observation_params(self) -> None:
        past_steps = int(self.latent_patch_past_steps)
        future_steps = int(self.latent_patch_future_steps)
        for term in (
            self.observations.expert_window.expert_motion,
            self.observations.expert_window.expert_anchor_pos_b,
            self.observations.expert_window.expert_anchor_ori_b,
        ):
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
        for term in (
            self.observations.expert_window.expert_ee_pos_b,
            self.observations.expert_window.expert_ee_ori_b,
        ):
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
            term.params["reference_body_names"] = tuple(self.command_ee_body_names)

    def __post_init__(self) -> None:
        super().__post_init__()  # type: ignore

        robot_preset = G1ImitationRobotCfg()
        for variant in (
            robot_preset.default,
            robot_preset.physx,
            robot_preset.newton_mjwarp,
        ):
            variant.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot_preset  # type: ignore
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Isaac Lab 3.0: SimulationCfg.physx was replaced by the backend-selecting
        # SimulationCfg.physics field. The PresetCfg resolves to PhysX by default
        # and to Newton with the `physics=newton_mjwarp` CLI override.
        self.sim.physics = G1ImitationPhysicsCfg()

        if self.scene.contact_forces is not None:
            # Per-backend sensor implementations; keep every variant's runtime
            # settings in sync because preset resolution happens after this.
            contact_preset = G1ImitationContactSensorCfg()
            for variant in (
                contact_preset.default,
                contact_preset.physx,
                contact_preset.newton_mjwarp,
            ):
                variant.update_period = self.sim.dt
                variant.force_threshold = 10.0
                variant.debug_vis = bool(self.enable_visualizers)
            self.scene.contact_forces = contact_preset

        # Reference marker visualizers are also gated by the master toggle.
        self.visualize_reference_arrows = bool(
            self.enable_visualizers and self.visualize_reference_arrows
        )

        self.scene.height_scanner = None

        if int(self.latent_patch_past_steps) < 0:
            raise ValueError("latent_patch_past_steps must be >= 0.")
        if int(self.latent_patch_future_steps) < 0:
            raise ValueError("latent_patch_future_steps must be >= 0.")
        if int(self.command_hold_steps) < 0:
            raise ValueError("command_hold_steps must be >= 0.")
        if int(self.command_hold_steps) > 0 and int(self.latent_patch_past_steps) > 0:
            raise ValueError(
                "command_hold_steps requires latent_patch_past_steps == 0; "
                "held chunk consumption is only defined for future-only windows."
            )
        normalized_policy_mode = (
            str(self.policy_command_mode).strip().lower().replace("-", "_")
        )
        if normalized_policy_mode not in {
            "reference",
            "full_body_chunk_current_slot",
        }:
            raise ValueError(
                "policy_command_mode must be reference or full_body_chunk_current_slot."
            )
        self.policy_command_mode = normalized_policy_mode
        if int(self.random_reset_step_min) < 0:
            raise ValueError("random_reset_step_min must be >= 0.")
        if int(self.random_reset_step_max) < int(self.random_reset_step_min):
            raise ValueError("random_reset_step_max must be >= random_reset_step_min.")
        if int(self.adaptive_failure_reset_bin_size) <= 0:
            raise ValueError("adaptive_failure_reset_bin_size must be positive.")
        if float(self.adaptive_failure_reset_init_num_failures) <= 0.0:
            raise ValueError(
                "adaptive_failure_reset_init_num_failures must be positive."
            )
        if not 0.0 <= float(self.adaptive_failure_reset_uniform_ratio) <= 1.0:
            raise ValueError("adaptive_failure_reset_uniform_ratio must be in [0, 1].")
        if int(self.adaptive_failure_reset_pre_failure_window) < 0:
            raise ValueError("adaptive_failure_reset_pre_failure_window must be >= 0.")
        if float(self.adaptive_failure_reset_failure_rate_max_over_mean) <= 0.0:
            raise ValueError(
                "adaptive_failure_reset_failure_rate_max_over_mean must be positive."
            )

        self._sync_expert_window_observation_params()


@configclass
class ImitationG1LafanTrackEnvCfg(ImitationG1BaseTrackingEnvCfg):
    """General 29-DoF motion-tracking env driven by a LAFAN1 manifest."""

    dataset_path: str | None = "data/lafan1/g1/"
    loader_type: str = "lafan1_csv"
    loader_kwargs: dict = {
        "dataset_name": "lafan1",
        "dataset": {"trajectories": {"lafan1_csv": []}},
        "control_freq": 50.0,
        "sim": {"dt": 0.005},
        "decimation": 4,
        "joint_names": G1_29DOF_ISAACLAB_JOINT_NAMES,
        "canonical_joint_names": G1_29DOF_ISAACLAB_JOINT_NAMES,
    }
    reset_schedule: str = "random"
    refresh_zarr_dataset: bool = False
    require_npz_body_states: bool = True
    lafan1_manifest_path: str | None = None
    motions: list[str] | None = None
    trajectories: list[str] | None = None
    wrap_steps: bool = False
    sync_control_rate_to_manifest: bool = True
    preferred_manifest_physics_fps: float = 240.0
    lafan1_loader_chunk_size: int | None = None
    lafan1_loader_shard_size: int | None = None
    reconstructed_reference_action: bool = True
    reconstructed_reference_action_mode = "next_pose"
    random_reset_full_trajectory: bool = True

    def _apply_optional_hydra_overrides(self, data: Mapping) -> dict:
        """Apply optional top-level overrides before Isaac Lab's strict type updater.

        Isaac Lab updates config objects by comparing the incoming value type against the
        runtime type of the existing attribute. That rejects Hydra overrides such as
        `None -> str` for optional public fields like `lafan1_manifest_path`.
        """
        remaining = dict(data)

        if "lafan1_manifest_path" in remaining:
            value = remaining.pop("lafan1_manifest_path")
            self.lafan1_manifest_path = None if value is None else str(value)

        if "dataset_path" in remaining:
            value = remaining.pop("dataset_path")
            self.dataset_path = None if value is None else str(value)

        if "motions" in remaining:
            value = remaining.pop("motions")
            if value is None:
                self.motions = None
            elif isinstance(value, (list, tuple)):
                self.motions = [str(item) for item in value]
            else:
                raise ValueError("motions must be a list of motion names or null.")

        if "trajectories" in remaining:
            value = remaining.pop("trajectories")
            if value is None:
                self.trajectories = None
            elif isinstance(value, (list, tuple)):
                self.trajectories = [str(item) for item in value]
            else:
                raise ValueError(
                    "trajectories must be a list of trajectory names or null."
                )

        return remaining

    def _lafan_source_entries(self) -> list[dict[str, object]]:
        try:
            entries = self.loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]
        except Exception as err:
            raise ValueError(
                "loader_kwargs must define dataset.trajectories.lafan1_csv with at least one source entry."
            ) from err
        if not isinstance(entries, list) or len(entries) == 0:
            raise ValueError(
                "loader_kwargs.dataset.trajectories.lafan1_csv must be a non-empty list."
            )
        return entries

    def _validate_source_path(self, source_path: Path) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(
                "LAFAN1 motion source is missing. "
                f"Expected: {source_path}. "
                "Set `lafan1_manifest_path` to a manifest that points at repo-local NPZ motions."
            )
        if self.require_npz_body_states and source_path.suffix.lower() != ".npz":
            raise ValueError(
                "This tracking env requires an npz source with body states "
                "(body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w). "
                f"Got: {source_path}. "
                "Generate repo-local NPZ files before loading this manifest."
            )

    def _normalize_sequence_overrides(self) -> None:
        if self.motions is not None:
            self.motions = list(self.motions)
        if self.trajectories is not None:
            self.trajectories = list(self.trajectories)

    def _validate_reset_schedule(self) -> None:
        allowed_reset_schedules = {"random", "sequential", "round_robin"}
        self.reset_schedule = self.reset_schedule.strip().lower()
        if self.reset_schedule not in allowed_reset_schedules:
            raise ValueError(
                f"Unsupported reset_schedule='{self.reset_schedule}'. "
                f"Allowed values: {sorted(allowed_reset_schedules)}."
            )

    def _validate_lafan_source_entries(
        self, source_entries: list[dict[str, object]]
    ) -> None:
        for source in source_entries:
            source_path = Path(str(source["path"])).expanduser().resolve()
            source["path"] = str(source_path)
            self._validate_source_path(source_path)

    def _set_control_frequency(self, control_freq: float) -> None:
        control_freq = float(control_freq)
        if control_freq <= 0.0:
            raise ValueError("control_freq must be positive.")

        def _integer_timing_for(
            physics_fps: float,
        ) -> tuple[float, int] | None:
            if physics_fps <= 0.0:
                return None
            decimation = max(int(round(physics_fps / control_freq)), 1)
            actual_control_freq = physics_fps / decimation
            if abs(actual_control_freq - control_freq) <= 1.0e-6:
                return 1.0 / physics_fps, decimation
            return None

        current_physics_fps = 1.0 / float(self.sim.dt)
        timing = _integer_timing_for(current_physics_fps)
        if timing is None:
            timing = _integer_timing_for(float(self.preferred_manifest_physics_fps))
        if timing is None:
            timing = (1.0 / control_freq, 1)

        self.sim.dt, self.decimation = timing
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            _set_contact_sensor_update_period(self.scene.contact_forces, self.sim.dt)

    def _sync_control_rate_to_manifest_entries(
        self,
        source_entries: list[dict[str, object]],
        *,
        timing_explicit: bool = False,
    ) -> None:
        if timing_explicit or not bool(self.sync_control_rate_to_manifest):
            return
        control_freq = infer_npz_manifest_control_freq(source_entries)
        if control_freq is None:
            return
        self._set_control_frequency(control_freq)

    def _resolve_manifest_config(
        self,
        *,
        dataset_path_explicit: bool = False,
        motions_explicit: bool = False,
        timing_explicit: bool = False,
    ) -> None:
        if self.lafan1_manifest_path is None:
            return

        _, manifest_entries = load_lafan1_manifest(self.lafan1_manifest_path)
        manifest_loader_options = load_lafan1_manifest_loader_options(
            self.lafan1_manifest_path
        )
        loader_chunk_size = self.lafan1_loader_chunk_size
        if loader_chunk_size is None:
            loader_chunk_size = manifest_loader_options.get("chunk_size")
        loader_shard_size = self.lafan1_loader_shard_size
        if loader_shard_size is None:
            loader_shard_size = manifest_loader_options.get("shard_size")
        self._sync_control_rate_to_manifest_entries(
            manifest_entries,
            timing_explicit=timing_explicit,
        )
        self.loader_type = "lafan1_csv"
        self.loader_kwargs = build_lafan1_loader_kwargs(
            entries=manifest_entries,
            sim_dt=float(self.sim.dt),
            decimation=int(self.decimation),
            joint_names=list(self.reference_joint_names),
            canonical_joint_names=list(self.target_joint_names),
        )

        if dataset_path_explicit and self.dataset_path is not None:
            self.dataset_path = str(Path(self.dataset_path).expanduser().resolve())
        else:
            self.dataset_path = dataset_path_from_entries(
                manifest_entries,
                manifest_path=self.lafan1_manifest_path,
            )

        if motions_explicit and self.motions is not None:
            self.motions = list(self.motions)
        else:
            self.motions = [str(entry["name"]) for entry in manifest_entries]

        self._validate_lafan_source_entries(
            self.loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]
        )

    def __post_init__(self) -> None:
        super().__post_init__()

        self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
        self._normalize_sequence_overrides()
        self._validate_reset_schedule()
        self._resolve_manifest_config()


# Backward-compatible aliases.
ImitationG1EnvCfg = ImitationG1LafanTrackEnvCfg


def _g1_lafan_track_env_cfg_from_dict(
    self: ImitationG1LafanTrackEnvCfg, data: dict
) -> None:
    dataset_path_explicit = isinstance(data, Mapping) and "dataset_path" in data
    motions_explicit = isinstance(data, Mapping) and "motions" in data
    timing_explicit = isinstance(data, Mapping) and (
        "sim" in data or "decimation" in data
    )

    if isinstance(data, Mapping):
        data = self._apply_optional_hydra_overrides(data)

    ImitationG1BaseTrackingEnvCfg.from_dict(self, data)
    self._sync_expert_window_observation_params()
    sync_goal_params = getattr(self, "_sync_expert_goal_observation_params", None)
    if callable(sync_goal_params):
        sync_goal_params()

    self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
    self._normalize_sequence_overrides()
    self._validate_reset_schedule()
    self._resolve_manifest_config(
        dataset_path_explicit=dataset_path_explicit,
        motions_explicit=motions_explicit,
        timing_explicit=timing_explicit,
    )


ImitationG1LafanTrackEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
