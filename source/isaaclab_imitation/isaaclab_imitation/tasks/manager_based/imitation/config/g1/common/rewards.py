# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward settings for the G1 tracking tasks (mimic baseline and SONIC)."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .... import mdp
from .constants import G1_TRACKED_BODY_NAMES


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

    # -- metrics
    # NOTE: this term is inert. RewardManager.compute() skips zero-weight terms
    # without calling them, so it never runs and Episode_Reward/mpjpe_m is a
    # constant zero. The live metric is logged by the env as Metrics/mpjpe_mm
    # and Metrics/mpjpe_mm_per_episode, driven by cfg.mpjpe_metric_body_names.
    # Kept only so the term name stays reserved and the reward table matches
    # historical runs; give it a non-zero weight only to make it a real reward.
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
