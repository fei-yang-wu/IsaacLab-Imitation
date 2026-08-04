# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward settings for the G1 tracking tasks (mimic baseline and SONIC)."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .... import mdp
from .constants import G1_FOOT_BODY_NAMES, G1_TRACKED_BODY_NAMES


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
    # SONIC `tracking_vr_5point_local`: pelvis + both wrists + both ankles, with
    # the wrists pushed 0.18 m forward and 0.025 m outboard so the tracked point
    # is the hand/controller rather than the wrist joint. This previously
    # tracked three points -- torso raised 0.5 m, plus bare wrists -- which
    # dropped both feet from the term entirely.
    #
    # The pelvis point is identically zero in the pelvis anchor frame for both
    # robot and reference, so it contributes no error and only divides the mean
    # by five instead of four. That is SONIC's own arrangement and is kept
    # deliberately: the term's scale must match theirs for the weight to mean
    # the same thing.
    tracking_reward_points = RewTerm(
        func=mdp.reference_local_reward_point_position_error_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "pelvis",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                preserve_order=True,
            ),
            "reference_body_names": [
                "pelvis",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
            "body_offsets": (
                (0.0, 0.0, 0.0),
                (0.18, -0.025, 0.0),
                (0.18, 0.025, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
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
    # The reward counterpart of the `foot_pos_xyz` termination, and a DELIBERATE
    # deviation from SONIC, which has no equivalent term.
    #
    # `foot_pos_xyz` is the only termination in either config that constrains
    # HORIZONTAL position -- `anchor_pos` and `ee_body_pos` test the Z component
    # alone, and `anchor_ori` is orientation. It is correspondingly the dominant
    # failure: measured over the 2026-08-03 5B runs it accounts for 66% of
    # LAFAN1 and 61% of BONES-SEED terminations that are not a timeout.
    #
    # Yet the feet were barely rewarded. Before the 5-point correction they
    # appeared only in `motion_body_pos`, weight 1.0 averaged over 14 bodies, so
    # the two ankles carried ~0.14 of effective weight while causing two thirds
    # of deaths. This term closes that gap directly: same reroot, same anchor,
    # same body set as the termination predicate
    # (`mdp.bad_reference_body_pos_relative`), so the policy is rewarded for
    # exactly the quantity that ends its episode.
    #
    # `std` 0.1 against the termination's 0.2 m threshold puts the kernel's
    # useful gradient inside the survivable band.
    #
    # UNSCREENED. The weight has not been through a hyperparameter screen; it is
    # a considered starting point, not a tuned value.
    motion_foot_pos = RewTerm(
        func=mdp.reference_relative_body_position_error_exp,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_FOOT_BODY_NAMES,
            "anchor_body_name": "pelvis",
            "std": 0.1,
        },
    )
    # SONIC weights this -2.5e-7; ours was -2.5e-6, a 10x stronger penalty.
    feet_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[r".*ankle.*"]),
        },
    )
