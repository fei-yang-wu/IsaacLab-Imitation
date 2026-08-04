# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward settings for the G1 tracking tasks (mimic baseline and SONIC)."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .... import mdp
from .constants import (
    G1_FOOT_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
    G1_WRIST_BODY_NAMES,
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
    # Coarse companion to `motion_global_anchor_pos`. INERT by default; enable
    # by override.
    #
    # World-frame anchor error is an INTEGRAL of past velocity error, so it is
    # not correctable within a step: once the robot has drifted, the only route
    # back is a sustained velocity bias. A narrow kernel cannot ask for that,
    # because it is numerically zero long before the drift is large. Measured on
    # the s15 setting (std 0.10, weight 2.0) against a mean training drift of
    # 0.215 m:
    #
    #   drift    s15 gradient   this term at std 0.5, w 1.0
    #   0.05 m         15.58                          0.40
    #   0.215 m         0.85                          1.43
    #   0.60 m          0.00                          1.14
    #   1.00 m          0.00                          0.15
    #
    # So s15 did not add global pull -- it traded far-field pull for near-field
    # precision, which is exactly why it moved strict MPJPE by 37% and left the
    # full-horizon pass untouched. This term restores a gradient that still
    # points home at 0.6-1.0 m while adding ~3% at 0.05 m, so it does not dilute
    # the precision s15 bought.
    motion_global_anchor_pos_wide = RewTerm(
        func=mdp.reference_global_anchor_position_error_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "pelvis",
            "std": 0.5,
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
    # World-frame body tracking -- the reward counterpart of MPJPE-G.
    #
    # `motion_body_pos` above is REROOTED, so it is blind to global drift by
    # construction, and drift is what dominates the global metrics: on the
    # 2026-08-04 screen world-frame body error tracked root drift almost 1:1,
    # and the arms that moved MPJPE-G moved the root rather than the rerooted
    # body term.
    #
    # Inert by default so the current contract is unchanged and
    # `RewardManager.compute` skips it entirely. Enable per run with
    # `env.rewards.motion_body_pos_global.weight=<w>`. UNSCREENED.
    motion_body_pos_global = RewTerm(
        func=mdp.reference_global_body_position_error_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_TRACKED_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 0.1,
        },
    )
    # Wrist end-effector tracking, INERT BY DEFAULT (weight 0.0).
    #
    # The wrists are the least-constrained bodies in the whole contract: no
    # termination bounds them horizontally -- `ee_body_pos` checks the Z
    # component alone and `foot_pos_xyz` covers only the ankles -- and their
    # only positional reward is their share of `tracking_reward_points`, where
    # they are 2 of 5 points. Feet by contrast have a 3D termination and a
    # dedicated 3D reward.
    #
    # Same geometry as `motion_foot_pos`, on the hands instead. Left at 0.0 so
    # the default is unchanged and the term costs nothing --
    # `RewardManager.compute` skips zero-weight terms without calling them --
    # and a screen arm enables it with
    # `env.rewards.motion_ee_pos.weight=<w>`.
    motion_ee_pos = RewTerm(
        func=mdp.reference_relative_body_position_error_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_WRIST_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_WRIST_BODY_NAMES,
            "anchor_body_name": "pelvis",
            "std": 0.1,
        },
    )
    # World-frame end-effector and foot tracking. Both INERT BY DEFAULT.
    #
    # Every position reward in this config except the two anchor terms is
    # REROOTED, and the anchor terms watch the root alone at weight 0.5 each --
    # the two lowest weights here. So roughly 8.0 of position-reward weight is
    # drift-blind and 1.0 anchors the robot in the world, which is why the
    # policy drifts: it is barely paid not to.
    #
    # A rerooted reward is also ambiguous as a target. It says "put the hand
    # here relative to your own pelvis", which a robot can satisfy perfectly
    # while standing somewhere else entirely. Measured: adding the LOCAL wrist
    # term (s13) improved root-relative EE to the best in the screen, 0.0284,
    # while root drift ROSE 0.0707 -> 0.0949 and MPJPE-G got 28% worse.
    #
    # These give the same body sets a world-frame target. Enable per run with
    # `env.rewards.motion_ee_pos_global.weight=<w>`. UNSCREENED.
    motion_ee_pos_global = RewTerm(
        func=mdp.reference_global_body_position_error_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_WRIST_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_WRIST_BODY_NAMES,
            "std": 0.1,
        },
    )
    motion_foot_pos_global = RewTerm(
        func=mdp.reference_global_body_position_error_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_FOOT_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_FOOT_BODY_NAMES,
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
