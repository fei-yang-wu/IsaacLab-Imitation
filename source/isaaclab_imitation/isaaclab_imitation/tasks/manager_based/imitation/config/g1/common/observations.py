# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vanilla (explicit-command) observation surface and shared obs-term params.

The ordered term list of each observation group IS the checkpoint input
contract (see ``tests/test_g1_task_layout_contract.py``): do not reorder,
add, or remove terms on an existing surface.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .... import mdp
from .constants import (
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_OBS_ANCHOR_BODY_NAME,
    G1_TRACKED_BODY_NAMES,
)


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
    """Return the expert joint command in the same pinned order as proprioception.

    The expert frame is stored in the live articulation order, which differs
    per physics backend. Without ``preserve_order=True`` the resolved indices
    are ascending in that live order, so the command would be delivered in a
    backend-specific permutation while ``joint_pos_rel`` and the action term
    stay pinned. Pinning here is what keeps the two pairable.
    """
    return {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
            preserve_order=True,
        )
    }


def _g1_expert_anchor_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
    }


def _g1_expert_window_motion_obs_params() -> dict[str, object]:
    """Window form of :func:`_g1_expert_motion_obs_params`; same pinning rule."""
    return {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
            preserve_order=True,
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


def _g1_expert_ee_obs_params() -> dict[str, object]:
    """Single-frame EE command params for the actor.

    The EE tracker is a single-frame consumer (126 = 90 proprioception + 36).
    Under ee_chunk_current_slot these terms return the phase-aligned slot of the
    held packet, mirroring how the full-body actor reads its command from the
    policy group rather than from the 10-frame expert_window group.
    """
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "reference_body_names": tuple(G1_EE_BODY_NAMES),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
    }


def _g1_expert_window_ee_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "reference_body_names": tuple(G1_EE_BODY_NAMES),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
        "past_steps": 0,
        "future_steps": 0,
    }


def _g1_expert_keypoint_obs_params() -> dict[str, object]:
    """Single-frame sparse-keypoint command params for the actor.

    Position and orientation are registered independently with these same body
    and anchor parameters. ``agent.command_components`` decides which terms the
    native tracker consumes.
    """
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "reference_body_names": tuple(G1_KEYPOINT5_BODY_NAMES),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
    }


def _g1_expert_window_keypoint_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "reference_body_names": tuple(G1_KEYPOINT5_BODY_NAMES),
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
        expert_motion_qpos = ObsTerm(
            func=mdp.policy_expert_motion_qpos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        expert_ee_pos_b = ObsTerm(
            func=mdp.policy_expert_ee_pos_b,
            params=_g1_expert_ee_obs_params(),
        )
        expert_ee_ori_b = ObsTerm(
            func=mdp.policy_expert_ee_ori_b,
            params=_g1_expert_ee_obs_params(),
        )
        expert_keypoint_pos_b = ObsTerm(
            func=mdp.policy_expert_keypoint_pos_b,
            params=_g1_expert_keypoint_obs_params(),
        )
        expert_keypoint_ori_b = ObsTerm(
            func=mdp.policy_expert_keypoint_ori_b,
            params=_g1_expert_keypoint_obs_params(),
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
        expert_motion_qpos = ObsTerm(
            func=mdp.policy_expert_motion_qpos,
            params=_g1_expert_motion_obs_params(),
        )
        expert_ee_pos_b = ObsTerm(
            func=mdp.policy_expert_ee_pos_b,
            params=_g1_expert_ee_obs_params(),
        )
        expert_ee_ori_b = ObsTerm(
            func=mdp.policy_expert_ee_ori_b,
            params=_g1_expert_ee_obs_params(),
        )
        expert_keypoint_pos_b = ObsTerm(
            func=mdp.policy_expert_keypoint_pos_b,
            params=_g1_expert_keypoint_obs_params(),
        )
        expert_keypoint_ori_b = ObsTerm(
            func=mdp.policy_expert_keypoint_ori_b,
            params=_g1_expert_keypoint_obs_params(),
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
        # Joint positions only (29), no velocities. Present so the DiffSR macro
        # state can be built over the root_qpos command space; unused unless
        # `expert_macro_state_terms` selects it.
        expert_motion_qpos = ObsTerm(
            func=mdp.expert_window_motion_qpos,
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
        expert_keypoint_pos_b = ObsTerm(
            func=mdp.expert_window_keypoint_pos_b,
            params=_g1_expert_window_keypoint_obs_params(),
        )
        expert_keypoint_ori_b = ObsTerm(
            func=mdp.expert_window_keypoint_ori_b,
            params=_g1_expert_window_keypoint_obs_params(),
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

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    expert_state: ExpertStateCfg = ExpertStateCfg()
    expert_window: ExpertWindowCfg = ExpertWindowCfg()
    reward_input: RewardInputCfg = RewardInputCfg()


# Policy-group command terms that `command_observation_terms` may retain. Only
# command terms are listed: proprioception is read by every command space, so
# pruning it would silently change the actor contract rather than save work.
_PRUNABLE_COMMAND_TERM_NAMES: tuple[str, ...] = (
    "expert_motion",
    "expert_motion_qpos",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
    "expert_ee_pos_b",
    "expert_ee_ori_b",
    "expert_keypoint_pos_b",
    "expert_keypoint_ori_b",
)

# Explicit command terms latent surfaces keep by default in latent mode: the
# historical "baseline test" terms exposed for posterior-mode baselines
# (the agent's input_keys decide what actually feeds each network).
_LATENT_MODE_DEFAULT_COMMAND_TERM_NAMES: tuple[str, ...] = (
    "expert_motion",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
)

# Every term declared on the expert_window group; the whitelist
# `expert_window_observation_terms` may retain any subset that still covers
# the active macro-state terms.
_EXPERT_WINDOW_TERM_NAMES: tuple[str, ...] = (
    "expert_motion",
    "expert_motion_qpos",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
    "expert_ee_pos_b",
    "expert_ee_ori_b",
    "expert_keypoint_pos_b",
    "expert_keypoint_ori_b",
)

# The full-body macro-state frame used when `expert_macro_state_terms` is None
# (expert_motion 58 + anchor_pos 3 + anchor_ori 6 = 67 per frame).
_DEFAULT_EXPERT_MACRO_STATE_TERMS: tuple[str, ...] = (
    "expert_motion",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
)


# Anchor-relative observation terms per group on the vanilla observation
# surface (no latent_command / expert_goal groups there).
_VANILLA_ANCHOR_TERM_NAMES_BY_GROUP: dict[str, tuple[str, ...]] = {
    # The keypoint term must follow the anchor body: root_points5's keypoints
    # and its root pose are one packet, re-expressed together by a single
    # anchor-frame transform. (The policy-group EE terms are deliberately not
    # listed: the abandoned EE tracker was trained with them pinned to
    # torso_link, and re-anchoring them now would break that checkpoint's
    # command contract.)
    "policy": (
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
        "expert_keypoint_pos_b",
        "expert_keypoint_ori_b",
    ),
    "critic": (
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
        "expert_keypoint_pos_b",
        "expert_keypoint_ori_b",
        "body_pos",
        "body_ori",
    ),
    "expert_state": ("expert_anchor_pos_b", "expert_anchor_ori_b"),
    "expert_window": (
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
        "expert_ee_pos_b",
        "expert_ee_ori_b",
        "expert_keypoint_pos_b",
        "expert_keypoint_ori_b",
    ),
    "reward_input": ("expert_anchor_pos_b", "expert_anchor_ori_b"),
}
