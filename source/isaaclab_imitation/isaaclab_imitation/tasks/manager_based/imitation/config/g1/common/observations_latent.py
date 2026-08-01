# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-command observation surfaces for the G1 tracking tasks.

The latent policy/critic groups also carry the explicit command superset
(pruned to None in latent command mode), so any surface here can serve an
explicit tracker via ``env.command_mode=explicit`` +
``command_observation_terms`` without a new env class.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .... import mdp
from .observations import (
    G1ObservationCfg,
    _g1_canonical_joint_obs_params,
    _g1_expert_anchor_obs_params,
    _g1_expert_ee_obs_params,
    _g1_expert_keypoint_obs_params,
    _g1_expert_motion_obs_params,
    _g1_tracked_body_obs_params,
)


# Anchor-relative observation terms per group on the latent observation
# surface. Unlike `_VANILLA_ANCHOR_TERM_NAMES_BY_GROUP`, the policy/critic
# groups carry robot body-pose terms and there is an `expert_goal` group.
# The explicit-command superset terms (EE/keypoint) follow the anchor here:
# they are new on this surface, so no legacy checkpoint pins them to
# torso_link the way the vanilla policy EE terms are pinned. In latent
# command mode they are pruned to None before anchoring, so this is inert
# for every latent task.
_LATENT_ANCHOR_TERM_NAMES_BY_GROUP: dict[str, tuple[str, ...]] = {
    "policy": (
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
        "expert_ee_pos_b",
        "expert_ee_ori_b",
        "expert_keypoint_pos_b",
        "expert_keypoint_ori_b",
        "body_pos",
        "body_ori",
    ),
    "critic": (
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
        "expert_ee_pos_b",
        "expert_ee_ori_b",
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
    ),
    "expert_goal": ("expert_anchor_pos_b", "expert_anchor_ori_b"),
    "reward_input": ("expert_anchor_pos_b", "expert_anchor_ori_b"),
}


@configclass
class G1LatentObservationCfg:
    """Latent-conditioned observation settings for the 29-DoF tracking environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        latent_command = ObsTerm(func=mdp.agent_latent_command)
        # baseline test
        expert_motion = ObsTerm(
            func=mdp.expert_motion_command,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # Explicit command superset (pruned to None in latent command mode):
        # present so this same observation surface can serve an explicit
        # tracker via `env.command_mode=explicit` + `command_observation_terms`
        # without a separate env class. Func bindings mirror the vanilla
        # policy group (`policy_*` variants honor chunk command adapters).
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
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params=_g1_tracked_body_obs_params(),
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params=_g1_tracked_body_obs_params(),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
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

        latent_command = ObsTerm(func=mdp.agent_latent_command)
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
        # Explicit command superset; see the policy-group comment. Pruned to
        # None in latent command mode so existing critics are unchanged.
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
            history_length=3,
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=3)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=3)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params=_g1_canonical_joint_obs_params(),
            history_length=3,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params=_g1_canonical_joint_obs_params(),
            history_length=3,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params=_g1_canonical_joint_obs_params(),
            history_length=3,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params=_g1_canonical_joint_obs_params(),
            history_length=3,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = False

    ExpertStateCfg = G1ObservationCfg.ExpertStateCfg
    ExpertWindowCfg = G1ObservationCfg.ExpertWindowCfg
    RewardInputCfg = G1ObservationCfg.RewardInputCfg

    @configclass
    class ExpertGoalCfg(ObsGroup):
        """Single future expert goal observations exposed for hierarchical skills."""

        expert_motion = ObsTerm(
            func=mdp.expert_goal_motion,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_goal_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_goal_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    expert_state: ExpertStateCfg = ExpertStateCfg()
    expert_window: ExpertWindowCfg = ExpertWindowCfg()
    expert_goal: ExpertGoalCfg = ExpertGoalCfg()
    reward_input: RewardInputCfg = RewardInputCfg()


@configclass
class G1SonicLatentObservationCfg(G1LatentObservationCfg):
    """Latent command plus the 10-step proprioceptive histories used by SONIC."""

    @configclass
    class PolicyCfg(G1LatentObservationCfg.PolicyCfg):
        # SONIC's actor consumes only the latent command and proprioceptive
        # history, but the expert reference terms stay EXPOSED in this group so
        # posterior-mode baselines keep their standard policy-group inputs.
        # The agent config's input_keys select what actually feeds each
        # network; do not strip terms here.
        body_pos = None
        body_ori = None
        base_lin_vel = None
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=10,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=10,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params=_g1_canonical_joint_obs_params(),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=10,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params=_g1_canonical_joint_obs_params(),
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=10,
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=10)

    @configclass
    class CriticCfg(G1LatentObservationCfg.CriticCfg):
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params=_g1_tracked_body_obs_params(),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params=_g1_canonical_joint_obs_params(),
            history_length=10,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params=_g1_canonical_joint_obs_params(),
            history_length=10,
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=10)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
