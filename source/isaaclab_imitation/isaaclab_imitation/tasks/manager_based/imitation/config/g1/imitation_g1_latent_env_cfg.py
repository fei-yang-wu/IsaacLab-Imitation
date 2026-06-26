# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import mdp
from .imitation_g1_env_cfg import (
    G1ObservationCfg,
    ImitationG1LafanTrackEnvCfg,
    _g1_lafan_track_env_cfg_from_dict,
    _g1_expert_anchor_obs_params,
    _g1_expert_motion_obs_params,
    _g1_tracked_body_obs_params,
)


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
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
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
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, history_length=3)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, history_length=3)
        joint_pos = ObsTerm(func=mdp.joint_pos, history_length=3)
        joint_vel = ObsTerm(func=mdp.joint_vel, history_length=3)
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
class ImitationG1LatentEnvCfg(ImitationG1LafanTrackEnvCfg):
    """Latent-conditioned G1 motion-tracking env driven by a LAFAN1 manifest."""

    observations = G1LatentObservationCfg()
    enable_latent_command: bool = True
    # Default skill-command width: skill code z (256) + sin_cos phase (2) = 258
    # (wandb run dh8k313e recipe, minus z_phi). Override per run as needed.
    latent_command_dim: int = 258
    latent_goal_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 0
        self.random_reset_step_min = 0
        self.random_reset_step_max = 200
        self.random_reset_full_trajectory = False
        self._sync_expert_window_observation_params()
        self._sync_expert_goal_observation_params()
        # No reference-based terminations in latent mode
        # self.terminations.anchor_pos = None
        # self.terminations.anchor_ori = None
        # self.terminations.ee_body_pos = None

    def _sync_expert_goal_observation_params(self) -> None:
        goal_steps = int(self.latent_goal_steps)
        if goal_steps < 0:
            raise ValueError("latent_goal_steps must be >= 0.")
        for term in (
            self.observations.expert_goal.expert_motion,
            self.observations.expert_goal.expert_anchor_pos_b,
            self.observations.expert_goal.expert_anchor_ori_b,
        ):
            term.params["goal_steps"] = goal_steps


@configclass
class ImitationG1LatentGoalEnvCfg(ImitationG1LatentEnvCfg):
    """Latent G1 env whose posterior command observes a held future goal state."""

    latent_command_dim: int = 128
    latent_goal_steps: int = 25


ImitationG1LatentEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
ImitationG1LatentGoalEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
