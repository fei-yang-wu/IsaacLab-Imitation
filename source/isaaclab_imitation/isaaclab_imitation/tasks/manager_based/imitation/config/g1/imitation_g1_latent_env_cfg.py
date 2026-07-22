# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ... import mdp
from .imitation_g1_env_cfg import (
    G1SonicActionsCfg,
    G1SonicEventCfg,
    G1SonicRewardsCfg,
    G1SonicRobotCfg,
    G1SonicTerminationCurriculumCfg,
    G1SonicTerminationsCfg,
    G1ObservationCfg,
    ImitationG1LafanTrackEnvCfg,
    _g1_canonical_joint_obs_params,
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
    PolicySupervisionCfg = G1ObservationCfg.PolicySupervisionCfg

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
    policy_supervision: PolicySupervisionCfg = PolicySupervisionCfg()


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
        self.sync_derived_fields()
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

    def _set_anchor_body(self, anchor_body_name: str) -> None:
        """Point every anchor-relative observation term at one body."""
        groups_and_terms = {
            "policy": (
                "expert_anchor_pos_b",
                "expert_anchor_ori_b",
                "body_pos",
                "body_ori",
            ),
            "critic": (
                "expert_anchor_pos_b",
                "expert_anchor_ori_b",
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
        for group_name, term_names in groups_and_terms.items():
            group = getattr(self.observations, group_name)
            for term_name in term_names:
                term = getattr(group, term_name)
                if term is None:
                    continue
                if "anchor_body_name" in term.params:
                    term.params["anchor_body_name"] = anchor_body_name

    def _set_reward_anchor_body(self, anchor_body_name: str) -> None:
        """Point the anchor-relative reward terms at one body."""
        for term_name in (
            "motion_global_anchor_pos",
            "motion_global_anchor_ori",
            "motion_body_pos",
            "motion_body_ori",
        ):
            getattr(self.rewards, term_name).params["anchor_body_name"] = (
                anchor_body_name
            )


@configclass
class ImitationG1LatentSonicEnvCfg(ImitationG1LatentEnvCfg):
    """Latent G1 task matched to the public SONIC release recipe.

    Termination thresholds are annealed from the release's base/eval values
    to its strict training values over the curriculum window; every frame
    after the window uses the strict release protocol. Disable with
    ``env.curriculum=null`` for strict-from-scratch release fidelity.
    """

    actions = G1SonicActionsCfg()
    observations = G1SonicLatentObservationCfg()
    rewards = G1SonicRewardsCfg()  # type: ignore
    terminations = G1SonicTerminationsCfg()  # type: ignore
    events = G1SonicEventCfg()
    curriculum = G1SonicTerminationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        robot_preset = G1SonicRobotCfg()
        for variant in (
            robot_preset.default,
            robot_preset.physx,
            robot_preset.newton_mjwarp,
        ):
            variant.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot_preset  # type: ignore

        # SONIC's motion library samples over the complete trajectory, with
        # adaptive failure weighting and a uniform component. The parent latent
        # task intentionally limits starts to [0, 200], so undo that only here.
        self.random_reset_step_min = 0
        self.random_reset_step_max = 0
        self.random_reset_full_trajectory = True
        self.adaptive_failure_reset_failure_rate_max_over_mean = 200.0
        self.expert_anchor_body_name = "pelvis"

        self._set_anchor_body("pelvis")
        self._set_reward_anchor_body("pelvis")


@configclass
class ImitationG1LatentStrictEnvCfg(ImitationG1LatentEnvCfg):
    """Pelvis-anchored legacy surface with annealed strict terminations.

    The evidence-backed middle ground from the 2026-07-19/20 investigation:
    keep the scaffolding that trains at single-GPU/1B scale (legacy [0, 200]
    reset starts, mimic actuators, single-frame observations, bundled G1
    asset, proven optimizer contract) and take from SONIC only the pelvis
    anchor and the strict adaptive termination functions, annealed from the
    release's base/eval thresholds to its strict values over 50M -> 300M
    frames. Requires a pelvis-anchored skill encoder (e.g.
    ``skill_encoder_sonic_pelvis_h25_20260719``, sha256 ``388d3e82...``).
    """

    terminations = G1SonicTerminationsCfg()  # type: ignore
    curriculum = G1SonicTerminationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.expert_anchor_body_name = "pelvis"
        self._set_anchor_body("pelvis")
        self._set_reward_anchor_body("pelvis")
        for term in (
            self.curriculum.anchor_pos_threshold,
            self.curriculum.anchor_ori_threshold,
            self.curriculum.ee_body_pos_threshold,
            self.curriculum.foot_pos_xyz_threshold,
        ):
            term.params["start_frames"] = 50_000_000
            term.params["end_frames"] = 300_000_000


@configclass
class ImitationG1LatentGoalEnvCfg(ImitationG1LatentEnvCfg):
    """Latent G1 env whose posterior command observes a held future goal state."""

    latent_command_dim: int = 128
    latent_goal_steps: int = 25


@configclass
class ImitationG1LatentFutureCVAEEnvCfg(ImitationG1LatentEnvCfg):
    """Latent G1 env exposing the current plus nine future reference frames."""

    latent_command_dim: int = 256

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 9
        self.command_hold_steps = 0
        self.sync_derived_fields()


@configclass
class ImitationG1LatentPerStepVQEnvCfg(ImitationG1LatentFutureCVAEEnvCfg):
    """Latent G1 env for ten-token, per-control-step command packets."""

    latent_command_dim: int = 64


ImitationG1LatentEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
ImitationG1LatentSonicEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
ImitationG1LatentStrictEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
ImitationG1LatentGoalEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
ImitationG1LatentFutureCVAEEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
ImitationG1LatentPerStepVQEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
