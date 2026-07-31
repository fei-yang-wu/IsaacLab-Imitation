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
    _apply_pelvis_protocol,
    _apply_strict_recipe,
    _bind_lafan_track_from_dict,
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
    # Latent surfaces default to the latent command; switch to an explicit
    # tracker with `env.command_mode=explicit` plus a matching
    # `command_observation_terms` / `agent.command_components` selection.
    command_mode: str = "latent"
    # Default skill-command width: skill code z (256) + sin_cos phase (2) = 258
    # (wandb run dh8k313e recipe, minus z_phi). Override per run as needed.
    latent_command_dim: int = 258
    latent_goal_steps: int = 1

    def _critic_prunable_command_term_names(self) -> tuple[str, ...]:
        # The supplemental explicit terms the latent critic gained for
        # explicit command mode; the historical latent critic terms
        # (expert_motion + anchors) are part of the latent critic contract
        # and are never pruned.
        return (
            "expert_motion_qpos",
            "expert_ee_pos_b",
            "expert_ee_ori_b",
            "expert_keypoint_pos_b",
            "expert_keypoint_ori_b",
        )

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

    def _anchor_term_names_by_group(self) -> dict[str, tuple[str, ...]]:
        return _LATENT_ANCHOR_TERM_NAMES_BY_GROUP


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
        _apply_pelvis_protocol(
            self,
            reset_step_max=0,
            random_reset_full_trajectory=True,
            failure_rate_max_over_mean=200.0,
        )


@configclass
class ImitationG1LatentSonicNoHistoryEnvCfg(ImitationG1LatentSonicEnvCfg):
    """SONIC release environment with this repo's single-frame observations.

    Everything on the environment side stays the SONIC release recipe --
    ``G1SonicRewardsCfg`` (pelvis anchor, 3-point local reward points,
    anti-shake, feet joint acceleration, elbow-exempt contact penalty),
    ``G1SonicTerminationsCfg`` (adaptive ``anchor_pos``/``ee_body_pos``,
    full ``anchor_ori``, ``foot_pos_xyz``, no ``base_too_low``),
    ``G1SonicTerminationCurriculumCfg``, ``G1SonicEventCfg`` (level0_4
    randomization), ``G1SonicActionsCfg``, ``G1SonicRobotCfg``, and SONIC's
    full-trajectory adaptive-failure reset sampling.

    The one deliberate departure is the observation set: the 2026-07-21
    isolated history ablation (``ImitationG1LatentStrictHistoryEnvCfg`` vs.
    the single-frame strict surface) showed SONIC's 10-step proprioceptive
    histories buy little at our scale, so this surface keeps the repo's
    single-frame ``G1LatentObservationCfg``. Term *names* are unchanged, so
    ``G1ImitationLatentSonicRLOptIPMDConfig``'s SONIC input-key selection
    (which adds ``projected_gravity`` and drops the robot body-pose terms
    from the actor) still resolves; only the per-term history length differs.
    """

    observations = G1LatentObservationCfg()


@configclass
class ImitationG1LatentStableEnvCfg(ImitationG1LatentSonicNoHistoryEnvCfg):
    """Default latent surface (2026-07-27): SONIC recipe, this repo's resets.

    The 2026-07-27 reset-sampling screen isolated why the SONIC environment
    trained so much worse at our scale. Holding geometry at 4096 x 24 and
    measuring windowed means over >=200M frames:

    ==============================  ========  =========
    arm                             ep_len    MPJPE
    ==============================  ========  =========
    Latent-v0, legacy resets          247.8    47.5 mm
    SONIC env (full-traj resets)       69.8    61.3 mm
    Latent-v0 + SONIC resets           44.6    59.7 mm
    ==============================  ========  =========

    Flipping *only* the reset sampler on an otherwise untouched Latent-v0
    collapsed episode length 5.6x -- more than the entire SONIC environment
    cost -- and `reference_finished` stayed at 0.0018, so those were real
    failures rather than starts landing near clip ends. SONIC trains at 64+
    GPU scale where full-trajectory adaptive-failure resets are affordable; at
    4096 environments they starve the policy of recoverable starts.

    This surface therefore keeps the whole SONIC release recipe --
    ``G1SonicRewardsCfg`` (including the 3-point local ``tracking_reward_points``
    precision term), ``G1SonicTerminationsCfg``, ``G1SonicEventCfg`` level0_4
    randomization, ``G1SonicActionsCfg``, ``G1SonicRobotCfg``, pelvis anchor,
    and single-frame observations -- and takes back only the legacy reset
    distribution: starts in [0, 200] with
    ``adaptive_failure_reset_failure_rate_max_over_mean=50``.

    Terminations are the strict release values from frame 0
    (``curriculum = None``), matching ``ImitationG1LatentStrictEnvCfg``. The
    SONIC anneal exists to avoid ~5-step early episodes, which legacy resets
    already prevent, and a moving threshold makes early episode-length and
    MPJPE curves incomparable across runs.

    Note that the terminations here are *identical* to the previous default's,
    not stricter -- both are ``G1SonicTerminationsCfg``. The only plausible
    source of an MPJPE gain over ``ImitationG1LatentStrictEnvCfg`` is the
    added reward terms, chiefly ``tracking_reward_points`` (weight 2.0,
    std 0.1).
    """

    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        # Undo `ImitationG1LatentSonicEnvCfg`'s full-trajectory adaptive-failure
        # sampler; this is the one axis this surface takes back from SONIC.
        # (The parent already anchored the surface at the pelvis.)
        _apply_pelvis_protocol(
            self, failure_rate_max_over_mean=50.0, set_anchor=False
        )


@configclass
class ImitationG1LatentStrictEnvCfg(ImitationG1LatentEnvCfg):
    """Strict recipe x latent command pin (`Isaac-Imitation-G1-Latent-Strict-v0`).

    Pelvis-anchored legacy surface with strict-from-scratch terminations;
    the recipe itself is shared with the explicit pin
    ``ImitationG1StrictTrackEnvCfg`` via ``_apply_strict_recipe``.

    The evidence-backed middle ground from the 2026-07-19/20 investigation:
    keep the scaffolding that trains at single-GPU/1B scale (legacy [0, 200]
    reset starts, mimic actuators, single-frame observations, bundled G1
    asset, proven optimizer contract) and take from SONIC only the pelvis
    anchor and the strict adaptive termination functions. Requires a
    pelvis-anchored skill encoder (e.g.
    ``skill_encoder_sonic_pelvis_h25_20260719``, sha256 ``388d3e82...``).

    Curriculum default removed (2026-07-21): the 50M -> 300M threshold anneal
    (``G1SonicTerminationCurriculumCfg``) made early training curves
    uninterpretable because the termination goalposts move while the policy
    learns. Thresholds are now the strict release values from frame 0.
    Opt back in with
    ``env.curriculum=G1SonicTerminationCurriculumCfg()``-style overrides if a
    run explicitly wants the anneal.
    """

    terminations = G1SonicTerminationsCfg()  # type: ignore
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        _apply_strict_recipe(self)


@configclass
class ImitationG1LatentStrictHistoryEnvCfg(ImitationG1LatentStrictEnvCfg):
    """Strict surface plus SONIC's 10-step proprioceptive history observations.

    Isolated history ablation (2026-07-21): identical to the default strict
    surface (pelvis anchor, strict-from-scratch terminations, no curriculum,
    legacy scaffolding) except policy/critic observations come from
    ``G1SonicLatentObservationCfg`` -- 10-step histories on the
    proprioceptive terms and SONIC's actor input set (adds
    ``projected_gravity``, drops the robot body-pose terms from the policy
    group). Pair with ``G1ImitationLatentSonicRLOptIPMDConfig`` (local
    optimizer contract default), which selects the SONIC input keys, so the
    only contract difference vs. ``Isaac-Imitation-G1-Latent-v0`` is the
    history/observation set: a low-cost stand-in for a recurrent policy.
    """

    observations = G1SonicLatentObservationCfg()


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
        self._sync_expert_window_observation_params()


@configclass
class ImitationG1LatentPerStepVQEnvCfg(ImitationG1LatentFutureCVAEEnvCfg):
    """Latent G1 env for ten-token, per-control-step command packets."""

    latent_command_dim: int = 64


@configclass
class ImitationG1LatentSonicOfficialFSQEnvCfg(ImitationG1LatentSonicEnvCfg):
    """SONIC release environment with a renewed 10-frame FSQ window command."""

    latent_command_dim: int = 64

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 9
        # Keep the sample-efficient reset sampler established by the Stable
        # reset screen; full-trajectory adaptive-failure starts need far more
        # data at our single-GPU scale. (Anchoring came from the SONIC parent.)
        _apply_pelvis_protocol(
            self, failure_rate_max_over_mean=50.0, set_anchor=False
        )
        # Zero means the observation window advances with the live reference.
        # The agent-side code_period=1 independently renews the quantized code.
        self.command_hold_steps = 0
        self._sync_expert_window_observation_params()


# Canonical-recipe alias: the Stable recipe is not latent-specific -- the
# command choice is pure configuration (`command_mode`) -- but the historical
# class name is pinned by the golden layout contract and by existing
# checkpoints/serialized configs, so the latent-named class stays canonical.
ImitationG1StableEnvCfg = ImitationG1LatentStableEnvCfg


_bind_lafan_track_from_dict(
    ImitationG1LatentEnvCfg,
    ImitationG1LatentSonicEnvCfg,
    ImitationG1LatentSonicNoHistoryEnvCfg,
    ImitationG1LatentStableEnvCfg,
    ImitationG1LatentStrictEnvCfg,
    ImitationG1LatentStrictHistoryEnvCfg,
    ImitationG1LatentGoalEnvCfg,
    ImitationG1LatentFutureCVAEEnvCfg,
    ImitationG1LatentPerStepVQEnvCfg,
    ImitationG1LatentSonicOfficialFSQEnvCfg,
)
