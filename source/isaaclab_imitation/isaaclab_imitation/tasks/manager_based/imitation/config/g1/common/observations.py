# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 observation surfaces, composed from the single-source term factories.

The ordered term list of each observation group IS the checkpoint input
contract (see ``tests/test_g1_task_layout_contract.py``): do not reorder,
add, or remove terms on a frozen surface.

Consolidation (2026-08-01): every observation term on every G1 surface is
built by the ``_*_term`` factories below -- one source for func / params /
noise / history. A surface is pure composition: it declares its term
membership by calling the factories with its noise and history profile.
The surfaces differ only in membership and profiles:

- :class:`G1ObservationCfg` -- the vanilla (explicit-command) surface, frozen
  for ``-G1-v0`` (no latent_command, explicit terms via the chunk-aware
  ``policy_*`` funcs, actor proprio noise).
- :class:`G1LatentObservationCfg` -- the latent surface, frozen for
  ``-G1-v1`` and the ASE-era latent tasks (latent_command, explicit
  superset, ASE-era h3 critic histories).
- :class:`G1SonicLatentObservationCfg` -- SONIC's 10-step proprio histories,
  applied in ``__post_init__`` (no-history-by-default convention), with the
  ASE-era dead critic terms deleted.
- :class:`G1V2ObservationCfg` -- the single v2 surface: policy + critic (plus
  the parked ``reward_input`` group), every command term read through the
  declared command interface's two channels.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .... import mdp
from ....command_components import LATENT_COMMAND_TERM_NAME
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


# ---------------------------------------------------------------------------
# Term factories: the single source of every ObsTerm on every G1 surface.
# ---------------------------------------------------------------------------


def _expert_motion_term(*, func, noise: Unoise | None = None) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_motion_obs_params(), noise=noise)


def _expert_motion_qpos_term(*, func) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_motion_obs_params())


def _expert_anchor_pos_term(*, func, noise: Unoise | None = None) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_anchor_obs_params(), noise=noise)


def _expert_anchor_ori_term(*, func, noise: Unoise | None = None) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_anchor_obs_params(), noise=noise)


def _expert_ee_pos_term(*, func) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_ee_obs_params())


def _expert_ee_ori_term(*, func) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_ee_obs_params())


def _expert_keypoint_pos_term(*, func) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_keypoint_obs_params())


def _expert_keypoint_ori_term(*, func) -> ObsTerm:
    return ObsTerm(func=func, params=_g1_expert_keypoint_obs_params())


def _window_motion_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_motion, params=_g1_expert_window_motion_obs_params()
    )


def _window_motion_qpos_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_motion_qpos, params=_g1_expert_window_motion_obs_params()
    )


def _window_anchor_pos_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_anchor_pos_b,
        params=_g1_expert_window_anchor_obs_params(),
    )


def _window_anchor_ori_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_anchor_ori_b,
        params=_g1_expert_window_anchor_obs_params(),
    )


def _window_ee_pos_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_ee_pos_b, params=_g1_expert_window_ee_obs_params()
    )


def _window_ee_ori_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_ee_ori_b, params=_g1_expert_window_ee_obs_params()
    )


def _window_keypoint_pos_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_keypoint_pos_b,
        params=_g1_expert_window_keypoint_obs_params(),
    )


def _window_keypoint_ori_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_window_keypoint_ori_b,
        params=_g1_expert_window_keypoint_obs_params(),
    )


def _goal_motion_term() -> ObsTerm:
    return ObsTerm(func=mdp.expert_goal_motion, params=_g1_expert_motion_obs_params())


def _goal_anchor_pos_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_goal_anchor_pos_b, params=_g1_expert_anchor_obs_params()
    )


def _goal_anchor_ori_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.expert_goal_anchor_ori_b, params=_g1_expert_anchor_obs_params()
    )


def _reward_input_motion_term() -> ObsTerm:
    return ObsTerm(func=mdp.robot_motion, params=_g1_expert_motion_obs_params())


def _reward_input_anchor_pos_term() -> ObsTerm:
    return ObsTerm(func=mdp.expert_anchor_pos_b, params=_g1_expert_anchor_obs_params())


def _reward_input_anchor_ori_term() -> ObsTerm:
    return ObsTerm(func=mdp.expert_anchor_ori_b, params=_g1_expert_anchor_obs_params())


def _reward_input_unit_motion_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.reward_robot_joint_pos, params=_g1_expert_motion_obs_params()
    )


def _reward_input_unit_desired_joint_pos_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.reward_expert_desired_joint_pos,
        params=_g1_expert_motion_obs_params(),
    )


def _reward_input_unit_anchor_pos_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.reward_expert_anchor_pos_b, params=_g1_expert_anchor_obs_params()
    )


def _reward_input_unit_anchor_ori_term() -> ObsTerm:
    return ObsTerm(
        func=mdp.reward_expert_anchor_ori_b, params=_g1_expert_anchor_obs_params()
    )


def _command_component_term(
    component: str,
    *,
    channel: str = "actor",
    noise: Unoise | None = None,
) -> ObsTerm:
    """One command component read through a command channel (the v2 surface).

    ``channel`` and the window are rewritten from the declared command
    interface at environment construction, so the values here are only the
    declaration's defaults; ``noise`` is the surface's own profile.
    """
    return ObsTerm(
        func=mdp.command_component,
        params={"channel": channel, "component": component},
        noise=noise,
    )


def _latent_command_term(func=mdp.agent_latent_command) -> ObsTerm:
    """The agent-published latent command (z + phase).

    ``mdp.agent_latent_command`` reads the env buffer directly; the v2
    single-producer surfaces pass ``mdp.reference_latent_command`` to read
    the ``command`` CommandManager term instead.
    """
    return ObsTerm(func=func)


def _projected_gravity_term(noise: Unoise | None = None, history: int = 0) -> ObsTerm:
    return ObsTerm(func=mdp.projected_gravity, noise=noise, history_length=history)


def _body_pos_term() -> ObsTerm:
    return ObsTerm(func=mdp.robot_body_pos_b, params=_g1_tracked_body_obs_params())


def _body_ori_term(history: int = 0) -> ObsTerm:
    return ObsTerm(
        func=mdp.robot_body_ori_b,
        params=_g1_tracked_body_obs_params(),
        history_length=history,
    )


def _base_lin_vel_term(noise: Unoise | None = None, history: int = 0) -> ObsTerm:
    return ObsTerm(func=mdp.base_lin_vel, noise=noise, history_length=history)


def _base_ang_vel_term(noise: Unoise | None = None, history: int = 0) -> ObsTerm:
    return ObsTerm(func=mdp.base_ang_vel, noise=noise, history_length=history)


def _joint_pos_rel_term(noise: Unoise | None = None, history: int = 0) -> ObsTerm:
    return ObsTerm(
        func=mdp.joint_pos_rel,
        params=_g1_canonical_joint_obs_params(),
        noise=noise,
        history_length=history,
    )


def _joint_vel_rel_term(noise: Unoise | None = None, history: int = 0) -> ObsTerm:
    return ObsTerm(
        func=mdp.joint_vel_rel,
        params=_g1_canonical_joint_obs_params(),
        noise=noise,
        history_length=history,
    )


def _joint_pos_term(history: int = 0) -> ObsTerm:
    return ObsTerm(
        func=mdp.joint_pos,
        params=_g1_canonical_joint_obs_params(),
        history_length=history,
    )


def _joint_vel_term(history: int = 0) -> ObsTerm:
    return ObsTerm(
        func=mdp.joint_vel,
        params=_g1_canonical_joint_obs_params(),
        history_length=history,
    )


def _last_action_term(history: int = 0) -> ObsTerm:
    return ObsTerm(func=mdp.last_action, history_length=history)


# ---------------------------------------------------------------------------
# Shared non-policy groups (aliased by every surface that carries them).
# ---------------------------------------------------------------------------


@configclass
class ExpertStateCfg(ObsGroup):
    """Single-frame expert observations exposed through the observation manager."""

    joint_pos = ObsTerm(
        func=mdp.expert_joint_pos, params=_g1_expert_motion_obs_params()
    )
    joint_vel = ObsTerm(
        func=mdp.expert_joint_vel, params=_g1_expert_motion_obs_params()
    )
    root_pos = ObsTerm(func=mdp.expert_root_pos)
    root_quat = ObsTerm(func=mdp.expert_root_quat)
    root_lin_vel = ObsTerm(func=mdp.expert_root_lin_vel)
    root_ang_vel = ObsTerm(func=mdp.expert_root_ang_vel)
    expert_motion = _expert_motion_term(func=mdp.expert_motion_command)
    expert_anchor_ori_b = _expert_anchor_ori_term(func=mdp.expert_anchor_ori_b)
    expert_anchor_pos_b = _expert_anchor_pos_term(func=mdp.expert_anchor_pos_b)

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class ExpertWindowCfg(ObsGroup):
    """Temporal expert observations exposed through the observation manager."""

    expert_motion = _window_motion_term()
    # Joint positions only (29), no velocities. Present so the DiffSR macro
    # state can be built over the root_qpos command space; unused unless
    # `expert_macro_state_terms` selects it.
    expert_motion_qpos = _window_motion_qpos_term()
    expert_anchor_pos_b = _window_anchor_pos_term()
    expert_anchor_ori_b = _window_anchor_ori_term()
    expert_ee_pos_b = _window_ee_pos_term()
    expert_ee_ori_b = _window_ee_ori_term()
    expert_keypoint_pos_b = _window_keypoint_pos_term()
    expert_keypoint_ori_b = _window_keypoint_ori_term()

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class ExpertGoalCfg(ObsGroup):
    """Single future expert goal observations exposed for hierarchical skills."""

    expert_motion = _goal_motion_term()
    expert_anchor_pos_b = _goal_anchor_pos_term()
    expert_anchor_ori_b = _goal_anchor_ori_term()

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class RewardInputCfg(ObsGroup):
    """Inputs consumed by discriminator / reward estimator networks.

    The frozen v0/v1 shape, served by ``ImitationRLEnvLegacy``'s expert-side
    cache: raw robot motion (joint_pos + joint_vel) plus raw anchor error. On
    rollout, terms are computed from the robot's actual state; on the expert
    minibatch the env's expert-observation mapper returns the idealized-expert
    counterpart (reference motion, zero anchor error). The v2 surface uses
    :class:`RewardInputUnitCfg` instead.
    """

    expert_motion = _reward_input_motion_term()
    expert_anchor_pos_b = _reward_input_anchor_pos_term()
    expert_anchor_ori_b = _reward_input_anchor_ori_term()

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class RewardInputUnitCfg(ObsGroup):
    """v2 IPMD reward-estimator inputs, all normalized into [0, 1].

    Three blocks: joint positions (soft-limit normalized), the relative root
    (anchor) position, and the relative root orientation as rot6d — see
    ``isaaclab_imitation.envs.reward_input_normalization``. On rollout, terms
    are computed from the robot's actual state; on the expert minibatch the
    composed env's data plane returns the idealized-expert counterpart
    (normalized reference joint positions, 0.5 anchor error, normalized
    identity rot6d) through the same helpers. The term names are the frozen
    key contract (`REWARD_INPUT_KEYS`), kept even though `expert_motion` here
    carries joint positions only.
    """

    expert_motion = _reward_input_unit_motion_term()
    expert_anchor_pos_b = _reward_input_unit_anchor_pos_term()
    expert_anchor_ori_b = _reward_input_unit_anchor_ori_term()
    # Goal conditioning (2026-08-23): the commanded reference joints, so the
    # estimator can learn r(s, g) — a tracking metric, not just
    # expert-likeness. REWARD-ESTIMATOR ONLY: expert information reaches the
    # actor exclusively through the command interface, never through this
    # group, which no actor or critic reads. Consumed only when the agent
    # declares `reward_estimation_pair_input=true`; otherwise computed and
    # ignored, keeping pre-pairing checkpoints resumable from this tree.
    expert_desired_joint_pos = _reward_input_unit_desired_joint_pos_term()

    def __post_init__(self):
        self.concatenate_terms = False


# ---------------------------------------------------------------------------
# Surfaces.
# ---------------------------------------------------------------------------


@configclass
class G1ObservationCfg:
    """Vanilla (explicit-command) observation settings; frozen for ``-G1-v0``."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        expert_motion = _expert_motion_term(func=mdp.policy_expert_motion_command)
        expert_anchor_pos_b = _expert_anchor_pos_term(
            func=mdp.policy_expert_anchor_pos_b
        )
        expert_anchor_ori_b = _expert_anchor_ori_term(
            func=mdp.policy_expert_anchor_ori_b
        )
        expert_motion_qpos = _expert_motion_qpos_term(
            func=mdp.policy_expert_motion_qpos
        )
        expert_ee_pos_b = _expert_ee_pos_term(func=mdp.policy_expert_ee_pos_b)
        expert_ee_ori_b = _expert_ee_ori_term(func=mdp.policy_expert_ee_ori_b)
        expert_keypoint_pos_b = _expert_keypoint_pos_term(
            func=mdp.policy_expert_keypoint_pos_b
        )
        expert_keypoint_ori_b = _expert_keypoint_ori_term(
            func=mdp.policy_expert_keypoint_ori_b
        )
        base_lin_vel = _base_lin_vel_term(noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = _base_ang_vel_term(noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = _joint_pos_rel_term(noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = _joint_vel_rel_term(noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = _last_action_term()

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        expert_motion = _expert_motion_term(func=mdp.expert_motion_command)
        expert_anchor_pos_b = _expert_anchor_pos_term(func=mdp.expert_anchor_pos_b)
        expert_anchor_ori_b = _expert_anchor_ori_term(func=mdp.expert_anchor_ori_b)
        expert_motion_qpos = _expert_motion_qpos_term(
            func=mdp.policy_expert_motion_qpos
        )
        expert_ee_pos_b = _expert_ee_pos_term(func=mdp.policy_expert_ee_pos_b)
        expert_ee_ori_b = _expert_ee_ori_term(func=mdp.policy_expert_ee_ori_b)
        expert_keypoint_pos_b = _expert_keypoint_pos_term(
            func=mdp.policy_expert_keypoint_pos_b
        )
        expert_keypoint_ori_b = _expert_keypoint_ori_term(
            func=mdp.policy_expert_keypoint_ori_b
        )
        body_pos = _body_pos_term()
        body_ori = _body_ori_term()
        base_lin_vel = _base_lin_vel_term()
        base_ang_vel = _base_ang_vel_term()
        joint_pos_rel = _joint_pos_rel_term()
        joint_vel_rel = _joint_vel_rel_term()
        last_action = _last_action_term()

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    expert_state: ExpertStateCfg = ExpertStateCfg()
    expert_window: ExpertWindowCfg = ExpertWindowCfg()
    reward_input: RewardInputCfg = RewardInputCfg()


@configclass
class G1LatentObservationCfg:
    """Latent-conditioned observation settings; frozen for ``-G1-v1`` / ASE-era tasks.

    The critic carries the ASE-era h3 histories and the three dead terms
    (``projected_gravity`` / ``joint_pos`` / ``joint_vel``); the v2 surfaces
    delete them (see :class:`G1SonicLatentObservationCfg` /
    :class:`G1FullSurfaceObservationCfg`).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        latent_command = _latent_command_term()
        # baseline test
        expert_motion = _expert_motion_term(func=mdp.expert_motion_command)
        expert_anchor_pos_b = _expert_anchor_pos_term(
            func=mdp.expert_anchor_pos_b, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        expert_anchor_ori_b = _expert_anchor_ori_term(
            func=mdp.expert_anchor_ori_b, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # Explicit command superset (pruned to None in latent command mode):
        # present so this same observation surface can serve an explicit
        # tracker via `env.command_mode=explicit` + `command_observation_terms`
        # without a separate env class. Func bindings mirror the vanilla
        # policy group (`policy_*` variants honor chunk command adapters).
        expert_motion_qpos = _expert_motion_qpos_term(
            func=mdp.policy_expert_motion_qpos
        )
        expert_ee_pos_b = _expert_ee_pos_term(func=mdp.policy_expert_ee_pos_b)
        expert_ee_ori_b = _expert_ee_ori_term(func=mdp.policy_expert_ee_ori_b)
        expert_keypoint_pos_b = _expert_keypoint_pos_term(
            func=mdp.policy_expert_keypoint_pos_b
        )
        expert_keypoint_ori_b = _expert_keypoint_ori_term(
            func=mdp.policy_expert_keypoint_ori_b
        )
        projected_gravity = _projected_gravity_term(
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        body_pos = _body_pos_term()
        body_ori = _body_ori_term()
        base_lin_vel = _base_lin_vel_term(noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = _base_ang_vel_term(noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = _joint_pos_rel_term(noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = _joint_vel_rel_term(noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = _last_action_term()

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        latent_command = _latent_command_term()
        expert_motion = _expert_motion_term(func=mdp.expert_motion_command)
        expert_anchor_pos_b = _expert_anchor_pos_term(func=mdp.expert_anchor_pos_b)
        expert_anchor_ori_b = _expert_anchor_ori_term(func=mdp.expert_anchor_ori_b)
        # Explicit command superset; see the policy-group comment. Pruned to
        # None in latent command mode so existing critics are unchanged.
        expert_motion_qpos = _expert_motion_qpos_term(
            func=mdp.policy_expert_motion_qpos
        )
        expert_ee_pos_b = _expert_ee_pos_term(func=mdp.policy_expert_ee_pos_b)
        expert_ee_ori_b = _expert_ee_ori_term(func=mdp.policy_expert_ee_ori_b)
        expert_keypoint_pos_b = _expert_keypoint_pos_term(
            func=mdp.policy_expert_keypoint_pos_b
        )
        expert_keypoint_ori_b = _expert_keypoint_ori_term(
            func=mdp.policy_expert_keypoint_ori_b
        )
        body_pos = _body_pos_term()
        body_ori = _body_ori_term(history=3)
        projected_gravity = _projected_gravity_term(history=3)
        base_lin_vel = _base_lin_vel_term(history=3)
        base_ang_vel = _base_ang_vel_term(history=3)
        joint_pos_rel = _joint_pos_rel_term(history=3)
        joint_vel_rel = _joint_vel_rel_term(history=3)
        joint_pos = _joint_pos_term(history=3)
        joint_vel = _joint_vel_term(history=3)
        last_action = _last_action_term()

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
    """Latent command plus the 10-step proprioceptive histories used by SONIC.

    No-history-by-default convention (2026-08-01): every observation term is
    declared history-free; this surface applies SONIC's 10-step proprio
    histories in ``__post_init__`` instead of declaring them inline. The
    ASE-era dead critic terms (``projected_gravity`` / ``joint_pos`` /
    ``joint_vel``) are deleted.
    """

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
        projected_gravity = _projected_gravity_term(
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = _base_ang_vel_term(noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = _joint_pos_rel_term(noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = _joint_vel_rel_term(noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = _last_action_term()

    @configclass
    class CriticCfg(G1LatentObservationCfg.CriticCfg):
        # Deleted: ASE-era terms no recipe's critic reads.
        projected_gravity = None
        joint_pos = None
        joint_vel = None
        body_ori = _body_ori_term()
        base_lin_vel = _base_lin_vel_term()
        base_ang_vel = _base_ang_vel_term()
        joint_pos_rel = _joint_pos_rel_term()
        joint_vel_rel = _joint_vel_rel_term()
        last_action = _last_action_term()

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

    def __post_init__(self):
        super().__post_init__()
        # SONIC's 10-step proprio histories, applied post-init so the
        # declarations stay history-free by default.
        for term in (
            self.policy.projected_gravity,
            self.policy.base_ang_vel,
            self.policy.joint_pos_rel,
            self.policy.joint_vel_rel,
            self.policy.last_action,
        ):
            term.history_length = 10
        for term in (
            self.critic.base_lin_vel,
            self.critic.base_ang_vel,
            self.critic.joint_pos_rel,
            self.critic.joint_vel_rel,
            self.critic.last_action,
        ):
            term.history_length = 10


@configclass
class G1V2ObservationCfg:
    """The single v2 observation surface: two groups, two command channels.

    Both groups declare every command component term; which ones survive, which
    channel each reads, and at what window is derived by the environment's
    ``CommandInterfaceCfg`` (``command_interface.apply_to_observations``) at
    construction. The declaration is therefore the complete surface and the
    resolution only ever narrows it.

    Policy: the actor's command (the latent command, or the explicit /
    chunk component terms) plus -- for a latent recipe -- the encoder's windowed
    reference view, plus the lean proprio set. Critic: its command view (which
    may span both channels) plus privileged state. ``reward_input`` stays the
    parked opt-in group for IPMD reward estimation.

    Single-frame design (2026-08-01): every proprio term is history-free.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: command terms + proprio."""

        latent_command = _command_component_term(LATENT_COMMAND_TERM_NAME)
        expert_motion = _command_component_term("joint_qpos_qvel")
        expert_anchor_pos_b = _command_component_term(
            "root_pos", noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        expert_anchor_ori_b = _command_component_term(
            "root_ori", noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        expert_motion_qpos = _command_component_term("joint_qpos")
        expert_ee_pos_b = _command_component_term("ee_pos")
        expert_ee_ori_b = _command_component_term("ee_ori")
        expert_keypoint_pos_b = _command_component_term("keypoint_pos")
        expert_keypoint_ori_b = _command_component_term("keypoint_ori")
        projected_gravity = _projected_gravity_term(
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = _base_ang_vel_term(noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = _joint_pos_rel_term(noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = _joint_vel_rel_term(noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = _last_action_term()

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: its command view + privileged state.

        Command terms here are noise-free and single-frame: the critic reads the
        reference channel directly (and the latent command when the actor
        publishes one), not the actor's windowed view.
        """

        latent_command = _command_component_term(LATENT_COMMAND_TERM_NAME)
        expert_motion = _command_component_term("joint_qpos_qvel", channel="reference")
        expert_anchor_pos_b = _command_component_term("root_pos", channel="reference")
        expert_anchor_ori_b = _command_component_term("root_ori", channel="reference")
        expert_motion_qpos = _command_component_term("joint_qpos", channel="reference")
        expert_ee_pos_b = _command_component_term("ee_pos", channel="reference")
        expert_ee_ori_b = _command_component_term("ee_ori", channel="reference")
        expert_keypoint_pos_b = _command_component_term(
            "keypoint_pos", channel="reference"
        )
        expert_keypoint_ori_b = _command_component_term(
            "keypoint_ori", channel="reference"
        )
        body_pos = _body_pos_term()
        body_ori = _body_ori_term()
        base_lin_vel = _base_lin_vel_term()
        base_ang_vel = _base_ang_vel_term()
        joint_pos_rel = _joint_pos_rel_term()
        joint_vel_rel = _joint_vel_rel_term()
        last_action = _last_action_term()

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    reward_input: RewardInputUnitCfg = RewardInputUnitCfg()


# ---------------------------------------------------------------------------
# Command-term pruning / whitelist constants and anchor tables.
# ---------------------------------------------------------------------------

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
        "expert_keypoint_pos_b",
        "expert_keypoint_ori_b",
    ),
    "expert_goal": ("expert_anchor_pos_b", "expert_anchor_ori_b"),
    "reward_input": ("expert_anchor_pos_b", "expert_anchor_ori_b"),
}

# NOTE: the v2 surface has no anchor-term table. `_resolve.stamp_anchor_body`
# finds them by asking the terms -- a term takes an anchor body exactly when it
# declares the parameter -- so adding a term to a group cannot leave it pointing
# at a stale anchor because someone forgot a list.
