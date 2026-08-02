# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v2``: the thin default G1 tracking environment (flagship).

The v2 default is the THINNEST surface, rebuilt from profiling the v1
superset: exactly the observation groups the latent recipe reads (policy +
critic only; no expert_window / expert_goal / expert_state / reward_input
groups) and ONE command term named ``command``
(:class:`~...mdp.commands.ReferenceCommandCfg`) that serves the
agent-published latent command, owns the reset-start samplers, and owns the
tracking metrics. The environment runs a single-compute step (one
observation compute per control step, at the returned next-reference
frame), so v2 is deliberately NOT bit-equivalent to the legacy env -- the
discarded mid-step compute also drew observation noise, so v2 has its own
fresh stochastic stream. Rewards, terminations, events, and actions are the
same frozen SONIC recipe components as v1.

The full surface (explicit command terms, expert_window / expert_goal
groups, the motion / skill / chunk command terms) is preserved as
:class:`ImitationG1FullSurfaceEnvCfg`, the base for the explicit and
reconstruction variants; the task registration points at the flagship
``ImitationG1EnvCfg`` (the lean default) since 2026-08-01.
"""

from isaaclab.utils.configclass import configclass

from ... import mdp
from ...mdp.commands import (
    HeldChunkCommandCfg,
    MotionCommandCfg,
    ReferenceCommandCfg,
    SkillCommandCfg,
)
from .common.actions import G1SonicActionsCfg
from .common.constants import G1_29DOF_ISAACLAB_JOINT_NAMES, G1_TRACKED_BODY_NAMES
from .common.events import G1SonicEventCfg
from .common.observations_latent import (
    G1FullSurfaceObservationCfg,
    G1LeanLatentObservationCfg,
    _LATENT_ANCHOR_TERM_NAMES_BY_GROUP,
)
from .common.presets import G1SonicRobotCfg
from .common.rewards import G1SonicRewardsCfg
from .common.terminations import G1SonicTerminationsCfg
from .common.tracking_env import (
    ImitationG1BaseTrackingEnvCfg,
    _apply_pelvis_protocol,
    _bind_lafan_track_from_dict,
)
from .imitation_g1_env_v1 import ImitationG1EnvV1Cfg

# The `*_chunk_current_slot` policy command modes (the same set the base cfg's
# `__post_init__` validation accepts next to "reference"): only under one of
# these does the env stream the actor's command from a held packet, so only
# then does the `chunk` command term exist.
_POLICY_CHUNK_COMMAND_MODES = (
    "explicit_chunk_current_slot",
    "full_body_chunk_current_slot",
    "ee_chunk_current_slot",
)

# The command-backed observation terms that the full v2 surface rebinds onto
# CommandManager terms (same values, one producer): the baseline explicit
# trio moves to the ``motion`` term and the agent-latent term moves to the
# ``skill`` term. Only these; the ``policy_*`` chunk-adapter terms keep their
# env funcs.
_COMMAND_MANAGER_BACKED_TERM_FUNCS = {
    "expert_motion": mdp.motion_command_joint,
    "expert_anchor_pos_b": mdp.motion_command_anchor_pos_b,
    "expert_anchor_ori_b": mdp.motion_command_anchor_ori_b,
    "latent_command": mdp.skill_command,
}


@configclass
class G1LeanCommandsCfg:
    """The lean v2 default carries one command term, named ``command``."""

    command: ReferenceCommandCfg = ReferenceCommandCfg(
        anchor_body_name="pelvis",
        mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
        # The term owns reference reset-start sampling (same sampler semantics
        # and cfg knobs as the full surface's `motion` term and v0/v1's
        # env-inline path).
        owns_reset=True,
    )


@configclass
class G1MotionCommandsCfg:
    """Command terms for the full v2 CommandManager surface."""

    motion: MotionCommandCfg = MotionCommandCfg(
        anchor_body_name="pelvis",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
        mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
        owns_reset=True,
    )

    # Placeholder width: the env cfg wires `latent_command_dim` from its own
    # field in `__post_init__` (and re-syncs it in the construction-time
    # refresh so plain-setattr overrides of `env.latent_command_dim` land).
    skill: SkillCommandCfg = SkillCommandCfg(latent_command_dim=258)

    # Held explicit-chunk term. None on the default latent task: the env cfg
    # instantiates it in `_sync_chunk_command_cfg` only when
    # `policy_command_mode` is a `*_chunk_current_slot` adapter (the
    # CommandManager skips None entries).
    chunk: HeldChunkCommandCfg | None = None


@configclass
class ImitationG1EnvCfg(ImitationG1BaseTrackingEnvCfg):
    """Thin v2 default: lean latent observations + one ``command`` term.

    Observation surface: policy (agent-published latent command + SONIC
    proprio histories) and critic (latent command + single-frame expert
    command + privileged state) only. No windowed / goal / state /
    reward-input groups -- nothing in the latent recipe reads them, and the
    offline skill-encoder sampler consumes ``current_expert_macro_transition_batch``
    (an env API), not the observation groups.
    """

    # -- components (shared SONIC blocks from common) --
    actions = G1SonicActionsCfg()
    observations = G1LeanLatentObservationCfg()
    rewards = G1SonicRewardsCfg()  # type: ignore
    terminations = G1SonicTerminationsCfg()  # type: ignore
    events = G1SonicEventCfg()
    curriculum = None

    # The single lean command term (latent buffer + reset ownership + metrics).
    # pyrefly: ignore[bad-override-mutable-attribute]  # configclass override idiom
    commands: G1LeanCommandsCfg = G1LeanCommandsCfg()

    # -- skill-conditioned command configuration --
    command_mode: str = "latent"
    latent_command_dim: int = 258
    # Future goal steps; unused by the lean surface (no expert_goal group) but
    # inherited by the full surface's goal sync.
    latent_goal_steps: int = 1
    # The lean surface has no expert_goal / reward_input groups; the parked
    # reward-estimation stack stays off.
    enable_reward_input_observations: bool = False

    def _anchor_term_names_by_group(self) -> dict[str, tuple[str, ...]]:
        # Only the critic carries anchor-relative terms on the lean surface.
        return {"critic": ("expert_anchor_pos_b", "expert_anchor_ori_b")}

    def _critic_prunable_command_term_names(self) -> tuple[str, ...]:
        # No explicit command superset on the lean surface.
        return ()

    def _sync_expert_window_observation_params(self) -> None:
        # The lean surface has no expert_window group; the offline skill
        # encoder samples via the env API, not this observation group.
        return

    def _sync_expert_goal_observation_params(self) -> None:
        # The lean surface has no expert_goal group.
        return

    def __post_init__(self):
        super().__post_init__()

        # Single-frame skill command over a live sliding reference window.
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 0
        self._sync_command_cfg()

        # SONIC robot asset (actuator contract matching G1SonicActionsCfg).
        robot_preset = G1SonicRobotCfg()
        for variant in (
            robot_preset.default,
            robot_preset.physx,
            robot_preset.newton_mjwarp,
        ):
            variant.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot_preset  # type: ignore

        # Pelvis expert anchor plus the legacy reset distribution: starts in
        # [0, 200], no full-trajectory adaptive-failure sampling,
        # failure_rate_max_over_mean=50.
        _apply_pelvis_protocol(self, failure_rate_max_over_mean=50.0)
        self._sync_command_cfg()

    def _sync_command_cfg(self) -> None:
        """Wire the lean ``command`` term's width and anchor from the env fields.

        The ReferenceCommand serves the env's ``_agent_latent_command`` buffer
        (width ``latent_command_dim``) and computes its metrics against the
        env's expert anchor; both must match or construction fails loudly.
        Idempotent. Inert on surfaces whose command set has no ``command``
        term (e.g. the full surface with motion/skill/chunk).
        """
        command = getattr(self.commands, "command", None)
        if command is None:
            return
        command.latent_command_dim = int(self.latent_command_dim)
        command.anchor_body_name = self.expert_anchor_body_name

    def _refresh_command_observation_terms(self) -> None:
        # The lean surface has nothing to prune/restore; keep the command
        # width/anchor in sync for late plain-setattr overrides
        # (`env.latent_command_dim=...` / `env.expert_anchor_body_name=...`).
        super()._refresh_command_observation_terms()
        self._sync_command_cfg()


@configclass
class ImitationG1FullSurfaceEnvCfg(ImitationG1EnvCfg):
    """Full v2 surface: the v1 observation layout plus the native command terms.

    The pre-thin-default v2 configuration: v1's complete observation groups
    (expert_state / expert_window / expert_goal / reward_input plus the
    explicit-command superset in policy/critic), the three command terms
    (``motion`` / ``skill`` / ``chunk``), and the v1 env-backed metric and
    reward-input behavior. Base for the explicit and reconstruction variants.
    """

    observations = G1FullSurfaceObservationCfg()  # type: ignore

    # pyrefly: ignore[bad-override-mutable-attribute]  # configclass override idiom
    commands: G1MotionCommandsCfg = G1MotionCommandsCfg()

    def _anchor_term_names_by_group(self) -> dict[str, tuple[str, ...]]:
        # The v1 latent anchor table (policy/critic body-pose + anchor terms,
        # expert_state/window/goal anchor terms).
        return _LATENT_ANCHOR_TERM_NAMES_BY_GROUP

    def _critic_prunable_command_term_names(self) -> tuple[str, ...]:
        # The supplemental explicit terms the latent critic gained for
        # explicit command mode (v1 behavior).
        return (
            "expert_motion_qpos",
            "expert_ee_pos_b",
            "expert_ee_ori_b",
            "expert_keypoint_pos_b",
            "expert_keypoint_ori_b",
        )

    def _sync_expert_window_observation_params(self) -> None:
        # The full surface HAS the expert_window group; restore the base sync
        # (the lean default no-ops it).
        ImitationG1BaseTrackingEnvCfg._sync_expert_window_observation_params(self)

    def _sync_expert_goal_observation_params(self) -> None:
        ImitationG1EnvV1Cfg._sync_expert_goal_observation_params(self)

    def __post_init__(self):
        super().__post_init__()
        # The lean post-init no-ops the goal sync; the full surface has the
        # expert_goal group, so restore the v1 goal_steps wiring.
        self._sync_expert_goal_observation_params()
        # Keep the command term's anchor in lockstep with the env protocol's
        # expert anchor (`_apply_pelvis_protocol` sets "pelvis").
        self.commands.motion.anchor_body_name = self.expert_anchor_body_name
        self._sync_skill_command_cfg()
        self._sync_chunk_command_cfg()
        self._rebind_command_manager_backed_terms()

    def _sync_skill_command_cfg(self) -> None:
        """Wire the ``skill`` term's width from the env's latent command dim.

        The SkillCommand adapter serves the env's ``_agent_latent_command``
        buffer, whose width the env derives from ``cfg.latent_command_dim``;
        the term cfg must carry the same value or SkillCommand fails loudly at
        construction. Idempotent.
        """
        self.commands.skill.latent_command_dim = int(self.latent_command_dim)

    def _sync_chunk_command_cfg(self) -> None:
        """Instantiate/prune the ``chunk`` term from ``policy_command_mode``.

        Only a ``*_chunk_current_slot`` mode streams the actor's command from
        the env's held window, so only then does the adapter term exist; the
        default latent task keeps ``chunk=None`` (the CommandManager skips
        None entries). When present, its knobs are wired in lockstep with the
        env fields the held-window machinery reads: ``hold_steps`` from
        ``command_hold_steps`` (HeldChunkCommand fails loudly on a mismatch),
        the pinned 29-DoF joint order of the chunk-mode
        ``policy_expert_motion_command`` observation term, and the expert
        anchor body. Normalizes the mode string the same way the base
        ``__post_init__`` does because plain-setattr overrides can arrive
        un-normalized before the construction-time refresh. Idempotent.
        """
        mode = str(self.policy_command_mode).strip().lower().replace("-", "_")
        if mode not in _POLICY_CHUNK_COMMAND_MODES:
            self.commands.chunk = None
            return
        if self.commands.chunk is None:
            self.commands.chunk = HeldChunkCommandCfg()
        self.commands.chunk.hold_steps = int(self.command_hold_steps)
        self.commands.chunk.joint_names = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
        self.commands.chunk.anchor_body_name = self.expert_anchor_body_name

    def _rebind_command_manager_backed_terms(self) -> None:
        """Serve the command-backed observation terms from CommandManager terms.

        Same values as the v1 env-backed funcs (baseline explicit trio ->
        ``motion``, ``latent_command`` -> ``skill``), but with the
        CommandManager term as the single producer. The new funcs take no
        params, so the old ``asset_cfg``/``anchor_body_name`` params are
        dropped; each term's noise (and everything else) is preserved.
        Idempotent. Terms pruned to None (e.g. ``latent_command`` under
        ``command_mode=explicit``) are skipped.
        """
        for group_name in ("policy", "critic"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            for term_name, func in _COMMAND_MANAGER_BACKED_TERM_FUNCS.items():
                term = getattr(group, term_name, None)
                if term is None:
                    continue
                term.func = func
                term.params = {}

    def _refresh_command_observation_terms(self) -> None:
        # The base refresh restores pruned terms from a fresh `type(group)()`,
        # which carries the declaration-time (v1 env-backed) func bindings, so
        # any restored command-backed term would silently stop reading its
        # command term. Re-apply the v2 rebinding (and the skill-width sync,
        # since `env.latent_command_dim` overrides can arrive as plain setattr
        # after `__post_init__`) after every refresh; every step is
        # idempotent, so refresh stays a fixed point.
        super()._refresh_command_observation_terms()
        self._sync_skill_command_cfg()
        self._sync_chunk_command_cfg()
        self._rebind_command_manager_backed_terms()


_bind_lafan_track_from_dict(ImitationG1EnvCfg, ImitationG1FullSurfaceEnvCfg)

# Back-compat alias: configs recorded against the pre-flip
# `imitation_g1_env_v2:ImitationG1EnvV2Cfg` entry point keep resolving.
ImitationG1EnvV2Cfg = ImitationG1EnvCfg

__all__ = [
    "G1LeanCommandsCfg",
    "G1MotionCommandsCfg",
    "ImitationG1EnvCfg",
    "ImitationG1EnvV2Cfg",
    "ImitationG1FullSurfaceEnvCfg",
]
