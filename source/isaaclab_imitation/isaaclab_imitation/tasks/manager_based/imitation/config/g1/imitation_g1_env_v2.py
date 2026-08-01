# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v2``: work-in-progress CommandManager-based redesign.

First increments of the v2 overhaul: identical to ``Isaac-Imitation-G1-v1``
(:class:`~.imitation_g1_env_v1.ImitationG1EnvCfg`) except that the motion
tracking command and the agent-published latent skill command are additionally
exposed through Isaac Lab's native CommandManager as the ``motion``
(``mdp.MotionCommandCfg``) and ``skill`` (``mdp.SkillCommandCfg``) terms. Both
are adapters over the existing ImitationRLEnv machinery: ``motion`` publishes
the 67-D explicit command tensor via ``command_manager.get_command("motion")``
and owns the tracking metrics, so the CommandManager logs
``Metrics/motion/...`` natively (the beyondmimic/SONIC idiom); ``skill``
serves the env's agent-latent buffer via
``command_manager.get_command("skill")`` and carries the published/hold
bookkeeping from ``mdp.PublishedCommandTerm``. Behavior is otherwise
byte-identical to v1.

The flagship class/task name moves here when v2 supersedes v1 as the default
(see the versioning convention in ``config/g1/__init__.py``).
"""

from isaaclab.utils.configclass import configclass

from ... import mdp
from ...mdp.commands import MotionCommandCfg, SkillCommandCfg
from .common.constants import G1_29DOF_ISAACLAB_JOINT_NAMES, G1_TRACKED_BODY_NAMES
from .common.tracking_env import _bind_lafan_track_from_dict
from .imitation_g1_env_v1 import ImitationG1EnvCfg

# The command-backed observation terms that v2 rebinds onto CommandManager
# terms (same values, one producer): the baseline explicit trio moves to the
# ``motion`` term and the agent-latent term moves to the ``skill`` term. Only
# these; the ``policy_*`` chunk-adapter terms keep their env funcs.
_COMMAND_MANAGER_BACKED_TERM_FUNCS = {
    "expert_motion": mdp.motion_command_joint,
    "expert_anchor_pos_b": mdp.motion_command_anchor_pos_b,
    "expert_anchor_ori_b": mdp.motion_command_anchor_ori_b,
    "latent_command": mdp.skill_command,
}


@configclass
class G1MotionCommandsCfg:
    """Command terms for the v2 CommandManager surface."""

    motion: MotionCommandCfg = MotionCommandCfg(
        anchor_body_name="pelvis",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
        mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
    )

    # Placeholder width: the env cfg wires `latent_command_dim` from its own
    # field in `__post_init__` (and re-syncs it in the construction-time
    # refresh so plain-setattr overrides of `env.latent_command_dim` land).
    skill: SkillCommandCfg = SkillCommandCfg(latent_command_dim=258)


@configclass
class ImitationG1EnvV2Cfg(ImitationG1EnvCfg):
    """v2 WIP: the v1 surface plus the native ``motion`` command term."""

    # pyrefly: ignore[bad-override-mutable-attribute]  # configclass override idiom
    commands: G1MotionCommandsCfg = G1MotionCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep the command term's anchor in lockstep with the env protocol's
        # expert anchor (v1's `_apply_pelvis_protocol` sets "pelvis").
        self.commands.motion.anchor_body_name = self.expert_anchor_body_name
        self._sync_skill_command_cfg()
        self._rebind_command_manager_backed_terms()

    def _sync_skill_command_cfg(self) -> None:
        """Wire the ``skill`` term's width from the env's latent command dim.

        The SkillCommand adapter serves the env's ``_agent_latent_command``
        buffer, whose width the env derives from ``cfg.latent_command_dim``;
        the term cfg must carry the same value or SkillCommand fails loudly at
        construction. Idempotent.
        """
        self.commands.skill.latent_command_dim = int(self.latent_command_dim)

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
        self._rebind_command_manager_backed_terms()


_bind_lafan_track_from_dict(ImitationG1EnvV2Cfg)

__all__ = ["G1MotionCommandsCfg", "ImitationG1EnvV2Cfg"]
