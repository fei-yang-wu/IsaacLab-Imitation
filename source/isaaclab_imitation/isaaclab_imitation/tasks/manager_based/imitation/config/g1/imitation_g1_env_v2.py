# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v2``: work-in-progress CommandManager-based redesign.

First increment of the v2 overhaul: identical to ``Isaac-Imitation-G1-v1``
(:class:`~.imitation_g1_env_v1.ImitationG1EnvCfg`) except that the motion
tracking command is additionally exposed through Isaac Lab's native
CommandManager as the ``motion`` term (``mdp.MotionCommandCfg``). The term is
an adapter over the existing ImitationRLEnv machinery: it publishes the
67-D explicit command tensor via ``command_manager.get_command("motion")``
and owns the tracking metrics, so the CommandManager logs
``Metrics/motion/...`` natively (the beyondmimic/SONIC idiom). Behavior is
otherwise byte-identical to v1.

The flagship class/task name moves here when v2 supersedes v1 as the default
(see the versioning convention in ``config/g1/__init__.py``).
"""

from isaaclab.utils.configclass import configclass

from ... import mdp
from ...mdp.commands import MotionCommandCfg
from .common.constants import G1_29DOF_ISAACLAB_JOINT_NAMES, G1_TRACKED_BODY_NAMES
from .common.tracking_env import _bind_lafan_track_from_dict
from .imitation_g1_env_v1 import ImitationG1EnvCfg

# The plain reference-command observation terms that v2 rebinds onto the
# CommandManager-served ``motion`` term (same values, one producer). Only the
# baseline trio; the ``policy_*`` chunk-adapter terms keep their env funcs.
_MOTION_COMMAND_BACKED_TERM_FUNCS = {
    "expert_motion": mdp.motion_command_joint,
    "expert_anchor_pos_b": mdp.motion_command_anchor_pos_b,
    "expert_anchor_ori_b": mdp.motion_command_anchor_ori_b,
}


@configclass
class G1MotionCommandsCfg:
    """Command terms for the v2 CommandManager surface."""

    motion: MotionCommandCfg = MotionCommandCfg(
        anchor_body_name="pelvis",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
        mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
    )


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
        self._rebind_motion_command_backed_terms()

    def _rebind_motion_command_backed_terms(self) -> None:
        """Serve the baseline explicit trio from the ``motion`` command term.

        Same values as the v1 env-backed funcs, but with the CommandManager
        term as the single producer. The new funcs take no params, so the old
        ``asset_cfg``/``anchor_body_name`` params are dropped; each term's
        noise (and everything else) is preserved. Idempotent.
        """
        for group_name in ("policy", "critic"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            for term_name, func in _MOTION_COMMAND_BACKED_TERM_FUNCS.items():
                term = getattr(group, term_name, None)
                if term is None:
                    continue
                term.func = func
                term.params = {}

    def _refresh_command_observation_terms(self) -> None:
        # The base refresh restores pruned terms from a fresh `type(group)()`,
        # which carries the declaration-time (v1 env-backed) func bindings, so
        # any restored trio term would silently stop reading the command term.
        # Re-apply the v2 rebinding after every refresh; both steps are
        # idempotent, so refresh stays a fixed point.
        super()._refresh_command_observation_terms()
        self._rebind_motion_command_backed_terms()


_bind_lafan_track_from_dict(ImitationG1EnvV2Cfg)

__all__ = ["G1MotionCommandsCfg", "ImitationG1EnvV2Cfg"]
