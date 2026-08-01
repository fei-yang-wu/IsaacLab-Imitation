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

from ...mdp.commands import MotionCommandCfg
from .common.constants import G1_29DOF_ISAACLAB_JOINT_NAMES, G1_TRACKED_BODY_NAMES
from .common.tracking_env import _bind_lafan_track_from_dict
from .imitation_g1_env_v1 import ImitationG1EnvCfg


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


_bind_lafan_track_from_dict(ImitationG1EnvV2Cfg)

__all__ = ["G1MotionCommandsCfg", "ImitationG1EnvV2Cfg"]
