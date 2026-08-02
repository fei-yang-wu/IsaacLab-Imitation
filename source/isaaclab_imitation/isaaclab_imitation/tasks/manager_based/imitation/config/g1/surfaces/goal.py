# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held future-goal surface on the single v2 env (rebase 2026-08-02).

The historical expert_goal observation group is gone; the held future goal
is the same command window machinery with ``latent_patch_future_steps=25``.
Config-only surface: its historical agent (bilinear skill commander) was
removed in the 2026-08-01 consolidation, so there is no registered task id;
reach it via the config class or ``env_cfg_entry_point`` overrides on
``Isaac-Imitation-G1-v2``.
"""

from isaaclab.utils.configclass import configclass

from ..imitation_g1_env_v2 import ImitationG1V2EnvCfg


@configclass
class ImitationG1GoalSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Single v2 env whose posterior observes a held 25-step future command.

    The 128-D command is held over a 25-frame future command window.
    """

    latent_command_dim: int = 128

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 25


__all__ = ["ImitationG1GoalSurfaceEnvCfg"]
