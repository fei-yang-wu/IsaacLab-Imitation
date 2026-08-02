# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held future-goal surface on the flat v2 full surface (migrated 2026-08-01).

Replaces the legacy ``variants.goal.ImitationG1LatentGoalEnvCfg`` (now
deleted). Config-only surface: its historical agent (bilinear skill
commander) was removed in the 2026-08-01 consolidation, so there is no
registered task id; reach it via the config class or
``env_cfg_entry_point`` overrides on ``Isaac-Imitation-G1-v2``.
"""

from isaaclab.utils.configclass import configclass

from ..imitation_g1_env_v2 import ImitationG1FullSurfaceEnvCfg


@configclass
class ImitationG1GoalSurfaceEnvCfg(ImitationG1FullSurfaceEnvCfg):
    """Flat v2 full surface whose posterior observes a held future goal state.

    The expert_goal group exposes 25 steps of future goal state for
    hierarchical skills; the published 128-D command is held over the goal
    horizon.
    """

    latent_command_dim: int = 128
    latent_goal_steps: int = 25


__all__ = ["ImitationG1GoalSurfaceEnvCfg"]
