# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held future-goal latent command variant (LafanTrack lineage)."""

from isaaclab.utils.configclass import configclass

from ..common.latent_env import ImitationG1LatentEnvCfg
from ..common.tracking_env import _bind_lafan_track_from_dict


@configclass
class ImitationG1LatentGoalEnvCfg(ImitationG1LatentEnvCfg):
    """Latent G1 env whose posterior command observes a held future goal state."""

    latent_command_dim: int = 128
    latent_goal_steps: int = 25


_bind_lafan_track_from_dict(ImitationG1LatentGoalEnvCfg)
