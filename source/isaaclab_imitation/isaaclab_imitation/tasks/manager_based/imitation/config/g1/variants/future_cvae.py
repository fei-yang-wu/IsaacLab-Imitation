# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Future-window latent command variants (LafanTrack lineage)."""

from isaaclab.utils.configclass import configclass

from ..common.latent_env import ImitationG1LatentEnvCfg
from ..common.tracking_env import _bind_lafan_track_from_dict


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


_bind_lafan_track_from_dict(
    ImitationG1LatentFutureCVAEEnvCfg,
    ImitationG1LatentPerStepVQEnvCfg,
)
