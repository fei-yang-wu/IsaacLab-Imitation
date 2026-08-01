# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-conditioned env variant tuned for the IPMD VQ-VAE skill codebook.

Differs from :class:`ImitationG1LatentEnvCfg` only in the size of the
expert observation window the encoder consumes; the rest of the latent
pipeline (latent_command obs term, observation groups, terminations,
events) is inherited unchanged.
"""

from isaaclab.utils.configclass import configclass

from ..common.latent_env import ImitationG1LatentEnvCfg
from ..common.tracking_env import _bind_lafan_track_from_dict


@configclass
class ImitationG1LatentVQVAEEnvCfg(ImitationG1LatentEnvCfg):
    """Latent-conditioned G1 env exposing a temporally-extended expert window."""

    def __post_init__(self):
        super().__post_init__()
        # Expert encoder consumes a causal 9-step window: 8 past frames plus
        # the current frame emitted by the trajectory manager.
        self.latent_patch_past_steps = 8
        self.latent_patch_future_steps = 0
        self._sync_expert_window_observation_params()


_bind_lafan_track_from_dict(ImitationG1LatentVQVAEEnvCfg)
