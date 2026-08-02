# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VQ-VAE surface on the flat v2 full surface (migrated 2026-08-01).

Replaces the legacy ``variants.vqvae.ImitationG1LatentVQVAEEnvCfg`` (now
deleted); its task id is re-registered on the v2 env as
``Isaac-Imitation-G1-VQVAE-v0``.
"""

from isaaclab.utils.configclass import configclass

from ..imitation_g1_env_v2 import ImitationG1FullSurfaceEnvCfg


@configclass
class ImitationG1VQVAESurfaceEnvCfg(ImitationG1FullSurfaceEnvCfg):
    """Flat v2 full surface exposing a causal 9-step expert window.

    Differs from the full surface only in the size of the expert observation
    window the in-loop VQ-VAE encoder consumes: 8 past frames plus the
    current frame emitted by the trajectory manager. The rest of the latent
    pipeline (latent_command obs term, observation groups, terminations,
    events) is inherited unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 8
        self.latent_patch_future_steps = 0
        # The env-construction resolution syncs the expert_window terms to
        # this window (see ImitationG1EnvCfg.resolve_late_overrides).


__all__ = ["ImitationG1VQVAESurfaceEnvCfg"]
