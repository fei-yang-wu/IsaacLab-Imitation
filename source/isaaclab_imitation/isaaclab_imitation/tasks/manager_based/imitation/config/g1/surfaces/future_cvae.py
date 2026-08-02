# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Future-window surfaces on the flat v2 full surface (migrated 2026-08-01).

Replaces the legacy ``variants.future_cvae`` classes (now deleted); their
task ids are re-registered on the v2 env as ``Isaac-Imitation-G1-CVAE-v0``
and ``Isaac-Imitation-G1-PerStepVQ-v0``.
"""

from isaaclab.utils.configclass import configclass

from ..imitation_g1_env_v2 import ImitationG1FullSurfaceEnvCfg


@configclass
class ImitationG1FutureCVAESurfaceEnvCfg(ImitationG1FullSurfaceEnvCfg):
    """Flat v2 full surface exposing the current plus nine future frames.

    The future-window CVAE encoder consumes a ten-frame future window from
    the expert_window group; the published 256-D command renews every control
    step (``command_hold_steps=0``).
    """

    latent_command_dim: int = 256

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 9
        self.command_hold_steps = 0


@configclass
class ImitationG1PerStepVQSurfaceEnvCfg(ImitationG1FutureCVAESurfaceEnvCfg):
    """Future-window surface for ten-token, per-control-step command packets."""

    latent_command_dim: int = 64


__all__ = [
    "ImitationG1FutureCVAESurfaceEnvCfg",
    "ImitationG1PerStepVQSurfaceEnvCfg",
]
