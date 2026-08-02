# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Future-window surfaces on the single v2 env (rebase 2026-08-02).

The encoders consume the windowed policy command terms
(``("policy", "expert_motion")`` etc.); ``latent_patch_future_steps=9``
widens the command window to the current plus nine future frames.
"""

from isaaclab.utils.configclass import configclass

from ..imitation_g1_env_v2 import ImitationG1V2EnvCfg


@configclass
class ImitationG1FutureCVAESurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Single v2 env exposing the current plus nine future command frames.

    The future-window CVAE encoder consumes a ten-frame command window from
    the policy group; the published 256-D command renews every control step
    (``command_hold_steps=0``).
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
