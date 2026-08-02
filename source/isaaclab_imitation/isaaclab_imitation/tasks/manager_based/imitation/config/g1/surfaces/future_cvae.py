# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Future-window surfaces on the single v2 env (rebase 2026-08-02).

The encoders consume the windowed policy command terms
(the encoder view of the reference channel); a future window of 9
widens the command window to the current plus nine future frames.
"""

from isaaclab.utils.configclass import configclass

from ..imitation_g1_env_v2 import ImitationG1V2EnvCfg


@configclass
class ImitationG1FutureCVAESurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Single v2 env exposing the current plus nine future command frames.

    The future-window CVAE encoder consumes a ten-frame command window from
    the policy group; the published 256-D command renews every control step
    (the command window advances with the live reference).
    """

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor.dim = 256
        self.command_interface.encoder.past_steps = 0
        self.command_interface.encoder.future_steps = 9


@configclass
class ImitationG1PerStepVQSurfaceEnvCfg(ImitationG1FutureCVAESurfaceEnvCfg):
    """Future-window surface for ten-token, per-control-step command packets."""

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor.dim = 64


__all__ = [
    "ImitationG1FutureCVAESurfaceEnvCfg",
    "ImitationG1PerStepVQSurfaceEnvCfg",
]
