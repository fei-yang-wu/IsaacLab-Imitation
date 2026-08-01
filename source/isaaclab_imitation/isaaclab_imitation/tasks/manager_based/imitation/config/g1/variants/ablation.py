# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict-protocol environment for controlled latent-representation ablations."""

from isaaclab.utils.configclass import configclass

from ..common.latent_env import ImitationG1LatentEnvCfg
from ..common.terminations import G1SonicTerminationsCfg
from ..common.tracking_env import (
    _apply_strict_recipe,
    _bind_lafan_track_from_dict,
)


@configclass
class ImitationG1LatentAblationEnvCfg(ImitationG1LatentEnvCfg):
    """Expose current + nine future frames on the strict LAFAN1 surface.

    The Strict recipe (strict SONIC terminations, pelvis anchor, [0, 200]
    reset starts, no curriculum -- ``_apply_strict_recipe``) on the latent
    surface, plus a ten-frame future observation window. The reconstruction
    learners publish a 64-value code plus a two-value within-chunk phase
    clock. Individual arms may override the command width (for example the
    phase-free CVAE row) without changing the environment protocol.
    """

    terminations = G1SonicTerminationsCfg()  # type: ignore
    curriculum = None
    latent_command_dim: int = 66

    def __post_init__(self):
        super().__post_init__()
        _apply_strict_recipe(self)
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 9
        self.command_hold_steps = 0
        self._sync_expert_window_observation_params()


_bind_lafan_track_from_dict(ImitationG1LatentAblationEnvCfg)
