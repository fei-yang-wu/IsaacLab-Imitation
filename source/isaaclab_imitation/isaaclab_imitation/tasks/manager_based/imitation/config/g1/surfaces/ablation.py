# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruction-ablation surface on the single v2 env (rebase 2026-08-02).

Config-only surface: its historical agent (``rlopt_ipmd_latent_ablation_cfg``)
was removed in the 2026-08-01 consolidation -- the reconstruction arms now
run on the generic latent agent with
``latent_learning.method=patch_autoencoder`` overrides (see the command-matrix
audit rows ``ae``/``vae``). Reach it via the config class or
``env_cfg_entry_point`` overrides on ``Isaac-Imitation-G1-v2``.
"""

from isaaclab.utils.configclass import configclass

from ..common.terminations import G1SonicTerminationsCfg
from ..imitation_g1_env_v2 import ImitationG1V2EnvCfg


@configclass
class ImitationG1AblationSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Single v2 env exposing a ten-frame command window on the strict protocol.

    The v2 defaults ARE the strict protocol (pelvis anchor, [0, 200] reset
    starts, no full-trajectory adaptive resets, no curriculum), so the delta
    is the strict SONIC termination set plus the ten-frame future command
    window and the 66-D (64 code + 2 phase) reconstruction command.
    """

    terminations = G1SonicTerminationsCfg()  # type: ignore
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor.dim = 66
        self.command_interface.encoder.past_steps = 0
        self.command_interface.encoder.future_steps = 9


__all__ = ["ImitationG1AblationSurfaceEnvCfg"]
