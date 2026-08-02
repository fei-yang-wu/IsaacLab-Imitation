# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VQ-VAE surface on the single v2 env (rebase 2026-08-02).

The encoder consumes the windowed policy command terms
(the encoder view of the reference channel); a past window of 8 widens it to a
causal 9-frame context.
"""

from isaaclab.utils.configclass import configclass

from ..imitation_g1_env_v2 import ImitationG1V2EnvCfg


@configclass
class ImitationG1VQVAESurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Single v2 env with a causal 9-step command window for the VQ-VAE encoder.

    Differs from the default only in the size of the command window the
    in-loop VQ-VAE encoder consumes: 8 past frames plus the current frame
    emitted by the trajectory manager.
    """

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.encoder.past_steps = 8
        self.command_interface.encoder.future_steps = 0
        # The env-construction resolution parameterizes the policy command
        # terms with this window (see ImitationG1V2EnvCfg.resolve_late_overrides).


__all__ = ["ImitationG1VQVAESurfaceEnvCfg"]
