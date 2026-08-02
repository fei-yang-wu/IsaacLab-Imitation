# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit and chunk actor surfaces on the single v2 env.

The environment is unchanged; only ``command_interface.actor`` differs, which is
the whole point of the declared interface. Both surfaces drop the encoder view
(nothing encodes a reference the actor already sees in full).

Selecting a command space is selecting components, on the command line as well::

    env.command_interface.actor.components='[joint_qpos,root_pos,root_ori]'

so a new explicit ablation needs no new config class.
"""

from isaaclab.utils.configclass import configclass

from ....command_interface import ChunkCommandCfg, ExplicitCommandCfg
from ..imitation_g1_env_v2 import ImitationG1V2EnvCfg


@configclass
class ImitationG1ExplicitSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """The oracle / direct explicit tracker row.

    The actor reads the reference channel directly, single-frame by default
    (the contract the explicit trackers are trained on). The critic mirrors the
    actor's components, so both are judged on the same command.
    """

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor = ExplicitCommandCfg()
        self.command_interface.encoder = None
        self.command_interface.critic_channels = ("reference",)


@configclass
class ImitationG1ChunkSurfaceEnvCfg(ImitationG1ExplicitSurfaceEnvCfg):
    """The planner-packet row: ten frames published per 5 Hz hold window.

    ``source="reference"`` is the oracle self-fill used to train the streamed
    interface and to certify it against the direct row; a real planner run sets
    ``env.command_interface.actor.source=external`` and publishes through the
    term.
    """

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor = ChunkCommandCfg(
            source="reference", horizon=10, hold_steps=10
        )


__all__ = [
    "ImitationG1ChunkSurfaceEnvCfg",
    "ImitationG1ExplicitSurfaceEnvCfg",
]
