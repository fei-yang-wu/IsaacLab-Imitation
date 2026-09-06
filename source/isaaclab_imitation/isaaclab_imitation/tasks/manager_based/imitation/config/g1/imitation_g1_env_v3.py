# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v3``: the `combo` recipe as the default (2026-09-06).

The same configurable environment as `-G1-v2` (one command interface, one
``resolve`` step, nothing derived in ``__post_init__``), with the
environment-side defaults moved to the recipe that
`2026-09-01-latent64-probe-10b` arm `combo` and the `combo-50b` chain pass as
overrides. Every field below is still a selection and every ``env.*`` override
still applies on top. `-G1-v2` keeps its exact kwargs, per the versioning
convention in ``config/g1/__init__.py``.

What moves against `-G1-v2`, and the override it replaces:

============================  ==========================================  =====
field                         v2 -> v3                                    was
============================  ==========================================  =====
rewards                       ``G1V3ComboRewardsCfg``                     four ``env.rewards.*.weight`` overrides
observations                  ``G1V3ObservationCfg`` (actor history 10)   five ``history_length=10`` overrides
actor command width           258 -> 66 (64-D code + sin/cos phase)       ``command_interface.actor.dim=66``
encoder view                  ``single`` (current frame)                  ``command_interface.encoder=single``
reset selection               sonic, uniform share 0.8 -> 0.2 over 4B     ``selection=sonic`` + three ratio fields
termination curriculum        on, 5M -> 30M frames                        three ``termination_curriculum*`` overrides
reference prefetch            ``next``                                    ``data.reference_prefetch_mode=next``
Newton constraint budget      ``njmax=320``                               ``sim.physics.solver_cfg.njmax=320``
============================  ==========================================  =====

The reset ramp is the single-segment form of the `combo-50b` schedule (0.8 ->
0.5 in segment 1, 0.5 -> 0.2 in segment 2, pinned 0.2 after): the ramp counter
restarts every cluster segment, so a chained run still passes the per-segment
stages explicitly. The agent side of the recipe is
``G1ImitationComboRLOptIPMDConfig`` (``rlopt_ipmd_cfg_entry_point`` of this
task). The skill encoder is an artifact, not a default: pass
``agent.ipmd.hl_skill_checkpoint_path`` to the past-5 affine merged-head
checkpoint (``p5_affine``, `2026-08-30-past-chunk-affine-64d`), and training
refuses to start without it.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ...command_interface import (
    ActorCommandPreset,
    CommandInterfaceCfg,
    EncoderViewPreset,
    ReferenceChannelCfg,
    ReferenceSelectionPreset,
)
from ...mdp.commands.actor import LatentCommandCfg
from ...mdp.commands.reference import ReferenceSelectionCfg
from .common.constants import (
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from .common.observations import G1V3ObservationCfg
from .common.presets import G1V3PhysicsCfg
from .common.rewards import G1V3ComboRewardsCfg
from .imitation_g1_env_v2 import ImitationG1V2EnvCfg

# `combo-50b`'s reset mix, in one segment: sonic's joint (rank, frame) draw,
# the uniform share ramped linearly from 80% to 20% over the first 4B frames,
# so the failure-weighted share grows 20% -> 80% and holds.
COMBO_RESET_SELECTION = ReferenceSelectionCfg(
    schedule="random",
    start_mode="auto",
    random_step_min=0,
    random_step_max=0,
    full_trajectory=True,
    adaptive_uniform_ratio=0.8,
    adaptive_uniform_ratio_final=0.2,
    adaptive_ratio_ramp_frames=4_000_000_000,
    adaptive_failure_rate_max_over_mean=200.0,
)

COMBO_ACTOR_COMMAND_DIM = 66
"""64-D skill code plus the sin/cos phase pair (constant at hold 1, kept)."""


@configclass
class G1V3ReferenceSelectionPreset(ReferenceSelectionPreset):
    """v2's reset-selection alternatives with the `combo` ramp as the default.

    ``combo`` names the same alternative explicitly, so a job that switches to
    ``sonic`` or ``random80_adaptive20`` for a comparison row can switch back
    by name.
    """

    default: ReferenceSelectionCfg = COMBO_RESET_SELECTION
    combo: ReferenceSelectionCfg = COMBO_RESET_SELECTION


@configclass
class G1V3ActorCommandPreset(ActorCommandPreset):
    """v2's actor alternatives with the 66-wide latent command as the default."""

    default: LatentCommandCfg = LatentCommandCfg(dim=COMBO_ACTOR_COMMAND_DIM)
    latent: LatentCommandCfg = LatentCommandCfg(dim=COMBO_ACTOR_COMMAND_DIM)


@configclass
class ImitationG1V3EnvCfg(ImitationG1V2EnvCfg):
    """`-G1-v3`: the v2 environment with the `combo` recipe as its defaults."""

    observations = G1V3ObservationCfg()  # type: ignore
    rewards = G1V3ComboRewardsCfg()  # type: ignore
    command_interface: CommandInterfaceCfg = CommandInterfaceCfg(
        reference=ReferenceChannelCfg(
            anchor_body_name="pelvis",
            joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
            mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
            ee_body_names=G1_EE_BODY_NAMES.copy(),
            keypoint_body_names=G1_KEYPOINT5_BODY_NAMES.copy(),
            selection=G1V3ReferenceSelectionPreset(),
        ),
        actor=G1V3ActorCommandPreset(),
        encoder=EncoderViewPreset(),
        critic_channels=("actor", "reference"),
    )
    enable_termination_curriculum: bool = True
    termination_curriculum_start_frames: int | None = 5_000_000
    termination_curriculum_end_frames: int | None = 30_000_000

    def __post_init__(self):
        super().__post_init__()
        self.data.reference_prefetch_mode = "next"
        self.sim.physics = G1V3PhysicsCfg()
