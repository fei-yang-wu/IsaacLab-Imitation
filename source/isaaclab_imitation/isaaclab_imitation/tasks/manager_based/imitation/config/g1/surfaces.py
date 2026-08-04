# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registered G1 surfaces: one configuration selection each, on the single v2 env.

A *surface* is not a kind of environment. It is a point in the v2 environment's
configuration space, and every class here says so: each selects a command
channel, an encoder window, and a reset profile, and holds no logic of its own.
Every one of them is reachable from the command line without a class at all::

    scripts/rlopt/train.py --task Isaac-Imitation-G1-v2 \\
        env.command_interface.actor=explicit \\
        env.command_interface.encoder=causal9

These classes exist because a gym registration names a config class, and a task
id is the stable, citable identity of a protocol (see the versioning convention
in ``__init__.py``). They are an id's *binding*, not its implementation --
which is why they are allowed to be this thin, and why a new ablation needs a
CLI override rather than a new file.

The one axis here that is genuinely not a command selection is the observation
profile: :class:`ImitationG1SonicSurfaceEnvCfg` applies SONIC's 10-step
proprioceptive histories, which change the actor's input *layout* rather than
its command.
"""

from isaaclab.utils.configclass import configclass

from ...command_interface import (
    ChunkCommandCfg,
    EncoderViewCfg,
    ExplicitCommandCfg,
    LatentCommandCfg,
    ReferenceSelectionPreset,
)
from .common.terminations import G1SonicTerminationCurriculumCfg
from .common.rewards import G1SonicRewardsCfg
from .imitation_g1_env_v2 import ImitationG1V2EnvCfg

_SONIC_HISTORY_TERMS_POLICY = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)
_SONIC_HISTORY_TERMS_CRITIC = (
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


# ---------------------------------------------------------------------------
# Explicit actor channels: the actor reads the reference command directly.
# ---------------------------------------------------------------------------


@configclass
class ImitationG1ExplicitSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """The oracle / direct explicit tracker row.

    Single-frame by default (the contract the explicit trackers are trained
    on), with no encoder: nothing encodes a reference the actor already sees in
    full. The critic mirrors the actor's components, so both are judged on the
    same command. Select a command space by its components::

        env.command_interface.actor.components='[joint_qpos,root_pos,root_ori]'
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


# ---------------------------------------------------------------------------
# Latent actor channels: what differs is the published width and the window
# the skill encoder reads. The environment does not know or care which encoder
# consumes the view -- that is the agent's config.
# ---------------------------------------------------------------------------


@configclass
class ImitationG1VQVAESurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Causal 9-frame command window (8 past frames plus the current one)."""

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.encoder = EncoderViewCfg(past_steps=8, future_steps=0)


@configclass
class ImitationG1FutureCVAESurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Current plus nine future command frames, published as a 256-D command."""

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor = LatentCommandCfg(dim=256)
        self.command_interface.encoder = EncoderViewCfg(past_steps=0, future_steps=9)


@configclass
class ImitationG1PerStepVQSurfaceEnvCfg(ImitationG1FutureCVAESurfaceEnvCfg):
    """Same window, published as ten 64-D per-control-step token packets."""

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor = LatentCommandCfg(dim=64)


@configclass
class ImitationG1GoalSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """A 128-D command held over a 25-frame future window (hierarchical skills).

    Config-only surface: its historical agent (the bilinear skill commander)
    was removed in the 2026-08-01 consolidation, so there is no registered task
    id for it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor = LatentCommandCfg(dim=128)
        self.command_interface.encoder = EncoderViewCfg(past_steps=0, future_steps=25)


@configclass
class ImitationG1AblationSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """Reconstruction-ablation protocol: a 66-D command (64 code + 2 phase).

    Config-only surface: the reconstruction arms run on the generic latent
    agent with ``latent_learning.method=patch_autoencoder`` overrides.
    """

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor = LatentCommandCfg(dim=66)
        self.command_interface.encoder = EncoderViewCfg(past_steps=0, future_steps=9)


# ---------------------------------------------------------------------------
# SONIC release recipe.
# ---------------------------------------------------------------------------


@configclass
class ImitationG1SonicSurfaceEnvCfg(ImitationG1V2EnvCfg):
    """The public SONIC release recipe.

    Three deltas over the v2 default: SONIC's full-trajectory adaptive-failure
    reset sampler, the termination-threshold anneal, and SONIC's 10-step
    proprioceptive histories. Thresholds anneal to the strict release values
    over the curriculum window; disable with ``env.curriculum=null`` for
    strict-from-scratch release fidelity.

    The rewards and the macro-state frame are PINNED to the release values
    rather than inherited. v2's defaults moved on 2026-08-04 -- tuned tracking
    weights and the ``root_qpos`` encoder frame -- and this surface exists to be
    the published recipe, so inheriting either would quietly make it something
    other than SONIC. Other v2 subclasses deliberately do track the default.
    """

    curriculum = G1SonicTerminationCurriculumCfg()
    rewards = G1SonicRewardsCfg()  # type: ignore
    expert_macro_state_terms: list[str] | None = [
        "expert_motion",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.reference.selection = ReferenceSelectionPreset().sonic
        for group, term_names in (
            (self.observations.policy, _SONIC_HISTORY_TERMS_POLICY),
            (self.observations.critic, _SONIC_HISTORY_TERMS_CRITIC),
        ):
            for term_name in term_names:
                term = getattr(group, term_name)
                if term is not None:
                    term.history_length = 10


@configclass
class ImitationG1SonicNoHistorySurfaceEnvCfg(ImitationG1SonicSurfaceEnvCfg):
    """SONIC release recipe with this repo's single-frame observations.

    The 2026-07-21 isolated history ablation showed SONIC's 10-step
    proprioceptive histories buy little at our scale; everything else stays the
    release recipe.
    """

    def __post_init__(self):
        super().__post_init__()
        for group, term_names in (
            (self.observations.policy, _SONIC_HISTORY_TERMS_POLICY),
            (self.observations.critic, _SONIC_HISTORY_TERMS_CRITIC),
        ):
            for term_name in term_names:
                term = getattr(group, term_name)
                if term is not None:
                    term.history_length = 0


@configclass
class ImitationG1SonicOfficialFSQSurfaceEnvCfg(ImitationG1SonicSurfaceEnvCfg):
    """SONIC environment with a renewed 10-frame 64-D FSQ window command.

    Keeps the sample-efficient reset sampler established by the 2026-07-27
    reset screen rather than SONIC's: full-trajectory adaptive-failure starts
    need far more data at our single-GPU scale. The command window advances
    with the live reference; the agent-side ``code_period=1`` independently
    renews the quantized code.
    """

    def __post_init__(self):
        super().__post_init__()
        self.command_interface.actor = LatentCommandCfg(dim=64)
        self.command_interface.encoder = EncoderViewCfg(past_steps=0, future_steps=9)
        self.command_interface.reference.selection = ReferenceSelectionPreset().default


__all__ = [
    "ImitationG1AblationSurfaceEnvCfg",
    "ImitationG1ChunkSurfaceEnvCfg",
    "ImitationG1ExplicitSurfaceEnvCfg",
    "ImitationG1FutureCVAESurfaceEnvCfg",
    "ImitationG1GoalSurfaceEnvCfg",
    "ImitationG1PerStepVQSurfaceEnvCfg",
    "ImitationG1SonicNoHistorySurfaceEnvCfg",
    "ImitationG1SonicOfficialFSQSurfaceEnvCfg",
    "ImitationG1SonicSurfaceEnvCfg",
    "ImitationG1VQVAESurfaceEnvCfg",
]
