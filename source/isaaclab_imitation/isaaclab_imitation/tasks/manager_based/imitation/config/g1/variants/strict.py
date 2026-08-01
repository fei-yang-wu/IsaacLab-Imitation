# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict-recipe pins: strict SONIC terminations on the legacy scaffolding.

The recipe itself lives in ``common.tracking_env._apply_strict_recipe``
(pelvis anchor, [0, 200] reset starts, no curriculum); the classes here
contribute only the observation surface and command configuration.
"""

from isaaclab.utils.configclass import configclass

from ..common.latent_env import ImitationG1LatentEnvCfg
from ..common.observations_latent import G1SonicLatentObservationCfg
from ..common.terminations import G1SonicTerminationsCfg
from ..common.tracking_env import (
    ImitationG1BaseTrackingEnvCfg,
    _apply_strict_recipe,
    _bind_lafan_track_from_dict,
)


@configclass
class ImitationG1StrictTrackEnvCfg(ImitationG1BaseTrackingEnvCfg):
    """Strict recipe x explicit command pin (`Isaac-Imitation-G1-Strict-v0`).

    The same Strict recipe as ``ImitationG1LatentStrictEnvCfg`` (see
    ``_apply_strict_recipe``: pelvis anchor, strict SONIC termination
    functions, [0, 200] reset starts, no curriculum) on the vanilla
    observation surface without a latent command, so explicit-interface
    trackers (single-frame full-body, full-body chunk, EE chunk via
    ``agent.command_space``) train on the same protocol as the latent tracker
    and differ only in the command space.
    """

    terminations = G1SonicTerminationsCfg()  # type: ignore
    curriculum = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_strict_recipe(self)


@configclass
class ImitationG1LatentStrictEnvCfg(ImitationG1LatentEnvCfg):
    """Strict recipe x latent command pin (`Isaac-Imitation-G1-Latent-Strict-v0`).

    Pelvis-anchored legacy surface with strict-from-scratch terminations;
    the recipe itself is shared with the explicit pin
    ``ImitationG1StrictTrackEnvCfg`` via ``_apply_strict_recipe``.

    The evidence-backed middle ground from the 2026-07-19/20 investigation:
    keep the scaffolding that trains at single-GPU/1B scale (legacy [0, 200]
    reset starts, mimic actuators, single-frame observations, bundled G1
    asset, proven optimizer contract) and take from SONIC only the pelvis
    anchor and the strict adaptive termination functions. Requires a
    pelvis-anchored skill encoder (e.g.
    ``skill_encoder_sonic_pelvis_h25_20260719``, sha256 ``388d3e82...``).

    Curriculum default removed (2026-07-21): the 50M -> 300M threshold anneal
    (``G1SonicTerminationCurriculumCfg``) made early training curves
    uninterpretable because the termination goalposts move while the policy
    learns. Thresholds are now the strict release values from frame 0.
    Opt back in with
    ``env.curriculum=G1SonicTerminationCurriculumCfg()``-style overrides if a
    run explicitly wants the anneal.
    """

    terminations = G1SonicTerminationsCfg()  # type: ignore
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        _apply_strict_recipe(self)


@configclass
class ImitationG1LatentStrictHistoryEnvCfg(ImitationG1LatentStrictEnvCfg):
    """Strict surface plus SONIC's 10-step proprioceptive history observations.

    Isolated history ablation (2026-07-21): identical to the default strict
    surface (pelvis anchor, strict-from-scratch terminations, no curriculum,
    legacy scaffolding) except policy/critic observations come from
    ``G1SonicLatentObservationCfg`` -- 10-step histories on the
    proprioceptive terms and SONIC's actor input set (adds
    ``projected_gravity``, drops the robot body-pose terms from the policy
    group). Pair with ``G1ImitationLatentSonicRLOptIPMDConfig`` (local
    optimizer contract default), which selects the SONIC input keys, so the
    only contract difference vs. ``Isaac-Imitation-G1-Latent-v0`` is the
    history/observation set: a low-cost stand-in for a recurrent policy.
    """

    observations = G1SonicLatentObservationCfg()


_bind_lafan_track_from_dict(
    ImitationG1StrictTrackEnvCfg,
    ImitationG1LatentStrictEnvCfg,
    ImitationG1LatentStrictHistoryEnvCfg,
)
