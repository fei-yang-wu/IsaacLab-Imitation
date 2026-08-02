# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC release-recipe surfaces on the flat v2 full surface (migrated
2026-08-01).

Replaces the legacy ``variants.sonic`` classes (now deleted); their task ids
are re-registered on the v2 env as ``Isaac-Imitation-G1-Sonic-v0``,
``Isaac-Imitation-G1-Sonic-NoHist-v0`` and
``Isaac-Imitation-G1-SonicOfficialFSQ-v0``.

The SONIC delta over the v2 full surface: the h10-history observation set,
the termination-threshold anneal curriculum, and SONIC's full-trajectory
adaptive-failure reset sampling (``random_reset_full_trajectory=True`` with
``failure_rate_max_over_mean=200``); the SONIC rewards/events/actions are
already the v2 flagship components.
"""

from isaaclab.utils.configclass import configclass

from ..common.observations import (
    G1LatentObservationCfg,
    G1SonicLatentObservationCfg,
)
from ..common.terminations import G1SonicTerminationCurriculumCfg
from ..imitation_g1_env_v2 import ImitationG1FullSurfaceEnvCfg


@configclass
class ImitationG1SonicSurfaceEnvCfg(ImitationG1FullSurfaceEnvCfg):
    """Flat v2 full surface matched to the public SONIC release recipe.

    Termination thresholds are annealed from the release's base/eval values
    to its strict training values over the curriculum window; every frame
    after the window uses the strict release protocol. Disable with
    ``env.curriculum=null`` for strict-from-scratch release fidelity.
    """

    observations = G1SonicLatentObservationCfg()  # type: ignore
    curriculum = G1SonicTerminationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # SONIC's motion library samples over the complete trajectory, with
        # adaptive failure weighting and a uniform component. The v2 flagship
        # intentionally limits starts to [0, 200], so undo that only here.
        self.random_reset_step_max = 0
        self.random_reset_full_trajectory = True
        self.adaptive_failure_reset_failure_rate_max_over_mean = 200.0


@configclass
class ImitationG1SonicNoHistorySurfaceEnvCfg(ImitationG1SonicSurfaceEnvCfg):
    """SONIC release environment with this repo's single-frame observations.

    Everything on the environment side stays the SONIC release recipe; the
    one deliberate departure is the observation set: the 2026-07-21 isolated
    history ablation showed SONIC's 10-step proprioceptive histories buy
    little at our scale, so this surface keeps the repo's single-frame
    ``G1LatentObservationCfg``. Term *names* are unchanged, so the SONIC
    input-key selection still resolves; only the per-term history length
    differs.
    """

    observations = G1LatentObservationCfg()  # type: ignore


@configclass
class ImitationG1SonicOfficialFSQSurfaceEnvCfg(ImitationG1SonicSurfaceEnvCfg):
    """SONIC release environment with a renewed 10-frame FSQ window command."""

    latent_command_dim: int = 64

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 9
        # Keep the sample-efficient reset sampler established by the Stable
        # reset screen; full-trajectory adaptive-failure starts need far more
        # data at our single-GPU scale.
        self.random_reset_step_max = 200
        self.random_reset_full_trajectory = False
        self.adaptive_failure_reset_failure_rate_max_over_mean = 50.0
        # Zero means the observation window advances with the live reference.
        # The agent-side code_period=1 independently renews the quantized code.
        self.command_hold_steps = 0


__all__ = [
    "ImitationG1SonicNoHistorySurfaceEnvCfg",
    "ImitationG1SonicOfficialFSQSurfaceEnvCfg",
    "ImitationG1SonicSurfaceEnvCfg",
]
