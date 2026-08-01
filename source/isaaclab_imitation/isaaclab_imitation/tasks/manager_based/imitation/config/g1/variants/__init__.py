# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Non-default G1 env variants, one file per variant family.

- ``strict``: strict SONIC terminations on the legacy scaffolding
  (explicit and latent pins, plus the history ablation).
- ``sonic``: the SONIC release recipe (deprecated defaults, official FSQ).
- ``goal`` / ``future_cvae`` / ``vqvae``: LafanTrack-lineage latent command
  studies.
- ``ablation``: the strict-protocol reconstruction-ablation surface.
"""

from .ablation import ImitationG1LatentAblationEnvCfg
from .future_cvae import (
    ImitationG1LatentFutureCVAEEnvCfg,
    ImitationG1LatentPerStepVQEnvCfg,
)
from .goal import ImitationG1LatentGoalEnvCfg
from .sonic import (
    ImitationG1LatentSonicEnvCfg,
    ImitationG1LatentSonicNoHistoryEnvCfg,
    ImitationG1LatentSonicOfficialFSQEnvCfg,
)
from .strict import (
    ImitationG1LatentStrictEnvCfg,
    ImitationG1LatentStrictHistoryEnvCfg,
    ImitationG1StrictTrackEnvCfg,
)
from .vqvae import ImitationG1LatentVQVAEEnvCfg

__all__ = [
    "ImitationG1LatentAblationEnvCfg",
    "ImitationG1LatentFutureCVAEEnvCfg",
    "ImitationG1LatentPerStepVQEnvCfg",
    "ImitationG1LatentGoalEnvCfg",
    "ImitationG1LatentSonicEnvCfg",
    "ImitationG1LatentSonicNoHistoryEnvCfg",
    "ImitationG1LatentSonicOfficialFSQEnvCfg",
    "ImitationG1LatentStrictEnvCfg",
    "ImitationG1LatentStrictHistoryEnvCfg",
    "ImitationG1StrictTrackEnvCfg",
    "ImitationG1LatentVQVAEEnvCfg",
]
