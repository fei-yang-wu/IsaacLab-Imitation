# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DEPRECATED module path -- import from ``common``/``v1``/``variants`` instead.

Kept as a pure re-export shim so historical imports, pickled configs, and
recorded ``module:Class`` entry-point strings keep resolving. Everything this
module used to define now lives in:

- ``common.observations`` (latent surfaces + anchor table),
- ``common.latent_env`` (the latent-lineage base / Legacy pin),
- ``imitation_g1_env_v1`` (the Stable default) and
  ``variants.{strict,sonic,goal,future_cvae}``.
"""

from .common.latent_env import ImitationG1LatentEnvCfg
from .common.observations import (
    _LATENT_ANCHOR_TERM_NAMES_BY_GROUP,
    G1LatentObservationCfg,
    G1SonicLatentObservationCfg,
)
from .imitation_g1_env_v1 import ImitationG1EnvV1Cfg
from .variants.future_cvae import (
    ImitationG1LatentFutureCVAEEnvCfg,
    ImitationG1LatentPerStepVQEnvCfg,
)
from .variants.goal import ImitationG1LatentGoalEnvCfg
from .variants.sonic import (
    ImitationG1LatentSonicEnvCfg,
    ImitationG1LatentSonicNoHistoryEnvCfg,
    ImitationG1LatentSonicOfficialFSQEnvCfg,
)
from .variants.strict import (
    ImitationG1LatentStrictEnvCfg,
    ImitationG1LatentStrictHistoryEnvCfg,
)

# Historical names for the stable (now frozen) v1 config (the flagship
# `ImitationG1EnvCfg` name moved to the v2 module in the 2026-08-01 flip);
# kept resolvable for old serialized configs and entry-point strings.
ImitationG1LatentStableEnvCfg = ImitationG1EnvV1Cfg
ImitationG1StableEnvCfg = ImitationG1EnvV1Cfg
ImitationG1EnvCfg = ImitationG1EnvV1Cfg

__all__ = [
    "G1LatentObservationCfg",
    "ImitationG1EnvCfg",
    "G1SonicLatentObservationCfg",
    "ImitationG1LatentEnvCfg",
    "ImitationG1LatentFutureCVAEEnvCfg",
    "ImitationG1LatentGoalEnvCfg",
    "ImitationG1LatentPerStepVQEnvCfg",
    "ImitationG1LatentSonicEnvCfg",
    "ImitationG1LatentSonicNoHistoryEnvCfg",
    "ImitationG1LatentSonicOfficialFSQEnvCfg",
    "ImitationG1LatentStableEnvCfg",
    "ImitationG1LatentStrictEnvCfg",
    "ImitationG1LatentStrictHistoryEnvCfg",
    "ImitationG1StableEnvCfg",
    "_LATENT_ANCHOR_TERM_NAMES_BY_GROUP",
]
