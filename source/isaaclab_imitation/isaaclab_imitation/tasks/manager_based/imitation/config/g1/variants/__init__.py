# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen legacy G1 env variants (v0/v1 lineage only, 2026-08-01).

Only ``strict`` survives here, as the frozen Strict-recipe pins
(``-G1-Strict-v0``, ``-G1-Latent-Strict-v0``, ``-G1-Latent-History-v0``).
The sonc / goal / future_cvae / vqvae / ablation families migrated to the
flat v2 base and now live in ``../surfaces/`` (their legacy classes were
deleted with their old task ids).

Do not add new variants here -- new surfaces go to ``../surfaces/`` on the
flat v2 full surface base.
"""

from .strict import (
    ImitationG1LatentStrictEnvCfg,
    ImitationG1LatentStrictHistoryEnvCfg,
    ImitationG1StrictTrackEnvCfg,
)

__all__ = [
    "ImitationG1LatentStrictEnvCfg",
    "ImitationG1LatentStrictHistoryEnvCfg",
    "ImitationG1StrictTrackEnvCfg",
]
