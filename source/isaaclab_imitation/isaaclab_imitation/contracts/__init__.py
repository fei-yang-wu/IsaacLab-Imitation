# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env-free shared data contracts for the imitation tasks.

Modules here define schemas that the live simulation env and offline
dataset/evaluation tooling must agree on byte-for-byte (e.g. the causal
planner observation). They deliberately import only torch -- no Isaac Sim,
no scene -- so contract tests run without a simulator and offline builders
share the exact code path the env uses.
"""

from .causal_planner_observation import (
    CAUSAL_PLANNER_FEATURE_WIDTHS,
    CAUSAL_PLANNER_FRAME_DIM,
    CAUSAL_PLANNER_OBSERVATION_VERSION,
    CausalPlannerHistory,
    build_causal_planner_frame,
    causal_planner_observation_spec,
)
from .command_channels import (
    ACTOR_TERM_NAME,
    COMMAND_CHANNEL_NAMES,
    REFERENCE_TERM_NAME,
)
from .command_publisher import (
    ChunkCommandPublisher,
    CommandPublisher,
    LatentCommandPublisher,
    renewal_env_ids,
)
from .planner_publish_schedule import planner_renew_env_ids

__all__ = [
    "ACTOR_TERM_NAME",
    "COMMAND_CHANNEL_NAMES",
    "REFERENCE_TERM_NAME",
    "CAUSAL_PLANNER_FEATURE_WIDTHS",
    "CAUSAL_PLANNER_FRAME_DIM",
    "CAUSAL_PLANNER_OBSERVATION_VERSION",
    "CausalPlannerHistory",
    "ChunkCommandPublisher",
    "CommandPublisher",
    "LatentCommandPublisher",
    "build_causal_planner_frame",
    "causal_planner_observation_spec",
    "planner_renew_env_ids",
    "renewal_env_ids",
]
