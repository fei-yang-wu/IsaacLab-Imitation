# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The two command-channel names, shared by the env and the command terms.

The imitation environments carry exactly two command terms (see
``tasks/manager_based/imitation/command_interface.py``): the always-present
dataset-backed reference channel, and the single actor channel. Their manager
names are part of the observation contract -- an observation term names the
channel it reads -- so they live here, in the env-free contracts package, where
both the environment and the term implementations can import them without a
circular dependency.
"""

REFERENCE_TERM_NAME = "reference"
"""Manager name of the privileged, always-present reference channel."""

ACTOR_TERM_NAME = "actor"
"""Manager name of the single actor command channel."""

COMMAND_CHANNEL_NAMES: tuple[str, ...] = (REFERENCE_TERM_NAME, ACTOR_TERM_NAME)

__all__ = ["ACTOR_TERM_NAME", "COMMAND_CHANNEL_NAMES", "REFERENCE_TERM_NAME"]
