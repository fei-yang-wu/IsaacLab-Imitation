# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""MDP terms for the source-parity Sharpa task.

Most terms are the shared CHORD kernels in :mod:`dexmanip.mdp`, bound to the
``sharpa_reference`` command. The few terms below exist because the released
Sharpa recipe differs from the Vega-Wuji surface: the action penalty is a sum
of squares over both hands, and the processed action of one hand is read from
that hand's action term.
"""

from __future__ import annotations

from typing import Any

import torch

from . import mdp as dex_mdp

COMMAND_NAME = "sharpa_reference"


def _command(env: Any, name: str = COMMAND_NAME) -> Any:
    return env.command_manager.get_term(name)


def action_norm_sum(env: Any, action_names: list[str]) -> torch.Tensor:
    """Sum of squared raw actions over the listed action terms (released ``action_norm``)."""

    total = None
    for name in action_names:
        value = env.action_manager.get_term(name).raw_actions.square().sum(dim=-1)
        total = value if total is None else total + value
    if total is None:
        raise ValueError("action_names must list at least one action term.")
    return total


def processed_action(env: Any, action_name: str) -> torch.Tensor:
    """Return the processed target of one hand's residual action term."""

    return env.action_manager.get_term(action_name).processed_actions


def finger_joint_pos(env: Any, command_name: str = COMMAND_NAME) -> torch.Tensor:
    """Return limit-scaled right and left finger positions in the live joint order."""

    command = _command(env, command_name)
    return torch.cat(
        (
            command.right_hand_finger_joint_pos_scaled,
            command.left_hand_finger_joint_pos_scaled,
        ),
        dim=-1,
    )


def reference_timeout(env: Any, command_name: str = COMMAND_NAME) -> torch.Tensor:
    """Time out when the reference horizon is reached (released ``timestep_timeout``)."""

    return dex_mdp.timestep_timeout(env, command_name)


# Shared CHORD kernels bound to the Sharpa command name.
wrist_position_e = dex_mdp.wrist_position_e
wrist_orientation_e = dex_mdp.wrist_orientation_e
wrist_velocity_b = dex_mdp.wrist_velocity_b
finger_joint_vel = dex_mdp.finger_joint_vel
object_position_e = dex_mdp.object_position_e
object_orientation_e = dex_mdp.object_orientation_e
contact_position_direction_in_wrist = dex_mdp.contact_position_direction_in_wrist
object_keypoints_tracking_exp = dex_mdp.object_keypoints_tracking_exp
hand_keypoints_tracking_exp = dex_mdp.hand_keypoints_tracking_exp
hand_joint_pos_tracking_exp = dex_mdp.hand_joint_pos_tracking_exp
contact_wrench_support_reward = dex_mdp.contact_wrench_support_reward
unintended_contact_penalty = dex_mdp.unintended_contact_penalty
missed_contact_penalty = dex_mdp.missed_contact_penalty
termination_penalty = dex_mdp.termination_penalty
hand_wrist_away_from_trajectory = dex_mdp.hand_wrist_away_from_trajectory
object_away_from_trajectory = dex_mdp.object_away_from_trajectory

__all__ = [name for name in globals() if not name.startswith("_")]
