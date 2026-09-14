# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-stage curriculum for virtual object control and reward weights.

This module ports the schedule from ``video_to_data`` CHORD. It uses only the
standard Isaac Lab manager interfaces. It has no PhysX or Newton dependency.
The schedule is global. Isaac Lab applies it when an environment resets.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase


DEFAULT_NUM_STEPS_PER_ENV = 24
"""Number of control steps in one released-recipe PPO rollout."""

DEFAULT_TIMESTEP_SCHEDULE = (
    2000,
    3500,
    5000,
    6500,
    8000,
    9500,
    11000,
    12500,
    14000,
    15500,
)
"""PPO update numbers at which the curriculum stages change."""

DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR = (
    1.0,
    0.75,
    0.5,
    0.25,
    0.1,
    0.05,
    0.025,
    0.01,
    0.0,
    0.0,
)
"""Virtual object control scale for each curriculum stage."""

DEFAULT_REWARD_WEIGHT_SCHEDULES: dict[str, float | tuple[float, ...]] = {
    "object_keypoints_tracking_exp": (
        0.0,
        0.1,
        0.25,
        0.25,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        20.0,
    ),
    "object_pose_tracking_exp": (
        0.0,
        0.1,
        0.25,
        0.25,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
    ),
    "object_goal_tracking_exp": (
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
    ),
    "object_lift_progress": (
        0.0,
        0.0,
        0.1,
        0.2,
        0.25,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
    ),
    "object_task_success": (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0),
    "hand_keypoints_tracking_exp": 0.25,
    "hand_joint_pos_tracking_exp": 0.25,
    "contact_wrench_support_reward": 10.0,
    "unintended_contact_penalty": -10.0,
    "missed_contact_penalty": -1.0,
}
"""Reward weights for the released stages."""


def fixed_schedule_thresholds(
    timestep_schedule: Sequence[int], num_steps_per_env: int
) -> tuple[int, ...]:
    """Convert PPO update numbers to control-step thresholds."""

    if (
        isinstance(num_steps_per_env, bool)
        or not isinstance(num_steps_per_env, int)
        or num_steps_per_env <= 0
    ):
        raise ValueError("num_steps_per_env must be a positive integer.")
    steps = tuple(timestep_schedule)
    if not steps:
        raise ValueError("timestep_schedule must contain at least one value.")
    if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
        raise TypeError("Each timestep_schedule value must be an integer.")
    if any(step < 0 for step in steps):
        raise ValueError("Each timestep_schedule value must be zero or positive.")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("timestep_schedule values must increase.")
    return tuple(step * num_steps_per_env for step in steps)


def fixed_schedule_stage_index(
    common_step_counter: int, thresholds: Sequence[int], stage_count: int
) -> int:
    """Return the active stage for one global control-step count."""

    if common_step_counter < 0:
        raise ValueError("common_step_counter must be zero or positive.")
    if stage_count <= 0:
        raise ValueError("stage_count must be positive.")
    if not thresholds:
        raise ValueError("thresholds must contain at least one value.")
    return min(bisect.bisect_right(thresholds, common_step_counter), stage_count - 1)


def _stage_values(
    value: Real | Sequence[float], stage_count: int, name: str
) -> tuple[float, ...]:
    """Expand one scalar or validate one stage sequence."""

    if isinstance(value, Real) and not isinstance(value, bool):
        values = (float(value),) * stage_count
    else:
        values = tuple(float(item) for item in value)
        if len(values) != stage_count:
            raise ValueError(
                f"{name} must contain {stage_count} values; got {len(values)}."
            )
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite values.")
    return values


SOURCE_REWARD_WEIGHT_SCHEDULES: dict[str, float | tuple[float, ...]] = {
    "object_keypoints_tracking_exp": DEFAULT_REWARD_WEIGHT_SCHEDULES[
        "object_keypoints_tracking_exp"
    ],
    "hand_keypoints_tracking_exp": 0.25,
    "hand_joint_pos_tracking_exp": 0.25,
    "contact_wrench_support_reward": 10.0,
    "unintended_contact_penalty": -10.0,
    "missed_contact_penalty": -1.0,
}
"""The released CHORD schedule with no added task-objective terms.

The released ``v2d_hand_env_cfg.py`` schedules exactly these six rewards.
:data:`DEFAULT_REWARD_WEIGHT_SCHEDULES` adds the object-goal terms of this
repository's own task variant.
"""


def build_curriculum_params(
    command_name: str = "motion",
    num_steps_per_env: int = DEFAULT_NUM_STEPS_PER_ENV,
    reward_weight_schedules: Mapping[str, float | Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Build independent Isaac Lab parameters for the released schedule.

    ``reward_weight_schedules`` selects which reward terms the curriculum
    drives; it defaults to :data:`DEFAULT_REWARD_WEIGHT_SCHEDULES`.
    """

    if not command_name:
        raise ValueError("command_name must not be empty.")
    # Validate the rollout length here so a bad config fails before Isaac starts.
    fixed_schedule_thresholds(DEFAULT_TIMESTEP_SCHEDULE, num_steps_per_env)
    if reward_weight_schedules is None:
        reward_weight_schedules = DEFAULT_REWARD_WEIGHT_SCHEDULES
    reward_schedules: dict[str, float | list[float]] = {}
    for name, value in reward_weight_schedules.items():
        if isinstance(value, Real):
            reward_schedules[name] = float(value)
        else:
            reward_schedules[name] = list(value)
    return {
        "command_name": command_name,
        "num_steps_per_env": int(num_steps_per_env),
        "timestep_schedule": list(DEFAULT_TIMESTEP_SCHEDULE),
        "virtual_object_control_scale_factor": list(
            DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR
        ),
        "reward_weight_schedules": reward_schedules,
    }


def apply_curriculum_stage(
    command: Any,
    reward_manager: Any,
    scale_factor: float,
    reward_weights: Mapping[str, float],
) -> torch.Tensor:
    """Apply one stage through the command and reward manager contracts."""

    scale = getattr(command, "virtual_object_controller_curriculum_scale", None)
    if not isinstance(scale, torch.Tensor):
        raise TypeError(
            "The motion command must provide the tensor "
            "virtual_object_controller_curriculum_scale."
        )
    scale.fill_(float(scale_factor))
    for reward_name, weight in reward_weights.items():
        reward_manager.get_term_cfg(reward_name).weight = float(weight)
    return scale.detach().clone()


class FixedTimestepCurriculum(ManagerTermBase):
    """Apply the released schedule at fixed global step thresholds."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        """Validate and cache the command, thresholds, and reward schedules."""

        super().__init__(cfg, env)
        params = cfg.params
        self._command = env.command_manager.get_term(params["command_name"])
        self._reward_manager = env.reward_manager
        self._thresholds = fixed_schedule_thresholds(
            params["timestep_schedule"], params["num_steps_per_env"]
        )
        stage_count = len(params["virtual_object_control_scale_factor"])
        if stage_count != len(self._thresholds):
            raise ValueError(
                "virtual_object_control_scale_factor and timestep_schedule "
                "must have the same length."
            )
        self._scale_factors = _stage_values(
            params["virtual_object_control_scale_factor"],
            stage_count,
            "virtual_object_control_scale_factor",
        )

        raw_reward_schedules = params["reward_weight_schedules"]
        available_reward_names = set(env.reward_manager.active_terms)
        unknown_reward_names = set(raw_reward_schedules) - available_reward_names
        if unknown_reward_names:
            names = ", ".join(sorted(unknown_reward_names))
            raise ValueError(f"The reward manager does not contain: {names}.")
        self._reward_weight_schedules = {
            name: _stage_values(value, stage_count, f"reward {name}")
            for name, value in raw_reward_schedules.items()
        }
        self._last_stage_index = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        num_steps_per_env: int,
        timestep_schedule: list[int],
        virtual_object_control_scale_factor: list[float],
        reward_weight_schedules: dict[str, float | list[float]],
    ) -> torch.Tensor:
        """Apply the active stage and return its virtual object control scale."""

        del env_ids, command_name, num_steps_per_env, timestep_schedule
        del virtual_object_control_scale_factor, reward_weight_schedules
        stage_index = fixed_schedule_stage_index(
            int(env.common_step_counter), self._thresholds, len(self._scale_factors)
        )
        if stage_index != self._last_stage_index:
            self._last_stage_index = stage_index
            reward_weights = {
                name: values[stage_index]
                for name, values in self._reward_weight_schedules.items()
            }
            return apply_curriculum_stage(
                self._command,
                self._reward_manager,
                self._scale_factors[stage_index],
                reward_weights,
            )
        scale = self._command.virtual_object_controller_curriculum_scale
        return scale.detach().clone()


__all__ = [
    "DEFAULT_NUM_STEPS_PER_ENV",
    "DEFAULT_REWARD_WEIGHT_SCHEDULES",
    "DEFAULT_TIMESTEP_SCHEDULE",
    "DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR",
    "FixedTimestepCurriculum",
    "apply_curriculum_stage",
    "build_curriculum_params",
    "fixed_schedule_stage_index",
    "fixed_schedule_thresholds",
]
