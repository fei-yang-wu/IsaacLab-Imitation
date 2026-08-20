"""Tests for the fixed-stage curriculum contract."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import CurriculumTermCfg

from isaaclab_imitation.tasks.manager_based.dexmanip.curriculum import (
    DEFAULT_NUM_STEPS_PER_ENV,
    DEFAULT_TIMESTEP_SCHEDULE,
    DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR,
    FixedTimestepCurriculum,
    build_curriculum_params,
    fixed_schedule_stage_index,
    fixed_schedule_thresholds,
)


@dataclass
class _RewardCfg:
    weight: float


class _RewardManager:
    def __init__(self, names: list[str]) -> None:
        self._configs = {name: _RewardCfg(99.0) for name in names}

    @property
    def active_terms(self) -> list[str]:
        return list(self._configs)

    def get_term_cfg(self, name: str) -> _RewardCfg:
        return self._configs[name]


class _CommandManager:
    def __init__(self, command: SimpleNamespace) -> None:
        self._command = command

    def get_term(self, name: str) -> SimpleNamespace:
        assert name == "motion"
        return self._command


def _fake_env(params: dict) -> SimpleNamespace:
    reward_names = list(params["reward_weight_schedules"])
    command = SimpleNamespace(
        virtual_object_controller_curriculum_scale=torch.tensor(-1.0)
    )
    return SimpleNamespace(
        common_step_counter=0,
        command_manager=_CommandManager(command),
        reward_manager=_RewardManager(reward_names),
    )


def test_released_schedule_values_are_exact() -> None:
    params = build_curriculum_params()

    assert params["num_steps_per_env"] == 24
    assert params["timestep_schedule"] == [
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
    ]
    assert params["virtual_object_control_scale_factor"] == [
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
    ]
    assert params["reward_weight_schedules"]["object_keypoints_tracking_exp"] == [
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
    ]


def test_stage_index_changes_at_exact_control_step_threshold() -> None:
    thresholds = fixed_schedule_thresholds(
        DEFAULT_TIMESTEP_SCHEDULE, DEFAULT_NUM_STEPS_PER_ENV
    )

    assert thresholds[0] == 48_000
    assert fixed_schedule_stage_index(0, thresholds, 10) == 0
    assert fixed_schedule_stage_index(47_999, thresholds, 10) == 0
    assert fixed_schedule_stage_index(48_000, thresholds, 10) == 1
    assert fixed_schedule_stage_index(335_999, thresholds, 10) == 8
    assert fixed_schedule_stage_index(336_000, thresholds, 10) == 9
    assert fixed_schedule_stage_index(1_000_000, thresholds, 10) == 9


@pytest.mark.parametrize(
    ("schedule", "rollout_steps", "error_type"),
    [
        ([], 24, ValueError),
        ([2, 2], 24, ValueError),
        ([2, 1], 24, ValueError),
        ([-1], 24, ValueError),
        ([1], 0, ValueError),
        ([1], 1.5, ValueError),
        ([1.5], 24, TypeError),
    ],
)
def test_threshold_validation_rejects_invalid_values(
    schedule: list[int], rollout_steps: int | float, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        fixed_schedule_thresholds(schedule, rollout_steps)


def test_parameter_builder_returns_independent_lists() -> None:
    first = build_curriculum_params()
    second = build_curriculum_params()

    first["timestep_schedule"][0] = -1
    first["reward_weight_schedules"]["object_keypoints_tracking_exp"][0] = -1.0

    assert second["timestep_schedule"][0] == 2000
    assert second["reward_weight_schedules"]["object_keypoints_tracking_exp"][0] == 0.0


def test_manager_term_applies_scale_and_all_reward_weights() -> None:
    params = build_curriculum_params()
    env = _fake_env(params)
    cfg = CurriculumTermCfg(func=FixedTimestepCurriculum, params=params)
    term = FixedTimestepCurriculum(cfg, env)

    state = term(env, slice(None), **params)
    assert state.item() == pytest.approx(1.0)
    assert env.command_manager.get_term(
        "motion"
    ).virtual_object_controller_curriculum_scale.item() == pytest.approx(1.0)
    assert env.reward_manager.get_term_cfg(
        "object_keypoints_tracking_exp"
    ).weight == pytest.approx(0.0)
    assert env.reward_manager.get_term_cfg(
        "contact_wrench_support_reward"
    ).weight == pytest.approx(10.0)

    env.common_step_counter = DEFAULT_TIMESTEP_SCHEDULE[-2] * DEFAULT_NUM_STEPS_PER_ENV
    state = term(env, slice(None), **params)
    assert state.item() == pytest.approx(
        DEFAULT_VIRTUAL_OBJECT_CONTROL_SCALE_FACTOR[-1]
    )
    assert env.reward_manager.get_term_cfg(
        "object_keypoints_tracking_exp"
    ).weight == pytest.approx(20.0)


def test_manager_term_rejects_a_missing_reward() -> None:
    params = build_curriculum_params()
    env = _fake_env(params)
    del env.reward_manager._configs["missed_contact_penalty"]
    cfg = CurriculumTermCfg(func=FixedTimestepCurriculum, params=params)

    with pytest.raises(ValueError, match="missed_contact_penalty"):
        FixedTimestepCurriculum(cfg, env)
