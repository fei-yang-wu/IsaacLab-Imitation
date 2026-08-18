"""Contract tests for shared MPJPE and transition health metrics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_imitation.contracts import mpjpe_local_global
from isaaclab_imitation.tasks.manager_based.imitation.mdp.commands.reference import (
    ReferenceCommandTerm,
)


def test_mpjpe_local_and_global_have_the_frozen_definition() -> None:
    robot_body = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    reference_body = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
    robot_root = torch.tensor([[1.0, 0.0, 0.0]])
    reference_root = torch.tensor([[0.0, 0.0, 0.0]])

    local, global_ = mpjpe_local_global(
        robot_body, robot_root, reference_body, reference_root
    )

    assert local.item() == pytest.approx(0.25)
    assert global_.item() == pytest.approx(0.75)


def _transition_term(window_steps: int = 200) -> ReferenceCommandTerm:
    term = object.__new__(ReferenceCommandTerm)
    term.cfg = SimpleNamespace(transition_metric_window_steps=window_steps)
    term._transition_mpjpe_l_sum = torch.zeros(())
    term._transition_mpjpe_g_sum = torch.zeros(())
    term._transition_weight = torch.zeros(())
    term._transition_metrics_initialized = False
    return term


def test_transition_health_is_transition_weighted_not_episode_weighted() -> None:
    term = _transition_term()
    env = SimpleNamespace(reset_buf=torch.tensor([False, False]))

    # One environment stays in a 20 mm episode. The other cycles through ten
    # short 60 mm episodes over the same 400 control steps. Each control step
    # still has one transition from each environment, so the transition mean
    # is 40 mm while the equal-episode mean is 620/11 = 56.36 mm.
    for _ in range(400):
        term._update_transition_metrics(
            env,
            torch.tensor([20.0, 60.0]),
            torch.tensor([30.0, 70.0]),
        )

    metrics = term.transition_metrics()
    assert metrics["TrainHealth/mpjpe_l_mm_transition_ewma"].item() == pytest.approx(
        40.0
    )
    assert metrics["TrainHealth/mpjpe_g_mm_transition_ewma"].item() == pytest.approx(
        50.0
    )


def test_transition_health_excludes_post_reset_samples() -> None:
    term = _transition_term(window_steps=1)
    env = SimpleNamespace(reset_buf=torch.tensor([False, True]))

    term._update_transition_metrics(
        env,
        torch.tensor([20.0, 999.0]),
        torch.tensor([30.0, 999.0]),
    )

    metrics = term.transition_metrics()
    assert metrics["TrainHealth/mpjpe_l_mm_transition_ewma"].item() == 20.0
    assert metrics["TrainHealth/mpjpe_g_mm_transition_ewma"].item() == 30.0
    assert metrics["TrainHealth/transition_ewma_weight"].item() == 1.0


def test_transition_health_is_empty_before_first_valid_sample() -> None:
    assert _transition_term().transition_metrics() == {}


def test_transition_health_returns_tensors_for_lazy_iteration_logging() -> None:
    term = _transition_term()
    env = SimpleNamespace(reset_buf=torch.tensor([False]))
    term._update_transition_metrics(env, torch.tensor([20.0]), torch.tensor([30.0]))

    assert all(isinstance(value, torch.Tensor) for value in term.transition_metrics().values())
