"""Tests for planner-only forward timing."""

from __future__ import annotations

import pytest
import torch

from audit_bones_seed_multigoal_language_comparison import _require_planner_latency
from planner_latency import PlannerForwardTimer


def test_timer_counts_only_enabled_root_forwards() -> None:
    planner = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    timer = PlannerForwardTimer(planner)
    planner(torch.ones(2, 3))
    with timer.enabled():
        planner(torch.ones(2, 3))
        planner(torch.ones(1, 3))

    summary = timer.summary(warmup_calls=1)
    timer.close()

    assert summary["total_call_count"] == 2
    assert summary["warmup_calls_excluded"] == 1
    assert summary["measured_call_count"] == 1
    assert summary["batch_sizes"] == [1, 2]
    assert float(summary["mean"]) >= 0.0


def test_timer_reports_no_measurement_after_only_warmup() -> None:
    planner = torch.nn.Linear(2, 2)
    timer = PlannerForwardTimer(planner)
    with timer.enabled():
        planner(torch.ones(1, 2))

    summary = timer.summary(warmup_calls=1)
    timer.close()

    assert summary["total_call_count"] == 1
    assert summary["measured_call_count"] == 0


def test_audit_allows_warmup_only_for_smoke_but_not_paper() -> None:
    summary = {
        "planner_inference_latency_ms": {
            "unit": "ms",
            "scope": "high_level_planner_forward_only",
            "total_call_count": 1,
            "warmup_calls_excluded": 1,
            "measured_call_count": 0,
            "mean": float("nan"),
        }
    }

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    _require_planner_latency(
        require,
        summary,
        label="smoke",
        require_measurement=False,
    )
    with pytest.raises(ValueError, match="no measured calls"):
        _require_planner_latency(
            require,
            summary,
            label="paper",
            require_measurement=True,
        )
