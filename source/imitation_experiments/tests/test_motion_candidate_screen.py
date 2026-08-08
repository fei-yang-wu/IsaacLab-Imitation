from __future__ import annotations

import pytest

from imitation_experiments.lowlevel.motion_candidate_screen import (
    aggregate_motion_screen,
    build_env_rank_assignment,
)


def test_build_env_rank_assignment_repeats_each_motion_stably() -> None:
    assert build_env_rank_assignment([7, 2, 11], 6) == [7, 7, 2, 2, 11, 11]


def test_build_env_rank_assignment_rejects_unbalanced_geometry() -> None:
    with pytest.raises(ValueError, match="positive multiple"):
        build_env_rank_assignment([1, 2, 3], 8)


def test_aggregate_motion_screen_reports_success_and_weighted_tracking() -> None:
    evaluation = {
        "metadata": {"action_sampling": "mode"},
        "per_environment": [
            {
                "trajectory_rank": 4,
                "motion_name": "walk",
                "completed_tracking_success": True,
                "survival_steps": 10,
                "tracking_metrics": {
                    "tracking_mpjpe_mm": 20.0,
                    "tracked_body_pos_error_m": 0.1,
                },
                "tracking_metric_counts": {
                    "tracking_mpjpe_mm": 10,
                    "tracked_body_pos_error_m": 10,
                },
            },
            {
                "trajectory_rank": 4,
                "motion_name": "walk",
                "completed_tracking_success": False,
                "survival_steps": 5,
                "tracking_metrics": {
                    "tracking_mpjpe_mm": 40.0,
                    "tracked_body_pos_error_m": 0.3,
                },
                "tracking_metric_counts": {
                    "tracking_mpjpe_mm": 5,
                    "tracked_body_pos_error_m": 5,
                },
            },
        ],
    }
    candidates = {
        "source": {"name": "test"},
        "candidates": [
            {
                "motion_name": "walk",
                "trajectory_rank": 4,
                "category": "walk",
                "language_goal": "walk",
                "reference_frames": 10,
                "historical_set": None,
            }
        ],
    }
    row = aggregate_motion_screen(evaluation, candidates)["results"][0]
    assert row["success_rate"] == 0.5
    assert row["mpjpe_l_mm_all"] == pytest.approx(80.0 / 3.0)
    assert row["mpjpe_g_m_all"] == pytest.approx(1.0 / 6.0)
    assert row["mpjpe_l_mm_successful"] == 20.0
    assert row["survival_steps_mean"] == 7.5
