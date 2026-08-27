"""Contract tests for the subset-selection sensitivity guard."""

from __future__ import annotations

import json

import pytest

from imitation_experiments.evaluation.clip_features import ClipFeatures
from imitation_experiments.evaluation.subset_sensitivity import (
    RULE_GRID_V1,
    ClipScore,
    load_clip_scores,
    micro_mpjpe,
    null_distribution,
    rule_grid,
)


def _scores(n: int = 400) -> list[ClipScore]:
    # Values spread 10..50 mm so a 50-clip subset can land anywhere in between.
    return [
        ClipScore(rank=i, frames=100 + (i % 7) * 10, value=10.0 + 40.0 * i / (n - 1))
        for i in range(n)
    ]


def _features(scores: list[ClipScore]) -> list[ClipFeatures]:
    return [
        ClipFeatures(
            rank=s.rank,
            motion=f"clip_{s.rank}",
            frames=120 + (s.rank % 12) * 110,
            pel_z_min=0.2 + 0.05 * (s.rank % 9),
            pel_z_mean=0.8,
            feet_z_max=0.1 + 0.02 * (s.rank % 5),
            root_speed_mean=0.5,
            root_speed_max=0.5 + 0.4 * (s.rank % 10),
            jvel_p99=3.0 + (s.rank % 6),
            wrist_z_max=1.2,
            travel_m=1.0,
        )
        for s in scores
    ]


def test_micro_is_frame_weighted_not_a_plain_mean():
    scores = [ClipScore(0, 100, 10.0), ClipScore(1, 300, 30.0)]
    assert micro_mpjpe(scores) == pytest.approx(25.0)  # not 20.0
    with pytest.raises(ValueError, match="no frames"):
        micro_mpjpe([ClipScore(0, 0, 10.0)])


def test_null_distribution_is_deterministic_and_brackets_the_target():
    scores = _scores()
    first = null_distribution(scores, target=30.0, size=50, draws=200)
    second = null_distribution(scores, target=30.0, size=50, draws=200)
    assert first == second
    assert first.minimum <= first.mean <= first.maximum
    assert 0.0 <= first.share_at_or_below <= 1.0
    # The pool centres near 30 mm, so the target sits mid-distribution.
    assert abs(first.z) < 1.0
    far = null_distribution(scores, target=5.0, size=50, draws=200)
    assert far.share_at_or_below == 0.0
    assert far.z < -1.0


def test_null_refuses_a_subset_larger_than_the_pool():
    with pytest.raises(ValueError, match="fewer than"):
        null_distribution(_scores(10), target=20.0, size=50)


def test_rule_grid_reports_the_search_size_it_used():
    scores = _scores()
    result = rule_grid(scores, _features(scores), target=25.0, size=50)
    expected = 1
    for axis in RULE_GRID_V1.values():
        expected *= len(axis)
    assert result.rules_tried == expected
    assert 0 < result.rules_valid <= expected
    assert result.minimum <= result.best_value <= result.maximum
    assert 0.0 <= result.hit_share <= 1.0
    assert set(result.best_rule) == set(RULE_GRID_V1)


def test_rule_grid_hit_count_follows_the_tolerance():
    scores = _scores()
    features = _features(scores)
    tight = rule_grid(scores, features, target=25.0, size=50, tolerance=0.1)
    loose = rule_grid(scores, features, target=25.0, size=50, tolerance=5.0)
    assert loose.hits >= tight.hits


def test_load_clip_scores_reads_both_per_clip_schemas(tmp_path):
    sonic = tmp_path / "sonic.json"
    sonic.write_text(
        json.dumps(
            {
                "per_environment": [
                    {
                        "trajectory_rank": 3,
                        "completed_tracking_success": True,
                        "survival_steps": 200,
                        "metrics": {"mpjpe_l_mm": 21.0},
                    },
                    {
                        "trajectory_rank": 4,
                        "completed_tracking_success": False,
                        "survival_steps": 40,
                        "metrics": {"mpjpe_l_mm": 90.0},
                    },
                ]
            }
        )
    )
    ours = tmp_path / "ours.json"
    ours.write_text(
        json.dumps(
            {
                "per_environment": [
                    {
                        "trajectory_rank": 3,
                        "completed_tracking_success": True,
                        "survival_steps": 200,
                        "tracking_metrics": {"tracking_mpjpe_mm": 21.0},
                    }
                ]
            }
        )
    )
    # Failed clips are dropped: a success-only metric never averages them in.
    assert [s.rank for s in load_clip_scores(sonic)] == [3]
    assert load_clip_scores(sonic)[0].value == pytest.approx(21.0)
    assert load_clip_scores(ours)[0].value == pytest.approx(21.0)


def test_load_clip_scores_refuses_a_missing_metric(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "per_environment": [
                    {
                        "trajectory_rank": 0,
                        "completed_tracking_success": True,
                        "survival_steps": 10,
                        "metrics": {},
                    }
                ]
            }
        )
    )
    with pytest.raises(KeyError, match="carries no"):
        load_clip_scores(path)
