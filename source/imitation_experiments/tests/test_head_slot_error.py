"""Per-slot statistics: the aggregation contract, without a head or a GPU."""

from __future__ import annotations

import math

import pytest
import torch

from imitation_experiments.planner.head_slot_error import per_slot_stats


def test_perfect_prediction_is_cosine_one_and_zero_rmse() -> None:
    target = torch.randn(8, 3, 5)
    stats = per_slot_stats(target.clone(), target)
    assert [s["slot"] for s in stats] == [0, 1, 2]
    for slot in stats:
        assert slot["count"] == 8
        assert slot["cosine"] == pytest.approx(1.0, abs=1e-5)
        assert slot["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_degrading_slots_are_reported_per_slot() -> None:
    target = torch.ones(16, 3, 4)
    prediction = target.clone()
    prediction[:, 1] += 1.0  # slot 1 offset by 1 in every dimension
    prediction[:, 2] *= -1.0  # slot 2 points the other way
    stats = per_slot_stats(prediction, target)
    assert stats[0]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert stats[1]["rmse"] == pytest.approx(1.0, abs=1e-5)
    assert stats[1]["cosine"] == pytest.approx(1.0, abs=1e-5)  # same direction
    assert stats[2]["cosine"] == pytest.approx(-1.0, abs=1e-5)


def test_valid_mask_selects_rows_and_empty_slot_is_nan() -> None:
    target = torch.ones(4, 2, 3)
    prediction = target.clone()
    prediction[0, 0] += 10.0
    valid = torch.ones(4, 2, dtype=torch.bool)
    valid[0, 0] = False  # drop the corrupted row from slot 0
    valid[:, 1] = False  # slot 1 has no valid rows at all
    stats = per_slot_stats(prediction, target, valid)
    assert stats[0]["count"] == 3
    assert stats[0]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert stats[1]["count"] == 0
    assert math.isnan(stats[1]["cosine"])


def test_shape_mismatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        per_slot_stats(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))
    with pytest.raises(ValueError, match=r"expected \[B, H, D\]"):
        per_slot_stats(torch.zeros(2, 3), torch.zeros(2, 3))
