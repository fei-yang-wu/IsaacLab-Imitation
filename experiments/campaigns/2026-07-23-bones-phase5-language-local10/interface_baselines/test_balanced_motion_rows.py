"""Tests for exact balanced planner-row selection."""

from __future__ import annotations

import pytest

from balanced_motion_rows import BalancedMotionRowSelector


def test_selector_enforces_exact_counts_across_uneven_batches() -> None:
    selector = BalancedMotionRowSelector(["walk", "wave"], rows_per_motion=3)

    assert selector.select(["walk", "walk", "unknown", "wave"]) == (0, 1, 3)
    assert selector.counts() == {"walk": 2, "wave": 1}
    assert selector.select(["walk", "walk", "wave", "wave", "wave"]) == (0, 2, 3)
    assert selector.complete is True
    assert selector.total_selected == 6
    assert selector.missing() == {}
    assert selector.select(["walk", "wave"]) == ()


@pytest.mark.parametrize(
    ("names", "rows"),
    [([], 1), ([""], 1), (["walk", "walk"], 1), (["walk"], 0)],
)
def test_selector_rejects_invalid_contract(names: list[str], rows: int) -> None:
    with pytest.raises(ValueError):
        BalancedMotionRowSelector(names, rows)
