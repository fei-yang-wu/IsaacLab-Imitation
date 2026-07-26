"""Exact per-motion row accounting for shared planner datasets."""

from __future__ import annotations

from collections.abc import Sequence


class BalancedMotionRowSelector:
    """Select at most a fixed number of rows for every named motion."""

    def __init__(self, motion_names: Sequence[str], rows_per_motion: int) -> None:
        names = tuple(str(name).strip() for name in motion_names)
        if not names or any(not name for name in names):
            raise ValueError("motion_names must contain non-empty names.")
        if len(names) != len(set(names)):
            raise ValueError("motion_names must be unique.")
        if int(rows_per_motion) <= 0:
            raise ValueError("rows_per_motion must be positive.")
        self.motion_names = names
        self.rows_per_motion = int(rows_per_motion)
        self._counts = {name: 0 for name in names}

    def select(self, row_motion_names: Sequence[str]) -> tuple[int, ...]:
        """Return row indices that still fit the exact per-motion budget."""

        selected: list[int] = []
        for index, raw_name in enumerate(row_motion_names):
            name = str(raw_name)
            count = self._counts.get(name)
            if count is None or count >= self.rows_per_motion:
                continue
            self._counts[name] = count + 1
            selected.append(index)
        return tuple(selected)

    @property
    def complete(self) -> bool:
        return all(count == self.rows_per_motion for count in self._counts.values())

    @property
    def total_selected(self) -> int:
        return sum(self._counts.values())

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def missing(self) -> dict[str, int]:
        return {
            name: self.rows_per_motion - count
            for name, count in self._counts.items()
            if count < self.rows_per_motion
        }


__all__ = ["BalancedMotionRowSelector"]
