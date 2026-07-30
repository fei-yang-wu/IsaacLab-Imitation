#!/usr/bin/env python3
"""Frozen grid for the enc380 shared-tracker capacity diagnostic."""

from __future__ import annotations

MOTIONS = (
    "walk1_subject1",
)
MODEL_SIZES = ("tiny", "small", "medium", "large")
PLANNER_SEEDS = (0, 1, 2)
PLANNER_BATCH_SIZE = 1024
PLANNER_UPDATES_BY_SIZE = {
    "tiny": 10_000,
    "small": 20_000,
    "medium": 30_000,
    "large": 50_000,
}
PLANNER_MICRO_BATCH_BY_SIZE = {
    "tiny": 1024,
    "small": 512,
    "medium": 256,
    "large": 128,
}
CELL_COUNT = len(MOTIONS) * len(MODEL_SIZES) * len(PLANNER_SEEDS)


def planner_dir_name(size: str) -> str:
    """Return the artifact namespace for the frozen capacity-aware budget."""
    if size not in MODEL_SIZES:
        raise ValueError(f"Unknown enc380 planner size: {size!r}")
    updates = PLANNER_UPDATES_BY_SIZE[size]
    return f"planner_oracle_u{updates}_b{PLANNER_BATCH_SIZE}"


def decode_cell(index: int) -> tuple[str, str, int]:
    """Map the exact 0-11 Slurm array onto motion x capacity x seed."""
    index = int(index)
    if not 0 <= index < CELL_COUNT:
        raise ValueError(f"enc380 cell index {index} is outside 0-{CELL_COUNT - 1}")
    motion_index = index % len(MOTIONS)
    size_index = (index // len(MOTIONS)) % len(MODEL_SIZES)
    seed_index = index // (len(MOTIONS) * len(MODEL_SIZES))
    return MOTIONS[motion_index], MODEL_SIZES[size_index], PLANNER_SEEDS[seed_index]
