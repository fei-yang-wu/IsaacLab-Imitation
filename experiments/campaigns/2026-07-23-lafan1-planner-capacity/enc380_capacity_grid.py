#!/usr/bin/env python3
"""Frozen grid for the enc380 shared-tracker capacity diagnostic."""

from __future__ import annotations

MOTIONS = (
    "walk1_subject1",
)
MODEL_SIZES = ("tiny", "small", "medium", "large")
PLANNER_SEEDS = (0, 1, 2)
ROUTES = ("root_qpos", "latent_skill")
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
ROUTE_TASK_COUNT = CELL_COUNT * len(ROUTES)


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


def decode_route_task(index: int) -> tuple[str, str, int, str]:
    """Map the exact 0-23 ICE array onto paired capacity cells and routes."""
    index = int(index)
    if not 0 <= index < ROUTE_TASK_COUNT:
        raise ValueError(
            f"enc380 route-task index {index} is outside 0-{ROUTE_TASK_COUNT - 1}"
        )
    cell_index, route_index = divmod(index, len(ROUTES))
    motion, size, seed = decode_cell(cell_index)
    return motion, size, seed, ROUTES[route_index]
