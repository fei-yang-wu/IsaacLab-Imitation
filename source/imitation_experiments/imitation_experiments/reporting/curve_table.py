#!/usr/bin/env python3
"""Reduce scored milestone rows to one tidy table: metrics against frames.

A budget-axis campaign writes one summary JSON per cell, named by the
convention every launcher uses::

    <arm>_seed<seed>_<row>_f<frames>.json

This module turns a directory of those into rows a plot can consume directly --
one row per (campaign, arm, seed, row, env_frames), carrying the canonical
three numbers together: success rate, success-only micro MPJPE-L, and
success-only MPJPE-G.

A COLLAPSED arm is kept, not dropped. `summarize_paper_boards` refuses to
report an MPJPE with no successful episodes, which is correct for a paper row
and wrong for a curve: dropping the row would silently remove the arm from the
figure and make a collapse look like missing data. Such a row carries its real
success rate with `mpjpe_l_micro_mm` and `mpjpe_g_micro_mm` left empty and
`collapsed=True`.

    python -m imitation_experiments.reporting.curve_table \\
        logs/interface_design_study_eval logs/pareto_stack_eval \\
        --row milestone --csv logs/report/curve.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys

from imitation_experiments.evaluation.summarize_paper_boards import summarize_board

__all__ = ["CurvePoint", "parse_cell_name", "collect_points", "write_csv"]

# `<arm>_seed<seed>_<row>_f<frames>.json`. The arm may itself contain
# underscores, so anchor on the fixed tail rather than splitting on "_".
_CELL_NAME = re.compile(
    r"^(?P<arm>.+)_seed(?P<seed>\d+)_(?P<row>[a-z]+)_f(?P<frames>\d+)$"
)

CSV_FIELDS = (
    "campaign",
    "arm",
    "seed",
    "row",
    "env_frames",
    "episodes",
    "successes",
    "success_rate",
    "mpjpe_l_micro_mm",
    "mpjpe_g_micro_mm",
    "velocity_distance_mps",
    "acceleration_distance_mps2",
    "collapsed",
)


@dataclass(frozen=True)
class CurvePoint:
    """One (arm, budget) point of a metric-against-frames curve."""

    campaign: str
    arm: str
    seed: int
    row: str
    env_frames: int
    episodes: int
    successes: int
    success_rate: float
    mpjpe_l_micro_mm: float | None
    mpjpe_g_micro_mm: float | None
    velocity_distance_mps: float | None
    acceleration_distance_mps2: float | None
    collapsed: bool


def parse_cell_name(stem: str) -> tuple[str, int, str, int] | None:
    """Return ``(arm, seed, row, frames)`` for a cell file stem, else None."""
    match = _CELL_NAME.match(stem)
    if match is None:
        return None
    return (
        match.group("arm"),
        int(match.group("seed")),
        match.group("row"),
        int(match.group("frames")),
    )


def _point(campaign: str, path: Path) -> CurvePoint | None:
    parsed = parse_cell_name(path.stem)
    if parsed is None:
        return None
    arm, seed, row, frames = parsed
    result = json.loads(path.read_text(encoding="utf-8"))
    episodes = list(result.get("per_environment", []))
    try:
        board = summarize_board(result)
    except ValueError:
        # Collapsed: real success rate, undefined success-only MPJPE.
        # Same success definition as the paper row: SONIC's completed-tracking
        # flag, never the instantaneous one.
        successes = sum(
            1 for item in episodes if bool(item["completed_tracking_success"])
        )
        return CurvePoint(
            campaign=campaign,
            arm=arm,
            seed=seed,
            row=row,
            env_frames=frames,
            episodes=len(episodes),
            successes=successes,
            success_rate=(successes / len(episodes)) if episodes else 0.0,
            mpjpe_l_micro_mm=None,
            mpjpe_g_micro_mm=None,
            velocity_distance_mps=None,
            acceleration_distance_mps2=None,
            collapsed=True,
        )
    return CurvePoint(
        campaign=campaign,
        arm=arm,
        seed=seed,
        row=row,
        env_frames=frames,
        episodes=board.episodes,
        successes=board.successes,
        success_rate=board.success_rate,
        mpjpe_l_micro_mm=board.mpjpe_l_micro_mm,
        mpjpe_g_micro_mm=board.mpjpe_g_micro_mm,
        velocity_distance_mps=board.velocity_distance_mps,
        acceleration_distance_mps2=board.acceleration_distance_mps2,
        collapsed=False,
    )


def collect_points(
    eval_dirs: list[Path], *, row: str | None = None
) -> list[CurvePoint]:
    """Collect every cell under ``eval_dirs``, sorted by arm then budget.

    The campaign name is the eval directory's own name, so points from several
    campaigns stay separable in one table.
    """
    points: list[CurvePoint] = []
    for eval_dir in eval_dirs:
        eval_dir = Path(eval_dir)
        campaign = eval_dir.name
        for path in sorted(eval_dir.glob("*.json")):
            point = _point(campaign, path)
            if point is None:
                continue
            if row is not None and point.row != row:
                continue
            points.append(point)
    points.sort(key=lambda p: (p.campaign, p.arm, p.seed, p.row, p.env_frames))
    return points


def write_csv(points: list[CurvePoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dirs", type=Path, nargs="+")
    parser.add_argument("--row", type=str, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    points = collect_points(list(args.eval_dirs), row=args.row)
    if not points:
        print("[INFO] no cells matched", file=sys.stderr)
        return 1
    if args.csv is not None:
        write_csv(points, args.csv)
        print(f"[INFO] wrote {len(points)} rows: {args.csv}")
        return 0
    for point in points:
        mpjpe_l = (
            "     -"
            if point.mpjpe_l_micro_mm is None
            else f"{point.mpjpe_l_micro_mm:6.2f}"
        )
        mpjpe_g = (
            "      -"
            if point.mpjpe_g_micro_mm is None
            else f"{point.mpjpe_g_micro_mm:7.2f}"
        )
        print(
            f"{point.campaign:28s} {point.arm:26s} s{point.seed} "
            f"{point.env_frames / 1e9:5.2f}B SR={point.success_rate:.4f} "
            f"L={mpjpe_l} mm G={mpjpe_g} mm"
            + ("  COLLAPSED" if point.collapsed else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
