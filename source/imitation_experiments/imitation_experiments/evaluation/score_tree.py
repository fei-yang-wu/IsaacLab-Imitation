#!/usr/bin/env python3
"""Plan the cells of a budget-axis evaluation over one checkpoint tree.

A tracker tree holds one checkpoint per milestone, in one of two layouts.

The MIRROR layout, written by a campaign's `mirror.sh`::

    <tree>/tracker/f<frames>/models/model_step_<frames>.pt

The directory name carries the TRUE cumulative frame count, because the
per-segment step counter restarts on every chained resume.

The TRAINER layout, as it stands on the cluster::

    <tree>/tracker/<timestamp>_wandb-<run id>/models/model_step_<steps>.pt

Here the frame count has to come from the file name, which is only the
cumulative count while the arm ran as ONE segment. A tree whose checkpoints are
spread over several run directories is refused rather than guessed: the second
segment's `model_step_250085376` is not 250M cumulative frames, and a silent
guess would put a curve's points at the wrong budgets. Mirror such a tree
first, where the rename is done with the resume record in hand.

Scoring every milestone of one arm gives the metric-against-frames curve;
scoring only the last one gives the row of record.

This module decides WHICH cells to run and where each row lands. It imports
nothing from Isaac Lab, so a launcher can plan the work -- on a login node, in
a test, or inside the evaluation process itself -- without starting a
simulation.

The output name is the convention every campaign launcher already writes::

    <output_root>/<arm>_seed<seed>_<row>_f<frames>.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

__all__ = [
    "AmbiguousTree",
    "Cell",
    "milestone_checkpoints",
    "milestone_frames",
    "plan_cells",
]

_FRAME_DIR = re.compile(r"^f(\d+)$")
_STEP_FILE = re.compile(r"^model_step_(\d+)\.pt$")


class AmbiguousTree(RuntimeError):
    """A trainer-layout tree whose checkpoints span several run directories."""


@dataclass(frozen=True)
class Cell:
    """One checkpoint scored into one summary file."""

    frames: int
    checkpoint: Path
    output_json: Path


def milestone_checkpoints(tree: Path) -> dict[int, Path]:
    """Map cumulative frame count -> checkpoint file, for either layout.

    The mirror layout wins when it is present: its directory names are the
    authority on cumulative frames. Otherwise the trainer layout is read, and a
    tree whose checkpoints span more than one run directory raises
    `AmbiguousTree`.
    """
    tracker = Path(tree) / "tracker"
    if not tracker.is_dir():
        return {}

    mirrored: dict[int, Path] = {}
    for entry in sorted(tracker.iterdir()):
        if not entry.is_dir():
            continue
        match = _FRAME_DIR.match(entry.name)
        if match is None:
            continue
        frames = int(match.group(1))
        checkpoint = entry / "models" / f"model_step_{frames}.pt"
        if checkpoint.is_file():
            mirrored[frames] = checkpoint
    if mirrored:
        return mirrored

    runs: dict[Path, dict[int, Path]] = {}
    for entry in sorted(tracker.iterdir()):
        models = entry / "models"
        if not entry.is_dir() or not models.is_dir():
            continue
        found: dict[int, Path] = {}
        for checkpoint in sorted(models.iterdir()):
            match = _STEP_FILE.match(checkpoint.name)
            if match is None or not checkpoint.is_file():
                continue
            found[int(match.group(1))] = checkpoint
        if found:
            runs[entry] = found
    if not runs:
        return {}
    if len(runs) > 1:
        names = ", ".join(sorted(path.name for path in runs))
        raise AmbiguousTree(
            f"{tree}: checkpoints span {len(runs)} run directories ({names}). "
            "The per-segment step counter restarts on a chained resume, so the "
            "file names are not cumulative frame counts. Mirror the tree into "
            "the f<frames> layout first."
        )
    return next(iter(runs.values()))


def milestone_frames(tree: Path) -> tuple[int, ...]:
    """Return the cumulative frame counts in ``tree``, ascending."""
    return tuple(sorted(milestone_checkpoints(tree)))


def plan_cells(
    tree: Path,
    output_root: Path,
    *,
    arm: str,
    seed: int,
    row: str,
    final_only: bool = False,
    skip_scored: bool = True,
) -> tuple[Cell, ...]:
    """Return the ordered cells to score for one arm.

    ``final_only`` keeps just the largest frame count -- the row of record.
    ``skip_scored`` drops a cell whose summary file already exists and is
    non-empty, so an interrupted run resumes instead of repeating work.

    A milestone directory without its checkpoint file is skipped: a tree can
    hold a directory whose transfer never finished.

    Raises `AmbiguousTree` for a trainer-layout tree spread over several run
    directories; see `milestone_checkpoints`.
    """
    tree = Path(tree)
    output_root = Path(output_root)
    checkpoints = milestone_checkpoints(tree)
    if not checkpoints:
        return ()
    frames = tuple(sorted(checkpoints))
    if final_only:
        frames = (frames[-1],)

    cells: list[Cell] = []
    for frame in frames:
        checkpoint = checkpoints[frame]
        output_json = output_root / f"{arm}_seed{seed}_{row}_f{frame}.json"
        if skip_scored and output_json.is_file() and output_json.stat().st_size > 0:
            continue
        cells.append(Cell(frames=frame, checkpoint=checkpoint, output_json=output_json))
    return tuple(cells)
