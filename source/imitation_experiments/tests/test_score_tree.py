"""Tests for the budget-axis cell planner."""

from __future__ import annotations

from pathlib import Path

import pytest

from imitation_experiments.evaluation.score_tree import (
    AmbiguousTree,
    Cell,
    milestone_frames,
    plan_cells,
)


def _tree(root: Path, frames: list[int], *, with_file: bool = True) -> Path:
    tree = root / "arm_seed0"
    for frame in frames:
        models = tree / "tracker" / f"f{frame}" / "models"
        models.mkdir(parents=True)
        if with_file:
            (models / f"model_step_{frame}.pt").write_bytes(b"weights")
    return tree


def test_milestone_frames_sorts_numerically(tmp_path: Path) -> None:
    tree = _tree(tmp_path, [1000341504, 250085376, 2000289792])
    assert milestone_frames(tree) == (250085376, 1000341504, 2000289792)


def test_milestone_frames_ignores_foreign_names(tmp_path: Path) -> None:
    tree = _tree(tmp_path, [250085376])
    (tree / "tracker" / "2026-08-19_11-00-00_wandb-ids-ctrl").mkdir()
    (tree / "tracker" / "fnot-a-number").mkdir()
    assert milestone_frames(tree) == (250085376,)


def test_milestone_frames_on_missing_tree(tmp_path: Path) -> None:
    assert milestone_frames(tmp_path / "absent") == ()


def test_plan_cells_names_rows_by_convention(tmp_path: Path) -> None:
    tree = _tree(tmp_path, [250085376, 500170752])
    cells = plan_cells(tree, tmp_path / "out", arm="ctrl", seed=0, row="milestone")
    assert cells == (
        Cell(
            frames=250085376,
            checkpoint=tree
            / "tracker"
            / "f250085376"
            / "models"
            / "model_step_250085376.pt",
            output_json=tmp_path / "out" / "ctrl_seed0_milestone_f250085376.json",
        ),
        Cell(
            frames=500170752,
            checkpoint=tree
            / "tracker"
            / "f500170752"
            / "models"
            / "model_step_500170752.pt",
            output_json=tmp_path / "out" / "ctrl_seed0_milestone_f500170752.json",
        ),
    )


def test_plan_cells_final_only_takes_the_largest_frame_count(tmp_path: Path) -> None:
    tree = _tree(tmp_path, [250085376, 2000289792, 500170752])
    cells = plan_cells(
        tree,
        tmp_path / "out",
        arm="ctrl",
        seed=1,
        row="clean",
        final_only=True,
    )
    assert [cell.frames for cell in cells] == [2000289792]
    assert cells[0].output_json.name == "ctrl_seed1_clean_f2000289792.json"


def test_plan_cells_skips_scored_rows(tmp_path: Path) -> None:
    tree = _tree(tmp_path, [250085376, 500170752])
    out = tmp_path / "out"
    out.mkdir()
    (out / "ctrl_seed0_milestone_f250085376.json").write_text("{}")
    cells = plan_cells(tree, out, arm="ctrl", seed=0, row="milestone")
    assert [cell.frames for cell in cells] == [500170752]


def test_plan_cells_keeps_an_empty_summary_file(tmp_path: Path) -> None:
    # A zero-byte file is a crashed write, not a scored row.
    tree = _tree(tmp_path, [250085376])
    out = tmp_path / "out"
    out.mkdir()
    (out / "ctrl_seed0_milestone_f250085376.json").write_text("")
    cells = plan_cells(tree, out, arm="ctrl", seed=0, row="milestone")
    assert [cell.frames for cell in cells] == [250085376]


def test_plan_cells_can_rescore(tmp_path: Path) -> None:
    tree = _tree(tmp_path, [250085376])
    out = tmp_path / "out"
    out.mkdir()
    (out / "ctrl_seed0_milestone_f250085376.json").write_text("{}")
    cells = plan_cells(
        tree, out, arm="ctrl", seed=0, row="milestone", skip_scored=False
    )
    assert [cell.frames for cell in cells] == [250085376]


def test_plan_cells_skips_a_milestone_without_its_checkpoint(tmp_path: Path) -> None:
    tree = _tree(tmp_path, [250085376], with_file=False)
    cells = plan_cells(tree, tmp_path / "out", arm="ctrl", seed=0, row="milestone")
    assert cells == ()


def _trainer_tree(root: Path, runs: dict[str, list[int]]) -> Path:
    """The layout the trainer writes: one run directory per submitted segment."""
    tree = root / "arm_seed0"
    for run, steps in runs.items():
        models = tree / "tracker" / run / "models"
        models.mkdir(parents=True)
        for step in steps:
            (models / f"model_step_{step}.pt").write_bytes(b"weights")
    return tree


def test_trainer_layout_reads_frames_from_the_file_name(tmp_path: Path) -> None:
    tree = _trainer_tree(
        tmp_path,
        {"2026-08-26_23-28-41_wandb-ids-usehold5-s0-0d49ed": [250085376, 500170752]},
    )
    assert milestone_frames(tree) == (250085376, 500170752)
    cells = plan_cells(tree, tmp_path / "out", arm="use_hold5", seed=0, row="milestone")
    assert [cell.frames for cell in cells] == [250085376, 500170752]
    assert cells[0].checkpoint.name == "model_step_250085376.pt"


def test_trainer_layout_ignores_a_run_directory_with_no_checkpoint(
    tmp_path: Path,
) -> None:
    # A resume that found the budget already met writes no checkpoint.
    tree = _trainer_tree(
        tmp_path,
        {
            "2026-08-26_23-28-41_wandb-ids-usehold5-s0-0d49ed": [250085376],
            "2026-08-27_03-52-31_wandb-ids-usehold5-s0-0d49ed": [],
        },
    )
    assert milestone_frames(tree) == (250085376,)


def test_trainer_layout_refuses_a_chained_tree(tmp_path: Path) -> None:
    # The step counter restarts per segment, so these file names are not
    # cumulative frame counts and guessing would misplace the curve's points.
    tree = _trainer_tree(
        tmp_path,
        {
            "2026-08-26_23-28-41_wandb-x": [250085376],
            "2026-08-27_03-52-31_wandb-x": [250085376],
        },
    )
    with pytest.raises(AmbiguousTree):
        milestone_frames(tree)


def test_mirror_layout_wins_over_the_trainer_layout(tmp_path: Path) -> None:
    # A mirrored tree keeps the run directory beside the renamed one; the
    # f<frames> names are the authority on cumulative frames.
    tree = _tree(tmp_path, [250085376])
    models = tree / "tracker" / "2026-08-26_23-28-41_wandb-x" / "models"
    models.mkdir(parents=True)
    (models / "model_step_999.pt").write_bytes(b"weights")
    assert milestone_frames(tree) == (250085376,)
