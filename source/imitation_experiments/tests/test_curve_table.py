"""Tests for the metrics-against-frames table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from imitation_experiments.reporting.curve_table import (
    collect_points,
    parse_cell_name,
    write_csv,
)


def _cell(
    directory: Path,
    name: str,
    *,
    successes: int,
    episodes: int,
    mpjpe_mm: float = 25.0,
    mpjpe_g_mm: float = 150.0,
) -> Path:
    per_env = []
    for index in range(episodes):
        success = index < successes
        entry = {
            "env_id": index,
            "trajectory_rank": index,
            "tracking_success": success,
            "completed_tracking_success": success,
            "survival_steps": 100,
            "termination_cause": "none" if success else "ee_body_pos",
        }
        if success:
            entry["tracking_metrics"] = {
                "tracking_mpjpe_mm": mpjpe_mm,
                "tracking_mpjpe_g_mm": mpjpe_g_mm,
            }
            entry["tracking_metric_counts"] = {
                "tracking_mpjpe_mm": 100,
                "tracking_mpjpe_g_mm": 100,
            }
        per_env.append(entry)
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"per_environment": per_env}), encoding="utf-8")
    return path


def test_parse_cell_name_keeps_underscored_arms() -> None:
    assert parse_cell_name("jepa_h1_ee_wide_seed0_milestone_f250085376") == (
        "jepa_h1_ee_wide",
        0,
        "milestone",
        250085376,
    )


def test_parse_cell_name_rejects_a_foreign_stem() -> None:
    assert parse_cell_name("summary") is None


def test_collect_points_orders_by_budget(tmp_path: Path) -> None:
    eval_dir = tmp_path / "pareto_stack_eval"
    eval_dir.mkdir()
    _cell(eval_dir, "ctrl_seed0_milestone_f500170752", successes=2, episodes=4)
    _cell(eval_dir, "ctrl_seed0_milestone_f250085376", successes=1, episodes=4)
    points = collect_points([eval_dir])
    assert [point.env_frames for point in points] == [250085376, 500170752]
    assert [point.success_rate for point in points] == [0.25, 0.5]
    assert points[0].campaign == "pareto_stack_eval"
    assert points[0].arm == "ctrl"


def test_collect_points_filters_by_row(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _cell(eval_dir, "ctrl_seed0_milestone_f250085376", successes=1, episodes=4)
    _cell(eval_dir, "ctrl_seed0_clean_f2000289792", successes=3, episodes=4)
    points = collect_points([eval_dir], row="milestone")
    assert [point.row for point in points] == ["milestone"]


def test_collect_points_keeps_a_collapsed_arm(tmp_path: Path) -> None:
    # No successful episode: the arm stays in the table with an empty
    # success-only MPJPE, so a collapse cannot look like missing data.
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _cell(eval_dir, "vq_ema_seed0_milestone_f250085376", successes=0, episodes=4)
    (point,) = collect_points([eval_dir])
    assert point.collapsed is True
    assert point.success_rate == 0.0
    assert point.mpjpe_l_micro_mm is None
    assert point.mpjpe_g_micro_mm is None


def test_write_csv_round_trips(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _cell(eval_dir, "ctrl_seed0_milestone_f250085376", successes=2, episodes=4)
    out = tmp_path / "report" / "curve.csv"
    write_csv(collect_points([eval_dir]), out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["arm"] == "ctrl"
    assert rows[0]["env_frames"] == "250085376"
    assert rows[0]["success_rate"] == "0.5"
