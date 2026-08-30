"""Checks for the suffix-k dose-response aggregator."""

from __future__ import annotations

import json

import pytest

from imitation_experiments.capacity.aggregate_window_suffix_arms import (
    aggregate,
    collect_runs,
    parse_run_name,
    resolved_against_reference,
    tail_means,
)


def _write_run(root, name, *, endpoint, ntp, final=50000, nested=False):
    directory = root / name / "encoder" if nested else root / name
    directory.mkdir(parents=True)
    lines = []
    for update in (1000, 30000, 46000, 48000, final):
        lines.append(
            json.dumps(
                {
                    "update": update,
                    # Early updates carry a much worse loss, so a tail mean and
                    # a whole-run mean cannot be confused.
                    "train/jepa_endpoint_loss_eval": endpoint
                    + (5.0 if update < 45000 else 0.0),
                    "train/jepa_ntp_loss_eval": ntp + (50.0 if update < 45000 else 0.0),
                }
            )
        )
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_run_name_defaults_bare_arm_to_seed_zero() -> None:
    assert parse_run_name("suffix2_seed1") == ("suffix2", 1)
    assert parse_run_name("suffix9") == ("suffix9", 0)
    with pytest.raises(ValueError, match="suffixN"):
        parse_run_name("intermediate_seed0")


def test_tail_means_ignores_early_updates() -> None:
    records = [
        {"update": 1000, "a": 100.0},
        {"update": 46000, "a": 2.0},
        {"update": 50000, "a": 4.0},
    ]
    means, count, final = tail_means(records, keys=("a",), tail_updates=5000)
    assert means["a"] == pytest.approx(3.0)
    assert count == 2
    assert final == 50000


def test_collect_runs_rejects_an_incomplete_arm(tmp_path) -> None:
    _write_run(tmp_path, "suffix2_seed0", endpoint=0.19, ntp=7.1)
    _write_run(tmp_path, "suffix9_seed0", endpoint=0.21, ntp=7.2, final=31000)
    with pytest.raises(ValueError, match="stopped at update"):
        collect_runs(tmp_path, tail_updates=5000, expected_updates=50000)


def test_collect_runs_reads_both_flat_and_nested_layouts(tmp_path) -> None:
    _write_run(tmp_path, "suffix2_seed0", endpoint=0.19, ntp=7.1)
    _write_run(tmp_path, "suffix9", endpoint=0.21, ntp=7.2, nested=True)
    runs = collect_runs(tmp_path, tail_updates=5000, expected_updates=50000)
    assert {(run["arm"], run["seed"]) for run in runs} == {
        ("suffix2", 0),
        ("suffix9", 0),
    }


def test_aggregate_orders_by_suffix_length_not_string(tmp_path) -> None:
    for name, endpoint in (
        ("suffix9_seed0", 0.21),
        ("suffix10_seed0", 0.22),
        ("suffix2_seed0", 0.19),
    ):
        _write_run(tmp_path, name, endpoint=endpoint, ntp=7.0)
    rows = aggregate(collect_runs(tmp_path, tail_updates=5000, expected_updates=50000))
    assert [row["arm"] for row in rows] == ["suffix2", "suffix9", "suffix10"]


def test_resolved_flags_overlapping_seed_ranges_as_unresolved(tmp_path) -> None:
    # suffix2 sits clearly below suffix9; suffix5's seeds straddle suffix9's.
    _write_run(tmp_path, "suffix2_seed0", endpoint=0.185, ntp=7.06)
    _write_run(tmp_path, "suffix2_seed1", endpoint=0.190, ntp=7.10)
    _write_run(tmp_path, "suffix5_seed0", endpoint=0.205, ntp=7.15)
    _write_run(tmp_path, "suffix5_seed1", endpoint=0.225, ntp=7.20)
    _write_run(tmp_path, "suffix9_seed0", endpoint=0.210, ntp=7.16)
    _write_run(tmp_path, "suffix9_seed1", endpoint=0.218, ntp=7.18)
    rows = aggregate(collect_runs(tmp_path, tail_updates=5000, expected_updates=50000))
    verdicts = resolved_against_reference(
        rows, reference_arm="suffix9", metric="endpoint_loss_eval"
    )
    assert verdicts["suffix2"]["resolved"]
    assert verdicts["suffix2"]["relative_change_vs_reference"] < 0
    assert not verdicts["suffix5"]["resolved"]
    assert verdicts["suffix5"]["seed_ranges_overlap"]
