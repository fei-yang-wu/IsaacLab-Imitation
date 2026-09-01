"""A convergence figure must not imply data it does not have."""

from __future__ import annotations

import json

import pytest
import yaml

from imitation_experiments.reporting.ablation_curves import (
    Curve,
    load_curves,
    plot_section,
    section_arms,
)


def _row(tmp_path, arm, frames, sr, local, glob, row="clean", seed=0):
    (tmp_path / f"{arm}_seed{seed}_{row}_f{frames}.json").write_text(
        json.dumps(
            {
                "aggregate": {"tracking_success_rate": sr},
                "successful_metrics": {
                    "tracking_mpjpe_mm": {"mean": local},
                    "tracking_mpjpe_g_mm": {"mean": glob},
                },
            }
        )
    )


def test_load_curves_orders_points_by_frames_not_filename(tmp_path):
    # Lexicographic order puts 1000... before 200..., which would draw the
    # curve backwards.
    _row(tmp_path, "hub", 200048640, 0.5, 40.0, 300.0)
    _row(tmp_path, "hub", 1000243200, 0.8, 30.0, 150.0)
    _row(tmp_path, "hub", 400097280, 0.6, 35.0, 220.0)
    curve = load_curves(tmp_path)["hub"]
    assert curve.frames == [200048640, 400097280, 1000243200]
    assert curve.values["success_rate"] == [0.5, 0.6, 0.8]


def test_load_curves_ignores_other_seeds_and_rows(tmp_path):
    _row(tmp_path, "hub", 200048640, 0.5, 40.0, 300.0)
    _row(tmp_path, "hub", 400097280, 0.6, 35.0, 220.0, seed=1)
    _row(tmp_path, "hub", 600145920, 0.7, 33.0, 200.0, row="robust")
    curve = load_curves(tmp_path)["hub"]
    assert curve.frames == [200048640]


def test_load_curves_survives_a_corrupt_row(tmp_path):
    _row(tmp_path, "hub", 200048640, 0.5, 40.0, 300.0)
    (tmp_path / "hub_seed0_clean_f400097280.json").write_text("{broken")
    assert load_curves(tmp_path)["hub"].frames == [200048640]


def test_missing_metric_becomes_none_not_zero(tmp_path):
    (tmp_path / "hub_seed0_clean_f200048640.json").write_text(
        json.dumps({"aggregate": {"tracking_success_rate": 0.5}})
    )
    values = load_curves(tmp_path)["hub"].values
    assert values["success_rate"] == [0.5]
    assert values["mpjpe_local_mm"] == [None]


def test_section_arms_groups_and_puts_the_hub_first(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "arms": {
                    "z_arm": {"vars": {"section": 1}},
                    "hub": {"vars": {"section": 1}},
                    "other": {"vars": {"section": 3}},
                    "untagged": {"vars": {}},
                }
            }
        )
    )
    from imitation_experiments.reporting.ablation_sections import load_campaign_arms

    grouped = section_arms(load_campaign_arms(path))
    assert grouped[1] == ["hub", "z_arm"]
    assert grouped[3] == ["other"]
    assert all("untagged" not in v for v in grouped.values())


def test_a_single_point_arm_is_reported_not_drawn(tmp_path):
    # One dot is not a convergence curve, and dropping it silently reads as
    # "this arm has no data" when it is only early.
    _row(tmp_path, "hub", 200048640, 0.5, 40.0, 300.0)
    _row(tmp_path, "hub", 400097280, 0.6, 35.0, 220.0)
    _row(tmp_path, "early", 200048640, 0.4, 45.0, 400.0)
    curves = load_curves(tmp_path)
    ok, skipped = plot_section(1, ["hub", "early"], curves, tmp_path / "f.png")
    assert ok
    assert skipped == ["early"]
    assert (tmp_path / "f.png").is_file()


def test_a_section_with_no_usable_arm_writes_nothing(tmp_path):
    _row(tmp_path, "early", 200048640, 0.4, 45.0, 400.0)
    curves = load_curves(tmp_path)
    ok, skipped = plot_section(2, ["early"], curves, tmp_path / "none.png")
    assert not ok
    assert skipped == ["early"]
    assert not (tmp_path / "none.png").exists()


def test_plot_skips_an_arm_absent_from_the_scored_rows(tmp_path):
    _row(tmp_path, "hub", 200048640, 0.5, 40.0, 300.0)
    _row(tmp_path, "hub", 400097280, 0.6, 35.0, 220.0)
    ok, skipped = plot_section(
        1, ["hub", "never_scored"], load_curves(tmp_path), tmp_path / "f.png"
    )
    assert ok
    assert skipped == ["never_scored"]


def test_curves_are_cut_at_the_table_budget(tmp_path):
    # A figure and the table beside it must describe the same runs at the same
    # budget. Arms train at different speeds, so an uncut figure would show one
    # arm to 5B and another to 1B and invite a comparison the table never makes.
    _row(tmp_path, "hub", 200048640, 0.5, 40.0, 300.0)
    _row(tmp_path, "hub", 2000486400, 0.9, 24.0, 99.0)
    _row(tmp_path, "hub", 5000232960, 0.93, 23.0, 92.0)
    assert load_curves(tmp_path)["hub"].frames == [200048640, 2000486400]
    assert load_curves(tmp_path, max_frames=None)["hub"].frames[-1] == 5000232960
    assert load_curves(tmp_path, max_frames=400097280)["hub"].frames == [200048640]


def test_plot_uses_the_paper_label_not_the_arm_id(tmp_path):
    _row(tmp_path, "hub", 200048640, 0.5, 40.0, 300.0)
    _row(tmp_path, "hub", 400097280, 0.6, 35.0, 220.0)
    ok, _ = plot_section(
        1, ["hub"], load_curves(tmp_path), tmp_path / "f.png", {"hub": "Ours"}
    )
    assert ok


def test_curve_dataclass_defaults_are_not_shared():
    a, b = Curve("a", [], {}), Curve("b", [], {})
    a.frames.append(1)
    assert b.frames == []


@pytest.mark.parametrize("count", [0, 1])
def test_load_curves_on_an_empty_or_missing_dir(tmp_path, count):
    target = tmp_path / "sub" if count else tmp_path
    if count:
        target.mkdir()
    assert load_curves(target) == {}
