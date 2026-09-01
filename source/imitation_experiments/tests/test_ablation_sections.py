"""The section report must never invent progress it cannot see."""

from __future__ import annotations

import json

import pytest
import yaml

from imitation_experiments.reporting.ablation_sections import (
    ArmRow,
    arm_status,
    build_sections,
    deepest_checkpoint_frames,
    load_campaign_arms,
    parse_slurm_states,
    render,
    scored_rows,
)


def _campaign(tmp_path, arms):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"name": "t", "arms": arms}))
    return path


def test_load_campaign_arms_rejects_an_empty_campaign(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({"name": "t", "arms": {}}))
    with pytest.raises(ValueError):
        load_campaign_arms(path)


def test_deepest_checkpoint_takes_the_largest_step_not_the_newest(tmp_path):
    models = tmp_path / "hub_seed0" / "tracker" / "run_a" / "models"
    models.mkdir(parents=True)
    for step in (200048640, 1000243200, 400097280):
        (models / f"model_step_{step}.pt").write_text("x")
    assert deepest_checkpoint_frames(tmp_path, "hub", 0) == 1000243200


def test_deepest_checkpoint_spans_chained_segments(tmp_path):
    root = tmp_path / "hub_seed0" / "tracker"
    for run, step in (("seg1", 200048640), ("seg2", 2000486400)):
        models = root / run / "models"
        models.mkdir(parents=True)
        (models / f"model_step_{step}.pt").write_text("x")
    assert deepest_checkpoint_frames(tmp_path, "hub", 0) == 2000486400


def test_missing_tree_is_none_not_zero(tmp_path):
    assert deepest_checkpoint_frames(tmp_path, "absent", 0) is None
    assert ArmRow(arm="a", section=1, label="x").frames_text() == "-"


def _eval_payload(sr, local_mm, global_mm):
    # Success rate is an aggregate; MPJPE is success-only and therefore lives
    # under `successful_metrics` as a {mean, count} entry.
    return {
        "aggregate": {"tracking_success_rate": sr},
        "successful_metrics": {
            "tracking_mpjpe_mm": {"mean": local_mm, "count": 10},
            "tracking_mpjpe_g_mm": {"mean": global_mm, "count": 10},
        },
    }


def test_scored_rows_keep_the_deepest_and_read_both_metric_homes(tmp_path):
    for frames, sr in ((200048640, 0.5), (2000486400, 0.9)):
        (tmp_path / f"hub_seed0_clean_f{frames}.json").write_text(
            json.dumps(_eval_payload(sr, 24.0, 85.0))
        )
    rows = scored_rows(tmp_path)
    assert rows["hub"]["frames"] == 2000486400
    assert rows["hub"]["success_rate"] == pytest.approx(0.9)
    # The regression this guards: reading MPJPE from `aggregate` returns None
    # and the table prints an empty column beside a populated success rate.
    assert rows["hub"]["mpjpe_local_mm"] == pytest.approx(24.0)
    assert rows["hub"]["mpjpe_global_mm"] == pytest.approx(85.0)


def test_at_frames_pins_the_table_to_one_budget(tmp_path):
    # The regression this guards: arms train at different speeds, so taking
    # each arm's deepest row compares a 2.6B checkpoint against a 2.0B one.
    (tmp_path / "hub_seed0_clean_f2000486400.json").write_text(
        json.dumps(_eval_payload(0.90, 24.0, 92.0))
    )
    (tmp_path / "hub_seed0_clean_f2600140800.json").write_text(
        json.dumps(_eval_payload(0.93, 23.0, 88.0))
    )
    (tmp_path / "other_seed0_clean_f2000486400.json").write_text(
        json.dumps(_eval_payload(0.80, 30.0, 200.0))
    )
    unpinned = scored_rows(tmp_path)
    assert unpinned["hub"]["frames"] == 2600140800

    pinned = scored_rows(tmp_path, at_frames=2000486400)
    assert pinned["hub"]["frames"] == 2000486400
    assert pinned["hub"]["success_rate"] == pytest.approx(0.90)
    assert set(pinned) == {"hub", "other"}


def test_at_frames_omits_an_arm_that_lacks_that_checkpoint(tmp_path):
    (tmp_path / "slow_seed0_clean_f400097280.json").write_text(
        json.dumps(_eval_payload(0.5, 40.0, 300.0))
    )
    assert scored_rows(tmp_path, at_frames=2000486400) == {}


def test_scored_rows_accept_a_plain_scalar_metric(tmp_path):
    payload = _eval_payload(0.9, 24.0, 85.0)
    payload["successful_metrics"]["tracking_mpjpe_mm"] = 24.0
    (tmp_path / "hub_seed0_clean_f100.json").write_text(json.dumps(payload))
    assert scored_rows(tmp_path)["hub"]["mpjpe_local_mm"] == pytest.approx(24.0)


def test_scored_rows_report_missing_mpjpe_as_none_not_zero(tmp_path):
    (tmp_path / "hub_seed0_clean_f100.json").write_text(
        json.dumps({"aggregate": {"tracking_success_rate": 0.9}})
    )
    row = scored_rows(tmp_path)["hub"]
    assert row["success_rate"] == pytest.approx(0.9)
    assert row["mpjpe_local_mm"] is None


def test_scored_rows_survive_a_corrupt_file(tmp_path):
    (tmp_path / "hub_seed0_clean_f200048640.json").write_text("{not json")
    assert scored_rows(tmp_path) == {}


def test_parse_slurm_states_folds_stages_into_one_arm():
    text = (
        "latent-star-v2-g2_mlp-s0-pretrain COMPLETED\n"
        "latent-star-v2-g2_mlp-s0-lowlevel1 RUNNING\n"
        "latent-star-v2-g2_mlp-s0-lowlevel2 PENDING\n"
        "unrelated-job RUNNING\n"
    )
    assert parse_slurm_states(text) == {"g2_mlp": {"COMPLETED", "RUNNING", "PENDING"}}


def test_parse_slurm_states_handles_arm_names_containing_dashes():
    text = "camp-with-dashes-g4_h5-s0-lowlevel1 RUNNING\n"
    assert "g4_h5" in parse_slurm_states(text)


@pytest.mark.parametrize(
    ("states", "has_score", "frames", "expected"),
    [
        (None, True, None, "SCORED"),
        ({"RUNNING", "PENDING"}, False, None, "pretraining"),
        ({"RUNNING"}, False, 1_000_000_000, "training"),
        ({"PENDING"}, False, None, "pending"),
        ({"COMPLETED"}, False, 5_000_000_000, "trained"),
        ({"FAILED", "COMPLETED"}, False, None, "completed/failed"),
        (None, False, None, "unknown"),
        (None, False, 1_000_000_000, "trained"),
    ],
)
def test_arm_status_collapses_stage_states(states, has_score, frames, expected):
    assert arm_status(states, has_score, frames) == expected


def test_a_failed_stage_is_never_hidden_behind_a_running_one():
    # RUNNING wins for the headline, but the failure must still be visible
    # somewhere, so the arm is not reported as merely pending.
    assert arm_status({"RUNNING", "FAILED"}, False, None) == "pretraining"
    assert arm_status({"FAILED"}, False, None) == "failed"


def test_build_sections_groups_orders_and_puts_the_default_first(tmp_path):
    path = _campaign(
        tmp_path,
        {
            "g3_cont128": {
                "vars": {
                    "section": 3,
                    "section_label": "w 128",
                    "section_group": "latent prior",
                }
            },
            "hub": {"vars": {"section": 1, "section_label": "chunk"}},
            "g2_endpoint": {"vars": {"section": 1, "section_label": "endpoint"}},
            "untagged": {"vars": {}},
        },
    )
    sections = build_sections(load_campaign_arms(path))
    assert [s.number for s in sections] == [1, 3]
    assert sections[0].rows[0].arm == "hub"
    assert sections[0].rows[0].is_default
    assert [r.arm for r in sections[1].rows] == ["g3_cont128"]
    # An arm with no `section` is omitted rather than dumped into a bucket.
    assert all(r.arm != "untagged" for s in sections for r in s.rows)


def test_render_marks_the_default_and_shows_dashes_for_unknowns(tmp_path):
    path = _campaign(
        tmp_path, {"hub": {"vars": {"section": 1, "section_label": "chunk"}}}
    )
    text = render(build_sections(load_campaign_arms(path)))
    assert "**hub**" in text
    assert "(default)" in text
    assert "| - | - | - | - |" in text
