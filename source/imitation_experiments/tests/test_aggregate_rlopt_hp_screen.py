"""Tests for the RLOpt hyperparameter-screen aggregator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from imitation_experiments.lowlevel.aggregate_rlopt_hp_screen import (
    ScreenError,
    arm_record_from_wandb_run,
    discover_arms,
    load_scalar_history,
    main,
    read_scalar_csv,
    render_markdown,
    require_matched_geometry,
    score_arms,
    tail_mean,
    warn_rate_gain_with_tracking_loss,
)


GEOMETRY = {
    "num_envs": 12288,
    "rollout_steps": 12,
    "frames_per_batch": 147456,
    "max_iterations": 340,
    "total_frames": 50135040,
    "seed": 0,
}


def write_arm(
    screen_root: Path,
    name: str,
    *,
    curves: dict[str, list[float]],
    geometry: dict | None = None,
    overrides: list[str] | None = None,
    wall_time_s: int = 1800,
) -> Path:
    """Materialise one arm directory in the layout the launcher produces."""
    arm_dir = screen_root / name
    scalars = arm_dir / "2026-08-02_00-00-00_run-abc123" / "IPMD_x" / "scalars"
    for metric, values in curves.items():
        path = scalars / f"{metric}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                f"{(i + 1) * GEOMETRY['frames_per_batch']},{v}\n"
                for i, v in enumerate(values)
            )
        )
    (arm_dir / "arm.json").write_text(
        json.dumps(
            {
                "arm": name,
                "description": f"{name} description",
                "overrides": overrides or [],
                "geometry": geometry or GEOMETRY,
                "wall_time_s": wall_time_s,
                "exit_status": 0,
            }
        )
    )
    return arm_dir


def test_read_scalar_csv_skips_torn_and_non_finite_lines(tmp_path):
    path = tmp_path / "lr.csv"
    path.write_text("100,0.001\n200,nan\n300,\n400,0.002\nnot-a-row\n500,0.0")
    assert read_scalar_csv(path) == [(100, 0.001), (400, 0.002), (500, 0.0)]


def test_load_scalar_history_joins_metrics_on_step(tmp_path):
    arm_dir = write_arm(
        tmp_path,
        "b0_baseline",
        curves={"train/lr": [1e-4, 2e-4], "episode/length": [10.0, 20.0]},
    )
    history = load_scalar_history(arm_dir)
    assert [point["step"] for point in history] == [147456, 294912]
    assert history[0]["train/lr"] == pytest.approx(1e-4)
    assert history[0]["episode/length"] == pytest.approx(10.0)
    assert history[1]["episode/length"] == pytest.approx(20.0)


def test_load_scalar_history_requires_a_scalars_tree(tmp_path):
    arm_dir = tmp_path / "b0_baseline"
    arm_dir.mkdir()
    with pytest.raises(ScreenError, match="no scalars/"):
        load_scalar_history(arm_dir)


def test_tail_mean_averages_only_the_trailing_fraction():
    history = [{"m": float(i)} for i in range(10)]
    # Trailing 20% of 10 points is the last 2: mean of 8 and 9.
    assert tail_mean(history, "m", 0.2) == pytest.approx(8.5)
    assert tail_mean(history, "m", 1.0) == pytest.approx(4.5)


def test_tail_mean_reports_nan_for_a_metric_no_arm_logged():
    assert math.isnan(tail_mean([{"a": 1.0}], "absent", 1.0))


def test_tail_mean_resolves_the_v2_mpjpe_alias():
    history = [{"Metrics/reference/mpjpe_mm": 42.0}]
    assert tail_mean(history, "mpjpe_mm", 1.0) == pytest.approx(42.0)
    legacy = [{"Metrics/mpjpe_mm": 38.0}]
    assert tail_mean(legacy, "mpjpe_mm", 1.0) == pytest.approx(38.0)


def test_tail_mean_rejects_an_out_of_range_fraction():
    with pytest.raises(ScreenError, match="tail_fraction"):
        tail_mean([{"m": 1.0}], "m", 0.0)


def test_mismatched_frame_budget_is_an_error_not_a_warning(tmp_path):
    write_arm(tmp_path, "b0_baseline", curves={"train/lr": [1e-4]})
    write_arm(
        tmp_path,
        "a1_short",
        curves={"train/lr": [1e-4]},
        geometry={**GEOMETRY, "total_frames": 25000000},
    )
    arms = discover_arms(tmp_path)
    with pytest.raises(ScreenError, match="matched screen"):
        require_matched_geometry(arms)


def test_differing_rollout_length_at_a_matched_budget_is_allowed(tmp_path):
    """The r12-vs-r24 question is the point, not a mistake to block.

    Both arms see the same 50M frames; only the batching differs.
    """
    write_arm(tmp_path, "b0_baseline", curves={"train/lr": [1e-4]})
    write_arm(
        tmp_path,
        "a8_r24_matched",
        curves={"train/lr": [1e-4]},
        geometry={**GEOMETRY, "rollout_steps": 24, "max_iterations": 170},
    )
    arms = discover_arms(tmp_path)
    geometry = require_matched_geometry(arms)
    assert geometry["total_frames"] == GEOMETRY["total_frames"]
    assert "rollout_steps" not in geometry


def test_updates_per_m_frames_is_epochs_over_minibatch(tmp_path):
    write_arm(
        tmp_path,
        "b0_baseline",
        curves={"train/lr": [1e-4]},
        overrides=["agent.loss.mini_batch_size=18432"],
    )
    write_arm(
        tmp_path,
        "a1_updates_2x",
        curves={"train/lr": [1e-4]},
        overrides=["agent.loss.mini_batch_size=9216"],
    )
    write_arm(
        tmp_path,
        "a7_epochs_3",
        curves={"train/lr": [1e-4]},
        overrides=["agent.loss.mini_batch_size=18432", "agent.loss.epochs=3"],
    )
    by_name = {a["arm"]: a for a in score_arms(discover_arms(tmp_path), 1.0)}
    assert by_name["b0_baseline"]["updates_per_m_frames"] == pytest.approx(
        271.3, abs=0.1
    )
    assert by_name["a1_updates_2x"]["updates_per_m_frames"] == pytest.approx(
        542.5, abs=0.1
    )
    assert by_name["a7_epochs_3"]["updates_per_m_frames"] == pytest.approx(
        162.8, abs=0.1
    )


def test_updates_per_m_frames_is_nan_when_the_minibatch_is_unpinned(tmp_path):
    write_arm(tmp_path, "b0_baseline", curves={"train/lr": [1e-4]}, overrides=[])
    (scored,) = score_arms(discover_arms(tmp_path), 1.0)
    assert math.isnan(scored["updates_per_m_frames"])


def test_render_markdown_reports_rollout_and_update_density(tmp_path):
    write_arm(
        tmp_path,
        "b0_baseline",
        curves={"episode/length": [100.0]},
        overrides=["agent.loss.mini_batch_size=18432"],
    )
    write_arm(
        tmp_path,
        "a8_r24_matched",
        curves={"episode/length": [100.0]},
        geometry={**GEOMETRY, "rollout_steps": 24},
        overrides=["agent.loss.mini_batch_size=18432"],
    )
    arms = discover_arms(tmp_path)
    report = render_markdown(
        score_arms(arms, 1.0), require_matched_geometry(arms), "b0_baseline", 1.0
    )
    assert "| steps |" in report
    assert "upd/Mf" in report
    # Same update density, different rollout length -- the isolation a8 exists for.
    assert "| 24 | 271" in report
    assert "| 12 | 271" in report


def test_score_arms_reports_lr_geomean_and_spread(tmp_path):
    write_arm(tmp_path, "b0_baseline", curves={"train/lr": [1e-5, 1e-3]})
    (scored,) = score_arms(discover_arms(tmp_path), tail_fraction=1.0)
    assert scored["scores"]["lr_geomean"] == pytest.approx(1e-4)
    assert scored["scores"]["lr_spread"] == pytest.approx(100.0)


def test_render_markdown_reports_deltas_against_the_baseline(tmp_path):
    write_arm(
        tmp_path,
        "b0_baseline",
        curves={"episode/length": [100.0], "Metrics/reference/mpjpe_mm": [50.0]},
    )
    write_arm(
        tmp_path,
        "a5_entropy_1e3",
        curves={"episode/length": [150.0], "Metrics/reference/mpjpe_mm": [40.0]},
        overrides=["agent.ppo.entropy_coeff=0.001"],
    )
    arms = discover_arms(tmp_path)
    report = render_markdown(
        score_arms(arms, 1.0), require_matched_geometry(arms), "b0_baseline", 1.0
    )
    assert "+50.0%" in report  # episode length
    assert "-20.0%" in report  # mpjpe
    assert "agent.ppo.entropy_coeff=0.001" in report


def test_render_markdown_tolerates_a_missing_baseline(tmp_path):
    write_arm(tmp_path, "a5_entropy_1e3", curves={"episode/length": [150.0]})
    arms = discover_arms(tmp_path)
    report = render_markdown(
        score_arms(arms, 1.0), require_matched_geometry(arms), "b0_baseline", 1.0
    )
    assert "is absent" in report
    assert "Change vs" not in report


def test_discover_arms_requires_at_least_one_arm(tmp_path):
    with pytest.raises(ScreenError, match="no \\*/arm.json"):
        discover_arms(tmp_path)


def test_main_refuses_to_overwrite_an_existing_report(tmp_path):
    write_arm(tmp_path, "b0_baseline", curves={"episode/length": [100.0]})
    out = tmp_path / "screen.md"
    out.write_text("previous aggregation")
    with pytest.raises(ScreenError, match="refusing to overwrite"):
        main(["--screen_root", str(tmp_path), "--out", str(out)])
    assert out.read_text() == "previous aggregation"


def test_main_writes_a_report(tmp_path, capsys):
    write_arm(
        tmp_path,
        "b0_baseline",
        curves={"episode/length": [10.0, 100.0], "train/lr": [1e-5, 1e-4]},
    )
    out = tmp_path / "reports" / "screen.md"
    assert main(["--screen_root", str(tmp_path), "--out", str(out)]) == 0
    report = out.read_text()
    assert "# RLOpt hyperparameter screen" in report
    assert "12288 envs, 50135040 frames per arm" in report
    assert "not a result" in report
    capsys.readouterr()


class _FakeWandbRun:
    """Minimal stand-in for a wandb Run.

    Only the attributes the aggregator actually reads. Using a fake rather than
    a recorded fixture keeps the test offline and makes the config shape the
    aggregator depends on explicit.
    """

    def __init__(self, exp_name, history, *, num_envs=12288, frames_per_batch=147456):
        self.id = f"id_{exp_name}"
        self.name = exp_name
        self.state = "finished"
        self.config = {
            "logger": {"exp_name": exp_name},
            "env": {"num_envs": num_envs},
            "collector": {"frames_per_batch": frames_per_batch},
            "seed": 0,
        }
        self._history = history

    def history(self, samples=None, pandas=True):
        # The aggregator asks for the non-pandas row form and caps the sample
        # count; assert that here so the stub cannot drift from the caller.
        assert pandas is False
        assert samples and samples > 0
        return list(self._history)[:samples]


def test_wandb_arm_record_matches_the_on_disk_shape():
    run = _FakeWandbRun(
        "rlopt_hp_screen_20260802_b0_baseline",
        [
            {"_step": 147456, "episode/length": 5.0, "train/lr": 1e-3},
            {"_step": 294912, "episode/length": 9.0, "train/lr": 1e-4},
        ],
    )
    arm = arm_record_from_wandb_run(run, arm_prefix="rlopt_hp_screen_20260802_")

    assert arm["arm"] == "b0_baseline"
    # rollout_steps is derived, not logged: frames_per_batch / num_envs.
    assert arm["geometry"]["rollout_steps"] == 12
    assert arm["geometry"]["num_envs"] == 12288
    # The history is re-keyed from wandb's _step to the aggregator's step so the
    # scoring code is shared with the CSV path verbatim.
    assert arm["history"][0]["step"] == 147456
    assert tail_mean(arm["history"], "episode/length", 0.5) == pytest.approx(9.0)


def test_wandb_total_frames_is_what_the_arm_saw_not_what_it_was_asked_to_run():
    """A truncated arm must not be rankable against a complete one.

    total_frames comes from the last logged step, so an arm that died at 20M is
    geometry-mismatched against one that finished 50M and gets rejected rather
    than quietly compared.
    """
    complete = _FakeWandbRun("s_b0", [{"_step": 50135040, "episode/length": 400.0}])
    truncated = _FakeWandbRun("s_a1", [{"_step": 20000000, "episode/length": 300.0}])

    arms = [
        arm_record_from_wandb_run(complete, arm_prefix="s_"),
        arm_record_from_wandb_run(truncated, arm_prefix="s_"),
    ]
    assert arms[0]["geometry"]["total_frames"] == 50135040
    assert arms[1]["geometry"]["total_frames"] == 20000000
    with pytest.raises(ScreenError, match="matched screen"):
        require_matched_geometry(arms)


def test_wandb_run_without_history_is_an_error():
    run = _FakeWandbRun("s_b0", [])
    with pytest.raises(ScreenError, match="no logged history"):
        arm_record_from_wandb_run(run, arm_prefix="s_")


def test_main_requires_exactly_one_screen_source(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "out.md")])
    with pytest.raises(SystemExit):
        main(
            [
                "--screen_root",
                str(tmp_path),
                "--wandb_group",
                "rlopt-hparam-search",
                "--out",
                str(tmp_path / "out.md"),
            ]
        )


def _scored(arm, rate, mpjpe):
    return {"arm": arm, "scores": {"return_per_min": rate, "mpjpe_mm": mpjpe}}


def test_rate_gain_with_worse_tracking_is_flagged():
    """The termination-curriculum trap: better rate, worse MPJPE."""
    scored = [_scored("b0_baseline", 0.495, 71.59), _scored("b5_curr", 0.892, 99.18)]
    out = warn_rate_gain_with_tracking_loss(scored, "b0_baseline")
    assert any("Rate gained, tracking lost" in line for line in out)
    assert any("b5_curr" in line and "99.18" in line for line in out)


def test_rate_gain_with_better_tracking_is_not_flagged():
    """A genuine win -- better rate AND better MPJPE -- passes silently."""
    scored = [_scored("b0_baseline", 0.495, 71.59), _scored("b4_silu", 0.528, 66.76)]
    assert warn_rate_gain_with_tracking_loss(scored, "b0_baseline") == []


def test_no_flag_without_a_baseline_or_without_mpjpe():
    assert warn_rate_gain_with_tracking_loss([_scored("x", 1.0, 50.0)], "absent") == []
    nan_mpjpe = [_scored("b0_baseline", 0.4, float("nan")), _scored("x", 0.9, 99.0)]
    assert warn_rate_gain_with_tracking_loss(nan_mpjpe, "b0_baseline") == []
