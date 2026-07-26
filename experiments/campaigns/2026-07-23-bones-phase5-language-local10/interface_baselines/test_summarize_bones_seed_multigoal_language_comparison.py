"""Tests for the paper-facing BONES-SEED per-seed summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from summarize_bones_seed_multigoal_language_comparison import (
    ROLLOUT_METRICS,
    main,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _raw_summary(interface: str, *, target_dim: int, parameter_count: int) -> dict:
    return {
        "metadata": {
            "planner_target_dim": target_dim,
            "planner_metadata": {
                "parameter_count": parameter_count,
                "num_samples": 200,
                "checkpoint_metadata": {"num_samples": 100},
                "sample_metadata": {"planner_rate_hz": 5.0},
            },
            "push_perturbation": {
                "enabled": True,
                "mode": "interval",
                "interval_range_s": [1.0, 3.0],
            },
        },
        "aggregate": {
            "return_sum_mean": 1.0,
            "survival_steps_mean": 1000.0,
            "tracking_success_rate": 0.9,
            "done_rate": 0.1,
        },
        "metrics": {
            name: {"mean": float(index + 1), "std": 0.0, "count": 10}
            for index, name in enumerate(ROLLOUT_METRICS)
        },
        "planner_inference_latency_ms": {
            "unit": "ms",
            "scope": "high_level_planner_forward_only",
            "total_call_count": 101,
            "warmup_calls_excluded": 1,
            "measured_call_count": 100,
            "mean": 2.5,
        },
        "steps_run": 1000,
        "interface": interface,
    }


def _make_run(tmp_path: Path, *, inconsistent_latent: bool = False) -> Path:
    run_root = tmp_path / "run"
    goals = ["kick", "stoop"]
    _write_json(run_root / "run_config.json", {"seed": 3, "goals": goals})
    final_summaries: dict[str, list[str]] = {}
    pretrained_summaries: dict[str, list[str]] = {}
    for interface, target_dim, parameter_count in (
        ("latent_skill", 256, 154_512),
        ("full_body_trajectory", 670, 156_304),
    ):
        paths: list[str] = []
        pretrained_paths: list[str] = []
        for index, goal in enumerate(goals):
            path = run_root / "raw" / interface / f"{goal}.json"
            count = parameter_count + (1 if inconsistent_latent and index == 1 else 0)
            _write_json(
                path,
                _raw_summary(interface, target_dim=target_dim, parameter_count=count),
            )
            paths.append(str(path))
            pretrained_path = run_root / "raw_pretrained" / interface / f"{goal}.json"
            pretrained = _raw_summary(
                interface, target_dim=target_dim, parameter_count=parameter_count
            )
            pretrained["aggregate"]["return_sum_mean"] = 0.5
            _write_json(pretrained_path, pretrained)
            pretrained_paths.append(str(pretrained_path))
        final_summaries[interface] = paths
        pretrained_summaries[interface] = pretrained_paths
    _write_json(
        run_root / "comparison_manifest.json",
        {
            "final_summaries": final_summaries,
            "pretrained_summaries": pretrained_summaries,
        },
    )
    return run_root


def test_summary_keeps_tracking_metrics_and_interface_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _make_run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["summary", "--run_root", str(run_root)])

    main()

    payload = json.loads(
        (run_root / "summary" / "final_results.json").read_text(encoding="utf-8")
    )
    assert payload["per_goal"][0]["tracking_mpjpe_mm"] == 1.0
    assert payload["per_goal"][0]["action_delta_l2"] == 8.0
    assert payload["per_goal"][0]["planner_latency_ms"] == 2.5
    assert payload["interface_specs"]["latent_skill"] == {
        "target_dim": 256,
        "planner_rate_hz": 5.0,
        "parameter_count": 154_512,
        "pretrain_sample_count": 100,
        "final_training_sample_count": 200,
        "planner_rollout_sample_count": 100,
        "values_per_second": 1280.0,
        "float32_bits_per_second": 40_960.0,
    }
    assert (
        payload["interface_specs"]["full_body_trajectory"]["values_per_second"]
        == 3350.0
    )
    assert payload["push_perturbation_protocol"]["enabled"] is True
    assert len(payload["per_goal_by_stage"]) == 8
    assert (
        payload["mean_change_after_planner_rollout"]["latent_skill"]["return_sum_mean"]
        == 0.5
    )


def test_summary_rejects_interface_metadata_that_changes_by_goal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _make_run(tmp_path, inconsistent_latent=True)
    monkeypatch.setattr(sys, "argv", ["summary", "--run_root", str(run_root)])

    with pytest.raises(ValueError, match="differs by goal"):
        main()


def test_summary_accepts_preliminary_latent_only_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _make_run(tmp_path)
    comparison_path = run_root / "comparison_manifest.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["interfaces"] = ["latent_skill"]
    comparison["final_summaries"] = {
        "latent_skill": comparison["final_summaries"]["latent_skill"]
    }
    comparison["pretrained_summaries"] = {
        "latent_skill": comparison["pretrained_summaries"]["latent_skill"]
    }
    _write_json(comparison_path, comparison)
    monkeypatch.setattr(sys, "argv", ["summary", "--run_root", str(run_root)])

    main()

    payload = json.loads(
        (run_root / "summary" / "final_results.json").read_text(encoding="utf-8")
    )
    assert set(payload["interface_specs"]) == {"latent_skill"}
    assert len(payload["per_goal"]) == 2


def test_summary_writes_unavailable_early_failure_metrics_as_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = _make_run(tmp_path)
    comparison = json.loads(
        (run_root / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    final_path = Path(comparison["final_summaries"]["latent_skill"][0])
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["steps_run"] = 1
    final["metrics"].pop("action_delta_l2")
    final["metrics"].pop("tracking_acceleration_distance_mps2")
    final["planner_inference_latency_ms"]["mean"] = None
    _write_json(final_path, final)
    monkeypatch.setattr(sys, "argv", ["summary", "--run_root", str(run_root)])

    main()

    payload = json.loads(
        (run_root / "summary" / "final_results.json").read_text(encoding="utf-8")
    )
    row = next(
        row
        for row in payload["per_goal"]
        if row["interface"] == "latent_skill" and row["goal"] == "kick"
    )
    assert row["action_delta_l2"] is None
    assert row["tracking_acceleration_distance_mps2"] is None
    assert row["planner_latency_ms"] is None
    assert (
        payload["aggregate_across_goals"]["latent_skill"]["action_delta_l2"][
            "goal_count"
        ]
        == 1
    )
