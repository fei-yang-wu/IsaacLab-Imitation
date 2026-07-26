"""Tests for BONES-SEED multi-seed result aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from aggregate_bones_seed_multiseed_results import METRICS, aggregate_runs


GOALS = ["kick", "stoop"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run(
    root: Path,
    *,
    seed: int,
    latent_offset: float,
    paper_complete: bool = True,
    checkpoint_hash: str = "same-checkpoint",
) -> Path:
    run_root = root / f"seed_{seed}"
    config = {
        "protocol": "bones_seed_shared_multigoal_language_v1",
        "goals": GOALS,
        "goal_count": len(GOALS),
        "demo_rows_per_goal": 1000,
        "rollout_rows_per_goal": 1000,
        "skip_pretrained_closed_loop": False,
        "expected_demo_rows_per_interface": 2000,
        "expected_rollout_rows_per_interface": 2000,
        "eval_steps": 1000,
        "seed": seed,
        "planner": {"model_size": "medium", "pretrain_updates": 2000},
        "interfaces": {
            "latent_skill": {"target_dim": 256},
            "full_body_trajectory": {"target_dim": 670},
        },
        "causal_state_dim": 930,
        "language": {"embedding_dim": 384},
        "submission_gates": {
            "fresh_preparation_checked": True,
            "low_level_gates_checked": True,
            "low_level": {
                "latent_success_rate": 0.8,
                "vanilla_success_rate": 1.0,
            },
        },
        "paper_protocol_complete": paper_complete,
        "workflow_source_sha256": {"runner.py": "source-hash"},
        "input_artifacts": {"vanilla_tracker_checkpoint": {"sha256": checkpoint_hash}},
    }
    _write_json(run_root / "run_config.json", config)

    rows: list[dict[str, object]] = []
    for interface in ("latent_skill", "full_body_trajectory"):
        for goal_index, goal in enumerate(GOALS):
            base = 10.0 + seed + goal_index
            value = base + latent_offset if interface == "latent_skill" else base
            rows.append(
                row := {
                    "interface": interface,
                    "goal": goal,
                    "seed": seed,
                    "steps_run": 1000,
                    "return_sum_mean": value,
                    "survival_steps_mean": 900.0 + value,
                    "tracking_success_rate": 0.5 + value / 100.0,
                    "done_rate": 0.5 - value / 100.0,
                }
            )
            for metric_index, metric in enumerate(METRICS):
                row.setdefault(metric, value + metric_index / 1000.0)
    summary = {
        "protocol": config["protocol"],
        "seed": seed,
        "goal_count": len(GOALS),
        "goals": GOALS,
        "per_goal": rows,
        "per_goal_by_stage": [
            {
                **row,
                "stage": stage,
                **(
                    {metric: float(row[metric]) - 0.5 for metric in METRICS}
                    if stage == "pretrained_demonstration"
                    else {}
                ),
            }
            for row in rows
            for stage in (
                "pretrained_demonstration",
                "finetuned_planner_rollout",
            )
        ],
        "interface_specs": {
            "latent_skill": {
                "target_dim": 256,
                "planner_rate_hz": 5.0,
                "parameter_count": 100,
                "values_per_second": 1280.0,
                "float32_bits_per_second": 40960.0,
            },
            "full_body_trajectory": {
                "target_dim": 670,
                "planner_rate_hz": 5.0,
                "parameter_count": 110,
                "values_per_second": 3350.0,
                "float32_bits_per_second": 107200.0,
            },
        },
        "push_perturbation_protocol": {
            "enabled": True,
            "mode": "interval",
            "interval_range_s": [1.0, 3.0],
        },
        "termination_cause_counts_across_goals": {
            "latent_skill": {"anchor_pos": 1, "reference_finished": 1},
            "full_body_trajectory": {"anchor_pos": 0, "reference_finished": 2},
        },
    }
    summary_path = run_root / "summary" / "final_results.json"
    _write_json(summary_path, summary)
    summary_csv = run_root / "summary" / "final_results.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit_path = run_root / "protocol_checks" / "multigoal_language_audit.json"
    _write_json(
        audit_path,
        {
            "passed": True,
            "paper_protocol_complete": paper_complete,
            "smoke_only": not paper_complete,
        },
    )
    manifest_path = run_root / "comparison_manifest.json"
    _write_json(manifest_path, {"protocol": config["protocol"]})
    cluster_submission_path = run_root / "cluster_submission.json"
    _write_json(
        cluster_submission_path,
        {
            "schema_version": 1,
            "seed": seed,
            "workspace_archive_sha256": "a" * 64,
            "repo_sync_manifest_sha256": "b" * 64,
            "jobs": {
                "prepare": 100 + seed * 10,
                "rollout_array": 101 + seed * 10,
                "finetune": 102 + seed * 10,
                "final_eval_array": 103 + seed * 10,
                "summarize": 104 + seed * 10,
            },
        },
    )
    artifacts = {
        "comparison_manifest": manifest_path,
        "summary_json": summary_path,
        "summary_csv": summary_csv,
        "protocol_audit": audit_path,
        "cluster_submission": cluster_submission_path,
    }
    _write_json(
        run_root / "stages" / "summarize.json",
        {
            "status": "complete",
            "stage": "summarize",
            "workflow_source_sha256": config["workflow_source_sha256"],
            "artifacts": {
                name: {"path": str(path), "kind": "file", "sha256": _sha256(path)}
                for name, path in artifacts.items()
            },
        },
    )
    return run_root


def test_aggregates_paired_seed_goal_results(tmp_path: Path) -> None:
    run_roots = [
        _make_run(tmp_path, seed=0, latent_offset=1.0),
        _make_run(tmp_path, seed=1, latent_offset=2.0),
        _make_run(tmp_path, seed=2, latent_offset=3.0),
    ]

    result = aggregate_runs(
        run_roots,
        output_dir=tmp_path / "aggregate",
        bootstrap_samples=200,
    )

    assert result["seeds"] == [0, 1, 2]
    assert result["paper_protocol_complete"] is True
    assert [item["seed"] for item in result["source_run_artifacts"]] == [0, 1, 2]
    assert all(
        len(item["artifacts"]["cluster_submission"]["sha256"]) == 64
        for item in result["source_run_artifacts"]
    )
    assert result["interface_specs"]["latent_skill"]["values_per_second"] == 1280.0
    assert result["push_perturbation_protocol"]["enabled"] is True
    assert result["termination_cause_counts_across_seeds"]["latent_skill"] == {
        "anchor_pos": 3,
        "reference_finished": 3,
    }
    paired = result["paired_statistics"]["return_sum_mean"]
    assert paired["mean_difference"] == pytest.approx(2.0)
    assert paired["seed_mean_differences"] == {"0": 1.0, "1": 2.0, "2": 3.0}
    assert paired["latent_better_pair_fraction"] == 1.0
    before_after = result["before_after_statistics"]["return_sum_mean"]
    assert before_after["interfaces"]["latent_skill"][
        "mean_change_finetuned_minus_pretrained"
    ] == pytest.approx(0.5)
    normalized = result["oracle_normalized_survival"]
    assert normalized["interfaces"]["latent_skill"][
        "oracle_tracking_success_rate"
    ] == pytest.approx(0.8)
    assert normalized["interfaces"]["full_body_trajectory"][
        "oracle_tracking_success_rate"
    ] == pytest.approx(1.0)
    assert (tmp_path / "aggregate" / "multiseed_results.json").is_file()
    assert (tmp_path / "aggregate" / "multiseed_per_goal.csv").is_file()
    assert (tmp_path / "aggregate" / "multiseed_paired_differences.csv").is_file()
    assert (tmp_path / "aggregate" / "multiseed_before_after.csv").is_file()
    paper_table = tmp_path / "aggregate" / "multiseed_results.md"
    assert paper_table.is_file()
    paper_text = paper_table.read_text(encoding="utf-8")
    assert "# BONES-SEED latent versus explicit planner results" in paper_text
    assert "| Survival without falling |" in paper_text
    assert "| DiffSR latent | 256 |" in paper_text
    assert "Tracking success relative to each low-level oracle" in paper_text
    aggregate_manifest = json.loads(
        (tmp_path / "aggregate" / "aggregation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert aggregate_manifest["schema_version"] == 1
    assert set(aggregate_manifest["outputs"]) == {
        "results_json",
        "per_goal_csv",
        "paired_differences_csv",
        "before_after_csv",
        "paper_markdown",
    }
    assert all(
        len(artifact["sha256"]) == 64
        for artifact in aggregate_manifest["outputs"].values()
    )


def test_refuses_to_overwrite_an_existing_aggregation(tmp_path: Path) -> None:
    run_roots = [_make_run(tmp_path, seed=seed, latent_offset=1.0) for seed in range(3)]
    output_dir = tmp_path / "aggregate"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        aggregate_runs(
            run_roots,
            output_dir=output_dir,
            bootstrap_samples=20,
        )


def test_rejects_mismatched_input_artifacts(tmp_path: Path) -> None:
    run_roots = [
        _make_run(tmp_path, seed=0, latent_offset=1.0),
        _make_run(
            tmp_path,
            seed=1,
            latent_offset=1.0,
            checkpoint_hash="different-checkpoint",
        ),
    ]

    with pytest.raises(ValueError, match="input artifacts differ"):
        aggregate_runs(
            run_roots,
            output_dir=tmp_path / "aggregate",
            minimum_seeds=2,
            expected_seeds=(0, 1),
            bootstrap_samples=20,
        )


def test_rejects_non_paper_run_by_default(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path, seed=0, latent_offset=1.0, paper_complete=False)

    with pytest.raises(ValueError, match="not paper-protocol complete"):
        aggregate_runs(
            [run_root],
            output_dir=tmp_path / "aggregate",
            minimum_seeds=1,
            expected_seeds=(0,),
            bootstrap_samples=20,
        )


def test_detects_changed_summarize_artifact(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path, seed=0, latent_offset=1.0)
    (run_root / "summary" / "final_results.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash changed"):
        aggregate_runs(
            [run_root],
            output_dir=tmp_path / "aggregate",
            minimum_seeds=1,
            expected_seeds=(0,),
            bootstrap_samples=20,
        )


def test_stage_artifact_paths_may_move_with_run_root(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path, seed=0, latent_offset=1.0)
    stage_path = run_root / "stages" / "summarize.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    for artifact in stage["artifacts"].values():
        relative = Path(artifact["path"]).relative_to(run_root)
        artifact["path"] = f"/container/output/{relative}"
    _write_json(stage_path, stage)

    result = aggregate_runs(
        [run_root],
        output_dir=tmp_path / "aggregate",
        minimum_seeds=1,
        expected_seeds=(0,),
        bootstrap_samples=20,
    )

    assert result["seeds"] == [0]


def test_rejects_a_different_three_seed_grid(tmp_path: Path) -> None:
    run_roots = [
        _make_run(tmp_path, seed=seed, latent_offset=1.0) for seed in (0, 1, 3)
    ]

    with pytest.raises(ValueError, match=r"exactly match \[0, 1, 2\]"):
        aggregate_runs(
            run_roots,
            output_dir=tmp_path / "aggregate",
            bootstrap_samples=20,
        )


def test_aggregate_keeps_pairs_with_finite_early_failure_metrics(
    tmp_path: Path,
) -> None:
    run_roots = [_make_run(tmp_path, seed=seed, latent_offset=1.0) for seed in range(3)]
    for run_root in run_roots:
        summary_path = run_root / "summary" / "final_results.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary["per_goal"]:
            if row["interface"] == "latent_skill" and row["goal"] == "kick":
                row["planner_latency_ms"] = None
        for row in summary["per_goal_by_stage"]:
            if row["interface"] == "latent_skill" and row["goal"] == "kick":
                row["planner_latency_ms"] = None
        _write_json(summary_path, summary)
        stage_path = run_root / "stages" / "summarize.json"
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        stage["artifacts"]["summary_json"]["sha256"] = _sha256(summary_path)
        _write_json(stage_path, stage)

    result = aggregate_runs(
        run_roots,
        output_dir=tmp_path / "aggregate",
        bootstrap_samples=50,
    )

    latent = result["interface_statistics"]["planner_latency_ms"]["latent_skill"]
    paired = result["paired_statistics"]["planner_latency_ms"]
    assert latent["value_count"] == 3
    assert paired["pair_count"] == 3
    assert paired["seed_count_with_pairs"] == 3
