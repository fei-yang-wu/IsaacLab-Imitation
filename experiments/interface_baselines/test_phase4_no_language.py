"""Tests for the guarded Phase-4 no-language matrix and launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from audit_focused_causal_interface_comparison import (
    PAPER_ROLLOUT_METRICS,
    _audit_closed_loop_outcome,
)
from aggregate_phase4_no_language_results import aggregate, write_aggregate_outputs
from phase4_no_language_matrix import motion_slug, resolve_task
from validate_phase4_no_language_submission import validate


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, names: list[str]) -> Path:
    _write_json(
        path,
        {
            "dataset": {
                "trajectories": {
                    "lafan1_csv": [
                        {"name": name, "path": f"../npz/{index}.npz"}
                        for index, name in enumerate(names)
                    ]
                }
            }
        },
    )
    return path


def test_matrix_maps_seed_motion_and_rounds_collection_rows(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json", ["walk one", "walk/two"])

    task = resolve_task(
        manifest,
        seeds=[0, 2, 5],
        task_index=3,
        sample_budgets=[1000, 10000, 50000],
        num_envs=16,
    )

    assert task["seed"] == 2
    assert task["motion_name"] == "walk/two"
    assert task["motion_slug"] == motion_slug("walk/two")
    assert task["total_tasks"] == 6
    assert task["planner_decisions_per_env"] == 3125
    assert task["available_demonstration_rows"] == 50000


def test_submission_validator_binds_all_qualified_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json", ["walk", "kick"])
    vanilla_checkpoint = tmp_path / "vanilla.pt"
    latent_checkpoint = tmp_path / "latent.pt"
    skill_checkpoint = tmp_path / "skill.pt"
    vanilla_checkpoint.write_bytes(b"vanilla")
    latent_checkpoint.write_bytes(b"latent")
    skill_checkpoint.write_bytes(b"skill")
    vanilla_audit = tmp_path / "vanilla_audit.json"
    latent_audit = tmp_path / "latent_audit.json"
    equivalence = tmp_path / "equivalence.json"
    _write_json(
        vanilla_audit,
        {
            "protocol_passed": True,
            "oracle_passed": True,
            "success_rate": 0.9,
            "checkpoint_sha256": _sha256(vanilla_checkpoint),
            "manifest_sha256": _sha256(manifest),
            "dataset_path": "/data/vanilla",
        },
    )
    _write_json(
        latent_audit,
        {
            "protocol_passed": True,
            "oracle_passed": True,
            "tracking_success_rate": 0.85,
            "low_level_checkpoint_sha256": _sha256(latent_checkpoint),
            "skill_checkpoint_sha256": _sha256(skill_checkpoint),
            "low_level_skill_binding": {
                "passed": True,
                "low_level_checkpoint_sha256": _sha256(latent_checkpoint),
                "skill_checkpoint_sha256": _sha256(skill_checkpoint),
            },
            "manifest_sha256": _sha256(manifest),
            "dataset_path": "/data/latent",
        },
    )
    _write_json(
        equivalence,
        {
            "passed": True,
            "observed_phases": list(range(10)),
            "missing_phases": [],
            "asynchronous_rephase_exercised": True,
            "policy_state_unchanged": True,
            "checkpoint_sha256": _sha256(vanilla_checkpoint),
            "motion_manifest_sha256": _sha256(manifest),
            "dataset_path": "/data/vanilla",
            "low_level_tracker": {"checkpoint_sha256": _sha256(vanilla_checkpoint)},
        },
    )
    args = argparse.Namespace(
        manifest=manifest,
        vanilla_checkpoint=vanilla_checkpoint,
        latent_checkpoint=latent_checkpoint,
        skill_checkpoint=skill_checkpoint,
        vanilla_audit=vanilla_audit,
        latent_audit=latent_audit,
        equivalence=equivalence,
        expected_latent_dataset_path="/data/latent",
        expected_vanilla_dataset_path="/data/vanilla",
        expected_manifest_sha256=_sha256(manifest),
        expected_motion_count=2,
        minimum_oracle_success=0.8,
    )

    validate(args)
    vanilla_payload = json.loads(vanilla_audit.read_text(encoding="utf-8"))
    vanilla_payload["dataset_path"] = "/data/stale"
    _write_json(vanilla_audit, vanilla_payload)
    with pytest.raises(ValueError, match="vanilla gate failed"):
        validate(args)
    vanilla_payload["dataset_path"] = "/data/vanilla"
    _write_json(vanilla_audit, vanilla_payload)

    payload = json.loads(equivalence.read_text(encoding="utf-8"))
    payload["motion_manifest_sha256"] = "wrong"
    _write_json(equivalence, payload)
    with pytest.raises(ValueError, match="equivalence certificate failed"):
        validate(args)


def test_paper_outcome_audit_requires_metrics_terminations_and_latency() -> None:
    summary = {
        "metadata": {"push_perturbation": {"enabled": True, "mode": "interval"}},
        "aggregate": {
            "tracking_success_rate": 1.0,
            "termination_cause_env_counts": {"reference_finished": 0},
        },
        "metrics": {
            name: {"mean": float(index + 1), "count": 10}
            for index, name in enumerate(PAPER_ROLLOUT_METRICS)
        },
        "planner_inference_latency_ms": {
            "scope": "high_level_planner_forward_only",
            "unit": "ms",
            "warmup_calls_excluded": 1,
            "total_call_count": 10,
            "measured_call_count": 9,
            "mean": 1.0,
        },
        "max_steps": 1000,
        "steps_run": 1000,
        "stop_reason": "max_steps",
        "per_environment": [
            {
                "tracking_success": True,
                "termination_terms": [],
            }
        ],
    }
    errors: list[str] = []

    push = _audit_closed_loop_outcome(
        summary,
        scope="latent",
        expected_steps=1000,
        expected_num_envs=1,
        require_planner_latency=True,
        errors=errors,
    )

    assert errors == []
    assert push["enabled"] is True
    summary["planner_inference_latency_ms"]["measured_call_count"] = 0
    _audit_closed_loop_outcome(
        summary,
        scope="latent",
        expected_steps=1000,
        expected_num_envs=1,
        require_planner_latency=True,
        errors=errors,
    )
    assert any("post-warmup" in error for error in errors)


def test_paper_outcome_audit_accepts_honest_first_step_failure() -> None:
    summary = {
        "metadata": {"push_perturbation": {"enabled": True}},
        "aggregate": {
            "tracking_success_rate": 0.0,
            "termination_cause_env_counts": {
                "reference_finished": 0,
                "bad_orientation": 1,
            },
        },
        "metrics": {
            name: {"mean": 1.0, "count": 1}
            for name in PAPER_ROLLOUT_METRICS
            if name not in {"action_delta_l2", "tracking_acceleration_distance_mps2"}
        },
        "planner_inference_latency_ms": {
            "scope": "high_level_planner_forward_only",
            "unit": "ms",
            "warmup_calls_excluded": 1,
            "total_call_count": 1,
            "measured_call_count": 0,
            "mean": float("nan"),
        },
        "steps_run": 1,
        "stop_reason": "all_envs_done",
        "per_environment": [
            {
                "tracking_success": False,
                "termination_terms": ["bad_orientation"],
            }
        ],
    }
    errors: list[str] = []

    _audit_closed_loop_outcome(
        summary,
        scope="early_failure",
        expected_steps=1000,
        expected_num_envs=1,
        require_planner_latency=True,
        errors=errors,
    )

    assert errors == []


def test_paper_outcome_audit_accepts_failure_before_second_publication() -> None:
    summary = {
        "metadata": {"push_perturbation": {"enabled": True}},
        "aggregate": {
            "tracking_success_rate": 0.0,
            "termination_cause_env_counts": {
                "reference_finished": 0,
                "bad_orientation": 1,
            },
        },
        "metrics": {name: {"mean": 1.0, "count": 2} for name in PAPER_ROLLOUT_METRICS},
        "planner_inference_latency_ms": {
            "scope": "high_level_planner_forward_only",
            "unit": "ms",
            "warmup_calls_excluded": 1,
            "total_call_count": 1,
            "measured_call_count": 0,
            "mean": float("nan"),
        },
        "steps_run": 2,
        "stop_reason": "all_envs_done",
        "per_environment": [
            {
                "tracking_success": False,
                "termination_terms": ["bad_orientation"],
            }
        ],
    }
    errors: list[str] = []

    _audit_closed_loop_outcome(
        summary,
        scope="early_failure",
        expected_steps=1000,
        expected_num_envs=1,
        require_planner_latency=True,
        errors=errors,
    )

    assert errors == []


def test_skynet_launcher_renders_one_seed_motion_array() -> None:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "VANILLA_TRACKER_CHECKPOINT": "logs/qualified/vanilla.pt",
        "LATENT_LOW_LEVEL_CHECKPOINT": "logs/qualified/latent.pt",
        "LATENT_SKILL_CHECKPOINT": "logs/qualified/skill.pt",
        "VANILLA_QUALIFICATION_AUDIT": "logs/qualified/vanilla.json",
        "LATENT_QUALIFICATION_AUDIT": "logs/qualified/latent.json",
        "STREAMED_EQUIVALENCE_CERTIFICATE": "logs/qualified/equivalence.json",
        "EXPECTED_MOTION_COUNT": "2",
        "SEEDS": "0 1 2",
        "SAMPLE_BUDGETS": "1000 10000 50000",
        "MAX_PARALLEL_TASKS": "2",
    }
    result = subprocess.run(
        ["bash", "experiments/interface_baselines/submit_phase4_no_language_skynet.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Phase-4 array covers 6 seed/motion tasks: 0-5%2" in result.stdout
    assert "CLUSTER_SLURM_ARRAY=0-5%2" in result.stdout
    assert "CLUSTER_SLURM_SUBMIT_SCRIPT=phase4" in result.stdout
    assert "CLUSTER_SLURM_SUBMISSION_RECORD_ROOT=" in result.stdout
    assert "MODE=phase4-no-language" in result.stdout


def test_skynet_launcher_rejects_changed_paper_budgets() -> None:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "VANILLA_TRACKER_CHECKPOINT": "logs/qualified/vanilla.pt",
        "LATENT_LOW_LEVEL_CHECKPOINT": "logs/qualified/latent.pt",
        "LATENT_SKILL_CHECKPOINT": "logs/qualified/skill.pt",
        "VANILLA_QUALIFICATION_AUDIT": "logs/qualified/vanilla.json",
        "LATENT_QUALIFICATION_AUDIT": "logs/qualified/latent.json",
        "STREAMED_EQUIVALENCE_CERTIFICATE": "logs/qualified/equivalence.json",
        "SAMPLE_BUDGETS": "1000 5000 50000",
    }

    result = subprocess.run(
        ["bash", "experiments/interface_baselines/submit_phase4_no_language_skynet.sh"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fixes SAMPLE_BUDGETS" in result.stderr


def test_phase4_submitter_records_array_job_and_workspace(tmp_path: Path) -> None:
    source = REPO_ROOT / "docker/cluster/submit_job_slurm_phase4.sh"
    workspace = tmp_path / "workspace"
    cluster_dir = workspace / "docker/cluster"
    cluster_dir.mkdir(parents=True)
    shutil.copy2(source, cluster_dir / source.name)
    generic = cluster_dir / "submit_job_slurm.sh"
    generic.write_text(
        "#!/usr/bin/env bash\necho 'Submitted batch job 777'\n",
        encoding="utf-8",
    )
    generic.chmod(0o755)
    (workspace / "workspace.tar.gz.sha256").write_text(
        f"{'a' * 64}  workspace.tar.gz\n", encoding="utf-8"
    )
    (workspace / "repo_sync_manifest.tsv").write_text(
        "phase4 snapshot\n", encoding="utf-8"
    )
    record_root = tmp_path / "persistent_output"
    env = {
        **os.environ,
        "CLUSTER_SLURM_SUBMISSION_RECORD_ROOT": str(record_root),
        "CLUSTER_SLURM_ARRAY": "0-119%4",
    }

    result = subprocess.run(
        [
            "bash",
            str(cluster_dir / source.name),
            str(workspace),
            "isaac-lab-base",
            "--mode",
            "phase4-no-language",
            "--env",
            "OUTPUT_ROOT=logs/interface_baselines/phase4_no_language_lafan1",
            "--env",
            "SEEDS=0 1 2",
            "--env",
            "SAMPLE_BUDGETS=1000 10000 50000",
            "--env",
            "EXPECTED_MOTION_COUNT=40",
        ],
        cwd=workspace,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    record = json.loads(
        (record_root / "cluster_submission.json").read_text(encoding="utf-8")
    )
    assert record["job_id"] == 777
    assert record["array"] == "0-119%4"
    assert record["seeds"] == "0 1 2"
    assert record["sample_budgets"] == "1000 10000 50000"
    assert record["workspace_archive_sha256"] == "a" * 64


def test_focused_dry_run_routes_existing_vanilla_cache(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "LATENT_LOW_LEVEL_CHECKPOINT": "latent.pt",
        "LATENT_SKILL_CHECKPOINT": "skill.pt",
        "VANILLA_TRACKER_CHECKPOINT": "vanilla.pt",
        "MANIFEST": "one_motion.json",
        "DATASET_PATH": "/data/latent_cache",
        "VANILLA_DATASET_PATH": "/data/vanilla_cache",
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "COLLECT_SAMPLES": "2",
        "SAMPLE_BUDGET": "2",
        "MODEL_SIZE": "tiny",
        "PRETRAIN_UPDATES": "1",
        "FINETUNE_UPDATES": "1",
    }
    result = subprocess.run(
        [
            "bash",
            "experiments/interface_baselines/run_focused_causal_interface_comparison.sh",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--dataset_path /data/vanilla_cache" in result.stdout
    assert "DATASET_PATH=/data/vanilla_cache" in result.stdout
    assert "focused_protocol_audit_2.json" not in result.stdout


def test_phase4_task_runner_dry_run_covers_each_budget(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json", ["walk", "kick"])
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MANIFEST": str(manifest),
        "LATENT_LOW_LEVEL_CHECKPOINT": "latent.pt",
        "LATENT_SKILL_CHECKPOINT": "skill.pt",
        "VANILLA_TRACKER_CHECKPOINT": "vanilla.pt",
        "VANILLA_QUALIFICATION_AUDIT": "vanilla_audit.json",
        "LATENT_QUALIFICATION_AUDIT": "latent_audit.json",
        "STREAMED_EQUIVALENCE_CERTIFICATE": "equivalence.json",
        "EXPECTED_MANIFEST_SHA256": _sha256(manifest),
        "DATASET_PATH": "/data/latent",
        "VANILLA_DATASET_PATH": "/data/vanilla",
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "SEEDS": "0 1",
        "SAMPLE_BUDGETS": "2 3",
        "NUM_ENVS": "2",
        "EXPECTED_MOTION_COUNT": "2",
        "PHASE4_TASK_INDEX": "0",
        "INTERFACE_BASELINE_PYTHON_CMD": sys.executable,
    }

    result = subprocess.run(
        ["bash", "experiments/interface_baselines/run_phase4_no_language_sweep.sh"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "budgets=2 3" in result.stdout
    assert "SAMPLE_BUDGET=2" in result.stdout
    assert "SAMPLE_BUDGET=3" in result.stdout
    assert len(list((tmp_path / "output" / "seed_0").glob("*/task_config.json"))) == 1


def _aggregate_summary(path: Path, success: float, latency: float) -> Path:
    _write_json(
        path,
        {
            "aggregate": {
                "return_sum_mean": success * 10.0,
                "survival_steps_mean": success * 100.0,
                "tracking_success_rate": success,
                "done_rate": 1.0 - success,
            },
            "metrics": {
                name: {"mean": float(index + 1), "count": 2}
                for index, name in enumerate(PAPER_ROLLOUT_METRICS)
            },
            "planner_inference_latency_ms": {"mean": latency},
        },
    )
    return path


def test_phase4_aggregate_uses_seed_motion_task_roots(tmp_path: Path) -> None:
    gate = {"passed": True, "manifest_sha256": "a" * 64}
    _write_json(
        tmp_path / "cluster_submission.json",
        {
            "schema_version": 1,
            "study": "phase4_no_language_sample_efficiency_v1",
            "submitted_at_utc": "2026-07-15T00:00:00Z",
            "output_root": str(tmp_path),
            "cluster_workspace": "/cluster/snapshot",
            "workspace_archive_sha256": "a" * 64,
            "repo_sync_manifest_sha256": "b" * 64,
            "array": "0-1%2",
            "seeds": "0",
            "sample_budgets": "10",
            "motion_count": 2,
            "job_id": 777,
        },
    )
    for motion_index, motion in enumerate(("walk", "kick")):
        task_root = tmp_path / "seed_0" / motion
        _write_json(
            task_root / "task_config.json",
            {"seed": 0, "motion_name": motion, "sample_budgets": [10]},
        )
        _write_json(task_root / "input" / "submission_gate.json", gate)
        planner_rows = {}
        oracle_rows = {}
        for interface, advantage in (
            ("latent_skill", 0.1),
            ("full_body_trajectory", 0.0),
        ):
            pretrained = _aggregate_summary(
                task_root / interface / "pretrained.json",
                0.4 + advantage,
                2.0,
            )
            final = _aggregate_summary(
                task_root / interface / "final.json",
                0.6 + advantage + 0.01 * motion_index,
                1.0 + advantage,
            )
            planner_rows[interface] = {
                "target_spec": {
                    "target_dim": 256 if interface == "latent_skill" else 670
                },
                "parameter_count": 1000,
                "pretrained_summary": str(pretrained),
                "summary": str(final),
            }
            oracle_rows[interface] = {"aggregate": {"tracking_success_rate": 0.9}}
        direct = _aggregate_summary(task_root / "direct.json", 0.95, 0.0)
        _write_json(
            task_root / "protocol_checks" / "focused_protocol_audit_10.json",
            {
                "passed": True,
                "expected": {"seed": 0, "rows_per_stage": 10},
                "planner_rows": planner_rows,
                "oracle_rows": oracle_rows,
                "ceiling": {"summary": str(direct)},
            },
        )

    result = aggregate(
        tmp_path,
        expected_seeds=[0],
        expected_budgets=[10],
        expected_motion_count=2,
        bootstrap_samples=100,
    )

    statistic = result["paired_statistics"]["budget_10/tracking_success_rate"]
    assert statistic["paired_motion_count"] == 2
    assert statistic["mean_latent_minus_explicit"] == pytest.approx(0.1)
    assert len(result["per_motion_rows"]) == 8
    assert result["cluster_submission"]["record"]["job_id"] == 777
    outputs = write_aggregate_outputs(
        result,
        output_dir=tmp_path / "aggregate",
        command="test phase4 aggregate",
    )
    assert all(path.is_file() for path in outputs.values())
    markdown = outputs["paper_markdown"].read_text(encoding="utf-8")
    assert "# Phase-4 no-language sample-efficiency results" in markdown
    assert "| 10 | Tracking success |" in markdown
    manifest = json.loads(outputs["aggregation_manifest"].read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {
        "results_json",
        "per_motion_csv",
        "paper_markdown",
    }

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_aggregate_outputs(
            result,
            output_dir=tmp_path / "aggregate",
            command="test phase4 aggregate",
        )
