"""Tests for the staged BONES-SEED Slurm dependency chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess


PIPELINE_SUBMITTER = (
    Path(__file__).resolve().parents[2]
    / "docker/cluster/submit_job_slurm_bones_pipeline.sh"
)
QUALIFICATION_RUNNER = (
    Path(__file__).resolve().parent / "run_bones_seed_low_level_qualification.sh"
)


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    cluster_dir = workspace / "docker/cluster"
    cluster_dir.mkdir(parents=True)
    shutil.copy2(PIPELINE_SUBMITTER, cluster_dir / PIPELINE_SUBMITTER.name)
    (workspace / "workspace.tar.gz.sha256").write_text(
        f"{'a' * 64}  workspace.tar.gz\n", encoding="utf-8"
    )
    (workspace / "repo_sync_manifest.tsv").write_text(
        "repo snapshot\n", encoding="utf-8"
    )

    call_log = tmp_path / "calls.tsv"
    counter = tmp_path / "counter"
    counter.write_text("100\n", encoding="utf-8")
    generic = cluster_dir / "submit_job_slurm.sh"
    generic.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
job_id=$(cat "${FAKE_COUNTER}")
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \\
    "${job_id}" \\
    "${CLUSTER_SLURM_DEPENDENCY:-}" \\
    "${CLUSTER_SLURM_ARRAY:-}" \\
    "${CLUSTER_SLURM_TIME_LIMIT:-}" \\
    "${CLUSTER_SLURM_JOB_NAME:-}" \\
    "$*" >> "${FAKE_CALL_LOG}"
printf '%s\n' "$((job_id + 1))" > "${FAKE_COUNTER}"
echo "Submitted batch job ${job_id}"
""",
        encoding="utf-8",
    )
    generic.chmod(0o755)
    return workspace, call_log, counter


def _run_pipeline(
    tmp_path: Path,
    *,
    array: str = "0-2%2",
    goal_limit: str = "3",
) -> subprocess.CompletedProcess[str]:
    workspace, call_log, counter = _workspace(tmp_path)
    env = dict(os.environ)
    env.update(
        {
            "CLUSTER_SLURM_PIPELINE_ARRAY": array,
            "CLUSTER_SLURM_DEPENDENCY": "afterok:77",
            "FAKE_CALL_LOG": str(call_log),
            "FAKE_COUNTER": str(counter),
            "CLUSTER_SLURM_SUBMISSION_RECORD_ROOT": str(
                tmp_path / "persistent_output"
            ),
        }
    )
    return subprocess.run(
        [
            "bash",
            str(workspace / "docker/cluster" / PIPELINE_SUBMITTER.name),
            str(workspace),
            "isaac-lab-base",
            "--mode",
            "bones-seed-multigoal-language",
            "--env",
            "OUTPUT_ROOT=logs/paper/bones-seed-seed0",
            "--env",
            f"GOAL_LIMIT={goal_limit}",
            "--env",
            "SEED=0",
        ],
        cwd=workspace,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_pipeline_submits_five_ordered_stages(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path)

    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls.tsv").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 5
    parsed = [line.split("\t") for line in calls]
    assert [(row[0], row[1], row[2], row[4]) for row in parsed] == [
        ("100", "afterok:77", "", "bones-prepare"),
        ("101", "afterok:100", "0-2%2", "bones-rollout"),
        ("102", "afterok:101", "", "bones-finetune"),
        ("103", "afterok:102", "0-2%2", "bones-final-eval"),
        ("104", "afterok:103", "", "bones-summarize"),
    ]
    for stage, row in zip(
        ("prepare", "rollout", "finetune", "final-eval", "summarize"),
        parsed,
        strict=True,
    ):
        assert f"--env PIPELINE_STAGE={stage}" in row[5]
        assert "--env RESUME=1" in row[5]

    record = next(
        workspace_path
        for workspace_path in (tmp_path / "workspace").glob(
            "bones_pipeline_submission_*.txt"
        )
    )
    record_text = record.read_text(encoding="utf-8")
    assert "goal_limit=3" in record_text
    assert "summarize_job_id=104" in record_text
    persistent = json.loads(
        (tmp_path / "persistent_output" / "cluster_submission.json").read_text(
            encoding="utf-8"
        )
    )
    assert persistent["seed"] == 0
    assert persistent["jobs"] == {
        "prepare": 100,
        "rollout_array": 101,
        "finetune": 102,
        "final_eval_array": 103,
        "summarize": 104,
    }
    assert persistent["workspace_archive_sha256"] == "a" * 64


def test_pipeline_rejects_array_goal_count_mismatch(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path, array="0-3", goal_limit="3")

    assert result.returncode == 2
    assert "cover exactly GOAL_LIMIT=3 goals" in result.stderr
    assert not (tmp_path / "calls.tsv").exists()


def test_pipeline_requires_positive_goal_limit(tmp_path: Path) -> None:
    result = _run_pipeline(tmp_path, goal_limit="0")

    assert result.returncode == 2
    assert "requires a positive GOAL_LIMIT" in result.stderr
    assert not (tmp_path / "calls.tsv").exists()


def test_qualification_audits_each_interface_cache() -> None:
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "VANILLA_TRACKER_CHECKPOINT": "/checkpoints/vanilla.pt",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/checkpoints/latent.pt",
            "LATENT_SKILL_CHECKPOINT": "/checkpoints/skill.pt",
            "MANIFEST": "/data/manifest.json",
            "VANILLA_DATASET_PATH": "/data/vanilla_cache",
            "LATENT_DATASET_PATH": "/data/latent_cache",
            "OUTPUT_ROOT": "/tmp/bones-qualification-dry-run",
        }
    )
    result = subprocess.run(
        ["bash", str(QUALIFICATION_RUNNER)],
        cwd=QUALIFICATION_RUNNER.parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    preflight_line = next(
        line
        for line in result.stdout.splitlines()
        if "scripts/audit_bones_seed_phase5.py" in line
    )
    assert "--expected_dataset_path" not in preflight_line
    assert (
        "audit_vanilla_tracker_qualification.py --summary " in result.stdout
        and "--expected_dataset_path /data/vanilla_cache" in result.stdout
    )
    latent_audit_line = next(
        line
        for line in result.stdout.splitlines()
        if "audit_diffsr_latent_qualification.py" in line
    )
    assert "--expected_dataset_path /data/latent_cache" in latent_audit_line
