"""Gate: status/logs/cancel reconcile squeue with sacct and never mutate without --yes."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import pytest

from imitation_experiments.pipeline.cluster.__main__ import main
from imitation_experiments.pipeline.cluster.status_cmd import parse_sacct, parse_squeue


def _submission_record(tmp_path: Path) -> Path:
    record = {
        "schema_version": 2,
        "kind": "cluster_submission",
        "status": "submitted",
        "plan_sha": "f" * 64,
        "plan_id": "demo-a1-s0-x",
        "drift": False,
        "cluster": {"profile": "t", "login": "ice"},
        "remote_plan_dir": str(tmp_path / "remote/plans/p1"),
        "workspace_archive_sha256": "0" * 64,
        "workspace_sync_mode": "archive",
        "jobs": [
            {
                "stage": "pretrain",
                "slurm_job_id": "100",
                "job_name": "demo-a1-s0-pretrain",
                "dependency": None,
                "slurm_log_path": str(
                    tmp_path / "slurm_logs/demo-a1-s0-pretrain_%j.log"
                ),
            },
            {
                "stage": "lowlevel",
                "slurm_job_id": "101",
                "job_name": "demo-a1-s0-lowlevel",
                "dependency": "afterok:100",
                "slurm_log_path": str(
                    tmp_path / "slurm_logs/demo-a1-s0-lowlevel_%j.log"
                ),
            },
        ],
    }
    path = tmp_path / "submission-20260815-000000.json"
    path.write_text(json.dumps(record))
    return path


def _install_fake_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    cmd_log = tmp_path / "ssh_cmd.log"
    # ssh_run sends the remote script on stdin ("ssh <host> bash -s"), so the
    # fake captures stdin, not argv, then prints its canned body.
    (bin_dir / "ssh").write_text(f'#!/bin/bash\nshift\ncat >> "{cmd_log}"\n{body}\n')
    (bin_dir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return cmd_log


def _run(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = main(argv)
    return rc, buffer.getvalue()


def test_parse_squeue_and_sacct() -> None:
    squeue = "100|RUNNING|1:23:45|node01\n"
    sacct = (
        "101|COMPLETED|02:00:00|0:0\n"
        "101.batch|COMPLETED|02:00:00|0:0\n"
        "101.extern|COMPLETED|02:00:00|0:0\n"
        "102_0|FAILED|00:01:00|1:0\n"
    )
    running = parse_squeue(squeue)
    finished = parse_sacct(sacct)
    assert running["100"]["state"] == "RUNNING"
    assert finished["101"]["exit"] == "0:0"
    assert "101.batch" not in finished
    assert finished["102_0"]["state"] == "FAILED"


def test_status_merges_running_and_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _submission_record(tmp_path)
    _install_fake_ssh(
        tmp_path,
        monkeypatch,
        "echo '100|RUNNING|0:10:00|node01'\n"
        "echo '---SACCT---'\n"
        "echo '101|COMPLETED|01:00:00|0:0'",
    )
    rc, stdout = _run(["status", "--submission", str(record)])
    assert rc == 0, stdout
    assert "pretrain" in stdout and "RUNNING" in stdout
    assert "lowlevel" in stdout and "COMPLETED" in stdout


def test_status_unknown_job_never_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _submission_record(tmp_path)
    _install_fake_ssh(tmp_path, monkeypatch, "echo '---SACCT---'")
    rc, stdout = _run(["status", "--submission", str(record)])
    assert rc == 0
    assert stdout.count("UNKNOWN") == 2


def test_logs_substitutes_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _submission_record(tmp_path)
    cmd_log = _install_fake_ssh(tmp_path, monkeypatch, "echo tailed")
    rc, stdout = _run(["logs", "--submission", str(record), "--stage", "pretrain"])
    assert rc == 0
    assert "tailed" in stdout
    assert "demo-a1-s0-pretrain_100.log" in cmd_log.read_text()
    assert "%j" not in cmd_log.read_text()


def test_logs_requires_stage_when_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _submission_record(tmp_path)
    _install_fake_ssh(tmp_path, monkeypatch, "echo tailed")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        rc = main(["logs", "--submission", str(record)])
    assert rc == 2
    assert "pass --stage" in buffer.getvalue()


def test_cancel_requires_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _submission_record(tmp_path)
    cmd_log = _install_fake_ssh(tmp_path, monkeypatch, "true")
    rc, stdout = _run(["cancel", "--submission", str(record)])
    assert rc == 2
    assert "refusing without --yes" in stdout
    assert not cmd_log.exists()
    rc, stdout = _run(["cancel", "--submission", str(record), "--yes"])
    assert rc == 0
    assert "scancel 100 101" in cmd_log.read_text()
