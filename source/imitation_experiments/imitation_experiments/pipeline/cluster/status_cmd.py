"""The ``status``, ``logs``, and ``cancel`` verbs over stored submission records."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from imitation_experiments.paper.common import PipelineError
from imitation_experiments.paths import REPO_ROOT

from .remote import ssh_run

_SACCT_SUFFIXES = (".batch", ".extern", ".interactive")


def _load_submission(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    if args.submission:
        path = Path(args.submission)
        if path.is_dir():
            candidates = sorted(path.glob("submission-*.json"))
            if not candidates:
                raise PipelineError(f"no submission record under {path}")
            path = candidates[-1]
    else:
        root = REPO_ROOT / "logs/cluster_control"
        candidates = sorted(root.glob("*/*/submission-*.json"))
        if args.campaign:
            candidates = [c for c in candidates if c.parts[-3] == args.campaign]
        if not candidates:
            raise PipelineError(
                f"no submission records found under {root}; pass --submission explicitly"
            )
        path = max(candidates, key=lambda c: c.stat().st_mtime)
    record = json.loads(path.read_text())
    if record.get("kind") != "cluster_submission":
        raise PipelineError(f"not a cluster_submission record: {path}")
    return record, path


def parse_squeue(text: str) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 4:
            parsed[parts[0]] = {
                "state": parts[1],
                "elapsed": parts[2],
                "reason": parts[3],
            }
    return parsed


def parse_sacct(text: str) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        job_id = parts[0]
        if job_id.endswith(_SACCT_SUFFIXES):
            continue
        parsed[job_id] = {"state": parts[1], "elapsed": parts[2], "exit": parts[3]}
    return parsed


def cmd_status(args: argparse.Namespace) -> int:
    record, path = _load_submission(args)
    login = record["cluster"]["login"]
    job_ids = [job["slurm_job_id"] for job in record["jobs"]]
    ids_csv = ",".join(job_ids)
    proc = ssh_run(
        login,
        f"squeue -h -j {shlex.quote(ids_csv)} -o '%i|%T|%M|%R' 2>/dev/null || true; "
        "echo '---SACCT---'; "
        f"sacct -n -P -j {shlex.quote(ids_csv)} "
        "--format=JobID,State,Elapsed,ExitCode 2>/dev/null || true",
    )
    stdout = proc.stdout.decode()
    queue_text, _, acct_text = stdout.partition("---SACCT---")
    running = parse_squeue(queue_text)
    finished = parse_sacct(acct_text)

    print(f"[STATUS] {path}")
    print(f"{'stage':<12} {'job_id':<10} {'state':<12} {'elapsed':<10} {'exit':<6} log")
    for job in record["jobs"]:
        job_id = job["slurm_job_id"]
        if job_id in running:
            entry = running[job_id]
            state, elapsed, exit_code = entry["state"], entry["elapsed"], "-"
        elif job_id in finished:
            entry = finished[job_id]
            state, elapsed, exit_code = entry["state"], entry["elapsed"], entry["exit"]
        else:
            state, elapsed, exit_code = "UNKNOWN", "-", "-"
        log_path = _substitute_log_path(job)
        print(
            f"{job['stage']:<12} {job_id:<10} {state:<12} {elapsed:<10} {exit_code:<6} {log_path}"
        )
    return 0


def _substitute_log_path(job: dict[str, Any]) -> str:
    return (
        job["slurm_log_path"]
        .replace("%j", job["slurm_job_id"])
        .replace("%x", job["job_name"])
    )


def _job_for_stage(record: dict[str, Any], stage: str | None) -> dict[str, Any]:
    jobs = record["jobs"]
    if stage is None:
        if len(jobs) == 1:
            return jobs[0]
        raise PipelineError(
            f"multiple stages in this submission ({[j['stage'] for j in jobs]}); pass --stage"
        )
    for job in jobs:
        if job["stage"] == stage:
            return job
    raise PipelineError(
        f"stage '{stage}' not in submission ({[j['stage'] for j in jobs]})"
    )


def cmd_logs(args: argparse.Namespace) -> int:
    record, _ = _load_submission(args)
    job = _job_for_stage(record, args.stage)
    log_path = _substitute_log_path(job)
    follow = "-F " if args.follow else ""
    proc = ssh_run(
        record["cluster"]["login"],
        f"if [ -f {shlex.quote(log_path)} ]; then tail {follow}-n {int(args.lines)} "
        f"{shlex.quote(log_path)}; else echo '[logs] no log file yet at "
        f"{log_path} — job may still be pending'; fi",
        check=False,
    )
    print(proc.stdout.decode(), end="")
    if proc.stderr:
        print(proc.stderr.decode(), end="")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    record, path = _load_submission(args)
    if args.stage:
        jobs = [_job_for_stage(record, args.stage)]
    else:
        jobs = record["jobs"]
    ids = [job["slurm_job_id"] for job in jobs]
    print(f"[CANCEL] would cancel jobs {ids} from {path}")
    if not args.yes:
        print("[CANCEL] refusing without --yes")
        return 2
    ssh_run(
        record["cluster"]["login"],
        "scancel " + " ".join(shlex.quote(job_id) for job_id in ids),
    )
    print(f"[CANCEL] cancelled {ids}")
    return 0
