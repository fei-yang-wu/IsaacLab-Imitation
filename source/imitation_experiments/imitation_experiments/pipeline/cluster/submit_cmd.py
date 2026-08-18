"""The ``submit`` verb: execute a confirmed plan, record what was submitted."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

from imitation_experiments.paper.common import PipelineError, utc_now
from imitation_experiments.paths import REPO_ROOT

from .planfile import (
    PLAN_SCHEMA_VERSION,
    atomic_write_json,
    git_seal_payload,
    load_plan,
)
from .remote import (
    build_workspace_archive,
    sbatch_parsable,
    ssh_run,
    ssh_upload,
    sync_workspace_archive,
)
from .slurm import validate_dependency


def _scancel(login: str, job_ids: list[str]) -> None:
    if not job_ids:
        return
    quoted = " ".join(shlex.quote(job_id) for job_id in job_ids)
    try:
        ssh_run(login, f"scancel {quoted}")
    except PipelineError as exc:
        print(f"[SUBMIT] WARNING: best-effort scancel failed: {exc}")


def cmd_submit(args: argparse.Namespace) -> int:
    record, plan_dir = load_plan(Path(args.plan))
    plan_sha: str = record["plan_sha"]
    if args.confirm != plan_sha:
        print("[SUBMIT] confirmation sha does not match this plan.")
        print(f"[SUBMIT] expected: --confirm {plan_sha}")
        return 2
    if record["preflight"]["status"] == "failed":
        print("[SUBMIT] refusing: this plan's preflight FAILED. Fix and re-plan.")
        return 2

    sealed = record["sealed"]
    profile: dict[str, Any] = sealed["profile"]
    login: str = profile["login"]
    remote_plan_dir: str = record["remote"]["plan_dir"]

    drift = git_seal_payload() != sealed["git"]
    if drift and not args.allow_drift:
        print(
            "[SUBMIT] refusing: the working tree changed since this plan was sealed "
            "(commit, diff, or untracked files). Re-plan, or pass --allow-drift to "
            "ship the CURRENT tree under the old plan sha."
        )
        return 2
    if drift:
        print("[SUBMIT] WARNING: submitting with git drift; recording drift=true.")

    existing = sorted(plan_dir.glob("submission-*.json"))
    if existing and not args.allow_resubmit:
        print(f"[SUBMIT] refusing: plan already submitted ({existing[-1].name}).")
        print("[SUBMIT] pass --allow-resubmit to submit again on purpose.")
        return 2

    archive_local = plan_dir / "workspace.tar.gz"
    print("[SUBMIT] packing workspace archive...")
    build_workspace_archive(REPO_ROOT, archive_local)
    print(f"[SUBMIT] archive: {archive_local.stat().st_size / 1e6:.1f} MB")
    archive_sha, store_path, reused = sync_workspace_archive(
        login,
        archive_local,
        control_root=record["remote"]["control_root"],
        remote_plan_dir=remote_plan_dir,
    )
    if reused:
        print(f"[SUBMIT] workspace already on the cluster, reused: {store_path}")
    else:
        print(f"[SUBMIT] uploaded + verified workspace: {store_path}")

    log_dirs = {job["directives"]["log_dir"] for job in sealed["jobs"]}
    for log_dir in sorted(log_dirs):
        ssh_run(login, f"mkdir -p {shlex.quote(log_dir)}")
    for job in sealed["jobs"]:
        stage = job["stage"]
        for rel_name in (f"batch_{stage}.sh", f"job_env.{stage}.resolved.sh"):
            ssh_upload(login, plan_dir / rel_name, f"{remote_plan_dir}/{rel_name}")

    submitted: list[dict[str, Any]] = []
    ids_by_stage: dict[str, str] = {}
    stamp = utc_now().translate(str.maketrans("", "", ":-Z")).replace("T", "-")
    for job in sealed["jobs"]:
        stage = job["stage"]
        dependency = None
        if job["depends_on"]:
            kind = str(job.get("dependency_kind", "afterok"))
            dependency = f"{kind}:{ids_by_stage[job['depends_on']]}"
            validate_dependency(dependency)
        try:
            job_id = sbatch_parsable(
                login,
                f"{remote_plan_dir}/batch_{stage}.sh",
                chdir=remote_plan_dir,
                dependency=dependency,
            )
        except PipelineError as exc:
            print(f"[SUBMIT] sbatch failed for stage '{stage}': {exc}")
            _scancel(login, [entry["slurm_job_id"] for entry in submitted])
            failure_record = _submission_record(
                record,
                archive_sha,
                submitted,
                drift=drift,
                status="failed",
                archive_path=store_path,
            )
            atomic_write_json(
                plan_dir / f"submission-{stamp}-failed.json", failure_record
            )
            return 1
        ids_by_stage[stage] = job_id
        submitted.append(
            {
                "stage": stage,
                "slurm_job_id": job_id,
                "job_name": job["job_name"],
                "dependency": dependency,
                "slurm_log_path": job["slurm_log_path"],
            }
        )
        dep_note = f" ({dependency})" if dependency else ""
        print(f"[SUBMIT] stage {stage}: job {job_id}{dep_note}")

    submission = _submission_record(
        record,
        archive_sha,
        submitted,
        drift=drift,
        status="submitted",
        archive_path=store_path,
    )
    local_record = plan_dir / f"submission-{stamp}.json"
    atomic_write_json(local_record, submission)
    ssh_upload(login, local_record, f"{remote_plan_dir}/{local_record.name}")
    print(f"[SUBMIT] submission record: {local_record}")
    return 0


def _submission_record(
    plan_record: dict[str, Any],
    archive_sha: str,
    jobs: list[dict[str, Any]],
    *,
    drift: bool,
    status: str,
    archive_path: str,
) -> dict[str, Any]:
    sealed = plan_record["sealed"]
    # Field names align with the legacy schema_version-1 records written by
    # docker/cluster/submit_job_slurm_bones_pipeline.sh.
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "cluster_submission",
        "status": status,
        "plan_sha": plan_record["plan_sha"],
        "plan_id": plan_record["plan_id"],
        "submitted_at_utc": utc_now(),
        "drift": drift,
        "cluster": {
            "profile": sealed["profile"]["name"],
            "login": sealed["profile"]["login"],
        },
        "remote_plan_dir": plan_record["remote"]["plan_dir"],
        "workspace_archive_sha256": archive_sha,
        "workspace_archive_path": archive_path,
        "workspace_sync_mode": "content_addressed_store",
        "jobs": jobs,
    }
