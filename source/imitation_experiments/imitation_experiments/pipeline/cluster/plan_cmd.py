"""The ``plan`` verb: resolve, validate, preflight, freeze. Never submits."""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

from imitation_experiments.paper.common import PipelineError, sha256_file, utc_now
from imitation_experiments.paths import REPO_ROOT

from .config import (
    ClusterProfile,
    ResolvedJobSet,
    ResolvedStage,
    load_campaign,
    load_profile,
    resolved_env,
)
from .envfile import render_job_env
from .planfile import (
    PLAN_SCHEMA_VERSION,
    build_sealed,
    compute_plan_sha,
    sealed_job_entry,
    write_plan,
)
from .preflight import CheckResult, run_preflight
from .slurm import SlurmDirectives, render_batch_script

_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _short(name: str, limit: int = 24) -> str:
    return _NAME_SANITIZE_RE.sub("-", name)[:limit].strip("-")


def _stage_directives(
    profile: ClusterProfile, stage: ResolvedStage, *, job_name: str
) -> SlurmDirectives:
    slurm = profile.slurm
    return SlurmDirectives(
        job_name=job_name,
        log_dir=slurm.log_dir,
        time_limit=stage.time_limit or slurm.time_limit,
        cpus_per_task=stage.cpus_per_task or slurm.cpus_per_task,
        gres=stage.gres or slurm.gres,
        mem=stage.mem or (slurm.mem if not slurm.mem_per_gpu else None),
        mem_per_gpu=None if (stage.mem or slurm.mem) else slurm.mem_per_gpu,
        account=slurm.account,
        qos=slurm.qos,
        partition=slurm.partition,
        nodes=slurm.nodes,
        ntasks=slurm.ntasks,
    )


def _select_stages(
    jobset: ResolvedJobSet, only_stage: str | None
) -> tuple[ResolvedStage, ...]:
    if only_stage is None:
        return jobset.stages
    # Comma-separated names keep multi-segment chains submittable against an
    # encoder already on disk (e.g. "lowlevel1,lowlevel2,lowlevel3,lowlevel4");
    # dependencies between the kept stages survive as afterok links.
    wanted = [name.strip() for name in only_stage.split(",") if name.strip()]
    by_name = {s.name: s for s in jobset.stages}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise PipelineError(
            f"--only-stage {missing} not in arm '{jobset.arm}' "
            f"(stages: {[s.name for s in jobset.stages]})"
        )
    selected = [by_name[name] for name in wanted]
    kept = {s.name for s in selected}
    # A dependency on a stage that is not part of this plan cannot become an
    # afterok, so it is dropped (e.g. resubmitting lowlevel against an encoder
    # already on disk).
    return tuple(
        replace(s, depends_on=s.depends_on if s.depends_on in kept else None)
        for s in selected
    )


def cmd_plan(args: argparse.Namespace) -> int:
    campaign_path = Path(args.campaign).resolve()
    jobset = load_campaign(
        campaign_path, arm=args.arm, seed=args.seed, overrides=list(args.set or [])
    )
    profile = load_profile(args.profile or jobset.profile_name)
    stages = _select_stages(jobset, args.only_stage)

    campaign_short = _short(jobset.campaign_name)
    stamp = utc_now().translate(str.maketrans("", "", ":-Z")).replace("T", "-")

    jobs = []
    stage_directives: list[SlurmDirectives] = []
    rendered: dict[str, str] = {}
    # Two-pass: seal semantic content first, then render files (whose headers
    # embed the sha) and record their hashes OUTSIDE the seal.
    for stage in stages:
        job_name = f"{campaign_short}-{jobset.arm}-s{jobset.seed}-{stage.name}"
        directives = _stage_directives(profile, stage, job_name=job_name)
        stage_directives.append(directives)
        env = resolved_env(profile, jobset, stage)
        jobs.append(
            sealed_job_entry(
                stage=stage.name,
                job_name=job_name,
                depends_on=stage.depends_on,
                dependency_kind=stage.dependency_kind,
                directives=directives,
                job_args=stage.args,
                env=env,
                slurm_log_path=f"{directives.log_dir}/{job_name}_%j.log",
            )
        )

    sealed = build_sealed(
        jobset=jobset,
        profile=profile,
        cli_overrides=list(args.set or []),
        jobs=jobs,
    )
    plan_sha = compute_plan_sha(sealed)
    plan_id = f"{campaign_short}-{jobset.arm}-s{jobset.seed}-{stamp}-{plan_sha[:8]}"
    remote_plan_dir = f"{profile.control_root}/plans/{plan_id}"

    out_root = (
        Path(args.out_root) if args.out_root else REPO_ROOT / "logs/cluster_control"
    )
    plan_dir = out_root / jobset.campaign_name / plan_id
    if plan_dir.exists():
        raise PipelineError(f"plan directory already exists: {plan_dir}")
    plan_dir.mkdir(parents=True)

    for stage, job, directives in zip(stages, jobs, stage_directives, strict=True):
        rendered[f"batch_{stage.name}.sh"] = render_batch_script(
            directives,
            remote_plan_dir=remote_plan_dir,
            stage=stage.name,
            job_args=stage.args,
            job_tmpdir_root=profile.job_tmpdir_root,
        )
        rendered[f"job_env.{stage.name}.resolved.sh"] = render_job_env(
            job["env"], plan_sha=plan_sha, stage=stage.name
        )
    for rel_name, content in rendered.items():
        (plan_dir / rel_name).write_text(content)

    checks: list[CheckResult] = []
    preflight_status = "skipped"
    if not args.skip_preflight:
        checks = run_preflight(profile, _with_stages(jobset, stages))
        preflight_status = "passed" if all(c.ok for c in checks) else "failed"

    record = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "cluster_plan",
        "plan_sha": plan_sha,
        "plan_id": plan_id,
        "created_at_utc": utc_now(),
        "remote": {
            "control_root": profile.control_root,
            "plan_dir": remote_plan_dir,
            "archive_path": f"{remote_plan_dir}/workspace.tar.gz",
        },
        "sealed": sealed,
        "artifacts": {name: sha256_file(plan_dir / name) for name in sorted(rendered)},
        "preflight": {
            "status": preflight_status,
            "results": [
                {"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks
            ],
        },
    }
    write_plan(plan_dir, record)

    print(f"[PLAN] {plan_id}")
    print(f"[PLAN] local dir:  {plan_dir}")
    print(f"[PLAN] remote dir: {remote_plan_dir}")
    for job in jobs:
        dep = (
            f" {job.get('dependency_kind', 'afterok')}:{job['depends_on']}"
            if job["depends_on"]
            else ""
        )
        print(f"[PLAN] stage {job['stage']}: {job['job_name']}{dep}")
    for check in checks:
        marker = "OK  " if check.ok else "FAIL"
        print(f"[PREFLIGHT] {marker} {check.name}: {check.detail}")
    if preflight_status == "failed":
        print(
            "[PLAN] preflight FAILED; plan written for inspection, not submittable as-is."
        )
        print(f"PLAN_SHA={plan_sha}")
        return 2
    print(
        "[PLAN] next: python -m imitation_experiments.pipeline.cluster submit "
        f"--plan {plan_dir} --confirm {plan_sha}"
    )
    print(f"PLAN_SHA={plan_sha}")
    return 0


def _with_stages(
    jobset: ResolvedJobSet, stages: tuple[ResolvedStage, ...]
) -> ResolvedJobSet:
    return replace(jobset, stages=stages)
