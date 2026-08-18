"""Plan records: sealed content, content hash, artifact verification.

``plan_sha`` covers the ``sealed`` subtree only — the semantic content of the
submission (resolved profile, stage argv, directives, frozen env dicts, git
state). Rendered artifact hashes, timestamps, and preflight results live
outside the seal, so re-planning an unchanged tree yields the same sha while
tampering with a rendered file is still detected via the ``artifacts`` map.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from imitation_experiments.paper.common import (
    PipelineError,
    canonical_json,
    git_state,
    sha256_file,
    sha256_text,
)
from imitation_experiments.paths import REPO_ROOT

from .config import ClusterProfile, ResolvedJobSet
from .slurm import SlurmDirectives

PLAN_SCHEMA_VERSION = 2


def git_seal_payload() -> dict[str, Any]:
    """Git state extended with a diff hash and untracked listing, so the
    submit-time drift gate also sees uncommitted and untracked changes."""
    payload = dict(git_state())

    def _git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else ""

    payload["diff_sha256"] = sha256_text(_git("diff", "HEAD"))
    payload["status_porcelain"] = _git("status", "--porcelain").strip()
    return payload


def build_sealed(
    *,
    jobset: ResolvedJobSet,
    profile: ClusterProfile,
    cli_overrides: list[str],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    # The remote plan dir embeds the sha itself, so it lives OUTSIDE the seal
    # (top-level "remote" in the record) to avoid a hash circularity.
    return {
        "campaign": {
            "name": jobset.campaign_name,
            "path": jobset.campaign_path,
            "arm": jobset.arm,
            "seed": jobset.seed,
            "cli_overrides": list(cli_overrides),
        },
        "profile": asdict(profile),
        "git": git_seal_payload(),
        "jobs": jobs,
    }


def compute_plan_sha(sealed: dict[str, Any]) -> str:
    return sha256_text(canonical_json(sealed))


def sealed_job_entry(
    *,
    stage: str,
    job_name: str,
    depends_on: str | None,
    dependency_kind: str,
    directives: SlurmDirectives,
    job_args: tuple[str, ...],
    env: dict[str, str],
    slurm_log_path: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "job_name": job_name,
        "depends_on": depends_on,
        "dependency_kind": dependency_kind,
        "directives": asdict(directives),
        "job_args": list(job_args),
        "env": env,
        "slurm_log_path": slurm_log_path,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    partial.replace(path)


def write_plan(plan_dir: Path, record: dict[str, Any]) -> Path:
    plan_path = plan_dir / "plan.json"
    atomic_write_json(plan_path, record)
    return plan_path


def load_plan(plan_ref: Path) -> tuple[dict[str, Any], Path]:
    """Load plan.json (or its directory), re-verify the sha and artifact hashes."""
    plan_path = plan_ref / "plan.json" if plan_ref.is_dir() else plan_ref
    if not plan_path.is_file():
        raise PipelineError(f"plan not found: {plan_path}")
    record = json.loads(plan_path.read_text())
    if (
        record.get("schema_version") != PLAN_SCHEMA_VERSION
        or record.get("kind") != "cluster_plan"
    ):
        raise PipelineError(
            f"not a v{PLAN_SCHEMA_VERSION} cluster_plan record: {plan_path}"
        )
    recomputed = compute_plan_sha(record["sealed"])
    if recomputed != record["plan_sha"]:
        raise PipelineError(
            f"plan sha mismatch in {plan_path}: recorded={record['plan_sha']} recomputed={recomputed}"
        )
    plan_dir = plan_path.parent
    for rel_name, recorded_sha in record.get("artifacts", {}).items():
        artifact = plan_dir / rel_name
        if not artifact.is_file():
            raise PipelineError(f"plan artifact missing: {artifact}")
        actual = sha256_file(artifact)
        if actual != recorded_sha:
            raise PipelineError(
                f"plan artifact modified since planning: {artifact} "
                f"(recorded {recorded_sha[:12]}, actual {actual[:12]})"
            )
    return record, plan_dir
