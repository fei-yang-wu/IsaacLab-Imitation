"""Typed Slurm directives and deterministic batch-script rendering.

One adapter serves every profile. The rendered script reproduces the proven
PACE job body (docker/cluster/submit_job_slurm_pace.sh): extract the uploaded
workspace archive into compute-local storage, install the per-stage frozen env
file, and hand off to run_singularity.sh. The dependency spec is deliberately
NOT a directive: job IDs are unknown at plan time, so ``submit`` passes
``--dependency`` on the sbatch command line and the script stays
content-stable (hashable) across the chain.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from imitation_experiments.paper.common import PipelineError

_ARRAY_RE = re.compile(r"^([0-9]+)-([0-9]+)(%[1-9][0-9]*)?$")
_DEPENDENCY_RE = re.compile(r"^(afterok|afterany):[0-9]+(:[0-9]+)*$")
_TIME_RE = re.compile(r"^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$")
_MEM_RE = re.compile(r"^[0-9]+[KMGT]?$")
_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class SlurmDirectives:
    job_name: str
    log_dir: str
    time_limit: str
    cpus_per_task: int
    gres: str
    mem: str | None = None
    mem_per_gpu: str | None = None
    account: str | None = None
    qos: str | None = None
    partition: str | None = None
    constraint: str | None = None
    nodelist: str | None = None
    exclude: str | None = None
    array: str | None = None
    nodes: int = 1
    ntasks: int = 1


def normalize_gres(gres: str) -> str:
    if not gres:
        raise PipelineError("gres must be non-empty")
    return gres if gres.startswith("gpu:") else f"gpu:{gres}"


def validate_directives(d: SlurmDirectives) -> None:
    if not _JOB_NAME_RE.match(d.job_name):
        raise PipelineError(
            f"job name may contain only letters, digits, '.', '_', '-': '{d.job_name}'"
        )
    if not d.log_dir.startswith("/"):
        raise PipelineError(f"Slurm log_dir must be absolute, got '{d.log_dir}'")
    if not _TIME_RE.match(d.time_limit):
        raise PipelineError(f"time limit must be [D-]HH:MM:SS, got '{d.time_limit}'")
    if (d.mem is None) == (d.mem_per_gpu is None):
        raise PipelineError("exactly one of mem / mem_per_gpu must be set")
    for label, mem_value in (("mem", d.mem), ("mem_per_gpu", d.mem_per_gpu)):
        if mem_value is not None and not _MEM_RE.match(mem_value):
            raise PipelineError(
                f"{label} must match ^[0-9]+[KMGT]?$, got '{mem_value}'"
            )
    if d.array is not None:
        match = _ARRAY_RE.match(d.array)
        if not match:
            raise PipelineError(
                f"array must use START-END or START-END%MAX_PARALLEL, got '{d.array}'"
            )
        if int(match.group(1)) > int(match.group(2)):
            raise PipelineError(f"array start must be <= end, got '{d.array}'")
    if d.cpus_per_task < 1 or d.nodes < 1 or d.ntasks < 1:
        raise PipelineError("cpus_per_task, nodes, and ntasks must be >= 1")
    normalize_gres(d.gres)


def validate_dependency(spec: str) -> None:
    if not _DEPENDENCY_RE.match(spec):
        raise PipelineError(
            f"dependency must be afterok/afterany with numeric job IDs, got '{spec}'"
        )


def _directive_lines(d: SlurmDirectives) -> list[str]:
    output_pattern = (
        f"{d.log_dir}/{d.job_name}_%A_%a.log"
        if d.array
        else f"{d.log_dir}/{d.job_name}_%j.log"
    )
    lines = [
        f"#SBATCH --job-name={d.job_name}",
        f"#SBATCH --output={output_pattern}",
        f"#SBATCH --error={output_pattern}",
    ]
    optional = (
        ("--account", d.account),
        ("--partition", d.partition),
        ("--qos", d.qos),
        ("--constraint", d.constraint),
        ("--nodelist", d.nodelist),
        ("--exclude", d.exclude),
        ("--array", d.array),
    )
    lines.extend(f"#SBATCH {flag}={value}" for flag, value in optional if value)
    lines.append(f"#SBATCH --nodes={d.nodes}")
    lines.append(f"#SBATCH --ntasks={d.ntasks}")
    lines.append(f"#SBATCH --cpus-per-task={d.cpus_per_task}")
    if d.mem is not None:
        lines.append(f"#SBATCH --mem={d.mem}")
    if d.mem_per_gpu is not None:
        lines.append(f"#SBATCH --mem-per-gpu={d.mem_per_gpu}")
    lines.append(f"#SBATCH --gres={normalize_gres(d.gres)}")
    lines.append(f"#SBATCH --time={d.time_limit}")
    # Deliver SIGTERM to every job step 5 minutes before the walltime kill.
    # Training runs route SIGTERM through their interrupt path and write a
    # final resume checkpoint, so a walltime-segmented chain loses minutes,
    # not a save_interval, at each segment boundary.
    lines.append("#SBATCH --signal=TERM@300")
    return lines


# W&B caps a run's tags at 64 characters and RLOpt adds a
# `logdir:<19-char timestamp>_wandb-<run id>` tag, which leaves 31 characters
# for the id itself. The random token and its separator take 7 of them.
_WANDB_RUN_ID_MAX = 31
_WANDB_RUN_ID_TOKEN_CHARS = 7


def _render_run_id_block(state_file: str) -> str:
    """Shell that pins one chain's W&B run id, creating it on first use."""
    quoted_state = shlex.quote(state_file)
    base_max = _WANDB_RUN_ID_MAX - _WANDB_RUN_ID_TOKEN_CHARS
    return f"""
# One W&B run per chain: generated once, then read by every resume.
env_file="$extracted_workspace/docker/cluster/job_env.resolved.sh"
declared_run_id="$(sed -n 's/^export WANDB_RUN_ID=//p' "$env_file" | tail -n 1)"
if [ -n "$declared_run_id" ]; then
    run_id_state={quoted_state}
    if [ -s "$run_id_state" ]; then
        resolved_run_id="$(tr -d '[:space:]' < "$run_id_state")"
        echo "[INFO] W&B run id from $run_id_state: $resolved_run_id"
    else
        run_id_base="$(printf '%s' "$declared_run_id" | cut -c1-{base_max})"
        resolved_run_id="${{run_id_base}}-$(tr -dc 'a-z0-9' < /dev/urandom | head -c 6)"
        mkdir -p "$(dirname "$run_id_state")"
        printf '%s' "$resolved_run_id" > "$run_id_state"
        echo "[INFO] W&B run id created: $resolved_run_id -> $run_id_state"
    fi
    # A later export wins; run_singularity.sh sources this file top to bottom.
    printf 'export WANDB_RUN_ID=%s\\n' "$resolved_run_id" >> "$env_file"
fi
"""


def render_batch_script(
    d: SlurmDirectives,
    *,
    remote_plan_dir: str,
    stage: str,
    job_args: list[str] | tuple[str, ...],
    job_tmpdir_root: str,
    container_profile: str = "isaac-lab-base",
    wandb_run_id_state_file: str | None = None,
) -> str:
    """Deterministic batch script for one stage of a plan.

    ``wandb_run_id_state_file`` is a host path inside the arm's own output
    tree. When the stage env declares ``WANDB_RUN_ID``, the first job of a
    chain appends a short random token to it and records the result there;
    every later job (a walltime resume, a chained segment) reads the same
    file, so the whole chain stays one W&B run and the evaluation sidecar
    keeps attaching to it. W&B refuses a run id that was ever deleted
    (``error 410``), which killed job 5580302 -- deleting this file is how a
    chain is deliberately rotated onto a fresh id.
    """
    validate_directives(d)
    if not remote_plan_dir.startswith("/"):
        raise PipelineError(
            f"remote plan dir must be absolute, got '{remote_plan_dir}'"
        )
    quoted_plan_dir = shlex.quote(remote_plan_dir)
    quoted_env_file = shlex.quote(f"{remote_plan_dir}/job_env.{stage}.resolved.sh")
    quoted_profile = shlex.quote(container_profile)
    quoted_args = " ".join(shlex.quote(str(a)) for a in job_args)
    quoted_tmp_root = shlex.quote(job_tmpdir_root)
    run_id_block = (
        _render_run_id_block(wandb_run_id_state_file) if wandb_run_id_state_file else ""
    )
    body = f"""#!/bin/bash

{chr(10).join(_directive_lines(d))}

set -euo pipefail
export PATH="/opt/slurm/current/bin:$PATH"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Job: ${{SLURM_JOB_ID:-unknown}}"
echo "[INFO] Plan dir: {remote_plan_dir}"
nvidia-smi || true

bootstrap_root={quoted_tmp_root}/isaaclab-bootstrap-${{SLURM_JOB_ID:-$$}}
rm -rf "$bootstrap_root"
mkdir -p "$bootstrap_root"
echo "[INFO] Extracting workspace archive into compute-local storage."
tar -xzf {quoted_plan_dir}/workspace.tar.gz -C "$bootstrap_root"
extracted_workspace="$bootstrap_root/isaaclab-submission-${{SLURM_JOB_ID:-$$}}"
mv "$bootstrap_root/workspace" "$extracted_workspace"

# Frozen per-stage env: run_singularity.sh sources ONLY this file when present.
cp {quoted_env_file} "$extracted_workspace/docker/cluster/job_env.resolved.sh"
{run_id_block}
sed 's/^/[ENV] /' "$extracted_workspace/docker/cluster/job_env.resolved.sh"

# stdbuf line-buffers output so failures are not swallowed by block buffering.
set +e
stdbuf -oL -eL bash "$extracted_workspace/docker/cluster/run_singularity.sh" \\
    "$extracted_workspace" {quoted_profile} {quoted_args}
job_status=$?
set -e
rm -rf "$bootstrap_root"

echo "[INFO] GPU status after job"
nvidia-smi || true
exit $job_status
"""
    return body
