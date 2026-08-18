"""Gate: the Slurm adapter renders valid, deterministic batch scripts and rejects bad specs.

Semantics mirror the legacy submitters (docker/cluster/submit_job_slurm_pace.sh
and submit_job_slurm.sh): dependency/array regexes, gres normalization, and the
extract-archive-then-run_singularity job body.
"""

from __future__ import annotations

import dataclasses
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from imitation_experiments.paper.common import PipelineError, sha256_text
from imitation_experiments.pipeline.cluster.slurm import (
    SlurmDirectives,
    normalize_gres,
    render_batch_script,
    validate_dependency,
    validate_directives,
)


_BASE_DIRECTIVES = SlurmDirectives(
    job_name="quant-fsq64-s0-pretrain",
    log_dir="/scratch/slurm_logs",
    time_limit="15:59:00",
    cpus_per_task=16,
    gres="gpu:h200:1",
    mem="160G",
    qos="coe-ice",
    partition="ice-gpu",
)


def _directives(**overrides: Any) -> SlurmDirectives:
    return dataclasses.replace(_BASE_DIRECTIVES, **overrides)


def _render(d: SlurmDirectives, job_args: tuple[str, ...] = ("--seed", "0")) -> str:
    return render_batch_script(
        d,
        remote_plan_dir="/scratch/cluster_control/plans/p1",
        stage="pretrain",
        job_args=job_args,
        job_tmpdir_root="/tmp",
    )


def test_directive_block_matches_pace_semantics() -> None:
    script = _render(_directives())
    assert "#SBATCH --gres=gpu:h200:1" in script
    assert "#SBATCH --qos=coe-ice" in script
    assert "#SBATCH --partition=ice-gpu" in script
    assert (
        "#SBATCH --output=/scratch/slurm_logs/quant-fsq64-s0-pretrain_%j.log" in script
    )
    assert "#SBATCH --time=15:59:00" in script
    assert "--account" not in script  # empty optionals are omitted, not blank
    assert "run_singularity.sh" in script


def test_array_switches_log_pattern_and_validates() -> None:
    script = _render(_directives(array="0-9%4"))
    assert "#SBATCH --array=0-9%4" in script
    assert "_%A_%a.log" in script
    with pytest.raises(PipelineError, match="array start must be <= end"):
        validate_directives(_directives(array="9-0"))
    with pytest.raises(PipelineError, match="START-END"):
        validate_directives(_directives(array="1,2,3"))


def test_dependency_validation_matrix() -> None:
    validate_dependency("afterok:12345")
    validate_dependency("afterany:1:2:3")
    for bad in ("after:1", "afterok:", "afterok:abc", "12345", "afterok:1;rm -rf /"):
        with pytest.raises(PipelineError, match="dependency"):
            validate_dependency(bad)


def test_bare_gres_normalized_and_mem_exclusivity() -> None:
    assert normalize_gres("h200:1") == "gpu:h200:1"
    assert normalize_gres("gpu:a40:2") == "gpu:a40:2"
    with pytest.raises(PipelineError, match="exactly one of mem"):
        validate_directives(_directives(mem=None))
    with pytest.raises(PipelineError, match="exactly one of mem"):
        validate_directives(_directives(mem_per_gpu="48G"))
    with pytest.raises(PipelineError, match="time limit"):
        validate_directives(_directives(time_limit="tomorrow"))
    with pytest.raises(PipelineError, match="log_dir must be absolute"):
        validate_directives(_directives(log_dir="logs/slurm"))


def test_render_is_deterministic() -> None:
    first = _render(_directives())
    second = _render(_directives())
    assert sha256_text(first) == sha256_text(second)


def test_batch_script_executes_run_singularity(tmp_path: Path) -> None:
    # Build a tiny real workspace archive whose run_singularity.sh logs its
    # argv and the installed frozen env file, then execute the rendered script.
    workspace = tmp_path / "workspace"
    (workspace / "docker/cluster").mkdir(parents=True)
    call_log = tmp_path / "calls.log"
    runner = workspace / "docker/cluster/run_singularity.sh"
    runner.write_text(
        f'#!/bin/bash\nprintf \'%s\\n\' "$@" >> "{call_log}"\n'
        f'cat "$(dirname "$0")/job_env.resolved.sh" >> "{call_log}"\n'
    )
    runner.chmod(0o755)

    plan_dir = tmp_path / "plans" / "p1"
    plan_dir.mkdir(parents=True)
    with tarfile.open(plan_dir / "workspace.tar.gz", "w:gz") as tar:
        tar.add(workspace, arcname="workspace")
    (plan_dir / "job_env.pretrain.resolved.sh").write_text(
        "export CLUSTER_SENTINEL='frozen'\n"
    )

    script = render_batch_script(
        _directives(),
        remote_plan_dir=str(plan_dir),
        stage="pretrain",
        job_args=("--seed", "0", "arg with spaces"),
        job_tmpdir_root=str(tmp_path / "jobtmp"),
    )
    script_path = tmp_path / "batch_pretrain.sh"
    script_path.write_text(script)
    proc = subprocess.run(
        ["bash", str(script_path)], check=False, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    logged = call_log.read_text()
    assert "arg with spaces" in logged.splitlines()  # quoting survived intact
    assert "export CLUSTER_SENTINEL='frozen'" in logged
    assert "[ENV] export CLUSTER_SENTINEL='frozen'" in proc.stdout
