"""Gate: the legacy cluster submission chain fails loudly, not silently.

Retired 2026-08-15 in favor of imitation_experiments.pipeline.cluster (plan
verified end-to-end on real ICE: jobs 5577564/5577565). Every entry point that
used to run a job now refuses with a clear pointer to the new CLI instead of
executing — this is the "warn and error if anyone refers them" contract, not
a silent no-op or a bare "file not found".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from imitation_experiments.paths import REPO_ROOT

DEPRECATED_CLUSTER_SCRIPTS = (
    "cluster_interface.sh",
    "submit_job_pbs.sh",
    "submit_job_slurm.sh",
    "submit_job_slurm_bones_pipeline.sh",
    "submit_job_slurm_pace.sh",
    "submit_job_slurm_phase4.sh",
    "submit_job_slurm_skynet.sh",
    "submit_job_slurm_skynet_pixi.sh",
)


@pytest.mark.parametrize("name", DEPRECATED_CLUSTER_SCRIPTS)
def test_legacy_cluster_script_errors_with_pointer(name: str) -> None:
    script = REPO_ROOT / "docker/cluster" / name
    assert script.is_file(), (
        f"{script} missing (retired scripts must still exist to warn)"
    )
    proc = subprocess.run(
        ["bash", str(script), "arg1", "arg2"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert "DEPRECATED" in proc.stderr
    assert "pipeline/cluster" in proc.stderr or "pipeline.cluster" in proc.stderr


def test_cluster_interface_orig_was_removed() -> None:
    assert not (REPO_ROOT / "docker/cluster/cluster_interface.sh.orig").exists()


def test_run_singularity_errors_without_frozen_env(tmp_path: Path) -> None:
    # No job_env.resolved.sh in this workspace: run_singularity.sh must refuse
    # the legacy .env.cluster fallback instead of silently reading it.
    real = REPO_ROOT / "docker/cluster/run_singularity.sh"
    workspace = tmp_path / "workspace"
    (workspace / "docker/cluster").mkdir(parents=True)
    (workspace / "docker/cluster/.env.cluster").write_text(
        "CLUSTER_ISAACLAB_DIR=should-never-be-read\n"
    )
    (workspace / "docker/.env.base").write_text("export ACCEPT_EULA=Y\n")
    proc = subprocess.run(
        ["bash", str(real), str(workspace), "isaac-lab-base"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "retired 2026-08-15" in proc.stderr
    assert "pipeline.cluster" in proc.stderr


def test_wandb_controls_cross_the_container_boundary() -> None:
    """`--containall` drops host env, so each W&B control must be forwarded.

    A campaign that exports WANDB_RUN_ID and gets an auto-generated id instead
    fails silently: the run exists, just not where anything expects it. The
    same boundary is what a sidecar attaching to the training run depends on
    (WANDB_MODE=shared + WANDB__PRIMARY).
    """
    script = (REPO_ROOT / "docker/cluster/run_singularity.sh").read_text()
    forwarded = script.split("for _wandb_var in", 1)[1].split("do", 1)[0]
    for variable in (
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_RUN_GROUP",
        "WANDB_MODE",
        "WANDB__PRIMARY",
        "WANDB__LABEL",
    ):
        assert variable in forwarded, f"{variable} never reaches the container"
    assert 'export "APPTAINERENV_${_wandb_var}=${_wandb_value}"' in script
    assert 'export "SINGULARITYENV_${_wandb_var}=${_wandb_value}"' in script


def test_pilot_run_sh_is_deprecation_shim() -> None:
    script = (
        REPO_ROOT / "experiments/campaigns/2026-08-14-latent-quant-ice-repeats/run.sh"
    )
    proc = subprocess.run(
        ["bash", str(script), "fsq64", "0"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "DEPRECATED" in proc.stderr
    assert "submit.sh" in proc.stderr
