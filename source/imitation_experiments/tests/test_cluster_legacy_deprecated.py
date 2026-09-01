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
    fails silently: the run exists, just not where anything expects it.
    WANDB_MODE still crosses for `offline`; the shared-mode pair
    (WANDB__PRIMARY / WANDB__LABEL) was retired on 2026-08-18 with the
    sidecar's attach path and must NOT come back by accident.
    """
    script = (REPO_ROOT / "docker/cluster/run_singularity.sh").read_text()
    forwarded = script.split("for _wandb_var in", 1)[1].split("do", 1)[0]
    for variable in (
        "WANDB_RUN_ID",
        "WANDB_RESUME",
        "WANDB_RUN_GROUP",
        "WANDB_MODE",
    ):
        assert variable in forwarded, f"{variable} never reaches the container"
    for retired in ("WANDB__PRIMARY", "WANDB__LABEL"):
        assert retired not in forwarded, f"{retired} is retired shared-mode plumbing"
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


def test_stage_exclude_reaches_the_sbatch_directives(tmp_path) -> None:
    """A stage's `exclude` must become `#SBATCH --exclude=...`.

    Added 2026-08-27 after `diffntp_chunk_50b` drew a node delivering 54.5k
    fps against its sibling's 127k; the field existed in SlurmDirectives but
    no campaign could set it.
    """
    from imitation_experiments.pipeline.cluster.slurm import (
        SlurmDirectives,
        _directive_lines,
    )

    directives = SlurmDirectives(
        job_name="probe",
        log_dir=str(tmp_path),
        time_limit="15:59:00",
        cpus_per_task=16,
        gres="gpu:h200:1",
        mem="160G",
        exclude="atl1-1-03-017-2-0",
    )
    header = "\n".join(_directive_lines(directives))
    assert "--exclude=atl1-1-03-017-2-0" in header

    plain = SlurmDirectives(
        job_name="probe",
        log_dir=str(tmp_path),
        time_limit="15:59:00",
        cpus_per_task=16,
        gres="gpu:h200:1",
        mem="160G",
    )
    assert "--exclude" not in "\n".join(_directive_lines(plain))


def test_stage_partition_and_qos_override_the_profile(tmp_path) -> None:
    """A stage's `partition` and `qos` must reach the sbatch directives.

    Added 2026-08-30 for `encoder-interface-500m`: the `ice-gpu` H100 nodes
    were full while `coe-gpu` held free ones, and only the profile could name
    a partition, so no arm could move without a second profile file.
    """
    from imitation_experiments.pipeline.cluster.config import (
        ClusterProfile,
        ResolvedStage,
        SlurmDefaults,
    )
    from imitation_experiments.pipeline.cluster.plan_cmd import _stage_directives
    from imitation_experiments.pipeline.cluster.slurm import _directive_lines

    profile = ClusterProfile(
        name="ice",
        login="ice",
        control_root="/control",
        data_dir="/data",
        shared_sif_path="/sif/image.sif",
        isaac_cache_dir="/cache",
        slurm=SlurmDefaults(
            gres="gpu:h100:1",
            mem="96G",
            qos="coe-ice",
            partition="ice-gpu",
            log_dir=str(tmp_path),
        ),
    )
    base = dict(
        name="lowlevel",
        executable="scripts/rlopt/train.py",
        args=(),
        env={},
        time_limit="15:59:00",
        gres="gpu:h100:1",
        mem="160G",
        cpus_per_task=16,
        exclude=None,
        depends_on=None,
        dependency_kind="afterok",
    )

    moved = ResolvedStage(**base, partition="coe-gpu", qos="coe-grade")
    header = "\n".join(
        _directive_lines(_stage_directives(profile, moved, job_name="probe"))
    )
    assert "--partition=coe-gpu" in header
    assert "--qos=coe-grade" in header

    default = ResolvedStage(**base, partition=None, qos=None)
    header = "\n".join(
        _directive_lines(_stage_directives(profile, default, job_name="probe"))
    )
    assert "--partition=ice-gpu" in header
    assert "--qos=coe-ice" in header
