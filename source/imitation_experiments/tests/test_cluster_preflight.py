"""Gate: preflight maps container paths through the real bind model and fails loudly.

Failure classes under test all have recorded incidents: unmapped dataset paths
("manifest missing", ICE 5577484), outputs outside the binds (node-local tmp,
ICE 5577507), and quota exhaustion (the ICE 300 GB cap).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from imitation_experiments.paper.common import PipelineError
from imitation_experiments.pipeline.cluster.config import load_campaign, load_profile
from imitation_experiments.pipeline.cluster.preflight import (
    container_to_remote,
    run_preflight,
)

CAMPAIGN_TEMPLATE = """\
name: demo
profile: ice
wandb_project: proj
wandb_group: grp
preflight:
  require_container_paths: [/data/ref_arrays]
  output_container_path: /data/out/${vars.arm}_seed${vars.seed}
arms:
  a1:
    stages:
      - name: pretrain
        executable: scripts/train.py
        args: ["env.data.reference_arrays_dir=/data/ref_arrays"]
"""


def _profile(tmp_path: Path, *, min_free_gb: int = 0):
    remote = tmp_path / "remote"
    profile_yaml = tmp_path / "profile_t.yaml"
    profile_yaml.write_text(
        f"name: t\n"
        f"login: ice\n"
        f"control_root: {remote / 'control'}\n"
        f"data_dir: {remote / 'data'}\n"
        f"shared_sif_path: {remote / 'sif/runtime.sif'}\n"
        f"isaac_cache_dir: {remote / 'cache'}\n"
        f"project_logs_dir: {remote / 'isaaclab/logs'}\n"
        f"extra_bind_paths: [{remote / 'shared-alloc'}]\n"
        f"min_free_gb: {min_free_gb}\n"
        f"slurm:\n"
        f"  gres: gpu:a40:1\n"
        f"  cpus_per_task: 4\n"
        f"  mem: 8G\n"
        f"  log_dir: {remote / 'slurm_logs'}\n"
    )
    return load_profile(str(profile_yaml))


def _jobset(tmp_path: Path):
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(CAMPAIGN_TEMPLATE)
    return load_campaign(campaign, arm="a1", seed=0)


def _install_fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_ssh = bin_dir / "ssh"
    fake_ssh.write_text('#!/bin/bash\nshift\nexec bash -c "$*"\n')
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_container_to_remote_mapping(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    remote = tmp_path / "remote"
    assert container_to_remote("/data/x/y.npz", profile) == f"{remote / 'data'}/x/y.npz"
    shared = str(remote / "shared-alloc")
    assert container_to_remote(f"{shared}/ds", profile) == f"{shared}/ds"
    assert (
        container_to_remote("/workspace/isaaclab/project/logs/run1", profile)
        == f"{remote / 'isaaclab/logs'}/run1"
    )
    with pytest.raises(PipelineError, match="not visible under the job's binds"):
        container_to_remote("/home/someone/output", profile)


def test_preflight_all_green(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ssh(tmp_path, monkeypatch)
    profile = _profile(tmp_path)
    remote = tmp_path / "remote"
    (remote / "data/ref_arrays").mkdir(parents=True)
    (remote / "data/out").mkdir()
    (remote / "sif").mkdir()
    (remote / "sif/runtime.sif").write_text("sif")
    results = run_preflight(profile, _jobset(tmp_path))
    failures = [r for r in results if not r.ok]
    assert not failures, failures
    names = {r.name for r in results}
    assert {
        "dataset:/data/ref_arrays",
        "output_writable",
        "slurm_log_dir",
        "shared_sif",
        "control_root",
    } <= names


def test_preflight_missing_dataset_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ssh(tmp_path, monkeypatch)
    profile = _profile(tmp_path)
    (tmp_path / "remote/data").mkdir(parents=True)
    (tmp_path / "remote/sif").mkdir()
    (tmp_path / "remote/sif/runtime.sif").write_text("sif")
    results = run_preflight(profile, _jobset(tmp_path))
    failed = {r.name for r in results if not r.ok}
    assert "dataset:/data/ref_arrays" in failed


def test_preflight_quota_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ssh(tmp_path, monkeypatch)
    # An absurd requirement fails; zero requirement passes on the same fs.
    demanding = _profile(tmp_path, min_free_gb=10_000_000)  # ~10 EB
    remote = tmp_path / "remote"
    (remote / "data/ref_arrays").mkdir(parents=True)
    (remote / "sif").mkdir()
    (remote / "sif/runtime.sif").write_text("sif")
    results = run_preflight(demanding, _jobset(tmp_path))
    by_name = {r.name: r for r in results}
    assert not by_name["output_free_space"].ok
    assert by_name["output_writable"].ok


def test_preflight_flags_unmapped_argv_path(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        CAMPAIGN_TEMPLATE.replace(
            "env.data.reference_arrays_dir=/data/ref_arrays",
            "env.data.reference_arrays_dir=/home/unbound/ref_arrays",
        )
    )
    jobset = load_campaign(campaign, arm="a1", seed=0)
    # Local argv mapping runs before any ssh, so no fake ssh is needed here.
    from imitation_experiments.pipeline.cluster.preflight import _argv_path_checks

    argv_failures = _argv_path_checks(profile, jobset)
    assert argv_failures and not argv_failures[0].ok
    assert "not visible" in argv_failures[0].detail
