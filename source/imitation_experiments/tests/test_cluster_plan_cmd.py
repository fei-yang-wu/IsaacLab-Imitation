"""Gate: ``plan`` freezes a submittable, sha-stable record and never touches the scheduler."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from imitation_experiments.pipeline.cluster.__main__ import main
from imitation_experiments.pipeline.cluster.planfile import load_plan

CAMPAIGN_YAML = """\
name: demo
profile: {profile}
wandb_project: proj
wandb_group: grp
vars:
  frame_cap: 100
preflight:
  require_container_paths: [/data/ref_arrays]
  output_container_path: /data/out/${{vars.arm}}_seed${{vars.seed}}
arms:
  a1:
    stages:
      - name: pretrain
        executable: scripts/train_pre.py
        args: ["--seed", "${{vars.seed}}", "env.data.reference_arrays_dir=/data/ref_arrays"]
      - name: lowlevel
        executable: scripts/train.py
        args: ["cap=${{vars.frame_cap}}", "out=/data/out/${{vars.arm}}_seed${{vars.seed}}"]
        depends_on: pretrain
"""


def _write_profile(tmp_path: Path) -> Path:
    remote = tmp_path / "remote"
    profile = tmp_path / "profile_t.yaml"
    profile.write_text(
        f"name: t\nlogin: ice\ncontrol_root: {remote / 'control'}\n"
        f"data_dir: {remote / 'data'}\nshared_sif_path: {remote / 'sif/runtime.sif'}\n"
        f"isaac_cache_dir: {remote / 'cache'}\n"
        f"slurm:\n  gres: gpu:a40:1\n  cpus_per_task: 4\n  mem: 8G\n"
        f"  log_dir: {remote / 'slurm_logs'}\n"
    )
    return profile


def _write_campaign(tmp_path: Path, profile: Path) -> Path:
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(CAMPAIGN_YAML.format(profile=profile))
    return campaign


def _seed_remote_fs(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    (remote / "data/ref_arrays").mkdir(parents=True)
    (remote / "data/out").mkdir()
    (remote / "sif").mkdir()
    (remote / "sif/runtime.sif").write_text("sif")


def _install_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "ssh").write_text('#!/bin/bash\nshift\nexec bash -c "$*"\n')
    (bin_dir / "ssh").chmod(0o755)
    sbatch_marker = tmp_path / "sbatch_was_called"
    (bin_dir / "sbatch").write_text(f'#!/bin/sh\ntouch "{sbatch_marker}"\nexit 97\n')
    (bin_dir / "sbatch").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return sbatch_marker


def _plan(tmp_path: Path, out_name: str, *extra: str) -> tuple[int, str, Path]:
    profile = _write_profile(tmp_path)
    campaign = _write_campaign(tmp_path, profile)
    out_root = tmp_path / out_name
    argv = [
        "plan",
        "--campaign",
        str(campaign),
        "--arm",
        "a1",
        "--seed",
        "0",
        "--out-root",
        str(out_root),
        *extra,
    ]
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = main(argv)
    return rc, buffer.getvalue(), out_root


def _extract_sha(stdout: str) -> str:
    match = re.search(r"^PLAN_SHA=([0-9a-f]{64})$", stdout, re.M)
    assert match, f"no PLAN_SHA line in output:\n{stdout}"
    return match.group(1)


def test_plan_writes_plan_dir_and_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(tmp_path, monkeypatch)
    _seed_remote_fs(tmp_path)
    rc, stdout, out_root = _plan(tmp_path, "out1")
    assert rc == 0, stdout
    sha = _extract_sha(stdout)
    plan_dirs = list((out_root / "demo").iterdir())
    assert len(plan_dirs) == 1
    record, plan_dir = load_plan(plan_dirs[0])  # re-verifies sha + artifact hashes
    assert record["plan_sha"] == sha
    assert {j["stage"] for j in record["sealed"]["jobs"]} == {"pretrain", "lowlevel"}
    assert (plan_dir / "batch_pretrain.sh").is_file()
    assert (plan_dir / "job_env.lowlevel.resolved.sh").is_file()
    env_text = (plan_dir / "job_env.pretrain.resolved.sh").read_text()
    assert "export CLUSTER_PYTHON_EXECUTABLE=scripts/train_pre.py" in env_text
    assert record["preflight"]["status"] == "passed"


def test_plan_never_submits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sbatch_marker = _install_fakes(tmp_path, monkeypatch)
    _seed_remote_fs(tmp_path)
    rc, stdout, _ = _plan(tmp_path, "out1")
    assert rc == 0, stdout
    assert not sbatch_marker.exists(), "plan invoked sbatch"


def test_plan_failed_preflight_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(tmp_path, monkeypatch)
    # No ref_arrays dir on the fake remote.
    (tmp_path / "remote/data").mkdir(parents=True)
    (tmp_path / "remote/sif").mkdir()
    (tmp_path / "remote/sif/runtime.sif").write_text("sif")
    rc, stdout, out_root = _plan(tmp_path, "out1")
    assert rc == 2
    assert "FAIL dataset:/data/ref_arrays" in stdout
    record = json.loads(next((out_root / "demo").glob("*/plan.json")).read_text())
    assert record["preflight"]["status"] == "failed"


def test_plan_sha_stable_and_sensitive_to_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(tmp_path, monkeypatch)
    _seed_remote_fs(tmp_path)
    rc1, out1, _ = _plan(tmp_path, "outA", "--skip-preflight")
    rc2, out2, _ = _plan(tmp_path, "outB", "--skip-preflight")
    rc3, out3, _ = _plan(
        tmp_path, "outC", "--skip-preflight", "--set", "vars.frame_cap=555"
    )
    assert rc1 == rc2 == rc3 == 0
    assert _extract_sha(out1) == _extract_sha(out2)
    assert _extract_sha(out3) != _extract_sha(out1)


def test_skip_preflight_needs_no_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/bin/sh\nexit 97\n")
    (bin_dir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    rc, stdout, _ = _plan(tmp_path, "out1", "--skip-preflight")
    assert rc == 0, stdout
    assert "PLAN_SHA=" in stdout


def test_only_stage_drops_external_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fakes(tmp_path, monkeypatch)
    rc, stdout, out_root = _plan(
        tmp_path, "out1", "--skip-preflight", "--only-stage", "lowlevel"
    )
    assert rc == 0, stdout
    record = json.loads(next((out_root / "demo").glob("*/plan.json")).read_text())
    jobs = record["sealed"]["jobs"]
    assert [j["stage"] for j in jobs] == ["lowlevel"]
    assert jobs[0]["depends_on"] is None
