"""Gate: ``submit`` executes only a confirmed, undrifted plan and records everything.

Includes the regression test for the config-forwarding failure class:
run_singularity.sh must prefer the frozen job_env.resolved.sh over the legacy
.env.cluster re-source.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

import imitation_experiments.paper.common as paper_common
import imitation_experiments.pipeline.cluster.planfile as planfile
import imitation_experiments.pipeline.cluster.submit_cmd as submit_cmd
from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.__main__ import main

CAMPAIGN_YAML = """\
name: demo
profile: {profile}
wandb_project: proj
wandb_group: grp
preflight:
  require_container_paths: [/data/ref_arrays]
  output_container_path: /data/out/${{vars.arm}}_seed${{vars.seed}}
arms:
  a1:
    stages:
      - name: pretrain
        executable: scripts/train_pre.py
        args: ["--seed", "${{vars.seed}}"]
      - name: lowlevel
        executable: scripts/train.py
        args: ["env.data.reference_arrays_dir=/data/ref_arrays"]
        depends_on: pretrain
"""


def _fixture_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "workrepo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/train.py").write_text("print('train')\n")
    (repo / "docker/cluster").mkdir(parents=True)
    (repo / "docker/cluster/run_singularity.sh").write_text(
        "#!/bin/bash\necho runner\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Fake ssh executing locally, counting fake sbatch, hermetic git repo."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text('#!/bin/bash\nshift\nexec bash -c "$*"\n')
    (bin_dir / "ssh").chmod(0o755)
    sbatch_log = tmp_path / "sbatch_argv.log"
    counter = tmp_path / "sbatch_counter"
    (bin_dir / "sbatch").write_text(
        f'#!/bin/sh\necho "$@" >> "{sbatch_log}"\n'
        f'n=$(cat "{counter}" 2>/dev/null || echo 100)\n'
        f'echo $((n+1)) > "{counter}"\necho $n\n'
    )
    (bin_dir / "sbatch").chmod(0o755)
    (bin_dir / "scancel").write_text(
        f'#!/bin/sh\necho "scancel $@" >> "{sbatch_log}"\n'
    )
    (bin_dir / "scancel").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    work_repo = _fixture_git_repo(tmp_path)
    monkeypatch.setattr(paper_common, "REPO_ROOT", work_repo)
    monkeypatch.setattr(planfile, "REPO_ROOT", work_repo)
    monkeypatch.setattr(submit_cmd, "REPO_ROOT", work_repo)

    remote = tmp_path / "remote"
    (remote / "data/ref_arrays").mkdir(parents=True)
    (remote / "sif").mkdir()
    (remote / "sif/runtime.sif").write_text("sif")

    profile = tmp_path / "profile_t.yaml"
    profile.write_text(
        f"name: t\nlogin: ice\ncontrol_root: {remote / 'control'}\n"
        f"data_dir: {remote / 'data'}\nshared_sif_path: {remote / 'sif/runtime.sif'}\n"
        f"isaac_cache_dir: {remote / 'cache'}\n"
        f"slurm:\n  gres: gpu:a40:1\n  cpus_per_task: 4\n  mem: 8G\n"
        f"  log_dir: {remote / 'slurm_logs'}\n"
    )
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(CAMPAIGN_YAML.format(profile=profile))
    return campaign, sbatch_log


def _run(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = main(argv)
    return rc, buffer.getvalue()


def _make_plan(tmp_path: Path, campaign: Path) -> tuple[Path, str]:
    rc, stdout = _run(
        [
            "plan",
            "--campaign",
            str(campaign),
            "--arm",
            "a1",
            "--seed",
            "0",
            "--out-root",
            str(tmp_path / "control_out"),
        ]
    )
    assert rc == 0, stdout
    sha = stdout.rsplit("PLAN_SHA=", 1)[1].strip()
    plan_dir = next((tmp_path / "control_out/demo").iterdir())
    return plan_dir, sha


def test_submit_refuses_without_matching_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, sbatch_log = _setup(tmp_path, monkeypatch)
    plan_dir, _ = _make_plan(tmp_path, campaign)
    rc, stdout = _run(["submit", "--plan", str(plan_dir), "--confirm", "deadbeef"])
    assert rc == 2
    assert "expected: --confirm" in stdout
    assert not sbatch_log.exists()


def test_submit_happy_path_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, sbatch_log = _setup(tmp_path, monkeypatch)
    plan_dir, sha = _make_plan(tmp_path, campaign)
    rc, stdout = _run(["submit", "--plan", str(plan_dir), "--confirm", sha])
    assert rc == 0, stdout

    argv_text = sbatch_log.read_text()
    assert "--dependency=afterok:100" in argv_text

    records = sorted(plan_dir.glob("submission-*.json"))
    assert len(records) == 1
    submission = json.loads(records[0].read_text())
    assert submission["plan_sha"] == sha
    assert submission["drift"] is False
    assert [j["slurm_job_id"] for j in submission["jobs"]] == ["100", "101"]
    assert submission["jobs"][1]["dependency"] == "afterok:100"

    remote_plan_dir = Path(submission["remote_plan_dir"])
    assert (remote_plan_dir / "workspace.tar.gz").is_file()
    assert (remote_plan_dir / "workspace.tar.gz.sha256").is_file()
    # The plan directory holds a link into the shared store, so a campaign of
    # N arms built from one tree keeps one archive, not N.
    store_entry = Path(submission["workspace_archive_path"])
    assert store_entry.name == f"{submission['workspace_archive_sha256']}.tar.gz"
    assert store_entry.parent.name == "workspaces"
    assert (remote_plan_dir / "workspace.tar.gz").resolve() == store_entry.resolve()
    assert (remote_plan_dir / "batch_pretrain.sh").is_file()
    assert (remote_plan_dir / "job_env.lowlevel.resolved.sh").is_file()
    assert (remote_plan_dir / records[0].name).is_file()


def test_submit_refuses_on_git_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, _ = _setup(tmp_path, monkeypatch)
    plan_dir, sha = _make_plan(tmp_path, campaign)
    (planfile.REPO_ROOT / "scripts/train.py").write_text("print('changed')\n")
    rc, stdout = _run(["submit", "--plan", str(plan_dir), "--confirm", sha])
    assert rc == 2
    assert "working tree changed" in stdout
    rc, stdout = _run(
        ["submit", "--plan", str(plan_dir), "--confirm", sha, "--allow-drift"]
    )
    assert rc == 0, stdout
    submission = json.loads(sorted(plan_dir.glob("submission-*.json"))[-1].read_text())
    assert submission["drift"] is True


def test_submit_records_are_versioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, _ = _setup(tmp_path, monkeypatch)
    plan_dir, sha = _make_plan(tmp_path, campaign)
    rc, _stdout = _run(["submit", "--plan", str(plan_dir), "--confirm", sha])
    assert rc == 0
    rc, stdout = _run(["submit", "--plan", str(plan_dir), "--confirm", sha])
    assert rc == 2
    assert "already submitted" in stdout


def test_run_singularity_prefers_resolved_env(tmp_path: Path) -> None:
    # Extract the env-loading region of the REAL run_singularity.sh: with the
    # frozen file present it sources that; the "no frozen file" branch is a
    # hard error since 2026-08-15 (test_cluster_legacy_deprecated.py covers
    # that branch end-to-end against the full script).
    real = REPO_ROOT / "docker/cluster/run_singularity.sh"
    stub_dir = tmp_path / "docker/cluster"
    stub_dir.mkdir(parents=True)
    (tmp_path / "docker/.env.base").write_text("export BASE_MARKER=base\n")
    (stub_dir / "job_env.resolved.sh").write_text("export CLUSTER_SENTINEL=frozen\n")

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/bin/bash\nset -e\n"
        f'SCRIPT_DIR="{stub_dir}"\n'
        f'eval "$(sed -n \'/^# load variables/,/\\.env\\.base$/p\' "{real}")"\n'
        'printf "%s" "$CLUSTER_SENTINEL"\n'
    )
    proc = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("frozen")
