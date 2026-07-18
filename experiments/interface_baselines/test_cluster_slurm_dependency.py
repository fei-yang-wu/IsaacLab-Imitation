"""Tests for guarded Slurm dependencies and arrays in the cluster submitter."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


SUBMIT_SCRIPT = (
    Path(__file__).resolve().parents[2] / "docker/cluster/submit_job_slurm.sh"
)


def _fake_sbatch(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    sbatch.chmod(0o755)
    return bin_dir


def test_afterok_dependency_is_rendered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = dict(os.environ)
    env["PATH"] = f"{_fake_sbatch(tmp_path)}:{env['PATH']}"
    env["CLUSTER_SLURM_DEPENDENCY"] = "afterok:3501873:3501960"

    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), str(workspace), "isaac-lab-base", "--help"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "#SBATCH --dependency=afterok:3501873:3501960" in result.stdout


def test_dependency_rejects_unrestricted_sbatch_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = dict(os.environ)
    env["PATH"] = f"{_fake_sbatch(tmp_path)}:{env['PATH']}"
    env["CLUSTER_SLURM_DEPENDENCY"] = "afterany:3501873"

    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), str(workspace), "isaac-lab-base", "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must use afterok" in result.stderr


def test_bounded_array_is_rendered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = dict(os.environ)
    env["PATH"] = f"{_fake_sbatch(tmp_path)}:{env['PATH']}"
    env["CLUSTER_SLURM_ARRAY"] = "0-99%8"

    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), str(workspace), "isaac-lab-base", "--help"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "#SBATCH --array=0-99%8" in result.stdout
    assert '#SBATCH --output="output_%A_%a.log"' in result.stdout


def test_array_rejects_invalid_or_descending_ranges(tmp_path: Path) -> None:
    for array_spec in ("0,2,4", "99-0", "0-99%0", "$(touch nope)"):
        workspace = tmp_path / f"workspace-{len(array_spec)}"
        workspace.mkdir(exist_ok=True)
        env = dict(os.environ)
        bin_root = tmp_path / f"bin-{len(array_spec)}"
        if not bin_root.exists():
            bin_root.mkdir()
            sbatch = bin_root / "sbatch"
            sbatch.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
            sbatch.chmod(0o755)
        env["PATH"] = f"{bin_root}:{env['PATH']}"
        env["CLUSTER_SLURM_ARRAY"] = array_spec

        result = subprocess.run(
            ["bash", str(SUBMIT_SCRIPT), str(workspace), "isaac-lab-base", "--help"],
            cwd=tmp_path,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert "CLUSTER_SLURM_ARRAY" in result.stderr
