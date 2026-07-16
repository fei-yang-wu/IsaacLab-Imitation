"""Tests for the fixed three-seed BONES-SEED paper launcher."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


LAUNCHER = Path(__file__).with_name(
    "submit_bones_seed_multiseed_pipeline_skynet.sh"
)


def _environment(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "VANILLA_TRACKER_CHECKPOINT": "/logs/vanilla.pt",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/logs/latent.pt",
            "LATENT_SKILL_CHECKPOINT": "/logs/skill.pt",
            "OUTPUT_ROOT_PREFIX": "logs/interface_baselines/test_paper_seed",
        }
    )
    return env


def test_multiseed_launcher_renders_exactly_three_unique_seeds(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=LAUNCHER.parents[2],
        env=_environment(tmp_path),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("mode=bones-seed-multigoal-language") == 3
    for seed in range(3):
        assert f"SEED={seed}" in result.stdout
        assert (
            f"OUTPUT_ROOT=logs/interface_baselines/test_paper_seed{seed}"
            in result.stdout
        )
    assert result.stdout.count("EVAL_STEPS=500") >= 3
    assert "EVAL_STEPS=1000" not in result.stdout


def test_multiseed_launcher_rejects_a_changed_seed_grid(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    env["SEEDS"] = "0 1 3"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=LAUNCHER.parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fixes SEEDS='0 1 2'" in result.stderr


def test_multiseed_launcher_rejects_extended_m3_episode(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    env["EVAL_STEPS"] = "1000"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=LAUNCHER.parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fixed to the normal 500-step episode" in result.stderr


def test_single_seed_preflight_cannot_be_a_dry_run() -> None:
    launcher = LAUNCHER.with_name("submit_bones_seed_multigoal_pipeline_skynet.sh")
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "PREFLIGHT_ONLY": "1",
            "VANILLA_TRACKER_CHECKPOINT": "/logs/vanilla.pt",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/logs/latent.pt",
            "LATENT_SKILL_CHECKPOINT": "/logs/skill.pt",
        }
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=launcher.parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "requires DRY_RUN=0" in result.stderr


def test_single_seed_preflight_refuses_an_existing_output_root(
    tmp_path: Path,
) -> None:
    launcher = LAUNCHER.with_name("submit_bones_seed_multigoal_pipeline_skynet.sh")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_ssh.chmod(0o755)
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DRY_RUN": "0",
            "PREFLIGHT_ONLY": "1",
            "VANILLA_TRACKER_CHECKPOINT": "logs/vanilla.pt",
            "LATENT_LOW_LEVEL_CHECKPOINT": "logs/latent.pt",
            "LATENT_SKILL_CHECKPOINT": "logs/skill.pt",
            "OUTPUT_ROOT": "logs/interface_baselines/already_exists",
        }
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=launcher.parents[2],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Refusing to reuse existing paper output root" in result.stderr
