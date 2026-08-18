"""Gate: the local latent-quant ablation launcher stays runnable and keeps LayerNorm a per-arm axis.

Regressions covered (originally found in both this local launcher and the
2026-08-14 ICE campaign's run.sh; the ICE launcher was retired 2026-08-15 in
favor of the control plane, so its coverage moved to
test_latent_quant_repeats_campaign_yaml.py):
- ``FRAME_CAP`` was interpolated before its definition, so the curriculum arms
  crashed under ``set -u`` before launching anything.
- ``--encoder_layer_norm`` was applied unconditionally, which made the ``_ln``
  arms byte-identical no-ops of their siblings.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from imitation_experiments.paths import REPO_ROOT

LOCAL_SCRIPT = (
    REPO_ROOT
    / "experiments/campaigns/2026-08-13-bones129k-latent-quant-ablation/run.sh"
)

# Non-dyn arms only: the dyn arms reuse a completed sibling encoder and must
# not be pretrained, so the test does not exercise that path.
LOCAL_PRETRAIN_ARMS = (
    "cont_det",
    "cont_det_ln",
    "vq",
    "group_vq",
    "fsq64",
    "fsq64_ln",
    "jepa_ntp",
    "jepa_sigreg_ebm",
    "jepa_pure",
    "fsq64_s5",
    "jepa_pure_s5",
    "jepa_sigreg_ebm_s5",
    "fsq64_curriculum",
    "fsq64_dyn_smooth_curriculum",
    "fsq64_smooth",
    "fsq64_ln_dyn_smooth",
)


def _fake_bin(tmp_path: Path, name: str, body: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / name
    fake.write_text(f"#!/bin/sh\n{body}\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _environment(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    env.update(extra)
    return env


def _run_local_pretrain(
    tmp_path: Path, arm: str, **extra: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    pixi_log = tmp_path / f"pixi_calls_{arm}.log"
    _fake_bin(tmp_path, "pixi", f'echo "$@" >> "{pixi_log}"')
    proc = subprocess.run(
        ["bash", str(LOCAL_SCRIPT), arm, "pretrain"],
        cwd=REPO_ROOT,
        env=_environment(tmp_path, OUTPUT_ROOT=str(tmp_path / "out" / arm), **extra),
        check=False,
        capture_output=True,
        text=True,
    )
    return proc, pixi_log


@pytest.mark.skipif(
    not LOCAL_SCRIPT.exists(), reason="untracked local campaign script absent"
)
@pytest.mark.parametrize("arm", LOCAL_PRETRAIN_ARMS)
def test_local_pretrain_launches_for_all_arms(tmp_path: Path, arm: str) -> None:
    proc, pixi_log = _run_local_pretrain(tmp_path, arm)
    assert proc.returncode == 0, proc.stderr
    assert pixi_log.exists(), "pretrain never reached the pixi launch"


@pytest.mark.skipif(
    not LOCAL_SCRIPT.exists(), reason="untracked local campaign script absent"
)
@pytest.mark.parametrize(
    ("plain_arm", "ln_arm"), [("fsq64", "fsq64_ln"), ("cont_det", "cont_det_ln")]
)
def test_local_layer_norm_is_per_arm(
    tmp_path: Path, plain_arm: str, ln_arm: str
) -> None:
    plain_proc, plain_log = _run_local_pretrain(tmp_path, plain_arm)
    ln_proc, ln_log = _run_local_pretrain(tmp_path, ln_arm)
    assert plain_proc.returncode == 0 and ln_proc.returncode == 0
    plain_cmd = plain_log.read_text()
    ln_cmd = ln_log.read_text()
    assert "--encoder_layer_norm" not in plain_cmd
    assert "--encoder_layer_norm" in ln_cmd
    assert plain_cmd != ln_cmd


@pytest.mark.skipif(
    not LOCAL_SCRIPT.exists(), reason="untracked local campaign script absent"
)
def test_local_curriculum_ramp_uses_frame_cap(tmp_path: Path) -> None:
    proc, pixi_log = _run_local_pretrain(
        tmp_path, "fsq64_curriculum", FRAME_CAP="2000000000"
    )
    assert proc.returncode == 0, proc.stderr
    assert pixi_log.exists()
