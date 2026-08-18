"""Gate: cluster profiles and campaign specs resolve deterministically and cover the legacy env.

The frozen-env design only removes the allow-list failure class if (a) every
key in a profile reaches the rendered env file, and (b) the ice profile never
silently drops a key the legacy ``.env.ice_runtime`` still carries.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from imitation_experiments.paper.common import PipelineError
from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import (
    load_campaign,
    load_profile,
    resolved_env,
)
from imitation_experiments.pipeline.cluster.envfile import (
    parse_env_file,
    render_job_env,
)

LEGACY_ICE_ENV = REPO_ROOT / "docker/cluster/.env.ice_runtime"

# Legacy keys that are deliberately NOT part of the frozen job env: they steer
# the submission side (scheduler choice, ssh, Slurm resources, sync mode),
# which the control plane owns through typed fields instead.
SUBMISSION_SIDE_KEYS = {
    "CLUSTER_JOB_SCHEDULER",
    "CLUSTER_LOGIN",
    "CLUSTER_REMOTE_LOGIN_SHELL",
    "CLUSTER_ARCHIVE_SYNC",
    "CLUSTER_SKIP_SINGULARITY_IMAGE_CHECK",
    "CLUSTER_SLURM_SUBMIT_SCRIPT",
    "CLUSTER_SLURM_GPU_GRES",
    "CLUSTER_SLURM_CPUS_PER_TASK",
    "CLUSTER_SLURM_MEM",
    "CLUSTER_SLURM_TIME_LIMIT",
    "CLUSTER_SLURM_JOB_NAME_PREFIX",
    "CLUSTER_SLURM_OUTPUT_DIR",
    "CLUSTER_SLURM_PRINT_JOB_SCRIPT",
}

CAMPAIGN_YAML = """\
name: demo
profile: ice
wandb_project: proj
wandb_group: grp
vars:
  frame_cap: 100
preflight:
  require_container_paths: [/data/ref]
  output_container_path: /data/out/${vars.arm}_seed${vars.seed}
shared_env:
  CLUSTER_WANDB_TAGS: demo,${vars.arm}
arms:
  a1:
    stages:
      - name: pretrain
        executable: scripts/rlopt/train_hl_skill_diffsr.py
        args: ["--seed", "${vars.seed}", "cap=${vars.frame_cap}"]
        env: {STAGE_ONLY: pre}
      - name: lowlevel
        executable: scripts/rlopt/train.py
        args: ["cap=${vars.frame_cap}"]
        depends_on: pretrain
  a2:
    vars: {frame_cap: 999}
    stages:
      - name: pretrain
        executable: x.py
        args: ["cap=${vars.frame_cap}"]
"""


def _campaign_file(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(CAMPAIGN_YAML)
    return path


def test_load_ice_profile_from_package() -> None:
    profile = load_profile("ice")
    assert profile.login == "ice"
    assert profile.slurm.log_dir.startswith("/")
    assert "/storage/ice-shared/vip-vwt" in profile.extra_bind_paths
    assert profile.min_free_gb > 0


def test_load_skynet_profile_from_package() -> None:
    profile = load_profile("skynet")
    assert profile.login == "skynet"
    assert profile.slurm.partition == "wu-lab"


def test_profile_rejects_relative_log_dir(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: t\nlogin: t\ncontrol_root: /a\ndata_dir: /b\nshared_sif_path: /c.sif\n"
        "isaac_cache_dir: /d\nslurm:\n  gres: gpu:a40:1\n  cpus_per_task: 4\n"
        "  mem: 8G\n  log_dir: logs/slurm\n"
    )
    with pytest.raises(PipelineError, match="log_dir must be absolute"):
        load_profile(str(bad))


def test_profile_rejects_fqdn_login(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: t\nlogin: login-ice.pace.gatech.edu\ncontrol_root: /a\ndata_dir: /b\n"
        "shared_sif_path: /c.sif\nisaac_cache_dir: /d\nslurm:\n  gres: gpu:a40:1\n"
        "  cpus_per_task: 4\n  mem: 8G\n  log_dir: /logs\n"
    )
    with pytest.raises(PipelineError, match="ssh alias"):
        load_profile(str(bad))


def test_campaign_resolution_interpolates_arm_and_seed(tmp_path: Path) -> None:
    jobset = load_campaign(_campaign_file(tmp_path), arm="a1", seed=3)
    assert jobset.output_container_path == "/data/out/a1_seed3"
    pretrain = jobset.stages[0]
    assert pretrain.args == ("--seed", "3", "cap=100")
    assert jobset.shared_env["CLUSTER_WANDB_TAGS"] == "demo,a1"
    assert jobset.stages[1].depends_on == "pretrain"


def test_campaign_arm_vars_and_cli_override(tmp_path: Path) -> None:
    path = _campaign_file(tmp_path)
    arm_local = load_campaign(path, arm="a2", seed=0)
    assert arm_local.stages[0].args == ("cap=999",)
    overridden = load_campaign(path, arm="a2", seed=0, overrides=["vars.frame_cap=555"])
    assert overridden.stages[0].args == ("cap=555",)


def test_campaign_unknown_arm_raises(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="unknown arm"):
        load_campaign(_campaign_file(tmp_path), arm="nope", seed=0)


def test_resolved_env_merge_order_and_determinism(tmp_path: Path) -> None:
    profile = load_profile("ice")
    profile.env["CLUSTER_WANDB_TAGS"] = "from-profile"
    jobset = load_campaign(_campaign_file(tmp_path), arm="a1", seed=0)
    env = resolved_env(profile, jobset, jobset.stages[0])
    # campaign shared_env beats profile.env; stage executable always wins.
    assert env["CLUSTER_WANDB_TAGS"] == "demo,a1"
    assert env["CLUSTER_PYTHON_EXECUTABLE"] == "scripts/rlopt/train_hl_skill_diffsr.py"
    assert env["STAGE_ONLY"] == "pre"
    assert env["CLUSTER_EXTRA_BIND_PATHS"] == "/storage/ice-shared/vip-vwt"
    again = resolved_env(profile, jobset, jobset.stages[0])
    assert list(env.items()) == list(again.items())


def test_render_job_env_round_trips_through_bash(tmp_path: Path) -> None:
    rendered = render_job_env(
        {"CLUSTER_A": "plain", "CLUSTER_B": "has spaces 'and quotes'"},
        plan_sha="abc123",
        stage="pretrain",
    )
    env_file = tmp_path / "job_env.resolved.sh"
    env_file.write_text(rendered)
    proc = subprocess.run(
        ["bash", "-c", f'source "{env_file}" && printf %s "$CLUSTER_B"'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout == "has spaces 'and quotes'"
    assert "plan_sha=abc123" in rendered


def test_render_job_env_rejects_bad_key() -> None:
    with pytest.raises(PipelineError, match="invalid environment variable name"):
        render_job_env({"BAD-KEY": "x"}, plan_sha="s", stage="pretrain")


def test_ice_profile_covers_env_ice_runtime(tmp_path: Path) -> None:
    legacy_keys = set(parse_env_file(LEGACY_ICE_ENV))
    profile = load_profile("ice")
    jobset = load_campaign(_campaign_file(tmp_path), arm="a1", seed=0)
    resolved_keys = set(resolved_env(profile, jobset, jobset.stages[0]))
    missing = legacy_keys - resolved_keys - SUBMISSION_SIDE_KEYS
    assert not missing, (
        f"legacy .env.ice_runtime keys absent from the frozen env: {sorted(missing)}; "
        "add them to profile_ice.yaml env or to SUBMISSION_SIDE_KEYS with a reason"
    )
