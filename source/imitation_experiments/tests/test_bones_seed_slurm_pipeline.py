"""Tests for BONES-SEED Slurm submission support that survives the 2026-08-15
retirement of docker/cluster/cluster_interface.sh and its submit_job_slurm_*
helpers. Rendering/dependency-chain tests against those helpers were removed
along with them; see test_cluster_legacy_deprecated.py for their replacement."""

from __future__ import annotations

import os
import subprocess
from imitation_experiments.paths import REPO_ROOT


QUALIFICATION_RUNNER = (
    REPO_ROOT
    / "experiments/campaigns/2026-07-23-bones-phase5-language-local10"
    / "interface_baselines/run_bones_seed_low_level_qualification.sh"
)


def test_qualification_audits_each_interface_cache() -> None:
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "VANILLA_TRACKER_CHECKPOINT": "/checkpoints/vanilla.pt",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/checkpoints/latent.pt",
            "LATENT_SKILL_CHECKPOINT": "/checkpoints/skill.pt",
            "MANIFEST": "/data/manifest.json",
            "VANILLA_DATASET_PATH": "/data/vanilla_cache",
            "LATENT_DATASET_PATH": "/data/latent_cache",
            "OUTPUT_ROOT": "/tmp/bones-qualification-dry-run",
        }
    )
    result = subprocess.run(
        ["bash", str(QUALIFICATION_RUNNER)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    preflight_line = next(
        line
        for line in result.stdout.splitlines()
        if "scripts/audit/audit_bones_seed_phase5.py" in line
    )
    assert "--expected_dataset_path" not in preflight_line
    assert (
        "imitation_experiments.audit.audit_vanilla_tracker_qualification --summary "
        in result.stdout
        and "--expected_dataset_path /data/vanilla_cache" in result.stdout
    )
    latent_audit_line = next(
        line
        for line in result.stdout.splitlines()
        if "imitation_experiments.audit.audit_diffsr_latent_qualification" in line
    )
    assert "--expected_dataset_path /data/latent_cache" in latent_audit_line
