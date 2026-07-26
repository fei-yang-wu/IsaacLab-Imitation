"""Regression tests for the strict DiffSR latent qualification audit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch


SCRIPT = Path(__file__).with_name("audit_diffsr_latent_qualification.py")


def test_reference_finished_is_not_a_tracking_failure(tmp_path: Path) -> None:
    """All motions may end successfully through ``reference_finished``."""

    checkpoint = tmp_path / "low_level.pt"
    skill_checkpoint = tmp_path / "skill.pt"
    manifest = tmp_path / "manifest.json"
    encoder = {"net.0.weight": torch.tensor([[1.0, 2.0]])}
    torch.save(
        {
            "hl_skill_command_sampler_state_dict": {
                "skill_encoder_state_dict": encoder,
                "finetune_updates": 0,
            }
        },
        checkpoint,
    )
    torch.save({"skill_encoder_state_dict": encoder}, skill_checkpoint)
    manifest.write_bytes(b"test")

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "metadata": {
                    "task": "Isaac-Imitation-G1-Latent-v0",
                    "algorithm": "IPMD",
                    "interface": "latent_skill",
                    "planner_target_dim": 256,
                    "num_envs": 100,
                    "seed": 0,
                    "reset_schedule": "sequential",
                    "wrap_steps": False,
                    "policy_observation_corruption_enabled": False,
                    "early_terminations_enabled": True,
                    "time_out_enabled": True,
                    "episode_length_extension_enabled": True,
                    "reward_clipping_enabled": False,
                    "checkpoint": str(checkpoint),
                    "motion_manifest": str(manifest),
                },
                "aggregate": {
                    "done_rate": 1.0,
                    "tracking_success_rate": 1.0,
                    "tracking_failure_rate": 0.0,
                    "survival_steps_mean": 400.0,
                },
                "skill_checkpoint_override": str(skill_checkpoint),
                "stop_reason": "all_envs_done",
                "max_steps": 1000,
                "steps_run": 400,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(summary),
            "--low_level_checkpoint",
            str(checkpoint),
            "--skill_checkpoint",
            str(skill_checkpoint),
            "--manifest",
            str(manifest),
            "--expected_num_envs",
            "100",
            "--expected_steps",
            "1000",
            "--expected_seed",
            "0",
            "--success_threshold",
            "0.8",
            "--require_pass",
            "--output_json",
            str(output),
        ],
        check=True,
    )

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["protocol_passed"] is True
    assert audit["oracle_passed"] is True
    assert audit["tracking_success_rate"] == 1.0
    assert audit["tracking_failure_rate"] == 0.0
