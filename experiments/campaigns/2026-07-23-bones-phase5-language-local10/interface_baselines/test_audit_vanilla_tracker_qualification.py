"""Regression tests for the strict direct-vanilla qualification audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).with_name("audit_vanilla_tracker_qualification.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vanilla_qualification_requires_the_full_fixed_protocol(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vanilla.pt"
    manifest = tmp_path / "manifest.json"
    dataset = tmp_path / "vanilla_cache"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_bytes(b"manifest")
    dataset.mkdir()
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "metadata": {
                    "task": "Isaac-Imitation-G1-v0",
                    "algorithm": "IPMD",
                    "checkpoint": str(checkpoint),
                    "motion_manifest": str(manifest),
                    "dataset_path": str(dataset),
                    "command_space": "single_frame_full_body",
                    "low_level_command_mode": "native",
                    "command_observation_source": "reference",
                    "command_past_steps": 0,
                    "command_future_steps": 0,
                    "planner_update_interval": 1,
                    "policy_only_checkpoint": True,
                    "low_level_tracker": {
                        "strict_policy_restore": True,
                        "policy_frozen": True,
                        "checkpoint_sha256": _sha256(checkpoint),
                    },
                    "num_envs": 100,
                    "steps_requested": 1000,
                    "seed": 0,
                    "reset_schedule": "sequential",
                    "reference_start_frame": 0,
                    "keep_after_done": False,
                    "observation_corruption_enabled": False,
                    "wrap_steps": False,
                    "early_terminations_enabled": True,
                    "time_out_enabled": True,
                    "episode_length_extension_enabled": True,
                    "reward_clipping_enabled": False,
                },
                "aggregate": {
                    "tracking_success_rate": 0.9,
                    "done_rate": 0.1,
                    "survival_steps_mean": 900.0,
                },
                "max_steps": 1000,
                "steps_run": 1000,
                "per_environment": [{"env_id": index} for index in range(100)],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(summary),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--expected_dataset_path",
            str(dataset),
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
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["protocol_passed"] is True
    assert audit["oracle_passed"] is True

    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["metadata"]["planner_update_interval"] = 10
    summary.write_text(json.dumps(payload), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--summary",
            str(summary),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--expected_dataset_path",
            str(dataset),
            "--expected_num_envs",
            "100",
            "--expected_steps",
            "1000",
            "--expected_seed",
            "0",
            "--output_json",
            str(output),
        ],
        check=False,
    )
    assert rejected.returncode == 1
