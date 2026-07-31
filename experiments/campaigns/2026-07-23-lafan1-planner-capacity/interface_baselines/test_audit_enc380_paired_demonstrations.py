from __future__ import annotations

from pathlib import Path

import pytest
import torch

from audit_enc380_paired_demonstrations import audit


def _write_pair(root: Path, *, rows: int, trajectories: int) -> tuple[Path, Path]:
    root_qpos = root / "root_qpos"
    latent = root / "latent_skill"
    root_qpos.mkdir(parents=True)
    latent.mkdir(parents=True)
    env_id = torch.arange(rows) % trajectories
    episode_id = torch.div(torch.arange(rows), trajectories, rounding_mode="floor")
    common = {
        "planner_state": torch.zeros(rows, 930),
        "expert_planner_state": torch.ones(rows, 930),
        "env_id": env_id,
        "episode_id": episode_id,
        "control_step": torch.arange(rows),
        "planner_step": torch.arange(rows),
        "trajectory_rank": torch.zeros(rows, dtype=torch.long),
        "motion_name": ["walk3_subject1"] * rows,
    }
    torch.save(
        {**common, "causal_target": torch.zeros(rows, 380)},
        root_qpos / "sample_step_000000.pt",
    )
    torch.save(
        {**common, "causal_target": torch.zeros(rows, 256)},
        latent / "sample_step_000000.pt",
    )
    return root_qpos, latent


def test_audit_accepts_exact_paired_causal_rows(tmp_path: Path) -> None:
    root_qpos, latent = _write_pair(tmp_path, rows=120, trajectories=100)
    result = audit(
        root_qpos,
        latent,
        expected_rows=120,
        min_trajectories=100,
        expected_motion="walk3_subject1",
    )
    assert result["passed"] is True
    assert result["trajectories"] == 120
    assert result["planner_input_key"] == "planner_state"


def test_audit_rejects_fewer_than_minimum_trajectories(tmp_path: Path) -> None:
    root_qpos, latent = _write_pair(tmp_path, rows=100, trajectories=10)
    # Repeated (env, episode) pairs leave only ten unique trajectories.
    for directory in (root_qpos, latent):
        path = directory / "sample_step_000000.pt"
        payload = torch.load(path, weights_only=False)
        payload["episode_id"].zero_()
        torch.save(payload, path)
    with pytest.raises(ValueError, match="at least 100 trajectories"):
        audit(
            root_qpos,
            latent,
            expected_rows=100,
            min_trajectories=100,
            expected_motion="walk3_subject1",
        )
