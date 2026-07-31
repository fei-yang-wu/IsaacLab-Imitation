from __future__ import annotations

from pathlib import Path

import pytest
import torch

from materialize_paired_interface_samples import _materialize_sample, _select_motion


def _sample() -> dict:
    return {
        "causal_target": torch.zeros(3, 256),
        "z_target": torch.zeros(3, 256),
        "latent_skill_target": torch.ones(3, 256),
        "encoder_input_packet_target": torch.ones(3, 380) * 2,
        "planner_state": torch.zeros(3, 930),
        "metadata": {
            "planner_interval_steps": 10,
            "paired_interface_target_specs": {
                "latent_skill_target": {
                    "interface": "latent_skill",
                    "target_dim": 256,
                },
                "encoder_input_packet_target": {
                    "interface": "root_qpos",
                    "target_dim": 380,
                },
            },
        },
    }


def test_materializes_root_packet_without_latent_aliases(tmp_path: Path) -> None:
    result = _materialize_sample(
        _sample(), key="encoder_input_packet_target", path=tmp_path / "sample.pt"
    )
    assert tuple(result["causal_target"].shape) == (3, 380)
    assert torch.equal(result["causal_target"], torch.full((3, 380), 2.0))
    assert "z_target" not in result
    assert "latent_skill_target" not in result
    assert result["metadata"]["interface"] == "root_qpos"
    assert result["metadata"]["command_future_steps"] == 9


def test_rejects_paired_target_row_mismatch(tmp_path: Path) -> None:
    sample = _sample()
    sample["latent_skill_target"] = torch.ones(2, 256)
    with pytest.raises(ValueError, match="row-count mismatch"):
        _materialize_sample(
            sample, key="latent_skill_target", path=tmp_path / "sample.pt"
        )


def test_motion_filter_preserves_only_matching_variable_length_rows(
    tmp_path: Path,
) -> None:
    sample = _sample()
    sample["motion_name"] = ["walk", "dance", "walk"]
    sample["env_id"] = torch.tensor([0, 1, 2])
    selected = _select_motion(sample, motion_name="walk", path=tmp_path / "sample.pt")
    assert selected is not None
    assert selected["motion_name"] == ["walk", "walk"]
    assert torch.equal(selected["env_id"], torch.tensor([0, 2]))
    assert tuple(selected["causal_target"].shape) == (2, 256)
    assert (
        _select_motion(sample, motion_name="run", path=tmp_path / "sample.pt") is None
    )
