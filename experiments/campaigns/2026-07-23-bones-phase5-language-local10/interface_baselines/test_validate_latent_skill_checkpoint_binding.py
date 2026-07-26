"""Tests for low-level to skill-encoder checkpoint binding."""

from __future__ import annotations

from pathlib import Path

import torch

from validate_latent_skill_checkpoint_binding import validate_binding


def _write_low_level(path: Path, encoder: dict[str, torch.Tensor]) -> None:
    torch.save(
        {
            "hl_skill_command_sampler_state_dict": {
                "skill_encoder_state_dict": encoder,
                "finetune_updates": 0,
            }
        },
        path,
    )


def _write_skill(path: Path, encoder: dict[str, torch.Tensor]) -> None:
    torch.save({"skill_encoder_state_dict": encoder}, path)


def test_binding_accepts_exact_embedded_encoder(tmp_path: Path) -> None:
    low_level = tmp_path / "low_level.pt"
    skill = tmp_path / "skill.pt"
    encoder = {
        "net.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "net.0.bias": torch.tensor([1.0, 2.0]),
    }
    _write_low_level(low_level, encoder)
    _write_skill(skill, {key: value.clone() for key, value in encoder.items()})

    result = validate_binding(low_level, skill)

    assert result["passed"] is True
    assert result["embedded_key_count"] == 2
    assert result["mismatched_keys"] == []


def test_binding_rejects_different_encoder_weights(tmp_path: Path) -> None:
    low_level = tmp_path / "low_level.pt"
    skill = tmp_path / "skill.pt"
    _write_low_level(low_level, {"net.0.weight": torch.tensor([[1.0]])})
    _write_skill(skill, {"net.0.weight": torch.tensor([[2.0]])})

    result = validate_binding(low_level, skill)

    assert result["passed"] is False
    assert result["mismatched_keys"] == ["net.0.weight"]


def test_binding_rejects_key_mismatch(tmp_path: Path) -> None:
    low_level = tmp_path / "low_level.pt"
    skill = tmp_path / "skill.pt"
    _write_low_level(low_level, {"net.0.weight": torch.tensor([[1.0]])})
    _write_skill(skill, {"net.1.weight": torch.tensor([[1.0]])})

    result = validate_binding(low_level, skill)

    assert result["passed"] is False
    assert result["missing_keys"] == ["net.0.weight"]
    assert result["unexpected_keys"] == ["net.1.weight"]
