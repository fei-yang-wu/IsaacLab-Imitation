from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from low_level_tracker import load_frozen_low_level_tracker


EXPECTED_KEYS = (
    ("policy", "expert_motion"),
    ("policy", "base_ang_vel"),
)


class _FakeAgent:
    def __init__(self, input_keys=EXPECTED_KEYS) -> None:
        self._policy_obs_keys = list(input_keys)
        self.policy = nn.Sequential(nn.Linear(3, 4), nn.ELU(), nn.Linear(4, 2))
        self.value_function = nn.Linear(3, 1)
        self.load_model_called = False

    @property
    def collector_policy(self) -> nn.Module:
        return self.policy

    def load_model(self, _path: str) -> None:
        self.load_model_called = True
        raise AssertionError("policy-only loading must not call agent.load_model")


def _save_checkpoint(path: Path, source: _FakeAgent, **extra) -> None:
    torch.save(
        {
            "policy_state_dict": source.policy.state_dict(),
            "value_state_dict": source.value_function.state_dict(),
            "optimizer_state_dict": {"ignored": True},
            **extra,
        },
        path,
    )


def test_strict_load_freezes_only_policy_and_returns_provenance(tmp_path: Path) -> None:
    torch.manual_seed(1)
    source = _FakeAgent()
    torch.manual_seed(2)
    target = _FakeAgent()
    original_value = {
        key: value.detach().clone()
        for key, value in target.value_function.state_dict().items()
    }
    checkpoint = tmp_path / "tracker.pt"
    _save_checkpoint(checkpoint, source)

    result = load_frozen_low_level_tracker(
        target,
        checkpoint,
        expected_input_keys=EXPECTED_KEYS,
    )

    assert result.policy is target.policy
    assert not target.load_model_called
    assert not result.policy.training
    assert all(not parameter.requires_grad for parameter in result.policy.parameters())
    for key, value in source.policy.state_dict().items():
        torch.testing.assert_close(target.policy.state_dict()[key], value)
    for key, value in original_value.items():
        torch.testing.assert_close(target.value_function.state_dict()[key], value)

    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert result.provenance == {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": expected_hash,
        "loaded_components": ["policy_state_dict"],
        "ignored_checkpoint_keys": [
            "optimizer_state_dict",
            "value_state_dict",
        ],
        "strict_policy_restore": True,
        "policy_frozen": True,
        "policy_training": False,
        "policy_parameter_count": sum(
            parameter.numel() for parameter in target.policy.parameters()
        ),
        "policy_trainable_parameter_count": 0,
        "policy_input_keys": [
            ["policy", "expert_motion"],
            ["policy", "base_ang_vel"],
        ],
    }


def test_rejects_vec_norm_before_mutating_policy(tmp_path: Path) -> None:
    source = _FakeAgent()
    target = _FakeAgent()
    original_policy = {
        key: value.detach().clone() for key, value in target.policy.state_dict().items()
    }
    checkpoint = tmp_path / "normalized.pt"
    _save_checkpoint(checkpoint, source, vec_norm_msg={"mean": torch.zeros(3)})

    with pytest.raises(ValueError, match="vec_norm_msg"):
        load_frozen_low_level_tracker(
            target,
            checkpoint,
            expected_input_keys=EXPECTED_KEYS,
        )

    for key, value in original_policy.items():
        torch.testing.assert_close(target.policy.state_dict()[key], value)


def test_rejects_input_key_order_mismatch_before_mutating_policy(
    tmp_path: Path,
) -> None:
    source = _FakeAgent()
    target = _FakeAgent(input_keys=reversed(EXPECTED_KEYS))
    original_policy = {
        key: value.detach().clone() for key, value in target.policy.state_dict().items()
    }
    checkpoint = tmp_path / "tracker.pt"
    _save_checkpoint(checkpoint, source)

    with pytest.raises(ValueError, match="ordered contract"):
        load_frozen_low_level_tracker(
            target,
            checkpoint,
            expected_input_keys=EXPECTED_KEYS,
        )

    for key, value in original_policy.items():
        torch.testing.assert_close(target.policy.state_dict()[key], value)


def test_strict_restore_rejects_wrong_policy_architecture(tmp_path: Path) -> None:
    source = _FakeAgent()
    target = _FakeAgent()
    target.policy = nn.Sequential(nn.Linear(4, 4), nn.ELU(), nn.Linear(4, 2))
    checkpoint = tmp_path / "tracker.pt"
    _save_checkpoint(checkpoint, source)

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_frozen_low_level_tracker(
            target,
            checkpoint,
            expected_input_keys=EXPECTED_KEYS,
        )
