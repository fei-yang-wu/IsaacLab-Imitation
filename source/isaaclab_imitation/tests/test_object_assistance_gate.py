"""Asynchronous reset evidence for object assistance changes."""

import importlib.util
from pathlib import Path

import torch
import pytest

_PATH = (
    Path(__file__).parent.parent
    / "isaaclab_imitation/contracts/object_assistance_gate.py"
)
_SPEC = importlib.util.spec_from_file_location("object_assistance_gate", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ObjectAssistanceGate = _MODULE.ObjectAssistanceGate


def test_startup_and_mixed_stage_episodes_cannot_advance_gate():
    gate = ObjectAssistanceGate(3, "cpu", 3, minimum_episodes=2, minimum_steps=0)
    success = torch.ones(3)
    length = torch.zeros(3)
    assert gate.reset([0, 1, 2], success, length, 0) == 0
    length.fill_(100)
    assert gate.reset([0, 1], success, length, 100) == 1
    assert gate.reset([2], success, length, 101) == 1
    assert gate.episodes == 0
    assert gate.reset([0, 1], success, length, 200) == 2
    assert gate.reset([0, 1], success, length, 300) == 2


def test_early_failure_counts_and_stage_dwell_time_is_required():
    gate = ObjectAssistanceGate(2, "cpu", 2, minimum_episodes=2, minimum_steps=100)
    length = torch.ones(2)
    gate.reset([0, 1], torch.ones(2), torch.zeros(2), 0)
    assert gate.reset([0, 1], torch.ones(2), length, 50) == 0
    assert gate.reset([0, 1], torch.zeros(2), length, 100) == 0
    assert gate.last_rate == 0.5
    assert gate.reset([0, 1], torch.ones(2), length, 150) == 1


def test_resume_preserves_evidence_but_discards_partial_episodes():
    gate = ObjectAssistanceGate(2, "cpu", 3, minimum_episodes=2, minimum_steps=0)
    gate.reset(slice(None), torch.ones(2), torch.zeros(2), 0)
    gate.reset([0, 1], torch.ones(2), torch.ones(2), 100)
    gate.reset([0], torch.ones(2), torch.ones(2), 200)
    saved = gate.state_dict()
    restored = ObjectAssistanceGate(2, "cpu", 3, minimum_episodes=2, minimum_steps=0)
    restored.load_state_dict(saved)
    assert restored.stage == 1
    assert restored.episodes == restored.successes == 1
    assert restored.reset(None, torch.ones(2), torch.zeros(2), 200) == 1
    assert restored.episodes == 1
    assert restored.reset([0], torch.ones(2), torch.ones(2), 300) == 2
    incompatible = ObjectAssistanceGate(
        2, "cpu", 3, minimum_episodes=4, minimum_steps=0
    )
    with pytest.raises(ValueError, match="configuration changed"):
        incompatible.load_state_dict(saved)
