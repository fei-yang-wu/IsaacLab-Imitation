"""The declared five-centimetre reward scale has measurable consequences."""

import importlib.util
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).parent.parent / "isaaclab_imitation/contracts/object_task_rewards.py"
)
_SPEC = importlib.util.spec_from_file_location("object_task_rewards", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_box_reward_uses_metric_distance_and_is_translation_invariant():
    actual = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.1, 0.0]])
    target = torch.zeros_like(actual)
    expected = torch.exp(torch.tensor([0.0, -1.0, -4.0]))
    torch.testing.assert_close(
        _MODULE.box_center_tracking_reward(actual, target), expected
    )
    torch.testing.assert_close(
        _MODULE.box_center_tracking_reward(actual + 3.0, target + 3.0), expected
    )


@pytest.mark.parametrize("scale", [0.0, -0.1, float("nan"), float("inf")])
def test_box_reward_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        _MODULE.box_center_tracking_reward(torch.zeros(1, 3), torch.zeros(1, 3), scale)
