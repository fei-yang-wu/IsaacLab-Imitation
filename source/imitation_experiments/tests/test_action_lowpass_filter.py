"""Guards for the inference-time action low-pass in `evaluate_checkpoint`.

The evaluator launches Isaac Sim at import time, so these tests inspect the
entrypoint source and re-implement the filter's arithmetic on tensors.
"""

from __future__ import annotations

import torch

from imitation_experiments.paths import REPO_ROOT


EVALUATOR = (
    REPO_ROOT
    / "source/imitation_experiments/imitation_experiments/lowlevel/evaluate_checkpoint.py"
)


def _source() -> str:
    return EVALUATOR.read_text(encoding="utf-8")


def test_lowpass_flag_defaults_to_off() -> None:
    source = _source()
    assert '"--action_lowpass_alpha"' in source
    assert "default=1.0" in source
    assert "if not 0.0 < action_lowpass_alpha <= 1.0:" in source


def test_lowpass_runs_before_the_action_metrics() -> None:
    """The filter must sit between the policy and the metrics.

    `action_delta_l2` has to describe what the actuator received, not what the
    policy proposed, or a filtered row would report the unfiltered roughness.
    """
    source = _source()
    apply_at = source.index('td.set("action", action)')
    metric_at = source.index('"action_delta_l2", action_delta_l2, step_active')
    step_at = source.index("td_step = env.step(td)")
    assert apply_at < metric_at, "filter must precede the action metrics"
    assert apply_at < step_at, "filter must precede env.step"


def test_lowpass_reseeds_on_reset() -> None:
    source = _source()
    assert "filter_valid" in source
    assert "filter_valid & ~done_any" in source
    assert "torch.where(" in source


def test_lowpass_arithmetic_matches_first_order_filter() -> None:
    alpha = 0.4
    raw = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    state = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    valid = torch.tensor([True, False])
    blended = (1.0 - alpha) * state + alpha * raw
    out = torch.where(valid.unsqueeze(-1), blended, raw)
    # Row 0 keeps its history; row 1 just reset and takes the raw action.
    torch.testing.assert_close(out[0], torch.tensor([0.4, 0.0]))
    torch.testing.assert_close(out[1], torch.tensor([0.0, 2.0]))


def test_lowpass_alpha_is_recorded_in_metadata() -> None:
    assert '"action_lowpass_alpha": action_lowpass_alpha,' in _source()
