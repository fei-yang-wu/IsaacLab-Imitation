import numpy as np
import pytest

from imitation_experiments.audit.vega_sharpa_training import summarize


def records(count=3):
    return {
        name: np.array([[24 * i, float(i)] for i in range(1, count + 1)])
        for name in ["train/loss_objective", "train/loss_critic", "train/grad_norm"]
    }


def test_partial_run_is_not_reported_complete():
    result = summarize(records(), num_envs=1, expected_iterations=5)
    assert result["iterations"] == 3
    assert not result["complete"]


def test_missing_optimizer_rows_are_not_hidden_by_final_frame_count():
    values = records()
    values["train/grad_norm"] = values["train/grad_norm"][[0, 2]]
    with pytest.raises(ValueError, match="Missing optimization records"):
        summarize(values, num_envs=1, expected_iterations=3)


def test_assistance_transition_is_reported_at_its_actual_log_step():
    values = records()
    values["Curriculum/fixed_timestep"] = np.array([[24, 1.0], [48, 1.0], [72, 0.75]])
    result = summarize(values, num_envs=1, expected_iterations=3)
    assert result["complete"]
    assert result["assistance_changes"] == [{"iteration": 3.0, "scale": 0.75}]
