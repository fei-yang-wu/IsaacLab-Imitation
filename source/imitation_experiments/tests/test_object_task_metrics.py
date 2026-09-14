import numpy as np
import pytest

from imitation_experiments.evaluation.object_task_metrics import object_endpoint_metrics


def test_endpoint_success_requires_end_time_and_uses_all_trials():
    goal = [0, 0, 0, 1, 0, 0, 0]
    actual = np.array([goal, goal, [0.051, 0, 0, 1, 0, 0, 0]])
    result = object_endpoint_metrics(actual, goal, [0, 0, 0], [True, False, True])
    assert result["position_success"] == [True, False, False]
    assert result["position_success_rate"] == pytest.approx(1 / 3)
    assert result["position_success_rate_by_tolerance_m"]["0.1"] == pytest.approx(2 / 3)


def test_box_center_and_root_origin_are_distinguished():
    goal = [0, 0, 0, 1, 0, 0, 0]
    # Rotate 180 degrees around the same COM, with a 10 cm offset from root.
    actual = [[0.2, 0, 0, 0, 0, 0, 1]]
    result = object_endpoint_metrics(actual, goal, [0.1, 0, 0], [True])
    assert result["position_error_m"]["mean"] < 1e-12
    assert result["root_position_error_m"]["mean"] == pytest.approx(0.2)
    assert result["position_success_rate"] == 1
    assert result["pose_success_rate"] == 0


def test_quaternion_sign_does_not_change_object_success():
    result = object_endpoint_metrics(
        [[1, 2, 3, -1, 0, 0, 0]], [1, 2, 3, 1, 0, 0, 0], [0.1, 0.2, 0.3], [True]
    )
    assert result["pose_success_rate"] == 1
    assert result["orientation_error_rad"]["mean"] == 0


def test_nonfinite_object_pose_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        object_endpoint_metrics(
            [[np.nan, 0, 0, 1, 0, 0, 0]], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0], [True]
        )
