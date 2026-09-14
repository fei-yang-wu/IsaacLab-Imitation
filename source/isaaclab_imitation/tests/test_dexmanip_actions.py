"""Pure tensor tests for the Vega residual action adapter."""

from __future__ import annotations

import pytest
import torch

from isaaclab_imitation.tasks.manager_based.dexmanip.actions import (
    SIDE_ORDER,
    VEGA_WUJI_JOINT_COUNT,
    VegaWujiReferenceResidualAction,
    clip_vector_norm,
    damped_least_squares_delta,
    reference_residual_joint_target,
    validate_reference_joint_order,
    wuji_anatomical_joint_limits,
)


def test_action_side_order_and_layout_match_released_recipe() -> None:
    assert SIDE_ORDER == ("right", "left")
    assert 2 * (3 + 3 + 20) == 52


def test_reference_residual_action_covers_all_59_live_actuators() -> None:
    names = tuple(f"joint_{index}" for index in range(VEGA_WUJI_JOINT_COUNT))

    assert validate_reference_joint_order(names, names) == names


def test_reference_residual_action_rejects_missing_or_reordered_joint() -> None:
    names = tuple(f"joint_{index}" for index in range(VEGA_WUJI_JOINT_COUNT))

    with pytest.raises(ValueError, match="exactly 59 live joints"):
        validate_reference_joint_order(names[:-1], names[:-1])
    with pytest.raises(ValueError, match="live actuator order"):
        validate_reference_joint_order(tuple(reversed(names)), names)


def test_processed_action_is_arm_then_finger_target_subset() -> None:
    action = object.__new__(VegaWujiReferenceResidualAction)
    action._processed_actions = torch.arange(2 * VEGA_WUJI_JOINT_COUNT).reshape(
        2, VEGA_WUJI_JOINT_COUNT
    )
    action._side_joint_ids = {
        "right": [*range(3, 10), *range(19, 39)],
        "left": [*range(10, 17), *range(39, 59)],
    }

    torch.testing.assert_close(
        action.processed_actions,
        action._processed_actions,
    )
    torch.testing.assert_close(
        action.processed_action("right"),
        action._processed_actions[:, action._side_joint_ids["right"]],
    )
    assert action.processed_action("left").shape == (2, 27)
    with pytest.raises(KeyError, match="Unknown Vega-Wuji hand side"):
        action.processed_action("middle")


def test_reference_residual_target_uses_tanh_ema_and_per_joint_scale() -> None:
    raw_action = torch.tensor([[0.0, 1.0, -2.0]])
    previous = torch.tensor([[0.1, -0.2, 0.3]])
    reference = torch.tensor([[0.5, 1.0, -1.0]])
    scale = torch.tensor([0.05, 0.15, 0.15])
    limits = torch.tensor([[[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]])

    residual, target = reference_residual_joint_target(
        raw_action,
        previous,
        reference,
        scale,
        limits,
        ema_factor=0.25,
    )

    expected_residual = 0.25 * previous + 0.75 * torch.tanh(raw_action) * scale
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(target, reference + expected_residual)


def test_reference_residual_target_clips_only_final_target_to_soft_limits() -> None:
    raw_action = torch.tensor([[10.0, -10.0]])
    previous = torch.zeros_like(raw_action)
    reference = torch.tensor([[0.19, -0.19]])
    scale = torch.tensor([0.15, 0.15])
    limits = torch.tensor([[[-0.2, 0.2], [-0.2, 0.2]]])

    residual, target = reference_residual_joint_target(
        raw_action,
        previous,
        reference,
        scale,
        limits,
        ema_factor=0.0,
    )

    torch.testing.assert_close(residual, torch.tanh(raw_action) * scale)
    torch.testing.assert_close(target, torch.tensor([[0.2, -0.2]]))


def test_wuji_runtime_limits_exclude_interphalangeal_hyperextension() -> None:
    names = (
        "R_arm_j1",
        "r_index_finger_mcp_abd",
        "r_index_finger_pip",
        "r_index_finger_dip",
        "r_thumb_mcp",
        "r_thumb_ip",
    )
    source = torch.tensor(
        [[[-2.0, 2.0] for _ in names], [[-1.0, 1.0] for _ in names]],
        dtype=torch.float64,
    )
    original = source.clone()

    actual = wuji_anatomical_joint_limits(names, source)

    torch.testing.assert_close(actual[:, 0], source[:, 0])
    torch.testing.assert_close(
        actual[:, 1],
        torch.tensor([[-0.6981317, 0.6981317]] * 2, dtype=torch.float64),
    )
    torch.testing.assert_close(actual[:, 2, 0], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        actual[:, 2, 1], torch.tensor([1.74532925, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(actual[:, 3, 0], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        actual[:, 3, 1], torch.tensor([1.3962634, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(actual[:, 4, 0], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(actual[:, 5, 0], torch.zeros(2, dtype=torch.float64))
    # The live thumb-IP upper bound is already tighter than the 100 deg cap.
    torch.testing.assert_close(
        actual[:, 5, 1], torch.tensor([1.74532925, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(source, original)


def test_damped_least_squares_solves_identity_task() -> None:
    jacobian = torch.eye(3).unsqueeze(0)
    error = torch.tensor([[1.0, -2.0, 0.5]])

    result = damped_least_squares_delta(jacobian, error, damping=1.0e-3)

    torch.testing.assert_close(result, error, atol=3.0e-6, rtol=0.0)


def test_damped_least_squares_uses_null_posture_only_outside_task() -> None:
    jacobian = torch.tensor([[[1.0, 0.0]]])
    result = damped_least_squares_delta(
        jacobian,
        torch.zeros(1, 1),
        damping=1.0e-3,
        posture_error=torch.tensor([[2.0, 3.0]]),
        posture_gain=0.5,
    )

    assert result[0, 0].item() == pytest.approx(0.0, abs=1.0e-5)
    assert result[0, 1].item() == pytest.approx(1.5)


def test_clip_vector_norm_preserves_direction_and_small_vectors() -> None:
    value = torch.tensor([[3.0, 4.0], [0.3, 0.4]])
    result = clip_vector_norm(value, 1.0)

    torch.testing.assert_close(result[0], torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(result[1], value[1])


@pytest.mark.parametrize("damping", (0.0, -1.0))
def test_damped_least_squares_rejects_nonpositive_damping(damping: float) -> None:
    with pytest.raises(ValueError, match="damping must be positive"):
        damped_least_squares_delta(
            torch.ones(1, 1, 1), torch.ones(1, 1), damping=damping
        )
