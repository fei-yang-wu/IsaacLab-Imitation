"""Focused tensor and wrapper tests for Vega-Wuji physics safety terms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab_imitation.tasks.manager_based.dexmanip import mdp


def _support_forces() -> torch.Tensor:
    forces = torch.zeros(3, 2, 2, 3)
    # Sum counterpart magnitudes per robot body: max(3 + 4, 6) = 7 N.
    forces[1, 0, 0, 0] = 3.0
    forces[1, 0, 1, 1] = 4.0
    forces[1, 1, 0, 0] = 6.0
    # Sensor corruption must fail closed.
    forces[2, 0, 0, 0] = float("nan")
    return forces


def test_support_force_aggregation_is_conservative_and_fail_closed() -> None:
    maximum = mdp.maximum_support_contact_force_tensors(_support_forces())

    assert maximum[0].item() == pytest.approx(0.0)
    assert maximum[1].item() == pytest.approx(7.0)
    assert torch.isposinf(maximum[2])


def test_support_force_penalty_is_bounded_and_matches_hard_termination() -> None:
    forces = _support_forces()

    penalty = mdp.support_contact_force_penalty_tensors(
        forces,
        penalty_start_force=5.0,
        saturation_force=15.0,
    )
    terminated = mdp.excessive_support_contact_force_tensors(
        forces,
        threshold=6.5,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.2, 1.0]))
    assert terminated.tolist() == [False, True, True]


def test_support_force_contract_rejects_missing_filters_and_bad_thresholds() -> None:
    with pytest.raises(ValueError, match="filtered support counterpart"):
        mdp.maximum_support_contact_force_tensors(torch.zeros(2, 3, 0, 3))
    with pytest.raises(ValueError, match="exceed penalty_start_force"):
        mdp.support_contact_force_penalty_tensors(
            torch.zeros(2, 3, 1, 3),
            penalty_start_force=5.0,
            saturation_force=5.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        mdp.excessive_support_contact_force_tensors(
            torch.zeros(2, 3, 1, 3),
            threshold=float("inf"),
        )


def test_support_force_wrappers_read_newton_proxy_force_matrix() -> None:
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            force_matrix_w=SimpleNamespace(torch=_support_forces()),
        )
    )
    env = SimpleNamespace(scene={"robot_support_contacts": sensor})

    torch.testing.assert_close(
        mdp.support_contact_force_penalty(
            env,
            penalty_start_force=5.0,
            saturation_force=15.0,
        ),
        torch.tensor([0.0, 0.2, 1.0]),
    )
    assert mdp.excessive_support_contact_force(
        env,
        threshold=6.5,
    ).tolist() == [False, True, True]


def test_nonfinite_physics_state_combines_all_live_state_tensors() -> None:
    joint_state = torch.zeros(3, 4)
    body_state = torch.zeros(3, 2, 7)
    joint_state[1, 0] = float("nan")
    body_state[2, 1, 3] = float("inf")

    invalid = mdp.nonfinite_physics_state_tensors(joint_state, body_state)

    assert invalid.tolist() == [False, True, True]


def test_nonfinite_physics_wrapper_checks_robot_and_object_pose_and_velocity() -> None:
    def proxy(value: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(torch=value)

    num_envs = 2
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=proxy(torch.zeros(num_envs, 2)),
            joint_vel=proxy(torch.zeros(num_envs, 2)),
            body_link_pose_w=proxy(torch.zeros(num_envs, 3, 7)),
            body_link_vel_w=proxy(torch.zeros(num_envs, 3, 6)),
        )
    )
    object_velocity = torch.zeros(num_envs, 6)
    object_velocity[1, 2] = float("nan")
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pose_w=proxy(torch.zeros(num_envs, 7)),
            root_link_vel_w=proxy(object_velocity),
        )
    )
    command = SimpleNamespace(robot=robot, objects=[object_asset])
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )

    assert mdp.nonfinite_physics_state(env, "motion").tolist() == [False, True]
