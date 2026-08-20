# SPDX-License-Identifier: BSD-3-Clause
"""Pure-Torch tests for the backend-independent dexmanip math adapter."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Load this torch-only module without importing the Isaac Lab task registry.
_MODULE_PATH = (
    Path(__file__).parent.parent
    / "isaaclab_imitation"
    / "tasks"
    / "manager_based"
    / "dexmanip"
    / "mdp.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location("mdp", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
mdp = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(mdp)


def _identity(batch: int, *middle: int) -> torch.Tensor:
    quaternion = torch.zeros(batch, *middle, 4)
    quaternion[..., 0] = 1.0
    return quaternion


def _z_rotation(angle: float, batch: int = 1) -> torch.Tensor:
    quaternion = torch.zeros(batch, 4)
    quaternion[:, 0] = math.cos(angle / 2.0)
    quaternion[:, 3] = math.sin(angle / 2.0)
    return quaternion


class _CommandManager:
    def __init__(self, command) -> None:
        self.command = command

    def get_term(self, name: str):
        assert name == "test_command"
        return self.command


def _env(command):
    return SimpleNamespace(command_manager=_CommandManager(command))


def test_pose_compose_relative_and_quaternion_sign_are_consistent() -> None:
    parent_position = torch.tensor([[1.0, 2.0, 0.0]])
    parent_wxyz = _z_rotation(math.pi / 2.0)
    child_position_o = torch.tensor([[1.0, 0.0, 0.0]])
    child_wxyz_o = _z_rotation(math.pi / 4.0)

    child_position, child_wxyz = mdp.compose_pose(
        parent_position,
        parent_wxyz,
        child_position_o,
        child_wxyz_o,
    )
    recovered_position, recovered_wxyz = mdp.relative_pose(
        parent_position,
        parent_wxyz,
        child_position,
        child_wxyz,
    )

    torch.testing.assert_close(child_position, torch.tensor([[1.0, 3.0, 0.0]]))
    torch.testing.assert_close(recovered_position, child_position_o)
    torch.testing.assert_close(recovered_wxyz, child_wxyz_o)
    torch.testing.assert_close(
        mdp.quat_error_magnitude(child_wxyz, -child_wxyz),
        torch.zeros(1),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_root_link_twist_is_converted_to_isaac_com_velocity() -> None:
    link_twist = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
    link_wxyz = _z_rotation(math.pi / 2.0)
    com_position_b = torch.tensor([[1.0, 0.0, 0.0]])

    com_twist = mdp.root_link_twist_to_com_twist_w(
        link_twist,
        link_wxyz,
        com_position_b,
    )

    # The rotated COM offset is world +Y. omega(+Z) x offset(+Y) is -X.
    torch.testing.assert_close(
        com_twist,
        torch.tensor([[-1.0, 0.0, 0.0, 0.0, 0.0, 2.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_object_relative_command_follows_live_object_pose() -> None:
    reference_object_position = torch.zeros(1, 3)
    reference_object_wxyz = _identity(1)
    reference_wrist_position = torch.tensor([[1.0, 0.0, 0.0]])
    reference_wrist_wxyz = _identity(1)
    live_object_position = torch.tensor([[2.0, 3.0, 0.0]])
    live_object_wxyz = _z_rotation(math.pi / 2.0)

    live_wrist_position, live_wrist_wxyz = mdp.retarget_pose_to_live_object(
        reference_object_position,
        reference_object_wxyz,
        live_object_position,
        live_object_wxyz,
        reference_wrist_position,
        reference_wrist_wxyz,
    )

    torch.testing.assert_close(
        live_wrist_position, torch.tensor([[2.0, 4.0, 0.0]]), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(live_wrist_wxyz, live_object_wxyz)


def test_world_contacts_are_rotated_to_object_wrench_support() -> None:
    object_position = torch.tensor([[[10.0, 0.0, 0.0]]])
    object_wxyz = _z_rotation(math.pi / 2.0).unsqueeze(1)
    contact_position = object_position.unsqueeze(2).clone()
    # World +Y becomes object-frame +X under the inverse object rotation.
    contact_force = torch.tensor([[[[0.0, 2.0, 0.0]]]])
    basis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    support = mdp.compute_contact_wrench_supports(
        contact_position,
        contact_force,
        object_position,
        object_wxyz,
        basis,
        object_radii=0.5,
        contact_active=torch.ones(1, 1, 1, dtype=torch.bool),
        num_friction_cone_edges=4,
        friction_coefficient=0.0,
    )

    torch.testing.assert_close(
        support,
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_contact_offset_produces_scaled_torque_support() -> None:
    contact_position = torch.tensor([[[[0.0, 1.0, 0.0]]]])
    contact_force = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    basis = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])

    support = mdp.compute_contact_wrench_supports(
        contact_position,
        contact_force,
        torch.zeros(1, 1, 3),
        _identity(1, 1),
        basis,
        object_radii=0.5,
        contact_active=torch.ones(1, 1, 1, dtype=torch.bool),
        friction_coefficient=0.0,
    )

    # r cross f = -Z, then the released recipe scales torque by the object radius.
    torch.testing.assert_close(support, torch.tensor([[[2.0]]]))


def test_friction_cone_default_preserves_released_pure_math_coefficient() -> None:
    normals = torch.tensor([[[0.0, 0.0, 1.0]]])
    cos_phase, sin_phase = mdp.friction_cone_phase(
        4,
        device="cpu",
        dtype=torch.float32,
    )

    default_edges = mdp.compute_friction_cone_edges(
        normals,
        cos_phase,
        sin_phase,
    )
    released_pure_edges = mdp.compute_friction_cone_edges(
        normals,
        cos_phase,
        sin_phase,
        friction_coefficient=0.5,
    )
    command_edges = mdp.compute_friction_cone_edges(
        normals,
        cos_phase,
        sin_phase,
        friction_coefficient=0.1,
    )

    torch.testing.assert_close(default_edges, released_pure_edges)
    assert not torch.allclose(default_edges, command_edges)


def test_start_frame_sampling_respects_each_reference_length() -> None:
    lengths = torch.tensor([2, 3, 5, 8, 13, 21, 34, 55])
    generator = torch.Generator().manual_seed(29)

    starts = mdp.sample_reference_start_frames(
        lengths,
        reset_to_first_frame_prob=0.0,
        virtual_object_control_scale=0.0,
        generator=generator,
    )

    assert torch.all(starts >= 0)
    assert torch.all(starts < lengths - 1)


def test_first_frame_mix_starts_only_after_voc_scale_is_below_threshold() -> None:
    lengths = torch.full((32,), 101, dtype=torch.long)
    expected_generator = torch.Generator().manual_seed(17)
    expected = mdp.sample_reference_start_frames(
        lengths,
        reset_to_first_frame_prob=0.0,
        virtual_object_control_scale=0.0,
        generator=expected_generator,
    )
    first_frame_mask = torch.rand(32, generator=expected_generator) < 0.5
    assert torch.any(first_frame_mask)
    assert torch.any(~first_frame_mask)
    expected[first_frame_mask] = 0

    actual = mdp.sample_reference_start_frames(
        lengths,
        reset_to_first_frame_prob=0.5,
        virtual_object_control_scale=0.099,
        generator=torch.Generator().manual_seed(17),
    )
    at_threshold = mdp.sample_reference_start_frames(
        lengths,
        reset_to_first_frame_prob=1.0,
        virtual_object_control_scale=0.1,
        generator=torch.Generator().manual_seed(17),
    )
    random_only = mdp.sample_reference_start_frames(
        lengths,
        reset_to_first_frame_prob=0.0,
        virtual_object_control_scale=0.0,
        generator=torch.Generator().manual_seed(17),
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(at_threshold, random_only)


def test_always_first_frame_overrides_random_sampling_and_voc_scale() -> None:
    starts = mdp.sample_reference_start_frames(
        torch.tensor([2, 5, 11, 23]),
        always_reset_to_first_frame=True,
        reset_to_first_frame_prob=0.0,
        virtual_object_control_scale=1.0,
        generator=torch.Generator().manual_seed(41),
    )

    torch.testing.assert_close(starts, torch.zeros(4, dtype=torch.long))


def test_virtual_object_controller_warmup_decays_smoothly() -> None:
    scale = mdp.virtual_object_controller_scale_schedule(
        torch.tensor([0, 5, 10, 15]),
        warmup_steps=10,
        curriculum_scale=0.2,
    )

    torch.testing.assert_close(scale.squeeze(-1), torch.tensor([1.0, 0.6, 0.2, 0.2]))


def test_unassisted_object_control_is_zero_from_reset_frame_zero() -> None:
    scale = mdp.virtual_object_controller_scale_schedule(
        torch.tensor([0, 1, 20]),
        warmup_steps=20,
        curriculum_scale=1.0,
        force_unassisted=True,
    )

    torch.testing.assert_close(scale, torch.zeros(3, 1))


def test_contact_support_reward_and_missed_fraction_have_expected_values() -> None:
    right_command = torch.tensor(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    right_current = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    left_command = torch.zeros_like(right_command)
    left_current = torch.zeros_like(right_current)

    reward = mdp.contact_wrench_support_reward_tensors(
        right_command,
        right_current,
        left_command,
        left_current,
    )
    missed = mdp.missed_contact_penalty_tensors(
        right_command,
        right_current,
        left_command,
        left_current,
    )

    torch.testing.assert_close(reward, torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(missed, torch.tensor([0.5, 0.0]))


def test_unintended_contact_combines_body_count_and_support_magnitude() -> None:
    right_command = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    right_current = torch.tensor([[[1.0, 1.0], [2.0, 0.0]]])
    left_command = torch.zeros_like(right_command)
    left_current = torch.zeros_like(right_current)

    penalty = mdp.unintended_contact_penalty_tensors(
        right_command,
        right_current,
        left_command,
        left_current,
    )

    # 0.5 unintended-body fraction + mean([2^2, 0^2]) = 2.5.
    torch.testing.assert_close(penalty, torch.tensor([2.5]))


def test_contact_reward_wrappers_use_only_the_command_manager_interface() -> None:
    command = SimpleNamespace(
        right_hand_contact_wrench_supports_command=torch.tensor([[[1.0, 1.0]]]),
        right_hand_contact_wrench_supports=torch.tensor([[[1.0, 1.0]]]),
        left_hand_contact_wrench_supports_command=torch.zeros(1, 1, 2),
        left_hand_contact_wrench_supports=torch.zeros(1, 1, 2),
    )

    reward = mdp.contact_wrench_support_reward(
        _env(command), command_name="test_command"
    )

    torch.testing.assert_close(reward, torch.ones(1))


def test_tracking_observations_and_terminations_use_command_tensors() -> None:
    identity = _identity(2)
    object_identity = _identity(2, 1)
    command = SimpleNamespace(
        right_hand_wrist_position_e=torch.tensor([[0.0, 0.0, 0.0], [0.31, 0.0, 0.0]]),
        left_hand_wrist_position_e=torch.zeros(2, 3),
        right_hand_wrist_wxyz_e=identity.clone(),
        left_hand_wrist_wxyz_e=identity.clone(),
        right_hand_wrist_velocity_b=torch.ones(2, 6),
        left_hand_wrist_velocity_b=-torch.ones(2, 6),
        right_hand_wrist_pose_command_e=torch.cat(
            (torch.zeros(2, 3), identity), dim=-1
        ),
        left_hand_wrist_pose_command_e=torch.cat((torch.zeros(2, 3), identity), dim=-1),
        object_position_e=torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.25]]]),
        object_orientation_e=object_identity.clone(),
        object_body_position_command_e=torch.zeros(2, 1, 3),
        object_body_wxyz_command_e=object_identity.clone(),
        timestep_counter=torch.tensor([3, 4]),
        retargeted_horizon=5,
    )
    env = _env(command)

    assert mdp.wrist_position_e(env, "test_command").shape == (2, 6)
    assert mdp.wrist_orientation_e(env, "test_command").shape == (2, 8)
    assert mdp.wrist_velocity_b(env, "test_command").shape == (2, 12)
    assert mdp.object_position_e(env, "test_command").shape == (2, 3)
    assert mdp.object_orientation_e(env, "test_command").shape == (2, 4)
    assert mdp.object_wrist_pose_command(env, "test_command").shape == (
        2,
        21,
    )
    assert mdp.hand_wrist_away_from_trajectory(
        env, "test_command", threshold=0.2
    ).tolist() == [False, True]
    assert mdp.object_away_from_trajectory_z(
        env, "test_command", threshold=0.2
    ).tolist() == [False, True]
    assert mdp.object_away_from_trajectory(
        env,
        "test_command",
        position_threshold=0.2,
        orientation_threshold=0.7,
    ).tolist() == [False, True]
    assert mdp.timestep_timeout(env, "test_command").tolist() == [False, True]


def test_tracking_rewards_are_one_at_the_reference() -> None:
    command = SimpleNamespace(
        object_position_e=torch.zeros(2, 1, 3),
        object_body_position_command_e=torch.zeros(2, 1, 3),
        object_orientation_e=_identity(2, 1),
        object_body_wxyz_command_e=_identity(2, 1),
        object_twist_w=torch.zeros(2, 1, 6),
        object_body_twist_command_w=torch.zeros(2, 1, 6),
        left_hand_wrist_pose_command_e=torch.cat(
            (torch.zeros(2, 3), _identity(2)), dim=-1
        ),
        right_hand_wrist_pose_command_e=torch.cat(
            (torch.zeros(2, 3), _identity(2)), dim=-1
        ),
        left_hand_fingertip_position_command_e=torch.zeros(2, 5, 3),
        right_hand_fingertip_position_command_e=torch.zeros(2, 5, 3),
        left_hand_wrist_position_e=torch.zeros(2, 3),
        right_hand_wrist_position_e=torch.zeros(2, 3),
        left_hand_fingertip_position_e=torch.zeros(2, 5, 3),
        right_hand_fingertip_position_e=torch.zeros(2, 5, 3),
    )
    env = _env(command)

    torch.testing.assert_close(
        mdp.object_position_tracking_exp(env, "test_command"), torch.ones(2)
    )
    torch.testing.assert_close(
        mdp.object_orientation_tracking_exp(env, "test_command"),
        torch.ones(2),
    )
    torch.testing.assert_close(
        mdp.object_twist_tracking_exp(env, "test_command"), torch.ones(2)
    )
    torch.testing.assert_close(
        mdp.hand_keypoints_tracking_exp(env, "test_command"), torch.ones(2)
    )


def test_object_twist_tracking_reward_drops_with_world_velocity_error() -> None:
    target = torch.zeros(2, 2, 6)
    current = target.clone()
    current[1, 0, :3] = 2.0
    current[1, 1, 3:] = 1.0

    reward = mdp.object_twist_tracking_exp_tensors(
        target,
        current,
        variance=1.0,
    )

    torch.testing.assert_close(reward[0], torch.tensor(1.0))
    assert 0.0 < reward[1] < reward[0]


def test_whole_robot_tracking_rewards_peak_at_reference_and_drop_with_error() -> None:
    target_position = torch.tensor([[0.0, 0.5, -0.5], [0.1, 0.2, 0.3]])
    target_velocity = torch.tensor([[0.0, 1.0, -1.0], [0.2, 0.0, -0.2]])
    live_position = target_position.clone()
    live_position[1, 2] += 0.6
    live_velocity = target_velocity.clone()
    live_velocity[1, 0] += 1.0
    command = SimpleNamespace(
        reference_joint_pos=target_position,
        reference_joint_vel=target_velocity,
        robot=SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=live_position),
                joint_vel=SimpleNamespace(torch=live_velocity),
            )
        ),
    )
    env = _env(command)

    position_reward = mdp.robot_joint_position_tracking_exp(
        env, "test_command", var=0.25
    )
    velocity_reward = mdp.robot_joint_velocity_tracking_exp(
        env, "test_command", var=1.0
    )

    torch.testing.assert_close(position_reward[0], torch.tensor(1.0))
    torch.testing.assert_close(velocity_reward[0], torch.tensor(1.0))
    assert 0.0 < float(position_reward[1]) < 1.0
    assert 0.0 < float(velocity_reward[1]) < 1.0


def test_whole_robot_deviation_is_range_normalized_and_warmup_masked() -> None:
    target = torch.zeros(3, 2)
    current = target.clone()
    current[0, 0] = 0.8  # 40% of the first joint's two-radian range.
    current[1, 1] = 0.7  # 35% of the second joint's two-radian range.
    limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]).repeat(3, 1, 1)
    active = torch.tensor([False, True, True])

    away = mdp.robot_joint_away_from_trajectory_tensors(
        target,
        current,
        limits,
        normalized_position_threshold=0.35,
        active=active,
    )

    # Env 0 exceeds the threshold but is still warming up; equality is allowed.
    assert away.tolist() == [False, False, False]
    active[0] = True
    assert mdp.robot_joint_away_from_trajectory_tensors(
        target,
        current,
        limits,
        normalized_position_threshold=0.35,
        active=active,
    ).tolist() == [True, False, False]


def test_contact_observation_rotates_position_and_force_to_wrist() -> None:
    wrist_position = torch.tensor([[1.0, 2.0, 0.0]])
    wrist_wxyz = _z_rotation(math.pi / 2.0)
    contact_position = torch.tensor([[[2.0, 2.0, 0.0]]])
    contact_force = torch.tensor([[[0.0, 3.0, 0.0]]])

    observation = mdp.contact_position_direction_in_wrist_frame(
        contact_position,
        contact_force,
        wrist_position,
        wrist_wxyz,
        torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(
        observation,
        torch.tensor([[[0.0, -1.0, 0.0, 1.0, 0.0, 0.0]]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_contact_observation_matches_released_feature_block_order() -> None:
    command = SimpleNamespace(
        left_hand_object_contact_positions_w=torch.tensor(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]]
        ),
        left_hand_object_contact_forces_w=torch.tensor(
            [[[[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]]
        ),
        left_hand_object_contact_active=torch.ones(1, 1, 2, dtype=torch.bool),
        left_hand_wrist_position_w=torch.zeros(1, 3),
        left_hand_wrist_wxyz_e=_identity(1),
        right_hand_object_contact_positions_w=torch.tensor(
            [[[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]]
        ),
        right_hand_object_contact_forces_w=torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, -4.0, 0.0]]]]
        ),
        right_hand_object_contact_active=torch.ones(1, 1, 2, dtype=torch.bool),
        right_hand_wrist_position_w=torch.zeros(1, 3),
        right_hand_wrist_wxyz_e=_identity(1),
    )

    observation = mdp.contact_position_direction_in_wrist(_env(command), "test_command")

    torch.testing.assert_close(
        observation,
        torch.tensor(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    3.0,
                    0.0,
                    0.0,
                    4.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                ]
            ]
        ),
    )


@pytest.mark.parametrize("invalid_radius", [0.0, -0.1])
def test_wrench_basis_rejects_non_positive_radius(invalid_radius: float) -> None:
    with pytest.raises(ValueError, match="object_radius must be positive"):
        mdp.sample_wrench_space_basis_scaled(
            8, invalid_radius, device=torch.device("cpu")
        )
