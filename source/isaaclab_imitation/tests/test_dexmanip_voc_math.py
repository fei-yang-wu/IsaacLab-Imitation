# SPDX-License-Identifier: BSD-3-Clause
"""Pure-Torch tests for the Newton virtual object controller math."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


_MODULE_PATH = (
    Path(__file__).parent.parent
    / "isaaclab_imitation"
    / "tasks"
    / "manager_based"
    / "dexmanip"
    / "voc_math.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location("voc_math", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
voc_math = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(voc_math)


def _identity(num_envs: int) -> torch.Tensor:
    quaternion = torch.zeros(num_envs, 4)
    quaternion[:, 3] = 1.0
    return quaternion


def _z_rotation(angle: float, num_envs: int = 1) -> torch.Tensor:
    quaternion = torch.zeros(num_envs, 4)
    quaternion[:, 2] = math.sin(angle / 2.0)
    quaternion[:, 3] = math.cos(angle / 2.0)
    return quaternion


def _wrench(
    *,
    current_position: torch.Tensor,
    current_orientation: torch.Tensor,
    velocity: torch.Tensor,
    target_position: torch.Tensor,
    target_orientation: torch.Tensor,
    scale: torch.Tensor | float = 1.0,
    linear_stiffness: float = 10.0,
    linear_damping: float = 2.0,
    angular_stiffness: float = 4.0,
    angular_damping: float = 0.5,
    max_force: float = 100.0,
    max_torque: float = 100.0,
    **gravity,
) -> tuple[torch.Tensor, torch.Tensor]:
    return voc_math.compute_pose_pd_wrench_xyzw(
        current_position,
        current_orientation,
        velocity,
        target_position,
        target_orientation,
        scale,
        linear_stiffness=linear_stiffness,
        linear_damping=linear_damping,
        angular_stiffness=angular_stiffness,
        angular_damping=angular_damping,
        max_force=max_force,
        max_torque=max_torque,
        **gravity,
    )


def test_pose_pd_force_and_damping_are_in_the_object_frame() -> None:
    orientation = _z_rotation(math.pi / 2.0)
    force, torque = _wrench(
        current_position=torch.zeros(1, 3),
        current_orientation=orientation,
        velocity=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        target_position=torch.tensor([[1.0, 0.0, 0.0]]),
        target_orientation=orientation,
    )

    # World +X is body -Y after a +90 degree object rotation.
    torch.testing.assert_close(
        force, torch.tensor([[0.0, -8.0, 0.0]]), atol=1.0e-5, rtol=0.0
    )
    torch.testing.assert_close(torque, torch.zeros(1, 3))


def test_orientation_pd_uses_xyzw_and_the_shortest_quaternion_sign() -> None:
    target = _z_rotation(math.pi / 2.0)
    common = dict(
        current_position=torch.zeros(1, 3),
        current_orientation=_identity(1),
        velocity=torch.zeros(1, 6),
        target_position=torch.zeros(1, 3),
    )
    _, torque = _wrench(target_orientation=target, **common)
    _, negative_torque = _wrench(target_orientation=-target, **common)

    expected = torch.tensor([[0.0, 0.0, 2.0 * math.pi]])
    torch.testing.assert_close(torque, expected, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(negative_torque, expected, atol=1.0e-6, rtol=0.0)


def test_gravity_compensation_uses_mass_scale_and_force_clamp() -> None:
    force, torque = _wrench(
        current_position=torch.zeros(2, 3),
        current_orientation=_identity(2),
        velocity=torch.zeros(2, 6),
        target_position=torch.zeros(2, 3),
        target_orientation=_identity(2),
        scale=torch.tensor([[0.5], [0.0]]),
        max_force=8.0,
        body_mass=torch.tensor([[[2.0]], [[1.0]]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        gravity_magnitude=10.0,
    )

    torch.testing.assert_close(force, torch.tensor([[0.0, 0.0, 8.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(torque, torch.zeros(2, 3))


def test_angular_damping_and_controller_scale_are_per_environment() -> None:
    force, torque = _wrench(
        current_position=torch.zeros(2, 3),
        current_orientation=_identity(2),
        velocity=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 4.0]]
        ),
        target_position=torch.zeros(2, 3),
        target_orientation=_identity(2),
        scale=torch.tensor([[1.0], [0.25]]),
    )

    torch.testing.assert_close(force, torch.zeros(2, 3))
    torch.testing.assert_close(torque[:, 2], torch.tensor([-2.0, -0.5]))


def test_partial_gravity_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="provided together"):
        _wrench(
            current_position=torch.zeros(1, 3),
            current_orientation=_identity(1),
            velocity=torch.zeros(1, 6),
            target_position=torch.zeros(1, 3),
            target_orientation=_identity(1),
            body_mass=torch.ones(1),
        )
