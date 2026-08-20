# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pure Torch pose-PD math for the virtual object controller.

This module adapts the virtual rigid-object controller from NVIDIA
``video_to_data`` commit ``afa9dffda748e301ddf79b7002ce4cc5cb552adb``.
It has no Isaac Lab or simulator imports. All quaternions use XYZW order.
"""

from __future__ import annotations

import torch


def _validate_batch(name: str, value: torch.Tensor, width: int) -> None:
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(
            f"{name} must have shape [num_envs, {width}]; got {tuple(value.shape)}."
        )


def quat_normalize_xyzw(
    quaternion: torch.Tensor, *, eps: float = 1.0e-8
) -> torch.Tensor:
    """Return unit XYZW quaternions."""

    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(eps)


def quat_multiply_xyzw(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose XYZW quaternions as ``first * second``."""

    first_xyz, first_w = first[..., :3], first[..., 3:]
    second_xyz, second_w = second[..., :3], second[..., 3:]
    xyz = (
        first_w * second_xyz
        + second_w * first_xyz
        + torch.linalg.cross(first_xyz, second_xyz, dim=-1)
    )
    w = first_w * second_w - (first_xyz * second_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_inverse_xyzw(quaternion: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Return the inverse of XYZW quaternions."""

    norm_squared = quaternion.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1) / norm_squared


def quat_rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors with broadcast-compatible XYZW quaternions."""

    quaternion = quat_normalize_xyzw(quaternion)
    quaternion_xyz = quaternion[..., :3]
    twice_cross = 2.0 * torch.linalg.cross(quaternion_xyz, vector, dim=-1)
    return (
        vector
        + quaternion[..., 3:] * twice_cross
        + torch.linalg.cross(quaternion_xyz, twice_cross, dim=-1)
    )


def quat_rotate_inverse_xyzw(
    quaternion: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Rotate world vectors into the frame of an XYZW quaternion."""

    return quat_rotate_xyzw(quat_inverse_xyzw(quaternion), vector)


def axis_angle_from_quat_xyzw(
    quaternion: torch.Tensor, *, eps: float = 1.0e-8
) -> torch.Tensor:
    """Return the shortest axis-angle vector for an XYZW quaternion."""

    quaternion = quat_normalize_xyzw(quaternion, eps=eps)
    # q and -q encode the same rotation. Select the representative whose
    # scalar part is non-negative so the result is the shortest rotation.
    quaternion = torch.where(quaternion[..., 3:] < 0.0, -quaternion, quaternion)
    xyz = quaternion[..., :3]
    xyz_norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, quaternion[..., 3:].clamp_min(eps))
    regular_scale = angle / xyz_norm.clamp_min(eps)
    scale = torch.where(xyz_norm > eps, regular_scale, torch.full_like(angle, 2.0))
    return xyz * scale


def _per_environment_column(
    value: torch.Tensor | float,
    *,
    num_envs: int,
    like: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if tensor.ndim == 0:
        return tensor.expand(num_envs).unsqueeze(-1)
    if tensor.shape == (num_envs,):
        return tensor.unsqueeze(-1)
    if tensor.shape == (num_envs, 1):
        return tensor
    if tensor.numel() == num_envs and tensor.shape[0] == num_envs:
        return tensor.reshape(num_envs, 1)
    raise ValueError(
        f"{name} must be scalar or contain one value per environment; "
        f"got {tuple(tensor.shape)}."
    )


def compute_pose_pd_wrench_xyzw(
    current_position_w: torch.Tensor,
    current_orientation_xyzw_w: torch.Tensor,
    current_com_velocity_w: torch.Tensor,
    target_position_w: torch.Tensor,
    target_orientation_xyzw_w: torch.Tensor,
    controller_scale: torch.Tensor | float,
    *,
    linear_stiffness: float,
    linear_damping: float,
    angular_stiffness: float,
    angular_damping: float,
    max_force: float,
    max_torque: float,
    body_mass: torch.Tensor | None = None,
    projected_gravity_b: torch.Tensor | None = None,
    gravity_magnitude: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a body-frame pose-PD wrench.

    Args:
        current_position_w: Current link positions in world coordinates.
        current_orientation_xyzw_w: Current link orientations in XYZW order.
        current_com_velocity_w: Current COM linear and angular world velocity.
        target_position_w: Target link positions in world coordinates.
        target_orientation_xyzw_w: Target link orientations in XYZW order.
        controller_scale: One curriculum scale for each environment.
        body_mass: Optional single-body mass for each environment.
        projected_gravity_b: Optional unit gravity direction in the body frame.
        gravity_magnitude: Optional non-negative gravity acceleration magnitude.

    Returns:
        Body-frame force and torque tensors, each with shape ``[num_envs, 3]``.
    """

    _validate_batch("current_position_w", current_position_w, 3)
    num_envs = current_position_w.shape[0]
    expected = {
        "current_orientation_xyzw_w": (current_orientation_xyzw_w, 4),
        "current_com_velocity_w": (current_com_velocity_w, 6),
        "target_position_w": (target_position_w, 3),
        "target_orientation_xyzw_w": (target_orientation_xyzw_w, 4),
    }
    for name, (value, width) in expected.items():
        _validate_batch(name, value, width)
        if value.shape[0] != num_envs:
            raise ValueError(f"{name} must have {num_envs} environment rows.")

    gains = {
        "linear_stiffness": linear_stiffness,
        "linear_damping": linear_damping,
        "angular_stiffness": angular_stiffness,
        "angular_damping": angular_damping,
    }
    for name, value in gains.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if max_force <= 0.0 or max_torque <= 0.0:
        raise ValueError("max_force and max_torque must be positive.")

    current_orientation_xyzw_w = quat_normalize_xyzw(current_orientation_xyzw_w)
    target_orientation_xyzw_w = quat_normalize_xyzw(target_orientation_xyzw_w)

    linear_velocity_b = quat_rotate_inverse_xyzw(
        current_orientation_xyzw_w, current_com_velocity_w[:, :3]
    )
    angular_velocity_b = quat_rotate_inverse_xyzw(
        current_orientation_xyzw_w, current_com_velocity_w[:, 3:]
    )
    position_error_b = quat_rotate_inverse_xyzw(
        current_orientation_xyzw_w, target_position_w - current_position_w
    )
    force_b = linear_stiffness * position_error_b - linear_damping * linear_velocity_b

    orientation_error_b = axis_angle_from_quat_xyzw(
        quat_multiply_xyzw(
            quat_inverse_xyzw(current_orientation_xyzw_w),
            target_orientation_xyzw_w,
        )
    )
    torque_b = (
        angular_stiffness * orientation_error_b - angular_damping * angular_velocity_b
    )

    gravity_inputs = (body_mass, projected_gravity_b, gravity_magnitude)
    if any(value is not None for value in gravity_inputs) and not all(
        value is not None for value in gravity_inputs
    ):
        raise ValueError(
            "body_mass, projected_gravity_b, and gravity_magnitude must be "
            "provided together."
        )
    if body_mass is not None:
        assert projected_gravity_b is not None
        assert gravity_magnitude is not None
        if gravity_magnitude < 0.0:
            raise ValueError("gravity_magnitude must be non-negative.")
        _validate_batch("projected_gravity_b", projected_gravity_b, 3)
        if projected_gravity_b.shape[0] != num_envs:
            raise ValueError(
                f"projected_gravity_b must have {num_envs} environment rows."
            )
        mass = _per_environment_column(
            body_mass,
            num_envs=num_envs,
            like=current_position_w,
            name="body_mass",
        )
        gravity_direction_b = projected_gravity_b / torch.linalg.vector_norm(
            projected_gravity_b, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        force_b = force_b - mass * gravity_magnitude * gravity_direction_b

    scale = _per_environment_column(
        controller_scale,
        num_envs=num_envs,
        like=current_position_w,
        name="controller_scale",
    )
    force_b = torch.clamp(force_b * scale, min=-max_force, max=max_force)
    torque_b = torch.clamp(torque_b * scale, min=-max_torque, max=max_torque)
    return force_b, torque_b


__all__ = [
    "axis_angle_from_quat_xyzw",
    "compute_pose_pd_wrench_xyzw",
    "quat_inverse_xyzw",
    "quat_multiply_xyzw",
    "quat_normalize_xyzw",
    "quat_rotate_inverse_xyzw",
    "quat_rotate_xyzw",
]
