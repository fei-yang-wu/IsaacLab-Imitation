# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-independent pose, contact, reward, and observation terms.

This module adapts the tensor math from NVIDIA's ``video_to_data`` CHORD task
at commit ``afa9dffda748e301ddf79b7002ce4cc5cb552adb``. The upstream files are
``tasks/v2d/mdp/{utils,utils_jit,rewards,observations,terminations}.py`` and
``tasks/v2d/mdp/commands/hand_object_commands.py``. The adapter removes Isaac
Lab and PhysX imports. It can therefore consume contacts supplied by Newton or
another simulator backend.

MDP wrappers use one small environment interface: ``env.command_manager`` must
provide ``get_term(name)``. The returned command term provides the tensors
listed on each wrapper. Pure tensor functions do not need an environment.
Tensor math in this module uses the WXYZ quaternion convention. The
Isaac Lab 3.0 boundary adapters convert to and from XYZW.
"""

from __future__ import annotations

import math
from typing import Any

import torch


DEFAULT_COMMAND_NAME = "dual_hands_object_tracking_command"
DEFAULT_SUPPORT_THRESHOLD = 1.0e-3
FIRST_FRAME_VOC_SCALE_THRESHOLD = 0.1


def _command(env: Any, command_name: str) -> Any:
    """Return one command term from the standard manager interface."""
    return env.command_manager.get_term(command_name)


def _positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")


def sample_reference_start_frames(
    lengths: torch.Tensor,
    *,
    always_reset_to_first_frame: bool = False,
    reset_to_first_frame_prob: float = 0.1,
    virtual_object_control_scale: float | torch.Tensor = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample valid reference start frames for variable-length References.

    The last Reference frame is not a valid start because it is the timeout
    frame. After virtual object control falls below 0.1, a configured fraction
    of starts moves to frame zero to match the released train/eval mix.
    """

    if lengths.ndim != 1:
        raise ValueError("Reference lengths must have shape [N].")
    if torch.any(lengths < 2):
        raise ValueError("Each Reference must contain at least two frames.")
    probability = float(reset_to_first_frame_prob)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("reset_to_first_frame_prob must be within [0, 1].")
    scale = torch.as_tensor(virtual_object_control_scale)
    if scale.numel() != 1 or not bool(torch.isfinite(scale).item()):
        raise ValueError("virtual_object_control_scale must be one finite value.")

    random_values = torch.rand(
        lengths.shape,
        device=lengths.device,
        dtype=torch.float32,
        generator=generator,
    )
    starts = (random_values * (lengths - 1).to(torch.float32)).to(lengths.dtype)
    if always_reset_to_first_frame:
        return torch.zeros_like(starts)
    if probability > 0.0 and float(scale.item()) < FIRST_FRAME_VOC_SCALE_THRESHOLD:
        first_frame_mask = (
            torch.rand(
                lengths.shape,
                device=lengths.device,
                dtype=torch.float32,
                generator=generator,
            )
            < probability
        )
        starts[first_frame_mask] = 0
    return starts


def virtual_object_controller_scale_schedule(
    steps_since_reset: torch.Tensor,
    *,
    warmup_steps: int,
    curriculum_scale: float | torch.Tensor,
    force_unassisted: bool = False,
) -> torch.Tensor:
    """Return a smooth reset-time virtual-object-controller scale.

    During a positive-length reset warmup, the scale linearly decays from one
    to the curriculum value.  This avoids a discontinuous controller drop just
    after a reset.  The Reference frame advances from the first step and is
    not held, because holding it froze the whole episode.  Evaluation can
    explicitly force zero assistance from reset frame zero.
    """

    if steps_since_reset.ndim != 1:
        raise ValueError("steps_since_reset must have shape [N].")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
    target = torch.as_tensor(
        curriculum_scale,
        device=steps_since_reset.device,
        dtype=torch.float32,
    )
    if target.numel() != 1 or not bool(torch.isfinite(target).item()):
        raise ValueError("curriculum_scale must be one finite value.")
    if not 0.0 <= float(target.item()) <= 1.0:
        raise ValueError("curriculum_scale must be within [0, 1].")
    if force_unassisted:
        return torch.zeros(
            steps_since_reset.shape[0],
            1,
            dtype=torch.float32,
            device=steps_since_reset.device,
        )
    if warmup_steps == 0:
        return target.expand(steps_since_reset.shape[0], 1).clone()
    alpha = (steps_since_reset.to(torch.float32) / float(warmup_steps)).clamp_(0.0, 1.0)
    return (1.0 + alpha * (target - 1.0)).unsqueeze(-1)


# -----------------------------------------------------------------------------
# WXYZ quaternion and object-relative command transforms
# -----------------------------------------------------------------------------


def quat_normalize(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return unit WXYZ quaternions."""
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(eps)


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of WXYZ quaternions."""
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_inverse(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return the inverse of WXYZ quaternions."""
    norm_squared = quaternion.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    return quat_conjugate(quaternion) / norm_squared


def quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose WXYZ quaternions as ``first * second``."""
    first_w, first_xyz = first[..., :1], first[..., 1:]
    second_w, second_xyz = second[..., :1], second[..., 1:]
    scalar = first_w * second_w - (first_xyz * second_xyz).sum(dim=-1, keepdim=True)
    vector = (
        first_w * second_xyz
        + second_w * first_xyz
        + torch.linalg.cross(first_xyz, second_xyz)
    )
    return torch.cat((scalar, vector), dim=-1)


def quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors with broadcast-compatible WXYZ quaternions."""
    quaternion = quat_normalize(quaternion)
    quaternion_xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(quaternion_xyz, vector)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.linalg.cross(quaternion_xyz, twice_cross)
    )


def quat_unique(quaternion: torch.Tensor) -> torch.Tensor:
    """Select the WXYZ quaternion representative with a non-negative scalar."""
    sign = torch.where(quaternion[..., :1] < 0.0, -1.0, 1.0)
    return quaternion * sign


def quat_error_magnitude(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return the shortest geodesic angle between WXYZ quaternions."""
    relative = quat_normalize(
        quat_multiply(first, quat_inverse(second, eps=eps)), eps=eps
    )
    vector_norm = torch.linalg.vector_norm(relative[..., 1:], dim=-1)
    return 2.0 * torch.atan2(vector_norm, relative[..., 0].abs().clamp_min(eps))


def compose_pose(
    parent_position: torch.Tensor,
    parent_wxyz: torch.Tensor,
    child_position_in_parent: torch.Tensor,
    child_wxyz_in_parent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Transform a child pose from a parent frame to the parent frame's frame."""
    child_position = parent_position + quat_rotate(
        parent_wxyz, child_position_in_parent
    )
    if child_wxyz_in_parent is None:
        return child_position, None
    child_wxyz = quat_unique(
        quat_normalize(quat_multiply(parent_wxyz, child_wxyz_in_parent))
    )
    return child_position, child_wxyz


def relative_pose(
    parent_position: torch.Tensor,
    parent_wxyz: torch.Tensor,
    child_position: torch.Tensor,
    child_wxyz: torch.Tensor | None = None,
    *,
    unique: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Express a child pose in a parent frame.

    ``unique=False`` keeps the raw relative quaternion, which is what the
    released ``subtract_frame_transforms`` command layout publishes.
    """
    parent_inverse = quat_inverse(parent_wxyz)
    relative_position = quat_rotate(parent_inverse, child_position - parent_position)
    if child_wxyz is None:
        return relative_position, None
    relative_wxyz = quat_normalize(quat_multiply(parent_inverse, child_wxyz))
    if unique:
        relative_wxyz = quat_unique(relative_wxyz)
    return relative_position, relative_wxyz


def retarget_pose_to_live_object(
    reference_object_position: torch.Tensor,
    reference_object_wxyz: torch.Tensor,
    live_object_position: torch.Tensor,
    live_object_wxyz: torch.Tensor,
    reference_target_position: torch.Tensor,
    reference_target_wxyz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move a reference target with the current object pose.

    This is the released object-relative command transform. It first stores the
    target in the reference object frame. It then composes that local target
    with the live object pose. The function works for wrist poses and for
    point-only fingertip commands.
    """
    target_position_o, target_wxyz_o = relative_pose(
        reference_object_position,
        reference_object_wxyz,
        reference_target_position,
        reference_target_wxyz,
    )
    return compose_pose(
        live_object_position,
        live_object_wxyz,
        target_position_o,
        target_wxyz_o,
    )


def pose_delta_command(
    current_position: torch.Tensor,
    current_wxyz: torch.Tensor,
    target_position: torch.Tensor,
    target_wxyz: torch.Tensor,
    *,
    unique: bool = True,
) -> torch.Tensor:
    """Return a target pose expressed in the current pose frame as XYZ+WXYZ."""
    position_delta, orientation_delta = relative_pose(
        current_position, current_wxyz, target_position, target_wxyz, unique=unique
    )
    assert orientation_delta is not None
    return torch.cat((position_delta, orientation_delta), dim=-1)


def root_link_twist_to_com_twist_w(
    link_twist_w: torch.Tensor,
    link_wxyz_w: torch.Tensor,
    com_position_b: torch.Tensor,
) -> torch.Tensor:
    """Convert a root-link-origin twist to Isaac's root-COM write convention."""

    if link_twist_w.shape[-1] != 6:
        raise ValueError("link_twist_w must end with six linear-then-angular values.")
    if link_wxyz_w.shape[-1] != 4 or com_position_b.shape[-1] != 3:
        raise ValueError("link_wxyz_w and com_position_b must end with 4 and 3 values.")
    angular_velocity_w = link_twist_w[..., 3:6]
    com_offset_w = quat_rotate(link_wxyz_w, com_position_b)
    com_linear_velocity_w = link_twist_w[..., :3] + torch.linalg.cross(
        angular_velocity_w,
        com_offset_w,
    )
    return torch.cat((com_linear_velocity_w, angular_velocity_w), dim=-1)


# -----------------------------------------------------------------------------
# Contact wrench support
# -----------------------------------------------------------------------------


def sample_wrench_space_basis_scaled(
    num_samples: int,
    object_radius: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    eps: float = 1.0e-8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample wrench directions on a scaled 6D unit sphere."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    _positive(object_radius, "object_radius")
    basis = torch.randn(
        num_samples,
        6,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    basis[:, 3:] = basis[:, 3:] / object_radius
    return basis / torch.linalg.vector_norm(basis, dim=-1, keepdim=True).clamp_min(eps)


def friction_cone_phase(
    num_edges: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine and sine phases with shapes ``(1, edges, 1)``."""
    if num_edges <= 0:
        raise ValueError(f"num_edges must be positive, got {num_edges}.")
    phase = (
        torch.arange(num_edges, device=device, dtype=dtype)
        * (2.0 * math.pi / num_edges)
    ).view(1, num_edges, 1)
    return torch.cos(phase), torch.sin(phase)


def compute_tangent_basis(
    normals: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Frisvad orthonormal tangent basis for normal vectors."""
    normals = normals / torch.linalg.vector_norm(
        normals, dim=-1, keepdim=True
    ).clamp_min(eps)
    normal_x, normal_y, normal_z = normals.unbind(dim=-1)
    sign = torch.where(normal_z >= 0.0, 1.0, -1.0)
    denominator = sign + normal_z
    denominator = torch.where(denominator.abs() < eps, sign * eps, denominator)
    factor = -1.0 / denominator
    cross_factor = normal_x * normal_y * factor
    tangent_1 = torch.stack(
        (
            1.0 + sign * normal_x * normal_x * factor,
            sign * cross_factor,
            -sign * normal_x,
        ),
        dim=-1,
    )
    tangent_2 = torch.stack(
        (
            cross_factor,
            sign + normal_y * normal_y * factor,
            -normal_y,
        ),
        dim=-1,
    )
    tangent_1 = tangent_1 / torch.linalg.vector_norm(
        tangent_1, dim=-1, keepdim=True
    ).clamp_min(eps)
    tangent_2 = tangent_2 / torch.linalg.vector_norm(
        tangent_2, dim=-1, keepdim=True
    ).clamp_min(eps)
    return tangent_1, tangent_2


def compute_friction_cone_edges(
    normals: torch.Tensor,
    cos_phase: torch.Tensor,
    sin_phase: torch.Tensor,
    friction_coefficient: float = 0.5,
    eps: float = 1.0e-6,
    append_normal: bool = True,
) -> torch.Tensor:
    """Build normalized rays for a polyhedral Coulomb friction cone."""
    batch_size, num_contacts, _ = normals.shape
    unit_normals = normals / torch.linalg.vector_norm(
        normals, dim=-1, keepdim=True
    ).clamp_min(eps)
    normals_flat = unit_normals.reshape(-1, 3)
    tangent_1, tangent_2 = compute_tangent_basis(normals_flat, eps=eps)
    edges = normals_flat.unsqueeze(1) + friction_coefficient * (
        cos_phase * tangent_1.unsqueeze(1) + sin_phase * tangent_2.unsqueeze(1)
    )
    edges = edges / torch.linalg.vector_norm(edges, dim=-1, keepdim=True).clamp_min(eps)
    if append_normal:
        edges = torch.cat((edges, normals_flat.unsqueeze(1)), dim=1)
    num_output_edges = cos_phase.shape[1] + int(append_normal)
    return edges.view(batch_size, num_contacts, num_output_edges, 3)


def compute_wrench_space(
    contact_points_o: torch.Tensor,
    contact_normals_o: torch.Tensor,
    cos_phase: torch.Tensor,
    sin_phase: torch.Tensor,
    object_radius: float,
    friction_coefficient: float = 0.5,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return primitive contact wrenches with shape ``(batch, 6, rays)``."""
    _positive(object_radius, "object_radius")
    batch_size = contact_points_o.shape[0]
    normal_magnitude = torch.linalg.vector_norm(contact_normals_o, dim=-1, keepdim=True)
    contact_active = normal_magnitude.squeeze(-1) > DEFAULT_SUPPORT_THRESHOLD
    unit_normals = contact_normals_o / normal_magnitude.clamp_min(eps)
    forces = compute_friction_cone_edges(
        unit_normals,
        cos_phase,
        sin_phase,
        friction_coefficient=friction_coefficient,
        eps=eps,
    )
    torques = torch.linalg.cross(contact_points_o.unsqueeze(2), forces)
    wrenches = torch.cat((forces, torques / object_radius), dim=-1)
    wrenches = wrenches * contact_active[..., None, None].to(wrenches.dtype)
    return wrenches.reshape(batch_size, -1, 6).transpose(1, 2).contiguous()


def compute_wrench_space_support_function(
    wrench_space: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the non-negative support of a primitive wrench set."""
    if wrench_space.shape[-1] == 0:
        return torch.zeros(
            wrench_space.shape[0],
            basis.shape[0],
            device=wrench_space.device,
            dtype=wrench_space.dtype,
        )
    return torch.matmul(basis.unsqueeze(0), wrench_space).amax(dim=-1).clamp_min(0.0)


def compute_contact_wrench_supports(
    contact_positions_w: torch.Tensor,
    contact_forces_w: torch.Tensor,
    object_com_positions_w: torch.Tensor,
    object_com_wxyz_w: torch.Tensor,
    basis: torch.Tensor,
    object_radii: torch.Tensor | float,
    *,
    contact_active: torch.Tensor | None = None,
    num_friction_cone_edges: int = 8,
    friction_coefficient: float = 0.5,
    contact_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> torch.Tensor:
    """Compute contact support values from backend-neutral contact tensors.

    Args:
        contact_positions_w: World contact positions, shape ``(N, B, C, 3)``.
        contact_forces_w: World contact forces, shape ``(N, B, C, 3)``.
        object_com_positions_w: Object COM positions, shape ``(N, B, 3)``.
        object_com_wxyz_w: Object COM orientations, shape ``(N, B, 4)``.
        basis: Per-body ``(B, K, 6)`` or shared ``(K, 6)`` directions.
        object_radii: Per-body radii or one radius shared by all bodies.
        contact_active: Optional backend contact mask, shape ``(N, B, C)``.

    Returns:
        Support values with shape ``(N, B, K)``.

    Newton integration should provide ``contact_active``. If it is absent,
    force magnitude greater than ``contact_threshold`` defines live contact.
    """
    if contact_positions_w.shape != contact_forces_w.shape:
        raise ValueError("Contact position and force tensors must have equal shapes.")
    if contact_positions_w.ndim != 4 or contact_positions_w.shape[-1] != 3:
        raise ValueError("Contact tensors must have shape (N, B, C, 3).")
    num_envs, num_bodies, num_contacts, _ = contact_positions_w.shape
    expected_com_position_shape = (num_envs, num_bodies, 3)
    expected_com_wxyz_shape = (num_envs, num_bodies, 4)
    if tuple(object_com_positions_w.shape) != expected_com_position_shape:
        raise ValueError(
            "Object COM positions must have shape "
            f"{expected_com_position_shape}, got {tuple(object_com_positions_w.shape)}."
        )
    if tuple(object_com_wxyz_w.shape) != expected_com_wxyz_shape:
        raise ValueError(
            "Object COM orientations must have shape "
            f"{expected_com_wxyz_shape}, got {tuple(object_com_wxyz_w.shape)}."
        )
    if contact_active is None:
        contact_active = (
            torch.linalg.vector_norm(contact_forces_w, dim=-1) > contact_threshold
        )
    if tuple(contact_active.shape) != (num_envs, num_bodies, num_contacts):
        raise ValueError("contact_active must have shape (N, B, C).")

    object_inverse = quat_inverse(object_com_wxyz_w).unsqueeze(2)
    contact_points_o = quat_rotate(
        object_inverse,
        contact_positions_w - object_com_positions_w.unsqueeze(2),
    )
    force_norm = torch.linalg.vector_norm(contact_forces_w, dim=-1, keepdim=True)
    contact_normals_o = quat_rotate(
        object_inverse,
        contact_forces_w / force_norm.clamp_min(1.0e-6),
    )
    active_float = contact_active.unsqueeze(-1).to(contact_points_o.dtype)
    contact_points_o = contact_points_o * active_float
    contact_normals_o = contact_normals_o * active_float

    if basis.ndim == 2:
        basis = basis.unsqueeze(0).expand(num_bodies, -1, -1)
    if basis.ndim != 3 or basis.shape[0] != num_bodies or basis.shape[-1] != 6:
        raise ValueError("basis must have shape (K, 6) or (B, K, 6).")
    radii = torch.as_tensor(
        object_radii,
        device=contact_positions_w.device,
        dtype=contact_positions_w.dtype,
    ).reshape(-1)
    if radii.numel() == 1:
        radii = radii.expand(num_bodies)
    if radii.numel() != num_bodies:
        raise ValueError("object_radii must contain one value or one value per body.")

    cos_phase, sin_phase = friction_cone_phase(
        num_friction_cone_edges,
        device=contact_positions_w.device,
        dtype=contact_positions_w.dtype,
    )
    support_per_body = []
    for body_index in range(num_bodies):
        wrench_space = compute_wrench_space(
            contact_points_o[:, body_index],
            contact_normals_o[:, body_index],
            cos_phase,
            sin_phase,
            object_radius=float(radii[body_index].item()),
            friction_coefficient=friction_coefficient,
        )
        support_per_body.append(
            compute_wrench_space_support_function(wrench_space, basis[body_index])
        )
    return torch.stack(support_per_body, dim=1)


# -----------------------------------------------------------------------------
# Pure reward kernels and manager wrappers
# -----------------------------------------------------------------------------


def contact_wrench_support_reward_tensors(
    right_command_supports: torch.Tensor,
    right_current_supports: torch.Tensor,
    left_command_supports: torch.Tensor,
    left_current_supports: torch.Tensor,
    *,
    tolerance: float = 0.1,
    variance: float = 0.1,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> torch.Tensor:
    """Reward support that is present and within the released tolerance band."""
    _positive(variance, "variance")
    command_supports = torch.stack(
        (right_command_supports, left_command_supports), dim=0
    )
    current_supports = torch.stack(
        (right_current_supports, left_current_supports), dim=0
    )
    command_active = command_supports > support_threshold
    current_active = current_supports > support_threshold
    command_active_per_body = command_active.any(dim=-1)
    command_count = command_active.sum(dim=-1).float().clamp_min(1.0e-6)
    command_body_count = command_active_per_body.sum(dim=-1).float().clamp_min(1.0e-6)
    active_hand_count = (
        (command_body_count > support_threshold).float().sum(dim=0).clamp_min(1.0e-6)
    )

    below = ((1.0 - tolerance) * command_supports - current_supports).clamp_min(0.0)
    above = (current_supports - (1.0 + tolerance) * command_supports).clamp_min(0.0)
    loss = below.square() + above.square()
    per_body = (
        (command_active & current_active).float() * torch.exp(-loss / variance)
    ).sum(dim=-1) / command_count
    per_hand = per_body.sum(dim=-1) / command_body_count
    return per_hand.sum(dim=0) / active_hand_count


def unintended_contact_penalty_tensors(
    right_command_supports: torch.Tensor,
    right_current_supports: torch.Tensor,
    left_command_supports: torch.Tensor,
    left_current_supports: torch.Tensor,
    *,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> torch.Tensor:
    """Return the penalty for support on bodies with no command contact."""
    command_supports = torch.stack(
        (right_command_supports, left_command_supports), dim=0
    )
    current_supports = torch.stack(
        (right_current_supports, left_current_supports), dim=0
    )
    command_active_per_body = (command_supports > support_threshold).any(dim=-1)
    current_active_per_body = (current_supports > support_threshold).any(dim=-1)
    unintended = ~command_active_per_body & current_active_per_body
    num_bodies = command_supports.shape[-2]
    command_body_count = command_active_per_body.sum(dim=-1).float()
    continuous = (~command_active_per_body).float() * current_supports.clamp_min(
        0.0
    ).square().mean(dim=-1)
    continuous = continuous.sum(dim=-1) / (
        float(num_bodies) - command_body_count
    ).clamp_min(1.0e-3)
    return unintended.float().mean(dim=-1).sum(dim=0) + continuous.sum(dim=0)


def missed_contact_penalty_tensors(
    right_command_supports: torch.Tensor,
    right_current_supports: torch.Tensor,
    left_command_supports: torch.Tensor,
    left_current_supports: torch.Tensor,
    *,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> torch.Tensor:
    """Return the commanded support fraction that is absent in the live state."""
    command_supports = torch.stack(
        (right_command_supports, left_command_supports), dim=0
    )
    current_supports = torch.stack(
        (right_current_supports, left_current_supports), dim=0
    )
    command_active = command_supports > support_threshold
    current_active = current_supports > support_threshold
    expected_count = command_active.sum(dim=-1).float()
    missed_count = (command_active & ~current_active).sum(dim=-1).float()
    missing_fraction = missed_count / expected_count.clamp_min(1.0e-6)
    body_active = expected_count > 0
    active_body_count = body_active.sum(dim=-1).float().clamp_min(1.0e-6)
    per_hand = (missing_fraction * body_active.float()).sum(dim=-1) / active_body_count
    active_hand_count = body_active.any(dim=-1).float().sum(dim=0).clamp_min(1.0e-6)
    return per_hand.sum(dim=0) / active_hand_count


def contact_wrench_support_reward(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    tolerance: float = 0.1,
    var: float = 0.1,
) -> torch.Tensor:
    """MDP wrapper for contact-wrench support inclusion."""
    command = _command(env, command_name)
    return contact_wrench_support_reward_tensors(
        command.right_hand_contact_wrench_supports_command,
        command.right_hand_contact_wrench_supports,
        command.left_hand_contact_wrench_supports_command,
        command.left_hand_contact_wrench_supports,
        tolerance=tolerance,
        variance=var,
    )


def unintended_contact_penalty(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """MDP wrapper for unintended contact support."""
    command = _command(env, command_name)
    return unintended_contact_penalty_tensors(
        command.right_hand_contact_wrench_supports_command,
        command.right_hand_contact_wrench_supports,
        command.left_hand_contact_wrench_supports_command,
        command.left_hand_contact_wrench_supports,
    )


def missed_contact_penalty(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """MDP wrapper for missed commanded contact support."""
    command = _command(env, command_name)
    return missed_contact_penalty_tensors(
        command.right_hand_contact_wrench_supports_command,
        command.right_hand_contact_wrench_supports,
        command.left_hand_contact_wrench_supports_command,
        command.left_hand_contact_wrench_supports,
    )


def object_position_tracking_exp_tensors(
    target_position: torch.Tensor,
    current_position: torch.Tensor,
    variance: float = 0.3,
) -> torch.Tensor:
    """Return mean exponential object-position tracking reward per environment."""
    _positive(variance, "variance")
    reward = torch.exp(
        -(target_position - current_position).square().sum(dim=-1) / variance
    )
    return reward.mean(dim=-1) if reward.ndim > 1 else reward


def object_orientation_tracking_exp_tensors(
    target_wxyz: torch.Tensor,
    current_wxyz: torch.Tensor,
    variance: float = 0.3,
) -> torch.Tensor:
    """Return mean exponential object-orientation tracking reward."""
    _positive(variance, "variance")
    reward = torch.exp(-quat_error_magnitude(target_wxyz, current_wxyz) / variance)
    return reward.mean(dim=-1) if reward.ndim > 1 else reward


def object_twist_tracking_exp_tensors(
    target_twist_w: torch.Tensor,
    current_twist_w: torch.Tensor,
    variance: float = 1.0,
) -> torch.Tensor:
    """Return mean exponential world-frame 6D object-twist tracking reward."""

    _positive(variance, "variance")
    if (
        target_twist_w.shape != current_twist_w.shape
        or target_twist_w.ndim != 3
        or target_twist_w.shape[-1] != 6
    ):
        raise ValueError(
            "Object twist target and current tensors must have equal [N, O, 6] "
            f"shapes; got {tuple(target_twist_w.shape)} and "
            f"{tuple(current_twist_w.shape)}."
        )
    per_object_error = (target_twist_w - current_twist_w).square().mean(dim=-1)
    return torch.exp(-per_object_error / variance).mean(dim=-1)


def object_position_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 0.3,
) -> torch.Tensor:
    """Track current object positions against the Reference."""
    command = _command(env, command_name)
    return object_position_tracking_exp_tensors(
        command.object_body_position_command_e,
        command.object_position_e,
        variance=var,
    )


def object_orientation_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 0.3,
) -> torch.Tensor:
    """Track current object orientations against the Reference."""
    command = _command(env, command_name)
    return object_orientation_tracking_exp_tensors(
        command.object_body_wxyz_command_e,
        command.object_orientation_e,
        variance=var,
    )


def object_twist_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 1.0,
) -> torch.Tensor:
    """Track world-frame object linear and angular velocity against Reference."""

    command = _command(env, command_name)
    return object_twist_tracking_exp_tensors(
        command.object_body_twist_command_w,
        command.object_twist_w,
        variance=var,
    )


def robot_joint_tracking_exp_tensors(
    target: torch.Tensor,
    current: torch.Tensor,
    variance: float = 0.25,
) -> torch.Tensor:
    """Return a whole-robot exponential joint-state tracking reward.

    The mean-squared error keeps the reward scale independent of the number of
    actuators.  This is the ReconBody component missing from the hand/contact
    terms inherited from the released recipe.
    """

    _positive(variance, "variance")
    if target.shape != current.shape or target.ndim != 2:
        raise ValueError(
            "Whole-robot target and current tensors must have equal [N, J] "
            f"shapes; got {tuple(target.shape)} and {tuple(current.shape)}."
        )
    return torch.exp(-(target - current).square().mean(dim=-1) / variance)


def robot_joint_position_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 0.25,
) -> torch.Tensor:
    """Track all runtime-order joint positions against the retargeted robot."""

    command = _command(env, command_name)
    return robot_joint_tracking_exp_tensors(
        command.reference_joint_pos,
        command.robot.data.joint_pos.torch,
        variance=var,
    )


def robot_joint_velocity_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 1.0,
) -> torch.Tensor:
    """Track all runtime-order joint velocities against the retargeted robot."""

    command = _command(env, command_name)
    return robot_joint_tracking_exp_tensors(
        command.reference_joint_vel,
        command.robot.data.joint_vel.torch,
        variance=var,
    )


def object_keypoints_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 0.1,
) -> torch.Tensor:
    """Track the six principal object keypoints used by the released recipe."""

    _positive(var, "var")
    command = _command(env, command_name)
    vectors = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=command.object_position_e.dtype,
        device=command.object_position_e.device,
    ).view(1, 1, 6, 3)
    current = command.object_position_e.unsqueeze(2) + quat_rotate(
        command.object_orientation_e.unsqueeze(2), vectors
    )
    target = command.object_body_position_command_e.unsqueeze(2) + quat_rotate(
        command.object_body_wxyz_command_e.unsqueeze(2), vectors
    )
    error = (current - target).square().sum(dim=-1)
    return torch.exp(-error / var).mean(dim=(-2, -1))


def hand_keypoints_tracking_exp_tensors(
    left_wrist_command: torch.Tensor,
    right_wrist_command: torch.Tensor,
    left_fingertip_command: torch.Tensor,
    right_fingertip_command: torch.Tensor,
    left_wrist_current: torch.Tensor,
    right_wrist_current: torch.Tensor,
    left_fingertip_current: torch.Tensor,
    right_fingertip_current: torch.Tensor,
    *,
    variance: float = 0.1,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Return the symmetric wrist-and-fingertip tracking reward."""
    _positive(variance, "variance")
    left_command = torch.cat(
        (left_wrist_command.unsqueeze(1), left_fingertip_command), dim=1
    )
    right_command = torch.cat(
        (right_wrist_command.unsqueeze(1), right_fingertip_command), dim=1
    )
    left_current = torch.cat(
        (left_wrist_current.unsqueeze(1), left_fingertip_current), dim=1
    )
    right_current = torch.cat(
        (right_wrist_current.unsqueeze(1), right_fingertip_current), dim=1
    )
    left_error = (left_command - left_current).square().sum(dim=-1)
    right_error = (right_command - right_current).square().sum(dim=-1)
    left_reward = torch.exp(-(left_error - threshold).clamp_min(0.0) / variance).mean(
        dim=-1
    )
    right_reward = torch.exp(-(right_error - threshold).clamp_min(0.0) / variance).mean(
        dim=-1
    )
    return (left_reward + right_reward) / 2.0


def hand_keypoints_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 0.1,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Track current wrists and fingertips against object-relative commands."""
    command = _command(env, command_name)
    return hand_keypoints_tracking_exp_tensors(
        command.left_hand_wrist_pose_command_e[..., :3],
        command.right_hand_wrist_pose_command_e[..., :3],
        command.left_hand_fingertip_position_command_e[..., :3],
        command.right_hand_fingertip_position_command_e[..., :3],
        command.left_hand_wrist_position_e,
        command.right_hand_wrist_position_e,
        command.left_hand_fingertip_position_e[..., :3],
        command.right_hand_fingertip_position_e[..., :3],
        variance=var,
        threshold=threshold,
    )


def hand_joint_pos_tracking_exp(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    var: float = 1.0,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Track the forty Wuji finger joints against the Reference."""

    _positive(var, "var")
    command = _command(env, command_name)
    left_error = (
        (
            command.left_hand_finger_joint_pos_command
            - command.left_hand_finger_joint_pos
        )
        .square()
        .sum(dim=-1)
    )
    right_error = (
        (
            command.right_hand_finger_joint_pos_command
            - command.right_hand_finger_joint_pos
        )
        .square()
        .sum(dim=-1)
    )
    return (
        torch.exp(-(left_error - threshold).clamp_min(0.0) / var)
        + torch.exp(-(right_error - threshold).clamp_min(0.0) / var)
    ) / 2.0


def action_norm(env: Any, action_name: str = "joint_residual") -> torch.Tensor:
    """Return the squared norm of the policy-controlled residual."""

    action = env.action_manager.get_term(action_name).raw_actions
    return action.square().sum(dim=-1)


def termination_penalty(env: Any) -> torch.Tensor:
    """Return the non-timeout termination mask as a float tensor."""

    return env.termination_manager.terminated.float()


# -----------------------------------------------------------------------------
# Physics safety
# -----------------------------------------------------------------------------


def _as_torch_tensor(value: Any, *, name: str) -> torch.Tensor:
    """Resolve an Isaac Lab ``ProxyArray`` or an ordinary torch tensor."""

    tensor = getattr(value, "torch", value)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must resolve to a torch.Tensor.")
    return tensor


def maximum_support_contact_force_tensors(
    force_matrix_w: torch.Tensor,
) -> torch.Tensor:
    """Return the maximum per-robot-body load from a filtered support sensor.

    ``force_matrix_w`` is the public Newton contact-sensor force matrix with
    shape ``[N, S, F, 3]``: environments, sensing robot bodies, filtered
    support bodies/shapes, and world XYZ. Magnitudes are summed over support
    counterparts before taking the maximum robot-body load. This is
    deliberately conservative when one support is represented by several
    collision shapes.

    A non-finite force makes the corresponding environment's result positive
    infinity. Safety terminations therefore fail closed instead of allowing a
    NaN comparison to evaluate false.
    """

    if force_matrix_w.ndim != 4 or force_matrix_w.shape[-1] != 3:
        raise ValueError(
            "Support force_matrix_w must have shape [N, S, F, 3]; got "
            f"{tuple(force_matrix_w.shape)}."
        )
    if force_matrix_w.shape[1] == 0 or force_matrix_w.shape[2] == 0:
        raise ValueError(
            "Support force_matrix_w must contain at least one sensing body "
            "and one filtered support counterpart."
        )
    if not force_matrix_w.is_floating_point():
        raise TypeError("Support force_matrix_w must use a floating-point dtype.")

    finite = torch.isfinite(force_matrix_w)
    finite_per_env = finite.reshape(force_matrix_w.shape[0], -1).all(dim=-1)
    finite_force = torch.where(finite, force_matrix_w, torch.zeros_like(force_matrix_w))
    pair_magnitudes = torch.linalg.vector_norm(finite_force, dim=-1)
    body_load = pair_magnitudes.sum(dim=-1)
    maximum = body_load.amax(dim=-1)
    return torch.where(
        finite_per_env,
        maximum,
        torch.full_like(maximum, float("inf")),
    )


def support_contact_force_penalty_tensors(
    force_matrix_w: torch.Tensor,
    *,
    penalty_start_force: float = 5.0,
    saturation_force: float = 50.0,
) -> torch.Tensor:
    """Return a bounded ``[0, 1]`` robot-vs-support contact-force penalty.

    Loads at or below ``penalty_start_force`` have zero cost. The cost rises
    linearly to one at ``saturation_force``. Non-finite sensor output maps to
    one, matching the fail-closed termination predicate.
    """

    start = float(penalty_start_force)
    saturation = float(saturation_force)
    if not math.isfinite(start) or start < 0.0:
        raise ValueError("penalty_start_force must be finite and non-negative.")
    if not math.isfinite(saturation) or saturation <= start:
        raise ValueError(
            "saturation_force must be finite and exceed penalty_start_force."
        )
    maximum = maximum_support_contact_force_tensors(force_matrix_w)
    penalty = ((maximum - start) / (saturation - start)).clamp(0.0, 1.0)
    return torch.where(torch.isfinite(maximum), penalty, torch.ones_like(penalty))


def excessive_support_contact_force_tensors(
    force_matrix_w: torch.Tensor,
    *,
    threshold: float = 75.0,
) -> torch.Tensor:
    """Return environments with an excessive robot-vs-support normal load."""

    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive.")
    return maximum_support_contact_force_tensors(force_matrix_w) > threshold


def _support_force_matrix(env: Any, sensor_name: str) -> torch.Tensor:
    """Read the filtered force matrix from one Newton contact sensor."""

    sensor = env.scene[sensor_name]
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        raise RuntimeError(
            f"Contact sensor {sensor_name!r} has no force_matrix_w. Configure "
            "at least one support filter counterpart."
        )
    return _as_torch_tensor(matrix, name=f"{sensor_name}.data.force_matrix_w")


def support_contact_force_penalty(
    env: Any,
    sensor_name: str = "robot_support_contacts",
    penalty_start_force: float = 5.0,
    saturation_force: float = 50.0,
) -> torch.Tensor:
    """MDP wrapper for the bounded robot-vs-support contact-force penalty."""

    return support_contact_force_penalty_tensors(
        _support_force_matrix(env, sensor_name),
        penalty_start_force=penalty_start_force,
        saturation_force=saturation_force,
    )


def excessive_support_contact_force(
    env: Any,
    sensor_name: str = "robot_support_contacts",
    threshold: float = 75.0,
) -> torch.Tensor:
    """MDP termination wrapper for excessive robot-vs-support force."""

    return excessive_support_contact_force_tensors(
        _support_force_matrix(env, sensor_name),
        threshold=threshold,
    )


def nonfinite_physics_state_tensors(*state_tensors: torch.Tensor) -> torch.Tensor:
    """Return a fail-closed mask for non-finite batched physics state tensors."""

    if not state_tensors:
        raise ValueError("At least one physics state tensor is required.")
    num_envs = state_tensors[0].shape[0] if state_tensors[0].ndim > 0 else -1
    if num_envs < 0:
        raise ValueError("Physics state tensors must include an environment dimension.")
    invalid = torch.zeros(
        num_envs,
        dtype=torch.bool,
        device=state_tensors[0].device,
    )
    for index, tensor in enumerate(state_tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Physics state {index} must be a torch.Tensor.")
        if tensor.ndim == 0 or tensor.shape[0] != num_envs:
            raise ValueError(
                "Every physics state tensor must share the leading environment "
                f"dimension {num_envs}; state {index} has shape {tuple(tensor.shape)}."
            )
        if tensor.device != invalid.device:
            raise ValueError("Every physics state tensor must be on the same device.")
        invalid |= ~torch.isfinite(tensor).reshape(num_envs, -1).all(dim=-1)
    return invalid


def nonfinite_physics_state(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Terminate when live robot or tracked-object state contains NaN/Inf.

    The wrapper reads Newton's public articulation/rigid-object state buffers.
    It intentionally checks both poses and velocities in addition to joints so
    a solver failure is caught even before it propagates to a policy feature.
    """

    command = _command(env, command_name)
    robot_data = command.robot.data
    states = [
        _as_torch_tensor(robot_data.joint_pos, name="robot joint_pos"),
        _as_torch_tensor(robot_data.joint_vel, name="robot joint_vel"),
        _as_torch_tensor(robot_data.body_link_pose_w, name="robot body_link_pose_w"),
        _as_torch_tensor(robot_data.body_link_vel_w, name="robot body_link_vel_w"),
    ]
    for object_index, object_asset in enumerate(command.objects):
        states.extend(
            (
                _as_torch_tensor(
                    object_asset.data.root_link_pose_w,
                    name=f"object {object_index} root_link_pose_w",
                ),
                _as_torch_tensor(
                    object_asset.data.root_link_vel_w,
                    name=f"object {object_index} root_link_vel_w",
                ),
            )
        )
    return nonfinite_physics_state_tensors(*states)


# -----------------------------------------------------------------------------
# Object, wrist, and contact observations
# -----------------------------------------------------------------------------


def wrist_position_e(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return right and left wrist positions in the environment frame."""
    command = _command(env, command_name)
    return torch.cat(
        (command.right_hand_wrist_position_e, command.left_hand_wrist_position_e),
        dim=-1,
    )


def current_robot_state(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return full live joint position and velocity state in runtime order."""

    return _command(env, command_name).current_robot_state


def current_object_state(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return flattened live object pose and world-frame twist states."""

    return _command(env, command_name).current_object_state


def next_desired_robot_state(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return the retargeted robot state for the next policy transition."""

    return _command(env, command_name).next_desired_robot_state


def next_desired_object_state(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return flattened target object XYZ+WXYZ poses and world-frame twists."""

    return _command(env, command_name).next_desired_object_state


def processed_action(
    env: Any,
    action_name: str = "joint_residual",
    side: str = "right",
) -> torch.Tensor:
    """Return one processed 7-arm-plus-20-finger target from the adapter."""

    return env.action_manager.get_term(action_name).processed_action(side)


def finger_joint_pos(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return scaled right and left finger positions."""

    command = _command(env, command_name)
    return torch.cat(
        (
            command.right_hand_finger_joint_pos_scaled,
            command.left_hand_finger_joint_pos_scaled,
        ),
        dim=-1,
    )


def finger_joint_vel(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return right and left finger velocities."""

    command = _command(env, command_name)
    return torch.cat(
        (
            command.right_hand_finger_joint_vel,
            command.left_hand_finger_joint_vel,
        ),
        dim=-1,
    )


def wrist_orientation_e(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return right and left WXYZ wrist orientations."""
    command = _command(env, command_name)
    return torch.cat(
        (command.right_hand_wrist_wxyz_e, command.left_hand_wrist_wxyz_e), dim=-1
    )


def wrist_velocity_b(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return right and left spatial wrist velocities in their body frames."""
    command = _command(env, command_name)
    return torch.cat(
        (command.right_hand_wrist_velocity_b, command.left_hand_wrist_velocity_b),
        dim=-1,
    )


def object_position_e(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return flattened current object positions."""
    command = _command(env, command_name)
    return command.object_position_e.reshape(command.object_position_e.shape[0], -1)


def object_orientation_e(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return flattened current WXYZ object orientations."""
    command = _command(env, command_name)
    return command.object_orientation_e.reshape(
        command.object_orientation_e.shape[0], -1
    )


def object_wrist_pose_command(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return current-to-target pose deltas for both wrists and all objects."""
    command = _command(env, command_name)
    right_delta = pose_delta_command(
        command.right_hand_wrist_position_e,
        command.right_hand_wrist_wxyz_e,
        command.right_hand_wrist_pose_command_e[..., :3],
        command.right_hand_wrist_pose_command_e[..., 3:7],
    )
    left_delta = pose_delta_command(
        command.left_hand_wrist_position_e,
        command.left_hand_wrist_wxyz_e,
        command.left_hand_wrist_pose_command_e[..., :3],
        command.left_hand_wrist_pose_command_e[..., 3:7],
    )
    object_delta = pose_delta_command(
        command.object_position_e,
        command.object_orientation_e,
        command.object_body_position_command_e,
        command.object_body_wxyz_command_e,
    )
    return torch.cat(
        (right_delta, left_delta, object_delta.reshape(object_delta.shape[0], -1)),
        dim=-1,
    )


def contact_position_direction_in_wrist_frame(
    contact_positions_w: torch.Tensor,
    contact_forces_w: torch.Tensor,
    wrist_position_w: torch.Tensor,
    wrist_wxyz_w: torch.Tensor,
    contact_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Express contact positions and normalized force directions in a wrist frame."""
    if contact_positions_w.shape != contact_forces_w.shape:
        raise ValueError("Contact position and force tensors must have equal shapes.")
    if contact_positions_w.ndim < 3 or contact_positions_w.shape[-1] != 3:
        raise ValueError("Contact tensors must end in a three-value XYZ dimension.")
    leading_singletons = contact_positions_w.ndim - 2
    wrist_position = wrist_position_w.view(
        wrist_position_w.shape[0], *([1] * leading_singletons), 3
    )
    wrist_wxyz = wrist_wxyz_w.view(
        wrist_wxyz_w.shape[0], *([1] * leading_singletons), 4
    )
    wrist_inverse = quat_inverse(wrist_wxyz)
    position_b = quat_rotate(wrist_inverse, contact_positions_w - wrist_position)
    force_direction_w = contact_forces_w / torch.linalg.vector_norm(
        contact_forces_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    direction_b = quat_rotate(wrist_inverse, force_direction_w)
    if contact_active is None:
        contact_active = torch.linalg.vector_norm(contact_forces_w, dim=-1) > 1.0e-3
    mask = contact_active.unsqueeze(-1).to(position_b.dtype)
    return torch.cat((position_b * mask, direction_b * mask), dim=-1)


def contact_position_direction_in_wrist(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Return flattened left and right contact position/direction observations.

    The command term must normalize Newton sensor history to current contact
    tensors with shape ``(N, B, C, 3)`` before this accessor is called.
    """
    command = _command(env, command_name)
    left = contact_position_direction_in_wrist_frame(
        command.left_hand_object_contact_positions_w,
        command.left_hand_object_contact_forces_w,
        command.left_hand_wrist_position_w,
        command.left_hand_wrist_wxyz_e,
        getattr(command, "left_hand_object_contact_active", None),
    )
    right = contact_position_direction_in_wrist_frame(
        command.right_hand_object_contact_positions_w,
        command.right_hand_object_contact_forces_w,
        command.right_hand_wrist_position_w,
        command.right_hand_wrist_wxyz_e,
        getattr(command, "right_hand_object_contact_active", None),
    )
    return torch.cat(
        (
            left[..., :3].reshape(left.shape[0], -1),
            left[..., 3:].reshape(left.shape[0], -1),
            right[..., :3].reshape(right.shape[0], -1),
            right[..., 3:].reshape(right.shape[0], -1),
        ),
        dim=-1,
    )


# -----------------------------------------------------------------------------
# Tracking terminations
# -----------------------------------------------------------------------------


def robot_joint_away_from_trajectory_tensors(
    target_position: torch.Tensor,
    current_position: torch.Tensor,
    soft_joint_position_limits: torch.Tensor,
    *,
    normalized_position_threshold: float = 0.35,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return environments whose normalized whole-robot error is excessive.

    Errors are normalized by each joint's soft range, which makes a single
    threshold meaningful across Lift, Vega arm, head, and Wuji finger joints.
    An optional per-environment mask suppresses the check during a deliberate
    reset warmup.
    """

    _positive(normalized_position_threshold, "normalized_position_threshold")
    if target_position.shape != current_position.shape or target_position.ndim != 2:
        raise ValueError(
            "Whole-robot target and current tensors must have equal [N, J] "
            f"shapes; got {tuple(target_position.shape)} and "
            f"{tuple(current_position.shape)}."
        )
    expected_limits_shape = (*target_position.shape, 2)
    if soft_joint_position_limits.shape != expected_limits_shape:
        raise ValueError(
            "soft_joint_position_limits must have shape [N, J, 2]; got "
            f"{tuple(soft_joint_position_limits.shape)}, expected "
            f"{expected_limits_shape}."
        )
    joint_range = (
        soft_joint_position_limits[..., 1] - soft_joint_position_limits[..., 0]
    )
    if not bool(torch.all(torch.isfinite(joint_range) & (joint_range > 0.0)).item()):
        raise ValueError("Every soft joint-position range must be finite and positive.")
    normalized_error = (target_position - current_position).abs() / joint_range
    away = (normalized_error > normalized_position_threshold).any(dim=-1)
    if active is not None:
        if active.shape != away.shape:
            raise ValueError(
                f"active must have shape {tuple(away.shape)}, got {tuple(active.shape)}."
            )
        away &= active.to(dtype=torch.bool, device=away.device)
    return away


def robot_joint_away_from_trajectory(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    normalized_position_threshold: float = 0.35,
) -> torch.Tensor:
    """Terminate on large whole-robot deviation after reset warmup."""

    command = _command(env, command_name)
    warmup_steps = int(getattr(getattr(command, "cfg", None), "warmup_steps", 0))
    active = getattr(command, "steps_since_last_reset", None)
    if active is not None:
        active = active >= warmup_steps
    return robot_joint_away_from_trajectory_tensors(
        command.reference_joint_pos,
        command.robot.data.joint_pos.torch,
        command.robot.data.soft_joint_pos_limits.torch,
        normalized_position_threshold=normalized_position_threshold,
        active=active,
    )


def hand_wrist_away_from_trajectory(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Terminate when either wrist position is too far from the Reference."""
    command = _command(env, command_name)
    right_error = torch.linalg.vector_norm(
        command.right_hand_wrist_pose_command_e[..., :3]
        - command.right_hand_wrist_position_e,
        dim=-1,
    )
    left_error = torch.linalg.vector_norm(
        command.left_hand_wrist_pose_command_e[..., :3]
        - command.left_hand_wrist_position_e,
        dim=-1,
    )
    return (right_error > threshold) | (left_error > threshold)


def object_away_from_trajectory_z(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Terminate when any object height is too far from the Reference."""
    command = _command(env, command_name)
    z_error = (
        command.object_body_position_command_e[..., 2]
        - command.object_position_e[..., 2]
    ).abs()
    return (z_error > threshold).any(dim=-1)


def object_away_from_trajectory(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
    position_threshold: float = 0.2,
    orientation_threshold: float = 0.7,
) -> torch.Tensor:
    """Terminate when any object pose is too far from the Reference."""
    command = _command(env, command_name)
    position_error = torch.linalg.vector_norm(
        command.object_body_position_command_e - command.object_position_e,
        dim=-1,
    )
    orientation_error = quat_error_magnitude(
        command.object_orientation_e,
        command.object_body_wxyz_command_e,
    )
    return (position_error > position_threshold).any(dim=-1) | (
        orientation_error > orientation_threshold
    ).any(dim=-1)


def timestep_timeout(
    env: Any,
    command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
    """Terminate when each environment reaches its Reference horizon."""
    command = _command(env, command_name)
    horizon = torch.as_tensor(
        command.retargeted_horizon,
        device=command.timestep_counter.device,
        dtype=command.timestep_counter.dtype,
    )
    return command.timestep_counter >= horizon - 1


__all__ = [
    "FIRST_FRAME_VOC_SCALE_THRESHOLD",
    "action_norm",
    "compose_pose",
    "compute_contact_wrench_supports",
    "compute_friction_cone_edges",
    "compute_tangent_basis",
    "compute_wrench_space",
    "compute_wrench_space_support_function",
    "contact_position_direction_in_wrist",
    "contact_position_direction_in_wrist_frame",
    "contact_wrench_support_reward",
    "contact_wrench_support_reward_tensors",
    "current_robot_state",
    "current_object_state",
    "excessive_support_contact_force",
    "excessive_support_contact_force_tensors",
    "friction_cone_phase",
    "finger_joint_pos",
    "finger_joint_vel",
    "hand_joint_pos_tracking_exp",
    "hand_keypoints_tracking_exp",
    "hand_keypoints_tracking_exp_tensors",
    "hand_wrist_away_from_trajectory",
    "maximum_support_contact_force_tensors",
    "missed_contact_penalty",
    "missed_contact_penalty_tensors",
    "next_desired_object_state",
    "next_desired_robot_state",
    "nonfinite_physics_state",
    "nonfinite_physics_state_tensors",
    "object_away_from_trajectory",
    "object_away_from_trajectory_z",
    "object_keypoints_tracking_exp",
    "object_orientation_e",
    "object_orientation_tracking_exp",
    "object_orientation_tracking_exp_tensors",
    "object_position_e",
    "object_position_tracking_exp",
    "object_position_tracking_exp_tensors",
    "object_twist_tracking_exp",
    "object_twist_tracking_exp_tensors",
    "object_wrist_pose_command",
    "pose_delta_command",
    "processed_action",
    "quat_conjugate",
    "quat_error_magnitude",
    "quat_inverse",
    "quat_multiply",
    "quat_normalize",
    "quat_rotate",
    "quat_unique",
    "relative_pose",
    "retarget_pose_to_live_object",
    "robot_joint_away_from_trajectory",
    "robot_joint_away_from_trajectory_tensors",
    "robot_joint_position_tracking_exp",
    "robot_joint_tracking_exp_tensors",
    "robot_joint_velocity_tracking_exp",
    "root_link_twist_to_com_twist_w",
    "sample_wrench_space_basis_scaled",
    "sample_reference_start_frames",
    "support_contact_force_penalty",
    "support_contact_force_penalty_tensors",
    "termination_penalty",
    "timestep_timeout",
    "unintended_contact_penalty",
    "unintended_contact_penalty_tensors",
    "virtual_object_controller_scale_schedule",
    "wrist_orientation_e",
    "wrist_position_e",
    "wrist_velocity_b",
]
