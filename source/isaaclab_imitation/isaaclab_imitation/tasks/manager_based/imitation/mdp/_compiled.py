from __future__ import annotations

import os

import torch

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    wrap_to_pi,
)


def heading_quat(quat: torch.Tensor) -> torch.Tensor:
    """Twist component about world Z, matching SONIC's ``get_heading_q``.

    Zero the x/y components and renormalize. This is exactly the twist of a
    swing-twist decomposition about Z, and it is what SONIC applies when it
    reroots a reference onto the robot anchor.

    ``isaaclab.utils.math.yaw_quat`` -- what this replaced -- returns the ZYX
    Euler yaw instead. The two agree while the anchor is near upright and
    diverge as it tilts, but the reason for the change is the singularity, not
    the disagreement: **ZYX yaw is degenerate at pitch = 90 degrees**, where a
    two-degree attitude change swings the extracted heading by 180 degrees.
    A pelvis can reach that (falling forward, a deep bow), and when it does the
    rerooted reference spins half a turn and every consumer -- ``motion_body_pos``,
    ``motion_body_ori``, ``foot_pos_xyz`` -- sees a fictitious catastrophic
    error. This form is continuous there, and stays continuous through full
    inversion; its own singularity needs an exact 180-degree rotation about a
    horizontal axis, which is far harder to reach.

    LAYOUT: Isaac Lab quaternions are ``(x, y, z, w)``, so the components to
    zero are indices 0 and 1. SONIC is scalar-first ``(w, x, y, z)`` and zeroes
    indices 1 and 2. Transcribing SONIC's indices literally would zero y and z,
    keep x and w, and yield a *roll* quaternion -- which runs, produces
    plausible numbers, and is silently wrong.
    """
    quat_heading = torch.zeros_like(quat)
    quat_heading[..., 2] = quat[..., 2]  # z
    quat_heading[..., 3] = quat[..., 3]  # w
    norm = torch.linalg.vector_norm(quat_heading, dim=-1, keepdim=True)
    return quat_heading / norm.clamp(min=1e-9)


def _maybe_compile(fn):
    """Keep Torch Inductor opt-in for small MDP kernels.

    These helpers are called with several batch/window shapes during startup and
    expert sampling. Compiling every shape can spawn many Inductor workers and
    exhaust memory before training reaches the first update.
    """
    enabled = os.environ.get("ISAACLAB_IMITATION_COMPILE_MDP", "").lower()
    if enabled in {"1", "true", "yes", "on"}:
        return torch.compile(fn)
    return fn


@_maybe_compile
def quat_to_rot6d_flat(quat: torch.Tensor) -> torch.Tensor:
    quat_mat = matrix_from_quat(quat)
    return quat_mat[..., :2].reshape(quat_mat.shape[0], -1)


@_maybe_compile
def heading_anchor_frame(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Yaw-only anchor frame: heading twist plus an xy-only origin.

    Cancels exactly the transformations the tracking reward's re-rooting is
    invariant under (global yaw and xy translation) and nothing more: poses
    expressed in this frame keep absolute height and roll/pitch relative to
    gravity, which a full anchor-pose cancellation destroys.
    """
    anchor_pos_xy = anchor_pos_w.clone()
    anchor_pos_xy[..., 2] = 0.0
    return anchor_pos_xy, heading_quat(anchor_quat_w)


@_maybe_compile
def body_pose_in_anchor_frame(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_bodies = body_pos_w.shape[1]
    return subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )


@_maybe_compile
def transform_root_pose_to_world(
    align_quat: torch.Tensor,
    align_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return quat_apply(align_quat, ref_pos) + align_pos, quat_mul(align_quat, ref_quat)


@_maybe_compile
def transform_root_velocity_to_world(
    align_quat: torch.Tensor,
    ref_lin_vel: torch.Tensor,
    ref_ang_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return quat_apply(align_quat, ref_lin_vel), quat_apply(align_quat, ref_ang_vel)


@_maybe_compile
def transform_body_pose_to_world(
    align_quat: torch.Tensor,
    align_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs, num_bodies = ref_pos.shape[0], ref_pos.shape[1]
    align_quat_expand = (
        align_quat.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    )
    ref_pos_w = quat_apply(align_quat_expand, ref_pos.reshape(-1, 3)).reshape(
        num_envs, num_bodies, 3
    )
    ref_pos_w = ref_pos_w + align_pos.unsqueeze(1)
    ref_quat_w = quat_mul(align_quat_expand, ref_quat.reshape(-1, 4)).reshape(
        num_envs, num_bodies, 4
    )
    return ref_pos_w, ref_quat_w


@_maybe_compile
def transform_body_velocity_to_world(
    align_quat: torch.Tensor,
    ref_cvel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs, num_bodies = ref_cvel.shape[0], ref_cvel.shape[1]
    align_quat_expand = (
        align_quat.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    )
    ref_ang_vel_w = quat_apply(
        align_quat_expand, ref_cvel[..., :3].reshape(-1, 3)
    ).reshape(num_envs, num_bodies, 3)
    ref_lin_vel_w = quat_apply(
        align_quat_expand, ref_cvel[..., 3:].reshape(-1, 3)
    ).reshape(num_envs, num_bodies, 3)
    return ref_ang_vel_w, ref_lin_vel_w


@_maybe_compile
def gaussian_from_squared_error(
    squared_error: torch.Tensor, sigma: float
) -> torch.Tensor:
    return torch.exp(-squared_error / (2.0 * sigma * sigma))


@_maybe_compile
def tracking_exp_from_squared_error(
    squared_error: torch.Tensor, std: float
) -> torch.Tensor:
    return torch.exp(-squared_error / (std * std))


@_maybe_compile
def replace_nan_with_default(
    values: torch.Tensor, defaults: torch.Tensor
) -> torch.Tensor:
    return torch.where(torch.isnan(values), defaults, values)


@_maybe_compile
def relative_pose_from_bodies(
    body_pos: torch.Tensor, body_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    main_pos = body_pos[:, :1, :]
    rel_pos_w = body_pos[:, 1:, :] - main_pos
    main_quat = body_quat[:, :1, :].expand_as(body_quat[:, 1:, :]).reshape(-1, 4)
    rel_pos = quat_apply_inverse(main_quat, rel_pos_w.reshape(-1, 3)).reshape(
        body_pos.shape[0], -1, 3
    )
    child_quat = body_quat[:, 1:, :].reshape(-1, 4)
    rel_quat = quat_mul(quat_inv(main_quat), child_quat).reshape(
        body_quat.shape[0], -1, 4
    )
    return rel_pos, rel_quat


@_maybe_compile
def relative_velocity_from_bodies(
    body_quat: torch.Tensor,
    body_ang_vel: torch.Tensor,
    body_lin_vel: torch.Tensor,
) -> torch.Tensor:
    main_quat = body_quat[:, :1, :].expand_as(body_quat[:, 1:, :])
    main_ang = body_ang_vel[:, :1, :]
    main_lin = body_lin_vel[:, :1, :]
    child_ang = body_ang_vel[:, 1:, :]
    child_lin = body_lin_vel[:, 1:, :]

    main_quat_flat = main_quat.reshape(-1, 4)
    rel_lin = quat_apply_inverse(
        main_quat_flat, (main_lin - child_lin).reshape(-1, 3)
    ).reshape(body_quat.shape[0], -1, 3)
    child_ang_main = quat_apply_inverse(
        main_quat_flat, child_ang.reshape(-1, 3)
    ).reshape(body_quat.shape[0], -1, 3)
    main_ang_main = quat_apply_inverse(
        main_quat_flat, main_ang.expand_as(child_ang).reshape(-1, 3)
    ).reshape(body_quat.shape[0], -1, 3)
    rel_ang = child_ang_main - main_ang_main
    return torch.cat([rel_ang, rel_lin], dim=-1)


@_maybe_compile
def rms_error(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((actual - target) ** 2, dim=1))


@_maybe_compile
def xy_error_norm(actual_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((actual_xy - target_xy) ** 2, dim=1))


@_maybe_compile
def joint_pos_target_l2_kernel(joint_pos: torch.Tensor, target: float) -> torch.Tensor:
    return torch.sum(torch.square(wrap_to_pi(joint_pos) - target), dim=1)


@_maybe_compile
def reroot_body_positions(
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    ref_pos_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
    num_bodies = ref_pos_w.shape[1]
    delta_ori = heading_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))
    delta_pos = robot_anchor_pos_w.clone()
    delta_pos[:, 2] = ref_anchor_pos_w[:, 2]
    delta_ori_exp = delta_ori.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    ref_anchor_pos_exp = ref_anchor_pos_w.unsqueeze(1).expand(-1, num_bodies, -1)
    ref_offset = ref_pos_w - ref_anchor_pos_exp
    ref_offset_rot = quat_apply(delta_ori_exp, ref_offset.reshape(-1, 3)).reshape(
        -1, num_bodies, 3
    )
    return delta_pos.unsqueeze(1).expand(-1, num_bodies, -1) + ref_offset_rot


@_maybe_compile
def reroot_body_orientations(
    robot_anchor_quat_w: torch.Tensor,
    ref_quat_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
    num_bodies = ref_quat_w.shape[1]
    delta_ori = heading_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))
    delta_ori_exp = delta_ori.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    return quat_mul(delta_ori_exp, ref_quat_w.reshape(-1, 4)).reshape(-1, num_bodies, 4)


@_maybe_compile
def quat_error_squared(
    actual_quat: torch.Tensor, target_quat: torch.Tensor
) -> torch.Tensor:
    return quat_error_magnitude(actual_quat, target_quat).square()


@_maybe_compile
def apply_reset_randomization(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    env_origins: torch.Tensor,
    pose_delta: torch.Tensor,
    velocity_delta: torch.Tensor,
    joint_noise: torch.Tensor,
    joint_limit_low: torch.Tensor,
    joint_limit_high: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    randomized_root_pos = root_pos + env_origins + pose_delta[:, :3]
    randomized_root_quat = quat_mul(
        quat_from_euler_xyz(pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5]),
        root_quat,
    )
    randomized_root_lin_vel = root_lin_vel + velocity_delta[:, :3]
    randomized_root_ang_vel = root_ang_vel + velocity_delta[:, 3:]
    randomized_joint_pos = torch.clip(
        joint_pos + joint_noise, joint_limit_low, joint_limit_high
    )
    return (
        randomized_root_pos,
        randomized_root_quat,
        randomized_root_lin_vel,
        randomized_root_ang_vel,
        randomized_joint_pos,
        joint_vel,
    )
