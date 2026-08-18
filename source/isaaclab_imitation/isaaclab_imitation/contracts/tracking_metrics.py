"""Environment-free tracking metric definitions shared by training and eval."""

from __future__ import annotations

import torch


def mpjpe_local_global(
    robot_body_pos_w: torch.Tensor,
    robot_root_pos_w: torch.Tensor,
    reference_body_pos_w: torch.Tensor,
    reference_root_pos_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample root-relative and world-frame MPJPE in metres.

    Root-relative MPJPE subtracts each side's own root position. It does not
    rotate either body set and is therefore not Procrustes-aligned MPJPE.
    Body tensors have shape ``[..., bodies, 3]`` and root tensors have shape
    ``[..., 3]``.
    """

    if robot_body_pos_w.shape != reference_body_pos_w.shape:
        raise ValueError(
            "Robot and reference body positions must have the same shape, got "
            f"{tuple(robot_body_pos_w.shape)} and "
            f"{tuple(reference_body_pos_w.shape)}."
        )
    if robot_body_pos_w.ndim < 2 or robot_body_pos_w.shape[-1] != 3:
        raise ValueError(
            "Body positions must have shape [..., bodies, 3], got "
            f"{tuple(robot_body_pos_w.shape)}."
        )
    expected_root_shape = robot_body_pos_w.shape[:-2] + (3,)
    if tuple(robot_root_pos_w.shape) != tuple(expected_root_shape):
        raise ValueError(
            "Robot root positions must have shape "
            f"{tuple(expected_root_shape)}, got {tuple(robot_root_pos_w.shape)}."
        )
    if tuple(reference_root_pos_w.shape) != tuple(expected_root_shape):
        raise ValueError(
            "Reference root positions must have shape "
            f"{tuple(expected_root_shape)}, got "
            f"{tuple(reference_root_pos_w.shape)}."
        )

    robot_relative = robot_body_pos_w - robot_root_pos_w.unsqueeze(-2)
    reference_relative = reference_body_pos_w - reference_root_pos_w.unsqueeze(-2)
    local = torch.linalg.vector_norm(
        robot_relative - reference_relative, dim=-1
    ).mean(dim=-1)
    global_ = torch.linalg.vector_norm(
        robot_body_pos_w - reference_body_pos_w, dim=-1
    ).mean(dim=-1)
    return local, global_


__all__ = ["mpjpe_local_global"]
