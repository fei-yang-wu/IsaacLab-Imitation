"""Shared closed-loop tracking metrics for interface evaluations."""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.utils import math as math_utils
from tensordict import TensorDictBase


def optional_flat_tensor(
    td: TensorDictBase,
    key: str | tuple[str, ...],
    *,
    num_envs: int,
    default: float | bool,
) -> torch.Tensor:
    """Read an optional TensorDict value as one CPU scalar per environment."""
    try:
        value = td.get(key)
    except KeyError:
        value = None
    if not isinstance(value, torch.Tensor):
        return torch.full((num_envs,), default)
    flat = value.detach().reshape(-1).cpu()
    if flat.numel() == 1 and num_envs > 1:
        flat = flat.expand(num_envs)
    if flat.numel() < num_envs:
        raise RuntimeError(
            f"Expected at least {num_envs} values for {key}, got {flat.numel()}."
        )
    return flat[:num_envs]


def resolve_existing_body_names(base_env: Any, requested_names: list[str]) -> list[str]:
    """Keep only metric bodies available in both robot and reference data."""
    names: list[str] = []
    for name in requested_names:
        try:
            base_env._get_robot_anchor_body_id_fast(name)
            base_env._get_reference_body_ids_fast((name,))
        except Exception as exc:
            print(f"[WARNING] Skipping unavailable body metric target {name!r}: {exc}")
            continue
        names.append(str(name))
    return names


def _as_torch_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    torch_value = getattr(value, "torch", None)
    if isinstance(torch_value, torch.Tensor):
        return torch_value
    return torch.as_tensor(value)


def _mean_body_pose_errors(
    base_env: Any,
    names: list[str],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if len(names) == 0:
        return None
    body_ids = [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]
    actual_pos, actual_quat = base_env._get_robot_body_pose_w_fast(body_ids)
    ref_pos, ref_quat = base_env._get_reference_body_pose_w_fast(tuple(names))
    actual_pos = _as_torch_tensor(actual_pos)
    actual_quat = _as_torch_tensor(actual_quat)
    ref_pos = _as_torch_tensor(ref_pos)
    ref_quat = _as_torch_tensor(ref_quat)
    pos_error = torch.linalg.vector_norm(actual_pos - ref_pos, dim=-1).mean(dim=-1)
    ori_error = math_utils.quat_error_magnitude(
        actual_quat.reshape(-1, 4),
        ref_quat.reshape(-1, 4),
    ).reshape(actual_quat.shape[0], -1)
    return pos_error, ori_error.mean(dim=-1)


def _body_tracking_tensors(
    base_env: Any,
    names: list[str],
) -> dict[str, torch.Tensor] | None:
    if len(names) == 0:
        return None
    body_ids = [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]
    actual_pos, actual_quat = base_env._get_robot_body_pose_w_fast(body_ids)
    ref_pos, ref_quat = base_env._get_reference_body_pose_w_fast(tuple(names))
    actual_ang_vel, actual_lin_vel = base_env._get_robot_body_velocity_w_fast(body_ids)
    ref_ang_vel, ref_lin_vel = base_env._get_reference_body_velocity_w_fast(
        tuple(names)
    )
    return {
        "actual_pos": _as_torch_tensor(actual_pos),
        "actual_quat": _as_torch_tensor(actual_quat),
        "actual_ang_vel": _as_torch_tensor(actual_ang_vel),
        "actual_lin_vel": _as_torch_tensor(actual_lin_vel),
        "ref_pos": _as_torch_tensor(ref_pos),
        "ref_quat": _as_torch_tensor(ref_quat),
        "ref_ang_vel": _as_torch_tensor(ref_ang_vel),
        "ref_lin_vel": _as_torch_tensor(ref_lin_vel),
    }


def tracking_metrics(
    base_env: Any,
    *,
    tracked_body_names: list[str],
    ee_body_names: list[str],
    tracking_success_root_height_threshold: float,
    tracking_success_root_ori_threshold: float,
) -> tuple[
    dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor
]:
    """Measure the common robot-to-reference errors after one control step."""
    robot_data = base_env.robot.data
    root_pos_ref, root_quat_ref, root_lin_vel_ref, root_ang_vel_ref = (
        base_env._get_reference_root_state_w_fast()
    )
    root_pos = _as_torch_tensor(robot_data.root_pos_w)
    root_quat = _as_torch_tensor(robot_data.root_quat_w)
    root_lin_vel = _as_torch_tensor(robot_data.root_lin_vel_w)
    root_ang_vel = _as_torch_tensor(robot_data.root_ang_vel_w)
    root_pos_ref = _as_torch_tensor(root_pos_ref)
    root_quat_ref = _as_torch_tensor(root_quat_ref)
    root_lin_vel_ref = _as_torch_tensor(root_lin_vel_ref)
    root_ang_vel_ref = _as_torch_tensor(root_ang_vel_ref)
    joint_pos = _as_torch_tensor(robot_data.joint_pos)
    joint_vel = _as_torch_tensor(robot_data.joint_vel)
    joint_pos_ref = _as_torch_tensor(base_env.current_expert_frame["joint_pos"])
    joint_vel_ref = _as_torch_tensor(base_env.current_expert_frame["joint_vel"])
    root_pos_error = root_pos - root_pos_ref
    root_ori_error = math_utils.quat_error_magnitude(
        root_quat, root_quat_ref
    )
    root_height_error = torch.abs(root_pos_error[:, 2])
    tracking_failure = torch.zeros_like(root_height_error, dtype=torch.bool)
    if float(tracking_success_root_height_threshold) > 0.0:
        tracking_failure |= root_height_error > float(
            tracking_success_root_height_threshold
        )
    if float(tracking_success_root_ori_threshold) > 0.0:
        tracking_failure |= root_ori_error > float(tracking_success_root_ori_threshold)
    metrics = {
        "tracking_failure": tracking_failure.float(),
        "root_pos_xyz_error_m": torch.linalg.vector_norm(root_pos_error, dim=-1),
        "root_pos_xy_error_m": torch.linalg.vector_norm(root_pos_error[:, :2], dim=-1),
        "root_height_error_m": root_height_error,
        "root_ori_error_rad": root_ori_error,
        "joint_pos_rmse_rad": torch.sqrt(
            torch.mean((joint_pos - joint_pos_ref).square(), dim=-1)
        ),
        "joint_vel_rmse_radps": torch.sqrt(
            torch.mean((joint_vel - joint_vel_ref).square(), dim=-1)
        ),
        "root_lin_vel_rmse_mps": torch.sqrt(
            torch.mean((root_lin_vel - root_lin_vel_ref).square(), dim=-1)
        ),
        "root_ang_vel_rmse_radps": torch.sqrt(
            torch.mean((root_ang_vel - root_ang_vel_ref).square(), dim=-1)
        ),
    }
    tracked_body_lin_vel: tuple[torch.Tensor, torch.Tensor] | None = None
    tracked_tensors = _body_tracking_tensors(base_env, tracked_body_names)
    if tracked_tensors is not None:
        tracked_pos_error = torch.linalg.vector_norm(
            tracked_tensors["actual_pos"] - tracked_tensors["ref_pos"], dim=-1
        )
        tracked_ori_error = math_utils.quat_error_magnitude(
            tracked_tensors["actual_quat"].reshape(-1, 4),
            tracked_tensors["ref_quat"].reshape(-1, 4),
        ).reshape(tracked_tensors["actual_quat"].shape[0], -1)
        actual_root_rel = (
            tracked_tensors["actual_pos"] - root_pos[:, None, :]
        )
        ref_root_rel = tracked_tensors["ref_pos"] - root_pos_ref[:, None, :]
        tracking_mpjpe_m = torch.linalg.vector_norm(
            actual_root_rel - ref_root_rel, dim=-1
        ).mean(dim=-1)
        body_lin_vel_error = torch.linalg.vector_norm(
            tracked_tensors["actual_lin_vel"] - tracked_tensors["ref_lin_vel"],
            dim=-1,
        ).mean(dim=-1)
        body_ang_vel_error = torch.linalg.vector_norm(
            tracked_tensors["actual_ang_vel"] - tracked_tensors["ref_ang_vel"],
            dim=-1,
        ).mean(dim=-1)
        metrics["tracked_body_pos_error_m"] = tracked_pos_error.mean(dim=-1)
        metrics["tracked_body_ori_error_rad"] = tracked_ori_error.mean(dim=-1)
        metrics["tracked_body_lin_vel_error_mps"] = body_lin_vel_error
        metrics["tracked_body_ang_vel_error_radps"] = body_ang_vel_error
        metrics["tracking_mpjpe_m"] = tracking_mpjpe_m
        metrics["tracking_mpjpe_mm"] = tracking_mpjpe_m * 1000.0
        metrics["tracking_velocity_distance_mps"] = body_lin_vel_error
        tracked_body_lin_vel = (
            tracked_tensors["actual_lin_vel"].detach(),
            tracked_tensors["ref_lin_vel"].detach(),
        )
    ee_errors = _mean_body_pose_errors(base_env, ee_body_names)
    if ee_errors is not None:
        metrics["ee_pos_error_m"] = ee_errors[0]
        metrics["ee_ori_error_rad"] = ee_errors[1]
    return metrics, tracked_body_lin_vel, tracking_failure


def accumulate_metric(
    stats: dict[str, list[torch.Tensor]],
    metric_name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Append masked per-environment values to a metric accumulator."""
    selected = values.detach().cpu()[mask.cpu()]
    if selected.numel() == 0:
        return
    stats.setdefault(metric_name, []).append(selected.float())


def finalize_metric_stats(
    stats: dict[str, list[torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Convert metric accumulators to a common mean/std/count schema."""
    finalized: dict[str, dict[str, float]] = {}
    for name, chunks in sorted(stats.items()):
        values = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        finalized[name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item())
            if values.numel() > 1
            else 0.0,
            "count": int(values.numel()),
        }
    return finalized


def tensor_mean_std(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float]:
    """Return population mean and standard deviation over a boolean mask."""
    selected = values[mask]
    if selected.numel() == 0:
        return float("nan"), float("nan")
    return (
        float(selected.mean().item()),
        float(selected.std(unbiased=False).item()) if selected.numel() > 1 else 0.0,
    )
