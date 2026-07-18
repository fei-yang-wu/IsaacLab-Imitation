from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_imitation.envs import ImitationRLEnv

from ._compiled import (
    gaussian_from_squared_error,
    joint_pos_target_l2_kernel,
    quat_error_squared,
    relative_pose_from_bodies,
    relative_velocity_from_bodies,
    reroot_body_orientations,
    reroot_body_positions,
    tracking_exp_from_squared_error,
)


def _select_last_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(-1, ids)


def _select_body_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(1, ids)


def joint_pos_target_l2(
    env: ImitationRLEnv, target: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    joint_pos = _select_last_dim(asset.data.joint_pos.torch, joint_ids)
    return joint_pos_target_l2_kernel(joint_pos, target)


def track_joint_pos(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    qpos_actual = _select_last_dim(
        env.scene[asset_cfg.name].data.joint_pos.torch, joint_ids
    )
    qpos_reference = _select_last_dim(env.current_expert_frame["joint_pos"], joint_ids)
    squared_error = torch.sum((qpos_actual - qpos_reference) ** 2, dim=1)
    return gaussian_from_squared_error(squared_error, sigma)


def track_joint_vel(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    qvel_actual = _select_last_dim(
        env.scene[asset_cfg.name].data.joint_vel.torch, joint_ids
    )
    qvel_reference = _select_last_dim(env.current_expert_frame["joint_vel"], joint_ids)
    squared_error = torch.sum((qvel_actual - qvel_reference) ** 2, dim=1)
    return gaussian_from_squared_error(squared_error, sigma)


def track_root_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_actual = asset.data.root_state_w.torch[:, :3]
    root_pos_reference_w, _, _, _ = env._get_reference_root_state_w_fast()
    squared_error_xy = torch.sum(
        (root_pos_actual[..., :2] - root_pos_reference_w[..., :2]) ** 2, dim=1
    )
    return gaussian_from_squared_error(squared_error_xy, sigma)


def track_root_quat(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_actual = asset.data.root_state_w.torch[:, 3:7]
    _, root_quat_reference_w, _, _ = env._get_reference_root_state_w_fast()
    squared_error = quat_error_squared(root_quat_actual, root_quat_reference_w)
    return gaussian_from_squared_error(squared_error, sigma)


def track_root_ang(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_actual = asset.data.root_quat_w.torch
    _, root_quat_reference_w, _, _ = env._get_reference_root_state_w_fast()
    squared_error = quat_error_squared(root_quat_actual, root_quat_reference_w)
    return gaussian_from_squared_error(squared_error, sigma)


def track_root_lin_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_link_state_actual = asset.data.root_link_state_w.torch
    root_quat_actual = root_link_state_actual[:, 3:7]
    root_lin_vel_actual_b = quat_apply_inverse(
        root_quat_actual, root_link_state_actual[:, 7:10]
    )
    _, _, root_lin_vel_reference_w, _ = env._get_reference_root_state_w_fast()
    root_lin_vel_reference_b = quat_apply_inverse(
        root_quat_actual, root_lin_vel_reference_w
    )
    squared_error = torch.sum(
        (root_lin_vel_actual_b[..., :2] - root_lin_vel_reference_b[..., :2]) ** 2,
        dim=-1,
    )
    return gaussian_from_squared_error(squared_error, sigma)


def track_root_ang_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg | None = None, sigma: float = 0.1
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_ang_vel_actual = asset.data.root_link_state_w.torch[:, 10:13]
    _, _, _, root_ang_vel_reference_w = env._get_reference_root_state_w_fast()
    squared_error = torch.sum(
        (root_ang_vel_actual - root_ang_vel_reference_w) ** 2, dim=-1
    )
    return gaussian_from_squared_error(squared_error, sigma)


def track_relative_body_pos(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    sigma: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = env._get_body_ids_tensor_fast(asset_cfg.body_ids)
    if isinstance(body_ids, slice):
        actual_pos = asset.data.body_link_pos_w.torch
        actual_quat = asset.data.body_link_quat_w.torch
    else:
        actual_pos = asset.data.body_link_pos_w.torch.index_select(1, body_ids)
        actual_quat = asset.data.body_link_quat_w.torch.index_select(1, body_ids)
    actual_rel_pos, _ = relative_pose_from_bodies(actual_pos, actual_quat)
    ref_pos_w, ref_quat_w = env._get_reference_body_pose_w_fast(reference_body_names)
    ref_rel_pos, _ = relative_pose_from_bodies(ref_pos_w, ref_quat_w)
    squared_error = torch.mean((actual_rel_pos - ref_rel_pos) ** 2, dim=(1, 2))
    return gaussian_from_squared_error(squared_error, sigma)


def track_relative_body_quat(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    sigma: float = 0.1,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = env._get_body_ids_tensor_fast(asset_cfg.body_ids)
    if isinstance(body_ids, slice):
        actual_pos = asset.data.body_link_pos_w.torch
        actual_quat = asset.data.body_link_quat_w.torch
    else:
        actual_pos = asset.data.body_link_pos_w.torch.index_select(1, body_ids)
        actual_quat = asset.data.body_link_quat_w.torch.index_select(1, body_ids)
    ref_pos_w, ref_quat_w = env._get_reference_body_pose_w_fast(reference_body_names)
    _, actual_rel_quat = relative_pose_from_bodies(actual_pos, actual_quat)
    _, ref_rel_quat = relative_pose_from_bodies(ref_pos_w, ref_quat_w)
    squared_error = quat_error_squared(
        actual_rel_quat.reshape(-1, 4), ref_rel_quat.reshape(-1, 4)
    ).reshape(actual_rel_quat.shape[0], -1)
    return gaussian_from_squared_error(torch.mean(squared_error, dim=1), sigma)


def track_relative_body_vel(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    sigma: float = 0.2,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = env._get_body_ids_tensor_fast(asset_cfg.body_ids)
    if isinstance(body_ids, slice):
        actual_quat = asset.data.body_link_quat_w.torch
        actual_ang_vel = asset.data.body_ang_vel_w.torch
        actual_lin_vel = asset.data.body_lin_vel_w.torch
    else:
        actual_quat = asset.data.body_link_quat_w.torch.index_select(1, body_ids)
        actual_ang_vel = asset.data.body_ang_vel_w.torch.index_select(1, body_ids)
        actual_lin_vel = asset.data.body_lin_vel_w.torch.index_select(1, body_ids)
    actual_rel_vel = relative_velocity_from_bodies(
        actual_quat, actual_ang_vel, actual_lin_vel
    )
    ref_quat_w = env._get_reference_body_pose_w_fast(reference_body_names)[1]
    ref_ang_vel_w, ref_lin_vel_w = env._get_reference_body_velocity_w_fast(
        reference_body_names
    )
    ref_rel_vel = relative_velocity_from_bodies(
        ref_quat_w, ref_ang_vel_w, ref_lin_vel_w
    )
    squared_error = torch.mean((actual_rel_vel - ref_rel_vel) ** 2, dim=(1, 2))
    return gaussian_from_squared_error(squared_error, sigma)


def reference_global_anchor_position_error_exp(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    anchor_body_name: str = "torso_link",
    std: float = 0.3,
) -> torch.Tensor:
    robot_anchor_pos_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[0]
    ref_anchor_pos_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[0][
        :, 0, :
    ]
    squared_error = torch.sum((ref_anchor_pos_w - robot_anchor_pos_w) ** 2, dim=-1)
    return tracking_exp_from_squared_error(squared_error, std)


def reference_global_anchor_orientation_error_exp(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    anchor_body_name: str = "torso_link",
    std: float = 0.4,
) -> torch.Tensor:
    robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[1]
    ref_anchor_quat_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[1][
        :, 0, :
    ]
    squared_error = quat_error_squared(ref_anchor_quat_w, robot_anchor_quat_w)
    return tracking_exp_from_squared_error(squared_error, std)


def reference_relative_body_position_error_exp(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    anchor_body_name: str = "torso_link",
    std: float = 0.3,
) -> torch.Tensor:
    body_ids = asset_cfg.body_ids
    robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
        anchor_body_name
    )
    ref_pos_w, _ = env._get_reference_body_pose_w_fast(reference_body_names)
    ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
        (anchor_body_name,)
    )
    target_pos_w = reroot_body_positions(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        ref_pos_w,
        ref_anchor_pos_w[:, 0, :],
        ref_anchor_quat_w[:, 0, :],
    )
    actual_pos_w = env._get_robot_body_pose_w_fast(body_ids)[0]
    squared_error = torch.sum((target_pos_w - actual_pos_w) ** 2, dim=-1).mean(-1)
    return tracking_exp_from_squared_error(squared_error, std)


def reference_relative_body_orientation_error_exp(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    anchor_body_name: str = "torso_link",
    std: float = 0.4,
) -> torch.Tensor:
    body_ids = asset_cfg.body_ids
    robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[1]
    ref_quat_w = env._get_reference_body_pose_w_fast(reference_body_names)[1]
    ref_anchor_quat_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[1][
        :, 0, :
    ]
    target_quat_w = reroot_body_orientations(
        robot_anchor_quat_w, ref_quat_w, ref_anchor_quat_w
    )
    actual_quat_w = env._get_robot_body_pose_w_fast(body_ids)[1]
    squared_error = quat_error_squared(
        target_quat_w.reshape(-1, 4), actual_quat_w.reshape(-1, 4)
    ).reshape(target_quat_w.shape[0], -1)
    return tracking_exp_from_squared_error(torch.mean(squared_error, dim=1), std)


def reference_global_body_linear_velocity_error_exp(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    std: float = 1.0,
) -> torch.Tensor:
    ref_lin_vel_w = env._get_reference_body_velocity_w_fast(reference_body_names)[1]
    actual_lin_vel_w = env._get_robot_body_velocity_w_fast(asset_cfg.body_ids)[1]
    squared_error = torch.sum((ref_lin_vel_w - actual_lin_vel_w) ** 2, dim=-1).mean(-1)
    return tracking_exp_from_squared_error(squared_error, std)


def reference_global_body_angular_velocity_error_exp(
    env: ImitationRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
    reference_body_names: Sequence[str] = (),
    std: float = 3.14,
) -> torch.Tensor:
    ref_ang_vel_w = env._get_reference_body_velocity_w_fast(reference_body_names)[0]
    actual_ang_vel_w = env._get_robot_body_velocity_w_fast(asset_cfg.body_ids)[0]
    squared_error = torch.sum((ref_ang_vel_w - actual_ang_vel_w) ** 2, dim=-1).mean(-1)
    return tracking_exp_from_squared_error(squared_error, std)
