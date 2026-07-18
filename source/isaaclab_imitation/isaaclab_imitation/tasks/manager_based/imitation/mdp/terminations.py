from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_imitation.envs import ImitationRLEnv

from ._compiled import quat_error_squared, rms_error, xy_error_norm


def _select_last_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(-1, ids)


def _select_body_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(1, ids)


def reference_joint_pos_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    joint_pos_actual = _select_last_dim(asset.data.joint_pos.torch, joint_ids)
    joint_pos_reference = _select_last_dim(
        env.current_expert_frame["joint_pos"], joint_ids
    )
    return rms_error(joint_pos_actual, joint_pos_reference) > threshold


def reference_root_position_xy_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_actual = asset.data.root_state_w.torch[:, :3]
    root_pos_reference_w = env._get_reference_root_state_w_fast()[0]
    return (
        xy_error_norm(root_pos_actual[:, :2], root_pos_reference_w[:, :2]) > threshold
    )


def reference_root_quat_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_actual = asset.data.root_state_w.torch[:, 3:7]
    root_quat_reference_w = env._get_reference_root_state_w_fast()[1]
    angular_error = torch.sqrt(
        quat_error_squared(root_quat_actual, root_quat_reference_w)
    )
    return angular_error > threshold


def bad_anchor_pos_z_only(
    env: ImitationRLEnv,
    threshold: float = 0.25,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot_anchor_pos_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[0]
    ref_anchor_pos_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[0][
        :, 0, :
    ]
    return torch.abs(ref_anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2]) > threshold


def bad_anchor_ori(
    env: ImitationRLEnv,
    threshold: float = 0.8,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[1]
    ref_anchor_quat_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[1][
        :, 0, :
    ]
    reference_projected_gravity_b = quat_apply_inverse(
        ref_anchor_quat_w, asset.data.GRAVITY_VEC_W.torch
    )
    robot_projected_gravity_b = quat_apply_inverse(
        robot_anchor_quat_w, asset.data.GRAVITY_VEC_W.torch
    )
    return (
        reference_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
    ).abs() > threshold


def bad_reference_body_pos_z_only(
    env: ImitationRLEnv,
    threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_body_names: Sequence[str] = (),
) -> torch.Tensor:
    actual_pos_w = env._get_robot_body_pose_w_fast(asset_cfg.body_ids)[0]
    ref_pos_w = env._get_reference_body_pose_w_fast(reference_body_names)[0]
    return torch.any(
        torch.abs(ref_pos_w[..., 2] - actual_pos_w[..., 2]) > threshold, dim=1
    )


def reference_trajectory_finished(env: ImitationRLEnv) -> torch.Tensor:
    return env.current_reference_is_final_frame()
