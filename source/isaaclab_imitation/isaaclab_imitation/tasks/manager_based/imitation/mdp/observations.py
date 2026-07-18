from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab_imitation.envs import ImitationRLEnv

from ._compiled import body_pose_in_anchor_frame, quat_to_rot6d_flat


def _select_last_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(-1, ids)


def expert_joint_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    return _select_last_dim(env.current_expert_frame["joint_pos"], joint_ids)


def expert_joint_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    return _select_last_dim(env.current_expert_frame["joint_vel"], joint_ids)


def expert_root_lin_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    del asset_cfg
    return env.current_expert_frame["root_lin_vel"]


def expert_root_ang_vel(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    del asset_cfg
    return env.current_expert_frame["root_ang_vel"]


def expert_root_pos(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    del asset_cfg
    return env.current_expert_frame["root_pos"]


def expert_root_quat(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    del asset_cfg
    return env.current_expert_frame["root_quat"]


def expert_motion_command(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return env._get_expert_motion_command_fast(asset_cfg.joint_ids)


def robot_motion(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    joint_pos = _select_last_dim(env.robot.data.joint_pos.torch, joint_ids)
    joint_vel = _select_last_dim(env.robot.data.joint_vel.torch, joint_ids)
    return torch.cat([joint_pos, joint_vel], dim=-1)


def agent_latent_command(
    env: ImitationRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    del asset_cfg
    return env.get_agent_latent_command()


def expert_anchor_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
        anchor_body_name
    )
    ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
        (anchor_body_name,)
    )
    anchor_pos_b, _ = body_pose_in_anchor_frame(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )
    return anchor_pos_b[:, 0, :]


def expert_anchor_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
        anchor_body_name
    )
    ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
        (anchor_body_name,)
    )
    _, anchor_ori_b = body_pose_in_anchor_frame(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        ref_anchor_pos_w,
        ref_anchor_quat_w,
    )
    return quat_to_rot6d_flat(anchor_ori_b[:, 0, :])


def expert_window_motion(
    env: ImitationRLEnv,
    past_steps: int = 1,
    future_steps: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.get_current_command_window_term(
        term_name="expert_motion",
        past_steps=past_steps,
        future_steps=future_steps,
        joint_ids=asset_cfg.joint_ids,
    )


def expert_window_anchor_pos_b(
    env: ImitationRLEnv,
    past_steps: int = 1,
    future_steps: int = 1,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    return env.get_current_command_window_term(
        term_name="expert_anchor_pos_b",
        past_steps=past_steps,
        future_steps=future_steps,
        anchor_body_name=anchor_body_name,
    )


def expert_window_anchor_ori_b(
    env: ImitationRLEnv,
    past_steps: int = 1,
    future_steps: int = 1,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    return env.get_current_command_window_term(
        term_name="expert_anchor_ori_b",
        past_steps=past_steps,
        future_steps=future_steps,
        anchor_body_name=anchor_body_name,
    )


def expert_window_ee_pos_b(
    env: ImitationRLEnv,
    past_steps: int = 1,
    future_steps: int = 1,
    reference_body_names: tuple[str, ...] = (),
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    return env.get_current_command_window_term(
        term_name="expert_ee_pos_b",
        past_steps=past_steps,
        future_steps=future_steps,
        anchor_body_name=anchor_body_name,
        reference_body_names=reference_body_names,
    )


def expert_window_ee_ori_b(
    env: ImitationRLEnv,
    past_steps: int = 1,
    future_steps: int = 1,
    reference_body_names: tuple[str, ...] = (),
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    return env.get_current_command_window_term(
        term_name="expert_ee_ori_b",
        past_steps=past_steps,
        future_steps=future_steps,
        anchor_body_name=anchor_body_name,
        reference_body_names=reference_body_names,
    )


def expert_goal_motion(
    env: ImitationRLEnv,
    goal_steps: int = 1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.get_current_expert_goal_term(
        term_name="expert_motion",
        goal_steps=goal_steps,
        joint_ids=asset_cfg.joint_ids,
    )


def expert_goal_anchor_pos_b(
    env: ImitationRLEnv,
    goal_steps: int = 1,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    return env.get_current_expert_goal_term(
        term_name="expert_anchor_pos_b",
        goal_steps=goal_steps,
        anchor_body_name=anchor_body_name,
    )


def expert_goal_anchor_ori_b(
    env: ImitationRLEnv,
    goal_steps: int = 1,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    return env.get_current_expert_goal_term(
        term_name="expert_anchor_ori_b",
        goal_steps=goal_steps,
        anchor_body_name=anchor_body_name,
    )


def robot_body_pos_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    body_pos_b, _ = env._get_robot_body_state_in_anchor_frame_fast(
        asset_cfg.body_ids, anchor_body_name
    )
    return body_pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ImitationRLEnv,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, body_ori_b = env._get_robot_body_state_in_anchor_frame_fast(
        asset_cfg.body_ids, anchor_body_name
    )
    return quat_to_rot6d_flat(body_ori_b)
