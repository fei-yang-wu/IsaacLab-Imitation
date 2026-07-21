from __future__ import annotations

from typing import Literal

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
from isaaclab_imitation.envs import ImitationRLEnv

from ._compiled import apply_reset_randomization, replace_nan_with_default


def _initialize_reset_bounds(
    env: ImitationRLEnv,
    pose_range: dict[str, tuple[float, float]] | None,
    velocity_range: dict[str, tuple[float, float]] | None,
    device: torch.device,
) -> None:
    if env._mdp_reset_pose_bounds is None:
        pose_range = pose_range or {}
        env._mdp_reset_pose_bounds = torch.tensor(
            [
                pose_range.get(key, (0.0, 0.0))
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            device=device,
        )
    if env._mdp_reset_velocity_bounds is None:
        velocity_range = velocity_range or {}
        env._mdp_reset_velocity_bounds = torch.tensor(
            [
                velocity_range.get(key, (0.0, 0.0))
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            device=device,
        )


def randomize_joint_default_pos(
    env: ImitationRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.default_joint_pos_nominal = torch.clone(
        asset.data.default_joint_pos.torch[0]
    )

    if pos_distribution_params is None:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.int, device=asset.device
        )

    randomized_pos = _randomize_prop_by_op(
        asset.data.default_joint_pos.torch.clone(),
        pos_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )

    env_ids_for_slice = env_ids[:, None] if joint_ids != slice(None) else env_ids
    selected_pos = randomized_pos[env_ids_for_slice, joint_ids]
    asset.data.default_joint_pos.torch[env_ids_for_slice, joint_ids] = selected_pos

    # ``default_joint_pos`` is indexed in the live articulation order, but the
    # action term's offset is indexed in the term's own pinned joint order.
    # Those two orders differ per physics backend, so the offset must be
    # gathered through the term's live-index mapping rather than copied
    # slot-for-slot. Writing it positionally scatters each joint's rest pose
    # onto a different joint on any backend whose enumeration is not the one
    # the pinned list was authored for.
    joint_pos_action_term = env.action_manager.get_term("joint_pos")
    offset = getattr(joint_pos_action_term, "_offset", None)
    if isinstance(offset, torch.Tensor):
        term_joint_ids = getattr(joint_pos_action_term, "_joint_ids", slice(None))
        if isinstance(term_joint_ids, slice):
            offset[env_ids] = randomized_pos[env_ids][:, term_joint_ids]
        else:
            term_joint_ids = torch.as_tensor(
                term_joint_ids, dtype=torch.long, device=asset.device
            )
            offset[env_ids] = randomized_pos[env_ids][:, term_joint_ids]


def reset_joints_to_reference(
    env: ImitationRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = env_ids.to(dtype=torch.int64, device=asset.device)
    reference_joint_pos = env.current_expert_frame["joint_pos"].index_select(0, env_ids)
    reference_joint_vel = env.current_expert_frame["joint_vel"].index_select(0, env_ids)
    joint_pos = replace_nan_with_default(
        reference_joint_pos, asset.data.default_joint_pos.torch.index_select(0, env_ids)
    ).clone()
    joint_vel = replace_nan_with_default(
        reference_joint_vel, asset.data.default_joint_vel.torch.index_select(0, env_ids)
    ).clone()
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.write_data_to_sim()
    env.scene.update(dt=0.0)
    asset.update(dt=0.0)
    env._invalidate_mdp_cache()


def reset_root_and_joints_to_reference_with_randomization(
    env: ImitationRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (0.0, 0.0),
):
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device
    env_ids = env_ids.to(dtype=torch.int64, device=device)
    reference = env.current_expert_frame
    _initialize_reset_bounds(env, pose_range, velocity_range, device)

    if env._mdp_reset_root_pose_source == "root":
        root_pos = reference["root_pos"].index_select(0, env_ids)
        root_quat = reference["root_quat"].index_select(0, env_ids)
    else:
        root_pos = reference["body_pos_w"].index_select(0, env_ids)[:, 0, :]
        root_quat = reference["body_quat_w"].index_select(0, env_ids)[:, 0, :]

    if env._mdp_reset_root_velocity_source == "root":
        root_lin_vel = reference["root_lin_vel"].index_select(0, env_ids)
        root_ang_vel = reference["root_ang_vel"].index_select(0, env_ids)
    elif env._mdp_reset_root_velocity_source == "body":
        root_lin_vel = reference["body_lin_vel_w"].index_select(0, env_ids)[:, 0, :]
        root_ang_vel = reference["body_ang_vel_w"].index_select(0, env_ids)[:, 0, :]
    else:
        root_lin_vel = torch.zeros_like(root_pos)
        root_ang_vel = torch.zeros_like(root_pos)

    joint_pos = replace_nan_with_default(
        reference["joint_pos"].index_select(0, env_ids),
        asset.data.default_joint_pos.torch.index_select(0, env_ids),
    ).clone()
    joint_vel = replace_nan_with_default(
        reference["joint_vel"].index_select(0, env_ids),
        asset.data.default_joint_vel.torch.index_select(0, env_ids),
    ).clone()

    pose_delta = sample_uniform(
        env._mdp_reset_pose_bounds[:, 0],
        env._mdp_reset_pose_bounds[:, 1],
        (env_ids.numel(), 6),
        device=device,
    )
    velocity_delta = sample_uniform(
        env._mdp_reset_velocity_bounds[:, 0],
        env._mdp_reset_velocity_bounds[:, 1],
        (env_ids.numel(), 6),
        device=device,
    )
    if joint_position_range[0] == 0.0 and joint_position_range[1] == 0.0:
        joint_noise = torch.zeros_like(joint_pos)
    else:
        joint_noise = sample_uniform(
            joint_position_range[0],
            joint_position_range[1],
            joint_pos.shape,
            device=device,
        )

    soft_joint_pos_limits = asset.data.soft_joint_pos_limits.torch.index_select(
        0, env_ids
    )
    (
        randomized_root_pos,
        randomized_root_quat,
        randomized_root_lin_vel,
        randomized_root_ang_vel,
        randomized_joint_pos,
        randomized_joint_vel,
    ) = apply_reset_randomization(
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        joint_pos,
        joint_vel,
        env.scene.env_origins.index_select(0, env_ids),
        pose_delta,
        velocity_delta,
        joint_noise,
        soft_joint_pos_limits[:, :, 0],
        soft_joint_pos_limits[:, :, 1],
    )

    asset.write_joint_state_to_sim(
        randomized_joint_pos,
        randomized_joint_vel,
        env_ids=env_ids,
    )
    root_state = torch.cat(
        [
            randomized_root_pos,
            randomized_root_quat,
            randomized_root_lin_vel,
            randomized_root_ang_vel,
        ],
        dim=-1,
    )
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_data_to_sim()
    env.scene.update(dt=0.0)
    asset.update(dt=0.0)
    env._invalidate_mdp_cache()
