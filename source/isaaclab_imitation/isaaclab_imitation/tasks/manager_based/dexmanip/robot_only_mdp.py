"""MDP terms for the robot-only Vega-Wuji tracking task."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg


class JointPositionClipToLimitsAction(JointPositionAction):
    """Apply position targets with the articulation's per-joint soft limits.

    The policy output is not clipped before the affine transform. The
    processed target keeps the configured default-joint offset, is clipped to
    the actual soft limits from the loaded asset, and is smoothed with an EMA.
    This differs from ``JointPositionToLimitsAction``: zero action remains the
    default joint pose instead of becoming the midpoint of an asymmetric
    limit.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._ema_actions = torch.zeros_like(self.raw_actions)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ema_actions[env_ids] = self._asset.data.joint_pos.torch[
            env_ids, self._joint_ids
        ]

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = torch.maximum(
            torch.minimum(self._processed_actions, limits[..., 1]), limits[..., 0]
        )
        alpha = float(self.cfg.alpha)
        self._ema_actions = alpha * target + (1.0 - alpha) * self._ema_actions
        self._processed_actions = torch.maximum(
            torch.minimum(self._ema_actions, limits[..., 1]), limits[..., 0]
        )


def joint_pos_rel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return isaac_mdp.joint_pos_rel(env, asset_cfg)


def joint_vel_rel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return isaac_mdp.joint_vel_rel(env, asset_cfg)


def last_action(env) -> torch.Tensor:
    return isaac_mdp.last_action(env)


def reference_joint_pos(env) -> torch.Tensor:
    return env.current_reference_joint_pos


def reference_joint_vel(env) -> torch.Tensor:
    return env.current_reference_joint_vel


def track_joint_pos(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.25,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos.torch - env.current_reference_joint_pos
    # Use mean per-joint error so the reward scale does not collapse when the
    # robot has 59 joints. The former sum made a modest error in every joint
    # indistinguishable from zero reward.
    return torch.exp(-torch.mean(error.square(), dim=-1) / (2.0 * sigma**2))


def track_joint_vel(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_vel.torch - env.current_reference_joint_vel
    return torch.exp(-torch.mean(error.square(), dim=-1) / (2.0 * sigma**2))


def action_rate_l2(env) -> torch.Tensor:
    return isaac_mdp.action_rate_l2(env)


__all__ = [
    "action_rate_l2",
    "joint_pos_rel",
    "joint_vel_rel",
    "JointPositionClipToLimitsAction",
    "last_action",
    "reference_joint_pos",
    "reference_joint_vel",
    "track_joint_pos",
    "track_joint_vel",
]
