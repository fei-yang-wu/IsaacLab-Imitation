"""Joint residuals for each seven-joint arm and its mounted Sharpa hand."""

from __future__ import annotations

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from .sharpa_command import _torch


@configclass
class VegaSharpaJointResidualCfg(ActionTermCfg):
    side: str = "right"
    command_name: str = "sharpa_reference"
    arm_scale: float = 0.1
    finger_scale: float = 0.15
    arm_clip: float = 0.5
    finger_clip: float = 1.0
    ema_factor: float = 0.3

    def __post_init__(self):
        self.class_type = VegaSharpaJointResidual


class VegaSharpaJointResidual(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.side not in ("left", "right"):
            raise ValueError("Mounted action side must be left or right.")
        self.command = env.command_manager.get_term(cfg.command_name)
        prefix = "L" if cfg.side == "left" else "R"
        names = [f"{prefix}_arm_j{i}" for i in range(1, 8)] + list(
            getattr(self.command, f"{cfg.side}_finger_joint_names")
        )
        self.joint_ids, _ = self._asset.find_joints(names, preserve_order=True)
        self._reference_indices = [
            self.command.reference_joint_names.index(n) for n in names
        ]
        self._raw_actions = torch.zeros((self.num_envs, len(names)), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._residual = torch.zeros_like(self._raw_actions)
        self._scale = torch.tensor(
            [cfg.arm_scale] * 7 + [cfg.finger_scale] * 22, device=self.device
        )
        self._clip = torch.tensor(
            [cfg.arm_clip] * 7 + [cfg.finger_clip] * 22, device=self.device
        )

    @property
    def action_dim(self):
        return len(self.joint_ids)

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._residual[:] = torch.clamp(
            self.cfg.ema_factor * self._residual
            + (1 - self.cfg.ema_factor) * actions * self._scale,
            -self._clip,
            self._clip,
        )
        reference = self.command._select(self.command._mounted_qpos)[
            :, self._reference_indices
        ]
        limits = _torch(self._asset.data.soft_joint_pos_limits)[:, self.joint_ids]
        self._processed_actions[:] = torch.clamp(
            reference + self._residual, limits[..., 0], limits[..., 1]
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self.joint_ids
        )
        offset = 0 if self.cfg.side == "left" else 7
        self._asset.set_joint_velocity_target(
            self.command.arm_velocity_command[:, offset : offset + 7],
            joint_ids=self.joint_ids[:7],
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self._residual[ids] = 0
        self._processed_actions[ids] = self.command._select(self.command._mounted_qpos)[
            ids
        ][:, self._reference_indices]
