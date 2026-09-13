# Copyright (c) 2026, IsaacLab-Imitation Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms specific to the imitation environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
# Explicit module path: `from isaaclab.utils import configclass` resolves to
# the configclass MODULE in the cluster container's Isaac Lab build (killed
# jobs 5597424/26/29 in 23s); only the workstation build re-exports the
# decorator at package level.
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EMAJointPositionAction(JointPositionAction):
    r"""Joint position action with an exponential moving average on the target.

    The published joint-position target is

    .. math::

        q^{target}_t = \alpha \, (\text{offset} + \text{scale} \times a_t)
                       + (1 - \alpha) \, q^{target}_{t-1}

    a trained-in first-order low-pass at the 50 Hz control rate. At
    ``ema_alpha=1.0`` the filter is the identity and this term is
    byte-for-byte :class:`JointPositionAction`. The one-step cutoff is
    ``f_c = f_s * ln(1 / (1 - alpha)) / (2 pi)`` — ``alpha=0.65`` is about
    8.4 Hz at 50 Hz.

    The filter state resets to the default joint pose on episode reset, so
    the first post-reset target is the nominal stand, not a stale target
    from the previous episode.
    """

    cfg: EMAJointPositionActionCfg

    def __init__(self, cfg: EMAJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        alpha = float(cfg.ema_alpha)
        if not 0.0 < alpha <= 1.0:
            msg = f"ema_alpha must be in (0, 1], got {alpha}."
            raise ValueError(msg)
        self._ema_alpha = alpha
        self._ema_state: torch.Tensor | None = None
        # Actuation delay in physics sub-steps (see the cfg docstring).
        decimation = int(getattr(env.cfg, "decimation", 1) or 1)
        low, high = int(cfg.delay_substeps_min), int(cfg.delay_substeps_max)
        if not 0 <= low <= high <= decimation:
            msg = (
                f"delay_substeps must satisfy 0 <= min <= max <= decimation "
                f"({decimation}), got ({low}, {high})."
            )
            raise ValueError(msg)
        self._delay_range = (low, high)
        self._delay_enabled = high > 0
        self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_target: torch.Tensor | None = None
        self._substep = 0
        if self._delay_enabled:
            self._sample_delays(slice(None))

    def _sample_delays(self, env_ids: Sequence[int] | slice) -> None:
        low, high = self._delay_range
        count = (
            self.num_envs
            if isinstance(env_ids, slice)
            else int(torch.as_tensor(env_ids).numel())
        )
        self._delay_steps[env_ids] = torch.randint(
            low, high + 1, (count,), device=self.device
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._delay_enabled:
            # The target the writer keeps applying until the new one lands.
            if self._previous_target is None:
                self._previous_target = self._offset_tensor().clone()
            else:
                self._previous_target.copy_(self._processed_actions)
            self._substep = 0
        super().process_actions(actions)
        if self._ema_alpha >= 1.0:
            return
        if self._ema_state is None:
            self._ema_state = self._processed_actions.clone()
        else:
            self._ema_state.mul_(1.0 - self._ema_alpha).add_(
                self._processed_actions, alpha=self._ema_alpha
            )
        self._processed_actions = self._ema_state

    def apply_actions(self) -> None:
        if not self._delay_enabled or self._previous_target is None:
            super().apply_actions()
            return
        # Sub-step k of the control step applies the previous target for the
        # environments whose delay exceeds k, the new one for the rest.
        stale = (self._delay_steps > self._substep).unsqueeze(-1)
        target = torch.where(stale, self._previous_target, self._processed_actions)
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)
        self._substep += 1

    def _offset_tensor(self) -> torch.Tensor:
        if isinstance(self._offset, torch.Tensor):
            return self._offset
        return torch.full(
            (self.num_envs, self.action_dim), float(self._offset), device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            reset_ids: Sequence[int] | slice = slice(None)
        else:
            reset_ids = env_ids
        if self._ema_state is not None:
            # Restart the filter at the nominal pose: offset is the default
            # joint position when use_default_offset is set.
            if isinstance(self._offset, torch.Tensor):
                self._ema_state[reset_ids] = self._offset[reset_ids]
            else:
                self._ema_state[reset_ids] = float(self._offset)
        if self._delay_enabled:
            self._sample_delays(reset_ids)
            if self._previous_target is not None:
                self._previous_target[reset_ids] = self._offset_tensor()[reset_ids]


@configclass
class EMAJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`EMAJointPositionAction`.

    ``ema_alpha=1.0`` (default) is the identity: existing arms are unchanged.
    A campaign enables the filter with e.g.
    ``env.actions.joint_pos.ema_alpha=0.65``.
    """

    class_type: type[EMAJointPositionAction] | str = (
        "isaaclab_imitation.tasks.manager_based.imitation.mdp.actions:EMAJointPositionAction"
    )

    ema_alpha: float = 1.0

    # Actuation delay randomization, in physics sub-steps of the control step
    # (decimation 4 at 200 Hz: one sub-step is 5 ms). Each environment draws
    # its delay uniformly from [min, max] at reset; for the first `delay`
    # sub-steps of every control step the PD loop still tracks the previous
    # target, as the robot's 500 Hz writer does while the new 50 Hz target is
    # computed and put on the wire (3-5 ms measured on 2026-09-13). (0, 0)
    # is the identity: the new target applies from the first sub-step, as
    # every arm before 2026-09-13 trained.
    delay_substeps_min: int = 0
    delay_substeps_max: int = 0
