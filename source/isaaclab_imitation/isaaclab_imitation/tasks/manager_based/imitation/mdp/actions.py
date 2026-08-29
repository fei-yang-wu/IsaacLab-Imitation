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
from isaaclab.utils import configclass

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

    def process_actions(self, actions: torch.Tensor) -> None:
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

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self._ema_state is not None:
            if env_ids is None:
                reset_ids: Sequence[int] | slice = slice(None)
            else:
                reset_ids = env_ids
            # Restart the filter at the nominal pose: offset is the default
            # joint position when use_default_offset is set.
            if isinstance(self._offset, torch.Tensor):
                self._ema_state[reset_ids] = self._offset[reset_ids]
            else:
                self._ema_state[reset_ids] = float(self._offset)


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
