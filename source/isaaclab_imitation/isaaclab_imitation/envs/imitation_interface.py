# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The capability surface an RLOpt agent sees of an imitation environment.

One small object instead of a dozen environment methods discovered by name.
:class:`~isaaclab_imitation.envs.imitation_rl_env_v2.ImitationRLEnv` exposes it
as ``env.imitation_interface``; RLOpt resolves it once
(``rlopt.env_interface.resolve_imitation_interface``) and reaches nothing else
on the environment. That keeps the agent side ignorant of how commands and
expert data are actually produced -- publishing an actor command is one call,
whether the actor channel is a skill latent or a planner packet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

    from .imitation_rl_env_v2 import ImitationRLEnv


class ImitationEnvInterface:
    """Expert data and actor-command publication, for the learning agent."""

    def __init__(self, env: ImitationRLEnv) -> None:
        self._env = env

    def __repr__(self) -> str:
        return f"ImitationEnvInterface({type(self._env).__name__})"

    # -- actor command ------------------------------------------------------

    @property
    def actor_command_dim(self) -> int:
        """Width of the actor command the agent publishes."""
        return int(self._env.actor_command.command.shape[-1])

    def publish_actor_command(
        self, command: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Publish the agent-produced actor command.

        Valid only where the actor channel is agent-published (a skill latent);
        an explicit channel is derived from the reference and a chunk channel is
        published by a planner, so both refuse.
        """
        actor = self._env.actor_command
        setter = getattr(actor, "set", None)
        if setter is None:
            raise RuntimeError(
                "The configured actor command channel "
                f"({type(actor).__name__}) is not agent-published."
            )
        setter(command, env_ids=env_ids)

    # -- expert data --------------------------------------------------------

    def sample_expert_batch(
        self, batch_size: int, required_keys: Sequence[Any]
    ) -> TensorDict | None:
        return self._env.sample_expert_batch(
            batch_size=batch_size, required_keys=required_keys
        )

    def expert_macro_frame_stride(self) -> int:
        return self._env.expert_macro_frame_stride()

    def sample_expert_macro_transition_batch(
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
        trajectory_ranks: Sequence[int] | torch.Tensor | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self._env.sample_expert_macro_transition_batch(
            batch_size=batch_size,
            horizon_steps=horizon_steps,
            split=split,
            eval_fraction=eval_fraction,
            split_seed=split_seed,
            trajectory_ranks=trajectory_ranks,
            state_history_steps=state_history_steps,
        )

    def current_expert_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self._env.current_expert_macro_transition_batch(
            horizon_steps=horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )

    def current_achieved_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self._env.current_achieved_macro_transition_batch(
            horizon_steps=horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )

    def expert_macro_feature_slices(self, horizon_steps: int) -> dict[str, Any]:
        return self._env.expert_macro_feature_slices(horizon_steps=horizon_steps)

    def expert_trajectory_motion_names(self) -> list[str]:
        return self._env.expert_trajectory_motion_names()

    def offline_dataset_mapper_params(self) -> dict[str, Any]:
        return self._env.get_offline_dataset_mapper_params()


__all__ = ["ImitationEnvInterface"]
