"""Concrete Isaac latent-command sampler driven by a GR00T action head.

Extends the frozen oracle sampler so hold length, phase channels, and
per-environment renewal stay byte-for-byte the machinery the low-level
policy was trained against; only the production of `z` is replaced.

The oracle latent is still computed each renewal (the base class needs the
expert macro batch anyway) and returned as a diagnostic, so a run can report
how far the planner's published latent sits from the latent the frozen
encoder would have produced — without that oracle ever reaching the policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from rlopt.agent.hl_skill_diffsr import FrozenHighLevelSkillCommandSampler

from imitation_experiments.planner.gr00t_isaac_sampler import Gr00tSkillCommandSampler


class Gr00tLatentCommandSampler(Gr00tSkillCommandSampler, FrozenHighLevelSkillCommandSampler):
    def __init__(
        self,
        *,
        causal_observation_fn: Callable[..., Any],
        state_history_steps: int,
        gr00t_checkpoint: str | Path,
        goal_features_path: str | Path,
        goal_name: str,
        num_envs: int,
        consumption: str = "open_loop",
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self._causal_observation_fn = causal_observation_fn
        self._causal_history_steps = int(state_history_steps)
        fsq_half = getattr(self.skill_encoder, "_half_levels", None)
        self.gr00t_provenance = self.configure_gr00t(
            checkpoint_path=gr00t_checkpoint,
            goal_features_path=goal_features_path,
            goal_name=goal_name,
            num_envs=int(num_envs),
            consumption=consumption,
            fsq_half_levels=None if fsq_half is None else fsq_half.detach(),
            device=self.device,
        )
        self.oracle_cosine: list[float] = []

    def _causal_planner_state(self, env_ids: Tensor) -> Tensor:
        batch = self._causal_observation_fn(
            env_ids=env_ids, history_steps=self._causal_history_steps
        )
        history = batch.get(("planner", "state_history"))
        return history.reshape(int(env_ids.numel()), -1).to(
            device=self.device, dtype=torch.float32
        )

    def _encode_current_macro_batch(
        self, env_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        oracle_z, state, future_window, target, _ = super()._encode_current_macro_batch(
            env_ids
        )
        planner_state = self._causal_planner_state(env_ids)
        z = self.gr00t_z(planner_state, env_ids).to(oracle_z.dtype)
        if z.shape != oracle_z.shape:
            msg = (
                f"GR00T head published {tuple(z.shape)} but the command "
                f"contract expects {tuple(oracle_z.shape)}."
            )
            raise ValueError(msg)
        with torch.no_grad():
            cosine = torch.nn.functional.cosine_similarity(z, oracle_z, dim=-1)
            self.oracle_cosine.append(float(cosine.mean()))
        return z, state, future_window, target, z

    def gr00t_report(self) -> dict[str, Any]:
        record = dict(self.gr00t_provenance)
        record.update(self.gr00t_stats())
        if self.oracle_cosine:
            values = torch.tensor(self.oracle_cosine)
            record["published_vs_oracle_z_cosine_mean"] = float(values.mean())
        return record


__all__ = ["Gr00tLatentCommandSampler"]
