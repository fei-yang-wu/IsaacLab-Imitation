"""Episode evidence for reducing global object assistance."""

from __future__ import annotations

import torch


class ObjectAssistanceGate:
    """Count only episodes that started and ended at the current stage."""

    def __init__(
        self,
        num_envs,
        device,
        stage_count,
        minimum_episodes=256,
        success_threshold=0.8,
        minimum_steps=2400,
    ):
        if stage_count < 1 or minimum_episodes < 1 or minimum_steps < 0:
            raise ValueError("Invalid assistance gate counts.")
        if not 0 <= success_threshold <= 1:
            raise ValueError("Success threshold must lie in [0, 1].")
        self.stage = 0
        self.stage_count = stage_count
        self.minimum_episodes = minimum_episodes
        self.success_threshold = success_threshold
        self.minimum_steps = minimum_steps
        self.stage_started_at = 0
        self.episodes = 0
        self.successes = 0
        self.last_rate = 0.0
        self.episode_stage = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )

    def reset(self, env_ids, success, episode_length, step):
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(
                len(self.episode_stage), device=self.episode_stage.device
            )
            if isinstance(env_ids, slice):
                ids = ids[env_ids]
        else:
            ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.episode_stage.device
            )
        eligible = (self.episode_stage[ids] == self.stage) & (episode_length[ids] > 0)
        self.episodes += int(eligible.sum())
        self.successes += int((success[ids].bool() & eligible).sum())
        if (
            self.episodes >= self.minimum_episodes
            and step - self.stage_started_at >= self.minimum_steps
        ):
            self.last_rate = self.successes / self.episodes
            if (
                self.last_rate >= self.success_threshold
                and self.stage < self.stage_count - 1
            ):
                self.stage += 1
                self.stage_started_at = step
            self.episodes = 0
            self.successes = 0
        self.episode_stage[ids] = self.stage
        return self.stage

    def state_dict(self):
        return {
            name: getattr(self, name)
            for name in (
                "stage",
                "stage_count",
                "minimum_episodes",
                "success_threshold",
                "minimum_steps",
                "stage_started_at",
                "episodes",
                "successes",
                "last_rate",
            )
        }

    def load_state_dict(self, state):
        for name in (
            "stage_count",
            "minimum_episodes",
            "success_threshold",
            "minimum_steps",
        ):
            if state[name] != getattr(self, name):
                raise ValueError(f"Assistance gate configuration changed: {name}.")
        if not 0 <= state["stage"] < self.stage_count:
            raise ValueError("Invalid saved assistance stage.")
        if (
            not 0 <= state["successes"] <= state["episodes"]
            or state["stage_started_at"] < 0
        ):
            raise ValueError("Invalid saved assistance evidence.")
        for name in ("stage", "stage_started_at", "episodes", "successes", "last_rate"):
            setattr(self, name, state[name])
        # The simulator resumes with new episodes, not the saved physical state.
        self.episode_stage.fill_(-1)
