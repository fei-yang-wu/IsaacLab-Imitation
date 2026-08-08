"""Execution rules for overlapping macro-latent forecasts."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F


class OverlappingLatentEnsembler:
    """Fuse forecasts of the same current H10 latent from successive plans."""

    MODES = ("first", "exponential", "clipped_gated")

    def __init__(
        self,
        *,
        num_envs: int,
        token_count: int,
        token_width: int,
        hold_steps: int,
        mode: str,
        decay: float,
        reference_std: torch.Tensor,
        clip_std: float = 1.0,
        gate_distance: float = 2.0,
        gate_cosine: float = 0.5,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}.")
        if token_count <= 0 or token_width <= 0 or hold_steps <= 0:
            raise ValueError(
                "token_count, token_width, and hold_steps must be positive."
            )
        if decay < 0 or clip_std <= 0 or gate_distance <= 0:
            raise ValueError(
                "decay must be non-negative and clip/gate scales positive."
            )
        if not -1.0 <= gate_cosine <= 1.0:
            raise ValueError("gate_cosine must be in [-1, 1].")
        if tuple(reference_std.shape) != (int(token_width),):
            raise ValueError(
                f"reference_std must be [{token_width}], got {tuple(reference_std.shape)}."
            )
        self.num_envs = int(num_envs)
        self.token_count = int(token_count)
        self.token_width = int(token_width)
        self.hold_steps = int(hold_steps)
        self.mode = str(mode)
        self.decay = float(decay)
        self.clip_std = float(clip_std)
        self.gate_distance = float(gate_distance)
        self.gate_cosine = float(gate_cosine)
        self.device = torch.device(device)
        self.dtype = dtype
        self.reference_std = reference_std.to(self.device, self.dtype).clamp_min(1.0e-6)
        self._chunks = torch.zeros(
            self.num_envs,
            self.token_count,
            self.token_count,
            self.token_width,
            device=self.device,
            dtype=self.dtype,
        )
        self._valid = torch.zeros(
            self.num_envs,
            self.token_count,
            device=self.device,
            dtype=torch.bool,
        )
        self._last_episode_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.publications = 0
        self.history_resets = 0
        self.rejected_candidates = 0
        self.candidate_histogram = [0 for _ in range(self.token_count + 1)]

    def update(
        self,
        *,
        env_ids: torch.Tensor,
        prediction: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> torch.Tensor:
        env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        prediction = prediction.to(device=self.device, dtype=self.dtype)
        expected = (int(env_ids.numel()), self.token_count * self.token_width)
        if tuple(prediction.shape) != expected:
            raise ValueError(
                f"Expected H3 prediction {expected}, got {tuple(prediction.shape)}."
            )
        episode_steps = episode_steps.to(device=self.device, dtype=torch.long).reshape(
            -1
        )
        if tuple(episode_steps.shape) != (int(env_ids.numel()),):
            raise ValueError("episode_steps must be row-aligned with env_ids.")
        chunks = prediction.reshape(-1, self.token_count, self.token_width)
        previous_steps = self._last_episode_step.index_select(0, env_ids)
        discontinuity = (episode_steps == 0) | (
            previous_steps + self.hold_steps != episode_steps
        )
        if bool(discontinuity.any()):
            reset_ids = env_ids[discontinuity]
            self._valid.index_fill_(0, reset_ids, False)
            self.history_resets += int(reset_ids.numel())

        if self.token_count > 1:
            self._chunks[env_ids, 1:] = self._chunks[env_ids, :-1].clone()
            self._valid[env_ids, 1:] = self._valid[env_ids, :-1].clone()
        self._chunks[env_ids, 0] = chunks
        self._valid[env_ids, 0] = True
        self._last_episode_step.index_copy_(0, env_ids, episode_steps)

        candidates = torch.stack(
            [self._chunks[env_ids, age, age] for age in range(self.token_count)],
            dim=1,
        )
        valid = self._valid.index_select(0, env_ids).clone()
        fresh = candidates[:, 0]
        if self.mode == "first":
            valid[:, 1:] = False
        elif self.mode == "clipped_gated" and self.token_count > 1:
            delta = candidates[:, 1:] - fresh[:, None]
            normalized = delta / self.reference_std.reshape(1, 1, -1)
            distance = normalized.square().mean(dim=-1).sqrt()
            cosine = F.cosine_similarity(candidates[:, 1:], fresh[:, None], dim=-1)
            coherent = (distance <= self.gate_distance) & (cosine >= self.gate_cosine)
            rejected = valid[:, 1:] & ~coherent
            self.rejected_candidates += int(rejected.sum().item())
            valid[:, 1:] &= coherent
            clipped = delta.clamp(
                min=-self.clip_std * self.reference_std.reshape(1, 1, -1),
                max=self.clip_std * self.reference_std.reshape(1, 1, -1),
            )
            candidates[:, 1:] = fresh[:, None] + clipped

        ages = torch.arange(
            self.token_count, device=self.device, dtype=self.dtype
        ).reshape(1, -1)
        weights = torch.exp(-self.decay * ages) * valid.to(self.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        fused = (candidates * weights[:, :, None]).sum(dim=1)
        counts = valid.sum(dim=1)
        for count in range(1, self.token_count + 1):
            self.candidate_histogram[count] += int((counts == count).sum().item())
        self.publications += int(env_ids.numel())
        return fused

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "token_count": self.token_count,
            "token_width": self.token_width,
            "hold_steps": self.hold_steps,
            "decay": self.decay,
            "normalized_weights_full_history": [
                float(value)
                for value in (
                    torch.exp(
                        -self.decay
                        * torch.arange(self.token_count, dtype=torch.float64)
                    )
                    / torch.exp(
                        -self.decay
                        * torch.arange(self.token_count, dtype=torch.float64)
                    ).sum()
                ).tolist()
            ],
            "clip_std": self.clip_std,
            "gate_distance": self.gate_distance,
            "gate_cosine": self.gate_cosine,
            "publications": self.publications,
            "history_resets": self.history_resets,
            "rejected_candidates": self.rejected_candidates,
            "candidate_histogram": {
                str(index): value
                for index, value in enumerate(self.candidate_histogram)
                if index > 0
            },
        }


def install_latent_receding_horizon(
    command_sampler: Any,
    *,
    env: Any,
    token_count: int,
    token_width: int,
    mode: str,
    decay: float,
    clip_std: float,
    gate_distance: float,
    gate_cosine: float,
) -> Callable[[], dict[str, Any]]:
    """Reduce the sampler's ordered H3 prediction to one H10 z command."""
    generator = getattr(command_sampler, "generator", None)
    if not isinstance(generator, torch.nn.Module):
        raise ValueError("Command sampler has no planner generator.")
    target_std = getattr(generator, "target_std", None)
    if not isinstance(target_std, torch.Tensor) or int(target_std.numel()) != (
        int(token_count) * int(token_width)
    ):
        raise ValueError("H3 planner target normalization does not match its tokens.")
    reference_std = target_std.reshape(int(token_count), int(token_width))[0]
    ensembler = OverlappingLatentEnsembler(
        num_envs=int(getattr(env, "num_envs")),
        token_count=int(token_count),
        token_width=int(token_width),
        hold_steps=int(getattr(command_sampler, "config").horizon_steps),
        mode=str(mode),
        decay=float(decay),
        reference_std=reference_std,
        clip_std=float(clip_std),
        gate_distance=float(gate_distance),
        gate_cosine=float(gate_cosine),
        device=next(generator.parameters()).device,
        dtype=next(generator.parameters()).dtype,
    )
    original = command_sampler._encode_current_macro_batch

    @torch.no_grad()
    def _encode_and_reduce(env_ids: torch.Tensor):
        prediction, state, future_window, target, _initial = original(env_ids)
        episode_steps = getattr(env, "episode_length_buf").index_select(
            0, env_ids.to(device=getattr(env, "episode_length_buf").device)
        )
        fused = ensembler.update(
            env_ids=env_ids,
            prediction=prediction,
            episode_steps=episode_steps,
        )
        return fused, state, future_window, target, fused

    command_sampler._encode_current_macro_batch = _encode_and_reduce
    return ensembler.stats
