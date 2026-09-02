"""Convex blending of two environments' latent commands, for composability probes.

A latent tracker is driven by a frozen sampler that produces one code per
environment each control step. ``LatentBlendSampler`` wraps that sampler: the
*target* environment receives ``(1 - a) * z_target + a * z_source`` where the
*source* environment tracks a different reference clip and ``a`` follows a
linear ramp in control steps. Only the code columns are mixed; the phase
columns (constant at hold 1) are left as the base sampler wrote them.

The question it serves (2026-09-02): does a phi that is affine in ``z`` give a
tracker whose behaviour under a convex combination of two skills is itself a
plausible motion, and does the concat phi differ?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class BlendSchedule:
    """``alpha(step)``: 0 before ``start_step``, linear to ``final_alpha`` over
    ``ramp_steps``, then held."""

    start_step: int
    ramp_steps: int
    final_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be >= 0.")
        if self.ramp_steps < 0:
            raise ValueError("ramp_steps must be >= 0.")
        if not 0.0 <= self.final_alpha <= 1.0:
            raise ValueError("final_alpha must be in [0, 1].")

    def alpha(self, step: int) -> float:
        if step < self.start_step:
            return 0.0
        if self.ramp_steps == 0:
            return float(self.final_alpha)
        fraction = min(1.0, (step - self.start_step) / float(self.ramp_steps))
        return float(self.final_alpha) * fraction


@dataclass
class BlendTrace:
    """Per-step record: alpha and the distance between the two source codes."""

    steps: list[int] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)
    code_distance: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "steps": len(self.steps),
            "alpha_first_nonzero_step": next(
                (s for s, a in zip(self.steps, self.alpha) if a > 0.0), None
            ),
            "alpha_final": self.alpha[-1] if self.alpha else None,
            "code_distance_mean": (
                sum(self.code_distance) / len(self.code_distance)
                if self.code_distance
                else None
            ),
            "code_distance_max": max(self.code_distance)
            if self.code_distance
            else None,
        }


class LatentBlendSampler:
    """Wrap a frozen command sampler and blend one environment's code into another.

    Every attribute not defined here is forwarded to the wrapped sampler, so
    checkpoint provenance, ``skill_encoder`` and the finetune hooks keep
    working. ``sample_for_step`` is the only call that changes.
    """

    def __init__(
        self,
        base: Any,
        *,
        target_env: int,
        source_env: int,
        schedule: BlendSchedule,
        code_dim: int,
    ) -> None:
        if target_env == source_env:
            raise ValueError("target_env and source_env must differ.")
        if target_env < 0 or source_env < 0:
            raise ValueError("environment indices must be >= 0.")
        if code_dim <= 0:
            raise ValueError("code_dim must be positive.")
        self._base = base
        self.target_env = int(target_env)
        self.source_env = int(source_env)
        self.schedule = schedule
        self.code_dim = int(code_dim)
        self.step = 0
        self.trace = BlendTrace()

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so our own fields stay ours.
        return getattr(self._base, name)

    @torch.no_grad()
    def sample_for_step(self, td: Any, *, device: Any, dtype: Any) -> torch.Tensor:
        latents = self._base.sample_for_step(td, device=device, dtype=dtype)
        batch = int(latents.shape[0])
        if batch <= max(self.target_env, self.source_env):
            raise ValueError(
                f"latent batch of {batch} cannot blend env {self.source_env} into "
                f"env {self.target_env}; run at least {max(self.target_env, self.source_env) + 1} environments."
            )
        alpha = self.schedule.alpha(self.step)
        z_t = latents[self.target_env, : self.code_dim]
        z_s = latents[self.source_env, : self.code_dim]
        self.trace.steps.append(self.step)
        self.trace.alpha.append(alpha)
        self.trace.code_distance.append(float((z_t - z_s).norm().item()))
        if alpha > 0.0:
            mixed = latents.clone()
            mixed[self.target_env, : self.code_dim] = (1.0 - alpha) * z_t + alpha * z_s
            latents = mixed
        self.step += 1
        return latents
