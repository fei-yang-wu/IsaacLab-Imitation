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
    """Per-step record: alpha, the distance between the two source codes, and
    what the target robot did (planar root speed and the size of its action
    step), read from the policy observation the sampler is handed."""

    steps: list[int] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)
    code_distance: list[float] = field(default_factory=list)
    target_root_speed: list[float] = field(default_factory=list)
    target_action_delta: list[float] = field(default_factory=list)

    def window(self, values: list[float], lo: int, hi: int) -> float | None:
        """Mean of ``values`` over steps in ``[lo, hi)``, ignoring NaNs."""
        picked = [v for s, v in zip(self.steps, values) if lo <= s < hi and v == v]
        return sum(picked) / len(picked) if picked else None

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
            "code_distance_max": (
                max(self.code_distance) if self.code_distance else None
            ),
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
        self._last_action: torch.Tensor | None = None

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so our own fields stay ours.
        return getattr(self._base, name)

    def _record_target_motion(self, td: Any) -> None:
        """Root speed and the action step of the target robot, when the
        observation carries them (the G1 v2 policy group does)."""
        speed = float("nan")
        delta = float("nan")
        getter = getattr(td, "get", None)
        if callable(getter):
            vel = getter(("policy", "base_lin_vel"), None)
            if vel is not None:
                vel = torch.as_tensor(vel).reshape(-1, vel.shape[-1])
                if vel.shape[0] > self.target_env:
                    speed = float(vel[self.target_env, :2].norm().item())
            act = getter(("policy", "last_action"), None)
            if act is not None:
                act = torch.as_tensor(act).reshape(-1, act.shape[-1])
                if act.shape[0] > self.target_env:
                    current = act[self.target_env].detach().float().cpu()
                    if self._last_action is not None:
                        delta = float((current - self._last_action).norm().item())
                    self._last_action = current
        self.trace.target_root_speed.append(speed)
        self.trace.target_action_delta.append(delta)

    @torch.no_grad()
    def sample_for_step(self, td: Any, *, device: Any, dtype: Any) -> torch.Tensor:
        latents = self._base.sample_for_step(td, device=device, dtype=dtype)
        batch = int(latents.shape[0])
        needed = max(self.target_env, self.source_env) + 1
        if batch < needed:
            raise ValueError(
                f"latent batch of {batch} cannot blend env {self.source_env} into "
                f"env {self.target_env}; run at least {needed} environments."
            )
        alpha = self.schedule.alpha(self.step)
        z_t = latents[self.target_env, : self.code_dim]
        z_s = latents[self.source_env, : self.code_dim]
        self.trace.steps.append(self.step)
        self.trace.alpha.append(alpha)
        self.trace.code_distance.append(float((z_t - z_s).norm().item()))
        self._record_target_motion(td)
        if alpha > 0.0:
            mixed = latents.clone()
            mixed[self.target_env, : self.code_dim] = (1.0 - alpha) * z_t + alpha * z_s
            latents = mixed
        self.step += 1
        return latents
