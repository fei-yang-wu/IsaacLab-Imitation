"""Mixing two (or three) environments' latent commands, for composability probes.

A latent tracker is driven by a frozen sampler that produces one code per
environment each control step. ``LatentBlendSampler`` wraps that sampler and,
for every :class:`BlendSpec`, rewrites the *target* environment's code:

* mix (two environments): ``z_t + a * (z_s - z_t)`` -- convex for ``a`` in
  ``[0, 1]``, extrapolation outside it;
* additive (three environments): ``z_t + a * (z_s - z_m)`` -- the *source*
  skill's offset from a *minus* baseline added to the target.

``a`` follows :class:`BlendSchedule` (0 before ``start_step``, linear ramp to
``final_alpha`` over ``ramp_steps``, then held; ``ramp_steps=0`` is a hard
switch, ``start_step=0`` with ``ramp_steps=0`` is a held mix from the first
step). Only the code columns are mixed; the phase columns (constant at hold 1)
stay as the base sampler wrote them. Every environment named by a spec gets a
:class:`BlendTrace` of what its robot did, read from the observation the
sampler is handed, so the composition metrics need no second pass.

The question it serves (2026-09-02): does a phi that is affine in ``z`` give a
tracker whose behaviour under composed codes is itself a plausible motion,
and does the concat phi differ?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

G1_JOINT_COUNT = 29


@dataclass(frozen=True)
class BlendSchedule:
    """``alpha(step)``: 0 before ``start_step``, linear to ``final_alpha`` over
    ``ramp_steps``, then held. ``final_alpha`` may leave ``[0, 1]`` on purpose
    (extrapolation probes)."""

    start_step: int
    ramp_steps: int
    final_alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be >= 0.")
        if self.ramp_steps < 0:
            raise ValueError("ramp_steps must be >= 0.")
        if self.final_alpha != self.final_alpha:
            raise ValueError("final_alpha must be a number.")

    def alpha(self, step: int) -> float:
        if step < self.start_step:
            return 0.0
        if self.ramp_steps == 0:
            return float(self.final_alpha)
        fraction = min(1.0, (step - self.start_step) / float(self.ramp_steps))
        return float(self.final_alpha) * fraction


@dataclass(frozen=True)
class BlendSpec:
    """Which environment is rewritten (``target``) from which (``source``), and
    for the additive form, which baseline (``minus``) is subtracted."""

    target: int
    source: int
    minus: int | None = None

    def __post_init__(self) -> None:
        envs = [self.target, self.source] + (
            [self.minus] if self.minus is not None else []
        )
        if any(e < 0 for e in envs):
            raise ValueError("environment indices must be >= 0.")
        if len(set(envs)) != len(envs):
            raise ValueError("target, source and minus must be distinct environments.")

    @property
    def envs(self) -> tuple[int, ...]:
        return (
            (self.target, self.source)
            if self.minus is None
            else (self.target, self.source, self.minus)
        )

    @property
    def needed_envs(self) -> int:
        return max(self.envs) + 1


def pair_specs(num_envs: int) -> list[BlendSpec]:
    """Environments ``(2i, 2i+1)`` as ``(target, source)`` pairs."""
    if num_envs % 2:
        raise ValueError("pairs layout needs an even number of environments.")
    return [BlendSpec(2 * i, 2 * i + 1) for i in range(num_envs // 2)]


def triple_specs(num_envs: int) -> list[BlendSpec]:
    """Environments ``(3i, 3i+1, 3i+2)`` as ``(target, source, minus)``."""
    if num_envs % 3:
        raise ValueError("triples layout needs a multiple of three environments.")
    return [BlendSpec(3 * i, 3 * i + 1, 3 * i + 2) for i in range(num_envs // 3)]


@dataclass
class BlendTrace:
    """Per-step record for one environment.

    ``alpha`` and ``code_distance`` are filled for targets only (distance is
    ``||z_target - z_source||`` before mixing). ``root_speed`` is the planar
    base velocity, ``upright`` is ``-projected_gravity_z`` (1 standing, ~0
    lying), ``action_delta`` the norm of the action step, ``joint_pos`` the
    29 ``joint_pos_rel`` values of the newest frame.
    """

    steps: list[int] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)
    code_distance: list[float] = field(default_factory=list)
    root_speed: list[float] = field(default_factory=list)
    action_delta: list[float] = field(default_factory=list)
    upright: list[float] = field(default_factory=list)
    joint_pos: list[list[float]] = field(default_factory=list)

    # Backward-compatible names used by the first probe's analysis.
    @property
    def target_root_speed(self) -> list[float]:
        return self.root_speed

    @property
    def target_action_delta(self) -> list[float]:
        return self.action_delta

    @property
    def target_upright(self) -> list[float]:
        return self.upright

    def window(self, values: list[float], lo: int, hi: int) -> float | None:
        """Mean of ``values`` over steps in ``[lo, hi)``, ignoring NaNs."""
        picked = [v for s, v in zip(self.steps, values) if lo <= s < hi and v == v]
        return sum(picked) / len(picked) if picked else None

    def summary(self) -> dict[str, Any]:
        finite_up = [v for v in self.upright if v == v]
        return {
            "steps": len(self.steps),
            "alpha_first_nonzero_step": next(
                (s for s, a in zip(self.steps, self.alpha) if a != 0.0), None
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
            "upright_min": min(finite_up) if finite_up else None,
            "fallen_steps": sum(1 for v in finite_up if v < 0.5),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": list(self.steps),
            "alpha": list(self.alpha),
            "code_distance": list(self.code_distance),
            "root_speed": list(self.root_speed),
            "action_delta": list(self.action_delta),
            "upright": list(self.upright),
            "joint_pos": [list(row) for row in self.joint_pos],
        }


def _newest_frame(values: torch.Tensor, width: int) -> torch.Tensor:
    """A history-stacked observation term ends with the newest frame."""
    if values.shape[-1] % width == 0:
        return values[..., -width:]
    return values


class LatentBlendSampler:
    """Wrap a frozen command sampler and rewrite target environments' codes.

    Every attribute not defined here is forwarded to the wrapped sampler, so
    checkpoint provenance, ``skill_encoder`` and the finetune hooks keep
    working. ``sample_for_step`` is the only call that changes.
    """

    def __init__(
        self,
        base: Any,
        *,
        schedule: BlendSchedule,
        code_dim: int,
        specs: Sequence[BlendSpec] | None = None,
        target_env: int | None = None,
        source_env: int | None = None,
        record_joints: bool = True,
    ) -> None:
        if specs is None:
            if target_env is None or source_env is None:
                raise ValueError("pass specs, or target_env and source_env.")
            specs = [BlendSpec(int(target_env), int(source_env))]
        if not specs:
            raise ValueError("at least one BlendSpec is needed.")
        if code_dim <= 0:
            raise ValueError("code_dim must be positive.")
        targets = [s.target for s in specs]
        if len(set(targets)) != len(targets):
            raise ValueError("a target environment may appear in one spec only.")
        self._base = base
        self.specs = list(specs)
        self.schedule = schedule
        self.code_dim = int(code_dim)
        self.record_joints = bool(record_joints)
        self.step = 0
        self.needed_envs = max(s.needed_envs for s in self.specs)
        self.traces: dict[int, BlendTrace] = {
            env: BlendTrace() for spec in self.specs for env in spec.envs
        }
        self._last_action: dict[int, torch.Tensor] = {}

    # Backward-compatible single-pair view.
    @property
    def target_env(self) -> int:
        return self.specs[0].target

    @property
    def source_env(self) -> int:
        return self.specs[0].source

    @property
    def trace(self) -> BlendTrace:
        return self.traces[self.specs[0].target]

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails, so our own fields stay ours.
        return getattr(self._base, name)

    def _record_motion(self, td: Any) -> None:
        getter = getattr(td, "get", None)
        vel = grav = act = joints = None
        if callable(getter):
            # The G1 v2 policy group carries no base_lin_vel; the critic group
            # does, and both ride in the same observation tensordict.
            vel = getter(("policy", "base_lin_vel"), None)
            if vel is None:
                vel = getter(("critic", "base_lin_vel"), None)
            grav = getter(("policy", "projected_gravity"), None)
            act = getter(("policy", "last_action"), None)
            if self.record_joints:
                joints = getter(("policy", "joint_pos_rel"), None)

        def _rows(value: Any, width: int | None = None) -> torch.Tensor | None:
            if value is None:
                return None
            tensor = torch.as_tensor(value)
            tensor = tensor.reshape(-1, tensor.shape[-1]).detach().float().cpu()
            return _newest_frame(tensor, width) if width else tensor

        vel_rows = _rows(vel, 3)
        grav_rows = _rows(grav, 3)
        act_rows = _rows(act)
        joint_rows = _rows(joints, G1_JOINT_COUNT)
        for env, trace in self.traces.items():
            speed = upright = delta = float("nan")
            if vel_rows is not None and vel_rows.shape[0] > env:
                speed = float(vel_rows[env, :2].norm().item())
            if grav_rows is not None and grav_rows.shape[0] > env:
                upright = float(-grav_rows[env, -1].item())
            if act_rows is not None and act_rows.shape[0] > env:
                current = act_rows[env]
                previous = self._last_action.get(env)
                if previous is not None:
                    delta = float((current - previous).norm().item())
                self._last_action[env] = current
            trace.root_speed.append(speed)
            trace.upright.append(upright)
            trace.action_delta.append(delta)
            if self.record_joints:
                if joint_rows is not None and joint_rows.shape[0] > env:
                    trace.joint_pos.append(joint_rows[env].tolist())
                else:
                    trace.joint_pos.append([])

    @torch.no_grad()
    def sample_for_step(self, td: Any, *, device: Any, dtype: Any) -> torch.Tensor:
        latents = self._base.sample_for_step(td, device=device, dtype=dtype)
        batch = int(latents.shape[0])
        if batch < self.needed_envs:
            raise ValueError(
                f"latent batch of {batch} cannot serve blend specs that need "
                f"{self.needed_envs} environments."
            )
        alpha = self.schedule.alpha(self.step)
        mixed = latents.clone() if alpha != 0.0 else latents
        for spec in self.specs:
            z_t = latents[spec.target, : self.code_dim]
            z_s = latents[spec.source, : self.code_dim]
            trace = self.traces[spec.target]
            trace.alpha.append(alpha)
            trace.code_distance.append(float((z_t - z_s).norm().item()))
            if alpha != 0.0:
                base = (
                    z_t if spec.minus is None else latents[spec.minus, : self.code_dim]
                )
                mixed[spec.target, : self.code_dim] = z_t + alpha * (z_s - base)
        for trace in self.traces.values():
            trace.steps.append(self.step)
        self._record_motion(td)
        self.step += 1
        return mixed

    def summary(self) -> dict[str, Any]:
        return {
            "schedule": {
                "start_step": self.schedule.start_step,
                "ramp_steps": self.schedule.ramp_steps,
                "final_alpha": self.schedule.final_alpha,
            },
            "specs": [
                {"target": s.target, "source": s.source, "minus": s.minus}
                for s in self.specs
            ],
            "targets": {
                str(s.target): self.traces[s.target].summary() for s in self.specs
            },
        }

    def traces_as_dict(self) -> dict[str, dict[str, Any]]:
        return {str(env): trace.as_dict() for env, trace in self.traces.items()}
