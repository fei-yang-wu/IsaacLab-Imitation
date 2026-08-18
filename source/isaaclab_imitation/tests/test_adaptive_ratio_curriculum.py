"""Linear ramp of the SONIC adaptive-reset uniform fraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab_imitation.tasks.manager_based.imitation.mdp.commands.reference import (
    ReferenceCommandTerm,
)


class _Sampler:
    def __init__(self) -> None:
        self.uniform_sampling_rate = 0.8


def _term(step: int, num_envs: int, *, final, ramp, start=0.8):
    term = object.__new__(ReferenceCommandTerm)
    term.cfg = SimpleNamespace(
        selection=SimpleNamespace(
            adaptive_uniform_ratio=start,
            adaptive_uniform_ratio_final=final,
            adaptive_ratio_ramp_frames=ramp,
        )
    )
    term._adaptive_failure_reset_sampler = _Sampler()
    # `num_envs` is a read-only property on the real term, so the fake env
    # supplies it exactly as the term reads it.
    term._env = SimpleNamespace(common_step_counter=step, num_envs=num_envs)
    return term


def test_ratio_ramps_linearly_and_holds_at_the_end() -> None:
    # 0.8 -> 0.2 uniform, i.e. adaptive share 20% -> 80%, over 1000 frames.
    # frames = step * num_envs; 10 envs so step 25 = 250/1000 = 25% of the ramp.
    for step, expected in ((0, 0.8), (25, 0.65), (50, 0.5), (100, 0.2), (500, 0.2)):
        term = _term(step, 10, final=0.2, ramp=1000)
        term._advance_adaptive_ratio_curriculum()
        assert term._adaptive_failure_reset_sampler.uniform_sampling_rate == (
            pytest.approx(expected)
        )


def test_disabled_when_no_final_ratio_is_configured() -> None:
    term = _term(9999, 100, final=None, ramp=1000)
    term._advance_adaptive_ratio_curriculum()
    # Untouched: the static ratio the sampler was constructed with.
    assert term._adaptive_failure_reset_sampler.uniform_sampling_rate == 0.8


def test_noop_without_an_adaptive_sampler() -> None:
    term = _term(10, 10, final=0.2, ramp=1000)
    term._adaptive_failure_reset_sampler = None
    term._advance_adaptive_ratio_curriculum()  # must not raise
