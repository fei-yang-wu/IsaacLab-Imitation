"""Contracts for the latent blend sampler used by the composability probe."""

from __future__ import annotations

import pytest
import torch

from imitation_experiments.evaluation.latent_blend import (
    BlendSchedule,
    LatentBlendSampler,
)


class _FakeSampler:
    """Two environments, constant codes, sin/cos phase pair appended."""

    def __init__(self) -> None:
        self.skill_encoder = "encoder-sentinel"
        self.calls = 0

    def sample_for_step(self, td, *, device, dtype):
        del td, device, dtype
        self.calls += 1
        codes = torch.stack([torch.full((4,), 1.0), torch.full((4,), 3.0)])
        phase = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        return torch.cat([codes, phase], dim=-1)


def test_schedule_ramps_linearly_then_holds():
    sched = BlendSchedule(start_step=10, ramp_steps=4, final_alpha=1.0)
    assert sched.alpha(0) == 0.0
    assert sched.alpha(9) == 0.0
    assert sched.alpha(10) == 0.0
    assert sched.alpha(12) == pytest.approx(0.5)
    assert sched.alpha(14) == pytest.approx(1.0)
    assert sched.alpha(100) == pytest.approx(1.0)
    assert BlendSchedule(start_step=3, ramp_steps=0, final_alpha=0.5).alpha(3) == 0.5
    with pytest.raises(ValueError):
        BlendSchedule(start_step=0, ramp_steps=1, final_alpha=1.5)


def test_blend_mixes_only_the_target_code_and_leaves_phase_alone():
    base = _FakeSampler()
    sampler = LatentBlendSampler(
        base,
        target_env=0,
        source_env=1,
        schedule=BlendSchedule(start_step=1, ramp_steps=2),
        code_dim=4,
    )
    out0 = sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    assert torch.equal(out0[0, :4], torch.full((4,), 1.0))  # alpha 0: untouched
    out1 = sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    assert torch.equal(out1[0, :4], torch.full((4,), 1.0))  # ramp start: alpha 0
    out2 = sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    assert torch.allclose(out2[0, :4], torch.full((4,), 2.0))  # alpha 0.5
    out3 = sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    assert torch.allclose(out3[0, :4], torch.full((4,), 3.0))  # alpha 1
    # The source environment and the phase columns never move.
    for out in (out0, out1, out2, out3):
        assert torch.equal(out[1, :4], torch.full((4,), 3.0))
        assert torch.equal(out[:, 4:], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert base.calls == 4
    assert sampler.trace.alpha == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert sampler.trace.code_distance == pytest.approx([4.0] * 4)
    assert sampler.trace.summary()["alpha_first_nonzero_step"] == 2


def test_attributes_forward_to_the_wrapped_sampler():
    base = _FakeSampler()
    sampler = LatentBlendSampler(
        base, target_env=0, source_env=1, schedule=BlendSchedule(0, 1), code_dim=4
    )
    assert sampler.skill_encoder == "encoder-sentinel"
    assert sampler.target_env == 0


def test_blend_refuses_a_batch_too_small_for_the_source_env():
    sampler = LatentBlendSampler(
        _FakeSampler(),
        target_env=0,
        source_env=5,
        schedule=BlendSchedule(0, 1),
        code_dim=4,
    )
    with pytest.raises(ValueError, match="run at least 6 environments"):
        sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
