"""Contracts for the latent blend sampler used by the composition probes."""

from __future__ import annotations

import pytest
import torch

from imitation_experiments.evaluation.latent_blend import (
    BlendSchedule,
    BlendSpec,
    LatentBlendSampler,
    pair_specs,
    triple_specs,
)


class _FakeTd(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class _FakeSampler:
    """N environments with constant codes 1, 3, 5, ...; sin/cos phase pair."""

    def __init__(self, n: int = 2) -> None:
        self.skill_encoder = "encoder-sentinel"
        self.calls = 0
        self.n = n

    def sample_for_step(self, td, *, device, dtype):
        del td, device, dtype
        self.calls += 1
        codes = torch.stack([torch.full((4,), 1.0 + 2.0 * i) for i in range(self.n)])
        phase = torch.tensor([[0.0, 1.0]] * self.n)
        return torch.cat([codes, phase], dim=-1)


def test_schedule_ramps_linearly_then_holds():
    sched = BlendSchedule(start_step=10, ramp_steps=4, final_alpha=1.0)
    assert sched.alpha(9) == 0.0
    assert sched.alpha(10) == 0.0
    assert sched.alpha(12) == pytest.approx(0.5)
    assert sched.alpha(14) == pytest.approx(1.0)
    assert sched.alpha(100) == pytest.approx(1.0)
    assert BlendSchedule(3, 0, 0.5).alpha(3) == 0.5
    # Extrapolation is allowed on purpose.
    assert BlendSchedule(0, 0, 1.5).alpha(0) == 1.5
    with pytest.raises(ValueError):
        BlendSchedule(start_step=-1, ramp_steps=1)


def test_single_pair_mixes_only_the_target_code_and_leaves_phase_alone():
    base = _FakeSampler()
    sampler = LatentBlendSampler(
        base, target_env=0, source_env=1, schedule=BlendSchedule(1, 2), code_dim=4
    )
    outs = [
        sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
        for _ in range(4)
    ]
    assert torch.equal(outs[0][0, :4], torch.full((4,), 1.0))
    assert torch.equal(outs[1][0, :4], torch.full((4,), 1.0))
    assert torch.allclose(outs[2][0, :4], torch.full((4,), 2.0))
    assert torch.allclose(outs[3][0, :4], torch.full((4,), 3.0))
    for out in outs:
        assert torch.equal(out[1, :4], torch.full((4,), 3.0))
        assert torch.equal(out[:, 4:], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert base.calls == 4
    assert sampler.trace.alpha == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert sampler.trace.code_distance == pytest.approx([4.0] * 4)
    assert sampler.trace.summary()["alpha_first_nonzero_step"] == 2
    assert sampler.skill_encoder == "encoder-sentinel"  # forwarded


def test_pairs_layout_mixes_every_even_environment_from_its_odd_partner():
    sampler = LatentBlendSampler(
        _FakeSampler(4),
        specs=pair_specs(4),
        schedule=BlendSchedule(0, 0, 0.5),
        code_dim=4,
    )
    out = sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    assert torch.allclose(out[0, :4], torch.full((4,), 2.0))  # 1 + 0.5 (3 - 1)
    assert torch.allclose(out[2, :4], torch.full((4,), 6.0))  # 5 + 0.5 (7 - 5)
    assert torch.equal(out[1, :4], torch.full((4,), 3.0))
    assert torch.equal(out[3, :4], torch.full((4,), 7.0))
    assert set(sampler.traces) == {0, 1, 2, 3}
    assert sampler.summary()["targets"].keys() == {"0", "2"}


def test_triples_layout_adds_the_source_minus_baseline_offset():
    sampler = LatentBlendSampler(
        _FakeSampler(3),
        specs=triple_specs(3),
        schedule=BlendSchedule(0, 0, 1.0),
        code_dim=4,
    )
    out = sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    # z_t + 1.0 (z_s - z_m) = 1 + (3 - 5) = -1
    assert torch.allclose(out[0, :4], torch.full((4,), -1.0))


def test_extrapolation_moves_past_the_source():
    sampler = LatentBlendSampler(
        _FakeSampler(),
        target_env=0,
        source_env=1,
        schedule=BlendSchedule(0, 0, 1.5),
        code_dim=4,
    )
    out = sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    assert torch.allclose(out[0, :4], torch.full((4,), 4.0))  # 1 + 1.5 * 2


def test_layout_helpers_and_spec_validation():
    assert [(s.target, s.source) for s in pair_specs(4)] == [(0, 1), (2, 3)]
    assert [(s.target, s.source, s.minus) for s in triple_specs(3)] == [(0, 1, 2)]
    with pytest.raises(ValueError):
        pair_specs(3)
    with pytest.raises(ValueError):
        BlendSpec(0, 0)
    with pytest.raises(ValueError):
        LatentBlendSampler(
            _FakeSampler(),
            specs=[BlendSpec(0, 1), BlendSpec(0, 2)],
            schedule=BlendSchedule(0, 1),
            code_dim=4,
        )


def test_batch_too_small_for_the_specs_is_refused():
    sampler = LatentBlendSampler(
        _FakeSampler(),
        target_env=0,
        source_env=5,
        schedule=BlendSchedule(0, 1),
        code_dim=4,
    )
    with pytest.raises(ValueError, match="need 6 environments"):
        sampler.sample_for_step(None, device="cpu", dtype=torch.float32)


def test_traces_record_motion_for_every_involved_environment():
    sampler = LatentBlendSampler(
        _FakeSampler(),
        target_env=0,
        source_env=1,
        schedule=BlendSchedule(0, 1),
        code_dim=4,
    )
    joints = torch.arange(2 * 29, dtype=torch.float32).reshape(2, 29)
    td0 = _FakeTd(
        {
            ("critic", "base_lin_vel"): torch.tensor(
                [[3.0, 4.0, 9.0], [0.6, 0.8, 0.0]]
            ),
            ("policy", "projected_gravity"): torch.tensor(
                [[0.0, 0.0, -1.0], [0.9, 0.0, -0.1]]
            ),
            ("policy", "last_action"): torch.tensor([[1.0, 0.0], [5.0, 5.0]]),
            ("policy", "joint_pos_rel"): torch.cat(
                [joints, joints], dim=-1
            ),  # 2-frame history
        }
    )
    td1 = _FakeTd(
        {
            ("critic", "base_lin_vel"): torch.tensor(
                [[0.0, 1.0, 9.0], [0.6, 0.8, 0.0]]
            ),
            ("policy", "projected_gravity"): torch.tensor(
                [[0.0, 0.0, -1.0], [0.9, 0.0, -0.1]]
            ),
            ("policy", "last_action"): torch.tensor([[1.0, 2.0], [5.0, 5.0]]),
            ("policy", "joint_pos_rel"): torch.cat([joints, joints], dim=-1),
        }
    )
    sampler.sample_for_step(td0, device="cpu", dtype=torch.float32)
    sampler.sample_for_step(td1, device="cpu", dtype=torch.float32)
    t, s = sampler.traces[0], sampler.traces[1]
    assert t.root_speed == pytest.approx([5.0, 1.0])
    assert s.root_speed == pytest.approx([1.0, 1.0])
    assert t.upright == pytest.approx([1.0, 1.0])
    assert s.upright == pytest.approx([0.1, 0.1])
    assert t.action_delta[0] != t.action_delta[0]  # NaN before a previous action
    assert t.action_delta[1] == pytest.approx(2.0)
    assert len(t.joint_pos[0]) == 29 and t.joint_pos[0][0] == 0.0
    assert s.joint_pos[0][0] == 29.0  # env 1's newest frame
    assert s.summary()["fallen_steps"] == 2
    assert t.summary()["fallen_steps"] == 0
    # No observation keys: NaNs, no raise.
    sampler.sample_for_step(None, device="cpu", dtype=torch.float32)
    assert len(t.root_speed) == 3
    assert set(sampler.traces_as_dict()) == {"0", "1"}
