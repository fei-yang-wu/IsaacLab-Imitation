from __future__ import annotations

import pytest
import torch

from iltools.datasets.reset_sampling import SonicAdaptiveResetSampler
from isaaclab_imitation.tasks.manager_based.imitation.mdp.commands.reset_sampling import (
    RandomTrajectoryAdaptiveResetSampler,
)


def test_fixed_motion_local_bins_and_sequence_length_weights() -> None:
    sampler = SonicAdaptiveResetSampler(
        torch.tensor([120, 55]),
        bin_size=50,
        pre_failure_sample_window=0,
    )

    assert sampler.bins.tolist() == [
        [0, 0, 50],
        [0, 50, 100],
        [0, 100, 120],
        [1, 0, 50],
        [1, 50, 55],
    ]
    expected_weights = torch.tensor([50 / 3, 50 / 3, 20 / 3, 50 / 2, 5 / 2])
    expected_weights /= expected_weights.sum()
    torch.testing.assert_close(sampler.sampling_probabilities(), expected_weights)


def test_visit_and_failure_statistics_match_sonic_updates() -> None:
    sampler = SonicAdaptiveResetSampler(
        torch.tensor([120, 55]),
        bin_size=50,
        pre_failure_sample_window=0,
    )
    sampler.record_visits(
        torch.tensor([0, 0, 1]),
        torch.tensor([10, 110, 54]),
    )
    sampler.record_failures(
        torch.tensor([0, 1]),
        torch.tensor([110, 54]),
    )

    expected_visits = torch.tensor([1.0 + 1 / 50, 1.0, 1.0 + 1 / 20, 1.0, 1.0 + 1 / 5])
    expected_failures = torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0])
    torch.testing.assert_close(sampler.num_visits, expected_visits)
    torch.testing.assert_close(sampler.num_failures, expected_failures)


def test_failure_rates_change_motion_and_bin_sampling_jointly() -> None:
    sampler = SonicAdaptiveResetSampler(
        torch.tensor([100, 100]),
        bin_size=50,
        uniform_sampling_rate=0.1,
        pre_failure_sample_window=0,
    )
    sampler.num_visits.fill_(100.0)
    sampler.num_failures.fill_(1.0)
    sampler.num_failures[3] = 90.0

    probabilities = sampler.sampling_probabilities()
    assert probabilities[3] > probabilities[0] * 20
    assert probabilities[2:].sum() > probabilities[:2].sum()


def test_random_full_trajectory_starts_apply_sonic_lead_in() -> None:
    lengths = torch.tensor([500, 260])
    raw_sampler = SonicAdaptiveResetSampler(
        lengths,
        bin_size=50,
        pre_failure_sample_window=0,
    )
    lead_in_sampler = SonicAdaptiveResetSampler(
        lengths,
        bin_size=50,
        pre_failure_sample_window=200,
    )

    torch.manual_seed(1234)
    raw_ranks, raw_steps = raw_sampler.sample(4096)
    torch.manual_seed(1234)
    lead_in_ranks, lead_in_steps = lead_in_sampler.sample(4096)

    torch.testing.assert_close(lead_in_ranks, raw_ranks)
    assert torch.all(lead_in_steps <= raw_steps)
    assert torch.all(raw_steps - lead_in_steps <= 199)
    assert torch.all(lead_in_steps >= 0)
    assert torch.all(lead_in_steps < lengths.index_select(0, lead_in_ranks))
    assert torch.unique(lead_in_steps).numel() > 100
    assert torch.any(lead_in_steps > 200)
    assert torch.any(lead_in_steps == 0)


def test_random_trajectory_branch_is_uniform_and_stays_in_leading_fraction() -> None:
    lengths = torch.tensor([1, 5, 20, 101])
    generator = torch.Generator().manual_seed(17)
    adaptive = SonicAdaptiveResetSampler(
        lengths,
        pre_failure_sample_window=0,
        generator=generator,
    )
    sampler = RandomTrajectoryAdaptiveResetSampler(
        lengths,
        adaptive=adaptive,
        random_sampling_ratio=1.0,
        random_start_fraction=0.5,
        generator=generator,
    )

    ranks, steps = sampler.sample(20_000)
    spans = torch.ceil(lengths.index_select(0, ranks) * 0.5).to(torch.long)
    assert torch.all(steps >= 0)
    assert torch.all(steps < spans)
    counts = torch.bincount(ranks, minlength=lengths.numel()).float()
    expected = ranks.numel() / lengths.numel()
    assert torch.all(torch.abs(counts - expected) < expected * 0.05)


def test_random80_adaptive20_is_an_explicit_per_reset_mixture() -> None:
    lengths = torch.tensor([100])
    generator = torch.Generator().manual_seed(23)
    adaptive = SonicAdaptiveResetSampler(
        lengths,
        bin_size=50,
        uniform_sampling_rate=0.0,
        pre_failure_sample_window=0,
        generator=generator,
    )
    sampler = RandomTrajectoryAdaptiveResetSampler(
        lengths,
        adaptive=adaptive,
        random_sampling_ratio=0.8,
        random_start_fraction=0.5,
        generator=generator,
    )
    # Force every adaptive candidate into the second half. The random branch
    # is restricted to the first half, so the returned frame identifies which
    # branch supplied each reset without exposing implementation-only state.
    probabilities = torch.tensor([0.0, 1.0])
    _ranks, steps = sampler.sample(20_000, probabilities=probabilities)

    observed_random_ratio = float((steps < 50).float().mean())
    assert observed_random_ratio == pytest.approx(0.8, abs=0.015)


def test_zero_random_ratio_exactly_delegates_to_sonic() -> None:
    lengths = torch.tensor([100, 175])
    direct_generator = torch.Generator().manual_seed(31)
    wrapped_generator = torch.Generator().manual_seed(31)
    direct = SonicAdaptiveResetSampler(
        lengths,
        pre_failure_sample_window=20,
        generator=direct_generator,
    )
    wrapped_adaptive = SonicAdaptiveResetSampler(
        lengths,
        pre_failure_sample_window=20,
        generator=wrapped_generator,
    )
    wrapped = RandomTrajectoryAdaptiveResetSampler(
        lengths,
        adaptive=wrapped_adaptive,
        random_sampling_ratio=0.0,
        generator=wrapped_generator,
    )

    expected_ranks, expected_steps = direct.sample(1024)
    actual_ranks, actual_steps = wrapped.sample(1024)
    torch.testing.assert_close(actual_ranks, expected_ranks)
    torch.testing.assert_close(actual_steps, expected_steps)


def test_reset_ramp_refuses_the_random_trajectory_wrapper() -> None:
    """The adaptive_uniform_ratio ramp under the random80 wrapper is a silent
    near-no-op (it rescales only the wrapped 20% branch); refuse it loudly."""
    import pytest

    from isaaclab_imitation.tasks.manager_based.imitation.mdp.commands.reference import (
        ReferenceSelectionCfg,
    )

    cfg = ReferenceSelectionCfg(
        schedule="random",
        full_trajectory=True,
        random_trajectory_sampling_ratio=0.8,
        adaptive_uniform_ratio=0.8,
        adaptive_uniform_ratio_final=0.2,
        adaptive_ratio_ramp_frames=1_000_000_000,
    )
    with pytest.raises(ValueError, match="rescale the adaptive branch"):
        cfg.resolve()
    # The sonic shape (no wrapper) stays valid.
    ok = ReferenceSelectionCfg(
        schedule="random",
        full_trajectory=True,
        adaptive_uniform_ratio=0.8,
        adaptive_uniform_ratio_final=0.2,
        adaptive_ratio_ramp_frames=1_000_000_000,
    )
    ok.resolve()
