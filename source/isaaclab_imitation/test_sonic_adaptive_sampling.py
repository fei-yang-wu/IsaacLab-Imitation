from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_MODULE_PATH = (
    Path(__file__).parent / "isaaclab_imitation" / "envs" / "sonic_adaptive_sampling.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "sonic_adaptive_sampling", _MODULE_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)
SonicAdaptiveResetSampler = _MODULE.SonicAdaptiveResetSampler


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
