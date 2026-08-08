from __future__ import annotations

import math

import torch

from imitation_experiments.planner.latent_receding_horizon import (
    OverlappingLatentEnsembler,
)


def _ensemble(mode: str, *, gate_cosine: float = 0.5) -> OverlappingLatentEnsembler:
    return OverlappingLatentEnsembler(
        num_envs=2,
        token_count=3,
        token_width=2,
        hold_steps=10,
        mode=mode,
        decay=0.5,
        reference_std=torch.ones(2),
        clip_std=1.0,
        gate_distance=2.0,
        gate_cosine=gate_cosine,
        device="cpu",
        dtype=torch.float32,
    )


def test_first_mode_executes_fresh_token_and_clears_on_reset() -> None:
    ensemble = _ensemble("first")
    first = ensemble.update(
        env_ids=torch.tensor([0]),
        prediction=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        episode_steps=torch.tensor([0]),
    )
    assert torch.equal(first, torch.tensor([[1.0, 2.0]]))
    reset = ensemble.update(
        env_ids=torch.tensor([0]),
        prediction=torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]),
        episode_steps=torch.tensor([0]),
    )
    assert torch.equal(reset, torch.tensor([[7.0, 8.0]]))
    assert ensemble.stats()["history_resets"] == 2


def test_exponential_mode_aligns_three_forecast_ages() -> None:
    ensemble = _ensemble("exponential")
    ensemble.update(
        env_ids=torch.tensor([0]),
        prediction=torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]]),
        episode_steps=torch.tensor([0]),
    )
    ensemble.update(
        env_ids=torch.tensor([0]),
        prediction=torch.tensor([[10.0, 10.0, 20.0, 20.0, 30.0, 30.0]]),
        episode_steps=torch.tensor([10]),
    )
    output = ensemble.update(
        env_ids=torch.tensor([0]),
        prediction=torch.tensor([[100.0, 100.0, 200.0, 200.0, 300.0, 300.0]]),
        episode_steps=torch.tensor([20]),
    )
    weights = torch.tensor([1.0, math.exp(-0.5), math.exp(-1.0)])
    expected = float(
        ((weights * torch.tensor([100.0, 20.0, 3.0])).sum() / weights.sum()).item()
    )
    assert torch.allclose(output, torch.full((1, 2), expected), atol=1.0e-6)
    assert ensemble.stats()["candidate_histogram"]["3"] == 1


def test_clipped_gated_rejects_incoherent_old_forecast() -> None:
    ensemble = _ensemble("clipped_gated", gate_cosine=0.5)
    ensemble.update(
        env_ids=torch.tensor([0]),
        prediction=torch.tensor([[1.0, 1.0, -10.0, -10.0, 0.0, 0.0]]),
        episode_steps=torch.tensor([0]),
    )
    output = ensemble.update(
        env_ids=torch.tensor([0]),
        prediction=torch.tensor([[2.0, 2.0, 4.0, 4.0, 6.0, 6.0]]),
        episode_steps=torch.tensor([10]),
    )
    assert torch.equal(output, torch.tensor([[2.0, 2.0]]))
    assert ensemble.stats()["rejected_candidates"] == 1
