"""Planted-truth checks for the latent-interpolation probe analyses."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlopt.agent.ipmd.module import build_bilinear_sr

from imitation_experiments.capacity.probe_latent_interpolation import (
    DEFAULT_ALPHAS,
    denoising_transfer,
    interpolant_geometry,
    pair_across_motions,
    score_affinity,
)

DEVICE = torch.device("cpu")
STATE_DIM = 6
Z_DIM = 5


def _head(parameterization: str, *, seed: int = 0):
    torch.manual_seed(seed)
    return build_bilinear_sr(
        "diffsr",
        obs_dim=STATE_DIM,
        next_obs_dim=STATE_DIM,
        action_dim=Z_DIM,
        feature_dim=4,
        embed_dim=6,
        g_hidden_dims=(8,),
        mu_hidden_dims=(8,),
        phi_parameterization=parameterization,
        num_noises=3,
        use_ema_for_policy=False,
        device="cpu",
    ).eval()


def _pair(seed: int = 0, count: int = 16):
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(count, STATE_DIM, generator=generator)
    z_left = torch.randn(count, Z_DIM, generator=generator)
    z_right = torch.randn(count, Z_DIM, generator=generator)
    return state, z_left, z_right


def test_affine_head_reports_zero_score_gap() -> None:
    """The whole point of the arm: phi at the mixed latent IS the mixed phi."""
    state, z_left, z_right = _pair()
    result = score_affinity(
        _head("affine"), state, z_left, z_right, DEFAULT_ALPHAS, device=DEVICE
    )
    for values in result.values():
        assert values["relative_gap_max"] < 1e-5


def test_nonlinear_heads_report_a_real_score_gap() -> None:
    """Negative control: with an MLP in the z path the interior is arbitrary,
    and only the two endpoints are exact."""
    state, z_left, z_right = _pair()
    for parameterization in ("concat", "bilinear"):
        result = score_affinity(
            _head(parameterization),
            state,
            z_left,
            z_right,
            DEFAULT_ALPHAS,
            device=DEVICE,
        )
        assert result["1"]["relative_gap_max"] < 1e-5
        assert result["0"]["relative_gap_max"] < 1e-5
        assert result["0.5"]["relative_gap_mean"] > 1e-3


def test_geometry_recovers_the_endpoints_exactly() -> None:
    """At alpha 0 and 1 the interpolant is a real latent, so its distance to
    the real set is zero and its norm ratio is the endpoints' own."""
    generator = torch.Generator().manual_seed(3)
    z_all = torch.randn(24, Z_DIM, generator=generator)
    left, right = z_all[:8], z_all[8:16]

    result = interpolant_geometry(z_all, left, right, (0.0, 0.5, 1.0))

    assert result["1"]["nearest_real_distance_mean"] == pytest.approx(0.0, abs=1e-6)
    assert result["0"]["nearest_real_distance_mean"] == pytest.approx(0.0, abs=1e-6)
    # A chord between independent Gaussian latents contracts toward the
    # origin, so the midpoint is nearer the mean than a real latent is.
    assert result["0.5"]["norm_ratio_to_real"] < 1.0
    assert result["_baseline"]["real_z_nearest_neighbor_mean"] > 0.0


def test_denoising_transfer_is_deterministic_and_alpha_indexed() -> None:
    head = _head("affine")
    state, z_left, z_right = _pair(seed=5)
    endpoint_left = torch.randn(state.shape[0], STATE_DIM)
    endpoint_right = torch.randn(state.shape[0], STATE_DIM)

    def run() -> dict:
        return denoising_transfer(
            head,
            state,
            endpoint_left,
            endpoint_right,
            z_left,
            z_right,
            DEFAULT_ALPHAS,
            device=DEVICE,
            seed=11,
        )

    first, second = run(), run()
    assert first == second
    assert set(first) == {f"{a:g}" for a in DEFAULT_ALPHAS}
    for values in first.values():
        assert values["eps_mse_left_target"] > 0.0
        assert values["eps_mse_right_target"] > 0.0


def test_pairs_never_share_a_motion() -> None:
    ranks = np.repeat(np.arange(20), 5)
    left, right = pair_across_motions(ranks, count=64, seed=0)

    assert left.shape == right.shape == (64,)
    assert np.all(ranks[left] != ranks[right])


def test_pairing_fails_loudly_when_one_motion_dominates() -> None:
    ranks = np.zeros(40, dtype=np.int64)
    with pytest.raises(ValueError, match="cross-motion pairs"):
        pair_across_motions(ranks, count=8, seed=0)
