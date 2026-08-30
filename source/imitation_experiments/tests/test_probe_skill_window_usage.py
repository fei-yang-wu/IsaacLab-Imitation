"""Planted-truth checks for the window-usage probe analyses."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from rlopt.agent.hl_skill_diffsr import HighLevelSkillDiffSRConfig
from rlopt.agent.hl_skill_encoder import build_skill_encoder

from imitation_experiments.capacity.probe_skill_window_usage import (
    FRAME_WIDTH,
    encode_windows,
    frame_sufficiency,
    group_split,
    heading_anchored_expert_windows,
    integrated_gradients_per_slot,
    load_encoder_bundle,
    per_offset_probes,
    sensitivity,
)

DEVICE = torch.device("cpu")


def _random_quat_xyzw(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    quat = torch.randn(*shape, 4, generator=generator, dtype=torch.float64)
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)


def test_windows_invariant_to_global_yaw_and_xy_shift() -> None:
    torch.manual_seed(0)
    n, t = 8, 5
    qpos = torch.randn(n, t, 29, dtype=torch.float64)
    pos = torch.randn(n, t, 3, dtype=torch.float64)
    quat = _random_quat_xyzw((n, t), seed=1)

    yaw = math.radians(37.0)
    half = yaw / 2.0
    yaw_quat = torch.tensor(
        [0.0, 0.0, math.sin(half), math.cos(half)], dtype=torch.float64
    )
    cos, sin = math.cos(yaw), math.sin(yaw)
    rotation = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    shift = torch.tensor([2.5, -1.0, 0.0], dtype=torch.float64)

    def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ax, ay, az, aw = a.unbind(-1)
        bx, by, bz, bw = b.unbind(-1)
        return torch.stack(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dim=-1,
        )

    moved_pos = pos @ rotation.T + shift
    moved_quat = _mul(yaw_quat.expand_as(quat), quat)

    base = heading_anchored_expert_windows(qpos, pos, quat)
    moved = heading_anchored_expert_windows(qpos, moved_pos, moved_quat)
    assert torch.allclose(base, moved, atol=1e-5)


def test_windows_slot0_anchor_and_rot6d_layout() -> None:
    n, t = 4, 3
    qpos = torch.zeros(n, t, 29, dtype=torch.float64)
    pos = torch.zeros(n, t, 3, dtype=torch.float64)
    pos[..., 2] = 0.8  # absolute height must survive the xy-only origin
    quat = torch.zeros(n, t, 4, dtype=torch.float64)
    quat[..., 3] = 1.0  # identity
    frames = heading_anchored_expert_windows(qpos, pos, quat)
    assert frames.shape == (n, t, FRAME_WIDTH)
    assert torch.allclose(frames[:, 0, 29:31], torch.zeros(n, 2))
    assert torch.allclose(frames[:, 0, 31], torch.full((n,), 0.8))
    # Interleaved quat_to_rot6d_flat layout: identity -> (1,0,0,1,0,0).
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert torch.allclose(frames[0, 0, 32:38], expected)


def _last_frame_only_encoder() -> tuple[torch.nn.Module, HighLevelSkillDiffSRConfig]:
    """Deterministic encoder whose z reads ONLY the last visible frame."""
    config = HighLevelSkillDiffSRConfig(
        horizon_steps=4,
        encoder_window_mode="intermediate",
        z_dim=8,
        encoder_hidden_dims=(32,),
        encoder_activation="relu",
        encoder_layer_norm=False,
        latent_mode="deterministic",
    )
    config.validate()
    torch.manual_seed(0)
    encoder = build_skill_encoder(
        state_dim=FRAME_WIDTH,
        window_steps=3,
        z_dim=config.z_dim,
        hidden_dims=config.encoder_hidden_dims,
        spec=config.latent_spec(),
        activation=config.encoder_activation,
        layer_norm=config.encoder_layer_norm,
    )
    with torch.no_grad():
        first = dict(encoder.named_parameters())["net.0.weight"]
        # Input layout: [state, w1, w2, w3] frame blocks; keep only w3.
        first[:, : 3 * FRAME_WIDTH] = 0.0
    encoder.eval()
    return encoder, config


@pytest.fixture(scope="module")
def planted() -> dict[str, object]:
    encoder, config = _last_frame_only_encoder()
    torch.manual_seed(1)
    windows = torch.randn(400, config.horizon_steps + 1, FRAME_WIDTH)
    ranks = np.repeat(np.arange(40), 10)
    train_mask, test_mask = group_split(ranks, test_fraction=0.2, seed=0)
    z = encode_windows(encoder, config, windows, device=DEVICE, batch_size=128)
    return {
        "encoder": encoder,
        "config": config,
        "windows": windows,
        "z": z,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


def test_group_split_never_splits_a_motion() -> None:
    ranks = np.repeat(np.arange(25), 4)
    train_mask, test_mask = group_split(ranks, test_fraction=0.2, seed=3)
    assert not np.intersect1d(ranks[train_mask], ranks[test_mask]).size
    assert train_mask.sum() + test_mask.sum() == ranks.size


def test_frame_sufficiency_detects_last_frame_collapse(planted) -> None:
    results = frame_sufficiency(
        planted["windows"],
        planted["z"],
        planted["train_mask"],
        planted["test_mask"],
        visible_steps=3,
        device=DEVICE,
        mlp_epochs=200,
    )
    # z is a ReLU MLP of the last visible frame: the nonlinear probe from that
    # frame alone must explain almost everything ...
    assert results["last1"]["mlp_r2"] > 0.85
    # ... and the unseen endpoint (independent noise here) must explain nothing.
    assert results["endpoint_unseen"]["ridge_r2"] < 0.1


def test_per_offset_probes_show_no_mid_frame_information(planted) -> None:
    rows = per_offset_probes(
        planted["windows"],
        planted["z"],
        planted["train_mask"],
        planted["test_mask"],
        visible_steps=3,
    )
    by_slot = {int(row["slot"]): row for row in rows}
    # Mid slots (1, 2) are independent noise: z adds no linear information.
    assert abs(by_slot[1]["z_increment_over_boundary"]) < 0.05
    assert abs(by_slot[2]["z_increment_over_boundary"]) < 0.05
    # The last visible frame (slot 3) is what z encodes. With z_dim 8 against
    # a 38-wide white target the linear ceiling is ~8/38, so expect a clear
    # positive R2, not a large one; the unseen endpoint (slot 4) gets none.
    assert by_slot[3]["r2_from_z"] > 0.08
    assert abs(by_slot[1]["r2_from_z"]) < 0.05
    assert abs(by_slot[4]["r2_from_z"]) < 0.05


def test_sensitivity_is_blind_to_mid_frames(planted) -> None:
    results = sensitivity(
        planted["encoder"],
        planted["config"],
        planted["windows"],
        planted["z"],
        visible_steps=3,
        device=DEVICE,
        batch_size=128,
        seed=0,
    )
    assert results["mid_swap"]["dz_norm_mean"] < 1e-6
    assert results["last_swap"]["dz_norm_mean"] > 0.1
    assert results["mid_interp"]["dz_norm_mean"] < 1e-6


def test_integrated_gradients_attribute_to_last_slot(planted) -> None:
    attribution = integrated_gradients_per_slot(
        planted["encoder"],
        planted["config"],
        planted["windows"],
        visible_steps=3,
        device=DEVICE,
        steps=16,
        max_windows=64,
        seed=0,
    )
    shares = attribution["share_per_slot"]
    assert len(shares) == 4  # state + three visible slots
    # A last-frame-only encoder: everything lands on slot 3, nothing on mids.
    assert shares[3] > 0.9
    assert shares[1] < 0.02 and shares[2] < 0.02
    # Completeness: the summed IG approximates ||z(x) - z(b)||^2.
    assert attribution["completeness_target"] > 0.0
    ratio = attribution["completeness_sum_ig"] / attribution["completeness_target"]
    assert 0.8 < ratio < 1.2


def test_load_encoder_bundle_round_trip(tmp_path) -> None:
    encoder, config = _last_frame_only_encoder()
    path = tmp_path / "encoder.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "skill_encoder_state_dict": encoder.state_dict(),
        },
        path,
    )
    restored, restored_config = load_encoder_bundle(path)
    assert restored_config.encoder_window_mode == "intermediate"
    windows = torch.randn(4, config.horizon_steps + 1, FRAME_WIDTH)
    z_original = encode_windows(encoder, config, windows, device=DEVICE, batch_size=4)
    z_restored = encode_windows(
        restored, restored_config, windows, device=DEVICE, batch_size=4
    )
    assert torch.allclose(z_original, z_restored)
