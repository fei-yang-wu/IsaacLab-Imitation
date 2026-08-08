"""Contract tests for the released SONIC G1 tracker adapter.

The 469 MB checkpoint is not a repository artifact, so the tests that need it
skip when it is absent. Everything that can be checked without it - the FSQ
lattice and the encoder window layout - always runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from imitation_experiments.lowlevel.sonic_release_actor import (
    ACTION_DIM,
    ENCODER_FRAMES,
    FSQ,
    PROPRIOCEPTION_DIM,
    load_sonic_release_actor,
    pack_encoder_window,
)


DEFAULT_CHECKPOINT = Path("/mnt/hsstorage/fwu91/sonic_release/last.pt")


def _checkpoint() -> Path | None:
    path = Path(os.environ.get("SONIC_RELEASE_CHECKPOINT", DEFAULT_CHECKPOINT))
    return path if path.is_file() else None


def test_fsq_lands_on_the_released_lattice() -> None:
    quantizer = FSQ(32)
    codes = quantizer(torch.randn(256, 64) * 4.0)
    assert torch.equal(codes * 16.0, (codes * 16.0).round())
    assert float(codes.min()) >= -1.0
    assert float(codes.max()) <= 15.0 / 16.0
    assert int(codes.unique().numel()) == 32


def test_fsq_snap_is_idempotent_on_lattice_values() -> None:
    quantizer = FSQ(32)
    codes = quantizer(torch.randn(64, 64) * 3.0)
    assert torch.equal(quantizer.snap(codes), codes)


def test_encoder_window_interleaves_two_frames_per_block() -> None:
    """Each 64-wide block is two frames of one term plus one orientation frame."""
    joint_pos = torch.arange(10 * 29, dtype=torch.float32).reshape(1, 10, 29)
    joint_vel = joint_pos + 1000.0
    anchor_ori = torch.arange(10 * 6, dtype=torch.float32).reshape(1, 10, 6) + 5000.0

    window = pack_encoder_window(joint_pos, joint_vel, anchor_ori)

    assert window.shape == (1, 640)
    for block in range(ENCODER_FRAMES // 2):
        start = block * 64
        assert torch.equal(window[0, start : start + 29], joint_pos[0, 2 * block])
        assert torch.equal(
            window[0, start + 29 : start + 58], joint_pos[0, 2 * block + 1]
        )
        assert torch.equal(window[0, start + 58 : start + 64], anchor_ori[0, block])
    for block in range(ENCODER_FRAMES // 2):
        start = (5 + block) * 64
        assert torch.equal(window[0, start : start + 29], joint_vel[0, 2 * block])
        assert torch.equal(
            window[0, start + 29 : start + 58], joint_vel[0, 2 * block + 1]
        )
        assert torch.equal(window[0, start + 58 : start + 64], anchor_ori[0, 5 + block])


def test_encoder_window_rejects_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="joint_vel"):
        pack_encoder_window(
            torch.zeros(1, 10, 29), torch.zeros(1, 10, 58), torch.zeros(1, 10, 6)
        )


@pytest.mark.skipif(_checkpoint() is None, reason="released SONIC checkpoint absent")
def test_released_actor_matches_the_published_shapes() -> None:
    actor = load_sonic_release_actor(_checkpoint())
    spec = actor.spec
    assert spec.encoder_input_dim == 640
    assert spec.token_dim == 64
    assert spec.decoder_input_dim == 64 + PROPRIOCEPTION_DIM
    assert spec.action_dim == ACTION_DIM
    assert spec.encoder_frame_width == 64


@pytest.mark.skipif(_checkpoint() is None, reason="released SONIC checkpoint absent")
def test_released_actor_emits_lattice_tokens_and_actions() -> None:
    actor = load_sonic_release_actor(_checkpoint())
    window = pack_encoder_window(
        torch.randn(3, 10, 29) * 0.3,
        torch.randn(3, 10, 29) * 0.3,
        torch.randn(3, 10, 6) * 0.3,
    )
    with torch.no_grad():
        token = actor.encode(window)
        action = actor(window, torch.randn(3, PROPRIOCEPTION_DIM) * 0.1)
    assert token.shape == (3, 64)
    assert torch.equal(token * 16.0, (token * 16.0).round())
    assert action.shape == (3, ACTION_DIM)
    assert torch.isfinite(action).all()
