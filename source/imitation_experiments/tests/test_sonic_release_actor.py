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
    PROPRIOCEPTION_TERMS,
    PROPRIOCEPTION_DIM,
    RELEASE_DECODER_HIDDEN_DIMS,
    V1_1_DECODER_HIDDEN_DIMS,
    _infer_sonic_version,
    assemble_proprioception,
    heading_relative_rot6d_from_full_relative,
    matrix_to_rot6d,
    quat_xyzw_to_matrix,
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


def test_checkpoint_shape_infers_sonic_version() -> None:
    assert _infer_sonic_version(RELEASE_DECODER_HIDDEN_DIMS) == "release"
    assert _infer_sonic_version(V1_1_DECODER_HIDDEN_DIMS) == "v1_1"
    with pytest.raises(ValueError, match="Unsupported SONIC g1_dyn decoder shape"):
        _infer_sonic_version((2048, 512))


def test_v1_1_heading_relative_orientation_keeps_robot_tilt() -> None:
    """A full-relative identity is not identity after heading-only rerooting."""
    angle = torch.tensor(0.5)
    robot_roll = torch.tensor(
        [[torch.sin(angle / 2.0), 0.0, 0.0, torch.cos(angle / 2.0)]],
        dtype=torch.float32,
    )
    identity_matrix = torch.eye(3).reshape(1, 1, 3, 3)
    full_relative_identity = matrix_to_rot6d(identity_matrix)

    heading_relative = heading_relative_rot6d_from_full_relative(
        full_relative_identity, robot_roll
    )

    expected = matrix_to_rot6d(quat_xyzw_to_matrix(robot_roll)).reshape(1, 1, 6)
    assert torch.allclose(heading_relative, expected, atol=1.0e-6)
    assert not torch.allclose(heading_relative, full_relative_identity, atol=1.0e-6)


def test_proprioception_uses_sonic_policy_field_order() -> None:
    assert [name for name, _ in PROPRIOCEPTION_TERMS] == [
        "base_ang_vel",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
        "gravity_dir",
    ]

    gravity = torch.full((1, 10, 3), 1.0)
    ang_vel = torch.full((1, 10, 3), 2.0)
    joint_pos = torch.full((1, 10, 29), 3.0)
    joint_vel = torch.full((1, 10, 29), 4.0)
    action = torch.full((1, 10, 29), 5.0)

    proprioception = assemble_proprioception(
        gravity, ang_vel, joint_pos, joint_vel, action
    )

    assert proprioception.shape == (1, PROPRIOCEPTION_DIM)
    assert torch.equal(proprioception[:, :30], ang_vel.reshape(1, -1))
    assert torch.equal(proprioception[:, -30:], gravity.reshape(1, -1))


@pytest.mark.skipif(_checkpoint() is None, reason="released SONIC checkpoint absent")
def test_released_actor_matches_the_published_shapes() -> None:
    actor = load_sonic_release_actor(_checkpoint())
    spec = actor.spec
    assert spec.encoder_input_dim == 640
    assert spec.token_dim == 64
    assert spec.decoder_input_dim == 64 + PROPRIOCEPTION_DIM
    assert spec.action_dim == ACTION_DIM
    assert spec.encoder_frame_width == 64
    assert spec.version == "release"
    assert spec.orientation_contract == "motion_anchor_ori_b_mf_nonflat"


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
