"""Tests for the GR00T training-table preparation logic."""

from __future__ import annotations

import torch

from imitation_experiments.planner.prepare_gr00t_dataset import (
    _encoder_flat_input,
    _join_slots,
)


def test_join_slots_matches_forward_rows() -> None:
    # Two envs; env 0 has rows at steps 0/10/20, env 1 at 0/10 only.
    env_id = torch.tensor([0, 0, 0, 1, 1])
    episode_id = torch.tensor([7, 7, 7, 3, 3])
    control_step = torch.tensor([0, 10, 20, 0, 10])
    values = torch.arange(5, dtype=torch.float32).reshape(5, 1) * 100.0
    target, valid = _join_slots(
        values, env_id, episode_id, control_step, slots=3, hold_steps=10
    )
    assert target.shape == (5, 3, 1)
    # Row 0 (env 0 step 0) sees rows 0/1/2.
    assert torch.equal(target[0, :, 0], torch.tensor([0.0, 100.0, 200.0]))
    assert valid[0].tolist() == [True, True, True]
    # Row 1 (env 0 step 10) sees rows 1/2, slot 2 invalid.
    assert valid[1].tolist() == [True, True, False]
    # Row 3 (env 1 step 0) must not join across environments.
    assert torch.equal(target[3, :2, 0], torch.tensor([300.0, 400.0]))
    assert valid[3].tolist() == [True, True, False]


def test_join_slots_does_not_cross_episodes() -> None:
    env_id = torch.tensor([0, 0])
    episode_id = torch.tensor([1, 2])
    control_step = torch.tensor([0, 10])
    values = torch.tensor([[1.0], [2.0]])
    _, valid = _join_slots(
        values, env_id, episode_id, control_step, slots=2, hold_steps=10
    )
    assert valid[0].tolist() == [True, False]


def test_join_slots_rejects_duplicate_keys() -> None:
    env_id = torch.tensor([0, 0])
    episode_id = torch.tensor([1, 1])
    control_step = torch.tensor([0, 0])
    values = torch.tensor([[1.0], [2.0]])
    try:
        _join_slots(values, env_id, episode_id, control_step, slots=1, hold_steps=10)
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("duplicate keys must be rejected")


def test_encoder_flat_input_layout() -> None:
    # Verified against stored targets: [frame 0; frames 1..9 frame-major].
    future = torch.arange(2 * 30 * 38, dtype=torch.float32).reshape(2, 30, 38)
    flat = _encoder_flat_input(future)
    assert flat.shape == (2, 38 * 10)
    assert torch.equal(flat[:, :38], future[:, 0])
    assert torch.equal(flat[:, 38:76], future[:, 1])
    assert torch.equal(flat[:, -38:], future[:, 9])
