from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from imitation_experiments.capacity.materialize_long_horizon_root_qpos import (
    TARGET_JOINT_NAMES,
    _extend_motion_rows,
)
from imitation_experiments.capacity.materialize_paired_interface_samples import _materialize_sample
from imitation_experiments.capacity.packet_to_latent_command import (
    OverlappingPacketEnsembler,
    PacketLayout,
    first_packet_window,
    frames_to_term_major,
    split_packet_for_encoder,
    term_major_to_frames,
)


ROOT_QPOS_TERM_WIDTHS = (
    ("expert_motion_qpos", 29),
    ("expert_anchor_pos_b", 3),
    ("expert_anchor_ori_b", 6),
)


def test_root_qpos_packet_roundtrip_and_encoder_split() -> None:
    spec = SimpleNamespace(
        term_names=tuple(name for name, _ in ROOT_QPOS_TERM_WIDTHS),
        term_widths=(290, 30, 60),
    )
    layout = PacketLayout.from_target_spec(spec, packet_frames=10)
    assert layout.term_widths == ROOT_QPOS_TERM_WIDTHS
    assert layout.frame_width == 38
    assert layout.packet_width == 380

    frames = torch.arange(2 * 10 * 38, dtype=torch.float32).reshape(2, 10, 38)
    packet = frames_to_term_major(frames, layout)
    assert torch.equal(term_major_to_frames(packet, layout), frames)
    state, future = split_packet_for_encoder(packet, layout)
    assert torch.equal(state, frames[:, 0])
    assert torch.equal(future, frames[:, 1:])


def test_packet_layout_rejects_nondivisible_target_width() -> None:
    spec = SimpleNamespace(term_names=("qpos",), term_widths=(291,))
    with pytest.raises(ValueError, match="not divisible"):
        PacketLayout.from_target_spec(spec, packet_frames=10)


def test_h30_first_window_executes_ten_and_discards_twenty() -> None:
    layout = PacketLayout(ROOT_QPOS_TERM_WIDTHS, packet_frames=30)
    frames = torch.arange(30 * 38, dtype=torch.float32).reshape(1, 30, 38)
    packet = frames_to_term_major(frames, layout)
    executed, executed_layout = first_packet_window(
        packet, prediction_layout=layout, execution_frames=10
    )
    assert executed_layout.packet_frames == 10
    assert torch.equal(term_major_to_frames(executed, executed_layout), frames[:, :10])


def _identity_rot6d(rows: int) -> torch.Tensor:
    value = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return value.repeat(rows, 1)


def test_temporal_ensemble_aligns_overlaps_and_clears_on_reset() -> None:
    layout = PacketLayout(
        (
            ("expert_motion_qpos", 1),
            ("expert_anchor_pos_b", 3),
            ("expert_anchor_ori_b", 6),
        ),
        packet_frames=30,
    )
    ensemble = OverlappingPacketEnsembler(
        num_envs=1,
        prediction_layout=layout,
        execution_frames=10,
        decay=0.5,
        device="cpu",
        dtype=torch.float32,
    )

    def packet(publication_step: int, anchor_x: float) -> torch.Tensor:
        absolute_steps = torch.arange(
            publication_step, publication_step + 30, dtype=torch.float32
        )
        frames = torch.cat(
            (
                absolute_steps[:, None],
                torch.stack(
                    (
                        absolute_steps - anchor_x,
                        torch.zeros(30),
                        torch.zeros(30),
                    ),
                    dim=-1,
                ),
                _identity_rot6d(30),
            ),
            dim=-1,
        ).unsqueeze(0)
        return frames_to_term_major(frames, layout)

    identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    first = ensemble.update(
        env_ids=torch.tensor([0]),
        packet=packet(0, 0.0),
        anchor_pos=torch.tensor([[0.0, 0.0, 0.0]]),
        anchor_quat=identity_quat,
        episode_steps=torch.tensor([0]),
    )
    second = ensemble.update(
        env_ids=torch.tensor([0]),
        packet=packet(10, 1.0),
        anchor_pos=torch.tensor([[1.0, 0.0, 0.0]]),
        anchor_quat=identity_quat,
        episode_steps=torch.tensor([10]),
    )
    execution_layout = PacketLayout(layout.term_widths, packet_frames=10)
    first_frames = term_major_to_frames(first, execution_layout)
    second_frames = term_major_to_frames(second, execution_layout)
    assert torch.allclose(first_frames[0, :, 0], torch.arange(10.0))
    assert torch.allclose(second_frames[0, :, 0], torch.arange(10.0, 20.0))
    assert torch.allclose(second_frames[0, :, 1], torch.arange(9.0, 19.0), atol=1.0e-6)

    reset_packet = packet(100, 5.0)
    after_reset = ensemble.update(
        env_ids=torch.tensor([0]),
        packet=reset_packet,
        anchor_pos=torch.tensor([[5.0, 0.0, 0.0]]),
        anchor_quat=identity_quat,
        episode_steps=torch.tensor([0]),
    )
    reset_frames = term_major_to_frames(after_reset, execution_layout)
    assert torch.allclose(reset_frames[0, :, 0], torch.arange(100.0, 110.0))
    assert ensemble.stats()["temporal_ensemble_history_resets"] == 2


def test_long_horizon_materialization_reuses_exact_source_rows() -> None:
    length = 40
    horizon = 30
    joint_pos = torch.arange(length * 29, dtype=torch.float64).reshape(length, 29)
    pelvis_pos = torch.stack(
        (torch.arange(float(length)), torch.zeros(length), torch.zeros(length)),
        dim=-1,
    )
    body_pos = torch.zeros(length, 1, 3, dtype=torch.float64)
    body_pos[:, 0] = pelvis_pos
    body_quat = torch.zeros(length, 1, 4, dtype=torch.float64)
    body_quat[..., 0] = 1.0  # dataset WXYZ identity
    trajectory = {
        "joint_pos": joint_pos.numpy(),
        "body_pos_w": body_pos.numpy(),
        "body_quat_w": body_quat.numpy(),
    }
    start = 3
    source_qpos = joint_pos[start : start + 10].reshape(1, -1).float()
    source_pos = pelvis_pos[start : start + 10].clone()
    source_pos[:, 0] -= pelvis_pos[start, 0] - 2.0
    source_ori = _identity_rot6d(10).double()
    source = torch.cat(
        (
            source_qpos,
            source_pos.reshape(1, -1).float(),
            source_ori.reshape(1, -1).float(),
        ),
        dim=-1,
    )
    extended, validation = _extend_motion_rows(
        source_target=source,
        control_steps=torch.tensor([start]),
        row_indices=torch.tensor([0]).numpy(),
        trajectory=trajectory,
        dataset_attrs={
            "joint_names": list(TARGET_JOINT_NAMES),
            "body_names": ["pelvis"],
        },
        anchor_body_name="pelvis",
        horizon_steps=horizon,
        tolerance=1.0e-5,
    )
    assert extended.shape == (1, horizon * 38)
    assert validation["max_abs"] < 1.0e-6
    assert torch.equal(extended[:, : 10 * 29], source[:, : 10 * 29])


def test_materialize_promotes_one_target_without_changing_states() -> None:
    causal_state = torch.randn(2, 930)
    latent_target = torch.randn(2, 258)
    packet_target = torch.randn(2, 380)
    sample = {
        "causal_target": packet_target,
        "demonstration_target": packet_target,
        "planner_state": causal_state,
        "latent_skill_target": latent_target,
        "encoder_input_packet_target": packet_target,
        "metadata": {
            "interface": "root_qpos",
            "planner_interval_steps": 10,
            "paired_interface_target_specs": {
                "latent_skill_target": {
                    "interface": "latent_skill",
                    "term_names": ["z"],
                    "term_widths": [258],
                    "target_dim": 258,
                },
                "encoder_input_packet_target": {
                    "interface": "root_qpos",
                    "term_names": [name for name, _ in ROOT_QPOS_TERM_WIDTHS],
                    "term_widths": [290, 30, 60],
                    "target_dim": 380,
                },
            },
        },
    }

    result = _materialize_sample(
        sample,
        key="latent_skill_target",
        path=Path("sample_step_000000.pt"),
    )

    assert result["planner_state"] is causal_state
    assert torch.equal(result["causal_target"], latent_target)
    assert torch.equal(result["demonstration_target"], latent_target)
    assert result["metadata"]["interface"] == "latent_skill"
    assert result["metadata"]["command_future_steps"] == 10
    assert "encoder_input_packet_target" not in result
