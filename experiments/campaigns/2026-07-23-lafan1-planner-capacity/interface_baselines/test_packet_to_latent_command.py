from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from materialize_paired_interface_samples import _materialize_sample
from packet_to_latent_command import (
    PacketLayout,
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
