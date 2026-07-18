"""Tests for the bounded frozen-skill latent scaling benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_latent_scale_benchmark import (  # noqa: E402
    ScaleConfig,
    build_train_command,
    dataset_metadata_fingerprint,
    parse_scale_config,
    parse_training_log,
    resolve_configs,
    validate_train_overrides,
)


def test_dataset_fingerprint_prunes_array_chunks(tmp_path: Path) -> None:
    (tmp_path / "zarr.json").write_text('{"node_type": "group"}')
    array_path = tmp_path / "trajectory" / "joint_pos"
    array_path.mkdir(parents=True)
    (array_path / "zarr.json").write_text('{"node_type": "array"}')
    fake_chunk_metadata = array_path / "c" / "0" / "zarr.json"
    fake_chunk_metadata.parent.mkdir(parents=True)
    fake_chunk_metadata.write_text('{"node_type": "group"}')

    identity = dataset_metadata_fingerprint(tmp_path)

    assert identity["file_count"] == 2
    assert [entry["path"] for entry in identity["files"]] == [
        "trajectory/joint_pos/zarr.json",
        "zarr.json",
    ]


def test_ridge_preserves_global_batch_and_optimizer_work() -> None:
    configs = resolve_configs("ridge", [])

    assert [config.global_rollout_batch for config in configs] == [98_304] * 5
    assert [config.optimizer_updates_per_rollout for config in configs] == [20] * 5


def test_remaining_preset_omits_only_a40_baseline() -> None:
    full = resolve_configs("a40-screen", [])
    remaining = resolve_configs("a40-remaining", [])

    assert len(full) == 10
    assert remaining == full[1:]


def test_confirmation_preset_contains_only_seed_zero_leaders() -> None:
    configs = resolve_configs("a40-confirm", [])

    assert [
        (config.num_envs, config.rollout_steps, config.minibatch_size)
        for config in configs
    ] == [
        (12_288, 8, 24_576),
        (8_192, 12, 12_288),
    ]


def test_production_preset_contains_only_validated_winner() -> None:
    configs = resolve_configs("a40-production", [])

    assert [
        (config.num_envs, config.rollout_steps, config.minibatch_size)
        for config in configs
    ] == [(8_192, 12, 12_288)]


def test_config_parser_rejects_invalid_or_oversized_minibatch() -> None:
    assert parse_scale_config("ok:4096:24:24576") == ScaleConfig("ok", 4096, 24, 24576)
    with pytest.raises(ValueError, match="Expected LABEL"):
        parse_scale_config("broken:4096:24")
    with pytest.raises(ValueError, match="exceeds"):
        parse_scale_config("broken:4:2:9")


def test_locked_profile_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="controlled key"):
        validate_train_overrides(["agent.ipmd.command_source=posterior"])
    validate_train_overrides(["agent.trainer.progress_bar=false"])


def test_command_pins_skill_source_and_requested_scaling(tmp_path: Path) -> None:
    args = argparse.Namespace(
        device="cuda:0",
        seed=3,
        log_interval_frames=5_000_000,
        run_label="test",
        skill_checkpoint=tmp_path / "skill.pt",
        manifest=tmp_path / "manifest.json",
        dataset_path=tmp_path / "dataset",
        train_override=[],
    )
    config = ScaleConfig("candidate", 8192, 12, 49_152)

    command = build_train_command(args, config, effective_frames=98_304 * 10)

    assert "--num_envs" in command
    assert command[command.index("--num_envs") + 1] == "8192"
    assert "agent.collector.frames_per_batch=12" in command
    assert "agent.loss.mini_batch_size=49152" in command
    assert "agent.ipmd.command_source=hl_skill" in command
    assert "agent.ipmd.hl_skill_finetune_enabled=false" in command
    assert "agent.logger.video=false" in command
    assert "--video" not in command


def test_log_parser_finds_first_and_sustained_threshold() -> None:
    log = """
[07/15/26 07:13:43] INFO iter=407/800 | frames=40000000/80000000 |
 r_step=0.03 | ep_len=170.0 | r_ep=5.64 |
 pi_loss=-0.01 | fps=24000.0
[07/15/26 07:20:52] INFO iter=509/800 | frames=50000000/80000000 |
 r_step=0.04 | ep_len=220.0 | r_ep=8.27 |
 pi_loss=-0.01 | fps=25000.0
[07/15/26 07:28:00] INFO iter=611/800 | frames=60000000/80000000 |
 r_step=0.04 | ep_len=250.0 | r_ep=9.44 |
 pi_loss=-0.01 | fps=26000.0
[07/15/26 07:35:10] INFO iter=713/800 | frames=70000000/80000000 |
 r_step=0.04 | ep_len=320.0 | r_ep=13.32 |
 pi_loss=-0.01 | fps=27000.0
[07/15/26 07:42:10] INFO iter=800/800 | frames=80000000/80000000 |
 r_step=0.04 | ep_len=330.0 | r_ep=14.00 |
 pi_loss=-0.01 | fps=28000.0
Training time: 1800.0 seconds
"""

    result = parse_training_log(log, target_return=7.5, sustain_points=3)

    assert result["first_hit"]["frames"] == 50_000_000
    assert result["sustained_start"]["frames"] == 50_000_000
    assert result["sustained_confirmed"]["frames"] == 70_000_000
    assert result["median_logged_fps"] == 26_000.0
    assert result["final_episode_return"] == 14.0
