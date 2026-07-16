from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest
import torch
from rlopt.agent.causal_interface_planner import (
    CausalInterfaceTransformerCategoricalPlanner,
)

from interface_planner_common import (
    InterfaceTargetSpec,
    load_planner_checkpoint,
    load_rollout_samples,
    save_planner_checkpoint,
)
from planner_sample_schema import (
    PlannerSampleWriter,
    add_sample_format_metadata,
    build_planner_sample,
    concatenate_planner_samples,
)


def _metadata() -> dict:
    return add_sample_format_metadata(
        {
            "interface": "latent_skill",
            "planner_observation_spec": {
                "history_frames": 10,
                "frame_dim": 93,
                "flat_dim": 930,
            },
        },
        collection_stage="demonstration",
        planner_interval_steps=10,
    )


def test_common_sample_keeps_unflattened_history_and_compatibility_aliases() -> None:
    causal = torch.randn(2, 10, 93)
    demonstration = torch.randn(2, 10, 93)
    causal_target = torch.randn(2, 256)
    demonstration_target = torch.randn(2, 256)
    sample = build_planner_sample(
        causal_state_history=causal,
        demonstration_state_history=demonstration,
        causal_target=causal_target,
        demonstration_target=demonstration_target,
        trajectory_rank=torch.tensor([3, 4]),
        episode_id=torch.tensor([7, 8]),
        control_step=torch.tensor([20, 30]),
        planner_step=torch.tensor([2, 3]),
        motion_names=["walk", "dance"],
        metadata=_metadata(),
    )
    assert sample["causal_state_history"].shape == (2, 10, 93)
    assert sample["planner_state"].shape == (2, 930)
    assert torch.equal(sample["causal_target"], causal_target)
    assert torch.equal(sample["demonstration_target"], demonstration_target)
    assert "target" not in sample
    assert sample["metadata"]["planner_rate_hz"] == 5.0


def test_sample_writer_chunks_rows_without_changing_values(tmp_path: Path) -> None:
    first = build_planner_sample(
        causal_state_history=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        demonstration_state_history=torch.ones(2, 3, 4),
        causal_target=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        demonstration_target=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        trajectory_rank=torch.tensor([0, 1]),
        episode_id=torch.tensor([0, 0]),
        control_step=torch.tensor([0, 10]),
        planner_step=torch.tensor([0, 1]),
        motion_names=["a", "b"],
        metadata=add_sample_format_metadata(
            {
                "interface": "latent_skill",
                "planner_observation_spec": {
                    "history_frames": 3,
                    "frame_dim": 4,
                    "flat_dim": 12,
                },
            },
            collection_stage="planner_rollout",
            planner_interval_steps=10,
        ),
    )
    second = build_planner_sample(
        causal_state_history=torch.full((1, 3, 4), 9.0),
        demonstration_state_history=torch.full((1, 3, 4), 8.0),
        causal_target=torch.tensor([[9.0, 10.0]]),
        demonstration_target=torch.tensor([[11.0, 12.0]]),
        trajectory_rank=torch.tensor([2]),
        episode_id=torch.tensor([1]),
        control_step=torch.tensor([20]),
        planner_step=torch.tensor([2]),
        motion_names=["c"],
        metadata=first["metadata"],
    )

    joined = concatenate_planner_samples([first, second])
    assert joined["motion_name"] == ["a", "b", "c"]
    assert joined["causal_target"].tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [9.0, 10.0],
    ]

    writer = PlannerSampleWriter(tmp_path / "samples", rows_per_file=3)
    writer.add(first)
    assert writer.file_count == 0
    writer.add(second)
    writer.flush()
    assert writer.file_count == 1
    assert writer.row_count == 3
    saved = torch.load(
        tmp_path / "samples/sample_step_000000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["motion_name"] == ["a", "b", "c"]
    assert torch.equal(saved["causal_target"], joined["causal_target"])


def test_common_sample_rejects_wrong_history_width() -> None:
    with pytest.raises(ValueError, match="flat width"):
        build_planner_sample(
            causal_state_history=torch.randn(1, 929),
            demonstration_state_history=torch.randn(1, 930),
            causal_target=torch.randn(1, 4),
            demonstration_target=torch.randn(1, 4),
            trajectory_rank=torch.tensor([0]),
            episode_id=0,
            control_step=0,
            planner_step=0,
            motion_names=["walk"],
            metadata=_metadata(),
        )


def test_common_sample_preserves_categorical_token_targets() -> None:
    metadata = _metadata()
    metadata["interface"] = "per_step_token_sequence"
    metadata["target_encoding"] = {
        "kind": "categorical_sequence",
        "horizon": 10,
        "codebook_size": 512,
    }
    tokens = torch.randint(0, 512, (2, 10))
    sample = build_planner_sample(
        causal_state_history=torch.randn(2, 10, 93),
        demonstration_state_history=torch.randn(2, 10, 93),
        causal_target=tokens,
        demonstration_target=tokens,
        trajectory_rank=torch.tensor([0, 1]),
        episode_id=torch.tensor([0, 0]),
        control_step=torch.tensor([0, 10]),
        planner_step=torch.tensor([0, 1]),
        motion_names=["walk", "dance"],
        metadata=metadata,
    )

    assert sample["causal_target"].dtype == torch.long
    assert sample["demonstration_target"].dtype == torch.long
    assert torch.equal(sample["causal_target"], tokens)


def test_language_conditioned_sample_roundtrip(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["target_spec"] = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [4],
        "target_dim": 4,
    }
    metadata["language_conditioning"] = {
        "enabled": True,
        "embedding_dim": 5,
        "embedding_path": "/tmp/test_language.pt",
    }
    language = torch.randn(2, 5)
    sample = build_planner_sample(
        causal_state_history=torch.randn(2, 10, 93),
        demonstration_state_history=torch.randn(2, 10, 93),
        causal_target=torch.randn(2, 4),
        demonstration_target=torch.randn(2, 4),
        trajectory_rank=torch.tensor([0, 1]),
        episode_id=torch.tensor([0, 0]),
        control_step=torch.tensor([0, 10]),
        planner_step=torch.tensor([0, 1]),
        motion_names=["walk", "dance"],
        metadata=metadata,
        language_embedding=language,
    )
    samples_dir = tmp_path / "language_samples"
    samples_dir.mkdir()
    torch.save(sample, samples_dir / "sample_step_000000.pt")

    data, loaded_metadata = load_rollout_samples(samples_dir)

    assert torch.equal(data["language_embedding"], language)
    assert loaded_metadata["language_conditioning"]["embedding_dim"] == 5


def test_state_only_sample_rejects_zero_width_language_tensor() -> None:
    metadata = _metadata()
    metadata["target_spec"] = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [4],
        "target_dim": 4,
    }
    metadata["language_conditioning"] = {
        "enabled": False,
        "embedding_dim": 0,
    }
    with pytest.raises(ValueError, match="omit it for a state-only"):
        build_planner_sample(
            causal_state_history=torch.randn(2, 10, 93),
            demonstration_state_history=torch.randn(2, 10, 93),
            causal_target=torch.randn(2, 4),
            demonstration_target=torch.randn(2, 4),
            trajectory_rank=torch.tensor([0, 0]),
            episode_id=torch.tensor([0, 0]),
            control_step=torch.tensor([0, 10]),
            planner_step=torch.tensor([0, 1]),
            motion_names=["walk", "walk"],
            metadata=metadata,
            language_embedding=torch.empty(2, 0),
        )


def test_language_samples_merge_across_explicit_goals(tmp_path: Path) -> None:
    base_metadata = _metadata()
    base_metadata["target_spec"] = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [4],
        "target_dim": 4,
    }
    base_metadata["language_conditioning"] = {
        "enabled": True,
        "embedding_dim": 5,
        "embedding_path": "/tmp/shared_language.pt",
        "embedding_sha256": "a" * 64,
        "backend": "test",
        "model": "test",
    }
    samples_dir = tmp_path / "multi_goal_samples"
    samples_dir.mkdir()
    for index, goal_name in enumerate(("walk", "kick")):
        metadata = copy.deepcopy(base_metadata)
        metadata["language_conditioning"].update(
            {
                "goal_name": goal_name,
                "goal_phrase": f"do {goal_name}",
                "motion_count": index + 1,
            }
        )
        sample = build_planner_sample(
            causal_state_history=torch.randn(1, 10, 93),
            demonstration_state_history=torch.randn(1, 10, 93),
            causal_target=torch.randn(1, 4),
            demonstration_target=torch.randn(1, 4),
            trajectory_rank=torch.tensor([index]),
            episode_id=index,
            control_step=index * 10,
            planner_step=index,
            motion_names=[goal_name],
            metadata=metadata,
            language_embedding=torch.randn(1, 5),
        )
        torch.save(sample, samples_dir / f"sample_step_{index:06d}.pt")

    data, _ = load_rollout_samples(samples_dir)
    assert data["planner_state"].shape[0] == 2
    assert data["language_embedding"].shape == (2, 5)

    incompatible = torch.load(
        samples_dir / "sample_step_000001.pt", map_location="cpu", weights_only=False
    )
    incompatible["metadata"]["language_conditioning"]["embedding_sha256"] = "b" * 64
    torch.save(incompatible, samples_dir / "sample_step_000001.pt")
    with pytest.raises(ValueError, match="metadata does not match"):
        load_rollout_samples(samples_dir)


def test_categorical_samples_and_checkpoint_roundtrip(tmp_path, monkeypatch) -> None:
    metadata = _metadata()
    metadata["interface"] = "per_step_token_sequence"
    metadata["target_spec"] = {
        "interface": "per_step_token_sequence",
        "term_names": ["token_ids"],
        "term_widths": [10],
        "target_dim": 10,
    }
    metadata["target_encoding"] = {
        "kind": "categorical_sequence",
        "horizon": 10,
        "codebook_size": 8,
    }
    tokens = torch.randint(0, 8, (3, 10))
    sample = build_planner_sample(
        causal_state_history=torch.randn(3, 10, 93),
        demonstration_state_history=torch.randn(3, 10, 93),
        causal_target=tokens,
        demonstration_target=tokens,
        trajectory_rank=torch.tensor([0, 1, 2]),
        episode_id=torch.tensor([0, 0, 0]),
        control_step=torch.tensor([0, 10, 20]),
        planner_step=torch.tensor([0, 1, 2]),
        motion_names=["a", "b", "c"],
        metadata=metadata,
    )
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    torch.save(sample, samples_dir / "sample_step_000000.pt")

    data, loaded_metadata = load_rollout_samples(samples_dir)
    assert data["causal_target"].dtype == torch.long
    assert data["demonstration_target"].dtype == torch.long
    assert torch.equal(data["causal_target"], tokens)
    assert loaded_metadata["target_encoding"] == metadata["target_encoding"]

    planner = CausalInterfaceTransformerCategoricalPlanner(
        state_dim=930,
        token_horizon=10,
        codebook_size=8,
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        num_state_tokens=1,
    )
    checkpoint_path = tmp_path / "run" / "checkpoints" / "latest.pt"
    save_planner_checkpoint(
        checkpoint_path,
        planner=planner,
        optimizer=None,
        target_spec=InterfaceTargetSpec.from_dict(metadata["target_spec"]),
        metadata=metadata,
    )
    loaded, target_spec, checkpoint_metadata = load_planner_checkpoint(checkpoint_path)
    assert isinstance(loaded, CausalInterfaceTransformerCategoricalPlanner)
    assert target_spec.interface == "per_step_token_sequence"
    assert checkpoint_metadata["target_encoding"] == metadata["target_encoding"]

    from train_categorical_token_planner import main as train_categorical_main

    train_dir = tmp_path / "trained"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_categorical_token_planner.py",
            "--samples_dir",
            str(samples_dir),
            "--output_dir",
            str(train_dir),
            "--state_key",
            "expert_planner_state",
            "--device",
            "cpu",
            "--model_size",
            "tiny",
            "--batch_size",
            "2",
            "--num_updates",
            "2",
            "--log_interval",
            "1",
        ],
    )
    train_categorical_main()
    assert (train_dir / "checkpoints" / "latest.pt").is_file()


def test_phase3_audit_accepts_match_and_rejects_backbone_mismatch(tmp_path) -> None:
    from audit_phase3_latent_interfaces import audit

    observation_spec = {
        "history_steps": 9,
        "history_frames": 10,
        "frame_dim": 93,
        "flat_dim": 930,
        "reference_features": [],
    }
    sample_base = {
        "sample_format": {
            "name": "causal_interface_planner_sample",
            "version": 1,
        },
        "planner_observation_spec": observation_spec,
        "state_history_steps": 9,
        "planner_interval_steps": 10,
        "control_rate_hz": 50.0,
        "seed": 7,
    }
    common_planner = {
        "state_dim": 930,
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 4,
        "feedforward_dim": 128,
        "num_state_tokens": 1,
        "dropout": 0.0,
    }

    paths: dict[str, Path] = {}
    for prefix, interface in (
        ("future", "future_cvae"),
        ("token", "per_step_token_sequence"),
    ):
        directory = tmp_path / prefix
        directory.mkdir()
        encoding = (
            {}
            if interface == "future_cvae"
            else {
                "target_encoding": {
                    "kind": "categorical_sequence",
                    "horizon": 10,
                    "codebook_size": 512,
                }
            }
        )
        planner_config = {
            **common_planner,
            **(
                {
                    "planner_type": "causal_interface_transformer_flow",
                    "target_dim": 256,
                    "patch_dim": 16,
                }
                if interface == "future_cvae"
                else {
                    "planner_type": "causal_interface_transformer_categorical",
                    "token_horizon": 10,
                    "codebook_size": 512,
                }
            ),
        }
        target_spec = {
            "interface": interface,
            "term_names": ["z" if interface == "future_cvae" else "token_ids"],
            "term_widths": [256 if interface == "future_cvae" else 10],
            "target_dim": 256 if interface == "future_cvae" else 10,
        }
        checkpoint_path = directory / "checkpoint.pt"
        torch.save(
            {
                "planner_config": planner_config,
                "target_spec": target_spec,
                "metadata": {
                    "pretrain_num_updates": 3,
                    "finetune_num_updates": 4,
                    "sample_metadata": {
                        **sample_base,
                        "interface": interface,
                        "target_spec": target_spec,
                        **encoding,
                    },
                },
            },
            checkpoint_path,
        )
        paths[f"{prefix}_checkpoint"] = checkpoint_path

        merge_path = directory / "merge.json"
        merge_path.write_text(
            json.dumps(
                {
                    "row_count": 10,
                    "sources": [
                        {
                            "collection_stage": "oracle_rollout",
                            "selected_row_count": 5,
                        },
                        {
                            "collection_stage": "planner_rollout",
                            "selected_row_count": 5,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths[f"{prefix}_merge_manifest"] = merge_path
        for name, stage in (
            ("oracle_summary", "oracle_rollout"),
            ("rollout_summary", "planner_rollout"),
        ):
            summary_path = directory / f"{name}.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "interface": interface,
                            "collection_stage": stage,
                            "planner_observation_spec": observation_spec,
                        },
                        "saved_rows": 8,
                        "control_steps": 40,
                    }
                ),
                encoding="utf-8",
            )
            paths[f"{prefix}_{name}"] = summary_path

        common_metrics = {
            metric_name: {"mean": 0.1, "std": 0.0, "count": 8}
            for metric_name in (
                "action_l2",
                "root_height_error_m",
                "root_ori_error_rad",
                "tracking_mpjpe_mm",
                "tracking_velocity_distance_mps",
            )
        }
        for name, stage in (
            ("oracle_eval_summary", "oracle_rollout"),
            ("pretrained_eval_summary", "planner_rollout"),
            ("finetuned_eval_summary", "planner_rollout"),
        ):
            planner_metrics = {}
            if stage == "planner_rollout":
                planner_metrics[
                    "planner_target_rmse"
                    if interface == "future_cvae"
                    else "planner_token_accuracy"
                ] = {"mean": 0.2, "std": 0.0, "count": 2}
            summary_path = directory / f"{name}.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "interface": interface,
                            "collection_stage": stage,
                        },
                        "aggregate": {
                            "survival_steps_mean": 12.0,
                            "done_rate": 1.0,
                            "tracking_success_rate": 0.5,
                        },
                        "metrics": {**common_metrics, **planner_metrics},
                        "max_steps": 20,
                        "steps_run": 12,
                        "evaluation_only": True,
                        "stop_after_done": True,
                        "saved_rows": 0,
                    }
                ),
                encoding="utf-8",
            )
            paths[f"{prefix}_{name}"] = summary_path

    args = argparse.Namespace(
        **paths,
        expected_seed=7,
        expected_history_steps=9,
        expected_planner_interval=10,
        expected_pretrain_updates=3,
        expected_finetune_updates=4,
        expected_rows_per_stage=5,
        expected_collected_rows_per_stage=8,
        expected_collection_control_steps=40,
        expected_eval_control_steps=20,
        expected_token_horizon=10,
        expected_codebook_size=512,
    )
    result = audit(args)
    assert result["passed"] is True

    token_checkpoint = torch.load(
        paths["token_checkpoint"], map_location="cpu", weights_only=False
    )
    token_checkpoint["planner_config"]["d_model"] = 32
    torch.save(token_checkpoint, paths["token_checkpoint"])
    mismatch = audit(args)
    assert mismatch["passed"] is False
    assert any("backbone d_model" in error for error in mismatch["errors"])
