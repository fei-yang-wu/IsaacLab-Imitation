#!/usr/bin/env python3
"""Pure-Python smoke tests for interface planner utilities."""

from __future__ import annotations

import csv
import json
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from rlopt.agent.skill_commander import SkillCommander

from aggregate_interface_comparison_seeds import PAPER_METRICS, _seed_for_root
from analyze_interface_sweep import (
    REQUIRED_SELECTED_METRICS,
    analyze_sweep,
    selected_sweep_rows,
    validate_selected_interfaces,
    validate_selected_metrics,
)
from audit_interface_comparison import (
    REQUIRED_AGGREGATE_METRICS,
    REQUIRED_BY_SEED_METRICS,
    REQUIRED_OFFLINE_METRICS,
    audit_aggregate_dir,
)
from backfill_planner_capacity_metadata import _backfill_root
from eval_interface_planner_offline import evaluate_planner_checkpoint
from eval_latent_skill_planner_offline import (
    _load_samples as load_latent_rollout_samples,
)
from eval_latent_skill_planner_offline import evaluate_latent_skill_planner_checkpoint
from interface_planner_common import (
    ChunkedTransformerFlowPlanner,
    InterfaceFlowPlanner,
    flatten_command_terms,
    load_planner_checkpoint,
    load_rollout_samples,
    parameter_counts,
    rmse_per_row,
    save_planner_checkpoint,
    unflatten_command_target,
)
from preflight_interface_comparison import run_preflight
from split_lafan1_manifest import _rebase_manifest_paths, split_manifest
from summarize_interface_comparison import (
    _row,
    _sample_count_from_config,
    _sample_count_from_samples_dir,
    _sample_count_from_summary,
    _wide_rows,
)


def _test_target_roundtrip() -> None:
    terms = {
        "expert_ee_pos_b": torch.randn(4, 12),
        "expert_ee_ori_b": torch.randn(4, 24),
    }
    flat, spec = flatten_command_terms("ee_trajectory", terms)
    restored = unflatten_command_target(flat, spec)
    assert list(restored) == ["expert_ee_pos_b", "expert_ee_ori_b"]
    for key, value in terms.items():
        assert torch.allclose(restored[key], value)


def _test_samples_and_checkpoint(tmp: Path) -> None:
    terms = {
        "expert_motion": torch.randn(3, 58),
        "expert_anchor_pos_b": torch.randn(3, 3),
        "expert_anchor_ori_b": torch.randn(3, 6),
    }
    target, spec = flatten_command_terms("full_body_trajectory", terms)
    sample = {
        "step": 0,
        "planner_state": torch.randn(3, 67),
        "expert_planner_state": torch.randn(3, 67),
        "target": target,
        "traj_rank": torch.zeros(3, dtype=torch.long),
        "metadata": {
            "interface": "full_body_trajectory",
            "target_spec": spec.to_dict(),
        },
    }
    torch.save(sample, tmp / "sample_step_000000.pt")
    data, metadata = load_rollout_samples(tmp)
    assert data["planner_state"].shape == (3, 67)
    assert data["target"].shape == (3, spec.target_dim)
    assert data["traj_rank"].shape == (3,)
    assert data["step"].tolist() == [0, 0, 0]
    assert metadata["target_spec"]["target_dim"] == spec.target_dim

    missing_metadata_dir = tmp / "missing_metadata"
    missing_metadata_dir.mkdir()
    broken_sample = dict(sample)
    broken_sample.pop("metadata")
    torch.save(broken_sample, missing_metadata_dir / "sample_step_000000.pt")
    try:
        load_rollout_samples(missing_metadata_dir)
    except KeyError as exc:
        assert "metadata" in str(exc)
    else:
        raise AssertionError("load_rollout_samples accepted a sample without metadata.")

    mismatched_metadata_dir = tmp / "mismatched_metadata"
    mismatched_metadata_dir.mkdir()
    torch.save(sample, mismatched_metadata_dir / "sample_step_000000.pt")
    other_sample = dict(sample)
    other_sample["step"] = 1
    other_sample["metadata"] = {
        "interface": "full_body_trajectory",
        "target_spec": spec.to_dict(),
        "seed": 123,
    }
    torch.save(other_sample, mismatched_metadata_dir / "sample_step_000001.pt")
    try:
        load_rollout_samples(mismatched_metadata_dir)
    except ValueError as exc:
        assert "metadata does not match" in str(exc)
    else:
        raise AssertionError("load_rollout_samples accepted mixed sample metadata.")

    planner = InterfaceFlowPlanner(
        state_dim=67, target_dim=spec.target_dim, hidden_dims=(32,)
    )
    assert parameter_counts(planner)["parameter_count"] > 0
    optimizer = torch.optim.AdamW(planner.parameters(), lr=1.0e-3)
    ckpt = tmp / "planner" / "checkpoints" / "latest.pt"
    save_planner_checkpoint(
        ckpt,
        planner=planner,
        optimizer=optimizer,
        target_spec=spec,
        metadata={"smoke": True},
    )
    loaded, loaded_spec, metadata = load_planner_checkpoint(ckpt)
    assert loaded_spec == spec
    assert metadata["smoke"] is True
    pred = loaded(torch.randn(2, 67), num_inference_steps=2)
    assert pred.shape == (2, spec.target_dim)

    offline_summary = evaluate_planner_checkpoint(
        samples_dir=tmp,
        planner_checkpoint=ckpt,
        output_json=tmp / "offline_eval" / "summary.json",
        output_csv=tmp / "offline_eval" / "summary.csv",
        interface="full_body_trajectory",
        state_key="expert_planner_state",
        setting="eval_pretrained_expert_state",
        batch_size=2,
        flow_num_inference_steps=2,
    )
    assert offline_summary["metadata"]["setting"] == "eval_pretrained_expert_state"
    assert offline_summary["aggregate"]["sample_count"] == 3
    assert "planner_target_rmse" in offline_summary["metrics"]
    assert "expert_motion_rmse" in offline_summary["metrics"]


def _test_chunked_transformer_checkpoint(tmp: Path) -> None:
    torch.manual_seed(11)
    terms = {
        "expert_ee_pos_b": torch.randn(5, 12),
        "expert_ee_ori_b": torch.randn(5, 24),
    }
    target, spec = flatten_command_terms("ee_trajectory", terms)
    state = torch.randn(5, 9)
    planner = ChunkedTransformerFlowPlanner(
        state_dim=9,
        target_dim=spec.target_dim,
        term_widths=spec.term_widths,
        d_model=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        patch_dim=8,
        num_state_tokens=1,
    )
    planner.set_normalization(
        state_mean=state.mean(dim=0),
        state_std=state.std(dim=0, unbiased=False),
        target_mean=target.mean(dim=0),
        target_std=target.std(dim=0, unbiased=False),
    )
    pred = planner(state, num_inference_steps=2)
    assert pred.shape == target.shape
    loss = planner.flow_matching_loss(state, target)
    assert torch.isfinite(loss)

    optimizer = torch.optim.AdamW(planner.parameters(), lr=1.0e-3)
    ckpt = tmp / "chunked" / "checkpoints" / "latest.pt"
    save_planner_checkpoint(
        ckpt,
        planner=planner,
        optimizer=optimizer,
        target_spec=spec,
        metadata={"planner_type": "chunked_transformer_flow"},
    )
    loaded, loaded_spec, metadata = load_planner_checkpoint(ckpt)
    assert loaded_spec == spec
    assert metadata["planner_type"] == "chunked_transformer_flow"
    loaded_pred = loaded(state, num_inference_steps=2)
    assert loaded_pred.shape == target.shape


def _test_chunked_transformer_microbatch_training(tmp: Path) -> None:
    torch.manual_seed(13)
    samples_dir = tmp / "microbatch_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    terms = {
        "expert_ee_pos_b": torch.randn(1, 6),
        "expert_ee_ori_b": torch.randn(1, 12),
    }
    _, spec = flatten_command_terms("ee_trajectory", terms)
    for step in range(4):
        target, _ = flatten_command_terms(
            "ee_trajectory",
            {
                "expert_ee_pos_b": torch.randn(1, 6),
                "expert_ee_ori_b": torch.randn(1, 12),
            },
        )
        torch.save(
            {
                "step": step,
                "planner_state": torch.randn(1, 9),
                "expert_planner_state": torch.randn(1, 9),
                "target": target,
                "traj_rank": torch.zeros(1, dtype=torch.long),
                "metadata": {
                    "interface": "ee_trajectory",
                    "target_spec": spec.to_dict(),
                },
            },
            samples_dir / f"sample_step_{step:06d}.pt",
        )

    output_dir = tmp / "microbatch_planner"
    subprocess.run(
        [
            sys.executable,
            "experiments/interface_baselines/train_chunked_transformer_planner.py",
            "--samples_dir",
            str(samples_dir),
            "--output_dir",
            str(output_dir),
            "--interface",
            "ee_trajectory",
            "--state_key",
            "expert_planner_state",
            "--model_size",
            "tiny",
            "--num_updates",
            "1",
            "--batch_size",
            "4",
            "--micro_batch_size",
            "2",
            "--eval_batch_size",
            "2",
            "--eval_max_samples",
            "4",
            "--flow_num_inference_steps",
            "2",
            "--flow_inference_noise_std",
            "0.125",
            "--endpoint_num_inference_steps",
            "1",
            "--log_interval",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["batch_size"] == 4
    assert config["micro_batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 2
    assert config["flow_num_inference_steps"] == 2
    assert config["flow_inference_noise_std"] == 0.125
    assert config["endpoint_num_inference_steps"] == 1
    assert config["pretrain_num_updates"] == 1
    assert config["finetune_num_updates"] is None
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    checkpoint = torch.load(
        output_dir / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["metadata"]["flow_inference_noise_std"] == 0.125
    assert checkpoint["metadata"]["pretrain_num_updates"] == 1
    assert checkpoint["metadata"]["finetune_num_updates"] is None
    assert (output_dir / "metrics.jsonl").is_file()


def _test_latent_skill_offline_eval(tmp: Path) -> None:
    torch.manual_seed(17)
    samples_dir = tmp / "latent_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    state = torch.randn(4, 5)
    expert_state = torch.randn(4, 5)
    lang = torch.randn(4, 3)
    z_target = torch.randn(4, 4)
    torch.save(
        {
            "step": 0,
            "planner_state": state,
            "expert_planner_state": expert_state,
            "lang": lang,
            "z_target": z_target,
            "traj_rank": torch.zeros(4, dtype=torch.long),
        },
        samples_dir / "sample_step_000000.pt",
    )
    latent_samples = load_latent_rollout_samples(samples_dir)
    assert latent_samples["traj_rank"].shape == (4,)
    assert latent_samples["step"].tolist() == [0, 0, 0, 0]
    generator = SkillCommander(
        state_dim=5,
        lang_embed_dim=3,
        z_dim=4,
        hidden_dims=(16,),
    )
    checkpoint = tmp / "latent_planner.pt"
    torch.save(
        {
            "config": {
                "planner_type": "mlp",
                "generator_hidden_dims": [16],
            },
            "generator_state_dict": generator.state_dict(),
            "update": 7,
        },
        checkpoint,
    )
    summary = evaluate_latent_skill_planner_checkpoint(
        samples_dir=samples_dir,
        planner_checkpoint=checkpoint,
        output_json=tmp / "latent_eval" / "summary.json",
        output_csv=tmp / "latent_eval" / "summary.csv",
        state_key="expert_planner_state",
        setting="eval_pretrained_expert_state",
        batch_size=2,
    )
    assert summary["metadata"]["interface"] == "latent_skill"
    assert summary["metadata"]["planner_target_dim"] == 4
    assert summary["aggregate"]["sample_count"] == 4
    assert "planner_target_rmse" in summary["metrics"]


def _test_flow_training_reduces_error() -> None:
    torch.manual_seed(5)
    state = torch.randn(64, 8)
    projection = torch.randn(8, 5) * 0.25
    target = state @ projection
    planner = InterfaceFlowPlanner(state_dim=8, target_dim=5, hidden_dims=(64, 64))
    optimizer = torch.optim.AdamW(planner.parameters(), lr=3.0e-3)
    with torch.no_grad():
        initial = rmse_per_row(planner(state, num_inference_steps=4), target).mean()
    for _ in range(80):
        loss = planner.flow_matching_loss(state, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = rmse_per_row(planner(state, num_inference_steps=4), target).mean()
    assert float(final.item()) < float(initial.item()), (initial.item(), final.item())


def _test_summary_config_sample_count(tmp: Path) -> None:
    json_config = tmp / "config.json"
    json_config.write_text('{"num_samples": 17}\n', encoding="utf-8")
    assert _sample_count_from_config(json_config) == 17

    yaml_config = tmp / "config.yaml"
    yaml_config.write_text("metadata:\n  num_samples: 23\n", encoding="utf-8")
    assert _sample_count_from_config(yaml_config) == 23

    samples_dir = tmp / "sample_count" / "rollout_training_samples"
    samples_dir.mkdir(parents=True)
    torch.save(
        {"planner_state": torch.randn(3, 5)},
        samples_dir / "sample_step_000000.pt",
    )
    assert _sample_count_from_samples_dir(samples_dir) == 3

    summary_path = samples_dir.parent / "summary.json"
    summary_path.write_text('{"saved_rows": 7, "saved_steps": 1}\n', encoding="utf-8")
    assert _sample_count_from_samples_dir(samples_dir) == 7

    summary_path.write_text(
        '{"saved_steps": 2, "metadata": {"num_envs": 4}}\n',
        encoding="utf-8",
    )
    assert _sample_count_from_samples_dir(samples_dir) == 8

    latent_summary = tmp / "latent_oracle_drive_samples" / "summary.json"
    latent_summary.parent.mkdir(parents=True)
    latent_summary.write_text(
        '{"num_metric_rows": 2, "saved_rows": 6, "sample_file_count": 2}\n',
        encoding="utf-8",
    )
    assert _sample_count_from_summary(latent_summary) == 6


def _test_summary_offline_eval_rows(tmp: Path) -> None:
    root = tmp / "results"
    path = (
        root
        / "ee_trajectory"
        / "chunked_transformer_medium_100"
        / "eval_pretrained_expert_state"
        / "summary.json"
    )
    payload = {
        "metadata": {
            "interface": "ee_trajectory",
            "planner_target_dim": 936,
            "planner_metadata": {
                "parameter_count": 1234,
                "state_dim": 67,
                "batch_size": 256,
                "micro_batch_size": 32,
                "gradient_accumulation_steps": 8,
                "num_updates": 2000,
                "pretrain_num_updates": 2000,
                "lr": 1.0e-4,
                "weight_decay": 1.0e-4,
                "flow_num_inference_steps": 16,
                "flow_inference_noise_std": 0.0,
                "endpoint_num_inference_steps": 4,
                "sample_metadata": {
                    "state_history_steps": 0,
                    "command_past_steps": 0,
                    "command_future_steps": 25,
                },
            },
        },
        "aggregate": {"sample_count": 4},
        "metrics": {
            "planner_target_rmse": {"mean": 0.2, "std": 0.0, "count": 4},
            "planner_normalized_target_rmse": {
                "mean": 0.5,
                "std": 0.0,
                "count": 4,
            },
        },
    }
    row = _row(path, payload, root)
    assert row["setting"] == "eval_pretrained_expert_state"
    assert row["planner_variant"] == "chunked_transformer_medium_100"
    assert row["planner_state_dim"] == 67
    assert row["planner_state_history_steps"] == 0
    assert row["planner_command_past_steps"] == 0
    assert row["planner_command_future_steps"] == 25
    assert row["planner_batch_size"] == 256
    assert row["planner_micro_batch_size"] == 32
    assert row["planner_gradient_accumulation_steps"] == 8
    assert row["planner_num_updates"] == 2000
    assert row["planner_pretrain_num_updates"] == 2000
    assert row["planner_finetune_num_updates"] == ""
    assert row["planner_lr"] == 1.0e-4
    assert row["planner_weight_decay"] == 1.0e-4
    assert row["planner_flow_num_inference_steps"] == 16
    assert row["planner_flow_inference_noise_std"] == 0.0
    assert row["planner_endpoint_num_inference_steps"] == 4
    wide = _wide_rows([row])
    assert wide[0]["planner_state_dim"] == 67
    assert wide[0]["planner_state_history_steps"] == 0
    assert wide[0]["planner_command_past_steps"] == 0
    assert wide[0]["planner_command_future_steps"] == 25
    assert wide[0]["planner_param_count"] == 1234
    assert wide[0]["planner_batch_size"] == 256
    assert wide[0]["planner_micro_batch_size"] == 32
    assert wide[0]["planner_gradient_accumulation_steps"] == 8
    assert wide[0]["planner_num_updates"] == 2000
    assert wide[0]["planner_pretrain_num_updates"] == 2000
    assert wide[0]["planner_finetune_num_updates"] == ""
    assert wide[0]["planner_lr"] == 1.0e-4
    assert wide[0]["planner_weight_decay"] == 1.0e-4
    assert wide[0]["planner_flow_num_inference_steps"] == 16
    assert wide[0]["planner_flow_inference_noise_std"] == 0.0
    assert wide[0]["planner_endpoint_num_inference_steps"] == 4
    assert wide[0]["pretrained_expert_target_rmse"] == 0.2
    assert wide[0]["pretrained_expert_normalized_target_rmse"] == 0.5
    assert wide[0]["pretrained_expert_eval_sample_count"] == 4

    merged = _wide_rows(
        [
            {
                "setting": "eval_finetuned_closed_loop",
                "interface": "ee_trajectory",
                "planner_variant": "chunked_transformer_medium_100",
                "planner_target_dim": 36,
                "planner_num_updates": 2000,
                "planner_finetune_num_updates": 2000,
                "planner_lr": 1.0e-4,
                "finetune_sample_count": 100,
            },
            {
                "setting": "eval_pretrained_closed_loop",
                "interface": "ee_trajectory",
                "planner_variant": "chunked_transformer_medium_100",
                "planner_target_dim": 36,
                "planner_num_updates": 777,
                "planner_pretrain_num_updates": 777,
                "planner_lr": 3.0e-4,
                "finetune_sample_count": 100,
            },
        ]
    )
    assert merged[0]["planner_num_updates"] == 2000
    assert merged[0]["planner_pretrain_num_updates"] == 777
    assert merged[0]["planner_finetune_num_updates"] == 2000
    assert merged[0]["planner_lr"] == 1.0e-4

    legacy_args_payload = {
        "metadata": {
            "interface": "ee_trajectory",
            "planner_target_dim": 36,
            "planner_metadata": {
                "parameter_count": 1234,
                "num_samples": 100,
                "checkpoint_metadata": {"args": {"num_updates": 777}},
                "sample_metadata": {
                    "state_history_steps": 10,
                    "command_past_steps": 1,
                    "command_future_steps": 25,
                },
                "args": {
                    "state_dim": 670,
                    "batch_size": 256,
                    "micro_batch_size": 32,
                    "gradient_accumulation_steps": 8,
                    "num_updates": 2000,
                    "lr": 1.0e-4,
                    "weight_decay": 1.0e-4,
                    "flow_num_inference_steps": 16,
                    "flow_inference_noise_std": 0.0,
                    "endpoint_num_inference_steps": 4,
                    "model_size": "medium",
                },
            },
        },
        "aggregate": {"sample_count": 4},
        "metrics": {},
    }
    legacy_args_row = _row(
        root
        / "ee_trajectory"
        / "chunked_transformer_medium_100"
        / "eval_finetuned_closed_loop"
        / "summary.json",
        legacy_args_payload,
        root,
    )
    assert legacy_args_row["planner_model_size"] == "medium"
    assert legacy_args_row["planner_state_dim"] == 670
    assert legacy_args_row["planner_state_history_steps"] == 10
    assert legacy_args_row["planner_command_past_steps"] == 1
    assert legacy_args_row["planner_command_future_steps"] == 25
    assert legacy_args_row["planner_batch_size"] == 256
    assert legacy_args_row["planner_num_updates"] == 2000
    assert legacy_args_row["planner_pretrain_num_updates"] == 777
    assert legacy_args_row["planner_finetune_num_updates"] == 2000
    assert legacy_args_row["planner_lr"] == 1.0e-4
    assert legacy_args_row["planner_weight_decay"] == 1.0e-4
    assert legacy_args_row["planner_flow_num_inference_steps"] == 16
    assert legacy_args_row["planner_flow_inference_noise_std"] == 0.0


def _test_summary_success_rates(tmp: Path) -> None:
    root = tmp / "results"
    rows = []
    for setting, done_rate in (
        ("oracle_low_level", 0.0),
        ("eval_pretrained_closed_loop", 0.25),
        ("eval_finetuned_closed_loop", 1.0),
    ):
        path = root / "latent_skill" / setting / "summary.json"
        payload = {
            "metadata": {
                "interface": "latent_skill",
                "planner_target_dim": 256,
            },
            "aggregate": {
                "done_rate": done_rate,
                "return_sum_mean": 1.0,
                "survival_steps_mean": 10.0,
            },
            "metrics": {},
        }
        rows.append(_row(path, payload, root))
    wide = _wide_rows(rows)[0]
    assert wide["oracle_success_rate"] == 1.0
    assert wide["pretrained_success_rate"] == 0.75
    assert wide["finetuned_success_rate"] == 0.0


def _test_aggregate_paper_metrics_include_offline() -> None:
    assert "planner_state_dim" in PAPER_METRICS
    assert "planner_state_history_steps" in PAPER_METRICS
    assert "planner_command_past_steps" in PAPER_METRICS
    assert "planner_command_future_steps" in PAPER_METRICS
    assert "planner_param_count" in PAPER_METRICS
    assert "planner_batch_size" in PAPER_METRICS
    assert "planner_micro_batch_size" in PAPER_METRICS
    assert "planner_gradient_accumulation_steps" in PAPER_METRICS
    assert "planner_num_updates" in PAPER_METRICS
    assert "planner_pretrain_num_updates" in PAPER_METRICS
    assert "planner_finetune_num_updates" in PAPER_METRICS
    assert "planner_lr" in PAPER_METRICS
    assert "planner_weight_decay" in PAPER_METRICS
    assert "planner_flow_num_inference_steps" in PAPER_METRICS
    assert "planner_flow_inference_noise_std" in PAPER_METRICS
    assert "planner_endpoint_num_inference_steps" in PAPER_METRICS
    assert "oracle_done_rate" in PAPER_METRICS
    assert "oracle_success_rate" in PAPER_METRICS
    assert "pretrained_done_rate" in PAPER_METRICS
    assert "pretrained_success_rate" in PAPER_METRICS
    assert "finetuned_done_rate" in PAPER_METRICS
    assert "finetuned_success_rate" in PAPER_METRICS
    assert "pretrained_expert_target_rmse" in PAPER_METRICS
    assert "finetuned_achieved_target_rmse" in PAPER_METRICS
    assert "pretrained_expert_eval_sample_count" in PAPER_METRICS


def _test_aggregate_seed_prefers_root_name(tmp: Path) -> None:
    root = tmp / "dance102_seed2"
    summary = root / "latent_skill" / "eval_finetuned_achieved_state" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text('{"metadata": {"seed": 0}, "aggregate": {}}\n', encoding="utf-8")
    assert _seed_for_root(root) == "2"


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _test_audit_interface_comparison(tmp: Path) -> None:
    aggregate_dir = tmp / "audit"
    expected_planner_metrics = {
        "planner_num_updates": 2000.0,
        "planner_finetune_num_updates": 2000.0,
        "planner_batch_size": 256.0,
        "planner_lr": 1.0e-4,
        "planner_weight_decay": 1.0e-4,
        "planner_flow_num_inference_steps": 16.0,
        "planner_flow_inference_noise_std": 0.0,
    }
    expected_hand_designed_planner_metrics = {
        "planner_state_history_steps": 0.0,
        "planner_command_past_steps": 0.0,
        "planner_command_future_steps": 25.0,
    }
    aggregate_metrics = sorted(
        set(REQUIRED_AGGREGATE_METRICS)
        | set(REQUIRED_OFFLINE_METRICS)
        | set(expected_planner_metrics)
        | set(expected_hand_designed_planner_metrics)
        | {"planner_state_dim"}
    )
    aggregate_columns = ["interface", "planner_variant", "num_seeds", "seeds"]
    for metric in aggregate_metrics:
        aggregate_columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_n",
            ]
        )

    aggregate_rows = []
    for interface, variant, output_dim in (
        ("latent_skill", "", 256),
        ("ee_trajectory", "chunked_transformer_medium_100", 936),
        ("full_body_trajectory", "chunked_transformer_medium_100", 1742),
    ):
        row: dict[str, object] = {
            "interface": interface,
            "planner_variant": variant,
            "num_seeds": 2,
            "seeds": "0,1",
        }
        for metric in aggregate_metrics:
            value = 1.0
            if metric == "output_dim":
                value = float(output_dim)
            elif metric == "planner_state_dim":
                value = 670.0 if interface == "latent_skill" else 67.0
            elif metric == "finetune_sample_count":
                value = 100.0
            elif metric in expected_planner_metrics:
                value = expected_planner_metrics[metric]
            elif metric in expected_hand_designed_planner_metrics:
                value = expected_hand_designed_planner_metrics[metric]
            elif metric.endswith("survival"):
                value = 1000.0
            if (
                interface == "latent_skill"
                and metric in expected_hand_designed_planner_metrics
            ):
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
                row[f"{metric}_min"] = ""
                row[f"{metric}_max"] = ""
                row[f"{metric}_n"] = 0
            else:
                row[f"{metric}_mean"] = value
                row[f"{metric}_std"] = 0.0
                row[f"{metric}_min"] = value
                row[f"{metric}_max"] = value
                row[f"{metric}_n"] = 2
        aggregate_rows.append(row)
    _write_csv(
        aggregate_dir / "interface_comparison_multiseed.csv",
        aggregate_rows,
        aggregate_columns,
    )

    by_seed_columns = [
        "result_root",
        "seed",
        "interface",
        "planner_variant",
        *REQUIRED_BY_SEED_METRICS,
        "planner_state_dim",
        *expected_planner_metrics,
        *expected_hand_designed_planner_metrics,
    ]
    aggregate_provenance = aggregate_dir / "interface_comparison_run_provenance.json"
    aggregate_provenance.write_text(
        json.dumps({"label": "smoke-aggregate"}) + "\n",
        encoding="utf-8",
    )
    seed_roots: dict[str, Path] = {}
    for seed in ("0", "1"):
        seed_root = tmp / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        (seed_root / "interface_comparison_run_provenance.json").write_text(
            json.dumps({"label": "smoke-seed", "seed": seed}) + "\n",
            encoding="utf-8",
        )
        seed_roots[seed] = seed_root
    by_seed_rows = []
    for seed in ("0", "1"):
        for interface, variant, output_dim in (
            ("latent_skill", "", 256),
            ("ee_trajectory", "chunked_transformer_medium_100", 936),
            ("full_body_trajectory", "chunked_transformer_medium_100", 1742),
        ):
            row = {
                "result_root": str(seed_roots[seed]),
                "seed": seed,
                "interface": interface,
                "planner_variant": variant,
            }
            for metric in REQUIRED_BY_SEED_METRICS:
                if metric == "output_dim":
                    row[metric] = output_dim
                elif metric == "finetune_sample_count":
                    row[metric] = 100
                else:
                    row[metric] = 1.0
            row["planner_state_dim"] = 670 if interface == "latent_skill" else 67
            row.update(expected_planner_metrics)
            if interface != "latent_skill":
                row.update(expected_hand_designed_planner_metrics)
            by_seed_rows.append(row)
    _write_csv(
        aggregate_dir / "interface_comparison_by_seed.csv",
        by_seed_rows,
        by_seed_columns,
    )

    selected_columns = [
        "interface",
        "planner_variant",
        *REQUIRED_SELECTED_METRICS,
    ]
    selected_rows = []
    for interface, variant, output_dim in (
        ("latent_skill", "", 256),
        ("ee_trajectory", "chunked_transformer_medium_100", 936),
        ("full_body_trajectory", "chunked_transformer_medium_100", 1742),
    ):
        row = {
            "interface": interface,
            "planner_variant": variant,
        }
        for metric in REQUIRED_SELECTED_METRICS:
            if metric == "output_dim":
                row[metric] = output_dim
            elif metric == "finetune_sample_count":
                row[metric] = 100
            else:
                row[metric] = 1.0
        selected_rows.append(row)
    _write_csv(
        aggregate_dir / "interface_sweep_selected.csv",
        selected_rows,
        selected_columns,
    )

    report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
    )
    assert report["status"] == "pass"
    assert len(report["selected_rows"]) == 3

    planner_budget_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        expected_planner_num_updates=2000,
        expected_planner_finetune_num_updates=2000,
        expected_planner_batch_size=256,
        expected_planner_lr=1.0e-4,
        expected_planner_weight_decay=1.0e-4,
        expected_planner_flow_num_inference_steps=16,
        expected_planner_flow_inference_noise_std=0.0,
        expected_hand_designed_planner_state_history_steps=0,
        expected_hand_designed_planner_command_past_steps=0,
        expected_hand_designed_planner_command_future_steps=25,
        require_selected=True,
    )
    assert planner_budget_report["status"] == "pass"
    assert planner_budget_report["selected_rows"][0]["planner_state_dim"] == 670.0
    assert (
        planner_budget_report["selected_rows"][0]["planner_state_history_steps"] is None
    )
    assert planner_budget_report["selected_rows"][0]["planner_num_updates"] == 2000.0
    assert (
        planner_budget_report["selected_rows"][0]["planner_finetune_num_updates"]
        == 2000.0
    )
    assert planner_budget_report["selected_rows"][0]["planner_batch_size"] == 256.0
    assert planner_budget_report["selected_rows"][0]["planner_lr"] == 1.0e-4
    assert planner_budget_report["selected_rows"][0]["planner_weight_decay"] == 1.0e-4
    assert any(
        "Planner metric planner_num_updates matches" in check["message"]
        for check in planner_budget_report["checks"]
    )
    assert any(
        "Planner metric planner_state_history_steps matches for ee_trajectory"
        in check["message"]
        for check in planner_budget_report["checks"]
    )
    assert not any(
        "Planner metric planner_state_history_steps" in check["message"]
        and "latent_skill" in check["message"]
        for check in planner_budget_report["checks"]
    )

    bad_planner_budget_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        expected_planner_batch_size=128,
        require_selected=True,
    )
    assert bad_planner_budget_report["status"] == "fail"
    assert any(
        "Planner metric planner_batch_size mismatch" in check["message"]
        for check in bad_planner_budget_report["checks"]
    )

    bad_history_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        expected_hand_designed_planner_state_history_steps=10,
        require_selected=True,
    )
    assert bad_history_report["status"] == "fail"
    assert any(
        "Planner metric planner_state_history_steps mismatch for ee_trajectory"
        in check["message"]
        for check in bad_history_report["checks"]
    )

    oracle_threshold_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
        min_oracle_survival=999.0,
        min_oracle_success_rate=0.99,
    )
    assert oracle_threshold_report["status"] == "pass"

    high_oracle_threshold_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
        min_oracle_survival=1001.0,
    )
    assert high_oracle_threshold_report["status"] == "fail"
    assert any(
        "oracle_survival" in check["message"] and "below minimum" in check["message"]
        for check in high_oracle_threshold_report["checks"]
    )

    ambiguous_rows = [dict(row) for row in aggregate_rows]
    extra_ee_row = dict(aggregate_rows[1])
    extra_ee_row["planner_variant"] = "chunked_transformer_large_100"
    ambiguous_rows.append(extra_ee_row)
    _write_csv(
        aggregate_dir / "interface_comparison_multiseed.csv",
        ambiguous_rows,
        aggregate_columns,
    )
    ambiguous_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
    )
    assert ambiguous_report["status"] == "fail"
    assert any(
        "Multiple comparable rows found for ee_trajectory" in check["message"]
        for check in ambiguous_report["checks"]
    )
    selected_variant_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
        use_selected_variants=True,
    )
    assert selected_variant_report["status"] == "pass"
    assert any(
        "Loaded planner variants from selected sweep CSV" in check["message"]
        for check in selected_variant_report["checks"]
    )
    _write_csv(
        aggregate_dir / "interface_comparison_multiseed.csv",
        aggregate_rows,
        aggregate_columns,
    )

    provenance_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
        require_provenance=True,
    )
    assert provenance_report["status"] == "pass"
    assert any(
        "Found valid aggregate provenance" in check["message"]
        for check in provenance_report["checks"]
    )
    assert any(
        "Found valid result-root provenance" in check["message"]
        for check in provenance_report["checks"]
    )

    aggregate_provenance.unlink()
    missing_aggregate_provenance_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
        require_provenance=True,
    )
    assert missing_aggregate_provenance_report["status"] == "fail"
    assert any(
        "Missing aggregate provenance" in check["message"]
        for check in missing_aggregate_provenance_report["checks"]
    )
    aggregate_provenance.write_text(
        json.dumps({"label": "smoke-aggregate"}) + "\n",
        encoding="utf-8",
    )

    seed_one_provenance = seed_roots["1"] / "interface_comparison_run_provenance.json"
    seed_one_provenance.unlink()
    missing_seed_provenance_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
        require_provenance=True,
    )
    assert missing_seed_provenance_report["status"] == "fail"
    assert any(
        "Missing result-root provenance" in check["message"]
        for check in missing_seed_provenance_report["checks"]
    )
    seed_one_provenance.write_text(
        json.dumps({"label": "smoke-seed", "seed": "1"}) + "\n",
        encoding="utf-8",
    )

    broken_selected = [dict(row) for row in selected_rows]
    broken_selected[0]["finetuned_action_delta"] = ""
    _write_csv(
        aggregate_dir / "interface_sweep_selected.csv",
        broken_selected,
        selected_columns,
    )
    broken_report = audit_aggregate_dir(
        aggregate_dir,
        expected_seeds={"0", "1"},
        expected_sample_count=100,
        require_selected=True,
    )
    assert broken_report["status"] == "fail"
    assert any(
        "Missing selected table metric finetuned_action_delta" in check["message"]
        for check in broken_report["checks"]
    )


def _test_analyze_interface_sweep() -> None:
    rows = [
        {
            "interface": "latent_skill",
            "planner_variant": "",
            "num_seeds": "2",
            "seeds": "0,1",
            "planner_param_count_mean": "123",
            "planner_batch_size_mean": "256",
            "planner_micro_batch_size_mean": "256",
            "planner_gradient_accumulation_steps_mean": "1",
            "planner_flow_num_inference_steps_mean": "16",
            "planner_endpoint_num_inference_steps_mean": "16",
            "finetune_sample_count_mean": "100",
            "finetuned_survival_mean": "1000",
            "oracle_done_rate_mean": "0.0",
            "oracle_success_rate_mean": "1.0",
            "pretrained_done_rate_mean": "0.0",
            "pretrained_success_rate_mean": "1.0",
            "finetuned_done_rate_mean": "0.0",
            "finetuned_success_rate_mean": "1.0",
            "finetuned_survival_oracle_ratio_mean": "1.0",
            "finetuned_return_mean": "70",
            "oracle_planner_target_rmse_mean": "0.001",
            "oracle_root_xy_error_mean": "0.011",
            "oracle_joint_rmse_mean": "0.021",
            "oracle_ee_pos_error_mean": "0.031",
            "oracle_action_delta_mean": "0.041",
            "pretrained_planner_target_rmse_mean": "0.005",
            "pretrained_root_xy_error_mean": "0.015",
            "pretrained_joint_rmse_mean": "0.025",
            "pretrained_ee_pos_error_mean": "0.035",
            "pretrained_action_delta_mean": "0.045",
            "finetuned_root_xy_error_mean": "0.01",
            "finetuned_joint_rmse_mean": "0.02",
            "finetuned_ee_pos_error_mean": "0.03",
            "finetuned_action_delta_mean": "0.04",
        },
        {
            "interface": "ee_trajectory",
            "planner_variant": "chunked_transformer_small_100",
            "num_seeds": "2",
            "seeds": "0,1",
            "planner_param_count_mean": "456",
            "planner_batch_size_mean": "256",
            "planner_micro_batch_size_mean": "32",
            "planner_gradient_accumulation_steps_mean": "8",
            "planner_flow_num_inference_steps_mean": "16",
            "planner_endpoint_num_inference_steps_mean": "4",
            "finetune_sample_count_mean": "100",
            "finetuned_survival_mean": "200",
            "oracle_done_rate_mean": "0.0",
            "oracle_success_rate_mean": "1.0",
            "pretrained_done_rate_mean": "1.0",
            "pretrained_success_rate_mean": "0.0",
            "finetuned_done_rate_mean": "0.8",
            "finetuned_success_rate_mean": "0.2",
            "finetuned_survival_oracle_ratio_mean": "0.2",
            "finetuned_return_mean": "20",
            "oracle_planner_target_rmse_mean": "0.101",
            "oracle_root_xy_error_mean": "0.111",
            "oracle_joint_rmse_mean": "0.121",
            "oracle_ee_pos_error_mean": "0.131",
            "oracle_action_delta_mean": "0.141",
            "pretrained_planner_target_rmse_mean": "0.151",
            "pretrained_root_xy_error_mean": "0.161",
            "pretrained_joint_rmse_mean": "0.171",
            "pretrained_ee_pos_error_mean": "0.181",
            "pretrained_action_delta_mean": "0.191",
            "finetuned_planner_target_rmse_mean": "0.1",
            "finetuned_root_xy_error_mean": "1.1",
            "finetuned_joint_rmse_mean": "1.2",
            "finetuned_ee_pos_error_mean": "1.3",
            "finetuned_action_delta_mean": "1.4",
        },
        {
            "interface": "ee_trajectory",
            "planner_variant": "chunked_transformer_medium_1000",
            "num_seeds": "2",
            "seeds": "0,1",
            "planner_param_count_mean": "789",
            "planner_batch_size_mean": "256",
            "planner_micro_batch_size_mean": "32",
            "planner_gradient_accumulation_steps_mean": "8",
            "planner_flow_num_inference_steps_mean": "16",
            "planner_endpoint_num_inference_steps_mean": "4",
            "finetune_sample_count_mean": "1000",
            "finetuned_survival_mean": "500",
            "oracle_done_rate_mean": "0.0",
            "oracle_success_rate_mean": "1.0",
            "pretrained_done_rate_mean": "0.9",
            "pretrained_success_rate_mean": "0.1",
            "finetuned_done_rate_mean": "0.5",
            "finetuned_success_rate_mean": "0.5",
            "finetuned_survival_oracle_ratio_mean": "0.5",
            "finetuned_return_mean": "30",
            "oracle_planner_target_rmse_mean": "0.201",
            "oracle_root_xy_error_mean": "0.211",
            "oracle_joint_rmse_mean": "0.221",
            "oracle_ee_pos_error_mean": "0.231",
            "oracle_action_delta_mean": "0.241",
            "pretrained_planner_target_rmse_mean": "0.251",
            "pretrained_root_xy_error_mean": "0.261",
            "pretrained_joint_rmse_mean": "0.271",
            "pretrained_ee_pos_error_mean": "0.281",
            "pretrained_action_delta_mean": "0.291",
            "finetuned_planner_target_rmse_mean": "0.2",
            "finetuned_root_xy_error_mean": "2.1",
            "finetuned_joint_rmse_mean": "2.2",
            "finetuned_ee_pos_error_mean": "2.3",
            "finetuned_action_delta_mean": "2.4",
        },
    ]
    for row_index, row in enumerate(rows):
        for setting in ("oracle", "pretrained", "finetuned"):
            row.setdefault(f"{setting}_return_mean", str(10.0 + row_index))
            row.setdefault(f"{setting}_survival_mean", str(100.0 + row_index))
            row.setdefault(f"{setting}_done_rate_mean", "0.0")
            row.setdefault(f"{setting}_success_rate_mean", "1.0")
            row.setdefault(f"{setting}_planner_target_rmse_mean", str(0.01 + row_index))
            row.setdefault(f"{setting}_root_xy_error_mean", str(0.02 + row_index))
            row.setdefault(f"{setting}_joint_rmse_mean", str(0.03 + row_index))
            row.setdefault(f"{setting}_ee_pos_error_mean", str(0.04 + row_index))
            row.setdefault(f"{setting}_action_delta_mean", str(0.05 + row_index))
        row.setdefault("output_dim_mean", str(256 + row_index))
        row.setdefault("pretrained_expert_target_rmse_mean", str(0.2 + row_index))
        row.setdefault("finetuned_achieved_target_rmse_mean", str(0.3 + row_index))
    summary = analyze_sweep(rows)
    latent = next(row for row in summary if row["interface"] == "latent_skill")
    best = next(row for row in summary if row["is_best_for_interface"])
    assert latent["is_latent_reference"] == 1
    assert best["planner_variant"] == "chunked_transformer_medium_1000"
    assert best["model_size"] == "medium"
    assert best["sample_budget"] == "1000"
    assert best["planner_param_count"] == 789.0
    assert best["planner_batch_size"] == 256.0
    assert best["planner_micro_batch_size"] == 32.0
    assert best["planner_gradient_accumulation_steps"] == 8.0
    assert best["planner_flow_num_inference_steps"] == 16.0
    assert best["planner_endpoint_num_inference_steps"] == 4.0
    assert best["finetuned_done_rate"] == 0.5
    assert best["finetuned_success_rate"] == 0.5
    assert best["oracle_root_xy_error"] == 0.211
    assert best["pretrained_root_xy_error"] == 0.261
    assert best["pretrained_action_delta"] == 0.291
    assert best["finetuned_root_xy_error"] == 2.1
    assert best["finetuned_joint_rmse"] == 2.2
    assert best["finetuned_ee_pos_error"] == 2.3
    assert best["finetuned_action_delta"] == 2.4
    assert best["rank_within_interface"] == 1
    assert best["gap_to_latent_finetuned_survival_oracle_ratio"] == -0.5
    selected = selected_sweep_rows(summary)
    assert [row["interface"] for row in selected] == [
        "latent_skill",
        "ee_trajectory",
    ]
    assert [row["planner_variant"] for row in selected] == [
        "",
        "chunked_transformer_medium_1000",
    ]
    filtered_summary = analyze_sweep(rows, selected_sample_count=100)
    filtered_best = next(
        row for row in filtered_summary if row["is_best_for_interface"]
    )
    filtered_medium = next(
        row
        for row in filtered_summary
        if row["planner_variant"] == "chunked_transformer_medium_1000"
    )
    assert filtered_best["planner_variant"] == "chunked_transformer_small_100"
    assert filtered_medium["is_sweep_candidate"] == 1
    assert filtered_medium["is_selection_candidate"] == 0
    validate_selected_interfaces(
        filtered_summary,
        ["latent_skill", "ee_trajectory"],
    )
    validate_selected_metrics(filtered_summary)
    broken_rows = [dict(row) for row in rows]
    broken_rows[1].pop("finetuned_action_delta_mean")
    broken_summary = analyze_sweep(broken_rows, selected_sample_count=100)
    try:
        validate_selected_metrics(broken_summary)
    except ValueError as exc:
        assert "Selected metric validation failed" in str(exc)
        assert "finetuned_action_delta" in str(exc)
    else:
        raise AssertionError("missing selected metric should fail validation")
    missing_filtered_summary = analyze_sweep(rows, selected_sample_count=999)
    try:
        validate_selected_interfaces(
            missing_filtered_summary,
            ["latent_skill", "ee_trajectory"],
        )
    except ValueError as exc:
        assert "missing" in str(exc)
        assert "ee_trajectory" in str(exc)
    else:
        raise AssertionError("missing selected interface should fail validation")
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        aggregate_csv = tmp / "aggregate.csv"
        fieldnames = sorted({key for row in rows for key in row})
        _write_csv(aggregate_csv, rows, fieldnames)
        script = Path(__file__).resolve().parent / "analyze_interface_sweep.py"
        selected_csv = tmp / "selected.csv"
        selected_md = tmp / "selected.md"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--input_csv",
                str(aggregate_csv),
                "--output_csv",
                str(tmp / "summary.csv"),
                "--output_md",
                str(tmp / "summary.md"),
                "--selected_csv",
                str(selected_csv),
                "--selected_md",
                str(selected_md),
                "--selected_sample_count",
                "100",
                "--expected_selected_interfaces",
                "latent_skill",
                "ee_trajectory",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with selected_csv.open("r", encoding="utf-8", newline="") as file:
            selected_disk_rows = list(csv.DictReader(file))
        assert len(selected_disk_rows) == 2
        assert "is_best_for_interface" not in selected_disk_rows[0]
        assert selected_disk_rows[0]["interface"] == "latent_skill"
        assert selected_disk_rows[1]["finetuned_done_rate"] == "0.8"
        assert selected_disk_rows[1]["finetuned_success_rate"] == "0.2"
        assert selected_disk_rows[1]["oracle_planner_target_rmse"] == "0.101"
        assert selected_disk_rows[1]["pretrained_root_xy_error"] == "0.161"
        assert selected_disk_rows[1]["pretrained_action_delta"] == "0.191"
        assert selected_disk_rows[1]["finetuned_root_xy_error"] == "1.1"
        assert selected_disk_rows[1]["finetuned_joint_rmse"] == "1.2"
        assert selected_disk_rows[1]["finetuned_ee_pos_error"] == "1.3"
        assert selected_disk_rows[1]["finetuned_action_delta"] == "1.4"
        assert selected_disk_rows[1]["planner_batch_size"] == "256.0"
        assert selected_disk_rows[1]["planner_micro_batch_size"] == "32.0"
        assert selected_disk_rows[1]["planner_flow_num_inference_steps"] == "16.0"
        assert selected_disk_rows[1]["planner_endpoint_num_inference_steps"] == "4.0"
        assert selected_disk_rows[1]["planner_variant"] == (
            "chunked_transformer_small_100"
        )
        assert "Selected Interface Baseline Rows" in selected_md.read_text(
            encoding="utf-8"
        )
        failed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--input_csv",
                str(aggregate_csv),
                "--output_csv",
                str(tmp / "failed_summary.csv"),
                "--output_md",
                str(tmp / "failed_summary.md"),
                "--selected_csv",
                str(tmp / "failed_selected.csv"),
                "--selected_md",
                str(tmp / "failed_selected.md"),
                "--selected_sample_count",
                "999",
                "--expected_selected_interfaces",
                "latent_skill",
                "ee_trajectory",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "Selected interface validation failed" in failed.stderr


def _test_capacity_metadata_backfill(tmp: Path) -> None:
    checkpoint = tmp / "checkpoint.pt"
    torch.save(
        {
            "planner_state_dict": {
                "layer.weight": torch.zeros(3, 4),
                "layer.bias": torch.zeros(3),
            },
            "planner_config": {"planner_type": "chunked_transformer_flow"},
            "metadata": {
                "batch_size": 256,
                "micro_batch_size": 32,
                "gradient_accumulation_steps": 8,
                "num_updates": 2000,
                "finetune_num_updates": 2000,
                "lr": 1.0e-4,
                "weight_decay": 1.0e-4,
                "flow_num_inference_steps": 16,
                "flow_inference_noise_std": 0.0,
                "endpoint_num_inference_steps": 4,
                "source_sample_count": 20,
                "selected_sample_count": 10,
                "heldout_sample_count": 10,
                "checkpoint_metadata": {
                    "num_updates": 123,
                },
            },
        },
        checkpoint,
    )
    summary_path = (
        tmp
        / "results"
        / "ee_trajectory"
        / "chunked_transformer_small_10"
        / "eval_finetuned_closed_loop"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        (
            '{"metadata": {"planner_checkpoint": "'
            + str(checkpoint)
            + '"}, "aggregate": {}, "metrics": {}}\n'
        ),
        encoding="utf-8",
    )
    latent_summary_path = (
        tmp / "results" / "latent_skill" / "eval_finetuned_closed_loop" / "summary.json"
    )
    latent_summary_path.parent.mkdir(parents=True, exist_ok=True)
    latent_summary_path.write_text(
        (
            '{"metadata": {"planner_checkpoint": "'
            + str(checkpoint)
            + '"}, "aggregate": {}, "metrics": {}}\n'
        ),
        encoding="utf-8",
    )
    latent_finetune_summary = (
        tmp
        / "results"
        / "latent_skill"
        / "planner_finetune_achieved_state"
        / "summary.json"
    )
    latent_finetune_summary.parent.mkdir(parents=True, exist_ok=True)
    latent_finetune_summary.write_text(
        json.dumps(
            {
                "num_samples": 100,
                "state_dim": 670,
                "z_dim": 256,
                "args": {
                    "batch_size": 256,
                    "num_updates": 2000,
                    "lr": 1.0e-4,
                    "weight_decay": 0.0,
                    "flow_num_inference_steps": 16,
                    "flow_inference_noise_std": 0.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _backfill_root(tmp / "results", dry_run=False) == 2
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    planner_metadata = payload["metadata"]["planner_metadata"]
    assert planner_metadata["parameter_count"] == 15
    assert planner_metadata["trainable_parameter_count"] == 15
    assert planner_metadata["model_size"] == "small"
    assert planner_metadata["sample_budget"] == "10"
    assert planner_metadata["batch_size"] == 256
    assert planner_metadata["micro_batch_size"] == 32
    assert planner_metadata["gradient_accumulation_steps"] == 8
    assert planner_metadata["num_updates"] == 2000
    assert planner_metadata["pretrain_num_updates"] == 123
    assert planner_metadata["finetune_num_updates"] == 2000
    assert planner_metadata["lr"] == 1.0e-4
    assert planner_metadata["weight_decay"] == 1.0e-4
    assert planner_metadata["flow_num_inference_steps"] == 16
    assert planner_metadata["flow_inference_noise_std"] == 0.0
    assert planner_metadata["endpoint_num_inference_steps"] == 4
    assert planner_metadata["source_sample_count"] == 20
    assert planner_metadata["selected_sample_count"] == 10
    assert planner_metadata["heldout_sample_count"] == 10

    latent_payload = json.loads(latent_summary_path.read_text(encoding="utf-8"))
    latent_metadata = latent_payload["metadata"]["planner_metadata"]
    assert latent_metadata["interface"] == "latent_skill"
    assert latent_metadata["source_sample_count"] == 100
    assert latent_metadata["selected_sample_count"] == 100
    assert latent_metadata["batch_size"] == 256
    assert latent_metadata["num_updates"] == 2000
    assert latent_metadata["finetune_num_updates"] == 2000
    assert latent_metadata["lr"] == 1.0e-4
    assert latent_metadata["weight_decay"] == 0.0
    assert latent_metadata["flow_num_inference_steps"] == 16
    assert latent_metadata["flow_inference_noise_std"] == 0.0
    assert latent_metadata["state_dim"] == 670
    assert latent_metadata["target_dim"] == 256


def _test_manifest_splitter() -> None:
    manifest = {
        "dataset_name": "lafan1",
        "dataset": {
            "trajectories": {
                "lafan1_csv": [
                    {"name": "walk", "path": "walk.npz", "input_fps": 50.0},
                    {"name": "dance", "path": "dance.npz", "input_fps": 50.0},
                    {"name": "jump", "path": "jump.npz", "input_fps": 50.0},
                ]
            }
        },
        "metadata": {"num_motions": 3},
    }
    train, heldout = split_manifest(manifest, heldout_names=["dance"])
    train_names = [
        entry["name"] for entry in train["dataset"]["trajectories"]["lafan1_csv"]
    ]
    heldout_names = [
        entry["name"] for entry in heldout["dataset"]["trajectories"]["lafan1_csv"]
    ]
    assert train_names == ["walk", "jump"]
    assert heldout_names == ["dance"]
    assert train["metadata"]["num_motions"] == 2
    assert heldout["metadata"]["split"] == "heldout"

    _, heldout_fraction = split_manifest(manifest, heldout_fraction=1 / 3, seed=7)
    assert len(heldout_fraction["dataset"]["trajectories"]["lafan1_csv"]) == 1

    try:
        split_manifest(manifest, heldout_fraction=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative heldout_fraction should fail")


def _test_manifest_splitter_rebases_relative_paths(tmp: Path) -> None:
    source_manifest = tmp / "manifests" / "g1_lafan1_manifest.json"
    target_manifest = (
        tmp / "manifests" / "interface_baselines" / "g1_lafan1_seed0_train.json"
    )
    motion_path = tmp / "npz" / "g1" / "dance1_subject2.npz"
    motion_path.parent.mkdir(parents=True)
    motion_path.write_bytes(b"placeholder")
    manifest = {
        "dataset": {
            "trajectories": {
                "lafan1_csv": [
                    {
                        "name": "dance1_subject2",
                        "path": "../npz/g1/dance1_subject2.npz",
                        "input_fps": 50.0,
                    }
                ]
            }
        },
        "metadata": {"paths_are_relative_to_manifest": True},
    }

    rebased = _rebase_manifest_paths(
        manifest,
        source_manifest_path=source_manifest,
        target_manifest_path=target_manifest,
    )
    entry = rebased["dataset"]["trajectories"]["lafan1_csv"][0]
    assert entry["path"] == "../../npz/g1/dance1_subject2.npz"
    assert (target_manifest.parent / entry["path"]).resolve() == motion_path.resolve()


def _test_lafan1_heldout_wrapper_missing_manifest(tmp: Path) -> None:
    script = (
        Path(__file__).resolve().parent
        / "run_lafan1_heldout_strong_interface_comparison.sh"
    )
    script_text = script.read_text(encoding="utf-8")
    assert (
        "${FULL_MANIFEST:-${CLUSTER_DATA_DIR:-data}/lafan1/manifests/"
        "g1_lafan1_manifest.json}" in script_text
    )
    assert (
        'SPLIT_OUTPUT_DIR="${SPLIT_OUTPUT_DIR:-${FULL_MANIFEST_DIR}/interface_baselines}"'
        in script_text
    )
    missing_manifest = tmp / "missing_manifest.json"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={"FULL_MANIFEST": str(missing_manifest), "PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Full multi-motion manifest not found" in result.stderr


def _test_lafan1_heldout_wrapper_manual_split_skips_full_manifest(tmp: Path) -> None:
    script = (
        Path(__file__).resolve().parent
        / "run_lafan1_heldout_strong_interface_comparison.sh"
    )
    missing_manifest = tmp / "missing_manifest.json"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "FULL_MANIFEST": str(missing_manifest),
            "TRAIN_MANIFEST": str(tmp / "train.json"),
            "EVAL_MANIFEST": str(tmp / "eval.json"),
            "OUTPUT_ROOT": str(tmp / "manual_split_result"),
            "FULL_BODY_TRAJECTORY_CHECKPOINT": "/tmp/full.pt",
            "EE_TRAJECTORY_CHECKPOINT": "/tmp/ee.pt",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Full multi-motion manifest not found" not in result.stderr
    assert str(tmp / "train.json") in result.stdout
    assert "run_dance102_strong_interface_comparison.sh" not in result.stdout
    assert "collect_interface_rollout_samples.py" in result.stdout


def _test_lafan1_heldout_wrapper_latent_requires_dataset_path(tmp: Path) -> None:
    script = (
        Path(__file__).resolve().parent
        / "run_lafan1_heldout_strong_interface_comparison.sh"
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "TRAIN_MANIFEST": str(tmp / "train.json"),
            "EVAL_MANIFEST": str(tmp / "eval.json"),
            "RUN_LATENT_BASELINE": "1",
            "FULL_BODY_TRAJECTORY_CHECKPOINT": "/tmp/full.pt",
            "EE_TRAJECTORY_CHECKPOINT": "/tmp/ee.pt",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/tmp/latent_low.pt",
            "LATENT_SKILL_CHECKPOINT": "/tmp/latent_skill.pt",
            "LATENT_PLANNER_CHECKPOINT": "/tmp/latent_planner.pt",
            "MIN_ORACLE_SURVIVAL": "800",
            "MIN_ORACLE_SUCCESS_RATE": "0.8",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Set LATENT_DATASET_PATH" in result.stderr


def _test_multiseed_wrapper_dry_run() -> None:
    script = (
        Path(__file__).resolve().parent / "run_dance102_strong_interface_multiseed.sh"
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "SEEDS": "0 1",
            "MODEL_SIZE": "medium",
            "SAMPLE_BUDGETS": "10000",
            "STATE_HISTORY_STEPS": "10",
            "COMMAND_PAST_STEPS": "1",
            "COMMAND_FUTURE_STEPS": "25",
            "OUTPUT_PREFIX": "/tmp/dance102_strong_10k",
            "LATENT_OUTPUT_PREFIX": "/tmp/dance102_latent_10k",
            "AGGREGATE_OUTPUT_DIR": "/tmp/dance102_strong_10k_multiseed",
            "RUN_LATENT_BASELINE": "1",
            "FULL_BODY_TRAJECTORY_CHECKPOINT": "/tmp/full.pt",
            "EE_TRAJECTORY_CHECKPOINT": "/tmp/ee.pt",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/tmp/latent_low.pt",
            "LATENT_SKILL_CHECKPOINT": "/tmp/latent_skill.pt",
            "LATENT_PLANNER_CHECKPOINT": "/tmp/latent_planner.pt",
            "LATENT_DATASET_PATH": "/tmp/latent_dataset",
            "SELECTED_SAMPLE_COUNT": "10000",
            "AUDIT_EXPECTED_PRETRAIN_UPDATES": "5000",
            "MIN_ORACLE_SURVIVAL": "800",
            "MIN_ORACLE_SUCCESS_RATE": "0.8",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "preflight_interface_comparison.py" in result.stdout
    assert "/tmp/dance102_strong_10k_seed0" in result.stdout
    assert "/tmp/dance102_latent_10k_seed0" in result.stdout
    assert "write_interface_run_provenance.py" in result.stdout
    assert (
        "--output_json /tmp/dance102_strong_10k_multiseed/interface_comparison_run_provenance.json"
        in result.stdout
    )
    assert "run_dance102_fair_interface_comparison.sh" in result.stdout
    assert "INTERFACES=" in result.stdout
    assert "backfill_planner_capacity_metadata.py" in result.stdout
    assert "--expected_interfaces" in result.stdout
    for interface in ("latent_skill", "ee_trajectory", "full_body_trajectory"):
        assert interface in result.stdout
    assert "--expected_sample_count 10000" in result.stdout
    assert "chunked_transformer_medium_10000" in result.stdout
    assert "--expected_selected_interfaces" in result.stdout
    assert "--selected_sample_count 10000" in result.stdout
    assert "--require_provenance" in result.stdout
    assert "--expected_planner_num_updates 2000" in result.stdout
    assert "--expected_planner_finetune_num_updates 2000" in result.stdout
    assert "--expected_planner_pretrain_num_updates 5000" in result.stdout
    assert "--expected_planner_batch_size 256" in result.stdout
    assert "--expected_planner_lr 1.0e-4" in result.stdout
    assert "--expected_planner_weight_decay 1.0e-4" in result.stdout
    assert "--expected_planner_flow_num_inference_steps 16" in result.stdout
    assert "--expected_planner_flow_inference_noise_std 0.0" in result.stdout
    assert "--expected_hand_designed_planner_state_history_steps 10" in result.stdout
    assert "--expected_hand_designed_planner_command_past_steps 1" in result.stdout
    assert "--expected_hand_designed_planner_command_future_steps 25" in result.stdout
    assert "COLLECT_STEPS=10000" in result.stdout
    assert "STATE_HISTORY_STEPS=10" in result.stdout
    assert "COMMAND_PAST_STEPS=1" in result.stdout
    assert "COMMAND_FUTURE_STEPS=25" in result.stdout
    assert "PRETRAIN_UPDATES=2000" in result.stdout
    assert "FINETUNE_WEIGHT_DECAY=1.0e-4" in result.stdout
    assert "--require_selected" in result.stdout
    assert "--use_selected_variants" in result.stdout
    assert "--min_oracle_survival 800" in result.stdout
    assert "--min_oracle_success_rate 0.8" in result.stdout
    assert (
        "--output_json /tmp/dance102_strong_10k_multiseed/interface_comparison_audit.json"
        in result.stdout
    )
    assert (
        "--output_md /tmp/dance102_strong_10k_multiseed/interface_comparison_audit.md"
        in result.stdout
    )

    latent_only = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "SEEDS": "0",
            "INTERFACES": "",
            "RUN_LATENT_BASELINE": "1",
            "OUTPUT_PREFIX": "/tmp/unused_hand_designed",
            "LATENT_OUTPUT_PREFIX": "/tmp/dance102_latent_only",
            "AGGREGATE_OUTPUT_DIR": "/tmp/dance102_latent_only_multiseed",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/tmp/latent_low.pt",
            "LATENT_SKILL_CHECKPOINT": "/tmp/latent_skill.pt",
            "LATENT_PLANNER_CHECKPOINT": "/tmp/latent_planner.pt",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--expected_interfaces latent_skill --expected_seeds" in latent_only.stdout
    assert "run_dance102_strong_interface_comparison.sh" not in latent_only.stdout
    assert "INTERFACES=" in latent_only.stdout


def _test_fair_runner_omits_empty_latent_motion_filter() -> None:
    script = (
        Path(__file__).resolve().parent / "run_dance102_fair_interface_comparison.sh"
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "RUN_LATENT": "1",
            "INTERFACES": "",
            "LATENT_MOTION_NAME": "",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/tmp/latent_low.pt",
            "LATENT_SKILL_CHECKPOINT": "/tmp/latent_skill.pt",
            "LATENT_PLANNER_CHECKPOINT": "/tmp/latent_planner.pt",
            "OUTPUT_ROOT": "/tmp/latent_no_motion_filter",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "eval_skill_commander_closed_loop.py" in result.stdout
    assert "write_interface_run_provenance.py" in result.stdout
    latent_eval_text = (
        Path(__file__).resolve().parents[2]
        / "scripts/rlopt/eval_skill_commander_closed_loop.py"
    ).read_text(encoding="utf-8")
    assert "saved_rows" in latent_eval_text
    assert "sample_file_count" in latent_eval_text
    assert "_skill_commander_planner_metadata" in latent_eval_text
    assert '"planner_metadata": planner_metadata' in latent_eval_text
    interface_trainer_text = (
        Path(__file__).resolve().parent / "train_interface_planner.py"
    ).read_text(encoding="utf-8")
    assert '"flow_inference_noise_std": float(args.flow_inference_noise_std)' in (
        interface_trainer_text
    )
    assert 'parser.add_argument("--weight_decay", type=float, default=1.0e-4)' in (
        interface_trainer_text
    )
    latent_finetune_text = (
        Path(__file__).resolve().parents[2]
        / "scripts/rlopt/finetune_skill_commander_rollout.py"
    ).read_text(encoding="utf-8")
    assert '"selected_sample_count": int(num_samples)' in latent_finetune_text
    assert 'output_checkpoint["metadata"] = checkpoint_metadata' in latent_finetune_text
    assert (
        "--output_json /tmp/latent_no_motion_filter/interface_comparison_run_provenance.json"
        in result.stdout
    )
    assert "--motion_name" not in result.stdout
    assert "--trajectory_name" not in result.stdout
    assert "--weight_decay 1.0e-4" in result.stdout
    assert "env.lafan1_manifest_path" in result.stdout


def _test_strong_runner_allows_empty_interfaces() -> None:
    script = (
        Path(__file__).resolve().parent / "run_dance102_strong_interface_comparison.sh"
    )
    script_text = script.read_text(encoding="utf-8")
    assert "MIN_REUSE_SAMPLE_COUNT" in script_text
    assert "sample_row_count()" in script_text
    assert "sample_step_*.pt" in script_text
    assert "Recollecting samples" in script_text
    assert "rows, need" in script_text
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "INTERFACES": "",
            "OUTPUT_ROOT": "/tmp/strong_no_interfaces",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "evaluate_checkpoint.py" not in result.stdout
    assert "collect_interface_rollout_samples.py" not in result.stdout
    assert "train_chunked_transformer_planner.py" not in result.stdout
    assert "write_interface_run_provenance.py" in result.stdout
    assert (
        "--output_json /tmp/strong_no_interfaces/interface_comparison_run_provenance.json"
        in result.stdout
    )
    assert "summarize_interface_comparison.py" in result.stdout


def _test_cluster_job_launcher_dry_run() -> None:
    script = Path(__file__).resolve().parent / "run_interface_baseline_job.py"
    script_text = script.read_text(encoding="utf-8")
    assert "INTERFACE_BASELINE_PYTHON_CMD" in script_text
    assert "/isaac-sim/python.sh" in script_text
    assert "ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT" in script_text
    assert '"lafan1" / "zarr_cache"' in script_text
    assert "ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT" in script_text
    assert '"unitree_usd_cache"' in script_text
    assert "manual-{host}-{os.getpid()}" in script_text
    assert "dance102-strong-multiseed" in script_text
    assert "multimotion-heldout" in script_text
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "dance102-strong",
            "--dry_run",
            "--env",
            "INTERFACES=",
            "--env",
            "OUTPUT_ROOT=/tmp/launcher_no_interfaces",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run_dance102_strong_interface_comparison.sh" in result.stdout
    assert "evaluate_checkpoint.py" not in result.stdout
    assert "summarize_interface_comparison.py" in result.stdout


def _test_run_provenance_writer(tmp: Path) -> None:
    script = Path(__file__).resolve().parent / "write_interface_run_provenance.py"
    output_json = tmp / "provenance" / "run.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--label",
            "smoke",
            "--output_json",
            str(output_json),
            "--result_root",
            "/tmp/seed0",
            "--note",
            "purpose=smoke",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "SAMPLE_BUDGETS": "10000",
            "MODEL_SIZES": "medium",
            "MIN_ORACLE_SURVIVAL": "800",
            "FINETUNE_WEIGHT_DECAY": "1.0e-4",
            "AUDIT_EXPECTED_PRETRAIN_UPDATES": "5000",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["label"] == "smoke"
    assert payload["env"]["SAMPLE_BUDGETS"] == "10000"
    assert payload["env"]["MODEL_SIZES"] == "medium"
    assert payload["env"]["MIN_ORACLE_SURVIVAL"] == "800"
    assert payload["env"]["FINETUNE_WEIGHT_DECAY"] == "1.0e-4"
    assert payload["env"]["AUDIT_EXPECTED_PRETRAIN_UPDATES"] == "5000"
    assert payload["result_roots"] == ["/tmp/seed0"]
    assert payload["notes"] == ["purpose=smoke"]
    assert "commit" in payload["git"]
    assert "submodule_status" in payload["git"]


def _test_unitree_usd_cache_env_hook() -> None:
    unitree_path = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_imitation/isaaclab_imitation/assets/robots/unitree.py"
    )
    text = unitree_path.read_text(encoding="utf-8")
    assert "ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT" in text
    assert "usd_dir: str | None = _unitree_usd_cache_dir()" in text


def _test_cluster_submitter_dry_run() -> None:
    script = Path(__file__).resolve().parent / "submit_cluster_interface_baselines.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "MODE": "lafan1-heldout-multiseed",
            "EXTRA_AGGREGATE_GLOBS": "/tmp/latent_seed[0-9]",
            "RUN_LATENT_BASELINE": "1",
            "INTERFACES": "",
            "MICRO_BATCH_SIZE": "16",
            "TRAIN_ENDPOINT_STEPS": "2",
            "STATE_HISTORY_STEPS": "10",
            "COMMAND_PAST_STEPS": "1",
            "COMMAND_FUTURE_STEPS": "25",
            "FINETUNE_WEIGHT_DECAY": "1.0e-4",
            "AUDIT_EXPECTED_PRETRAIN_UPDATES": "5000",
            "LATENT_DATASET_PATH": "/tmp/latent_dataset",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/tmp/latent_low.pt",
            "LATENT_SKILL_CHECKPOINT": "/tmp/latent_skill.pt",
            "LATENT_PLANNER_CHECKPOINT": "/tmp/latent_planner.pt",
            "MIN_ORACLE_SURVIVAL": "800",
            "MIN_ORACLE_SUCCESS_RATE": "0.8",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "CLUSTER_PYTHON_EXECUTABLE=experiments/interface_baselines/run_interface_baseline_job.py"
        in result.stdout
    )
    assert "CLUSTER_GIT_SYNC_FIRST=0" in result.stdout
    assert "CLUSTER_EXTRA_RSYNC_EXCLUDES=IsaacLab/ RLOpt/ ImitationLearningTools/" in (
        result.stdout
    )
    assert "CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=1" in result.stdout
    assert "CLUSTER_SKIP_CACHE_COPY=1" in result.stdout
    assert "CLUSTER_OVERLAY_SIZE_MB=8192" in result.stdout
    assert "CLUSTER_USE_SHARED_SIF=1" in result.stdout
    assert "CLUSTER_RLOPT_LOCAL_PATH=" in result.stdout
    assert "CLUSTER_IMITATION_TOOLS_LOCAL_PATH=" in result.stdout
    assert "CLUSTER_G1_MANIFEST_PATH is empty" in result.stdout
    assert "--mode lafan1-heldout-multiseed" in result.stdout
    assert r"--env EXTRA_AGGREGATE_GLOBS=/tmp/latent_seed\[0-9\]" in result.stdout
    assert "--env RUN_LATENT_BASELINE=1" in result.stdout
    assert "--env INTERFACES=" in result.stdout
    assert "--env MICRO_BATCH_SIZE=16" in result.stdout
    assert "--env TRAIN_ENDPOINT_STEPS=2" in result.stdout
    assert "--env STATE_HISTORY_STEPS=10" in result.stdout
    assert "--env COMMAND_PAST_STEPS=1" in result.stdout
    assert "--env COMMAND_FUTURE_STEPS=25" in result.stdout
    assert "--env FINETUNE_WEIGHT_DECAY=1.0e-4" in result.stdout
    assert "--env AUDIT_EXPECTED_PRETRAIN_UPDATES=5000" in result.stdout
    assert "--env LATENT_DATASET_PATH=/tmp/latent_dataset" in result.stdout
    assert "--env MIN_ORACLE_SURVIVAL=800" in result.stdout
    assert "--env MIN_ORACLE_SUCCESS_RATE=0.8" in result.stdout


def _test_cluster_submitter_auto_syncs_local_checkpoints(tmp: Path) -> None:
    checkpoint = tmp / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    script = Path(__file__).resolve().parent / "submit_cluster_interface_baselines.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "MODE": "dance102-strong",
            "FULL_BODY_TRAJECTORY_CHECKPOINT": str(checkpoint),
            "INTERFACES": "full_body_trajectory",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Auto-syncing FULL_BODY_TRAJECTORY_CHECKPOINT" in result.stdout
    assert f"{checkpoint.parent}" in result.stdout
    assert "CLUSTER_EXTRA_RSYNC_SPECS=" in result.stdout
    assert (
        f"{checkpoint.parent}:external/interface_baseline_checkpoints/"
        "FULL_BODY_TRAJECTORY_CHECKPOINT" in result.stdout
    )
    assert (
        "external/interface_baseline_checkpoints/FULL_BODY_TRAJECTORY_CHECKPOINT"
        in (result.stdout)
    )
    assert (
        "--env FULL_BODY_TRAJECTORY_CHECKPOINT=external/interface_baseline_checkpoints/FULL_BODY_TRAJECTORY_CHECKPOINT/model.pt"
        in result.stdout
    )


def _test_cluster_submitter_auto_syncs_extra_aggregate_artifacts(
    tmp: Path,
) -> None:
    root = tmp / "latent_seed0_full"
    root.mkdir()
    (root / "summary.json").write_text("{}", encoding="utf-8")
    glob_root = tmp / "latent_seed1_full"
    glob_root.mkdir()
    (glob_root / "summary.json").write_text("{}", encoding="utf-8")

    script = Path(__file__).resolve().parent / "submit_cluster_interface_baselines.sh"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "MODE": "dance102-strong-multiseed",
            "EXTRA_AGGREGATE_ROOTS": str(root),
            "EXTRA_AGGREGATE_GLOBS": str(tmp / "latent_seed[1]_full"),
            "RUN_LATENT_BASELINE": "0",
            "INTERFACES": "",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Auto-syncing EXTRA_AGGREGATE_ROOTS" in result.stdout
    assert "Auto-syncing EXTRA_AGGREGATE_GLOBS match" in result.stdout
    assert "CLUSTER_EXTRA_RSYNC_SPECS=" in result.stdout
    assert (
        f"{root}:external/interface_baseline_results/latent_seed0_full" in result.stdout
    )
    assert (
        f"{glob_root}:external/interface_baseline_results/latent_seed1_full"
        in result.stdout
    )
    assert (
        "--env EXTRA_AGGREGATE_ROOTS=external/interface_baseline_results/latent_seed0_full"
        in result.stdout
    )
    assert (
        "--env EXTRA_AGGREGATE_GLOBS=external/interface_baseline_results/latent_seed\\[1\\]_full"
        in result.stdout
    )


def _test_cluster_interface_forwards_remote_env_args() -> None:
    script = Path(__file__).resolve().parents[2] / "docker/cluster/cluster_interface.sh"
    text = script.read_text(encoding="utf-8")
    assert "build_remote_job_env_args" in text
    assert 'env "${REMOTE_JOB_ENV_ARGS[@]}"' in text
    assert 'submit_job "$@"' in text
    assert "add_remote_env_arg_if_set CLUSTER_G1_MANIFEST_PATH" in text
    assert "add_remote_env_arg_if_set CLUSTER_G1_MANIFEST_REFRESH_POLICY" in text
    assert "CLUSTER_RLOPT_LOCAL_PATH" in text
    assert "CLUSTER_IMITATION_TOOLS_LOCAL_PATH" in text
    assert "CLUSTER_GIT_SYNC_FIRST" in text
    assert "CLUSTER_EXTRA_RSYNC_EXCLUDES" in text
    assert "CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS" in text
    assert "link_isaaclab_from_previous_sync" in text
    assert "prefix_home_if_relative()" in text
    assert (
        'CLUSTER_G1_MANIFEST_PATH="$(prefix_home_if_relative '
        '"$CLUSTER_REMOTE_HOME" "$CLUSTER_G1_MANIFEST_PATH")' in text
    )
    assert "CLUSTER_EXTRA_PYTHONPATH_REL" in text
    assert "CLUSTER_EXTRA_RSYNC_SPECS" in text
    assert "Using CLUSTER_EXTRA_RSYNC_SPECS for artifact sync." in text
    assert "sync_tree_to_cluster" in text
    assert '--exclude=".pixi/"' in text
    assert '--exclude=".codex/"' in text
    assert '--exclude="logs/"' in text

    runtime_script = (
        Path(__file__).resolve().parents[2] / "docker/cluster/run_singularity.sh"
    )
    runtime_text = runtime_script.read_text(encoding="utf-8")
    assert "prefix_home_if_relative()" in runtime_text
    assert 'base_tmpdir="$(prefix_home_if_relative "$HOME" "$base_tmpdir")"' in (
        runtime_text
    )
    assert (
        'CLUSTER_G1_MANIFEST_PATH="$(prefix_home_if_relative "$HOME" '
        '"$CLUSTER_G1_MANIFEST_PATH")' in runtime_text
    )
    assert "sync_project_logs_back()" in runtime_text
    assert "Syncing per-job project logs back to permanent workspace" in runtime_text
    assert "seed_shared_project_logs_from_submission()" in runtime_text
    assert "Seeding shared project logs from submitted workspace" in runtime_text
    assert "seed_shared_project_logs_from_submission\n\ncontainer_image=" in (
        runtime_text
    )
    assert 'if [ "${PROJECT_LOGS_SYNCED:-0}" = "1" ]; then' in runtime_text
    assert "No submitted workspace name yet; skipping per-job project log sync." in (
        runtime_text
    )
    assert "PROJECT_LOGS_SYNCED=1" in runtime_text
    assert (
        "cleanup_job_tmpdir() {\n    local status=$?\n    set +e\n    sync_project_logs_back"
        in (runtime_text)
    )
    assert "sync_project_logs_back || true" in runtime_text
    assert 'exit "$workload_status"' in runtime_text
    assert "Copying submitted workspace into per-job TMPDIR." in runtime_text
    assert "Extracting container image into per-job TMPDIR." in runtime_text


def _test_shell_wrappers_do_not_require_git_repo() -> None:
    script_dir = Path(__file__).resolve().parent
    scripts = [
        script_dir / "run_dance102_fair_interface_comparison.sh",
        script_dir / "run_dance102_strong_interface_comparison.sh",
        script_dir / "run_dance102_strong_interface_multiseed.sh",
        script_dir / "run_lafan1_heldout_strong_interface_comparison.sh",
        script_dir / "run_lafan1_heldout_strong_interface_multiseed.sh",
        script_dir / "run_multimotion_heldout_interface_comparison.sh",
    ]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "git rev-parse --show-toplevel" not in text
        assert 'git -C "${SCRIPT_DIR}" rev-parse --show-toplevel' in text
        assert 'REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"' in text
        assert "INTERFACE_BASELINE_PYTHON_CMD" in text


def _test_lafan1_heldout_multiseed_wrapper_dry_run() -> None:
    script = (
        Path(__file__).resolve().parent
        / "run_lafan1_heldout_strong_interface_multiseed.sh"
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "SEEDS": "0 1",
            "MODEL_SIZE": "medium",
            "SAMPLE_BUDGETS": "10000",
            "STATE_HISTORY_STEPS": "10",
            "COMMAND_PAST_STEPS": "1",
            "COMMAND_FUTURE_STEPS": "25",
            "OUTPUT_PREFIX": "/tmp/lafan1_heldout",
            "AGGREGATE_OUTPUT_DIR": "/tmp/lafan1_heldout_multiseed",
            "FULL_BODY_TRAJECTORY_CHECKPOINT": "/tmp/full.pt",
            "EE_TRAJECTORY_CHECKPOINT": "/tmp/ee.pt",
            "SELECTED_SAMPLE_COUNT": "10000",
            "AUDIT_EXPECTED_PRETRAIN_UPDATES": "5000",
            "MIN_ORACLE_SURVIVAL": "800",
            "MIN_ORACLE_SUCCESS_RATE": "0.8",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run_lafan1_heldout_strong_interface_comparison.sh" in result.stdout
    assert "/tmp/lafan1_heldout_seed0" in result.stdout
    assert "write_interface_run_provenance.py" in result.stdout
    assert (
        "--output_json /tmp/lafan1_heldout_multiseed/interface_comparison_run_provenance.json"
        in result.stdout
    )
    assert "backfill_planner_capacity_metadata.py" in result.stdout
    assert "aggregate_interface_comparison_seeds.py" in result.stdout
    assert "--expected_interfaces ee_trajectory full_body_trajectory" in result.stdout
    assert "analyze_interface_sweep.py" in result.stdout
    assert "--expected_selected_interfaces ee_trajectory full_body_trajectory" in (
        result.stdout
    )
    assert "--selected_sample_count 10000" in result.stdout
    assert "--require_provenance" in result.stdout
    assert "--expected_planner_num_updates 2000" in result.stdout
    assert "--expected_planner_finetune_num_updates 2000" in result.stdout
    assert "--expected_planner_pretrain_num_updates 5000" in result.stdout
    assert "--expected_planner_batch_size 256" in result.stdout
    assert "--expected_planner_lr 1.0e-4" in result.stdout
    assert "--expected_planner_weight_decay 1.0e-4" in result.stdout
    assert "--expected_planner_flow_num_inference_steps 16" in result.stdout
    assert "--expected_planner_flow_inference_noise_std 0.0" in result.stdout
    assert "--expected_hand_designed_planner_state_history_steps 10" in result.stdout
    assert "--expected_hand_designed_planner_command_past_steps 1" in result.stdout
    assert "--expected_hand_designed_planner_command_future_steps 25" in result.stdout
    assert "PRETRAIN_UPDATES=2000" in result.stdout
    assert "STATE_HISTORY_STEPS=10" in result.stdout
    assert "COMMAND_PAST_STEPS=1" in result.stdout
    assert "COMMAND_FUTURE_STEPS=25" in result.stdout
    assert "--require_selected" in result.stdout
    assert "--use_selected_variants" in result.stdout
    assert "--min_oracle_survival 800" in result.stdout
    assert "--min_oracle_success_rate 0.8" in result.stdout
    assert (
        "--output_json /tmp/lafan1_heldout_multiseed/interface_comparison_audit.json"
        in result.stdout
    )
    assert (
        "--output_md /tmp/lafan1_heldout_multiseed/interface_comparison_audit.md"
        in result.stdout
    )

    with_latent = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "SEEDS": "0",
            "MODEL_SIZE": "medium",
            "SAMPLE_BUDGETS": "10000",
            "OUTPUT_PREFIX": "/tmp/lafan1_heldout_with_latent",
            "AGGREGATE_OUTPUT_DIR": "/tmp/lafan1_heldout_with_latent_multiseed",
            "FULL_BODY_TRAJECTORY_CHECKPOINT": "/tmp/full.pt",
            "EE_TRAJECTORY_CHECKPOINT": "/tmp/ee.pt",
            "RUN_LATENT_BASELINE": "1",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/tmp/latent_low.pt",
            "LATENT_SKILL_CHECKPOINT": "/tmp/latent_skill.pt",
            "LATENT_PLANNER_CHECKPOINT": "/tmp/latent_planner.pt",
            "LATENT_DATASET_PATH": "/tmp/latent_dataset",
            "SELECTED_SAMPLE_COUNT": "10000",
            "MIN_ORACLE_SURVIVAL": "800",
            "MIN_ORACLE_SUCCESS_RATE": "0.8",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "RUN_LATENT_BASELINE=1" in with_latent.stdout
    assert "LATENT_MOTION_NAME=" in with_latent.stdout
    assert "LATENT_TRAJECTORY_NAME=" in with_latent.stdout
    assert "LATENT_DATASET_PATH=/tmp/latent_dataset" in with_latent.stdout
    assert (
        "--expected_interfaces latent_skill ee_trajectory full_body_trajectory"
        in with_latent.stdout
    )
    assert (
        "--expected_selected_interfaces latent_skill ee_trajectory full_body_trajectory"
        in with_latent.stdout
    )
    assert "--require_provenance" in with_latent.stdout
    assert "--expected_planner_num_updates 2000" in with_latent.stdout
    assert "--expected_planner_finetune_num_updates 2000" in with_latent.stdout
    assert "--expected_planner_batch_size 256" in with_latent.stdout
    assert "--expected_planner_lr 1.0e-4" in with_latent.stdout
    assert "--expected_planner_weight_decay 1.0e-4" in with_latent.stdout
    assert "--expected_planner_flow_num_inference_steps 16" in with_latent.stdout
    assert "--expected_planner_flow_inference_noise_std 0.0" in with_latent.stdout
    assert "PRETRAIN_UPDATES=2000" in with_latent.stdout
    assert "--require_selected" in with_latent.stdout
    assert "--use_selected_variants" in with_latent.stdout
    assert "--min_oracle_survival 800" in with_latent.stdout
    assert "--min_oracle_success_rate 0.8" in with_latent.stdout
    assert (
        "--output_json /tmp/lafan1_heldout_with_latent_multiseed/interface_comparison_audit.json"
        in with_latent.stdout
    )
    assert (
        "--output_md /tmp/lafan1_heldout_with_latent_multiseed/interface_comparison_audit.md"
        in with_latent.stdout
    )

    latent_only = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "DRY_RUN": "1",
            "SEEDS": "0",
            "INTERFACES": "",
            "RUN_LATENT_BASELINE": "1",
            "MODEL_SIZE": "medium",
            "SAMPLE_BUDGETS": "10000",
            "OUTPUT_PREFIX": "/tmp/lafan1_heldout_latent_only",
            "AGGREGATE_OUTPUT_DIR": "/tmp/lafan1_heldout_latent_only_multiseed",
            "LATENT_LOW_LEVEL_CHECKPOINT": "/tmp/latent_low.pt",
            "LATENT_SKILL_CHECKPOINT": "/tmp/latent_skill.pt",
            "LATENT_PLANNER_CHECKPOINT": "/tmp/latent_planner.pt",
            "LATENT_DATASET_PATH": "/tmp/latent_dataset",
            "SELECTED_SAMPLE_COUNT": "10000",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--expected_interfaces latent_skill --expected_seeds" in latent_only.stdout
    assert "--expected_selected_interfaces latent_skill" in latent_only.stdout
    assert "ee_trajectory=chunked_transformer" not in latent_only.stdout
    assert "full_body_trajectory=chunked_transformer" not in latent_only.stdout


def _test_lafan1_heldout_multiseed_wrapper_missing_manifest(tmp: Path) -> None:
    script = (
        Path(__file__).resolve().parent
        / "run_lafan1_heldout_strong_interface_multiseed.sh"
    )
    missing_manifest = tmp / "missing_manifest.json"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            "PATH": "/usr/bin:/bin",
            "FULL_MANIFEST": str(missing_manifest),
            "SEEDS": "0 1",
            "FULL_BODY_TRAJECTORY_CHECKPOINT": "/tmp/full.pt",
            "EE_TRAJECTORY_CHECKPOINT": "/tmp/ee.pt",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Full multi-motion manifest not found" in result.stderr
    assert "run_lafan1_heldout_strong_interface_comparison.sh" not in result.stdout


def _test_preflight_interface_comparison(tmp: Path) -> None:
    motion_path = tmp / "motion.npz"
    motion_path.write_bytes(b"placeholder")
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "trajectories": {
                        "lafan1_csv": [
                            {
                                "name": "motion",
                                "path": "motion.npz",
                                "input_fps": 50.0,
                            }
                        ]
                    }
                },
                "metadata": {"paths_are_relative_to_manifest": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    args = argparse.Namespace(
        train_manifest=manifest_path,
        eval_manifest=manifest_path,
        interfaces=["ee_trajectory"],
        run_latent=False,
        full_body_checkpoint=None,
        ee_checkpoint=checkpoint,
        latent_low_level_checkpoint=None,
        latent_skill_checkpoint=None,
        latent_planner_checkpoint=None,
        latent_dataset_path=None,
        require_latent_dataset_path=False,
        allow_missing_latent_dataset=False,
        model_sizes=["tiny"],
        sample_budgets=["10"],
        output_roots=[tmp / "new_output"],
        fail_if_output_exists=False,
        allow_missing_checkpoints=False,
        allow_missing_motion_files=False,
    )
    report = run_preflight(args)
    assert report["status"] == "pass"
    assert report["interfaces"] == ["ee_trajectory"]

    args.ee_checkpoint = tmp / "missing.pt"
    try:
        run_preflight(args)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing checkpoint should fail preflight")

    args.ee_checkpoint = checkpoint
    args.run_latent = True
    args.latent_low_level_checkpoint = checkpoint
    args.latent_skill_checkpoint = checkpoint
    args.latent_planner_checkpoint = checkpoint
    args.latent_dataset_path = None
    args.require_latent_dataset_path = True
    try:
        run_preflight(args)
    except FileNotFoundError as exc:
        assert "latent_dataset" in str(exc)
    else:
        raise AssertionError("missing latent dataset should fail preflight")

    latent_dataset = tmp / "latent_dataset"
    latent_dataset.mkdir()
    args.latent_dataset_path = latent_dataset
    report = run_preflight(args)
    assert report["run_latent"] is True
    assert report["latent_dataset_path"] == str(latent_dataset)


def main() -> None:
    _test_target_roundtrip()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _test_samples_and_checkpoint(tmp_path)
        _test_chunked_transformer_checkpoint(tmp_path)
        _test_chunked_transformer_microbatch_training(tmp_path)
        _test_latent_skill_offline_eval(tmp_path)
        _test_summary_config_sample_count(tmp_path)
        _test_summary_offline_eval_rows(tmp_path)
        _test_summary_success_rates(tmp_path)
        _test_aggregate_paper_metrics_include_offline()
        _test_aggregate_seed_prefers_root_name(tmp_path)
        _test_audit_interface_comparison(tmp_path)
        _test_capacity_metadata_backfill(tmp_path)
        _test_lafan1_heldout_wrapper_missing_manifest(tmp_path)
        _test_lafan1_heldout_wrapper_manual_split_skips_full_manifest(tmp_path)
        _test_lafan1_heldout_wrapper_latent_requires_dataset_path(tmp_path)
        _test_multiseed_wrapper_dry_run()
        _test_fair_runner_omits_empty_latent_motion_filter()
        _test_strong_runner_allows_empty_interfaces()
        _test_cluster_job_launcher_dry_run()
        _test_run_provenance_writer(tmp_path)
        _test_unitree_usd_cache_env_hook()
        _test_cluster_submitter_dry_run()
        _test_cluster_submitter_auto_syncs_local_checkpoints(tmp_path)
        _test_cluster_submitter_auto_syncs_extra_aggregate_artifacts(tmp_path)
        _test_cluster_interface_forwards_remote_env_args()
        _test_shell_wrappers_do_not_require_git_repo()
        _test_lafan1_heldout_multiseed_wrapper_dry_run()
        _test_lafan1_heldout_multiseed_wrapper_missing_manifest(tmp_path)
        _test_manifest_splitter_rebases_relative_paths(tmp_path)
        _test_preflight_interface_comparison(tmp_path)
    _test_manifest_splitter()
    _test_analyze_interface_sweep()
    _test_flow_training_reduces_error()
    print("[INFO] interface planner smoke tests passed")


if __name__ == "__main__":
    main()
