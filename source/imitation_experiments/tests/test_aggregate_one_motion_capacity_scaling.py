from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.capacity.aggregate_one_motion_capacity_scaling import INTERFACES, METRICS, aggregate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary(starts: list[int], scale: float = 1.0) -> dict:
    return {
        "aggregate": {
            "survival_rate": 1.0,
            "survival_steps_mean": 500.0,
            "threshold_tracking_success_rate": 0.8,
        },
        "metrics": {name: {"mean": scale} for name in METRICS},
        "start_trajectories": {"local_steps": starts},
        "planner_inference_latency_ms": {"mean": 2.0},
    }


def _target_dim(interface: str) -> int:
    return {
        "latent_skill": 256,
        "full_body_trajectory": 670,
        "ee_trajectory": 360,
        "root_qpos": 380,
        "root_points5": 240,
    }[interface]


def _config(interface: str, stage: str) -> dict:
    target_dim = _target_dim(interface)
    return {
        "planner_type": "causal_interface_transformer_flow",
        "parameter_count": 1000 + target_dim,
        "target_dim": target_dim,
        "state_key": "planner_state",
        "source_sample_count": 1000 if stage == "demonstration_only" else 2000,
        "selected_sample_count": 1000 if stage == "demonstration_only" else 2000,
        "num_updates": 2000,
        "batch_size": 256,
        "micro_batch_size": 32,
        "flow_num_inference_steps": 16,
        "flow_inference_noise_std": 0.0,
        "args": {"seed": 0},
        "sample_metadata": {"motion_name": "walk1_subject1"},
    }


# Distinct oracle scales per interface so normalization is checkable.
_ORACLE_SCALE = {
    "latent_skill": 0.5,
    "full_body_trajectory": 0.25,
    "ee_trajectory": 0.2,
    "root_qpos": 0.4,
    "root_points5": 0.3,
}


def _tree(
    tmp_path: Path, *, mismatched_start: bool = False
) -> tuple[Path, dict[str, Path]]:
    scaling = tmp_path / "scaling"
    oracle_paths: dict[str, Path] = {}
    for interface in INTERFACES:
        oracle = tmp_path / f"{interface}_oracle.json"
        _write_json(oracle, _summary([1, 2], _ORACLE_SCALE[interface]))
        oracle_paths[interface] = oracle
        for stage, eval_dir in (
            ("demonstration_only", "eval_pretrained_10starts"),
            ("rollout_finetuned", "eval_finetuned_10starts"),
        ):
            starts = (
                [1, 3]
                if mismatched_start and interface == "full_body_trajectory"
                else [1, 2]
            )
            root = scaling / "tiny" / interface
            _write_json(root / eval_dir / "summary.json", _summary(starts))
            train_dir = (
                "planner_pretrain"
                if stage == "demonstration_only"
                else "planner_finetune"
            )
            _write_json(root / train_dir / "config.json", _config(interface, stage))
            checkpoint = root / train_dir / "checkpoints/latest.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            if stage == "rollout_finetuned":
                _write_json(
                    root / "demonstration_and_rollout_samples/merge_manifest.json",
                    {
                        "row_count": 2000,
                        "sources": [
                            {"selected_row_count": 1000},
                            {"selected_row_count": 1000},
                        ],
                    },
                )
    return scaling, oracle_paths


def test_aggregate_records_both_stages_and_oracle_normalization(tmp_path: Path) -> None:
    scaling, oracle_paths = _tree(tmp_path)
    payload = aggregate(
        scaling_root=scaling,
        oracle_paths=oracle_paths,
        sizes=["tiny"],
    )
    assert len(payload["rows"]) == 2 * len(INTERFACES)
    latent = next(row for row in payload["rows"] if row["interface"] == "latent_skill")
    explicit = next(
        row for row in payload["rows"] if row["interface"] == "full_body_trajectory"
    )
    ee = next(row for row in payload["rows"] if row["interface"] == "ee_trajectory")
    # Row metric is 1.0; oracle-normalized = 1.0 / oracle_scale.
    assert latent["oracle_normalized_metrics"]["tracking_mpjpe_mm"] == 2.0
    assert explicit["oracle_normalized_metrics"]["tracking_mpjpe_mm"] == 4.0
    assert ee["oracle_normalized_metrics"]["tracking_mpjpe_mm"] == 5.0
    assert latent["planner_family"] == "causal_interface_transformer_flow"
    assert latent["planner_seed"] == 0
    assert payload["evaluation_starts"] == [1, 2]


def test_aggregate_over_available_interfaces(tmp_path: Path) -> None:
    # Aggregation covers whichever interfaces have an oracle, not a fixed set:
    # a partial grid must still produce a table for the interfaces present.
    scaling, oracle_paths = _tree(tmp_path)
    for absent in set(INTERFACES) - {"latent_skill", "full_body_trajectory"}:
        oracle_paths.pop(absent)
    payload = aggregate(scaling_root=scaling, oracle_paths=oracle_paths, sizes=["tiny"])
    assert set(payload["interfaces"]) == {"latent_skill", "full_body_trajectory"}
    assert len(payload["rows"]) == 2 * 2  # 2 interfaces x 2 stages


def test_aggregate_requires_latent_and_one_explicit(tmp_path: Path) -> None:
    scaling, oracle_paths = _tree(tmp_path)
    for absent in set(INTERFACES) - {"latent_skill"}:
        oracle_paths.pop(absent)
    with pytest.raises(ValueError, match="at least one explicit"):
        aggregate(scaling_root=scaling, oracle_paths=oracle_paths, sizes=["tiny"])


def test_aggregate_rejects_different_evaluation_starts(tmp_path: Path) -> None:
    scaling, oracle_paths = _tree(tmp_path, mismatched_start=True)
    with pytest.raises(ValueError, match="Mismatched evaluation starts"):
        aggregate(
            scaling_root=scaling,
            oracle_paths=oracle_paths,
            sizes=["tiny"],
        )
