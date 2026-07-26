from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aggregate_one_motion_capacity_seeds import CORE_METRICS, aggregate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capacity_summary(root: Path, seed: int, *, starts: list[int]) -> Path:
    root.mkdir(parents=True)
    rows = []
    for interface in ("latent_skill", "full_body_trajectory"):
        for stage in ("demonstration_only", "rollout_finetuned"):
            config = root / f"{interface}_{stage}_config.json"
            config.write_text(
                json.dumps(
                    {
                        "args": {"seed": seed},
                        "planner_type": "causal_interface_transformer_flow",
                    }
                )
            )
            mpjpe = 1.0 + seed + (1.0 if interface == "full_body_trajectory" else 0.0)
            rows.append(
                {
                    "size": "tiny",
                    "interface": interface,
                    "stage": stage,
                    "parameter_count": 1000 if interface == "latent_skill" else 1010,
                    "target_dim": 256 if interface == "latent_skill" else 670,
                    "planner_inference_latency_ms": 2.0 + seed,
                    "survival_rate": 1.0,
                    "metrics": {name: mpjpe for name in CORE_METRICS},
                    "oracle_normalized_metrics": {"tracking_mpjpe_mm": mpjpe / 2.0},
                    "artifacts": {
                        "config": str(config),
                        "config_sha256": _sha256(config),
                    },
                }
            )
    path = root / "capacity_results.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_starts": starts,
                "oracles": {
                    "latent_skill": {"sha256": "latent"},
                    "full_body_trajectory": {"sha256": "explicit"},
                },
                "rows": rows,
            }
        )
    )
    return path


def test_aggregate_repeated_seeds_and_fixed_performance_minimum(tmp_path: Path) -> None:
    inputs = [
        _capacity_summary(tmp_path / f"seed{seed}", seed, starts=[1, 2])
        for seed in (0, 1)
    ]
    payload = aggregate(
        inputs=inputs,
        min_seeds=2,
        survival_target=0.9,
        normalized_mpjpe_target=2.0,
    )
    latent_demo = next(
        item
        for item in payload["summaries"]
        if item["interface"] == "latent_skill" and item["stage"] == "demonstration_only"
    )
    assert latent_demo["seeds"] == [0, 1]
    assert latent_demo["metrics"]["tracking_mpjpe_mm"]["mean"] == 1.5
    assert latent_demo["planner_inference_latency_ms"]["mean"] == 2.5
    assert latent_demo["output_values_per_second"] == 1280
    latent_minimum = next(
        item
        for item in payload["minimum_tested_sizes"]
        if item["interface"] == "latent_skill" and item["stage"] == "demonstration_only"
    )
    assert latent_minimum["minimum_tested_parameter_count"] == 1000
    assert len(payload["paired_seed_rows"]) == 4


def test_aggregate_rejects_mismatched_starts(tmp_path: Path) -> None:
    inputs = [
        _capacity_summary(tmp_path / "seed0", 0, starts=[1, 2]),
        _capacity_summary(tmp_path / "seed1", 1, starts=[1, 3]),
    ]
    with pytest.raises(ValueError, match="Mismatched evaluation starts"):
        aggregate(
            inputs=inputs,
            min_seeds=2,
            survival_target=0.9,
            normalized_mpjpe_target=2.0,
        )
