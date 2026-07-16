#!/usr/bin/env python3
"""Audit one matched Future-CVAE and per-step-token planner comparison."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


INTERFACES = ("future_cvae", "per_step_token_sequence")
COMMON_BACKBONE_KEYS = (
    "state_dim",
    "d_model",
    "num_layers",
    "num_heads",
    "feedforward_dim",
    "num_state_tokens",
    "dropout",
)
SAMPLE_FORMAT = {"name": "causal_interface_planner_sample", "version": 1}
COMMON_EVAL_METRICS = (
    "action_l2",
    "root_height_error_m",
    "root_ori_error_rad",
    "tracking_mpjpe_mm",
    "tracking_velocity_distance_mps",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("future", "token"):
        parser.add_argument(f"--{prefix}_checkpoint", type=Path, required=True)
        parser.add_argument(f"--{prefix}_merge_manifest", type=Path, required=True)
        parser.add_argument(f"--{prefix}_oracle_summary", type=Path, required=True)
        parser.add_argument(f"--{prefix}_rollout_summary", type=Path, required=True)
        parser.add_argument(
            f"--{prefix}_oracle_eval_summary", type=Path, required=True
        )
        parser.add_argument(
            f"--{prefix}_pretrained_eval_summary", type=Path, required=True
        )
        parser.add_argument(
            f"--{prefix}_finetuned_eval_summary", type=Path, required=True
        )
    parser.add_argument("--expected_seed", type=int, required=True)
    parser.add_argument("--expected_history_steps", type=int, required=True)
    parser.add_argument("--expected_planner_interval", type=int, required=True)
    parser.add_argument("--expected_pretrain_updates", type=int, required=True)
    parser.add_argument("--expected_finetune_updates", type=int, required=True)
    parser.add_argument("--expected_rows_per_stage", type=int, required=True)
    parser.add_argument("--expected_collected_rows_per_stage", type=int, default=0)
    parser.add_argument("--expected_collection_control_steps", type=int, default=0)
    parser.add_argument("--expected_eval_control_steps", type=int, required=True)
    parser.add_argument("--expected_token_horizon", type=int, default=10)
    parser.add_argument("--expected_codebook_size", type=int, default=512)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _paths(args: argparse.Namespace, interface: str) -> tuple[Path, ...]:
    prefix = "future" if interface == "future_cvae" else "token"
    return tuple(
        getattr(args, f"{prefix}_{suffix}").expanduser().resolve()
        for suffix in (
            "checkpoint",
            "merge_manifest",
            "oracle_summary",
            "rollout_summary",
            "oracle_eval_summary",
            "pretrained_eval_summary",
            "finetuned_eval_summary",
        )
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    """Return a machine-readable protocol audit; do not raise for mismatches."""
    errors: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    expected_collected_rows = (
        int(args.expected_collected_rows_per_stage)
        if int(args.expected_collected_rows_per_stage) > 0
        else int(args.expected_rows_per_stage)
    )

    for interface in INTERFACES:
        (
            checkpoint_path,
            merge_path,
            oracle_path,
            rollout_path,
            oracle_eval_path,
            pretrained_eval_path,
            finetuned_eval_path,
        ) = _paths(args, interface)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = _mapping(checkpoint.get("metadata"))
        sample_metadata = _mapping(metadata.get("sample_metadata"))
        planner_config = _mapping(checkpoint.get("planner_config"))
        target_spec = _mapping(checkpoint.get("target_spec"))
        observation_spec = _mapping(sample_metadata.get("planner_observation_spec"))
        merge = _load_json(merge_path)
        oracle = _load_json(oracle_path)
        rollout = _load_json(rollout_path)
        evaluations = {
            "oracle": _load_json(oracle_eval_path),
            "pretrained": _load_json(pretrained_eval_path),
            "finetuned": _load_json(finetuned_eval_path),
        }
        sources = merge.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        record = {
            "checkpoint": str(checkpoint_path),
            "merge_manifest": str(merge_path),
            "oracle_summary": str(oracle_path),
            "rollout_summary": str(rollout_path),
            "planner_config": dict(planner_config),
            "target_spec": dict(target_spec),
            "sample_metadata": dict(sample_metadata),
            "merge_rows": merge.get("row_count"),
            "source_selected_rows": [
                _mapping(source).get("selected_row_count") for source in sources
            ],
            "evaluations": {
                name: {
                    "path": str(path),
                    "max_steps": summary.get("max_steps"),
                    "steps_run": summary.get("steps_run"),
                    "aggregate": summary.get("aggregate"),
                    "metrics": summary.get("metrics"),
                }
                for name, path, summary in (
                    ("oracle", oracle_eval_path, evaluations["oracle"]),
                    ("pretrained", pretrained_eval_path, evaluations["pretrained"]),
                    ("finetuned", finetuned_eval_path, evaluations["finetuned"]),
                )
            },
        }
        records[interface] = record

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(f"{interface}: {message}")

        require(target_spec.get("interface") == interface, "target interface mismatch")
        expected_planner_type = (
            "causal_interface_transformer_flow"
            if interface == "future_cvae"
            else "causal_interface_transformer_categorical"
        )
        require(
            planner_config.get("planner_type") == expected_planner_type,
            "planner type mismatch",
        )
        require(
            sample_metadata.get("sample_format") == SAMPLE_FORMAT,
            "sample format is not causal_interface_planner_sample v1",
        )
        require(
            int(observation_spec.get("history_steps", -1))
            == int(args.expected_history_steps),
            "history_steps mismatch",
        )
        require(
            int(observation_spec.get("frame_dim", -1)) == 93,
            "causal frame width is not 93",
        )
        require(
            int(observation_spec.get("flat_dim", -1))
            == (int(args.expected_history_steps) + 1) * 93,
            "causal flat width mismatch",
        )
        require(
            observation_spec.get("reference_features") == [],
            "planner observation contains reference features",
        )
        require(
            int(sample_metadata.get("planner_interval_steps", -1))
            == int(args.expected_planner_interval),
            "planner interval mismatch",
        )
        require(
            abs(float(sample_metadata.get("control_rate_hz", -1.0)) - 50.0) < 1.0e-6,
            "control rate is not 50 Hz",
        )
        require(
            int(sample_metadata.get("seed", -1)) == int(args.expected_seed),
            "seed mismatch",
        )
        require(
            int(metadata.get("pretrain_num_updates", -1))
            == int(args.expected_pretrain_updates),
            "pretrain update count mismatch",
        )
        require(
            int(metadata.get("finetune_num_updates", -1))
            == int(args.expected_finetune_updates),
            "finetune update count mismatch",
        )
        require(len(sources) == 2, "merge must contain exactly two stages")
        require(
            [_mapping(source).get("collection_stage") for source in sources]
            == ["oracle_rollout", "planner_rollout"],
            "merge stages must be oracle_rollout then planner_rollout",
        )
        require(
            [int(_mapping(source).get("selected_row_count", -1)) for source in sources]
            == [int(args.expected_rows_per_stage)] * 2,
            "selected rows per stage mismatch",
        )
        require(
            int(merge.get("row_count", -1)) == 2 * int(args.expected_rows_per_stage),
            "merged row count mismatch",
        )
        for summary, expected_stage, name in (
            (oracle, "oracle_rollout", "oracle summary"),
            (rollout, "planner_rollout", "rollout summary"),
        ):
            summary_metadata = _mapping(summary.get("metadata"))
            require(
                summary_metadata.get("interface") == interface,
                f"{name} interface mismatch",
            )
            require(
                summary_metadata.get("collection_stage") == expected_stage,
                f"{name} collection stage mismatch",
            )
            require(
                int(summary.get("saved_rows", -1)) == expected_collected_rows,
                f"{name} row count mismatch",
            )
            if int(args.expected_collection_control_steps) > 0:
                require(
                    int(summary.get("control_steps", -1))
                    == int(args.expected_collection_control_steps),
                    f"{name} control-step count mismatch",
                )
            require(
                _mapping(summary_metadata.get("planner_observation_spec"))
                == observation_spec,
                f"{name} observation specification mismatch",
            )

        for eval_name, evaluation in evaluations.items():
            evaluation_metadata = _mapping(evaluation.get("metadata"))
            aggregate = _mapping(evaluation.get("aggregate"))
            metrics = _mapping(evaluation.get("metrics"))
            expected_stage = (
                "oracle_rollout" if eval_name == "oracle" else "planner_rollout"
            )
            require(
                evaluation_metadata.get("interface") == interface,
                f"{eval_name} evaluation interface mismatch",
            )
            require(
                evaluation_metadata.get("collection_stage") == expected_stage,
                f"{eval_name} evaluation command source mismatch",
            )
            require(
                int(evaluation.get("max_steps", -1))
                == int(args.expected_eval_control_steps),
                f"{eval_name} evaluation step budget mismatch",
            )
            steps_run = int(evaluation.get("steps_run", -1))
            require(
                1 <= steps_run <= int(args.expected_eval_control_steps),
                f"{eval_name} evaluation actual step count is invalid",
            )
            require(
                evaluation.get("evaluation_only") is True,
                f"{eval_name} evaluation saved training samples",
            )
            require(
                evaluation.get("stop_after_done") is True,
                f"{eval_name} evaluation does not stop metrics after done",
            )
            require(
                int(evaluation.get("saved_rows", -1)) == 0,
                f"{eval_name} evaluation saved planner rows",
            )
            for aggregate_key in (
                "survival_steps_mean",
                "done_rate",
                "tracking_success_rate",
            ):
                require(
                    aggregate_key in aggregate,
                    f"{eval_name} evaluation lacks {aggregate_key}",
                )
            for metric_name in COMMON_EVAL_METRICS:
                require(
                    metric_name in metrics,
                    f"{eval_name} evaluation lacks {metric_name}",
                )
            if eval_name != "oracle":
                planner_metric = (
                    "planner_target_rmse"
                    if interface == "future_cvae"
                    else "planner_token_accuracy"
                )
                require(
                    planner_metric in metrics,
                    f"{eval_name} evaluation lacks {planner_metric}",
                )

        encoding = sample_metadata.get("target_encoding")
        if interface == "future_cvae":
            require(
                not encoding or _mapping(encoding).get("kind") == "continuous",
                "Future-CVAE target must be continuous",
            )
            require(
                int(target_spec.get("target_dim", -1))
                == int(planner_config.get("target_dim", -2)),
                "Future-CVAE target width mismatch",
            )
        else:
            encoding_map = _mapping(encoding)
            require(
                encoding_map.get("kind") == "categorical_sequence",
                "token target is not categorical_sequence",
            )
            require(
                int(encoding_map.get("horizon", -1))
                == int(args.expected_token_horizon),
                "token horizon mismatch",
            )
            require(
                int(encoding_map.get("codebook_size", -1))
                == int(args.expected_codebook_size),
                "codebook size mismatch",
            )
            require(
                int(planner_config.get("token_horizon", -1))
                == int(args.expected_token_horizon),
                "categorical planner horizon mismatch",
            )
            require(
                int(planner_config.get("codebook_size", -1))
                == int(args.expected_codebook_size),
                "categorical planner codebook mismatch",
            )

    future_config = records["future_cvae"]["planner_config"]
    token_config = records["per_step_token_sequence"]["planner_config"]
    for key in COMMON_BACKBONE_KEYS:
        if future_config.get(key) != token_config.get(key):
            errors.append(
                "per_step_token_sequence: planner backbone "
                f"{key}={token_config.get(key)!r} does not match "
                f"future_cvae {future_config.get(key)!r}"
            )

    return {
        "passed": not errors,
        "errors": errors,
        "expected": {
            "seed": int(args.expected_seed),
            "history_steps": int(args.expected_history_steps),
            "planner_interval_steps": int(args.expected_planner_interval),
            "pretrain_updates": int(args.expected_pretrain_updates),
            "finetune_updates": int(args.expected_finetune_updates),
            "rows_per_stage": int(args.expected_rows_per_stage),
            "collected_rows_per_stage": expected_collected_rows,
            "collection_control_steps": int(args.expected_collection_control_steps),
            "eval_control_steps": int(args.expected_eval_control_steps),
            "token_horizon": int(args.expected_token_horizon),
            "codebook_size": int(args.expected_codebook_size),
        },
        "interfaces": records,
    }


def main() -> None:
    args = _parse_args()
    payload = audit(args)
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["errors"]:
        raise SystemExit("Phase 3 audit failed:\n- " + "\n- ".join(payload["errors"]))
    print(f"[PASS] Phase 3 latent-interface audit: {output}")


if __name__ == "__main__":
    main()
