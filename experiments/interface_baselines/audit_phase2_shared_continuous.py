#!/usr/bin/env python3
"""Audit one matched Phase 2 DiffSR/full-body/EE comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


INTERFACES = ("latent_skill", "full_body_trajectory", "ee_trajectory")
BACKBONE_KEYS = (
    "planner_type",
    "state_dim",
    "d_model",
    "num_layers",
    "num_heads",
    "feedforward_dim",
    "patch_dim",
    "num_state_tokens",
    "dropout",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for interface in INTERFACES:
        flag = interface.replace("_trajectory", "").replace("_skill", "")
        parser.add_argument(f"--{flag}_checkpoint", type=Path, required=True)
        parser.add_argument(f"--{flag}_merge_manifest", type=Path, required=True)
        parser.add_argument(f"--{flag}_summary", type=Path, required=True)
    parser.add_argument("--expected_seed", type=int, required=True)
    parser.add_argument("--expected_history_steps", type=int, required=True)
    parser.add_argument("--expected_planner_interval", type=int, required=True)
    parser.add_argument("--expected_pretrain_updates", type=int, required=True)
    parser.add_argument("--expected_finetune_updates", type=int, required=True)
    parser.add_argument("--expected_rows_per_stage", type=int, required=True)
    parser.add_argument("--expected_eval_steps", type=int, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _paths(args: argparse.Namespace, interface: str) -> tuple[Path, Path, Path]:
    flag = interface.replace("_trajectory", "").replace("_skill", "")
    return (
        getattr(args, f"{flag}_checkpoint").expanduser().resolve(),
        getattr(args, f"{flag}_merge_manifest").expanduser().resolve(),
        getattr(args, f"{flag}_summary").expanduser().resolve(),
    )


def _summary_step_budget(summary: dict[str, Any]) -> int:
    if summary.get("max_steps") is None:
        raise ValueError("Summary does not record its requested max_steps budget.")
    return int(summary["max_steps"])


def main() -> None:
    args = _parse_args()
    errors: list[str] = []
    records: dict[str, dict[str, Any]] = {}

    for interface in INTERFACES:
        checkpoint_path, merge_path, summary_path = _paths(args, interface)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata", {})
        sample_metadata = metadata.get("sample_metadata", {})
        planner_config = checkpoint.get("planner_config", {})
        target_spec = checkpoint.get("target_spec", {})
        merge = _load_json(merge_path)
        summary = _load_json(summary_path)
        observation_spec = sample_metadata.get("planner_observation_spec", {})
        sources = merge.get("sources", [])

        record = {
            "checkpoint": str(checkpoint_path),
            "merge_manifest": str(merge_path),
            "summary": str(summary_path),
            "sample_format": sample_metadata.get("sample_format"),
            "observation_spec": observation_spec,
            "planner_config": planner_config,
            "target_spec": target_spec,
            "seed": sample_metadata.get("seed"),
            "planner_interval_steps": sample_metadata.get("planner_interval_steps"),
            "control_rate_hz": sample_metadata.get("control_rate_hz"),
            "pretrain_updates": metadata.get("pretrain_num_updates"),
            "finetune_updates": metadata.get("finetune_num_updates"),
            "merge_rows": merge.get("row_count"),
            "source_selected_rows": [
                source.get("selected_row_count") for source in sources
            ],
            "evaluation_step_budget": _summary_step_budget(summary),
            "evaluation_steps_run": summary.get("steps_run"),
        }
        records[interface] = record

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(f"{interface}: {message}")

        require(target_spec.get("interface") == interface, "target interface mismatch")
        require(
            sample_metadata.get("sample_format")
            == {"name": "causal_interface_planner_sample", "version": 1},
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
            observation_spec.get("reference_features") == [],
            "planner observation contains reference features",
        )
        require(
            int(sample_metadata.get("planner_interval_steps", -1))
            == int(args.expected_planner_interval),
            "planner interval mismatch",
        )
        require(
            abs(float(sample_metadata.get("control_rate_hz", -1.0)) - 50.0) < 1e-6,
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
            [source.get("collection_stage") for source in sources]
            == ["oracle_rollout", "planner_rollout"],
            "merge stages must be oracle_rollout then planner_rollout",
        )
        require(
            [int(source.get("selected_row_count", -1)) for source in sources]
            == [int(args.expected_rows_per_stage)] * 2,
            "selected rows per stage mismatch",
        )
        require(
            int(merge.get("row_count", -1)) == 2 * int(args.expected_rows_per_stage),
            "merged row count mismatch",
        )
        require(
            _summary_step_budget(summary) == int(args.expected_eval_steps),
            "closed-loop evaluation step budget mismatch",
        )

    baseline = records["latent_skill"]["planner_config"]
    for interface in INTERFACES[1:]:
        candidate = records[interface]["planner_config"]
        for key in BACKBONE_KEYS:
            if candidate.get(key) != baseline.get(key):
                errors.append(
                    f"{interface}: planner backbone {key}={candidate.get(key)!r} "
                    f"does not match latent_skill {baseline.get(key)!r}"
                )

    payload = {
        "passed": not errors,
        "errors": errors,
        "expected": {
            "seed": int(args.expected_seed),
            "history_steps": int(args.expected_history_steps),
            "planner_interval_steps": int(args.expected_planner_interval),
            "pretrain_updates": int(args.expected_pretrain_updates),
            "finetune_updates": int(args.expected_finetune_updates),
            "rows_per_stage": int(args.expected_rows_per_stage),
            "eval_steps": int(args.expected_eval_steps),
        },
        "interfaces": records,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("Phase 2 audit failed:\n- " + "\n- ".join(errors))
    print(f"[PASS] Phase 2 shared continuous-interface audit: {output}")


if __name__ == "__main__":
    main()
