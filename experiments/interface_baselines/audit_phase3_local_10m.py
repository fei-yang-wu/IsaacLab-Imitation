#!/usr/bin/env python3
"""Audit matched local 10M-frame Future-CVAE and token oracle runs."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


INTERFACES = ("future_cvae", "per_step_token_sequence")
REQUIRED_METRICS = (
    "action_l2",
    "root_height_error_m",
    "root_ori_error_rad",
    "tracking_mpjpe_mm",
    "tracking_velocity_distance_mps",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("future", "token"):
        parser.add_argument(f"--{prefix}_checkpoint", type=Path)
        parser.add_argument(f"--{prefix}_oracle_summary", type=Path)
    parser.add_argument(
        "--interfaces",
        nargs="+",
        choices=INTERFACES,
        default=list(INTERFACES),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--expected_seed", type=int, required=True)
    parser.add_argument("--target_frames", type=int, required=True)
    parser.add_argument("--frames_per_batch", type=int, required=True)
    parser.add_argument("--eval_control_steps", type=int, required=True)
    parser.add_argument("--oracle_success_threshold", type=float, default=0.8)
    parser.add_argument("--bc_coef", type=float, default=0.0)
    parser.add_argument("--rollout_bc_coef", type=float, default=0.0)
    parser.add_argument("--bc_pretrain_updates", type=int, default=0)
    parser.add_argument(
        "--reconstructed_action_mode",
        choices=("next_pose", "pd_compensated"),
        default="next_pose",
    )
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _checkpoint_frames(path: Path) -> int:
    match = re.fullmatch(r"model_step_([0-9]+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Checkpoint does not encode its frame count: {path}")
    return int(match.group(1))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    manifest = args.manifest.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    expected_frames = (
        (int(args.target_frames) + int(args.frames_per_batch) - 1)
        // int(args.frames_per_batch)
    ) * int(args.frames_per_batch)

    selected = set(args.interfaces)
    for prefix, interface in (
        ("future", "future_cvae"),
        ("token", "per_step_token_sequence"),
    ):
        if interface not in selected:
            continue
        checkpoint_arg = getattr(args, f"{prefix}_checkpoint")
        summary_arg = getattr(args, f"{prefix}_oracle_summary")
        if checkpoint_arg is None or summary_arg is None:
            raise ValueError(
                f"{interface} requires --{prefix}_checkpoint and "
                f"--{prefix}_oracle_summary."
            )
        checkpoint = checkpoint_arg.expanduser().resolve()
        summary_path = summary_arg.expanduser().resolve()
        summary = _load_json(summary_path)
        metadata = _mapping(summary.get("metadata"))
        aggregate = _mapping(summary.get("aggregate"))
        metrics = _mapping(summary.get("metrics"))
        frames = _checkpoint_frames(checkpoint)
        success = float(aggregate.get("closed_loop_success_rate", float("nan")))
        oracle_passed = success >= float(args.oracle_success_threshold)
        record = {
            "checkpoint": str(checkpoint),
            "checkpoint_frames": frames,
            "oracle_summary": str(summary_path),
            "oracle_success_rate": success,
            "oracle_passed": oracle_passed,
            "survival_steps_mean": aggregate.get("survival_steps_mean"),
            "survival_fraction_mean": aggregate.get("survival_fraction_mean"),
            "horizon_completion_rate": aggregate.get("horizon_completion_rate"),
            "tracking_success_rate": aggregate.get("tracking_success_rate"),
            "done_rate": aggregate.get("done_rate"),
            "tracking_mpjpe_mm": _mapping(metrics.get("tracking_mpjpe_mm")).get("mean"),
        }
        records[interface] = record

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(f"{interface}: {message}")

        require(checkpoint.is_file(), "checkpoint is missing")
        require(frames == expected_frames, "checkpoint frame count mismatch")
        require(metadata.get("interface") == interface, "summary interface mismatch")
        require(
            metadata.get("collection_stage") == "oracle_rollout",
            "summary is not an oracle rollout",
        )
        require(
            int(metadata.get("seed", -1)) == int(args.expected_seed), "seed mismatch"
        )
        summary_manifest = metadata.get("motion_manifest")
        require(
            summary_manifest is not None
            and Path(str(summary_manifest)).expanduser().resolve() == manifest,
            "manifest mismatch",
        )
        summary_dataset = metadata.get("dataset_path")
        require(
            summary_dataset is not None
            and Path(str(summary_dataset)).expanduser().resolve() == dataset_path,
            "dataset path mismatch",
        )
        require(
            int(summary.get("max_steps", -1)) == int(args.eval_control_steps),
            "evaluation step budget mismatch",
        )
        steps_run = int(summary.get("steps_run", -1))
        require(
            1 <= steps_run <= int(args.eval_control_steps),
            "actual evaluation step count is invalid",
        )
        require(summary.get("evaluation_only") is True, "evaluation saved samples")
        require(summary.get("stop_after_done") is True, "evaluation ignores done")
        require(
            "closed_loop_success_rate" in aggregate,
            "missing closed-loop success rate",
        )
        for metric_name in REQUIRED_METRICS:
            require(metric_name in metrics, f"missing metric {metric_name}")

    quality_failures = [
        interface
        for interface, record in records.items()
        if not bool(record["oracle_passed"])
    ]
    return {
        "passed": not errors and not quality_failures,
        "protocol_passed": not errors,
        "oracle_gate_passed": not quality_failures,
        "errors": errors,
        "quality_failures": quality_failures,
        "expected": {
            "manifest": str(manifest),
            "dataset_path": str(dataset_path),
            "seed": int(args.expected_seed),
            "target_frames": int(args.target_frames),
            "frames_per_batch": int(args.frames_per_batch),
            "effective_frames": expected_frames,
            "eval_control_steps": int(args.eval_control_steps),
            "oracle_success_threshold": float(args.oracle_success_threshold),
            "interfaces": sorted(selected),
            "bc_coef": float(args.bc_coef),
            "rollout_bc_coef": float(args.rollout_bc_coef),
            "bc_pretrain_updates": int(args.bc_pretrain_updates),
            "reconstructed_action_mode": str(args.reconstructed_action_mode),
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
        raise SystemExit(
            "10M protocol audit failed:\n- " + "\n- ".join(payload["errors"])
        )
    if payload["quality_failures"]:
        raise SystemExit(
            "10M oracle gate failed for: " + ", ".join(payload["quality_failures"])
        )
    print(f"[PASS] Phase 3 local 10M qualification: {output}")


if __name__ == "__main__":
    main()
