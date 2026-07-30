#!/usr/bin/env python3
"""Summarize final per-goal results from a BONES-SEED interface comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


AGGREGATE_METRICS = (
    "return_sum_mean",
    "survival_steps_mean",
    "tracking_success_rate",
    "done_rate",
)
ROLLOUT_METRICS = (
    "tracking_mpjpe_mm",
    "root_pos_xy_error_m",
    "root_height_error_m",
    "root_ori_error_rad",
    "joint_pos_rmse_rad",
    "ee_pos_error_m",
    "ee_ori_error_rad",
    "action_delta_l2",
    "tracking_velocity_distance_mps",
    "tracking_acceleration_distance_mps2",
)
SUMMARY_METRICS = ("planner_latency_ms",)
METRICS = AGGREGATE_METRICS + ROLLOUT_METRICS + SUMMARY_METRICS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _metric(summary: dict[str, Any], name: str) -> float:
    aggregate = summary.get("aggregate", {})
    metrics = summary.get("metrics", {})
    if name == "planner_latency_ms":
        value = summary.get("planner_inference_latency_ms", {}).get("mean", math.nan)
    elif name in ROLLOUT_METRICS:
        value = metrics.get(name, {}).get("mean", math.nan)
    elif name == "tracking_success_rate" and name not in aggregate:
        value = aggregate.get("threshold_tracking_success_rate", math.nan)
    else:
        value = aggregate.get(name, math.nan)
    return float(value) if value is not None else math.nan


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with JSON null without changing CSV rows."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else run_root / "summary" / "final_results.json"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else run_root / "summary" / "final_results.csv"
    )
    run_config = _json(run_root / "run_config.json")
    comparison = _json(run_root / "comparison_manifest.json")
    goals = [str(value) for value in run_config.get("goals", [])]
    interfaces = [str(value) for value in comparison.get("interfaces", [])]
    if not interfaces:
        interfaces = [
            interface
            for interface in ("latent_skill", "full_body_trajectory")
            if interface in comparison.get("final_summaries", {})
        ]
    if not interfaces:
        raise ValueError("Comparison manifest does not contain any interfaces.")

    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    changes_after_rollout: dict[str, dict[str, float]] = {}
    termination_counts: dict[str, Counter[str]] = {}
    interface_specs: dict[str, dict[str, float | int]] = {}
    push_perturbation_protocols: list[dict[str, Any]] = []
    for interface in interfaces:
        paths = comparison.get("final_summaries", {}).get(interface, [])
        pretrained_paths = comparison.get("pretrained_summaries", {}).get(interface, [])
        if len(paths) != len(goals):
            raise ValueError(f"{interface}: expected {len(goals)} final summaries")
        if pretrained_paths and len(pretrained_paths) != len(goals):
            raise ValueError(f"{interface}: expected {len(goals)} pretrained summaries")
        observed_specs: list[dict[str, float | int]] = []
        termination_counts[interface] = Counter()
        for goal_index, (goal, raw_path) in enumerate(zip(goals, paths, strict=True)):
            summary = _json(Path(str(raw_path)).expanduser().resolve())
            metadata = summary.get("metadata", {})
            push_perturbation_protocols.append(metadata.get("push_perturbation", {}))
            planner_metadata = metadata.get("planner_metadata", {})
            sample_metadata = planner_metadata.get("sample_metadata", {})
            target_dim = int(metadata.get("planner_target_dim", -1))
            planner_rate_hz = float(sample_metadata.get("planner_rate_hz", math.nan))
            observed_specs.append(
                {
                    "target_dim": target_dim,
                    "planner_rate_hz": planner_rate_hz,
                    "parameter_count": int(planner_metadata.get("parameter_count", -1)),
                    "pretrain_sample_count": int(
                        planner_metadata.get("checkpoint_metadata", {}).get(
                            "num_samples", -1
                        )
                    ),
                    "final_training_sample_count": int(
                        planner_metadata.get("num_samples", -1)
                    ),
                }
            )
            per_environment = summary.get("per_environment", [])
            termination_terms = sorted(
                {
                    str(term)
                    for environment in per_environment
                    for term in environment.get("termination_terms", [])
                }
            )
            termination_counts[interface].update(
                {
                    str(term): int(count)
                    for term, count in summary.get("aggregate", {})
                    .get("termination_cause_env_counts", {})
                    .items()
                }
            )
            row: dict[str, Any] = {
                "interface": interface,
                "goal": goal,
                "seed": int(run_config["seed"]),
                "steps_run": int(summary.get("steps_run", 0)),
                "stop_reason": str(summary.get("stop_reason", "")),
                "termination_terms": ";".join(termination_terms),
            }
            row.update({name: _metric(summary, name) for name in METRICS})
            rows.append(row)
            stage_rows.append({**row, "stage": "finetuned_planner_rollout"})
            if pretrained_paths:
                pretrained = _json(
                    Path(str(pretrained_paths[goal_index])).expanduser().resolve()
                )
                pretrained_environment = pretrained.get("per_environment", [])
                pretrained_terms = sorted(
                    {
                        str(term)
                        for environment in pretrained_environment
                        for term in environment.get("termination_terms", [])
                    }
                )
                pretrained_row: dict[str, Any] = {
                    "interface": interface,
                    "goal": goal,
                    "seed": int(run_config["seed"]),
                    "steps_run": int(pretrained.get("steps_run", 0)),
                    "stop_reason": str(pretrained.get("stop_reason", "")),
                    "termination_terms": ";".join(pretrained_terms),
                    "stage": "pretrained_demonstration",
                }
                pretrained_row.update(
                    {name: _metric(pretrained, name) for name in METRICS}
                )
                stage_rows.append(pretrained_row)

        if any(spec != observed_specs[0] for spec in observed_specs[1:]):
            raise ValueError(f"{interface}: planner interface metadata differs by goal")
        interface_spec = observed_specs[0]
        target_dim = int(interface_spec["target_dim"])
        planner_rate_hz = float(interface_spec["planner_rate_hz"])
        interface_specs[interface] = {
            **interface_spec,
            "planner_rollout_sample_count": int(
                interface_spec["final_training_sample_count"]
            )
            - int(interface_spec["pretrain_sample_count"]),
            "values_per_second": target_dim * planner_rate_hz,
            "float32_bits_per_second": target_dim * planner_rate_hz * 32.0,
        }

        interface_rows = [row for row in rows if row["interface"] == interface]
        aggregates[interface] = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in interface_rows]
            finite = [value for value in values if math.isfinite(value)]
            aggregates[interface][metric] = {
                "mean": fmean(finite) if finite else math.nan,
                "std_across_goals": pstdev(finite) if finite else math.nan,
                "goal_count": len(finite),
            }
        pretrained_interface_rows = [
            row
            for row in stage_rows
            if row["interface"] == interface
            and row["stage"] == "pretrained_demonstration"
        ]
        changes_after_rollout[interface] = {}
        if pretrained_interface_rows:
            final_by_goal = {str(row["goal"]): row for row in interface_rows}
            pretrained_by_goal = {
                str(row["goal"]): row for row in pretrained_interface_rows
            }
            for metric in METRICS:
                changes = [
                    float(final_by_goal[goal][metric])
                    - float(pretrained_by_goal[goal][metric])
                    for goal in goals
                    if math.isfinite(float(final_by_goal[goal][metric]))
                    and math.isfinite(float(pretrained_by_goal[goal][metric]))
                ]
                changes_after_rollout[interface][metric] = (
                    fmean(changes) if changes else math.nan
                )

    if any(
        protocol != push_perturbation_protocols[0]
        for protocol in push_perturbation_protocols[1:]
    ):
        raise ValueError("Push perturbation metadata differs across final summaries")
    payload = {
        "protocol": run_config.get("protocol"),
        "seed": int(run_config["seed"]),
        "goal_count": len(goals),
        "goals": goals,
        "per_goal": rows,
        "per_goal_by_stage": stage_rows,
        "interface_specs": interface_specs,
        "push_perturbation_protocol": push_perturbation_protocols[0],
        "termination_cause_counts_across_goals": {
            interface: dict(sorted(counts.items()))
            for interface, counts in termination_counts.items()
        },
        "aggregate_across_goals": aggregates,
        "mean_change_after_planner_rollout": changes_after_rollout,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote multi-goal summary: {output_json}")
    print(f"[INFO] Wrote per-goal table: {output_csv}")


if __name__ == "__main__":
    main()
