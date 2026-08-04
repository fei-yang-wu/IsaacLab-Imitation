#!/usr/bin/env python3
"""Aggregate completed BONES-SEED interface runs across training seeds."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
from statistics import fmean, stdev
import sys
from typing import Any


PROTOCOL = "bones_seed_shared_multigoal_language_v1"
INTERFACES = ("latent_skill", "full_body_trajectory")
METRICS = (
    "return_sum_mean",
    "survival_rate",
    "fall_rate",
    "survival_steps_mean",
    "tracking_success_rate",
    "done_rate",
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
    "planner_latency_ms",
)
OPTIONAL_EARLY_FAILURE_METRICS = {
    "action_delta_l2",
    "tracking_acceleration_distance_mps2",
    "planner_latency_ms",
}
PAPER_TABLE_METRICS = (
    ("survival_rate", "Survival without falling", "higher"),
    ("tracking_mpjpe_mm", "Root-relative MPJPE (mm)", "lower"),
    ("root_pos_xy_error_m", "Root XY error (m)", "lower"),
    ("root_ori_error_rad", "Root orientation error (rad)", "lower"),
    ("joint_pos_rmse_rad", "Joint RMSE (rad)", "lower"),
    ("ee_pos_error_m", "End-effector error (m)", "lower"),
    ("action_delta_l2", "Action change L2", "lower"),
    ("planner_latency_ms", "Planner latency (ms)", "lower"),
)
HIGHER_IS_BETTER = {
    "return_sum_mean": True,
    "survival_rate": True,
    "fall_rate": False,
    "survival_steps_mean": True,
    "tracking_success_rate": True,
    "done_rate": False,
    "tracking_mpjpe_mm": False,
    "root_pos_xy_error_m": False,
    "root_height_error_m": False,
    "root_ori_error_rad": False,
    "joint_pos_rmse_rad": False,
    "ee_pos_error_m": False,
    "ee_ori_error_rad": False,
    "action_delta_l2": False,
    "tracking_velocity_distance_mps": False,
    "tracking_acceleration_distance_mps2": False,
    "planner_latency_ms": False,
}
PROTOCOL_FIELDS = (
    "protocol",
    "goals",
    "goal_count",
    "demo_rows_per_goal",
    "rollout_rows_per_goal",
    "skip_pretrained_closed_loop",
    "expected_demo_rows_per_interface",
    "expected_rollout_rows_per_interface",
    "eval_steps",
    "planner",
    "interfaces",
    "causal_state_dim",
    "language",
    "submission_gates",
    "paper_protocol_complete",
    "workflow_source_sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--minimum_seeds", type=int, default=3)
    parser.add_argument(
        "--expected_seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Exact training-seed set. The paper protocol fixes 0 1 2.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument(
        "--allow_non_paper_runs",
        action="store_true",
        help="Allow smoke or otherwise incomplete runs. Never use this for paper results.",
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty list.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _hierarchical_bootstrap_ci(
    values_by_seed: list[list[float]],
    *,
    samples: int,
    seed: int,
    label: str,
) -> tuple[float, float]:
    """Resample seeds, then goals within each selected seed."""
    if samples <= 0:
        raise ValueError("--bootstrap_samples must be positive.")
    if not values_by_seed or any(not values for values in values_by_seed):
        raise ValueError("Hierarchical bootstrap requires values for every seed.")
    rng = random.Random(f"{seed}:{label}")
    estimates: list[float] = []
    for _ in range(samples):
        selected_seed_indices = rng.choices(
            range(len(values_by_seed)), k=len(values_by_seed)
        )
        selected_seed_means: list[float] = []
        for seed_index in selected_seed_indices:
            goal_values = values_by_seed[seed_index]
            selected_seed_means.append(
                fmean(rng.choices(goal_values, k=len(goal_values)))
            )
        estimates.append(fmean(selected_seed_means))
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def _hierarchical_statistics(
    values_by_seed: dict[int, list[float]],
    *,
    samples: int,
    bootstrap_seed: int,
    label: str,
) -> dict[str, Any]:
    active = {seed: values for seed, values in values_by_seed.items() if values}
    if not active:
        return {
            "mean": None,
            "std_across_seed_means": None,
            "hierarchical_bootstrap_95_ci": None,
            "seed_means": {},
            "seed_count_with_values": 0,
            "value_count": 0,
        }
    ordered_seeds = sorted(active)
    nested = [active[seed] for seed in ordered_seeds]
    seed_means = [fmean(values) for values in nested]
    ci_low, ci_high = _hierarchical_bootstrap_ci(
        nested,
        samples=samples,
        seed=bootstrap_seed,
        label=label,
    )
    return {
        "mean": fmean(seed_means),
        "std_across_seed_means": stdev(seed_means) if len(seed_means) > 1 else 0.0,
        "hierarchical_bootstrap_95_ci": [ci_low, ci_high],
        "seed_means": dict(zip((str(seed) for seed in ordered_seeds), seed_means)),
        "seed_count_with_values": len(ordered_seeds),
        "value_count": sum(len(values) for values in nested),
    }


def _as_metric(value: Any) -> float:
    return float(value) if value is not None else math.nan


def _estimate_with_ci(statistics: dict[str, Any]) -> str:
    mean = statistics.get("mean")
    interval = statistics.get("hierarchical_bootstrap_95_ci")
    if mean is None or not math.isfinite(float(mean)):
        return "N/A"
    if not isinstance(interval, list) or len(interval) != 2:
        return f"{float(mean):.4g}"
    return f"{float(mean):.4g} [{float(interval[0]):.4g}, {float(interval[1]):.4g}]"


def _paired_estimate_with_ci(statistics: dict[str, Any]) -> str:
    return _estimate_with_ci(
        {
            "mean": statistics.get("mean_difference"),
            "hierarchical_bootstrap_95_ci": statistics.get(
                "hierarchical_bootstrap_95_ci"
            ),
        }
    )


def _write_paper_markdown(payload: dict[str, Any], output: Path) -> None:
    interface_statistics = payload["interface_statistics"]
    paired_statistics = payload["paired_statistics"]
    interface_specs = payload["interface_specs"]
    normalized = payload.get(
        "oracle_normalized_survival",
        payload.get("oracle_normalized_tracking_success"),
    )
    lines = [
        "# BONES-SEED latent versus explicit planner results",
        "",
        f"Seeds: {', '.join(str(seed) for seed in payload['seeds'])}. "
        f"Goals per seed: {payload['goal_count']}.",
        "",
        "Values are seed-balanced means with hierarchical 95% bootstrap intervals. "
        "The difference column is latent minus explicit.",
        "",
        "## Final closed-loop tracking",
        "",
        "| Metric | DiffSR latent | Explicit vanilla packet | Difference | Better |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for metric, label, better in PAPER_TABLE_METRICS:
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    _estimate_with_ci(interface_statistics[metric]["latent_skill"]),
                    _estimate_with_ci(
                        interface_statistics[metric]["full_body_trajectory"]
                    ),
                    _paired_estimate_with_ci(paired_statistics[metric]),
                    better,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interface cost",
            "",
            "| Interface | Output width | Values/s | Float32 bits/s | Planner parameters |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for interface, label in (
        ("latent_skill", "DiffSR latent"),
        ("full_body_trajectory", "Explicit vanilla packet"),
    ):
        spec = interface_specs[interface]
        lines.append(
            f"| {label} | {int(spec['target_dim'])} | "
            f"{float(spec['values_per_second']):.4g} | "
            f"{float(spec['float32_bits_per_second']):.4g} | "
            f"{int(spec['parameter_count'])} |"
        )

    if isinstance(normalized, dict):
        lines.extend(
            [
                "",
                "## Tracking success relative to each low-level oracle",
                "",
                "| Interface | Oracle success | Planner/oracle | 95% interval |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        normalized_interfaces = normalized["interfaces"]
        for interface, label in (
            ("latent_skill", "DiffSR latent"),
            ("full_body_trajectory", "Explicit vanilla packet"),
        ):
            values = normalized_interfaces[interface]
            interval = values["hierarchical_bootstrap_95_ci"]
            lines.append(
                f"| {label} | {float(values['oracle_tracking_success_rate']):.4g} "
                f"| {float(values['mean_planner_over_oracle']):.4g} | "
                f"[{float(interval[0]):.4g}, {float(interval[1]):.4g}] |"
            )

    lines.extend(
        [
            "",
            "This table is valid only with the accompanying passing protocol audits "
            "and cluster submission records.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _validate_stage_record(run_root: Path, run_config: dict[str, Any]) -> None:
    stage_path = run_root / "stages" / "summarize.json"
    stage = _json(stage_path)
    if stage.get("status") != "complete" or stage.get("stage") != "summarize":
        raise ValueError(f"Summarize stage is not complete: {stage_path}")
    if stage.get("workflow_source_sha256") != run_config.get("workflow_source_sha256"):
        raise ValueError(f"Summarize stage source hashes differ: {stage_path}")
    artifacts = stage.get("artifacts", {})
    expected_paths = {
        "comparison_manifest": run_root / "comparison_manifest.json",
        "summary_json": run_root / "summary" / "final_results.json",
        "summary_csv": run_root / "summary" / "final_results.csv",
        "protocol_audit": run_root
        / "protocol_checks"
        / "multigoal_language_audit.json",
    }
    if run_config.get("paper_protocol_complete") is True:
        expected_paths["cluster_submission"] = run_root / "cluster_submission.json"
    for required, path in expected_paths.items():
        artifact = artifacts.get(required)
        if not isinstance(artifact, dict):
            raise ValueError(
                f"Summarize stage lacks artifact {required!r}: {stage_path}"
            )
        recorded_path = Path(str(artifact.get("path", "")))
        relative_parts = path.relative_to(run_root).parts
        if recorded_path.parts[-len(relative_parts) :] != relative_parts:
            raise ValueError(
                f"Summarize artifact path does not identify {required!r}: "
                f"{recorded_path}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Summarize artifact is missing: {path}")
        if artifact.get("kind", "file") != "file":
            raise ValueError(f"Summarize artifact changed type: {path}")
        if artifact.get("sha256") != _sha256(path):
            raise ValueError(f"Summarize artifact hash changed: {path}")


def _protocol_contract(run_config: dict[str, Any]) -> dict[str, Any]:
    contract = {field: run_config.get(field) for field in PROTOCOL_FIELDS}
    contract["input_artifact_sha256"] = {
        name: artifact.get("sha256")
        for name, artifact in sorted(run_config.get("input_artifacts", {}).items())
    }
    return contract


def _load_run(
    run_root: Path, *, allow_non_paper_runs: bool
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = run_root.expanduser().resolve()
    run_config = _json(run_root / "run_config.json")
    summary = _json(run_root / "summary" / "final_results.json")
    audit = _json(run_root / "protocol_checks" / "multigoal_language_audit.json")
    _validate_stage_record(run_root, run_config)

    if run_config.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol in {run_root}")
    if audit.get("passed") is not True:
        raise ValueError(f"Protocol audit did not pass in {run_root}")
    if not allow_non_paper_runs:
        if run_config.get("paper_protocol_complete") is not True:
            raise ValueError(f"Run is not paper-protocol complete: {run_root}")
        if audit.get("paper_protocol_complete") is not True:
            raise ValueError(f"Audit is not paper-protocol complete: {run_root}")
        if audit.get("smoke_only") is not False:
            raise ValueError(f"Smoke run cannot be used for paper results: {run_root}")

    goals = [str(value) for value in run_config.get("goals", [])]
    rows = summary.get("per_goal")
    if not isinstance(rows, list):
        raise ValueError(f"Summary lacks per-goal rows: {run_root}")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise TypeError(f"Summary row is not an object: {run_root}")
        key = (str(raw_row.get("interface")), str(raw_row.get("goal")))
        if key in indexed:
            raise ValueError(f"Duplicate summary row {key}: {run_root}")
        indexed[key] = raw_row
    expected = {(interface, goal) for interface in INTERFACES for goal in goals}
    if set(indexed) != expected:
        raise ValueError(
            f"Summary interface/goal rows do not match run config: {run_root}"
        )

    seed = int(run_config["seed"])
    normalized: list[dict[str, Any]] = []
    for interface in INTERFACES:
        for goal in goals:
            row = indexed[(interface, goal)]
            if int(row.get("seed", -1)) != seed:
                raise ValueError(f"Summary seed mismatch for {(interface, goal)}")
            normalized_row: dict[str, Any] = {
                "run_root": str(run_root),
                "seed": seed,
                "goal": goal,
                "interface": interface,
                "steps_run": int(row.get("steps_run", -1)),
                "stop_reason": str(row.get("stop_reason", "")),
                "termination_terms": str(row.get("termination_terms", "")),
            }
            for metric in METRICS:
                value = _as_metric(row.get(metric, math.nan))
                if (
                    not math.isfinite(value)
                    and metric not in OPTIONAL_EARLY_FAILURE_METRICS
                ):
                    raise ValueError(
                        f"Non-finite {metric} for {(seed, interface, goal)}"
                    )
                normalized_row[metric] = value
            normalized.append(normalized_row)

    raw_stage_rows = summary.get("per_goal_by_stage")
    if not isinstance(raw_stage_rows, list):
        raise ValueError(f"Summary lacks before/after rows: {run_root}")
    stages = ("pretrained_demonstration", "finetuned_planner_rollout")
    stage_indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_row in raw_stage_rows:
        if not isinstance(raw_row, dict):
            raise TypeError(f"Before/after summary row is not an object: {run_root}")
        key = (
            str(raw_row.get("interface")),
            str(raw_row.get("goal")),
            str(raw_row.get("stage")),
        )
        if key in stage_indexed:
            raise ValueError(f"Duplicate before/after row {key}: {run_root}")
        stage_indexed[key] = raw_row
    expected_stage_rows = {
        (interface, goal, stage)
        for interface in INTERFACES
        for goal in goals
        for stage in stages
    }
    if set(stage_indexed) != expected_stage_rows:
        raise ValueError(f"Before/after rows do not match run config: {run_root}")
    normalized_stage_rows: list[dict[str, Any]] = []
    for interface in INTERFACES:
        for goal in goals:
            for stage in stages:
                row = stage_indexed[(interface, goal, stage)]
                if int(row.get("seed", -1)) != seed:
                    raise ValueError(
                        f"Before/after seed mismatch for {(interface, goal, stage)}"
                    )
                normalized_row = {
                    "run_root": str(run_root),
                    "seed": seed,
                    "goal": goal,
                    "interface": interface,
                    "stage": stage,
                    "steps_run": int(row.get("steps_run", -1)),
                    "stop_reason": str(row.get("stop_reason", "")),
                    "termination_terms": str(row.get("termination_terms", "")),
                }
                for metric in METRICS:
                    value = _as_metric(row.get(metric, math.nan))
                    if (
                        not math.isfinite(value)
                        and metric not in OPTIONAL_EARLY_FAILURE_METRICS
                    ):
                        raise ValueError(
                            f"Non-finite {metric} for {(seed, interface, goal, stage)}"
                        )
                    normalized_row[metric] = value
                normalized_stage_rows.append(normalized_row)
    return run_config, normalized, normalized_stage_rows


def aggregate_runs(
    run_roots: list[Path],
    *,
    output_dir: Path,
    minimum_seeds: int = 3,
    expected_seeds: tuple[int, ...] | list[int] = (0, 1, 2),
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    allow_non_paper_runs: bool = False,
) -> dict[str, Any]:
    if minimum_seeds <= 0:
        raise ValueError("--minimum_seeds must be positive.")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing aggregation directory: {output_dir}"
        )
    expected_seed_list = sorted(int(seed) for seed in expected_seeds)
    if len(expected_seed_list) != len(set(expected_seed_list)):
        raise ValueError(f"Expected seeds must be unique: {expected_seed_list}")
    if len(expected_seed_list) < minimum_seeds:
        raise ValueError(
            "Expected seed count cannot be smaller than --minimum_seeds: "
            f"{expected_seed_list} versus {minimum_seeds}"
        )
    loaded = [
        _load_run(path, allow_non_paper_runs=allow_non_paper_runs) for path in run_roots
    ]
    configs = [item[0] for item in loaded]
    rows = [row for item in loaded for row in item[1]]
    stage_rows = [row for item in loaded for row in item[2]]
    seeds = [int(config["seed"]) for config in configs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Training seeds must be unique; received {seeds}")
    if len(seeds) < minimum_seeds:
        raise ValueError(
            f"Need at least {minimum_seeds} seeds; received {len(seeds)} ({seeds})"
        )
    if sorted(seeds) != expected_seed_list:
        raise ValueError(
            f"Training seeds must exactly match {expected_seed_list}; received "
            f"{sorted(seeds)}"
        )

    contracts = [_protocol_contract(config) for config in configs]
    contract_hashes = [_canonical_sha256(contract) for contract in contracts]
    if len(set(contract_hashes)) != 1:
        raise ValueError("Run protocols, source hashes, or input artifacts differ.")
    interface_specs = [
        _json(path.expanduser().resolve() / "summary" / "final_results.json").get(
            "interface_specs"
        )
        for path in run_roots
    ]
    if not isinstance(interface_specs[0], dict) or any(
        specs != interface_specs[0] for specs in interface_specs[1:]
    ):
        raise ValueError("Planner interface dimensions, rates, or sizes differ.")
    push_protocols = [
        _json(path.expanduser().resolve() / "summary" / "final_results.json").get(
            "push_perturbation_protocol"
        )
        for path in run_roots
    ]
    if not isinstance(push_protocols[0], dict) or any(
        protocol != push_protocols[0] for protocol in push_protocols[1:]
    ):
        raise ValueError("Push perturbation protocols differ between seeds.")
    termination_counts_across_seeds: dict[str, Counter[str]] = {
        interface: Counter() for interface in INTERFACES
    }
    for path in run_roots:
        summary = _json(path.expanduser().resolve() / "summary" / "final_results.json")
        raw_counts = summary.get("termination_cause_counts_across_goals", {})
        for interface in INTERFACES:
            counts = raw_counts.get(interface)
            if not isinstance(counts, dict):
                raise ValueError(
                    f"Termination counts are missing for {interface}: {path}"
                )
            termination_counts_across_seeds[interface].update(
                {str(term): int(count) for term, count in counts.items()}
            )
    goals = [str(value) for value in configs[0]["goals"]]
    seeds = sorted(seeds)

    indexed = {
        (int(row["seed"]), str(row["interface"]), str(row["goal"])): row for row in rows
    }
    interface_statistics: dict[str, dict[str, Any]] = {}
    paired_statistics: dict[str, dict[str, Any]] = {}
    paired_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        interface_statistics[metric] = {}
        for interface in INTERFACES:
            values_by_seed = {
                seed: [
                    value
                    for goal in goals
                    if math.isfinite(
                        value := float(indexed[(seed, interface, goal)][metric])
                    )
                ]
                for seed in seeds
            }
            statistics = _hierarchical_statistics(
                values_by_seed,
                samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                label=f"interface:{metric}:{interface}",
            )
            interface_statistics[metric][interface] = {
                **statistics,
            }

        deltas_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
        direction_adjusted: list[float] = []
        for seed in seeds:
            for goal in goals:
                latent = float(indexed[(seed, "latent_skill", goal)][metric])
                explicit = float(indexed[(seed, "full_body_trajectory", goal)][metric])
                if not (math.isfinite(latent) and math.isfinite(explicit)):
                    continue
                delta = latent - explicit
                deltas_by_seed[seed].append(delta)
                adjusted = delta if HIGHER_IS_BETTER[metric] else -delta
                direction_adjusted.append(adjusted)
                paired_rows.append(
                    {
                        "seed": seed,
                        "goal": goal,
                        "metric": metric,
                        "latent_skill": latent,
                        "full_body_trajectory": explicit,
                        "latent_minus_full_body": delta,
                        "higher_is_better": HIGHER_IS_BETTER[metric],
                        "latent_better": adjusted > 0.0,
                    }
                )
        delta_statistics = _hierarchical_statistics(
            deltas_by_seed,
            samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"paired:{metric}",
        )
        paired_statistics[metric] = {
            "difference": "latent_skill - full_body_trajectory",
            "higher_is_better": HIGHER_IS_BETTER[metric],
            "mean_difference": delta_statistics["mean"],
            "std_across_seed_mean_differences": delta_statistics[
                "std_across_seed_means"
            ],
            "hierarchical_bootstrap_95_ci": delta_statistics[
                "hierarchical_bootstrap_95_ci"
            ],
            "seed_mean_differences": delta_statistics["seed_means"],
            "seed_count_with_pairs": delta_statistics["seed_count_with_values"],
            "pair_count": delta_statistics["value_count"],
            "latent_better_pair_fraction": (
                sum(value > 0.0 for value in direction_adjusted)
                / len(direction_adjusted)
                if direction_adjusted
                else None
            ),
            "tie_pair_fraction": (
                sum(value == 0.0 for value in direction_adjusted)
                / len(direction_adjusted)
                if direction_adjusted
                else None
            ),
        }

    stage_indexed = {
        (
            int(row["seed"]),
            str(row["interface"]),
            str(row["goal"]),
            str(row["stage"]),
        ): row
        for row in stage_rows
    }
    before_after_statistics: dict[str, Any] = {}
    for metric in METRICS:
        interface_changes: dict[str, dict[int, dict[str, float]]] = {}
        metric_statistics: dict[str, Any] = {"interfaces": {}}
        for interface in INTERFACES:
            pretrained_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
            finetuned_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
            changes_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
            changes_by_goal: dict[int, dict[str, float]] = {seed: {} for seed in seeds}
            for seed in seeds:
                for goal in goals:
                    pretrained = float(
                        stage_indexed[
                            (seed, interface, goal, "pretrained_demonstration")
                        ][metric]
                    )
                    finetuned = float(
                        stage_indexed[
                            (seed, interface, goal, "finetuned_planner_rollout")
                        ][metric]
                    )
                    if math.isfinite(pretrained):
                        pretrained_by_seed[seed].append(pretrained)
                    if math.isfinite(finetuned):
                        finetuned_by_seed[seed].append(finetuned)
                    if math.isfinite(pretrained) and math.isfinite(finetuned):
                        change = finetuned - pretrained
                        changes_by_seed[seed].append(change)
                        changes_by_goal[seed][goal] = change
            interface_changes[interface] = changes_by_goal
            pretrained_statistics = _hierarchical_statistics(
                pretrained_by_seed,
                samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                label=f"pretrained:{metric}:{interface}",
            )
            finetuned_statistics = _hierarchical_statistics(
                finetuned_by_seed,
                samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                label=f"finetuned:{metric}:{interface}",
            )
            change_statistics = _hierarchical_statistics(
                changes_by_seed,
                samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                label=f"before_after:{metric}:{interface}",
            )
            adjusted_changes = [
                value if HIGHER_IS_BETTER[metric] else -value
                for values in changes_by_seed
                for value in changes_by_seed[values]
            ]
            metric_statistics["interfaces"][interface] = {
                "pretrained_mean": pretrained_statistics["mean"],
                "finetuned_mean": finetuned_statistics["mean"],
                "mean_change_finetuned_minus_pretrained": change_statistics["mean"],
                "std_across_seed_mean_changes": change_statistics[
                    "std_across_seed_means"
                ],
                "change_hierarchical_bootstrap_95_ci": change_statistics[
                    "hierarchical_bootstrap_95_ci"
                ],
                "seed_mean_changes": change_statistics["seed_means"],
                "change_pair_count": change_statistics["value_count"],
                "improved_pair_fraction": (
                    sum(value > 0.0 for value in adjusted_changes)
                    / len(adjusted_changes)
                    if adjusted_changes
                    else None
                ),
            }
        paired_change_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
        for seed in seeds:
            for goal in goals:
                latent_change = interface_changes["latent_skill"][seed].get(goal)
                explicit_change = interface_changes["full_body_trajectory"][seed].get(
                    goal
                )
                if latent_change is not None and explicit_change is not None:
                    paired_change_by_seed[seed].append(latent_change - explicit_change)
        paired_change_statistics = _hierarchical_statistics(
            paired_change_by_seed,
            samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"before_after_paired:{metric}",
        )
        metric_statistics["higher_is_better"] = HIGHER_IS_BETTER[metric]
        metric_statistics["paired_change_difference"] = {
            "definition": "latent change - full-body change",
            "mean": paired_change_statistics["mean"],
            "std_across_seed_means": paired_change_statistics["std_across_seed_means"],
            "hierarchical_bootstrap_95_ci": paired_change_statistics[
                "hierarchical_bootstrap_95_ci"
            ],
            "seed_means": paired_change_statistics["seed_means"],
            "pair_count": paired_change_statistics["value_count"],
        }
        before_after_statistics[metric] = metric_statistics

    low_level_gates = configs[0].get("submission_gates", {}).get("low_level", {})
    oracle_success_rates = {
        "latent_skill": float(low_level_gates.get("latent_success_rate", math.nan)),
        "full_body_trajectory": float(
            low_level_gates.get("vanilla_success_rate", math.nan)
        ),
    }
    if not allow_non_paper_runs and any(
        not math.isfinite(value) or value <= 0.0
        for value in oracle_success_rates.values()
    ):
        raise ValueError("Paper runs lack valid low-level oracle success rates.")
    oracle_normalized: dict[str, Any] | None = None
    if all(
        math.isfinite(value) and value > 0.0 for value in oracle_success_rates.values()
    ):
        normalized_values: dict[str, list[list[float]]] = {}
        normalized_interface_stats: dict[str, Any] = {}
        for interface in INTERFACES:
            oracle_rate = oracle_success_rates[interface]
            values_by_seed = [
                [
                    float(indexed[(seed, interface, goal)]["survival_rate"])
                    / oracle_rate
                    for goal in goals
                ]
                for seed in seeds
            ]
            normalized_values[interface] = values_by_seed
            seed_means = [fmean(values) for values in values_by_seed]
            ci_low, ci_high = _hierarchical_bootstrap_ci(
                values_by_seed,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                label=f"oracle_normalized:{interface}",
            )
            normalized_interface_stats[interface] = {
                "oracle_tracking_success_rate": oracle_rate,
                "mean_planner_over_oracle": fmean(seed_means),
                "std_across_seed_means": stdev(seed_means)
                if len(seed_means) > 1
                else 0.0,
                "hierarchical_bootstrap_95_ci": [ci_low, ci_high],
                "seed_means": dict(zip((str(seed) for seed in seeds), seed_means)),
            }
        normalized_differences = [
            [
                latent - explicit
                for latent, explicit in zip(
                    normalized_values["latent_skill"][seed_index],
                    normalized_values["full_body_trajectory"][seed_index],
                    strict=True,
                )
            ]
            for seed_index in range(len(seeds))
        ]
        normalized_seed_mean_differences = [
            fmean(values) for values in normalized_differences
        ]
        ci_low, ci_high = _hierarchical_bootstrap_ci(
            normalized_differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            label="oracle_normalized:paired",
        )
        oracle_normalized = {
            "definition": "planner tracking success rate / low-level oracle tracking success rate",
            "interfaces": normalized_interface_stats,
            "paired_difference": {
                "difference": "latent_skill ratio - full_body_trajectory ratio",
                "mean_difference": fmean(normalized_seed_mean_differences),
                "std_across_seed_mean_differences": stdev(
                    normalized_seed_mean_differences
                )
                if len(normalized_seed_mean_differences) > 1
                else 0.0,
                "hierarchical_bootstrap_95_ci": [ci_low, ci_high],
                "seed_mean_differences": dict(
                    zip(
                        (str(seed) for seed in seeds),
                        normalized_seed_mean_differences,
                    )
                ),
            },
        }

    output_dir.mkdir(parents=True)
    source_run_artifacts: list[dict[str, Any]] = []
    for run_root, config in zip(run_roots, configs, strict=True):
        resolved_root = run_root.expanduser().resolve()
        artifact_paths = {
            "run_config": resolved_root / "run_config.json",
            "comparison_manifest": resolved_root / "comparison_manifest.json",
            "final_summary": resolved_root / "summary" / "final_results.json",
            "protocol_audit": resolved_root
            / "protocol_checks"
            / "multigoal_language_audit.json",
            "summarize_stage": resolved_root / "stages" / "summarize.json",
            "cluster_submission": resolved_root / "cluster_submission.json",
        }
        source_run_artifacts.append(
            {
                "seed": int(config["seed"]),
                "run_root": str(resolved_root),
                "artifacts": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in artifact_paths.items()
                    if path.is_file()
                },
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "aggregation_source_sha256": _sha256(Path(__file__).resolve()),
        "aggregation_command": " ".join(shlex.quote(value) for value in sys.argv),
        "paper_protocol_complete": all(
            config.get("paper_protocol_complete") is True for config in configs
        ),
        "run_roots": [str(path.expanduser().resolve()) for path in run_roots],
        "source_run_artifacts": sorted(
            source_run_artifacts, key=lambda item: int(item["seed"])
        ),
        "seeds": seeds,
        "expected_seeds": expected_seed_list,
        "seed_count": len(seeds),
        "goals": goals,
        "goal_count": len(goals),
        "interface_specs": interface_specs[0],
        "push_perturbation_protocol": push_protocols[0],
        "termination_cause_counts_across_seeds": {
            interface: dict(sorted(counts.items()))
            for interface, counts in termination_counts_across_seeds.items()
        },
        "protocol_contract_sha256": contract_hashes[0],
        "statistics": {
            "unit": "paired goal within training seed",
            "seed_summary": "mean across goals for each training seed",
            "confidence_interval": "hierarchical bootstrap over seeds, then goals",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "interface_statistics": interface_statistics,
        "paired_statistics": paired_statistics,
        "before_after_statistics": before_after_statistics,
        "oracle_normalized_survival": oracle_normalized,
        "oracle_normalized_tracking_success": oracle_normalized,
    }
    output_json = output_dir / "multiseed_results.json"
    per_goal_csv = output_dir / "multiseed_per_goal.csv"
    paired_csv = output_dir / "multiseed_paired_differences.csv"
    before_after_csv = output_dir / "multiseed_before_after.csv"
    paper_markdown = output_dir / "multiseed_results.md"
    aggregation_manifest = output_dir / "aggregation_manifest.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with per_goal_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["seed"], row["goal"], row["interface"]))
        )
    with paired_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    with before_after_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(
            sorted(
                stage_rows,
                key=lambda row: (
                    row["seed"],
                    row["goal"],
                    row["interface"],
                    row["stage"],
                ),
            )
        )
    _write_paper_markdown(payload, paper_markdown)
    aggregate_outputs = {
        "results_json": output_json,
        "per_goal_csv": per_goal_csv,
        "paired_differences_csv": paired_csv,
        "before_after_csv": before_after_csv,
        "paper_markdown": paper_markdown,
    }
    aggregation_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "aggregation_source_sha256": payload["aggregation_source_sha256"],
                "aggregation_command": payload["aggregation_command"],
                "source_run_artifacts": payload["source_run_artifacts"],
                "outputs": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in aggregate_outputs.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote multi-seed result summary: {output_json}")
    print(f"[INFO] Wrote per-goal measurements: {per_goal_csv}")
    print(f"[INFO] Wrote paired differences: {paired_csv}")
    print(f"[INFO] Wrote before/after measurements: {before_after_csv}")
    print(f"[INFO] Wrote paper table: {paper_markdown}")
    print(f"[INFO] Wrote aggregation manifest: {aggregation_manifest}")
    return payload


def main() -> None:
    args = _parse_args()
    aggregate_runs(
        args.run_roots,
        output_dir=args.output_dir,
        minimum_seeds=args.minimum_seeds,
        expected_seeds=args.expected_seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        allow_non_paper_runs=args.allow_non_paper_runs,
    )


if __name__ == "__main__":
    main()
