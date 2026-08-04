#!/usr/bin/env python3
"""Aggregate the paired Phase-4 no-language sample-efficiency study."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
from statistics import fmean, stdev
import sys
from typing import Any


INTERFACES = ("latent_skill", "full_body_trajectory")
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
METRICS = AGGREGATE_METRICS + ROLLOUT_METRICS + ("planner_latency_ms",)
HIGHER_IS_BETTER = {
    metric: metric
    in {"return_sum_mean", "survival_steps_mean", "tracking_success_rate"}
    for metric in METRICS
}
PAPER_TABLE_METRICS = (
    ("tracking_success_rate", "Tracking success"),
    ("oracle_normalized_tracking_success", "Tracking success / oracle"),
    ("tracking_mpjpe_mm", "Root-relative MPJPE (mm)"),
    ("planner_latency_ms", "Planner latency (ms)"),
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(summary: dict[str, Any], name: str) -> float:
    if name == "planner_latency_ms":
        value = summary.get("planner_inference_latency_ms", {}).get("mean", math.nan)
    elif name in ROLLOUT_METRICS:
        value = summary.get("metrics", {}).get(name, {}).get("mean", math.nan)
    else:
        value = summary.get("aggregate", {}).get(name, math.nan)
    return float(value) if value is not None else math.nan


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_paired_difference(
    paired: dict[int, dict[str, float]],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    seeds = sorted(paired)
    draws: list[float] = []
    for _ in range(samples):
        selected_seeds = [rng.choice(seeds) for _ in seeds]
        seed_values: list[float] = []
        for seed in selected_seeds:
            motion_values = list(paired[seed].values())
            selected_motions = [rng.choice(motion_values) for _ in motion_values]
            seed_values.append(fmean(selected_motions))
        draws.append(fmean(seed_values))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _parse_ints(values: list[str], *, label: str) -> list[int]:
    parsed = [int(value) for value in values]
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must be a nonempty unique integer list.")
    return parsed


def _validate_cluster_submission(
    run_root: Path,
    *,
    expected_seeds: list[int],
    expected_budgets: list[int],
    expected_motion_count: int,
) -> dict[str, Any]:
    path = run_root / "cluster_submission.json"
    if not path.is_file():
        raise ValueError(f"Phase-4 cluster submission record is missing: {path}")
    record = _json(path)
    recorded_output = Path(str(record.get("output_root", ""))).expanduser()
    if "logs" in recorded_output.parts:
        logs_index = recorded_output.parts.index("logs")
        recorded_suffix = recorded_output.parts[logs_index:]
        output_matches = run_root.parts[-len(recorded_suffix) :] == recorded_suffix
    else:
        output_matches = recorded_output.resolve() == run_root
    expected_array_end = len(expected_seeds) * expected_motion_count - 1
    if not (
        record.get("schema_version") == 1
        and record.get("study") == "phase4_no_language_sample_efficiency_v1"
        and int(record.get("job_id", -1)) > 0
        and int(record.get("motion_count", -1)) == expected_motion_count
        and str(record.get("seeds", ""))
        == " ".join(str(seed) for seed in expected_seeds)
        and str(record.get("sample_budgets", ""))
        == " ".join(str(budget) for budget in expected_budgets)
        and re.fullmatch(
            rf"0-{expected_array_end}(%[1-9][0-9]*)?",
            str(record.get("array", "")),
        )
        is not None
        and bool(str(record.get("cluster_workspace", "")).strip())
        and bool(str(record.get("submitted_at_utc", "")).strip())
        and output_matches
    ):
        raise ValueError("Phase-4 cluster submission record does not match the grid.")
    for key in ("workspace_archive_sha256", "repo_sync_manifest_sha256"):
        value = str(record.get(key, ""))
        if not (
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Phase-4 cluster submission {key} is invalid.")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "record": record,
    }


def aggregate(
    run_root: Path,
    *,
    expected_seeds: list[int],
    expected_budgets: list[int],
    expected_motion_count: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if expected_budgets != sorted(expected_budgets):
        raise ValueError("expected_budgets must be in increasing order.")
    task_configs = sorted(run_root.glob("seed_*/*/task_config.json"))
    expected_task_count = len(expected_seeds) * expected_motion_count
    if len(task_configs) != expected_task_count:
        raise ValueError(
            f"Expected {expected_task_count} completed seed/motion tasks, "
            f"found {len(task_configs)}."
        )
    cluster_submission = _validate_cluster_submission(
        run_root,
        expected_seeds=expected_seeds,
        expected_budgets=expected_budgets,
        expected_motion_count=expected_motion_count,
    )

    rows: list[dict[str, Any]] = []
    audit_hashes: dict[str, str] = {}
    gate_identity: dict[str, Any] | None = None
    motions_by_seed: dict[int, set[str]] = defaultdict(set)
    interface_specs: dict[str, dict[str, Any]] = {}
    oracle_success: dict[tuple[int, str, str], float] = {}
    oracle_measurements: dict[tuple[int, str, str], dict[str, Any]] = {}
    direct_success: dict[tuple[int, str], float] = {}
    direct_rows: list[dict[str, Any]] = []
    for task_config_path in task_configs:
        task_root = task_config_path.parent
        config = _json(task_config_path)
        seed = int(config["seed"])
        motion = str(config["motion_name"])
        if seed not in expected_seeds:
            raise ValueError(f"Unexpected seed in {task_config_path}: {seed}")
        if [int(value) for value in config["sample_budgets"]] != expected_budgets:
            raise ValueError(f"Sample budgets changed in {task_config_path}.")
        motions_by_seed[seed].add(motion)
        gate = _json(task_root / "input" / "submission_gate.json")
        if gate.get("passed") is not True:
            raise ValueError(f"Submission gate did not pass: {task_root}")
        if gate_identity is None:
            gate_identity = gate
        elif gate != gate_identity:
            raise ValueError("Qualified data or low-level artifacts differ across tasks.")

        for budget in expected_budgets:
            audit_path = (
                task_root
                / "protocol_checks"
                / f"focused_protocol_audit_{budget}.json"
            )
            audit = _json(audit_path)
            if audit.get("passed") is not True:
                raise ValueError(f"Focused protocol audit failed: {audit_path}")
            expected = audit.get("expected", {})
            if int(expected.get("seed", -1)) != seed:
                raise ValueError(f"Audit seed mismatch: {audit_path}")
            if int(expected.get("rows_per_stage", -1)) != budget:
                raise ValueError(f"Audit row budget mismatch: {audit_path}")
            audit_hashes[str(audit_path.relative_to(run_root))] = _sha256(audit_path)

            for interface in INTERFACES:
                planner = audit["planner_rows"][interface]
                spec = {
                    "target_dim": int(planner["target_spec"]["target_dim"]),
                    "parameter_count": int(planner["parameter_count"]),
                    "planner_rate_hz": 5.0,
                }
                spec["values_per_second"] = (
                    float(spec["target_dim"]) * float(spec["planner_rate_hz"])
                )
                spec["float32_bits_per_second"] = (
                    float(spec["values_per_second"]) * 32.0
                )
                if interface in interface_specs and interface_specs[interface] != spec:
                    raise ValueError(f"{interface} planner specification changed.")
                interface_specs[interface] = spec
                oracle = audit["oracle_rows"][interface]
                oracle_success[(seed, motion, interface)] = float(
                    oracle["aggregate"]["tracking_success_rate"]
                )
                oracle_key = (seed, motion, interface)
                if oracle_key not in oracle_measurements:
                    oracle_summary = {
                        "aggregate": oracle.get("aggregate", {}),
                        "metrics": oracle.get("metrics", {}),
                    }
                    oracle_measurements[oracle_key] = {
                        "seed": seed,
                        "motion": motion,
                        "interface": interface,
                        **{
                            metric: _metric(oracle_summary, metric)
                            for metric in AGGREGATE_METRICS + ROLLOUT_METRICS
                        },
                    }
                for stage, key in (
                    ("pretrained_demonstration", "pretrained_summary"),
                    ("finetuned_planner_rollout", "summary"),
                ):
                    summary = _json(Path(planner[key]).expanduser().resolve())
                    row: dict[str, Any] = {
                        "seed": seed,
                        "motion": motion,
                        "sample_budget": budget,
                        "interface": interface,
                        "stage": stage,
                    }
                    row.update({metric: _metric(summary, metric) for metric in METRICS})
                    oracle_rate = oracle_success[(seed, motion, interface)]
                    row["oracle_tracking_success_rate"] = oracle_rate
                    row["oracle_normalized_tracking_success"] = (
                        row["tracking_success_rate"] / oracle_rate
                        if oracle_rate > 0.0
                        else math.nan
                    )
                    rows.append(row)
            ceiling = audit["ceiling"]
            direct_summary = _json(Path(ceiling["summary"]).expanduser().resolve())
            direct_key = (seed, motion)
            if direct_key not in direct_success:
                direct_success[direct_key] = float(
                    direct_summary["aggregate"]["tracking_success_rate"]
                )
                direct_rows.append(
                    {
                        "seed": seed,
                        "motion": motion,
                        "interface": "direct_vanilla_50hz_ceiling",
                        "target_dim": 67,
                        "command_rate_hz": 50.0,
                        "values_per_second": 3350.0,
                        "float32_bits_per_second": 107200.0,
                        **{
                            metric: _metric(direct_summary, metric)
                            for metric in AGGREGATE_METRICS + ROLLOUT_METRICS
                        },
                    }
                )

    expected_motion_set: set[str] | None = None
    for seed in expected_seeds:
        motions = motions_by_seed[seed]
        if len(motions) != expected_motion_count:
            raise ValueError(f"Seed {seed} has {len(motions)} motions, expected all.")
        if expected_motion_set is None:
            expected_motion_set = motions
        elif motions != expected_motion_set:
            raise ValueError("Motion set differs across planner seeds.")

    paired_statistics: dict[str, dict[str, Any]] = {}
    rng = random.Random(0)
    final_rows = [row for row in rows if row["stage"] == "finetuned_planner_rollout"]
    for budget in expected_budgets:
        for metric in METRICS + ("oracle_normalized_tracking_success",):
            values: dict[int, dict[str, dict[str, float]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            for row in final_rows:
                if int(row["sample_budget"]) != budget:
                    continue
                values[int(row["seed"])][str(row["motion"])][str(row["interface"])] = float(
                    row[metric]
                )
            paired: dict[int, dict[str, float]] = defaultdict(dict)
            for seed, by_motion in values.items():
                for motion, by_interface in by_motion.items():
                    if set(by_interface) != set(INTERFACES):
                        raise ValueError(f"Incomplete pair: seed={seed} motion={motion}")
                    if not all(_finite(value) for value in by_interface.values()):
                        continue
                    paired[seed][motion] = (
                        by_interface["latent_skill"]
                        - by_interface["full_body_trajectory"]
                    )
            seed_means = {
                seed: fmean(by_motion.values())
                for seed, by_motion in paired.items()
                if by_motion
            }
            if seed_means:
                ci_low, ci_high = _bootstrap_paired_difference(
                    {seed: paired[seed] for seed in seed_means},
                    samples=bootstrap_samples,
                    rng=rng,
                )
                mean_difference: float | None = fmean(seed_means.values())
                std_difference: float | None = (
                    stdev(seed_means.values()) if len(seed_means) > 1 else 0.0
                )
                interval: list[float] | None = [ci_low, ci_high]
            else:
                mean_difference = None
                std_difference = None
                interval = None
            key = f"budget_{budget}/{metric}"
            paired_statistics[key] = {
                "sample_budget": budget,
                "metric": metric,
                "higher_is_better": HIGHER_IS_BETTER.get(metric, True),
                "paired_motion_count": sum(len(value) for value in paired.values()),
                "seed_count_with_finite_pairs": len(seed_means),
                "mean_latent_minus_explicit": mean_difference,
                "std_across_seed_means": std_difference,
                "seed_mean_differences": {
                    str(seed): value for seed, value in sorted(seed_means.items())
                },
                "hierarchical_bootstrap_95ci": interval,
            }

    return {
        "protocol": "phase4_no_language_sample_efficiency_v1",
        "paper_protocol_complete": True,
        "aggregation_source_sha256": _sha256(Path(__file__).resolve()),
        "cluster_submission": cluster_submission,
        "seeds": expected_seeds,
        "sample_budgets": expected_budgets,
        "motion_count": expected_motion_count,
        "motions": sorted(expected_motion_set or []),
        "interface_specs": interface_specs,
        "qualified_artifacts": gate_identity,
        "per_motion_rows": rows,
        "oracle_rows": list(oracle_measurements.values()),
        "direct_ceiling_rows": direct_rows,
        "paired_statistics": paired_statistics,
        "direct_ceiling_tracking_success": {
            f"seed_{seed}/{motion}": value
            for (seed, motion), value in sorted(direct_success.items())
        },
        "audit_sha256": audit_hashes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--expected_seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument(
        "--expected_budgets", nargs="+", default=["1000", "10000", "50000"]
    )
    parser.add_argument("--expected_motion_count", type=int, default=40)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    return parser.parse_args()


def _format_difference(statistic: dict[str, Any]) -> str:
    mean = statistic.get("mean_latent_minus_explicit")
    interval = statistic.get("hierarchical_bootstrap_95ci")
    if mean is None:
        return "N/A"
    if not isinstance(interval, list) or len(interval) != 2:
        return f"{float(mean):.4g}"
    return (
        f"{float(mean):.4g} "
        f"[{float(interval[0]):.4g}, {float(interval[1]):.4g}]"
    )


def _write_paper_markdown(result: dict[str, Any], output: Path) -> None:
    lines = [
        "# Phase-4 no-language sample-efficiency results",
        "",
        f"Seeds: {', '.join(str(seed) for seed in result['seeds'])}. "
        f"Motions per seed: {result['motion_count']}.",
        "",
        "Differences are paired within motion and seed and are reported as "
        "latent minus explicit with hierarchical 95% bootstrap intervals.",
        "",
        "| Planner rows | Metric | Difference | Better direction |",
        "| ---: | --- | ---: | --- |",
    ]
    statistics = result["paired_statistics"]
    for budget in result["sample_budgets"]:
        for metric, label in PAPER_TABLE_METRICS:
            statistic = statistics[f"budget_{budget}/{metric}"]
            lines.append(
                f"| {budget} | {label} | {_format_difference(statistic)} | "
                f"{'higher' if statistic['higher_is_better'] else 'lower'} |"
            )
    lines.extend(
        [
            "",
            "The direct 50 Hz vanilla tracker is a low-level ceiling and is not a "
            "planner row. Use this table only with the passing task audits and "
            "cluster submission record.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_aggregate_outputs(
    result: dict[str, Any],
    *,
    output_dir: Path,
    command: str,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Phase-4 aggregate: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    output_json = output_dir / "phase4_no_language_results.json"
    output_csv = output_dir / "phase4_no_language_per_motion.csv"
    paper_markdown = output_dir / "phase4_no_language_results.md"
    aggregation_manifest = output_dir / "aggregation_manifest.json"
    output_payload = {**result, "aggregation_command": command}
    output_json.write_text(
        json.dumps(_json_safe(output_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = result["per_motion_rows"]
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_paper_markdown(result, paper_markdown)
    outputs = {
        "results_json": output_json,
        "per_motion_csv": output_csv,
        "paper_markdown": paper_markdown,
    }
    aggregation_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": result["protocol"],
                "aggregation_source_sha256": result[
                    "aggregation_source_sha256"
                ],
                "aggregation_command": command,
                "cluster_submission": result["cluster_submission"],
                "outputs": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in outputs.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**outputs, "aggregation_manifest": aggregation_manifest}


def main() -> None:
    args = _parse_args()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.run_root.expanduser().resolve() / "aggregate"
    )
    result = aggregate(
        args.run_root,
        expected_seeds=_parse_ints(list(args.expected_seeds), label="seeds"),
        expected_budgets=_parse_ints(list(args.expected_budgets), label="budgets"),
        expected_motion_count=int(args.expected_motion_count),
        bootstrap_samples=int(args.bootstrap_samples),
    )
    outputs = write_aggregate_outputs(
        result,
        output_dir=output_dir,
        command=" ".join(shlex.quote(value) for value in sys.argv),
    )
    print(f"[PASS] Phase-4 aggregate: {outputs['results_json']}")


if __name__ == "__main__":
    main()
