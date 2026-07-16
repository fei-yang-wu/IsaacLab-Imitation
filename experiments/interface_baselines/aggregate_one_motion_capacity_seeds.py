#!/usr/bin/env python3
"""Aggregate repeated seeds from one-motion planner-capacity summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


INTERFACES = ("latent_skill", "full_body_trajectory")
CORE_METRICS = (
    "tracking_mpjpe_mm",
    "root_pos_xyz_error_m",
    "joint_pos_rmse_rad",
    "ee_pos_error_m",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_seeds", type=int, default=2)
    parser.add_argument("--survival_target", type=float, default=0.9)
    parser.add_argument("--normalized_mpjpe_target", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"Expected finite values, got {values}")
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "seed_count": len(values),
    }


def _row_identity(row: dict[str, Any]) -> tuple[int, str]:
    config_path = Path(row["artifacts"]["config"])
    config = _load(config_path)
    expected_hash = row["artifacts"].get("config_sha256")
    if expected_hash is not None and _sha256(config_path) != expected_hash:
        raise ValueError(f"Config changed after aggregation: {config_path}")
    seed = int(row.get("planner_seed", config["args"]["seed"]))
    family = str(row.get("planner_family", config["planner_type"]))
    return seed, family


def aggregate(
    *,
    inputs: list[Path],
    min_seeds: int,
    survival_target: float,
    normalized_mpjpe_target: float,
) -> dict[str, Any]:
    if min_seeds < 1:
        raise ValueError("min_seeds must be positive")
    payloads = [_load(path) for path in inputs]
    starts = {tuple(payload["evaluation_starts"]) for payload in payloads}
    if len(starts) != 1:
        raise ValueError(f"Mismatched evaluation starts: {sorted(starts)}")
    oracle_hashes = {
        interface: {payload["oracles"][interface]["sha256"] for payload in payloads}
        for interface in INTERFACES
    }
    if any(len(values) != 1 for values in oracle_hashes.values()):
        raise ValueError(f"Mismatched low-level oracle summaries: {oracle_hashes}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, str, str]] = set()
    point_members: defaultdict[tuple[int, str, str], set[tuple[str, str]]] = (
        defaultdict(set)
    )
    for source_path, payload in zip(inputs, payloads, strict=True):
        for original in payload["rows"]:
            row = dict(original)
            seed, family = _row_identity(row)
            row["planner_seed"] = seed
            row["planner_family"] = family
            row["source_capacity_summary"] = str(source_path.resolve())
            key = (seed, family, row["size"], row["interface"], row["stage"])
            if key in seen:
                raise ValueError(f"Duplicate capacity row: {key}")
            seen.add(key)
            point_members[(seed, family, row["size"])].add(
                (row["interface"], row["stage"])
            )
            rows.append(row)
    expected_members = {
        (interface, stage)
        for interface in INTERFACES
        for stage in ("demonstration_only", "rollout_finetuned")
    }
    incomplete = {
        str(key): sorted(expected_members - members)
        for key, members in point_members.items()
        if members != expected_members
    }
    if incomplete:
        raise ValueError(f"Incomplete seed/size points: {incomplete}")

    grouped: defaultdict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (row["planner_family"], row["size"], row["interface"], row["stage"])
        ].append(row)

    summaries: list[dict[str, Any]] = []
    parameter_counts: dict[tuple[str, str, str], int] = {}
    for (family, size, interface, stage), group in sorted(grouped.items()):
        counts = {int(row["parameter_count"]) for row in group}
        if len(counts) != 1:
            raise ValueError(
                f"Parameter count changed across seeds for {(family, size, interface, stage)}: {counts}"
            )
        count = counts.pop()
        parameter_counts[(family, size, interface)] = count
        target_dims = {int(row["target_dim"]) for row in group}
        if len(target_dims) != 1:
            raise ValueError(
                f"Target width changed across seeds for {(family, size, interface, stage)}: "
                f"{target_dims}"
            )
        target_dim = target_dims.pop()
        summaries.append(
            {
                "planner_family": family,
                "size": size,
                "interface": interface,
                "stage": stage,
                "parameter_count": count,
                "target_dim": target_dim,
                "output_values_per_second": target_dim * 5,
                "seeds": sorted(int(row["planner_seed"]) for row in group),
                "planner_inference_latency_ms": _mean_std(
                    [float(row["planner_inference_latency_ms"]) for row in group]
                ),
                "survival_rate": _mean_std(
                    [float(row["survival_rate"]) for row in group]
                ),
                "metrics": {
                    name: _mean_std([float(row["metrics"][name]) for row in group])
                    for name in CORE_METRICS
                },
                "oracle_normalized_mpjpe": _mean_std(
                    [
                        float(row["oracle_normalized_metrics"]["tracking_mpjpe_mm"])
                        for row in group
                    ]
                ),
            }
        )

    by_point: defaultdict[tuple[int, str, str, str], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in rows:
        by_point[
            (row["planner_seed"], row["planner_family"], row["size"], row["stage"])
        ][row["interface"]] = row
    paired_rows: list[dict[str, Any]] = []
    for (seed, family, size, stage), pair in sorted(by_point.items()):
        latent = pair["latent_skill"]
        explicit = pair["full_body_trajectory"]
        paired_rows.append(
            {
                "planner_seed": seed,
                "planner_family": family,
                "size": size,
                "stage": stage,
                "latent_minus_explicit": {
                    "survival_rate": float(latent["survival_rate"])
                    - float(explicit["survival_rate"]),
                    "tracking_mpjpe_mm": float(latent["metrics"]["tracking_mpjpe_mm"])
                    - float(explicit["metrics"]["tracking_mpjpe_mm"]),
                    "oracle_normalized_mpjpe": float(
                        latent["oracle_normalized_metrics"]["tracking_mpjpe_mm"]
                    )
                    - float(explicit["oracle_normalized_metrics"]["tracking_mpjpe_mm"]),
                },
            }
        )

    minimums: list[dict[str, Any]] = []
    families = sorted({row["planner_family"] for row in rows})
    for family in families:
        for interface in INTERFACES:
            for stage in ("demonstration_only", "rollout_finetuned"):
                candidates = [
                    summary
                    for summary in summaries
                    if summary["planner_family"] == family
                    and summary["interface"] == interface
                    and summary["stage"] == stage
                    and summary["survival_rate"]["seed_count"] >= min_seeds
                    and summary["survival_rate"]["mean"] >= survival_target
                    and summary["oracle_normalized_mpjpe"]["mean"]
                    <= normalized_mpjpe_target
                ]
                winner = min(
                    candidates, key=lambda item: item["parameter_count"], default=None
                )
                minimums.append(
                    {
                        "planner_family": family,
                        "interface": interface,
                        "stage": stage,
                        "minimum_tested_parameter_count": None
                        if winner is None
                        else winner["parameter_count"],
                        "size": None if winner is None else winner["size"],
                    }
                )

    return {
        "study": "one_motion_planner_capacity_repeated_seeds",
        "paper_result": False,
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in inputs
        ],
        "evaluation_starts": list(next(iter(starts))),
        "fixed_performance_target": {
            "minimum_seed_count": min_seeds,
            "mean_survival_rate_at_least": survival_target,
            "mean_oracle_normalized_mpjpe_at_most": normalized_mpjpe_target,
        },
        "summaries": summaries,
        "paired_seed_rows": paired_rows,
        "minimum_tested_sizes": minimums,
        "interpretation_limits": [
            "This local study uses one motion and is not a paper result.",
            "Minimums are observed tested sizes; no interpolation is used.",
            "Parameter efficiency and inference latency are reported separately.",
            "A paper scaling curve requires at least three planner seeds and multiple motions.",
        ],
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Repeated-seed planner capacity diagnostic",
        "",
        "This is a local one-motion diagnostic, not a paper result.",
        "",
        "| Family | Size | Interface | Stage | Seeds | Parameters | Values/s | Latency mean ± std (ms) | Survival mean ± std | MPJPE mean ± std (mm) | MPJPE / oracle mean ± std |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["summaries"]:
        survival = item["survival_rate"]
        mpjpe = item["metrics"]["tracking_mpjpe_mm"]
        normalized = item["oracle_normalized_mpjpe"]
        latency = item["planner_inference_latency_ms"]
        lines.append(
            f"| {item['planner_family']} | {item['size']} | {item['interface']} | {item['stage']} | "
            f"{survival['seed_count']} | {item['parameter_count']:,} | "
            f"{item['output_values_per_second']:,} | "
            f"{_fmt(latency['mean'])} ± {_fmt(latency['std'])} | "
            f"{_fmt(survival['mean'])} ± {_fmt(survival['std'])} | "
            f"{_fmt(mpjpe['mean'])} ± {_fmt(mpjpe['std'])} | "
            f"{_fmt(normalized['mean'])} ± {_fmt(normalized['std'])} |"
        )
    lines.extend(["", "## Minimum tested size at the fixed target", ""])
    target = payload["fixed_performance_target"]
    lines.append(
        f"Target: mean survival ≥ {target['mean_survival_rate_at_least']}, "
        f"mean MPJPE / oracle ≤ {target['mean_oracle_normalized_mpjpe_at_most']}, "
        f"using at least {target['minimum_seed_count']} seeds."
    )
    lines.append("")
    for item in payload["minimum_tested_sizes"]:
        result = (
            "not reached"
            if item["minimum_tested_parameter_count"] is None
            else f"{item['size']} ({item['minimum_tested_parameter_count']:,} parameters)"
        )
        lines.append(
            f"- {item['planner_family']}, {item['interface']}, {item['stage']}: {result}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = aggregate(
        inputs=args.input,
        min_seeds=args.min_seeds,
        survival_target=args.survival_target,
        normalized_mpjpe_target=args.normalized_mpjpe_target,
    )
    json_path = args.output_dir / "capacity_multiseed_results.json"
    markdown_path = args.output_dir / "capacity_multiseed_results.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(payload))
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {markdown_path}")


if __name__ == "__main__":
    main()
