#!/usr/bin/env python3
"""Aggregate the matched one-motion planner-capacity diagnostic.

This is intentionally a diagnostic aggregator, not a paper-result aggregator.  It
requires the two interfaces to use identical starts and training budgets, retains
both training stages, and normalizes physical errors by each interface's own
low-level oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


INTERFACES = ("latent_skill", "full_body_trajectory")
STAGES = {
    "demonstration_only": "eval_pretrained_10starts/summary.json",
    "rollout_finetuned": "eval_finetuned_10starts/summary.json",
}
METRICS = (
    "tracking_mpjpe_mm",
    "root_pos_xyz_error_m",
    "joint_pos_rmse_rad",
    "ee_pos_error_m",
    "action_delta_l2",
    "tracking_velocity_distance_mps",
    "tracking_acceleration_distance_mps2",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling_root", type=Path, required=True)
    parser.add_argument("--latent_oracle", type=Path, required=True)
    parser.add_argument("--explicit_oracle", type=Path, required=True)
    parser.add_argument(
        "--sizes", nargs="+", default=["tiny", "small", "medium", "large"]
    )
    parser.add_argument(
        "--size_root",
        action="append",
        default=[],
        metavar="SIZE=PATH",
        help="Override the artifact root for one size without copying checkpoints.",
    )
    parser.add_argument(
        "--interface_root",
        action="append",
        default=[],
        metavar="SIZE:INTERFACE=PATH",
        help="Override one size/interface artifact root.",
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def _metric(summary: dict[str, Any], name: str) -> float:
    value = summary.get("metrics", {}).get(name, {}).get("mean")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Missing finite metric {name!r}")
    return float(value)


def _survival(summary: dict[str, Any]) -> float:
    aggregate = summary.get("aggregate", {})
    for key in ("survival_rate", "fall_free_rate", "horizon_completion_rate"):
        value = aggregate.get(key)
        if value is not None:
            return float(value)
    raise ValueError("Summary does not contain a survival measure")


def _starts(summary: dict[str, Any]) -> tuple[int, ...]:
    values = summary.get("start_trajectories", {}).get("local_steps")
    if not isinstance(values, list) or not values:
        raise ValueError(
            "Summary does not contain non-empty start_trajectories.local_steps"
        )
    return tuple(int(value) for value in values)


def _latency(summary: dict[str, Any]) -> float | None:
    value = summary.get("planner_inference_latency_ms", {}).get("mean")
    return None if value is None else float(value)


def _assert_equal(label: str, values: list[Any]) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"Mismatched {label}: {values}")
    return first


def _oracle_record(path: Path) -> dict[str, Any]:
    summary = _load(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "survival_rate": _survival(summary),
        "metrics": {name: _metric(summary, name) for name in METRICS},
    }


def _row(
    *,
    interface_root: Path,
    size: str,
    interface: str,
    stage: str,
    oracle: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    summary_path = interface_root / STAGES[stage]
    config_path = (
        interface_root
        / ("planner_pretrain" if stage == "demonstration_only" else "planner_finetune")
        / "config.json"
    )
    checkpoint_path = config_path.parent / "checkpoints/latest.pt"
    summary = _load(summary_path)
    config = _load(config_path)
    metrics = {name: _metric(summary, name) for name in METRICS}
    normalized = {
        name: metrics[name] / float(oracle["metrics"][name]) for name in METRICS
    }
    row = {
        "size": size,
        "interface": interface,
        "stage": stage,
        "planner_family": str(config["planner_type"]),
        "planner_seed": int(config["args"]["seed"]),
        "parameter_count": int(config["parameter_count"]),
        "target_dim": int(config["target_dim"]),
        "survival_rate": _survival(summary),
        "threshold_tracking_success_rate": float(
            summary.get("aggregate", {}).get(
                "threshold_tracking_success_rate", float("nan")
            )
        ),
        "survival_steps_mean": float(
            summary.get("aggregate", {}).get("survival_steps_mean", float("nan"))
        ),
        "planner_inference_latency_ms": _latency(summary),
        "metrics": metrics,
        "oracle_normalized_metrics": normalized,
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
    }
    contract = {
        "seed": int(config["args"]["seed"]),
        "state_key": str(config["state_key"]),
        "source_sample_count": int(config["source_sample_count"]),
        "selected_sample_count": int(config["selected_sample_count"]),
        "num_updates": int(config["num_updates"]),
        "batch_size": int(config["batch_size"]),
        "micro_batch_size": int(config["micro_batch_size"]),
        "flow_num_inference_steps": int(config["flow_num_inference_steps"]),
        "flow_inference_noise_std": float(config["flow_inference_noise_std"]),
        "motion_name": str(config["sample_metadata"]["motion_name"]),
    }
    if stage == "rollout_finetuned":
        merge_path = (
            interface_root / "demonstration_and_rollout_samples/merge_manifest.json"
        )
        merge = _load(merge_path)
        row["artifacts"]["merge_manifest"] = str(merge_path.resolve())
        row["artifacts"]["merge_manifest_sha256"] = _sha256(merge_path)
        contract["merge_total_rows"] = int(merge["row_count"])
        contract["merge_source_rows"] = tuple(
            int(item["selected_row_count"]) for item in merge["sources"]
        )
    return row, _starts(summary), contract


def aggregate(
    *,
    scaling_root: Path,
    latent_oracle_path: Path,
    explicit_oracle_path: Path,
    sizes: list[str],
    size_roots: dict[str, Path] | None = None,
    interface_roots: dict[tuple[str, str], Path] | None = None,
) -> dict[str, Any]:
    scaling_root = scaling_root.resolve()
    resolved_size_roots = {
        size: (size_roots or {}).get(size, scaling_root / size).resolve()
        for size in sizes
    }
    resolved_interface_roots = {
        (size, interface): (interface_roots or {})
        .get((size, interface), resolved_size_roots[size] / interface)
        .resolve()
        for size in sizes
        for interface in INTERFACES
    }
    oracles = {
        "latent_skill": _oracle_record(latent_oracle_path),
        "full_body_trajectory": _oracle_record(explicit_oracle_path),
    }
    rows: list[dict[str, Any]] = []
    starts: list[tuple[int, ...]] = []
    contracts_by_stage: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in STAGES
    }
    for size in sizes:
        for interface in INTERFACES:
            for stage in STAGES:
                row, row_starts, contract = _row(
                    interface_root=resolved_interface_roots[(size, interface)],
                    size=size,
                    interface=interface,
                    stage=stage,
                    oracle=oracles[interface],
                )
                rows.append(row)
                starts.append(row_starts)
                contracts_by_stage[stage].append(contract)
    common_starts = _assert_equal("evaluation starts", starts)
    for stage, contracts in contracts_by_stage.items():
        for key in contracts[0]:
            _assert_equal(
                f"{stage} contract field {key}",
                [contract[key] for contract in contracts],
            )
    empirical_minimums: list[dict[str, Any]] = []
    for stage in STAGES:
        for interface in INTERFACES:
            candidates = [
                row
                for row in rows
                if row["stage"] == stage
                and row["interface"] == interface
                and row["survival_rate"] >= 1.0
                and row["metrics"]["tracking_mpjpe_mm"] <= 75.0
            ]
            empirical_minimums.append(
                {
                    "stage": stage,
                    "interface": interface,
                    "criterion": "survival_rate == 1 and MPJPE <= 75 mm",
                    "minimum_tested_parameter_count": min(
                        (row["parameter_count"] for row in candidates), default=None
                    ),
                }
            )
    return {
        "study": "one_motion_planner_capacity_diagnostic",
        "paper_result": False,
        "scaling_root": str(scaling_root),
        "sizes": sizes,
        "size_roots": {size: str(root) for size, root in resolved_size_roots.items()},
        "interface_roots": {
            f"{size}:{interface}": str(root)
            for (size, interface), root in resolved_interface_roots.items()
        },
        "interfaces": list(INTERFACES),
        "stages": list(STAGES),
        "evaluation_starts": list(common_starts),
        "oracles": oracles,
        "rows": rows,
        "empirical_minimums": empirical_minimums,
        "interpretation_limits": [
            "One motion and one seed do not support a paper-level capacity claim.",
            "The tested curve may be non-monotonic; minimums are observed tested points, not interpolated estimates.",
            "Each interface is normalized by its own frozen low-level oracle.",
        ],
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# One-motion planner capacity diagnostic",
        "",
        "This is a local, one-motion, one-seed diagnostic. It is not a paper result.",
        "",
        "| Size | Interface | Stage | Parameters | Survival | MPJPE (mm) | MPJPE / oracle | Root XYZ (m) | Joint RMSE | Latency (ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {size} | {interface} | {stage} | {params:,} | {survival} | {mpjpe} | {norm} | {root} | {joint} | {latency} |".format(
                size=row["size"],
                interface=row["interface"],
                stage=row["stage"],
                params=row["parameter_count"],
                survival=_fmt(row["survival_rate"]),
                mpjpe=_fmt(row["metrics"]["tracking_mpjpe_mm"]),
                norm=_fmt(row["oracle_normalized_metrics"]["tracking_mpjpe_mm"]),
                root=_fmt(row["metrics"]["root_pos_xyz_error_m"]),
                joint=_fmt(row["metrics"]["joint_pos_rmse_rad"]),
                latency=_fmt(row["planner_inference_latency_ms"]),
            )
        )
    lines.extend(["", "## Empirical minimum tested size", ""])
    for item in payload["empirical_minimums"]:
        count = item["minimum_tested_parameter_count"]
        rendered = "not reached" if count is None else f"{count:,} parameters"
        lines.append(f"- {item['interface']}, {item['stage']}: {rendered}")
    lines.extend(
        [
            "",
            "The minimum criterion is 100% fall-free survival and MPJPE at most 75 mm.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    size_roots: dict[str, Path] = {}
    for item in args.size_root:
        if "=" not in item:
            raise ValueError(f"Expected --size_root SIZE=PATH, got {item!r}")
        size, raw_path = item.split("=", 1)
        if size in size_roots:
            raise ValueError(f"Duplicate --size_root for {size!r}")
        size_roots[size] = Path(raw_path)
    unknown_sizes = sorted(set(size_roots) - set(args.sizes))
    if unknown_sizes:
        raise ValueError(f"--size_root supplied for unrequested sizes: {unknown_sizes}")
    interface_roots: dict[tuple[str, str], Path] = {}
    for item in args.interface_root:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Expected --interface_root SIZE:INTERFACE=PATH, got {item!r}"
            )
        key, raw_path = item.split("=", 1)
        size, interface = key.split(":", 1)
        pair = (size, interface)
        if pair in interface_roots:
            raise ValueError(f"Duplicate --interface_root for {key!r}")
        if size not in args.sizes or interface not in INTERFACES:
            raise ValueError(f"Unknown --interface_root key {key!r}")
        interface_roots[pair] = Path(raw_path)
    output_dir = (args.output_dir or args.scaling_root / "capacity_summary").resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = aggregate(
        scaling_root=args.scaling_root,
        latent_oracle_path=args.latent_oracle,
        explicit_oracle_path=args.explicit_oracle,
        sizes=[str(size) for size in args.sizes],
        size_roots=size_roots,
        interface_roots=interface_roots,
    )
    json_path = output_dir / "capacity_results.json"
    markdown_path = output_dir / "capacity_results.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {markdown_path}")


if __name__ == "__main__":
    main()
