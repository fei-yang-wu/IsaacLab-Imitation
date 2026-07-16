#!/usr/bin/env python3
"""Summarize LAFAN1 motion-tracking results across interfaces.

Walks the per-motion result roots produced by
``run_lafan1_from_scratch_comparison.sh`` (or any
``run_lafan1_motion_tracking_evaluation.sh`` runs) and builds one table with:

- oracle rows for every interface (latent hl_skill, EE chunk, full-body chunk),
  so low-level competence is visible before any planner row is interpreted;
- an oracle-competence gate: planner rows for an interface only count as
  paper-facing on motions where that interface's oracle passes the success
  threshold;
- the SONIC-style tracking metrics used by PR#19.

Latent success uses ``aggregate.tracking_success_rate``. The EE/full-body
oracle harness (``evaluate_checkpoint.py``) does not emit that metric, so those
rows fall back to ``1 - done_rate``; the ``success_source`` column records
which definition each row used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

METRIC_COLUMNS = [
    "tracking_mpjpe_mm",
    "root_height_error_m",
    "root_ori_error_rad",
    "ee_pos_error_m",
    "joint_pos_rmse_rad",
    "joint_vel_rmse_radps",
    "tracking_velocity_distance_mps",
    "tracking_acceleration_distance_mps2",
]

# method key -> (interface, setting, glob patterns relative to a rank_* dir)
METHOD_SPECS: dict[str, dict[str, Any]] = {
    "latent_oracle": {
        "interface": "latent_skill",
        "setting": "oracle",
        "patterns": ["oracle_ll_eval/summary.json"],
    },
    "latent_planner_base": {
        "interface": "latent_skill",
        "setting": "planner_base",
        "patterns": ["planner_ll_base/summary.json"],
    },
    "latent_planner_finetuned": {
        "interface": "latent_skill",
        "setting": "planner_finetuned",
        "patterns": ["planner_ll_finetuned/summary.json"],
    },
    "ee_oracle": {
        "interface": "ee_trajectory",
        "setting": "oracle",
        "patterns": [
            "hand_designed_chunk_baselines/ee_trajectory/oracle_low_level/summary.json"
        ],
    },
    "ee_planner_base": {
        "interface": "ee_trajectory",
        "setting": "planner_base",
        "patterns": [
            "hand_designed_chunk_baselines/ee_trajectory/chunked_transformer_*/eval_pretrained_closed_loop/summary.json"
        ],
    },
    "ee_planner_finetuned": {
        "interface": "ee_trajectory",
        "setting": "planner_finetuned",
        "patterns": [
            "hand_designed_chunk_baselines/ee_trajectory/chunked_transformer_*/eval_finetuned_closed_loop/summary.json"
        ],
    },
    "full_body_oracle": {
        "interface": "full_body_trajectory",
        "setting": "oracle",
        "patterns": [
            "hand_designed_chunk_baselines/full_body_trajectory/oracle_low_level/summary.json"
        ],
    },
    "full_body_planner_base": {
        "interface": "full_body_trajectory",
        "setting": "planner_base",
        "patterns": [
            "hand_designed_chunk_baselines/full_body_trajectory/chunked_transformer_*/eval_pretrained_closed_loop/summary.json"
        ],
    },
    "full_body_planner_finetuned": {
        "interface": "full_body_trajectory",
        "setting": "planner_finetuned",
        "patterns": [
            "hand_designed_chunk_baselines/full_body_trajectory/chunked_transformer_*/eval_finetuned_closed_loop/summary.json"
        ],
    },
}


def _load_summary(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"[WARN] Skipping unreadable summary {path}: {error}")
        return None


def _success_rate(summary: dict[str, Any]) -> tuple[float | None, str]:
    aggregate = summary.get("aggregate", {})
    value = aggregate.get("tracking_success_rate")
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value), "tracking_success_rate"
    done_rate = aggregate.get("done_rate")
    if isinstance(done_rate, (int, float)) and not math.isnan(float(done_rate)):
        return 1.0 - float(done_rate), "1-done_rate"
    return None, "missing"


def _metric_mean(summary: dict[str, Any], name: str) -> float | None:
    entry = summary.get("metrics", {}).get(name)
    if isinstance(entry, dict):
        value = entry.get("mean")
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return float(value)
    value = summary.get("metric_means", {}).get(name)
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def _survival_steps(summary: dict[str, Any]) -> float | None:
    value = summary.get("aggregate", {}).get("survival_steps_mean")
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def collect_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rank_dirs = sorted(run_root.glob("*/per_trajectory/rank_*"))
    if not rank_dirs:
        rank_dirs = sorted(run_root.glob("per_trajectory/rank_*"))
    for rank_dir in rank_dirs:
        motion = rank_dir.name
        for method, spec in METHOD_SPECS.items():
            summary_paths: list[Path] = []
            for pattern in spec["patterns"]:
                summary_paths.extend(sorted(rank_dir.glob(pattern)))
            if not summary_paths:
                continue
            if len(summary_paths) > 1:
                print(
                    f"[WARN] {motion}/{method}: {len(summary_paths)} summaries "
                    f"found, using {summary_paths[-1]}"
                )
            summary = _load_summary(summary_paths[-1])
            if summary is None:
                continue
            success, success_source = _success_rate(summary)
            row: dict[str, Any] = {
                "motion": motion,
                "method": method,
                "interface": spec["interface"],
                "setting": spec["setting"],
                "success": success,
                "success_source": success_source,
                "survival_steps": _survival_steps(summary),
                "summary_path": str(summary_paths[-1]),
            }
            for metric in METRIC_COLUMNS:
                row[metric] = _metric_mean(summary, metric)
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None and not math.isnan(v)]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def aggregate_methods(
    rows: list[dict[str, Any]], oracle_success_threshold: float
) -> list[dict[str, Any]]:
    # Oracle pass/fail per (interface, motion) drives the gate for planner rows.
    oracle_pass: dict[tuple[str, str], bool] = {}
    for row in rows:
        if row["setting"] == "oracle" and row["success"] is not None:
            oracle_pass[(row["interface"], row["motion"])] = (
                row["success"] >= oracle_success_threshold
            )

    aggregates: list[dict[str, Any]] = []
    for method, spec in METHOD_SPECS.items():
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        gated_rows = [
            row
            for row in method_rows
            if oracle_pass.get((row["interface"], row["motion"]), False)
        ]
        entry: dict[str, Any] = {
            "method": method,
            "interface": spec["interface"],
            "setting": spec["setting"],
            "num_motions": len(method_rows),
            "num_oracle_pass": len(gated_rows),
            "success": _mean([row["success"] for row in method_rows]),
            "success_gated": _mean([row["success"] for row in gated_rows]),
            "success_source": ",".join(
                sorted({row["success_source"] for row in method_rows})
            ),
            "survival_steps": _mean([row["survival_steps"] for row in method_rows]),
        }
        for metric in METRIC_COLUMNS:
            entry[metric] = _mean([row[metric] for row in method_rows])
            entry[f"{metric}_gated"] = _mean([row[metric] for row in gated_rows])
        aggregates.append(entry)
    return aggregates


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_outputs(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    output_dir: Path,
    oracle_success_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_motion_csv = output_dir / "per_motion.csv"
    if rows:
        with per_motion_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    aggregate_csv = output_dir / "method_summary.csv"
    if aggregates:
        with aggregate_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0].keys()))
            writer.writeheader()
            writer.writerows(aggregates)

    markdown_columns = [
        "method",
        "num_motions",
        "num_oracle_pass",
        "success",
        "success_gated",
        "success_source",
        "survival_steps",
        *METRIC_COLUMNS,
    ]
    lines = [
        "# LAFAN1 Motion Tracking Summary",
        "",
        f"Oracle gate: interface oracle success >= {oracle_success_threshold} "
        "per motion. `success_gated` and `*_gated` columns average only over "
        "motions passing that interface's oracle gate.",
        "",
        "Planner rows for an interface whose oracle fails the gate on most "
        "motions should not be used for interface-comparison claims: they "
        "measure low-level incompetence, not planner or interface quality.",
        "",
        "| " + " | ".join(markdown_columns) + " |",
        "| " + " | ".join(["---"] * len(markdown_columns)) + " |",
    ]
    for entry in aggregates:
        lines.append(
            "| "
            + " | ".join(_format_cell(entry.get(column)) for column in markdown_columns)
            + " |"
        )
    (output_dir / "method_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"[INFO] Wrote {per_motion_csv}")
    print(f"[INFO] Wrote {aggregate_csv}")
    print(f"[INFO] Wrote {output_dir / 'method_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_root",
        type=Path,
        required=True,
        help=(
            "Experiment root containing latent/, ee/, fb/ subroots (or a single "
            "root with per_trajectory/ directly)."
        ),
    )
    parser.add_argument("--oracle_success_threshold", type=float, default=0.8)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <run_root>/summary.",
    )
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Run root not found: {run_root}")

    rows = collect_rows(run_root)
    if not rows:
        raise SystemExit(
            f"No per-motion summaries found under {run_root}. Expected "
            "<root>/{latent,ee,fb}/per_trajectory/rank_*/... result dirs."
        )
    aggregates = aggregate_methods(rows, args.oracle_success_threshold)
    output_dir = args.output_dir or (run_root / "summary")
    write_outputs(rows, aggregates, output_dir, args.oracle_success_threshold)

    for entry in aggregates:
        if entry["setting"] == "oracle" and entry["success"] is not None:
            if entry["success"] < args.oracle_success_threshold:
                print(
                    f"[WARN] {entry['interface']} oracle mean success "
                    f"{entry['success']:.3f} is below the "
                    f"{args.oracle_success_threshold} gate: planner rows for "
                    "this interface are not paper-facing."
                )


if __name__ == "__main__":
    main()
