#!/usr/bin/env python3
"""Aggregate interface-comparison tables across multiple result roots."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from summarize_interface_comparison import PRIMARY_COLUMNS, _format  # noqa: E402


PAPER_METRICS = [
    "output_dim",
    "planner_state_dim",
    "planner_state_history_steps",
    "planner_command_past_steps",
    "planner_command_future_steps",
    "planner_param_count",
    "planner_batch_size",
    "planner_micro_batch_size",
    "planner_gradient_accumulation_steps",
    "planner_num_updates",
    "planner_pretrain_num_updates",
    "planner_finetune_num_updates",
    "planner_lr",
    "planner_weight_decay",
    "planner_flow_num_inference_steps",
    "planner_flow_inference_noise_std",
    "planner_endpoint_num_inference_steps",
    "finetune_sample_count",
    "oracle_survival",
    "oracle_done_rate",
    "oracle_success_rate",
    "pretrained_survival",
    "pretrained_done_rate",
    "pretrained_success_rate",
    "finetuned_survival",
    "finetuned_done_rate",
    "finetuned_success_rate",
    "finetuned_survival_oracle_ratio",
    "pretrained_expert_target_rmse",
    "finetuned_achieved_target_rmse",
    "pretrained_expert_normalized_target_rmse",
    "finetuned_achieved_normalized_target_rmse",
    "pretrained_expert_eval_sample_count",
    "finetuned_achieved_eval_sample_count",
    "oracle_return",
    "pretrained_return",
    "finetuned_return",
    "finetuned_return_oracle_ratio",
    "finetuned_planner_target_rmse",
    "finetuned_root_xy_error",
    "finetuned_joint_rmse",
    "finetuned_ee_pos_error",
    "finetuned_action_delta",
]
INTEGER_METRICS = {
    "output_dim",
    "planner_state_dim",
    "planner_state_history_steps",
    "planner_command_past_steps",
    "planner_command_future_steps",
    "planner_param_count",
    "planner_batch_size",
    "planner_micro_batch_size",
    "planner_gradient_accumulation_steps",
    "planner_num_updates",
    "planner_pretrain_num_updates",
    "planner_finetune_num_updates",
    "planner_flow_num_inference_steps",
    "planner_endpoint_num_inference_steps",
    "finetune_sample_count",
    "pretrained_expert_eval_sample_count",
    "finetuned_achieved_eval_sample_count",
    "oracle_survival",
    "pretrained_survival",
    "finetuned_survival",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_roots",
        type=Path,
        nargs="*",
        default=[],
        help="Concrete result roots, one per seed/run.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern for result roots. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs/interface_baselines/multiseed_interface_comparison"),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate per-root wide tables before aggregating.",
    )
    return parser.parse_args()


def _discover_roots(args: argparse.Namespace) -> list[Path]:
    explicit_roots = [root.expanduser().resolve() for root in args.result_roots]
    roots = list(explicit_roots)
    for pattern in args.glob:
        roots.extend(Path(path) for path in glob.glob(pattern))
    roots = [root.expanduser().resolve() for root in roots if root.is_dir()]
    unique_by_path = {str(root): root for root in roots}
    explicit_paths = {str(root) for root in explicit_roots}
    unique = []
    for root in sorted(unique_by_path.values()):
        if _has_result_summaries(root):
            unique.append(root)
            continue
        if str(root) in explicit_paths:
            raise ValueError(f"Explicit result root has no summary.json files: {root}")
        print(f"[WARN] Skipping non-result root matched by glob: {root}")
    if not unique:
        raise ValueError("No result roots found.")
    return unique


def _has_result_summaries(root: Path) -> bool:
    return any(root.rglob("summary.json"))


def _ensure_wide_csv(root: Path, *, refresh: bool) -> Path:
    wide_csv = root / "interface_comparison_wide.csv"
    if refresh or not wide_csv.is_file():
        script = Path(__file__).resolve().parent / "summarize_interface_comparison.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--result_root",
                str(root),
            ],
            check=True,
        )
    if not wide_csv.is_file():
        raise FileNotFoundError(f"Missing wide table for {root}: {wide_csv}")
    return wide_csv


def _read_wide_rows(root: Path, *, refresh: bool) -> list[dict[str, Any]]:
    wide_csv = _ensure_wide_csv(root, refresh=refresh)
    with wide_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["result_root"] = str(root)
        row["seed"] = _seed_for_root(root)
    return rows


def _seed_for_root(root: Path) -> str:
    seed_from_name = re.search(r"seed([0-9]+)", root.name)
    if seed_from_name:
        return seed_from_name.group(1)
    fallback = root.name
    for path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        seed = metadata.get("seed")
        if seed not in (None, ""):
            return str(seed)
        planner_metadata = metadata.get("planner_metadata")
        if isinstance(planner_metadata, dict):
            sample_metadata = planner_metadata.get("sample_metadata")
            if isinstance(sample_metadata, dict):
                seed = sample_metadata.get("seed")
                if seed not in (None, ""):
                    return str(seed)
    return fallback


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for preferred in PRIMARY_COLUMNS:
        if preferred not in {"interface", "planner_variant"}:
            seen.add(preferred)
            columns.append(preferred)
    for row in rows:
        for key, value in row.items():
            if key in seen or key in {
                "interface",
                "planner_variant",
                "result_root",
                "seed",
            }:
                continue
            if _as_float(value) is not None:
                seen.add(key)
                columns.append(key)
    return columns


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("interface", "")), str(row.get("planner_variant", "")))
        groups[key].append(row)

    numeric_columns = _numeric_columns(rows)
    output_rows: list[dict[str, Any]] = []
    for (interface, variant), group_rows in sorted(groups.items()):
        seeds = sorted({str(row.get("seed", "")) for row in group_rows})
        output_row: dict[str, Any] = {
            "interface": interface,
            "planner_variant": variant,
            "num_seeds": len(seeds),
            "seeds": ",".join(seeds),
        }
        for column in numeric_columns:
            values = [
                value
                for value in (_as_float(row.get(column)) for row in group_rows)
                if value is not None
            ]
            output_row[f"{column}_n"] = len(values)
            if not values:
                output_row[f"{column}_mean"] = ""
                output_row[f"{column}_std"] = ""
                output_row[f"{column}_min"] = ""
                output_row[f"{column}_max"] = ""
                continue
            output_row[f"{column}_mean"] = mean(values)
            output_row[f"{column}_std"] = stdev(values) if len(values) > 1 else 0.0
            output_row[f"{column}_min"] = min(values)
            output_row[f"{column}_max"] = max(values)
        output_rows.append(output_row)
    return output_rows, numeric_columns


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_columns(numeric_columns: list[str]) -> list[str]:
    columns = ["interface", "planner_variant", "num_seeds", "seeds"]
    for column in numeric_columns:
        columns.extend(
            [
                f"{column}_mean",
                f"{column}_std",
                f"{column}_min",
                f"{column}_max",
                f"{column}_n",
            ]
        )
    return columns


def _paper_value(row: dict[str, Any], metric: str) -> str:
    metric_mean = row.get(f"{metric}_mean")
    metric_std = row.get(f"{metric}_std")
    metric_n = int(row.get(f"{metric}_n") or 0)
    if metric_mean in (None, "") or metric_n == 0:
        return ""
    formatted_mean = _format_paper_number(metric, float(metric_mean))
    if metric_n == 1:
        return formatted_mean
    metric_std_float = _as_float(metric_std) or 0.0
    if metric in INTEGER_METRICS and abs(metric_std_float) < 1.0e-6:
        return formatted_mean
    return f"{formatted_mean} +/- {_format(metric_std)}"


def _format_paper_number(metric: str, value: float) -> str:
    if metric in INTEGER_METRICS and abs(value - round(value)) < 1.0e-6:
        return str(int(round(value)))
    return _format(value)


def _paper_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if int(row.get("pretrained_survival_n") or 0) > 0
        or int(row.get("finetuned_survival_n") or 0) > 0
    ]
    return filtered or rows


def _write_paper_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["interface", "planner_variant", "num_seeds", "seeds", *PAPER_METRICS]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in _paper_rows(rows):
        cells = [
            str(row.get("interface", "")),
            str(row.get("planner_variant", "")),
            str(row.get("num_seeds", "")),
            str(row.get("seeds", "")),
        ]
        cells.extend(_paper_value(row, metric) for metric in PAPER_METRICS)
        body.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, sep, *body]) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    roots = _discover_roots(args)
    rows: list[dict[str, Any]] = []
    for root in roots:
        rows.extend(_read_wide_rows(root, refresh=bool(args.refresh)))
    if not rows:
        raise ValueError("No wide-table rows found.")

    output_dir = args.output_dir.expanduser().resolve()
    per_seed_columns = ["result_root", "seed", *PRIMARY_COLUMNS]
    _write_csv(rows, output_dir / "interface_comparison_by_seed.csv", per_seed_columns)

    aggregate_rows, numeric_columns = _aggregate(rows)
    _write_csv(
        aggregate_rows,
        output_dir / "interface_comparison_multiseed.csv",
        _aggregate_columns(numeric_columns),
    )
    _write_paper_markdown(
        aggregate_rows,
        output_dir / "interface_comparison_multiseed.md",
    )
    print(f"[INFO] Aggregated {len(rows)} rows from {len(roots)} roots.")
    print(f"[INFO] Wrote outputs under {output_dir}.")


if __name__ == "__main__":
    main()
