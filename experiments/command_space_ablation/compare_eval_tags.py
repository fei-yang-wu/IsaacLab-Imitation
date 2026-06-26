#!/usr/bin/env python3
"""Compare merged command-space evaluation tags against a baseline tag."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_eval_csv import _format_number, _markdown_table, _parse_float


METRICS = [
    "return_sum_mean",
    "survival_steps_mean",
    "done_rate",
    "root_pos_xy_error_m_mean",
    "joint_pos_rmse_rad_mean",
    "ee_pos_error_m_mean",
    "action_delta_l2_mean",
]

OUTPUT_COLUMNS = [
    "tag",
    "command_space",
    "planner_mode",
    "planner_update_interval",
    "planner_noise_std",
    "n",
    "return_sum_mean",
    "return_vs_baseline",
    "survival_steps_mean",
    "survival_vs_baseline",
    "done_rate",
    "joint_pos_rmse_rad_mean",
    "ee_pos_error_m_mean",
    "action_delta_l2_mean",
]


def _read_rows(csv_path: Path, tag: str) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["tag"] = tag
    return rows


def _resolve_tag(eval_root: Path, tag_spec: str) -> tuple[str, Path]:
    if "=" in tag_spec:
        label, dirname = tag_spec.split("=", 1)
    else:
        label = dirname = tag_spec
    tag_dir = (eval_root / dirname).expanduser().resolve()
    for filename in ("merged.csv", "summary.csv"):
        csv_path = tag_dir / filename
        if csv_path.is_file():
            return label, csv_path
    raise FileNotFoundError(f"No merged.csv or summary.csv found under {tag_dir}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_std_cell(values: list[float]) -> str:
    if len(values) == 0:
        return ""
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return f"{_format_number(mean)} +/- {_format_number(variance**0.5)}"


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    group_keys = [
        "tag",
        "command_space",
        "planner_mode",
        "planner_update_interval",
        "planner_noise_std",
    ]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(column, "") for column in group_keys)
        groups.setdefault(key, []).append(row)

    outputs: list[dict[str, str]] = []
    for key, group_rows in sorted(groups.items()):
        output = {column: value for column, value in zip(group_keys, key)}
        output["n"] = str(len(group_rows))
        for metric in METRICS:
            values = [
                parsed
                for row in group_rows
                if (parsed := _parse_float(row.get(metric, ""))) is not None
            ]
            output[f"{metric}__mean_value"] = str(_mean(values)) if values else ""
            output[metric] = _mean_std_cell(values)
        outputs.append(output)
    return outputs


def _baseline_by_command_space(
    rows: list[dict[str, str]],
    *,
    baseline_tag: str,
    baseline_planner_mode: str,
) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("tag") != baseline_tag:
            continue
        if baseline_planner_mode and row.get("planner_mode") != baseline_planner_mode:
            continue
        command_space = row.get("command_space", "")
        baselines[command_space] = {}
        for metric in ("return_sum_mean", "survival_steps_mean"):
            value = _parse_float(row.get(f"{metric}__mean_value", ""))
            if value is not None:
                baselines[command_space][metric] = value
    return baselines


def _format_ratio(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline == 0.0:
        return ""
    return f"{_format_number(100.0 * value / baseline)}%"


def _comparison_rows(
    rows: list[dict[str, str]],
    *,
    baseline_tag: str,
    baseline_planner_mode: str,
) -> list[dict[str, str]]:
    baselines = _baseline_by_command_space(
        rows,
        baseline_tag=baseline_tag,
        baseline_planner_mode=baseline_planner_mode,
    )
    outputs: list[dict[str, str]] = []
    for row in rows:
        command_space = row.get("command_space", "")
        baseline = baselines.get(command_space, {})
        output = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
        return_value = _parse_float(row.get("return_sum_mean__mean_value", ""))
        survival_value = _parse_float(row.get("survival_steps_mean__mean_value", ""))
        output["return_vs_baseline"] = _format_ratio(
            return_value, baseline.get("return_sum_mean")
        )
        output["survival_vs_baseline"] = _format_ratio(
            survival_value, baseline.get("survival_steps_mean")
        )
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_root",
        type=Path,
        required=True,
        help="Directory containing per-tag evaluation result directories.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        required=True,
        help="Evaluation tag to include. Use LABEL=DIRNAME to rename a tag.",
    )
    parser.add_argument(
        "--baseline_tag",
        type=str,
        required=True,
        help="Included tag label used as the per-command-space baseline.",
    )
    parser.add_argument(
        "--baseline_planner_mode",
        type=str,
        default="reference",
        help="Planner mode within the baseline tag. Use an empty string to ignore.",
    )
    parser.add_argument("--output_md", type=Path, default=None)
    args = parser.parse_args()

    eval_root = args.eval_root.expanduser().resolve()
    all_rows: list[dict[str, str]] = []
    for tag_spec in args.tag:
        label, csv_path = _resolve_tag(eval_root, tag_spec)
        all_rows.extend(_read_rows(csv_path, label))

    aggregate_rows = _aggregate(all_rows)
    comparison_rows = _comparison_rows(
        aggregate_rows,
        baseline_tag=args.baseline_tag,
        baseline_planner_mode=args.baseline_planner_mode,
    )
    table = _markdown_table(comparison_rows, OUTPUT_COLUMNS)
    print(table, end="")

    if args.output_md is not None:
        output_md = args.output_md.expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(table, encoding="utf-8")
        print(f"[INFO] Wrote comparison Markdown: {output_md}")


if __name__ == "__main__":
    main()
