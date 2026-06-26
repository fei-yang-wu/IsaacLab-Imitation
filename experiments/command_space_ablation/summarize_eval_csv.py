#!/usr/bin/env python3
"""Render a compact Markdown table from command-space evaluation CSV rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_COLUMNS = [
    "command_space",
    "planner_mode",
    "planner_update_interval",
    "seed",
    "return_sum_mean",
    "survival_steps_mean",
    "done_rate",
    "root_pos_xy_error_m_mean",
    "root_ori_error_rad_mean",
    "joint_pos_rmse_rad_mean",
    "joint_vel_rmse_radps_mean",
    "ee_pos_error_m_mean",
    "action_delta_l2_mean",
    "checkpoint",
]

DEFAULT_AGGREGATE_METRICS = [
    "return_sum_mean",
    "survival_steps_mean",
    "done_rate",
    "root_pos_xy_error_m_mean",
    "root_ori_error_rad_mean",
    "joint_pos_rmse_rad_mean",
    "joint_vel_rmse_radps_mean",
    "ee_pos_error_m_mean",
    "action_delta_l2_mean",
]


def _format_number(number: float) -> str:
    if abs(number) >= 1000.0:
        return f"{number:.1f}"
    if abs(number) >= 10.0:
        return f"{number:.2f}"
    return f"{number:.4f}"


def _format_cell(value: str | float | int) -> str:
    if not isinstance(value, str):
        return _format_number(float(value))
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped.lstrip("-").isdigit():
        return stripped
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return _format_number(number)


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_format_cell(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _aggregate_rows(
    rows: list[dict[str, str]],
    *,
    group_by: list[str],
    metrics: list[str],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(column, "") for column in group_by)
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, str]] = []
    for key, group_rows in sorted(groups.items()):
        output = {column: value for column, value in zip(group_by, key)}
        output["n"] = str(len(group_rows))
        for metric in metrics:
            values = [
                parsed
                for row in group_rows
                if (parsed := _parse_float(row.get(metric, ""))) is not None
            ]
            if len(values) == 0:
                output[metric] = ""
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            output[metric] = (
                f"{_format_number(mean)} +/- {_format_number(variance**0.5)}"
            )
        aggregate_rows.append(output)
    return aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        default=False,
        help="Aggregate numeric metric columns across rows with the same group keys.",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="command_space planner_mode",
        help="Whitespace-separated group columns used with --aggregate.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=" ".join(DEFAULT_COLUMNS),
        help="Whitespace-separated CSV columns to include.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=" ".join(DEFAULT_AGGREGATE_METRICS),
        help="Whitespace-separated numeric metric columns used with --aggregate.",
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.aggregate:
        group_by = [column for column in args.group_by.split() if column]
        metrics = [column for column in args.metrics.split() if column]
        rows = _aggregate_rows(rows, group_by=group_by, metrics=metrics)
        columns = [*group_by, "n", *metrics]
    else:
        columns = [column for column in args.columns.split() if column]
        rows.sort(key=lambda row: (row.get("command_space", ""), row.get("seed", "")))
    table = _markdown_table(rows, columns)
    print(table, end="")

    if args.output_md is not None:
        output_md = args.output_md.expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(table, encoding="utf-8")
        print(f"[INFO] Wrote Markdown summary: {output_md}")


if __name__ == "__main__":
    main()
