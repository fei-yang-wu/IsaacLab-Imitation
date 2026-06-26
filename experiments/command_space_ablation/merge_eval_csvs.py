#!/usr/bin/env python3
"""Merge per-checkpoint evaluation CSVs and render summary tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_eval_csv import (
    DEFAULT_AGGREGATE_METRICS,
    DEFAULT_COLUMNS,
    _aggregate_rows,
    _markdown_table,
)


def _read_rows(input_dir: Path, pattern: str, output_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(input_dir.glob(pattern)):
        if csv_path.resolve() == output_csv.resolve():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                continue
            rows.extend(dict(row) for row in reader)
    return rows


def _union_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for fieldname in row:
            if fieldname in seen:
                continue
            seen.add(fieldname)
            fieldnames.append(fieldname)
    return fieldnames


def _write_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    if len(rows) == 0:
        raise ValueError("No evaluation rows found to merge.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _union_fieldnames(rows)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote merged CSV: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.csv")
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--summary_md", type=Path, default=None)
    parser.add_argument("--aggregate_md", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else input_dir / "summary.csv"
    )
    rows = _read_rows(input_dir, args.pattern, output_csv)
    rows.sort(
        key=lambda row: (
            row.get("command_space", ""),
            row.get("planner_mode", ""),
            row.get("seed", ""),
            row.get("label", ""),
        )
    )
    _write_csv(rows, output_csv)

    summary_md = (
        args.summary_md.expanduser().resolve()
        if args.summary_md is not None
        else input_dir / "summary.md"
    )
    summary = _markdown_table(rows, DEFAULT_COLUMNS)
    summary_md.write_text(summary, encoding="utf-8")
    print(f"[INFO] Wrote Markdown summary: {summary_md}")

    aggregate_md = (
        args.aggregate_md.expanduser().resolve()
        if args.aggregate_md is not None
        else input_dir / "aggregate.md"
    )
    aggregate_rows = _aggregate_rows(
        rows,
        group_by=["command_space", "planner_mode"],
        metrics=DEFAULT_AGGREGATE_METRICS,
    )
    aggregate = _markdown_table(
        aggregate_rows,
        ["command_space", "planner_mode", "n", *DEFAULT_AGGREGATE_METRICS],
    )
    aggregate_md.write_text(aggregate, encoding="utf-8")
    print(f"[INFO] Wrote aggregate Markdown summary: {aggregate_md}")


if __name__ == "__main__":
    main()
