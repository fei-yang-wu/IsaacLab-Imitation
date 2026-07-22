#!/usr/bin/env python3
"""Aggregate per-trajectory eval summary.json files into long/wide/mean tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PRIMARY_KEYS = (
    "success_rate",
    "mpjpe_l_mm",
    "e_vel",
    "e_acc",
    "survival_steps_mean",
    "joint_pos_rmse_rad_mean",
)

STATUS_KEY = "status"

NUMERIC_PRIMARY_KEYS = PRIMARY_KEYS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Table root containing W*/interface/trajectories/... cells.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--table-name", type=str, default="comparison")
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
        return None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        if math.isfinite(number):
            return number
    return None


def _load_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    summaries: list[Path] = []

    def add_summary(path: Path) -> None:
        if not path.is_file():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        summaries.append(path)

    # Avoid broad recursive scans through generated caches.
    for window_dir in sorted(path for path in root.glob("W*") if path.is_dir()):
        for interface_dir in sorted(path for path in window_dir.iterdir() if path.is_dir()):
            trajectory_root = interface_dir / "trajectories"
            if trajectory_root.is_dir():
                for trajectory_dir in sorted(
                    path for path in trajectory_root.iterdir() if path.is_dir()
                ):
                    eval_root = trajectory_dir / "eval"
                    if not eval_root.is_dir():
                        continue
                    for setting_dir in sorted(
                        path for path in eval_root.iterdir() if path.is_dir()
                    ):
                        add_summary(setting_dir / "summary.json")

            shared_eval_root = interface_dir / "eval"
            if shared_eval_root.is_dir():
                for setting_dir in sorted(
                    path for path in shared_eval_root.iterdir() if path.is_dir()
                ):
                    add_summary(setting_dir / "summary.json")

    for summary in summaries:
        try:
            payload = json.loads(summary.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "interface": "",
                    "window": "",
                    "rank": "",
                    "trajectory": "",
                    "setting": summary.parent.name,
                    STATUS_KEY: f"invalid_json:{exc}",
                    "json_path": str(summary),
                }
            )
            continue
        aggregate = (
            payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
        )
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        row = {
            "interface": payload.get("interface", ""),
            "window": payload.get("window", ""),
            "rank": payload.get("rank", ""),
            "trajectory": payload.get("trajectory", ""),
            "setting": payload.get("setting", summary.parent.name),
            "table": payload.get("table", ""),
            "planner_unit": payload.get("planner_unit", "trajectory"),
            STATUS_KEY: payload.get(STATUS_KEY, ""),
            "json_path": str(summary),
        }
        for key in NUMERIC_PRIMARY_KEYS:
            value = payload.get(key)
            if value is None:
                value = aggregate.get(key)
            if value is None and isinstance(metrics.get(key), dict):
                value = metrics[key].get("mean")
            row[key] = value
        if not row["interface"]:
            metadata = (
                payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            )
            row["interface"] = metadata.get("interface", "")
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_wide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("interface"),
            row.get("window"),
            row.get("rank"),
            row.get("trajectory"),
        )
        wide = grouped.setdefault(
            key,
            {
                "interface": row.get("interface"),
                "window": row.get("window"),
                "rank": row.get("rank"),
                "trajectory": row.get("trajectory"),
                "table": row.get("table"),
            },
        )
        setting = str(row.get("setting") or "")
        for metric in (*NUMERIC_PRIMARY_KEYS, STATUS_KEY):
            wide[f"{setting}_{metric}"] = row.get(metric)
        oracle_succ = _as_float(wide.get("oracle_success_rate"))
        ft_succ = _as_float(wide.get("finetuned_success_rate"))
        if oracle_succ is not None and ft_succ is not None and oracle_succ != 0.0:
            wide["finetuned_succ_oracle_ratio"] = ft_succ / oracle_succ
        else:
            wide.setdefault("finetuned_succ_oracle_ratio", None)
    return list(grouped.values())


def _mean_std(values: list[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    mean = float(statistics.fmean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean, std, len(values)


def _mean_over_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std across trajectories for each interface, window, and setting."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get(STATUS_KEY) or "").startswith("invalid_json"):
            continue
        key = (row.get("interface"), row.get("window"), row.get("setting"), row.get("table"))
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (interface, window, setting, table), group in sorted(
        groups.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        summary: dict[str, Any] = {
            "interface": interface,
            "window": window,
            "setting": setting,
            "table": table,
            "n_trajectories": len(group),
        }
        for key in NUMERIC_PRIMARY_KEYS:
            values = [
                number
                for number in (_as_float(row.get(key)) for row in group)
                if number is not None
            ]
            mean, std, count = _mean_std(values)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
            summary[f"{key}_n"] = count
        out.append(summary)
    return out


def _mean_wide(mean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot mean/std rows so oracle/pretrained/finetuned sit on one line."""
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in mean_rows:
        key = (row.get("interface"), row.get("window"), row.get("table"))
        wide = grouped.setdefault(
            key,
            {
                "interface": row.get("interface"),
                "window": row.get("window"),
                "table": row.get("table"),
                "n_trajectories": row.get("n_trajectories"),
            },
        )
        # Keep the max n if settings differ slightly in coverage.
        n_existing = int(wide.get("n_trajectories") or 0)
        n_row = int(row.get("n_trajectories") or 0)
        wide["n_trajectories"] = max(n_existing, n_row)
        setting = str(row.get("setting") or "")
        for key_name in NUMERIC_PRIMARY_KEYS:
            wide[f"{setting}_{key_name}_mean"] = row.get(f"{key_name}_mean")
            wide[f"{setting}_{key_name}_std"] = row.get(f"{key_name}_std")
            wide[f"{setting}_{key_name}_n"] = row.get(f"{key_name}_n")

        oracle_succ = _as_float(wide.get("oracle_success_rate_mean"))
        ft_succ = _as_float(wide.get("finetuned_success_rate_mean"))
        if oracle_succ is not None and ft_succ is not None and oracle_succ != 0.0:
            wide["finetuned_succ_oracle_ratio_mean"] = ft_succ / oracle_succ
        else:
            wide.setdefault("finetuned_succ_oracle_ratio_mean", None)
    return list(grouped.values())


def _write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_table(root: Path, output_dir: Path, table_name: str) -> dict[str, Path]:
    rows = _load_summaries(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_cols = [
        "interface",
        "window",
        "rank",
        "trajectory",
        "setting",
        "table",
        *NUMERIC_PRIMARY_KEYS,
        STATUS_KEY,
        "json_path",
    ]
    long_csv = output_dir / f"{table_name}_long.csv"
    long_md = output_dir / f"{table_name}_long.md"
    _write_csv(long_csv, rows, long_cols)
    _write_markdown(
        long_md,
        rows,
        [
            "interface",
            "window",
            "rank",
            "trajectory",
            "setting",
            STATUS_KEY,
            "success_rate",
        ],
    )

    wide_rows = _to_wide(rows)
    wide_cols = [
        "interface",
        "window",
        "rank",
        "trajectory",
        "table",
        "oracle_success_rate",
        "pretrained_success_rate",
        "finetuned_success_rate",
        "finetuned_succ_oracle_ratio",
        "oracle_status",
        "finetuned_status",
    ]
    wide_csv = output_dir / f"{table_name}_wide.csv"
    wide_md = output_dir / f"{table_name}_wide.md"
    _write_csv(wide_csv, wide_rows, wide_cols)
    _write_markdown(
        wide_md,
        wide_rows,
        [
            "interface",
            "window",
            "rank",
            "trajectory",
            "oracle_status",
            "finetuned_status",
            "finetuned_succ_oracle_ratio",
        ],
    )

    mean_rows = _mean_over_trajectories(rows)
    mean_cols = [
        "interface",
        "window",
        "setting",
        "table",
        "n_trajectories",
    ]
    for key in NUMERIC_PRIMARY_KEYS:
        mean_cols.extend([f"{key}_mean", f"{key}_std", f"{key}_n"])
    mean_csv = output_dir / f"{table_name}_mean_by_setting.csv"
    mean_md = output_dir / f"{table_name}_mean_by_setting.md"
    _write_csv(mean_csv, mean_rows, mean_cols)
    _write_markdown(
        mean_md,
        mean_rows,
        [
            "interface",
            "window",
            "setting",
            "n_trajectories",
            "success_rate_mean",
            "success_rate_std",
            "mpjpe_l_mm_mean",
            "mpjpe_l_mm_std",
        ],
    )

    mean_wide_rows = _mean_wide(mean_rows)
    mean_wide_cols = [
        "interface",
        "window",
        "table",
        "n_trajectories",
        "oracle_success_rate_mean",
        "oracle_success_rate_std",
        "pretrained_success_rate_mean",
        "pretrained_success_rate_std",
        "finetuned_success_rate_mean",
        "finetuned_success_rate_std",
        "finetuned_succ_oracle_ratio_mean",
        "oracle_mpjpe_l_mm_mean",
        "oracle_mpjpe_l_mm_std",
        "finetuned_mpjpe_l_mm_mean",
        "finetuned_mpjpe_l_mm_std",
    ]
    mean_wide_csv = output_dir / f"{table_name}_mean_wide.csv"
    mean_wide_md = output_dir / f"{table_name}_mean_wide.md"
    _write_csv(mean_wide_csv, mean_wide_rows, mean_wide_cols)
    _write_markdown(
        mean_wide_md,
        mean_wide_rows,
        [
            "interface",
            "window",
            "n_trajectories",
            "oracle_success_rate_mean",
            "oracle_success_rate_std",
            "finetuned_success_rate_mean",
            "finetuned_success_rate_std",
            "finetuned_succ_oracle_ratio_mean",
        ],
    )

    print(f"[aggregate] wrote {long_csv}")
    print(f"[aggregate] wrote {wide_csv}")
    print(f"[aggregate] wrote {mean_csv}")
    print(f"[aggregate] wrote {mean_wide_csv}")
    print(
        f"[aggregate] rows={len(rows)} wide_rows={len(wide_rows)} "
        f"mean_rows={len(mean_rows)} mean_wide_rows={len(mean_wide_rows)}"
    )
    return {
        "long": long_csv,
        "wide": wide_csv,
        "mean_by_setting": mean_csv,
        "mean_wide": mean_wide_csv,
    }


def main() -> None:
    args = _parse_args()
    aggregate_table(args.root, args.output_dir, args.table_name)


if __name__ == "__main__":
    main()
