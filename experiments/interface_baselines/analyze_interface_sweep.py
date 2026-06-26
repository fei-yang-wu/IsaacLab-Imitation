#!/usr/bin/env python3
"""Summarize data/model sweeps for strong internal interface baselines."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_KEY_METRIC = "finetuned_survival_oracle_ratio"
SECONDARY_KEY_METRICS = (
    "finetuned_survival",
    "finetuned_return",
)
LOWER_IS_BETTER_TIEBREAKERS = (
    "finetuned_planner_target_rmse",
    "finetuned_root_xy_error",
    "finetuned_joint_rmse",
    "finetuned_ee_pos_error",
)
INTERFACE_ORDER = {
    "latent_skill": 0,
    "ee_trajectory": 1,
    "full_body_trajectory": 2,
}
DIAGNOSTIC_METRIC_NAMES = (
    "planner_target_rmse",
    "root_xy_error",
    "joint_rmse",
    "ee_pos_error",
    "action_delta",
)
PLANNER_INPUT_METRICS = (
    "planner_state_dim",
    "planner_state_history_steps",
    "planner_command_past_steps",
    "planner_command_future_steps",
)
PLANNER_CAPACITY_METRICS = (
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
)
SETTING_DIAGNOSTIC_COLUMNS = [
    f"{setting}_{metric}"
    for setting in ("oracle", "pretrained", "finetuned")
    for metric in DIAGNOSTIC_METRIC_NAMES
]
REQUIRED_SELECTED_METRICS = (
    "output_dim",
    "planner_param_count",
    "finetune_sample_count",
    "oracle_survival",
    "oracle_done_rate",
    "oracle_success_rate",
    "oracle_return",
    "oracle_root_xy_error",
    "oracle_joint_rmse",
    "oracle_ee_pos_error",
    "oracle_action_delta",
    "pretrained_survival",
    "pretrained_done_rate",
    "pretrained_success_rate",
    "pretrained_return",
    "pretrained_planner_target_rmse",
    "pretrained_root_xy_error",
    "pretrained_joint_rmse",
    "pretrained_ee_pos_error",
    "pretrained_action_delta",
    "finetuned_survival",
    "finetuned_done_rate",
    "finetuned_success_rate",
    "finetuned_return",
    "finetuned_planner_target_rmse",
    "finetuned_root_xy_error",
    "finetuned_joint_rmse",
    "finetuned_ee_pos_error",
    "finetuned_action_delta",
    "pretrained_expert_target_rmse",
    "finetuned_achieved_target_rmse",
)
SUMMARY_COLUMNS = [
    "interface",
    "planner_variant",
    "model_size",
    "sample_budget",
    "num_seeds",
    "seeds",
    "is_latent_reference",
    "is_sweep_candidate",
    "is_selection_candidate",
    "is_best_for_interface",
    "rank_within_interface",
    "output_dim",
    *PLANNER_INPUT_METRICS,
    "planner_param_count",
    *PLANNER_CAPACITY_METRICS,
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
    "oracle_return",
    "pretrained_return",
    "finetuned_return",
    "finetuned_return_oracle_ratio",
    *SETTING_DIAGNOSTIC_COLUMNS,
    "pretrained_expert_target_rmse",
    "finetuned_achieved_target_rmse",
    "gap_to_latent_finetuned_survival_oracle_ratio",
    "gap_to_latent_finetuned_survival",
    "gap_to_latent_finetuned_return",
]
SELECTED_COLUMNS = [
    "interface",
    "planner_variant",
    "model_size",
    "sample_budget",
    "num_seeds",
    "seeds",
    "rank_within_interface",
    "output_dim",
    *PLANNER_INPUT_METRICS,
    "planner_param_count",
    *PLANNER_CAPACITY_METRICS,
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
    "oracle_return",
    "pretrained_return",
    "finetuned_return",
    "finetuned_return_oracle_ratio",
    *SETTING_DIAGNOSTIC_COLUMNS,
    "pretrained_expert_target_rmse",
    "finetuned_achieved_target_rmse",
    "gap_to_latent_finetuned_survival_oracle_ratio",
    "gap_to_latent_finetuned_survival",
    "gap_to_latent_finetuned_return",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--aggregate_dir",
        type=Path,
        help="Directory containing interface_comparison_multiseed.csv.",
    )
    input_group.add_argument(
        "--input_csv",
        type=Path,
        help="Aggregate CSV to analyze directly.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Defaults to <aggregate_dir>/interface_sweep_summary.csv.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=None,
        help="Defaults to <aggregate_dir>/interface_sweep_summary.md.",
    )
    parser.add_argument(
        "--selected_csv",
        type=Path,
        default=None,
        help="Defaults to <aggregate_dir>/interface_sweep_selected.csv.",
    )
    parser.add_argument(
        "--selected_md",
        type=Path,
        default=None,
        help="Defaults to <aggregate_dir>/interface_sweep_selected.md.",
    )
    parser.add_argument(
        "--key_metric",
        default=DEFAULT_KEY_METRIC,
        help=(
            "Metric used to select best variants. For multiseed aggregates, pass "
            "the base metric name without _mean."
        ),
    )
    parser.add_argument(
        "--selected_sample_count",
        type=int,
        default=None,
        help=(
            "Only rank/select hand-designed variants with this finetune sample "
            "count. Omit to select the strongest available variant."
        ),
    )
    parser.add_argument(
        "--expected_selected_interfaces",
        nargs="*",
        default=None,
        help=(
            "Require the selected table to contain exactly one row for each "
            "listed interface. Useful when --selected_sample_count is set."
        ),
    )
    return parser.parse_args()


def _input_csv(args: argparse.Namespace) -> Path:
    if args.input_csv is not None:
        return args.input_csv.expanduser().resolve()
    return (
        args.aggregate_dir.expanduser().resolve() / "interface_comparison_multiseed.csv"
    )


def _default_output(args: argparse.Namespace, suffix: str) -> Path:
    if args.aggregate_dir is not None:
        return (
            args.aggregate_dir.expanduser().resolve()
            / f"interface_sweep_summary.{suffix}"
        )
    return (
        args.input_csv.expanduser()
        .resolve()
        .with_name(f"interface_sweep_summary.{suffix}")
    )


def _default_selected_output(args: argparse.Namespace, suffix: str) -> Path:
    if args.aggregate_dir is not None:
        return (
            args.aggregate_dir.expanduser().resolve()
            / f"interface_sweep_selected.{suffix}"
        )
    return (
        args.input_csv.expanduser()
        .resolve()
        .with_name(f"interface_sweep_selected.{suffix}")
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def _metric(row: dict[str, Any], metric: str) -> float | None:
    if f"{metric}_mean" in row:
        return _as_float(row.get(f"{metric}_mean"))
    return _as_float(row.get(metric))


def _metric_text(row: dict[str, Any], metric: str) -> str:
    value = _metric(row, metric)
    if value is None:
        return ""
    return _format(value)


def _parse_variant(variant: str) -> tuple[str, str]:
    match = re.fullmatch(r"chunked_transformer_([^_]+)_(.+)", variant)
    if match is None:
        return "", ""
    return match.group(1), match.group(2)


def _is_candidate(row: dict[str, Any]) -> bool:
    interface = str(row.get("interface", ""))
    variant = str(row.get("planner_variant", ""))
    if interface == "latent_skill":
        return False
    return bool(variant) and _metric(row, "finetuned_survival") is not None


def _matches_selected_sample_count(
    row: dict[str, Any], selected_sample_count: int | None
) -> bool:
    if selected_sample_count is None:
        return True
    sample_count = _metric(row, "finetune_sample_count")
    return (
        sample_count is not None and abs(sample_count - selected_sample_count) < 1.0e-6
    )


def _rank_key(row: dict[str, Any], key_metric: str) -> tuple[float, ...]:
    values: list[float] = []
    for metric in (key_metric, *SECONDARY_KEY_METRICS):
        values.append(_metric(row, metric) or float("-inf"))
    for metric in LOWER_IS_BETTER_TIEBREAKERS:
        value = _metric(row, metric)
        values.append(-(value if value is not None else float("inf")))
    sample_count = _metric(row, "finetune_sample_count")
    values.append(-(sample_count if sample_count is not None else float("inf")))
    return tuple(values)


def _interface_sort_key(interface: str) -> tuple[int, str]:
    return (INTERFACE_ORDER.get(interface, len(INTERFACE_ORDER)), interface)


def _best_latent_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    latent_rows = [
        row
        for row in rows
        if row.get("interface") == "latent_skill"
        and str(row.get("planner_variant", "")) == ""
        and _metric(row, "finetuned_survival") is not None
    ]
    if not latent_rows:
        return None
    return max(latent_rows, key=lambda row: _rank_key(row, DEFAULT_KEY_METRIC))


def analyze_sweep(
    rows: list[dict[str, str]],
    *,
    key_metric: str = DEFAULT_KEY_METRIC,
    selected_sample_count: int | None = None,
) -> list[dict[str, Any]]:
    latent_row = _best_latent_row(rows)
    candidates_by_interface: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if _is_candidate(row) and _matches_selected_sample_count(
            row,
            selected_sample_count,
        ):
            candidates_by_interface.setdefault(
                str(row.get("interface", "")), []
            ).append(row)

    rank_by_identity: dict[tuple[int, str], int] = {}
    best_identities: set[tuple[int, str]] = set()
    for interface_rows in candidates_by_interface.values():
        ranked = sorted(
            interface_rows, key=lambda row: _rank_key(row, key_metric), reverse=True
        )
        for index, row in enumerate(ranked, start=1):
            identity = (id(row), str(row.get("planner_variant", "")))
            rank_by_identity[identity] = index
            if index == 1:
                best_identities.add(identity)

    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        interface = str(row.get("interface", ""))
        variant = str(row.get("planner_variant", ""))
        is_latent_reference = latent_row is row
        is_candidate = _is_candidate(row)
        is_selection_candidate = is_candidate and _matches_selected_sample_count(
            row,
            selected_sample_count,
        )
        identity = (id(row), variant)
        model_size, sample_budget = _parse_variant(variant)
        output_row: dict[str, Any] = {
            "interface": interface,
            "planner_variant": variant,
            "model_size": model_size,
            "sample_budget": sample_budget,
            "num_seeds": row.get("num_seeds", ""),
            "seeds": row.get("seeds", ""),
            "is_latent_reference": int(is_latent_reference),
            "is_sweep_candidate": int(is_candidate),
            "is_selection_candidate": int(is_selection_candidate),
            "is_best_for_interface": int(identity in best_identities),
            "rank_within_interface": rank_by_identity.get(identity, ""),
        }
        for metric in (
            "output_dim",
            *PLANNER_INPUT_METRICS,
            "planner_param_count",
            *PLANNER_CAPACITY_METRICS,
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
            "oracle_return",
            "pretrained_return",
            "finetuned_return",
            "finetuned_return_oracle_ratio",
            *SETTING_DIAGNOSTIC_COLUMNS,
            "pretrained_expert_target_rmse",
            "finetuned_achieved_target_rmse",
        ):
            output_row[metric] = _metric(row, metric)
        if latent_row is not None:
            for metric in (
                "finetuned_survival_oracle_ratio",
                "finetuned_survival",
                "finetuned_return",
            ):
                row_value = _metric(row, metric)
                latent_value = _metric(latent_row, metric)
                gap_key = f"gap_to_latent_{metric}"
                if row_value is None or latent_value is None:
                    output_row[gap_key] = ""
                else:
                    output_row[gap_key] = row_value - latent_value
        summary_rows.append(output_row)

    return sorted(
        summary_rows,
        key=lambda row: (
            _interface_sort_key(str(row["interface"])),
            0 if row["is_latent_reference"] else 1,
            int(row["rank_within_interface"] or 10**9),
            str(row["planner_variant"]),
        ),
    )


def selected_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if int(row.get("is_latent_reference") or 0)
        or int(row.get("is_best_for_interface") or 0)
    ]
    return sorted(
        selected,
        key=lambda row: (
            _interface_sort_key(str(row.get("interface", ""))),
            int(row.get("rank_within_interface") or 10**9),
            str(row.get("planner_variant", "")),
        ),
    )


def validate_selected_interfaces(
    rows: list[dict[str, Any]],
    expected_interfaces: list[str] | None,
) -> None:
    if expected_interfaces is None:
        return
    expected = [interface for interface in expected_interfaces if interface]
    selected_rows = selected_sweep_rows(rows)
    selected_by_interface: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        selected_by_interface.setdefault(str(row.get("interface", "")), []).append(row)

    selected_interfaces = set(selected_by_interface)
    expected_set = set(expected)
    missing = sorted(expected_set - selected_interfaces)
    extra = sorted(selected_interfaces - expected_set)
    duplicates = sorted(
        interface
        for interface, interface_rows in selected_by_interface.items()
        if len(interface_rows) != 1
    )
    if missing or extra or duplicates:
        details = {
            "missing": missing,
            "extra": extra,
            "duplicates": duplicates,
            "selected": sorted(selected_interfaces),
            "expected": sorted(expected_set),
        }
        raise ValueError(f"Selected interface validation failed: {details}")


def validate_selected_metrics(rows: list[dict[str, Any]]) -> None:
    failures: dict[str, list[str]] = {}
    for row in selected_sweep_rows(rows):
        interface = str(row.get("interface", ""))
        missing = [
            metric
            for metric in REQUIRED_SELECTED_METRICS
            if _as_float(row.get(metric)) is None
        ]
        if missing:
            failures[interface] = missing
    if failures:
        raise ValueError(f"Selected metric validation failed: {failures}")


def _format(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000.0:
            return f"{value:.1f}"
        if abs(value) >= 10.0:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _write_selected_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SELECTED_COLUMNS,
            restval="",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _markdown(
    rows: list[dict[str, Any]],
    *,
    selected_sample_count: int | None,
) -> str:
    best_rows = selected_sweep_rows(rows)
    all_candidate_rows = [row for row in rows if row["is_sweep_candidate"]]
    selection_note = (
        "Selected rows are ranked across all available sample counts."
        if selected_sample_count is None
        else f"Selected rows are restricted to `{selected_sample_count}` samples."
    )
    sections = [
        "# Interface Sweep Summary",
        "",
        "## Selected Rows",
        "",
        selection_note,
        "",
        _table(
            best_rows,
            [
                "interface",
                "planner_variant",
                "num_seeds",
                "planner_param_count",
                "planner_batch_size",
                "planner_micro_batch_size",
                "planner_pretrain_num_updates",
                "planner_finetune_num_updates",
                "planner_lr",
                "planner_weight_decay",
                "planner_flow_num_inference_steps",
                "planner_endpoint_num_inference_steps",
                "finetune_sample_count",
                "finetuned_survival",
                "finetuned_done_rate",
                "finetuned_success_rate",
                "finetuned_survival_oracle_ratio",
                "finetuned_return",
                "finetuned_root_xy_error",
                "finetuned_joint_rmse",
                "finetuned_ee_pos_error",
                "finetuned_action_delta",
                "gap_to_latent_finetuned_survival_oracle_ratio",
            ],
        ),
        "",
        "## Sweep Candidates",
        "",
        _table(
            all_candidate_rows,
            [
                "interface",
                "planner_variant",
                "rank_within_interface",
                "is_selection_candidate",
                "num_seeds",
                "planner_param_count",
                "planner_batch_size",
                "planner_micro_batch_size",
                "planner_pretrain_num_updates",
                "planner_finetune_num_updates",
                "planner_lr",
                "planner_weight_decay",
                "planner_flow_num_inference_steps",
                "planner_endpoint_num_inference_steps",
                "finetune_sample_count",
                "pretrained_survival",
                "finetuned_survival",
                "finetuned_done_rate",
                "finetuned_success_rate",
                "finetuned_survival_oracle_ratio",
                "finetuned_planner_target_rmse",
                "finetuned_root_xy_error",
                "finetuned_joint_rmse",
                "finetuned_ee_pos_error",
                "finetuned_action_delta",
            ],
        ),
    ]
    return "\n".join(sections) + "\n"


def _selected_markdown(
    rows: list[dict[str, Any]],
    *,
    key_metric: str,
    selected_sample_count: int | None,
) -> str:
    selected_rows = selected_sweep_rows(rows)
    selection_scope = (
        "all available sample counts"
        if selected_sample_count is None
        else f"variants with `{selected_sample_count}` finetune samples"
    )
    sections = [
        "# Selected Interface Baseline Rows",
        "",
        "These are the rows to use for the paper-facing comparison table. "
        "`latent_skill` is kept as the reference row; each hand-designed "
        f"interface uses the top-ranked sweep candidate by `{key_metric}` with "
        "the tie-breakers declared in `analyze_interface_sweep.py`, considering "
        f"{selection_scope}.",
        "",
        _table(selected_rows, SELECTED_COLUMNS),
    ]
    return "\n".join(sections) + "\n"


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def main() -> None:
    args = _parse_args()
    input_csv = _input_csv(args)
    rows = analyze_sweep(
        _read_rows(input_csv),
        key_metric=args.key_metric,
        selected_sample_count=args.selected_sample_count,
    )
    validate_selected_interfaces(rows, args.expected_selected_interfaces)
    validate_selected_metrics(rows)
    output_csv = args.output_csv or _default_output(args, "csv")
    output_md = args.output_md or _default_output(args, "md")
    selected_csv = args.selected_csv or _default_selected_output(args, "csv")
    selected_md = args.selected_md or _default_selected_output(args, "md")
    _write_csv(rows, output_csv.expanduser().resolve())
    _write_selected_csv(selected_sweep_rows(rows), selected_csv.expanduser().resolve())
    output_md = output_md.expanduser().resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        _markdown(rows, selected_sample_count=args.selected_sample_count),
        encoding="utf-8",
    )
    selected_md = selected_md.expanduser().resolve()
    selected_md.parent.mkdir(parents=True, exist_ok=True)
    selected_md.write_text(
        _selected_markdown(
            rows,
            key_metric=args.key_metric,
            selected_sample_count=args.selected_sample_count,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] Wrote sweep summary to {output_csv} and {output_md}.")
    print(f"[INFO] Wrote selected sweep rows to {selected_csv} and {selected_md}.")


if __name__ == "__main__":
    main()
