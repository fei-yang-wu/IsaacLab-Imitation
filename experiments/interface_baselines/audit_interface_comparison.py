#!/usr/bin/env python3
"""Audit aggregated interface-comparison outputs before using them in tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analyze_interface_sweep import REQUIRED_SELECTED_METRICS


DEFAULT_INTERFACES = ("latent_skill", "ee_trajectory", "full_body_trajectory")
PROVENANCE_FILENAME = "interface_comparison_run_provenance.json"
REQUIRED_AGGREGATE_METRICS = (
    "output_dim",
    "planner_param_count",
    "finetune_sample_count",
    "oracle_return",
    "oracle_survival",
    "oracle_done_rate",
    "oracle_success_rate",
    "pretrained_return",
    "pretrained_survival",
    "pretrained_done_rate",
    "pretrained_success_rate",
    "finetuned_return",
    "finetuned_survival",
    "finetuned_done_rate",
    "finetuned_success_rate",
    "finetuned_survival_oracle_ratio",
    "finetuned_root_xy_error",
    "finetuned_joint_rmse",
    "finetuned_ee_pos_error",
    "finetuned_action_delta",
    "finetuned_planner_target_rmse",
)
REQUIRED_OFFLINE_METRICS = (
    "pretrained_expert_target_rmse",
    "finetuned_achieved_target_rmse",
)
REQUIRED_BY_SEED_METRICS = (
    "output_dim",
    "planner_param_count",
    "finetune_sample_count",
    "oracle_return",
    "oracle_survival",
    "oracle_done_rate",
    "oracle_success_rate",
    "pretrained_return",
    "pretrained_survival",
    "pretrained_done_rate",
    "pretrained_success_rate",
    "finetuned_return",
    "finetuned_survival",
    "finetuned_done_rate",
    "finetuned_success_rate",
    "finetuned_planner_target_rmse",
    "pretrained_expert_target_rmse",
    "finetuned_achieved_target_rmse",
)


@dataclass
class Check:
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate_dir",
        type=Path,
        required=True,
        help="Directory containing interface_comparison_multiseed.csv.",
    )
    parser.add_argument(
        "--expected_interfaces",
        nargs="+",
        default=list(DEFAULT_INTERFACES),
    )
    parser.add_argument(
        "--expected_seeds",
        nargs="*",
        default=None,
        help="Expected seed ids, for example: --expected_seeds 0 1 2.",
    )
    parser.add_argument(
        "--expected_sample_count",
        type=int,
        default=None,
        help="Require every comparable row to use this finetune sample count.",
    )
    parser.add_argument(
        "--expected_planner_num_updates",
        type=int,
        default=None,
        help=(
            "Require comparable rows to report this effective planner update count. "
            "Use --expected_planner_finetune_num_updates for the explicit "
            "achieved-state finetune budget."
        ),
    )
    parser.add_argument(
        "--expected_planner_finetune_num_updates",
        type=int,
        default=None,
        help=(
            "Require comparable rows to report this achieved-state planner "
            "finetune update count."
        ),
    )
    parser.add_argument(
        "--expected_planner_pretrain_num_updates",
        type=int,
        default=None,
        help=(
            "Optionally require comparable rows to report this expert/offline "
            "pretrain update count. Do not use when latent and hand-designed "
            "interfaces intentionally use different offline pretraining recipes."
        ),
    )
    parser.add_argument(
        "--expected_planner_batch_size",
        type=int,
        default=None,
        help="Require comparable rows to use this effective planner batch size.",
    )
    parser.add_argument(
        "--expected_planner_lr",
        type=float,
        default=None,
        help="Require comparable rows to use this planner learning rate.",
    )
    parser.add_argument(
        "--expected_planner_weight_decay",
        type=float,
        default=None,
        help="Require comparable rows to use this planner weight decay.",
    )
    parser.add_argument(
        "--expected_planner_flow_num_inference_steps",
        type=int,
        default=None,
        help="Require comparable rows to report this flow inference step count.",
    )
    parser.add_argument(
        "--expected_planner_flow_inference_noise_std",
        type=float,
        default=None,
        help="Require comparable rows to report this flow inference noise std.",
    )
    parser.add_argument(
        "--expected_hand_designed_planner_state_history_steps",
        type=int,
        default=None,
        help=(
            "Require hand-designed interfaces to report this achieved-state "
            "history length. Latent rows are skipped because older latent "
            "summaries may not store the history count explicitly."
        ),
    )
    parser.add_argument(
        "--expected_hand_designed_planner_command_past_steps",
        type=int,
        default=None,
        help="Require hand-designed interfaces to report this command past window.",
    )
    parser.add_argument(
        "--expected_hand_designed_planner_command_future_steps",
        type=int,
        default=None,
        help="Require hand-designed interfaces to report this command future window.",
    )
    parser.add_argument(
        "--planner_variant",
        action="append",
        default=[],
        metavar="INTERFACE=VARIANT",
        help=(
            "Pin the audited planner variant for an interface. Repeat for "
            "multiple interfaces."
        ),
    )
    parser.add_argument(
        "--require_offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require offline target-error summaries for every interface.",
    )
    parser.add_argument(
        "--require_selected",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require and audit interface_sweep_selected.csv.",
    )
    parser.add_argument(
        "--use_selected_variants",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use interface_sweep_selected.csv to choose planner variants before "
            "auditing aggregate/by-seed metrics. Explicit --planner_variant "
            "settings take precedence."
        ),
    )
    parser.add_argument(
        "--require_provenance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Require aggregate and selected per-result-root run provenance JSONs."),
    )
    parser.add_argument(
        "--min_oracle_survival",
        type=float,
        default=None,
        help=(
            "Require selected aggregate rows to reach this oracle survival. "
            "Use this to gate paper-facing planner comparisons on low-level "
            "oracle competence."
        ),
    )
    parser.add_argument(
        "--min_oracle_success_rate",
        type=float,
        default=None,
        help=(
            "Require selected aggregate rows to reach this oracle success rate. "
            "Use with --min_oracle_survival for held-out fairness checks."
        ),
    )
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    if abs(number - round(number)) > 1.0e-6:
        return None
    return int(round(number))


def _parse_seed_set(raw: Any) -> set[str]:
    if raw in (None, ""):
        return set()
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _parse_variant_specs(raw_specs: list[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Expected INTERFACE=VARIANT, got: {raw}")
        interface, variant = raw.split("=", 1)
        interface = interface.strip()
        if not interface:
            raise ValueError(f"Missing interface in planner variant spec: {raw}")
        specs[interface] = variant.strip()
    return specs


def _selected_variant_specs(
    selected_table_rows: list[dict[str, str]] | None,
    expected_interfaces: list[str] | tuple[str, ...],
) -> dict[str, str]:
    if selected_table_rows is None:
        return {}
    expected = {str(interface) for interface in expected_interfaces}
    specs: dict[str, str] = {}
    for row in selected_table_rows:
        interface = str(row.get("interface", ""))
        if interface in expected and interface not in specs:
            specs[interface] = str(row.get("planner_variant", ""))
    return specs


def _check_json_file(
    checks: list[Check],
    path: Path,
    *,
    label: str,
    details: dict[str, Any] | None = None,
) -> bool:
    details = dict(details or {})
    details["path"] = str(path)
    if not path.is_file():
        checks.append(Check("fail", f"Missing {label}.", details))
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        details["error"] = str(exc)
        checks.append(Check("fail", f"Invalid {label} JSON.", details))
        return False
    except OSError as exc:
        details["error"] = str(exc)
        checks.append(Check("fail", f"Could not read {label}.", details))
        return False
    checks.append(Check("pass", f"Found valid {label}.", details))
    return True


def _metric_n(row: dict[str, Any], metric: str) -> int:
    return _as_int(row.get(f"{metric}_n")) or 0


def _metric_mean(row: dict[str, Any], metric: str) -> float | None:
    return _as_float(row.get(f"{metric}_mean"))


def _has_aggregate_metric(row: dict[str, Any], metric: str) -> bool:
    return _metric_n(row, metric) > 0 and _metric_mean(row, metric) is not None


def _matches_sample_count(
    row: dict[str, Any],
    expected_sample_count: int | None,
) -> bool:
    if expected_sample_count is None:
        return True
    values = [
        _metric_mean(row, "finetune_sample_count"),
        _as_float(row.get("finetune_sample_count_min")),
        _as_float(row.get("finetune_sample_count_max")),
    ]
    return all(
        value is not None and abs(value - expected_sample_count) <= 1.0e-6
        for value in values
    )


def _expected_planner_metric_specs(
    *,
    expected_planner_num_updates: int | None,
    expected_planner_finetune_num_updates: int | None,
    expected_planner_pretrain_num_updates: int | None,
    expected_planner_batch_size: int | None,
    expected_planner_lr: float | None,
    expected_planner_weight_decay: float | None,
    expected_planner_flow_num_inference_steps: int | None,
    expected_planner_flow_inference_noise_std: float | None,
) -> dict[str, float]:
    specs: dict[str, float] = {}
    for metric, value in (
        ("planner_num_updates", expected_planner_num_updates),
        ("planner_finetune_num_updates", expected_planner_finetune_num_updates),
        ("planner_pretrain_num_updates", expected_planner_pretrain_num_updates),
        ("planner_batch_size", expected_planner_batch_size),
        ("planner_lr", expected_planner_lr),
        ("planner_weight_decay", expected_planner_weight_decay),
        (
            "planner_flow_num_inference_steps",
            expected_planner_flow_num_inference_steps,
        ),
        (
            "planner_flow_inference_noise_std",
            expected_planner_flow_inference_noise_std,
        ),
    ):
        if value is not None:
            specs[metric] = float(value)
    return specs


def _expected_hand_designed_planner_metric_specs(
    *,
    expected_hand_designed_planner_state_history_steps: int | None,
    expected_hand_designed_planner_command_past_steps: int | None,
    expected_hand_designed_planner_command_future_steps: int | None,
) -> dict[str, float]:
    specs: dict[str, float] = {}
    for metric, value in (
        (
            "planner_state_history_steps",
            expected_hand_designed_planner_state_history_steps,
        ),
        (
            "planner_command_past_steps",
            expected_hand_designed_planner_command_past_steps,
        ),
        (
            "planner_command_future_steps",
            expected_hand_designed_planner_command_future_steps,
        ),
    ):
        if value is not None:
            specs[metric] = float(value)
    return specs


def _select_aggregate_row(
    rows: list[dict[str, str]],
    *,
    interface: str,
    variant_specs: dict[str, str],
    expected_sample_count: int | None,
    checks: list[Check],
) -> dict[str, str] | None:
    interface_rows = [row for row in rows if row.get("interface") == interface]
    if not interface_rows:
        checks.append(Check("fail", f"Missing aggregate row for {interface}."))
        return None

    if interface in variant_specs:
        variant = variant_specs[interface]
        candidates = [
            row
            for row in interface_rows
            if str(row.get("planner_variant", "")) == variant
        ]
    elif interface == "latent_skill":
        candidates = [
            row for row in interface_rows if str(row.get("planner_variant", "")) == ""
        ]
    else:
        candidates = [
            row
            for row in interface_rows
            if str(row.get("planner_variant", ""))
            and _has_aggregate_metric(row, "pretrained_survival")
            and _has_aggregate_metric(row, "finetuned_survival")
        ]
        candidates = [
            row
            for row in candidates
            if _matches_sample_count(row, expected_sample_count)
        ]

    if len(candidates) == 1:
        row = candidates[0]
        checks.append(
            Check(
                "pass",
                f"Selected comparable row for {interface}.",
                {"planner_variant": row.get("planner_variant", "")},
            )
        )
        return row
    if not candidates:
        checks.append(
            Check(
                "fail",
                f"No comparable aggregate row found for {interface}.",
                {
                    "available_variants": sorted(
                        {r.get("planner_variant", "") for r in interface_rows}
                    )
                },
            )
        )
        return None
    checks.append(
        Check(
            "fail",
            f"Multiple comparable rows found for {interface}; specify --planner_variant.",
            {
                "available_variants": sorted(
                    {r.get("planner_variant", "") for r in candidates}
                )
            },
        )
    )
    return None


def _check_aggregate_metric(
    checks: list[Check],
    row: dict[str, Any],
    *,
    interface: str,
    metric: str,
    expected_n: int | None,
) -> None:
    mean_value = _metric_mean(row, metric)
    metric_n = _metric_n(row, metric)
    if mean_value is None or metric_n == 0:
        checks.append(
            Check("fail", f"Missing aggregate metric {metric} for {interface}.")
        )
        return
    if expected_n is not None and metric_n != expected_n:
        checks.append(
            Check(
                "fail",
                f"Aggregate metric {metric} for {interface} has wrong n.",
                {"expected": expected_n, "actual": metric_n},
            )
        )
        return
    checks.append(
        Check(
            "pass",
            f"Aggregate metric {metric} present for {interface}.",
            {"mean": mean_value, "n": metric_n},
        )
    )


def _check_positive_metric(
    checks: list[Check],
    row: dict[str, Any],
    *,
    interface: str,
    metric: str,
) -> None:
    value = _metric_mean(row, metric)
    if value is None or value <= 0.0:
        checks.append(
            Check(
                "fail",
                f"Aggregate metric {metric} for {interface} must be positive.",
                {"actual": value},
            )
        )


def _check_sample_count(
    checks: list[Check],
    row: dict[str, Any],
    *,
    interface: str,
    expected_sample_count: int | None,
) -> None:
    if expected_sample_count is None:
        return
    if _matches_sample_count(row, expected_sample_count):
        checks.append(
            Check(
                "pass",
                f"Sample count matches for {interface}.",
                {"expected": expected_sample_count},
            )
        )
        return
    checks.append(
        Check(
            "fail",
            f"Sample count mismatch for {interface}.",
            {
                "expected": expected_sample_count,
                "mean": row.get("finetune_sample_count_mean"),
                "min": row.get("finetune_sample_count_min"),
                "max": row.get("finetune_sample_count_max"),
            },
        )
    )


def _check_expected_planner_metric(
    checks: list[Check],
    row: dict[str, Any],
    *,
    interface: str,
    metric: str,
    expected: float,
    tolerance: float = 1.0e-8,
) -> None:
    values = [
        _metric_mean(row, metric),
        _as_float(row.get(f"{metric}_min")),
        _as_float(row.get(f"{metric}_max")),
    ]
    if all(
        value is not None and abs(float(value) - float(expected)) <= tolerance
        for value in values
    ):
        checks.append(
            Check(
                "pass",
                f"Planner metric {metric} matches for {interface}.",
                {"expected": expected},
            )
        )
        return
    checks.append(
        Check(
            "fail",
            f"Planner metric {metric} mismatch for {interface}.",
            {
                "expected": expected,
                "mean": row.get(f"{metric}_mean"),
                "min": row.get(f"{metric}_min"),
                "max": row.get(f"{metric}_max"),
            },
        )
    )


def _check_metric_minimum(
    checks: list[Check],
    row: dict[str, Any],
    *,
    interface: str,
    metric: str,
    minimum: float | None,
) -> None:
    if minimum is None:
        return
    value = _metric_mean(row, metric)
    if value is not None and value >= minimum:
        checks.append(
            Check(
                "pass",
                f"Aggregate metric {metric} for {interface} clears minimum.",
                {"minimum": minimum, "actual": value},
            )
        )
        return
    checks.append(
        Check(
            "fail",
            f"Aggregate metric {metric} for {interface} is below minimum.",
            {"minimum": minimum, "actual": value},
        )
    )


def _check_seed_set(
    checks: list[Check],
    row: dict[str, Any],
    *,
    interface: str,
    expected_seeds: set[str] | None,
) -> int | None:
    actual_seeds = _parse_seed_set(row.get("seeds"))
    actual_num_seeds = _as_int(row.get("num_seeds"))
    if expected_seeds is None:
        if actual_num_seeds is None or actual_num_seeds <= 0 or not actual_seeds:
            checks.append(Check("fail", f"Missing seed metadata for {interface}."))
            return None
        checks.append(
            Check(
                "pass",
                f"Seed metadata present for {interface}.",
                {"seeds": sorted(actual_seeds), "num_seeds": actual_num_seeds},
            )
        )
        return actual_num_seeds
    if actual_seeds != expected_seeds:
        checks.append(
            Check(
                "fail",
                f"Seed set mismatch for {interface}.",
                {"expected": sorted(expected_seeds), "actual": sorted(actual_seeds)},
            )
        )
        return len(expected_seeds)
    if actual_num_seeds != len(expected_seeds):
        checks.append(
            Check(
                "fail",
                f"num_seeds mismatch for {interface}.",
                {"expected": len(expected_seeds), "actual": actual_num_seeds},
            )
        )
        return len(expected_seeds)
    checks.append(
        Check(
            "pass",
            f"Seed set matches for {interface}.",
            {"seeds": sorted(expected_seeds)},
        )
    )
    return len(expected_seeds)


def _check_by_seed_rows(
    checks: list[Check],
    by_seed_rows: list[dict[str, str]],
    *,
    interface: str,
    variant: str,
    expected_seeds: set[str] | None,
    expected_sample_count: int | None,
    expected_planner_metrics: dict[str, float] | None = None,
    require_provenance: bool = False,
    checked_provenance_paths: set[Path] | None = None,
) -> None:
    expected_planner_metrics = dict(expected_planner_metrics or {})
    candidates = [
        row
        for row in by_seed_rows
        if row.get("interface") == interface
        and str(row.get("planner_variant", "")) == variant
    ]
    if not candidates:
        checks.append(
            Check(
                "fail",
                f"Missing by-seed rows for {interface}.",
                {"planner_variant": variant},
            )
        )
        return
    actual_seeds = {str(row.get("seed", "")) for row in candidates}
    if expected_seeds is not None and actual_seeds != expected_seeds:
        checks.append(
            Check(
                "fail",
                f"By-seed seed set mismatch for {interface}.",
                {"expected": sorted(expected_seeds), "actual": sorted(actual_seeds)},
            )
        )
    else:
        checks.append(
            Check(
                "pass",
                f"By-seed rows present for {interface}.",
                {"seeds": sorted(actual_seeds), "planner_variant": variant},
            )
        )

    for row in candidates:
        seed = str(row.get("seed", ""))
        for metric in REQUIRED_BY_SEED_METRICS:
            value = _as_float(row.get(metric))
            if value is None:
                checks.append(
                    Check(
                        "fail",
                        f"Missing by-seed metric {metric} for {interface}.",
                        {"seed": seed, "planner_variant": variant},
                    )
                )
        if expected_sample_count is not None:
            sample_count = _as_int(row.get("finetune_sample_count"))
            if sample_count != expected_sample_count:
                checks.append(
                    Check(
                        "fail",
                        f"By-seed sample count mismatch for {interface}.",
                        {
                            "seed": seed,
                            "expected": expected_sample_count,
                            "actual": sample_count,
                        },
                    )
                )

        for metric, expected in expected_planner_metrics.items():
            value = _as_float(row.get(metric))
            if value is not None and abs(float(value) - float(expected)) <= 1.0e-8:
                checks.append(
                    Check(
                        "pass",
                        f"By-seed planner metric {metric} matches for {interface}.",
                        {
                            "seed": seed,
                            "expected": expected,
                            "actual": value,
                        },
                    )
                )
            else:
                checks.append(
                    Check(
                        "fail",
                        f"By-seed planner metric {metric} mismatch for {interface}.",
                        {
                            "seed": seed,
                            "expected": expected,
                            "actual": row.get(metric),
                        },
                    )
                )

        if require_provenance:
            raw_result_root = str(row.get("result_root", "")).strip()
            if not raw_result_root:
                checks.append(
                    Check(
                        "fail",
                        f"Missing result_root for {interface} by-seed row.",
                        {"seed": seed, "planner_variant": variant},
                    )
                )
                continue
            provenance_path = (
                Path(raw_result_root).expanduser().resolve() / PROVENANCE_FILENAME
            )
            if (
                checked_provenance_paths is not None
                and provenance_path in checked_provenance_paths
            ):
                continue
            if checked_provenance_paths is not None:
                checked_provenance_paths.add(provenance_path)
            _check_json_file(
                checks,
                provenance_path,
                label="result-root provenance",
                details={
                    "interface": interface,
                    "seed": seed,
                    "planner_variant": variant,
                    "result_root": raw_result_root,
                },
            )


def _check_selected_table(
    checks: list[Check],
    selected_table_rows: list[dict[str, str]] | None,
    *,
    selected_rows: list[dict[str, Any]],
    expected_interfaces: list[str] | tuple[str, ...],
    expected_sample_count: int | None,
    require_offline: bool,
) -> None:
    if selected_table_rows is None:
        return

    by_interface: dict[str, list[dict[str, str]]] = {}
    for row in selected_table_rows:
        by_interface.setdefault(str(row.get("interface", "")), []).append(row)

    expected_set = {str(interface) for interface in expected_interfaces}
    actual_set = set(by_interface)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    duplicates = sorted(
        interface for interface, rows in by_interface.items() if len(rows) != 1
    )
    if missing or extra or duplicates:
        checks.append(
            Check(
                "fail",
                "Selected table interface set mismatch.",
                {
                    "missing": missing,
                    "extra": extra,
                    "duplicates": duplicates,
                    "expected": sorted(expected_set),
                    "actual": sorted(actual_set),
                },
            )
        )
        return
    checks.append(
        Check(
            "pass",
            "Selected table contains exactly the expected interfaces.",
            {"interfaces": sorted(expected_set)},
        )
    )

    selected_variant_by_interface = {
        str(row.get("interface", "")): str(row.get("planner_variant", ""))
        for row in selected_rows
    }
    required_metrics = list(REQUIRED_SELECTED_METRICS)
    if not require_offline:
        required_metrics = [
            metric
            for metric in required_metrics
            if metric
            not in {
                "pretrained_expert_target_rmse",
                "finetuned_achieved_target_rmse",
            }
        ]
    for interface in expected_interfaces:
        interface = str(interface)
        row = by_interface[interface][0]
        expected_variant = selected_variant_by_interface.get(interface)
        actual_variant = str(row.get("planner_variant", ""))
        if expected_variant is not None and actual_variant != expected_variant:
            checks.append(
                Check(
                    "fail",
                    f"Selected table variant mismatch for {interface}.",
                    {"expected": expected_variant, "actual": actual_variant},
                )
            )
        else:
            checks.append(
                Check(
                    "pass",
                    f"Selected table variant matches for {interface}.",
                    {"planner_variant": actual_variant},
                )
            )

        if expected_sample_count is not None:
            sample_count = _as_int(row.get("finetune_sample_count"))
            if sample_count != expected_sample_count:
                checks.append(
                    Check(
                        "fail",
                        f"Selected table sample count mismatch for {interface}.",
                        {
                            "expected": expected_sample_count,
                            "actual": sample_count,
                        },
                    )
                )
            else:
                checks.append(
                    Check(
                        "pass",
                        f"Selected table sample count matches for {interface}.",
                        {"expected": expected_sample_count},
                    )
                )

        for metric in required_metrics:
            if _as_float(row.get(metric)) is None:
                checks.append(
                    Check(
                        "fail",
                        f"Missing selected table metric {metric} for {interface}.",
                        {"planner_variant": actual_variant},
                    )
                )


def audit_aggregate_dir(
    aggregate_dir: Path,
    *,
    expected_interfaces: list[str] | tuple[str, ...] = DEFAULT_INTERFACES,
    expected_seeds: set[str] | None = None,
    expected_sample_count: int | None = None,
    expected_planner_num_updates: int | None = None,
    expected_planner_finetune_num_updates: int | None = None,
    expected_planner_pretrain_num_updates: int | None = None,
    expected_planner_batch_size: int | None = None,
    expected_planner_lr: float | None = None,
    expected_planner_weight_decay: float | None = None,
    expected_planner_flow_num_inference_steps: int | None = None,
    expected_planner_flow_inference_noise_std: float | None = None,
    expected_hand_designed_planner_state_history_steps: int | None = None,
    expected_hand_designed_planner_command_past_steps: int | None = None,
    expected_hand_designed_planner_command_future_steps: int | None = None,
    variant_specs: dict[str, str] | None = None,
    require_offline: bool = True,
    require_selected: bool = False,
    use_selected_variants: bool = False,
    require_provenance: bool = False,
    min_oracle_survival: float | None = None,
    min_oracle_success_rate: float | None = None,
) -> dict[str, Any]:
    aggregate_dir = aggregate_dir.expanduser().resolve()
    variant_specs = dict(variant_specs or {})
    expected_planner_metrics = _expected_planner_metric_specs(
        expected_planner_num_updates=expected_planner_num_updates,
        expected_planner_finetune_num_updates=expected_planner_finetune_num_updates,
        expected_planner_pretrain_num_updates=expected_planner_pretrain_num_updates,
        expected_planner_batch_size=expected_planner_batch_size,
        expected_planner_lr=expected_planner_lr,
        expected_planner_weight_decay=expected_planner_weight_decay,
        expected_planner_flow_num_inference_steps=expected_planner_flow_num_inference_steps,
        expected_planner_flow_inference_noise_std=expected_planner_flow_inference_noise_std,
    )
    expected_hand_designed_planner_metrics = _expected_hand_designed_planner_metric_specs(
        expected_hand_designed_planner_state_history_steps=expected_hand_designed_planner_state_history_steps,
        expected_hand_designed_planner_command_past_steps=expected_hand_designed_planner_command_past_steps,
        expected_hand_designed_planner_command_future_steps=expected_hand_designed_planner_command_future_steps,
    )
    checks: list[Check] = []
    multiseed_path = aggregate_dir / "interface_comparison_multiseed.csv"
    by_seed_path = aggregate_dir / "interface_comparison_by_seed.csv"
    selected_path = aggregate_dir / "interface_sweep_selected.csv"
    provenance_path = aggregate_dir / PROVENANCE_FILENAME
    checked_provenance_paths: set[Path] = set()

    try:
        aggregate_rows = _read_csv(multiseed_path)
        checks.append(Check("pass", "Found multiseed aggregate CSV."))
    except FileNotFoundError:
        aggregate_rows = []
        checks.append(
            Check(
                "fail",
                "Missing multiseed aggregate CSV.",
                {"path": str(multiseed_path)},
            )
        )

    try:
        by_seed_rows = _read_csv(by_seed_path)
        checks.append(Check("pass", "Found by-seed aggregate CSV."))
    except FileNotFoundError:
        by_seed_rows = []
        checks.append(
            Check("fail", "Missing by-seed aggregate CSV.", {"path": str(by_seed_path)})
        )

    try:
        selected_table_rows = _read_csv(selected_path)
        checks.append(Check("pass", "Found selected sweep CSV."))
    except FileNotFoundError:
        selected_table_rows = None
        if require_selected:
            checks.append(
                Check(
                    "fail",
                    "Missing selected sweep CSV.",
                    {"path": str(selected_path)},
                )
            )
        else:
            checks.append(
                Check(
                    "warn",
                    "Selected sweep CSV not found; skipping selected-table audit.",
                    {"path": str(selected_path)},
                )
            )

    if require_provenance:
        _check_json_file(
            checks,
            provenance_path,
            label="aggregate provenance",
        )

    if use_selected_variants:
        if selected_table_rows is None:
            checks.append(
                Check(
                    "fail",
                    "Cannot use selected variants because selected sweep CSV is missing.",
                    {"path": str(selected_path)},
                )
            )
        else:
            for interface, variant in _selected_variant_specs(
                selected_table_rows,
                expected_interfaces,
            ).items():
                variant_specs.setdefault(interface, variant)
            checks.append(
                Check(
                    "pass",
                    "Loaded planner variants from selected sweep CSV.",
                    {"variant_specs": dict(sorted(variant_specs.items()))},
                )
            )

    selected_rows: list[dict[str, Any]] = []
    for interface in expected_interfaces:
        row = _select_aggregate_row(
            aggregate_rows,
            interface=interface,
            variant_specs=variant_specs,
            expected_sample_count=expected_sample_count,
            checks=checks,
        )
        if row is None:
            continue
        selected_rows.append(
            {
                "interface": interface,
                "planner_variant": row.get("planner_variant", ""),
                "seeds": sorted(_parse_seed_set(row.get("seeds"))),
                "output_dim": _metric_mean(row, "output_dim"),
                "planner_state_dim": _metric_mean(row, "planner_state_dim"),
                "planner_state_history_steps": _metric_mean(
                    row, "planner_state_history_steps"
                ),
                "planner_param_count": _metric_mean(row, "planner_param_count"),
                "planner_num_updates": _metric_mean(row, "planner_num_updates"),
                "planner_pretrain_num_updates": _metric_mean(
                    row, "planner_pretrain_num_updates"
                ),
                "planner_finetune_num_updates": _metric_mean(
                    row, "planner_finetune_num_updates"
                ),
                "planner_batch_size": _metric_mean(row, "planner_batch_size"),
                "planner_lr": _metric_mean(row, "planner_lr"),
                "planner_weight_decay": _metric_mean(row, "planner_weight_decay"),
                "planner_flow_num_inference_steps": _metric_mean(
                    row, "planner_flow_num_inference_steps"
                ),
                "planner_flow_inference_noise_std": _metric_mean(
                    row, "planner_flow_inference_noise_std"
                ),
                "finetune_sample_count": _metric_mean(row, "finetune_sample_count"),
            }
        )
        expected_n = _check_seed_set(
            checks,
            row,
            interface=interface,
            expected_seeds=expected_seeds,
        )
        for metric in REQUIRED_AGGREGATE_METRICS:
            _check_aggregate_metric(
                checks,
                row,
                interface=interface,
                metric=metric,
                expected_n=expected_n,
            )
        if require_offline:
            for metric in REQUIRED_OFFLINE_METRICS:
                _check_aggregate_metric(
                    checks,
                    row,
                    interface=interface,
                    metric=metric,
                    expected_n=expected_n,
                )
        _check_positive_metric(checks, row, interface=interface, metric="output_dim")
        _check_positive_metric(
            checks, row, interface=interface, metric="oracle_survival"
        )
        _check_metric_minimum(
            checks,
            row,
            interface=interface,
            metric="oracle_survival",
            minimum=min_oracle_survival,
        )
        _check_metric_minimum(
            checks,
            row,
            interface=interface,
            metric="oracle_success_rate",
            minimum=min_oracle_success_rate,
        )
        _check_sample_count(
            checks,
            row,
            interface=interface,
            expected_sample_count=expected_sample_count,
        )
        for metric, expected in expected_planner_metrics.items():
            _check_expected_planner_metric(
                checks,
                row,
                interface=interface,
                metric=metric,
                expected=expected,
            )
        row_expected_planner_metrics = dict(expected_planner_metrics)
        if interface != "latent_skill":
            for metric, expected in expected_hand_designed_planner_metrics.items():
                _check_expected_planner_metric(
                    checks,
                    row,
                    interface=interface,
                    metric=metric,
                    expected=expected,
                )
                row_expected_planner_metrics[metric] = expected
        _check_by_seed_rows(
            checks,
            by_seed_rows,
            interface=interface,
            variant=str(row.get("planner_variant", "")),
            expected_seeds=expected_seeds,
            expected_sample_count=expected_sample_count,
            expected_planner_metrics=row_expected_planner_metrics,
            require_provenance=require_provenance,
            checked_provenance_paths=checked_provenance_paths,
        )

    _check_selected_table(
        checks,
        selected_table_rows,
        selected_rows=selected_rows,
        expected_interfaces=expected_interfaces,
        expected_sample_count=expected_sample_count,
        require_offline=bool(require_offline),
    )

    failed = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warn"]
    return {
        "status": "fail" if failed else "pass",
        "aggregate_dir": str(aggregate_dir),
        "selected_rows": selected_rows,
        "num_checks": len(checks),
        "num_failed": len(failed),
        "num_warnings": len(warnings),
        "checks": [check.to_dict() for check in checks],
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    def cell(value: Any) -> Any:
        return "" if value is None else value

    lines = [
        "# Interface Comparison Audit",
        "",
        f"Status: `{report['status'].upper()}`",
        f"Aggregate dir: `{report['aggregate_dir']}`",
        f"Checks: {report['num_checks']} total, {report['num_failed']} failed",
        "",
        "## Selected Rows",
        "",
        "| Interface | Planner variant | Seeds | Output dim | State dim | State history | Params | Pretrain updates | Finetune updates | Batch | LR | Weight decay | Flow steps | Flow noise | Samples |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["selected_rows"]:
        lines.append(
            (
                "| {interface} | {variant} | {seeds} | {output_dim} | "
                "{state_dim} | {state_history} | {params} | "
                "{pretrain_updates} | {finetune_updates} | {batch} | {lr} | "
                "{weight_decay} | {flow_steps} | {flow_noise} | {samples} |"
            ).format(
                interface=row["interface"],
                variant=row["planner_variant"],
                seeds=",".join(row["seeds"]),
                output_dim=cell(row["output_dim"]),
                state_dim=cell(row.get("planner_state_dim")),
                state_history=cell(row.get("planner_state_history_steps")),
                params=cell(row.get("planner_param_count")),
                pretrain_updates=cell(row.get("planner_pretrain_num_updates")),
                finetune_updates=cell(row.get("planner_finetune_num_updates")),
                batch=cell(row.get("planner_batch_size")),
                lr=cell(row.get("planner_lr")),
                weight_decay=cell(row.get("planner_weight_decay")),
                flow_steps=cell(row.get("planner_flow_num_inference_steps")),
                flow_noise=cell(row.get("planner_flow_inference_noise_std")),
                samples=cell(row["finetune_sample_count"]),
            )
        )
    lines.extend(["", "## Failed Checks", ""])
    failed = [check for check in report["checks"] if check["status"] == "fail"]
    if failed:
        for check in failed:
            lines.append(f"- {check['message']} `{json.dumps(check['details'])}`")
    else:
        lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    expected_seeds = (
        {str(seed) for seed in args.expected_seeds}
        if args.expected_seeds is not None
        else None
    )
    report = audit_aggregate_dir(
        args.aggregate_dir,
        expected_interfaces=args.expected_interfaces,
        expected_seeds=expected_seeds,
        expected_sample_count=args.expected_sample_count,
        expected_planner_num_updates=args.expected_planner_num_updates,
        expected_planner_finetune_num_updates=args.expected_planner_finetune_num_updates,
        expected_planner_pretrain_num_updates=args.expected_planner_pretrain_num_updates,
        expected_planner_batch_size=args.expected_planner_batch_size,
        expected_planner_lr=args.expected_planner_lr,
        expected_planner_weight_decay=args.expected_planner_weight_decay,
        expected_planner_flow_num_inference_steps=args.expected_planner_flow_num_inference_steps,
        expected_planner_flow_inference_noise_std=args.expected_planner_flow_inference_noise_std,
        expected_hand_designed_planner_state_history_steps=args.expected_hand_designed_planner_state_history_steps,
        expected_hand_designed_planner_command_past_steps=args.expected_hand_designed_planner_command_past_steps,
        expected_hand_designed_planner_command_future_steps=args.expected_hand_designed_planner_command_future_steps,
        variant_specs=_parse_variant_specs(args.planner_variant),
        require_offline=bool(args.require_offline),
        require_selected=bool(args.require_selected),
        use_selected_variants=bool(args.use_selected_variants),
        require_provenance=bool(args.require_provenance),
        min_oracle_survival=args.min_oracle_survival,
        min_oracle_success_rate=args.min_oracle_success_rate,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md is not None:
        _write_markdown(report, args.output_md)

    failed = [check for check in report["checks"] if check["status"] == "fail"]
    for check in failed[:20]:
        print(f"[FAIL] {check['message']} {check['details']}", file=sys.stderr)
    if len(failed) > 20:
        print(f"[FAIL] ... {len(failed) - 20} more failures", file=sys.stderr)
    if failed:
        raise SystemExit(1)
    print(
        "[PASS] Audited {num} checks for {path}.".format(
            num=report["num_checks"],
            path=report["aggregate_dir"],
        )
    )


if __name__ == "__main__":
    main()
