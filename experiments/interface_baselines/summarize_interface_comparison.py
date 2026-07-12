#!/usr/bin/env python3
"""Summarize fair interface-baseline result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_COLUMNS = [
    "setting",
    "interface",
    "planner_variant",
    "return_sum_mean",
    "survival_steps_mean",
    "done_rate",
    "root_pos_xy_error_m_mean",
    "joint_pos_rmse_rad_mean",
    "ee_pos_error_m_mean",
    "action_delta_l2_mean",
    "planner_target_rmse_mean",
    "planner_target_dim",
    "planner_state_dim",
    "planner_state_history_steps",
    "planner_command_past_steps",
    "planner_command_future_steps",
    "finetune_sample_count",
    "planner_model_size",
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
    "planner_source_sample_count",
    "planner_selected_sample_count",
    "planner_heldout_sample_count",
    "eval_sample_count",
    "json_path",
]

SKIP_TABLE_SETTINGS = frozenset({"oracle_drive_samples"})
PRIMARY_SETTING_COLUMNS = {
    "oracle_low_level": "oracle",
    "eval_pretrained_closed_loop": "pretrained",
    "eval_finetuned_closed_loop": "finetuned",
}
OFFLINE_SETTING_COLUMNS = {
    "eval_pretrained_expert_state": "pretrained_expert",
    "eval_finetuned_achieved_state": "finetuned_achieved",
}
PRIMARY_METRICS = [
    ("return", "return_sum_mean"),
    ("survival", "survival_steps_mean"),
    ("done_rate", "done_rate"),
    ("root_xy_error", "root_pos_xy_error_m_mean"),
    ("joint_rmse", "joint_pos_rmse_rad_mean"),
    ("ee_pos_error", "ee_pos_error_m_mean"),
    ("action_delta", "action_delta_l2_mean"),
    ("planner_target_rmse", "planner_target_rmse_mean"),
]
OFFLINE_METRICS = [
    ("target_rmse", "planner_target_rmse_mean"),
    ("normalized_target_rmse", "planner_normalized_target_rmse_mean"),
]
PLANNER_METADATA_COLUMNS = (
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
    "planner_source_sample_count",
    "planner_selected_sample_count",
    "planner_heldout_sample_count",
    "finetune_sample_count",
)
PRIMARY_COLUMNS = [
    "interface",
    "planner_variant",
    "output_dim",
    *PLANNER_METADATA_COLUMNS,
    *[
        f"{setting}_{metric}"
        for setting in PRIMARY_SETTING_COLUMNS.values()
        for metric, _ in PRIMARY_METRICS
    ],
    *[f"{setting}_success_rate" for setting in PRIMARY_SETTING_COLUMNS.values()],
    *[
        f"{setting}_{metric}"
        for setting in OFFLINE_SETTING_COLUMNS.values()
        for metric, _ in OFFLINE_METRICS
    ],
    *[f"{setting}_eval_sample_count" for setting in OFFLINE_SETTING_COLUMNS.values()],
    "pretrained_return_oracle_ratio",
    "pretrained_survival_oracle_ratio",
    "finetuned_return_oracle_ratio",
    "finetuned_survival_oracle_ratio",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--output_wide_csv", type=Path, default=None)
    parser.add_argument("--output_wide_md", type=Path, default=None)
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Recursive glob pattern under result_root.",
    )
    return parser.parse_args()


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000.0:
            return f"{value:.1f}"
        if abs(value) >= 10.0:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def _maybe_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "metadata" in payload and "aggregate" in payload:
        return payload
    if "metric_means" not in payload:
        return None
    return payload


def _interface_from_path(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return None


def _setting_from_path(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    known_settings = (
        set(PRIMARY_SETTING_COLUMNS)
        | set(OFFLINE_SETTING_COLUMNS)
        | SKIP_TABLE_SETTINGS
    )
    for part in relative.parts[1:-1]:
        if part in known_settings:
            return part
    if len(relative.parts) >= 3:
        return relative.parts[1]
    return None


def _variant_from_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    if len(relative.parts) < 4:
        return ""
    setting = _setting_from_path(path, root)
    if setting is not None and relative.parts[1] != setting:
        return relative.parts[1]
    return ""


def _variant_root_for(path: Path, root: Path) -> Path | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    setting = _setting_from_path(path, root)
    if (
        len(relative.parts) >= 4
        and setting is not None
        and relative.parts[1] != setting
    ):
        return root / relative.parts[0] / relative.parts[1]
    return None


def _sample_count_from_config(config_path: Path) -> int | None:
    if not config_path.is_file():
        return None
    raw = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        return None
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        payload = payload["metadata"]
    sample_count = payload.get("num_samples")
    if sample_count in (None, ""):
        return None
    return int(sample_count)


def _sample_count_from_any_config(config_stem: Path) -> int | None:
    for suffix in (".json", ".yaml"):
        sample_count = _sample_count_from_config(config_stem.with_suffix(suffix))
        if sample_count is not None:
            return sample_count
    return None


def _sample_count_from_summary(summary_path: Path) -> int | None:
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    saved_rows = payload.get("saved_rows")
    if saved_rows not in (None, ""):
        return int(saved_rows)
    saved_steps = payload.get("saved_steps")
    metadata = payload.get("metadata")
    if saved_steps not in (None, "") and isinstance(metadata, dict):
        num_envs = metadata.get("num_envs")
        if num_envs not in (None, ""):
            return int(saved_steps) * int(num_envs)
    if saved_steps not in (None, ""):
        return int(saved_steps)
    return None


def _sample_count_from_samples_dir(samples_dir: Path) -> int | None:
    summary_count = _sample_count_from_summary(samples_dir.parent / "summary.json")
    if summary_count is not None:
        return summary_count
    sample_paths = sorted(samples_dir.glob("sample_step_*.pt"))
    if not sample_paths:
        return None
    count = 0
    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        tensor = None
        for key in ("planner_state", "target", "z_target"):
            candidate = sample.get(key)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
                break
        if not isinstance(tensor, torch.Tensor):
            raise KeyError(
                f"Sample {sample_path} is missing planner_state, target, and z_target tensors."
            )
        count += 1 if tensor.ndim == 1 else int(tensor.shape[0])
    return count


def _planner_sample_count_from_metadata(metadata: dict[str, Any]) -> int | None:
    planner_metadata = metadata.get("planner_metadata")
    if not isinstance(planner_metadata, dict):
        return None
    sample_count = planner_metadata.get("num_samples")
    if sample_count in (None, ""):
        return None
    return int(sample_count)


def _planner_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    planner_metadata = metadata.get("planner_metadata")
    if isinstance(planner_metadata, dict):
        return planner_metadata
    return {}


def _nested_value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return "" if current in (None, "") else current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _planner_value(planner_metadata: dict[str, Any], key: str) -> Any:
    value = planner_metadata.get(key)
    if value not in (None, ""):
        return value
    return _nested_value(planner_metadata, "args", key)


def _planner_sample_value(planner_metadata: dict[str, Any], key: str) -> Any:
    value = _nested_value(planner_metadata, "sample_metadata", key)
    if value not in (None, ""):
        return value
    return _nested_value(
        planner_metadata, "checkpoint_metadata", "sample_metadata", key
    )


def _planner_checkpoint_value(planner_metadata: dict[str, Any], key: str) -> Any:
    checkpoint_metadata = planner_metadata.get("checkpoint_metadata")
    if not isinstance(checkpoint_metadata, dict):
        return ""
    value = checkpoint_metadata.get(key)
    if value not in (None, ""):
        return value
    return _nested_value(checkpoint_metadata, "args", key)


def _planner_pretrain_num_updates(
    planner_metadata: dict[str, Any], setting: str
) -> Any:
    value = _planner_value(planner_metadata, "pretrain_num_updates")
    if value not in (None, ""):
        return value
    value = _planner_checkpoint_value(planner_metadata, "pretrain_num_updates")
    if value not in (None, ""):
        return value
    value = _planner_checkpoint_value(planner_metadata, "num_updates")
    if value not in (None, ""):
        return value
    if setting in {"eval_pretrained_closed_loop", "eval_pretrained_expert_state"}:
        return _planner_value(planner_metadata, "num_updates")
    return ""


def _planner_finetune_num_updates(
    planner_metadata: dict[str, Any], setting: str
) -> Any:
    value = _planner_value(planner_metadata, "finetune_num_updates")
    if value not in (None, ""):
        return value
    if setting in {"eval_finetuned_closed_loop", "eval_finetuned_achieved_state"}:
        return _planner_value(planner_metadata, "num_updates")
    return ""


def _sample_count_for(path: Path, root: Path) -> int | None:
    variant_root = _variant_root_for(path, root)
    if variant_root is not None:
        for config_stem in (
            variant_root / "planner_finetune_achieved_state" / "config",
            variant_root / "planner_pretrain_expert_state" / "config",
        ):
            sample_count = _sample_count_from_any_config(config_stem)
            if sample_count is not None:
                return sample_count
    interface = _interface_from_path(path, root)
    if interface is not None:
        samples_dir = (
            root / interface / "oracle_drive_samples" / "rollout_training_samples"
        )
        if samples_dir.is_dir():
            return _sample_count_from_samples_dir(samples_dir)
    for parent in [path.parent, *path.parents]:
        samples_dir = parent / "rollout_training_samples"
        if samples_dir.is_dir():
            return _sample_count_from_samples_dir(samples_dir)
    return None


def _path_or_payload_setting(path: Path, root: Path, metadata: dict[str, Any]) -> str:
    return (
        _setting_from_path(path, root)
        or metadata.get("setting")
        or metadata.get("label")
        or path.parent.name
    )


def _row(path: Path, payload: dict[str, Any], root: Path) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    aggregate = dict(payload.get("aggregate", {}))
    metrics = dict(payload.get("metrics", {}))
    planner_metadata = _planner_metadata(metadata)
    interface = (
        _interface_from_path(path, root)
        or metadata.get("interface")
        or metadata.get("command_space")
        or metadata.get("planner_mode")
        or "unknown"
    )
    setting = _path_or_payload_setting(path, root, metadata)
    planner_pretrain_num_updates = _planner_pretrain_num_updates(
        planner_metadata, str(setting)
    )
    planner_finetune_num_updates = _planner_finetune_num_updates(
        planner_metadata, str(setting)
    )
    row: dict[str, Any] = {
        "setting": setting,
        "interface": interface,
        "planner_variant": _variant_from_path(path, root),
        "json_path": str(path.relative_to(root)),
        "label": metadata.get("label", ""),
        "planner_target_dim": metadata.get("planner_target_dim", ""),
        "planner_state_dim": _first_present(
            _planner_value(planner_metadata, "state_dim"),
            _planner_checkpoint_value(planner_metadata, "state_dim"),
            metadata.get("planner_state_dim", ""),
        ),
        "planner_state_history_steps": _first_present(
            metadata.get("state_history_steps", ""),
            _planner_sample_value(planner_metadata, "state_history_steps"),
            _planner_value(planner_metadata, "state_history_steps"),
            _planner_checkpoint_value(planner_metadata, "state_history_steps"),
        ),
        "planner_command_past_steps": _first_present(
            metadata.get("command_past_steps", ""),
            _planner_sample_value(planner_metadata, "command_past_steps"),
            _planner_value(planner_metadata, "command_past_steps"),
            _planner_checkpoint_value(planner_metadata, "command_past_steps"),
        ),
        "planner_command_future_steps": _first_present(
            metadata.get("command_future_steps", ""),
            _planner_sample_value(planner_metadata, "command_future_steps"),
            _planner_value(planner_metadata, "command_future_steps"),
            _planner_checkpoint_value(planner_metadata, "command_future_steps"),
        ),
        "finetune_sample_count": _planner_sample_count_from_metadata(metadata)
        or _sample_count_for(path, root),
        "planner_model_size": _planner_value(planner_metadata, "model_size"),
        "planner_param_count": _planner_value(planner_metadata, "parameter_count"),
        "planner_batch_size": _planner_value(planner_metadata, "batch_size"),
        "planner_micro_batch_size": _planner_value(
            planner_metadata, "micro_batch_size"
        ),
        "planner_gradient_accumulation_steps": _planner_value(
            planner_metadata, "gradient_accumulation_steps"
        ),
        "planner_num_updates": _planner_value(planner_metadata, "num_updates"),
        "planner_pretrain_num_updates": planner_pretrain_num_updates,
        "planner_finetune_num_updates": planner_finetune_num_updates,
        "planner_lr": _planner_value(planner_metadata, "lr"),
        "planner_weight_decay": _planner_value(planner_metadata, "weight_decay"),
        "planner_flow_num_inference_steps": _planner_value(
            planner_metadata, "flow_num_inference_steps"
        ),
        "planner_flow_inference_noise_std": _planner_value(
            planner_metadata, "flow_inference_noise_std"
        ),
        "planner_endpoint_num_inference_steps": _planner_value(
            planner_metadata, "endpoint_num_inference_steps"
        ),
        "planner_source_sample_count": _planner_value(
            planner_metadata, "source_sample_count"
        ),
        "planner_selected_sample_count": _planner_value(
            planner_metadata, "selected_sample_count"
        ),
        "planner_heldout_sample_count": _planner_value(
            planner_metadata, "heldout_sample_count"
        ),
        "eval_sample_count": aggregate.get("sample_count", ""),
    }
    if row["planner_selected_sample_count"] in (None, ""):
        row["planner_selected_sample_count"] = _planner_value(
            planner_metadata, "num_samples"
        )
    row.update(aggregate)
    for metric_name, stats in metrics.items():
        if isinstance(stats, dict):
            for stat_name, value in stats.items():
                row[f"{metric_name}_{stat_name}"] = value
    metric_means = dict(payload.get("metric_means", {}))
    if metric_means:
        row["steps_run"] = payload.get("steps_run", "")
        row["stop_reason"] = payload.get("stop_reason", "")
        if not row.get("planner_target_dim"):
            row["planner_target_dim"] = payload.get("planner_target_dim", "")
        if "survival_steps_mean" not in row:
            row["survival_steps_mean"] = payload.get("steps_run", "")
        if "done_rate" not in row and payload.get("stop_reason") is not None:
            row["done_rate"] = 1.0 if payload.get("stop_reason") == "env_done" else 0.0
        if "planner_target_rmse_mean" not in row and "m3/z_mse" in metric_means:
            row["planner_target_rmse_mean"] = math.sqrt(
                max(float(metric_means["m3/z_mse"]), 0.0)
            )
        for metric_name, value in metric_means.items():
            row[f"{metric_name}_mean"] = value
    return row


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = list(DEFAULT_COLUMNS)
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_columns(
    rows: list[dict[str, Any]], output_csv: Path, columns: list[str]
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict[str, Any]]) -> str:
    header = "| " + " | ".join(DEFAULT_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in DEFAULT_COLUMNS) + " |"
    body = [
        "| "
        + " | ".join(_format(row.get(column, "")) for column in DEFAULT_COLUMNS)
        + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"


def _markdown_columns(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"


def _wide_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        interface = str(row.get("interface", "unknown"))
        variant = str(row.get("planner_variant", ""))
        setting = str(row.get("setting", ""))
        setting_column = PRIMARY_SETTING_COLUMNS.get(setting)
        group_key = f"{interface}/{variant}" if variant else interface
        metadata_priority = _planner_metadata_priority(setting)
        wide_row = grouped.setdefault(
            group_key,
            {
                "interface": interface,
                "planner_variant": variant,
                "output_dim": "",
                "_planner_metadata_priority": -1,
                **{column: "" for column in PLANNER_METADATA_COLUMNS},
            },
        )
        if row.get("planner_target_dim") not in (None, ""):
            wide_row["output_dim"] = row.get("planner_target_dim")
        for column in PLANNER_METADATA_COLUMNS:
            if row.get(column) in (None, ""):
                continue
            if wide_row.get(column) in (None, "") or metadata_priority >= int(
                wide_row["_planner_metadata_priority"]
            ):
                wide_row[column] = row.get(column)
                wide_row["_planner_metadata_priority"] = max(
                    int(wide_row["_planner_metadata_priority"]),
                    metadata_priority,
                )
        if setting_column is not None:
            for metric_column, source_column in PRIMARY_METRICS:
                wide_row[f"{setting_column}_{metric_column}"] = row.get(
                    source_column, ""
                )
            if row.get("tracking_success_rate") not in (None, ""):
                wide_row[f"{setting_column}_tracking_success_rate"] = row.get(
                    "tracking_success_rate"
                )
            continue
        offline_setting_column = OFFLINE_SETTING_COLUMNS.get(setting)
        if offline_setting_column is None:
            continue
        if row.get("eval_sample_count") not in (None, ""):
            wide_row[f"{offline_setting_column}_eval_sample_count"] = row.get(
                "eval_sample_count"
            )
        for metric_column, source_column in OFFLINE_METRICS:
            wide_row[f"{offline_setting_column}_{metric_column}"] = row.get(
                source_column, ""
            )
    _copy_interface_oracle_context(grouped)
    for wide_row in grouped.values():
        _add_success_rates(wide_row)
        _add_oracle_ratios(wide_row)
        wide_row.pop("_planner_metadata_priority", None)
    return [grouped[key] for key in sorted(grouped)]


def _planner_metadata_priority(setting: str) -> int:
    if setting == "eval_finetuned_closed_loop":
        return 40
    if setting == "eval_finetuned_achieved_state":
        return 30
    if setting == "eval_pretrained_closed_loop":
        return 20
    if setting == "eval_pretrained_expert_state":
        return 10
    return 0


def _copy_interface_oracle_context(grouped: dict[str, dict[str, Any]]) -> None:
    interface_oracle_rows = {
        str(row.get("interface", "")): row
        for row in grouped.values()
        if str(row.get("planner_variant", "")) == ""
    }
    oracle_columns = [
        column
        for column in PRIMARY_COLUMNS
        if column.startswith("oracle_")
        or column in {"output_dim", "finetune_sample_count"}
    ]
    for row in grouped.values():
        if str(row.get("planner_variant", "")) == "":
            continue
        oracle_row = interface_oracle_rows.get(str(row.get("interface", "")))
        if oracle_row is None:
            continue
        for column in oracle_columns:
            if row.get(column) in (None, "") and oracle_row.get(column) not in (
                None,
                "",
            ):
                row[column] = oracle_row.get(column)


def _add_oracle_ratios(row: dict[str, Any]) -> None:
    for setting in ("pretrained", "finetuned"):
        for metric in ("return", "survival"):
            numerator = _as_float(row.get(f"{setting}_{metric}"))
            denominator = _as_float(row.get(f"oracle_{metric}"))
            ratio_key = f"{setting}_{metric}_oracle_ratio"
            if numerator is None or denominator is None or denominator == 0.0:
                row[ratio_key] = ""
            else:
                row[ratio_key] = numerator / denominator


def _add_success_rates(row: dict[str, Any]) -> None:
    for setting in PRIMARY_SETTING_COLUMNS.values():
        tracking_success_rate = _as_float(row.get(f"{setting}_tracking_success_rate"))
        done_rate = _as_float(row.get(f"{setting}_done_rate"))
        success_key = f"{setting}_success_rate"
        if tracking_success_rate is not None:
            row[success_key] = tracking_success_rate
        elif done_rate is None:
            row[success_key] = ""
        else:
            row[success_key] = 1.0 - done_rate


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = _parse_args()
    root = args.result_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob(args.pattern)):
        if path.name in {"config.json", "config.yaml"}:
            continue
        payload = _maybe_summary(path)
        if payload is None:
            continue
        row = _row(path, payload, root)
        if row.get("setting") in SKIP_TABLE_SETTINGS:
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (str(row.get("interface", "")), str(row.get("setting", "")))
    )
    if not rows:
        raise ValueError(f"No result summaries found under {root}.")
    output_csv = args.output_csv or root / "interface_comparison.csv"
    output_md = args.output_md or root / "interface_comparison.md"
    output_wide_csv = args.output_wide_csv or root / "interface_comparison_wide.csv"
    output_wide_md = args.output_wide_md or root / "interface_comparison_wide.md"
    _write_csv(rows, output_csv.expanduser().resolve())
    output_md = output_md.expanduser().resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(rows), encoding="utf-8")
    wide_rows = _wide_rows(rows)
    _write_csv_columns(
        wide_rows,
        output_wide_csv.expanduser().resolve(),
        PRIMARY_COLUMNS,
    )
    output_wide_md = output_wide_md.expanduser().resolve()
    output_wide_md.parent.mkdir(parents=True, exist_ok=True)
    output_wide_md.write_text(
        _markdown_columns(wide_rows, PRIMARY_COLUMNS), encoding="utf-8"
    )
    print(f"[INFO] Wrote CSV: {output_csv}")
    print(f"[INFO] Wrote Markdown: {output_md}")
    print(f"[INFO] Wrote wide CSV: {output_wide_csv}")
    print(f"[INFO] Wrote wide Markdown: {output_wide_md}")


if __name__ == "__main__":
    main()
