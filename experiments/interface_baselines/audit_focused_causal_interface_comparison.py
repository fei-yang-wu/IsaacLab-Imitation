#!/usr/bin/env python3
"""Audit the focused latent-skill versus streamed-vanilla comparison."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch

from planner_sample_schema import (
    PAIRED_TARGET_CONTRACT,
    PLANNER_SAMPLE_FORMAT,
    PLANNER_SAMPLE_VERSION,
)


PLANNER_INTERFACES = ("latent_skill", "full_body_trajectory")
PAPER_ROLLOUT_METRICS = (
    "tracking_mpjpe_mm",
    "root_pos_xy_error_m",
    "root_height_error_m",
    "root_ori_error_rad",
    "joint_pos_rmse_rad",
    "ee_pos_error_m",
    "ee_ori_error_rad",
    "action_delta_l2",
    "tracking_velocity_distance_mps",
    "tracking_acceleration_distance_mps2",
)
TEMPORAL_ROLLOUT_METRICS = {
    "action_delta_l2",
    "tracking_acceleration_distance_mps2",
}
BACKBONE_KEYS = (
    "planner_type",
    "state_dim",
    "d_model",
    "num_layers",
    "num_heads",
    "feedforward_dim",
    "patch_dim",
    "num_state_tokens",
    "language_dim",
    "num_language_tokens",
    "dropout",
)
TRAINING_KEYS = (
    "batch_size",
    "micro_batch_size",
    "lr",
    "weight_decay",
    "flow_num_inference_steps",
    "flow_inference_noise_std",
    "endpoint_num_inference_steps",
    "model_size",
)
FOCUSED_PROTOCOL = {
    "reset_schedule": "sequential",
    "wrap_steps": False,
    "policy_observation_corruption_enabled": False,
    "early_terminations_enabled": True,
    "time_out_enabled": True,
    "episode_length_extension_enabled": True,
    "reward_clipping_enabled": False,
}
VANILLA_POLICY_INPUT_KEYS = [
    ["policy", "expert_motion"],
    ["policy", "expert_anchor_pos_b"],
    ["policy", "expert_anchor_ori_b"],
    ["policy", "base_ang_vel"],
    ["policy", "joint_pos_rel"],
    ["policy", "joint_vel_rel"],
    ["policy", "last_action"],
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("latent", "full_body"):
        parser.add_argument(f"--{prefix}_checkpoint", type=Path, required=True)
        parser.add_argument(f"--{prefix}_merge_manifest", type=Path, required=True)
        parser.add_argument(
            f"--{prefix}_pretrained_summary", type=Path, required=True
        )
        parser.add_argument(f"--{prefix}_summary", type=Path, required=True)
    parser.add_argument("--direct_vanilla_summary", type=Path, required=True)
    parser.add_argument("--latent_oracle_summary", type=Path, required=True)
    parser.add_argument("--full_body_oracle_summary", type=Path, required=True)
    parser.add_argument("--streamed_equivalence", type=Path, required=True)
    parser.add_argument("--expected_seed", type=int, required=True)
    parser.add_argument("--expected_num_envs", type=int, required=True)
    parser.add_argument("--expected_history_steps", type=int, required=True)
    parser.add_argument("--expected_horizon_steps", type=int, required=True)
    parser.add_argument("--expected_full_body_future_steps", type=int, required=True)
    parser.add_argument("--expected_planner_interval", type=int, required=True)
    parser.add_argument("--expected_pretrain_updates", type=int, required=True)
    parser.add_argument("--expected_finetune_updates", type=int, required=True)
    parser.add_argument("--expected_rows_per_stage", type=int, required=True)
    parser.add_argument("--expected_eval_steps", type=int, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _path_identity(value: object) -> str:
    text = str(value or "").strip()
    return str(Path(text).expanduser().resolve()) if text else ""


def _as_int(value: object, default: int = -1) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)


def _planner_paths(
    args: argparse.Namespace, interface: str
) -> tuple[Path, Path, Path, Path]:
    prefix = "latent" if interface == "latent_skill" else "full_body"
    return (
        getattr(args, f"{prefix}_checkpoint").expanduser().resolve(),
        getattr(args, f"{prefix}_merge_manifest").expanduser().resolve(),
        getattr(args, f"{prefix}_pretrained_summary").expanduser().resolve(),
        getattr(args, f"{prefix}_summary").expanduser().resolve(),
    )


def _audit_focused_protocol(
    value: object,
    *,
    scope: str,
    errors: list[str],
) -> None:
    metadata = _mapping(value)
    for key, expected in FOCUSED_PROTOCOL.items():
        if metadata.get(key) != expected:
            errors.append(
                f"{scope}: {key}={metadata.get(key)!r} does not match "
                f"focused protocol {expected!r}"
            )


def _tracker_record(
    value: object,
    *,
    scope: str,
    errors: list[str],
) -> dict[str, Any]:
    tracker = _mapping(value)

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{scope}: {message}")

    require(bool(tracker), "missing frozen-tracker provenance")
    require(
        tracker.get("loaded_components") == ["policy_state_dict"],
        "tracker did not load only policy_state_dict",
    )
    require(tracker.get("strict_policy_restore") is True, "restore was not strict")
    require(tracker.get("policy_frozen") is True, "policy was not frozen")
    require(tracker.get("policy_training") is False, "policy was not in eval mode")
    require(
        tracker.get("policy_input_keys") == VANILLA_POLICY_INPUT_KEYS,
        "ordered vanilla policy input contract mismatch",
    )
    require(
        _as_int(tracker.get("policy_parameter_count")) > 0,
        "tracker parameter count is not positive",
    )
    require(
        _as_int(tracker.get("policy_trainable_parameter_count")) == 0,
        "tracker has trainable parameters",
    )
    checksum = tracker.get("checkpoint_sha256")
    require(
        isinstance(checksum, str) and len(checksum) == 64,
        "missing checkpoint SHA-256",
    )
    return tracker


def _audit_closed_loop_outcome(
    summary: dict[str, Any],
    *,
    scope: str,
    expected_steps: int,
    expected_num_envs: int,
    require_planner_latency: bool,
    errors: list[str],
) -> dict[str, Any]:
    metadata = _mapping(summary.get("metadata"))
    aggregate = _mapping(summary.get("aggregate"))
    metrics = _mapping(summary.get("metrics"))

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{scope}: {message}")

    steps_run = _as_int(summary.get("steps_run"))
    require(0 < steps_run <= expected_steps, "evaluation length is invalid")
    require(
        summary.get("stop_reason")
        == ("max_steps" if steps_run == expected_steps else "all_envs_done"),
        "stop reason does not explain the evaluation length",
    )
    per_environment = summary.get("per_environment", [])
    require(
        isinstance(per_environment, list)
        and len(per_environment) == expected_num_envs,
        "per-environment outcomes are missing",
    )
    successes = [
        bool(item.get("tracking_success"))
        for item in per_environment
        if isinstance(item, Mapping)
    ]
    require(
        bool(successes)
        and math.isclose(
            float(aggregate.get("tracking_success_rate", math.nan)),
            sum(successes) / len(successes),
            abs_tol=1.0e-6,
        ),
        "aggregate strict success differs from per-environment outcomes",
    )
    for item in per_environment:
        if not isinstance(item, Mapping):
            continue
        failure_terms = set(item.get("termination_terms", []))
        failure_terms.difference_update({"time_out", "reference_finished"})
        require(
            bool(item.get("tracking_success")) == (not failure_terms),
            "strict success differs from termination causes",
        )
    termination_counts = aggregate.get("termination_cause_env_counts")
    require(
        isinstance(termination_counts, Mapping)
        and "reference_finished" in termination_counts,
        "termination-cause counts are missing",
    )
    for metric_name in PAPER_ROLLOUT_METRICS:
        metric = _mapping(metrics.get(metric_name))
        # A rollout in which every environment fails on its first control step
        # has no action delta or finite-difference acceleration by definition.
        # Keep that early failure as a valid result; all instantaneous metrics
        # must still be present, and temporal metrics are required as soon as a
        # second control step was observed.
        temporal_metric_unavailable = (
            metric_name in TEMPORAL_ROLLOUT_METRICS and steps_run == 1
        )
        require(
            temporal_metric_unavailable
            or (
                math.isfinite(float(metric.get("mean", math.nan)))
                and _as_int(metric.get("count"), 0) > 0
            ),
            f"required metric {metric_name} is missing",
        )
    push = _mapping(metadata.get("push_perturbation"))
    require(push.get("enabled") is True, "push perturbation metadata is missing")
    if require_planner_latency:
        latency = _mapping(summary.get("planner_inference_latency_ms"))
        require(
            latency.get("scope") == "high_level_planner_forward_only"
            and latency.get("unit") == "ms",
            "planner-only latency metadata is missing",
        )
        total_calls = _as_int(latency.get("total_call_count"), 0)
        measured_calls = _as_int(latency.get("measured_call_count"), 0)
        require(total_calls > 0, "planner latency observed no publication call")
        require(
            _as_int(latency.get("warmup_calls_excluded")) == 1,
            "planner latency warmup policy mismatch",
        )
        require(
            (total_calls == 1 and measured_calls == 0)
            or (
                measured_calls > 0
                and math.isfinite(float(latency.get("mean", math.nan)))
            ),
            "planner latency has no post-warmup measurement",
        )
    return push


def _audit_planner_row(
    args: argparse.Namespace,
    interface: str,
    errors: list[str],
) -> dict[str, Any]:
    checkpoint_path, merge_path, pretrained_summary_path, summary_path = _planner_paths(
        args, interface
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Planner checkpoint is not a mapping: {checkpoint_path}")
    metadata = _mapping(checkpoint.get("metadata"))
    sample_metadata = _mapping(metadata.get("sample_metadata"))
    observation_spec = _mapping(sample_metadata.get("planner_observation_spec"))
    planner_config = _mapping(checkpoint.get("planner_config"))
    target_spec = _mapping(checkpoint.get("target_spec"))
    pretrain_metadata = _mapping(metadata.get("checkpoint_metadata"))
    merge = _load_json(merge_path)
    pretrained_summary = _load_json(pretrained_summary_path)
    summary = _load_json(summary_path)
    summary_metadata = _mapping(summary.get("metadata"))
    pretrained_summary_metadata = _mapping(pretrained_summary.get("metadata"))
    sources = merge.get("sources")
    if not isinstance(sources, list):
        sources = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"{interface}: {message}")

    require(target_spec.get("interface") == interface, "target interface mismatch")
    require(
        sample_metadata.get("sample_format")
        == {"name": PLANNER_SAMPLE_FORMAT, "version": PLANNER_SAMPLE_VERSION},
        f"sample format is not {PLANNER_SAMPLE_FORMAT} v{PLANNER_SAMPLE_VERSION}",
    )
    require(
        sample_metadata.get("paired_target_contract") == PAIRED_TARGET_CONTRACT,
        "sample metadata does not enforce paired causal and demonstration targets",
    )
    require(
        _as_int(sample_metadata.get("state_history_steps"))
        == args.expected_history_steps,
        "sample history length mismatch",
    )
    require(
        _as_int(observation_spec.get("history_steps")) == args.expected_history_steps,
        "observation history length mismatch",
    )
    require(_as_int(observation_spec.get("frame_dim")) == 93, "frame width is not 93")
    require(
        _as_int(observation_spec.get("history_frames"))
        == args.expected_history_steps + 1,
        "planner input does not contain nine past frames plus current",
    )
    expected_flat_dim = (args.expected_history_steps + 1) * 93
    require(
        _as_int(observation_spec.get("flat_dim")) == expected_flat_dim,
        "flattened causal planner width mismatch",
    )
    require(
        _as_int(planner_config.get("state_dim")) == expected_flat_dim,
        "planner state_dim does not match the 10x93 causal input",
    )
    require(
        observation_spec.get("feature_names")
        == [
            "joint_pos_rel",
            "joint_vel_rel",
            "base_ang_vel",
            "projected_gravity",
            "last_action",
        ],
        "causal planner feature order mismatch",
    )
    require(
        observation_spec.get("feature_widths") == [29, 29, 3, 3, 29],
        "causal planner feature widths mismatch",
    )
    require(
        observation_spec.get("reference_features") == [],
        "planner observation contains reference features",
    )
    _audit_focused_protocol(
        sample_metadata,
        scope=f"{interface}_samples",
        errors=errors,
    )
    _audit_focused_protocol(
        summary_metadata,
        scope=f"{interface}_evaluation",
        errors=errors,
    )
    _audit_focused_protocol(
        pretrained_summary_metadata,
        scope=f"{interface}_pretrained_evaluation",
        errors=errors,
    )
    require(
        _as_int(sample_metadata.get("command_past_steps")) == 0,
        "command history must be zero",
    )
    expected_future_steps = (
        args.expected_horizon_steps
        if interface == "latent_skill"
        else args.expected_full_body_future_steps
    )
    require(
        _as_int(sample_metadata.get("command_future_steps")) == expected_future_steps,
        "command horizon mismatch",
    )
    require(
        _as_int(sample_metadata.get("planner_interval_steps"))
        == args.expected_planner_interval,
        "planner interval mismatch",
    )
    control_rate_hz = float(sample_metadata.get("control_rate_hz", -1.0))
    planner_rate_hz = float(sample_metadata.get("planner_rate_hz", -1.0))
    require(math.isclose(control_rate_hz, 50.0), "control rate is not 50 Hz")
    require(
        math.isclose(
            planner_rate_hz,
            50.0 / float(args.expected_planner_interval),
        ),
        "planner rate does not match the interval",
    )
    require(
        _as_int(sample_metadata.get("seed")) == args.expected_seed,
        "sample seed mismatch",
    )
    require(
        _as_int(metadata.get("selection_seed")) == args.expected_seed,
        "finetune selection seed mismatch",
    )
    require(
        _as_int(pretrain_metadata.get("selection_seed")) == args.expected_seed,
        "pretrain selection seed mismatch",
    )
    require(
        _as_int(metadata.get("pretrain_num_updates")) == args.expected_pretrain_updates,
        "pretrain update budget mismatch",
    )
    require(
        _as_int(metadata.get("finetune_num_updates")) == args.expected_finetune_updates,
        "finetune update budget mismatch",
    )
    require(
        _as_int(metadata.get("parameter_count")) > 0,
        "planner parameter count is not positive",
    )
    require(
        _as_int(pretrain_metadata.get("selected_sample_count"))
        == args.expected_rows_per_stage,
        "pretrain sample budget mismatch",
    )
    require(
        _as_int(metadata.get("selected_sample_count"))
        == 2 * args.expected_rows_per_stage,
        "finetune sample budget mismatch",
    )
    require(len(sources) == 2, "merge must contain exactly two stages")
    require(
        [
            source.get("collection_stage")
            for source in sources
            if isinstance(source, Mapping)
        ]
        == ["oracle_rollout", "planner_rollout"],
        "merge stages must be oracle_rollout then planner_rollout",
    )
    require(
        [
            _as_int(source.get("selected_row_count"))
            for source in sources
            if isinstance(source, Mapping)
        ]
        == [args.expected_rows_per_stage] * 2,
        "selected rows per stage mismatch",
    )
    require(
        _as_int(merge.get("row_count")) == 2 * args.expected_rows_per_stage,
        "merged row count mismatch",
    )
    require(
        _as_int(summary.get("max_steps")) == args.expected_eval_steps,
        "evaluation step budget mismatch",
    )
    require(
        summary_metadata.get("interface") == interface,
        "closed-loop summary interface mismatch",
    )
    require(
        _as_int(summary_metadata.get("seed")) == args.expected_seed,
        "closed-loop summary seed mismatch",
    )
    require(
        _as_int(summary_metadata.get("num_envs")) == args.expected_num_envs,
        "closed-loop num_envs mismatch",
    )
    require(
        _as_int(pretrained_summary_metadata.get("num_envs"))
        == args.expected_num_envs,
        "pretrained closed-loop num_envs mismatch",
    )
    final_push = _audit_closed_loop_outcome(
        summary,
        scope=f"{interface}_finetuned",
        expected_steps=args.expected_eval_steps,
        expected_num_envs=args.expected_num_envs,
        require_planner_latency=True,
        errors=errors,
    )
    pretrained_push = _audit_closed_loop_outcome(
        pretrained_summary,
        scope=f"{interface}_pretrained",
        expected_steps=args.expected_eval_steps,
        expected_num_envs=args.expected_num_envs,
        require_planner_latency=True,
        errors=errors,
    )
    require(
        pretrained_push == final_push,
        "push perturbation changed between planner training stages",
    )

    return {
        "checkpoint": str(checkpoint_path),
        "merge_manifest": str(merge_path),
        "summary": str(summary_path),
        "pretrained_summary": str(pretrained_summary_path),
        "planner_config": planner_config,
        "training_config": {key: metadata.get(key) for key in TRAINING_KEYS},
        "parameter_count": metadata.get("parameter_count"),
        "target_spec": target_spec,
        "sample_metadata": sample_metadata,
        "summary_metadata": summary_metadata,
        "pretrained_summary_metadata": pretrained_summary_metadata,
        "push_perturbation": final_push,
        "merge_rows": merge.get("row_count"),
    }


def main() -> None:
    args = _parse_args()
    errors: list[str] = []
    rows = {
        interface: _audit_planner_row(args, interface, errors)
        for interface in PLANNER_INTERFACES
    }

    latent = rows["latent_skill"]
    full_body = rows["full_body_trajectory"]
    for key in BACKBONE_KEYS:
        if full_body["planner_config"].get(key) != latent["planner_config"].get(key):
            errors.append(
                "full_body_trajectory: planner backbone "
                f"{key}={full_body['planner_config'].get(key)!r} does not match "
                f"latent_skill {latent['planner_config'].get(key)!r}"
            )

    oracle_rows: dict[str, dict[str, Any]] = {}
    for interface, raw_path in (
        ("latent_skill", args.latent_oracle_summary),
        ("full_body_trajectory", args.full_body_oracle_summary),
    ):
        oracle_path = raw_path.expanduser().resolve()
        oracle_summary = _load_json(oracle_path)
        oracle_push = _audit_closed_loop_outcome(
            oracle_summary,
            scope=f"{interface}_oracle",
            expected_steps=args.expected_eval_steps,
            expected_num_envs=args.expected_num_envs,
            require_planner_latency=False,
            errors=errors,
        )
        oracle_rows[interface] = {
            "summary": str(oracle_path),
            "aggregate": _mapping(oracle_summary.get("aggregate")),
            "metrics": _mapping(oracle_summary.get("metrics")),
            "metadata": _mapping(oracle_summary.get("metadata")),
            "push_perturbation": oracle_push,
        }
        if oracle_push != rows[interface]["push_perturbation"]:
            errors.append(
                f"{interface}: push perturbation differs between oracle and planner"
            )
    for key in TRAINING_KEYS:
        if full_body["training_config"].get(key) != latent["training_config"].get(key):
            errors.append(
                "full_body_trajectory: planner training setting "
                f"{key}={full_body['training_config'].get(key)!r} does not match "
                f"latent_skill {latent['training_config'].get(key)!r}"
            )

    latent_train_manifest = _path_identity(
        _mapping(latent["sample_metadata"].get("provenance")).get("motion_manifest")
    )
    full_train_manifest = _path_identity(
        _mapping(full_body["sample_metadata"].get("provenance")).get("motion_manifest")
    )
    if not latent_train_manifest or latent_train_manifest != full_train_manifest:
        errors.append(
            "focused_protocol: latent and full-body planner samples use different "
            "training manifests"
        )
    latent_eval_manifest = _path_identity(
        latent["summary_metadata"].get("motion_manifest")
    )
    full_eval_manifest = _path_identity(
        full_body["summary_metadata"].get("motion_manifest")
    )
    if not latent_eval_manifest or latent_eval_manifest != full_eval_manifest:
        errors.append(
            "focused_protocol: latent and full-body rows use different evaluation "
            "manifests"
        )

    full_metadata = full_body["summary_metadata"]
    if full_metadata.get("low_level_command_mode") != "streamed_vanilla":
        errors.append("full_body_trajectory: low-level mode is not streamed_vanilla")
    if full_metadata.get("low_level_command_space") != "single_frame_full_body":
        errors.append("full_body_trajectory: low-level command space is not vanilla")
    if full_metadata.get("policy_command_mode") != "full_body_chunk_current_slot":
        errors.append(
            "full_body_trajectory: policy does not consume the current chunk slot"
        )
    if _as_int(full_metadata.get("command_past_steps")) != 0:
        errors.append("full_body_trajectory: command history must be zero")
    if (
        _as_int(full_metadata.get("command_future_steps"))
        != args.expected_full_body_future_steps
    ):
        errors.append("full_body_trajectory: deployed command horizon mismatch")
    if (
        _as_int(full_metadata.get("planner_update_interval"))
        != args.expected_planner_interval
    ):
        errors.append("full_body_trajectory: deployed planner interval mismatch")
    full_tracker = _tracker_record(
        full_metadata.get("low_level_tracker"),
        scope="full_body_trajectory",
        errors=errors,
    )
    full_sample_metadata = full_body["sample_metadata"]
    if full_sample_metadata.get("low_level_command_mode") != "streamed_vanilla":
        errors.append(
            "full_body_trajectory: training samples were not streamed_vanilla"
        )
    if full_sample_metadata.get("low_level_command_space") != "single_frame_full_body":
        errors.append(
            "full_body_trajectory: training samples did not use vanilla tracker inputs"
        )
    if (
        full_sample_metadata.get("policy_command_mode")
        != "full_body_chunk_current_slot"
    ):
        errors.append(
            "full_body_trajectory: training samples did not use current-slot adapter"
        )
    sample_tracker = _tracker_record(
        full_sample_metadata.get("low_level_tracker"),
        scope="full_body_trajectory_samples",
        errors=errors,
    )
    if sample_tracker != full_tracker:
        errors.append(
            "full_body_trajectory: planner samples and evaluation do not have "
            "identical frozen-tracker provenance"
        )
    full_target = full_body["target_spec"]
    expected_term_widths = [
        58 * args.expected_horizon_steps,
        3 * args.expected_horizon_steps,
        6 * args.expected_horizon_steps,
    ]
    if full_target.get("term_names") != [
        "expert_motion",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    ]:
        errors.append("full_body_trajectory: target terms are not the vanilla command")
    if full_target.get("term_widths") != expected_term_widths:
        errors.append(
            "full_body_trajectory: target is not exactly ten 67-D command frames"
        )
    if _as_int(full_target.get("target_dim")) != 67 * args.expected_horizon_steps:
        errors.append("full_body_trajectory: target width is not 670")

    direct_path = args.direct_vanilla_summary.expanduser().resolve()
    direct_summary = _load_json(direct_path)
    direct_metadata = _mapping(direct_summary.get("metadata"))
    direct_push = _audit_closed_loop_outcome(
        direct_summary,
        scope="direct_vanilla_ceiling",
        expected_steps=args.expected_eval_steps,
        expected_num_envs=args.expected_num_envs,
        require_planner_latency=False,
        errors=errors,
    )

    def require_direct(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"direct_vanilla_ceiling: {message}")

    require_direct(
        direct_metadata.get("command_space") == "single_frame_full_body",
        "command space mismatch",
    )
    require_direct(
        direct_metadata.get("low_level_command_mode") == "native",
        "must use native single-frame commands",
    )
    require_direct(
        direct_metadata.get("low_level_command_space") == "single_frame_full_body",
        "low-level command space mismatch",
    )
    require_direct(
        direct_metadata.get("policy_command_mode") == "reference",
        "policy command mode is not direct reference",
    )
    require_direct(
        direct_metadata.get("policy_only_checkpoint") is True,
        "did not use the frozen policy-only loader",
    )
    require_direct(
        direct_metadata.get("command_observation_source") == "reference",
        "command source is not reference",
    )
    require_direct(
        _as_int(direct_metadata.get("command_past_steps")) == 0,
        "command history must be zero",
    )
    require_direct(
        _as_int(direct_metadata.get("command_future_steps")) == 0,
        "direct ceiling must not receive a future window",
    )
    require_direct(
        direct_metadata.get("planner_mode") == "none",
        "direct ceiling must not run a planner",
    )
    require_direct(
        _as_int(direct_metadata.get("planner_update_interval")) == 1,
        "direct ceiling must receive commands at 50 Hz",
    )
    require_direct(
        _as_int(direct_metadata.get("seed")) == args.expected_seed,
        "seed mismatch",
    )
    require_direct(
        _as_int(direct_metadata.get("num_envs")) == args.expected_num_envs,
        "num_envs mismatch",
    )
    require_direct(
        _as_int(direct_metadata.get("steps_requested")) == args.expected_eval_steps,
        "evaluation step budget mismatch",
    )
    _audit_focused_protocol(
        direct_metadata,
        scope="direct_vanilla_ceiling",
        errors=errors,
    )
    direct_tracker = _tracker_record(
        direct_metadata.get("low_level_tracker"),
        scope="direct_vanilla_ceiling",
        errors=errors,
    )
    require_direct(
        full_tracker == direct_tracker,
        "frozen-tracker provenance differs from the streamed full-body tracker",
    )
    require_direct(
        full_metadata.get("checkpoint") == direct_metadata.get("checkpoint"),
        "checkpoint path differs from the streamed full-body tracker",
    )
    require_direct(
        full_metadata.get("motion_manifest") == direct_metadata.get("motion_manifest"),
        "evaluation manifest differs from the streamed full-body row",
    )
    require_direct(
        direct_push == full_body["push_perturbation"]
        and direct_push == latent["push_perturbation"],
        "push perturbation differs across planner rows and direct ceiling",
    )

    equivalence_path = args.streamed_equivalence.expanduser().resolve()
    equivalence = _load_json(equivalence_path)

    def require_equivalence(condition: bool, message: str) -> None:
        if not condition:
            errors.append(f"streamed_equivalence: {message}")

    equivalence_tracker = _tracker_record(
        equivalence.get("low_level_tracker"),
        scope="streamed_equivalence",
        errors=errors,
    )
    require_equivalence(equivalence.get("passed") is True, "certificate did not pass")
    require_equivalence(
        _as_int(equivalence.get("hold_steps")) == args.expected_planner_interval,
        "hold length mismatch",
    )
    require_equivalence(
        _as_int(equivalence.get("window_steps")) == args.expected_horizon_steps,
        "command-frame count mismatch",
    )
    require_equivalence(
        equivalence.get("observed_phases")
        == list(range(args.expected_planner_interval)),
        "not every held-command phase was observed",
    )
    require_equivalence(
        equivalence.get("missing_phases") == [],
        "certificate reports missing phases",
    )
    require_equivalence(
        equivalence.get("asynchronous_rephase_required") is True
        and equivalence.get("asynchronous_rephase_exercised") is True,
        "asynchronous row republication was not exercised",
    )
    require_equivalence(
        equivalence.get("policy_state_unchanged") is True,
        "frozen actor state changed",
    )
    tolerance = float(equivalence.get("atol", -1.0))
    for key in (
        "max_all_policy_input_abs",
        "max_command_abs",
        "max_action_abs",
    ):
        value = float(equivalence.get(key, math.inf))
        require_equivalence(
            tolerance >= 0.0 and math.isfinite(value) and value <= tolerance,
            f"{key} exceeds tolerance",
        )
    require_equivalence(
        equivalence_tracker == full_tracker,
        "certificate did not use identical frozen-tracker provenance",
    )
    latent_episode_length = float(
        latent["summary_metadata"].get("episode_length_s", math.nan)
    )
    full_episode_length = float(full_metadata.get("episode_length_s", math.nan))
    direct_episode_length = float(direct_metadata.get("episode_length_s", math.nan))
    if not (
        math.isfinite(latent_episode_length)
        and math.isclose(latent_episode_length, full_episode_length)
        and math.isclose(full_episode_length, direct_episode_length)
    ):
        errors.append(
            "focused_protocol: final evaluation episode lengths differ across "
            "latent, streamed full-body, and direct ceiling"
        )

    payload = {
        "passed": not errors,
        "errors": errors,
        "scope": {
            "planner_rows": list(PLANNER_INTERFACES),
            "ceiling": "direct_vanilla_50hz",
            "ceiling_is_planner_row": False,
        },
        "expected": {
            "seed": args.expected_seed,
            "num_envs": args.expected_num_envs,
            "history_steps": args.expected_history_steps,
            "horizon_steps": args.expected_horizon_steps,
            "full_body_future_steps": args.expected_full_body_future_steps,
            "planner_interval_steps": args.expected_planner_interval,
            "pretrain_updates": args.expected_pretrain_updates,
            "finetune_updates": args.expected_finetune_updates,
            "rows_per_stage": args.expected_rows_per_stage,
            "eval_steps": args.expected_eval_steps,
        },
        "planner_rows": rows,
        "oracle_rows": oracle_rows,
        "ceiling": {
            "summary": str(direct_path),
            "metadata": direct_metadata,
            "push_perturbation": direct_push,
        },
        "streamed_equivalence": {
            "summary": str(equivalence_path),
            "result": equivalence,
        },
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("Focused comparison audit failed:\n- " + "\n- ".join(errors))
    print(f"[PASS] Focused causal-interface audit: {output}")


if __name__ == "__main__":
    main()
