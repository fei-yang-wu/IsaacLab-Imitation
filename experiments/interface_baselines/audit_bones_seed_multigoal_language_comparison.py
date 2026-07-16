#!/usr/bin/env python3
"""Audit a shared multi-goal BONES-SEED language interface comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import load_rollout_samples  # noqa: E402
from planner_sample_schema import (  # noqa: E402
    PAIRED_TARGET_CONTRACT,
    PLANNER_SAMPLE_FORMAT,
    PLANNER_SAMPLE_VERSION,
)


PROTOCOL = "bones_seed_shared_multigoal_language_v1"
INTERFACE_TARGET_DIMS = {"latent_skill": 256, "full_body_trajectory": 670}
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


def _require_paper_rollout_metrics(
    require: Any,
    summary: dict[str, Any],
    *,
    label: str,
) -> None:
    steps_run = int(summary.get("steps_run", -1))
    metric_payload = summary.get("metrics", {})
    for metric_name in PAPER_ROLLOUT_METRICS:
        metric = metric_payload.get(metric_name, {})
        metric_mean = float(metric.get("mean", math.nan))
        temporal_metric_unavailable = (
            metric_name in TEMPORAL_ROLLOUT_METRICS and steps_run == 1
        )
        require(
            temporal_metric_unavailable
            or (math.isfinite(metric_mean) and int(metric.get("count", 0)) > 0),
            f"{label}: required metric {metric_name} is missing",
        )


def _require_planner_latency(
    require: Any,
    summary: dict[str, Any],
    *,
    label: str,
    require_measurement: bool,
) -> None:
    latency = summary.get("planner_inference_latency_ms", {})
    require(
        isinstance(latency, dict)
        and latency.get("unit") == "ms"
        and latency.get("scope") == "high_level_planner_forward_only",
        f"{label}: planner-only latency metadata is missing",
    )
    if not isinstance(latency, dict):
        return
    require(
        int(latency.get("warmup_calls_excluded", -1)) == 1,
        f"{label}: planner latency warmup policy mismatch",
    )
    if require_measurement:
        total_calls = int(latency.get("total_call_count", 0))
        measured_calls = int(latency.get("measured_call_count", 0))
        honest_early_failure = (
            summary.get("stop_reason") == "all_envs_done"
            and int(summary.get("steps_run", -1)) > 0
        )
        require(
            total_calls > 0,
            f"{label}: planner latency did not observe a publication call",
        )
        require(
            (total_calls == 1 and measured_calls == 0 and honest_early_failure)
            or (
                measured_calls > 0
                and math.isfinite(float(latency.get("mean", math.nan)))
            ),
            f"{label}: planner latency has no measured calls",
        )
    else:
        require(
            int(latency.get("total_call_count", 0)) > 0,
            f"{label}: planner latency did not observe a publication call",
        )


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a checkpoint mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _load_motion_names(samples_dir: Path) -> list[str]:
    names: list[str] = []
    for sample_path in sorted(samples_dir.glob("sample_step_*.pt")):
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        target = sample.get("causal_target")
        if not isinstance(target, torch.Tensor) or target.ndim == 0:
            raise ValueError(f"Sample has no row-shaped causal target: {sample_path}")
        row_count = int(target.shape[0])
        raw_names = sample.get("motion_name")
        if isinstance(raw_names, str):
            raw_names = [raw_names] * row_count
        if not isinstance(raw_names, list) or len(raw_names) != row_count:
            raise ValueError(
                f"Sample motion_name rows do not match its target: {sample_path}"
            )
        names.extend(str(value) for value in raw_names)
    return names


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    output = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else run_root / "protocol_checks" / "multigoal_language_audit.json"
    )
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    run_config = _json(run_root / "run_config.json")
    comparison = _json(run_root / "comparison_manifest.json")
    preflight = _json(run_root / "protocol_checks" / "bones_seed_preflight.json")
    require(run_config.get("protocol") == PROTOCOL, "run protocol mismatch")
    require(comparison.get("protocol") == PROTOCOL, "comparison protocol mismatch")
    require(preflight.get("passed") is True, "BONES-SEED preflight did not pass")
    require(not comparison.get("failures"), "comparison recorded command failures")
    smoke_only = bool(run_config.get("skip_pretrained_closed_loop", False))
    gate_metadata = run_config.get("submission_gates", {})
    submission_gates_complete = bool(
        gate_metadata.get("fresh_preparation_checked")
        and gate_metadata.get("low_level_gates_checked")
    )
    expected_paper_status = bool(not smoke_only and submission_gates_complete)
    require(
        comparison.get("skip_pretrained_closed_loop") == smoke_only,
        "pretrained evaluation setting differs between run records",
    )
    require(
        comparison.get("pretrained_closed_loop_complete") == (not smoke_only)
        and run_config.get("pretrained_closed_loop_complete") == (not smoke_only),
        "pretrained closed-loop status is inconsistent",
    )
    rollout_collection = run_config.get("planner_rollout_collection", {})
    require(
        rollout_collection.get("mode") == "parallel_identical_goal"
        and int(rollout_collection.get("num_envs", 0)) > 0
        and rollout_collection.get("goal_source") == "explicit_language_argument"
        and rollout_collection.get("reference_scope") == "same_named_motion_only",
        "planner-rollout parallel collection contract is invalid",
    )
    require(
        comparison.get("planner_rollout_collection")
        == {
            "mode": rollout_collection.get("mode"),
            "num_envs": rollout_collection.get("num_envs"),
            "goal_source": rollout_collection.get("goal_source"),
            "reference_scope": rollout_collection.get("reference_scope"),
        },
        "planner-rollout collection differs between run records",
    )
    require(
        comparison.get("submission_gates_complete") == submission_gates_complete,
        "submission gate status is inconsistent",
    )
    require(
        comparison.get("paper_protocol_complete") == expected_paper_status
        and run_config.get("paper_protocol_complete") == expected_paper_status,
        "paper protocol status is inconsistent",
    )
    cluster_submission_path = run_root / "cluster_submission.json"
    cluster_submission: dict[str, Any] | None = None
    if cluster_submission_path.is_file():
        cluster_submission = _json(cluster_submission_path)
        jobs = cluster_submission.get("jobs", {})
        job_ids: list[int] = []
        if isinstance(jobs, dict):
            try:
                job_ids = [int(value) for value in jobs.values()]
            except (TypeError, ValueError):
                job_ids = []
        require(
            cluster_submission.get("schema_version") == 1,
            "cluster submission schema mismatch",
        )
        require(
            int(cluster_submission.get("seed", -1)) == int(run_config.get("seed", -2)),
            "cluster submission seed mismatch",
        )
        require(
            int(cluster_submission.get("goal_limit", -1))
            == int(run_config.get("goal_count", -2)),
            "cluster submission goal count mismatch",
        )
        require(
            re.fullmatch(
                rf"0-{len(run_config.get('goals', [])) - 1}(%[1-9][0-9]*)?",
                str(cluster_submission.get("pipeline_array", "")),
            )
            is not None,
            "cluster submission array does not cover every goal",
        )
        require(
            _as_path(cluster_submission.get("output_root")) == run_root,
            "cluster submission output root mismatch",
        )
        require(
            bool(str(cluster_submission.get("cluster_workspace", "")).strip()),
            "cluster submission workspace is missing",
        )
        require(
            bool(str(cluster_submission.get("submitted_at_utc", "")).strip()),
            "cluster submission timestamp is missing",
        )
        require(
            isinstance(jobs, dict)
            and set(jobs)
            == {
                "prepare",
                "rollout_array",
                "finetune",
                "final_eval_array",
                "summarize",
            }
            and len(job_ids) == 5
            and len(set(job_ids)) == 5
            and all(value > 0 for value in job_ids),
            "cluster submission job IDs are incomplete or invalid",
        )
        for key in ("workspace_archive_sha256", "repo_sync_manifest_sha256"):
            value = str(cluster_submission.get(key, ""))
            require(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"cluster submission {key} is not a SHA-256",
            )
        recorded = comparison.get("cluster_submission")
        require(
            isinstance(recorded, dict)
            and recorded.get("sha256") == _sha256(cluster_submission_path),
            "comparison manifest does not bind the cluster submission record",
        )
    require(
        not expected_paper_status or cluster_submission is not None,
        "paper run is missing the cluster submission record",
    )

    goals = [str(value) for value in run_config.get("goals", [])]
    require(bool(goals), "run has no selected goals")
    require(len(goals) == len(set(goals)), "selected goals are not unique")
    require(comparison.get("goals") == goals, "comparison goals differ from run config")
    require(run_config.get("goal_count") == len(goals), "run goal count mismatch")
    require(
        comparison.get("goal_count") == len(goals), "comparison goal count mismatch"
    )
    run_paths = run_config.get("paths", {})
    expected_dataset_paths = {
        "full_body_trajectory": _as_path(run_paths.get("vanilla_dataset_path")),
        "latent_skill": _as_path(run_paths.get("latent_dataset_path")),
    }
    for interface, dataset_path in expected_dataset_paths.items():
        require(dataset_path.is_dir(), f"{interface} dataset cache is missing")

    state_dim = int(run_config.get("causal_state_dim", -1))
    language_dim = int(run_config.get("language", {}).get("embedding_dim", -1))
    demo_rows = int(run_config.get("demo_rows_per_goal", -1))
    rollout_rows = int(run_config.get("rollout_rows_per_goal", -1))
    final_rows = demo_rows + rollout_rows
    expected_rows = {
        "demonstration_samples": demo_rows * len(goals),
        "planner_rollout_samples": rollout_rows * len(goals),
        "demonstration_and_rollout_samples": final_rows * len(goals),
    }
    demonstration_collection = run_config.get("demonstration_collection", {})
    require(
        demonstration_collection.get("mode") == "balanced_multi_environment",
        "demonstration collection is not the balanced multi-environment path",
    )
    require(
        comparison.get("demonstration_collection", {}).get("mode")
        == demonstration_collection.get("mode"),
        "demonstration collection mode differs between run records",
    )
    for interface, relative_summary in (
        ("full_body_trajectory", Path("demonstration_batched/full_body/summary.json")),
        ("latent_skill", Path("demonstration_batched/latent_skill/summary.json")),
    ):
        summary_path = run_root / relative_summary
        require(summary_path.is_file(), f"{interface} batched demonstration is missing")
        if not summary_path.is_file():
            continue
        demonstration_summary = _json(summary_path)
        demonstration_metadata = demonstration_summary.get("metadata", {})
        require(
            _as_path(demonstration_metadata.get("dataset_path"))
            == expected_dataset_paths[interface],
            f"{interface} batched demonstration used the wrong dataset cache",
        )
        balanced = demonstration_summary.get("balanced_collection", {})
        counts = balanced.get("counts", {})
        require(
            balanced.get("complete") is True,
            f"{interface} batched demonstration did not complete",
        )
        require(
            balanced.get("motion_names") == goals,
            f"{interface} batched demonstration goal order mismatch",
        )
        require(
            balanced.get("rows_per_motion") == demo_rows,
            f"{interface} batched demonstration row budget mismatch",
        )
        require(
            counts == {goal: demo_rows for goal in goals},
            f"{interface} batched demonstration counts are not balanced",
        )
        require(
            demonstration_summary.get("saved_rows") == demo_rows * len(goals),
            f"{interface} batched demonstration total row count mismatch",
        )

    language_path = _as_path(run_config.get("paths", {}).get("language_embeddings"))
    require(language_path.is_file(), "language embedding table is missing")
    language_sha = _sha256(language_path) if language_path.is_file() else ""
    language_table = (
        torch.load(language_path, map_location="cpu", weights_only=False)
        if language_path.is_file()
        else {}
    )
    table_names = [str(value) for value in language_table.get("names", [])]
    table_embeddings = language_table.get("embeddings")
    table_by_name: dict[str, torch.Tensor] = {}
    if isinstance(table_embeddings, torch.Tensor) and table_embeddings.ndim == 2:
        require(
            int(table_embeddings.shape[0]) == len(table_names),
            "language table names and embeddings differ in length",
        )
        table_by_name = {
            name: table_embeddings[index].detach().cpu()
            for index, name in enumerate(table_names)
        }
    else:
        require(False, "language table has no rank-2 embeddings")
    require(
        all(goal in table_by_name for goal in goals),
        "selected goal missing from language table",
    )
    artifacts = run_config.get("input_artifacts", {})
    required_artifact_names = (
        "manifest",
        "language_embeddings",
        "latent_low_level_checkpoint",
        "latent_skill_checkpoint",
        "vanilla_tracker_checkpoint",
    )
    require(
        all(name in artifacts for name in required_artifact_names),
        "required input artifact hashes are missing",
    )
    for artifact_name, artifact in artifacts.items():
        artifact_path = _as_path(artifact.get("path"))
        require(artifact_path.is_file(), f"input artifact is missing: {artifact_name}")
        if artifact_path.is_file():
            require(
                artifact.get("sha256") == _sha256(artifact_path),
                f"input artifact hash mismatch: {artifact_name}",
            )
    repository = run_config.get("repository", {})
    require(
        len(str(repository.get("commit", ""))) == 40,
        "repository commit was not recorded",
    )
    require(
        isinstance(repository.get("submodule_status"), list),
        "submodule revisions were not recorded",
    )

    checkpoints: dict[str, dict[str, dict[str, Any]]] = {
        "pretrain": {},
        "final": {},
    }
    checkpoint_hashes: list[str] = []
    sample_checks: dict[str, dict[str, Any]] = {}
    for interface, target_dim in INTERFACE_TARGET_DIMS.items():
        sample_checks[interface] = {}
        for stage, rows_per_goal in (
            ("demonstration_samples", demo_rows),
            ("planner_rollout_samples", rollout_rows),
            ("demonstration_and_rollout_samples", final_rows),
        ):
            samples_dir = run_root / interface / stage
            tensors, metadata = load_rollout_samples(samples_dir)
            row_count = int(tensors["causal_target"].shape[0])
            require(
                row_count == expected_rows[stage],
                f"{interface}/{stage}: row count mismatch",
            )
            require(
                int(tensors["planner_state"].shape[-1]) == state_dim,
                f"{interface}/{stage}: causal state width mismatch",
            )
            require(
                int(tensors["causal_target"].shape[-1]) == target_dim,
                f"{interface}/{stage}: target width mismatch",
            )
            language = tensors.get("language_embedding")
            require(
                language is not None, f"{interface}/{stage}: language embedding missing"
            )
            if isinstance(language, torch.Tensor):
                require(
                    int(language.shape[-1]) == language_dim,
                    f"{interface}/{stage}: language width mismatch",
                )
            motion_names = _load_motion_names(samples_dir)
            require(
                Counter(motion_names)
                == Counter({goal: rows_per_goal for goal in goals}),
                f"{interface}/{stage}: rows are not balanced by selected goal",
            )
            if isinstance(language, torch.Tensor) and len(motion_names) == row_count:
                for row, goal in enumerate(motion_names):
                    expected = table_by_name.get(goal)
                    require(
                        expected is not None,
                        f"{interface}/{stage}: unknown row goal {goal}",
                    )
                    if expected is not None:
                        require(
                            torch.equal(language[row].detach().cpu(), expected),
                            f"{interface}/{stage}: row {row} language vector does not match {goal}",
                        )
            require(
                metadata.get("sample_format")
                == {"name": PLANNER_SAMPLE_FORMAT, "version": PLANNER_SAMPLE_VERSION},
                f"{interface}/{stage}: planner sample schema mismatch",
            )
            require(
                metadata.get("paired_target_contract") == PAIRED_TARGET_CONTRACT,
                f"{interface}/{stage}: paired target contract mismatch",
            )
            require(
                _as_path(metadata.get("dataset_path"))
                == expected_dataset_paths[interface],
                f"{interface}/{stage}: planner samples used the wrong dataset cache",
            )
            observation = metadata.get("planner_observation_spec", {})
            require(
                observation.get("reference_features") == [],
                f"{interface}/{stage}: planner input contains reference features",
            )
            language_metadata = metadata.get("language_conditioning", {})
            require(
                language_metadata.get("enabled") is True,
                f"{interface}/{stage}: language disabled",
            )
            require(
                language_metadata.get("embedding_sha256") == language_sha,
                f"{interface}/{stage}: language table hash mismatch",
            )
            merge = _json(samples_dir / "merge_manifest.json")
            require(
                merge.get("row_count") == row_count,
                f"{interface}/{stage}: merge row count mismatch",
            )
            sample_checks[interface][stage] = {
                "rows": row_count,
                "rows_per_goal": rows_per_goal,
            }

        for checkpoint_stage, manifest_key in (
            ("pretrain", "pretrain_checkpoints"),
            ("final", "final_checkpoints"),
        ):
            checkpoint_path = _as_path(comparison.get(manifest_key, {}).get(interface))
            checkpoint = _checkpoint(checkpoint_path)
            checkpoints[checkpoint_stage][interface] = checkpoint
            config = checkpoint.get("planner_config", {})
            require(
                int(config.get("state_dim", -1)) == state_dim,
                f"{interface}/{checkpoint_stage}: state_dim mismatch",
            )
            require(
                int(config.get("target_dim", -1)) == target_dim,
                f"{interface}/{checkpoint_stage}: target_dim mismatch",
            )
            require(
                int(config.get("language_dim", -1)) == language_dim,
                f"{interface}/{checkpoint_stage}: language_dim mismatch",
            )
            metadata = checkpoint.get("metadata", {})
            expected_count = expected_rows[
                "demonstration_samples"
                if checkpoint_stage == "pretrain"
                else "demonstration_and_rollout_samples"
            ]
            require(
                int(metadata.get("num_samples", -1)) == expected_count,
                f"{interface}/{checkpoint_stage}: training sample count mismatch",
            )
            language_metadata = metadata.get("sample_metadata", {}).get(
                "language_conditioning", {}
            )
            checkpoint_hashes.append(str(language_metadata.get("embedding_sha256", "")))
            require(
                language_metadata.get("embedding_sha256") == language_sha,
                f"{interface}/{checkpoint_stage}: checkpoint language hash mismatch",
            )
            if checkpoint_stage == "final":
                expected_init = _as_path(
                    comparison.get("pretrain_checkpoints", {}).get(interface)
                )
                require(
                    _as_path(metadata.get("init_checkpoint")) == expected_init,
                    f"{interface}: final planner did not initialize from shared pretrain planner",
                )

    for checkpoint_stage, stage_checkpoints in checkpoints.items():
        configs = {
            interface: checkpoint.get("planner_config", {})
            for interface, checkpoint in stage_checkpoints.items()
        }
        for key in BACKBONE_KEYS:
            require(
                configs["latent_skill"].get(key)
                == configs["full_body_trajectory"].get(key),
                f"{checkpoint_stage}: planner backbone mismatch for {key}",
            )
    require(
        bool(checkpoint_hashes) and set(checkpoint_hashes) == {language_sha},
        "planner checkpoints do not share the same language table",
    )

    final_summary_checks: dict[str, list[dict[str, Any]]] = {}
    push_perturbation_protocols: list[dict[str, Any]] = []
    expected_seed = int(run_config.get("seed", -1))
    expected_steps = int(run_config.get("eval_steps", -1))
    for interface in INTERFACE_TARGET_DIMS:
        summary_paths = comparison.get("final_summaries", {}).get(interface, [])
        require(
            len(summary_paths) == len(goals),
            f"{interface}: final summary count mismatch",
        )
        final_summary_checks[interface] = []
        expected_planner = _as_path(
            comparison.get("final_checkpoints", {}).get(interface)
        )
        for goal, raw_path in zip(goals, summary_paths, strict=False):
            summary = _json(_as_path(raw_path))
            metadata = summary.get("metadata", {})
            push_perturbation = metadata.get("push_perturbation", {})
            push_perturbation_protocols.append(push_perturbation)
            require(
                push_perturbation.get("enabled") is True
                and push_perturbation.get("mode") == "interval",
                f"{interface}/{goal}: interval push perturbation is not enabled",
            )
            planner_metadata = metadata.get("planner_metadata", {})
            planner_sample_metadata = planner_metadata.get("sample_metadata", {})
            require(
                _as_path(metadata.get("dataset_path"))
                == expected_dataset_paths[interface],
                f"{interface}/{goal}: final evaluation used the wrong dataset cache",
            )
            require(
                _as_path(planner_sample_metadata.get("dataset_path"))
                == expected_dataset_paths[interface],
                f"{interface}/{goal}: planner checkpoint used the wrong dataset cache",
            )
            require(
                int(planner_metadata.get("parameter_count", -1)) > 0,
                f"{interface}/{goal}: planner parameter count is missing",
            )
            require(
                int(planner_metadata.get("num_samples", -1))
                == expected_rows["demonstration_and_rollout_samples"],
                f"{interface}/{goal}: final planner sample count mismatch",
            )
            require(
                int(
                    planner_metadata.get("checkpoint_metadata", {}).get(
                        "num_samples", -1
                    )
                )
                == expected_rows["demonstration_samples"],
                f"{interface}/{goal}: pretrained planner sample count mismatch",
            )
            require(
                int(metadata.get("planner_target_dim", -1))
                == INTERFACE_TARGET_DIMS[interface],
                f"{interface}/{goal}: planner target dimension mismatch",
            )
            require(
                float(planner_sample_metadata.get("control_rate_hz", math.nan)) == 50.0,
                f"{interface}/{goal}: control rate is not 50 Hz",
            )
            require(
                int(planner_sample_metadata.get("planner_interval_steps", -1)) == 10,
                f"{interface}/{goal}: planner interval is not ten steps",
            )
            require(
                float(planner_sample_metadata.get("planner_rate_hz", math.nan)) == 5.0,
                f"{interface}/{goal}: planner rate is not 5 Hz",
            )
            _require_paper_rollout_metrics(
                require,
                summary,
                label=f"{interface}/{goal}",
            )
            _require_planner_latency(
                require,
                summary,
                label=f"{interface}/{goal}",
                require_measurement=not smoke_only,
            )
            language_metadata = metadata.get("language_conditioning", {})
            require(
                language_metadata.get("goal_name") == goal,
                f"{interface}/{goal}: explicit language goal mismatch",
            )
            require(
                language_metadata.get("embedding_sha256") == language_sha,
                f"{interface}/{goal}: evaluation language hash mismatch",
            )
            require(
                metadata.get("motion_name") == goal,
                f"{interface}/{goal}: reference motion mismatch",
            )
            require(
                _as_path(metadata.get("planner_checkpoint")) == expected_planner,
                f"{interface}/{goal}: final planner checkpoint mismatch",
            )
            require(
                int(metadata.get("seed", -1)) == expected_seed,
                f"{interface}/{goal}: seed mismatch",
            )
            steps_run = int(summary.get("steps_run", -1))
            require(
                0 < steps_run <= expected_steps,
                f"{interface}/{goal}: evaluation length is outside the budget",
            )
            require(
                summary.get("stop_reason")
                == ("max_steps" if steps_run == expected_steps else "all_envs_done"),
                f"{interface}/{goal}: evaluation stop reason is inconsistent",
            )
            per_environment = summary.get("per_environment", [])
            require(
                len(per_environment) == int(metadata.get("num_envs", -1)),
                f"{interface}/{goal}: per-environment results are missing",
            )
            per_environment_success = [
                bool(environment.get("tracking_success"))
                for environment in per_environment
            ]
            per_environment_survival = [
                bool(environment.get("survived_without_fall"))
                for environment in per_environment
            ]
            aggregate_success = float(
                summary.get("aggregate", {}).get("tracking_success_rate", math.nan)
            )
            aggregate_survival = float(
                summary.get("aggregate", {}).get("survival_rate", math.nan)
            )
            require(
                bool(per_environment_success)
                and math.isclose(
                    aggregate_success,
                    sum(per_environment_success) / len(per_environment_success),
                    abs_tol=1.0e-6,
                ),
                f"{interface}/{goal}: aggregate strict success is inconsistent",
            )
            require(
                bool(per_environment_survival)
                and math.isclose(
                    aggregate_survival,
                    sum(per_environment_survival) / len(per_environment_survival),
                    abs_tol=1.0e-6,
                ),
                f"{interface}/{goal}: aggregate fall-free survival is inconsistent",
            )
            require(
                metadata.get("tracking_terminations_enabled") is False
                and set(metadata.get("disabled_tracking_termination_terms", []))
                == {"anchor_pos", "anchor_ori", "ee_body_pos"}
                and metadata.get("survival_definition")
                == "no_base_too_low_termination",
                f"{interface}/{goal}: M3 termination protocol mismatch",
            )
            require(
                int(metadata.get("random_reset_step_min", -1)) == 0
                and int(metadata.get("random_reset_step_max", -1)) == 200,
                f"{interface}/{goal}: M3 reset range is not 0-200",
            )
            require(
                metadata.get("episode_length_extension_enabled") is False
                and math.isclose(
                    float(metadata.get("episode_length_s", math.nan)),
                    10.0,
                    abs_tol=1.0e-6,
                ),
                f"{interface}/{goal}: M3 episode duration is not 500 steps",
            )
            for environment in per_environment:
                strict_failure_terms = set(environment.get("termination_terms", []))
                strict_failure_terms.difference_update(
                    {"time_out", "reference_finished"}
                )
                require(
                    bool(environment.get("tracking_success"))
                    == (not strict_failure_terms),
                    f"{interface}/{goal}: strict success and termination causes differ",
                )
                require(
                    bool(environment.get("survived_without_fall"))
                    == ("base_too_low" not in strict_failure_terms),
                    f"{interface}/{goal}: fall-free survival and termination causes differ",
                )
            termination_counts = summary.get("aggregate", {}).get(
                "termination_cause_env_counts"
            )
            require(
                isinstance(termination_counts, dict)
                and "reference_finished" in termination_counts,
                f"{interface}/{goal}: termination causes are missing",
            )
            require(
                metadata.get("policy_observation_corruption_enabled") is False,
                f"{interface}/{goal}: policy observation corruption enabled",
            )
            if interface == "latent_skill":
                require(
                    summary.get("motion_name") == goal,
                    f"{interface}/{goal}: selected latent motion mismatch",
                )
                start_names = summary.get("start_trajectories", {}).get(
                    "motion_names", []
                )
                require(
                    bool(start_names) and set(start_names) == {goal},
                    f"{interface}/{goal}: latent rollout started on another motion",
                )
            else:
                require(
                    metadata.get("low_level_command_mode") == "streamed_vanilla",
                    f"{interface}/{goal}: explicit tracker is not streamed vanilla",
                )
                tracker = metadata.get("low_level_tracker", {})
                require(
                    tracker.get("strict_policy_restore") is True,
                    f"{interface}/{goal}: tracker restore was not strict",
                )
                require(
                    tracker.get("policy_frozen") is True,
                    f"{interface}/{goal}: tracker was not frozen",
                )
                require(
                    tracker.get("policy_training") is False,
                    f"{interface}/{goal}: tracker remained in train mode",
                )
                require(
                    int(tracker.get("policy_trainable_parameter_count", -1)) == 0,
                    f"{interface}/{goal}: tracker has trainable parameters",
                )
                observation = metadata.get("planner_observation_spec", {})
                require(
                    observation.get("reference_features") == [],
                    f"{interface}/{goal}: evaluation planner input contains reference features",
                )
            final_summary_checks[interface].append(
                {
                    "goal": goal,
                    "steps_run": summary.get("steps_run"),
                    "tracking_success_rate": summary.get("aggregate", {}).get(
                        "tracking_success_rate"
                    ),
                    "survival_rate": summary.get("aggregate", {}).get("survival_rate"),
                    "fall_rate": summary.get("aggregate", {}).get("fall_rate"),
                    "survival_steps_mean": summary.get("aggregate", {}).get(
                        "survival_steps_mean"
                    ),
                }
            )

    if not smoke_only:
        for interface in INTERFACE_TARGET_DIMS:
            pretrained_paths = comparison.get("pretrained_summaries", {}).get(
                interface, []
            )
            require(
                len(pretrained_paths) == len(goals),
                f"{interface}: pretrained closed-loop summary count mismatch",
            )
            expected_planner = _as_path(
                comparison.get("pretrain_checkpoints", {}).get(interface)
            )
            for goal, raw_path in zip(goals, pretrained_paths, strict=False):
                summary = _json(_as_path(raw_path))
                metadata = summary.get("metadata", {})
                require(
                    _as_path(metadata.get("dataset_path"))
                    == expected_dataset_paths[interface],
                    f"{interface}/{goal}: pretrained evaluation used the wrong dataset cache",
                )
                _require_paper_rollout_metrics(
                    require,
                    summary,
                    label=f"{interface}/{goal} pretrained",
                )
                _require_planner_latency(
                    require,
                    summary,
                    label=f"{interface}/{goal} pretrained",
                    require_measurement=True,
                )
                pretrained_steps = int(summary.get("steps_run", -1))
                require(
                    0 < pretrained_steps <= expected_steps,
                    f"{interface}/{goal}: pretrained evaluation length is invalid",
                )
                require(
                    summary.get("stop_reason")
                    == (
                        "max_steps"
                        if pretrained_steps == expected_steps
                        else "all_envs_done"
                    ),
                    f"{interface}/{goal}: pretrained stop reason is inconsistent",
                )
                require(
                    len(summary.get("per_environment", []))
                    == int(metadata.get("num_envs", -1)),
                    f"{interface}/{goal}: pretrained termination results are missing",
                )
                pretrained_environments = summary.get("per_environment", [])
                pretrained_successes = [
                    bool(environment.get("tracking_success"))
                    for environment in pretrained_environments
                ]
                require(
                    bool(pretrained_successes)
                    and math.isclose(
                        float(
                            summary.get("aggregate", {}).get(
                                "tracking_success_rate", math.nan
                            )
                        ),
                        sum(pretrained_successes) / len(pretrained_successes),
                        abs_tol=1.0e-6,
                    ),
                    f"{interface}/{goal}: pretrained strict success is inconsistent",
                )
                require(
                    metadata.get("motion_name") == goal,
                    f"{interface}/{goal}: pretrained reference motion mismatch",
                )
                require(
                    metadata.get("language_conditioning", {}).get("goal_name") == goal,
                    f"{interface}/{goal}: pretrained language goal mismatch",
                )
                require(
                    _as_path(metadata.get("planner_checkpoint")) == expected_planner,
                    f"{interface}/{goal}: pretrained planner checkpoint mismatch",
                )

    require(
        bool(push_perturbation_protocols)
        and all(
            protocol == push_perturbation_protocols[0]
            for protocol in push_perturbation_protocols[1:]
        ),
        "push perturbation protocol differs between final evaluations",
    )

    report = {
        "passed": not errors,
        "errors": errors,
        "protocol": PROTOCOL,
        "goals": goals,
        "goal_count": len(goals),
        "seed": expected_seed,
        "smoke_only": smoke_only,
        "submission_gates_complete": submission_gates_complete,
        "paper_protocol_complete": expected_paper_status,
        "cluster_submission": cluster_submission,
        "state_dim": state_dim,
        "language_dim": language_dim,
        "language_embedding_sha256": language_sha,
        "target_dims": INTERFACE_TARGET_DIMS,
        "sample_checks": sample_checks,
        "shared_backbone_keys": list(BACKBONE_KEYS),
        "required_paper_rollout_metrics": list(PAPER_ROLLOUT_METRICS),
        "push_perturbation_protocol": (
            push_perturbation_protocols[0] if push_perturbation_protocols else None
        ),
        "final_summary_checks": final_summary_checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(f"[FAIL] {error}" for error in errors))
    print(f"[PASS] BONES-SEED multi-goal language comparison audit: {output}")


if __name__ == "__main__":
    main()
