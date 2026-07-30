#!/usr/bin/env python3
"""Audit and aggregate the focused H30 explicit-packet temporal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MODES = ("execute_first10", "temporal_exponential")
METRICS = (
    "tracking_mpjpe_mm",
    "root_pos_xyz_error_m",
    "joint_pos_rmse_rad",
    "ee_pos_error_m",
    "action_delta_l2",
    "tracking_velocity_distance_mps",
    "tracking_acceleration_distance_mps2",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--source_study_root", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite.")
    return result


def _metric(summary: dict[str, Any], name: str) -> float | None:
    means = summary.get("metric_means", {})
    for key in (name, f"deterministic_tracking/{name}"):
        if key in means:
            return _finite(means[key], key)
    metrics = summary.get("metrics", {})
    for key in (name, f"deterministic_tracking/{name}"):
        if key in metrics and metrics[key].get("mean") is not None:
            return _finite(metrics[key]["mean"], key)
    if (
        name
        in {
            "action_delta_l2",
            "tracking_velocity_distance_mps",
            "tracking_acceleration_distance_mps2",
        }
        and int(summary.get("steps_run", -1)) <= 1
    ):
        return None
    raise ValueError(f"Missing full-horizon metric {name!r}.")


def _survival(summary: dict[str, Any]) -> float:
    aggregate = summary.get("aggregate", {})
    for key in ("survival_rate", "fall_free_rate"):
        if aggregate.get(key) is not None:
            return _finite(aggregate[key], key)
    raise ValueError("Survival pass has no survival/fall-free rate.")


def _validate_passes(
    *,
    survival: dict[str, Any],
    full: dict[str, Any],
    expected_mode: str,
    expected_checkpoint: Path,
) -> list[dict[str, str]]:
    survival_meta = survival.get("metadata", {})
    full_meta = full.get("metadata", {})
    if survival_meta.get("early_terminations_enabled") is not True:
        raise ValueError("Survival pass did not keep base_too_low active.")
    if survival_meta.get("tracking_terminations_enabled") is not False:
        raise ValueError("Survival pass did not disable tracking terminations.")
    if full_meta.get("early_terminations_enabled") is not False:
        raise ValueError("Full-horizon pass did not disable all terminations.")
    starts = survival.get("start_trajectories", {}).get("local_steps")
    if starts != full.get("start_trajectories", {}).get("local_steps"):
        raise ValueError("Survival and full-horizon passes used different starts.")
    for summary in (survival, full):
        packet = summary.get("metadata", {}).get("packet_encoder_command")
        if not isinstance(packet, dict):
            raise ValueError("Evaluation did not use the explicit packet encoder.")
        expected = {
            "packet_source": "planner",
            "packet_interface": "root_qpos",
            "packet_target_dim": 1140,
            "encoder_input_width": 380,
            "packet_frames": 10,
            "planner_prediction_frames": 30,
            "planner_prediction_width": 1140,
            "packet_temporal_ensemble": expected_mode,
        }
        for key, value in expected.items():
            if packet.get(key) != value:
                raise ValueError(f"{key}={packet.get(key)!r}, expected {value!r}.")
        used = Path(str(packet.get("packet_planner_checkpoint", ""))).resolve()
        if used != expected_checkpoint.resolve():
            raise ValueError("Evaluation used the wrong H30 planner checkpoint.")
    raw_video_dir = full.get("video_dir")
    video_dir = Path(str(raw_video_dir)).expanduser().resolve()
    videos = sorted(video_dir.rglob("*.mp4")) if video_dir.is_dir() else []
    if not videos:
        raise ValueError(f"Full-horizon video missing under {video_dir}.")
    return [{"path": str(path), "sha256": _sha256(path)} for path in videos]


def _h30_row(output_root: Path, mode: str, checkpoint: Path) -> dict[str, Any]:
    eval_root = output_root / "evaluation" / mode
    survival = _load(eval_root / "survival/summary.json")
    full = _load(eval_root / "full_horizon/summary.json")
    expected_mode = "none" if mode == "execute_first10" else "exponential"
    videos = _validate_passes(
        survival=survival,
        full=full,
        expected_mode=expected_mode,
        expected_checkpoint=checkpoint,
    )
    return {
        "method": f"h30_{mode}",
        "prediction_horizon_steps": 30,
        "execution_horizon_steps": 10,
        "published_values_per_second": 5700,
        "survival_rate": _survival(survival),
        "metrics": {name: _metric(full, name) for name in METRICS},
        "planner_latency_ms": full.get("planner_inference_latency_ms"),
        "videos": videos,
        "survival_summary": str((eval_root / "survival/summary.json").resolve()),
        "full_horizon_summary": str(
            (eval_root / "full_horizon/summary.json").resolve()
        ),
    }


def _h10_row(source_root: Path) -> dict[str, Any]:
    route_root = (
        source_root / "motions/walk1_subject1/capacity/medium/seed0/matched/root_qpos"
    )
    survival_path = route_root / "eval_oracle_trained_survival/summary.json"
    full_path = route_root / "eval_oracle_trained_full_horizon/summary.json"
    survival = _load(survival_path)
    full = _load(full_path)
    packet = full.get("metadata", {}).get("packet_encoder_command", {})
    if (
        packet.get("packet_target_dim") != 380
        or packet.get("packet_frames") != 10
        or packet.get("packet_source") != "planner"
    ):
        raise ValueError("Source H10 baseline is not the frozen explicit protocol.")
    if survival.get("metadata", {}).get("tracking_terminations_enabled") is not False:
        raise ValueError("Source H10 survival pass kept tracking terminations.")
    if full.get("metadata", {}).get("early_terminations_enabled") is not False:
        raise ValueError("Source H10 full-horizon pass kept early terminations.")
    return {
        "method": "h10_explicit_baseline",
        "prediction_horizon_steps": 10,
        "execution_horizon_steps": 10,
        "published_values_per_second": 1900,
        "survival_rate": _survival(survival),
        "metrics": {name: _metric(full, name) for name in METRICS},
        "planner_latency_ms": full.get("planner_inference_latency_ms"),
        "survival_summary": str(survival_path.resolve()),
        "full_horizon_summary": str(full_path.resolve()),
    }


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    source_root = args.source_study_root.expanduser().resolve()
    aggregate_dir = output_root / "aggregate"
    if aggregate_dir.exists():
        raise FileExistsError(f"Refusing existing aggregate: {aggregate_dir}")

    materialization_path = (
        output_root / "demonstrations/root_qpos_h30/materialization_manifest.json"
    )
    materialization = _load(materialization_path)
    if (
        int(materialization.get("row_count", -1)) != 4864
        or int(materialization.get("source_horizon_steps", -1)) != 10
        or int(materialization.get("target_horizon_steps", -1)) != 30
    ):
        raise ValueError("H30 materialization does not preserve the 4,864 H10 rows.")
    if _finite(
        materialization.get("validation", {}).get("max_abs"),
        "H10 reconstruction max_abs",
    ) > _finite(
        materialization.get("validation", {}).get("tolerance"),
        "H10 reconstruction tolerance",
    ):
        raise ValueError("H30 materialization failed exact H10 reconstruction.")

    planner_dir = output_root / "planner/medium/seed0/planner_oracle_u30000_b1024"
    config_path = planner_dir / "config.json"
    checkpoint = planner_dir / "checkpoints/best.pt"
    config = _load(config_path)
    expected_config = {
        "interface": "root_qpos",
        "state_dim": 930,
        "target_dim": 1140,
        "model_size": "medium",
        "batch_size": 1024,
        "micro_batch_size": 256,
        "num_updates": 30000,
        "best_validation_metric_name": "val/normalized_target_rmse_mean",
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(f"Planner {key}={config.get(key)!r}, expected {value!r}.")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    rows = [_h10_row(source_root)]
    rows.extend(_h30_row(output_root, mode, checkpoint) for mode in MODES)
    aggregate_dir.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "scope": "strong_explicit_h30_diagnostic",
        "fair_reuse": {
            "same_causal_rows": True,
            "same_trajectory_split_seed": 0,
            "row_count": 4864,
            "h10_reconstruction_max_abs": materialization["validation"]["max_abs"],
            "materialization_manifest": str(materialization_path),
            "materialization_manifest_sha256": _sha256(materialization_path),
        },
        "rows": rows,
    }
    result_path = aggregate_dir / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# H30 explicit temporal diagnostic",
        "",
        "| Method | Survival | MPJPE (mm) | Planner latency (ms) | Values/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        latency = row["planner_latency_ms"]
        latency_mean = latency.get("mean") if isinstance(latency, dict) else None
        latency_text = "N/A" if latency_mean is None else f"{float(latency_mean):.3f}"
        markdown.append(
            f"| {row['method']} | {row['survival_rate']:.3f} | "
            f"{row['metrics']['tracking_mpjpe_mm']:.2f} | {latency_text} | "
            f"{row['published_values_per_second']} |"
        )
    markdown_path = aggregate_dir / "results.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"[PASS] {result_path}")
    print(f"[PASS] {markdown_path}")


if __name__ == "__main__":
    main()
