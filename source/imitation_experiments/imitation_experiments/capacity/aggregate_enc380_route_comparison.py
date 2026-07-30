#!/usr/bin/env python3
"""Aggregate the walk1 enc380 shared-tracker capacity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from imitation_experiments.capacity.enc380_capacity_grid import (
    MODEL_SIZES as DEFAULT_SIZES,
    MOTIONS as DEFAULT_MOTIONS,
    PLANNER_SEEDS as DEFAULT_SEEDS,
)

ROUTES = ("root_qpos", "latent_skill")
STAGES = ("oracle_trained",)
TRAJECTORIES_PER_MOTION = 100
METRICS = (
    "tracking_mpjpe_mm",
    "root_pos_xyz_error_m",
    "joint_pos_rmse_rad",
    "ee_pos_error_m",
    "action_delta_l2",
    "tracking_velocity_distance_mps",
    "tracking_acceleration_distance_mps2",
)
TEMPORAL_METRICS = {
    "action_delta_l2",
    "tracking_velocity_distance_mps",
    "tracking_acceleration_distance_mps2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--motions", nargs="+", default=list(DEFAULT_MOTIONS))
    parser.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get("metrics", {}).get(name, {}).get("mean")
    if value is not None and math.isfinite(float(value)):
        return float(value)
    if name in TEMPORAL_METRICS and int(summary.get("steps_run", -1)) <= 1:
        return None
    raise ValueError(f"Missing finite full-horizon metric {name!r}.")


def _survival(summary: dict[str, Any]) -> float:
    aggregate = summary.get("aggregate", {})
    for key in ("survival_rate", "fall_free_rate", "horizon_completion_rate"):
        value = aggregate.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    raise ValueError("Survival summary has no finite survival measure.")


def _starts(summary: dict[str, Any]) -> tuple[int, ...]:
    values = summary.get("start_trajectories", {}).get("local_steps")
    if not isinstance(values, list) or not values:
        raise ValueError("Summary has no start_trajectories.local_steps.")
    return tuple(int(value) for value in values)


def _latency(summary: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    block = summary.get("planner_inference_latency_ms")
    if not isinstance(block, dict):
        raise ValueError("Evaluation did not record planner-only latency provenance.")
    if block.get("scope") not in (None, "high_level_planner_forward_only"):
        raise ValueError("Planner latency used the wrong timing scope.")
    if int(block.get("warmup_calls_excluded", 0)) > 1:
        raise ValueError("Planner latency excluded more than one warmup call.")
    value = block.get("mean")
    if value is not None and math.isfinite(float(value)):
        return float(value), block
    measured = int(block.get("measured_call_count", 0))
    total = int(block.get("total_call_count", 0))
    if measured == 0 and total <= 1:
        # Required failure semantics: retain a rollout that ended before the
        # second publication and report latency as unavailable.
        return None, block
    raise ValueError("Planner latency is non-finite despite post-warmup calls.")


def _assert_same(label: str, values: list[Any]) -> Any:
    if not values:
        raise ValueError(f"No values supplied for {label}.")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"Mismatched {label}: {values}")
    return first


def _validate_pass_contract(
    survival: dict[str, Any], full: dict[str, Any]
) -> list[dict[str, str]]:
    survival_meta = survival.get("metadata", {})
    full_meta = full.get("metadata", {})
    if survival_meta.get("early_terminations_enabled") is not True:
        raise ValueError("Survival pass did not keep base_too_low active.")
    if survival_meta.get("tracking_terminations_enabled") is not False:
        raise ValueError("Survival pass did not disable tracking terminations.")
    if full_meta.get("early_terminations_enabled") is not False:
        raise ValueError("Full-horizon pass did not disable all early terminations.")
    raw_video_dir = full.get("video_dir")
    if not raw_video_dir:
        raise ValueError("Full-horizon summary did not record a video directory.")
    video_dir = Path(str(raw_video_dir)).expanduser().resolve()
    videos = sorted(video_dir.rglob("*.mp4")) if video_dir.is_dir() else []
    if not videos:
        raise ValueError(f"No retained full-horizon MP4 found under {video_dir}.")
    return [{"path": str(video), "sha256": _sha256(video)} for video in videos]


def _summary_contract(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    return {
        "task": metadata.get("task"),
        "checkpoint": metadata.get("checkpoint"),
        "skill_checkpoint": summary.get("skill_checkpoint_override"),
        "manifest": metadata.get("motion_manifest"),
        "dataset_path": metadata.get("dataset_path"),
        "motion_name": metadata.get("motion_name"),
        "seed": metadata.get("seed"),
        "num_envs": metadata.get("num_envs"),
        "max_steps": summary.get("max_steps"),
        "starts": _starts(summary),
    }


def _route_contract(
    summary: dict[str, Any], *, route: str, planner_checkpoint: Path
) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    packet = metadata.get("packet_encoder_command")
    if route == "root_qpos":
        if not isinstance(packet, dict):
            raise ValueError(
                "root_qpos evaluation did not use the packet encoder route."
            )
        expected = {
            "packet_source": "planner",
            "packet_interface": "root_qpos",
            "packet_target_dim": 380,
            "encoder_input_width": 380,
            "packet_frames": 10,
            "packet_frame_width": 38,
            "packet_width": 380,
            "layout_verified": True,
        }
        for key, value in expected.items():
            if packet.get(key) != value:
                raise ValueError(
                    f"root_qpos packet route has {key}={packet.get(key)!r}, "
                    f"expected {value!r}."
                )
        if int(packet.get("publishes", 0)) <= 0:
            raise ValueError("root_qpos packet route recorded no encoder publications.")
        used = Path(str(packet.get("packet_planner_checkpoint", ""))).resolve()
        if used != planner_checkpoint.resolve():
            raise ValueError(
                "root_qpos packet route used the wrong planner checkpoint."
            )
        return {**expected, "packet_planner_checkpoint": str(used)}
    if packet is not None:
        raise ValueError("latent_skill evaluation unexpectedly used a packet encoder.")
    used = Path(str(metadata.get("planner_checkpoint", ""))).resolve()
    if used != planner_checkpoint.resolve():
        raise ValueError("latent_skill route used the wrong planner checkpoint.")
    return {"packet_encoder_command": None, "planner_checkpoint": str(used)}


def _row(
    point_root: Path,
    *,
    motion: str,
    size: str,
    seed: int,
    route: str,
    stage: str,
    demonstration_audit: dict[str, Any],
) -> dict[str, Any]:
    route_root = point_root / "matched" / route
    survival_path = route_root / f"eval_{stage}_survival/summary.json"
    full_path = route_root / f"eval_{stage}_full_horizon/summary.json"
    planner_dir = route_root / "planner_oracle"
    config_path = planner_dir / "config.json"
    checkpoint_path = planner_dir / "checkpoints/latest.pt"
    survival = _load(survival_path)
    full = _load(full_path)
    config = _load(config_path)
    videos = _validate_pass_contract(survival, full)
    if config.get("interface") != route:
        raise ValueError(f"Planner config interface does not match {route!r}.")
    expected_dim = 380 if route == "root_qpos" else 256
    if int(config.get("target_dim", -1)) != expected_dim:
        raise ValueError(f"{route} planner target is not {expected_dim}-D.")
    if int(config.get("state_dim", -1)) != 930:
        raise ValueError("Planner input is not the required causal 10 x 93 history.")
    if str(config.get("model_size")) != size:
        raise ValueError(f"Planner size does not match path size {size!r}.")
    args = config.get("args", {})
    if config.get("training_stage") != "oracle":
        raise ValueError("Planner training_stage is not 'oracle'.")
    if str(args.get("state_key")) != "planner_state":
        raise ValueError("Planner was not trained from causal planner_state.")
    if int(args.get("seed", -1)) != seed:
        raise ValueError("Planner seed does not match its grid cell.")
    trajectory_split = config.get("trajectory_split")
    if not isinstance(trajectory_split, dict):
        raise ValueError(
            "Planner config has no trajectory-wise train/validation split."
        )
    split_total = int(trajectory_split.get("num_train_trajectories", 0)) + int(
        trajectory_split.get("num_val_trajectories", 0)
    )
    if split_total != int(demonstration_audit["trajectories"]):
        raise ValueError(
            "Planner trajectory split does not cover the exact oracle dataset."
        )
    survival_route = _route_contract(
        survival, route=route, planner_checkpoint=checkpoint_path
    )
    full_route = _route_contract(full, route=route, planner_checkpoint=checkpoint_path)
    _assert_same(
        f"{motion}/{size}/{seed}/{route}/{stage} route", [survival_route, full_route]
    )
    survival_contract = _summary_contract(survival)
    full_contract = _summary_contract(full)
    _assert_same(
        f"{motion}/{size}/{seed}/{route}/{stage} eval contract",
        [survival_contract, full_contract],
    )
    if survival_contract["motion_name"] != motion:
        raise ValueError("Evaluation motion does not match its specialist planner.")
    if int(survival_contract["seed"]) != seed:
        raise ValueError("Evaluation seed does not match its planner seed.")
    latency, latency_record = _latency(full)
    return {
        "motion_name": motion,
        "route": route,
        "stage": stage,
        "planner_family": config["planner_type"],
        "planner_seed": seed,
        "model_size": size,
        "parameter_count": int(config["parameter_count"]),
        "target_dim": int(config["target_dim"]),
        "output_bandwidth_values_per_second": int(config["target_dim"]) * 5,
        "source_sample_count": int(config["source_sample_count"]),
        "selected_sample_count": int(config["selected_sample_count"]),
        "num_updates": int(config["num_updates"]),
        "demonstration_rows": int(demonstration_audit["rows"]),
        "demonstration_trajectories": int(demonstration_audit["trajectories"]),
        "training_trajectories": int(trajectory_split["num_train_trajectories"]),
        "validation_trajectories": int(trajectory_split["num_val_trajectories"]),
        "survival_rate": _survival(survival),
        "full_horizon_metrics": {name: _metric(full, name) for name in METRICS},
        "planner_inference_latency_ms": latency,
        "planner_inference_latency_record": latency_record,
        "evaluation_contract": full_contract,
        "artifacts": {
            "survival_summary": str(survival_path.resolve()),
            "survival_summary_sha256": _sha256(survival_path),
            "full_horizon_summary": str(full_path.resolve()),
            "full_horizon_summary_sha256": _sha256(full_path),
            "planner_config": str(config_path.resolve()),
            "planner_config_sha256": _sha256(config_path),
            "planner_checkpoint": str(checkpoint_path.resolve()),
            "planner_checkpoint_sha256": _sha256(checkpoint_path),
            "full_horizon_videos": videos,
        },
    }


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    return {
        "mean": statistics.mean(materialized),
        "std": statistics.pstdev(materialized),
    }


def _optional_mean_std(values: Iterable[float | None]) -> dict[str, float] | None:
    materialized = [float(value) for value in values if value is not None]
    return _mean_std(materialized) if materialized else None


def aggregate(
    study_root: Path,
    *,
    motions: tuple[str, ...] = DEFAULT_MOTIONS,
    sizes: tuple[str, ...] = DEFAULT_SIZES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    study_root = study_root.expanduser().resolve()
    completion_path = study_root / "qualification/tracker_completion.json"
    selection_path = study_root / "qualification/motion_selection.json"
    binding_path = study_root / "qualification/skill_binding.json"
    qualification_path = study_root / "qualification/latent_qualification_audit.json"
    collection_path = study_root / "demonstrations/paired_raw/summary.json"
    completion = _load(completion_path)
    selection = _load(selection_path)
    binding = _load(binding_path)
    qualification = _load(qualification_path)
    collection = _load(collection_path)
    if (
        completion.get("passed") is not True
        or int(completion.get("cumulative_credited_frames", 0)) < 5_000_000_000
    ):
        raise ValueError("Cross-segment enc380 5B completion did not pass.")
    if (
        selection.get("passed") is not True
        or tuple(selection.get("motions", ())) != motions
        or selection.get("performance_data_used") is not True
        or selection.get("paper_representative_motion_selection") is not False
    ):
        raise ValueError(
            "Manifest-bound walk1 diagnostic selection did not pass."
        )
    if binding.get("passed") is not True:
        raise ValueError("Frozen tracker/encoder tensor binding did not pass.")
    if (
        qualification.get("protocol_passed") is not True
        or qualification.get("oracle_passed") is not True
    ):
        raise ValueError("enc380 oracle qualification did not pass.")
    trajectory_collection = collection.get("balanced_trajectory_collection")
    if (
        not isinstance(trajectory_collection, dict)
        or trajectory_collection.get("complete") is not True
        or int(trajectory_collection.get("completed_trajectory_count", -1))
        != TRAJECTORIES_PER_MOTION * len(motions)
        or set(trajectory_collection.get("counts", {})) != set(motions)
        or any(
            int(value) != TRAJECTORIES_PER_MOTION
            for value in trajectory_collection.get("counts", {}).values()
        )
    ):
        raise ValueError(
            "Single-session oracle collection did not produce the exact "
            f"{TRAJECTORIES_PER_MOTION} complete trajectories per motion."
        )

    rows: list[dict[str, Any]] = []
    demo_artifacts: dict[str, dict[str, Any]] = {}
    for motion in motions:
        demo_path = (
            study_root
            / f"motions/{motion}/demonstrations/paired_demonstration_audit.json"
        )
        demo = _load(demo_path)
        if demo.get("passed") is not True or demo.get("motion_name") != motion:
            raise ValueError(f"Paired demonstrations did not pass for {motion}.")
        if int(demo.get("trajectories", 0)) != TRAJECTORIES_PER_MOTION:
            raise ValueError(
                f"{motion} does not have the fixed "
                f"{TRAJECTORIES_PER_MOTION} completed oracle trajectories."
            )
        demo_artifacts[motion] = {
            "path": str(demo_path.resolve()),
            "sha256": _sha256(demo_path),
            "rows": int(demo["rows"]),
            "trajectories": int(demo["trajectories"]),
        }
        for size in sizes:
            for seed in seeds:
                point_root = study_root / f"motions/{motion}/capacity/{size}/seed{seed}"
                point_rows = [
                    _row(
                        point_root,
                        motion=motion,
                        size=size,
                        seed=seed,
                        route=route,
                        stage=stage,
                        demonstration_audit=demo,
                    )
                    for stage in STAGES
                    for route in ROUTES
                ]
                for stage in STAGES:
                    pair = [row for row in point_rows if row["stage"] == stage]
                    _assert_same(
                        f"{motion}/{size}/{seed}/{stage} eval",
                        [row["evaluation_contract"] for row in pair],
                    )
                    _assert_same(
                        f"{motion}/{size}/{seed}/{stage} updates",
                        [row["num_updates"] for row in pair],
                    )
                    _assert_same(
                        f"{motion}/{size}/{seed}/{stage} selected rows",
                        [row["selected_sample_count"] for row in pair],
                    )
                    _assert_same(
                        f"{motion}/{size}/{seed}/{stage} source rows",
                        [row["source_sample_count"] for row in pair],
                    )
                _assert_same(
                    f"{motion}/{size}/{seed} evaluation starts/protocol",
                    [row["evaluation_contract"] for row in point_rows],
                )
                rows.extend(point_rows)

    expected_rows = len(motions) * len(sizes) * len(seeds) * len(STAGES) * len(ROUTES)
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} result rows, got {len(rows)}.")
    shared = _assert_same(
        "frozen tracker/encoder binding",
        [
            (
                row["evaluation_contract"]["checkpoint"],
                row["evaluation_contract"]["skill_checkpoint"],
            )
            for row in rows
        ],
    )

    paired: list[dict[str, Any]] = []
    for motion in motions:
        for size in sizes:
            for seed in seeds:
                for stage in STAGES:
                    by_route = {
                        row["route"]: row
                        for row in rows
                        if row["motion_name"] == motion
                        and row["model_size"] == size
                        and row["planner_seed"] == seed
                        and row["stage"] == stage
                    }
                    latent, root = by_route["latent_skill"], by_route["root_qpos"]
                    paired.append(
                        {
                            "motion_name": motion,
                            "model_size": size,
                            "planner_seed": seed,
                            "stage": stage,
                            "latent_minus_root_qpos": {
                                "survival_rate": latent["survival_rate"]
                                - root["survival_rate"],
                                **{
                                    name: (
                                        latent["full_horizon_metrics"][name]
                                        - root["full_horizon_metrics"][name]
                                        if latent["full_horizon_metrics"][name]
                                        is not None
                                        and root["full_horizon_metrics"][name]
                                        is not None
                                        else None
                                    )
                                    for name in METRICS
                                },
                            },
                        }
                    )

    capacity_summary: list[dict[str, Any]] = []
    for stage in STAGES:
        for size in sizes:
            for route in ROUTES:
                group = [
                    row
                    for row in rows
                    if row["stage"] == stage
                    and row["model_size"] == size
                    and row["route"] == route
                ]
                available_latencies = [
                    row["planner_inference_latency_ms"]
                    for row in group
                    if row["planner_inference_latency_ms"] is not None
                ]
                capacity_summary.append(
                    {
                        "stage": stage,
                        "model_size": size,
                        "route": route,
                        "n_motion_seed_cells": len(group),
                        "parameter_count": _mean_std(
                            row["parameter_count"] for row in group
                        ),
                        "survival_rate": _mean_std(
                            row["survival_rate"] for row in group
                        ),
                        "planner_inference_latency_ms": (
                            _mean_std(available_latencies)
                            if available_latencies
                            else None
                        ),
                        "planner_latency_available_cells": len(available_latencies),
                        "planner_latency_unavailable_cells": len(group)
                        - len(available_latencies),
                        "full_horizon_metrics": {
                            name: _optional_mean_std(
                                row["full_horizon_metrics"][name] for row in group
                            )
                            for name in METRICS
                        },
                    }
                )

    paired_capacity_summary: list[dict[str, Any]] = []
    for stage in STAGES:
        for size in sizes:
            group = [
                item
                for item in paired
                if item["stage"] == stage and item["model_size"] == size
            ]
            paired_capacity_summary.append(
                {
                    "stage": stage,
                    "model_size": size,
                    "n_motion_seed_pairs": len(group),
                    "latent_minus_root_qpos": {
                        "survival_rate": _mean_std(
                            item["latent_minus_root_qpos"]["survival_rate"]
                            for item in group
                        ),
                        **{
                            name: _optional_mean_std(
                                item["latent_minus_root_qpos"][name] for item in group
                            )
                            for name in METRICS
                        },
                    },
                }
            )

    return {
        "study": "enc380_shared_tracker_planner_route_capacity_comparison",
        "protocol": {
            "explicit_route": "root_qpos planner -> frozen enc380 encoder -> latent tracker",
            "latent_route": "latent planner -> latent tracker",
            "shared_tracker_checkpoint": shared[0],
            "shared_skill_checkpoint": shared[1],
            "planner_input": "causal planner_state (10 x 93)",
            "training": (
                "single oracle-supervised fit; no planner pretrain, learned-planner "
                "rollout collection, or finetune"
            ),
            "oracle_collection": {
                "summary": str(collection_path.resolve()),
                "summary_sha256": _sha256(collection_path),
                "sessions": 1,
                "parallel_environments": int(collection["num_envs"]),
                "completed_trajectories_total": (
                    TRAJECTORIES_PER_MOTION * len(motions)
                ),
                "completed_trajectories_per_motion": TRAJECTORIES_PER_MOTION,
                "partial_trajectories_at_cutoff_included": False,
            },
            "motion_selection": {
                "rule": (
                    "user-requested walk1_subject1 continuity diagnostic; "
                    "not an unbiased paper-representative motion selection"
                ),
                "motions": list(motions),
            },
            "sizes": list(sizes),
            "seeds": list(seeds),
            "survival_source": "base_too_low-active pass",
            "metric_source": "all-early-terminations-disabled full-horizon pass",
            "latency_scope": "high-level planner root forward only, one warmup excluded",
            "qualification": {
                "tracker_completion": {
                    "path": str(completion_path.resolve()),
                    "sha256": _sha256(completion_path),
                    "cumulative_credited_frames": completion[
                        "cumulative_credited_frames"
                    ],
                },
                "motion_selection": {
                    "path": str(selection_path.resolve()),
                    "sha256": _sha256(selection_path),
                    "manifest_sha256": selection["manifest_sha256"],
                },
                "binding": {
                    "path": str(binding_path.resolve()),
                    "sha256": _sha256(binding_path),
                },
                "oracle": {
                    "path": str(qualification_path.resolve()),
                    "sha256": _sha256(qualification_path),
                },
            },
        },
        "demonstrations": demo_artifacts,
        "rows": rows,
        "paired_differences": paired,
        "capacity_summary": capacity_summary,
        "paired_capacity_summary": paired_capacity_summary,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# enc380 shared-tracker planner-route capacity comparison",
        "",
        "walk1_subject1 × four capacities × three planner seeds. Values are mean ± population SD across three seed cells.",
        "",
        "| stage | size | route | n | params | survival | MPJPE (mm) | root (m) | joint (rad) | EE (m) | latency (ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["capacity_summary"]:
        metrics = row["full_horizon_metrics"]

        def fmt(block: dict[str, float] | None, digits: int) -> str:
            if block is None:
                return "unavailable"
            return f"{block['mean']:.{digits}f} ± {block['std']:.{digits}f}"

        lines.append(
            f"| {row['stage']} | {row['model_size']} | {row['route']} | {row['n_motion_seed_cells']} | "
            f"{fmt(row['parameter_count'], 0)} | {fmt(row['survival_rate'], 3)} | "
            f"{fmt(metrics['tracking_mpjpe_mm'], 2)} | {fmt(metrics['root_pos_xyz_error_m'], 4)} | "
            f"{fmt(metrics['joint_pos_rmse_rad'], 4)} | {fmt(metrics['ee_pos_error_m'], 4)} | "
            f"{fmt(row['planner_inference_latency_ms'], 3)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    payload = aggregate(
        args.study_root,
        motions=tuple(args.motions),
        sizes=tuple(args.sizes),
        seeds=tuple(args.seeds),
    )
    output_dir.mkdir(parents=True)
    results_path = output_dir / "results.json"
    markdown_path = output_dir / "results.md"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"[PASS] {results_path}")
    print(f"[PASS] {markdown_path}")


if __name__ == "__main__":
    main()
