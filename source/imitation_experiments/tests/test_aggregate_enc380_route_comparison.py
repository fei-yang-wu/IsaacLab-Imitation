from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.capacity.aggregate_enc380_route_comparison import METRICS, aggregate
from imitation_experiments.capacity.enc380_capacity_grid import (
    PLANNER_BATCH_SIZE,
    PLANNER_MICRO_BATCH_BY_SIZE,
    PLANNER_UPDATES_BY_SIZE,
    ROUTE_TASK_COUNT,
    decode_route_task,
    planner_dir_name,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_route_task_grid_gives_each_pair_two_independent_jobs() -> None:
    tasks = [decode_route_task(index) for index in range(ROUTE_TASK_COUNT)]
    assert len(tasks) == 24
    assert tasks[0] == ("walk1_subject1", "tiny", 0, "root_qpos")
    assert tasks[1] == ("walk1_subject1", "tiny", 0, "latent_skill")
    assert tasks[-2:] == [
        ("walk1_subject1", "large", 2, "root_qpos"),
        ("walk1_subject1", "large", 2, "latent_skill"),
    ]


def _summary(route: str, planner: Path, *, publishes: int) -> dict:
    packet = None
    planner_checkpoint = str(planner.resolve())
    if route == "root_qpos":
        packet = {
            "packet_source": "planner",
            "packet_interface": "root_qpos",
            "packet_target_dim": 380,
            "encoder_input_width": 380,
            "packet_frames": 10,
            "packet_frame_width": 38,
            "packet_width": 380,
            "layout_verified": True,
            "publishes": publishes,
            "packet_planner_checkpoint": planner_checkpoint,
        }
        planner_checkpoint = None
    return {
        "metadata": {
            "task": "Isaac-Imitation-G1-Latent-Strict-v0",
            "checkpoint": "/checkpoints/tracker.pt",
            "planner_checkpoint": planner_checkpoint,
            "packet_encoder_command": packet,
            "motion_manifest": "/data/manifest.json",
            "dataset_path": "/data/zarr",
            "motion_name": "motion_a",
            "seed": 0,
            "num_envs": 10,
            "early_terminations_enabled": True,
            "tracking_terminations_enabled": False,
        },
        "skill_checkpoint_override": "/checkpoints/encoder.pt",
        "max_steps": 500,
        "steps_run": 500,
        "start_trajectories": {"local_steps": [0, 20, 40]},
        "aggregate": {"survival_rate": 0.8},
        "metrics": {name: {"mean": 1.0} for name in METRICS},
        "planner_inference_latency_ms": {
            "scope": "high_level_planner_forward_only",
            "total_call_count": 20,
            "warmup_calls_excluded": 1,
            "measured_call_count": 19,
            "mean": 2.5,
        },
    }


def _study(tmp_path: Path) -> Path:
    root = tmp_path / "study"
    _write(
        root / "qualification/tracker_completion.json",
        {"passed": True, "cumulative_credited_frames": 5_000_085_504},
    )
    _write(
        root / "qualification/motion_selection.json",
        {
            "passed": True,
            "motions": ["motion_a"],
            "performance_data_used": True,
            "paper_representative_motion_selection": False,
            "manifest_sha256": "manifest-sha",
        },
    )
    _write(root / "qualification/skill_binding.json", {"passed": True})
    _write(
        root / "qualification/latent_qualification_audit.json",
        {"protocol_passed": True, "oracle_passed": True},
    )
    _write(
        root / "demonstrations/paired_raw/summary.json",
        {
            "num_envs": 10,
            "balanced_trajectory_collection": {
                "complete": True,
                "completed_trajectory_count": 100,
                "counts": {"motion_a": 100},
            },
        },
    )
    _write(
        root / "motions/motion_a/demonstrations/paired_demonstration_audit.json",
        {
            "passed": True,
            "motion_name": "motion_a",
            "rows": 5000,
            "trajectories": 100,
        },
    )
    point = root / "motions/motion_a/capacity/medium/seed0"
    for stage in ("oracle_trained",):
        planner_name = planner_dir_name("medium")
        samples = 5000
        for route, target_dim, parameters in (
            ("root_qpos", 380, 1_100_000),
            ("latent_skill", 256, 1_000_000),
        ):
            route_root = point / "matched" / route
            planner_dir = route_root / planner_name
            checkpoint = planner_dir / "checkpoints/best.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{route}-{stage}".encode())
            _write(
                planner_dir / "config.json",
                {
                    "interface": route,
                    "planner_type": "flow",
                    "training_stage": "oracle",
                    "args": {
                        "seed": 0,
                        "state_key": "planner_state",
                        "training_stage": "oracle",
                    },
                    "model_size": "medium",
                    "parameter_count": parameters,
                    "state_dim": 930,
                    "target_dim": target_dim,
                    "source_sample_count": samples,
                    "selected_sample_count": samples,
                    "num_updates": PLANNER_UPDATES_BY_SIZE["medium"],
                    "batch_size": PLANNER_BATCH_SIZE,
                    "micro_batch_size": PLANNER_MICRO_BATCH_BY_SIZE["medium"],
                    "best_validation_metric_name": (
                        "val/normalized_target_rmse_mean"
                    ),
                    "best_validation_metric": 0.25,
                    "best_validation_update": 25_000,
                    "best_checkpoint": str(checkpoint.resolve()),
                    "trajectory_split": {
                        "num_trajectories": 100,
                        "num_train_trajectories": 80,
                        "num_val_trajectories": 20,
                    },
                },
            )
            _write(
                route_root / f"eval_{stage}_survival/summary.json",
                _summary(route, checkpoint, publishes=20),
            )
            full_dir = route_root / f"eval_{stage}_full_horizon"
            video_dir = full_dir / "videos/play"
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / "rollout.mp4").write_bytes(b"video")
            full_summary = _summary(route, checkpoint, publishes=40)
            full_summary["metadata"]["early_terminations_enabled"] = False
            full_summary["video_dir"] = str(video_dir)
            _write(full_dir / "summary.json", full_summary)
    return root


def test_aggregate_accepts_full_single_cell_contract(tmp_path: Path) -> None:
    payload = aggregate(
        _study(tmp_path), motions=("motion_a",), sizes=("medium",), seeds=(0,)
    )
    assert payload["protocol"]["shared_tracker_checkpoint"] == "/checkpoints/tracker.pt"
    assert len(payload["rows"]) == 2
    assert len(payload["capacity_summary"]) == 2
    assert payload["rows"][0]["demonstration_trajectories"] == 100
    assert (
        payload["paired_differences"][0]["latent_minus_root_qpos"]["tracking_mpjpe_mm"]
        == 0.0
    )


def test_aggregate_accepts_cluster_to_local_checkpoint_relocation(
    tmp_path: Path,
) -> None:
    root = _study(tmp_path)
    for route in ("root_qpos", "latent_skill"):
        config_path = root / (
            "motions/motion_a/capacity/medium/seed0/matched/"
            f"{route}/{planner_dir_name('medium')}/config.json"
        )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        local_checkpoint = Path(payload["best_checkpoint"])
        suffix = local_checkpoint.parts[local_checkpoint.parts.index("motions") :]
        payload["best_checkpoint"] = str(Path("/workspace/IsaacLab/project") / Path(*suffix))
        _write(config_path, payload)

    result = aggregate(
        root, motions=("motion_a",), sizes=("medium",), seeds=(0,)
    )
    assert len(result["rows"]) == 2


def test_aggregate_rejects_latent_route_using_packet_encoder(tmp_path: Path) -> None:
    root = _study(tmp_path)
    path = (
        root
        / "motions/motion_a/capacity/medium/seed0/matched/latent_skill/eval_oracle_trained_survival/summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["packet_encoder_command"] = {"packet_source": "planner"}
    _write(path, payload)
    with pytest.raises(ValueError, match="unexpectedly used a packet encoder"):
        aggregate(root, motions=("motion_a",), sizes=("medium",), seeds=(0,))


def test_aggregate_rejects_noncausal_oracle_training_key(tmp_path: Path) -> None:
    root = _study(tmp_path)
    path = (
        root
        / (
            "motions/motion_a/capacity/medium/seed0/matched/root_qpos/"
            f"{planner_dir_name('medium')}/config.json"
        )
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["args"]["state_key"] = "expert_planner_state"
    _write(path, payload)
    with pytest.raises(ValueError, match="causal planner_state"):
        aggregate(root, motions=("motion_a",), sizes=("medium",), seeds=(0,))


def test_aggregate_retains_legitimate_latency_unavailable_failure(
    tmp_path: Path,
) -> None:
    root = _study(tmp_path)
    path = (
        root
        / "motions/motion_a/capacity/medium/seed0/matched/root_qpos/eval_oracle_trained_full_horizon/summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["planner_inference_latency_ms"].update(
        {
            "total_call_count": 1,
            "warmup_calls_excluded": 1,
            "measured_call_count": 0,
            "mean": float("nan"),
        }
    )
    _write(path, payload)
    result = aggregate(root, motions=("motion_a",), sizes=("medium",), seeds=(0,))
    row = next(
        row
        for row in result["rows"]
        if row["route"] == "root_qpos" and row["stage"] == "oracle_trained"
    )
    assert row["planner_inference_latency_ms"] is None


def test_aggregate_retains_first_step_temporal_metrics_as_unavailable(
    tmp_path: Path,
) -> None:
    root = _study(tmp_path)
    for route in ("root_qpos", "latent_skill"):
        path = root / (
            "motions/motion_a/capacity/medium/seed0/matched/"
            f"{route}/eval_oracle_trained_full_horizon/summary.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["steps_run"] = 1
        payload["metrics"].pop("action_delta_l2")
        _write(path, payload)
    result = aggregate(root, motions=("motion_a",), sizes=("medium",), seeds=(0,))
    rows = [row for row in result["rows"] if row["stage"] == "oracle_trained"]
    assert all(row["full_horizon_metrics"]["action_delta_l2"] is None for row in rows)


def test_aggregate_rejects_incomplete_oracle_trajectory_split(
    tmp_path: Path,
) -> None:
    root = _study(tmp_path)
    path = root / (
        "motions/motion_a/capacity/medium/seed0/matched/"
        f"root_qpos/{planner_dir_name('medium')}/config.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trajectory_split"]["num_train_trajectories"] = 79
    _write(path, payload)
    with pytest.raises(ValueError, match="exact oracle dataset"):
        aggregate(root, motions=("motion_a",), sizes=("medium",), seeds=(0,))
