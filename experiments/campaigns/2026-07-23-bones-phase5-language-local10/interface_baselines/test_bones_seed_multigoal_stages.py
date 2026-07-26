"""Tests for resumable BONES-SEED multi-goal stage contracts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from run_bones_seed_multigoal_language_comparison import (
    DEMONSTRATION_COLLECTION_SAFETY_MULTIPLIER,
    HORIZON_STEPS,
    Runner,
    _explicit_eval_command,
    _goal_index_for_stage,
    _latent_eval_command,
    _repository_provenance,
    _require_stage_record,
    _validate_existing_run_config,
    _write_run_config,
    _write_stage_record,
)
import run_bones_seed_multigoal_language_comparison as staged_runner
from audit_bones_seed_multigoal_language_comparison import (
    PAPER_ROLLOUT_METRICS,
    _require_paper_rollout_metrics,
    _require_planner_latency,
)


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "stage": "prepare",
        "goal_index": None,
        "demo_rows_per_goal": 3,
        "rollout_rows_per_goal": 4,
        "rollout_num_envs": 2,
        "skip_pretrained_closed_loop": False,
        "eval_steps": 20,
        "seed": 7,
        "model_size": "tiny",
        "pretrain_updates": 1,
        "finetune_updates": 2,
        "batch_size": 4,
        "micro_batch_size": 2,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "flow_steps": 2,
        "train_endpoint_steps": 1,
        "flow_noise_std": 0.0,
        "interfaces": ["latent_skill", "full_body_trajectory"],
        "latent_dataset_path": Path("/tmp/bones-latent-cache"),
        "vanilla_dataset_path": Path("/tmp/bones-vanilla-cache"),
        "python_cmd": "python",
        "isaaclab_python_cmd": "python",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_goal_stage_uses_only_explicit_index_or_array_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _goal_index_for_stage(_args(stage="rollout", goal_index=1), 3) == 1

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    assert _goal_index_for_stage(_args(stage="final-eval"), 3) == 2

    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    with pytest.raises(ValueError, match="requires --goal_index"):
        _goal_index_for_stage(_args(stage="rollout"), 3)
    with pytest.raises(ValueError, match="outside"):
        _goal_index_for_stage(_args(stage="rollout", goal_index=3), 3)
    with pytest.raises(ValueError, match="only valid"):
        _goal_index_for_stage(_args(stage="finetune", goal_index=0), 3)


def test_archive_snapshot_does_not_require_git_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(staged_runner, "_git_output", lambda *args: "")

    provenance = _repository_provenance()

    assert provenance["git_metadata_available"] is False
    assert provenance["snapshot_without_git_metadata"] is True


def test_stage_record_detects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"original")
    record = _write_stage_record(
        output_root=tmp_path,
        stage="rollout",
        goal_index=0,
        goal_name="walk",
        artifacts={"checkpoint": artifact},
    )

    assert _require_stage_record(record)["status"] == "complete"
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        _require_stage_record(record)


def test_stage_record_hashes_sample_directory_and_goal_identity(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    sample = samples / "sample_step_000000.pt"
    sample.write_bytes(b"rows")
    record = _write_stage_record(
        output_root=tmp_path,
        stage="rollout",
        goal_index=2,
        goal_name="turn left",
        artifacts={"samples": samples},
    )

    _require_stage_record(
        record,
        expected_stage="rollout",
        expected_goal_index=2,
        expected_goal_name="turn left",
    )
    with pytest.raises(ValueError, match="wrong goal name"):
        _require_stage_record(record, expected_goal_name="turn right")
    sample.write_bytes(b"changed rows")
    with pytest.raises(ValueError, match="hash changed"):
        _require_stage_record(record)


def test_later_stage_validates_original_run_contract(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    manifest = tmp_path / "manifest.json"
    language = tmp_path / "language.pt"
    manifest.write_text("{}\n", encoding="utf-8")
    language.write_bytes(b"language")
    input_paths = {"manifest": manifest, "language_embeddings": language}
    args = _args()
    paths = {
        **input_paths,
        "output_root": output_root,
        "latent_dataset_path": Path(args.latent_dataset_path),
        "vanilla_dataset_path": Path(args.vanilla_dataset_path),
    }
    goals = ["walk", "wave"]
    _write_run_config(
        output_root=output_root,
        args=args,
        goals=goals,
        language_metadata={"embedding_dim": 384},
        gate_metadata={
            "fresh_preparation_checked": True,
            "low_level_gates_checked": True,
        },
        paths=paths,
    )

    config_path = output_root / "run_config.json"
    config = staged_runner._json_object(config_path)
    assert config["demonstration_collection"] == {
        "mode": "balanced_multi_environment",
        "simulator_launches": 2,
        "rows_per_goal": 3,
        "max_control_steps": (
            3 * HORIZON_STEPS * DEMONSTRATION_COLLECTION_SAFETY_MULTIPLIER
        ),
        "safety_multiplier": DEMONSTRATION_COLLECTION_SAFETY_MULTIPLIER,
    }
    _validate_existing_run_config(
        path=config_path,
        args=args,
        goals=goals,
        output_root=output_root,
        input_paths=input_paths,
    )

    with pytest.raises(ValueError, match="seed differs"):
        _validate_existing_run_config(
            path=config_path,
            args=_args(seed=8),
            goals=goals,
            output_root=output_root,
            input_paths=input_paths,
        )

    with pytest.raises(ValueError, match="environment count differs"):
        _validate_existing_run_config(
            path=config_path,
            args=_args(rollout_num_envs=3),
            goals=goals,
            output_root=output_root,
            input_paths=input_paths,
        )

    with pytest.raises(ValueError, match="selected interfaces differ"):
        _validate_existing_run_config(
            path=config_path,
            args=_args(interfaces=["latent_skill"]),
            goals=goals,
            output_root=output_root,
            input_paths=input_paths,
        )

    manifest.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        _validate_existing_run_config(
            path=config_path,
            args=args,
            goals=goals,
            output_root=output_root,
            input_paths=input_paths,
        )


def test_paper_audit_keeps_early_failure_without_temporal_latency_samples() -> None:
    summary = {
        "steps_run": 1,
        "stop_reason": "all_envs_done",
        "metrics": {
            name: {"mean": 1.0, "count": 1}
            for name in PAPER_ROLLOUT_METRICS
            if name not in {"action_delta_l2", "tracking_acceleration_distance_mps2"}
        },
        "planner_inference_latency_ms": {
            "unit": "ms",
            "scope": "high_level_planner_forward_only",
            "total_call_count": 1,
            "warmup_calls_excluded": 1,
            "measured_call_count": 0,
            "mean": float("nan"),
        },
    }
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    _require_paper_rollout_metrics(require, summary, label="latent/walk")
    _require_planner_latency(
        require,
        summary,
        label="latent/walk",
        require_measurement=True,
    )

    assert errors == []


def test_rollout_commands_batch_only_one_explicit_goal(tmp_path: Path) -> None:
    args = _args(rollout_rows_per_goal=13, rollout_num_envs=3)
    runner = Runner(args, tmp_path)
    explicit = _explicit_eval_command(
        runner,
        goal="kick",
        planner_checkpoint=tmp_path / "planner.pt",
        output_json=tmp_path / "explicit.json",
        steps=50,
        num_envs=3,
        balanced_rows_per_motion=13,
        save_samples=True,
        samples_output_dir=tmp_path / "explicit_samples",
        label="explicit",
        manifest=tmp_path / "manifest.json",
        language_embeddings=tmp_path / "language.pt",
        checkpoint=tmp_path / "tracker.pt",
        dataset_path=tmp_path / "vanilla_cache",
    )
    assert explicit[explicit.index("--num_envs") + 1] == "3"
    assert explicit[explicit.index("--motion_name") + 1] == "kick"
    assert explicit[explicit.index("--language_goal_name") + 1] == "kick"
    assert explicit[explicit.index("--balanced_rows_per_motion") + 1] == "13"
    assert "--disable_tracking_terminations" in explicit
    assert "--keep_configured_episode_length" in explicit

    latent = _latent_eval_command(
        runner,
        goal="kick",
        balanced_rows_per_motion=13,
        num_envs=3,
        output_dir=tmp_path / "latent",
        max_steps=50,
        command_source="skill_commander",
        planner_checkpoint=tmp_path / "planner.pt",
        save_samples=True,
        refresh=False,
        label="latent",
        manifest=tmp_path / "manifest.json",
        dataset_path=tmp_path / "latent_cache",
        language_embeddings=tmp_path / "language.pt",
        low_level_checkpoint=tmp_path / "latent_low_level.pt",
        skill_checkpoint=tmp_path / "skill.pt",
    )
    assert latent[latent.index("--num_envs") + 1] == "3"
    assert latent[latent.index("--motion_name") + 1] == "kick"
    balanced_index = latent.index("--balanced_motion_names")
    assert latent[balanced_index + 1] == "kick"
    assert latent[latent.index("--balanced_rows_per_motion") + 1] == "13"
    assert "--disable_tracking_terminations" in latent
    assert "--allow_random_reset" in latent
    assert "--extend_episode_length_for_max_steps" not in latent
    assert "agent.ipmd.skill_commander_goal_name=kick" in latent
