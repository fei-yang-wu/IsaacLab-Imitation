from __future__ import annotations

import argparse
from pathlib import Path

from imitation_experiments.pipeline.run_language_planner_oracle_pretrain import (
    _collection_command,
    _eval_command,
    _train_command,
    aggregate_milestone_evaluations,
)


def _eval_args(**overrides) -> argparse.Namespace:
    values = {
        "low_level_checkpoint": Path("low_level.pt"),
        "skill_checkpoint": Path("skill.pt"),
        "language_embeddings": Path("language.pt"),
        "eval_trajectories_per_motion": 100,
        "seed": 0,
        "flow_steps": 16,
        "reference_arrays_dir": Path("reference_arrays"),
        "reference_arrays_persist_id": "selected10@test",
        "skill_z_dim": 256,
        "policy_num_cells": None,
        "policy_activation": None,
        "solver_njmax": 289,
        "solver_nconmax": 200,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _summary(successes: int, total: int, mpjpe: float) -> dict:
    return {
        "aggregate": {"completed_tracking_success_count": successes},
        "per_environment": [{} for _ in range(total)],
        "successful_trajectory_metrics": {
            "tracking_mpjpe_mm": {"mean": mpjpe, "count": successes * 10},
            "tracked_body_pos_error_m": {"mean": 0.1, "count": successes * 10},
        },
        "metrics": {"tracking_mpjpe_mm": {"mean": mpjpe + 1, "count": total * 10}},
    }


def test_budget_curve_detects_two_stable_intervals() -> None:
    curve = aggregate_milestone_evaluations(
        {
            2000: [_summary(90, 100, 20.0)],
            4000: [_summary(90, 100, 19.4)],
            6000: [_summary(90, 100, 19.0)],
        }
    )
    assert curve["rows"][0]["success_rate"] == 0.9
    assert curve["rows"][2]["plateau_candidate"] is True
    assert curve["plateau_candidate_update"] == 6000


def test_eval_command_binds_explicit_goal_to_trajectory_rank() -> None:
    command = _eval_command(
        _eval_args(),
        python=["python"],
        goal="goal_motion",
        goal_rank=7,
        checkpoint=Path("planner.pt"),
        output_dir=Path("eval"),
    )
    rank_index = command.index("--trajectory_ranks")
    assert command[rank_index + 1] == "7"
    assert "--require_goal_motion_match" in command
    assert "--disable_push_event" in command
    assert "--sonic_success_terminations" in command
    assert "--flow_inference_noise_std" in command
    noise_index = command.index("--flow_inference_noise_std")
    assert command[noise_index + 1] == "0.0"


def test_discrete_fsq64_geometry_reaches_both_isaac_stages() -> None:
    cells = [2048, 2048, 1024, 1024, 512, 512]
    args = _eval_args(
        skill_z_dim=64,
        policy_num_cells=cells,
        policy_activation="silu",
        solver_njmax=320,
        trajectories_per_motion=100,
        collection_max_steps=1200,
        sample_rows_per_file=8192,
        future_window_frames=30,
    )
    rendered = "[" + ",".join(str(width) for width in cells) + "]"
    for command in (
        _eval_command(
            args,
            python=["python"],
            goal="goal_motion",
            goal_rank=0,
            checkpoint=Path("planner.pt"),
            output_dir=Path("eval"),
        ),
        _collection_command(
            args,
            python=["python"],
            names=["goal_motion"],
            num_envs=100,
            output_dir=Path("collection"),
        ),
    ):
        assert "env.command_interface.actor.dim=66" in command
        assert "agent.ipmd.latent_dim=66" in command
        assert "agent.ipmd.latent_learning.code_latent_dim=64" in command
        assert f"agent.policy.num_cells={rendered}" in command
        assert f"agent.value_function.num_cells={rendered}" in command
        assert "agent.policy.activation_fn=silu" in command
        assert "env.sim.physics.solver_cfg.njmax=320" in command
        assert not any(item.endswith("actor.dim=258") for item in command)


def test_train_command_resumes_from_latest_optimizer_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "planner"
    latest = output_dir / "checkpoints" / "latest.pt"
    latest.parent.mkdir(parents=True)
    latest.touch()
    args = argparse.Namespace(
        output_root=tmp_path,
        model_size="medium",
        seed=0,
        num_updates=20000,
        milestone_interval=2000,
        batch_size=256,
        micro_batch_size=32,
        lr=1.0e-4,
        weight_decay=1.0e-4,
        flow_steps=16,
        endpoint_steps=4,
        resume=True,
    )
    command = _train_command(args, python=["python"], output_dir=output_dir)
    resume_index = command.index("--resume_checkpoint")
    assert command[resume_index + 1] == str(latest)
