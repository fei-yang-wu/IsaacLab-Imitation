"""Compare the 670D- and 380D-input latent oracles on one matched protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_capacity_entry import _runtime_env


ROWS = {
    "full_body_670": {
        "macro_terms": "[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]",
    },
    "root_qpos_380": {
        "macro_terms": ("[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]"),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    print("[CMD]", " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def _runtime_python(env: dict[str, str]) -> str:
    configured = env.get("ISAAC_PY", "")
    if configured:
        return configured
    candidates = []
    runtime_root = os.environ.get("ISAACLAB_CU130_RUNTIME_ROOT", "")
    if runtime_root:
        candidates.append(Path(runtime_root) / "bin" / "python")
    candidates.append(
        Path(
            "/opt/isaaclab-imitation-runtime-spec/.pixi/envs/"
            "container-runtime/bin/python"
        )
    )
    candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Could not resolve an IsaacLab runtime Python interpreter.")


def _summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "task": payload["metadata"]["task"],
        "num_envs": int(payload["num_envs"]),
        "steps_run": int(payload["steps_run"]),
        "valid_transition_count": int(payload["aggregate"]["valid_transition_count"]),
        "tracking_success_rate": float(payload["aggregate"]["tracking_success_rate"]),
        "threshold_tracking_success_rate": float(
            payload["aggregate"]["threshold_tracking_success_rate"]
        ),
        "root_relative_mpjpe_mm": float(
            payload["metrics"]["tracking_mpjpe_mm"]["mean"]
        ),
        "early_terminations_enabled": bool(
            payload["metadata"]["early_terminations_enabled"]
        ),
        "deterministic_tracking": payload["metadata"]["deterministic_tracking"],
        "push_perturbation": payload["metadata"]["push_perturbation"],
        "termination_cause_env_counts": payload["aggregate"][
            "termination_cause_env_counts"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--full_body_checkpoint", type=Path, required=True)
    parser.add_argument("--full_body_skill_checkpoint", type=Path, required=True)
    parser.add_argument("--root_qpos_checkpoint", type=Path, required=True)
    parser.add_argument("--root_qpos_skill_checkpoint", type=Path, required=True)
    parser.add_argument("--full_body_checkpoint_sha256", required=True)
    parser.add_argument("--full_body_skill_sha256", required=True)
    parser.add_argument("--root_qpos_checkpoint_sha256", required=True)
    parser.add_argument("--root_qpos_skill_sha256", required=True)
    parser.add_argument("--num_envs", type=int, default=40)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.num_envs != 40 or args.max_steps != 1000 or args.seed != 0:
        parser.error("The matched diagnostic is fixed to 40 envs, 1000 steps, seed 0.")
    if args.output_root.exists() and not args.dry_run:
        parser.error(f"Refusing existing output root: {args.output_root}")

    row_paths = {
        "full_body_670": (
            args.full_body_checkpoint,
            args.full_body_skill_checkpoint,
            args.full_body_checkpoint_sha256,
            args.full_body_skill_sha256,
        ),
        "root_qpos_380": (
            args.root_qpos_checkpoint,
            args.root_qpos_skill_checkpoint,
            args.root_qpos_checkpoint_sha256,
            args.root_qpos_skill_sha256,
        ),
    }
    if not args.dry_run:
        for label, (
            checkpoint,
            skill,
            expected_checkpoint,
            expected_skill,
        ) in row_paths.items():
            if not checkpoint.is_file() or not skill.is_file():
                parser.error(f"{label} checkpoint input is missing.")
            actual_checkpoint = _sha256(checkpoint)
            actual_skill = _sha256(skill)
            if actual_checkpoint != expected_checkpoint:
                parser.error(
                    f"{label} policy SHA mismatch: {actual_checkpoint} "
                    f"!= {expected_checkpoint}"
                )
            if actual_skill != expected_skill:
                parser.error(
                    f"{label} encoder SHA mismatch: {actual_skill} != {expected_skill}"
                )
        if not args.manifest.is_file() or not args.dataset_path.is_dir():
            parser.error("The corrected manifest or dataset cache is missing.")
        args.output_root.mkdir(parents=True)

    evaluator = Path("scripts/rlopt/eval_skill_commander_closed_loop.py")
    runtime_env = _runtime_env()
    runtime_python = _runtime_python(runtime_env)
    binding_script = Path(
        "experiments/campaigns/2026-07-23-bones-phase5-language-local10/"
        "interface_baselines/validate_latent_skill_checkpoint_binding.py"
    )
    for label, spec in ROWS.items():
        checkpoint, skill, _, _ = row_paths[label]
        row_root = args.output_root / label
        binding_path = row_root / "skill_binding.json"
        _run(
            [
                runtime_python,
                str(binding_script),
                "--low_level_checkpoint",
                str(checkpoint),
                "--skill_checkpoint",
                str(skill),
                "--output_json",
                str(binding_path),
            ],
            dry_run=args.dry_run,
            env=runtime_env,
        )
        common = [
            runtime_python,
            str(evaluator),
            "--headless",
            "--device",
            "cuda:0",
            "--task",
            "Isaac-Imitation-G1-Latent-Strict-v0",
            "--algorithm",
            "IPMD",
            "--checkpoint",
            str(checkpoint),
            "--skill_checkpoint",
            str(skill),
            "--state_history_steps",
            "9",
            "--num_envs",
            str(args.num_envs),
            "--max_steps",
            str(args.max_steps),
            "--seed",
            str(args.seed),
            "--keep_time_out",
            "--extend_episode_length_for_max_steps",
            "--disable_reward_clipping",
            "--kit_args=--/app/extensions/fsWatcherEnabled=false",
            "agent.logger.backend=",
            "agent.ipmd.command_source=hl_skill",
            f"agent.ipmd.hl_skill_checkpoint_path={skill}",
            "agent.ipmd.hl_skill_finetune_enabled=false",
            f"env.lafan1_manifest_path={args.manifest}",
            f"env.dataset_path={args.dataset_path}",
            "env.refresh_zarr_dataset=false",
            "env.reset_schedule=sequential",
            "env.wrap_steps=false",
            "env.random_reset_step_min=0",
            "env.random_reset_step_max=0",
            "env.random_reset_full_trajectory=false",
            "env.reference_start_frame=0",
            "env.observations.policy.enable_corruption=false",
            f"env.expert_macro_state_terms={spec['macro_terms']}",
            "env.latent_command_dim=258",
            "agent.ipmd.latent_dim=258",
            "agent.ipmd.hl_skill_horizon_steps=10",
            "agent.ipmd.hl_skill_command_mode=z",
            "agent.ipmd.latent_steps_min=10",
            "agent.ipmd.latent_steps_max=10",
            "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
            "agent.ipmd.latent_learning.code_latent_dim=256",
            "agent.ipmd.latent_learning.code_period=10",
            "agent.ipmd.reward_loss_coeff=0.0",
            "agent.ipmd.reward_l2_coeff=0.0",
            "agent.ipmd.reward_grad_penalty_coeff=0.0",
            "agent.ipmd.reward_logit_reg_coeff=0.0",
            "agent.ipmd.reward_param_weight_decay_coeff=0.0",
            "physics=newton_mjwarp",
            "env.sim.physics.solver_cfg.njmax=320",
            "env.sim.physics.solver_cfg.nconmax=40",
        ]
        _run(
            [
                *common,
                "--metric_interval",
                "1",
                "--keep_early_terminations",
                "--output_dir",
                str(row_root / "strict_oracle"),
                "--label",
                f"{label}_strict_oracle",
            ],
            dry_run=args.dry_run,
            env=runtime_env,
        )
        _run(
            [
                *common,
                "--metric_interval",
                "10",
                "--deterministic_tracking",
                "--video",
                "--video_length",
                str(args.max_steps),
                "--output_dir",
                str(row_root / "full_horizon_deterministic"),
                "--label",
                f"{label}_full_horizon_deterministic",
            ],
            dry_run=args.dry_run,
            env=runtime_env,
        )

    if args.dry_run:
        return 0

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "task": "Isaac-Imitation-G1-Latent-Strict-v0",
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "manifest": str(args.manifest.resolve()),
            "dataset_path": str(args.dataset_path.resolve()),
            "oracle_command_source": "hl_skill",
        },
        "rows": {},
    }
    for label, (checkpoint, skill, _, _) in row_paths.items():
        row_root = args.output_root / label
        result["rows"][label] = {
            "macro_terms": ROWS[label]["macro_terms"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "skill_checkpoint": str(skill.resolve()),
            "skill_checkpoint_sha256": _sha256(skill),
            "strict": _summary(row_root / "strict_oracle" / "summary.json"),
            "full_horizon_deterministic": _summary(
                row_root / "full_horizon_deterministic" / "summary.json"
            ),
        }
        video = (
            row_root
            / "full_horizon_deterministic"
            / "videos"
            / "play"
            / "rl-video-step-0.mp4"
        )
        if not video.is_file():
            raise RuntimeError(f"Retained diagnostic video missing: {video}")
        print(f"[RESULT] retained video: {video.resolve()}", flush=True)
    output = args.output_root / "comparison.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[RESULT] comparison: {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
