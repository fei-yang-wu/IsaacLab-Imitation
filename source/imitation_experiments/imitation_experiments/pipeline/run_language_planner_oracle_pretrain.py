"""Collect complete oracle-policy trajectories, pretrain, and score milestones."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence

from imitation_experiments.paths import REPO_ROOT


HORIZON_STEPS = 10
STATE_HISTORY_STEPS = 9
ROOT_QPOS_DIM = 38
SKILL_Z_DIM = 256
PHASE_CHANNELS = 2
LATENT_COMMAND_DIM = SKILL_Z_DIM + PHASE_CHANNELS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low_level_checkpoint", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language_embeddings", type=Path, required=True)
    parser.add_argument("--reference_arrays_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_arrays_persist_id",
        default="bones_seed_language10_v1@60a5b7a5",
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "collect", "train", "eval"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skill_z_dim",
        type=int,
        default=SKILL_Z_DIM,
        help="Skill-code width. The actor command adds two sin/cos phase channels.",
    )
    parser.add_argument(
        "--policy_num_cells",
        type=int,
        nargs="+",
        default=None,
        help="Actor/critic hidden widths of the frozen tracker. Omit for the "
        "agent entry-point default.",
    )
    parser.add_argument(
        "--policy_activation",
        default=None,
        help="Actor/critic activation of the frozen tracker, for example silu.",
    )
    parser.add_argument("--solver_njmax", type=int, default=289)
    parser.add_argument("--solver_nconmax", type=int, default=200)
    parser.add_argument("--trajectories_per_motion", type=int, default=100)
    parser.add_argument(
        "--collection_num_envs",
        type=int,
        default=0,
        help="Zero means motions * trajectories_per_motion (1,000 for ten motions).",
    )
    parser.add_argument("--collection_max_steps", type=int, default=1200)
    parser.add_argument("--future_window_frames", type=int, default=30)
    parser.add_argument("--sample_rows_per_file", type=int, default=8192)
    parser.add_argument(
        "--model_size", choices=("tiny", "small", "medium", "large"), default="medium"
    )
    parser.add_argument("--num_updates", type=int, default=10000)
    parser.add_argument("--milestone_interval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--endpoint_steps", type=int, default=4)
    parser.add_argument("--eval_trajectories_per_motion", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python_cmd", default="pixi run python")
    parser.add_argument("--isaaclab_python_cmd", default="pixi run -e isaaclab python")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _manifest_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no motion entries: {path}.")
    names = [str(entry.get("name", "")).strip() for entry in entries]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Manifest motion names must be unique and non-empty.")
    return names


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


class Runner:
    def __init__(self, *, dry_run: bool, resume: bool) -> None:
        self.dry_run = bool(dry_run)
        self.resume = bool(resume)

    def run(self, command: Sequence[str], *, expected: Path | None = None) -> None:
        if expected is not None and expected.exists() and self.resume:
            print(f"[SKIP] Existing output: {expected}", flush=True)
            return
        command = [str(item) for item in command]
        print(f"[CMD] {shlex.join(command)}", flush=True)
        if self.dry_run:
            return
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if expected is not None and not expected.exists():
            raise FileNotFoundError(f"Command did not create {expected}.")


def _tracker_overrides(args: argparse.Namespace) -> list[str]:
    """Rebuild the frozen tracker's geometry so a strict restore succeeds."""
    overrides: list[str] = []
    cells = getattr(args, "policy_num_cells", None)
    if cells:
        rendered = "[" + ",".join(str(int(width)) for width in cells) + "]"
        overrides.extend(
            [
                f"agent.policy.num_cells={rendered}",
                f"agent.value_function.num_cells={rendered}",
            ]
        )
    activation = str(getattr(args, "policy_activation", None) or "").strip()
    if activation:
        overrides.extend(
            [
                f"agent.policy.activation_fn={activation}",
                f"agent.value_function.activation_fn={activation}",
            ]
        )
    return overrides


def _hydra_overrides(
    *,
    reference_arrays_dir: Path,
    reference_arrays_persist_id: str,
    skill_checkpoint: Path,
    command_source: str,
    skill_z_dim: int = SKILL_Z_DIM,
    tracker_overrides: Sequence[str] = (),
    solver_njmax: int = 289,
    solver_nconmax: int = 200,
    planner_checkpoint: Path | None = None,
    language_embeddings: Path | None = None,
    goal_name: str | None = None,
    flow_steps: int = 16,
) -> list[str]:
    skill_z_dim = int(skill_z_dim)
    latent_command_dim = skill_z_dim + PHASE_CHANNELS
    overrides = [
        "physics=newton_mjwarp",
        "env.data.manifest=null",
        "env.data.cache_dir=null",
        f"env.data.reference_arrays_dir={reference_arrays_dir}",
        f"env.data.persist_id={reference_arrays_persist_id}",
        "env.data.persist_dir=null",
        "env.data.reference_arrays_warm_workers=2",
        "env.data.macro_cache_device=cuda:0",
        "env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]",
        "env.data.wrap_steps=false",
        f"env.command_interface.actor.dim={latent_command_dim}",
        "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]",
        "agent.logger.backend=",
        f"agent.ipmd.command_source={command_source}",
        f"agent.ipmd.hl_skill_checkpoint_path={skill_checkpoint}",
        "agent.ipmd.hl_skill_finetune_enabled=false",
        f"agent.ipmd.latent_dim={latent_command_dim}",
        f"agent.ipmd.latent_steps_min={HORIZON_STEPS}",
        f"agent.ipmd.latent_steps_max={HORIZON_STEPS}",
        f"agent.ipmd.hl_skill_horizon_steps={HORIZON_STEPS}",
        "agent.ipmd.hl_skill_command_mode=z",
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
        f"agent.ipmd.latent_learning.code_latent_dim={skill_z_dim}",
        f"agent.ipmd.latent_learning.code_period={HORIZON_STEPS}",
        f"env.sim.physics.solver_cfg.njmax={int(solver_njmax)}",
        f"env.sim.physics.solver_cfg.nconmax={int(solver_nconmax)}",
        *tracker_overrides,
    ]
    if command_source == "skill_commander":
        if planner_checkpoint is None or language_embeddings is None or not goal_name:
            raise ValueError(
                "Planner control requires checkpoint, language table, and goal."
            )
        overrides.extend(
            [
                f"agent.ipmd.skill_commander_checkpoint_path={planner_checkpoint}",
                f"agent.ipmd.skill_commander_embeddings_path={language_embeddings}",
                f"agent.ipmd.skill_commander_goal_name={goal_name}",
                "agent.ipmd.skill_commander_use_achieved_state=true",
                f"agent.ipmd.skill_commander_flow_num_inference_steps={flow_steps}",
                "agent.ipmd.skill_commander_flow_inference_noise_std=0.0",
            ]
        )
    return overrides


def _collection_command(
    args: argparse.Namespace,
    *,
    python: list[str],
    names: list[str],
    num_envs: int,
    output_dir: Path,
) -> list[str]:
    command = [
        *python,
        "scripts/rlopt/eval_skill_commander_closed_loop.py",
        "--headless",
        "--task",
        "Isaac-Imitation-G1-v2",
        "--algorithm",
        "IPMD",
        "--agent_entry_point",
        "rlopt_ipmd_tuned_cfg_entry_point",
        "--checkpoint",
        str(args.low_level_checkpoint),
        "--skill_checkpoint",
        str(args.skill_checkpoint),
        "--language_embeddings",
        str(args.language_embeddings),
        "--state_history_steps",
        str(STATE_HISTORY_STEPS),
        "--output_dir",
        str(output_dir),
        "--label",
        "language_oracle_trajectory_collection",
        "--num_envs",
        str(num_envs),
        "--max_steps",
        str(args.collection_max_steps),
        "--seed",
        str(args.seed),
        "--metric_interval",
        str(max(int(args.collection_max_steps), 1)),
        "--motion_names",
        *names,
        "--trajectory_ranks",
        *[str(rank) for rank in range(len(names))],
        "--balanced_motion_names",
        *names,
        "--balanced_trajectories_per_motion",
        str(args.trajectories_per_motion),
        "--save_rollout_training_samples",
        "--sample_rows_per_file",
        str(args.sample_rows_per_file),
        "--sample_future_window_frames",
        str(args.future_window_frames),
        "--require_root_qpos_samples",
        "--sonic_success_terminations",
        "--disable_push_event",
        "--disable_reward_clipping",
        "--assert-kitless",
        *_hydra_overrides(
            reference_arrays_dir=args.reference_arrays_dir,
            reference_arrays_persist_id=args.reference_arrays_persist_id,
            skill_checkpoint=args.skill_checkpoint,
            command_source="hl_skill",
            skill_z_dim=int(args.skill_z_dim),
            tracker_overrides=_tracker_overrides(args),
            solver_njmax=int(args.solver_njmax),
            solver_nconmax=int(args.solver_nconmax),
        ),
    ]
    return command


def _train_command(
    args: argparse.Namespace, *, python: list[str], output_dir: Path
) -> list[str]:
    log_interval = (
        100 if int(args.milestone_interval) % 100 == 0 else int(args.milestone_interval)
    )
    command = [
        *python,
        "-m",
        "imitation_experiments.planner.train_chunked_transformer_planner",
        "--samples_dir",
        str(args.output_root / "collection" / "rollout_training_samples"),
        "--output_dir",
        str(output_dir),
        "--interface",
        "latent_skill",
        "--state_key",
        "planner_state",
        "--training_stage",
        "oracle",
        "--model_size",
        str(args.model_size),
        "--seed",
        str(args.seed),
        "--num_updates",
        str(args.num_updates),
        "--milestone_interval",
        str(args.milestone_interval),
        "--log_interval",
        str(log_interval),
        "--batch_size",
        str(args.batch_size),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--flow_num_inference_steps",
        str(args.flow_steps),
        "--endpoint_num_inference_steps",
        str(args.endpoint_steps),
        "--val_trajectory_fraction",
        "0.2",
        "--val_split_seed",
        "0",
    ]
    latest_checkpoint = output_dir / "checkpoints" / "latest.pt"
    if bool(args.resume) and latest_checkpoint.is_file():
        command.extend(["--resume_checkpoint", str(latest_checkpoint)])
    return command


def _eval_command(
    args: argparse.Namespace,
    *,
    python: list[str],
    goal: str,
    goal_rank: int,
    checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    return [
        *python,
        "scripts/rlopt/eval_skill_commander_closed_loop.py",
        "--headless",
        "--task",
        "Isaac-Imitation-G1-v2",
        "--algorithm",
        "IPMD",
        "--agent_entry_point",
        "rlopt_ipmd_tuned_cfg_entry_point",
        "--checkpoint",
        str(args.low_level_checkpoint),
        "--planner_checkpoint",
        str(checkpoint),
        "--skill_checkpoint",
        str(args.skill_checkpoint),
        "--language_embeddings",
        str(args.language_embeddings),
        "--state_history_steps",
        str(STATE_HISTORY_STEPS),
        "--output_dir",
        str(output_dir),
        "--label",
        f"oracle_pretrain_{checkpoint.stem}_{_slug(goal)}",
        "--num_envs",
        str(args.eval_trajectories_per_motion),
        "--max_steps",
        "0",
        "--seed",
        str(args.seed),
        "--metric_interval",
        "100",
        "--motion_name",
        goal,
        "--trajectory_ranks",
        str(goal_rank),
        "--require_goal_motion_match",
        "--sonic_success_terminations",
        "--disable_push_event",
        "--disable_reward_clipping",
        "--flow_num_inference_steps",
        str(args.flow_steps),
        "--flow_inference_noise_std",
        "0.0",
        "--assert-kitless",
        *_hydra_overrides(
            reference_arrays_dir=args.reference_arrays_dir,
            reference_arrays_persist_id=args.reference_arrays_persist_id,
            skill_checkpoint=args.skill_checkpoint,
            command_source="skill_commander",
            skill_z_dim=int(args.skill_z_dim),
            tracker_overrides=_tracker_overrides(args),
            solver_njmax=int(args.solver_njmax),
            solver_nconmax=int(args.solver_nconmax),
            planner_checkpoint=checkpoint,
            language_embeddings=args.language_embeddings,
            goal_name=goal,
            flow_steps=int(args.flow_steps),
        ),
    ]


def _weighted_metric(
    summaries: Sequence[dict[str, Any]], *, section: str, metric: str
) -> float:
    total = 0.0
    count = 0
    for summary in summaries:
        row = summary.get(section, {}).get(metric, {})
        metric_count = int(row.get("count", 0)) if isinstance(row, dict) else 0
        value = row.get("mean") if isinstance(row, dict) else None
        if metric_count > 0 and value is not None and math.isfinite(float(value)):
            total += float(value) * metric_count
            count += metric_count
    return total / count if count else float("nan")


def aggregate_milestone_evaluations(
    milestone_summaries: dict[int, list[dict[str, Any]]],
    *,
    training_metrics: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a closed-loop budget curve and a conservative plateau diagnostic."""
    rows: list[dict[str, Any]] = []
    stable_intervals = 0
    for update in sorted(milestone_summaries):
        summaries = milestone_summaries[update]
        env_count = sum(
            len(summary.get("per_environment", [])) for summary in summaries
        )
        successes = sum(
            int(summary.get("aggregate", {}).get("completed_tracking_success_count", 0))
            for summary in summaries
        )
        row: dict[str, Any] = {
            "update": int(update),
            "motion_count": len(summaries),
            "trajectory_count": env_count,
            "completed_success_count": successes,
            "success_rate": successes / env_count if env_count else float("nan"),
            "mpjpe_l_mm_successful": _weighted_metric(
                summaries,
                section="successful_trajectory_metrics",
                metric="tracking_mpjpe_mm",
            ),
            "mpjpe_g_m_successful": _weighted_metric(
                summaries,
                section="successful_trajectory_metrics",
                metric="tracked_body_pos_error_m",
            ),
            "mpjpe_l_mm_all": _weighted_metric(
                summaries, section="metrics", metric="tracking_mpjpe_mm"
            ),
        }
        if training_metrics and update in training_metrics:
            row["offline_validation"] = training_metrics[update]
        if rows:
            previous = rows[-1]
            row["success_rate_delta"] = row["success_rate"] - previous["success_rate"]
            row["mpjpe_l_mm_successful_delta"] = (
                row["mpjpe_l_mm_successful"] - previous["mpjpe_l_mm_successful"]
            )
            stable = (
                abs(row["success_rate_delta"]) < 0.01
                and abs(row["mpjpe_l_mm_successful_delta"]) < 1.0
            )
            stable_intervals = stable_intervals + 1 if stable else 0
        row["plateau_candidate"] = stable_intervals >= 2
        rows.append(row)
    plateau_update = next(
        (int(row["update"]) for row in rows if row["plateau_candidate"]), None
    )
    return {
        "schema": "language_planner_oracle_pretrain_budget_curve_v1",
        "success_definition": "reference finished without SONIC tracking termination",
        "mpjpe_definition": "root-relative MPJPE over successful trajectories",
        "plateau_heuristic": (
            "two consecutive 2k intervals with |delta SR| < 0.01 and "
            "|delta success-only MPJPE-L| < 1 mm"
        ),
        "plateau_candidate_update": plateau_update,
        "rows": rows,
    }


def _training_metrics_by_update(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[int(row["update"])] = {
            key: value for key, value in row.items() if str(key).startswith("val/")
        }
    return result


def _write_curve(curve: dict[str, Any], output_root: Path) -> None:
    json_path = output_root / "milestone_curve.json"
    json_path.write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Oracle-pretrained planner budget curve",
        "",
        "| update | SONIC SR | successful MPJPE-L (mm) | successful MPJPE-G (m) | plateau? |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in curve["rows"]:
        lines.append(
            f"| {row['update']} | {row['success_rate']:.3f} | "
            f"{row['mpjpe_l_mm_successful']:.2f} | "
            f"{row['mpjpe_g_m_successful']:.4f} | "
            f"{'yes' if row['plateau_candidate'] else 'no'} |"
        )
    (output_root / "milestone_curve.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    for field in (
        "low_level_checkpoint",
        "skill_checkpoint",
        "manifest",
        "language_embeddings",
        "reference_arrays_dir",
        "output_root",
    ):
        setattr(args, field, _resolve(getattr(args, field)))
    for artifact in (
        args.low_level_checkpoint,
        args.skill_checkpoint,
        args.manifest,
        args.language_embeddings,
        args.reference_arrays_dir / "reference_arrays_manifest.json",
    ):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    names = _manifest_names(args.manifest)
    if args.trajectories_per_motion <= 0 or args.eval_trajectories_per_motion <= 0:
        raise ValueError("Trajectory budgets must be positive.")
    expected_envs = len(names) * int(args.trajectories_per_motion)
    collection_num_envs = int(args.collection_num_envs) or expected_envs
    if collection_num_envs != expected_envs:
        raise ValueError(
            "One complete episode per environment requires collection_num_envs == "
            f"motions * trajectories_per_motion ({expected_envs}), got "
            f"{collection_num_envs}."
        )
    if int(args.skill_z_dim) <= 0:
        raise ValueError("skill_z_dim must be positive.")
    if args.policy_num_cells is not None and any(
        int(width) <= 0 for width in args.policy_num_cells
    ):
        raise ValueError("policy_num_cells widths must be positive.")
    if args.num_updates <= 0 or args.milestone_interval <= 0:
        raise ValueError("num_updates and milestone_interval must be positive.")
    if args.num_updates % args.milestone_interval:
        raise ValueError("num_updates must be divisible by milestone_interval.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    runner = Runner(dry_run=args.dry_run, resume=args.resume)
    python = shlex.split(args.python_cmd)
    isaaclab_python = shlex.split(args.isaaclab_python_cmd)
    binding = args.output_root / "latent_skill_binding.json"
    runner.run(
        [
            *python,
            "-m",
            "imitation_experiments.audit.validate_latent_skill_checkpoint_binding",
            "--low_level_checkpoint",
            str(args.low_level_checkpoint),
            "--skill_checkpoint",
            str(args.skill_checkpoint),
            "--output_json",
            str(binding),
        ],
        expected=binding,
    )

    stages = {args.stage} if args.stage != "all" else {"collect", "train", "eval"}
    collection_dir = args.output_root / "collection"
    if "collect" in stages:
        runner.run(
            _collection_command(
                args,
                python=isaaclab_python,
                names=names,
                num_envs=collection_num_envs,
                output_dir=collection_dir,
            ),
            expected=collection_dir / "summary.json",
        )
    train_dir = args.output_root / "planner_oracle_pretrain"
    if "train" in stages:
        runner.run(
            _train_command(args, python=python, output_dir=train_dir),
            expected=(
                train_dir / "checkpoints" / f"update_{int(args.num_updates):07d}.pt"
            ),
        )
    if "eval" in stages:
        milestone_summaries: dict[int, list[dict[str, Any]]] = {}
        for update in range(
            int(args.milestone_interval),
            int(args.num_updates) + 1,
            int(args.milestone_interval),
        ):
            checkpoint = train_dir / "checkpoints" / f"update_{update:07d}.pt"
            if not args.dry_run and not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            summaries: list[dict[str, Any]] = []
            for goal_rank, goal in enumerate(names):
                output_dir = (
                    args.output_root
                    / "milestone_eval"
                    / f"update_{update:07d}"
                    / _slug(goal)
                )
                summary_path = output_dir / "summary.json"
                runner.run(
                    _eval_command(
                        args,
                        python=isaaclab_python,
                        goal=goal,
                        goal_rank=goal_rank,
                        checkpoint=checkpoint,
                        output_dir=output_dir,
                    ),
                    expected=summary_path,
                )
                if not args.dry_run:
                    summaries.append(
                        json.loads(summary_path.read_text(encoding="utf-8"))
                    )
            if not args.dry_run:
                milestone_summaries[update] = summaries
        if not args.dry_run:
            curve = aggregate_milestone_evaluations(
                milestone_summaries,
                training_metrics=_training_metrics_by_update(
                    train_dir / "metrics.jsonl"
                ),
            )
            _write_curve(curve, args.output_root)
            print(f"[PASS] Budget curve: {args.output_root / 'milestone_curve.md'}")


if __name__ == "__main__":
    main()
