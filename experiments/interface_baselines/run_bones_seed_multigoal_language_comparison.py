#!/usr/bin/env python3
"""Run the shared multi-goal BONES-SEED latent versus explicit comparison."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
HORIZON_STEPS = 10
STATE_HISTORY_STEPS = 9
LATENT_DIM = 258
SKILL_Z_DIM = 256
DEMONSTRATION_COLLECTION_SAFETY_MULTIPLIER = 8
PROTOCOL = "bones_seed_shared_multigoal_language_v1"
WORKFLOW_SOURCE_PATHS = (
    "experiments/interface_baselines/run_bones_seed_multigoal_language_comparison.py",
    "experiments/interface_baselines/submit_bones_seed_multigoal_pipeline_skynet.sh",
    "experiments/interface_baselines/submit_bones_seed_multiseed_pipeline_skynet.sh",
    "docker/cluster/submit_job_slurm_bones_pipeline.sh",
    "experiments/interface_baselines/validate_bones_seed_planner_submission.py",
    "experiments/interface_baselines/collect_interface_rollout_samples.py",
    "experiments/interface_baselines/balanced_motion_rows.py",
    "experiments/interface_baselines/closed_loop_metrics.py",
    "experiments/interface_baselines/low_level_tracker.py",
    "experiments/interface_baselines/planner_publish_schedule.py",
    "experiments/interface_baselines/paper_protocol_metadata.py",
    "experiments/interface_baselines/planner_latency.py",
    "scripts/rlopt/eval_skill_commander_closed_loop.py",
    "experiments/interface_baselines/eval_interface_planner_closed_loop.py",
    "experiments/interface_baselines/train_chunked_transformer_planner.py",
    "experiments/interface_baselines/eval_interface_planner_offline.py",
    "experiments/interface_baselines/merge_planner_samples.py",
    "experiments/interface_baselines/planner_sample_schema.py",
    "experiments/interface_baselines/interface_planner_common.py",
    "experiments/interface_baselines/audit_bones_seed_multigoal_language_comparison.py",
    "experiments/interface_baselines/summarize_bones_seed_multigoal_language_comparison.py",
    "RLOpt/rlopt/agent/causal_interface_planner.py",
    "RLOpt/rlopt/agent/skill_commander.py",
    "source/isaaclab_imitation/isaaclab_imitation/envs/causal_planner_observation.py",
    "source/isaaclab_imitation/isaaclab_imitation/envs/imitation_rl_env.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_low_level_checkpoint", type=Path, required=True)
    parser.add_argument("--latent_skill_checkpoint", type=Path, required=True)
    parser.add_argument("--vanilla_tracker_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/bones_seed/manifests/g1_bones_seed_10_manifest.json"),
    )
    parser.add_argument(
        "--language_embeddings",
        type=Path,
        default=Path(
            "data/bones_seed/language/g1_bones_seed_10_minilm_goal_embeddings.pt"
        ),
    )
    parser.add_argument("--latent_dataset_path", type=Path, required=True)
    parser.add_argument("--vanilla_dataset_path", type=Path, required=True)
    parser.add_argument(
        "--preparation_record",
        type=Path,
        default=None,
        help="Optional fresh-export preparation.json; required by the cluster wrapper.",
    )
    parser.add_argument("--vanilla_qualification_audit", type=Path, default=None)
    parser.add_argument("--latent_qualification_audit", type=Path, default=None)
    parser.add_argument("--streamed_equivalence_certificate", type=Path, default=None)
    parser.add_argument("--min_oracle_success", type=float, default=0.8)
    parser.add_argument(
        "--interfaces",
        nargs="+",
        choices=("latent_skill", "full_body_trajectory"),
        default=["latent_skill", "full_body_trajectory"],
        help=(
            "Interfaces to execute. The paper launcher keeps both; selecting only "
            "one interface is a preliminary diagnostic and is never paper-complete."
        ),
    )
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument(
        "--goal_names",
        nargs="*",
        default=None,
        help="Optional manifest motion names. The default uses every motion.",
    )
    parser.add_argument(
        "--goal_limit",
        type=int,
        default=0,
        help="Use only the first N selected goals; <=0 uses all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo_rows_per_goal", type=int, default=1000)
    parser.add_argument("--rollout_rows_per_goal", type=int, default=1000)
    parser.add_argument(
        "--rollout_num_envs",
        type=int,
        default=10,
        help=(
            "Parallel environments used only for planner-driven sample collection. "
            "Every environment is restricted to the same explicit goal and motion."
        ),
    )
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument(
        "--model_size",
        choices=("tiny", "small", "medium", "large"),
        default="medium",
    )
    parser.add_argument("--pretrain_updates", type=int, default=2000)
    parser.add_argument("--finetune_updates", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--train_endpoint_steps", type=int, default=4)
    parser.add_argument("--flow_noise_std", type=float, default=0.0)
    parser.add_argument(
        "--skip_pretrained_closed_loop",
        action="store_true",
        default=False,
        help="Skip the before-rollout closed-loop evaluation; useful only for smoke runs.",
    )
    parser.add_argument(
        "--refresh_datasets",
        action="store_true",
        default=False,
        help="Rebuild each manifest-backed dataset on the first collection call.",
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "rollout", "finetune", "final-eval", "summarize"),
        default="all",
        help=(
            "Run the full workflow or one resumable stage. Rollout and final-eval "
            "require one explicit --goal_index or SLURM_ARRAY_TASK_ID."
        ),
    )
    parser.add_argument(
        "--goal_index",
        type=int,
        default=None,
        help="Zero-based explicit goal index for rollout or final-eval stages.",
    )
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--python_cmd", default="pixi run python")
    parser.add_argument("--isaaclab_python_cmd", default="pixi run -e isaaclab python")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _as_path(value: Any) -> Path:
    return Path(str(value or "")).expanduser().resolve()


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT
        / "logs"
        / "interface_baselines"
        / f"bones_seed_multigoal_language_{timestamp}"
    )


def _manifest_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no dataset.trajectories.lafan1_csv: {path}")
    names = [str(entry.get("name", "")).strip() for entry in entries]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Every manifest entry must have a unique non-empty name.")
    return names


def _select_goals(
    manifest_names: list[str], requested: list[str] | None, limit: int
) -> list[str]:
    goals = list(manifest_names) if not requested else [str(x) for x in requested]
    unknown = [goal for goal in goals if goal not in manifest_names]
    if unknown:
        raise ValueError(f"Requested goals are absent from the manifest: {unknown}")
    goals = list(dict.fromkeys(goals))
    if int(limit) > 0:
        goals = goals[: int(limit)]
    if not goals:
        raise ValueError("At least one goal is required.")
    return goals


def _validate_language_table(path: Path, goals: list[str]) -> dict[str, Any]:
    table = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(table, dict):
        raise TypeError(f"Language table is not a mapping: {path}")
    names = table.get("names")
    embeddings = table.get("embeddings")
    if not isinstance(names, list) or not isinstance(embeddings, torch.Tensor):
        raise ValueError("Language table requires names and embeddings.")
    missing = [goal for goal in goals if goal not in names]
    if missing:
        raise ValueError(f"Language table is missing selected goals: {missing}")
    if embeddings.ndim != 2 or int(embeddings.shape[0]) != len(names):
        raise ValueError("Language embedding table shape does not match its names.")
    return {
        "backend": table.get("backend"),
        "model": table.get("model"),
        "embedding_dim": int(embeddings.shape[-1]),
        "table_motion_count": len(names),
    }


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return result or "goal"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_sha256(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _validate_preparation_record(path: Path, manifest: Path) -> dict[str, Any]:
    record = _json_object(path)
    manifest_sha = _sha256(manifest)
    if record.get("status") != "complete":
        raise ValueError(f"BONES-SEED preparation is not complete: {path}")
    recorded_sha = record.get("artifacts", {}).get("manifest_sha256")
    if recorded_sha != manifest_sha:
        raise ValueError("Preparation record does not match the selected manifest.")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": "complete",
        "manifest_sha256": manifest_sha,
    }


def _validate_low_level_gates(
    *,
    vanilla_audit_path: Path,
    latent_audit_path: Path,
    equivalence_path: Path,
    manifest: Path,
    vanilla_checkpoint: Path,
    latent_checkpoint: Path,
    skill_checkpoint: Path,
    vanilla_dataset_path: Path,
    latent_dataset_path: Path,
    min_success: float,
) -> dict[str, Any]:
    vanilla = _json_object(vanilla_audit_path)
    latent = _json_object(latent_audit_path)
    equivalence = _json_object(equivalence_path)
    manifest_sha = _sha256(manifest)
    vanilla_sha = _sha256(vanilla_checkpoint)
    latent_sha = _sha256(latent_checkpoint)
    skill_sha = _sha256(skill_checkpoint)
    vanilla_success = float(vanilla.get("success_rate", float("nan")))
    latent_success = float(latent.get("tracking_success_rate", float("nan")))
    failures: list[str] = []
    if vanilla.get("protocol_passed") is not True:
        failures.append("vanilla qualification protocol failed")
    if vanilla.get("oracle_passed") is not True or vanilla_success < min_success:
        failures.append("vanilla oracle success gate failed")
    if vanilla.get("checkpoint_sha256") != vanilla_sha:
        failures.append("vanilla qualification checkpoint hash mismatch")
    if vanilla.get("manifest_sha256") != manifest_sha:
        failures.append("vanilla qualification manifest hash mismatch")
    if _as_path(vanilla.get("dataset_path")) != vanilla_dataset_path:
        failures.append("vanilla qualification dataset path mismatch")
    if (
        latent.get("protocol_passed") is not True
        or latent.get("oracle_passed") is not True
        or latent_success < min_success
    ):
        failures.append("latent oracle success gate failed")
    if latent.get("low_level_checkpoint_sha256") != latent_sha:
        failures.append("latent qualification checkpoint hash mismatch")
    if latent.get("skill_checkpoint_sha256") != skill_sha:
        failures.append("latent qualification skill checkpoint hash mismatch")
    skill_binding = latent.get("low_level_skill_binding", {})
    if (
        skill_binding.get("passed") is not True
        or skill_binding.get("low_level_checkpoint_sha256") != latent_sha
        or skill_binding.get("skill_checkpoint_sha256") != skill_sha
    ):
        failures.append("latent low-level and skill checkpoint binding mismatch")
    if latent.get("manifest_sha256") != manifest_sha:
        failures.append("latent qualification manifest hash mismatch")
    if _as_path(latent.get("dataset_path")) != latent_dataset_path:
        failures.append("latent qualification dataset path mismatch")
    tracker = equivalence.get("low_level_tracker", {})
    if equivalence.get("passed") is not True:
        failures.append("streamed-vanilla equivalence certificate failed")
    if tracker.get("checkpoint_sha256") != vanilla_sha:
        failures.append("equivalence certificate checkpoint hash mismatch")
    if equivalence.get("checkpoint_sha256") != vanilla_sha:
        failures.append("equivalence top-level checkpoint hash mismatch")
    if equivalence.get("motion_manifest_sha256") != manifest_sha:
        failures.append("equivalence manifest hash mismatch")
    if _as_path(equivalence.get("dataset_path")) != vanilla_dataset_path:
        failures.append("equivalence dataset path mismatch")
    if equivalence.get("observed_phases") != list(range(HORIZON_STEPS)):
        failures.append("equivalence certificate phase coverage mismatch")
    if equivalence.get("missing_phases") != []:
        failures.append("equivalence certificate does not cover all packet phases")
    if equivalence.get("asynchronous_rephase_exercised") is not True:
        failures.append("equivalence certificate did not exercise asynchronous renewal")
    if equivalence.get("policy_state_unchanged") is not True:
        failures.append("equivalence certificate changed tracker state")
    if failures:
        raise ValueError("; ".join(failures))
    return {
        "minimum_success": float(min_success),
        "vanilla_success_rate": vanilla_success,
        "latent_success_rate": latent_success,
        "manifest_sha256": manifest_sha,
        "vanilla_checkpoint_sha256": vanilla_sha,
        "latent_checkpoint_sha256": latent_sha,
        "skill_checkpoint_sha256": skill_sha,
        "vanilla_dataset_path": str(vanilla_dataset_path),
        "latent_dataset_path": str(latent_dataset_path),
        "vanilla_qualification_audit": {
            "path": str(vanilla_audit_path),
            "sha256": _sha256(vanilla_audit_path),
        },
        "latent_qualification_audit": {
            "path": str(latent_audit_path),
            "sha256": _sha256(latent_audit_path),
        },
        "streamed_equivalence_certificate": {
            "path": str(equivalence_path),
            "sha256": _sha256(equivalence_path),
        },
    }


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _repository_provenance() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    if not commit:
        return {
            "git_metadata_available": False,
            "snapshot_without_git_metadata": True,
            "working_tree_dirty": None,
            "status_short": [],
            "submodule_status": [],
        }
    status = _git_output("status", "--short")
    return {
        "git_metadata_available": True,
        "commit": commit,
        "working_tree_dirty": bool(status),
        "status_short": status.splitlines(),
        "submodule_status": _git_output(
            "submodule", "status", "--recursive"
        ).splitlines(),
    }


def _workflow_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in WORKFLOW_SOURCE_PATHS:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Workflow source is missing: {path}")
        hashes[relative_path] = _sha256(path)
    return hashes


class Runner:
    def __init__(self, args: argparse.Namespace, output_root: Path) -> None:
        self.args = args
        self.output_root = output_root
        self.python = shlex.split(str(args.python_cmd))
        self.isaaclab_python = shlex.split(str(args.isaaclab_python_cmd))
        self.failures: list[dict[str, Any]] = []

    def run(self, command: list[str], *, expected: Path | None = None) -> bool:
        if expected is not None and expected.exists() and self.args.resume:
            print(f"[SKIP] Existing output: {expected}")
            return True
        print(f"[CMD] {shlex.join(command)}", flush=True)
        if self.args.dry_run:
            return True
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            self.failures.append(
                {"returncode": int(exc.returncode), "command": command}
            )
            if not self.args.continue_on_error:
                raise
            return False
        if expected is not None and not expected.exists():
            raise FileNotFoundError(
                f"Command did not create expected output: {expected}"
            )
        return True


def _latent_hydra_overrides(
    *,
    manifest: Path,
    dataset_path: Path,
    skill_checkpoint: Path,
    refresh: bool,
) -> list[str]:
    return [
        f"agent.ipmd.hl_skill_checkpoint_path={skill_checkpoint}",
        "agent.logger.backend=",
        "agent.ipmd.hl_skill_finetune_enabled=false",
        f"env.lafan1_manifest_path={manifest}",
        f"env.dataset_path={dataset_path}",
        "env.reset_schedule=sequential",
        "env.wrap_steps=false",
        "env.observations.policy.enable_corruption=false",
        f"env.refresh_zarr_dataset={str(bool(refresh)).lower()}",
        f"env.latent_command_dim={LATENT_DIM}",
        f"agent.ipmd.latent_dim={LATENT_DIM}",
        f"agent.ipmd.latent_steps_min={HORIZON_STEPS}",
        f"agent.ipmd.latent_steps_max={HORIZON_STEPS}",
        f"agent.ipmd.hl_skill_horizon_steps={HORIZON_STEPS}",
        "agent.ipmd.hl_skill_command_mode=z",
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
        f"agent.ipmd.latent_learning.code_latent_dim={SKILL_Z_DIM}",
        f"agent.ipmd.latent_learning.code_period={HORIZON_STEPS}",
        "agent.ipmd.reward_loss_coeff=0.0",
        "agent.ipmd.reward_l2_coeff=0.0",
        "agent.ipmd.reward_grad_penalty_coeff=0.0",
        "agent.ipmd.reward_logit_reg_coeff=0.0",
        "agent.ipmd.reward_param_weight_decay_coeff=0.0",
    ]


def _latent_eval_command(
    runner: Runner,
    *,
    goal: str | None,
    motion_names: list[str] | None = None,
    balanced_rows_per_motion: int = 0,
    num_envs: int | None = None,
    metric_interval: int = 1,
    output_dir: Path,
    max_steps: int,
    command_source: str,
    planner_checkpoint: Path | None,
    save_samples: bool,
    refresh: bool,
    label: str,
    manifest: Path,
    dataset_path: Path,
    language_embeddings: Path,
    low_level_checkpoint: Path,
    skill_checkpoint: Path,
) -> list[str]:
    args = runner.args
    if (goal is None) == (motion_names is None):
        raise ValueError("Provide exactly one goal or a list of motion_names.")
    if num_envs is not None and int(num_envs) <= 0:
        raise ValueError("num_envs must be positive when provided.")
    if metric_interval <= 0:
        raise ValueError("metric_interval must be positive.")
    default_num_envs = len(motion_names) if motion_names is not None else 1
    resolved_num_envs = default_num_envs if num_envs is None else int(num_envs)
    if motion_names is not None and resolved_num_envs != len(motion_names):
        raise ValueError(
            "Multi-motion latent collection requires one environment per motion."
        )
    command = [
        *runner.isaaclab_python,
        "scripts/rlopt/eval_skill_commander_closed_loop.py",
        "--headless",
        "--task",
        "Isaac-Imitation-G1-Latent-v0",
        "--algorithm",
        "IPMD",
        "--checkpoint",
        str(low_level_checkpoint),
        "--skill_checkpoint",
        str(skill_checkpoint),
        "--language_embeddings",
        str(language_embeddings),
        "--state_history_steps",
        str(STATE_HISTORY_STEPS),
        "--output_dir",
        str(output_dir),
        "--label",
        label,
        "--num_envs",
        str(resolved_num_envs),
        "--max_steps",
        str(max_steps),
        "--seed",
        str(args.seed),
        "--metric_interval",
        str(metric_interval),
        "--keep_time_out",
        "--allow_random_reset",
        "--keep_early_terminations",
        "--disable_tracking_terminations",
        "--disable_reward_clipping",
        "--flow_num_inference_steps",
        str(args.flow_steps),
        "--flow_inference_noise_std",
        str(args.flow_noise_std),
        "--kit_args=--/app/extensions/fsWatcherEnabled=false",
    ]
    if motion_names is not None:
        command.extend(["--motion_names", *motion_names])
    else:
        command.extend(["--motion_name", str(goal), "--require_goal_motion_match"])
    if balanced_rows_per_motion > 0:
        balanced_names = motion_names if motion_names is not None else [str(goal)]
        command.extend(
            [
                "--balanced_motion_names",
                *balanced_names,
                "--balanced_rows_per_motion",
                str(balanced_rows_per_motion),
            ]
        )
    if planner_checkpoint is not None:
        command.extend(
            [
                "--planner_checkpoint",
                str(planner_checkpoint),
            ]
        )
    if save_samples:
        rows_per_file = (
            int(balanced_rows_per_motion)
            if int(balanced_rows_per_motion) > 0
            else int(args.rollout_rows_per_goal)
        )
        command.extend(
            [
                "--save_rollout_training_samples",
                "--continue_after_reset",
                "--sample_rows_per_file",
                str(rows_per_file),
            ]
        )
    command.append(f"agent.ipmd.command_source={command_source}")
    if command_source == "skill_commander":
        if planner_checkpoint is None or goal is None:
            raise ValueError(
                "skill_commander requires a planner checkpoint and one explicit goal"
            )
        command.extend(
            [
                f"agent.ipmd.skill_commander_checkpoint_path={planner_checkpoint}",
                f"agent.ipmd.skill_commander_embeddings_path={language_embeddings}",
                f"agent.ipmd.skill_commander_goal_name={goal}",
                "agent.ipmd.skill_commander_use_achieved_state=true",
                f"agent.ipmd.skill_commander_flow_num_inference_steps={args.flow_steps}",
                f"agent.ipmd.skill_commander_flow_inference_noise_std={args.flow_noise_std}",
            ]
        )
    command.extend(
        _latent_hydra_overrides(
            manifest=manifest,
            dataset_path=dataset_path,
            skill_checkpoint=skill_checkpoint,
            refresh=refresh,
        )
    )
    return command


def _explicit_demo_command(
    runner: Runner,
    *,
    goals: list[str],
    output_dir: Path,
    control_steps: int,
    rows_per_goal: int,
    refresh: bool,
    manifest: Path,
    language_embeddings: Path,
    checkpoint: Path,
    dataset_path: Path,
) -> list[str]:
    command = [
        *runner.isaaclab_python,
        "experiments/interface_baselines/collect_interface_rollout_samples.py",
        "--headless",
        "--task",
        "Isaac-Imitation-G1-v0",
        "--algo",
        "IPMD",
        "--checkpoint",
        str(checkpoint),
        "--interface",
        "full_body_trajectory",
        "--output_dir",
        str(output_dir),
        "--motion_manifest",
        str(manifest),
        "--dataset_path",
        str(dataset_path),
        "--motion_names",
        *goals,
        "--balanced_motion_names",
        *goals,
        "--balanced_rows_per_motion",
        str(rows_per_goal),
        "--sample_rows_per_file",
        str(rows_per_goal),
        "--language_embeddings",
        str(language_embeddings),
        "--num_envs",
        str(len(goals)),
        "--control_steps",
        str(control_steps),
        "--seed",
        str(runner.args.seed),
        "--state_history_steps",
        str(STATE_HISTORY_STEPS),
        "--planner_interval_steps",
        str(HORIZON_STEPS),
        "--command_past_steps",
        "0",
        "--command_future_steps",
        str(HORIZON_STEPS - 1),
        "--reset_schedule",
        "sequential",
        "--low_level_command_mode",
        "streamed_vanilla",
        "--keep_configured_episode_length",
        "--disable_tracking_terminations",
        "--kit_args=--/app/extensions/fsWatcherEnabled=false",
        "agent.logger.backend=",
        "env.observations.policy.enable_corruption=false",
    ]
    if refresh:
        command.append("--refresh_zarr_dataset")
    return command


def _explicit_eval_command(
    runner: Runner,
    *,
    goal: str,
    planner_checkpoint: Path,
    output_json: Path,
    steps: int,
    num_envs: int = 1,
    balanced_rows_per_motion: int = 0,
    save_samples: bool,
    samples_output_dir: Path | None,
    label: str,
    manifest: Path,
    language_embeddings: Path,
    checkpoint: Path,
    dataset_path: Path,
) -> list[str]:
    args = runner.args
    if int(num_envs) <= 0:
        raise ValueError("num_envs must be positive.")
    if int(balanced_rows_per_motion) < 0:
        raise ValueError("balanced_rows_per_motion must be non-negative.")
    if int(balanced_rows_per_motion) > 0 and not save_samples:
        raise ValueError("Balanced collection requires save_samples=True.")
    command = [
        *runner.isaaclab_python,
        "experiments/interface_baselines/eval_interface_planner_closed_loop.py",
        "--headless",
        "--task",
        "Isaac-Imitation-G1-v0",
        "--algo",
        "IPMD",
        "--checkpoint",
        str(checkpoint),
        "--low_level_command_mode",
        "streamed_vanilla",
        "--planner_checkpoint",
        str(planner_checkpoint),
        "--output_json",
        str(output_json),
        "--label",
        label,
        "--motion_manifest",
        str(manifest),
        "--dataset_path",
        str(dataset_path),
        "--motion_name",
        goal,
        "--language_embeddings",
        str(language_embeddings),
        "--language_goal_name",
        goal,
        "--num_envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--seed",
        str(args.seed),
        "--state_history_steps",
        str(STATE_HISTORY_STEPS),
        "--command_past_steps",
        "0",
        "--command_future_steps",
        str(HORIZON_STEPS - 1),
        "--planner_update_interval",
        str(HORIZON_STEPS),
        "--flow_num_inference_steps",
        str(args.flow_steps),
        "--flow_inference_noise_std",
        str(args.flow_noise_std),
        "--reset_schedule",
        "sequential",
        "--keep_configured_episode_length",
        "--disable_tracking_terminations",
        "--kit_args=--/app/extensions/fsWatcherEnabled=false",
        "agent.logger.backend=",
        "env.observations.policy.enable_corruption=false",
    ]
    if save_samples:
        if samples_output_dir is None:
            raise ValueError("samples_output_dir is required when saving samples")
        command.extend(
            [
                "--keep_after_done",
                "--save_rollout_training_samples",
                "--samples_output_dir",
                str(samples_output_dir),
                "--sample_rows_per_file",
                str(args.rollout_rows_per_goal),
            ]
        )
        if int(balanced_rows_per_motion) > 0:
            command.extend(
                [
                    "--balanced_rows_per_motion",
                    str(balanced_rows_per_motion),
                ]
            )
    return command


def _merge_command(
    runner: Runner,
    *,
    sources: list[Path],
    limits: list[int],
    output_dir: Path,
) -> list[str]:
    if len(sources) != len(limits):
        raise ValueError("Every merge source requires a row limit.")
    command = [
        *runner.python,
        "experiments/interface_baselines/merge_planner_samples.py",
        "--replace_incomplete",
    ]
    for source, limit in zip(sources, limits, strict=True):
        command.extend(["--source", str(source), "--source_limit", str(limit)])
    command.extend(["--seed", str(runner.args.seed), "--output_dir", str(output_dir)])
    return command


def _train_command(
    runner: Runner,
    *,
    interface: str,
    samples_dir: Path,
    output_dir: Path,
    state_key: str,
    num_updates: int,
    checkpoint: Path | None = None,
) -> list[str]:
    args = runner.args
    command = [
        *runner.python,
        "experiments/interface_baselines/train_chunked_transformer_planner.py",
        "--samples_dir",
        str(samples_dir),
        "--output_dir",
        str(output_dir),
        "--interface",
        interface,
        "--state_key",
        state_key,
        "--model_size",
        str(args.model_size),
        "--seed",
        str(args.seed),
        "--max_samples",
        "0",
        "--num_updates",
        str(num_updates),
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
        str(args.train_endpoint_steps),
        "--flow_inference_noise_std",
        str(args.flow_noise_std),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    return command


def _offline_eval_command(
    runner: Runner,
    *,
    interface: str,
    samples_dir: Path,
    planner_checkpoint: Path,
    output_json: Path,
    state_key: str,
    setting: str,
) -> list[str]:
    return [
        *runner.python,
        "experiments/interface_baselines/eval_interface_planner_offline.py",
        "--samples_dir",
        str(samples_dir),
        "--planner_checkpoint",
        str(planner_checkpoint),
        "--output_json",
        str(output_json),
        "--interface",
        interface,
        "--state_key",
        state_key,
        "--setting",
        setting,
        "--seed",
        str(runner.args.seed),
        "--flow_num_inference_steps",
        str(runner.args.flow_steps),
        "--flow_inference_noise_std",
        str(runner.args.flow_noise_std),
    ]


def _write_run_config(
    *,
    output_root: Path,
    args: argparse.Namespace,
    goals: list[str],
    language_metadata: dict[str, Any],
    gate_metadata: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    active_interfaces = tuple(args.interfaces)
    interface_specs = {
        "latent_skill": {"target_dim": 256},
        "full_body_trajectory": {"target_dim": 670},
    }
    paired_interfaces = set(active_interfaces) == set(interface_specs)
    payload = {
        "protocol": PROTOCOL,
        "goals": goals,
        "goal_count": len(goals),
        "demo_rows_per_goal": int(args.demo_rows_per_goal),
        "rollout_rows_per_goal": int(args.rollout_rows_per_goal),
        "planner_rollout_collection": {
            "mode": "parallel_identical_goal",
            "num_envs": int(args.rollout_num_envs),
            "rows_per_goal": int(args.rollout_rows_per_goal),
            "max_control_steps_per_goal": (
                (int(args.rollout_rows_per_goal) + int(args.rollout_num_envs) - 1)
                // int(args.rollout_num_envs)
            )
            * HORIZON_STEPS,
            "goal_source": "explicit_language_argument",
            "reference_scope": "same_named_motion_only",
        },
        "skip_pretrained_closed_loop": bool(args.skip_pretrained_closed_loop),
        "pretrained_closed_loop_complete": not bool(args.skip_pretrained_closed_loop),
        "paper_protocol_complete": (
            not bool(args.skip_pretrained_closed_loop)
            and bool(gate_metadata.get("fresh_preparation_checked"))
            and bool(gate_metadata.get("low_level_gates_checked"))
            and paired_interfaces
        ),
        "expected_demo_rows_per_interface": len(goals) * int(args.demo_rows_per_goal),
        "demonstration_collection": {
            "mode": "balanced_multi_environment",
            "simulator_launches": len(active_interfaces),
            "rows_per_goal": int(args.demo_rows_per_goal),
            "max_control_steps": (
                int(args.demo_rows_per_goal)
                * HORIZON_STEPS
                * DEMONSTRATION_COLLECTION_SAFETY_MULTIPLIER
            ),
            "safety_multiplier": DEMONSTRATION_COLLECTION_SAFETY_MULTIPLIER,
        },
        "expected_rollout_rows_per_interface": len(goals)
        * int(args.rollout_rows_per_goal),
        "eval_steps": int(args.eval_steps),
        "seed": int(args.seed),
        "planner": _planner_contract(args),
        "interfaces": {
            interface: interface_specs[interface] for interface in active_interfaces
        },
        "preliminary_unqualified": not bool(
            paired_interfaces and gate_metadata.get("low_level_gates_checked")
        ),
        "causal_state_dim": 930,
        "language": language_metadata,
        "submission_gates": gate_metadata,
        "paths": {key: str(value) for key, value in paths.items()},
        "input_artifacts": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
            if path.is_file()
        },
        "repository": _repository_provenance(),
        "workflow_source_sha256": _workflow_source_hashes(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _planner_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_size": args.model_size,
        "pretrain_updates": int(args.pretrain_updates),
        "finetune_updates": int(args.finetune_updates),
        "batch_size": int(args.batch_size),
        "micro_batch_size": int(args.micro_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "flow_steps": int(args.flow_steps),
        "train_endpoint_steps": int(args.train_endpoint_steps),
        "flow_noise_std": float(args.flow_noise_std),
    }


def _validate_existing_run_config(
    *,
    path: Path,
    args: argparse.Namespace,
    goals: list[str],
    output_root: Path,
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    payload = _json_object(path)
    mismatches: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            mismatches.append(message)

    check(payload.get("protocol") == PROTOCOL, "protocol differs")
    check(payload.get("goals") == goals, "selected goals or their order differ")
    check(payload.get("seed") == int(args.seed), "seed differs")
    check(
        list(payload.get("interfaces", {})) == list(args.interfaces),
        "selected interfaces differ",
    )
    check(
        payload.get("demo_rows_per_goal") == int(args.demo_rows_per_goal),
        "demonstration row budget differs",
    )
    check(
        payload.get("rollout_rows_per_goal") == int(args.rollout_rows_per_goal),
        "planner-rollout row budget differs",
    )
    rollout_collection = payload.get("planner_rollout_collection", {})
    check(
        int(rollout_collection.get("num_envs", -1)) == int(args.rollout_num_envs),
        "planner-rollout environment count differs",
    )
    check(payload.get("eval_steps") == int(args.eval_steps), "eval steps differ")
    check(
        payload.get("skip_pretrained_closed_loop")
        == bool(args.skip_pretrained_closed_loop),
        "pretrained closed-loop setting differs",
    )
    check(payload.get("planner") == _planner_contract(args), "planner setup differs")
    recorded_output_root = Path(
        str(payload.get("paths", {}).get("output_root", ""))
    ).expanduser()
    check(
        recorded_output_root.resolve() == output_root.resolve(),
        "output root differs",
    )
    for path_key, current_path in (
        ("latent_dataset_path", _resolve(args.latent_dataset_path)),
        ("vanilla_dataset_path", _resolve(args.vanilla_dataset_path)),
    ):
        recorded_path = _as_path(payload.get("paths", {}).get(path_key))
        check(recorded_path == current_path, f"{path_key} differs")
    artifacts = payload.get("input_artifacts", {})
    for name, artifact_path in input_paths.items():
        if not artifact_path.is_file():
            continue
        artifact = artifacts.get(name, {})
        check(bool(artifact), f"input artifact record is missing for {name}")
        if artifact:
            check(
                artifact.get("sha256") == _sha256(artifact_path),
                f"input artifact hash differs for {name}",
            )
    check(
        payload.get("workflow_source_sha256") == _workflow_source_hashes(),
        "workflow source hashes differ",
    )
    if mismatches:
        raise ValueError(
            "Staged run does not match its original run_config.json: "
            + "; ".join(mismatches)
        )
    return payload


def _goal_index_for_stage(args: argparse.Namespace, goal_count: int) -> int | None:
    goal_index = args.goal_index
    array_index = str(os.environ.get("SLURM_ARRAY_TASK_ID", "")).strip()
    if goal_index is None and array_index:
        goal_index = int(array_index)
    if args.stage in {"rollout", "final-eval"}:
        if goal_index is None:
            raise ValueError(
                f"--stage {args.stage} requires --goal_index or SLURM_ARRAY_TASK_ID."
            )
        if not 0 <= int(goal_index) < int(goal_count):
            raise ValueError(
                f"Goal index {goal_index} is outside [0, {int(goal_count) - 1}]."
            )
        return int(goal_index)
    if goal_index is not None:
        raise ValueError("--goal_index is only valid for rollout or final-eval.")
    return None


def _write_stage_record(
    *,
    output_root: Path,
    stage: str,
    artifacts: dict[str, Path],
    goal_index: int | None = None,
    goal_name: str | None = None,
) -> Path:
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot mark stage {stage!r} complete; missing artifacts: {missing}."
        )
    suffix = (
        f"/{int(goal_index):04d}_{_slug(str(goal_name))}.json"
        if goal_index is not None
        else ".json"
    )
    record_path = output_root / "stages" / f"{stage}{suffix}"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "stage": stage,
        "goal_index": goal_index,
        "goal_name": goal_name,
        "completed_at": datetime.now().astimezone().isoformat(),
        "workflow_source_sha256": _workflow_source_hashes(),
        "artifacts": {
            name: {
                "path": str(path),
                "kind": "directory" if path.is_dir() else "file",
                "sha256": _artifact_sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    record_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return record_path


def _require_stage_record(
    path: Path,
    *,
    expected_stage: str | None = None,
    expected_goal_index: int | None = None,
    expected_goal_name: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required stage record is missing: {path}")
    record = _json_object(path)
    if record.get("status") != "complete":
        raise ValueError(f"Stage record is not complete: {path}")
    if expected_stage is not None and record.get("stage") != expected_stage:
        raise ValueError(f"Stage record has the wrong stage: {path}")
    if expected_goal_index is not None and record.get("goal_index") != int(
        expected_goal_index
    ):
        raise ValueError(f"Stage record has the wrong goal index: {path}")
    if expected_goal_name is not None and record.get("goal_name") != str(
        expected_goal_name
    ):
        raise ValueError(f"Stage record has the wrong goal name: {path}")
    if record.get("workflow_source_sha256") != _workflow_source_hashes():
        raise ValueError(f"Stage record used different workflow source: {path}")
    for name, artifact in record.get("artifacts", {}).items():
        artifact_path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Stage artifact {name!r} is missing: {artifact_path}"
            )
        actual_kind = "directory" if artifact_path.is_dir() else "file"
        if artifact.get("kind", "file") != actual_kind:
            raise ValueError(f"Stage artifact {name!r} changed type: {artifact_path}")
        if artifact.get("sha256") != _artifact_sha256(artifact_path):
            raise ValueError(f"Stage artifact {name!r} hash changed: {artifact_path}")
    return record


def main() -> None:
    args = _parse_args()
    if len(set(args.interfaces)) != len(args.interfaces):
        raise ValueError("--interfaces must not contain duplicates.")
    interface_order = ("latent_skill", "full_body_trajectory")
    selected = set(args.interfaces)
    active_interfaces = tuple(
        interface for interface in interface_order if interface in selected
    )
    args.interfaces = list(active_interfaces)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    for name in (
        "demo_rows_per_goal",
        "rollout_rows_per_goal",
        "rollout_num_envs",
        "eval_steps",
        "batch_size",
        "micro_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.pretrain_updates < 0 or args.finetune_updates < 0:
        raise ValueError("Planner update counts must be non-negative.")
    if not 0.0 <= float(args.min_oracle_success) <= 1.0:
        raise ValueError("--min_oracle_success must be between zero and one.")

    manifest = _resolve(args.manifest)
    language_embeddings = _resolve(args.language_embeddings)
    latent_dataset_path = _resolve(args.latent_dataset_path)
    vanilla_dataset_path = _resolve(args.vanilla_dataset_path)
    latent_low_level = _resolve(args.latent_low_level_checkpoint)
    latent_skill = _resolve(args.latent_skill_checkpoint)
    vanilla_tracker = _resolve(args.vanilla_tracker_checkpoint)
    required_files = {
        "manifest": manifest,
        "language_embeddings": language_embeddings,
        "latent_low_level_checkpoint": latent_low_level,
        "latent_skill_checkpoint": latent_skill,
        "vanilla_tracker_checkpoint": vanilla_tracker,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    gate_metadata: dict[str, Any] = {
        "fresh_preparation_checked": False,
        "low_level_gates_checked": False,
    }
    if args.preparation_record is not None:
        preparation_record = _resolve(args.preparation_record)
        if not preparation_record.is_file():
            raise FileNotFoundError(
                f"preparation_record not found: {preparation_record}"
            )
        required_files["preparation_record"] = preparation_record
        gate_metadata["fresh_preparation_checked"] = True
        gate_metadata["preparation"] = _validate_preparation_record(
            preparation_record, manifest
        )

    qualification_args = (
        args.vanilla_qualification_audit,
        args.latent_qualification_audit,
        args.streamed_equivalence_certificate,
    )
    if any(value is not None for value in qualification_args) and not all(
        value is not None for value in qualification_args
    ):
        raise ValueError("Provide all three low-level gate artifacts or none of them.")
    if all(value is not None for value in qualification_args):
        vanilla_audit = _resolve(args.vanilla_qualification_audit)
        latent_audit = _resolve(args.latent_qualification_audit)
        equivalence = _resolve(args.streamed_equivalence_certificate)
        for label, path in (
            ("vanilla_qualification_audit", vanilla_audit),
            ("latent_qualification_audit", latent_audit),
            ("streamed_equivalence_certificate", equivalence),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
            required_files[label] = path
        gate_metadata["low_level_gates_checked"] = True
        gate_metadata["low_level"] = _validate_low_level_gates(
            vanilla_audit_path=vanilla_audit,
            latent_audit_path=latent_audit,
            equivalence_path=equivalence,
            manifest=manifest,
            vanilla_checkpoint=vanilla_tracker,
            latent_checkpoint=latent_low_level,
            skill_checkpoint=latent_skill,
            vanilla_dataset_path=vanilla_dataset_path,
            latent_dataset_path=latent_dataset_path,
            min_success=float(args.min_oracle_success),
        )

    manifest_names = _manifest_names(manifest)
    goals = _select_goals(manifest_names, args.goal_names, args.goal_limit)
    stage_goal_index = _goal_index_for_stage(args, len(goals))
    language_metadata = _validate_language_table(language_embeddings, goals)
    output_root = (
        _resolve(args.output_root)
        if args.output_root is not None
        else _default_output_root()
    )
    starts_run = args.stage in {"all", "prepare"}
    run_config_path = output_root / "run_config.json"
    if starts_run and output_root.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"Output root already exists; use --resume or a new path: {output_root}"
        )
    if not starts_run and not args.dry_run and not run_config_path.is_file():
        raise FileNotFoundError(
            f"Stage {args.stage!r} requires the prepare-stage run config: "
            f"{run_config_path}"
        )
    paths = {
        **required_files,
        "latent_dataset_path": latent_dataset_path,
        "vanilla_dataset_path": vanilla_dataset_path,
        "output_root": output_root,
    }
    if not args.dry_run and starts_run:
        if run_config_path.is_file() and args.resume:
            _validate_existing_run_config(
                path=run_config_path,
                args=args,
                goals=goals,
                output_root=output_root,
                input_paths=required_files,
            )
        else:
            _write_run_config(
                output_root=output_root,
                args=args,
                goals=goals,
                language_metadata=language_metadata,
                gate_metadata=gate_metadata,
                paths=paths,
            )
    elif not args.dry_run:
        _validate_existing_run_config(
            path=run_config_path,
            args=args,
            goals=goals,
            output_root=output_root,
            input_paths=required_files,
        )

    runner = Runner(args, output_root)
    preflight = output_root / "protocol_checks" / "bones_seed_preflight.json"
    if starts_run:
        runner.run(
            [
                *runner.python,
                "scripts/audit_bones_seed_phase5.py",
                "--manifest",
                str(manifest),
                "--report",
                str(preflight),
                "--require-body-names",
                "--require-temporal-events",
            ],
            expected=preflight,
        )
    elif not args.dry_run:
        preflight_payload = _json_object(preflight)
        if preflight_payload.get("passed") is not True:
            raise ValueError(
                f"Prepare-stage BONES-SEED preflight did not pass: {preflight}"
            )

    demo_sources: dict[str, list[Path]] = {
        interface: [] for interface in active_interfaces
    }
    demo_control_steps = (
        int(args.demo_rows_per_goal)
        * HORIZON_STEPS
        * DEMONSTRATION_COLLECTION_SAFETY_MULTIPLIER
    )
    explicit_dir = output_root / "demonstration_batched" / "full_body"
    explicit_samples = explicit_dir / "rollout_training_samples"
    if "full_body_trajectory" in selected:
        demo_sources["full_body_trajectory"].append(explicit_samples)

    latent_dir = output_root / "demonstration_batched" / "latent_skill"
    latent_samples = latent_dir / "rollout_training_samples"
    if "latent_skill" in selected:
        demo_sources["latent_skill"].append(latent_samples)
    if starts_run:
        if "full_body_trajectory" in selected:
            runner.run(
                _explicit_demo_command(
                    runner,
                    goals=goals,
                    output_dir=explicit_dir,
                    control_steps=demo_control_steps,
                    rows_per_goal=int(args.demo_rows_per_goal),
                    refresh=bool(args.refresh_datasets),
                    manifest=manifest,
                    language_embeddings=language_embeddings,
                    checkpoint=vanilla_tracker,
                    dataset_path=vanilla_dataset_path,
                ),
                expected=explicit_dir / "summary.json",
            )
        if "latent_skill" in selected:
            runner.run(
                _latent_eval_command(
                    runner,
                    goal=None,
                    motion_names=goals,
                    balanced_rows_per_motion=int(args.demo_rows_per_goal),
                    metric_interval=demo_control_steps + 1,
                    output_dir=latent_dir,
                    max_steps=demo_control_steps,
                    command_source="hl_skill",
                    planner_checkpoint=None,
                    save_samples=True,
                    refresh=bool(args.refresh_datasets),
                    label="bones_seed_demo_latent_batched",
                    manifest=manifest,
                    dataset_path=latent_dataset_path,
                    language_embeddings=language_embeddings,
                    low_level_checkpoint=latent_low_level,
                    skill_checkpoint=latent_skill,
                ),
                expected=latent_dir / "summary.json",
            )

    merged_demo: dict[str, Path] = {}
    pretrain_checkpoints: dict[str, Path] = {}
    for interface in active_interfaces:
        interface_root = output_root / interface
        merged_demo[interface] = interface_root / "demonstration_samples"
        pretrain_dir = interface_root / "planner_pretrain_demonstration"
        pretrain_checkpoints[interface] = pretrain_dir / "checkpoints" / "latest.pt"
        if starts_run:
            runner.run(
                _merge_command(
                    runner,
                    sources=demo_sources[interface],
                    limits=[0],
                    output_dir=merged_demo[interface],
                ),
                expected=merged_demo[interface] / "merge_manifest.json",
            )
            runner.run(
                _train_command(
                    runner,
                    interface=interface,
                    samples_dir=merged_demo[interface],
                    output_dir=pretrain_dir,
                    state_key="expert_planner_state",
                    num_updates=int(args.pretrain_updates),
                ),
                expected=pretrain_checkpoints[interface],
            )
            runner.run(
                _offline_eval_command(
                    runner,
                    interface=interface,
                    samples_dir=merged_demo[interface],
                    planner_checkpoint=pretrain_checkpoints[interface],
                    output_json=interface_root
                    / "eval_pretrained_offline"
                    / "summary.json",
                    state_key="expert_planner_state",
                    setting="eval_pretrained_demonstration",
                ),
                expected=interface_root / "eval_pretrained_offline" / "summary.json",
            )

    if starts_run and not args.dry_run:
        prepare_artifacts: dict[str, Path] = {"preflight": preflight}
        if "full_body_trajectory" in selected:
            prepare_artifacts.update(
                {
                    "explicit_demonstration_summary": explicit_dir / "summary.json",
                    "explicit_demonstration_merge": merged_demo["full_body_trajectory"]
                    / "merge_manifest.json",
                    "explicit_demonstration_samples": merged_demo[
                        "full_body_trajectory"
                    ],
                    "explicit_pretrain_checkpoint": pretrain_checkpoints[
                        "full_body_trajectory"
                    ],
                    "explicit_pretrain_offline_summary": output_root
                    / "full_body_trajectory"
                    / "eval_pretrained_offline"
                    / "summary.json",
                }
            )
        if "latent_skill" in selected:
            prepare_artifacts.update(
                {
                    "latent_demonstration_summary": latent_dir / "summary.json",
                    "latent_demonstration_merge": merged_demo["latent_skill"]
                    / "merge_manifest.json",
                    "latent_demonstration_samples": merged_demo["latent_skill"],
                    "latent_pretrain_checkpoint": pretrain_checkpoints["latent_skill"],
                    "latent_pretrain_offline_summary": output_root
                    / "latent_skill"
                    / "eval_pretrained_offline"
                    / "summary.json",
                }
            )
        _write_stage_record(
            output_root=output_root,
            stage="prepare",
            artifacts=prepare_artifacts,
        )

    goal_slugs = [f"{index:04d}_{_slug(goal)}" for index, goal in enumerate(goals)]
    all_rollout_sources: dict[str, list[Path]] = {
        "full_body_trajectory": [
            output_root
            / "full_body_trajectory"
            / "planner_rollout_per_goal"
            / slug
            / "rollout_training_samples"
            for slug in goal_slugs
        ],
        "latent_skill": [
            output_root
            / "latent_skill"
            / "planner_rollout_per_goal"
            / slug
            / "rollout_training_samples"
            for slug in goal_slugs
        ],
    }
    rollout_sources = {
        interface: all_rollout_sources[interface] for interface in active_interfaces
    }
    all_pretrained_summaries: dict[str, list[Path]] = {
        "full_body_trajectory": [
            output_root
            / "full_body_trajectory"
            / "eval_pretrained_per_goal"
            / slug
            / "summary.json"
            for slug in goal_slugs
        ],
        "latent_skill": [
            output_root
            / "latent_skill"
            / "eval_pretrained_per_goal"
            / slug
            / "summary.json"
            for slug in goal_slugs
        ],
    }
    pretrained_summaries = {
        interface: all_pretrained_summaries[interface]
        for interface in active_interfaces
    }
    rollout_publications = (
        int(args.rollout_rows_per_goal) + int(args.rollout_num_envs) - 1
    ) // int(args.rollout_num_envs)
    rollout_control_steps = rollout_publications * HORIZON_STEPS
    if args.stage in {"all", "rollout"}:
        if not args.dry_run:
            _require_stage_record(
                output_root / "stages" / "prepare.json",
                expected_stage="prepare",
            )
        rollout_goal_indices = (
            range(len(goals)) if args.stage == "all" else (int(stage_goal_index),)
        )
        for goal_index in rollout_goal_indices:
            goal = goals[goal_index]
            slug = goal_slugs[goal_index]
            explicit_pre = (
                pretrained_summaries["full_body_trajectory"][goal_index]
                if "full_body_trajectory" in selected
                else None
            )
            latent_pre = (
                pretrained_summaries["latent_skill"][goal_index]
                if "latent_skill" in selected
                else None
            )
            if not args.skip_pretrained_closed_loop:
                if explicit_pre is not None:
                    runner.run(
                        _explicit_eval_command(
                            runner,
                            goal=goal,
                            planner_checkpoint=pretrain_checkpoints[
                                "full_body_trajectory"
                            ],
                            output_json=explicit_pre,
                            steps=int(args.eval_steps),
                            save_samples=False,
                            samples_output_dir=None,
                            label=f"bones_seed_pretrained_full_body_{slug}",
                            manifest=manifest,
                            language_embeddings=language_embeddings,
                            checkpoint=vanilla_tracker,
                            dataset_path=vanilla_dataset_path,
                        ),
                        expected=explicit_pre,
                    )
                if latent_pre is not None:
                    runner.run(
                        _latent_eval_command(
                            runner,
                            goal=goal,
                            output_dir=latent_pre.parent,
                            max_steps=int(args.eval_steps),
                            command_source="skill_commander",
                            planner_checkpoint=pretrain_checkpoints["latent_skill"],
                            save_samples=False,
                            refresh=False,
                            label=f"bones_seed_pretrained_latent_{slug}",
                            manifest=manifest,
                            dataset_path=latent_dataset_path,
                            language_embeddings=language_embeddings,
                            low_level_checkpoint=latent_low_level,
                            skill_checkpoint=latent_skill,
                        ),
                        expected=latent_pre,
                    )

            rollout_artifacts: dict[str, Path] = {}
            if "full_body_trajectory" in selected:
                explicit_rollout_samples = rollout_sources["full_body_trajectory"][
                    goal_index
                ]
                explicit_rollout_summary = (
                    explicit_rollout_samples.parent / "summary.json"
                )
                runner.run(
                    _explicit_eval_command(
                        runner,
                        goal=goal,
                        planner_checkpoint=pretrain_checkpoints["full_body_trajectory"],
                        output_json=explicit_rollout_summary,
                        steps=rollout_control_steps,
                        num_envs=int(args.rollout_num_envs),
                        balanced_rows_per_motion=int(args.rollout_rows_per_goal),
                        save_samples=True,
                        samples_output_dir=explicit_rollout_samples,
                        label=f"bones_seed_rollout_full_body_{slug}",
                        manifest=manifest,
                        language_embeddings=language_embeddings,
                        checkpoint=vanilla_tracker,
                        dataset_path=vanilla_dataset_path,
                    ),
                    expected=explicit_rollout_summary,
                )
                rollout_artifacts.update(
                    {
                        "explicit_rollout_summary": explicit_rollout_summary,
                        "explicit_rollout_samples": explicit_rollout_samples,
                    }
                )
            if "latent_skill" in selected:
                latent_rollout_samples = rollout_sources["latent_skill"][goal_index]
                latent_rollout_summary = latent_rollout_samples.parent / "summary.json"
                runner.run(
                    _latent_eval_command(
                        runner,
                        goal=goal,
                        balanced_rows_per_motion=int(args.rollout_rows_per_goal),
                        num_envs=int(args.rollout_num_envs),
                        output_dir=latent_rollout_samples.parent,
                        max_steps=rollout_control_steps,
                        command_source="skill_commander",
                        planner_checkpoint=pretrain_checkpoints["latent_skill"],
                        save_samples=True,
                        refresh=False,
                        label=f"bones_seed_rollout_latent_{slug}",
                        manifest=manifest,
                        dataset_path=latent_dataset_path,
                        language_embeddings=language_embeddings,
                        low_level_checkpoint=latent_low_level,
                        skill_checkpoint=latent_skill,
                    ),
                    expected=latent_rollout_summary,
                )
                rollout_artifacts.update(
                    {
                        "latent_rollout_summary": latent_rollout_summary,
                        "latent_rollout_samples": latent_rollout_samples,
                    }
                )
            if not args.dry_run:
                if not args.skip_pretrained_closed_loop:
                    if explicit_pre is not None:
                        rollout_artifacts["explicit_pretrained_summary"] = explicit_pre
                    if latent_pre is not None:
                        rollout_artifacts["latent_pretrained_summary"] = latent_pre
                _write_stage_record(
                    output_root=output_root,
                    stage="rollout",
                    goal_index=goal_index,
                    goal_name=goal,
                    artifacts=rollout_artifacts,
                )

    final_checkpoints: dict[str, Path] = {}
    merged_final: dict[str, Path] = {}
    if args.stage in {"all", "finetune"} and not args.dry_run:
        for goal_index, goal in enumerate(goals):
            _require_stage_record(
                output_root
                / "stages"
                / "rollout"
                / f"{goal_index:04d}_{_slug(goal)}.json",
                expected_stage="rollout",
                expected_goal_index=goal_index,
                expected_goal_name=goal,
            )
    for interface in active_interfaces:
        interface_root = output_root / interface
        merged_rollout = interface_root / "planner_rollout_samples"
        merged_final[interface] = interface_root / "demonstration_and_rollout_samples"
        finetune_dir = interface_root / "planner_finetune_planner_rollout"
        final_checkpoints[interface] = finetune_dir / "checkpoints" / "latest.pt"
        if args.stage in {"all", "finetune"}:
            runner.run(
                _merge_command(
                    runner,
                    sources=rollout_sources[interface],
                    limits=[int(args.rollout_rows_per_goal)] * len(goals),
                    output_dir=merged_rollout,
                ),
                expected=merged_rollout / "merge_manifest.json",
            )
            runner.run(
                _merge_command(
                    runner,
                    sources=[merged_demo[interface], merged_rollout],
                    limits=[0, 0],
                    output_dir=merged_final[interface],
                ),
                expected=merged_final[interface] / "merge_manifest.json",
            )
            runner.run(
                _train_command(
                    runner,
                    interface=interface,
                    samples_dir=merged_final[interface],
                    output_dir=finetune_dir,
                    state_key="planner_state",
                    num_updates=int(args.finetune_updates),
                    checkpoint=pretrain_checkpoints[interface],
                ),
                expected=final_checkpoints[interface],
            )
            runner.run(
                _offline_eval_command(
                    runner,
                    interface=interface,
                    samples_dir=merged_final[interface],
                    planner_checkpoint=final_checkpoints[interface],
                    output_json=interface_root
                    / "eval_finetuned_offline"
                    / "summary.json",
                    state_key="planner_state",
                    setting="eval_finetuned_planner_rollout",
                ),
                expected=interface_root / "eval_finetuned_offline" / "summary.json",
            )

    if args.stage in {"all", "finetune"} and not args.dry_run:
        finetune_artifacts: dict[str, Path] = {}
        for interface in active_interfaces:
            prefix = "latent" if interface == "latent_skill" else "explicit"
            finetune_artifacts.update(
                {
                    f"{prefix}_rollout_merge": output_root
                    / interface
                    / "planner_rollout_samples"
                    / "merge_manifest.json",
                    f"{prefix}_final_samples": merged_final[interface],
                    f"{prefix}_final_checkpoint": final_checkpoints[interface],
                    f"{prefix}_finetuned_offline_summary": output_root
                    / interface
                    / "eval_finetuned_offline"
                    / "summary.json",
                }
            )
        _write_stage_record(
            output_root=output_root,
            stage="finetune",
            artifacts=finetune_artifacts,
        )

    all_final_summaries: dict[str, list[Path]] = {
        "full_body_trajectory": [
            output_root
            / "full_body_trajectory"
            / "eval_finetuned_per_goal"
            / slug
            / "summary.json"
            for slug in goal_slugs
        ],
        "latent_skill": [
            output_root
            / "latent_skill"
            / "eval_finetuned_per_goal"
            / slug
            / "summary.json"
            for slug in goal_slugs
        ],
    }
    final_summaries = {
        interface: all_final_summaries[interface] for interface in active_interfaces
    }
    if args.stage in {"all", "final-eval"}:
        if not args.dry_run:
            _require_stage_record(
                output_root / "stages" / "finetune.json",
                expected_stage="finetune",
            )
        final_goal_indices = (
            range(len(goals)) if args.stage == "all" else (int(stage_goal_index),)
        )
        for goal_index in final_goal_indices:
            goal = goals[goal_index]
            slug = goal_slugs[goal_index]
            final_eval_artifacts: dict[str, Path] = {}
            if "full_body_trajectory" in selected:
                explicit_summary = final_summaries["full_body_trajectory"][goal_index]
                runner.run(
                    _explicit_eval_command(
                        runner,
                        goal=goal,
                        planner_checkpoint=final_checkpoints["full_body_trajectory"],
                        output_json=explicit_summary,
                        steps=int(args.eval_steps),
                        save_samples=False,
                        samples_output_dir=None,
                        label=f"bones_seed_finetuned_full_body_{slug}",
                        manifest=manifest,
                        language_embeddings=language_embeddings,
                        checkpoint=vanilla_tracker,
                        dataset_path=vanilla_dataset_path,
                    ),
                    expected=explicit_summary,
                )
                final_eval_artifacts["explicit_final_summary"] = explicit_summary
            if "latent_skill" in selected:
                latent_summary = final_summaries["latent_skill"][goal_index]
                runner.run(
                    _latent_eval_command(
                        runner,
                        goal=goal,
                        output_dir=latent_summary.parent,
                        max_steps=int(args.eval_steps),
                        command_source="skill_commander",
                        planner_checkpoint=final_checkpoints["latent_skill"],
                        save_samples=False,
                        refresh=False,
                        label=f"bones_seed_finetuned_latent_{slug}",
                        manifest=manifest,
                        dataset_path=latent_dataset_path,
                        language_embeddings=language_embeddings,
                        low_level_checkpoint=latent_low_level,
                        skill_checkpoint=latent_skill,
                    ),
                    expected=latent_summary,
                )
                final_eval_artifacts["latent_final_summary"] = latent_summary
            if not args.dry_run:
                _write_stage_record(
                    output_root=output_root,
                    stage="final-eval",
                    goal_index=goal_index,
                    goal_name=goal,
                    artifacts=final_eval_artifacts,
                )

    if args.stage not in {"all", "summarize"}:
        if runner.failures:
            raise SystemExit(
                f"Stage {args.stage!r} completed with {len(runner.failures)} failures."
            )
        print(f"[INFO] Multi-goal stage {args.stage!r} complete: {output_root}")
        return

    if not args.dry_run:
        _require_stage_record(
            output_root / "stages" / "finetune.json",
            expected_stage="finetune",
        )
        for goal_index, goal in enumerate(goals):
            _require_stage_record(
                output_root
                / "stages"
                / "final-eval"
                / f"{goal_index:04d}_{_slug(goal)}.json",
                expected_stage="final-eval",
                expected_goal_index=goal_index,
                expected_goal_name=goal,
            )

    comparison_manifest = {
        "protocol": PROTOCOL,
        "interfaces": list(active_interfaces),
        "preliminary_unqualified": not bool(
            set(active_interfaces) == {"latent_skill", "full_body_trajectory"}
            and gate_metadata.get("low_level_gates_checked")
        ),
        "goals": goals,
        "goal_count": len(goals),
        "demo_rows_per_goal": int(args.demo_rows_per_goal),
        "demonstration_collection": {
            "mode": "balanced_multi_environment",
            "simulator_launches": len(active_interfaces),
            "rows_per_goal": int(args.demo_rows_per_goal),
        },
        "rollout_rows_per_goal": int(args.rollout_rows_per_goal),
        "planner_rollout_collection": {
            "mode": "parallel_identical_goal",
            "num_envs": int(args.rollout_num_envs),
            "goal_source": "explicit_language_argument",
            "reference_scope": "same_named_motion_only",
        },
        "skip_pretrained_closed_loop": bool(args.skip_pretrained_closed_loop),
        "pretrained_closed_loop_complete": not bool(args.skip_pretrained_closed_loop),
        "submission_gates_complete": bool(
            gate_metadata.get("fresh_preparation_checked")
            and gate_metadata.get("low_level_gates_checked")
        ),
        "paper_protocol_complete": bool(
            not args.skip_pretrained_closed_loop
            and gate_metadata.get("fresh_preparation_checked")
            and gate_metadata.get("low_level_gates_checked")
            and set(active_interfaces) == {"latent_skill", "full_body_trajectory"}
        ),
        "pretrain_checkpoints": {
            key: str(value) for key, value in pretrain_checkpoints.items()
        },
        "final_checkpoints": {
            key: str(value) for key, value in final_checkpoints.items()
        },
        "merged_samples": {key: str(value) for key, value in merged_final.items()},
        "final_summaries": {
            key: [str(path) for path in values]
            for key, values in final_summaries.items()
        },
        "pretrained_summaries": {
            key: (
                []
                if args.skip_pretrained_closed_loop
                else [str(path) for path in values]
            )
            for key, values in pretrained_summaries.items()
        },
        "failures": runner.failures,
    }
    cluster_submission = output_root / "cluster_submission.json"
    if cluster_submission.is_file():
        comparison_manifest["cluster_submission"] = {
            "path": str(cluster_submission),
            "sha256": _sha256(cluster_submission),
        }
    else:
        comparison_manifest["cluster_submission"] = None
    manifest_output = output_root / "comparison_manifest.json"
    if not args.dry_run:
        manifest_output.write_text(
            json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(f"[DRY-RUN] Would write comparison manifest: {manifest_output}")
    summary_json = output_root / "summary" / "final_results.json"
    summary_csv = output_root / "summary" / "final_results.csv"
    audit_json = output_root / "protocol_checks" / "multigoal_language_audit.json"
    runner.run(
        [
            *runner.python,
            "experiments/interface_baselines/summarize_bones_seed_multigoal_language_comparison.py",
            "--run_root",
            str(output_root),
            "--output_json",
            str(summary_json),
            "--output_csv",
            str(summary_csv),
        ],
        expected=summary_json,
    )
    run_paper_audit = bool(comparison_manifest["paper_protocol_complete"])
    if run_paper_audit:
        runner.run(
            [
                *runner.python,
                "experiments/interface_baselines/audit_bones_seed_multigoal_language_comparison.py",
                "--run_root",
                str(output_root),
                "--output_json",
                str(audit_json),
            ],
            expected=audit_json,
        )
    elif not args.dry_run:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(
            json.dumps(
                {
                    "passed": False,
                    "paper_protocol_complete": False,
                    "preliminary_unqualified": True,
                    "interfaces": list(active_interfaces),
                    "reason": (
                        "Paper audit intentionally skipped because this run does not "
                        "contain both qualified interfaces."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    comparison_manifest["summary_json"] = str(summary_json)
    comparison_manifest["summary_csv"] = str(summary_csv)
    comparison_manifest["audit_json"] = str(audit_json)
    comparison_manifest["paper_audit_executed"] = run_paper_audit
    comparison_manifest["failures"] = runner.failures
    if not args.dry_run:
        manifest_output.write_text(
            json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8"
        )
        summarize_artifacts = {
            "comparison_manifest": manifest_output,
            "summary_json": summary_json,
            "summary_csv": summary_csv,
            "protocol_audit": audit_json,
        }
        if cluster_submission.is_file():
            summarize_artifacts["cluster_submission"] = cluster_submission
        _write_stage_record(
            output_root=output_root,
            stage="summarize",
            artifacts=summarize_artifacts,
        )
    if runner.failures:
        raise SystemExit(f"Comparison completed with {len(runner.failures)} failures.")
    print(f"[INFO] Multi-goal comparison complete: {output_root}")


if __name__ == "__main__":
    main()
