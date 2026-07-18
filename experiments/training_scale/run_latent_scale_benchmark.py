#!/usr/bin/env python3
"""Run a bounded sequential scaling benchmark for frozen-skill latent IPMD.

This runner intentionally launches one fresh training process per configuration.
It keeps the corrected-LAFAN task, frozen DiffSR encoder, PPO epochs, and all
environment protocol settings fixed while varying only:

* number of simulation environments;
* PPO rollout steps per environment; and
* PPO minibatch size.

The authoritative result is wall-clock time to a sustained episodic-return
threshold, not peak simulator FPS.  Every planned run is accounted for before
the first child process starts, and the full matrix is rejected if it would
exceed ``--aggregate-frame-cap``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

# The baseline rollout batch is 4096 environments x 24 steps.  The ridge keeps
# that batch and the optimizer workload fixed while increasing simulator
# parallelism and shortening the GAE fragment.
RIDGE_CONFIGS = (
    "ridge_4096x24_m24576:4096:24:24576",
    "ridge_6144x16_m24576:6144:16:24576",
    "ridge_8192x12_m24576:8192:12:24576",
    "ridge_12288x8_m24576:12288:8:24576",
    "ridge_16384x6_m24576:16384:6:24576",
)

# Ten bounded A40 points.  The 16k x 6 ridge endpoint is intentionally omitted:
# the local qualification showed the same early-return degradation as 12k x 8
# for only a small extra throughput gain.  The two balanced points retain at
# least 12 rollout steps and four minibatches per epoch while using more of the
# A40 for simulation and a moderately larger global rollout batch.
A40_SCREEN_CONFIGS = (
    *RIDGE_CONFIGS[:4],
    "balanced_8192x15_m30720:8192:15:30720",
    "balanced_10240x12_m30720:10240:12:30720",
    "rollout_6144x12_m24576:6144:12:24576",
    "rollout_6144x24_m24576:6144:24:24576",
    "minibatch_8192x12_m12288:8192:12:12288",
    "minibatch_8192x12_m49152:8192:12:49152",
)

# Seed-confirm only the two seed-0 leaders: maximum steady-state collection
# throughput, and the smaller-minibatch point with the best launch-to-target
# time.  Keep this separate from the exploratory screen so confirmations cannot
# silently widen into another sweep.
A40_CONFIRM_CONFIGS = (
    "confirm_throughput_12288x8_m24576:12288:8:24576",
    "confirm_optimizer_8192x12_m12288:8192:12:12288",
)

# Single production run using the three-seed A40 winner.  Keep this preset
# separate from the screening and confirmation matrices so a long convergence
# job cannot accidentally launch an additional configuration.
A40_PRODUCTION_CONFIGS = (
    "production_8192x12_m12288:8192:12:12288",
)

PRESETS = {
    "ridge": RIDGE_CONFIGS,
    "a40-screen": A40_SCREEN_CONFIGS,
    "a40-remaining": A40_SCREEN_CONFIGS[1:],
    "a40-confirm": A40_CONFIRM_CONFIGS,
    "a40-production": A40_PRODUCTION_CONFIGS,
}

LOCKED_OVERRIDE_KEYS = {
    "agent.collector.frames_per_batch",
    "agent.loss.mini_batch_size",
    "agent.loss.epochs",
    "agent.logger.backend",
    "agent.logger.video",
    "agent.ipmd.command_source",
    "agent.ipmd.hl_skill_checkpoint_path",
    "agent.ipmd.hl_skill_finetune_enabled",
    "agent.ipmd.hl_skill_horizon_steps",
    "agent.ipmd.hl_skill_command_mode",
    "agent.ipmd.latent_dim",
    "agent.ipmd.latent_steps_min",
    "agent.ipmd.latent_steps_max",
    "agent.ipmd.latent_learning.command_phase_mode",
    "agent.ipmd.latent_learning.code_latent_dim",
    "agent.ipmd.latent_learning.code_period",
    "agent.ipmd.reward_loss_coeff",
    "agent.ipmd.reward_l2_coeff",
    "agent.ipmd.reward_grad_penalty_coeff",
    "agent.ipmd.reward_logit_reg_coeff",
    "agent.ipmd.reward_param_weight_decay_coeff",
    "agent.ipmd.use_estimated_rewards_for_ppo",
    "agent.ipmd.env_reward_weight",
    "agent.ipmd.bc_coef",
    "agent.ipmd.rollout_bc_coef",
    "env.lafan1_manifest_path",
    "env.dataset_path",
    "env.refresh_zarr_dataset",
    "env.random_reset_step_min",
    "env.random_reset_step_max",
    "env.random_reset_full_trajectory",
    "env.reconstructed_reference_action",
    "env.latent_command_dim",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PROGRESS_RE = re.compile(
    r"iter=(?P<iteration>\d+)/(?P<total_iterations>\d+)\s*\|\s*"
    r"frames=(?P<frames>\d+)/(?P<total_frames>\d+)"
)
RETURN_RE = re.compile(r"r_ep=(?P<return>-?\d+(?:\.\d+)?)")
FPS_RE = re.compile(r"fps=(?P<fps>\d+(?:\.\d+)?)")
TRAINING_TIME_RE = re.compile(r"Training time:\s*(?P<seconds>\d+(?:\.\d+)?)\s*seconds")
TIMESTAMP_RE = re.compile(r"\[(?P<stamp>\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\]")


@dataclass(frozen=True)
class ScaleConfig:
    """One scaling point."""

    label: str
    num_envs: int
    rollout_steps: int
    minibatch_size: int

    @property
    def global_rollout_batch(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def minibatches_per_epoch(self) -> int:
        return math.ceil(self.global_rollout_batch / self.minibatch_size)

    @property
    def optimizer_updates_per_rollout(self) -> int:
        return 5 * self.minibatches_per_epoch

    def iterations_for_frames(self, target_frames: int) -> int:
        return math.ceil(target_frames / self.global_rollout_batch)

    def effective_frames(self, target_frames: int) -> int:
        return self.iterations_for_frames(target_frames) * self.global_rollout_batch


@dataclass(frozen=True)
class ProgressPoint:
    """One periodic RLOpt summary."""

    iteration: int
    frames: int
    episode_return: float | None
    fps: float | None
    timestamp: str | None


def parse_scale_config(raw: str) -> ScaleConfig:
    """Parse ``LABEL:NUM_ENVS:ROLLOUT_STEPS:MINIBATCH_SIZE``."""

    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Expected LABEL:NUM_ENVS:ROLLOUT_STEPS:MINIBATCH_SIZE, got {raw!r}."
        )
    label = parts[0].strip()
    if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"Invalid configuration label {label!r}.")
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"Non-integer scaling value in {raw!r}.") from exc
    if any(value <= 0 for value in values):
        raise ValueError(f"Scaling values must be positive in {raw!r}.")
    config = ScaleConfig(label, *values)
    if config.minibatch_size > config.global_rollout_batch:
        raise ValueError(
            f"{label}: minibatch_size={config.minibatch_size} exceeds "
            f"global_rollout_batch={config.global_rollout_batch}."
        )
    return config


def resolve_configs(preset: str | None, raw_configs: list[str]) -> list[ScaleConfig]:
    """Resolve a preset plus optional explicit configurations."""

    specs: list[str] = []
    if preset:
        specs.extend(PRESETS[preset])
    specs.extend(raw_configs)
    if not specs:
        raise ValueError("Select --preset or provide at least one --config.")
    configs = [parse_scale_config(spec) for spec in specs]
    labels = [config.label for config in configs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Configuration labels must be unique: {labels!r}.")
    return configs


def validate_train_overrides(overrides: list[str]) -> None:
    """Prevent extra overrides from changing the controlled experiment."""

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Expected KEY=VALUE for --train-override, got {override!r}."
            )
        key = override.split("=", 1)[0].strip()
        if not key:
            raise ValueError(f"Empty key in --train-override {override!r}.")
        if key in LOCKED_OVERRIDE_KEYS:
            raise ValueError(f"--train-override may not change controlled key {key!r}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_metadata_fingerprint(dataset_path: Path) -> dict[str, Any]:
    """Hash every Zarr metadata root without scanning array chunks."""

    metadata_files: list[Path] = []
    for root, directory_names, file_names in os.walk(dataset_path, topdown=True):
        root_path = Path(root)
        if "zarr.json" not in file_names:
            continue
        metadata_file = root_path / "zarr.json"
        metadata_files.append(metadata_file)
        try:
            node_type = json.loads(metadata_file.read_text())["node_type"]
        except (KeyError, json.JSONDecodeError):
            node_type = None
        if node_type == "array":
            # Zarr v3 stores array chunks below ``c/``.  A recursive glob walks
            # every one of those directories on NFS even though they cannot
            # contain metadata nodes, which can leave a cluster GPU idle for
            # many minutes during preflight.
            directory_names.clear()
    metadata_files.sort()
    if not metadata_files:
        raise ValueError(f"No zarr.json files found under dataset: {dataset_path}")
    digest = hashlib.sha256()
    entries: list[dict[str, str]] = []
    for metadata_file in metadata_files:
        relative = metadata_file.relative_to(dataset_path).as_posix()
        file_hash = _sha256(metadata_file)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        entries.append({"path": relative, "sha256": file_hash})
    return {
        "algorithm": "sha256(relative_zarr_json_path\\0file_sha256\\n)",
        "file_count": len(entries),
        "fingerprint": digest.hexdigest(),
        "files": entries,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Validate and hash the fixed benchmark inputs."""

    manifest = args.manifest.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    skill_checkpoint = args.skill_checkpoint.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_path)
    if not skill_checkpoint.is_file():
        raise FileNotFoundError(skill_checkpoint)

    manifest_hash = _sha256(manifest)
    skill_hash = _sha256(skill_checkpoint)
    dataset_identity = dataset_metadata_fingerprint(dataset_path)
    expected = {
        "manifest": args.expected_manifest_sha256,
        "skill_checkpoint": args.expected_skill_sha256,
        "dataset_metadata": args.expected_dataset_fingerprint,
    }
    actual = {
        "manifest": manifest_hash,
        "skill_checkpoint": skill_hash,
        "dataset_metadata": dataset_identity["fingerprint"],
    }
    for name, expected_hash in expected.items():
        if expected_hash and actual[name] != expected_hash:
            raise ValueError(
                f"{name} hash mismatch: expected {expected_hash}, got {actual[name]}."
            )
    return {
        "manifest": {"path": str(manifest), "sha256": manifest_hash},
        "dataset": {"path": str(dataset_path), **dataset_identity},
        "skill_checkpoint": {
            "path": str(skill_checkpoint),
            "sha256": skill_hash,
        },
    }


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_provenance() -> dict[str, Any]:
    """Capture source provenance without modifying the worktree."""

    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "status_short": (_git_output("status", "--short") or "").splitlines(),
        "submodules": (_git_output("submodule", "status") or "").splitlines(),
    }


def build_train_command(
    args: argparse.Namespace,
    config: ScaleConfig,
    *,
    effective_frames: int,
) -> list[str]:
    """Construct the fully pinned low-level training command."""

    max_iterations = effective_frames // config.global_rollout_batch
    if Path("/isaac-sim/python.sh").is_file():
        python_command = "/isaac-sim/python.sh"
    else:
        python_command = sys.executable
    command = [
        python_command,
        "scripts/rlopt/train.py",
        "--headless",
        "--device",
        args.device,
        "--num_envs",
        str(config.num_envs),
        "--task",
        "Isaac-Imitation-G1-Latent-v0",
        "--algo",
        "IPMD",
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(max_iterations),
        "--log_interval",
        str(args.log_interval_frames),
        "--kit_args=--/app/extensions/fsWatcherEnabled=false",
        f"agent.collector.frames_per_batch={config.rollout_steps}",
        f"agent.loss.mini_batch_size={config.minibatch_size}",
        "agent.loss.epochs=5",
        "agent.logger.backend=",
        "agent.logger.video=false",
        f"agent.logger.exp_name={args.run_label}_{config.label}_seed{args.seed}",
        "agent.save_interval=1000000000000",
        "agent.ipmd.command_source=hl_skill",
        f"agent.ipmd.hl_skill_checkpoint_path={args.skill_checkpoint.expanduser().resolve()}",
        "agent.ipmd.hl_skill_finetune_enabled=false",
        f"env.lafan1_manifest_path={args.manifest.expanduser().resolve()}",
        f"env.dataset_path={args.dataset_path.expanduser().resolve()}",
        "env.refresh_zarr_dataset=false",
        "env.random_reset_step_min=0",
        "env.random_reset_step_max=200",
        "env.random_reset_full_trajectory=false",
        "env.reconstructed_reference_action=true",
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
        "agent.ipmd.use_estimated_rewards_for_ppo=false",
        "agent.ipmd.env_reward_weight=1.0",
        "agent.ipmd.bc_coef=0.0",
        "agent.ipmd.rollout_bc_coef=0.0",
    ]
    command.extend(args.train_override)
    return command


def _strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_training_log(
    text: str,
    *,
    target_return: float,
    sustain_points: int,
) -> dict[str, Any]:
    """Parse periodic summaries and derive time-to-threshold metrics."""

    points: list[ProgressPoint] = []
    pending: dict[str, Any] | None = None
    training_time_s: float | None = None
    for raw_line in text.splitlines():
        line = _strip_ansi(raw_line)
        progress_match = PROGRESS_RE.search(line)
        if progress_match:
            timestamp_match = TIMESTAMP_RE.search(line)
            pending = {
                "iteration": int(progress_match.group("iteration")),
                "frames": int(progress_match.group("frames")),
                "episode_return": None,
                "fps": None,
                "timestamp": timestamp_match.group("stamp")
                if timestamp_match
                else None,
            }
        if pending is not None:
            return_match = RETURN_RE.search(line)
            fps_match = FPS_RE.search(line)
            if return_match:
                pending["episode_return"] = float(return_match.group("return"))
            if fps_match:
                pending["fps"] = float(fps_match.group("fps"))
            if return_match or fps_match:
                if return_match and not fps_match:
                    continue
                points.append(ProgressPoint(**pending))
                pending = None
        training_time_match = TRAINING_TIME_RE.search(line)
        if training_time_match:
            training_time_s = float(training_time_match.group("seconds"))

    if pending is not None and (
        pending["episode_return"] is not None or pending["fps"] is not None
    ):
        points.append(ProgressPoint(**pending))

    return_points = [point for point in points if point.episode_return is not None]
    first_hit_index = next(
        (
            index
            for index, point in enumerate(return_points)
            if point.episode_return is not None
            and point.episode_return >= target_return
        ),
        None,
    )
    sustained_start_index: int | None = None
    sustained_confirm_index: int | None = None
    if sustain_points > 0:
        for start in range(0, len(return_points) - sustain_points + 1):
            window = return_points[start : start + sustain_points]
            if all(
                point.episode_return is not None
                and point.episode_return >= target_return
                for point in window
            ):
                sustained_start_index = start
                sustained_confirm_index = start + sustain_points - 1
                break

    parsed_timestamps: list[datetime | None] = []
    for point in return_points:
        if point.timestamp is None:
            parsed_timestamps.append(None)
            continue
        parsed_timestamps.append(
            datetime.strptime(point.timestamp, "%m/%d/%y %H:%M:%S")
        )

    final_timestamp = next(
        (
            timestamp
            for timestamp in reversed(parsed_timestamps)
            if timestamp is not None
        ),
        None,
    )

    def _point_payload(index: int | None) -> dict[str, Any] | None:
        if index is None:
            return None
        point = return_points[index]
        payload = asdict(point)
        point_timestamp = parsed_timestamps[index]
        if (
            training_time_s is not None
            and final_timestamp is not None
            and point_timestamp is not None
        ):
            seconds_after_training_start = (
                training_time_s - (final_timestamp - point_timestamp).total_seconds()
            )
            payload["estimated_training_elapsed_s"] = max(
                0.0, seconds_after_training_start
            )
        return payload

    fps_values = [
        point.fps for point in points if point.fps is not None and point.fps > 0
    ]
    return {
        "target_return": target_return,
        "sustain_points": sustain_points,
        "training_time_s": training_time_s,
        "first_hit": _point_payload(first_hit_index),
        "sustained_start": _point_payload(sustained_start_index),
        "sustained_confirmed": _point_payload(sustained_confirm_index),
        "median_logged_fps": statistics.median(fps_values) if fps_values else None,
        "final_episode_return": (
            return_points[-1].episode_return if return_points else None
        ),
        "progress_points": [asdict(point) for point in points],
    }


def _monitor_gpu_memory(
    process: subprocess.Popen[str],
    stop_event: threading.Event,
    samples: list[int],
    *,
    poll_seconds: float,
    vram_limit_mib: int,
    limit_event: threading.Event,
) -> None:
    """Poll total device memory and terminate a run that crosses the guardrail."""

    while not stop_event.wait(poll_seconds):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            continue
        try:
            used_mib = int(result.stdout.strip().splitlines()[0])
        except (IndexError, ValueError):
            continue
        samples.append(used_mib)
        if vram_limit_mib > 0 and used_mib > vram_limit_mib:
            limit_event.set()
            if process.poll() is None:
                process.terminate()
            return


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_one(
    args: argparse.Namespace,
    config: ScaleConfig,
    *,
    output_root: Path,
    effective_frames: int,
) -> dict[str, Any]:
    """Run and summarize one fresh low-level training process."""

    run_dir = output_root / config.label
    run_dir.mkdir(parents=True, exist_ok=False)
    command = build_train_command(args, config, effective_frames=effective_frames)
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    print(f"[RUN] {config.label}: {shlex.join(command)}", flush=True)
    start_wall = datetime.now(timezone.utc)
    start_monotonic = time.perf_counter()
    train_log_path = run_dir / "train.log"
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    stop_event = threading.Event()
    limit_event = threading.Event()
    memory_samples: list[int] = []
    monitor = threading.Thread(
        target=_monitor_gpu_memory,
        args=(process, stop_event, memory_samples),
        kwargs={
            "poll_seconds": args.memory_poll_seconds,
            "vram_limit_mib": args.vram_limit_mib,
            "limit_event": limit_event,
        },
        daemon=True,
    )
    monitor.start()
    with train_log_path.open("w", encoding="utf-8") as train_log:
        for line in process.stdout:
            train_log.write(line)
            train_log.flush()
            print(line, end="", flush=True)
    returncode = process.wait()
    stop_event.set()
    monitor.join(timeout=max(2.0, args.memory_poll_seconds + 1.0))
    end_wall = datetime.now(timezone.utc)
    process_elapsed_s = time.perf_counter() - start_monotonic
    parsed = parse_training_log(
        train_log_path.read_text(encoding="utf-8", errors="replace"),
        target_return=args.target_return,
        sustain_points=args.sustain_points,
    )
    return {
        "config": {
            **asdict(config),
            "global_rollout_batch": config.global_rollout_batch,
            "minibatches_per_epoch": config.minibatches_per_epoch,
            "optimizer_updates_per_rollout": config.optimizer_updates_per_rollout,
        },
        "planned_frames": effective_frames,
        "max_iterations": config.iterations_for_frames(args.target_frames_per_run),
        "returncode": returncode,
        "termination_reason": "vram_limit" if limit_event.is_set() else "completed",
        "peak_vram_mib": max(memory_samples) if memory_samples else None,
        "memory_sample_count": len(memory_samples),
        "process_started_at_utc": start_wall.isoformat(),
        "process_ended_at_utc": end_wall.isoformat(),
        "process_elapsed_s": process_elapsed_s,
        "log_path": str(train_log_path),
        "metrics": parsed,
    }


def build_plan(
    args: argparse.Namespace,
    configs: list[ScaleConfig],
    input_identity: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate the complete pre-execution plan."""

    runs = []
    for config in configs:
        effective_frames = config.effective_frames(args.target_frames_per_run)
        runs.append(
            {
                "config": {
                    **asdict(config),
                    "global_rollout_batch": config.global_rollout_batch,
                    "minibatches_per_epoch": config.minibatches_per_epoch,
                    "optimizer_updates_per_rollout": config.optimizer_updates_per_rollout,
                },
                "requested_frames": args.target_frames_per_run,
                "effective_frames": effective_frames,
                "max_iterations": config.iterations_for_frames(
                    args.target_frames_per_run
                ),
                "command": build_train_command(
                    args, config, effective_frames=effective_frames
                ),
            }
        )
    planned_frames = sum(run["effective_frames"] for run in runs)
    if planned_frames > args.aggregate_frame_cap:
        raise ValueError(
            f"Planned matrix uses {planned_frames:,} frames, exceeding "
            f"--aggregate-frame-cap={args.aggregate_frame_cap:,}."
        )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "preset": args.preset,
        "seed": args.seed,
        "task": "Isaac-Imitation-G1-Latent-v0",
        "algorithm": "IPMD",
        "target_return": args.target_return,
        "sustain_points": args.sustain_points,
        "log_interval_frames": args.log_interval_frames,
        "requested_frames_per_run": args.target_frames_per_run,
        "planned_aggregate_frames": planned_frames,
        "aggregate_frame_cap": args.aggregate_frame_cap,
        "vram_limit_mib": args.vram_limit_mib,
        "fixed_profile": {
            "command_source": "hl_skill",
            "hl_skill_horizon_steps": 10,
            "hl_skill_command_mode": "z",
            "code_latent_dim": 256,
            "command_phase_mode": "sin_cos",
            "latent_command_dim": 258,
            "ppo_epochs": 5,
            "reward_source": "environment",
            "skill_finetune": False,
            "video": False,
            "random_reset_step_range": [0, 200],
        },
        "input_identity": input_identity,
        "git": git_provenance(),
        "runs": runs,
    }


def print_plan_summary(plan: dict[str, Any]) -> None:
    """Print the actionable plan without dumping every dataset metadata file."""

    input_identity = plan["input_identity"]
    print(
        "[PLAN] "
        f"runs={len(plan['runs'])} "
        f"planned_frames={plan['planned_aggregate_frames']:,} "
        f"cap={plan['aggregate_frame_cap']:,} "
        f"target_return={plan['target_return']} "
        f"sustain_points={plan['sustain_points']}",
        flush=True,
    )
    print(
        "[INPUT] "
        f"manifest_sha256={input_identity['manifest']['sha256']} "
        f"skill_sha256={input_identity['skill_checkpoint']['sha256']} "
        f"dataset_fingerprint={input_identity['dataset']['fingerprint']} "
        f"dataset_metadata_files={input_identity['dataset']['file_count']}",
        flush=True,
    )
    for run in plan["runs"]:
        config = run["config"]
        print(
            "[CONFIG] "
            f"{config['label']} "
            f"num_envs={config['num_envs']} "
            f"rollout_steps={config['rollout_steps']} "
            f"global_batch={config['global_rollout_batch']} "
            f"minibatch={config['minibatch_size']} "
            f"updates_per_rollout={config['optimizer_updates_per_rollout']} "
            f"iterations={run['max_iterations']} "
            f"effective_frames={run['effective_frames']}",
            flush=True,
        )
        print(f"[COMMAND] {shlex.join(run['command'])}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="LABEL:NUM_ENVS:ROLLOUT_STEPS:MINIBATCH_SIZE",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--skill-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=None)
    parser.add_argument("--expected-skill-sha256", default=None)
    parser.add_argument("--expected-dataset-fingerprint", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", default="latent_scale")
    parser.add_argument("--target-frames-per-run", type=int, default=10_000_000)
    parser.add_argument("--aggregate-frame-cap", type=int, default=50_000_000)
    parser.add_argument("--target-return", type=float, default=7.5)
    parser.add_argument("--sustain-points", type=int, default=3)
    parser.add_argument("--log-interval-frames", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vram-limit-mib", type=int, default=0)
    parser.add_argument("--memory-poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Additional Hydra override; recorded verbatim and appended last.",
    )
    args = parser.parse_args()
    for name in (
        "target_frames_per_run",
        "aggregate_frame_cap",
        "sustain_points",
        "log_interval_frames",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.memory_poll_seconds <= 0:
        parser.error("--memory-poll-seconds must be positive.")
    if args.vram_limit_mib < 0:
        parser.error("--vram-limit-mib cannot be negative.")
    return args


def main() -> None:
    args = parse_args()
    validate_train_overrides(list(args.train_override))
    configs = resolve_configs(args.preset, list(args.config))
    input_identity = validate_inputs(args)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite benchmark output root: {output_root}"
        )
    plan = build_plan(args, configs, input_identity)
    print_plan_summary(plan)
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(output_root / "benchmark_plan.json", plan)
    results: list[dict[str, Any]] = []
    aggregate = {
        "schema_version": 1,
        "plan_path": str(output_root / "benchmark_plan.json"),
        "status": "running",
        "results": results,
    }
    _atomic_write_json(output_root / "benchmark_results.json", aggregate)
    for config, run_plan in zip(configs, plan["runs"], strict=True):
        result = run_one(
            args,
            config,
            output_root=output_root,
            effective_frames=int(run_plan["effective_frames"]),
        )
        results.append(result)
        aggregate["results"] = results
        _atomic_write_json(output_root / "benchmark_results.json", aggregate)
        if result["returncode"] != 0 and not args.continue_on_error:
            aggregate["status"] = "failed"
            _atomic_write_json(output_root / "benchmark_results.json", aggregate)
            raise RuntimeError(
                f"Configuration {config.label} failed with return code "
                f"{result['returncode']}."
            )
    aggregate["status"] = (
        "complete"
        if all(result["returncode"] == 0 for result in results)
        else "complete_with_failures"
    )
    aggregate["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(output_root / "benchmark_results.json", aggregate)
    print(f"[RESULT] {output_root / 'benchmark_results.json'}", flush=True)


if __name__ == "__main__":
    main()
