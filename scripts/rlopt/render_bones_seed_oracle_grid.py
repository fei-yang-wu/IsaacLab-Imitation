#!/usr/bin/env python3
# ruff: noqa: E402

"""Render a grid of BONES-SEED motions driven by oracle skill commands.

Each Isaac environment is permanently assigned one randomly selected expert
trajectory. The frozen high-level skill encoder converts the current 25-step
reference segment into the latent command consumed by the low-level IPMD policy.
Selections are sampled without replacement and recorded next to the video. When
a reference finishes, that humanoid resets to frame zero and repeats the same
motion while the shared recording continues.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Render distinct BONES-SEED trajectories in parallel with an oracle "
        "high-level skill encoder and a trained low-level IPMD policy."
    )
)
parser.add_argument(
    "--checkpoint",
    required=True,
    help=(
        "Low-level model checkpoint, or a run directory containing "
        "models/model_step_*.pt."
    ),
)
parser.add_argument(
    "--skill-checkpoint",
    "--skill_checkpoint",
    dest="skill_checkpoint",
    required=True,
    help="Frozen high-level skill encoder checkpoint used by the low-level policy.",
)
parser.add_argument(
    "--data-root",
    "--data_root",
    dest="data_root",
    default="data/bones_seed_100",
    help="Prepared BONES-SEED-100 root containing manifests/ and g1_hl_diffsr/.",
)
parser.add_argument(
    "--output-dir",
    "--output_dir",
    dest="output_dir",
    default=None,
    help="Output directory. Defaults to a timestamped logs/rlopt_eval run.",
)
parser.add_argument(
    "--task",
    default="Isaac-Imitation-G1-Latent-v0",
    help="Latent-conditioned Isaac Lab task.",
)
parser.add_argument(
    "--num-envs",
    "--num_envs",
    dest="num_envs",
    type=int,
    default=64,
    help="Number of distinct motions/humanoids to render (default: 64).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for environment setup and motion selection.",
)
parser.add_argument(
    "--max-steps",
    "--max_steps",
    dest="max_steps",
    type=int,
    default=None,
    help="Optional rollout limit. By default, run through the longest selection.",
)
parser.add_argument(
    "--grid-spacing",
    "--grid_spacing",
    dest="grid_spacing",
    type=float,
    default=2.5,
    help=(
        "Spacing in meters between humanoid environments; defaults to the "
        "G1 training environment spacing (2.5)."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record the 64-humanoid overview video.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Throttle playback to the environment control rate when possible.",
)
parser.add_argument(
    "--disable-fabric",
    "--disable_fabric",
    dest="disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import random
import re
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMD
from tensordict.nn import InteractionType
from torch import Tensor
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401


MODEL_STEP_PATTERN = re.compile(r"^model_step_(\d+)\.pt$")
DEFAULT_MANIFEST = Path("manifests/g1_bones_seed_100_manifest.json")
DEFAULT_DATASET_CACHE = Path("g1_hl_diffsr")
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
# Match the low, three-quarter side view used by the policy/reference playback.
# The vector is scaled to the selected environment grid in _set_overview_camera.
TRAINING_CAMERA_OFFSET = (3.0, -5.0, 2.0)


def _resolve_checkpoint(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates: list[tuple[int, Path]] = []
    for search_root in (path, path / "models"):
        if not search_root.is_dir():
            continue
        for candidate in search_root.glob("model_step_*.pt"):
            match = MODEL_STEP_PATTERN.match(candidate.name)
            if match is not None:
                candidates.append((int(match.group(1)), candidate.resolve()))
    if not candidates:
        raise FileNotFoundError(
            f"No model_step_*.pt checkpoints found in {path} or {path / 'models'}."
        )
    return max(candidates, key=lambda item: item[0])[1]


def _state_dict_from_mapping(value: Any, *, label: str) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or is not a state dictionary.")
    state = {str(key): tensor for key, tensor in value.items()}
    if not state or not all(isinstance(tensor, Tensor) for tensor in state.values()):
        raise ValueError(f"{label} does not contain tensor parameters.")
    return state


def _validate_checkpoint_pair(policy_path: Path, skill_path: Path) -> dict[str, Any]:
    policy_checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    skill_checkpoint = torch.load(skill_path, map_location="cpu", weights_only=False)

    sampler_state = policy_checkpoint.get("hl_skill_command_sampler_state_dict")
    if not isinstance(sampler_state, Mapping):
        raise ValueError(
            "The low-level checkpoint does not contain an oracle high-level skill "
            "sampler state. Use a checkpoint trained with command_source=hl_skill."
        )
    policy_encoder = _state_dict_from_mapping(
        sampler_state.get("skill_encoder_state_dict"),
        label="Low-level checkpoint skill encoder",
    )
    skill_encoder = _state_dict_from_mapping(
        skill_checkpoint.get("skill_encoder_state_dict"),
        label="Skill checkpoint encoder",
    )

    if set(policy_encoder) != set(skill_encoder):
        missing = sorted(set(policy_encoder) - set(skill_encoder))
        extra = sorted(set(skill_encoder) - set(policy_encoder))
        raise ValueError(
            "Low-level and skill checkpoints have different encoder parameters: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    mismatches = [
        key
        for key in policy_encoder
        if policy_encoder[key].shape != skill_encoder[key].shape
        or not torch.equal(policy_encoder[key], skill_encoder[key])
    ]
    if mismatches:
        raise ValueError(
            "The supplied skill checkpoint does not match the encoder embedded in "
            f"the low-level checkpoint. First mismatches: {mismatches[:5]}."
        )

    config = skill_checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Skill checkpoint is missing its config mapping.")
    horizon_steps = int(config.get("horizon_steps", 0))
    z_dim = int(config.get("z_dim", 0))
    if horizon_steps != 25 or z_dim != 256:
        raise ValueError(
            "This BONES-100 policy expects a horizon-25, z-dim-256 skill encoder; "
            f"received horizon_steps={horizon_steps}, z_dim={z_dim}."
        )
    return {"horizon_steps": horizon_steps, "z_dim": z_dim}


def _validate_manifest(manifest_path: Path, num_envs: int) -> int:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["dataset"]["trajectories"]["lafan1_csv"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid BONES-SEED manifest: {manifest_path}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"Manifest trajectory list is not a list: {manifest_path}")
    names = [str(entry.get("name", "")).strip() for entry in entries]
    unique_names = {name for name in names if name}
    if len(unique_names) != len(entries):
        raise ValueError(
            "BONES-100 grid playback requires one unique motion name per manifest entry."
        )
    if num_envs > len(unique_names):
        raise ValueError(
            f"Requested {num_envs} distinct motions, but the manifest has "
            f"{len(unique_names)}."
        )
    return len(unique_names)


def _unwrap_imitation_env(env: Any) -> ImitationRLEnv:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ImitationRLEnv):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if isinstance(unwrapped, ImitationRLEnv):
            return unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError("Could not unwrap an ImitationRLEnv from the environment.")


def _disable_terms(
    config_group: Any, *, label: str, keep: frozenset[str] = frozenset()
) -> None:
    if config_group is None:
        return
    disabled: list[str] = []
    for name in getattr(config_group, "__dataclass_fields__", {}):
        if name in keep:
            continue
        if getattr(config_group, name, None) is None:
            continue
        setattr(config_group, name, None)
        disabled.append(name)
    if disabled:
        print(f"[INFO] Disabled {label}: {', '.join(sorted(disabled))}")


def _configure_exact_reference_reset(env_cfg: Any) -> None:
    events = getattr(env_cfg, "events", None)
    if events is None:
        raise ValueError("Task configuration does not expose reset events.")

    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        if hasattr(events, name):
            setattr(events, name, None)

    reset_term = getattr(events, "reset_reference_state", None)
    if reset_term is None:
        raise ValueError("Task configuration is missing reset_reference_state.")
    reset_term.params["pose_range"] = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_term.params["velocity_range"] = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_term.params["joint_position_range"] = (0.0, 0.0)


def _ordered_trajectories(base_env: ImitationRLEnv) -> list[tuple[str, str, str]]:
    ordered = getattr(base_env.trajectory_manager, "_ordered_traj_list", None)
    if not ordered:
        raise RuntimeError("The trajectory manager does not expose trajectories.")
    return [
        (str(dataset), str(motion), str(trajectory))
        for dataset, motion, trajectory in ordered
    ]


def _select_distinct_motion_ranks(
    base_env: ImitationRLEnv, *, count: int, seed: int
) -> Tensor:
    ordered = _ordered_trajectories(base_env)
    unique_ranks: list[int] = []
    seen_motions: set[str] = set()
    for rank, (_dataset, motion, _trajectory) in enumerate(ordered):
        if motion in seen_motions:
            continue
        seen_motions.add(motion)
        unique_ranks.append(rank)
    if count > len(unique_ranks):
        raise ValueError(
            f"Requested {count} distinct motions, but the loaded cache has "
            f"{len(unique_ranks)}."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(len(unique_ranks), generator=generator)[:count]
    return torch.tensor(
        [unique_ranks[int(index)] for index in permutation], dtype=torch.long
    )


def _force_assignments_on_reset(
    base_env: ImitationRLEnv, selected_ranks: Tensor
) -> None:
    tm = base_env.trajectory_manager
    selected_cpu = selected_ranks.detach().cpu().to(dtype=torch.long)

    def _custom_reset_fn(env_ids: Tensor, _num_trajectories: int) -> Tensor:
        lookup = selected_cpu.to(device=env_ids.device)
        return lookup.index_select(0, env_ids.to(dtype=torch.long))

    tm.reset_schedule = "custom"
    tm.custom_reset_fn = _custom_reset_fn
    tm.reset_start_step = 0
    tm.wrap_steps = False
    base_env._random_reset_full_trajectory = False
    base_env._random_reset_step_min = 0
    base_env._random_reset_step_max = 0


def _set_overview_camera(base_env: ImitationRLEnv) -> None:
    origins = base_env.scene.env_origins.detach().cpu()
    minimum = origins.amin(dim=0)
    maximum = origins.amax(dim=0)
    center = 0.5 * (minimum + maximum)
    span = max(
        float(maximum[0] - minimum[0]),
        float(maximum[1] - minimum[1]),
        float(args_cli.grid_spacing),
    )
    lookat = center.clone()
    lookat[2] = 0.9
    # Preserve the familiar training/playback camera direction while scaling
    # its distance so the complete multi-environment grid remains in frame.
    camera_offset = torch.tensor(TRAINING_CAMERA_OFFSET, dtype=lookat.dtype)
    # A quarter-span scale keeps approximately the old camera-to-grid distance,
    # avoiding crop while lowering the elevation angle substantially.
    eye = lookat + camera_offset * (0.25 * span)
    base_env.sim.set_camera_view(eye.tolist(), lookat.tolist())
    print(f"[INFO] Overview camera eye={eye.tolist()} lookat={lookat.tolist()}")


def _assignment_payload(
    base_env: ImitationRLEnv,
    *,
    policy_path: Path,
    skill_path: Path,
    manifest_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    tm = base_env.trajectory_manager
    rows: list[dict[str, Any]] = []
    for env_id in range(base_env.num_envs):
        dataset, motion, trajectory = tm.get_env_traj_info(env_id)
        rank = int(tm.env_traj_rank[env_id].item())
        rows.append(
            {
                "env_id": env_id,
                "trajectory_rank": rank,
                "dataset": dataset,
                "motion": motion,
                "trajectory": trajectory,
                "trajectory_length_steps": int(tm._length[rank].item()),
            }
        )
    return {
        "seed": int(args_cli.seed),
        "num_envs": int(base_env.num_envs),
        "max_steps": int(max_steps),
        "policy_checkpoint": str(policy_path),
        "skill_checkpoint": str(skill_path),
        "manifest": str(manifest_path),
        "command_source": "hl_skill",
        "command_mode": "z",
        "horizon_steps": 25,
        "restart_finished_motions": True,
        "assignments": rows,
    }


def _run_directory() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs", "rlopt_eval", "bones_seed_oracle_grid", timestamp).resolve()


@hydra_task_config(args_cli.task, "rlopt_ipmd_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: Any,
) -> None:
    if args_cli.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args_cli.max_steps is not None and args_cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive when provided.")
    if args_cli.grid_spacing <= 0.0:
        raise ValueError("--grid-spacing must be positive.")

    data_root = Path(args_cli.data_root).expanduser().resolve()
    manifest_path = data_root / DEFAULT_MANIFEST
    dataset_path = data_root / DEFAULT_DATASET_CACHE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BONES-SEED manifest not found: {manifest_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"BONES-SEED cache not found: {dataset_path}")
    manifest_motion_count = _validate_manifest(manifest_path, args_cli.num_envs)

    policy_path = _resolve_checkpoint(args_cli.checkpoint)
    skill_path = Path(args_cli.skill_checkpoint).expanduser().resolve()
    if not skill_path.is_file():
        raise FileNotFoundError(f"Skill checkpoint not found: {skill_path}")
    skill_config = _validate_checkpoint_pair(policy_path, skill_path)
    print(f"[INFO] Resolved policy checkpoint: {policy_path}")
    print(
        "[INFO] Verified matching skill encoder: "
        f"{skill_path} (horizon={skill_config['horizon_steps']}, "
        f"z_dim={skill_config['z_dim']})"
    )
    print(f"[INFO] BONES-SEED manifest exposes {manifest_motion_count} unique motions.")

    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.scene.env_spacing = float(args_cli.grid_spacing)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.lafan1_manifest_path = str(manifest_path)
    env_cfg.dataset_path = str(dataset_path)
    resolve_manifest_config = getattr(env_cfg, "_resolve_manifest_config", None)
    if not callable(resolve_manifest_config):
        raise TypeError(
            f"Task {args_cli.task!r} does not expose manifest-driven configuration."
        )
    resolve_manifest_config(dataset_path_explicit=True)
    env_cfg.refresh_zarr_dataset = False
    env_cfg.wrap_steps = False
    env_cfg.random_reset_step_min = 0
    env_cfg.random_reset_step_max = 0
    env_cfg.random_reset_full_trajectory = False
    env_cfg.episode_length_s = 1.0e9
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.resolution = (VIDEO_WIDTH, VIDEO_HEIGHT)

    _disable_terms(
        getattr(env_cfg, "terminations", None),
        label="termination terms",
        keep=frozenset({"reference_finished"}),
    )
    _disable_terms(getattr(env_cfg, "rewards", None), label="reward terms")
    _configure_exact_reference_reset(env_cfg)

    agent_cfg.env.num_envs = int(args_cli.num_envs)
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.env.device = str(env_cfg.sim.device)
    agent_cfg.device = str(env_cfg.sim.device)
    agent_cfg.optim.device = str(env_cfg.sim.device)
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.collector.frames_per_batch *= int(args_cli.num_envs)
    agent_cfg.logger.backend = ""
    agent_cfg.logger.video = False
    agent_cfg.ipmd.use_latent_command = True
    agent_cfg.ipmd.command_source = "hl_skill"
    agent_cfg.ipmd.hl_skill_checkpoint_path = str(skill_path)
    agent_cfg.ipmd.hl_skill_horizon_steps = 25
    agent_cfg.ipmd.hl_skill_command_mode = "z"
    agent_cfg.ipmd.hl_skill_finetune_enabled = False
    agent_cfg.ipmd.latent_steps_min = 25
    agent_cfg.ipmd.latent_steps_max = 25

    log_dir = _run_directory()
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Writing BONES-SEED oracle grid output to: {log_dir}")

    raw_gym_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(raw_gym_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    base_env = _unwrap_imitation_env(raw_gym_env)
    selected_ranks = _select_distinct_motion_ranks(
        base_env, count=int(args_cli.num_envs), seed=int(args_cli.seed)
    )
    _force_assignments_on_reset(base_env, selected_ranks)

    tm = base_env.trajectory_manager
    selected_lengths = tm._length.index_select(
        0, selected_ranks.to(device=tm._state_device)
    )
    natural_max_steps = int(selected_lengths.max().item())
    max_steps = (
        int(args_cli.max_steps) if args_cli.max_steps is not None else natural_max_steps
    )
    max_steps = max(1, min(max_steps, natural_max_steps))
    if args_cli.video:
        _set_overview_camera(base_env)

    gym_env: Any = raw_gym_env
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(log_dir / "videos" / "bones_seed_oracle_grid"),
            "step_trigger": lambda step: step == 0,
            "video_length": max_steps,
            "disable_logger": True,
        }
        print("[INFO] Recording BONES-SEED oracle grid video.")
        print_dict(video_kwargs, nesting=4)
        gym_env = gym.wrappers.RecordVideo(gym_env, **video_kwargs)

    wrapped_env = IsaacLabWrapper(gym_env)
    wrapped_env = wrapped_env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=wrapped_env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=wrapped_env,
        transform=Compose(
            RewardSum(),
            StepCounter(max_steps + 1),
            RewardClipping(-10.0, 5.0),
        ),
    )

    agent = IPMD(env=env, config=agent_cfg)
    print(f"[INFO] Loading low-level checkpoint: {policy_path}")
    agent.load_model(str(policy_path))
    collector_policy = agent.collector_policy
    collector_policy.eval()

    td = env.reset()
    actual_ranks = tm.env_traj_rank.detach().cpu().to(dtype=torch.long)
    if not torch.equal(actual_ranks, selected_ranks):
        raise RuntimeError(
            "Trajectory assignment changed during reset: "
            f"expected={selected_ranks.tolist()}, actual={actual_ranks.tolist()}."
        )

    assignments = _assignment_payload(
        base_env,
        policy_path=policy_path,
        skill_path=skill_path,
        manifest_path=manifest_path,
        max_steps=max_steps,
    )
    assignment_path = log_dir / "assignments.json"
    assignment_path.write_text(
        json.dumps(assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[INFO] Wrote environment-to-motion assignments: {assignment_path}")
    for row in assignments["assignments"]:
        print(
            "[INFO] "
            f"env={row['env_id']:02d} rank={row['trajectory_rank']:03d} "
            f"steps={row['trajectory_length_steps']:04d} motion={row['motion']}"
        )

    dt = getattr(base_env, "step_dt", None)
    print(
        f"[INFO] Starting deterministic oracle rollout for {max_steps} steps "
        f"({args_cli.num_envs} distinct motions)."
    )
    print(
        "[INFO] Finished references reset independently and repeat their assigned "
        "motion from frame zero."
    )
    timestep = 0
    try:
        while simulation_app.is_running() and timestep < max_steps:
            start_time = time.time()
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                td = collector_policy(td)
                td = env.step(td)
                td = step_mdp(
                    td, exclude_reward=True, exclude_done=False, exclude_action=True
                )
            timestep += 1

            if args_cli.real_time and dt is not None:
                sleep_time = float(dt) - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        final_ranks = tm.env_traj_rank.detach().cpu().to(dtype=torch.long)
        if not torch.equal(final_ranks, selected_ranks):
            raise RuntimeError(
                "Trajectory assignment changed during rollout: "
                f"expected={selected_ranks.tolist()}, actual={final_ranks.tolist()}."
            )
    finally:
        env.close()

    print(f"[INFO] Completed {timestep} rollout steps.")
    if args_cli.video:
        print(
            f"[INFO] Video directory: {log_dir / 'videos' / 'bones_seed_oracle_grid'}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
