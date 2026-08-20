#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate one Vega-Wuji policy checkpoint against one reference motion.

This is the policy-side companion to ``replay_vega_wuji_reference.py``. It
uses the same selected Reference, Newton backend, and 20 Hz horizon, records a
deterministic policy video, and writes joint tracking metrics. The reference
articulation is not teleported during this run; only the policy controls the
robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--motion-index", type=int, default=0)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task", default="Isaac-Imitation-Vega-Wuji-v0")
parser.add_argument("--agent", default="rlopt_ppo_cfg_entry_point")
parser.add_argument(
    "--physics", choices=("default", "newton_mjwarp"), default="newton_mjwarp"
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video-length", type=int, default=None)
parser.add_argument(
    "--output-dir", type=Path, default=Path("logs/vega_wuji_policy_eval")
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--debug-vis",
    action="store_true",
    default=False,
    help="Draw command markers: live and commanded wrist, object, and "
    "fingertip frames. Contact markers appear only when the Reference "
    "carries active contact geometry.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_motion(path: Path, index: int) -> tuple[Path, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")
    if path.suffix.lower() != ".json":
        return path, {"name": path.stem}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("motions") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Reference manifest has no motions: {path}")
    if not 0 <= index < len(entries):
        raise IndexError(f"motion-index {index} is outside [0, {len(entries)})")
    entry = entries[index]
    metadata = dict(entry) if isinstance(entry, dict) else {"path": entry}
    motion_value = metadata.get("path")
    if not isinstance(motion_value, str) or not motion_value:
        raise ValueError(f"Manifest motion {index} has no path: {path}")
    motion_path = Path(motion_value).expanduser()
    if not motion_path.is_absolute():
        motion_path = path.parent / motion_path
    motion_path = motion_path.resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Manifest motion is missing: {motion_path}")
    expected_hash = metadata.get("sha256")
    if expected_hash is not None and str(expected_hash).lower() != _sha256(motion_path):
        raise ValueError(f"Manifest hash mismatch: {motion_path}")
    return motion_path, metadata


reference_path = args_cli.reference.expanduser().resolve()
motion_path, motion_metadata = _select_motion(reference_path, args_cli.motion_index)
with np.load(motion_path, allow_pickle=False) as data:
    if "qpos" not in data:
        raise ValueError(f"Reference NPZ must contain qpos: {motion_path}")
    reference_frames = int(np.asarray(data["qpos"]).shape[0])
if reference_frames < 2:
    raise ValueError(f"Reference must contain at least two frames: {motion_path}")
replay_steps = args_cli.video_length or reference_frames - 1
if replay_steps <= 0 or replay_steps > reference_frames - 1:
    raise ValueError(
        f"video length must be in [1, {reference_frames - 1}], got {replay_steps}"
    )

if not args_cli.checkpoint.expanduser().is_file():
    raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")

hydra_args.extend(
    [
        f"env.reference_path={reference_path}",
        "env.randomize_reference=false",
        "env.commands.motion.always_reset_to_first_frame=true",
        "env.commands.motion.warmup_steps=0",
        "env.commands.motion.force_unassisted_object_control=true",
        f"env.reference_motion_index={args_cli.motion_index}",
        f"physics={args_cli.physics}",
    ]
)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import PPO
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp
from tensordict.nn import InteractionType


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
        bind_command_interface,
    )

    if bind_command_interface(agent_cfg, env_cfg) is None:
        sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
        if callable(sync_input_keys):
            sync_input_keys()

    env_cfg.scene.num_envs = 1
    agent_cfg.env.num_envs = 1
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.commands.motion.debug_vis = args_cli.debug_vis
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.episode_length_s = 1.0e9
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None and hasattr(terminations, "time_out"):
        terminations.time_out = None
    logger_cfg = getattr(agent_cfg, "logger", None)
    if logger_cfg is not None:
        logger_cfg.backend = ""
        logger_cfg.video = False

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output_dir)
    raw_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    task_env = raw_env.unwrapped
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(output_dir / "videos"),
            # Episode-triggered recording includes the synchronized reset frame.
            "episode_trigger": lambda episode: episode == 0,
            "video_length": replay_steps,
            "disable_logger": True,
        }
        print("[INFO] Recording Vega-Wuji policy video")
        print_dict(video_kwargs, nesting=4)
        raw_env = gym.wrappers.RecordVideo(raw_env, **video_kwargs)

    env = IsaacLabWrapper(raw_env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(
            RewardSum(), StepCounter(replay_steps + 1), RewardClipping(-10.0, 5.0)
        ),
    )
    agent = PPO(env=env, config=agent_cfg)
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(str(checkpoint_path))
    collector_policy = getattr(agent, "deployment_policy", None)
    if collector_policy is None:
        collector_policy = agent.collector_policy
    collector_policy.eval()

    with torch.inference_mode():
        td = env.reset()
    motion_command = task_env.command_manager.get_term("motion")
    try:
        support_contact_sensor = task_env.scene["robot_support_contacts"]
    except KeyError:
        support_contact_sensor = None
    if torch.any(motion_command.virtual_object_controller_scale != 0.0):
        raise RuntimeError(
            "Policy evaluation requires virtual object control to be exactly "
            "zero from reset frame zero."
        )
    metrics: list[dict[str, float | int]] = []
    with torch.inference_mode(), set_exploration_type(InteractionType.DETERMINISTIC):
        for step_index in range(replay_steps):
            td = collector_policy(td)
            td = env.step(td)
            reference_pos = task_env.current_reference_joint_pos[0]
            reference_vel = task_env.current_reference_joint_vel[0]
            joint_pos = task_env.robot.data.joint_pos.torch[0]
            joint_vel = task_env.robot.data.joint_vel.torch[0]
            pos_error = (joint_pos - reference_pos).abs()
            vel_error = (joint_vel - reference_vel).abs()
            object_position_error = torch.linalg.vector_norm(
                motion_command.object_position_e
                - motion_command.object_body_position_command_e,
                dim=-1,
            )
            object_quaternion_dot = (
                motion_command.object_orientation_e
                * motion_command.object_body_wxyz_command_e
            ).sum(dim=-1)
            object_orientation_error = 2.0 * torch.acos(
                object_quaternion_dot.abs().clamp(0.0, 1.0)
            )
            object_twist_error = torch.linalg.vector_norm(
                motion_command.object_twist_w
                - motion_command.object_body_twist_command_w,
                dim=-1,
            )
            support_force = 0.0
            if support_contact_sensor is not None:
                force_matrix = support_contact_sensor.data.force_matrix_w
                if force_matrix is None:
                    raise RuntimeError(
                        "robot_support_contacts did not publish force_matrix_w."
                    )
                force_matrix = getattr(force_matrix, "torch", force_matrix)
                support_force = float(
                    torch.linalg.vector_norm(
                        force_matrix[0],
                        dim=-1,
                    )
                    .max()
                    .item()
                )
            metrics.append(
                {
                    "step": step_index + 1,
                    "reference_frame": int(task_env.reference_frame[0].item()),
                    "joint_position_mae_rad": float(pos_error.mean().item()),
                    "joint_position_rmse_rad": float(
                        torch.sqrt((joint_pos - reference_pos).square().mean()).item()
                    ),
                    "joint_position_max_error_rad": float(pos_error.max().item()),
                    "joint_velocity_mae_rad_s": float(vel_error.mean().item()),
                    "joint_velocity_rmse_rad_s": float(
                        torch.sqrt((joint_vel - reference_vel).square().mean()).item()
                    ),
                    "object_position_mae_m": float(object_position_error.mean().item()),
                    "object_orientation_mae_rad": float(
                        object_orientation_error.mean().item()
                    ),
                    "object_twist_mae": float(object_twist_error.mean().item()),
                    "support_contact_force_max_n": support_force,
                    "environment_reward": float(task_env.reward_buf[0].item()),
                    "virtual_object_controller_scale": float(
                        motion_command.virtual_object_controller_scale.max().item()
                    ),
                }
            )
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )

    env.close()

    def _mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in metrics]))

    summary = {
        "steps": replay_steps,
        "reference_frames": reference_frames,
        "mean_joint_position_mae_rad": _mean("joint_position_mae_rad"),
        "mean_joint_position_rmse_rad": _mean("joint_position_rmse_rad"),
        "max_joint_position_error_rad": max(
            float(row["joint_position_max_error_rad"]) for row in metrics
        ),
        "mean_joint_velocity_mae_rad_s": _mean("joint_velocity_mae_rad_s"),
        "mean_joint_velocity_rmse_rad_s": _mean("joint_velocity_rmse_rad_s"),
        "mean_object_position_mae_m": _mean("object_position_mae_m"),
        "mean_object_orientation_mae_rad": _mean("object_orientation_mae_rad"),
        "mean_object_twist_mae": _mean("object_twist_mae"),
        "mean_environment_reward": _mean("environment_reward"),
        "max_support_contact_force_n": max(
            float(row["support_contact_force_max_n"]) for row in metrics
        ),
        "max_virtual_object_controller_scale": max(
            float(row["virtual_object_controller_scale"]) for row in metrics
        ),
    }
    payload = {
        "task": args_cli.task,
        "physics": args_cli.physics,
        "seed": args_cli.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "reference_manifest_or_npz": str(args_cli.reference.expanduser().resolve()),
        "selected_motion": str(motion_path),
        "selected_motion_metadata": motion_metadata,
        "selected_motion_sha256": _sha256(motion_path),
        "summary": summary,
        "per_step": metrics,
        "video_paths": [
            str(path.resolve()) for path in sorted(output_dir.rglob("*.mp4"))
        ],
    }
    metrics_path = output_dir / "policy_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("[POLICY_TRACKING_SUMMARY] " + json.dumps(summary, sort_keys=True))
    for video in payload["video_paths"]:
        print(f"[VIDEO] {video}")
    print(f"[INFO] Policy metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
