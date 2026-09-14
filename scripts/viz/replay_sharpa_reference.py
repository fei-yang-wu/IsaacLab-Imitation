#!/usr/bin/env python3
"""Run a finite zero-residual replay of a Sharpa ILTools reference."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference_manifest", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--video", action="store_true", help="Record an MP4 replay.")
parser.add_argument(
    "--video_dir",
    default="logs/rsl_rl/sharpa_v2d/reference_videos",
    help="Directory for the recorded MP4 (used with --video).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Video length in control steps; zero uses --steps.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ["SHARPA_REFERENCE_MANIFEST"] = args.reference_manifest
if args.video:
    args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_imitation.tasks  # noqa: F401, E402


def main() -> None:
    cfg = parse_env_cfg(
        "Isaac-Sharpa-V2D-PhysX-v0",
        device=args.device,
        num_envs=args.num_envs,
    )
    cfg.reference_manifest = args.reference_manifest
    cfg.commands.sharpa_reference.reference_manifest = args.reference_manifest
    env = gym.make(
        "Isaac-Sharpa-V2D-PhysX-v0",
        cfg=cfg,
        render_mode="rgb_array" if args.video else None,
    )
    video_length = args.video_length if args.video_length > 0 else args.steps
    if args.video:
        video_dir = os.path.abspath(os.path.expanduser(args.video_dir))
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print(f"Recording Sharpa reference video in {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    observation, _ = env.reset()
    rollout_steps = min(args.steps, video_length) if args.video else args.steps
    for _ in range(rollout_steps):
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        observation, reward, terminated, truncated, _ = env.step(action)
        tensors = [reward, terminated, truncated]
        if isinstance(observation, dict):
            tensors.extend(
                value
                for value in observation.values()
                if isinstance(value, torch.Tensor)
            )
        if any(not torch.isfinite(value.float()).all() for value in tensors):
            raise RuntimeError("Sharpa replay produced a non-finite tensor.")
    print(
        f"Sharpa replay completed {rollout_steps} steps in {args.num_envs} environment(s)."
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
