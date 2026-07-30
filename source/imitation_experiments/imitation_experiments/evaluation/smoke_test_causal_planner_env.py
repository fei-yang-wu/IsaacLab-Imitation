#!/usr/bin/env python3
"""Exercise the causal planner observation contract in a live Isaac environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Imitation-G1-Latent-v0")
parser.add_argument("--motion_manifest", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--history_steps", type=int, default=9)
parser.add_argument("--horizon_steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_imitation.tasks  # noqa: E402, F401
import isaaclab_tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=int(args_cli.num_envs),
        use_fabric=True,
    )
    env_cfg.lafan1_manifest_path = str(args_cli.motion_manifest.expanduser().resolve())
    resolve_manifest = getattr(env_cfg, "_resolve_manifest_config", None)
    if callable(resolve_manifest):
        resolve_manifest()
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    try:
        env.reset()
        spec = base_env.causal_planner_observation_spec(
            history_steps=int(args_cli.history_steps)
        )
        live_before = base_env.current_causal_planner_observation(
            history_steps=int(args_cli.history_steps)
        ).get(("planner", "state_history"))
        assert tuple(live_before.shape) == (
            int(args_cli.num_envs),
            int(args_cli.history_steps) + 1,
            93,
        )

        # Moving only the reference cursor must not affect the robot-only input.
        tm = base_env.trajectory_manager
        original_steps = tm.env_step.clone()
        next_steps = original_steps + 1
        tm._set_env_steps(torch.arange(int(args_cli.num_envs)), next_steps)
        base_env._refresh_current_expert_frame(advance=False)
        live_after = base_env.current_causal_planner_observation(
            history_steps=int(args_cli.history_steps)
        ).get(("planner", "state_history"))
        torch.testing.assert_close(live_before, live_after)
        tm._set_env_steps(torch.arange(int(args_cli.num_envs)), original_steps)
        base_env._refresh_current_expert_frame(advance=False)

        training_batch = base_env.sample_causal_planner_training_batch(
            batch_size=int(args_cli.num_envs),
            horizon_steps=int(args_cli.horizon_steps),
            split="all",
            history_steps=int(args_cli.history_steps),
        )
        offline_history = training_batch.get(("planner", "state_history"))
        assert tuple(offline_history.shape) == tuple(live_before.shape)
        assert int(spec["flat_dim"]) == int(offline_history[0].numel())

        zeros = torch.zeros(env.action_space.shape, device=base_env.device)
        env.step(zeros)
        env.reset()
        reset_history = base_env.current_causal_planner_observation(
            history_steps=int(args_cli.history_steps)
        ).get(("planner", "state_history"))
        torch.testing.assert_close(
            reset_history,
            reset_history[:, -1:].expand_as(reset_history),
        )
        print(
            "[INFO] Causal planner smoke passed: "
            f"history={tuple(live_before.shape)}, flat_dim={spec['flat_dim']}."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
