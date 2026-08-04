# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark steady-state G1 imitation env step, reward, and observation time."""

# ruff: noqa: E402

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Benchmark G1 imitation MDP performance.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=10, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
parser.add_argument(
    "--warmup_steps", type=int, default=100, help="Warmup steps before timing."
)
parser.add_argument("--steps", type=int, default=200, help="Timed steps.")
parser.add_argument(
    "--output", type=str, default=None, help="Optional path to write benchmark JSON."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _make_sync(device: torch.device) -> Callable[[], None]:
    if device.type == "cuda" and torch.cuda.is_available():
        return lambda: torch.cuda.synchronize(device)
    return lambda: None


class _Timer:
    def __init__(self, sync: Callable[[], None]) -> None:
        self._sync = sync
        self.active = False
        self.values_ms: list[float] = []

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self.active:
                return fn(*args, **kwargs)
            self._sync()
            start = time.perf_counter()
            value = fn(*args, **kwargs)
            self._sync()
            self.values_ms.append((time.perf_counter() - start) * 1000.0)
            return value

        return _wrapped


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = torch.device(str(unwrapped.device))
    sync = _make_sync(device)

    reward_timer = _Timer(sync)
    obs_timer = _Timer(sync)
    unwrapped.reward_manager.compute = reward_timer.wrap(
        unwrapped.reward_manager.compute
    )
    unwrapped.observation_manager.compute = obs_timer.wrap(
        unwrapped.observation_manager.compute
    )

    env.reset(seed=args_cli.seed)
    actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

    with torch.inference_mode():
        for _ in range(args_cli.warmup_steps):
            env.step(actions)

        sync()
        reward_timer.active = True
        obs_timer.active = True
        step_values_ms: list[float] = []

        for _ in range(args_cli.steps):
            sync()
            start = time.perf_counter()
            env.step(actions)
            sync()
            step_values_ms.append((time.perf_counter() - start) * 1000.0)

    metrics = {
        "task": args_cli.task,
        "num_envs": int(args_cli.num_envs),
        "warmup_steps": int(args_cli.warmup_steps),
        "steps": int(args_cli.steps),
        "step_ms": _mean(step_values_ms),
        "reward_ms": _mean(reward_timer.values_ms),
        "observation_ms": _mean(obs_timer.values_ms),
    }
    if args_cli.output:
        output_path = Path(args_cli.output)
        output_path.write_text(json.dumps(metrics, sort_keys=True) + "\n")
    print("BENCHMARK_RESULT " + json.dumps(metrics, sort_keys=True), flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
