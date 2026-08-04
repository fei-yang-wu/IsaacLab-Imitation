#!/usr/bin/env python3
"""Profile the G1 imitation env step: per-component ms breakdown.

Wraps every step() sub-component (pre-step reference refresh, action
process, physics write/sim/update, terminations, rewards, commands, events,
observation compute, resets, post-step refresh, planner history) plus the
known-heavy expert-window builders, and prints mean ms + % of step time.

Example (run from the repository root):

.. code-block:: bash

    pixi run -e isaaclab python scripts/bench/profile_g1_env_step.py \
        --task Isaac-Imitation-G1-v2 --num_envs 256 --steps 200 \
        --manifest ./data/unitree/manifests/g1_unitree_dance102_manifest.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v2")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("./data/unitree/manifests/g1_unitree_dance102_manifest.json"),
    )
    parser.add_argument(
        "--env-overrides",
        default="",
        help="Comma-separated env.* overrides (plain-setattr path).",
    )
    parser.add_argument("--output", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args(argv)
    return args


class _Timer:
    def __init__(self, sync: Callable[[], None], name: str) -> None:
        self._sync = sync
        self.name = name
        self.active = False
        self.values_ms: list[float] = []

    def wrap(self, fn: Callable[..., object]) -> Callable[..., object]:
        def _wrapped(*args: object, **kwargs: object) -> object:
            if not self.active:
                return fn(*args, **kwargs)
            self._sync()
            start = time.perf_counter()
            value = fn(*args, **kwargs)
            self._sync()
            self.values_ms.append((time.perf_counter() - start) * 1000.0)
            return value

        _wrapped._timer = self  # type: ignore[attr-defined]
        return _wrapped


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = _parse_args(None)
    app_launcher = AppLauncher(
        {
            k: v
            for k, v in vars(args).items()
            if k in ("headless", "device", "enable_cameras")
        }
        | {"headless": True}
    )
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_imitation.tasks  # noqa: F401

    env_cfg = parse_env_cfg(
        args.task, device="cuda:0", num_envs=args.num_envs, use_fabric=True
    )
    env_cfg.data.manifest = str(args.manifest.resolve())
    for override in _split_overrides(args.env_overrides):
        key, value = override.split("=", 1)
        if key.startswith("env."):
            key = key[len("env.") :]
        parts = key.split(".")
        target = env_cfg
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], _coerce(value))
    if args.env_overrides.strip():
        # Plain-setattr overrides land after __post_init__; re-run it so
        # pruning/validation knobs (e.g. expert_window_observation_terms,
        # enable_expert_goal_observations) take effect. Idempotent for the
        # default values.
        env_cfg.__post_init__()

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = torch.device(str(unwrapped.device))
    sync = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda" and torch.cuda.is_available()
        else (lambda: None)
    )

    timers: dict[str, _Timer] = {}

    def add(name: str, fn: Callable[..., object]) -> None:
        timer = _Timer(sync, name)
        timers[name] = timer
        return timer.wrap(fn)

    # Env-level step hooks.
    unwrapped._refresh_current_expert_frame = add(  # type: ignore[method-assign]
        "refresh_frame", unwrapped._refresh_current_expert_frame
    )
    unwrapped._append_causal_planner_history = add(  # type: ignore[method-assign]
        "planner_history", unwrapped._append_causal_planner_history
    )
    unwrapped._compute_rollout_reference_state_log = add(  # type: ignore[method-assign]
        "rollout_state_log", unwrapped._compute_rollout_reference_state_log
    )
    unwrapped._reset_idx = add("resets", unwrapped._reset_idx)  # type: ignore[method-assign]

    # Manager components inside the base step.
    unwrapped.action_manager.process_action = add(
        "action_process", unwrapped.action_manager.process_action
    )
    unwrapped.scene.write_data_to_sim = add(
        "scene_write_to_sim", unwrapped.scene.write_data_to_sim
    )
    unwrapped.sim.step = add("physics_sim_step", unwrapped.sim.step)
    unwrapped.scene.update = add("scene_update", unwrapped.scene.update)
    unwrapped.termination_manager.compute = add(
        "terminations", unwrapped.termination_manager.compute
    )
    unwrapped.reward_manager.compute = add("rewards", unwrapped.reward_manager.compute)
    unwrapped.command_manager.compute = add(
        "commands", unwrapped.command_manager.compute
    )
    unwrapped.event_manager.apply = add("events", unwrapped.event_manager.apply)
    unwrapped.observation_manager.compute = add(
        "observations", unwrapped.observation_manager.compute
    )

    # Expert-window builders (the known-heavy obs-side suspects). Legacy envs
    # keep the builders inline; only the composed v2 env has the plane.
    plane = getattr(unwrapped, "expert_data_plane", None)
    if plane is not None:
        plane._ensure_mdp_step_cache = add(  # type: ignore[method-assign]
            "mdp_step_cache", plane._ensure_mdp_step_cache
        )
        plane._get_current_expert_window_terms = add(  # type: ignore[method-assign]
            "expert_window_build", plane._get_current_expert_window_terms
        )
        plane._get_current_expert_goal_terms = add(  # type: ignore[method-assign]
            "expert_goal_build", plane._get_current_expert_goal_terms
        )

    # Per-observation-group timing inside the observation manager.
    group_timers: dict[str, _Timer] = {}
    compute_group = unwrapped.observation_manager.compute_group

    def _wrapped_compute_group(group_name: str, update_history: bool = False):
        timer = group_timers.get(group_name)
        if timer is None or not timer.active:
            return compute_group(group_name, update_history=update_history)
        timer._sync()
        start = time.perf_counter()
        value = compute_group(group_name, update_history=update_history)
        timer._sync()
        timer.values_ms.append((time.perf_counter() - start) * 1000.0)
        return value

    unwrapped.observation_manager.compute_group = _wrapped_compute_group  # type: ignore[method-assign]
    for group_name in unwrapped.observation_manager.active_terms:
        group_timers[group_name] = _Timer(sync, f"obs_group:{group_name}")

    env.reset(seed=args.seed)
    actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

    with torch.inference_mode():
        for _ in range(args.warmup_steps):
            env.step(actions)
        sync()
        for timer in timers.values():
            timer.active = True
        for timer in group_timers.values():
            timer.active = True
        step_values_ms: list[float] = []
        for _ in range(args.steps):
            sync()
            start = time.perf_counter()
            env.step(actions)
            sync()
            step_values_ms.append((time.perf_counter() - start) * 1000.0)

    step_ms = _mean(step_values_ms)
    rows = []
    for name, timer in timers.items():
        mean_ms = _mean(timer.values_ms)
        rows.append(
            {
                "component": name,
                "mean_ms": round(mean_ms, 3),
                "pct_of_step": round(100.0 * mean_ms / step_ms, 2) if step_ms else 0.0,
                "calls": len(timer.values_ms),
            }
        )
    rows.sort(key=lambda row: row["mean_ms"], reverse=True)
    group_rows = []
    for name, timer in group_timers.items():
        mean_ms = _mean(timer.values_ms)
        group_rows.append(
            {
                "component": name,
                "mean_ms": round(mean_ms, 3),
                "pct_of_obs": round(
                    100.0 * mean_ms / _mean(timers["observations"].values_ms), 2
                )
                if timers["observations"].values_ms
                else 0.0,
                "pct_of_step": round(100.0 * mean_ms / step_ms, 2) if step_ms else 0.0,
                "calls": len(timer.values_ms),
            }
        )
    group_rows.sort(key=lambda row: row["mean_ms"], reverse=True)
    result = {
        "task": args.task,
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "step_ms": round(step_ms, 3),
        "components": rows,
        "obs_groups": group_rows,
    }
    if args.output:
        args.output.write_text(json.dumps(result, sort_keys=True) + "\n")
    print("PROFILE_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


def _coerce(value: str):
    import yaml

    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        # List-form strings like "[a,b,c]" (Hydra convention) are kept raw;
        # the cfg's prune/validation methods parse both forms themselves.
        return value


def _split_overrides(raw: str) -> list[str]:
    """Split comma-separated overrides, keeping bracketed list values intact."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in raw:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        parts.append(token)
    return parts


if __name__ == "__main__":
    main()
