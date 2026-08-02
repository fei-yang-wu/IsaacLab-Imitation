#!/usr/bin/env python3
"""Fixed-seed A/B equivalence certificate: legacy vs v2 env classes.

Both env classes accept the same ``-G1-v2`` cfg, so the fork's switch-over
condition is a fixed-seed A/B: run the identical cfg and action sequence once
with the legacy class (``ImitationRLEnvLegacy``) and once with the flagship
``ImitationRLEnv`` (one class per process), then ``torch.equal``-compare the
full observation dict, rewards, dones, and the command-manager commands at
every step.

ROLE CHANGE (2026-08-01, lean v2): the v2 env now runs a single-compute step
while the legacy env keeps the base env's double observation compute; the
discarded mid-step compute also drew observation noise, so v2 has its own
fresh stochastic stream and is deliberately NOT bit-equivalent to the legacy
env. This tool therefore certifies:

- legacy == legacy (cross-process determinism of the frozen v0/v1 env), and
- v2 == v2 (determinism of the thin env) for a fixed cfg/seed/action
  sequence.

Capturing the same class twice and comparing is the regression gate; a
legacy-vs-v2 comparison is expected to diverge after the first noise draw.
"""

Examples (run from the repository root):

.. code-block:: bash

    pixi run -e isaaclab python scripts/audit/certify_v2_env_equivalence.py \
        capture --class legacy --output logs/v2_cert/legacy.pt \
        --manifest ./data/unitree/manifests/g1_unitree_dance102_manifest.json

    pixi run -e isaaclab python scripts/audit/certify_v2_env_equivalence.py \
        capture --class v2 --output logs/v2_cert/v2.pt \
        --manifest ./data/unitree/manifests/g1_unitree_dance102_manifest.json

    python scripts/audit/certify_v2_env_equivalence.py compare \
        logs/v2_cert/legacy.pt logs/v2_cert/v2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_TASK = "Isaac-Imitation-G1-v2"
DEFAULT_MANIFEST = Path("./data/unitree/manifests/g1_unitree_dance102_manifest.json")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    capture = subparsers.add_parser(
        "capture", help="Run one env class and save the rollout."
    )
    capture.add_argument(
        "--class", dest="class_name", choices=("legacy", "v2"), required=True
    )
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--task", default=DEFAULT_TASK)
    capture.add_argument("--num_envs", type=int, default=4)
    capture.add_argument("--steps", type=int, default=12)
    capture.add_argument("--seed", type=int, default=42)
    capture.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    capture.add_argument(
        "--env-overrides",
        default="",
        help="Comma-separated env.* overrides applied to the cfg (e.g. "
        "env.reset_start_mode=adaptive,env.random_reset_step_min=0).",
    )
    capture.add_argument("--headless", action="store_true", default=True)
    capture.set_defaults(headless=True)

    compare = subparsers.add_parser(
        "compare", help="torch.equal-compare two captures from the two classes."
    )
    compare.add_argument("legacy_capture", type=Path)
    compare.add_argument("v2_capture", type=Path)
    return parser.parse_args(argv)


def _split_override_tokens(raw: str) -> list[str]:
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


def _apply_env_override(env_cfg: object, key: str, value: str) -> object:
    """Apply a dotted env.* override with YAML coercion (plain-setattr path)."""
    import yaml

    if key.startswith("env."):
        key = key[len("env.") :]
    parts = key.split(".")
    target = env_cfg
    for part in parts[:-1]:
        target = getattr(target, part)
    try:
        coerced = yaml.safe_load(value)
    except yaml.YAMLError:
        # List-form strings like "[a,b,c]" (Hydra convention) are kept raw;
        # the cfg's prune/validation methods parse both forms themselves.
        coerced = value
    setattr(target, parts[-1], coerced)
    return env_cfg


def _capture(args: argparse.Namespace) -> None:
    """Run one env class on the fixed protocol and save the per-step capture."""
    import numpy as np

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True, "device": "cuda:0"})
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_imitation.tasks  # noqa: F401
    from isaaclab_imitation.envs import ImitationRLEnv, ImitationRLEnvLegacy

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env_cfg = parse_env_cfg(
        args.task,
        device="cuda:0",
        num_envs=args.num_envs,
        use_fabric=True,
    )
    setattr(env_cfg, "lafan1_manifest_path", str(args.manifest.resolve()))
    for override in _split_override_tokens(args.env_overrides):
        key, value = override.split("=", 1)
        env_cfg = _apply_env_override(env_cfg, key, value)

    env_cls = ImitationRLEnvLegacy if args.class_name == "legacy" else ImitationRLEnv
    env = env_cls(env_cfg)

    action_gen = torch.Generator(device=env.device).manual_seed(args.seed)
    action_shape = tuple(env.action_space.shape)

    def snapshot(step_index: int) -> dict:
        obs = _to_cpu_dict(env.obs_buf)
        commands = {}
        command_manager = getattr(env, "command_manager", None)
        active_terms = tuple(getattr(command_manager, "active_terms", ()))
        for term_name in ("command", "motion", "skill", "chunk"):
            if term_name in active_terms:
                commands[term_name] = (
                    command_manager.get_command(term_name).detach().cpu().clone()
                )
        log = {}
        for key, value in env.extras.get("log", {}).items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu().clone()
            else:
                log[key] = value
        record = {
            "step": step_index,
            "obs": obs,
            "commands": commands,
            "log": log,
        }
        for attr in ("reward_buf", "reset_terminated", "reset_time_outs"):
            value = getattr(env, attr, None)
            if value is not None:
                record[attr] = value.detach().cpu().clone()
        return record

    records: list[dict] = []
    obs, _ = env.reset()
    env.obs_buf = obs
    records.append(snapshot(0))
    for step_index in range(1, args.steps + 1):
        action = torch.randn(*action_shape, device=env.device, generator=action_gen)
        env.step(action)
        records.append(snapshot(step_index))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "class_name": args.class_name,
            "task": args.task,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "records": records,
        },
        args.output,
    )
    print(f"[INFO] captured {args.class_name} rollout -> {args.output}")

    # NOTE: simulation_app.close() below terminates the process on some
    # headless Isaac Sim builds, so the capture is persisted first.
    env.close()
    simulation_app.close()


def _to_cpu_dict(value: object) -> dict | torch.Tensor:
    """Recursively clone an observation structure to CPU tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(key): _to_cpu_dict(item) for key, item in value.items()}
    if hasattr(value, "keys") and hasattr(value, "get"):
        return {
            str(key): _to_cpu_dict(value.get(key))
            for key in value.keys()
            if value.get(key) is not None
        }
    return value


def _compare_entry(label: str, legacy: object, v2: object) -> list[str]:
    failures: list[str] = []
    if isinstance(legacy, dict) and isinstance(v2, dict):
        legacy_keys = set(legacy)
        v2_keys = set(v2)
        if legacy_keys != v2_keys:
            failures.append(
                f"{label}: key sets differ (legacy-only: {sorted(legacy_keys - v2_keys)}, "
                f"v2-only: {sorted(v2_keys - legacy_keys)})"
            )
            return failures
        for key in sorted(legacy_keys):
            failures.extend(_compare_entry(f"{label}.{key}", legacy[key], v2[key]))
        return failures
    if isinstance(legacy, torch.Tensor) and isinstance(v2, torch.Tensor):
        if legacy.shape != v2.shape:
            failures.append(
                f"{label}: shape {tuple(legacy.shape)} vs {tuple(v2.shape)}"
            )
        elif not torch.equal(legacy, v2):
            failures.append(f"{label}: values differ")
        return failures
    if legacy != v2:
        failures.append(f"{label}: {legacy!r} vs {v2!r}")
    return failures


def _compare(args: argparse.Namespace) -> None:
    legacy_data = torch.load(
        args.legacy_capture, map_location="cpu", weights_only=False
    )
    v2_data = torch.load(args.v2_capture, map_location="cpu", weights_only=False)
    if legacy_data["seed"] != v2_data["seed"]:
        raise SystemExit(
            f"captures use different seeds: {legacy_data['seed']} vs {v2_data['seed']}"
        )
    if legacy_data["num_envs"] != v2_data["num_envs"]:
        raise SystemExit("captures use different num_envs.")
    legacy_records = legacy_data["records"]
    v2_records = v2_data["records"]
    if len(legacy_records) != len(v2_records):
        raise SystemExit("captures have different step counts.")

    failures: list[str] = []
    for legacy_rec, v2_rec in zip(legacy_records, v2_records):
        step = legacy_rec["step"]
        for key in (
            "obs",
            "reward_buf",
            "reset_terminated",
            "reset_time_outs",
            "commands",
            "log",
        ):
            if key not in legacy_rec and key not in v2_rec:
                continue
            failures.extend(
                _compare_entry(f"step {step} {key}", legacy_rec[key], v2_rec[key])
            )
    if failures:
        print(f"[FAIL] {len(failures)} mismatched entries")
        for failure in failures[:100]:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(
        f"[PASS] legacy({legacy_data['class_name']}) == v2 for "
        f"{len(legacy_records)} steps x {legacy_data['num_envs']} envs "
        "(obs dict, rewards, dones, commands, log) on seed "
        f"{legacy_data['seed']}."
    )


def main() -> None:
    args = _parse_args(sys.argv[1:])
    if args.mode == "capture":
        _capture(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
