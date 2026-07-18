#!/usr/bin/env python3
"""Benchmark RLOpt training throughput across Isaac Lab physics backends.

Runs the same training job once per backend (sequential subprocesses, one
Isaac Sim app each) and reports wall-clock training time, effective FPS, and
the final logged step reward. Intended for routine performance regression
checks after dependency bumps.

Run from the repo root through Pixi:

.. code-block:: bash

    # Quick sanity sweep (a few iterations per backend)
    pixi run -e isaaclab bench-backends-quick

    # Full comparison (~10M frames per backend)
    pixi run -e isaaclab bench-backends

    # Custom
    pixi run -e isaaclab python scripts/benchmark_physics_backends.py \
        --backends physx newton_mjwarp --num_envs 4096 --iterations 102
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "./data/unitree/manifests/g1_unitree_dance102_manifest.json"
# RLOpt PPO collects num_envs * FRAMES_PER_BATCH frames per iteration.
FRAMES_PER_BATCH = 24

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--backends",
    nargs="+",
    default=["physx", "newton_mjwarp"],
    help="Physics preset names understood by the task cfg (physics=<name>).",
)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v0")
parser.add_argument("--algo", type=str, default="PPO")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument(
    "--iterations",
    type=int,
    default=102,
    help="Training iterations per backend (102 x 4096 envs x 24 ~= 10M frames).",
)
parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="JSON summary path (default: logs/benchmarks/physics_backends_<ts>.json).",
)
parser.add_argument(
    "--extra",
    nargs=argparse.REMAINDER,
    default=[],
    help="Extra args forwarded verbatim to train.py (after --extra).",
)


def _last_float(pattern: str, text: str) -> float | None:
    """Return the last regex capture as float, tolerating rich line wrapping."""
    collapsed = re.sub(r"\s+", " ", text)
    hits = re.findall(pattern, collapsed)
    if not hits:
        return None
    try:
        return float(hits[-1])
    except ValueError:
        return None


def run_backend(args: argparse.Namespace, backend: str, log_path: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/rlopt/train.py",
        "--task",
        args.task,
        "--algo",
        args.algo,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.iterations),
        "--headless",
        "--kit_args=--/app/extensions/fsWatcherEnabled=false",
        f"env.lafan1_manifest_path={args.manifest}",
        "agent.logger.backend=",
        f"physics={backend}",
        *args.extra,
    ]
    frames = args.num_envs * args.iterations * FRAMES_PER_BATCH
    print(f"[bench] backend={backend} frames={frames:,} -> {log_path}", flush=True)
    start = time.monotonic()
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT
        )
    wall_s = time.monotonic() - start
    text = log_path.read_text(errors="replace")

    training_time = _last_float(r"Training time: ([0-9.]+) seconds", text)
    result = {
        "backend": backend,
        "exit_code": proc.returncode,
        "frames": frames,
        "wall_time_s": round(wall_s, 1),
        "training_time_s": training_time,
        "effective_fps": (round(frames / training_time, 1) if training_time else None),
        "reported_fps_last": _last_float(r"fps=([0-9.]+)", text),
        "r_step_last": _last_float(r"r_step=(-?[0-9.]+)", text),
        "ep_len_last": _last_float(r"ep_len=(-?[0-9.]+)", text),
        "log": str(log_path),
    }
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[bench] backend={backend} {status} | {result}", flush=True)
    return result


def main() -> int:
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "logs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (out_dir / f"physics_backends_{stamp}.json")

    results = [
        run_backend(args, backend, out_dir / f"train_{backend}_{stamp}.log")
        for backend in args.backends
    ]

    summary = {
        "timestamp_utc": stamp,
        "task": args.task,
        "algo": args.algo,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "frames_per_batch": FRAMES_PER_BATCH,
        "manifest": args.manifest,
        "results": results,
    }
    output.write_text(json.dumps(summary, indent=2))

    print("\n=== physics backend benchmark summary ===")
    header = f"{'backend':<16}{'exit':<6}{'train_s':<10}{'eff_fps':<12}{'r_step':<10}{'ep_len':<8}"
    print(header)
    for r in results:
        print(
            f"{r['backend']:<16}{r['exit_code']:<6}"
            f"{r['training_time_s'] or '-':<10}{r['effective_fps'] or '-':<12}"
            f"{r['r_step_last'] if r['r_step_last'] is not None else '-':<10}"
            f"{r['ep_len_last'] if r['ep_len_last'] is not None else '-':<8}"
        )
    print(f"Summary written to {output}")

    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
