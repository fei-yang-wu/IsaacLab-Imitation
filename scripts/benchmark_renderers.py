#!/usr/bin/env python3
# ruff: noqa: E402
"""Benchmark Isaac Lab 3.0 camera renderers on the G1 scene.

The G1 imitation tasks are state-based (no cameras), so the training
benchmark (`scripts/benchmark_physics_backends.py`) is renderer-independent.
This script measures the camera-renderer axis in isolation: it spawns the G1
robot scene with a tiled RGB camera per environment, steps the simulation,
and reports steps/sec and rendered frames/sec for each renderer backend.

Renderers:

- ``none``      — no camera in the scene (baseline: pure physics stepping)
- ``rtx``       — Isaac Sim RTX tiled renderer (``isaaclab_physx``)
- ``newton_warp`` — Newton Warp raytracer (``isaaclab_newton``)

Run from the repo root through Pixi:

.. code-block:: bash

    pixi run -e isaaclab bench-renderers
    # or custom:
    pixi run -e isaaclab python scripts/benchmark_renderers.py \
        --renderers none rtx newton_warp --num_envs 256 --steps 200

Each renderer runs in its own subprocess (one Isaac Sim app per run); a JSON
summary is written under ``logs/benchmarks/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_CHOICES = ("none", "rtx", "newton_warp")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--renderers",
    nargs="+",
    default=list(RENDERER_CHOICES),
    choices=RENDERER_CHOICES,
)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="JSON summary path (default: logs/benchmarks/renderers_<ts>.json).",
)
parser.add_argument(
    "--worker_renderer",
    type=str,
    default=None,
    help=argparse.SUPPRESS,  # internal: run as measurement worker
)


# ---------------------------------------------------------------------------
# Worker: one renderer measurement inside a live Isaac Sim app.
# ---------------------------------------------------------------------------


def run_worker(args: argparse.Namespace) -> None:
    renderer = args.worker_renderer

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args([])
    app_args.headless = True
    # RTX tiled rendering requires camera support in the Kit app.
    app_args.enable_cameras = renderer != "none"
    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    import torch  # noqa: F401
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.configclass import configclass

    from isaaclab_imitation.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG

    renderer_cfg = None
    if renderer == "rtx":
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        renderer_cfg = IsaacRtxRendererCfg()
    elif renderer == "newton_warp":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        renderer_cfg = NewtonWarpRendererCfg()

    @configclass
    class RendererBenchSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=750.0),
        )
        robot: ArticulationCfg = UNITREE_G1_29DOF_MIMIC_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

    scene_cfg = RendererBenchSceneCfg(num_envs=args.num_envs, env_spacing=2.5)
    if renderer_cfg is not None:
        scene_cfg.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(2.0, 2.0, 1.2),
                # Scalar-last (x, y, z, w) per Isaac Lab 3.0 convention.
                rot=(0.2706, 0.6533, 0.6533, 0.2706),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 20.0)),
            width=args.width,
            height=args.height,
            renderer_cfg=renderer_cfg,
        )

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    camera = scene.sensors.get("camera") if renderer_cfg is not None else None

    def step_once() -> None:
        sim.step(render=renderer == "rtx")
        scene.update(sim.get_physics_dt())
        if camera is not None:
            _ = camera.data.output["rgb"]

    for _ in range(args.warmup_steps):
        step_once()

    start = time.monotonic()
    for _ in range(args.steps):
        step_once()
    elapsed = time.monotonic() - start

    steps_per_s = args.steps / elapsed
    result = {
        "renderer": renderer,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "resolution": [args.width, args.height],
        "elapsed_s": round(elapsed, 3),
        "steps_per_s": round(steps_per_s, 2),
        "env_steps_per_s": round(steps_per_s * args.num_envs, 1),
        "rendered_frames_per_s": (
            round(steps_per_s * args.num_envs, 1) if camera is not None else None
        ),
    }
    print(f"RESULT_JSON: {json.dumps(result)}", flush=True)
    simulation_app.close()


# ---------------------------------------------------------------------------
# Driver: run each renderer in its own subprocess and summarize.
# ---------------------------------------------------------------------------


def run_driver(args: argparse.Namespace) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "logs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (out_dir / f"renderers_{stamp}.json")

    results = []
    for renderer in args.renderers:
        log_path = out_dir / f"renderer_{renderer}_{stamp}.log"
        cmd = [
            sys.executable,
            __file__,
            "--worker_renderer",
            renderer,
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--warmup_steps",
            str(args.warmup_steps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ]
        print(f"[bench] renderer={renderer} -> {log_path}", flush=True)
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
        with log_path.open("w") as log_file:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, env=env
            )
        text = log_path.read_text(errors="replace")
        match = re.findall(r"RESULT_JSON: (\{.*\})", text)
        result = json.loads(match[-1]) if match else {"renderer": renderer}
        result["exit_code"] = proc.returncode
        result["log"] = str(log_path)
        print(f"[bench] renderer={renderer} -> {result}", flush=True)
        results.append(result)

    summary = {
        "timestamp_utc": stamp,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "resolution": [args.width, args.height],
        "results": results,
    }
    output.write_text(json.dumps(summary, indent=2))

    print("\n=== renderer benchmark summary ===")
    print(f"{'renderer':<14}{'exit':<6}{'steps/s':<10}{'frames/s':<12}")
    for r in results:
        print(
            f"{r['renderer']:<14}{r['exit_code']:<6}"
            f"{r.get('steps_per_s', '-'):<10}{r.get('rendered_frames_per_s') or '-':<12}"
        )
    print(f"Summary written to {output}")
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


def main() -> int:
    args = parser.parse_args()
    if args.worker_renderer is not None:
        run_worker(args)
        return 0
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
