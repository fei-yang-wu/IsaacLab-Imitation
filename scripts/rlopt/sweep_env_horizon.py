#!/usr/bin/env python3
"""Sweep (num_envs x per-env horizon) for the low-level IPMD policy: peak VRAM + throughput.

Runs a quick skill-encoder pretrain once (to build the Zarr cache + a checkpoint),
then for each (num_envs, horizon) config runs the low-level train.py for a fixed
number of iterations while polling ``nvidia-smi`` for peak GPU memory. Parses the
training stdout for steady-state throughput (frames/s). Writes a summary JSON and
prints a table so the best-fitting config for the target GPU can be picked.

train.py auto-rescales replay_buffer.size / mini_batch_size to num_envs x horizon
(see the on-policy block in train.py), so each config is a valid single-knob change.

Example (inside the ICE Singularity job, /data is mounted):
    /isaac-sim/python.sh scripts/rlopt/sweep_env_horizon.py \
        --data-root /data/bones_seed_100 --iters 120 \
        --configs 4096x24,4096x48,8192x32
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_ITER_RE = re.compile(r"iter=(\d+)/\d+\s*\|\s*frames=(\d+)/")
_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})")


def _parse_configs(spec: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        n, h = token.lower().split("x")
        configs.append((int(n), int(h)))
    return configs


class VramPoller:
    """Background nvidia-smi poller capturing peak used memory (MiB)."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.peak_mib = 0
        self.available = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll_once(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            )
        except Exception:
            self.available = False
            return None
        vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return max(vals) if vals else None

    def _run(self) -> None:
        while not self._stop.is_set():
            v = self._poll_once()
            if v is None:
                return
            self.peak_mib = max(self.peak_mib, v)
            self._stop.wait(self.interval)

    def __enter__(self) -> "VramPoller":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _throughput_from_log(text: str, warmup_frac: float = 0.4) -> tuple[float, int]:
    """Steady-state frames/s from iter log lines (skip a warmup fraction)."""
    samples: list[tuple[float, int]] = []
    for line in text.splitlines():
        m = _ITER_RE.search(line)
        t = _TS_RE.search(line)
        if not m or not t:
            continue
        hh, mm, ss = (int(g) for g in t.groups())
        wall = hh * 3600 + mm * 60 + ss
        samples.append((wall, int(m.group(2))))
    if len(samples) < 4:
        return 0.0, len(samples)
    start = int(len(samples) * warmup_frac)
    seg = samples[start:]
    # handle midnight wrap defensively by using frame deltas / time deltas
    dt = seg[-1][0] - seg[0][0]
    df = seg[-1][1] - seg[0][1]
    if dt <= 0:
        dt += 24 * 3600
    fps = df / dt if dt > 0 else 0.0
    return float(fps), len(samples)


def _run_capture(cmd: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    with subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc, log_path.open("w", encoding="utf-8") as fh:
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            chunks.append(line)
        proc.wait()
        return int(proc.returncode), "".join(chunks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/data/bones_seed_100")
    p.add_argument("--task", default="Isaac-Imitation-G1-Latent-v0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--configs", default="4096x24,4096x48,8192x32")
    p.add_argument("--skill-ckpt", default=None, help="Reuse an existing best.pt; else pretrain once.")
    p.add_argument("--pretrain-updates", type=int, default=200)
    p.add_argument("--run-root", default=None)
    args = p.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    manifest = data_root / "manifests" / "g1_bones_seed_100_manifest.json"
    dataset_path = data_root / "g1_hl_diffsr"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root or (REPO_ROOT / "logs" / "sweep_env_horizon" / ts)).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    configs = _parse_configs(args.configs)
    print(f"[INFO] sweep configs: {configs}  iters={args.iters}  run_root={run_root}", flush=True)

    # Stage 0: pretrain once (also builds the Zarr cache with refresh=true).
    skill_ckpt = Path(args.skill_ckpt).resolve() if args.skill_ckpt else (run_root / "skill_encoder" / "checkpoints" / "best.pt")
    if not args.skill_ckpt:
        rc, _ = _run_capture(
            [
                sys.executable, "scripts/rlopt/train_hl_skill_diffsr.py",
                "--headless", "--device", args.device, "--task", args.task,
                "--num_envs", "4096", "--seed", str(args.seed),
                "--output_dir", str(run_root / "skill_encoder"),
                "--horizon_steps", "25", "--encoder_window_mode", "intermediate",
                "--z_dim", "256", "--diffsr_feature_dim", "128", "--diffsr_embed_dim", "512",
                "--batch_size", "8192", "--num_updates", str(args.pretrain_updates),
                "--log_interval", "100", "--train_split", "all", "--eval_split", "all",
                "--eval_trajectory_fraction", "0.5", "--trajectory_split_seed", str(args.seed),
                f"env.lafan1_manifest_path={manifest}", f"env.dataset_path={dataset_path}",
                "env.refresh_zarr_dataset=true",
            ],
            run_root / "pretrain.log",
        )
        if rc != 0 or not skill_ckpt.is_file():
            raise SystemExit(f"[ERROR] pretrain failed (rc={rc}) or no checkpoint at {skill_ckpt}")

    results = []
    for num_envs, horizon in configs:
        tag = f"{num_envs}x{horizon}"
        log_path = run_root / f"lowlevel_{tag}.log"
        print(f"[INFO] === config {tag}: num_envs={num_envs} horizon={horizon} batch={num_envs*horizon} ===", flush=True)
        cmd = [
            sys.executable, "scripts/rlopt/train.py",
            "--headless", "--device", args.device, "--num_envs", str(num_envs),
            "--task", args.task, "--algo", "IPMD", "--seed", str(args.seed),
            "--max_iterations", str(args.iters),
            f"agent.collector.frames_per_batch={horizon}",
            "agent.logger.backend=none", "agent.logger.video=false",
            "agent.save_interval=100000000000",
            f"agent.ipmd.hl_skill_checkpoint_path={skill_ckpt}",
            f"env.lafan1_manifest_path={manifest}", f"env.dataset_path={dataset_path}",
            "env.refresh_zarr_dataset=false",
        ]
        t0 = time.time()
        with VramPoller() as poller:
            rc, out = _run_capture(cmd, log_path)
        elapsed = time.time() - t0
        fps, n_iter_lines = _throughput_from_log(out)
        oom = ("out of memory" in out.lower()) or ("CUDA error" in out)
        results.append({
            "config": tag, "num_envs": num_envs, "horizon": horizon,
            "batch": num_envs * horizon, "returncode": rc, "fit": rc == 0 and not oom,
            "peak_vram_mib": poller.peak_mib, "vram_available": poller.available,
            "frames_per_s": round(fps, 1), "iter_lines": n_iter_lines,
            "wallclock_s": round(elapsed, 1),
        })
        print(f"[INFO] {tag}: fit={results[-1]['fit']} peak_vram={poller.peak_mib}MiB fps={fps:.0f} rc={rc}", flush=True)

    summary_path = run_root / "sweep_summary.json"
    summary_path.write_text(json.dumps({"configs": args.configs, "iters": args.iters, "results": results}, indent=2) + "\n")
    print("\n=== SWEEP SUMMARY ===")
    print(f"{'config':>12} {'batch':>9} {'fit':>5} {'peak_vram_MiB':>14} {'peak_GB':>8} {'frames/s':>10}")
    for r in results:
        print(f"{r['config']:>12} {r['batch']:>9} {str(r['fit']):>5} {r['peak_vram_mib']:>14} {r['peak_vram_mib']/1024:>8.1f} {r['frames_per_s']:>10.0f}")
    print(f"\n[INFO] wrote {summary_path}")


if __name__ == "__main__":
    main()
