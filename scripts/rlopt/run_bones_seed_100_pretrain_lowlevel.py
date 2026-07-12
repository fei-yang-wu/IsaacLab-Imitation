#!/usr/bin/env python3
"""Cluster entry: pretrain DiffSR skill encoder + low-level oracle IPMD on BONES-SEED-100.

Mirrors the validated local launcher (scripts/rlopt/run_local_pretrain_lowlevel.sh)
as a single Python entry the cluster interface can invoke via
``pixi run -e isaaclab python scripts/rlopt/run_bones_seed_100_pretrain_lowlevel.py <args>``.

Stage 1 builds the Zarr cache from scratch (refresh_zarr_dataset=true) and pretrains
the skill encoder; Stage 2 trains the low-level IPMD policy conditioned on that encoder
(command_source=hl_skill; all hl_skill/latent params are baked config defaults).

Expects the dataset already staged at ``--data-root`` (default /data/bones_seed_100):
    <data-root>/manifests/g1_bones_seed_100_manifest.json
    <data-root>/npz/g1/*.npz
The cluster submit script remaps a leading ``/data`` to the node-local data dir.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/data/bones_seed_100")
    p.add_argument("--run-root", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--task", default="Isaac-Imitation-G1-Latent-v0")
    p.add_argument("--algorithm", default="IPMD")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-cap", type=int, default=2_000_000_000)
    p.add_argument("--train-num-envs", type=int, default=4096)
    p.add_argument("--horizon-steps", type=int, default=25)
    p.add_argument("--z-dim", type=int, default=256)
    p.add_argument("--encoder-window-mode", default="intermediate")
    p.add_argument("--diffsr-feature-dim", type=int, default=128)
    p.add_argument("--diffsr-embed-dim", type=int, default=512)
    p.add_argument("--pretrain-updates", type=int, default=5000)
    p.add_argument("--pretrain-batch-size", type=int, default=8192)
    p.add_argument("--video-length", type=int, default=500)
    p.add_argument("--video-interval", type=int, default=2500)
    p.add_argument("--save-interval", type=int, default=100_000_000)
    p.add_argument("--logger-backend", default="wandb")
    p.add_argument("--wandb-project", default="g1-bones-seed-100-hl-skill-2b")
    p.add_argument("--exp-name", default=None)
    p.add_argument("--skip-pretrain", action="store_true")
    p.add_argument("--skip-low-level", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("[CMD] " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest = data_root / "manifests" / "g1_bones_seed_100_manifest.json"
    dataset_path = data_root / "g1_hl_diffsr"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.exp_name or f"bones_seed_100_h{args.horizon_steps}_z{args.z_dim}_pretrain_lowlevel_seed{args.seed}"
    run_root = Path(args.run_root or (REPO_ROOT / "logs" / "bones_seed_100_pretrain_lowlevel" / f"{run_id}_{timestamp}")).resolve()
    skill_dir = run_root / f"skill_encoder_h{args.horizon_steps}_z{args.z_dim}"
    skill_ckpt = skill_dir / "checkpoints" / "best.pt"
    run_root.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not manifest.is_file():
        raise SystemExit(f"[ERROR] manifest not found: {manifest} (is the dataset staged at {data_root}?)")

    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] manifest={manifest}")
    print(f"[INFO] run_root={run_root}")

    # Stage 1: skill-encoder pretrain (builds the Zarr cache with refresh=true).
    if not args.skip_pretrain:
        _run(
            [
                sys.executable, "scripts/rlopt/train_hl_skill_diffsr.py",
                "--headless", "--device", args.device,
                "--task", args.task,
                "--num_envs", str(args.train_num_envs),
                "--seed", str(args.seed),
                "--output_dir", str(skill_dir),
                "--horizon_steps", str(args.horizon_steps),
                "--encoder_window_mode", args.encoder_window_mode,
                "--z_dim", str(args.z_dim),
                "--diffsr_feature_dim", str(args.diffsr_feature_dim),
                "--diffsr_embed_dim", str(args.diffsr_embed_dim),
                "--batch_size", str(args.pretrain_batch_size),
                "--num_updates", str(args.pretrain_updates),
                "--log_interval", "100",
                "--eval_batches", "4",
                "--eval_batch_size", str(args.pretrain_batch_size),
                "--train_split", "all", "--eval_split", "all",
                "--eval_trajectory_fraction", "0.5",
                "--trajectory_split_seed", str(args.seed),
                "--reconstruction_eval", "--window_probe_eval",
                "--window_probe_train_batches", "8",
                "--window_probe_eval_batches", "4",
                f"env.lafan1_manifest_path={manifest}",
                f"env.dataset_path={dataset_path}",
                "env.refresh_zarr_dataset=true",
            ],
            dry_run=args.dry_run,
        )

    if not args.dry_run and not args.skip_pretrain and not skill_ckpt.is_file():
        raise SystemExit(f"[ERROR] skill checkpoint not produced: {skill_ckpt}")

    # Stage 2: low-level oracle IPMD (hl_skill; reuses the Stage-1 Zarr cache).
    if not args.skip_low_level:
        _run(
            [
                sys.executable, "scripts/rlopt/train.py",
                "--headless", "--video",
                "--video_length", str(args.video_length),
                "--video_interval", str(args.video_interval),
                "--device", args.device,
                "--num_envs", str(args.train_num_envs),
                "--task", args.task,
                "--algo", args.algorithm,
                "--seed", str(args.seed),
                f"agent.collector.total_frames={args.frame_cap}",
                f"agent.logger.backend={args.logger_backend}",
                f"agent.logger.project_name={args.wandb_project}",
                f"agent.logger.exp_name={run_id}_oracle_low_level",
                "agent.logger.video=true",
                f"agent.save_interval={args.save_interval}",
                f"agent.ipmd.hl_skill_checkpoint_path={skill_ckpt}",
                f"env.lafan1_manifest_path={manifest}",
                f"env.dataset_path={dataset_path}",
                "env.refresh_zarr_dataset=false",
            ],
            dry_run=args.dry_run,
        )

    print(f"[INFO] DONE: pretrain + low-level complete. run_root={run_root}")


if __name__ == "__main__":
    main()
