#!/usr/bin/env python3
"""Cluster entrypoint for one LAFAN1 horizon cell."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--interfaces", type=str, required=True)
    parser.add_argument("--ranks", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--budget", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--full-manifest", type=str, default=None)
    parser.add_argument("--dry-run", type=str, default=None)
    args, _unknown = parser.parse_known_args()
    return args


def _ensure_lafan1_data_link(repo_root: Path, data_root: Path) -> None:
    link = repo_root / "data" / "lafan1"
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if data_root.is_dir():
        link.symlink_to(data_root, target_is_directory=True)
        print(f"[lafan1_ablation] linked {link} -> {data_root}")


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    script = Path(__file__).resolve().parent / "run_horizon_ablation.sh"
    if not script.is_file():
        print(f"[ERROR] missing {script}", file=sys.stderr)
        return 2

    flash_user = os.environ.get("SKYNET_FLASH_USER") or os.environ.get("USER", "unknown")
    data_dir = Path(
        os.environ.get(
            "CLUSTER_DATA_DIR",
            f"/coc/flash12/{flash_user}/Research/IsaacLab/data",
        )
    )
    lafan1_root = Path(os.environ.get("CLUSTER_G1_DATA_ROOT", str(data_dir / "lafan1")))
    default_manifest = Path(
        os.environ.get(
            "CLUSTER_G1_MANIFEST_PATH",
            str(lafan1_root / "manifests" / "g1_lafan1_manifest.json"),
        )
    )
    default_output = (
        data_dir.parent / "logs" / "lafan1_ablation" / f"seed{os.environ.get('SEED', '0')}"
    )

    env = os.environ.copy()
    env.setdefault("LAFAN1_ABLATION_PYTHON_CMD", "python")
    env.setdefault("LAFAN1_ABLATION_ISAACLAB_PYTHON_CMD", "python")
    env.setdefault("FULL_MANIFEST", str(default_manifest))
    env.setdefault("LL_MANIFEST", env["FULL_MANIFEST"])
    env.setdefault("OUTPUT_ROOT", str(default_output))
    env.setdefault("DRY_RUN", "0")
    env.setdefault("BUDGET", "full")
    env.setdefault("LOGGER_BACKEND", "wandb")
    env.setdefault("WANDB_PROJECT", "G1-Imitation-LAFAN1-Ablation")
    env.setdefault(
        "WANDB_GROUP",
        f"lafan1_ablation_seed{env.get('SEED', os.environ.get('SEED', '0'))}",
    )

    env["WINDOWS"] = str(args.window)
    env["INTERFACES"] = args.interfaces
    if args.ranks is not None:
        env["RANKS"] = args.ranks
    if args.seed is not None:
        env["SEED"] = str(args.seed)
        env["OUTPUT_ROOT"] = str(
            data_dir.parent / "logs" / "lafan1_ablation" / f"seed{args.seed}"
        )
    if args.budget is not None:
        env["BUDGET"] = args.budget
    if args.output_root is not None:
        env["OUTPUT_ROOT"] = args.output_root
    if args.full_manifest is not None:
        env["FULL_MANIFEST"] = args.full_manifest
        env["LL_MANIFEST"] = args.full_manifest
    if args.dry_run is not None:
        env["DRY_RUN"] = args.dry_run

    _ensure_lafan1_data_link(repo_root, lafan1_root)

    print("[lafan1_ablation] cluster horizon cell entrypoint")
    print(f"  repo={repo_root}")
    print(f"  WINDOWS={env['WINDOWS']} INTERFACES={env['INTERFACES']}")
    print(f"  FULL_MANIFEST={env['FULL_MANIFEST']}")
    print(f"  OUTPUT_ROOT={env['OUTPUT_ROOT']}")
    print(f"  RANKS={env.get('RANKS', '(budget default)')}")
    print(f"  BUDGET={env['BUDGET']} DRY_RUN={env['DRY_RUN']}")
    print(f"  PHYSICS_BACKEND={env.get('PHYSICS_BACKEND', '<task-default>')}")
    print(
        f"  LOGGER_BACKEND={env['LOGGER_BACKEND']} "
        f"WANDB_PROJECT={env.get('WANDB_PROJECT')} "
        f"WANDB_GROUP={env.get('WANDB_GROUP')}"
    )

    return subprocess.call(["bash", str(script)], cwd=str(repo_root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
