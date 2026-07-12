#!/usr/bin/env python3
"""Cluster-friendly launcher for interface-baseline shell workflows."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


SCRIPT_BY_MODE = {
    "dance102-fair": "run_dance102_fair_interface_comparison.sh",
    "dance102-strong": "run_dance102_strong_interface_comparison.sh",
    "dance102-strong-multiseed": "run_dance102_strong_interface_multiseed.sh",
    "lafan1-heldout": "run_lafan1_heldout_strong_interface_comparison.sh",
    "lafan1-heldout-multiseed": "run_lafan1_heldout_strong_interface_multiseed.sh",
    "lafan1-motion-tracking": "run_lafan1_motion_tracking_evaluation.sh",
    "lafan1-single-trajectory": "run_lafan1_motion_tracking_evaluation.sh",
    "multimotion-heldout": "run_multimotion_heldout_interface_comparison.sh",
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(SCRIPT_BY_MODE), required=True)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment assignment forwarded to the selected workflow.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override DRY_RUN for the selected workflow.",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=None,
        help="Defaults to the repository root inferred from this file.",
    )
    return parser.parse_known_args()


def _repo_root(args: argparse.Namespace) -> Path:
    if args.repo_root is not None:
        return args.repo_root.expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _apply_env_assignments(env: dict[str, str], assignments: list[str]) -> None:
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Expected KEY=VALUE for --env, got: {assignment}")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty environment key in --env {assignment!r}")
        env[key] = value


def _apply_container_python_defaults(env: dict[str, str]) -> None:
    isaac_python = Path("/isaac-sim/python.sh")
    if not isaac_python.is_file():
        return
    env.setdefault("INTERFACE_BASELINE_PYTHON_CMD", str(isaac_python))
    env.setdefault("INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD", str(isaac_python))
    cluster_data_dir = env.get("CLUSTER_DATA_DIR")
    if cluster_data_dir:
        cache_root = Path(cluster_data_dir).expanduser() / "lafan1" / "zarr_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        env.setdefault(
            "ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT",
            str(cache_root),
        )
        cache_run_id = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")
        if not cache_run_id:
            host = env.get("HOSTNAME", "host").replace("/", "_")
            cache_run_id = f"manual-{host}-{os.getpid()}"
        unitree_usd_cache_root = (
            Path(cluster_data_dir).expanduser()
            / "isaaclab_imitation"
            / "unitree_usd_cache"
            / str(cache_run_id)
        )
        unitree_usd_cache_root.mkdir(parents=True, exist_ok=True)
        env.setdefault(
            "ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT",
            str(unitree_usd_cache_root),
        )


def main() -> None:
    args, unknown = _parse_args()
    repo_root = _repo_root(args)
    script = (
        repo_root / "experiments" / "interface_baselines" / SCRIPT_BY_MODE[args.mode]
    )
    if not script.is_file():
        raise FileNotFoundError(script)

    env = dict(os.environ)
    _apply_env_assignments(env, list(args.env))
    _apply_container_python_defaults(env)
    if args.dry_run is not None:
        env["DRY_RUN"] = "1" if args.dry_run else "0"

    ignored_unknown = [arg for arg in unknown if arg]
    if ignored_unknown:
        print(
            "[WARN] Ignoring unrecognized launcher args: "
            + " ".join(shlex.quote(arg) for arg in ignored_unknown)
        )

    cmd = ["bash", str(script)]
    print("[CMD] " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
