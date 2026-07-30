#!/usr/bin/env python3
"""Audit the paired enc380 demonstrations before either planner is trained."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROUTES = {
    "root_qpos": 380,
    "latent_skill": 256,
}
IDENTITY_TENSORS = (
    "planner_state",
    "expert_planner_state",
    "env_id",
    "episode_id",
    "control_step",
    "planner_step",
    "trajectory_rank",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a sample mapping: {path}")
    return payload


def _paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("sample_step_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No sample shards found in {root}.")
    return paths


def audit(
    root_qpos_dir: Path,
    latent_skill_dir: Path,
    *,
    expected_rows: int,
    min_trajectories: int,
    expected_trajectories: int = 0,
    expected_motion: str,
) -> dict[str, Any]:
    route_dirs = {
        "root_qpos": root_qpos_dir.expanduser().resolve(),
        "latent_skill": latent_skill_dir.expanduser().resolve(),
    }
    route_paths = {route: _paths(path) for route, path in route_dirs.items()}
    names = [[path.name for path in paths] for paths in route_paths.values()]
    if names[0] != names[1]:
        raise ValueError("The two materialized routes have different shard names.")

    rows = 0
    trajectory_keys: list[torch.Tensor] = []
    shard_records: list[dict[str, Any]] = []
    for root_path, latent_path in zip(
        route_paths["root_qpos"], route_paths["latent_skill"], strict=True
    ):
        root = _load(root_path)
        latent = _load(latent_path)
        for key in IDENTITY_TENSORS:
            root_value = root.get(key)
            latent_value = latent.get(key)
            if not isinstance(root_value, torch.Tensor) or not isinstance(
                latent_value, torch.Tensor
            ):
                raise ValueError(f"{root_path.name} is missing tensor {key!r}.")
            if not torch.equal(root_value, latent_value):
                raise ValueError(
                    f"Paired routes differ in {key!r} for {root_path.name}."
                )
        root_target = root.get("causal_target")
        latent_target = latent.get("causal_target")
        if not isinstance(root_target, torch.Tensor) or root_target.ndim != 2:
            raise ValueError(f"{root_path} has no rank-2 causal_target.")
        if not isinstance(latent_target, torch.Tensor) or latent_target.ndim != 2:
            raise ValueError(f"{latent_path} has no rank-2 causal_target.")
        if int(root_target.shape[1]) != ROUTES["root_qpos"]:
            raise ValueError(f"{root_path} target is not 380-D root+qpos.")
        if int(latent_target.shape[1]) != ROUTES["latent_skill"]:
            raise ValueError(f"{latent_path} target is not 256-D latent skill.")
        if int(root_target.shape[0]) != int(latent_target.shape[0]):
            raise ValueError(f"Paired row count differs in {root_path.name}.")
        motion_names = root.get("motion_name")
        if not isinstance(motion_names, list) or set(motion_names) != {expected_motion}:
            raise ValueError(
                f"{root_path} is not restricted to motion {expected_motion!r}."
            )
        shard_rows = int(root_target.shape[0])
        rows += shard_rows
        trajectory_keys.append(
            torch.stack(
                [
                    root["env_id"].reshape(-1).to(dtype=torch.long),
                    root["episode_id"].reshape(-1).to(dtype=torch.long),
                ],
                dim=1,
            )
        )
        shard_records.append(
            {
                "name": root_path.name,
                "rows": shard_rows,
                "root_qpos_sha256": _sha256(root_path),
                "latent_skill_sha256": _sha256(latent_path),
            }
        )

    if expected_rows > 0 and rows != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} rows, found {rows}.")
    all_trajectory_keys = torch.cat(trajectory_keys, dim=0)
    unique_trajectory_keys, trajectory_row_counts = torch.unique(
        all_trajectory_keys, dim=0, return_counts=True
    )
    trajectories = int(unique_trajectory_keys.shape[0])
    if expected_trajectories > 0 and trajectories != expected_trajectories:
        raise ValueError(
            f"Expected exactly {expected_trajectories} trajectories, "
            f"found {trajectories}."
        )
    if trajectories < min_trajectories:
        raise ValueError(
            f"Expected at least {min_trajectories} trajectories, found {trajectories}."
        )
    return {
        "format": "enc380_paired_demonstration_audit",
        "version": 1,
        "passed": True,
        "motion_name": expected_motion,
        "rows": rows,
        "trajectories": trajectories,
        "expected_trajectories": (
            expected_trajectories if expected_trajectories > 0 else None
        ),
        "minimum_trajectories": min_trajectories,
        "trajectory_rows_min": int(trajectory_row_counts.min().item()),
        "trajectory_rows_max": int(trajectory_row_counts.max().item()),
        "trajectory_rows_mean": float(trajectory_row_counts.float().mean().item()),
        "trajectory_key": ["env_id", "episode_id"],
        "planner_input_key": "planner_state",
        "planner_input_is_causal": True,
        "routes": ROUTES,
        "shards": shard_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_qpos_dir", type=Path, required=True)
    parser.add_argument("--latent_skill_dir", type=Path, required=True)
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=0,
        help="Exact row count; <=0 accepts variable rows from completed trajectories.",
    )
    parser.add_argument("--min_trajectories", type=int, default=100)
    parser.add_argument("--expected_trajectories", type=int, default=0)
    parser.add_argument("--expected_motion", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.root_qpos_dir,
        args.latent_skill_dir,
        expected_rows=args.expected_rows,
        min_trajectories=args.min_trajectories,
        expected_trajectories=args.expected_trajectories,
        expected_motion=args.expected_motion,
    )
    output = args.output_json.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"[PASS] {result['motion_name']}: {result['rows']} rows, "
        f"{result['trajectories']} trajectories -> {output}"
    )


if __name__ == "__main__":
    main()
