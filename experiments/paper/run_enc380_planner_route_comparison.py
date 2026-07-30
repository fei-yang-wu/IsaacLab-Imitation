#!/usr/bin/env python3
"""Run the walk1 enc380 shared-tracker capacity comparison."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_RUNNER = (
    REPO_ROOT
    / "experiments/campaigns/2026-07-23-lafan1-planner-capacity"
    / "run_enc380_planner_route_comparison.sh"
)
from imitation_experiments.capacity.enc380_capacity_grid import (
    MODEL_SIZES,
    MOTIONS,
    PLANNER_BATCH_SIZE,
    PLANNER_MICRO_BATCH_BY_SIZE,
    PLANNER_SEEDS,
    PLANNER_UPDATES_BY_SIZE,
    planner_dir_name,
)

CELL_STAGES = ("train", "eval")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def _gate_file(path: Path, expected_sha256: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}."
        )


def _run(environment: dict[str, str], *, stages: Iterable[str]) -> None:
    child = dict(environment)
    child["STAGES"] = " ".join(stages)
    result = subprocess.run(
        ["bash", str(CAMPAIGN_RUNNER)],
        cwd=REPO_ROOT,
        env=child,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@hydra.main(version_base=None, config_path="conf", config_name="enc380_planner_route")
def main(cfg: DictConfig) -> None:
    dry_run = bool(cfg.execution.dry_run)
    low_level = Path(_resolved(cfg.paths.low_level_checkpoint))
    encoder = Path(_resolved(cfg.paths.skill_checkpoint))
    completion_record = Path(_resolved(cfg.paths.tracker_completion_record))
    manifest = Path(_resolved(cfg.paths.manifest))
    dataset = Path(_resolved(cfg.paths.dataset_path))
    _gate_file(low_level, str(cfg.checkpoint.low_level_sha256), dry_run=dry_run)
    _gate_file(encoder, str(cfg.checkpoint.skill_sha256), dry_run=dry_run)
    if not dry_run and not completion_record.is_file():
        raise FileNotFoundError(completion_record)
    if not dry_run and not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not dry_run and not dataset.is_dir():
        raise NotADirectoryError(dataset)

    motions = [str(value) for value in cfg.protocol.motion_names]
    sizes = [str(value) for value in cfg.planner.model_sizes]
    seeds = [int(value) for value in cfg.planner.seeds]
    updates_by_size = {
        str(key): int(value)
        for key, value in cfg.planner.train_updates_by_size.items()
    }
    micro_batch_by_size = {
        str(key): int(value)
        for key, value in cfg.planner.micro_batch_size_by_size.items()
    }
    stages = [str(value) for value in cfg.stages]
    if (
        tuple(motions) != MOTIONS
        or tuple(sizes) != MODEL_SIZES
        or tuple(seeds) != PLANNER_SEEDS
    ):
        raise ValueError(
            "Hydra enc380 grid differs from the frozen campaign grid; update both "
            "intentionally instead of running a partial or substituted matrix."
        )
    if updates_by_size != PLANNER_UPDATES_BY_SIZE:
        raise ValueError(
            "Hydra optimizer budgets differ from the frozen capacity-aware schedule."
        )
    if int(cfg.planner.batch_size) != PLANNER_BATCH_SIZE:
        raise ValueError(
            f"enc380 effective planner batch must be {PLANNER_BATCH_SIZE}."
        )
    if micro_batch_by_size != PLANNER_MICRO_BATCH_BY_SIZE:
        raise ValueError(
            "Hydra microbatches differ from the frozen capacity-aware schedule."
        )
    if int(cfg.data.trajectories_per_motion) * len(motions) != 100:
        raise ValueError(
            "enc380 oracle collection must contain exactly 100 trajectories."
        )
    if int(cfg.data.collection_envs) != 100:
        raise ValueError(
            "enc380 oracle collection must use exactly 100 parallel environments."
        )
    environment = dict(os.environ)
    environment.update(
        {
            "LOW_LEVEL_CHECKPOINT": str(low_level),
            "SKILL_CHECKPOINT": str(encoder),
            "TRACKER_COMPLETION_RECORD": str(completion_record),
            "EXPECTED_LOW_LEVEL_SHA256": str(cfg.checkpoint.low_level_sha256),
            "EXPECTED_SKILL_SHA256": str(cfg.checkpoint.skill_sha256),
            "MANIFEST": str(manifest),
            "DATASET_PATH": str(dataset),
            "OUTPUT_ROOT": _resolved(cfg.paths.output_root),
            "TASK": str(cfg.protocol.task),
            "DRY_RUN": "1" if dry_run else "0",
            "DEVICE": str(cfg.execution.device),
            "COLLECT_ENVS": str(cfg.data.collection_envs),
            "DEMO_TRAJECTORIES_PER_MOTION": str(cfg.data.trajectories_per_motion),
            "COLLECT_STEPS": str(cfg.data.collection_control_steps),
            "BATCH_SIZE": str(cfg.planner.batch_size),
            "FLOW_STEPS": str(cfg.planner.flow_inference_steps),
            "EVAL_STEPS": str(cfg.protocol.evaluation_steps),
            "EVAL_ENVS": str(cfg.protocol.evaluation_envs),
            "QUALIFY_STEPS": str(cfg.qualification.steps),
            "QUALIFY_ENVS": str(cfg.qualification.num_envs),
            "MIN_ORACLE_SUCCESS": str(cfg.qualification.min_oracle_success),
            "RENDER_VIDEO": "1" if bool(cfg.protocol.render_video) else "0",
        }
    )
    provenance = {
        "format": "enc380_planner_route_capacity_run",
        "version": 3,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "grid": {
            "motions": motions,
            "model_sizes": sizes,
            "seeds": seeds,
            "capacity_cells": len(motions) * len(sizes) * len(seeds),
            "routes_per_cell": 2,
            "effective_batch_size": PLANNER_BATCH_SIZE,
            "updates_by_size": updates_by_size,
            "micro_batch_by_size": micro_batch_by_size,
        },
    }
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)

    if "qualify" in stages:
        _run(environment, stages=("qualify",))
    if "demo" in stages:
        environment.pop("MOTION_NAME", None)
        _run(environment, stages=("demo",))
    selected_cell_stages = tuple(stage for stage in CELL_STAGES if stage in stages)
    if selected_cell_stages:
        for seed in seeds:
            for size in sizes:
                for motion in motions:
                    environment.update(
                        {
                            "MOTION_NAME": motion,
                            "MODEL_SIZE": size,
                            "SEED": str(seed),
                            "TRAIN_UPDATES": str(updates_by_size[size]),
                            "MICRO_BATCH_SIZE": str(micro_batch_by_size[size]),
                            "PLANNER_DIR_NAME": planner_dir_name(size),
                        }
                    )
                    _run(environment, stages=selected_cell_stages)
    if "aggregate" in stages:
        _run(environment, stages=("aggregate",))


if __name__ == "__main__":
    main()
