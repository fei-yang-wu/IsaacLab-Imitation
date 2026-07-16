#!/usr/bin/env python3
"""Write launch provenance for interface-baseline aggregate runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENV_KEYS = (
    "TASK",
    "ALGORITHM",
    "LOW_LEVEL_CHECKPOINT",
    "FULL_BODY_TRAJECTORY_CHECKPOINT",
    "EE_TRAJECTORY_CHECKPOINT",
    "MANIFEST",
    "TRAIN_MANIFEST",
    "EVAL_MANIFEST",
    "FULL_MANIFEST",
    "INTERFACES",
    "SEED",
    "SEEDS",
    "OUTPUT_ROOT",
    "OUTPUT_PREFIX",
    "LATENT_OUTPUT_PREFIX",
    "AGGREGATE_OUTPUT_DIR",
    "EXTRA_AGGREGATE_ROOTS",
    "EXTRA_AGGREGATE_GLOBS",
    "RUN_LATENT",
    "RUN_LATENT_BASELINE",
    "RUN_PREFLIGHT",
    "RUN_CAPACITY_BACKFILL",
    "RUN_AGGREGATE",
    "RUN_AUDIT",
    "RUN_SWEEP_ANALYSIS",
    "SELECTED_SAMPLE_COUNT",
    "AUDIT_PLANNER_VARIANTS",
    "AUDIT_EXPECTED_SEEDS",
    "AUDIT_EXPECTED_PRETRAIN_UPDATES",
    "MIN_ORACLE_SURVIVAL",
    "MIN_ORACLE_SUCCESS_RATE",
    "MODEL_SIZE",
    "MODEL_SIZES",
    "SAMPLE_BUDGETS",
    "MICRO_BATCH_SIZE",
    "TRAIN_ENDPOINT_STEPS",
    "PRETRAIN_UPDATES",
    "FINETUNE_UPDATES",
    "FINETUNE_BATCH_SIZE",
    "BATCH_SIZE",
    "LR",
    "FINETUNE_LR",
    "WEIGHT_DECAY",
    "FINETUNE_WEIGHT_DECAY",
    "FLOW_STEPS",
    "FLOW_NOISE_STD",
    "STATE_HISTORY_STEPS",
    "COMMAND_PAST_STEPS",
    "COMMAND_FUTURE_STEPS",
    "NUM_ENVS",
    "STEPS",
    "EVAL_STEPS",
    "COLLECT_STEPS",
    "COLLECT_SAMPLES",
    "SAMPLE_BUDGET",
    "VANILLA_DATASET_PATH",
    "USE_CHECKPOINT_NORMALIZATION",
    "FORCE_COLLECT",
    "RUN_ORACLE",
    "LATENT_TASK",
    "LATENT_ALGORITHM",
    "LATENT_LOW_LEVEL_CHECKPOINT",
    "LATENT_SKILL_CHECKPOINT",
    "LATENT_PLANNER_CHECKPOINT",
    "LATENT_LANGUAGE_EMBEDDINGS",
    "LATENT_MOTION_NAME",
    "LATENT_TRAJECTORY_NAME",
    "LATENT_DATASET_PATH",
    "LATENT_COMMAND_MODE",
    "LATENT_DIM",
    "LATENT_CODE_DIM",
    "LATENT_STEPS",
    "HELDOUT_NAMES",
    "HELDOUT_PATTERNS",
    "HELDOUT_COUNT",
    "HELDOUT_FRACTION",
    "SPLIT_OUTPUT_DIR",
    "SPLIT_PREFIX",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo_root", type=Path, default=None)
    parser.add_argument("--env_key", action="append", default=[])
    parser.add_argument("--result_root", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def _repo_root(args: argparse.Namespace) -> Path:
    if args.repo_root is not None:
        return args.repo_root.expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _git(
    repo_root: Path,
    *args: str,
    check: bool = False,
) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if check:
            raise RuntimeError(result.stderr.strip() or "git command failed")
        return None
    return result.stdout.strip()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--short") or ""
    submodules = _git(repo_root, "submodule", "status", "--recursive") or ""
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "branch", "--show-current"),
        "is_dirty": bool(status),
        "status_short": status.splitlines(),
        "submodule_status": submodules.splitlines(),
    }


def _env_payload(keys: list[str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key in keys:
        if key in os.environ:
            payload[key] = os.environ[key]
    return payload


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root(args)
    env_keys = sorted(set(DEFAULT_ENV_KEYS) | {str(key) for key in args.env_key})
    output_json = args.output_json.expanduser().resolve()
    payload = {
        "label": str(args.label),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "argv": sys.argv,
        "env": _env_payload(env_keys),
        "result_roots": [str(root) for root in args.result_root],
        "notes": [str(note) for note in args.note],
        "git": _git_metadata(repo_root),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote interface run provenance: {output_json}")


if __name__ == "__main__":
    main()
