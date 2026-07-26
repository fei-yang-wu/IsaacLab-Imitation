#!/usr/bin/env python3
"""Resolve one seed/motion task in the Phase-4 no-language matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


def _manifest_entries(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.expanduser().resolve().read_text(encoding="utf-8"))
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest has no dataset.trajectories.lafan1_csv entries.")
    if not all(isinstance(entry, dict) for entry in entries):
        raise TypeError("Manifest trajectory entries must be JSON objects.")
    return entries


def motion_names(manifest: Path) -> list[str]:
    names = [
        str(entry.get("name", "")).strip() for entry in _manifest_entries(manifest)
    ]
    if any(not name for name in names):
        raise ValueError("Every Phase-4 trajectory must have a nonempty name.")
    if len(set(names)) != len(names):
        raise ValueError("Phase-4 trajectory names must be unique.")
    return names


def motion_slug(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "motion"
    suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{prefix[:80]}-{suffix}"


def parse_integer_list(
    values: list[str],
    *,
    label: str,
    allow_zero: bool,
) -> list[int]:
    parsed = [int(value) for value in values]
    minimum = 0 if allow_zero else 1
    if not parsed or any(value < minimum for value in parsed):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must contain {qualifier} integers.")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must not contain duplicates.")
    return parsed


def resolve_task(
    manifest: Path,
    *,
    seeds: list[int],
    task_index: int,
    sample_budgets: list[int],
    num_envs: int,
) -> dict[str, Any]:
    names = motion_names(manifest)
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique non-negative integers.")
    if (
        not sample_budgets
        or any(value <= 0 for value in sample_budgets)
        or len(set(sample_budgets)) != len(sample_budgets)
    ):
        raise ValueError("Sample budgets must be unique positive integers.")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    total_tasks = len(names) * len(seeds)
    if task_index < 0 or task_index >= total_tasks:
        raise ValueError(f"task_index must be in [0, {total_tasks}), got {task_index}.")
    seed_index, motion_index = divmod(task_index, len(names))
    maximum_budget = max(sample_budgets)
    planner_decisions_per_env = math.ceil(maximum_budget / num_envs)
    return {
        "task_index": task_index,
        "total_tasks": total_tasks,
        "motion_count": len(names),
        "motion_index": motion_index,
        "motion_name": names[motion_index],
        "motion_slug": motion_slug(names[motion_index]),
        "seed": seeds[seed_index],
        "sample_budgets": sample_budgets,
        "maximum_sample_budget": maximum_budget,
        "num_envs": num_envs,
        "planner_decisions_per_env": planner_decisions_per_env,
        "available_demonstration_rows": planner_decisions_per_env * num_envs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument(
        "--sample_budgets", nargs="+", default=["1000", "10000", "50000"]
    )
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = parse_integer_list(list(args.seeds), label="seeds", allow_zero=True)
    budgets = parse_integer_list(
        list(args.sample_budgets), label="sample_budgets", allow_zero=False
    )
    result = resolve_task(
        args.manifest,
        seeds=seeds,
        task_index=int(args.task_index),
        sample_budgets=budgets,
        num_envs=int(args.num_envs),
    )
    if args.format == "lines":
        for key in (
            "motion_name",
            "motion_slug",
            "seed",
            "total_tasks",
            "motion_count",
            "maximum_sample_budget",
            "planner_decisions_per_env",
            "available_demonstration_rows",
        ):
            print(result[key])
        return
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
