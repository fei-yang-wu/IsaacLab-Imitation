"""Utilities for one-process, repeated per-motion low-level screening."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


LOCAL_METRIC = "tracking_mpjpe_mm"
GLOBAL_METRIC = "tracked_body_pos_error_m"


def build_env_rank_assignment(
    trajectory_ranks: Sequence[int], num_envs: int
) -> list[int]:
    """Assign an equal contiguous block of environments to each trajectory."""

    ranks = [int(rank) for rank in trajectory_ranks]
    if not ranks:
        raise ValueError("At least one trajectory rank is required.")
    if any(rank < 0 for rank in ranks):
        raise ValueError("Trajectory ranks must be non-negative.")
    if len(set(ranks)) != len(ranks):
        raise ValueError("Trajectory ranks must be unique.")
    if int(num_envs) < len(ranks) or int(num_envs) % len(ranks) != 0:
        raise ValueError(
            "num_envs must be a positive multiple of the trajectory-rank count "
            f"(got num_envs={num_envs}, ranks={len(ranks)})."
        )
    repeats = int(num_envs) // len(ranks)
    return [rank for rank in ranks for _ in range(repeats)]


def _weighted_metric(
    rows: Sequence[dict[str, Any]], metric_name: str, *, successful_only: bool
) -> float:
    total = 0.0
    count = 0
    for row in rows:
        if successful_only and not bool(row.get("completed_tracking_success")):
            continue
        value = row.get("tracking_metrics", {}).get(metric_name)
        metric_count = int(row.get("tracking_metric_counts", {}).get(metric_name, 0))
        if value is None or metric_count <= 0 or not math.isfinite(float(value)):
            continue
        total += float(value) * metric_count
        count += metric_count
    return total / count if count > 0 else float("nan")


def aggregate_motion_screen(
    evaluation: dict[str, Any], candidates: dict[str, Any]
) -> dict[str, Any]:
    """Group repeated environment outcomes into one record per candidate motion."""

    candidate_rows = candidates.get("candidates")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        raise ValueError("Candidate file must contain a non-empty 'candidates' list.")
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation.get("per_environment", []):
        by_rank[int(row["trajectory_rank"])].append(row)

    results: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        rank = int(candidate["trajectory_rank"])
        rows = by_rank.get(rank, [])
        if not rows:
            raise ValueError(
                f"Evaluation has no environments for trajectory rank {rank}."
            )
        expected_name = str(candidate["motion_name"])
        observed_names = {str(row["motion_name"]) for row in rows}
        if observed_names != {expected_name}:
            raise ValueError(
                f"Trajectory rank {rank} name mismatch: expected {expected_name!r}, "
                f"observed {sorted(observed_names)!r}."
            )
        successful = sum(bool(row.get("completed_tracking_success")) for row in rows)
        repetitions = len(rows)
        result = dict(candidate)
        result.update(
            {
                "repetitions": repetitions,
                "completed_successes": successful,
                "success_rate": successful / repetitions,
                "mpjpe_l_mm_all": _weighted_metric(
                    rows, LOCAL_METRIC, successful_only=False
                ),
                "mpjpe_g_m_all": _weighted_metric(
                    rows, GLOBAL_METRIC, successful_only=False
                ),
                "mpjpe_l_mm_successful": _weighted_metric(
                    rows, LOCAL_METRIC, successful_only=True
                ),
                "mpjpe_g_m_successful": _weighted_metric(
                    rows, GLOBAL_METRIC, successful_only=True
                ),
                "survival_steps_mean": sum(int(row["survival_steps"]) for row in rows)
                / repetitions,
            }
        )
        results.append(result)

    results.sort(
        key=lambda row: (
            -float(row["success_rate"]),
            float(row["mpjpe_l_mm_all"]),
            float(row["mpjpe_g_m_all"]),
        )
    )
    return {
        "schema": "bones_language_motion_screen_v1",
        "evaluation_metadata": evaluation.get("metadata", {}),
        "candidate_source": candidates.get("source", {}),
        "ranking_rule": "success_rate desc, MPJPE-L all asc, MPJPE-G all asc",
        "results": results,
    }


def _write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "motion_name",
        "trajectory_rank",
        "category",
        "language_goal",
        "reference_frames",
        "repetitions",
        "completed_successes",
        "success_rate",
        "mpjpe_l_mm_all",
        "mpjpe_g_m_all",
        "mpjpe_l_mm_successful",
        "mpjpe_g_m_successful",
        "survival_steps_mean",
        "historical_set",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_json", type=Path, required=True)
    parser.add_argument("--candidates_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation_json.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    output = aggregate_motion_screen(evaluation, candidates)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_csv, output["results"])
    print(f"[PASS] Ranked {len(output['results'])} candidate motions.")
    print(f"[INFO] JSON: {args.output_json.resolve()}")
    print(f"[INFO] CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
