#!/usr/bin/env python3
"""Audit H30 future-planner fusion curves and report per-motion results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from imitation_experiments.pipeline.run_language_planner_oracle_pretrain import (
    aggregate_milestone_evaluations,
)


VARIANTS: tuple[tuple[str, str], ...] = (
    ("h30_future_exponential", "exponential"),
    ("h30_future_clipped_gated", "clipped_gated"),
)
DEFAULT_UPDATES: tuple[int, ...] = (2000, 4000, 6000, 8000, 10000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_goals", type=int, default=10)
    parser.add_argument("--expected_trajectories_per_goal", type=int, default=100)
    parser.add_argument("--updates", type=int, nargs="+", default=DEFAULT_UPDATES)
    return parser.parse_args()


def _weighted_metric(
    summaries: Sequence[dict[str, Any]], *, section: str, metric: str
) -> float:
    total = 0.0
    count = 0
    for summary in summaries:
        row = summary.get(section, {}).get(metric, {})
        if not isinstance(row, dict):
            continue
        metric_count = int(row.get("count", 0))
        value = row.get("mean")
        if metric_count > 0 and value is not None and math.isfinite(float(value)):
            total += float(value) * metric_count
            count += metric_count
    return total / count if count else float("nan")


def _load_variant_update(
    root: Path,
    *,
    update: int,
    variant: str,
    expected_mode: str,
    expected_goals: int,
    expected_trajectories_per_goal: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(
        (root / "evaluation" / f"update_{update:07d}" / variant).glob("*/summary.json")
    )
    if len(paths) != int(expected_goals):
        raise ValueError(
            f"{variant} update {update} has {len(paths)} goal summaries; "
            f"expected {expected_goals}."
        )
    summaries: list[dict[str, Any]] = []
    per_motion: list[dict[str, Any]] = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        per_environment = summary.get("per_environment", [])
        if len(per_environment) != int(expected_trajectories_per_goal):
            raise ValueError(
                f"{variant} update {update} motion {path.parent.name} has "
                f"{len(per_environment)} trajectories; expected "
                f"{expected_trajectories_per_goal}."
            )
        receding = summary.get("metadata", {}).get("latent_receding_horizon")
        if not isinstance(receding, dict):
            raise ValueError(f"{path} has no latent receding-horizon metadata.")
        if receding.get("target_frame") != "future_publication":
            raise ValueError(
                f"{path} target frame {receding.get('target_frame')!r} is not "
                "'future_publication'."
            )
        if receding.get("mode") != expected_mode:
            raise ValueError(
                f"{path} mode {receding.get('mode')!r} != {expected_mode!r}."
            )
        successes = int(
            summary.get("aggregate", {}).get("completed_tracking_success_count", 0)
        )
        per_motion.append(
            {
                "motion": path.parent.name,
                "trajectory_count": len(per_environment),
                "completed_success_count": successes,
                "success_rate": successes / len(per_environment),
                "mpjpe_l_mm_successful": _weighted_metric(
                    [summary],
                    section="successful_trajectory_metrics",
                    metric="tracking_mpjpe_mm",
                ),
                "mpjpe_l_mm_all": _weighted_metric(
                    [summary], section="metrics", metric="tracking_mpjpe_mm"
                ),
            }
        )
        summaries.append(summary)
    return summaries, per_motion


def aggregate_h30_fusion_results(
    root: Path,
    *,
    updates: Sequence[int] = DEFAULT_UPDATES,
    expected_goals: int,
    expected_trajectories_per_goal: int,
) -> dict[str, Any]:
    """Validate both fusion modes at every milestone and return their curves."""
    normalized_updates = tuple(sorted({int(update) for update in updates}))
    if not normalized_updates or normalized_updates[0] <= 0:
        raise ValueError("updates must contain positive integers.")
    variants: dict[str, Any] = {}
    for variant, mode in VARIANTS:
        milestone_summaries: dict[int, list[dict[str, Any]]] = {}
        per_motion_by_update: dict[str, list[dict[str, Any]]] = {}
        for update in normalized_updates:
            summaries, per_motion = _load_variant_update(
                root,
                update=update,
                variant=variant,
                expected_mode=mode,
                expected_goals=expected_goals,
                expected_trajectories_per_goal=expected_trajectories_per_goal,
            )
            milestone_summaries[update] = summaries
            per_motion_by_update[str(update)] = per_motion
        variants[variant] = {
            "execution_mode": mode,
            "curve": aggregate_milestone_evaluations(milestone_summaries),
            "per_motion_by_update": per_motion_by_update,
        }
    final_update = normalized_updates[-1]
    ranking = sorted(
        VARIANTS,
        key=lambda item: (
            -float(variants[item[0]]["curve"]["rows"][-1]["success_rate"]),
            float(variants[item[0]]["curve"]["rows"][-1]["mpjpe_l_mm_successful"]),
        ),
    )
    return {
        "schema": "bones_language_h30_future_fusion_v1",
        "protocol": {
            "updates": list(normalized_updates),
            "goals": int(expected_goals),
            "trajectories_per_goal": int(expected_trajectories_per_goal),
            "target_frame": "future_publication",
            "prediction_horizon_steps": 30,
            "execution_horizon_steps": 10,
            "policy_action_selection": "deterministic",
            "push_enabled": False,
            "other_randomization_kept": True,
        },
        "variants": variants,
        "final_update": final_update,
        "final_ranking": [name for name, _ in ranking],
        "best_final_variant": ranking[0][0],
    }


def _write_results(payload: dict[str, Any], output_root: Path) -> None:
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    json_path = aggregate_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# BONES language10 H30 future-fusion comparison",
        "",
        "Clipped/gated fusion includes the same exponential age weighting as the raw exponential row.",
        "",
        "| variant | update | SONIC SR | successful MPJPE-L (mm) | plateau? |",
        "|---|---:|---:|---:|:---:|",
    ]
    for variant, _ in VARIANTS:
        for row in payload["variants"][variant]["curve"]["rows"]:
            lines.append(
                f"| {variant} | {row['update']} | {row['success_rate']:.3f} | "
                f"{row['mpjpe_l_mm_successful']:.2f} | "
                f"{'yes' if row['plateau_candidate'] else 'no'} |"
            )
    final_update = str(payload["final_update"])
    for variant, _ in VARIANTS:
        lines += [
            "",
            f"## {variant}: per-motion results at update {final_update}",
            "",
            "| motion | successes | SONIC SR | successful MPJPE-L (mm) | all-rollout MPJPE-L (mm) |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in payload["variants"][variant]["per_motion_by_update"][final_update]:
            lines.append(
                f"| {row['motion']} | {row['completed_success_count']}/"
                f"{row['trajectory_count']} | {row['success_rate']:.3f} | "
                f"{row['mpjpe_l_mm_successful']:.2f} | "
                f"{row['mpjpe_l_mm_all']:.2f} |"
            )
    lines += ["", f"Best final row: **{payload['best_final_variant']}**.", ""]
    markdown_path = aggregate_dir / "results.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PASS] {markdown_path}")
    print(f"[PASS] {json_path}")


def main() -> None:
    args = _parse_args()
    root = args.output_root.expanduser().resolve()
    payload = aggregate_h30_fusion_results(
        root,
        updates=args.updates,
        expected_goals=int(args.expected_goals),
        expected_trajectories_per_goal=int(args.expected_trajectories_per_goal),
    )
    _write_results(payload, root)


if __name__ == "__main__":
    main()
