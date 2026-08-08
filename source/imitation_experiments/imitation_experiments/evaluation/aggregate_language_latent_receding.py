#!/usr/bin/env python3
"""Audit and aggregate the BONES language10 latent receding-horizon grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from imitation_experiments.pipeline.run_language_planner_oracle_pretrain import (
    aggregate_milestone_evaluations,
)


VARIANTS: tuple[tuple[str, str | None, str], ...] = (
    ("h1_baseline", None, "first"),
    ("h3_future_first", "future_publication", "first"),
    ("h3_future_exponential", "future_publication", "exponential"),
    ("h3_future_clipped_gated", "future_publication", "clipped_gated"),
    ("h3_current_first", "current_publication", "first"),
    ("h3_current_exponential", "current_publication", "exponential"),
    ("h3_current_clipped_gated", "current_publication", "clipped_gated"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_goals", type=int, default=10)
    parser.add_argument("--expected_trajectories_per_goal", type=int, default=100)
    return parser.parse_args()


def _summaries(path: Path) -> list[dict[str, Any]]:
    paths = sorted(path.glob("*/summary.json"))
    return [json.loads(item.read_text(encoding="utf-8")) for item in paths]


def aggregate_receding_results(
    root: Path,
    *,
    expected_goals: int,
    expected_trajectories_per_goal: int,
) -> dict[str, Any]:
    """Validate the fixed grid and return its ranked aggregate."""
    rows: list[dict[str, Any]] = []
    for name, expected_frame, expected_mode in VARIANTS:
        summaries = _summaries(root / "evaluation" / name)
        if len(summaries) != int(expected_goals):
            raise ValueError(
                f"{name} has {len(summaries)} goal summaries; "
                f"expected {expected_goals}."
            )
        for summary in summaries:
            per_environment = summary.get("per_environment", [])
            if len(per_environment) != int(expected_trajectories_per_goal):
                raise ValueError(
                    f"{name} summary has {len(per_environment)} trajectories; "
                    f"expected {expected_trajectories_per_goal}."
                )
            receding = summary.get("metadata", {}).get("latent_receding_horizon")
            if expected_frame is None:
                if receding is not None:
                    raise ValueError(
                        f"H1 baseline unexpectedly reports H3 metadata: {receding}"
                    )
            else:
                if not isinstance(receding, dict):
                    raise ValueError(f"{name} has no latent receding-horizon record.")
                if receding.get("target_frame") != expected_frame:
                    raise ValueError(
                        f"{name} target frame {receding.get('target_frame')!r} "
                        f"!= {expected_frame!r}."
                    )
                if receding.get("mode") != expected_mode:
                    raise ValueError(
                        f"{name} mode {receding.get('mode')!r} != {expected_mode!r}."
                    )
        curve = aggregate_milestone_evaluations({10000: summaries})
        row = dict(curve["rows"][0])
        row["variant"] = name
        row["target_frame"] = expected_frame
        row["execution_mode"] = expected_mode
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["success_rate"]),
            float(row.get("mpjpe_l_mm_successful", float("inf"))),
        ),
    )
    payload = {
        "protocol": {
            "goals": int(expected_goals),
            "trajectories_per_goal": int(expected_trajectories_per_goal),
            "planner_update_budget": 10000,
            "planner_rate_hz": 5.0,
            "prediction_horizon_steps": 30,
            "execution_horizon_steps": 10,
            "policy_action_selection": "deterministic",
            "push_enabled": False,
            "other_randomization_kept": True,
        },
        "rows": rows,
        "ranking": [str(row["variant"]) for row in ranked],
        "best_variant": str(ranked[0]["variant"]),
    }
    return payload


def main() -> None:
    args = _parse_args()
    root = args.output_root.expanduser().resolve()
    payload = aggregate_receding_results(
        root,
        expected_goals=int(args.expected_goals),
        expected_trajectories_per_goal=int(args.expected_trajectories_per_goal),
    )
    ranked = sorted(
        payload["rows"],
        key=lambda row: (
            -float(row["success_rate"]),
            float(row.get("mpjpe_l_mm_successful", float("inf"))),
        ),
    )
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    json_path = aggregate_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# BONES language10 latent receding-horizon comparison",
        "",
        "| variant | target frame | execution | SONIC SR | successful MPJPE-L (mm) |",
        "|---|---|---|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['variant']} | {row['target_frame'] or 'H1'} | "
            f"{row['execution_mode']} | {float(row['success_rate']):.4f} | "
            f"{float(row['mpjpe_l_mm_successful']):.2f} |"
        )
    lines += ["", f"Best: **{ranked[0]['variant']}**.", ""]
    markdown_path = aggregate_dir / "results.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[PASS] {markdown_path}")
    print(f"[PASS] {json_path}")


if __name__ == "__main__":
    main()
