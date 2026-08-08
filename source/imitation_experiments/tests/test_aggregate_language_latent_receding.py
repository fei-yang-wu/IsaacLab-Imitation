from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.evaluation.aggregate_language_latent_receding import (
    VARIANTS,
    aggregate_receding_results,
)


def _write_summary(
    root: Path,
    *,
    variant: str,
    target_frame: str | None,
    mode: str,
    successes: int,
    mpjpe_mm: float,
) -> None:
    receding = None
    if target_frame is not None:
        receding = {"target_frame": target_frame, "mode": mode}
    summary = {
        "metadata": {"latent_receding_horizon": receding},
        "per_environment": [{}, {}],
        "aggregate": {"completed_tracking_success_count": successes},
        "successful_trajectory_metrics": {
            "tracking_mpjpe_mm": {
                "mean": mpjpe_mm,
                "count": max(successes, 1),
            },
            "tracked_body_pos_error_m": {
                "mean": mpjpe_mm / 1000.0,
                "count": max(successes, 1),
            },
        },
        "metrics": {
            "tracking_mpjpe_mm": {"mean": mpjpe_mm, "count": 2},
        },
    }
    output = root / "evaluation" / variant / "goal"
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_aggregate_receding_results_ranks_success_then_mpjpe(tmp_path: Path) -> None:
    for index, (variant, target_frame, mode) in enumerate(VARIANTS):
        _write_summary(
            tmp_path,
            variant=variant,
            target_frame=target_frame,
            mode=mode,
            successes=2 if index in (2, 3) else 1,
            mpjpe_mm=40.0 if index == 3 else 50.0 + index,
        )

    result = aggregate_receding_results(
        tmp_path, expected_goals=1, expected_trajectories_per_goal=2
    )

    assert result["best_variant"] == "h3_future_clipped_gated"
    assert result["ranking"][:2] == [
        "h3_future_clipped_gated",
        "h3_future_exponential",
    ]


def test_aggregate_receding_results_rejects_mislabeled_frame(tmp_path: Path) -> None:
    for variant, target_frame, mode in VARIANTS:
        _write_summary(
            tmp_path,
            variant=variant,
            target_frame=target_frame,
            mode=mode,
            successes=1,
            mpjpe_mm=50.0,
        )
    bad = tmp_path / "evaluation" / "h3_future_first" / "goal" / "summary.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["metadata"]["latent_receding_horizon"]["target_frame"] = (
        "current_publication"
    )
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="target frame"):
        aggregate_receding_results(
            tmp_path, expected_goals=1, expected_trajectories_per_goal=2
        )
