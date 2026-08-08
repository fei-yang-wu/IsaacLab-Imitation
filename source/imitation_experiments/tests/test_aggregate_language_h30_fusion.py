from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.evaluation.aggregate_language_h30_fusion import (
    VARIANTS,
    aggregate_h30_fusion_results,
)


def _write_summary(
    root: Path,
    *,
    update: int,
    variant: str,
    mode: str,
    successes: int,
    mpjpe_mm: float,
) -> Path:
    output = root / "evaluation" / f"update_{update:07d}" / variant / "motion"
    output.mkdir(parents=True)
    summary = {
        "metadata": {
            "latent_receding_horizon": {
                "target_frame": "future_publication",
                "mode": mode,
            }
        },
        "per_environment": [{}, {}],
        "aggregate": {"completed_tracking_success_count": successes},
        "successful_trajectory_metrics": {
            "tracking_mpjpe_mm": {"mean": mpjpe_mm, "count": max(successes, 1)},
            "tracked_body_pos_error_m": {
                "mean": mpjpe_mm / 1000.0,
                "count": max(successes, 1),
            },
        },
        "metrics": {"tracking_mpjpe_mm": {"mean": mpjpe_mm + 5.0, "count": 2}},
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_aggregate_h30_fusion_tracks_milestones_and_ranks_final(
    tmp_path: Path,
) -> None:
    for update in (2000, 4000):
        for variant, mode in VARIANTS:
            _write_summary(
                tmp_path,
                update=update,
                variant=variant,
                mode=mode,
                successes=2 if variant.endswith("clipped_gated") else 1,
                mpjpe_mm=40.0 if variant.endswith("clipped_gated") else 45.0,
            )

    result = aggregate_h30_fusion_results(
        tmp_path,
        updates=(2000, 4000),
        expected_goals=1,
        expected_trajectories_per_goal=2,
    )

    assert result["best_final_variant"] == "h30_future_clipped_gated"
    assert (
        result["variants"]["h30_future_exponential"]["curve"]["rows"][0]["success_rate"]
        == 0.5
    )
    assert (
        result["variants"]["h30_future_clipped_gated"]["per_motion_by_update"]["4000"][
            0
        ]["mpjpe_l_mm_all"]
        == 45.0
    )


def test_aggregate_h30_fusion_rejects_wrong_execution_mode(tmp_path: Path) -> None:
    paths = []
    for variant, mode in VARIANTS:
        paths.append(
            _write_summary(
                tmp_path,
                update=2000,
                variant=variant,
                mode=mode,
                successes=1,
                mpjpe_mm=45.0,
            )
        )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["metadata"]["latent_receding_horizon"]["mode"] = "first"
    paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mode"):
        aggregate_h30_fusion_results(
            tmp_path,
            updates=(2000,),
            expected_goals=1,
            expected_trajectories_per_goal=2,
        )
