"""Tests for the results-page builder.

The reductions are the point of this module, so they are pinned here: the
headline MPJPE is the episode mean, not the transition-weighted or the
successful-only value, and an oracle ceiling is scored on the arm's own motion
set. Getting either wrong changes a published number without changing anything
visible on the page.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.reporting.build import build_report
from imitation_experiments.reporting.records import load_summary
from imitation_experiments.reporting.render import (
    arm_attributes,
    render_html,
    restricted_mpjpe,
)
from imitation_experiments.reporting.spec import RunRef
from imitation_experiments.reporting.spec import load_spec


# Stable board ranks so an arm and its wider oracle ceiling agree on identity.
_RANKS = {"walk": 0, "jump": 1, "panic": 2}


def _summary(
    *,
    label: str,
    episodes: list[tuple[str, float, bool]],
    transition_weighted: float,
    successful_only: float,
    planner: bool,
    max_steps: int = 2000,
    inference_knobs: dict | None = None,
) -> dict:
    """Build a summary.json payload with the fields the reducer reads.

    The protocol/board blocks are what make a record comparable: an oracle
    ceiling is only restricted to an arm's board when both carry the same
    pinned protocol hash and complete episode identities.
    """
    return {
        "task": "Isaac-Imitation-G1-v2",
        "protocol": {
            "protocol_id": "test_protocol_v1",
            "content_hash": "p" * 64,
            "backend": "isaac",
        },
        "board": {"board_id": "test_board_v1", "content_hash": "b" * 64},
        "algorithm": "IPMD",
        "max_steps": max_steps,
        "disable_push_event": True,
        "sonic_termination_profile": "fall_only",
        "metadata": {
            "label": label,
            "interface": "latent_skill",
            "episode_length_s": 10.0,
            "fall_height_m": 0.4,
            "num_envs": len(episodes),
            "seed": 0,
            "low_level_tracker": {
                "checkpoint_path": "logs/tracker/model.pt",
                "checkpoint_sha256": "a" * 64,
                "policy_frozen": True,
                "policy_parameter_count": 8474170,
            },
            "gr00t_planner": (
                {
                    "checkpoint": "outputs/arm/checkpoints/latest.pt",
                    "update": 12000,
                    "action_dim": 64,
                    "action_horizon": 3,
                    "consumption": "open_loop",
                    "quantizer": "fsq",
                    "planner_latency_ms": {"p50": 93.7},
                    **(inference_knobs or {}),
                }
                if planner
                else None
            ),
        },
        "aggregate": {
            "done_rate": 1.0,
            "fall_free_rate": 1.0
            - sum(fell for _, _, fell in episodes) / len(episodes),
            "tracking_success_rate": 1.0,
            "survival_steps_mean": 495.0,
            "valid_transition_count": 1000,
        },
        "metric_means": {"tracking_mpjpe_mm": transition_weighted},
        "successful_trajectory_metrics": {"tracking_mpjpe_mm": successful_only},
        "per_environment": [
            {
                "env_id": index,
                "motion_name": motion,
                "fell": fell,
                # Board identity, not env order: the ceiling covers more
                # motions than the arm, so matching happens on these keys.
                "trajectory_rank": _RANKS[motion],
                "start_frame": 0,
                "env_seed": 0,
                "repeat_index": sum(
                    1 for earlier, _, _ in episodes[:index] if earlier == motion
                ),
                "tracking_metrics": {"tracking_mpjpe_mm": value},
            }
            for index, (motion, value, fell) in enumerate(episodes)
        ],
    }


def _write(root: Path, run: str, payload: dict) -> None:
    path = root / "logs" / run / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A miniature repository holding one planner arm and its oracle ceiling."""
    (tmp_path / "pixi.toml").write_text("", encoding="utf-8")
    (tmp_path / "source").mkdir()

    _write(
        tmp_path,
        "eval/arm",
        _summary(
            label="arm",
            episodes=[
                ("walk", 40.0, False),
                ("walk", 44.0, False),
                ("jump", 50.0, False),
                ("jump", 54.0, True),
            ],
            transition_weighted=61.0,
            successful_only=44.6,
            planner=True,
        ),
    )
    _write(
        tmp_path,
        "eval/ceiling",
        _summary(
            label="ceiling",
            # Same repeats per motion as the arm, plus one motion the arm
            # never ran -- the case the restriction exists for.
            episodes=[
                ("walk", 20.0, False),
                ("walk", 20.0, False),
                ("jump", 24.0, False),
                ("jump", 24.0, False),
                ("panic", 90.0, True),
            ],
            transition_weighted=30.0,
            successful_only=22.0,
            planner=False,
        ),
    )
    return tmp_path


def test_headline_mpjpe_is_the_episode_mean(workspace: Path) -> None:
    record = load_summary(workspace / "logs/eval/arm/summary.json", workspace)

    assert record.mpjpe_mm == pytest.approx(47.0)
    assert record.mpjpe_mm_transition_weighted == pytest.approx(61.0)
    assert record.mpjpe_mm_successful_only == pytest.approx(44.6)


def test_per_motion_reduction_and_balance(workspace: Path) -> None:
    record = load_summary(workspace / "logs/eval/arm/summary.json", workspace)

    scores = {score.motion_name: score for score in record.per_motion}
    assert scores["walk"].mpjpe_mm == pytest.approx(42.0)
    assert scores["jump"].mpjpe_mm == pytest.approx(52.0)
    assert scores["jump"].fall_free_rate == pytest.approx(0.5)
    assert record.is_balanced
    assert record.episodes_per_motion == 2


def test_ceiling_is_restricted_to_the_arm_motion_set(workspace: Path) -> None:
    arm = load_summary(workspace / "logs/eval/arm/summary.json", workspace)
    ceiling = load_summary(workspace / "logs/eval/ceiling/summary.json", workspace)

    matched, motion_count = restricted_mpjpe(ceiling, arm)

    assert motion_count == 2
    assert matched == pytest.approx(22.0)
    # The unrestricted ceiling is far worse because it keeps a motion the arm
    # never ran; reporting that as the bound would understate the planner cost.
    # (20 + 20 + 24 + 24 + 90) / 5 over the ceiling's own board.
    assert ceiling.mpjpe_mm == pytest.approx(35.6, rel=1e-4)


def test_planner_and_oracle_records_are_distinguished(workspace: Path) -> None:
    arm = load_summary(workspace / "logs/eval/arm/summary.json", workspace)
    ceiling = load_summary(workspace / "logs/eval/ceiling/summary.json", workspace)

    assert arm.kind == "planner"
    assert arm.planner_update == 12000
    assert ceiling.kind == "oracle"
    assert ceiling.planner_checkpoint is None


def test_recorded_inference_knobs_override_the_spec_declaration(
    workspace: Path,
) -> None:
    """A hand-written attribute must not contradict the run it describes."""
    _write(
        workspace,
        "eval/recorded",
        _summary(
            label="recorded",
            episodes=[("walk", 40.0, False)],
            transition_weighted=40.0,
            successful_only=40.0,
            planner=True,
            inference_knobs={
                "temporal_ensemble": "exponential",
                "temporal_ensemble_decay": 0.5,
                "num_inference_timesteps": 16,
                "samples_per_publication": 4,
                "consume_slots": 3,
            },
        ),
    )
    record = load_summary(workspace / "logs/eval/recorded/summary.json", workspace)
    # The spec is stale: it claims no ensembling for a run that ensembled.
    ref = RunRef(
        id="r", run="eval/recorded", name="Arm", attributes={"ensemble": "none"}
    )

    merged = arm_attributes(ref, record)

    assert record.planner_temporal_ensemble == "exponential"
    assert merged["ensemble"] == "exponential 0.5"
    assert merged["samples"] == "4"
    assert merged["ODE steps"] == "16"


def test_unrecorded_inference_knobs_leave_the_spec_declaration_alone(
    workspace: Path,
) -> None:
    record = load_summary(workspace / "logs/eval/arm/summary.json", workspace)
    ref = RunRef(
        id="r", run="eval/arm", name="Arm", attributes={"ensemble": "exponential 0.5"}
    )

    merged = arm_attributes(ref, record)

    assert record.planner_temporal_ensemble is None
    assert merged["ensemble"] == "exponential 0.5"


_SPEC = """
meta:
  title: Test report
  subtitle: fixture
  updated: "2026-08-16"
  noise_band_pct: 15.0
low_level:
  headline:
    id: ceiling
    run: eval/ceiling
    name: Oracle ceiling
  rows:
    - {id: ceiling, run: eval/ceiling, name: Oracle ceiling}
planner:
  baseline: arm
  rows:
    - {id: arm, run: eval/arm, name: Planner arm, ceiling: eval/ceiling}
ablations:
  - id: only
    title: Only comparison
    variable: nothing
    question: fixture
    baseline: arm
    rows:
      - {id: arm, run: eval/arm, name: Planner arm, ceiling: eval/ceiling}
method_cards: [mpjpe, fsq64]
caveats: ["fixture caveat"]
"""


def test_build_report_writes_a_self_contained_page(workspace: Path) -> None:
    spec_path = workspace / "report.yaml"
    spec_path.write_text(_SPEC, encoding="utf-8")
    output = workspace / "out/index.html"

    result = build_report(spec_path, output, repo_root=workspace)

    page = output.read_text(encoding="utf-8")
    assert result.record_count == 2
    assert result.data_path.is_file()
    # The headline reduction, the matched ceiling, and the resulting gap.
    assert "47.00" in page
    assert "22.00" in page
    assert "25.00" in page
    # Self-contained: no request leaves the page.
    for scheme in ("http://", "https://", "//cdn", "src="):
        assert scheme not in page
    assert "<math" in page


def test_missing_run_fails_loudly(workspace: Path) -> None:
    spec_path = workspace / "report.yaml"
    spec_path.write_text(_SPEC.replace("eval/ceiling", "eval/absent"), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="absent"):
        build_report(spec_path, workspace / "out/index.html", repo_root=workspace)


def test_baseline_must_be_one_of_the_rows(workspace: Path) -> None:
    spec_path = workspace / "report.yaml"
    spec_path.write_text(
        _SPEC.replace(
            "  baseline: arm\n  rows:\n    - {id: arm",
            "  baseline: ghost\n  rows:\n    - {id: arm",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ghost"):
        load_spec(spec_path, workspace)


def test_differences_inside_the_noise_band_are_marked_unresolved(
    workspace: Path,
) -> None:
    _write(
        workspace,
        "eval/near",
        _summary(
            label="near",
            episodes=[("walk", 42.0, False), ("walk", 44.0, False)],
            transition_weighted=43.0,
            successful_only=43.0,
            planner=True,
        ),
    )
    spec_path = workspace / "report.yaml"
    spec_path.write_text(
        _SPEC.replace(
            "    - {id: arm, run: eval/arm, name: Planner arm, ceiling: eval/ceiling}\n"
            "ablations:",
            "    - {id: arm, run: eval/arm, name: Planner arm, ceiling: eval/ceiling}\n"
            "    - {id: near, run: eval/near, name: Near arm}\n"
            "ablations:",
        ),
        encoding="utf-8",
    )
    spec = load_spec(spec_path, workspace)
    records = {
        run: load_summary(spec.summary_path(run), workspace)
        for run in spec.referenced_runs()
    }

    page = render_html(spec, records)

    # 43.0 against a 47.0 baseline is -8.5%, inside the band.
    assert "unresolved" in page
    assert "-8.5%" in page
