"""The canonical paper-facing boards, protocols, and the deployable-clip rule."""

from __future__ import annotations

import pytest

from imitation_experiments.evaluation.clip_features import (
    DEPLOYABLE_CLIP_RULE_V1,
    ClipFeatures,
    is_deployable,
    select_deployable_ranks,
)
from imitation_experiments.evaluation.summarize_paper_boards import summarize_board
from imitation_experiments.evaluation.protocol import (
    BOARDS,
    DEPLOYABLE123_MOTIONS,
    PROFILES,
    PROTOCOLS,
)


def _clip(rank: int, **overrides: float) -> ClipFeatures:
    """One clip that passes the rule, before the requested field is broken."""
    values = dict(
        frames=300,
        pel_z_min=0.75,
        pel_z_mean=0.78,
        feet_z_max=0.20,
        root_speed_mean=0.6,
        root_speed_max=1.2,
        jvel_p99=3.0,
        wrist_z_max=1.2,
        travel_m=1.5,
    )
    values.update(overrides)
    return ClipFeatures(rank=rank, motion=f"motion_{rank}", **values)  # type: ignore[arg-type]


def test_deployable_rule_rejects_each_axis_on_its_own() -> None:
    assert is_deployable(_clip(0))
    for field, value in (
        ("pel_z_min", 0.40),
        ("root_speed_max", 3.5),
        ("jvel_p99", 9.0),
        ("feet_z_max", 0.60),
        ("frames", 60),
        ("frames", 900),
    ):
        assert not is_deployable(_clip(0, **{field: value})), field


def test_deployable_selection_depends_only_on_rule_and_seed() -> None:
    pool = [_clip(rank) for rank in range(200)]
    first = select_deployable_ranks(pool, count=20)
    shuffled = list(reversed(pool))
    assert select_deployable_ranks(shuffled, count=20) == first
    assert select_deployable_ranks(pool, count=20, seed=1) != first
    assert first == sorted(first)


def test_deployable_selection_refuses_to_pad_a_short_pool() -> None:
    with pytest.raises(ValueError, match="fewer than the requested"):
        select_deployable_ranks([_clip(rank) for rank in range(5)], count=123)


def test_frozen_deployable123_is_a_subset_of_the_canonical_block() -> None:
    ranks = [rank for rank, _ in DEPLOYABLE123_MOTIONS]
    assert len(ranks) == 123
    assert ranks == sorted(set(ranks))
    assert all(12288 <= rank <= 16383 for rank in ranks)
    board = BOARDS["bones_deployable123_v1"]
    assert [case.trajectory_rank for case in board.cases] == ranks


def test_heldout_block_is_disjoint_from_the_canonical_block() -> None:
    canonical = {
        case.trajectory_rank for case in BOARDS["bones_scoreboard4096_v1"].cases
    }
    heldout = {case.trajectory_rank for case in BOARDS["bones_heldout4096_v1"].cases}
    assert len(heldout) == 4096
    assert canonical.isdisjoint(heldout)


def test_clean_and_robust_protocols_differ_only_in_randomization() -> None:
    clean = PROTOCOLS["sonic_sr_clean_v1"]
    robust = PROTOCOLS["sonic_sr_v1"]
    assert clean.randomization_profile == "none"
    assert dict(clean.randomization_kept) == {
        "push": False,
        "reset": False,
        "startup": False,
    }
    assert robust.randomization_profile == "no_push"
    assert dict(robust.randomization_kept)["startup"] is True
    assert clean.success_definition == robust.success_definition
    assert clean.active_terminations == robust.active_terminations
    assert clean.disabled_terminations == robust.disabled_terminations
    assert clean.content_hash() != robust.content_hash()


def test_clean_protocol_publishes_the_success_only_frame_weighted_reduction() -> None:
    assert PROTOCOLS["sonic_sr_clean_v1"].reduction == "frame_weighted_success_only"
    assert PROTOCOLS["sonic_sr_clean_v1"].mpjpe_definition == (
        "root_position_subtracted_and_world_frame"
    )
    assert PROTOCOLS["sonic_sr_clean_v1"].tracked_body_names == (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )


@pytest.mark.parametrize(
    ("profile_id", "protocol_id", "board_id"),
    (
        ("paper_deployable123_v1", "sonic_sr_clean_v1", "bones_deployable123_v1"),
        ("paper_scoreboard4096_v1", "sonic_sr_clean_v1", "bones_scoreboard4096_v1"),
        (
            "paper_scoreboard4096_robust_v1",
            "sonic_sr_v1",
            "bones_scoreboard4096_v1",
        ),
    ),
)
def test_paper_profiles_are_registered(
    profile_id: str, protocol_id: str, board_id: str
) -> None:
    profile = PROFILES[profile_id]
    assert profile.protocol_id == protocol_id
    assert profile.board_id == board_id
    assert profile.protocol_hash == PROTOCOLS[protocol_id].content_hash()
    assert profile.board_hash == BOARDS[board_id].content_hash()


def test_rule_thresholds_are_the_frozen_ones() -> None:
    assert dict(DEPLOYABLE_CLIP_RULE_V1) == {
        "pel_z_min_min": 0.65,
        "root_speed_max_max": 2.0,
        "jvel_p99_max": 6.0,
        "feet_z_max_max": 0.35,
        "frames_min": 150.0,
        "frames_max": 600.0,
    }


def _result(label: str, episodes: list[dict]) -> dict:
    return {"label": label, "per_environment": episodes}


def _episode(rank: int, *, ok: bool, steps: int, local: float, world: float) -> dict:
    return {
        "trajectory_rank": rank,
        "completed_tracking_success": ok,
        "survival_steps": steps,
        "metrics": {"mpjpe_l_mm": local, "mpjpe_g_mm": world},
    }


def test_row_is_frame_weighted_and_success_only() -> None:
    result = _result(
        "fixture",
        [
            _episode(0, ok=True, steps=100, local=20.0, world=50.0),
            _episode(1, ok=True, steps=300, local=40.0, world=90.0),
            # A failure must not enter either mean, however good it looked.
            _episode(2, ok=False, steps=10, local=1.0, world=1.0),
        ],
    )
    row = summarize_board(result)
    assert row.episodes == 3
    assert row.success_rate == pytest.approx(2 / 3)
    assert row.mpjpe_l_micro_mm == pytest.approx((20.0 * 100 + 40.0 * 300) / 400)
    assert row.mpjpe_l_macro_mm == pytest.approx(30.0)
    assert row.mpjpe_g_micro_mm == pytest.approx((50.0 * 100 + 90.0 * 300) / 400)
    assert row.successful_frames == 400


def test_subset_refuses_to_score_a_board_missing_requested_ranks() -> None:
    result = _result("fixture", [_episode(0, ok=True, steps=10, local=1.0, world=2.0)])
    with pytest.raises(ValueError, match="absent from the result"):
        summarize_board(result, ranks=(0, 1))


def test_row_refuses_to_publish_without_a_success() -> None:
    result = _result("fixture", [_episode(0, ok=False, steps=10, local=1.0, world=2.0)])
    with pytest.raises(ValueError, match="no successful episodes"):
        summarize_board(result)
