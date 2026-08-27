"""The canonical paper-facing boards, protocols, and the testbed-clip rule."""

from __future__ import annotations

import pytest

from imitation_experiments.evaluation.clip_features import (
    TESTBED_CLIP_RULE_V1,
    TESTBED_EXCLUDED_NAME_TOKENS,
    ClipFeatures,
    difficulty_index,
    select_testbed_ranks,
)
from imitation_experiments.evaluation.protocol import (
    BOARDS,
    MILESTONE_TESTBED256_RANKS,
    PROFILES,
    PROTOCOLS,
    TESTBED4096_RANKS,
)
from imitation_experiments.evaluation.summarize_paper_boards import summarize_board


def _clip(
    rank: int, motion: str = "walk_forward_001", **overrides: float
) -> ClipFeatures:
    values = dict(
        frames=300,
        pel_z_min=0.60,
        pel_z_mean=0.72,
        feet_z_max=0.20,
        root_speed_mean=0.6,
        root_speed_max=1.2,
        jvel_p99=3.0,
        wrist_z_max=1.2,
        travel_m=1.5,
    )
    values.update(overrides)
    return ClipFeatures(rank=rank, motion=motion, **values)  # type: ignore[arg-type]


def _corpus(size: int = 500) -> list[ClipFeatures]:
    """A spread-out corpus so the difficulty band is not degenerate."""
    return [
        _clip(
            rank,
            pel_z_min=0.30 + 0.001 * rank,
            root_speed_max=0.01 * rank,
            jvel_p99=0.02 * rank,
            feet_z_max=0.001 * rank,
        )
        for rank in range(size)
    ]


def test_difficulty_index_is_a_population_percentile() -> None:
    scores = difficulty_index(_corpus(100))
    assert len(scores) == 100
    assert all(0.0 < score <= 1.0 for score in scores)
    # Every axis increases with rank in this fixture, so difficulty must too.
    assert scores == sorted(scores)


def test_difficulty_index_averages_ties() -> None:
    identical = [_clip(rank) for rank in range(4)]
    assert difficulty_index(identical) == pytest.approx([0.625] * 4)


def test_testbed_selection_drops_scene_dependent_clips() -> None:
    corpus = _corpus()
    corpus.append(_clip(900, motion="open_door_handle_003_A012", pel_z_min=0.40))
    ranks = select_testbed_ranks(corpus, count=50)
    assert 900 not in ranks
    assert "door" in TESTBED_EXCLUDED_NAME_TOKENS


def test_testbed_selection_keeps_crouch_kneel_and_crawl() -> None:
    """SONIC deploys these on hardware; a board that drops them measures ease."""
    for motion in ("kneeling_start_002", "crawl_ff_loop_225_001", "squat_down_004"):
        assert not (set(motion.split("_")) & TESTBED_EXCLUDED_NAME_TOKENS), motion


def test_testbed_selection_drops_length_and_below_floor_artifacts() -> None:
    corpus = _corpus()
    corpus += [
        _clip(901, frames=40),
        _clip(902, frames=4000),
        _clip(903, pel_z_min=-0.13),
    ]
    ranks = select_testbed_ranks(corpus, count=50)
    assert not ({901, 902, 903} & set(ranks))


def test_testbed_selection_is_deterministic_and_order_independent() -> None:
    corpus = _corpus()
    first = select_testbed_ranks(corpus, count=50)
    assert select_testbed_ranks(list(reversed(corpus)), count=50) == first
    assert select_testbed_ranks(corpus, count=50, seed=1) != first
    assert first == sorted(first)


def test_testbed_selection_refuses_to_pad_a_short_band() -> None:
    with pytest.raises(ValueError, match="fewer than the requested"):
        select_testbed_ranks(_corpus(20), count=4096)


def test_frozen_testbed_board_matches_the_registered_ranks() -> None:
    assert len(TESTBED4096_RANKS) == 4096
    assert list(TESTBED4096_RANKS) == sorted(set(TESTBED4096_RANKS))
    assert all(0 <= rank < 129785 for rank in TESTBED4096_RANKS)
    board = BOARDS["bones_testbed4096_v1"]
    assert [case.trajectory_rank for case in board.cases] == list(TESTBED4096_RANKS)


def test_testbed_is_drawn_from_the_whole_corpus_not_one_block() -> None:
    legacy = {case.trajectory_rank for case in BOARDS["bones_scoreboard4096_v1"].cases}
    overlap = legacy & set(TESTBED4096_RANKS)
    assert len(overlap) < 200
    assert max(TESTBED4096_RANKS) - min(TESTBED4096_RANKS) > 100_000


def test_rule_thresholds_are_the_frozen_ones() -> None:
    assert dict(TESTBED_CLIP_RULE_V1) == {
        "frames_min": 100.0,
        "frames_max": 1500.0,
        "pel_z_min_floor": 0.0,
        "difficulty_min": 0.25,
    }


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
    clean = PROTOCOLS["sonic_sr_clean_v1"]
    assert clean.reduction == "frame_weighted_success_only"
    assert clean.mpjpe_definition == "root_position_subtracted_and_world_frame"
    assert clean.tracked_body_names == (
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
        ("paper_testbed4096_v1", "sonic_sr_clean_v1", "bones_testbed4096_v1"),
        ("paper_testbed4096_robust_v1", "sonic_sr_v1", "bones_testbed4096_v1"),
        ("paper_scoreboard4096_v1", "sonic_sr_clean_v1", "bones_scoreboard4096_v1"),
        ("paper_scoreboard4096_robust_v1", "sonic_sr_v1", "bones_scoreboard4096_v1"),
        (
            "paper_milestone_testbed256_v1",
            "sonic_sr_clean_v1",
            "bones_milestone_testbed256_v1",
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


def test_milestone_board_is_a_subset_of_the_testbed_population() -> None:
    """The budget-axis curve and its endpoint must be the same population.

    A curve scored on the legacy `bones_milestone256_v1` block cannot be read
    against a testbed row, because the two boards are different populations.
    """
    assert len(MILESTONE_TESTBED256_RANKS) == 256
    assert set(MILESTONE_TESTBED256_RANKS) <= set(TESTBED4096_RANKS)
    assert list(MILESTONE_TESTBED256_RANKS) == sorted(set(MILESTONE_TESTBED256_RANKS))
    board = BOARDS["bones_milestone_testbed256_v1"]
    assert [case.trajectory_rank for case in board.cases] == list(
        MILESTONE_TESTBED256_RANKS
    )


def test_milestone_board_is_derived_not_transcribed() -> None:
    """It must move with the frozen testbed, not drift away from it."""
    assert MILESTONE_TESTBED256_RANKS == TESTBED4096_RANKS[::16]


def test_milestone_and_testbed_rows_share_one_protocol() -> None:
    """Only the population may differ; a protocol difference would make the
    curve and its endpoint incomparable."""
    milestone = PROFILES["paper_milestone_testbed256_v1"]
    headline = PROFILES["paper_testbed4096_v1"]
    assert milestone.protocol_id == headline.protocol_id
    assert milestone.protocol_hash == headline.protocol_hash
    assert milestone.board_id != headline.board_id


def test_milestone_board_is_not_the_retired_legacy_milestone() -> None:
    legacy = {case.trajectory_rank for case in BOARDS["bones_milestone256_v1"].cases}
    assert not legacy & set(MILESTONE_TESTBED256_RANKS)


def _result(label: str, episodes: list[dict]) -> dict:
    return {"label": label, "per_environment": episodes}


def _episode(
    rank: int,
    *,
    ok: bool,
    steps: int,
    local: float,
    world: float,
    velocity: float | None = None,
    acceleration: float | None = None,
    terms: tuple[str, ...] = (),
) -> dict:
    metrics: dict[str, float] = {"mpjpe_l_mm": local, "mpjpe_g_mm": world}
    if velocity is not None:
        metrics["tracking_velocity_distance_mps"] = velocity
    if acceleration is not None:
        metrics["tracking_acceleration_distance_mps2"] = acceleration
    episode = {
        "trajectory_rank": rank,
        "completed_tracking_success": ok,
        "survival_steps": steps,
        "metrics": metrics,
    }
    if terms:
        episode["termination_terms"] = list(terms)
    return episode


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


def test_velocity_and_acceleration_are_success_only_and_frame_weighted() -> None:
    """SONIC's other two published tracker metrics ride on the same reduction."""
    result = _result(
        "fixture",
        [
            _episode(
                0,
                ok=True,
                steps=100,
                local=20.0,
                world=50.0,
                velocity=0.2,
                acceleration=8.0,
            ),
            _episode(
                1,
                ok=True,
                steps=300,
                local=40.0,
                world=90.0,
                velocity=0.6,
                acceleration=4.0,
            ),
            _episode(
                2,
                ok=False,
                steps=10,
                local=1.0,
                world=1.0,
                velocity=9.9,
                acceleration=9.9,
            ),
        ],
    )
    row = summarize_board(result)
    assert row.velocity_distance_mps == pytest.approx((0.2 * 100 + 0.6 * 300) / 400)
    assert row.acceleration_distance_mps2 == pytest.approx(
        (8.0 * 100 + 4.0 * 300) / 400
    )


def test_a_metric_the_result_predates_stays_absent() -> None:
    """Acceleration was board-wide only until it was accumulated per episode.

    Reporting the all-transition mean in a success-only column would be a
    different quantity, so an older file must report absent, not zero.
    """
    result = _result(
        "fixture",
        [_episode(0, ok=True, steps=100, local=20.0, world=50.0, velocity=0.2)],
    )
    row = summarize_board(result)
    assert row.velocity_distance_mps == pytest.approx(0.2)
    assert row.acceleration_distance_mps2 is None


def test_termination_counts_cover_every_episode_most_frequent_first() -> None:
    """Failure terms occur exactly on the episodes a success-only mean drops."""
    result = _result(
        "fixture",
        [
            _episode(
                0,
                ok=True,
                steps=100,
                local=20.0,
                world=50.0,
                terms=("reference_finished",),
            ),
            _episode(
                1, ok=False, steps=10, local=1.0, world=1.0, terms=("ee_body_pos",)
            ),
            _episode(
                2, ok=False, steps=10, local=1.0, world=1.0, terms=("ee_body_pos",)
            ),
            _episode(
                3, ok=False, steps=10, local=1.0, world=1.0, terms=("anchor_ori",)
            ),
        ],
    )
    row = summarize_board(result)
    assert row.termination_counts == (
        ("ee_body_pos", 2),
        ("anchor_ori", 1),
        ("reference_finished", 1),
    )


def test_env_frames_come_from_metadata_then_from_the_checkpoint_path() -> None:
    """A chained segment restarts its step counter, so the checkpoint tree name
    carries the true cumulative frame count."""
    episodes = [_episode(0, ok=True, steps=100, local=20.0, world=50.0)]
    from_path = _result("fixture", episodes)
    from_path["metadata"] = {
        "checkpoint": "logs/mirror/arm_seed0/tracker/f10000269312/model.pt"
    }
    assert summarize_board(from_path).env_frames == 10_000_269_312

    declared = _result("fixture", episodes)
    declared["metadata"] = {
        "cumulative_env_frames": 2_000_000_000,
        "checkpoint": "logs/mirror/arm_seed0/tracker/f10000269312/model.pt",
    }
    assert summarize_board(declared).env_frames == 2_000_000_000

    assert summarize_board(_result("fixture", episodes)).env_frames is None


def test_row_refuses_to_publish_without_a_success() -> None:
    result = _result("fixture", [_episode(0, ok=False, steps=10, local=1.0, world=2.0)])
    with pytest.raises(ValueError, match="no successful episodes"):
        summarize_board(result)
