"""Contract tests for the SONIC-paper-facing proxy population."""

from __future__ import annotations

import json

import pytest

from imitation_experiments.evaluation import protocol
from imitation_experiments.evaluation.sonic_paper_proxy import (
    BONES_CATEGORY_TO_SONIC_GROUP,
    SONIC_TABLE2_TEST_CONTENT,
    SONIC_TABLE2_TEST_REPETITION,
    CorpusClip,
    load_corpus_clips,
    mixture_shares,
    select_proxy_ranks,
    sonic_group,
)


def _corpus() -> list[CorpusClip]:
    """A synthetic corpus with enough headroom in every mappable group."""
    categories = sorted(BONES_CATEGORY_TO_SONIC_GROUP)
    clips = []
    rank = 0
    for category in categories:
        for index in range(800):
            clips.append(
                CorpusClip(rank=rank, motion=f"{category}_{index}", category=category)
            )
            rank += 1
    for index in range(800):
        clips.append(
            CorpusClip(
                rank=rank,
                motion=f"injured_leg_walk_{index}",
                category="Basic Locomotion Styles",
            )
        )
        rank += 1
    return clips


def test_table2_columns_match_published_totals():
    # SONIC Table 2: test-repetition 6,306 clips, test-content 6,998.
    assert sum(SONIC_TABLE2_TEST_REPETITION.values()) == 6306
    assert sum(SONIC_TABLE2_TEST_CONTENT.values()) == 6998


def test_injured_name_wins_over_category():
    injured = CorpusClip(rank=0, motion="inj_torso_idle_turn_360", category="Gestures")
    assert sonic_group(injured) == "Injured"
    plain = CorpusClip(rank=1, motion="walk_forward_loop", category="Gestures")
    assert sonic_group(plain) == "Gestures"


def test_unmapped_category_raises_rather_than_binning():
    with pytest.raises(KeyError, match="no SONIC Table 2 group"):
        sonic_group(CorpusClip(rank=0, motion="x", category="Underwater Basketry"))


def test_absent_groups_are_dropped_and_the_rest_renormalized():
    shares = mixture_shares(
        SONIC_TABLE2_TEST_REPETITION,
        ["Locomotion", "Gestures", "Props", "Dance", "Injured", "ActionTool", "Others"],
    )
    # Acting (20 clips) and Combat (0) have no BONES-SEED counterpart.
    assert "Acting" not in shares and "Combat" not in shares
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["Locomotion"] == pytest.approx(2683 / 6286)


def test_selection_is_exact_size_deterministic_and_mixture_matched():
    clips = _corpus()
    ranks = select_proxy_ranks(clips, count=4096)
    assert len(ranks) == 4096
    assert len(set(ranks)) == 4096
    assert ranks == sorted(ranks)
    assert ranks == select_proxy_ranks(clips, count=4096)

    by_rank = {clip.rank: clip for clip in clips}
    counts: dict[str, int] = {}
    for rank in ranks:
        group = sonic_group(by_rank[rank])
        counts[group] = counts.get(group, 0) + 1
    shares = mixture_shares(SONIC_TABLE2_TEST_REPETITION, counts)
    for group, share in shares.items():
        # Quotas are rounded per group, so one clip of slack each way, plus the
        # rounding remainder that the largest group absorbs.
        assert abs(counts[group] - share * 4096) <= 2


def test_a_different_seed_draws_a_different_population():
    clips = _corpus()
    assert select_proxy_ranks(clips, count=512, seed=1) != select_proxy_ranks(
        clips, count=512, seed=2
    )


def test_frozen_board_is_registered_and_consistent():
    board = protocol.BOARDS["sonic_proxy_testrep4096_v1"]
    ranks = tuple(case.trajectory_rank for case in board.cases)
    assert ranks == protocol.SONIC_PROXY_TESTREP4096_RANKS
    assert len(set(ranks)) == 4096
    assert max(ranks) < 129785
    # Frame-0 starts, one repeat, seed 0 -- the clean-protocol contract.
    assert {case.start_frame for case in board.cases} == {0}
    assert {case.repeat_index for case in board.cases} == {0}

    clean = protocol.PROFILES["paper_sonic_proxy_testrep4096_v1"]
    robust = protocol.PROFILES["paper_sonic_proxy_testrep4096_robust_v1"]
    assert clean.protocol_id == "sonic_sr_clean_v1"
    assert robust.protocol_id == "sonic_sr_v1"
    assert clean.board_id == robust.board_id == "sonic_proxy_testrep4096_v1"


def test_loader_join_tolerates_a_missing_selection_record(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "reference_arrays_manifest.json").write_text(
        json.dumps(
            {
                "traj_info": {
                    "ordered_traj_list": [
                        ["src", "walk_forward_loop_001_A001", "trajectory_0"],
                        ["src", "orphan_clip_002_A002", "trajectory_0"],
                    ]
                }
            }
        )
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "motions": [
                    {
                        "filename": "walk_forward_loop_001__A001",
                        "category": "Basic Locomotion Neutral",
                    }
                ]
            }
        )
    )
    clips = load_corpus_clips(tree, selection)
    assert [clip.category for clip in clips] == ["Basic Locomotion Neutral", "Other"]
    assert [sonic_group(clip) for clip in clips] == ["Locomotion", "Others"]


def test_deployment_families_are_matched_once_and_in_order():
    from imitation_experiments.evaluation.sonic_paper_proxy import (
        SONIC_DEPLOYMENT_FAMILIES,
        deployment_family,
    )

    # `bow_saw_cutting_tree` is tool use, excluded from stage_bow.
    assert (
        deployment_family(CorpusClip(0, "bow_saw_cutting_tree_loop_R_002", "Household"))
        is None
    )
    assert deployment_family(CorpusClip(1, "knightly_bow_R_002_A429", "Gestures")) == (
        "stage_bow"
    )
    # A clip matching two families takes the first in declaration order, so a
    # quota cannot be inflated by double counting.
    both = CorpusClip(2, "dance_hiphop_crouch_step_001", "Dancing")
    assert deployment_family(both) == "hiphop_dance"
    assert list(SONIC_DEPLOYMENT_FAMILIES)[0] == "hiphop_dance"
    assert deployment_family(CorpusClip(3, "walk_forward_loop_001", "Gestures")) is None


def test_deployment_selection_is_exact_balanced_and_capped():
    from imitation_experiments.evaluation.sonic_paper_proxy import (
        deployment_family,
        select_deployment_ranks,
    )

    clips = []
    rank = 0
    # `grovel` deliberately smaller than an equal share, to exercise the
    # shortfall redistribution.
    sizes = {"dance_hiphop": 60, "high_jump": 60, "kick": 60, "grovel": 3}
    for stem, size in sizes.items():
        for index in range(size):
            clips.append(CorpusClip(rank, f"{stem}_{index}_A001", "Other"))
            rank += 1

    ranks = select_deployment_ranks(clips, count=123)
    assert len(ranks) == 123
    assert len(set(ranks)) == 123
    assert ranks == sorted(ranks)
    assert ranks == select_deployment_ranks(clips, count=123)

    by_rank = {clip.rank: clip for clip in clips}
    counts: dict[str, int] = {}
    for r in ranks:
        family = deployment_family(by_rank[r])
        counts[family] = counts.get(family, 0) + 1
    assert counts["grovel"] == 3  # capped at what the corpus holds
    assert sum(counts.values()) == 123
    # The shortfall lands on the families with headroom, evenly.
    roomy = ("hiphop_dance", "high_jump", "kick")
    assert max(counts[f] for f in roomy) - min(counts[f] for f in roomy) <= 1


def test_frozen_deployment_board_is_registered():
    board = protocol.BOARDS["sonic_deploy123_v1"]
    ranks = tuple(case.trajectory_rank for case in board.cases)
    assert ranks == protocol.SONIC_DEPLOY123_RANKS
    assert len(set(ranks)) == 123
    assert protocol.PROFILES["diag_sonic_deploy123_v1"].board_id == "sonic_deploy123_v1"
