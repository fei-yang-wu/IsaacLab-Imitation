"""Evaluation identity and evaluator-capability contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from imitation_experiments.evaluation.protocol import (
    BOARDS,
    PROTOCOLS,
    STRAT64_BUCKET_POPULATION,
    STRAT64_MOTIONS,
    EvalBoardV1,
    EvalEpisodeCaseV1,
    ProtocolMismatch,
    ProtocolUnsupported,
    eval_skill_commander_argv,
    evaluate_checkpoint_argv,
    validate_realized_protocol,
)


def test_protocol_hash_is_stable_and_semantic() -> None:
    protocol = PROTOCOLS["tracker_fall_only_v1"]
    assert protocol.content_hash() == protocol.content_hash()
    assert (
        replace(protocol, description="new prose").content_hash()
        == protocol.content_hash()
    )
    assert (
        replace(protocol, outer_safety_cap_steps=9999).content_hash()
        != protocol.content_hash()
    )


def test_board_identity_uses_episode_keys_not_human_motion_label() -> None:
    first = EvalBoardV1(
        board_id="fixture",
        cases=(EvalEpisodeCaseV1(7, 0, 3, motion_name="old name"),),
    )
    renamed = EvalBoardV1(
        board_id="fixture",
        cases=(EvalEpisodeCaseV1(7, 0, 3, motion_name="new name"),),
    )
    assert first.content_hash() == renamed.content_hash()


def test_stratified_board_weights_restore_the_population_shares() -> None:
    board = BOARDS["ec_strat64_v1"]
    repeats = {case.repeat_index for case in board.cases}
    motions = {case.trajectory_rank for case in board.cases}
    assert len(motions) == 64
    assert len(board.cases) == 64 * len(repeats)

    # One repeat of the whole board is one population: the weights of a single
    # repeat sum to 1, and each difficulty bucket's weight equals its share of
    # the 4,096 scoreboard motions.
    single = [case for case in board.cases if case.repeat_index == 0]
    assert sum(case.population_weight for case in single) == pytest.approx(1.0)

    total = sum(STRAT64_BUCKET_POPULATION.values())
    failing_of = {rank: failing for rank, _, _, failing in STRAT64_MOTIONS}
    per_bucket: dict[int, float] = {}
    for case in single:
        bucket = failing_of[case.trajectory_rank]
        per_bucket[bucket] = per_bucket.get(bucket, 0.0) + case.population_weight
    for bucket, weight in per_bucket.items():
        assert weight == pytest.approx(STRAT64_BUCKET_POPULATION[bucket] / total)


def test_board_weights_are_part_of_board_identity() -> None:
    plain = EvalBoardV1(board_id="fixture", cases=(EvalEpisodeCaseV1(7, 0, 3),))
    weighted = EvalBoardV1(
        board_id="fixture",
        cases=(EvalEpisodeCaseV1(7, 0, 3, population_weight=0.25),),
    )
    assert plain.content_hash() != weighted.content_hash()


def test_milestone_board_is_strict_subset_of_final_board() -> None:
    milestone = {case.identity() for case in BOARDS["bones_milestone256_v1"].cases}
    final = {case.identity() for case in BOARDS["bones_scoreboard4096_v1"].cases}
    assert milestone < final


def test_evaluate_checkpoint_adapter_emits_fall_only_contract() -> None:
    args = evaluate_checkpoint_argv(
        PROTOCOLS["tracker_fall_only_v1"],
        BOARDS["bones_milestone256_v1"],
        checkpoint="model.pt",
        output_json="summary.json",
    )
    assert args[:4] == ["--checkpoint", "model.pt", "--output_json", "summary.json"]
    assert "--randomization" in args
    assert "no_push" in args
    assert "env.terminations.foot_pos_xyz=null" in args
    assert "env.terminations.base_too_low=null" not in args
    assert args[-1] != "--disable_early_terminations"


def test_skill_commander_adapter_emits_gr00t_2000_step_contract() -> None:
    args = eval_skill_commander_argv(
        PROTOCOLS["gr00t_planner_v1"],
        BOARDS["gr00t28x20_v1"],
        checkpoint="tracker.pt",
        output_dir="eval",
    )
    assert args[args.index("--max_steps") + 1] == "2000"
    assert args[args.index("--metric_interval") + 1] == "10"
    assert "--fall_only_success" in args
    assert "--disable_tracking_terminations" in args


def test_unsupported_pairing_names_the_missing_field() -> None:
    protocol = replace(PROTOCOLS["tracker_fall_only_v1"], metric_interval=10)
    with pytest.raises(ProtocolUnsupported, match="metric_interval"):
        evaluate_checkpoint_argv(
            protocol,
            BOARDS["bones_milestone256_v1"],
            checkpoint="model.pt",
            output_json="summary.json",
        )


def test_requested_realized_mismatch_names_only_differing_fields() -> None:
    requested = PROTOCOLS["tracker_fall_only_v1"]
    realized = replace(requested, randomization_profile="all")
    with pytest.raises(ProtocolMismatch) as error:
        validate_realized_protocol(requested, realized)
    assert set(error.value.differences) == {"randomization_profile"}
