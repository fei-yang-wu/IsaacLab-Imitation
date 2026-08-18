"""Tests for the EC/MuJoCo evaluation sidecar orchestrator.

The worker itself needs the Embodied-Control ``lowlevel-sim`` environment and
is exercised by the live smoke; here the worker's output rows are synthetic
fixtures, and the tests pin the parts that must not drift:

* ``to_summary`` emits the canonical schema the real
  :func:`imitation_experiments.reporting.records.load_summary` reads;
* the episode-status semantics (``reference_finished`` is a success,
  ``no_command`` is an artifact failure, only ``fell`` is a fall);
* the atomic claim protocol that keeps two overlapping sidecars from
  duplicating or clobbering work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.evaluation.ec_tracker_sidecar import (
    SidecarError,
    build_contract,
    claim_path,
    discover_checkpoints,
    to_summary,
    try_claim,
)
from imitation_experiments.evaluation.protocol import (
    BOARDS,
    PROFILES,
    PROTOCOLS,
    STRAT64_BUCKET_POPULATION,
    STRAT64_MOTIONS,
)
from imitation_experiments.reporting.records import load_summary

PROFILE = PROFILES["sidecar_ec_v1"]
PROTOCOL = PROTOCOLS[PROFILE.protocol_id]
BOARD = BOARDS[PROFILE.board_id]

CHECKPOINT_SHA = "c" * 64


def _episode_row(
    rank: int,
    *,
    status: str = "reference_finished",
    fell: bool = False,
    steps: int = 339,
    motion_length: int = 340,
    mpjpe_l: float = 25.0,
    mpjpe_g: float = 60.0,
) -> dict:
    frames = steps
    return {
        "trajectory_rank": rank,
        "motion_name": f"motion_{rank}",
        "start_frame": 0,
        "env_seed": 0,
        "repeat_index": 0,
        "status": status,
        "steps": steps,
        "motion_length": motion_length,
        "min_base_height_m": 0.55,
        "fell": fell,
        "termination": "base_too_low" if fell else "reference_finished",
        "success": not fell,
        "reference_finished": status == "reference_finished",
        "complete_motion": status == "reference_finished",
        "frames_scored": frames,
        "mpjpe_l_mm": mpjpe_l,
        "mpjpe_g_mm": mpjpe_g,
        "mpjpe_l_mm_p95": mpjpe_l * 2,
        "mpjpe_g_mm_p95": mpjpe_g * 2,
        "sonic": {"success": True, "complete_motion": True},
        "eval_seconds": 1.0,
    }


def _worker_result(rows: list[dict], *, noise: dict | None = None) -> dict:
    return {
        "schema_version": "ec_sidecar_worker_result_v1",
        "execution_mode": "sync_lockstep",
        "observation_noise": (
            {key: value for key, value in PROTOCOL.observation_noise}
            if noise is None
            else noise
        ),
        "episodes": rows,
        "runtime": {"playground_load_seconds": 2.0, "eval_seconds": 10.0},
    }


def _contract():
    return build_contract(
        checkpoint_sha=CHECKPOINT_SHA,
        facts={"cumulative_env_frames": 500_000_000},
        preset="fsq64_v2",
        manifest={"interface": "latent", "models": {}},
        reference_manifest_sha="d" * 64,
        task_id="Isaac-Imitation-G1-v2",
        algorithm="IPMD",
        protocol_tracked_bodies=PROTOCOL.tracked_body_names,
        job_config_sha="e" * 64,
    )


def _summary(rows: list[dict], *, noise: dict | None = None) -> dict:
    return to_summary(
        worker_result=_worker_result(rows, noise=noise),
        profile=PROFILE,
        contract=_contract(),
        checkpoint_path="logs/run/model_step_500000000.pt",
        checkpoint_sha=CHECKPOINT_SHA,
        bundle_dir=Path("/tmp/bundle"),
        bundle_manifest={"interface": "latent"},
        label="ec-sidecar test",
        runtime={"export_seconds": 30.0, "eval_seconds": 60.0, "cpu_count": 8},
    )


def test_summary_round_trips_through_the_real_reporting_reader(
    tmp_path: Path,
) -> None:
    rows = [_episode_row(rank) for rank in range(10)]
    rows[3] = _episode_row(3, status="fell", fell=True, steps=120, mpjpe_l=80.0)
    summary = _summary(rows)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))

    record = load_summary(summary_path, tmp_path)

    assert record.protocol_id == PROTOCOL.protocol_id
    assert record.protocol_hash == PROTOCOL.content_hash()
    assert record.protocol_backend == "ec_mujoco"
    assert record.board_id == BOARD.board_id
    assert record.board_hash == BOARD.content_hash()
    assert record.profile_id == PROFILE.profile_id
    assert record.contract_hash is not None
    assert record.tracker_sha256 == CHECKPOINT_SHA
    assert record.episode_count == 10
    assert record.motion_count == 10
    assert record.fall_free_rate == pytest.approx(0.9)
    # Headline = equal-episode mean of per-environment MPJPE-L.
    expected = (9 * 25.0 + 80.0) / 10
    assert record.mpjpe_mm == pytest.approx(expected)
    # Transition-weighted mean weights the short fallen episode less.
    assert record.mpjpe_mm_transition_weighted is not None
    assert record.mpjpe_mm_transition_weighted < expected
    # Successful-only drops the fallen episode entirely.
    assert record.mpjpe_mm_successful_only == pytest.approx(25.0)


def test_reference_finished_counts_as_success_and_only_fell_is_a_fall() -> None:
    rows = [_episode_row(rank) for rank in range(10)]
    rows[0] = _episode_row(0, status="completed", steps=1200, motion_length=2000)
    rows[0]["termination"] = "max_steps"
    rows[0]["reference_finished"] = False
    summary = _summary(rows)
    aggregate = summary["aggregate"]
    assert aggregate["fall_count"] == 0
    assert aggregate["fall_free_rate"] == 1.0
    finished = [row for row in summary["per_environment"] if row["reference_finished"]]
    assert len(finished) == 9


def test_stratified_board_reports_population_weighted_rates() -> None:
    """The screen over-samples hard motions; only the weighted rate is a
    population number, and it must be reconstructed from the board's weights."""
    profile = PROFILES["sidecar_ec_strat64_v1"]
    board = BOARDS[profile.board_id]
    failing_of = {rank: failing for rank, _, _, failing in STRAT64_MOTIONS}

    rows = []
    for case in board.cases:
        # Only the motions no arm failed succeed here, so the weighted success
        # rate must land on that bucket's share of the population.
        succeeds = failing_of[case.trajectory_rank] == 0
        row = _episode_row(case.trajectory_rank, fell=not succeeds)
        row["repeat_index"] = case.repeat_index
        row["sonic"] = {"success": succeeds, "complete_motion": succeeds}
        rows.append(row)

    summary = to_summary(
        worker_result=_worker_result(rows),
        profile=profile,
        contract=_contract(),
        checkpoint_path="logs/run/model_step_500000000.pt",
        checkpoint_sha=CHECKPOINT_SHA,
        bundle_dir=Path("/tmp/bundle"),
        bundle_manifest={"interface": "latent"},
        label="ec-sidecar strat64 test",
        runtime={"export_seconds": 30.0, "eval_seconds": 60.0, "cpu_count": 8},
    )
    aggregate = summary["aggregate"]
    easy_share = STRAT64_BUCKET_POPULATION[0] / sum(STRAT64_BUCKET_POPULATION.values())
    assert aggregate["population_weighted_sonic_success_rate"] == pytest.approx(
        easy_share
    )
    # The raw rate counts episodes, not population, and is far lower.
    assert aggregate["sonic_success_rate"] < easy_share / 2


def test_off_board_worker_row_fails_loudly() -> None:
    rows = [_episode_row(rank) for rank in range(10)]
    rows[3]["trajectory_rank"] = 4096
    with pytest.raises(SidecarError, match="not on the board"):
        _summary(rows)


def test_no_command_row_invalidates_the_checkpoint() -> None:
    rows = [_episode_row(rank) for rank in range(10)]
    rows[5] = {
        "trajectory_rank": 5,
        "motion_name": "motion_5",
        "start_frame": 0,
        "env_seed": 0,
        "repeat_index": 0,
        "status": "no_command",
        "steps": 40,
        "motion_length": 340,
        "min_base_height_m": 0.7,
        "artifact_failure": "tracker_received_no_command",
    }
    with pytest.raises(SidecarError, match="artifact failures"):
        _summary(rows)


def test_frame_count_skew_fails_loudly() -> None:
    rows = [_episode_row(rank) for rank in range(10)]
    rows[2]["frames_scored"] = rows[2]["steps"] - 5
    with pytest.raises(SidecarError, match="scored"):
        _summary(rows)


def test_summary_carries_sync_lockstep_and_uncertified_authority() -> None:
    summary = _summary([_episode_row(rank) for rank in range(10)])
    assert summary["authority_status"] == "uncertified"
    assert summary["realized_protocol"]["execution_mode"] == "sync_lockstep"
    assert summary["disable_push_event"] is True


def test_rehearsal_protocol_injects_sonic_training_noise() -> None:
    """A hardware rehearsal must not silently run on perfect sensor readings."""
    assert PROTOCOL.observation_corruption is True
    assert dict(PROTOCOL.observation_noise) == {
        "projected_gravity": 0.05,
        "base_ang_vel": 0.2,
        "joint_pos": 0.01,
        "joint_vel": 0.5,
    }
    summary = _summary([_episode_row(rank) for rank in range(10)])
    realized = summary["realized_protocol"]
    assert realized["observation_corruption"] is True
    assert realized["observation_noise"]["joint_vel"] == 0.5


def test_noise_free_run_cannot_be_reported_under_the_rehearsal_protocol() -> None:
    """The realized-versus-requested guard, on the field that matters most."""
    rows = [_episode_row(rank) for rank in range(10)]
    with pytest.raises(SidecarError, match="observation noise mismatch"):
        _summary(rows, noise={})


def test_noise_magnitudes_are_part_of_the_protocol_identity() -> None:
    """Two noise levels must never share a protocol hash."""
    from dataclasses import replace

    quieter = replace(
        PROTOCOL, observation_noise=(("base_ang_vel", 0.1), ("joint_pos", 0.01))
    )
    assert quieter.content_hash() != PROTOCOL.content_hash()


def test_claim_is_exclusive_and_skips_completed_work(tmp_path: Path) -> None:
    claim = claim_path(tmp_path, CHECKPOINT_SHA, "p" * 64)
    assert try_claim(claim, stale_minutes=45.0, payload={})
    # A second contender loses while the claim is fresh.
    assert not try_claim(claim, stale_minutes=45.0, payload={})
    claim.unlink()
    # Finished work is never re-claimed even without a live claim.
    claim.with_name("summary.json").write_text("{}")
    assert not try_claim(claim, stale_minutes=45.0, payload={})


def test_stale_claim_is_reclaimed(tmp_path: Path) -> None:
    claim = claim_path(tmp_path, CHECKPOINT_SHA, "p" * 64)
    assert try_claim(claim, stale_minutes=45.0, payload={})
    import os

    old = claim.stat().st_mtime - 60 * 60
    os.utime(claim, (old, old))
    assert try_claim(claim, stale_minutes=45.0, payload={})


def test_fallen_episode_is_scored_not_rejected() -> None:
    """A fall is a measurement, not an artifact failure.

    Under sensor noise falls are common, and a fallen episode legitimately
    stops short of the reference end. The completeness check must fire only
    for episodes claiming ``reference_finished``.
    """
    rows = [_episode_row(rank) for rank in range(10)]
    rows[4] = _episode_row(
        4, status="fell", fell=True, steps=60, motion_length=800, mpjpe_l=95.0
    )
    rows[4]["reference_finished"] = False
    rows[4]["complete_motion"] = False

    summary = _summary(rows)

    assert summary["aggregate"]["fall_count"] == 1
    fallen = summary["per_environment"][4]
    assert fallen["fell"] is True
    assert fallen["termination_terms"] == ["base_too_low"]
    assert fallen["tracking_metrics"]["tracking_mpjpe_mm"] == 95.0


def test_five_repeats_per_motion_stay_balanced_and_carry_repeat_index(
    tmp_path: Path,
) -> None:
    """The rehearsal board is 10 motions x 5 noise draws."""
    rows = [
        _episode_row(rank, mpjpe_l=20.0 + repeat)
        for rank in range(10)
        for repeat in range(5)
    ]
    for index, row in enumerate(rows):
        row["repeat_index"] = index % 5
    summary = _summary(rows)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))

    record = load_summary(summary_path, tmp_path)

    assert record.episode_count == 50
    assert record.motion_count == 10
    assert record.is_balanced
    assert record.episodes_per_motion == 5
    assert {episode.repeat_index for episode in record.episodes} == {0, 1, 2, 3, 4}


def test_incomplete_reference_finished_run_is_rejected() -> None:
    """Silent truncation would flatter the arm exactly where it is weakest."""
    rows = [_episode_row(rank) for rank in range(10)]
    rows[6] = _episode_row(6, steps=100, motion_length=800)
    rows[6]["reference_finished"] = True

    with pytest.raises(SidecarError, match="reference_finished"):
        _summary(rows)


def test_load_checkpoint_facts_rejects_a_checkpoint_without_frames(
    tmp_path: Path,
) -> None:
    """`cumulative_env_frames` is the x-axis every eval is plotted against."""
    import torch

    from imitation_experiments.evaluation.ec_tracker_sidecar import (
        load_checkpoint_facts,
    )

    path = tmp_path / "model_step_10.pt"
    torch.save({"policy": {}}, path)

    with pytest.raises(SidecarError, match="cumulative_env_frames"):
        load_checkpoint_facts(path)


def test_stage_checkpoint_is_content_addressed_and_verified(tmp_path: Path) -> None:
    from imitation_experiments.evaluation.ec_tracker_sidecar import stage_checkpoint

    source = tmp_path / "model_step_10.pt"
    source.write_bytes(b"weights")
    staging = tmp_path / "staging"

    staged, sha = stage_checkpoint(source, staging)

    assert staged.name == f"{sha}.pt"
    assert staged.read_bytes() == b"weights"
    # A second call reuses the content-addressed copy rather than recopying.
    again, sha_again = stage_checkpoint(source, staging)
    assert (again, sha_again) == (staged, sha)


def test_wandb_payload_is_keyed_on_cumulative_frames() -> None:
    """A sidecar point must land on the training run's own x-axis.

    Segment-local step counters restart on every chained segment, so anything
    but `cumulative_env_frames` would fold four segments onto each other.
    """
    from imitation_experiments.evaluation.ec_tracker_sidecar import wandb_payload

    summary = _summary([_episode_row(rank) for rank in range(10)])
    payload, frames = wandb_payload(summary)

    assert frames == 500_000_000
    assert payload["Eval/cumulative_env_frames"] == 500_000_000
    assert payload["Eval/mpjpe_l_mm"] == pytest.approx(25.0)
    assert payload["Eval/fall_free_rate"] == pytest.approx(1.0)
    assert payload["Eval/mpjpe_l_mm_successful"] == pytest.approx(25.0)


def test_wandb_payload_drops_absent_fields() -> None:
    """A bundle with no frame count still publishes its metrics."""
    from imitation_experiments.evaluation.ec_tracker_sidecar import wandb_payload

    summary = _summary([_episode_row(rank) for rank in range(10)])
    summary["contract"] = dict(summary["contract"])
    summary["contract"]["cumulative_env_frames"] = None
    summary["successful_trajectory_metrics"] = {"tracking_mpjpe_mm": {}}

    payload, frames = wandb_payload(summary)

    assert frames is None
    assert "Eval/cumulative_env_frames" not in payload
    assert "Eval/mpjpe_l_mm_successful" not in payload
    assert payload["Eval/mpjpe_l_mm"] == pytest.approx(25.0)


def test_training_run_id_comes_from_the_checkpoint_path(tmp_path: Path) -> None:
    """Attaching to the right run must not depend on a name or a registry.

    RLOpt nests checkpoints under `<timestamp>_wandb-<run_id>`, so the run that
    produced a checkpoint is knowable from the file itself.
    """
    from imitation_experiments.evaluation.ec_tracker_sidecar import (
        infer_training_run_id,
    )

    checkpoint = (
        tmp_path
        / "arm/tracker/2026-08-17_14-43-48_wandb-nfcqoq1s/models/model_step_10.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")

    assert infer_training_run_id(checkpoint) == "nfcqoq1s"
    assert infer_training_run_id(tmp_path / "loose/model_step_10.pt") is None


def test_skill_checkpoint_is_discovered_from_the_arm_tree(tmp_path: Path) -> None:
    """FSQ export dies without the encoder, so the pairing must be automatic."""
    from imitation_experiments.evaluation.ec_tracker_sidecar import (
        resolve_skill_checkpoint,
    )

    arm = tmp_path / "fsq64_hold10_seed0"
    encoder = arm / "encoder" / "checkpoints" / "latest.pt"
    encoder.parent.mkdir(parents=True)
    encoder.write_bytes(b"encoder")
    tracker = arm / "tracker" / "run" / "models" / "model_step_10.pt"
    tracker.parent.mkdir(parents=True)
    tracker.write_bytes(b"tracker")

    assert resolve_skill_checkpoint(tracker) == encoder.resolve()
    # No arm tree above the checkpoint: report nothing rather than guess, and
    # let the exporter's own FSQ error name the missing input.
    assert resolve_skill_checkpoint(tmp_path / "loose" / "model_step_10.pt") is None


def test_explicit_skill_checkpoint_wins_and_must_exist(tmp_path: Path) -> None:
    from imitation_experiments.evaluation.ec_tracker_sidecar import (
        resolve_skill_checkpoint,
    )

    arm = tmp_path / "arm"
    discovered = arm / "encoder" / "checkpoints" / "latest.pt"
    discovered.parent.mkdir(parents=True)
    discovered.write_bytes(b"encoder")
    tracker = arm / "tracker" / "run" / "models" / "model_step_10.pt"
    tracker.parent.mkdir(parents=True)
    tracker.write_bytes(b"tracker")
    explicit = tmp_path / "chosen.pt"
    explicit.write_bytes(b"chosen")

    assert resolve_skill_checkpoint(tracker, explicit) == explicit.resolve()

    with pytest.raises(SidecarError, match="skill checkpoint not found"):
        resolve_skill_checkpoint(tracker, tmp_path / "missing.pt")


def test_export_command_carries_the_skill_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FSQ lattice lives only in the skill checkpoint's config."""
    import subprocess

    from imitation_experiments.evaluation import ec_tracker_sidecar

    recorded: dict = {}

    def fake_run(command, **kwargs):  # noqa: ANN001 - test double
        recorded["command"] = command
        bundle = Path(command[command.index("--output") + 1])
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text(json.dumps({"interface": "latent"}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ec_tracker_sidecar.subprocess, "run", fake_run)

    ec_tracker_sidecar.export_bundle(
        tmp_path / "staged.pt",
        preset="fsq64_v2",
        bundle_dir=tmp_path / "bundle",
        pixi_bin="pixi",
        timeout_s=60.0,
        skill_checkpoint=tmp_path / "encoder.pt",
    )

    command = recorded["command"]
    assert "--skill-checkpoint" in command
    assert command[command.index("--skill-checkpoint") + 1] == str(
        tmp_path / "encoder.pt"
    )


def test_discover_checkpoints_orders_by_step_and_skips_partials(
    tmp_path: Path,
) -> None:
    (tmp_path / "model_step_1000.pt").write_bytes(b"x")
    (tmp_path / "model_step_200.pt").write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "model_step_30000.pt").write_bytes(b"x")
    (tmp_path / "model_step_500.pt.partial").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")

    found = discover_checkpoints(tmp_path)
    assert [path.name for path in found] == [
        "model_step_200.pt",
        "model_step_1000.pt",
        "model_step_30000.pt",
    ]
