"""Tests for the Vega-Wuji runtime-promotion attestation.

The attestation exists to stop an offline-fitted Reference from reaching
training without Newton evidence, so the tests that matter are the refusals:
a failed replay, a short replay, re-engaged object assistance, a motion that
was edited after promotion, and a swapped robot model must all be rejected.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from imitation_experiments.audit.vega_wuji_promotion import (
    ATTESTATION_SCHEMA_VERSION,
    STATE_LOCK_MODE,
    UNASSISTED_MODE,
    attestation_path_for_manifest,
    build_attestation,
    canonical_qpos_sha256,
    load_attestation,
    verify_attestation_for_reference,
)


MODEL_SHA = "a" * 64


def _state_lock_payload(**overrides):
    payload = {
        "mode": STATE_LOCK_MODE,
        "task": "Isaac-Imitation-Vega-Wuji-v0",
        "physics": "newton_mjwarp",
        "selected_motion": "/tmp/ref.npz",
        "full_reference_horizon_requested": True,
        "full_reference_horizon_completed": True,
        "completed_replay_steps": 26,
        "termination_step": None,
        "fired_termination_terms": [],
        "state_lock": {
            "passed": True,
            "max_joint_position_error_rad": 1.0e-6,
            "max_object_position_error_m": 2.0e-6,
        },
    }
    payload.update(overrides)
    return payload


def _unassisted_payload(**overrides):
    payload = {
        "mode": UNASSISTED_MODE,
        "task": "Isaac-Imitation-Vega-Wuji-v0",
        "physics": "newton_mjwarp",
        "selected_motion": "/tmp/ref.npz",
        "full_reference_horizon_requested": True,
        "full_reference_horizon_completed": True,
        "completed_without_termination": True,
        "zero_virtual_object_controller_verified": True,
        "assistance_violation_step": None,
        "completed_replay_steps": 27,
        "termination_step": None,
        "unassisted_dynamics": {
            "max_object_position_error_m": 0.01,
            "per_step": [{"joint_position_mae_rad": 0.001}],
        },
    }
    payload.update(overrides)
    return payload


def _motion(qpos, **overrides):
    entry = {
        "sequence_id": "snack_box",
        "qpos_sha256": canonical_qpos_sha256(qpos),
        "state_lock": _state_lock_payload(),
        "unassisted": _unassisted_payload(),
    }
    entry.update(overrides)
    return entry


QPOS = np.linspace(0.0, 1.0, 27 * 4, dtype=np.float64).reshape(27, 4)


def test_build_and_verify_round_trip() -> None:
    attestation = build_attestation(
        motions=[_motion(QPOS)], model_sha256=MODEL_SHA, tool="test"
    )
    assert attestation["schema_version"] == ATTESTATION_SCHEMA_VERSION
    record = verify_attestation_for_reference(
        attestation, sequence_id="snack_box", qpos=QPOS, model_sha256=MODEL_SHA
    )
    assert record["sequence_id"] == "snack_box"
    # The bulky per-step table must not be copied into the attestation.
    assert "per_step" not in record["unassisted"]["unassisted_dynamics"]


def test_failed_state_lock_cannot_be_promoted() -> None:
    motion = _motion(QPOS)
    motion["state_lock"] = _state_lock_payload(
        state_lock={"passed": False, "max_joint_position_error_rad": 0.4}
    )
    with pytest.raises(ValueError, match="did not pass its tolerances"):
        build_attestation(motions=[motion], model_sha256=MODEL_SHA, tool="test")


def test_partial_horizon_replay_cannot_be_promoted() -> None:
    motion = _motion(QPOS)
    motion["unassisted"] = _unassisted_payload(
        full_reference_horizon_completed=False,
        termination_step=11,
        fired_termination_terms=["wrist_away_from_trajectory"],
    )
    with pytest.raises(ValueError, match="did not complete the full Reference"):
        build_attestation(motions=[motion], model_sha256=MODEL_SHA, tool="test")


def test_reengaged_object_assistance_cannot_be_promoted() -> None:
    motion = _motion(QPOS)
    motion["unassisted"] = _unassisted_payload(assistance_violation_step=7)
    with pytest.raises(ValueError, match="assistance re-engaged"):
        build_attestation(motions=[motion], model_sha256=MODEL_SHA, tool="test")


def test_nonzero_virtual_controller_cannot_be_promoted() -> None:
    motion = _motion(QPOS)
    motion["unassisted"] = _unassisted_payload(
        zero_virtual_object_controller_verified=False
    )
    with pytest.raises(ValueError, match="not verified to be zero"):
        build_attestation(motions=[motion], model_sha256=MODEL_SHA, tool="test")


def test_swapped_replay_modes_are_rejected() -> None:
    motion = _motion(QPOS)
    motion["state_lock"] = _unassisted_payload()
    with pytest.raises(ValueError, match=f"expected a '{STATE_LOCK_MODE}'"):
        build_attestation(motions=[motion], model_sha256=MODEL_SHA, tool="test")


def test_edited_motion_is_rejected_at_verify() -> None:
    """Promotion must not transfer to a Reference whose motion changed."""

    attestation = build_attestation(
        motions=[_motion(QPOS)], model_sha256=MODEL_SHA, tool="test"
    )
    edited = QPOS.copy()
    edited[3, 1] += 1.0e-3
    with pytest.raises(ValueError, match="does not match its promotion record"):
        verify_attestation_for_reference(
            attestation, sequence_id="snack_box", qpos=edited, model_sha256=MODEL_SHA
        )


def test_swapped_model_is_rejected_at_verify() -> None:
    attestation = build_attestation(
        motions=[_motion(QPOS)], model_sha256=MODEL_SHA, tool="test"
    )
    with pytest.raises(ValueError, match="was promoted against model"):
        verify_attestation_for_reference(
            attestation, sequence_id="snack_box", qpos=QPOS, model_sha256="b" * 64
        )


def test_uncovered_motion_is_rejected_at_verify() -> None:
    attestation = build_attestation(
        motions=[_motion(QPOS)], model_sha256=MODEL_SHA, tool="test"
    )
    with pytest.raises(ValueError, match="no runtime-promotion record"):
        verify_attestation_for_reference(
            attestation, sequence_id="other", qpos=QPOS, model_sha256=MODEL_SHA
        )


def test_hand_edited_attestation_cannot_flip_a_failure() -> None:
    """Verification re-checks the evidence, not just the hashes."""

    attestation = build_attestation(
        motions=[_motion(QPOS)], model_sha256=MODEL_SHA, tool="test"
    )
    attestation["motions"][0]["state_lock"]["state_lock"]["passed"] = False
    with pytest.raises(ValueError, match="failing state-lock record"):
        verify_attestation_for_reference(
            attestation, sequence_id="snack_box", qpos=QPOS, model_sha256=MODEL_SHA
        )


def test_duplicate_motions_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than once"):
        build_attestation(
            motions=[_motion(QPOS), _motion(QPOS)],
            model_sha256=MODEL_SHA,
            tool="test",
        )


def test_load_rejects_a_foreign_schema(tmp_path) -> None:
    path = tmp_path / "m.json.promotion.json"
    path.write_text(json.dumps({"schema_version": "other/v9", "motions": [{}]}))
    with pytest.raises(ValueError, match="unsupported attestation schema"):
        load_attestation(path)


def test_load_rejects_an_empty_attestation(tmp_path) -> None:
    path = tmp_path / "m.json.promotion.json"
    path.write_text(
        json.dumps({"schema_version": ATTESTATION_SCHEMA_VERSION, "motions": []})
    )
    with pytest.raises(ValueError, match="covers no motions"):
        load_attestation(path)


def test_attestation_path_is_derived_from_the_manifest() -> None:
    assert str(attestation_path_for_manifest("/a/b/m.json")).endswith(
        "m.json.promotion.json"
    )


def test_qpos_digest_is_packaging_independent() -> None:
    """Re-saving an NPZ must not change the motion identity."""

    assert canonical_qpos_sha256(QPOS) == canonical_qpos_sha256(
        np.asarray(QPOS, dtype=np.float32).astype(np.float64)
    )
    assert canonical_qpos_sha256(QPOS) != canonical_qpos_sha256(QPOS[:-1])
