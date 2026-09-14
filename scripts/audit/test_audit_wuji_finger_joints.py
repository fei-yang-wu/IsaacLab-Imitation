"""Focused tests for fail-closed stored Wuji finger evidence."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from audit_wuji_finger_joints import (
    _canonical_float32_array_sha256,
    _embedded_evidence_binding,
    _embedded_gate_measurements,
)


def test_embedded_gate_measurements_are_read_from_final_fk_record() -> None:
    metadata = {
        "finger_mapping": {
            "final_acceptance": {
                "sides": {
                    "left": {
                        "wrist_mean_mm": 1.25,
                        "fingertip_residuals": {"tip_p95_mm": 3.5},
                    }
                }
            }
        }
    }

    assert _embedded_gate_measurements(metadata, "left") == (3.5, 1.25)
    assert _embedded_gate_measurements(metadata, "right") == (None, None)


def test_embedded_gate_measurements_reject_corrupt_evidence() -> None:
    metadata = {
        "finger_mapping": {
            "final_acceptance": {
                "sides": {
                    "left": {
                        "wrist_mean_mm": 1.25,
                        "fingertip_residuals": {"tip_p95_mm": float("nan")},
                    }
                }
            }
        }
    }

    with pytest.raises(ValueError, match="invalid"):
        _embedded_gate_measurements(metadata, "left")


def test_embedded_evidence_binding_rejects_changed_qpos(tmp_path) -> None:
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>", encoding="utf-8")
    qpos = np.zeros((2, 1), dtype=np.float32)
    joint_names = ("joint",)
    metadata = {
        "finger_mapping": {
            "final_acceptance": {
                "evidence_binding": {
                    "schema": "isaaclab_imitation_wuji_final_fk_evidence/v1",
                    "qpos_float32_sha256": _canonical_float32_array_sha256(qpos),
                    "fingertip_targets_float32_sha256": "1" * 64,
                    "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    "joint_names_sha256": hashlib.sha256(
                        json.dumps(list(joint_names), separators=(",", ":")).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                }
            }
        }
    }

    assert _embedded_evidence_binding(
        metadata, qpos=qpos, joint_names=joint_names, model_path=model
    ) == (True, "verified")
    changed = qpos.copy()
    changed[1, 0] = 0.1
    bound, reason = _embedded_evidence_binding(
        metadata, qpos=changed, joint_names=joint_names, model_path=model
    )
    assert bound is False
    assert "qpos_float32_sha256" in reason
