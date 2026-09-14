"""Tests for the Vega-Wuji promotion CLI's evidence extraction.

The attestation logic is tested in
``source/imitation_experiments/tests/test_vega_wuji_promotion.py``. These
tests cover what this script adds on top: refusing to describe contact
provenance that does not exist, and rebuilding the collision-clearance record
from the Reference's own stored audit rather than from an assumption.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from promote_vega_wuji_reference import _collision_clearance, _contact_provenance


def _reference(metadata: dict, *, active: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        metadata=metadata,
        contacts=SimpleNamespace(
            active=np.asarray([[[active]]], dtype=bool),
        ),
        frame_count=27,
    )


def _recovered_metadata(**overrides) -> dict:
    recovery = {
        "status": "recovered",
        "recovery_rate": 0.81,
        "per_side_recovery_rate": {"left": 0.78, "right": 0.84},
        "minimum_distance_m": -0.0004,
    }
    recovery.update(overrides)
    return {
        "contacts": {"contact_recovery": recovery},
        "geometry_clearance_audit": {
            "robot_self_collision_20hz": {"minimum_signed_distance_m": -0.0009},
        },
    }


def test_contact_provenance_records_the_measured_recovery() -> None:
    provenance = _contact_provenance(_reference(_recovered_metadata()))
    assert "SOMA-X mesh reconstruction" in provenance
    assert "0.78" in provenance and "0.84" in provenance


def test_contact_provenance_refuses_a_blocked_recovery() -> None:
    metadata = _recovered_metadata(status="blocked")
    with pytest.raises(ValueError, match="no recovered robot contact geometry"):
        _contact_provenance(_reference(metadata))


def test_contact_provenance_refuses_an_inspection_reference() -> None:
    """An inspection Reference has named but inactive contact slots."""

    metadata = {"contacts": {}, "geometry_clearance_audit": {}}
    with pytest.raises(ValueError, match="no recovered robot contact geometry"):
        _contact_provenance(_reference(metadata))


def test_contact_provenance_refuses_all_inactive_contacts() -> None:
    with pytest.raises(ValueError, match="no active frames"):
        _contact_provenance(_reference(_recovered_metadata(), active=False))


def test_collision_clearance_takes_the_worst_stored_distance() -> None:
    record = _collision_clearance(_reference(_recovered_metadata()), 27)
    assert record.qualified is True
    assert record.checked_frame_count == 27
    # -0.0009 self-collision versus -0.0004 contact recovery.
    assert record.minimum_signed_distance_m == pytest.approx(-0.0009)
    assert record.penetration_tolerance_m == pytest.approx(0.001)


def test_collision_clearance_refuses_without_an_audit() -> None:
    with pytest.raises(ValueError, match="no geometry clearance audit"):
        _collision_clearance(_reference({"contacts": {}}), 27)


def test_collision_clearance_refuses_without_signed_distance_evidence() -> None:
    metadata = {
        "contacts": {"contact_recovery": {"status": "recovered"}},
        "geometry_clearance_audit": {"robot_self_collision_20hz": {}},
    }
    with pytest.raises(ValueError, match="no signed-distance evidence"):
        _collision_clearance(_reference(metadata), 27)
