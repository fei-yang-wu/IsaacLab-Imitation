"""Tests for paper protocol metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

from paper_protocol_metadata import (
    disable_domain_randomization,
    interval_event_metadata,
)


def test_interval_event_metadata_records_push_without_mutation() -> None:
    velocity_range = {"x": (-0.5, 0.5), "yaw": (-0.78, 0.78)}
    term = SimpleNamespace(
        func=interval_event_metadata,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": velocity_range},
    )
    env_cfg = SimpleNamespace(events=SimpleNamespace(push_robot=term))

    result = interval_event_metadata(env_cfg, "push_robot")

    assert result == {
        "enabled": True,
        "term_name": "push_robot",
        "mode": "interval",
        "interval_range_s": [1.0, 3.0],
        "function": ("paper_protocol_metadata.interval_event_metadata"),
        "velocity_range": {"x": [-0.5, 0.5], "yaw": [-0.78, 0.78]},
    }
    assert term.params["velocity_range"] is velocity_range


def test_interval_event_metadata_records_disabled_event() -> None:
    result = interval_event_metadata(SimpleNamespace(events=None), "push_robot")

    assert result == {"enabled": False, "term_name": "push_robot"}


def _g1_like_events() -> SimpleNamespace:
    """Mirror the shape of G1EventCfg closely enough to exercise every branch."""
    return SimpleNamespace(
        reset_reference_state=SimpleNamespace(
            params={
                "pose_range": {"x": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
                "velocity_range": {"x": (-0.5, 0.5)},
                "joint_position_range": (-0.1, 0.1),
            }
        ),
        add_joint_default_pos=SimpleNamespace(
            params={"pos_distribution_params": (-0.01, 0.01)}
        ),
        physics_material=SimpleNamespace(params={"static_friction_range": (0.3, 1.6)}),
        base_com=SimpleNamespace(params={"com_range": {"x": (-0.025, 0.025)}}),
        randomize_rigid_body_mass=SimpleNamespace(
            params={"mass_distribution_params": (0.8, 2.5)}
        ),
        push_robot=SimpleNamespace(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": {"x": (-0.5, 0.5)}},
        ),
    )


def test_disable_domain_randomization_neutralizes_the_reset_perturbation() -> None:
    events = _g1_like_events()
    env_cfg = SimpleNamespace(events=events)

    record = disable_domain_randomization(env_cfg)

    reset_params = events.reset_reference_state.params
    assert reset_params["pose_range"] == {}
    assert reset_params["velocity_range"] == {}
    assert reset_params["joint_position_range"] == (0.0, 0.0)
    assert events.add_joint_default_pos.params["pos_distribution_params"] is None
    assert record["enabled"] is True
    assert record["reset_ranges_zeroed"]["reset_reference_state"] == [
        "joint_position_range",
        "pose_range",
        "velocity_range",
    ]
    assert record["noise_params_disabled"] == [
        "add_joint_default_pos.pos_distribution_params"
    ]


def test_disable_domain_randomization_removes_pushes_and_startup_events() -> None:
    events = _g1_like_events()
    env_cfg = SimpleNamespace(events=events)

    record = disable_domain_randomization(env_cfg)

    for term_name in (
        "push_robot",
        "physics_material",
        "base_com",
        "randomize_rigid_body_mass",
    ):
        assert getattr(events, term_name) is None, term_name
    assert record["events_disabled"] == [
        "base_com",
        "physics_material",
        "push_robot",
        "randomize_rigid_body_mass",
    ]
    # The recorded protocol must agree with the config it produced.
    assert interval_event_metadata(env_cfg, "push_robot") == {
        "enabled": False,
        "term_name": "push_robot",
    }


def test_disable_domain_randomization_is_idempotent() -> None:
    """Re-applying must not error on already-removed events."""
    env_cfg = SimpleNamespace(events=_g1_like_events())

    disable_domain_randomization(env_cfg)
    second = disable_domain_randomization(env_cfg)

    assert second["events_disabled"] == []


def test_disable_domain_randomization_tolerates_no_events() -> None:
    record = disable_domain_randomization(SimpleNamespace(events=None))

    assert record == {
        "enabled": True,
        "reset_ranges_zeroed": {},
        "noise_params_disabled": [],
        "events_disabled": [],
    }
