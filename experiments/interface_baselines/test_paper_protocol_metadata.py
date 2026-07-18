"""Tests for paper protocol metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace

from paper_protocol_metadata import interval_event_metadata


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
