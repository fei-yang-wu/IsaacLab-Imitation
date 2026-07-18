"""Small metadata helpers shared by paper-facing evaluation entrypoints."""

from __future__ import annotations

from typing import Any


def interval_event_metadata(env_cfg: Any, term_name: str) -> dict[str, Any]:
    """Describe one configured interval event without changing it."""
    events = getattr(env_cfg, "events", None)
    term = getattr(events, term_name, None) if events is not None else None
    if term is None:
        return {"enabled": False, "term_name": term_name}
    function = getattr(term, "func", None)
    params = getattr(term, "params", {}) or {}
    velocity_range = params.get("velocity_range")
    return {
        "enabled": True,
        "term_name": term_name,
        "mode": str(getattr(term, "mode", "")),
        "interval_range_s": list(getattr(term, "interval_range_s", ()) or ()),
        "function": (
            f"{getattr(function, '__module__', '')}."
            f"{getattr(function, '__qualname__', getattr(function, '__name__', ''))}"
        ).strip("."),
        "velocity_range": {
            str(axis): [float(bounds[0]), float(bounds[1])]
            for axis, bounds in sorted((velocity_range or {}).items())
        },
    }
