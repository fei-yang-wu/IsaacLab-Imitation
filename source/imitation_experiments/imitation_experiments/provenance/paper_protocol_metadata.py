"""Small metadata helpers shared by paper-facing evaluation entrypoints."""

from __future__ import annotations

from typing import Any


# Reset-event range parameters and their "no perturbation" value. Empty dicts
# and a zero-width tuple are how the reset event spells "no offset"; it does
# not accept None for these.
_RESET_RANGE_NEUTRAL: dict[str, Any] = {
    "pose_range": {},
    "velocity_range": {},
    "joint_position_range": (0.0, 0.0),
}

# Randomization parameters that document None as "no randomization".
_NULLABLE_NOISE_PARAMS = ("pos_distribution_params",)

# Randomization events removed wholesale, because their range parameters have
# no neutral value to substitute.
_RANDOMIZATION_EVENTS = (
    "push_robot",
    "physics_material",
    "base_com",
    "randomize_rigid_body_mass",
)


def disable_domain_randomization(env_cfg: Any) -> dict[str, Any]:
    """Turn an env config into a deterministic tracking-fidelity measurement.

    Root-relative MPJPE subtracts root position but not root orientation, so
    the reset event's roll/pitch/yaw perturbation rigidly rotates every body in
    the root-relative frame and lands in the metric at full strength -- about
    39 mm on the G1's 14-body set before the policy has done anything. Interval
    pushes then keep injecting disturbance for the rest of the episode.

    That is the right protocol for a robustness number, and for the paired
    interface comparison where both rows see identical perturbations. It is the
    wrong protocol for an absolute tracking claim or for comparison against
    externally published MPJPE, which is measured on an unperturbed rollout.

    Returns a record of everything changed, so a result file can prove which
    protocol produced it. Mutates ``env_cfg`` in place.
    """
    record: dict[str, Any] = {
        "enabled": True,
        "reset_ranges_zeroed": {},
        "noise_params_disabled": [],
        "events_disabled": [],
    }
    events = getattr(env_cfg, "events", None)
    if events is None:
        return record

    for term_name in dir(events):
        if term_name.startswith("_"):
            continue
        params = getattr(getattr(events, term_name, None), "params", None)
        if not isinstance(params, dict):
            continue
        for key, neutral in _RESET_RANGE_NEUTRAL.items():
            if key in params:
                params[key] = type(neutral)(neutral)
                record["reset_ranges_zeroed"].setdefault(term_name, []).append(key)
        for key in _NULLABLE_NOISE_PARAMS:
            if key in params:
                params[key] = None
                record["noise_params_disabled"].append(f"{term_name}.{key}")

    for term_name in _RANDOMIZATION_EVENTS:
        if getattr(events, term_name, None) is not None:
            setattr(events, term_name, None)
            record["events_disabled"].append(term_name)

    record["reset_ranges_zeroed"] = {
        name: sorted(keys) for name, keys in record["reset_ranges_zeroed"].items()
    }
    record["noise_params_disabled"].sort()
    record["events_disabled"].sort()
    return record


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
