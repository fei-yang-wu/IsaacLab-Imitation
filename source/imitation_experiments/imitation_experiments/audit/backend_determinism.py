# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Make a cross-backend probe a controlled comparison, or fail loudly.

Two Isaac Lab physics backends can only be compared when the *only* thing that
differs between the two processes is the solver. Every probe in this repo
therefore has to pin the same reference start frame, the same trajectory
assignment, and the same (absent) randomization on both sides.

That pinning used to be three lines copied into each probe::

    env_cfg.random_reset_step_min = 0
    env_cfg.random_reset_step_max = 0
    env_cfg.random_reset_full_trajectory = False

which is correct on the legacy ``-G1-v0`` / ``-G1-v1`` lineage and a **silent
no-op** on the v2 surface, whose reset sampling moved onto
``command_interface.reference.selection``. A configclass accepts the unknown
attributes without complaint, so a v2 probe kept its default random start
(uniform over frames 0-200, random trajectory schedule) while reporting itself
as deterministic. Two backends then scored different motions from different
frames, and the difference was read as a solver gap.

:func:`pin_reference_start` writes whichever surface the config actually has and
raises when it recognizes neither, so the failure mode is a traceback rather
than a plausible number.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RANDOMIZATION_PROFILES",
    "apply_randomization_profile",
    "describe_reference_selection",
    "pin_reference_start",
]

RANDOMIZATION_PROFILES = ("none", "startup", "reset", "all")
"""Randomization kept by :func:`apply_randomization_profile`.

``none`` is the only setting under which two backends are comparable; the
others exist to attribute a gap to a specific randomization family.
"""

# Startup-mode domain randomization (asset properties). Every one of these
# draws from the global RNG, so leaving one on desynchronizes the two runs even
# when its physical effect is small.
_STARTUP_EVENT_NAMES = (
    "physics_material",
    "add_joint_default_pos",
    "base_com",
    "randomize_rigid_body_mass",
)

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _selection_cfg(env_cfg: Any) -> Any | None:
    """The v2 reference-selection config, with any preset collapsed.

    Returns ``None`` when the config is not a v2 command-interface surface.
    """
    interface = getattr(env_cfg, "command_interface", None)
    reference = getattr(interface, "reference", None)
    selection = getattr(reference, "selection", None)
    if selection is None:
        return None
    # `PresetCfg` alternatives are collapsed by `CommandInterfaceCfg.resolve()`,
    # which has not run yet at probe-configuration time. Walk to `.default`
    # ourselves rather than importing Isaac Lab: this module is imported by the
    # default (Isaac-free) Pixi environment's tests.
    seen = 0
    while not hasattr(selection, "start_mode"):
        selection = getattr(selection, "default", None)
        seen += 1
        if selection is None or seen > 8:
            return None
    return selection


def pin_reference_start(env_cfg: Any, *, start_frame: int = 0) -> str:
    """Pin every environment to the same trajectory schedule and start frame.

    Returns the name of the surface that was written (``"command_interface"``
    or ``"legacy"``) so a probe can record which contract it exercised.

    Raises:
        TypeError: the config exposes neither reset-sampling surface. Silence
            here is what produced uncontrolled backend comparisons before, so
            an unrecognized config is an error, never a no-op.
    """
    start_frame = int(start_frame)
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}.")

    selection = _selection_cfg(env_cfg)
    if selection is not None:
        # `round_robin` assigns trajectory ranks from the environment index, so
        # environment i tracks the same motion on both backends. `random` would
        # draw from an RNG the two processes consume at different points.
        selection.schedule = "round_robin"
        selection.custom_fn = None
        selection.start_mode = "fixed"
        selection.start_frame = start_frame
        selection.full_trajectory = False
        env_cfg.command_interface.reference.selection = selection
        return "command_interface"

    if hasattr(env_cfg, "random_reset_step_min"):
        env_cfg.random_reset_step_min = start_frame
        env_cfg.random_reset_step_max = start_frame
        env_cfg.random_reset_full_trajectory = False
        if hasattr(env_cfg, "reset_schedule"):
            env_cfg.reset_schedule = "round_robin"
        return "legacy"

    raise TypeError(
        f"{type(env_cfg).__name__} exposes neither "
        "`command_interface.reference.selection` (v2) nor `random_reset_step_min` "
        "(legacy), so the reference start frame cannot be pinned. A cross-backend "
        "probe must not run against a config whose reset sampling it cannot "
        "control."
    )


def describe_reference_selection(env_cfg: Any) -> dict[str, Any]:
    """Record the reset-sampling settings a probe actually ran with.

    Written into the probe's output so a later reader can tell a pinned run
    from an uncontrolled one without re-deriving it from the launch command.
    """
    selection = _selection_cfg(env_cfg)
    if selection is not None:
        return {
            "surface": "command_interface",
            "schedule": str(selection.schedule),
            "start_mode": str(selection.start_mode),
            "start_frame": int(selection.start_frame),
            "random_step_min": int(selection.random_step_min),
            "random_step_max": int(selection.random_step_max),
            "full_trajectory": bool(selection.full_trajectory),
        }
    if hasattr(env_cfg, "random_reset_step_min"):
        return {
            "surface": "legacy",
            "random_step_min": int(env_cfg.random_reset_step_min),
            "random_step_max": int(getattr(env_cfg, "random_reset_step_max", 0)),
            "full_trajectory": bool(
                getattr(env_cfg, "random_reset_full_trajectory", False)
            ),
        }
    return {"surface": None}


def apply_randomization_profile(env_cfg: Any, profile: str) -> dict[str, bool]:
    """Keep only the randomization families named by ``profile``.

    Returns which families survived, for the probe's record.

    Raises:
        ValueError: unknown profile name.
    """
    if profile not in RANDOMIZATION_PROFILES:
        raise ValueError(
            f"Unknown randomization profile {profile!r}; expected one of "
            f"{list(RANDOMIZATION_PROFILES)}."
        )
    events = getattr(env_cfg, "events", None)
    if events is None:
        return {"startup": False, "reset": False, "push": False}

    keep_startup = profile in ("startup", "all")
    keep_reset = profile in ("reset", "all")
    keep_push = profile == "all"

    if not keep_startup:
        for name in _STARTUP_EVENT_NAMES:
            if hasattr(events, name):
                setattr(events, name, None)
    if not keep_push and hasattr(events, "push_robot"):
        events.push_robot = None
    if not keep_reset:
        # The reset event also *places* the robot on the reference, so it is
        # zeroed rather than removed: dropping it would leave the robot at its
        # spawn pose and make the probe measure nothing.
        reset_term = getattr(events, "reset_reference_state", None)
        if reset_term is not None:
            reset_term.params["pose_range"] = {key: (0.0, 0.0) for key in _POSE_KEYS}
            reset_term.params["velocity_range"] = {
                key: (0.0, 0.0) for key in _POSE_KEYS
            }
            reset_term.params["joint_position_range"] = (0.0, 0.0)

    return {"startup": keep_startup, "reset": keep_reset, "push": keep_push}
