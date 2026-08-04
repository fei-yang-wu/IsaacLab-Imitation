# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The pinning helper must write a real surface or raise.

These use plain stand-ins rather than the Isaac Lab configs so they run in the
default (Isaac-free) Pixi environment, which is where the experiment-library
tests live. The shapes mirror the two real lineages: v2 keeps reset sampling on
``command_interface.reference.selection``, the legacy lineage on flat
``random_reset_*`` fields.
"""

from __future__ import annotations

import pytest

from imitation_experiments.audit.backend_determinism import (
    apply_randomization_profile,
    describe_reference_selection,
    pin_reference_start,
)


class _Selection:
    def __init__(self):
        self.schedule = "random"
        self.custom_fn = None
        self.start_mode = "auto"
        self.start_frame = 0
        self.random_step_min = 0
        self.random_step_max = 200
        self.full_trajectory = False


class _SelectionPreset:
    """Stands in for an unresolved ``PresetCfg`` (no ``start_mode`` of its own)."""

    def __init__(self):
        self.default = _Selection()
        self.frame0 = _Selection()


class _Reference:
    def __init__(self, selection):
        self.selection = selection


class _Interface:
    def __init__(self, selection):
        self.reference = _Reference(selection)


class _EventTerm:
    def __init__(self, **params):
        self.params = dict(params)


class _Events:
    def __init__(self):
        self.physics_material = _EventTerm()
        self.add_joint_default_pos = _EventTerm(pos_distribution_params=(-0.01, 0.01))
        self.base_com = _EventTerm()
        self.randomize_rigid_body_mass = _EventTerm()
        self.push_robot = _EventTerm()
        self.reset_reference_state = _EventTerm(
            pose_range={"x": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
            velocity_range={"x": (-0.5, 0.5)},
            joint_position_range=(-0.1, 0.1),
        )


class _V2Cfg:
    def __init__(self, selection=None):
        self.command_interface = _Interface(selection or _SelectionPreset())
        self.events = _Events()


class _LegacyCfg:
    def __init__(self):
        self.random_reset_step_min = 0
        self.random_reset_step_max = 200
        self.random_reset_full_trajectory = True
        self.events = _Events()


class _ForeignCfg:
    """A config with neither surface -- the case that used to pass silently."""

    def __init__(self):
        self.events = _Events()


def test_pins_the_v2_selection_through_an_unresolved_preset():
    cfg = _V2Cfg()
    assert pin_reference_start(cfg) == "command_interface"
    selection = cfg.command_interface.reference.selection
    assert selection.start_mode == "fixed"
    assert selection.start_frame == 0
    assert selection.schedule == "round_robin"
    assert selection.full_trajectory is False


def test_pins_an_already_resolved_v2_selection():
    cfg = _V2Cfg(selection=_Selection())
    assert pin_reference_start(cfg, start_frame=37) == "command_interface"
    assert cfg.command_interface.reference.selection.start_frame == 37


def test_pins_the_legacy_surface():
    cfg = _LegacyCfg()
    assert pin_reference_start(cfg) == "legacy"
    assert cfg.random_reset_step_min == 0
    assert cfg.random_reset_step_max == 0
    assert cfg.random_reset_full_trajectory is False


def test_refuses_a_config_it_cannot_pin():
    # The regression this module exists for: writing `random_reset_step_min`
    # onto a v2 config is accepted by a configclass and changes nothing, so an
    # unrecognized surface must raise instead of reporting success.
    with pytest.raises(TypeError, match="neither"):
        pin_reference_start(_ForeignCfg())


def test_rejects_a_negative_start_frame():
    with pytest.raises(ValueError, match="start_frame"):
        pin_reference_start(_V2Cfg(), start_frame=-1)


def test_describe_reports_the_pinned_settings():
    cfg = _V2Cfg()
    pin_reference_start(cfg, start_frame=5)
    described = describe_reference_selection(cfg)
    assert described["surface"] == "command_interface"
    assert described["start_mode"] == "fixed"
    assert described["start_frame"] == 5


def test_profile_none_removes_every_randomization_source():
    cfg = _V2Cfg()
    kept = apply_randomization_profile(cfg, "none")
    assert kept == {"startup": False, "reset": False, "push": False}
    events = cfg.events
    assert events.physics_material is None
    assert events.add_joint_default_pos is None
    assert events.base_com is None
    assert events.randomize_rigid_body_mass is None
    assert events.push_robot is None
    # The reset event survives -- it is what places the robot on the reference --
    # but with every perturbation range collapsed to zero.
    reset = events.reset_reference_state
    assert set(reset.params["pose_range"].values()) == {(0.0, 0.0)}
    assert set(reset.params["velocity_range"].values()) == {(0.0, 0.0)}
    assert reset.params["joint_position_range"] == (0.0, 0.0)


def test_profile_startup_keeps_asset_randomization_but_drops_pushes():
    cfg = _V2Cfg()
    kept = apply_randomization_profile(cfg, "startup")
    assert kept == {"startup": True, "reset": False, "push": False}
    assert cfg.events.physics_material is not None
    assert cfg.events.push_robot is None
    assert cfg.events.reset_reference_state.params["joint_position_range"] == (0.0, 0.0)


def test_profile_all_changes_nothing():
    cfg = _V2Cfg()
    kept = apply_randomization_profile(cfg, "all")
    assert kept == {"startup": True, "reset": True, "push": True}
    assert cfg.events.push_robot is not None
    assert cfg.events.reset_reference_state.params["joint_position_range"] == (
        -0.1,
        0.1,
    )


def test_rejects_an_unknown_profile():
    with pytest.raises(ValueError, match="Unknown randomization profile"):
        apply_randomization_profile(_V2Cfg(), "sometimes")
