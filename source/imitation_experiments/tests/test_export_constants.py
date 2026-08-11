"""Exporter constants that need no torch export path (always collected)."""

import pytest

from imitation_experiments.lowlevel import export_policy_bundle as epb


def test_actuation_table_matches_known_values():
    actuation = epb._per_joint_actuation()
    names = epb.G1_ISAAC_JOINT_NAMES
    by_name = dict(zip(names, actuation["action_scale"], strict=True))
    assert by_name["left_hip_pitch_joint"] == pytest.approx(0.35066147, abs=1e-6)
    assert by_name["waist_yaw_joint"] == pytest.approx(0.54754647, abs=1e-6)
    assert by_name["left_ankle_pitch_joint"] == pytest.approx(0.43857731, abs=1e-6)
    assert by_name["left_wrist_pitch_joint"] == pytest.approx(0.07450087, abs=1e-6)
    defaults_by_name = dict(zip(names, actuation["default_joint_pos"], strict=True))
    assert defaults_by_name["left_knee_joint"] == pytest.approx(0.669)
    assert defaults_by_name["right_shoulder_roll_joint"] == pytest.approx(-0.2)
    assert defaults_by_name["left_shoulder_roll_joint"] == pytest.approx(0.2)
    assert min(actuation["stiffness"]) > 0 and min(actuation["damping"]) > 0
    effort_by_name = dict(zip(names, actuation["effort_limit"], strict=True))
    assert effort_by_name["left_hip_pitch_joint"] == pytest.approx(139.0)
    assert effort_by_name["left_wrist_yaw_joint"] == pytest.approx(5.0)
    armature_by_name = dict(zip(names, actuation["armature"], strict=True))
    assert armature_by_name["left_ankle_pitch_joint"] == pytest.approx(2 * 0.003609725)
    assert armature_by_name["left_hip_pitch_joint"] == pytest.approx(0.025101925)


def test_sdk_permutation_round_trip():
    sdk = epb._sdk_joint_names()
    assert len(sdk) == 29
    isaac_to_sdk = [sdk.index(name) for name in epb.G1_ISAAC_JOINT_NAMES]
    assert sorted(isaac_to_sdk) == list(range(29))
    assert (
        sdk[isaac_to_sdk[epb.G1_ISAAC_JOINT_NAMES.index("waist_yaw_joint")]]
        == "waist_yaw_joint"
    )
