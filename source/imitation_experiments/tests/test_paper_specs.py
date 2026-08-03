from __future__ import annotations

import pytest

from imitation_experiments.paper.specs import (
    build_explicit_interface,
    get_interface,
)


def test_root_points5_pose_width_and_term_order() -> None:
    spec = get_interface("root_points5_pose")
    assert spec.command_components == (
        "keypoint_pos",
        "keypoint_ori",
        "root_pos",
        "root_ori",
    )
    assert spec.command_terms == (
        "expert_keypoint_pos_b",
        "expert_keypoint_ori_b",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    )
    assert spec.packet_values(horizon_steps=10) == 540


def test_custom_explicit_components_are_canonically_ordered() -> None:
    spec = build_explicit_interface(
        name="custom_root_qpos",
        command_components=("root_ori", "joint_qpos", "root_pos"),
    )
    assert spec.command_components == ("joint_qpos", "root_pos", "root_ori")
    assert spec.command_terms == (
        "expert_motion_qpos",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    )
    assert spec.packet_values(horizon_steps=10) == 380


def test_shared_vanilla_rejects_reduced_packet() -> None:
    with pytest.raises(ValueError, match="full 67D"):
        build_explicit_interface(
            name="invalid_shared",
            command_components=("joint_qpos", "root_pos", "root_ori"),
            tracker_binding="shared_vanilla",
        )
