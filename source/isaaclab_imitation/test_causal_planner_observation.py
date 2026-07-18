from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_MODULE_PATH = (
    Path(__file__).parent
    / "isaaclab_imitation"
    / "envs"
    / "causal_planner_observation.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "causal_planner_observation", _MODULE_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)

CAUSAL_PLANNER_FRAME_DIM = _MODULE.CAUSAL_PLANNER_FRAME_DIM
CausalPlannerHistory = _MODULE.CausalPlannerHistory
build_causal_planner_frame = _MODULE.build_causal_planner_frame
build_offline_causal_planner_frame = _MODULE.build_offline_causal_planner_frame
causal_planner_observation_spec = _MODULE.causal_planner_observation_spec


def _features(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "joint_pos_rel": torch.zeros(batch_size, 29),
        "joint_vel_rel": torch.zeros(batch_size, 29),
        "base_ang_vel": torch.zeros(batch_size, 3),
        "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch_size, 1),
        "last_action": torch.zeros(batch_size, 29),
    }


def test_schema_is_ten_by_ninety_three() -> None:
    spec = causal_planner_observation_spec(history_steps=9)
    assert spec["feature_names"] == [
        "joint_pos_rel",
        "joint_vel_rel",
        "base_ang_vel",
        "projected_gravity",
        "last_action",
    ]
    assert spec["frame_dim"] == 93
    assert spec["history_frames"] == 10
    assert spec["flat_dim"] == 930
    assert spec["reference_features"] == []


def test_robot_feature_changes_only_its_slice() -> None:
    baseline = build_causal_planner_frame(_features(batch_size=1))
    changed_features = _features(batch_size=1)
    changed_features["base_ang_vel"][0, 1] = 2.0
    changed = build_causal_planner_frame(changed_features)
    nonzero = (changed - baseline).nonzero(as_tuple=False)
    assert nonzero.tolist() == [[0, 59]]


def test_unrelated_reference_changes_cannot_change_frame() -> None:
    features = _features(batch_size=1)
    frame_before = build_causal_planner_frame(features)
    fake_reference = torch.randn(1, 10, 67)
    fake_reference.add_(1000.0)
    frame_after = build_causal_planner_frame(features)
    torch.testing.assert_close(frame_before, frame_after)


def test_history_reset_repeats_new_initial_frame_only_for_reset_rows() -> None:
    initial = build_causal_planner_frame(_features())
    history = CausalPlannerHistory(initial, history_steps=9)
    updated = initial.clone()
    updated[:, 0] = torch.tensor([1.0, 2.0])
    history.append(updated)

    reset_frame = updated[1:2].clone()
    reset_frame[:, 0] = 7.0
    history.reset(torch.tensor([1]), reset_frame)

    selected = history.select(torch.tensor([0, 1]), history_steps=9)
    assert selected.shape == (2, 10, CAUSAL_PLANNER_FRAME_DIM)
    assert selected[0, -1, 0].item() == 1.0
    assert selected[0, 0, 0].item() == 0.0
    assert torch.all(selected[1, :, 0] == 7.0)


def test_offline_builder_matches_live_order_for_identity_root() -> None:
    features = _features(batch_size=1)
    features["joint_pos_rel"].normal_()
    features["joint_vel_rel"].normal_()
    features["base_ang_vel"].normal_()
    features["last_action"].normal_()
    live = build_causal_planner_frame(features)
    offline = build_offline_causal_planner_frame(
        joint_pos=features["joint_pos_rel"],
        joint_vel=features["joint_vel_rel"],
        root_quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_ang_vel_w=features["base_ang_vel"],
        last_action=features["last_action"],
        default_joint_pos=torch.zeros(1, 29),
        default_joint_vel=torch.zeros(1, 29),
    )
    torch.testing.assert_close(offline, live)


def test_offline_builder_requires_previous_action() -> None:
    try:
        build_offline_causal_planner_frame(
            joint_pos=torch.zeros(1, 29),
            joint_vel=torch.zeros(1, 29),
            root_quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_ang_vel_w=torch.zeros(1, 3),
            last_action=None,
            default_joint_pos=torch.zeros(1, 29),
            default_joint_vel=torch.zeros(1, 29),
        )
    except ValueError as error:
        assert "previous action" in str(error)
    else:
        raise AssertionError("missing previous action must fail")
