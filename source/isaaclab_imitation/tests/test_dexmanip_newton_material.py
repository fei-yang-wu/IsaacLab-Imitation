"""Pure contracts for the Newton Wuji hand material override."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "isaaclab_imitation/tasks/manager_based/dexmanip/newton_material.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "_newton_material_contract", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
NEWTON_MATERIAL = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = NEWTON_MATERIAL
MODULE_SPEC.loader.exec_module(NEWTON_MATERIAL)

NEWTON_COLLIDE_SHAPES_FLAG = NEWTON_MATERIAL.NEWTON_COLLIDE_SHAPES_FLAG
WUJI_HAND_BODY_NAMES = NEWTON_MATERIAL.WUJI_HAND_BODY_NAMES
WujiHandMaterialSpec = NEWTON_MATERIAL.WujiHandMaterialSpec
sample_bucketed_wuji_materials = NEWTON_MATERIAL.sample_bucketed_wuji_materials
select_wuji_collision_shape_ids = NEWTON_MATERIAL.select_wuji_collision_shape_ids


def _complete_labels_and_shapes() -> tuple[list[str], list[int], list[int]]:
    body_names = [
        "base_link",
        "L_arm_l8",
        "R_arm_l8",
        *WUJI_HAND_BODY_NAMES,
    ]
    labels = [f"/World/envs/env_0/Robot/{name}" for name in body_names]
    shape_body_ids = list(range(len(labels)))
    shape_flags = [NEWTON_COLLIDE_SHAPES_FLAG] * len(labels)
    return labels, shape_body_ids, shape_flags


def test_released_material_exposes_exact_ranges_and_bucket_count() -> None:
    spec = WujiHandMaterialSpec()

    assert spec.static_friction_range == (2.0, 2.01)
    assert spec.dynamic_friction_range == (2.0, 2.01)
    assert spec.restitution_range == (0.0, 0.0)
    assert spec.num_buckets == 64


def test_mjwarp_rejects_different_static_and_dynamic_friction() -> None:
    with pytest.raises(ValueError, match="must be equal"):
        WujiHandMaterialSpec(dynamic_friction_range=(1.9, 2.0))


def test_bucketed_material_sampling_uses_run_seed_and_released_ranges() -> None:
    first = sample_bucketed_wuji_materials(200, seed=123)
    repeated = sample_bucketed_wuji_materials(200, seed=123)
    different_seed = sample_bucketed_wuji_materials(200, seed=124)

    assert first == repeated
    assert first != different_seed
    friction, restitution = first
    assert all(2.0 <= value <= 2.01 for value in friction)
    assert set(restitution) == {0.0}
    assert len(set(friction)) <= 64


def test_shape_selection_changes_wuji_collisions_and_preserves_vega() -> None:
    labels, body_ids, flags = _complete_labels_and_shapes()

    selected = select_wuji_collision_shape_ids(labels, body_ids, flags)

    selected_names = {labels[body_ids[index]].rsplit("/", 1)[-1] for index in selected}
    assert "l_wrist" in selected_names
    assert "r_wrist" in selected_names
    assert "L_arm_l8" not in selected_names
    assert "R_arm_l8" not in selected_names
    assert "base_link" not in selected_names
    assert len(selected) == len(WUJI_HAND_BODY_NAMES)


def test_shape_selection_excludes_exact_hand_name_outside_robot_path() -> None:
    labels, body_ids, flags = _complete_labels_and_shapes()
    labels.append("/World/envs/env_0/l_wrist")
    body_ids.append(len(labels) - 1)
    flags.append(NEWTON_COLLIDE_SHAPES_FLAG)

    selected = select_wuji_collision_shape_ids(labels, body_ids, flags)

    assert len(selected) == len(WUJI_HAND_BODY_NAMES)
    assert len(labels) - 1 not in selected


def test_leaf_label_fallback_requires_exact_mjcf_body_names() -> None:
    labels = [*WUJI_HAND_BODY_NAMES, "l_scene_object", "r_workpiece"]
    body_ids = list(range(len(labels)))
    flags = [NEWTON_COLLIDE_SHAPES_FLAG] * len(labels)

    selected = select_wuji_collision_shape_ids(labels, body_ids, flags)

    assert len(selected) == len(WUJI_HAND_BODY_NAMES)
    assert len(labels) - 2 not in selected
    assert len(labels) - 1 not in selected


def test_shape_selection_ignores_sites_and_visual_only_shapes() -> None:
    labels, body_ids, flags = _complete_labels_and_shapes()
    labels.append("/World/envs/env_0/Robot/l_wrist")
    body_ids.extend((3, 3))
    flags.extend((1, 8))

    selected = select_wuji_collision_shape_ids(labels, body_ids, flags)

    assert len(selected) == len(WUJI_HAND_BODY_NAMES)


def test_shape_selection_fails_closed_when_one_hand_is_incomplete() -> None:
    labels, body_ids, flags = _complete_labels_and_shapes()
    missing_index = labels.index("/World/envs/env_0/Robot/r_pinky_distal")
    flags[missing_index] = 1

    with pytest.raises(ValueError, match="r_pinky_distal"):
        select_wuji_collision_shape_ids(labels, body_ids, flags)


def test_shape_selection_fails_when_one_robot_has_no_hand_collisions() -> None:
    labels, body_ids, flags = _complete_labels_and_shapes()
    second_labels = [label.replace("env_0", "env_1") for label in labels]
    second_offset = len(labels)
    labels.extend(second_labels)
    body_ids.extend(range(second_offset, len(labels)))
    flags.extend([1] * len(second_labels))

    with pytest.raises(ValueError, match="env_1/Robot.*Missing bodies"):
        select_wuji_collision_shape_ids(labels, body_ids, flags)


def test_shape_selection_rejects_invalid_shape_body_id() -> None:
    labels, body_ids, flags = _complete_labels_and_shapes()
    body_ids[-1] = len(labels)

    with pytest.raises(ValueError, match="uses body ID"):
        select_wuji_collision_shape_ids(labels, body_ids, flags)
