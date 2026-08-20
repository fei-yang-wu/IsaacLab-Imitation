"""Pure contracts for the Newton ILTools scene-material override."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "isaaclab_imitation/tasks/manager_based/dexmanip/newton_scene_material.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "_newton_scene_material_contract", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
SCENE_MATERIAL = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = SCENE_MATERIAL
MODULE_SPEC.loader.exec_module(SCENE_MATERIAL)

COLLIDE = SCENE_MATERIAL.NEWTON_COLLIDE_SHAPES_FLAG
build_scene_material_assignments = SCENE_MATERIAL.build_scene_material_assignments
configure_scene_material = SCENE_MATERIAL.configure_scene_material
select_scene_collision_shape_ids = SCENE_MATERIAL.select_scene_collision_shape_ids


def _replicated_scene() -> tuple[list[str], list[int]]:
    labels = [
        "/World/ground",
        "/World/envs/env_0/Robot/cube",
        "/World/envs/env_0/cube_backup",
        "/World/envs/env_0/cube",
        "/World/envs/env_0/cube/visual",
        "/World/envs/env_0/cube/collisions/mesh",
        "/World/envs/env_0/table/collider",
        "/World/envs/env_1/cube/collisions/mesh_0",
        "/World/envs/env_1/cube/collisions/mesh_1",
        "/World/envs/env_1/table",
    ]
    flags = [
        COLLIDE,
        COLLIDE,
        COLLIDE,
        COLLIDE,
        0,
        COLLIDE,
        COLLIDE,
        COLLIDE,
        COLLIDE,
        COLLIDE,
    ]
    return labels, flags


def test_selector_matches_exact_asset_roots_and_descendants_per_env() -> None:
    labels, flags = _replicated_scene()

    selected = select_scene_collision_shape_ids(
        labels,
        flags,
        num_envs=2,
        object_names=("cube",),
        support_surface_names=("table",),
    )

    assert selected.object_shape_ids_by_env == (((3, 5),), ((7, 8),))
    assert selected.support_shape_ids_by_env == (((6,),), ((9,),))
    assert 1 not in selected.object_shape_ids_by_env[0][0]
    assert 2 not in selected.object_shape_ids_by_env[0][0]
    assert 4 not in selected.object_shape_ids_by_env[0][0]


def test_selector_fails_when_one_replica_has_no_colliding_asset_shape() -> None:
    labels, flags = _replicated_scene()
    flags[9] = 0

    with pytest.raises(ValueError, match="env_1/table"):
        select_scene_collision_shape_ids(
            labels,
            flags,
            num_envs=2,
            object_names=("cube",),
            support_surface_names=("table",),
        )


def test_selector_rejects_ambiguous_or_non_segment_asset_names() -> None:
    labels, flags = _replicated_scene()

    with pytest.raises(ValueError, match="globally unique"):
        select_scene_collision_shape_ids(
            labels,
            flags,
            num_envs=2,
            object_names=("cube",),
            support_surface_names=("cube",),
        )
    with pytest.raises(ValueError, match="path segments"):
        select_scene_collision_shape_ids(
            labels,
            flags,
            num_envs=2,
            object_names=("nested/cube",),
            support_surface_names=("table",),
        )


def test_material_assignments_use_dynamic_friction_for_every_replica() -> None:
    labels, flags = _replicated_scene()
    selected = select_scene_collision_shape_ids(
        labels,
        flags,
        num_envs=2,
        object_names=("cube",),
        support_surface_names=("table",),
    )

    shape_ids, friction, restitution = build_scene_material_assignments(
        selected,
        object_dynamic_friction=(0.7,),
        object_restitution=(0.1,),
        support_dynamic_friction=(0.5,),
        support_restitution=(0.0,),
    )

    assert shape_ids == (3, 5, 7, 8, 6, 9)
    assert friction == pytest.approx((0.7, 0.7, 0.7, 0.7, 0.5, 0.5))
    assert restitution == pytest.approx((0.1, 0.1, 0.1, 0.1, 0.0, 0.0))


def test_material_assignments_reject_invalid_typed_values() -> None:
    labels, flags = _replicated_scene()
    selected = select_scene_collision_shape_ids(
        labels,
        flags,
        num_envs=2,
        object_names=("cube",),
        support_surface_names=("table",),
    )

    with pytest.raises(ValueError, match="non-negative"):
        build_scene_material_assignments(
            selected,
            object_dynamic_friction=(-0.1,),
            object_restitution=(0.0,),
            support_dynamic_friction=(0.5,),
            support_restitution=(0.0,),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_scene_material_assignments(
            selected,
            object_dynamic_friction=(0.7,),
            object_restitution=(1.1,),
            support_dynamic_friction=(0.5,),
            support_restitution=(0.0,),
        )


def test_missing_scene_physics_is_only_allowed_for_explicit_inspection() -> None:
    inspection_env = SimpleNamespace(
        cfg=SimpleNamespace(require_training_qualified_reference=False),
        _scene_physics=None,
    )
    configure_scene_material(inspection_env, None)

    training_env = SimpleNamespace(
        cfg=SimpleNamespace(require_training_qualified_reference=True),
        _scene_physics=None,
    )
    with pytest.raises(RuntimeError, match="must declare ScenePhysics"):
        configure_scene_material(training_env, None)


def test_scene_material_event_rejects_non_startup_env_ids() -> None:
    env = SimpleNamespace(
        cfg=SimpleNamespace(require_training_qualified_reference=False),
        _scene_physics=None,
    )

    with pytest.raises(ValueError, match="startup mode only"):
        configure_scene_material(env, (0,))
