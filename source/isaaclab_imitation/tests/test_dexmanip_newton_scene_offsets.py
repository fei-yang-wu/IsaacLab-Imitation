"""Pure contracts for the Newton ILTools scene contact-offset override."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


PACKAGE_DIR = Path(__file__).parents[1] / (
    "isaaclab_imitation/tasks/manager_based/dexmanip"
)
# Load the two sibling modules under a stub package, so the relative import
# inside newton_scene_offsets resolves without importing Isaac Lab.
_STUB = "_dexmanip_offsets_stub"
_package = types.ModuleType(_STUB)
_package.__path__ = [str(PACKAGE_DIR)]
sys.modules[_STUB] = _package


def _load(module_name: str):
    qualified = f"{_STUB}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        qualified, PACKAGE_DIR / f"{module_name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


SCENE_MATERIAL = _load("newton_scene_material")
SCENE_OFFSETS = _load("newton_scene_offsets")

COLLIDE = SCENE_MATERIAL.NEWTON_COLLIDE_SHAPES_FLAG
NewtonSceneShapeSelection = SCENE_MATERIAL.NewtonSceneShapeSelection
select_scene_collision_shape_ids = SCENE_MATERIAL.select_scene_collision_shape_ids
build_scene_contact_offset_assignments = (
    SCENE_OFFSETS.build_scene_contact_offset_assignments
)
scene_shape_offsets = SCENE_OFFSETS.scene_shape_offsets


def _selection(objects, supports) -> NewtonSceneShapeSelection:
    return NewtonSceneShapeSelection(
        object_shape_ids_by_env=objects,
        support_shape_ids_by_env=supports,
    )


def test_every_selected_shape_receives_the_declared_offsets() -> None:
    # Two environments, one object with two collision meshes, one support.
    selection = _selection(
        objects=(((0, 1),), ((4, 5),)),
        supports=(((2,),), ((6,),)),
    )
    shape_ids, margins, gaps = build_scene_contact_offset_assignments(
        selection,
        object_contact_margin=0.0,
        object_contact_gap=0.0,
        support_contact_margin=0.0,
        support_contact_gap=0.0,
    )
    assert shape_ids == (0, 1, 4, 5, 2, 6)
    assert margins == (0.0,) * 6
    assert gaps == (0.0,) * 6


def test_object_and_support_offsets_stay_separate() -> None:
    selection = _selection(objects=(((0,),),), supports=(((1,),),))
    shape_ids, margins, gaps = build_scene_contact_offset_assignments(
        selection,
        object_contact_margin=0.001,
        object_contact_gap=0.002,
        support_contact_margin=0.003,
        support_contact_gap=0.004,
    )
    assert shape_ids == (0, 1)
    assert margins == (0.001, 0.003)
    assert gaps == (0.002, 0.004)


@pytest.mark.parametrize(
    "field",
    [
        "object_contact_margin",
        "object_contact_gap",
        "support_contact_margin",
        "support_contact_gap",
    ],
)
@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_a_non_physical_offset_is_refused(field: str, bad: float) -> None:
    values = {
        "object_contact_margin": 0.0,
        "object_contact_gap": 0.0,
        "support_contact_margin": 0.0,
        "support_contact_gap": 0.0,
    }
    values[field] = bad
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_scene_contact_offset_assignments(
            _selection(objects=(((0,),),), supports=(((1,),),)), **values
        )


def test_a_shape_claimed_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="more than one scene offset asset"):
        build_scene_contact_offset_assignments(
            _selection(objects=(((0,),),), supports=(((0,),),)),
            object_contact_margin=0.0,
            object_contact_gap=0.0,
            support_contact_margin=0.0,
            support_contact_gap=0.0,
        )


def test_the_measured_corn_can_layout_reaches_every_collider() -> None:
    """The measured corn-can layout, with the visual mesh excluded.

    A URDF object keeps its geometry under instanced meshes, so Isaac Lab
    cannot apply collision properties there. Shape selection works on Newton
    labels and the COLLIDE flag, not on USD instancing, so it must reach the
    collider and the support, and must leave the visual mesh alone.
    """
    labels = (
        # Visual only: measured flags = 1, so COLLIDE_SHAPES is clear.
        "/World/envs/env_0/corn_can/Geometry/object/textured_mesh/textured_mesh",
        "/World/envs/env_0/corn_can/Geometry/object/textured_mesh_1/textured_mesh",
        "/World/envs/env_0/corn_can_support/object_0",
    )
    selection = select_scene_collision_shape_ids(
        labels,
        (1, COLLIDE, COLLIDE),
        num_envs=1,
        object_names=("corn_can",),
        support_surface_names=("corn_can_support",),
    )
    shape_ids, margins, gaps = build_scene_contact_offset_assignments(
        selection,
        object_contact_margin=0.0,
        object_contact_gap=0.0,
        support_contact_margin=0.0,
        support_contact_gap=0.0,
    )
    assert sorted(shape_ids) == [1, 2]
    assert set(gaps) == {0.0}
    assert set(margins) == {0.0}


def test_scene_shape_offsets_reports_every_shape() -> None:
    record = scene_shape_offsets(("a", "b"), (0.0, 0.1), (0.2, 0.3))
    assert record == {"a": (0.0, 0.2), "b": (0.1, 0.3)}
    with pytest.raises(ValueError, match="equal lengths"):
        scene_shape_offsets(("a",), (0.0, 0.1), (0.2,))
