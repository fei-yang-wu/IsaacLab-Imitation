"""Pure contracts for the Newton-safe scene layer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "isaaclab_imitation/tasks/manager_based/dexmanip/scene.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("_scene_contract", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
SCENE_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = SCENE_MODULE
MODULE_SPEC.loader.exec_module(SCENE_MODULE)

ContactMaterialSpec = SCENE_MODULE.ContactMaterialSpec
RigidObjectSpec = SCENE_MODULE.RigidObjectSpec
SceneSpec = SCENE_MODULE.SceneSpec
SupportSurfaceSpec = SCENE_MODULE.SupportSurfaceSpec
wxyz_to_xyzw = SCENE_MODULE.wxyz_to_xyzw


def _write_rigid_urdf(path: Path) -> None:
    path.write_text(
        """
        <robot name="cube">
          <link name="base"/>
          <link name="visual"/>
          <joint name="visual_mount" type="fixed">
            <parent link="base"/>
            <child link="visual"/>
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )


def test_valid_scene_has_one_primary_object_and_valid_assets(tmp_path: Path) -> None:
    object_path = tmp_path / "cube.urdf"
    support_path = tmp_path / "support.usda"
    _write_rigid_urdf(object_path)
    support_path.write_text("#usda 1.0\n", encoding="utf-8")

    scene = SceneSpec(
        objects=(
            RigidObjectSpec(
                name="cube",
                asset_path=str(object_path),
                init_pos=[0, 0, 0.5],
                mass=0.2,
                material=ContactMaterialSpec(
                    static_friction=1.2,
                    dynamic_friction=0.9,
                ),
            ),
        ),
        support_surfaces=(
            SupportSurfaceSpec(
                name="support_surface",
                asset_path=str(support_path),
            ),
        ),
    )

    scene.validate_assets()
    assert scene.primary_object_name == "cube"
    assert scene.object_names == ("cube",)
    assert scene.support_surface_names == ("support_surface",)
    assert scene.objects[0].init_pos == (0.0, 0.0, 0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "bad-name", "asset_path": "cube.usd"}, "USD path segment"),
        (
            {
                "name": "cube",
                "asset_path": "cube.usd",
                "init_quat_wxyz": (2.0, 0.0, 0.0, 0.0),
            },
            "unit quaternion",
        ),
        (
            {"name": "cube", "asset_path": "cube.usd", "scale": (1.0, 0.0, 1.0)},
            "scale values must be positive",
        ),
        (
            {"name": "cube", "asset_path": "cube.obj"},
            "asset must use one of",
        ),
    ],
)
def test_rigid_object_spec_rejects_invalid_contract(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RigidObjectSpec(**kwargs)


def test_scene_rejects_duplicate_names_across_roles() -> None:
    with pytest.raises(ValueError, match="asset names must be unique"):
        SceneSpec(
            objects=(RigidObjectSpec(name="item", asset_path="item.usd"),),
            support_surfaces=(
                SupportSurfaceSpec(name="item", asset_path="support.usda"),
            ),
        )


def test_scene_rejects_primary_name_that_is_not_dynamic() -> None:
    with pytest.raises(ValueError, match="does not name a rigid object"):
        SceneSpec(
            objects=(RigidObjectSpec(name="item", asset_path="item.usd"),),
            primary_object_name="other",
        )


def test_asset_validation_rejects_missing_file(tmp_path: Path) -> None:
    scene = SceneSpec(
        objects=(
            RigidObjectSpec(
                name="item",
                asset_path=str(tmp_path / "missing.usd"),
            ),
        ),
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        scene.validate_assets()


def test_asset_validation_rejects_articulated_urdf(tmp_path: Path) -> None:
    object_path = tmp_path / "hinge.urdf"
    object_path.write_text(
        """
        <robot name="hinge">
          <link name="base"/>
          <link name="lid"/>
          <joint name="hinge" type="revolute">
            <parent link="base"/>
            <child link="lid"/>
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )
    scene = SceneSpec(
        objects=(RigidObjectSpec(name="hinge", asset_path=str(object_path)),),
    )

    with pytest.raises(ValueError, match="non-fixed joint"):
        scene.validate_assets()


def test_quaternion_conversion_is_explicit_at_isaac_boundary() -> None:
    assert wxyz_to_xyzw((0.5, 0.5, -0.5, 0.5)) == (0.5, -0.5, 0.5, 0.5)


def test_dynamic_object_preserves_authored_material_by_default() -> None:
    object_spec = RigidObjectSpec(name="item", asset_path="item.usd")
    support_spec = SupportSurfaceSpec(name="support_surface", asset_path="support.usda")

    assert object_spec.material is None
    assert support_spec.material == ContactMaterialSpec()


def test_scene_contacts_default_to_zero_newton_offsets() -> None:
    """Do not inherit Newton's locomotion-oriented one-centimetre shape gap."""

    object_spec = RigidObjectSpec(name="item", asset_path="item.usd")
    support_spec = SupportSurfaceSpec(name="support_surface", asset_path="support.usda")

    assert object_spec.contact_margin == 0.0
    assert object_spec.contact_gap == 0.0
    assert support_spec.contact_margin == 0.0
    assert support_spec.contact_gap == 0.0


def test_scene_module_has_no_physx_or_live_stage_dependency() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "isaaclab_physx" not in source
    assert "activate_contact_sensors" not in source
    assert "from pxr" not in source
    assert "Usd.Stage" not in source
    assert "rigid_body_enabled=False" in source
