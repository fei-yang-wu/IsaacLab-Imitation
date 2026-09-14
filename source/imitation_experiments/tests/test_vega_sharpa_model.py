import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from imitation_experiments.data.vega_sharpa_model import (
    compose_model,
    base_ground_alignment,
)
from imitation_experiments.paths import REPO_ROOT


def test_assembly_keeps_sharpa_mount_frames_and_collision_exclusions(tmp_path):
    assets = REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets"
    path = compose_model(assets, tmp_path / "vega_sharpa.xml")
    root = ET.parse(path).getroot()
    exclusions = {
        (e.get("body1"), e.get("body2")) for e in root.findall("contact/exclude")
    }
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nq == model.nv == model.nu == 58
    np.testing.assert_allclose(model.body_gravcomp[1:], 1.0)
    for name in (
        "L_arm_l3_collision",
        "R_arm_l5_collision",
        "torso_flip_link_collision",
    ):
        geom = model.geom(name).id
        assert model.geom_contype[geom] == model.geom_conaffinity[geom] == 1
    names = {model.joint(i).name for i in range(model.njnt)}
    assert not names & {"Lift", "torso_flip", "head_j1", "head_j2", "head_j3"}
    assert not any(
        n.startswith(("l_thumb", "r_thumb", "l_index", "r_index")) for n in names
    )
    data = mujoco.MjData(model)
    data.qpos[:] = model.jnt_range.mean(axis=1)
    mujoco.mj_kinematics(model, data)
    for side, prefix in (("left", "L"), ("right", "R")):
        hand = ET.parse(
            assets / f"sharpa_wave/xmls/sharpawave/{side}_sharpawave.xml"
        ).getroot()
        vendor_exclusions = {
            (e.get("body1"), e.get("body2")) for e in hand.findall("contact/exclude")
        }
        assert vendor_exclusions <= exclusions
        body = model.body(f"{side}_hand_C_MC").id
        flange = model.body(f"{prefix}_arm_l8").id
        np.testing.assert_allclose(data.xpos[body], data.xpos[flange], atol=1e-10)
        np.testing.assert_allclose(data.xmat[body], data.xmat[flange], atol=1e-10)
        wrist = model.site(f"{side}_mounted_wrist").id
        np.testing.assert_allclose(data.site_xmat[wrist], data.xmat[body], atol=1e-10)
        assert ("torso_flip_link", f"{prefix}_arm_l1") in exclusions


def test_usd_body_exclusions_expand_only_to_owned_shapes(tmp_path):
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdPhysics
    from imitation_experiments.data.vega_sharpa_model import (
        author_robot_shape_collision_exclusions,
    )

    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    stage.SetDefaultPrim(root)
    for body in ("/robot/torso", "/robot/torso/arm", "/robot/hand"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, body).GetPrim())
        for name in ("a", "b"):
            shape = UsdGeom.Cube.Define(stage, body + "/" + name).GetPrim()
            UsdPhysics.CollisionAPI.Apply(shape)
    UsdPhysics.FilteredPairsAPI.Apply(
        stage.GetPrimAtPath("/robot/torso")
    ).CreateFilteredPairsRel().AddTarget("/robot/hand")
    stage.GetRootLayer().Save()
    assert author_robot_shape_collision_exclusions(path) == 4
    stage.Reload()
    expected = {"/robot/hand/a", "/robot/hand/b"}
    for name in ("a", "b"):
        shape = stage.GetPrimAtPath("/robot/torso/" + name)
        assert {
            str(p) for p in shape.GetRelationship("physics:filteredPairs").GetTargets()
        } == expected
        assert not stage.GetPrimAtPath("/robot/torso/arm/" + name).HasRelationship(
            "physics:filteredPairs"
        )


def test_ground_alignment_uses_base_mesh_bottom_instead_of_frame_origin(tmp_path):
    from types import SimpleNamespace
    from imitation_experiments.data.vega_sharpa_reference import publish_local_reference

    assets = REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets"
    model = compose_model(assets, tmp_path / "vega_sharpa.xml")
    pose = np.array([-0.04, 0.55, 0.0, 2**-0.5, 0, 0, -(2**-0.5)])
    old = base_ground_alignment(model, pose)
    assert old["base_ground_clearance_m"] == pytest.approx(-0.1869, abs=1e-4)
    with pytest.raises(ValueError, match="base intersects"):
        publish_local_reference(
            SimpleNamespace(fixed_root_pose_w=pose), model, {}, {}, tmp_path / "invalid"
        )
    pose[2] = old["ground_aligned_root_z_m"]
    aligned = base_ground_alignment(model, pose)
    assert aligned["base_ground_clearance_m"] == pytest.approx(0, abs=1e-9)
    pose[2] -= 0.05
    assert base_ground_alignment(model, pose)[
        "base_ground_clearance_m"
    ] == pytest.approx(-0.05)
