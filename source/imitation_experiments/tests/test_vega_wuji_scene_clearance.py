import hashlib
from pathlib import Path

import numpy as np
import pytest

from imitation_experiments.audit.vega_wuji_scene_clearance import (
    Cylinder,
    _compile_xml_include_model,
    audit_reference,
    compose_scaled_translation,
    load_support_cylinder,
    mesh_above_cylinder_clearance,
    quat_wxyz_to_matrix,
    relative_pose_wxyz,
)
from imitation_experiments.paths import REPO_ROOT


def test_quaternion_rotation_and_scaled_translation() -> None:
    quat = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    rotation = quat_wxyz_to_matrix(quat)
    assert rotation @ np.asarray([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    pose = np.concatenate((np.asarray([2.0, 3.0, 4.0]), quat))
    point = compose_scaled_translation(pose, [0.5, 0.5, 0.5], [2.0, 0.0, 0.0])
    assert point == pytest.approx([2.0, 4.0, 4.0])


def test_load_support_cylinder_applies_root_pose_and_uniform_scale(
    tmp_path: Path,
) -> None:
    support = tmp_path / "support.usda"
    support.write_text(
        """#usda 1.0
def Xform \"root\" {
  def Cylinder \"surface\" {
    uniform token axis = \"Z\"
    double height = 0.02
    double radius = 0.4
    double3 xformOp:translate = (1.0, 2.0, 3.0)
  }
}
""",
        encoding="utf-8",
    )
    cylinder = load_support_cylinder(
        support,
        [10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    )
    assert cylinder.center == pytest.approx((10.5, 21.0, 31.5))
    assert cylinder.radius == pytest.approx(0.2)
    assert cylinder.half_height == pytest.approx(0.005)


def test_mesh_above_cylinder_clearance() -> None:
    cylinder = Cylinder(
        center=(1.0, 2.0, 3.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        radius=0.5,
        half_height=0.1,
    )
    points = np.asarray([[1.0, 2.0, 3.125], [1.3, 2.0, 3.2]])
    clearance, max_radius = mesh_above_cylinder_clearance(points, cylinder)
    assert clearance == pytest.approx(0.025)
    assert max_radius == pytest.approx(0.3)


def test_xml_include_keeps_far_disabled_mesh_distance(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    (tmp_path / "tetra.obj").write_text(
        """v 0 0 0
v 0.1 0 0
v 0 0.1 0
v 0 0 0.1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
""",
        encoding="utf-8",
    )
    robot = tmp_path / "robot.xml"
    robot.write_text(
        """<mujoco model="robot">
  <asset><mesh name="tetra" file="tetra.obj"/></asset>
  <worldbody>
    <body name="robot"><geom name="disabled_mesh" type="mesh" mesh="tetra"
                              group="4" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    model = _compile_xml_include_model(
        mujoco,
        robot,
        '<worldbody><geom name="far" type="sphere" pos="1 0 0" size="0.1"/>'
        "</worldbody>",
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    disabled_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "disabled_mesh")
    far_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "far")
    distance = mujoco.mj_geomDistance(
        model, data, disabled_id, far_id, 2.0, np.empty(6)
    )
    assert model.geom_group[disabled_id] == 4
    assert model.geom_contype[disabled_id] == 0
    assert model.geom_conaffinity[disabled_id] == 0
    assert distance > 0.2


def test_relative_pose_keeps_robot_at_model_origin() -> None:
    yaw_90 = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    parent = np.concatenate(([1.0, 2.0, 3.0], yaw_90))
    child = np.concatenate(([1.0, 3.0, 4.0], yaw_90))
    relative = relative_pose_wxyz(parent, child)
    assert relative[:3] == pytest.approx([1.0, 0.0, 1.0])
    assert relative[3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


_REPLACEMENT_REFERENCE = Path(
    "/tmp/corn_can_handover_vega_wuji_dual_path_geometry_qualified_inspection.npz"
)
_REPLACEMENT_SHA256 = "5f73250f85c980593d98ca3419bbef0027d5e0340a5e0b3ff583c1ad79f159a5"


@pytest.mark.skipif(
    not _REPLACEMENT_REFERENCE.is_file(),
    reason="The local replacement Vega/Wuji Reference is not available.",
)
def test_replacement_reference_passes_its_declared_threshold() -> None:
    digest = hashlib.sha256(_REPLACEMENT_REFERENCE.read_bytes()).hexdigest()
    assert digest == _REPLACEMENT_SHA256
    robot_mjcf = (
        REPO_ROOT
        / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
        / "vega_u_wuji_v2_beta1_with_mount.xml"
    )
    result = audit_reference(_REPLACEMENT_REFERENCE, robot_mjcf)
    assert result["required_clearance_m"] == pytest.approx(0.005)
    assert result["robot_scene_threshold_gate_passed"] is True
    for scene in ("intended_robot_can", "intended_robot_support"):
        summary = result["categories"][scene]
        assert summary["passed"] is True
        assert summary["minimum_clearance_lower_bound_m"] == pytest.approx(0.005)
        assert summary["minimum_query_below_cutoff"] is None
    assert result["can_support"]["obj_vertex_vertical"]["minimum"][
        "clearance_m"
    ] == pytest.approx(0.0008340203224225552)
