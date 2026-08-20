"""Tests for the external Vega-Wuji MJCF contract audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from iltools.core import (
    CollisionClearanceQualification,
    ContactSequence,
    DexterousReference,
    ScenePhysics,
    TrainingQualification,
    load_dexterous_reference_manifest,
    save_dexterous_reference_npz,
)

from write_vega_wuji_reference_manifest import (
    JOINT_POSITION_LIMIT_TOLERANCE,
    MAX_ABS_REFERENCE_JOINT_VELOCITY,
    build_manifest,
)
from validate_vega_wuji_asset import validate_vega_wuji_asset


_JOINT_NAMES = tuple(f"joint_{index:02d}" for index in range(59))
_FINGER_FRAME_SUFFIXES = (
    "thumb_distal",
    "index_finger_distal",
    "middle_finger_distal",
    "ring_finger_distal",
    "pinky_distal",
)
_HAND_FRAME_NAMES = {
    side: tuple(f"{side[0]}_{suffix}" for suffix in _FINGER_FRAME_SUFFIXES)
    for side in ("right", "left")
}
_CONTACT_LINKS = {
    "right": ("r_index_finger_distal",),
    "left": ("l_index_finger_distal",),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_robot_model(
    tmp_path: Path,
    *,
    include_scene: bool = False,
    missing_fingertip: bool = False,
) -> Path:
    special_body_names = (
        "R_ee",
        "L_ee",
        *_HAND_FRAME_NAMES["right"],
        *_HAND_FRAME_NAMES["left"],
    )
    body_lines: list[str] = []
    for index, joint_name in enumerate(_JOINT_NAMES):
        body_name = (
            special_body_names[index]
            if index < len(special_body_names)
            else f"link_{index:02d}"
        )
        if missing_fingertip and body_name == _HAND_FRAME_NAMES["right"][0]:
            body_name = "renamed_right_thumb"
        mount = {
            0: '<body name="r_mount"><site name="right_palm"/></body>',
            1: '<body name="l_mount"><site name="left_palm"/></body>',
        }.get(index, "")
        body_lines.append(
            f'<body name="{body_name}" pos="{index * 0.01:.2f} 0 0">'
            f'<joint name="{joint_name}" type="hinge" axis="0 0 1" '
            'range="-1 1"/>'
            '<geom type="sphere" size="0.005" mass="0.01"/>'
            f"{mount}</body>"
        )
    if include_scene:
        body_lines.append('<body name="desk"/>')
    actuator_lines = [
        f'<position name="actuator_{index:02d}" joint="{joint_name}"/>'
        for index, joint_name in enumerate(_JOINT_NAMES)
    ]
    model_path = tmp_path / "vega_wuji.xml"
    model_path.write_text(
        '<mujoco model="vega_wuji_test">'
        '<compiler angle="radian" autolimits="true"/><worldbody>'
        + "".join(body_lines)
        + "</worldbody><actuator>"
        + "".join(actuator_lines)
        + "</actuator></mujoco>",
        encoding="utf-8",
    )
    return model_path


def _identity_poses(shape: tuple[int, ...]) -> np.ndarray:
    poses = np.zeros((*shape, 7), dtype=np.float32)
    poses[..., 3] = 1.0
    return poses


def _contacts(
    frame_count: int,
    *,
    hand_sides: tuple[str, ...] = ("right", "left"),
    link_names: dict[str, tuple[str, ...]] | None = None,
    active_contacts: bool = True,
) -> ContactSequence:
    links = dict(_CONTACT_LINKS if link_names is None else link_names)
    slot_count = max((len(links[side]) for side in hand_sides), default=1)
    names = np.full((len(hand_sides), slot_count), "", dtype="<U64")
    for side_index, side in enumerate(hand_sides):
        row = links[side]
        names[side_index, : len(row)] = row
    active = np.broadcast_to(
        names[None, ...] != "",
        (frame_count, len(hand_sides), slot_count),
    ).copy()
    if not active_contacts:
        active.fill(False)
    vector_shape = (*active.shape, 3)
    link_normals = np.zeros(vector_shape, dtype=np.float32)
    object_normals = np.zeros(vector_shape, dtype=np.float32)
    link_normals[..., 2] = active
    object_normals[..., 2] = active
    object_indices = np.full(active.shape, -1, dtype=np.int32)
    object_indices[active] = 0
    link_positions = np.zeros(vector_shape, dtype=np.float32)
    object_positions = np.zeros(vector_shape, dtype=np.float32)
    link_positions[active] = (0.2, 0.1, 0.3)
    object_positions[active] = (0.201, 0.1, 0.3)
    return ContactSequence(
        hand_sides=hand_sides,
        link_names=names,
        link_positions_w=link_positions,
        link_normals_w=link_normals,
        object_positions_w=object_positions,
        object_normals_w=object_normals,
        object_indices=object_indices,
        active=active,
    )


def _write_scene_assets(tmp_path: Path) -> tuple[Path, Path]:
    object_path = tmp_path / "cube.usda"
    object_path.write_text('#usda 1.0\ndef Cube "cube" {}\n', encoding="utf-8")
    support_path = tmp_path / "table.usda"
    support_path.write_text('#usda 1.0\ndef Cube "table" {}\n', encoding="utf-8")
    return object_path, support_path


def _reference(
    tmp_path: Path,
    *,
    sequence_id: str = "motion",
    object_path: Path,
    support_path: Path,
) -> DexterousReference:
    frame_count = 2
    wrist_poses = _identity_poses((frame_count,))
    return DexterousReference(
        sequence_id=sequence_id,
        robot_name="vega_wuji",
        fps=20.0,
        joint_names=_JOINT_NAMES,
        qpos=np.zeros((frame_count, len(_JOINT_NAMES)), dtype=np.float32),
        fixed_root_pose_w=_identity_poses(()),
        left_wrist_pose_w=wrist_poses,
        right_wrist_pose_w=wrist_poses,
        left_wrist_frame_name="left_palm",
        right_wrist_frame_name="right_palm",
        object_names=("cube",),
        object_poses_w=_identity_poses((frame_count, 1)),
        object_asset_paths=(str(object_path),),
        object_asset_sha256=(_sha256(object_path),),
        object_radii=np.asarray([0.04], dtype=np.float32),
        left_hand_frame_names=_HAND_FRAME_NAMES["left"],
        left_hand_frame_poses_w=_identity_poses(
            (frame_count, len(_HAND_FRAME_NAMES["left"]))
        ),
        right_hand_frame_names=_HAND_FRAME_NAMES["right"],
        right_hand_frame_poses_w=_identity_poses(
            (frame_count, len(_HAND_FRAME_NAMES["right"]))
        ),
        support_surface_names=("table",),
        support_surface_asset_paths=(str(support_path),),
        support_surface_asset_sha256=(_sha256(support_path),),
        support_surface_poses_w=_identity_poses((1,)),
        scene_physics=ScenePhysics(
            object_mass_kg=np.asarray([0.35]),
            object_center_of_mass_m=np.asarray([[0.0, 0.0, 0.01]]),
            object_diagonal_inertia_kg_m2=np.asarray([[0.001, 0.001, 0.001]]),
            object_static_friction=np.asarray([0.8]),
            object_dynamic_friction=np.asarray([0.6]),
            object_restitution=np.asarray([0.05]),
            support_static_friction=np.asarray([0.9]),
            support_dynamic_friction=np.asarray([0.7]),
            support_restitution=np.asarray([0.02]),
        ),
        contacts=_contacts(frame_count),
        training_qualification=TrainingQualification(
            runtime_qualified=True,
            isaac_runtime_qualified=True,
            inspection_only=False,
            contact_geometry_provenance="manual test contact annotation",
            collision_clearance=CollisionClearanceQualification(
                qualified=True,
                method="signed-distance replay",
                scope="robot, object, and support geometry for every frame",
                provenance="Isaac Newton test replay",
                checked_frame_count=frame_count,
                minimum_signed_distance_m=0.0,
                penetration_tolerance_m=0.0002,
            ),
        ),
    )


def _save_reference(reference: DexterousReference, tmp_path: Path) -> Path:
    return save_dexterous_reference_npz(
        reference,
        tmp_path / f"{reference.sequence_id}.npz",
    )


def test_validate_robot_only_mjcf_and_reference(tmp_path) -> None:
    model_path = tmp_path / "robot.xml"
    model_path.write_text(
        """
        <mujoco model="vega_wuji_test">
          <compiler angle="radian" autolimits="true"/>
          <worldbody>
            <body name="base">
              <body name="R_ee">
                <joint name="joint_a" type="hinge" axis="0 0 1" range="-1 2"/>
                <geom type="sphere" size="0.1" mass="1"/>
                <body name="r_mount">
                  <site name="right_palm" pos="0 0 0"/>
                </body>
              </body>
              <body name="L_ee">
                <joint name="joint_b" type="hinge" axis="0 0 1" range="-0.5 0.75"/>
                <geom type="sphere" size="0.1" mass="1"/>
                <body name="l_mount">
                  <site name="left_palm" pos="0 0 0"/>
                </body>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="actuator_b" joint="joint_b"/>
            <position name="actuator_a" joint="joint_a"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, qpos=np.zeros((2, 2)), joint_names=["joint_b", "joint_a"])

    record = validate_vega_wuji_asset(
        model_path,
        reference_path=reference_path,
        expected_actuator_count=2,
    )

    assert record["valid"] is True
    assert record["actuator_joint_names"] == ["joint_b", "joint_a"]
    assert record["actuator_joint_position_limited"] == [True, True]
    np.testing.assert_allclose(
        record["actuator_joint_position_limits"],
        [[-0.5, 0.75], [-1.0, 2.0]],
    )
    assert record["actuator_joint_types"] == ["mjJNT_HINGE", "mjJNT_HINGE"]
    assert record["forbidden_scene_bodies"] == []
    assert record["wrist_frames"]["right"]["runtime_body"] == "r_mount"
    assert record["wrist_frames"]["right"]["identity_on_runtime_body"] is True


def test_vendored_palm_sites_are_mount_frames_not_ee_frames() -> None:
    model_path = (
        Path(__file__).resolve().parents[2]
        / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
        / "vega_u_wuji_v2_beta1_with_mount.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for site_name, mount_name, ee_name in (
        ("right_palm", "r_mount", "R_ee"),
        ("left_palm", "l_mount", "L_ee"),
    ):
        site_id = model.site(site_name).id
        mount_id = model.body(mount_name).id
        ee_id = model.body(ee_name).id
        assert int(model.site_bodyid[site_id]) == mount_id
        assert int(model.body_parentid[mount_id]) == ee_id
        np.testing.assert_allclose(model.site_pos[site_id], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            np.abs(model.site_quat[site_id]),
            (1.0, 0.0, 0.0, 0.0),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(data.site_xpos[site_id], data.xpos[mount_id])
        np.testing.assert_allclose(data.site_xmat[site_id], data.xmat[mount_id])
        np.testing.assert_allclose(
            np.abs(model.body_quat[mount_id]),
            (0.0, 1.0, 0.0, 0.0),
            atol=1.0e-12,
        )
        assert not np.allclose(data.xmat[mount_id], data.xmat[ee_id])


def test_validate_rejects_non_identity_palm_site(tmp_path) -> None:
    model_path = _write_robot_model(tmp_path)
    source = model_path.read_text(encoding="utf-8")
    model_path.write_text(
        source.replace(
            '<site name="right_palm"/>',
            '<site name="right_palm" pos="0.01 0 0"/>',
        ),
        encoding="utf-8",
    )

    record = validate_vega_wuji_asset(model_path)

    assert record["valid"] is False
    assert "right_palm must be an identity site on r_mount" in record["issues"]


def test_validate_rejects_actuated_joint_without_position_limit(tmp_path) -> None:
    model_path = _write_robot_model(tmp_path)
    source = model_path.read_text(encoding="utf-8")
    model_path.write_text(
        source.replace(' range="-1 1"', "", 1),
        encoding="utf-8",
    )

    record = validate_vega_wuji_asset(model_path)

    assert record["valid"] is False
    assert "actuated robot joints without position limits" in record["issues"][0]
    assert _JOINT_NAMES[0] in record["issues"][0]


def test_validate_rejects_scene_body_and_wrong_reference_order(tmp_path) -> None:
    model_path = tmp_path / "scene.xml"
    model_path.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base">
                  <body name="R_ee">
                    <joint name="joint_a" type="hinge"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                    <body name="r_mount"><site name="right_palm"/></body>
                  </body>
                  <body name="L_ee">
                    <joint name="joint_b" type="hinge"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                    <body name="l_mount"><site name="left_palm"/></body>
              </body>
            </body>
            <body name="desk"/>
          </worldbody>
          <actuator>
            <position name="actuator_a" joint="joint_a"/>
            <position name="actuator_b" joint="joint_b"/>
          </actuator>
        </mujoco>
        """,
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, qpos=np.zeros((2, 2)), joint_names=["joint_b", "joint_a"])

    record = validate_vega_wuji_asset(
        model_path,
        reference_path=reference_path,
        expected_actuator_count=2,
    )

    assert record["valid"] is False
    assert "desk" in record["forbidden_scene_bodies"]
    assert "reference joint_names do not match actuator joint order" in record["issues"]


def test_reference_manifest_uses_iltools_schema_and_relocatable_paths(tmp_path) -> None:
    model_path = _write_robot_model(tmp_path)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference_paths = [
        _save_reference(
            _reference(
                tmp_path,
                sequence_id=sequence_id,
                object_path=object_path,
                support_path=support_path,
            ),
            tmp_path,
        )
        for sequence_id in ("first", "second")
    ]
    output_path = tmp_path / "manifests" / "reference.json"

    payload = build_manifest(
        reference_paths, output_path=output_path, model_path=model_path
    )

    assert payload["schema"] == "iltools_dexterous_reference_manifest/v1"
    assert payload["robot_name"] == "vega_wuji"
    assert payload["reference_fps"] == 20.0
    assert payload["joint_names"] == list(_JOINT_NAMES)
    assert payload["left_wrist_frame_name"] == "left_palm"
    assert payload["right_wrist_frame_name"] == "right_palm"
    assert payload["motion_count"] == 2
    assert payload["model"] == "../vega_wuji.xml"
    assert payload["model_sha256"] == _sha256(model_path)
    assert not str(payload["model"]).startswith("/")
    assert payload["motions"][0]["path"] == "../first.npz"
    assert payload["metadata"]["training_qualified"] is True
    assert payload["metadata"]["inspection_only"] is False
    manifest = load_dexterous_reference_manifest(output_path)
    loaded = manifest.load_references()
    assert len(loaded) == 2
    assert loaded[0].object_asset_sha256 == (_sha256(object_path),)

    with pytest.raises(ValueError, match="must declare its robot model"):
        build_manifest(
            reference_paths,
            output_path=tmp_path / "missing_model.json",
            model_path=None,
        )


def test_manifest_writer_rejects_metadata_only_inspection_reference_by_default(
    tmp_path,
) -> None:
    model_path = _write_robot_model(tmp_path)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference = _reference(
        tmp_path,
        object_path=object_path,
        support_path=support_path,
    )
    reference.training_qualification = None
    reference.metadata = {
        "runtime_qualified": False,
        "isaac_runtime_qualified": False,
        "inspection_only": True,
        "contacts": {"contact_geometry": "unavailable"},
        "geometry_clearance_audit": {"passed": True},
    }
    reference_path = _save_reference(reference, tmp_path)
    output_path = tmp_path / "training_manifest.json"

    with pytest.raises(ValueError, match="no typed training qualification"):
        build_manifest(
            [reference_path],
            output_path=output_path,
            model_path=model_path,
        )

    assert not output_path.exists()


def test_manifest_writer_has_explicit_inspection_only_escape(tmp_path) -> None:
    model_path = _write_robot_model(tmp_path)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference = _reference(
        tmp_path,
        object_path=object_path,
        support_path=support_path,
    )
    reference.training_qualification = None
    reference.metadata = {
        "runtime_qualified": False,
        "isaac_runtime_qualified": False,
        "inspection_only": True,
    }
    reference_path = _save_reference(reference, tmp_path)
    output_path = tmp_path / "inspection_manifest.json"

    payload = build_manifest(
        [reference_path],
        output_path=output_path,
        model_path=model_path,
        allow_unqualified_inspection=True,
    )

    assert payload["metadata"]["training_qualified"] is False
    assert payload["metadata"]["inspection_only"] is True


def test_manifest_writer_rejects_undeclared_scene_physics(tmp_path) -> None:
    model_path = _write_robot_model(tmp_path)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference = _reference(
        tmp_path,
        object_path=object_path,
        support_path=support_path,
    )
    reference.scene_physics = None
    reference_path = _save_reference(reference, tmp_path)
    output_path = tmp_path / "missing_scene_physics.json"

    with pytest.raises(ValueError, match="no typed ScenePhysics"):
        build_manifest(
            [reference_path],
            output_path=output_path,
            model_path=model_path,
        )

    assert not output_path.exists()


def test_manifest_writer_rejects_invalid_robot_before_output(tmp_path) -> None:
    model_path = _write_robot_model(tmp_path, include_scene=True)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference_path = _save_reference(
        _reference(
            tmp_path,
            object_path=object_path,
            support_path=support_path,
        ),
        tmp_path,
    )
    output_path = tmp_path / "invalid_model.json"

    with pytest.raises(ValueError, match="robot-only MJCF"):
        build_manifest(
            [reference_path],
            output_path=output_path,
            model_path=model_path,
        )

    assert not output_path.exists()


def test_manifest_writer_rejects_missing_fingertip_body_before_output(
    tmp_path,
) -> None:
    model_path = _write_robot_model(tmp_path, missing_fingertip=True)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference_path = _save_reference(
        _reference(
            tmp_path,
            object_path=object_path,
            support_path=support_path,
        ),
        tmp_path,
    )
    output_path = tmp_path / "missing_fingertip_body.json"

    with pytest.raises(ValueError, match="missing required fingertip bodies"):
        build_manifest(
            [reference_path],
            output_path=output_path,
            model_path=model_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("robot_name", "expected 'vega_wuji'"),
        ("fps", "expected 20.0 Hz"),
        ("joint_order", "actuator joint order"),
        ("qpos_limit", "outside the Vega-Wuji joint position limits"),
        ("qvel_nonfinite", "qvel contains non-finite values"),
        ("qvel_implausible", "qvel is implausible"),
        ("qvel_inconsistent", "qvel is inconsistent with qpos and 20 Hz"),
        ("wrist_frame", "right wrist frame"),
        ("no_object", "at least one rigid object"),
        ("object_asset", "object without an asset path"),
        ("object_radius", "non-positive object radius"),
        ("right_fingertip", "missing right hand frames"),
        ("left_contact_group", "has no left contact group"),
        ("right_contact_names", "no named right contact links"),
        ("missing_contact_body", "contact links are not Vega-Wuji MJCF bodies"),
    ),
)
def test_manifest_writer_rejects_invalid_reference_before_output(
    tmp_path,
    case: str,
    match: str,
) -> None:
    model_path = _write_robot_model(tmp_path)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference = _reference(
        tmp_path,
        object_path=object_path,
        support_path=support_path,
    )
    if case == "robot_name":
        reference.robot_name = "other_robot"
    elif case == "fps":
        reference.fps = 50.0
    elif case == "joint_order":
        reference.joint_names = tuple(reversed(reference.joint_names))
        reference.qpos = reference.qpos[:, ::-1]
        reference.qvel = reference.qvel[:, ::-1]
    elif case == "qpos_limit":
        reference.qpos[:, 0] = 1.0 + 2.0 * JOINT_POSITION_LIMIT_TOLERANCE
    elif case == "qvel_nonfinite":
        reference.qvel[0, 0] = np.nan
    elif case == "qvel_implausible":
        reference.qvel[0, 0] = MAX_ABS_REFERENCE_JOINT_VELOCITY + 1.0
    elif case == "qvel_inconsistent":
        reference.qpos[1, 0] = 0.1
    elif case == "wrist_frame":
        reference.right_wrist_frame_name = "R_ee"
    elif case == "no_object":
        reference.object_names = ()
        reference.object_poses_w = _identity_poses((reference.frame_count, 0))
        reference.object_twists_w = np.empty(
            (reference.frame_count, 0, 6), dtype=np.float32
        )
        reference.object_asset_paths = ()
        reference.object_asset_sha256 = ()
        reference.object_scales = np.empty((0, 3), dtype=np.float32)
        reference.object_radii = np.empty((0,), dtype=np.float32)
        reference.scene_physics = None
        reference.contacts = _contacts(
            reference.frame_count,
            active_contacts=False,
        )
    elif case == "object_asset":
        reference.object_asset_paths = ("",)
        reference.object_asset_sha256 = ("",)
    elif case == "object_radius":
        reference.object_radii[:] = 0.0
    elif case == "right_fingertip":
        reference.right_hand_frame_names = reference.right_hand_frame_names[:-1]
        reference.right_hand_frame_poses_w = reference.right_hand_frame_poses_w[:, :-1]
    elif case == "left_contact_group":
        reference.contacts = _contacts(
            reference.frame_count,
            hand_sides=("right",),
        )
    elif case == "right_contact_names":
        reference.contacts = _contacts(
            reference.frame_count,
            link_names={"right": ("",), "left": _CONTACT_LINKS["left"]},
        )
    elif case == "missing_contact_body":
        reference.contacts = _contacts(
            reference.frame_count,
            link_names={
                "right": ("r_unknown_contact",),
                "left": _CONTACT_LINKS["left"],
            },
        )
    reference_path = _save_reference(reference, tmp_path)
    output_path = tmp_path / f"{case}.json"

    with pytest.raises(ValueError, match=match):
        build_manifest(
            [reference_path],
            output_path=output_path,
            model_path=model_path,
        )

    assert not output_path.exists()


def test_manifest_writer_accepts_tiny_qpos_limit_roundoff(tmp_path) -> None:
    model_path = _write_robot_model(tmp_path)
    object_path, support_path = _write_scene_assets(tmp_path)
    reference = _reference(
        tmp_path,
        object_path=object_path,
        support_path=support_path,
    )
    reference.qpos[:, 0] = 1.0 + 0.5 * JOINT_POSITION_LIMIT_TOLERANCE
    reference_path = _save_reference(reference, tmp_path)
    output_path = tmp_path / "position_roundoff.json"

    payload = build_manifest(
        [reference_path],
        output_path=output_path,
        model_path=model_path,
    )

    assert payload["metadata"]["training_qualified"] is True
    assert output_path.is_file()


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("scene_name", "changes object_names"),
        ("root_pose", "changes fixed_root_pose_w"),
        ("contact_links", "changes right contact links"),
    ),
)
def test_manifest_writer_rejects_layout_changes_between_motions(
    tmp_path,
    case: str,
    match: str,
) -> None:
    model_path = _write_robot_model(tmp_path)
    object_path, support_path = _write_scene_assets(tmp_path)
    first = _reference(
        tmp_path,
        sequence_id="first",
        object_path=object_path,
        support_path=support_path,
    )
    second = _reference(
        tmp_path,
        sequence_id="second",
        object_path=object_path,
        support_path=support_path,
    )
    if case == "scene_name":
        second.object_names = ("renamed_cube",)
    elif case == "root_pose":
        second.fixed_root_pose_w[0] = 0.01
    elif case == "contact_links":
        second.contacts = _contacts(
            second.frame_count,
            link_names={
                "right": ("r_middle_finger_distal",),
                "left": _CONTACT_LINKS["left"],
            },
        )
    reference_paths = [
        _save_reference(reference, tmp_path) for reference in (first, second)
    ]
    output_path = tmp_path / f"shared_{case}.json"

    with pytest.raises(ValueError, match=match):
        build_manifest(
            reference_paths,
            output_path=output_path,
            model_path=model_path,
        )

    assert not output_path.exists()
