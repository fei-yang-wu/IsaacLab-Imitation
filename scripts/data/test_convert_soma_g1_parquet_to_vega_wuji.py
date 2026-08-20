"""Focused tests for the pinned SOMA-to-Vega-Wuji converter."""

from __future__ import annotations

import hashlib
import pickle

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from convert_soma_g1_parquet_to_vega_wuji import (
    DEFAULT_LOCAL_GEOMETRY_SCALE,
    DEFAULT_OBJECT_ANCHOR_TARGET,
    DEFAULT_SOURCE,
    EXPECTED_SOMA_JOINT_NAMES_SHA256,
    PINNED_SUPPORT_CENTER_SOURCE,
    SOMA_JOINT_NAMES_SHA256,
    ContactPhaseProjection,
    _contact_phase_weights,
    _densify_scene_trajectory,
    _collision_safe_finger_projection,
    _has_exact_contact_geometry,
    _inspection_contacts,
    _project_palm_positions_nonpenetrating,
    _safe_payload,
    _target_times,
    _usd_safe_identifier,
    convert_soma_g1_parquet,
)


def test_pinned_soma_joint_order_hash() -> None:
    assert SOMA_JOINT_NAMES_SHA256 == EXPECTED_SOMA_JOINT_NAMES_SHA256


def test_safe_payload_rejects_unpinned_and_global_pickles() -> None:
    primitive = pickle.dumps({"safe": [1, 2, 3]})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _safe_payload(primitive, expected_sha256="0" * 64)

    global_pickle = pickle.dumps(len)
    digest = hashlib.sha256(global_pickle).hexdigest()
    with pytest.raises(pickle.UnpicklingError, match="forbidden global"):
        _safe_payload(global_pickle, expected_sha256=digest)


def test_target_times_use_physical_time_and_drop_partial_tail() -> None:
    source, target = _target_times(
        start_frame=522,
        end_frame_exclusive=840,
        source_fps=200.0,
    )
    assert source.shape == (318,)
    assert target.shape == (32,)
    np.testing.assert_allclose(np.diff(target), 0.05, atol=1.0e-12)
    assert target[0] == pytest.approx(522 / 200.0)
    assert target[-1] == pytest.approx(832 / 200.0)
    assert target[-1] < source[-1]


def test_dense_scene_audit_samples_intervals_and_preserves_endpoints() -> None:
    qpos = np.asarray(((0.0, 1.0), (1.0, 3.0)), dtype=np.float64)
    poses = np.zeros((2, 1, 7), dtype=np.float64)
    poses[:, 0, 0] = (0.0, 1.0)
    poses[..., 3] = 1.0
    dense_qpos, dense_poses = _densify_scene_trajectory(
        qpos,
        poses,
        source_fps=2.0,
        audit_fps=4.0,
    )
    np.testing.assert_allclose(dense_qpos, ((0.0, 1.0), (0.5, 2.0), (1.0, 3.0)))
    np.testing.assert_allclose(dense_poses[:, 0, 0], (0.0, 0.5, 1.0))
    np.testing.assert_allclose(dense_poses[..., 3], 1.0)


def test_contact_phase_weights_are_side_local_and_smooth() -> None:
    active = np.zeros((2, 201), dtype=np.float64)
    active[0, 80:121] = 1.0
    weights = _contact_phase_weights(
        active,
        dilation_frames=41,
        gaussian_sigma_frames=15.0,
    )
    assert weights.shape == active.shape
    assert np.all((0.0 <= weights) & (weights <= 1.0))
    assert weights[0, 100] > 0.99
    assert 0.0 < weights[0, 50] < weights[0, 70] < weights[0, 100]
    np.testing.assert_allclose(weights[1], 0.0, atol=0.0)


def test_palm_projection_enforces_object_and_support_standoffs() -> None:
    objects = np.zeros((3, 1, 3), dtype=np.float64)
    objects[..., 2] = 1.0
    active = np.broadcast_to(objects, (3, 2, 3)).copy()
    active[..., 0] += 0.1
    neutral = np.asarray(((0.8, 0.2, 1.0), (0.8, -0.2, 1.0)))
    weights = np.asarray(((1.0, 0.5, 0.0), (0.0, 0.5, 1.0)))
    projected, projected_active, rest = _project_palm_positions_nonpenetrating(
        active,
        objects,
        neutral,
        weights,
        support_top_z=1.1,
        palm_object_standoff_m=0.25,
        palm_support_standoff_m=0.2,
    )
    assert np.all(np.linalg.norm(projected_active - objects, axis=-1) >= 0.25 - 1.0e-12)
    assert np.all(projected_active[..., 2] >= 1.3)
    np.testing.assert_allclose(rest[:, 2], 1.3)
    np.testing.assert_allclose(projected[0, 0], projected_active[0, 0])
    np.testing.assert_allclose(projected[0, 1], rest[1])


def test_finger_closure_line_search_returns_a_qualified_uniform_scale() -> None:
    class ThresholdAuditor:
        def audit(
            self, qpos: np.ndarray, object_poses: np.ndarray
        ) -> dict[str, object]:
            del object_poses
            return {
                "qualified": bool(np.max(np.abs(qpos)) <= 0.4),
                "scenes": {},
            }

    qpos = np.ones((2, 2), dtype=np.float64)
    object_poses = np.zeros((2, 1, 7), dtype=np.float64)
    object_poses[..., 3] = 1.0
    projected, scales, audit = _collision_safe_finger_projection(
        qpos,
        joint_names=("l_index_finger_mcp_flex", "r_index_finger_mcp_flex"),
        object_poses=object_poses,
        auditor=ThresholdAuditor(),  # type: ignore[arg-type]
    )
    assert audit["qualified"] is True
    assert 0.398 < scales["left"] < 0.4
    assert 0.398 < scales["right"] < 0.4
    assert np.max(np.abs(projected)) < 0.4


def test_geom_distance_uses_scalar_when_far_pair_leaves_fromto_zero() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="left" type="sphere" size="0.1" pos="0 0 0"/>
            <geom name="right" type="sphere" size="0.1" pos="2 0 0"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6, dtype=np.float64)
    distance = mujoco.mj_geomDistance(
        model,
        data,
        model.geom("left").id,
        model.geom("right").id,
        0.1,
        fromto,
    )
    assert distance == pytest.approx(0.1)
    np.testing.assert_array_equal(fromto, 0.0)


def test_two_scale_projection_preserves_local_geometry() -> None:
    rotation = Rotation.from_euler("z", 0.4)
    source_anchor = np.asarray((-1.0, 0.2, 0.9))
    projection = ContactPhaseProjection(
        rotation=rotation,
        source_object_anchor=source_anchor,
        target_object_anchor=DEFAULT_OBJECT_ANCHOR_TARGET,
        global_motion_scale=0.4,
        local_geometry_scale=DEFAULT_LOCAL_GEOMETRY_SCALE,
    )
    objects = np.asarray([[[*source_anchor]]], dtype=np.float64)
    wrists = np.asarray(
        [
            [
                [
                    source_anchor[0] + 0.2,
                    source_anchor[1] - 0.1,
                    source_anchor[2] + 0.05,
                ],
                [
                    source_anchor[0] - 0.1,
                    source_anchor[1] + 0.3,
                    source_anchor[2] - 0.02,
                ],
            ]
        ],
        dtype=np.float64,
    )
    mapped_object = projection.object_positions(objects)
    mapped_wrists = projection.active_wrist_positions(wrists, objects)
    np.testing.assert_allclose(mapped_object[0, 0], DEFAULT_OBJECT_ANCHOR_TARGET)
    source_distances = np.linalg.norm(wrists[0] - objects[0, 0], axis=-1)
    target_distances = np.linalg.norm(mapped_wrists[0] - mapped_object[0, 0], axis=-1)
    np.testing.assert_allclose(
        target_distances,
        DEFAULT_LOCAL_GEOMETRY_SCALE * source_distances,
        atol=1.0e-12,
    )
    support_pose = projection.support_asset_pose_wxyz()
    mapped_support_center = support_pose[:3] + DEFAULT_LOCAL_GEOMETRY_SCALE * (
        rotation.apply(PINNED_SUPPORT_CENTER_SOURCE)
    )
    expected_support_center = DEFAULT_OBJECT_ANCHOR_TARGET + (
        DEFAULT_LOCAL_GEOMETRY_SCALE
        * rotation.apply(PINNED_SUPPORT_CENTER_SOURCE - source_anchor)
    )
    np.testing.assert_allclose(mapped_support_center, expected_support_center)


def test_inspection_contacts_are_named_but_never_active() -> None:
    contacts = _inspection_contacts(7)
    contacts.validate(frame_count=7, object_count=1)
    assert contacts.link_names.shape == (2, 5)
    assert np.all(contacts.link_names != "")
    assert not np.any(contacts.active)
    assert np.all(contacts.object_indices == -1)
    assert np.all(contacts.link_positions_w == 0.0)
    assert np.all(contacts.object_positions_w == 0.0)


def test_exact_contact_geometry_detection_fails_on_partial_groups() -> None:
    names = (
        "hand_contact_link_names",
        "hand_link_contact_positions",
        "hand_link_contact_normals",
        "hand_object_contact_positions",
        "hand_object_contact_normals",
        "hand_object_contact_part_ids",
    )
    empty = {name: [[], []] for name in names}
    assert _has_exact_contact_geometry(empty) is False
    partial = dict(empty)
    partial["hand_contact_link_names"] = [["left_tip"], []]
    with pytest.raises(ValueError, match="incomplete"):
        _has_exact_contact_geometry(partial)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("corn-can", "corn_can"),
        ("9 can", "_9_can"),
        ("---", "corn_can"),
    ),
)
def test_usd_safe_identifier(source: str, expected: str) -> None:
    assert _usd_safe_identifier(source, fallback="corn_can") == expected


@pytest.mark.skipif(not DEFAULT_SOURCE.is_file(), reason="Pinned SOMA source absent")
def test_real_inspection_conversion_is_bounded_and_fails_closed() -> None:
    with pytest.raises(ValueError, match="Refusing to invent"):
        convert_soma_g1_parquet(DEFAULT_SOURCE)

    reference = convert_soma_g1_parquet(DEFAULT_SOURCE, inspection_only=True)
    assert reference.object_names == ("corn_can",)
    assert reference.frame_count == 32
    assert reference.fps == 20.0
    assert reference.metadata["runtime_qualified"] is False
    assert reference.metadata["inspection_scene_geometry_qualified"] is True
    assert reference.metadata["contacts"]["contact_geometry"] == "unavailable"
    audit = reference.metadata["geometry_clearance_audit"]
    assert audit["qualified"] is True
    assert audit["required_clearance_m"] == pytest.approx(0.005)
    assert audit["source_200hz"]["frame_count"] == 318
    assert audit["emitted_20hz"]["frame_count"] == 32
    assert audit["source_path_dense_1khz"]["frame_count"] == 1586
    assert audit["source_path_dense_1khz"]["fps"] == 1000.0
    assert audit["emitted_path_dense_1khz"]["frame_count"] == 1551
    assert audit["emitted_path_dense_1khz"]["fps"] == 1000.0
    for rate in (
        "source_200hz",
        "emitted_20hz",
        "source_path_dense_1khz",
        "emitted_path_dense_1khz",
    ):
        assert audit[rate]["robot_collision_geometry_count"] == 76
        for scene in ("can", "support"):
            assert audit[rate]["scenes"][scene]["violation_frame_count"] == 0
            assert (
                audit[rate]["scenes"][scene]["minimum_clearance_lower_bound_m"]
                >= audit["required_clearance_m"]
            )
    for rate in (
        "rendered_geometry_20hz",
        "source_path_rendered_dense_1khz",
        "emitted_path_rendered_dense_1khz",
    ):
        assert audit[rate]["robot_collision_geometry_count"] == 142
        assert audit[rate]["qualified"] is True
        for scene in ("can", "support"):
            assert audit[rate]["scenes"][scene]["violation_frame_count"] == 0
            assert (
                audit[rate]["scenes"][scene]["minimum_clearance_lower_bound_m"]
                >= audit["required_clearance_m"]
            )
    for rate, frame_count in (
        ("source_path_can_support_dense_1khz", 1586),
        ("emitted_path_can_support_dense_1khz", 1551),
    ):
        assert audit[rate]["frame_count"] == frame_count
        assert audit[rate]["qualified"] is True
        assert audit[rate]["minimum_vertical_clearance_m"] >= 0.0
    assert reference.metadata["wrist_ik"]["max_position_error_m"] < 0.05
    assert not np.any(reference.contacts.active)
    assert np.all(reference.contacts.object_indices == -1)
    assert np.allclose(reference.object_scales, 0.75)
    assert np.allclose(reference.support_surface_scales, 0.75)
    assert len(reference.collision_asset_dependencies) == 1
    dependency = reference.collision_asset_dependencies[0]
    assert dependency.asset_role == "object"
    assert dependency.asset_index == 0
    assert dependency.uri == "textured_mesh.obj"
    assert dependency.sha256 == (
        "d0668549cbcf73a9c7935b6690485a6a51e475fcbf2e68d73108be0eb5b532ed"
    )
