"""Focused tests for the pinned SOMA-to-Vega-Wuji converter."""

from __future__ import annotations

import hashlib
import importlib.util
import pickle

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from convert_soma_g1_parquet_to_vega_wuji import (
    DEFAULT_FINGER_POSTURE_PRIOR,
    DEFAULT_IK_ORIENTATION_WEIGHT,
    DEFAULT_LOCAL_GEOMETRY_SCALE,
    DEFAULT_OBJECT_ANCHOR_TARGET,
    DEFAULT_SOMA_TO_WUJI_HAND_SCALE,
    DEFAULT_SEQUENCE_KEY,
    DEFAULT_SOURCE,
    EXPECTED_SOMA_JOINT_NAMES_SHA256,
    PINNED_END_FRAME_EXCLUSIVE,
    PINNED_OBJECT_ASSET_SHA256,
    PINNED_PARQUET_SHA256,
    PINNED_PAYLOAD_SHA256,
    PINNED_SEQUENCE_ID,
    PINNED_SEQUENCES,
    PINNED_START_FRAME,
    PINNED_SUPPORT_CENTER_SOURCE,
    PINNED_SUPPORT_HEIGHT_SOURCE,
    PINNED_SUPPORT_RADIUS_SOURCE,
    PINNED_SUPPORT_SHA256,
    SOMA_CHORD_LINK_KEYS,
    SOMA_HAND_KNUCKLE_NAMES,
    SOMA_JOINT_NAMES_SHA256,
    ContactPhaseProjection,
    _contact_phase_weights,
    _calibrate_soma_wrist_orientations,
    _contact_recovery_gate,
    _densify_scene_trajectory,
    _collision_safe_finger_projection,
    _has_exact_contact_geometry,
    _finger_specs,
    _inspection_contacts,
    _infer_phase_object_anchors,
    _phase_object_anchor_trajectory,
    _project_palm_positions_nonpenetrating,
    _resample_chord_targets_nearest,
    _audit_can_support_vertical_clearance,
    _safe_payload,
    _sha256_file,
    _source_frame_change_bound,
    _target_times,
    _usd_safe_identifier,
    _verify_locked_joint_trajectory,
    build_parser,
    convert_soma_g1_parquet,
)


def test_pinned_soma_joint_order_hash() -> None:
    assert SOMA_JOINT_NAMES_SHA256 == EXPECTED_SOMA_JOINT_NAMES_SHA256


def test_legacy_pin_names_alias_the_default_registry_entry() -> None:
    """The historical PINNED_* names must keep the corn-can contract."""

    pin = PINNED_SEQUENCES[DEFAULT_SEQUENCE_KEY]
    assert DEFAULT_SEQUENCE_KEY == "corn_can_handover"
    assert pin.sequence_id == PINNED_SEQUENCE_ID
    assert pin.source_path == DEFAULT_SOURCE
    assert pin.parquet_sha256 == PINNED_PARQUET_SHA256
    assert pin.payload_sha256 == PINNED_PAYLOAD_SHA256
    assert pin.support_sha256 == PINNED_SUPPORT_SHA256
    assert pin.object_asset_sha256 == PINNED_OBJECT_ASSET_SHA256
    assert (pin.start_frame, pin.end_frame_exclusive) == (
        PINNED_START_FRAME,
        PINNED_END_FRAME_EXCLUSIVE,
    )
    assert pin.support_height_source == PINNED_SUPPORT_HEIGHT_SOURCE
    assert pin.support_radius_source == PINNED_SUPPORT_RADIUS_SOURCE
    assert np.allclose(pin.support_center_source, PINNED_SUPPORT_CENTER_SOURCE)
    assert np.allclose(pin.object_anchor_target, DEFAULT_OBJECT_ANCHOR_TARGET)


def test_every_pin_declares_a_distinct_complete_contract() -> None:
    """A registry entry must be self-sufficient and not reuse another's data."""

    for key, pin in PINNED_SEQUENCES.items():
        assert pin.sequence_id in str(pin.source_path), key
        assert pin.sequence_id in str(pin.support_path), key
        assert pin.support_path.suffix == ".usda", key
        assert 0 <= pin.start_frame < pin.end_frame_exclusive, key
        assert set(pin.object_asset_sha256) == {
            "textured_mesh.urdf",
            "textured_mesh.obj",
        }, key
        for digest in (
            pin.parquet_sha256,
            pin.payload_sha256,
            pin.support_sha256,
            *pin.object_asset_sha256.values(),
        ):
            assert len(digest) == 64 and set(digest) <= set("0123456789abcdef"), key
        assert pin.support_radius_source > 0.0 and pin.support_height_source > 0.0, key
        assert len(pin.object_anchor_target) == 3, key
        assert pin.runtime_object_name.isidentifier(), key
    # Distinct motions must not share a source, a support, or a payload.
    for field in ("source_path", "support_path", "payload_sha256", "parquet_sha256"):
        values = [getattr(pin, field) for pin in PINNED_SEQUENCES.values()]
        assert len(set(values)) == len(values), field


def test_snack_box_pin_matches_the_frozen_bimanual_grasp_crop() -> None:
    """The crop holds the grasp and lift and stops before the carry."""

    pin = PINNED_SEQUENCES["snack_box_pick"]
    assert pin.sequence_id == "2026-03-06_10-24-18_snack_box_pick_and_place_01"
    # Both contacts close at ~f440; the walk starts after f480.
    assert (pin.start_frame, pin.end_frame_exclusive) == (400, 480)
    assert pin.runtime_object_name == "snack_box"
    # Support cylinder copied verbatim from the pinned support USDA prim.
    assert pin.support_radius_source == pytest.approx(0.22521347938612124)
    assert pin.support_height_source == pytest.approx(0.01)
    assert np.allclose(
        pin.support_center_source,
        (1.8040543894793986, 0.05697129174318872, 0.7503690559680148),
    )


@pytest.mark.parametrize("key", sorted(PINNED_SEQUENCES))
def test_pinned_assets_resolve_and_hash_as_declared(key: str) -> None:
    pin = PINNED_SEQUENCES[key]
    if not pin.source_path.is_file() or not pin.support_path.is_file():
        pytest.skip(f"Pinned {key} source assets are absent")
    assert _sha256_file(pin.source_path) == pin.parquet_sha256
    assert _sha256_file(pin.support_path) == pin.support_sha256
    object_dir = pin.source_path.parent.parent / "object"
    for name, digest in pin.object_asset_sha256.items():
        asset = object_dir / name
        assert asset.is_file(), asset
        assert _sha256_file(asset) == digest, name


def test_support_projection_uses_the_pin_cylinder_not_a_global() -> None:
    """A non-corn-can pin must move the support with its own geometry."""

    pin = PINNED_SEQUENCES["snack_box_pick"]
    source_anchor = np.asarray((1.80, 0.06, 0.90), dtype=np.float64)
    target_anchor = np.asarray((0.55, 0.0, 1.05), dtype=np.float64)
    projection = ContactPhaseProjection(
        rotation=Rotation.identity(),
        source_object_anchor=source_anchor,
        target_object_anchor=target_anchor,
        global_motion_scale=1.0,
        local_geometry_scale=1.0,
        support_center_source=pin.support_center_source,
        support_height_source=pin.support_height_source,
    )
    expected_center = target_anchor + (
        np.asarray(pin.support_center_source, dtype=np.float64) - source_anchor
    )
    assert np.allclose(projection.support_center_target(), expected_center)
    assert projection.support_top_z() == pytest.approx(
        float(expected_center[2]) + 0.5 * pin.support_height_source
    )
    # The corn-can globals must not leak into this pin's scene.
    assert not np.allclose(
        projection.support_center_target(),
        target_anchor + (PINNED_SUPPORT_CENTER_SOURCE - source_anchor),
    )


def test_soma_non_thumb_chains_use_mcp_knuckle_and_anatomical_hinges() -> None:
    assert SOMA_HAND_KNUCKLE_NAMES == {
        "left": (
            "LeftHandIndex2",
            "LeftHandMiddle2",
            "LeftHandRing2",
            "LeftHandPinky2",
        ),
        "right": (
            "RightHandIndex2",
            "RightHandMiddle2",
            "RightHandRing2",
            "RightHandPinky2",
        ),
    }
    limits = {
        f"l_{finger}_{joint}": (-1.0, 1.0)
        for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")
        for joint in ("mcp_flex", "pip", "dip")
    }
    limits.update(
        {
            "l_thumb_cmc_flex": (-1.0, 1.0),
            "l_thumb_mcp": (-1.0, 1.0),
            "l_thumb_ip": (-1.0, 1.0),
        }
    )
    specs = {spec.target_name: spec for spec in _finger_specs("left", limits)}
    assert (
        specs["l_index_finger_mcp_flex"].parent,
        specs["l_index_finger_mcp_flex"].joint,
        specs["l_index_finger_mcp_flex"].child,
    ) == ("LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3")
    assert (
        specs["l_index_finger_pip"].parent,
        specs["l_index_finger_pip"].joint,
        specs["l_index_finger_pip"].child,
    ) == ("LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4")
    assert (
        specs["l_index_finger_dip"].parent,
        specs["l_index_finger_dip"].joint,
        specs["l_index_finger_dip"].child,
    ) == ("LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd")


def test_parser_defaults_to_the_adopted_wuji_recipe() -> None:
    args = build_parser().parse_args(["--output", "candidate.npz"])
    assert args.split_wrist_hand_ik is True
    assert args.anatomical_finger_limits is True
    assert args.soma_hand_scale == DEFAULT_SOMA_TO_WUJI_HAND_SCALE == 1.0
    assert args.finger_posture_prior == DEFAULT_FINGER_POSTURE_PRIOR == 2.0e-3
    assert args.wrist_orientation_weight == DEFAULT_IK_ORIENTATION_WEIGHT == 0.03
    assert args.require_finger_gates is False


def test_soma_wrist_orientation_uses_geometry_correction() -> None:
    source = Rotation.from_euler("xyz", [[10.0, -20.0, 30.0]], degrees=True)
    alignment = Rotation.from_euler("z", 40.0, degrees=True)
    corrections = {
        "left": {
            "source_to_robot_rotation": Rotation.from_euler(
                "y", 2.0, degrees=True
            ).as_matrix()
        },
        "right": {
            "source_to_robot_rotation": Rotation.from_euler(
                "y", -3.0, degrees=True
            ).as_matrix()
        },
    }
    source_wxyz = np.repeat(
        np.roll(source.as_quat(), 1, axis=-1)[:, None, :], 2, axis=1
    )

    actual_wxyz, offsets = _calibrate_soma_wrist_orientations(
        source_wxyz,
        alignment_rotation=alignment,
        palm_frame_calibration=corrections,
    )

    for side_index, side in enumerate(("left", "right")):
        expected = (
            alignment
            * source
            * Rotation.from_matrix(corrections[side]["source_to_robot_rotation"])
        )
        actual = Rotation.from_quat(np.roll(actual_wxyz[:, side_index], -1, axis=-1))
        np.testing.assert_allclose(
            (expected.inv() * actual).magnitude(), 0.0, atol=1.0e-12
        )
        assert len(offsets[side]) == 4


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


def test_target_times_stretch_motion_without_changing_reference_fps() -> None:
    source, target = _target_times(
        start_frame=522,
        end_frame_exclusive=840,
        source_fps=200.0,
        motion_time_scale=4.0,
    )
    assert source.shape == (318,)
    assert target.shape == (127,)
    np.testing.assert_allclose(np.diff(target), 0.0125, atol=1.0e-12)
    assert (len(target) - 1) / 20.0 == pytest.approx(4.0 * (target[-1] - target[0]))
    assert target[-1] < source[-1]


def test_source_frame_change_bound_preserves_emitted_limit() -> None:
    source_bound = _source_frame_change_bound(
        source_fps=200.0,
        target_fps=20.0,
        motion_time_scale=1.0,
        target_frame_change_rad=float(np.radians(34.99)),
    )
    assert np.degrees(source_bound) == pytest.approx(3.499)
    assert np.degrees(10.0 * source_bound) == pytest.approx(34.99)

    stretched_bound = _source_frame_change_bound(
        source_fps=200.0,
        target_fps=20.0,
        motion_time_scale=4.0,
        target_frame_change_rad=float(np.radians(34.99)),
    )
    assert np.degrees(stretched_bound) == pytest.approx(13.996)


def test_target_times_reject_invalid_motion_time_scale() -> None:
    with pytest.raises(ValueError, match="motion_time_scale"):
        _target_times(
            start_frame=522,
            end_frame_exclusive=840,
            source_fps=200.0,
            motion_time_scale=0.0,
        )


def test_chord_resampling_uses_crop_relative_indices() -> None:
    active = np.zeros((12, 2, 1), dtype=bool)
    active[0, 0, 0] = True
    active[10, 1, 0] = True
    positions = np.arange(active.size * 3, dtype=np.float64).reshape(*active.shape, 3)
    normals = -positions

    sampled_active, sampled_positions, sampled_normals = (
        _resample_chord_targets_nearest(
            active,
            positions,
            normals,
            source_fps=200.0,
            target_times=np.asarray((2.61, 2.66)),
            source_start_time=2.61,
        )
    )

    np.testing.assert_array_equal(sampled_active, active[[0, 10]])
    np.testing.assert_array_equal(sampled_positions, positions[[0, 10]])
    np.testing.assert_array_equal(sampled_normals, normals[[0, 10]])


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


def test_palm_projection_can_keep_inactive_rest_below_finite_support() -> None:
    objects = np.zeros((2, 1, 3), dtype=np.float64)
    objects[..., 2] = 1.0
    active = np.broadcast_to(objects, (2, 2, 3)).copy()
    active[..., 0] += 0.3
    rest_target = np.asarray(((0.3, 0.35, 0.75), (0.3, -0.35, 0.75)))
    weights = np.zeros((2, 2), dtype=np.float64)

    projected, projected_active, rest = _project_palm_positions_nonpenetrating(
        active,
        objects,
        rest_target,
        weights,
        support_top_z=1.1,
        palm_object_standoff_m=0.25,
        palm_support_standoff_m=0.2,
        raise_rest_above_support=False,
    )

    assert np.all(projected_active[..., 2] >= 1.3)
    np.testing.assert_allclose(rest, rest_target)
    np.testing.assert_allclose(projected, np.broadcast_to(rest_target, projected.shape))


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


def test_split_wrist_hand_lock_allows_fingers_but_rejects_arm_drift() -> None:
    joint_names = ("left_arm", "left_finger", "right_arm", "right_finger")
    wrist_solution = np.zeros((3, 4), dtype=np.float64)
    fingers_refined = wrist_solution.copy()
    fingers_refined[:, (1, 3)] = 0.4
    report = _verify_locked_joint_trajectory(
        wrist_solution,
        fingers_refined,
        joint_names=joint_names,
        locked_joint_names=("left_arm", "right_arm"),
    )
    assert report["verified"] is True
    assert report["maximum_absolute_drift_rad_or_m"] == 0.0

    arm_changed = fingers_refined.copy()
    arm_changed[2, 0] = 1.0e-5
    with pytest.raises(RuntimeError, match="left_arm"):
        _verify_locked_joint_trajectory(
            wrist_solution,
            arm_changed,
            joint_names=joint_names,
            locked_joint_names=("left_arm", "right_arm"),
        )


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


def test_phase_object_anchor_blend_returns_to_base_when_hands_are_inactive() -> None:
    base = np.asarray((0.8, 0.15, 1.14), dtype=np.float64)
    phases = np.asarray(((0.65, 0.15, 1.14), (0.8, -0.20, 1.14)))
    weights = np.asarray(
        (
            (0.0, 1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0, 1.0),
        )
    )
    anchors = _phase_object_anchor_trajectory(base, phases, weights)
    np.testing.assert_allclose(anchors[0], base)
    np.testing.assert_allclose(anchors[1], phases[0])
    np.testing.assert_allclose(anchors[2], (0.725, -0.025, 1.14))
    np.testing.assert_allclose(anchors[3], phases[1])


def test_phase_anchor_inference_only_moves_xy_and_is_bounded() -> None:
    source_object = np.zeros((3, 1, 3), dtype=np.float64)
    source_wrists = np.zeros((3, 2, 3), dtype=np.float64)
    source_wrists[:, 0, 0] = -0.1
    source_wrists[:, 1, 1] = -0.2
    active = np.asarray(((True, True, False), (False, True, True)))
    neutral = np.asarray(((0.8, 0.2, 1.0), (0.8, -0.2, 1.0)))
    base = np.asarray((0.8, 0.15, 1.14))
    anchors = _infer_phase_object_anchors(
        source_wrists,
        source_object,
        active,
        rotation=Rotation.identity(),
        local_geometry_scale=1.0,
        neutral_positions=neutral,
        base_anchor=base,
        max_shift_m=0.1,
    )
    np.testing.assert_allclose(anchors[:, 2], base[2])
    np.testing.assert_allclose(anchors[0], (0.9, 0.2, 1.14))
    np.testing.assert_allclose(anchors[1], (0.8, 0.05, 1.14))


def test_inspection_contacts_are_named_but_never_active() -> None:
    contacts = _inspection_contacts(7)
    contacts.validate(frame_count=7, object_count=1)
    assert contacts.link_names.shape == (2, 5)
    assert np.all(contacts.link_names != "")
    assert not np.any(contacts.active)
    assert np.all(contacts.object_indices == -1)
    assert np.all(contacts.link_positions_w == 0.0)
    assert np.all(contacts.object_positions_w == 0.0)


def test_contact_recovery_gate_is_bilateral() -> None:
    qualified, rates = _contact_recovery_gate(
        {
            "per_side_recovered": {"left": 8, "right": 4},
            "per_side_unreached": {"left": 2, "right": 6},
        }
    )
    assert qualified is False
    assert rates == {"left": 0.8, "right": 0.4}

    qualified, rates = _contact_recovery_gate(
        {
            "per_side_recovered": {"left": 5, "right": 6},
            "per_side_unreached": {"left": 5, "right": 4},
        }
    )
    assert qualified is True
    assert rates == {"left": 0.5, "right": 0.6}


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


@pytest.mark.skipif(
    not DEFAULT_SOURCE.is_file() or importlib.util.find_spec("pyarrow") is None,
    reason="Pinned SOMA source or pyarrow absent",
)
def test_real_inspection_conversion_is_bounded_and_fails_closed() -> None:
    with pytest.raises(ValueError, match="wrist_orientation_weight must equal 0.03"):
        convert_soma_g1_parquet(DEFAULT_SOURCE, wrist_orientation_weight=0.01)

    with pytest.raises(ValueError, match="Refusing to invent"):
        convert_soma_g1_parquet(DEFAULT_SOURCE)

    reference = convert_soma_g1_parquet(DEFAULT_SOURCE, inspection_only=True)
    assert reference.object_names == ("corn_can",)
    assert reference.frame_count == 32
    assert reference.fps == 20.0
    assert reference.metadata["runtime_qualified"] is False
    assert reference.metadata["isaac_runtime_qualified"] is False
    assert reference.metadata["inspection_only"] is True
    assert reference.training_qualification is None
    assert reference.metadata["inspection_scene_geometry_qualified"] is False
    assert reference.metadata["contacts"]["contact_geometry"] == "unavailable"
    audit = reference.metadata["geometry_clearance_audit"]
    assert audit["qualified"] is False
    assert audit["robot_self_collision_20hz"]["qualified"] is False
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
    assert reference.metadata["wrist_ik"]["max_position_error_m"] > 0.0025
    assert reference.metadata["wrist_ik"][
        "solver_orientation_objective_weight"
    ] == pytest.approx(0.03**2)
    finger_acceptance = reference.metadata["finger_mapping"]["final_acceptance"]
    assert finger_acceptance["qualified"] is False
    assert finger_acceptance["source_to_robot_scale"] == 1.0
    finger_mapping = reference.metadata["finger_mapping"]
    projection_sides = finger_mapping["soma_chord_hand_keypoint_projection"]["sides"]
    for side in ("left", "right"):
        gates = finger_acceptance["sides"][side]["gates"]
        assert gates["pass"] is False
        assert gates["not_measured"] == []
        assert "wrist_mean_mm" in gates["failures"]
        calibration = finger_mapping["palm_frame_calibration"][side]
        assert calibration["calibration_source"] == (
            "identity-specific full-resolution SOMA zero-pose MCP geometry"
        )
        assert calibration["scale_applied_to_targets"] == 1.0
        projection = projection_sides[side]
        assert projection["frame_count"] == 318
        assert projection["solve_rate_hz"] == 200.0
        assert projection["source_frame_change_limit_deg"] == pytest.approx(3.499)
        assert projection["emitted_frame_change_limit_deg"] == pytest.approx(34.99)
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


def _box_poses(z: float, roll_deg: float = 0.0) -> np.ndarray:
    wxyz = Rotation.from_euler("x", roll_deg, degrees=True).as_quat(scalar_first=True)
    pose = [0.0, 0.0, float(z)] + [float(value) for value in wxyz]
    return np.asarray([[pose]], dtype=np.float64)


def test_support_audit_ignores_vertices_past_the_table_edge(tmp_path) -> None:
    """A wide object overhanging a small support must not read as penetration.

    The support is a finite cylinder. A corner hanging beyond its edge is in
    free space, so restricting the test to vertices actually over the disc is
    the physical question; the infinite-plane reading stays reported.
    """

    # A 2 m wide, thin slab centred on a 0.2 m radius support.
    mesh = tmp_path / "slab.obj"
    mesh.write_text(
        "\n".join(
            f"v {x} {y} {z}"
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (0.0, 0.05)
        )
        + "\n",
        encoding="utf-8",
    )
    # Tilt the slab so its far corners drop well below the support top while
    # the material over the disc stays above it.
    poses = _box_poses(z=0.5, roll_deg=5.0)

    conservative = _audit_can_support_vertical_clearance(
        poses, object_mesh=mesh, object_scale=1.0, support_top_z=0.45
    )
    assert conservative["qualified"] is False
    assert conservative["support_disc_restricted"] is False

    restricted = _audit_can_support_vertical_clearance(
        poses,
        object_mesh=mesh,
        object_scale=1.0,
        support_top_z=0.45,
        support_center_xy=(0.0, 0.0),
        support_radius_m=0.2,
    )
    assert restricted["qualified"] is True
    assert restricted["support_disc_restricted"] is True
    # The stricter reading must remain visible, not be discarded.
    assert restricted[
        "minimum_vertical_clearance_to_infinite_plane_m"
    ] == pytest.approx(conservative["minimum_vertical_clearance_m"])


def test_support_audit_still_catches_real_penetration(tmp_path) -> None:
    """Restricting to the disc must not hide an object sunk into the table."""

    mesh = tmp_path / "slab.obj"
    mesh.write_text(
        "\n".join(
            f"v {x} {y} {z}"
            for x in (-0.05, 0.05)
            for y in (-0.05, 0.05)
            for z in (0.0, 0.05)
        )
        + "\n",
        encoding="utf-8",
    )
    poses = _box_poses(z=0.44)  # bottom at 0.44, support top at 0.45
    restricted = _audit_can_support_vertical_clearance(
        poses,
        object_mesh=mesh,
        object_scale=1.0,
        support_top_z=0.45,
        support_center_xy=(0.0, 0.0),
        support_radius_m=0.2,
    )
    assert restricted["qualified"] is False
    assert restricted["minimum_vertical_clearance_m"] == pytest.approx(-0.01)


def test_support_audit_tolerance_matches_the_other_penetration_gates() -> None:
    """1 um was a numerical epsilon; every other gate here allows 1 mm."""

    import inspect

    signature = inspect.signature(_audit_can_support_vertical_clearance)
    assert signature.parameters["penetration_tolerance_m"].default == pytest.approx(
        1.0e-3
    )


def test_chord_distal_indices_are_the_fingertip_links() -> None:
    from convert_soma_g1_parquet_to_vega_wuji import (
        CHORD_DISTAL_LINK_INDICES,
        SOMA_CHORD_LINK_KEYS,
    )

    assert [SOMA_CHORD_LINK_KEYS[i] for i in CHORD_DISTAL_LINK_INDICES] == [
        "thumb3",
        "index3",
        "middle3",
        "ring3",
        "pinky3",
    ]


def test_object_anchored_targets_follow_contact_and_fall_back_off_contact() -> None:
    """On a true-size object the contact point, not the human fingertip, is
    the physical requirement - but only while the finger is touching."""

    from convert_soma_g1_parquet_to_vega_wuji import (
        _object_anchored_fingertip_targets,
    )

    frames, links = 41, len(SOMA_CHORD_LINK_KEYS)
    wrist_anchored = np.zeros((frames, 2, 5, 3), dtype=np.float64)
    chord_targets = np.zeros((frames, 2, links, 3), dtype=np.float64)
    chord_targets[..., 0] = 1.0  # every contact point sits 1 m away in x
    chord_active = np.zeros((frames, 2, links), dtype=bool)
    # Contact for a long, solidly-held middle stretch.
    chord_active[10:31, :, :] = True

    blended, report = _object_anchored_fingertip_targets(
        wrist_anchored,
        chord_targets=chord_targets,
        chord_active=chord_active,
        blend_frames=5,
    )
    assert blended.shape == wrist_anchored.shape
    # Deep inside contact the target is the object-anchored contact point.
    assert blended[20, 0, 0, 0] == pytest.approx(1.0)
    # Far outside contact it is the wrist-anchored human fingertip.
    assert blended[0, 0, 0, 0] == pytest.approx(0.0)
    assert blended[-1, 0, 0, 0] == pytest.approx(0.0)
    assert report["max_target_displacement_m"] == pytest.approx(1.0)


def test_object_anchored_targets_never_jump_at_contact_onset() -> None:
    """A binary contact mask must not produce a discontinuous target path."""

    from convert_soma_g1_parquet_to_vega_wuji import (
        _object_anchored_fingertip_targets,
    )

    frames, links = 61, len(SOMA_CHORD_LINK_KEYS)
    wrist_anchored = np.zeros((frames, 2, 5, 3), dtype=np.float64)
    chord_targets = np.zeros((frames, 2, links, 3), dtype=np.float64)
    chord_targets[..., 0] = 1.0
    chord_active = np.zeros((frames, 2, links), dtype=bool)
    chord_active[30:, :, :] = True  # hard switch on

    blend_frames = 11
    blended, _ = _object_anchored_fingertip_targets(
        wrist_anchored,
        chord_targets=chord_targets,
        chord_active=chord_active,
        blend_frames=blend_frames,
    )
    steps = np.abs(np.diff(blended[:, 0, 0, 0]))
    # A hard switch would step the full 1 m in one frame; the cross-fade must
    # spread it over the blend window instead.
    assert steps.max() <= 1.0 / (blend_frames - 1) + 1.0e-9
    assert np.all(np.diff(blended[:, 0, 0, 0]) >= -1.0e-9)  # monotone onset


def test_object_anchored_targets_reject_misaligned_inputs() -> None:
    from convert_soma_g1_parquet_to_vega_wuji import (
        _object_anchored_fingertip_targets,
    )

    wrist_anchored = np.zeros((4, 2, 5, 3))
    chord_targets = np.zeros((4, 2, len(SOMA_CHORD_LINK_KEYS), 3))
    with pytest.raises(ValueError, match="does not align"):
        _object_anchored_fingertip_targets(
            wrist_anchored,
            chord_targets=chord_targets,
            chord_active=np.zeros((4, 2, 3), dtype=bool),
            blend_frames=3,
        )
    with pytest.raises(ValueError, match="positive odd frame count"):
        _object_anchored_fingertip_targets(
            wrist_anchored,
            chord_targets=chord_targets,
            chord_active=np.zeros((4, 2, len(SOMA_CHORD_LINK_KEYS)), dtype=bool),
            blend_frames=4,
        )


def test_object_anchored_targets_ignore_the_origin_placeholder() -> None:
    """CHORD parks inactive slots at the world origin, not at a real point.

    Averaging the raw target array over time therefore drags fingertips toward
    the origin on frames next to a contact edge. On the real clip that showed
    up as a 518 mm fingertip residual, so it is pinned here.
    """

    from convert_soma_g1_parquet_to_vega_wuji import (
        _object_anchored_fingertip_targets,
    )

    frames, links = 41, len(SOMA_CHORD_LINK_KEYS)
    # The hand works a metre away from the origin, like the real scene.
    wrist_anchored = np.full((frames, 2, 5, 3), 1.0, dtype=np.float64)
    chord_targets = np.zeros((frames, 2, links, 3), dtype=np.float64)
    chord_active = np.zeros((frames, 2, links), dtype=bool)
    # A short contact whose points sit right beside the wrist-anchored target,
    # while every inactive slot stays at the origin placeholder.
    chord_targets[18:24, :, :, :] = 1.01
    chord_active[18:24, :, :] = True

    blended, report = _object_anchored_fingertip_targets(
        wrist_anchored,
        chord_targets=chord_targets,
        chord_active=chord_active,
        blend_frames=9,
    )
    # No target may be pulled anywhere near the origin.
    assert float(np.abs(blended - 1.0).max()) < 0.05
    assert report["max_target_displacement_m"] < 0.05
    # Frames far from the contact are untouched.
    assert blended[0, 0, 0, 0] == pytest.approx(1.0)


def test_object_anchored_targets_stand_off_by_the_fingertip_radius() -> None:
    """The IK aims a point; contact happens on a sphere around it."""

    from convert_soma_g1_parquet_to_vega_wuji import (
        _object_anchored_fingertip_targets,
    )

    frames, links = 21, len(SOMA_CHORD_LINK_KEYS)
    wrist_anchored = np.zeros((frames, 2, 5, 3), dtype=np.float64)
    chord_targets = np.zeros((frames, 2, links, 3), dtype=np.float64)
    chord_normals = np.zeros((frames, 2, links, 3), dtype=np.float64)
    chord_normals[..., 2] = 1.0  # surface faces +z
    chord_active = np.ones((frames, 2, links), dtype=bool)

    blended, report = _object_anchored_fingertip_targets(
        wrist_anchored,
        chord_targets=chord_targets,
        chord_active=chord_active,
        blend_frames=3,
        chord_normals=chord_normals,
        tip_radius_m=0.009,
    )
    # The target sits one radius outside the surface, along its normal.
    assert blended[10, 0, 0, 2] == pytest.approx(0.009)
    assert blended[10, 0, 0, 0] == pytest.approx(0.0)
    assert report["fingertip_collision_radius_offset_m"] == pytest.approx(0.009)


def test_object_anchored_targets_require_normals_for_the_offset() -> None:
    from convert_soma_g1_parquet_to_vega_wuji import (
        _object_anchored_fingertip_targets,
    )

    with pytest.raises(ValueError, match="needs the CHORD normals"):
        _object_anchored_fingertip_targets(
            np.zeros((4, 2, 5, 3)),
            chord_targets=np.zeros((4, 2, len(SOMA_CHORD_LINK_KEYS), 3)),
            chord_active=np.zeros((4, 2, len(SOMA_CHORD_LINK_KEYS)), dtype=bool),
            blend_frames=3,
            tip_radius_m=0.009,
        )
