"""Tests for the dependency-light EgoDex hand retargeter."""

from __future__ import annotations

import numpy as np
import pytest

from retarget_egodex import _keypoint_specs, _resample_qpos, _source_names


def test_right_hand_specs_cover_three_hinges_per_finger() -> None:
    limits = {
        name: (-2.0, 2.0)
        for name in (
            "r_thumb_cmc_flex",
            "r_thumb_mcp",
            "r_thumb_ip",
            "r_index_finger_mcp_flex",
            "r_index_finger_pip",
            "r_index_finger_dip",
            "r_middle_finger_mcp_flex",
            "r_middle_finger_pip",
            "r_middle_finger_dip",
            "r_ring_finger_mcp_flex",
            "r_ring_finger_pip",
            "r_ring_finger_dip",
            "r_pinky_mcp_flex",
            "r_pinky_pip",
            "r_pinky_dip",
        )
    }
    specs = _keypoint_specs("right", limits)

    assert len(_source_names("right")) == 21
    assert len(specs) == 15
    assert all(spec.target_name.startswith("r_") for spec in specs)
    assert all(spec.scale == -1.0 for spec in specs)


def test_resample_qpos_preserves_endpoints_and_changes_rate() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)

    result = _resample_qpos(source, input_fps=2.0, output_fps=4.0)

    np.testing.assert_allclose(result[[0, -1], 0], [0.0, 2.0])
    assert result.shape == (5, 1)


def test_hdf5_loader_accepts_hand_level_confidence(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    from retarget_egodex import _read_hdf5

    path = tmp_path / "hora_layout.hdf5"
    names = _source_names("right")
    with h5py.File(path, "w") as handle:
        transforms = handle.create_group("transforms")
        confidences = handle.create_group("confidences")
        for index, name in enumerate(names):
            values = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
            values[:, 0, 3] = np.arange(3, dtype=np.float32) + index
            transforms.create_dataset(name, data=values)
        confidences.create_dataset("rightHand", data=np.ones(3, dtype=np.float32))

    trajectory, _ = _read_hdf5(
        path,
        hand="right",
        fps=30.0,
        confidence_threshold=0.5,
    )

    assert trajectory.observations["keypoints"].shape == (3, 21, 3)
    assert set(trajectory.infos["confidence_sources"].values()) == {
        "confidences/rightHand"
    }
