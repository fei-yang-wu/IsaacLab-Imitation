"""Tests for the Sharpa reference parity comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("iltools.core")
from iltools.core import (  # noqa: E402
    create_dexterous_reference_manifest,
    save_dexterous_reference_npz,
    sha256_file,
)
from iltools.datasets import ManoSharpaLoader  # noqa: E402


def _load_module():
    path = Path(__file__).with_name("compare_sharpa_references.py")
    spec = importlib.util.spec_from_file_location("compare_sharpa_references", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(*, wrist_shift: float = 0.0, joint_shift: float = 0.0) -> dict:
    frame_count = 3
    frames = np.zeros((frame_count, 67, 7), dtype=np.float32)
    frames[..., 3] = 1.0
    identity = [[1.0, 0.0, 0.0, 0.0]] * frame_count
    return {
        "sequence_id": "fixture",
        "robot_name": "sharpa_wave",
        "fps": 20.0,
        "mano_to_robot_scale": 1.0,
        "mano_link_names": [],
        "left_robot_finger_joint_names": [f"left_{i}" for i in range(22)],
        "right_robot_finger_joint_names": [f"right_{i}" for i in range(22)],
        "left_robot_frame_names": [f"frame_{i}" for i in range(67)],
        "right_robot_frame_names": [f"frame_{i}" for i in range(67)],
        "left_robot_frame_task_names": ["left_hand_C_MC"],
        "right_robot_frame_task_names": ["right_hand_C_MC"],
        "robot_left_finger_joints": np.full((frame_count, 22), joint_shift).tolist(),
        "robot_right_finger_joints": np.zeros((frame_count, 22)).tolist(),
        "robot_left_wrist_position": [
            [wrist_shift, -0.2, 0.5],
            [0.1, -0.2, 0.5],
            [0.2, -0.2, 0.5],
        ],
        "robot_right_wrist_position": [[0.0, 0.2, 0.5]] * frame_count,
        "robot_left_wrist_wxyz": identity,
        "robot_right_wrist_wxyz": identity,
        "robot_left_frames": frames.tolist(),
        "robot_right_frames": frames.tolist(),
        "object_name": "box",
        "object_body_names": ["box"],
        "object_body_position": [[[0.0, 0.0, 0.4]]] * frame_count,
        "object_body_wxyz": [[identity[0]]] * frame_count,
        "object_articulation": [[] for _ in range(frame_count)],
        "object_mesh_radius": [0.05],
    }


def _write_manifest(tmp_path: Path, name: str, row: dict) -> Path:
    object_path = tmp_path / "box.obj"
    object_path.write_text("o box\n", encoding="utf-8")
    reference = ManoSharpaLoader.to_reference(
        row, object_asset_path=object_path, object_asset_sha256=sha256_file(object_path)
    )
    npz_path = save_dexterous_reference_npz(reference, tmp_path / name / "00000.npz")
    return create_dexterous_reference_manifest(
        [npz_path], tmp_path / name / "manifest.json", dataset_name=name
    )


def test_quaternion_angle_measures_relative_rotation() -> None:
    module = _load_module()
    half = np.sqrt(0.5)
    angle = module.quaternion_angle_rad(
        np.asarray([[1.0, 0.0, 0.0, 0.0], [half, 0.0, 0.0, half]]),
        np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    assert angle == pytest.approx([0.0, np.pi / 2.0], abs=1.0e-6)
    # Antipodal quaternions describe the same rotation.
    same = module.quaternion_angle_rad(
        np.asarray([[1.0, 0.0, 0.0, 0.0]]), np.asarray([[-1.0, 0.0, 0.0, 0.0]])
    )
    assert same == pytest.approx([0.0], abs=1.0e-6)


def test_identical_references_pass_tight_thresholds(tmp_path) -> None:
    module = _load_module()
    first = _write_manifest(tmp_path, "a", _row())
    second = _write_manifest(tmp_path, "b", _row())
    report = module.build_report(
        first, second, {"max_wrist_position_m": 1.0e-6, "max_joint_rad": 1.0e-6}
    )
    assert report["passed"]
    motion = report["motions"][0]
    assert motion["left_wrist_position_m"]["max"] == 0.0
    assert motion["finger_joint_rad"]["max"] == 0.0
    assert motion["metadata_a"]["retarget_source"] == "parquet_robot_columns"


def test_differences_are_reported_and_fail_thresholds(tmp_path) -> None:
    module = _load_module()
    first = _write_manifest(tmp_path, "a", _row())
    second = _write_manifest(tmp_path, "b", _row(wrist_shift=0.05, joint_shift=0.2))
    report = module.build_report(
        first, second, {"max_wrist_position_m": 0.01, "max_joint_rad": 0.1}
    )
    assert not report["passed"]
    motion = report["motions"][0]
    assert motion["left_wrist_position_m"]["max"] == pytest.approx(0.05)
    assert motion["right_wrist_position_m"]["max"] == 0.0
    assert motion["finger_joint_rad"]["max"] == pytest.approx(0.2)
    assert motion["finger_joint_worst"]["joint"].startswith("left_")
    assert len(report["failures"]) == 2


def test_cli_returns_nonzero_on_failure(tmp_path, capsys) -> None:
    module = _load_module()
    first = _write_manifest(tmp_path, "a", _row())
    second = _write_manifest(tmp_path, "b", _row(wrist_shift=0.05))
    output = tmp_path / "report.json"
    code = module.main(
        [
            "--reference",
            str(first),
            "--reference",
            str(second),
            "--max-wrist-position-m",
            "0.01",
            "--output",
            str(output),
        ]
    )
    assert code == 1
    assert output.is_file()
    assert "FAIL" in capsys.readouterr().out
