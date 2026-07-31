#!/usr/bin/env python3
"""Extend collected root+qpos packets without changing their causal input rows.

The collected packet contains the current reference pelvis pose expressed in
the live robot pelvis frame. Together with the reference pelvis pose in the
frozen Zarr, that relative transform recovers the robot anchor at collection
time. We can therefore express additional future reference frames in the exact
same anchor frame, preserving every planner input, trajectory ID, and split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr


# Newton/MJWarp enumerates the G1 articulation depth-first. This is the public
# Unitree G1_29_JointIndex order and the order of the collected qpos packet.
TARGET_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
TERM_NAMES = (
    "expert_motion_qpos",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
)
SOURCE_FRAMES = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=Path, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--horizon_steps", type=int, default=30)
    parser.add_argument("--anchor_body_name", default="pelvis")
    parser.add_argument("--validation_tolerance", type=float, default=2.0e-5)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert normalized scalar-first quaternions to rotation matrices."""
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1.0e-12)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Project the stored first-two-column representation onto SO(3)."""
    columns = np.asarray(rot6d, dtype=np.float64).reshape(*rot6d.shape[:-1], 3, 2)
    first = columns[..., :, 0]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True).clip(min=1.0e-12)
    second = columns[..., :, 1]
    second = second - first * np.sum(first * second, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True).clip(min=1.0e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[..., :, :2].reshape(*matrix.shape[:-2], 6)


def _dataset_group(root: Any, motion_name: str) -> tuple[Any, Any]:
    for dataset_name in root.group_keys():
        dataset = root[dataset_name]
        if motion_name in dataset:
            motion = dataset[motion_name]
            trajectory_names = sorted(motion.group_keys())
            if trajectory_names != ["trajectory_0"]:
                raise ValueError(
                    f"{motion_name} must have exactly trajectory_0, got "
                    f"{trajectory_names}."
                )
            return dataset, motion["trajectory_0"]
    raise KeyError(f"Dataset has no motion {motion_name!r}.")


def _row_aligned_motion_names(
    sample: dict[str, Any], rows: int, path: Path
) -> list[str]:
    names = sample.get("motion_name")
    if not isinstance(names, list) or len(names) != rows:
        raise ValueError(f"{path} has no row-aligned motion_name field.")
    return [str(name) for name in names]


def _extend_motion_rows(
    *,
    source_target: torch.Tensor,
    control_steps: torch.Tensor,
    row_indices: np.ndarray,
    trajectory: Any,
    dataset_attrs: dict[str, Any],
    anchor_body_name: str,
    horizon_steps: int,
    tolerance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    joint_names = [str(name) for name in dataset_attrs.get("joint_names", ())]
    body_names = [str(name) for name in dataset_attrs.get("body_names", ())]
    if set(TARGET_JOINT_NAMES) - set(joint_names):
        raise ValueError("Reference Zarr is missing required G1 joint names.")
    if anchor_body_name not in body_names:
        raise ValueError(f"Reference Zarr has no anchor body {anchor_body_name!r}.")
    joint_ids = np.asarray(
        [joint_names.index(name) for name in TARGET_JOINT_NAMES], dtype=np.int64
    )
    anchor_id = body_names.index(anchor_body_name)
    steps = control_steps.index_select(
        0, torch.as_tensor(row_indices, dtype=torch.long)
    ).numpy()
    length = int(trajectory["joint_pos"].shape[0])
    offsets = np.arange(horizon_steps, dtype=np.int64)
    frame_indices = np.minimum(steps[:, None] + offsets[None, :], length - 1)
    flat_indices = frame_indices.reshape(-1)
    batch_size = int(frame_indices.shape[0])

    joint_pos = np.asarray(
        trajectory["joint_pos"][flat_indices], dtype=np.float64
    ).reshape(batch_size, horizon_steps, -1)
    joint_pos = joint_pos[..., joint_ids]
    anchor_pos = np.asarray(
        trajectory["body_pos_w"][flat_indices], dtype=np.float64
    ).reshape(batch_size, horizon_steps, -1, 3)[..., anchor_id, :]
    anchor_quat = np.asarray(
        trajectory["body_quat_w"][flat_indices], dtype=np.float64
    ).reshape(batch_size, horizon_steps, -1, 4)[..., anchor_id, :]
    reference_rot = _quat_wxyz_to_matrix(anchor_quat)

    source = source_target.index_select(
        0, torch.as_tensor(row_indices, dtype=torch.long)
    ).numpy()
    if source.shape[1] != SOURCE_FRAMES * 38:
        raise ValueError(
            f"Expected a {SOURCE_FRAMES * 38}-D root_qpos source packet, got "
            f"{source.shape[1]}."
        )
    source_pos = source[:, SOURCE_FRAMES * 29 : SOURCE_FRAMES * 32].reshape(
        -1, SOURCE_FRAMES, 3
    )
    source_ori = source[:, SOURCE_FRAMES * 32 :].reshape(-1, SOURCE_FRAMES, 6)
    relative_rot_0 = _rot6d_to_matrix(source_ori[:, 0])
    robot_rot = reference_rot[:, 0] @ np.swapaxes(relative_rot_0, -1, -2)
    robot_pos = anchor_pos[:, 0] - np.einsum("nij,nj->ni", robot_rot, source_pos[:, 0])
    robot_rot_inv = np.swapaxes(robot_rot, -1, -2)
    relative_pos = np.einsum(
        "nij,nhj->nhi", robot_rot_inv, anchor_pos - robot_pos[:, None, :]
    )
    relative_rot = np.einsum("nij,nhjk->nhik", robot_rot_inv, reference_rot)
    relative_ori = _matrix_to_rot6d(relative_rot)

    reconstructed_h10 = np.concatenate(
        (
            joint_pos[:, :SOURCE_FRAMES].reshape(-1, SOURCE_FRAMES * 29),
            relative_pos[:, :SOURCE_FRAMES].reshape(-1, SOURCE_FRAMES * 3),
            relative_ori[:, :SOURCE_FRAMES].reshape(-1, SOURCE_FRAMES * 6),
        ),
        axis=-1,
    )
    absolute_error = np.abs(reconstructed_h10 - source)
    max_error = float(absolute_error.max(initial=0.0))
    if max_error > tolerance:
        raise ValueError(
            "Offline reconstruction does not reproduce the collected H10 packet: "
            f"max_abs={max_error:.9g} > tolerance={tolerance:.9g}."
        )
    extended = np.concatenate(
        (
            joint_pos.reshape(-1, horizon_steps * 29),
            relative_pos.reshape(-1, horizon_steps * 3),
            relative_ori.reshape(-1, horizon_steps * 6),
        ),
        axis=-1,
    )
    return torch.from_numpy(extended).float(), {
        "max_abs": max_error,
        "mean_abs": float(absolute_error.mean()),
    }


def main() -> None:
    args = _parse_args()
    if args.horizon_steps <= SOURCE_FRAMES:
        raise ValueError("--horizon_steps must exceed the collected H10 packet.")
    if args.validation_tolerance <= 0:
        raise ValueError("--validation_tolerance must be positive.")
    samples_dir = args.samples_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_paths = sorted(samples_dir.glob("sample_step_*.pt"))
    if not source_paths:
        raise FileNotFoundError(f"No sample_step_*.pt files under {samples_dir}.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    zarr_root = zarr.open_group(str(dataset_path), mode="r")

    manifest: dict[str, Any] = {
        "format": "long_horizon_root_qpos_materialization",
        "version": 1,
        "source_samples_dir": str(samples_dir),
        "dataset_path": str(dataset_path),
        "anchor_body_name": str(args.anchor_body_name),
        "source_horizon_steps": SOURCE_FRAMES,
        "target_horizon_steps": int(args.horizon_steps),
        "planner_interval_steps": SOURCE_FRAMES,
        "row_count": 0,
        "source_files": [],
        "output_files": [],
        "validation": {
            "tolerance": float(args.validation_tolerance),
            "max_abs": 0.0,
            "weighted_mean_abs": 0.0,
        },
    }
    weighted_error_sum = 0.0
    for source_path in source_paths:
        sample = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(sample, dict):
            raise TypeError(f"{source_path} is not a sample mapping.")
        source_target = sample.get("causal_target")
        control_steps = sample.get("control_step")
        if (
            not isinstance(source_target, torch.Tensor)
            or source_target.ndim != 2
            or not isinstance(control_steps, torch.Tensor)
        ):
            raise ValueError(f"{source_path} lacks target/control_step tensors.")
        rows = int(source_target.shape[0])
        motion_names = _row_aligned_motion_names(sample, rows, source_path)
        extended = torch.empty((rows, int(args.horizon_steps) * 38))
        file_max_error = 0.0
        file_weighted_error = 0.0
        for motion_name in sorted(set(motion_names)):
            row_indices = np.asarray(
                [i for i, name in enumerate(motion_names) if name == motion_name],
                dtype=np.int64,
            )
            dataset_group, trajectory = _dataset_group(zarr_root, motion_name)
            target, validation = _extend_motion_rows(
                source_target=source_target,
                control_steps=control_steps.reshape(-1).long(),
                row_indices=row_indices,
                trajectory=trajectory,
                dataset_attrs=dict(dataset_group.attrs),
                anchor_body_name=str(args.anchor_body_name),
                horizon_steps=int(args.horizon_steps),
                tolerance=float(args.validation_tolerance),
            )
            extended.index_copy_(
                0, torch.as_tensor(row_indices, dtype=torch.long), target
            )
            file_max_error = max(file_max_error, validation["max_abs"])
            file_weighted_error += validation["mean_abs"] * len(row_indices)

        result = copy.deepcopy(sample)
        result["source_h10_target"] = source_target.detach().cpu().contiguous()
        result["causal_target"] = extended.contiguous()
        result["demonstration_target"] = extended.contiguous()
        metadata = copy.deepcopy(sample.get("metadata"))
        if not isinstance(metadata, dict):
            raise ValueError(f"{source_path} has no metadata mapping.")
        target_spec = {
            "interface": "root_qpos",
            "term_names": list(TERM_NAMES),
            "term_widths": [
                int(args.horizon_steps) * 29,
                int(args.horizon_steps) * 3,
                int(args.horizon_steps) * 6,
            ],
            "target_dim": int(args.horizon_steps) * 38,
        }
        metadata.update(
            {
                "interface": "root_qpos",
                "target_spec": target_spec,
                "command_future_steps": int(args.horizon_steps) - 1,
                "long_horizon_materialization": {
                    "source_file": str(source_path.resolve()),
                    "source_file_sha256": _sha256(source_path),
                    "source_horizon_steps": SOURCE_FRAMES,
                    "target_horizon_steps": int(args.horizon_steps),
                    "anchor_body_name": str(args.anchor_body_name),
                    "h10_reconstruction_max_abs": file_max_error,
                    "validation_tolerance": float(args.validation_tolerance),
                },
            }
        )
        result["metadata"] = metadata
        output_path = output_dir / source_path.name
        torch.save(result, output_path)
        source_record = {
            "name": source_path.name,
            "sha256": _sha256(source_path),
            "rows": rows,
        }
        output_record = {
            "name": output_path.name,
            "sha256": _sha256(output_path),
            "rows": rows,
            "h10_reconstruction_max_abs": file_max_error,
        }
        manifest["source_files"].append(source_record)
        manifest["output_files"].append(output_record)
        manifest["row_count"] += rows
        manifest["validation"]["max_abs"] = max(
            manifest["validation"]["max_abs"], file_max_error
        )
        weighted_error_sum += file_weighted_error

    manifest["validation"]["weighted_mean_abs"] = weighted_error_sum / max(
        int(manifest["row_count"]), 1
    )
    manifest_path = output_dir / "materialization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "[PASS] Reused "
        f"{manifest['row_count']} collected rows as H{args.horizon_steps} targets; "
        f"H10 reconstruction max_abs={manifest['validation']['max_abs']:.3e}."
    )
    print(f"[PASS] {manifest_path}")


if __name__ == "__main__":
    main()
