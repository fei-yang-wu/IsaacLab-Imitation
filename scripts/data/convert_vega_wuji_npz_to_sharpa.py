#!/usr/bin/env python3
"""Adapt a qualified Vega-Wuji reference into the Sharpa floating-hand layout.

This is a transitional bridge for the real SOMA references already produced by
the Vega-Wuji pipeline.  It keeps the measured object/support/contact geometry
and wrist poses, while remapping the 20-DOF Vega finger state into the 22-DOF
Sharpa hand model (the extra thumb MCP-AA and pinky CMC joints are initialized
to zero).  It is deliberately separate from the MANO/Pink converter: it does
not claim to retarget MANO again or to improve the underlying human estimate.

Example::

    pixi run -e isaaclab python scripts/data/convert_vega_wuji_npz_to_sharpa.py \
      --input logs/reference_replay/vega_wuji/snack_box_pick_2026-08-24/reference/snack_box_pick_contact_20hz.npz \
      --output-dir data/dexmanip/sharpa/snack_box_pick_contact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iltools.core import (
    ContactSequence,
    DexterousReference,
    create_dexterous_reference_manifest,
    load_dexterous_reference_npz,
    save_dexterous_reference_npz,
    sha256_file,
)


LEFT_SHARPA_JOINTS = (
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)
RIGHT_SHARPA_JOINTS = tuple(
    name.replace("left_", "right_", 1) for name in LEFT_SHARPA_JOINTS
)
SHARPA_FINGERTIP_NAMES = (
    "thumb_DP",
    "index_DP",
    "middle_DP",
    "ring_DP",
    "pinky_DP",
)


def _source_columns(reference: DexterousReference, side: str) -> dict[str, int]:
    prefix = f"{side[0]}_"
    columns = {
        name: index
        for index, name in enumerate(reference.joint_names)
        if name.startswith(prefix)
    }
    if len(columns) != 20:
        raise ValueError(
            f"Expected 20 {side} Vega finger joints, found {len(columns)}: "
            f"{sorted(columns)}"
        )
    return columns


def _map_finger_qpos(
    reference: DexterousReference,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Map the Vega 20-DOF finger block into the 22-DOF Sharpa order."""

    left = _source_columns(reference, "left")
    right = _source_columns(reference, "right")
    source = np.asarray(reference.qpos, dtype=np.float32)
    zeros = np.zeros((reference.frame_count,), dtype=np.float32)

    def build(side: str, columns: dict[str, int]) -> np.ndarray:
        p = side[0]
        values = {
            f"{side}_thumb_CMC_FE": source[:, columns[f"{p}_thumb_cmc_flex"]],
            f"{side}_thumb_CMC_AA": source[:, columns[f"{p}_thumb_cmc_abd"]],
            f"{side}_thumb_MCP_FE": source[:, columns[f"{p}_thumb_mcp"]],
            f"{side}_thumb_MCP_AA": zeros,
            f"{side}_thumb_IP": source[:, columns[f"{p}_thumb_ip"]],
            f"{side}_index_MCP_FE": source[:, columns[f"{p}_index_finger_mcp_flex"]],
            f"{side}_index_MCP_AA": source[:, columns[f"{p}_index_finger_mcp_abd"]],
            f"{side}_index_PIP": source[:, columns[f"{p}_index_finger_pip"]],
            f"{side}_index_DIP": source[:, columns[f"{p}_index_finger_dip"]],
            f"{side}_middle_MCP_FE": source[:, columns[f"{p}_middle_finger_mcp_flex"]],
            f"{side}_middle_MCP_AA": source[:, columns[f"{p}_middle_finger_mcp_abd"]],
            f"{side}_middle_PIP": source[:, columns[f"{p}_middle_finger_pip"]],
            f"{side}_middle_DIP": source[:, columns[f"{p}_middle_finger_dip"]],
            f"{side}_ring_MCP_FE": source[:, columns[f"{p}_ring_finger_mcp_flex"]],
            f"{side}_ring_MCP_AA": source[:, columns[f"{p}_ring_finger_mcp_abd"]],
            f"{side}_ring_PIP": source[:, columns[f"{p}_ring_finger_pip"]],
            f"{side}_ring_DIP": source[:, columns[f"{p}_ring_finger_dip"]],
            f"{side}_pinky_CMC": zeros,
            f"{side}_pinky_MCP_FE": source[:, columns[f"{p}_pinky_mcp_flex"]],
            f"{side}_pinky_MCP_AA": source[:, columns[f"{p}_pinky_mcp_abd"]],
            f"{side}_pinky_PIP": source[:, columns[f"{p}_pinky_pip"]],
            f"{side}_pinky_DIP": source[:, columns[f"{p}_pinky_dip"]],
        }
        names = LEFT_SHARPA_JOINTS if side == "left" else RIGHT_SHARPA_JOINTS
        return np.stack([values[name] for name in names], axis=-1)

    return (
        build("left", left),
        build("right", right),
        LEFT_SHARPA_JOINTS,
        RIGHT_SHARPA_JOINTS,
    )


def _rename_contacts(contacts: ContactSequence | None) -> ContactSequence | None:
    if contacts is None:
        return None
    link_map = {
        "l_mount": "left_hand_C_MC",
        "l_thumb_proximal_abd": "left_thumb_MC",
        "l_thumb_middle": "left_thumb_PP",
        "l_thumb_distal": "left_thumb_DP",
        "l_index_finger_proximal_abd": "left_index_PP",
        "l_index_finger_middle": "left_index_MP",
        "l_index_finger_distal": "left_index_DP",
        "l_middle_finger_proximal_abd": "left_middle_PP",
        "l_middle_finger_middle": "left_middle_MP",
        "l_middle_finger_distal": "left_middle_DP",
        "l_ring_finger_proximal_abd": "left_ring_PP",
        "l_ring_finger_middle": "left_ring_MP",
        "l_ring_finger_distal": "left_ring_DP",
        "l_pinky_proximal_abd": "left_pinky_PP",
        "l_pinky_middle": "left_pinky_MP",
        "l_pinky_distal": "left_pinky_DP",
        "r_mount": "right_hand_C_MC",
        "r_thumb_proximal_abd": "right_thumb_MC",
        "r_thumb_middle": "right_thumb_PP",
        "r_thumb_distal": "right_thumb_DP",
        "r_index_finger_proximal_abd": "right_index_PP",
        "r_index_finger_middle": "right_index_MP",
        "r_index_finger_distal": "right_index_DP",
        "r_middle_finger_proximal_abd": "right_middle_PP",
        "r_middle_finger_middle": "right_middle_MP",
        "r_middle_finger_distal": "right_middle_DP",
        "r_ring_finger_proximal_abd": "right_ring_PP",
        "r_ring_finger_middle": "right_ring_MP",
        "r_ring_finger_distal": "right_ring_DP",
        "r_pinky_proximal_abd": "right_pinky_PP",
        "r_pinky_middle": "right_pinky_MP",
        "r_pinky_distal": "right_pinky_DP",
    }
    names = np.asarray(
        [
            [link_map.get(str(name), str(name)) for name in row]
            for row in contacts.link_names
        ]
    )
    return ContactSequence(
        hand_sides=contacts.hand_sides,
        link_names=names,
        link_positions_w=contacts.link_positions_w.copy(),
        link_normals_w=contacts.link_normals_w.copy(),
        object_positions_w=contacts.object_positions_w.copy(),
        object_normals_w=contacts.object_normals_w.copy(),
        object_indices=contacts.object_indices.copy(),
        active=contacts.active.copy(),
    )


def convert(reference: DexterousReference) -> DexterousReference:
    if reference.robot_name != "vega_wuji":
        raise ValueError(
            f"Expected a Vega-Wuji reference, got {reference.robot_name!r}."
        )
    if len(reference.object_names) != 1:
        raise ValueError("Sharpa v1 accepts exactly one rigid object.")
    if reference.metadata.get("object_kind", "rigid") != "rigid":
        raise ValueError("Sharpa v1 accepts only rigid-object references.")
    if not np.isclose(reference.fps, 20.0):
        raise ValueError("Sharpa references must be sampled at 20 Hz.")
    left_qpos, right_qpos, left_names, right_names = _map_finger_qpos(reference)
    metadata: dict[str, Any] = dict(reference.metadata)
    metadata.update(
        {
            "source_format": "iltools_dexterous_reference/v1",
            "source_robot_name": "vega_wuji",
            "source_adapter": "scripts/data/convert_vega_wuji_npz_to_sharpa.py",
            "source_reference": str(reference.source_path or ""),
            "object_kind": "rigid",
            "physics_backend": "physx",
            "physics_dt": 0.01,
            "control_dt": 0.05,
            "finger_mapping": {
                "left_extra_joints_zeroed": ["left_thumb_MCP_AA", "left_pinky_CMC"],
                "right_extra_joints_zeroed": ["right_thumb_MCP_AA", "right_pinky_CMC"],
                "source_joint_count_per_hand": 20,
                "target_joint_count_per_hand": 22,
            },
        }
    )
    contacts = _rename_contacts(reference.contacts)
    fingertip_left = tuple(f"left_{name}" for name in SHARPA_FINGERTIP_NAMES)
    fingertip_right = tuple(f"right_{name}" for name in SHARPA_FINGERTIP_NAMES)
    return DexterousReference(
        sequence_id=f"{reference.sequence_id}_sharpa",
        robot_name="sharpa_wave",
        robot_layout="dual_floating_hand",
        fps=reference.fps,
        joint_names=left_names + right_names,
        left_joint_names=left_names,
        right_joint_names=right_names,
        qpos=np.concatenate((left_qpos, right_qpos), axis=-1),
        fixed_root_pose_w=np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        left_wrist_pose_w=reference.left_wrist_pose_w,
        right_wrist_pose_w=reference.right_wrist_pose_w,
        left_wrist_frame_name="left_hand_C_MC",
        right_wrist_frame_name="right_hand_C_MC",
        left_hand_frame_names=fingertip_left,
        left_hand_frame_poses_w=reference.left_hand_frame_poses_w,
        right_hand_frame_names=fingertip_right,
        right_hand_frame_poses_w=reference.right_hand_frame_poses_w,
        object_names=reference.object_names,
        object_poses_w=reference.object_poses_w,
        object_asset_paths=reference.object_asset_paths,
        object_asset_sha256=reference.object_asset_sha256,
        object_scales=reference.object_scales,
        object_radii=reference.object_radii,
        support_surface_names=reference.support_surface_names,
        support_surface_asset_paths=reference.support_surface_asset_paths,
        support_surface_asset_sha256=reference.support_surface_asset_sha256,
        support_surface_scales=reference.support_surface_scales,
        support_surface_poses_w=reference.support_surface_poses_w,
        scene_physics=reference.scene_physics,
        contacts=contacts,
        metadata=metadata,
    )


def _write_zarr(reference: DexterousReference, output: Path) -> None:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "zarr is required for the canonical trajectory store"
        ) from exc
    root = zarr.open_group(str(output / "trajectories.zarr"), mode="w")
    group = root.create_group("trajectories").create_group(
        "00000_" + reference.sequence_id
    )
    group.attrs.update(
        {
            "sequence_id": reference.sequence_id,
            "fps": reference.fps,
            "robot_layout": reference.robot_layout,
            "joint_names": list(reference.joint_names),
        }
    )
    arrays = {
        "qpos": reference.qpos,
        "qvel": reference.qvel,
        "left_wrist_pose_w": reference.left_wrist_pose_w,
        "right_wrist_pose_w": reference.right_wrist_pose_w,
        "left_wrist_twist_w": reference.left_wrist_twist_w,
        "right_wrist_twist_w": reference.right_wrist_twist_w,
        "object_poses_w": reference.object_poses_w,
        "object_twists_w": reference.object_twists_w,
        "left_hand_frame_poses_w": reference.left_hand_frame_poses_w,
        "right_hand_frame_poses_w": reference.right_hand_frame_poses_w,
    }
    for name, value in arrays.items():
        group.create_array(name, data=np.asarray(value), overwrite=True)
    if reference.contacts is not None:
        contacts = group.create_group("contacts")
        for name, value in reference.contacts.to_npz_fields().items():
            contacts.create_array(name, data=np.asarray(value), overwrite=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Vega-Wuji v1 NPZ")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_path = args.input.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = convert(load_dexterous_reference_npz(source_path))
    reference.verify_scene_assets(require_hashes=True)
    npz_path = save_dexterous_reference_npz(
        reference, output / "references" / "00000.npz"
    )
    _write_zarr(reference, output)
    manifest = create_dexterous_reference_manifest(
        [npz_path],
        output / "manifest.json",
        dataset_name="sharpa_real_contact",
        metadata={
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "trajectory_store": "trajectories.zarr",
            "adapter": "vega_wuji_fixed_base_to_sharpa_dual_floating_hand",
            "contact_slots_active": int(np.count_nonzero(reference.contacts.active))
            if reference.contacts is not None
            else 0,
        },
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "reference": str(npz_path),
                "frames": reference.frame_count,
                "fps": reference.fps,
                "left_joints": len(reference.left_joint_names),
                "right_joints": len(reference.right_joint_names),
                "active_contact_slots": int(np.count_nonzero(reference.contacts.active))
                if reference.contacts is not None
                else 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
