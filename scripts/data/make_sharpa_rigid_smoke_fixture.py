#!/usr/bin/env python3
"""Create a small rigid Sharpa reference for import and simulation smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from iltools.core import (
    ContactSequence,
    DexterousReference,
    ScenePhysics,
    create_dexterous_reference_manifest,
    save_dexterous_reference_npz,
    sha256_file,
)
from iltools.datasets import (
    ManoSharpaLoader,
    make_rigid_proxy_row,
    resample_mano_sharpa_row,
)
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = (
    REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets/sharpa_wave"
)


def _names(urdf: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = ET.parse(urdf).getroot()
    joints = tuple(
        str(item.attrib["name"])
        for item in root.findall("joint")
        if item.attrib.get("type") != "fixed"
    )
    links = tuple(str(item.attrib["name"]) for item in root.findall("link"))
    if len(joints) != 22:
        raise ValueError(f"Expected 22 Sharpa joints in {urdf}; found {len(joints)}.")
    return joints, links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument(
        "--source-parquet",
        type=Path,
        help="Optional processed ManoSharpaData fixture; its first object body becomes the rigid proxy.",
    )
    return parser.parse_args()


def _proxy_physics() -> ScenePhysics:
    return ScenePhysics(
        object_mass_kg=np.asarray([0.2], dtype=np.float32),
        object_center_of_mass_m=np.zeros((1, 3), dtype=np.float32),
        object_diagonal_inertia_kg_m2=np.asarray(
            [[5.33e-4, 5.33e-4, 5.33e-4]], dtype=np.float32
        ),
        object_static_friction=np.asarray([1.0], dtype=np.float32),
        object_dynamic_friction=np.asarray([1.0], dtype=np.float32),
        object_restitution=np.asarray([0.0], dtype=np.float32),
        support_static_friction=np.zeros((0,), dtype=np.float32),
        support_dynamic_friction=np.zeros((0,), dtype=np.float32),
        support_restitution=np.zeros((0,), dtype=np.float32),
    )


def _source_proxy_reference(source: Path, object_asset: Path) -> DexterousReference:
    rows = ManoSharpaLoader(source).rows()
    if not rows:
        raise ValueError("The source fixture contains no trajectories.")
    row = make_rigid_proxy_row(rows[0], object_body_index=0)
    row = resample_mano_sharpa_row(row, target_fps=20.0)
    reference = ManoSharpaLoader.to_reference(
        row,
        object_asset_path=object_asset,
        object_asset_sha256=sha256_file(object_asset),
    )
    reference.object_names = ("object",)
    reference.scene_physics = _proxy_physics()
    reference.metadata.update(
        {
            "object_kind": "rigid",
            "fixture": True,
            "rigid_proxy_for_source_articulation": True,
            "physics_backend": "physx",
            "physics_dt": 0.01,
            "control_dt": 0.05,
        }
    )
    return reference


def main() -> None:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least two.")
    left_joints, left_links = _names(
        ASSET_ROOT / "urdfs/sharpawave/left_sharpa_wave.urdf"
    )
    right_joints, right_links = _names(
        ASSET_ROOT / "urdfs/sharpawave/right_sharpa_wave.urdf"
    )
    frames = int(args.frames)
    time = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    left_wrist = np.zeros((frames, 7), dtype=np.float32)
    right_wrist = np.zeros((frames, 7), dtype=np.float32)
    object_pose = np.zeros((frames, 1, 7), dtype=np.float32)
    left_wrist[:, :3] = np.stack(
        (0.03 * time, -0.15 + 0.0 * time, 0.55 + 0.0 * time), axis=-1
    )
    right_wrist[:, :3] = np.stack(
        (0.03 * time, 0.15 + 0.0 * time, 0.55 + 0.0 * time), axis=-1
    )
    object_pose[:, 0, :3] = np.stack(
        (0.03 * time, 0.0 * time, 0.48 + 0.0 * time), axis=-1
    )
    left_wrist[:, 3] = right_wrist[:, 3] = object_pose[:, 0, 3] = 1.0

    def link_poses(names: tuple[str, ...], wrist: np.ndarray) -> np.ndarray:
        poses = np.repeat(wrist[:, None, :], len(names), axis=1)
        tip_ids = [index for index, name in enumerate(names) if name.endswith("_DP")]
        for offset, index in enumerate(tip_ids):
            poses[:, index, 0] += 0.05 + 0.01 * offset
        return poses

    qpos = np.zeros((frames, 44), dtype=np.float32)
    qpos[:, :] = 0.15 * np.sin(time[:, None] * np.pi)
    contact_names = (("left_index_DP",), ("right_index_DP",))
    active = np.zeros((frames, 2, 1), dtype=bool)
    active[frames // 3 : 2 * frames // 3] = True
    contact_positions = np.zeros((frames, 2, 1, 3), dtype=np.float32)
    contact_positions[active] = np.tile(
        np.asarray([0.0, 0.0, 0.48]), (int(active.sum()), 1)
    )
    contacts = ContactSequence(
        hand_sides=("left", "right"),
        link_names=np.asarray(contact_names),
        link_positions_w=contact_positions,
        link_normals_w=np.where(active[..., None], np.asarray([0.0, 0.0, 1.0]), 0.0),
        object_positions_w=contact_positions.copy(),
        object_normals_w=np.where(active[..., None], np.asarray([0.0, 0.0, -1.0]), 0.0),
        object_indices=np.zeros((frames, 2, 1), dtype=np.int32),
        active=active,
    )
    object_asset = ASSET_ROOT / "rigid_box_proxy.usda"
    if args.source_parquet is not None:
        reference = _source_proxy_reference(args.source_parquet, object_asset)
    else:
        reference = DexterousReference(
            sequence_id="sharpa_rigid_smoke",
            robot_name="sharpa_wave",
            robot_layout="dual_floating_hand",
            fps=20.0,
            joint_names=left_joints + right_joints,
            left_joint_names=left_joints,
            right_joint_names=right_joints,
            qpos=qpos,
            fixed_root_pose_w=np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            left_wrist_pose_w=left_wrist,
            right_wrist_pose_w=right_wrist,
            left_wrist_frame_name="left_hand_C_MC",
            right_wrist_frame_name="right_hand_C_MC",
            left_hand_frame_names=left_links,
            left_hand_frame_poses_w=link_poses(left_links, left_wrist),
            right_hand_frame_names=right_links,
            right_hand_frame_poses_w=link_poses(right_links, right_wrist),
            object_names=("object",),
            object_poses_w=object_pose,
            object_asset_paths=(str(object_asset.resolve()),),
            object_asset_sha256=(sha256_file(object_asset),),
            object_radii=np.asarray([0.04], dtype=np.float32),
            contacts=contacts,
            scene_physics=_proxy_physics(),
            metadata={
                "object_kind": "rigid",
                "fixture": True,
                "physics_backend": "physx",
                "physics_dt": 0.01,
                "control_dt": 0.05,
            },
        )
    args.output.mkdir(parents=True, exist_ok=True)
    npz = save_dexterous_reference_npz(
        reference, args.output / "sharpa_rigid_smoke.npz"
    )
    manifest = create_dexterous_reference_manifest(
        [npz], args.output / "manifest.json", dataset_name="sharpa_rigid_smoke"
    )
    print(manifest)


if __name__ == "__main__":
    main()
