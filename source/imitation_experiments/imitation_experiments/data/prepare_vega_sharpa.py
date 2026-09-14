"""Reproducible mounted-arm preparation for the first local PPO test."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from iltools.core import (
    load_dexterous_reference_set,
    save_dexterous_reference_npz,
    sha256_file,
)
from iltools.retarget.mounted_hands import (
    MountedHandConfig,
    crop_mounted_reference,
    project_mounted_self_collisions,
    refresh_mounted_geometry,
    retarget_mounted_hands,
    retime_mounted_reference,
)
from iltools.retarget.mounted_scene import project_contact_constraints
from imitation_experiments.data.vega_sharpa_model import base_ground_alignment
from imitation_experiments.data.vega_sharpa_reference import publish_local_reference
from imitation_experiments.data.vega_sharpa_scene import build_audit_model


def prepare(args):
    source = load_dexterous_reference_set(args.source_manifest)[0]
    scene = json.loads(args.scene.read_text())
    asset = json.loads(args.robot_asset.read_text())
    if sha256_file(args.model) != asset["model_sha256"]:
        raise ValueError("The converted robot asset uses a different model.")
    args.output.mkdir(parents=True, exist_ok=True)
    intermediate = args.output / "intermediate"
    intermediate.mkdir(exist_ok=True)
    automatic_height = args.base_xyz_yaw is None
    x, y, z, yaw = args.base_xyz_yaw or (-0.04, 0.55, 0.0, -90)
    base = (
        x,
        y,
        z,
        *Rotation.from_euler("z", yaw, degrees=True).as_quat(scalar_first=True),
    )
    alignment = base_ground_alignment(args.model, base)
    if automatic_height:
        base = (*base[:2], alignment["ground_aligned_root_z_m"], *base[3:])
        alignment = base_ground_alignment(args.model, base)
    if alignment["base_ground_clearance_m"] < -1e-5:
        raise ValueError(
            "The robot base intersects the z=0 floor. Omit --base-xyz-yaw for "
            f"automatic alignment or set root z >= {alignment['ground_aligned_root_z_m']:.6f} m."
        )
    print(
        f"Fixed root z={base[2]:.6f} m; base floor clearance={alignment['base_ground_clearance_m']:.6f} m",
        flush=True,
    )
    cfg = MountedHandConfig(
        arm_joint_names=tuple(
            f"{side}_arm_j{i}" for side in ("L", "R") for i in range(1, 8)
        ),
        left_wrist_site="left_mounted_wrist",
        right_wrist_site="right_mounted_wrist",
        fixed_root_pose_w=base,
        robot_name="vega_u_sharpa",
        initial_arm_qpos=(0.0, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0) * 2,
    )
    print("Retargeting arms against the converted assembly...", flush=True)
    mounted, ik = retarget_mounted_hands(source, args.model, cfg)
    save_dexterous_reference_npz(mounted, intermediate / "full_arm_retarget.npz")
    np.savez_compressed(intermediate / "arm_ik_audit.npz", **ik)
    mounted = crop_mounted_reference(mounted, args.start_frame, args.stop_frame)
    print("Projecting hand self collisions...", flush=True)
    mounted, self_report = project_mounted_self_collisions(mounted, args.model)
    fingers = tuple(n for n in mounted.joint_names if "_arm_j" not in n)
    model = build_audit_model(args.model, mounted.fixed_root_pose_w, scene)
    print("Correcting hand/object contacts with fixed wrists...", flush=True)
    qpos, finger_report = project_contact_constraints(
        model, mounted, variable_joint_names=fingers
    )
    mounted = refresh_mounted_geometry(mounted, args.model, qpos)
    print("Correcting residual contacts with bounded arm motion...", flush=True)
    qpos, arm_report = project_contact_constraints(
        model, mounted, joint_change_limits={n: 0.05 for n in cfg.arm_joint_names}
    )
    if arm_report["maximum_penetration_after_m"] > 0.001:
        raise ValueError("Bounded contact correction failed the 1 mm limit.")
    mounted = refresh_mounted_geometry(mounted, args.model, qpos)
    save_dexterous_reference_npz(mounted, intermediate / "contact_projected.npz")
    print(
        f"Retiming by {args.time_stretch} and checking interpolated poses...",
        flush=True,
    )
    mounted = retime_mounted_reference(mounted, args.model, args.time_stretch)
    qpos, retimed_report = project_contact_constraints(
        model, mounted, variable_joint_names=fingers
    )
    mounted = refresh_mounted_geometry(mounted, args.model, qpos)
    if retimed_report["maximum_penetration_after_m"] > 0.001:
        raise ValueError("Retimed contacts failed the 1 mm limit.")

    source_times = np.arange(source.frame_count)
    sample_times = np.linspace(
        args.start_frame, args.stop_frame - 1, mounted.frame_count
    )
    source_errors = {}
    for side in ("left", "right"):
        target = getattr(source, f"{side}_wrist_pose_w")
        achieved = getattr(mounted, f"{side}_wrist_pose_w")
        positions = np.stack(
            [np.interp(sample_times, source_times, target[:, i]) for i in range(3)],
            axis=-1,
        )
        rotations = Slerp(
            source_times, Rotation.from_quat(target[:, 3:], scalar_first=True)
        )(sample_times)
        position_error = np.linalg.norm(achieved[:, :3] - positions, axis=-1)
        angle_error = (
            Rotation.from_quat(achieved[:, 3:], scalar_first=True) * rotations.inv()
        ).magnitude()
        source_errors[f"{side}_wrist_position_error_m"] = position_error
        source_errors[f"{side}_wrist_orientation_error_rad"] = angle_error
        if position_error.max() > 0.05 or angle_error.max() > 0.2:
            raise ValueError(
                f"{side} wrist correction exceeds the declared 50 mm / 0.2 rad source-fidelity limit."
            )
    np.savez_compressed(args.output / "source_fidelity_audit.npz", **source_errors)
    recipe = {
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "source_reference_sha256": sha256_file(source.source_path),
        "model_sha256": sha256_file(args.model),
        "arm_config": asdict(cfg),
        "base_ground_alignment": alignment,
        "source_frame_range": [args.start_frame, args.stop_frame],
        "time_stretch": args.time_stretch,
        "source_wrist_position_limit_m": 0.05,
        "source_wrist_orientation_limit_rad": 0.2,
        "collision_penetration_limit_m": 0.001,
        "self_projection": self_report,
        "finger_projection": finger_report,
        "bounded_arm_projection": arm_report,
        "retimed_projection": retimed_report,
        "source_fidelity_maximum": {
            name: float(values.max()) for name, values in source_errors.items()
        },
    }
    mounted.metadata["mounted_preparation_recipe"] = {
        key: value for key, value in recipe.items() if not key.endswith("projection")
    }
    (args.output / "preparation.json").write_text(json.dumps(recipe, indent=2) + "\n")
    manifest = publish_local_reference(mounted, args.model, scene, asset, args.output)
    print(f"Prepared local Reference: {manifest}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--robot-asset", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=603)
    parser.add_argument("--stop-frame", type=int, default=765)
    parser.add_argument("--time-stretch", type=int, default=3)
    parser.add_argument(
        "--base-xyz-yaw",
        type=float,
        nargs=4,
        default=None,
        help="Explicit station XYZ/yaw (degrees). Default: x=-0.04, y=0.55, yaw=-90, z derived from the base collision mesh and z=0 floor.",
    )
    args = parser.parse_args()
    for name in ("source_manifest", "model", "robot_asset", "scene", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
