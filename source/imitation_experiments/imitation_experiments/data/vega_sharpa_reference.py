"""Publish an audited mounted Reference for local Isaac integration tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from iltools.core import (
    ScenePhysics,
    create_dexterous_reference_manifest,
    save_dexterous_reference_npz,
)
from iltools.retarget.mounted_scene import contact_geometry_and_clearance
from imitation_experiments.data.vega_sharpa_model import (
    HAND_CONTACT_SUFFIXES,
    base_ground_alignment,
)
from imitation_experiments.data.vega_sharpa_scene import build_audit_model


def publish_local_reference(
    reference, model_path: Path, scene: dict, robot_asset: dict, output: Path
) -> Path:
    """Recompute contacts and record collision-checked, speed-bounded resets.

    This publishes a native-audited input for runtime checks. It does not
    claim Isaac qualification or successful unassisted replay.
    """
    output.mkdir(parents=True, exist_ok=True)
    alignment = base_ground_alignment(model_path, reference.fixed_root_pose_w)
    if alignment["base_ground_clearance_m"] < -1e-5:
        raise ValueError("The fixed robot base intersects the z=0 floor.")
    model = build_audit_model(model_path, reference.fixed_root_pose_w, scene)
    contacts, audit = contact_geometry_and_clearance(
        model,
        reference,
        {
            side: [f"{side}_{suffix}" for suffix in HAND_CONTACT_SUFFIXES]
            for side in ("left", "right")
        },
    )
    arms = [i for i, n in enumerate(reference.joint_names) if "_arm_j" in n]
    audit["base_ground_clearance_m"] = np.full(
        reference.frame_count, alignment["base_ground_clearance_m"]
    )
    fingers = [i for i, n in enumerate(reference.joint_names) if "_arm_j" not in n]
    audit["arm_speed_rad_s"] = np.abs(reference.qvel[:, arms]).max(-1)
    audit["finger_speed_rad_s"] = np.abs(reference.qvel[:, fingers]).max(-1)
    for key, columns in (("arm_speed_rad_s", arms), ("finger_speed_rad_s", fingers)):
        interval_speed = np.abs(
            np.diff(reference.qpos[:, columns], axis=0) * reference.fps
        ).max(-1)
        audit[key][:-1] = np.maximum(audit[key][:-1], interval_speed)
        audit[key][1:] = np.maximum(audit[key][1:], interval_speed)
    if (
        audit["arm_speed_rad_s"].max() > 2.4
        or audit["finger_speed_rad_s"].max() > 11.62
    ):
        raise ValueError(
            "The Reference exceeds joint-speed limits; retime and re-audit it."
        )
    collision_ok = np.ones(reference.frame_count, dtype=bool)
    for key in (
        "self_penetration_m",
        "robot_support_penetration_m",
        "robot_object_penetration_m",
        "object_support_penetration_m",
    ):
        collision_ok &= audit[key] <= 0.001
    if not collision_ok.all():
        raise ValueError("The local Reference still exceeds 1 mm penetration.")
    good = (
        collision_ok
        & (audit["arm_speed_rad_s"] <= 2.4)
        & (audit["finger_speed_rad_s"] <= 11.62)
    )
    good[-24:] = False
    reset_frames = np.flatnonzero(good).tolist()
    if not reset_frames:
        raise ValueError("No valid local-training reset frames remain.")
    metadata = {
        **reference.metadata,
        "contact_geometry_pending": False,
        "mounted_robot_asset": robot_asset,
        "mounted_scene": scene,
        "base_ground_alignment": alignment,
        "mounted_retarget": {
            **reference.metadata["mounted_retarget"],
            "reset_frames": reset_frames,
            "geometry_and_speed_feasible_frames": list(range(reference.frame_count)),
            "final_geometry_and_speed_checked": True,
        },
        "mounted_data_status": "native collision audit passed; Isaac runtime checks pending",
        "contact_geometry_provenance": "MuJoCo signed closest points on explicit runtime convex pieces; active within 10 mm",
    }
    reference = replace(
        reference,
        contacts=contacts,
        metadata=metadata,
        object_asset_paths=(scene["object_usd"],),
        object_asset_sha256=(scene["object_usd_sha256"],),
        support_surface_names=("stand",),
        support_surface_asset_paths=(scene["stand_usd"],),
        support_surface_asset_sha256=(scene["stand_usd_sha256"],),
        support_surface_scales=np.ones((1, 3)),
        support_surface_poses_w=np.asarray([scene["stand_pose_w"]]),
        collision_asset_dependencies=(),
        scene_physics=ScenePhysics(
            object_mass_kg=np.array([scene["object_mass_kg"]]),
            object_center_of_mass_m=np.asarray([scene["object_center_of_mass_m"]]),
            object_diagonal_inertia_kg_m2=np.asarray(
                [scene["object_diagonal_inertia_kg_m2"]]
            ),
            object_static_friction=np.array([1.0]),
            object_dynamic_friction=np.array([1.0]),
            object_restitution=np.array([0.0]),
            support_static_friction=np.array([1.0]),
            support_dynamic_friction=np.array([1.0]),
            support_restitution=np.array([0.0]),
        ),
    )
    path = output / "reference.npz"
    save_dexterous_reference_npz(reference, path)
    np.savez_compressed(output / "native_audit.npz", **audit)
    manifest = output / "manifest.json"
    create_dexterous_reference_manifest(
        [path],
        manifest,
        dataset_name="arctic_vega_sharpa_local",
        model_path=model_path,
        metadata={
            "purpose": "local 100-iteration integration test",
            "qualification": "native geometry only; Isaac checks pending",
        },
    )
    return manifest
