"""Object-only endpoint metrics in the Reference's world coordinate frame."""

from __future__ import annotations

import numpy as np


def box_pose_errors(actual_pose_w, reference_pose_w, center_of_mass_b) -> dict:
    """Compare box COM positions and root orientations; poses use XYZ/WXYZ."""
    from scipy.spatial.transform import Rotation

    actual = np.array(actual_pose_w, dtype=np.float64, copy=True)
    target = np.broadcast_to(
        np.asarray(reference_pose_w, dtype=np.float64), actual.shape
    ).copy()
    if actual.ndim != 2 or actual.shape[1] != 7:
        raise ValueError("Box poses must have shape (N, 7).")
    if not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError("Box poses must be finite.")
    offset = np.broadcast_to(
        np.asarray(center_of_mass_b, dtype=np.float64), (len(actual), 3)
    ).copy()
    if not np.isfinite(offset).all():
        raise ValueError("The box center-of-mass offset must be finite.")
    actual_rotation = Rotation.from_quat(actual[:, 3:], scalar_first=True)
    target_rotation = Rotation.from_quat(target[:, 3:], scalar_first=True)
    actual_center = actual[:, :3] + actual_rotation.apply(offset)
    target_center = target[:, :3] + target_rotation.apply(offset)
    return {
        "actual_center_position_w": actual_center,
        "reference_center_position_w": target_center,
        "center_position_error_m": np.linalg.norm(
            actual_center - target_center, axis=-1
        ),
        "root_position_error_m": np.linalg.norm(actual[:, :3] - target[:, :3], axis=-1),
        "orientation_error_rad": (actual_rotation * target_rotation.inv()).magnitude(),
    }


def object_endpoint_metrics(
    actual_pose_w,
    goal_pose_w,
    center_of_mass_b,
    reached_end,
    *,
    position_tolerance_m=0.05,
    orientation_tolerance_rad=0.35,
) -> dict:
    """Score the final object outcome, with every requested trial in the denominator."""
    if (
        not np.isfinite([position_tolerance_m, orientation_tolerance_rad]).all()
        or position_tolerance_m <= 0
        or orientation_tolerance_rad <= 0
    ):
        raise ValueError("Object success tolerances must be positive.")
    errors = box_pose_errors(actual_pose_w, goal_pose_w, center_of_mass_b)
    reached = np.asarray(reached_end, dtype=bool)
    distance = errors["center_position_error_m"]
    angle = errors["orientation_error_rad"]
    if reached.shape != distance.shape or len(reached) == 0:
        raise ValueError("Reached-end flags must cover every evaluated box pose.")
    position_success = reached & (distance <= position_tolerance_m)
    pose_success = position_success & (angle <= orientation_tolerance_rad)

    def summary(values):
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "maximum": float(np.max(values)),
            "per_trial": values.tolist(),
        }

    return {
        "position_point": "box center of mass, transformed consistently for actual and Reference poses",
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_rad": float(orientation_tolerance_rad),
        "trial_count": len(reached),
        "reached_reference_end": reached.tolist(),
        "goal_root_pose_wxyz": np.asarray(goal_pose_w).tolist(),
        "actual_final_root_pose_wxyz": np.asarray(actual_pose_w).tolist(),
        "goal_center_position_w": errors["reference_center_position_w"][0].tolist(),
        "actual_final_center_positions_w": errors["actual_center_position_w"].tolist(),
        "position_error_m": summary(distance),
        "root_position_error_m": summary(errors["root_position_error_m"]),
        "orientation_error_rad": summary(angle),
        "position_success": position_success.tolist(),
        "position_success_count": int(position_success.sum()),
        "position_success_rate": float(position_success.mean()),
        "pose_success": pose_success.tolist(),
        "pose_success_count": int(pose_success.sum()),
        "pose_success_rate": float(pose_success.mean()),
        "position_success_rate_by_tolerance_m": {
            f"{tolerance:g}": float((reached & (distance <= tolerance)).mean())
            for tolerance in sorted({0.02, 0.05, 0.1, float(position_tolerance_m)})
        },
    }
