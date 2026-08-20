"""Restore the joint-trajectory invariants of a retargeted Vega-Wuji Reference.

A retargeted joint trajectory can carry two defects that a Reference consumer
has no way to detect. Redundant wrist joints flip between IK solutions, so the
commanded target jumps within one control frame. Separately, ``qvel`` can be
carried over from a different source than ``qpos``: on the corn-can inspection
Reference the two disagree by up to 17.5 rad/s, so a reset writes a velocity
the Reference's own positions never produce.

This tool low-pass filters the joint trajectory and restores two invariants:

* ``qvel`` is the finite difference of ``qpos`` at the Reference rate.
* every stored Cartesian frame is the forward kinematics of ``qpos``.

Object, support-surface, and contact arrays are copied through unchanged: the
virtual object controller drives the object, and this tool never invents
contact evidence. The output stays an inspection Reference; smoothing changes
the motion and cannot by itself qualify a Reference for training.

Measured scope, on the corn-can scene-clear inspection Reference: filtering
reduces the worst per-frame joint step from 0.90 rad to 0.31 rad (window 13)
and makes ``qvel`` exactly consistent with ``qpos``. It does **not** by itself
lengthen an episode on that Reference, because the binding termination there is
object-driven and this tool does not touch object data.

Usage:

    pixi run -e isaaclab python scripts/data/smooth_vega_wuji_reference.py \\
        --reference IN.npz --output OUT.npz [--window 5] [--polyorder 2]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MJCF = Path(
    "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji/"
    "vega_u_wuji_v2_beta1_with_mount.xml"
)

# Cartesian frame arrays that must stay the forward kinematics of qpos.
_WRIST_FRAMES = (
    ("right_wrist_pose_w", "right_wrist_frame_name"),
    ("left_wrist_pose_w", "left_wrist_frame_name"),
)
_HAND_FRAMES = (
    ("right_hand_frame_poses_w", "right_hand_frame_names"),
    ("left_hand_frame_poses_w", "left_hand_frame_names"),
)


def savitzky_golay(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Filter along axis 0, keeping the endpoints anchored."""

    if window <= 1:
        return values.copy()
    frames = values.shape[0]
    # The window must be odd, at least polyorder + 2, and no longer than the clip.
    window = min(window, frames if frames % 2 else frames - 1)
    if window <= polyorder + 1:
        return values.copy()
    if window % 2 == 0:
        window -= 1
    from scipy.signal import savgol_filter

    # Endpoints are filtered like every other frame. Anchoring them would
    # preserve exactly the discontinuity this tool exists to remove when the
    # worst IK flip sits on the first or last transition, and the episode
    # simply resets to whatever the filtered first frame is.
    return savgol_filter(
        values, window_length=window, polyorder=polyorder, axis=0, mode="interp"
    )


def _frame_pose(model: Any, data: Any, name: str, mujoco: Any) -> np.ndarray:
    """Return one frame pose as XYZ + WXYZ, accepting a site or a body."""

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id >= 0:
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        return np.concatenate([data.site_xpos[site_id], quat])
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"{name!r} is neither a site nor a body in the MJCF.")
    return np.concatenate([data.xpos[body_id], data.xquat[body_id]])


def recompute_kinematics(
    payload: dict[str, Any], qpos: np.ndarray, mjcf_path: Path
) -> dict[str, np.ndarray]:
    """Recompute every stored Cartesian frame from ``qpos`` by forward kinematics."""

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    joint_names = [str(name) for name in payload["joint_names"]]
    address: dict[int, int] = {}
    for index, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Reference joint {name!r} is absent from the MJCF.")
        address[index] = int(model.jnt_qposadr[joint_id])

    root = np.asarray(payload["fixed_root_pose_w"], dtype=np.float64)
    frames = qpos.shape[0]
    wanted: list[tuple[str, list[str]]] = []
    for array_key, name_key in _WRIST_FRAMES:
        if array_key in payload:
            wanted.append((array_key, [str(payload[name_key])]))
    for array_key, name_key in _HAND_FRAMES:
        if array_key in payload:
            wanted.append((array_key, [str(n) for n in payload[name_key]]))

    out = {
        key: np.zeros((frames, len(names), 7), dtype=np.float32)
        for key, names in wanted
    }
    for step in range(frames):
        data.qpos[:] = 0.0
        for index, qadr in address.items():
            data.qpos[qadr] = qpos[step, index]
        mujoco.mj_forward(model, data)
        for key, names in wanted:
            for slot, name in enumerate(names):
                pose = _frame_pose(model, data, name, mujoco)
                pose[:3] = pose[:3] + root[:3]
                out[key][step, slot] = pose
    # A single-frame array is stored without its frame axis.
    for key, names in wanted:
        if len(names) == 1:
            out[key] = out[key][:, 0, :]
    return out


def smooth_reference(
    payload: dict[str, Any], *, window: int, polyorder: int, mjcf_path: Path
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return the repaired payload and a before/after report."""

    qpos = np.asarray(payload["qpos"], dtype=np.float64)
    fps = float(np.asarray(payload["fps"]).reshape(()))
    if fps <= 0.0:
        raise ValueError("Reference fps must be positive.")

    smoothed = savitzky_golay(qpos, window, polyorder)
    # Recompute velocity from the trajectory it belongs to. A Reference whose
    # qvel disagrees with its qpos resets the robot into a state its own
    # positions never produce.
    qvel = np.gradient(smoothed, 1.0 / fps, axis=0)

    out = dict(payload)
    out["qpos"] = smoothed.astype(np.float32)
    out["qvel"] = qvel.astype(np.float32)
    out.update(recompute_kinematics(payload, smoothed, mjcf_path))

    before_step = np.abs(np.diff(qpos, axis=0)).max()
    after_step = np.abs(np.diff(smoothed, axis=0)).max()
    original_qvel = np.asarray(payload["qvel"], dtype=np.float64)
    finite_difference = np.gradient(qpos, 1.0 / fps, axis=0)
    report = {
        "max_joint_step_before_rad": float(before_step),
        "max_joint_step_after_rad": float(after_step),
        "max_qvel_before": float(np.abs(original_qvel).max()),
        "max_qvel_after": float(np.abs(qvel).max()),
        "qvel_qpos_mismatch_before": float(
            np.abs(original_qvel - finite_difference).max()
        ),
        "qvel_qpos_mismatch_after": float(
            np.abs(qvel - np.gradient(smoothed, 1.0 / fps, axis=0)).max()
        ),
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    args = parser.parse_args(argv)

    with np.load(args.reference, allow_pickle=True) as handle:
        payload = {key: handle[key] for key in handle.files}
    repaired, report = smooth_reference(
        payload,
        window=args.window,
        polyorder=args.polyorder,
        mjcf_path=args.mjcf,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **repaired)
    for name, value in report.items():
        print(f"{name}: {value:.4f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
