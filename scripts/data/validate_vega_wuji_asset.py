#!/usr/bin/env python3
"""Validate the external, robot-only Vega-Wuji MJCF contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco as _mujoco
import numpy as np

mujoco: Any = _mujoco


DEFAULT_ACTUATOR_COUNT = 59
WRIST_FRAME_CONTRACT = {
    "right": {"reference_site": "right_palm", "runtime_body": "r_mount"},
    "left": {"reference_site": "left_palm", "runtime_body": "l_mount"},
}
REQUIRED_BODIES = ("R_ee", "L_ee", "r_mount", "l_mount")
REQUIRED_SITES = ("right_palm", "left_palm")
FORBIDDEN_BODIES = ("grasp_cube", "desk")
SCALAR_JOINT_TYPES = frozenset(
    (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
)


def _name_list(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int
) -> tuple[str, ...]:
    names: list[str] = []
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        names.append(name or f"<unnamed:{object_id}>")
    return tuple(names)


def _reference_joint_names(path: Path) -> tuple[str, ...]:
    with np.load(path, allow_pickle=False) as data:
        if "qpos" not in data or "joint_names" not in data:
            raise ValueError("Reference NPZ must contain qpos and joint_names.")
        qpos = np.asarray(data["qpos"])
        if qpos.ndim != 2:
            raise ValueError(
                f"Reference qpos must have shape [frames, joints], got {qpos.shape}."
            )
        names: list[str] = []
        for value in np.asarray(data["joint_names"]).reshape(-1).tolist():
            names.append(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
            )
        if len(names) != qpos.shape[1]:
            raise ValueError(
                "Reference joint_names width does not match qpos: "
                f"{len(names)} != {qpos.shape[1]}."
            )
    if not names or len(set(names)) != len(names):
        raise ValueError("Reference joint_names must be non-empty and unique.")
    return tuple(names)


def validate_vega_wuji_asset(
    model_path: str | Path,
    *,
    reference_path: str | Path | None = None,
    expected_actuator_count: int = DEFAULT_ACTUATOR_COUNT,
) -> dict[str, Any]:
    """Inspect one robot MJCF and return a fail-closed contract record."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Vega-Wuji MJCF not found: {path}")
    if expected_actuator_count < 1:
        raise ValueError("expected_actuator_count must be positive.")

    model = mujoco.MjModel.from_xml_path(str(path))
    joint_names = _name_list(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = _name_list(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    actuator_joint_names: list[str] = []
    actuator_joint_position_limited: list[bool] = []
    actuator_joint_position_limits: list[list[float]] = []
    actuator_joint_types: list[str] = []
    unresolved_actuators: list[str] = []
    for actuator_id, actuator_name in enumerate(actuator_names):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            unresolved_actuators.append(actuator_name)
            continue
        actuator_joint_names.append(joint_names[joint_id])
        actuator_joint_position_limited.append(bool(model.jnt_limited[joint_id]))
        actuator_joint_position_limits.append(
            np.asarray(model.jnt_range[joint_id], dtype=np.float64).tolist()
        )
        actuator_joint_types.append(mujoco.mjtJoint(int(model.jnt_type[joint_id])).name)

    scalar_joint_names = tuple(
        name
        for joint_id, name in enumerate(joint_names)
        if int(model.jnt_type[joint_id]) in SCALAR_JOINT_TYPES
    )
    body_names = _name_list(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    body_name_set = set(body_names)
    site_names = set(_name_list(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite))
    missing_bodies = [name for name in REQUIRED_BODIES if name not in body_name_set]
    missing_sites = [name for name in REQUIRED_SITES if name not in site_names]
    forbidden_bodies = [name for name in FORBIDDEN_BODIES if name in body_name_set]
    wrist_frames: dict[str, dict[str, Any]] = {}
    wrist_frame_issues: list[str] = []
    for side, frame_contract in WRIST_FRAME_CONTRACT.items():
        site_name = frame_contract["reference_site"]
        runtime_body_name = frame_contract["runtime_body"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        runtime_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, runtime_body_name
        )
        if site_id < 0 or runtime_body_id < 0:
            continue
        site_body_id = int(model.site_bodyid[site_id])
        site_body_name = body_names[site_body_id]
        site_position = np.asarray(model.site_pos[site_id], dtype=np.float64)
        site_quaternion = np.asarray(model.site_quat[site_id], dtype=np.float64)
        identity = bool(
            np.allclose(site_position, 0.0, rtol=0.0, atol=1.0e-9)
            and np.allclose(
                np.abs(site_quaternion),
                np.asarray((1.0, 0.0, 0.0, 0.0)),
                rtol=0.0,
                atol=1.0e-9,
            )
        )
        wrist_frames[side] = {
            "reference_site": site_name,
            "runtime_body": runtime_body_name,
            "site_body": site_body_name,
            "site_pos": site_position.tolist(),
            "site_quat_wxyz": site_quaternion.tolist(),
            "identity_on_runtime_body": site_body_id == runtime_body_id and identity,
        }
        if site_body_id != runtime_body_id or not identity:
            wrist_frame_issues.append(
                f"{site_name} must be an identity site on {runtime_body_name}"
            )

    reference_names: tuple[str, ...] | None = None
    if reference_path is not None:
        reference_names = _reference_joint_names(
            Path(reference_path).expanduser().resolve()
        )

    issues: list[str] = []
    if model.nu != expected_actuator_count:
        issues.append(f"expected {expected_actuator_count} actuators, found {model.nu}")
    if unresolved_actuators:
        issues.append(
            "actuators without a joint transmission: " + ", ".join(unresolved_actuators)
        )
    if missing_bodies:
        issues.append("missing required bodies: " + ", ".join(missing_bodies))
    if missing_sites:
        issues.append("missing required sites: " + ", ".join(missing_sites))
    if forbidden_bodies:
        issues.append(
            "robot-only MJCF contains scene bodies: " + ", ".join(forbidden_bodies)
        )
    issues.extend(wrist_frame_issues)
    if len(actuator_joint_names) != len(set(actuator_joint_names)):
        issues.append("actuator joint names are not unique")
    if set(actuator_joint_names) != set(scalar_joint_names):
        issues.append(
            "actuator transmissions do not cover exactly the scalar robot joints"
        )
    unlimited_actuator_joints = [
        name
        for name, limited in zip(
            actuator_joint_names,
            actuator_joint_position_limited,
            strict=True,
        )
        if not limited
    ]
    if unlimited_actuator_joints:
        issues.append(
            "actuated robot joints without position limits: "
            + ", ".join(unlimited_actuator_joints)
        )
    if reference_names is not None and reference_names != tuple(actuator_joint_names):
        issues.append("reference joint_names do not match actuator joint order")

    return {
        "model_path": str(path),
        "valid": not issues,
        "issues": issues,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        # The control/reference order is the MuJoCo actuator transmission
        # order. XML joint declaration order is not a control contract.
        "joint_names": actuator_joint_names,
        "actuator_names": list(actuator_names),
        "actuator_joint_names": actuator_joint_names,
        "actuator_joint_types": actuator_joint_types,
        "actuator_joint_position_limited": actuator_joint_position_limited,
        "actuator_joint_position_limits": actuator_joint_position_limits,
        "body_names": list(body_names),
        "missing_required_bodies": missing_bodies,
        "missing_required_sites": missing_sites,
        "wrist_frames": wrist_frames,
        "forbidden_scene_bodies": forbidden_bodies,
        "reference_joint_names": (
            None if reference_names is None else list(reference_names)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the external robot-only Vega-Wuji MJCF contract."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional named joint-reference NPZ to compare with actuator order.",
    )
    parser.add_argument(
        "--expected-actuator-count",
        type=int,
        default=DEFAULT_ACTUATOR_COUNT,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    record = validate_vega_wuji_asset(
        args.model,
        reference_path=args.reference,
        expected_actuator_count=args.expected_actuator_count,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
