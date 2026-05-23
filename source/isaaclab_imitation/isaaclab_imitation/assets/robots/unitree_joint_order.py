"""Unitree G1 joint-order constants derived from vendored robot assets."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


UNITREE_G1_DESCRIPTION_ROOT = (
    Path(__file__).resolve().parents[1] / "unitree" / "g1_description"
)
UNITREE_G1_29DOF_XML_FILE = UNITREE_G1_DESCRIPTION_ROOT / "g1_29dof_rev_1_0.xml"
UNITREE_G1_29DOF_URDF_FILE = UNITREE_G1_DESCRIPTION_ROOT / "g1_29dof_rev_1_0.urdf"
UNITREE_G1_29DOF_USD_FILE = UNITREE_G1_DESCRIPTION_ROOT / "g1_29dof_rev_1_0.usd"

_G1_29DOF_JOINT_COUNT = 29

# This matches Unitree's public G1_29_JointIndex enum in
# https://github.com/unitreerobotics/unitree_lerobot and is kept here as an
# upstream sanity check for the vendored XML/URDF files.
UNITREE_LEROBOT_G1_29DOF_JOINT_NAMES: tuple[str, ...] = (
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


def _validate_joint_names(
    joint_names: tuple[str, ...], *, label: str
) -> tuple[str, ...]:
    if len(joint_names) != _G1_29DOF_JOINT_COUNT:
        raise RuntimeError(
            f"{label} must contain {_G1_29DOF_JOINT_COUNT} G1 joints, "
            f"got {len(joint_names)}."
        )
    duplicates = sorted({name for name in joint_names if joint_names.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"{label} contains duplicate joints: {duplicates}.")
    return joint_names


def _read_mujoco_actuator_joint_names(xml_path: Path) -> tuple[str, ...]:
    root = ElementTree.parse(xml_path).getroot()
    joint_names = tuple(
        motor.attrib["joint"]
        for motor in root.findall("./actuator/motor")
        if "joint" in motor.attrib
    )
    return _validate_joint_names(joint_names, label=f"{xml_path.name} actuator order")


def _read_urdf_revolute_joint_names(urdf_path: Path) -> tuple[str, ...]:
    root = ElementTree.parse(urdf_path).getroot()
    joint_names = tuple(
        joint.attrib["name"]
        for joint in root.findall("./joint")
        if joint.attrib.get("type") in {"revolute", "continuous"}
    )
    return _validate_joint_names(joint_names, label=f"{urdf_path.name} revolute order")


UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES = _read_mujoco_actuator_joint_names(
    UNITREE_G1_29DOF_XML_FILE
)
UNITREE_G1_29DOF_URDF_REVOLUTE_JOINT_NAMES = _read_urdf_revolute_joint_names(
    UNITREE_G1_29DOF_URDF_FILE
)

if UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES != UNITREE_G1_29DOF_URDF_REVOLUTE_JOINT_NAMES:
    raise RuntimeError(
        "Vendored Unitree G1 XML actuator order does not match URDF revolute "
        "joint order."
    )

if UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES != UNITREE_LEROBOT_G1_29DOF_JOINT_NAMES:
    raise RuntimeError(
        "Vendored Unitree G1 XML actuator order does not match Unitree's public "
        "G1_29_JointIndex order."
    )

UNITREE_G1_29DOF_SDK_JOINT_NAMES = UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES
UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES = UNITREE_G1_29DOF_SDK_JOINT_NAMES
UNITREE_G1_29DOF_JOINT_ORDER_SOURCE = (
    "Unitree G1_29_JointIndex order, cross-checked against vendored MuJoCo "
    "actuator order and URDF revolute joint order."
)
