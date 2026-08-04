# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared G1 name tables and randomization ranges.

These lists are contract constants: joint/body orders are pinned here rather
than derived from the live articulation because the PhysX and Newton backends
enumerate joints/bodies differently. External tools also parse this module
textually (see ``scripts/data/convert_bones_seed_full.py``), so keep the
assignments as plain literals.
"""

from isaaclab_imitation.assets.robots.unitree import (
    UNITREE_G1_29DOF_SDK_JOINT_NAMES,
)

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

G1_29DOF_JOINT_NAMES: list[str] = list(UNITREE_G1_29DOF_SDK_JOINT_NAMES)

# IsaacLab G1 articulation (USD) joint order, i.e. the order of
# ``robot.joint_names`` / ``robot.data.joint_pos.torch`` at runtime. This is a
# breadth-first (level-order) traversal and is NOT the Unitree SDK/URDF order.
# The env applies the reference directly to the articulation, so this is the
# ground-truth ``target_joint_names``. Verified against a live articulation via
# ``robot.joint_names``; guarded at runtime in the env.
G1_29DOF_ISAACLAB_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Canonical body order of the recorded NPZ reference datasets. The NPZ body
# arrays carry no body-name metadata; they were recorded from the PhysX
# articulation, whose breadth-first (level-order) body enumeration is captured
# here. Do NOT derive this from the live robot at runtime: the Newton backend
# enumerates bodies depth-first per limb, which silently permutes the mapping.
G1_29DOF_DATASET_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# Body tracking set aligned with the original Unitree G1 mimic tracking config.
G1_TRACKED_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_EE_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

# Sparse keypoint set for explicit-interface ablations: the four end-effectors
# plus the pelvis. Position and rot6d orientation are exposed as separate terms,
# so configs can select point targets (5 x 3) or full poses (5 x 9). The pelvis
# is not redundant with the torso_link anchor -- the waist joints separate them.
G1_WRIST_BODY_NAMES: list[str] = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
"""The hands, for an end-effector tracking reward.

Separate from :data:`G1_EE_BODY_NAMES`, which also contains the ankles: those
already carry a dedicated 3D reward (`motion_foot_pos`) and a 3D termination
(`foot_pos_xyz`), so including them here would double-count the feet and
dilute the wrist signal this term exists to supply.
"""


G1_FOOT_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
"""The bodies `foot_pos_xyz` terminates on.

Named separately so the reward that mirrors that termination cannot drift away
from the body set the termination actually uses.
"""


G1_KEYPOINT5_BODY_NAMES: list[str] = [
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

G1_OBS_ANCHOR_BODY_NAME = "torso_link"
