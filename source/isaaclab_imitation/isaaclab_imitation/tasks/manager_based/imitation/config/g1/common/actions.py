# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configurations for the 29-DoF G1 tracking tasks."""

from isaaclab.utils.configclass import configclass

from isaaclab_imitation.assets.robots.unitree import (
    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    UNITREE_G1_29DOF_SONIC_ACTION_SCALE,
)

from .... import mdp
from .constants import G1_29DOF_ISAACLAB_JOINT_NAMES


@configclass
class G1ActionsCfg:
    """Action settings for 29-DoF mimic G1."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
        preserve_order=True,
        scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class G1SonicActionsCfg(G1ActionsCfg):
    """Action scale induced by SONIC's released actuator configuration."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
        preserve_order=True,
        scale=UNITREE_G1_29DOF_SONIC_ACTION_SCALE,
        use_default_offset=True,
    )
