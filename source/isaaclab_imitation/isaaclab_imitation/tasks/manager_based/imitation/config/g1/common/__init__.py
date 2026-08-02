# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared building blocks for the G1 imitation task configs.

Layout:

- ``constants``: joint/body name tables and randomization ranges.
- ``presets``: physics/robot/contact-sensor backend presets.
- ``actions`` / ``events`` / ``rewards`` / ``terminations``: manager term
  blocks for the mimic baseline and the SONIC release recipe.
- ``observations``: every G1 observation surface (vanilla / latent / sonic / lean / full), composed from the shared term factories
  observation surfaces (checkpoint input contracts).
- ``tracking_env``: the shared base env class (vanilla component defaults +
  LAFAN1 manifest machinery), recipe helpers, Hydra plumbing.
- ``latent_env``: the latent-lineage base env class.

Release assemblies live in ``imitation_g1_env_v0.py`` / ``..._v1.py`` and
non-default assemblies in ``variants/``; both compose from this package.
"""

from .actions import G1ActionsCfg, G1SonicActionsCfg
from .constants import (
    G1_29DOF_DATASET_BODY_NAMES,
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_29DOF_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_OBS_ANCHOR_BODY_NAME,
    G1_TRACKED_BODY_NAMES,
    VELOCITY_RANGE,
)
from .events import G1EventCfg, G1SonicEventCfg
from .latent_env import ImitationG1LatentEnvCfg
from .observations import G1ObservationCfg
from .observations import (
    G1LatentObservationCfg,
    G1SonicLatentObservationCfg,
)
from .presets import (
    G1ImitationContactSensorCfg,
    G1ImitationPhysicsCfg,
    G1ImitationRobotCfg,
    G1SonicRobotCfg,
)
from .rewards import G1RewardsCfg, G1SonicRewardsCfg
from .terminations import (
    G1SonicTerminationCurriculumCfg,
    G1SonicTerminationsCfg,
    G1TerminationsCfg,
)
from .tracking_env import ImitationG1BaseTrackingEnvCfg

__all__ = [
    "G1ActionsCfg",
    "G1SonicActionsCfg",
    "G1_29DOF_DATASET_BODY_NAMES",
    "G1_29DOF_ISAACLAB_JOINT_NAMES",
    "G1_29DOF_JOINT_NAMES",
    "G1_EE_BODY_NAMES",
    "G1_KEYPOINT5_BODY_NAMES",
    "G1_OBS_ANCHOR_BODY_NAME",
    "G1_TRACKED_BODY_NAMES",
    "VELOCITY_RANGE",
    "G1EventCfg",
    "G1SonicEventCfg",
    "ImitationG1LatentEnvCfg",
    "G1ObservationCfg",
    "G1LatentObservationCfg",
    "G1SonicLatentObservationCfg",
    "G1ImitationContactSensorCfg",
    "G1ImitationPhysicsCfg",
    "G1ImitationRobotCfg",
    "G1SonicRobotCfg",
    "G1RewardsCfg",
    "G1SonicRewardsCfg",
    "G1SonicTerminationCurriculumCfg",
    "G1SonicTerminationsCfg",
    "G1TerminationsCfg",
    "ImitationG1BaseTrackingEnvCfg",
]
