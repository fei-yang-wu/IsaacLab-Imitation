# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DEPRECATED module path -- import from ``common``/``v0``/``variants`` instead.

Kept as a pure re-export shim so historical imports, pickled configs, and
recorded ``module:Class`` entry-point strings keep resolving. Everything this
module used to define now lives in:

- ``common.constants`` (name tables), ``common.presets``, ``common.actions``,
  ``common.events``, ``common.rewards``, ``common.terminations``,
  ``common.observations`` (vanilla surface + obs-param helpers),
- ``common.tracking_env`` (the base class, recipes, Hydra plumbing),
- ``imitation_g1_env_v0`` (the ``Isaac-Imitation-G1-v0`` release) and
  ``variants.strict``.
"""

from .common.actions import G1ActionsCfg, G1SonicActionsCfg
from .common.constants import (
    G1_29DOF_DATASET_BODY_NAMES,
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_29DOF_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_OBS_ANCHOR_BODY_NAME,
    G1_TRACKED_BODY_NAMES,
    VELOCITY_RANGE,
)
from .common.events import G1EventCfg, G1SonicEventCfg
from .common.observations import (
    _LATENT_MODE_DEFAULT_COMMAND_TERM_NAMES,
    _PRUNABLE_COMMAND_TERM_NAMES,
    _VANILLA_ANCHOR_TERM_NAMES_BY_GROUP,
    G1ObservationCfg,
    _g1_canonical_joint_obs_params,
    _g1_expert_anchor_obs_params,
    _g1_expert_ee_obs_params,
    _g1_expert_keypoint_obs_params,
    _g1_expert_motion_obs_params,
    _g1_expert_window_anchor_obs_params,
    _g1_expert_window_ee_obs_params,
    _g1_expert_window_keypoint_obs_params,
    _g1_expert_window_motion_obs_params,
    _g1_tracked_body_asset_cfg,
    _g1_tracked_body_obs_params,
)
from .common.presets import (
    G1ImitationContactSensorCfg,
    G1ImitationPhysicsCfg,
    G1ImitationRobotCfg,
    G1SonicRobotCfg,
    _set_contact_sensor_update_period,
)
from .common.rewards import G1RewardsCfg, G1SonicRewardsCfg
from .common.terminations import (
    G1SonicTerminationCurriculumCfg,
    G1SonicTerminationsCfg,
    G1TerminationsCfg,
    _sonic_threshold_anneal_params,
)
from .common.tracking_env import (
    ImitationG1BaseTrackingEnvCfg,
    _apply_pelvis_protocol,
    _apply_strict_recipe,
    _bind_lafan_track_from_dict,
    _g1_lafan_track_env_cfg_from_dict,
)
from .imitation_g1_env_v0 import ImitationG1LafanTrackEnvCfg
from .variants.strict import ImitationG1StrictTrackEnvCfg

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
    "G1ObservationCfg",
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
    "ImitationG1LafanTrackEnvCfg",
    "ImitationG1StrictTrackEnvCfg",
    "_LATENT_MODE_DEFAULT_COMMAND_TERM_NAMES",
    "_PRUNABLE_COMMAND_TERM_NAMES",
    "_VANILLA_ANCHOR_TERM_NAMES_BY_GROUP",
    "_apply_pelvis_protocol",
    "_apply_strict_recipe",
    "_bind_lafan_track_from_dict",
    "_g1_canonical_joint_obs_params",
    "_g1_expert_anchor_obs_params",
    "_g1_expert_ee_obs_params",
    "_g1_expert_keypoint_obs_params",
    "_g1_expert_motion_obs_params",
    "_g1_expert_window_anchor_obs_params",
    "_g1_expert_window_ee_obs_params",
    "_g1_expert_window_keypoint_obs_params",
    "_g1_expert_window_motion_obs_params",
    "_g1_lafan_track_env_cfg_from_dict",
    "_g1_tracked_body_asset_cfg",
    "_g1_tracked_body_obs_params",
    "_set_contact_sensor_update_period",
    "_sonic_threshold_anneal_params",
]
