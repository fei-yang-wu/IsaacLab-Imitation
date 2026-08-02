# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v1``: the frozen v1 G1 tracking environment.

Skill-conditioned tracker on the SONIC release recipe with this repo's
legacy reset distribution. The command space is configuration, not identity:
the class defaults to the latent 258-D skill command, and
``env.command_mode=explicit`` + ``command_observation_terms`` turn the same
surface into an explicit tracker.

Superseded as the default by ``-G1-v2`` on 2026-08-01 (the flagship
``ImitationG1EnvCfg`` name moved to ``imitation_g1_env_v2.py``); this module
keeps the exact class as ``ImitationG1EnvV1Cfg`` (with a back-compat
``ImitationG1EnvCfg`` alias) so the frozen ``-G1-v1`` task and old
serialized configs keep resolving.

This file is the complete v1 release definition: it inherits only the
generic machinery base (``ImitationG1BaseTrackingEnvCfg``) and states every
component pin and protocol choice below.
"""

from isaaclab.utils.configclass import configclass

from .common.actions import G1SonicActionsCfg
from .common.events import G1SonicEventCfg
from .common.observations import (
    _LATENT_ANCHOR_TERM_NAMES_BY_GROUP,
    G1LatentObservationCfg,
)
from .common.presets import G1SonicRobotCfg
from .common.rewards import G1SonicRewardsCfg
from .common.terminations import G1SonicTerminationsCfg
from .common.tracking_env import (
    ImitationG1BaseTrackingEnvCfg,
    _apply_pelvis_protocol,
    _bind_lafan_track_from_dict,
)


@configclass
class ImitationG1EnvV1Cfg(ImitationG1BaseTrackingEnvCfg):
    """Skill-conditioned 29-DoF G1 tracking env (`Isaac-Imitation-G1-v1`).

    Validated 2026-07-27: the full SONIC release recipe, taking back only the
    legacy reset distribution (full-trajectory adaptive-failure resets
    collapsed episode length 5.6x at 4096 envs in the reset-sampling screen).
    Terminations are the strict release values from frame 0; the anneal
    curriculum is off because legacy resets already prevent the ~5-step early
    episodes it exists to avoid, and a moving threshold makes early curves
    incomparable across runs.
    """

    # -- components (shared blocks from common) --
    actions = G1SonicActionsCfg()
    observations = G1LatentObservationCfg()
    rewards = G1SonicRewardsCfg()  # type: ignore
    terminations = G1SonicTerminationsCfg()  # type: ignore
    events = G1SonicEventCfg()
    curriculum = None

    # -- skill-conditioned command configuration --
    # The actor consumes the agent-published latent skill command by default;
    # `command_mode=explicit` prunes it and re-enables the explicit terms.
    command_mode: str = "latent"
    # Skill-command width: skill code z (256) + sin_cos phase (2) = 258
    # (wandb run dh8k313e recipe, minus z_phi). Override per run as needed.
    latent_command_dim: int = 258
    # The expert_goal group exposes this many steps of future goal state for
    # hierarchical skills; agents that do not read it can drop the group with
    # `enable_expert_goal_observations=false`.
    latent_goal_steps: int = 1

    def _anchor_term_names_by_group(self) -> dict[str, tuple[str, ...]]:
        # The latent observation surface carries robot body-pose terms and an
        # expert_goal group, so its anchor re-pointing table differs from the
        # vanilla one the base class uses.
        return _LATENT_ANCHOR_TERM_NAMES_BY_GROUP

    def _critic_prunable_command_term_names(self) -> tuple[str, ...]:
        # The supplemental explicit terms the latent critic gained for
        # explicit command mode; the historical latent critic terms
        # (expert_motion + anchors) are part of the critic contract and are
        # never pruned.
        return (
            "expert_motion_qpos",
            "expert_ee_pos_b",
            "expert_ee_ori_b",
            "expert_keypoint_pos_b",
            "expert_keypoint_ori_b",
        )

    def _sync_expert_goal_observation_params(self) -> None:
        goal_steps = int(self.latent_goal_steps)
        if goal_steps < 0:
            raise ValueError("latent_goal_steps must be >= 0.")
        # The group is None when `enable_expert_goal_observations=False`
        # dropped it; nothing to sync then.
        if getattr(self.observations, "expert_goal", None) is None:
            return
        for term in (
            self.observations.expert_goal.expert_motion,
            self.observations.expert_goal.expert_anchor_pos_b,
            self.observations.expert_goal.expert_anchor_ori_b,
        ):
            term.params["goal_steps"] = goal_steps

    def __post_init__(self):
        super().__post_init__()

        # Single-frame skill command over a live sliding reference window.
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 0
        self._sync_expert_window_observation_params()
        self._sync_expert_goal_observation_params()

        # SONIC robot asset (actuator contract matching G1SonicActionsCfg).
        robot_preset = G1SonicRobotCfg()
        for variant in (
            robot_preset.default,
            robot_preset.physx,
            robot_preset.newton_mjwarp,
        ):
            variant.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot_preset  # type: ignore

        # Pelvis expert anchor plus the legacy reset distribution: starts in
        # [0, 200], no full-trajectory adaptive-failure sampling,
        # failure_rate_max_over_mean=50.
        _apply_pelvis_protocol(self, failure_rate_max_over_mean=50.0)


_bind_lafan_track_from_dict(ImitationG1EnvV1Cfg)

# Back-compat alias: old imports and serialized configs that resolve
# `imitation_g1_env_v1:ImitationG1EnvCfg` (including the frozen
# `_LATENT_STABLE_TASK_KWARGS` entry-point string) keep resolving to the v1
# class. The flagship `ImitationG1EnvCfg` name now belongs to the v2 module.
ImitationG1EnvCfg = ImitationG1EnvV1Cfg

__all__ = ["ImitationG1EnvCfg", "ImitationG1EnvV1Cfg"]
