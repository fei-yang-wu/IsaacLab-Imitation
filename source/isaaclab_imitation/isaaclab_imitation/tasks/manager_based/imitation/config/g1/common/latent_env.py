# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-lineage base env config (also the deprecated Legacy pin).

``ImitationG1LatentEnvCfg`` is the parent of every latent variant and of the
Stable default (``v1.py``); on its own it is the pre-migration
beyondmimic-style surface frozen as ``Isaac-Imitation-G1-Latent-Legacy-v0``.
"""

from isaaclab.utils.configclass import configclass

from .observations_latent import (
    _LATENT_ANCHOR_TERM_NAMES_BY_GROUP,
    G1LatentObservationCfg,
)
from .tracking_env import (
    ImitationG1BaseTrackingEnvCfg,
    _bind_lafan_track_from_dict,
)


@configclass
class ImitationG1LatentEnvCfg(ImitationG1BaseTrackingEnvCfg):
    """Latent-conditioned G1 motion-tracking env driven by a LAFAN1 manifest."""

    observations = G1LatentObservationCfg()
    # Latent surfaces default to the latent command; switch to an explicit
    # tracker with `env.command_mode=explicit` plus a matching
    # `command_observation_terms` / `agent.command_components` selection.
    command_mode: str = "latent"
    # Default skill-command width: skill code z (256) + sin_cos phase (2) = 258
    # (wandb run dh8k313e recipe, minus z_phi). Override per run as needed.
    latent_command_dim: int = 258
    latent_goal_steps: int = 1

    def _critic_prunable_command_term_names(self) -> tuple[str, ...]:
        # The supplemental explicit terms the latent critic gained for
        # explicit command mode; the historical latent critic terms
        # (expert_motion + anchors) are part of the latent critic contract
        # and are never pruned.
        return (
            "expert_motion_qpos",
            "expert_ee_pos_b",
            "expert_ee_ori_b",
            "expert_keypoint_pos_b",
            "expert_keypoint_ori_b",
        )

    def __post_init__(self):
        super().__post_init__()
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 0
        self.random_reset_step_min = 0
        self.random_reset_step_max = 200
        self.random_reset_full_trajectory = False
        self._sync_expert_window_observation_params()
        self._sync_expert_goal_observation_params()
        # No reference-based terminations in latent mode
        # self.terminations.anchor_pos = None
        # self.terminations.anchor_ori = None
        # self.terminations.ee_body_pos = None

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

    def _anchor_term_names_by_group(self) -> dict[str, tuple[str, ...]]:
        return _LATENT_ANCHOR_TERM_NAMES_BY_GROUP


_bind_lafan_track_from_dict(ImitationG1LatentEnvCfg)
