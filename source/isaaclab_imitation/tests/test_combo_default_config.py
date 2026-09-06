"""`Isaac-Imitation-G1-v3` resolves to the `combo` recipe, and v2 is untouched.

The recipe was carried as command-line overrides for every campaign since
2026-09-01; `-G1-v3` makes those the class defaults. Two things must hold:

* the v3 task's env and agent defaults equal the overrides the `combo-50b`
  chain passes (rewards, actor history, command width, reset ramp, curriculum,
  prefetch, optimizer, critic schedule, latent wiring);
* `-G1-v2` and the tuned / full-batch agent contracts keep their exact values,
  so every recorded run stays reproducible from its id.
"""

from __future__ import annotations

import gymnasium as gym
import importlib

import isaaclab_imitation.tasks  # noqa: F401  (registers the gym tasks)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    G1ImitationComboRLOptIPMDConfig,
    G1ImitationTunedFullBatchRLOptIPMDConfig,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_v2 import (
    ImitationG1V2EnvCfg,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_v3 import (
    COMBO_ACTOR_COMMAND_DIM,
    ImitationG1V3EnvCfg,
)

COMBO_CELLS = [2048, 2048, 1024, 1024, 512, 512]
HISTORY_TERMS = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


def _entry(task_id: str, key: str):
    module_name, class_name = gym.spec(task_id).kwargs[key].split(":")
    return getattr(importlib.import_module(module_name), class_name)


def test_v3_is_registered_on_the_combo_env_and_agent():
    assert _entry("Isaac-Imitation-G1-v3", "env_cfg_entry_point") is ImitationG1V3EnvCfg
    assert (
        _entry("Isaac-Imitation-G1-v3", "rlopt_ipmd_cfg_entry_point")
        is G1ImitationComboRLOptIPMDConfig
    )
    assert (
        _entry("Isaac-Imitation-G1-v3", "rlopt_ipmd_combo_cfg_entry_point")
        is G1ImitationComboRLOptIPMDConfig
    )
    # Still selectable on the frozen v2 task, never its default.
    assert (
        _entry("Isaac-Imitation-G1-v2", "rlopt_ipmd_combo_cfg_entry_point")
        is G1ImitationComboRLOptIPMDConfig
    )
    assert (
        _entry("Isaac-Imitation-G1-v2", "rlopt_ipmd_cfg_entry_point")
        is not G1ImitationComboRLOptIPMDConfig
    )
    assert _entry("Isaac-Imitation-G1-v2", "env_cfg_entry_point") is ImitationG1V2EnvCfg


def test_combo_agent_defaults_match_the_campaign_overrides():
    cfg = G1ImitationComboRLOptIPMDConfig()
    assert cfg.loss.epochs == 3
    assert cfg.loss.mini_batch_size == 1 << 30
    assert cfg.loss.gamma == 0.97
    assert cfg.collector.frames_per_batch == 24
    assert cfg.policy.num_cells == COMBO_CELLS
    assert cfg.value_function.num_cells == COMBO_CELLS
    assert cfg.policy.activation_fn == "silu"
    assert cfg.optim.weight_decay == 1.0e-2
    assert cfg.ipmd.critic_lr_schedule == "linear"
    assert cfg.ipmd.critic_lr_final == 1.0e-5
    assert cfg.ipmd.expert_batch_size == 24576
    assert cfg.ipmd.use_latent_command is True
    assert cfg.ipmd.command_source == "hl_skill"
    assert cfg.ipmd.latent_dim == COMBO_ACTOR_COMMAND_DIM
    assert cfg.ipmd.hl_skill_horizon_steps == 10
    assert cfg.ipmd.hl_skill_command_mode == "z"
    assert cfg.ipmd.hl_skill_finetune_enabled is False
    assert (cfg.ipmd.latent_steps_min, cfg.ipmd.latent_steps_max) == (1, 1)
    assert cfg.ipmd.latent_learning.code_period == 1
    assert cfg.ipmd.latent_learning.command_phase_mode == "sin_cos"
    assert cfg.ipmd.latent_learning.code_latent_dim == 64
    # No repo-default encoder: the path stays empty and is required per run.
    assert cfg.ipmd.hl_skill_checkpoint_path == ""


def test_fullbatch_contract_is_unchanged_by_the_combo_class():
    cfg = G1ImitationTunedFullBatchRLOptIPMDConfig()
    assert cfg.optim.weight_decay == 0.0
    assert cfg.ipmd.critic_lr_schedule == "constant"
    assert cfg.policy.num_cells == [1024, 1024, 512]
    assert cfg.ipmd.latent_dim == 258


def test_v3_env_defaults_match_the_campaign_overrides():
    cfg = ImitationG1V3EnvCfg()
    cfg.resolve()
    assert cfg.rewards.motion_ee_pos.weight == 1.0
    assert cfg.rewards.motion_global_anchor_pos_wide.weight == 1.0
    assert cfg.rewards.tracking_reward_points.weight == 4.0
    assert cfg.rewards.action_rate_l2.weight == -0.03
    assert cfg.rewards.feet_acc.weight == -2.5e-6
    for name in HISTORY_TERMS:
        assert getattr(cfg.observations.policy, name).history_length == 10, name
        assert getattr(cfg.observations.critic, name, None) is None or (
            getattr(cfg.observations.critic, name).history_length in (0, None)
        )
    assert cfg.command_interface.actor.dim == COMBO_ACTOR_COMMAND_DIM
    selection = cfg.command_interface.reference.selection
    assert selection.full_trajectory is True
    assert selection.adaptive_uniform_ratio == 0.8
    assert selection.adaptive_uniform_ratio_final == 0.2
    assert selection.adaptive_ratio_ramp_frames == 4_000_000_000
    assert selection.adaptive_failure_rate_max_over_mean == 200.0
    assert cfg.enable_termination_curriculum is True
    assert cfg.termination_curriculum_start_frames == 5_000_000
    assert cfg.termination_curriculum_end_frames == 30_000_000
    assert cfg.data.reference_prefetch_mode == "next"


def test_v3_newton_preset_raises_the_constraint_budget():
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.presets import (
        G1ImitationPhysicsCfg,
        G1V3PhysicsCfg,
    )

    assert G1V3PhysicsCfg().newton_mjwarp.solver_cfg.njmax == 320
    assert G1V3PhysicsCfg().newton_mjwarp.solver_cfg.nconmax == 200
    assert G1ImitationPhysicsCfg().newton_mjwarp.solver_cfg.njmax == 288


def test_v2_env_defaults_are_frozen():
    cfg = ImitationG1V2EnvCfg()
    cfg.resolve()
    assert cfg.rewards.motion_ee_pos.weight == 0.0
    assert cfg.rewards.action_rate_l2.weight == -0.1
    assert cfg.observations.policy.projected_gravity.history_length in (0, None)
    assert cfg.command_interface.actor.dim == 258
    assert cfg.enable_termination_curriculum is False
    assert cfg.data.reference_prefetch_mode == "off"
