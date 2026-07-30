from __future__ import annotations

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_latent_env_cfg import (
    ImitationG1LatentSonicOfficialFSQEnvCfg,
)


def test_official_sonic_fsq_encodes_one_future_window_and_renews_each_step() -> None:
    cfg = G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig()
    latent = cfg.ipmd.latent_learning

    assert cfg.sonic_release_optimizer is True
    assert cfg.ipmd.command_source == "posterior"
    assert cfg.ipmd.latent_dim == 64
    assert cfg.ipmd.latent_steps_min == cfg.ipmd.latent_steps_max == 1
    assert latent.method == "patch_vqvae"
    assert latent.quantizer == "fsq"
    assert latent.fsq_levels == [32] * 64
    assert latent.fsq_normalize_codes is True
    assert latent.posterior_input_keys == [
        ("expert_window", "expert_motion"),
        ("expert_window", "expert_anchor_pos_b"),
        ("expert_window", "expert_anchor_ori_b"),
    ]
    assert latent.patch_past_steps == 0
    assert latent.patch_future_steps == 9
    assert latent.code_period == 1
    assert latent.posterior_command_period == 1
    assert latent.command_phase_mode == "none"
    assert latent.train_posterior_through_policy is True
    assert latent.recon_coeff == 0.01
    assert latent.encoder_hidden_dims == [2048, 1024, 512, 512]
    assert cfg.loss.epochs == 5
    assert cfg.loss.mini_batch_size == 4096 * 24 // 4
    assert cfg.policy.normalize_input is False
    assert cfg.value_function is not None
    assert cfg.value_function.normalize_input is True
    assert cfg.ipmd.actor_learning_rate == 2.0e-5
    assert cfg.ipmd.critic_learning_rate == 1.0e-3
    assert cfg.optim.max_grad_norm == 0.1


def test_official_sonic_fsq_environment_exposes_ten_advancing_frames() -> None:
    cfg = ImitationG1LatentSonicOfficialFSQEnvCfg()

    assert cfg.latent_command_dim == 64
    assert cfg.latent_patch_past_steps == 0
    assert cfg.latent_patch_future_steps == 9
    assert cfg.command_hold_steps == 0
    assert cfg.random_reset_step_min == 0
    assert cfg.random_reset_step_max == 200
    assert cfg.random_reset_full_trajectory is False
    assert cfg.adaptive_failure_reset_failure_rate_max_over_mean == 50.0
    assert cfg.observations.policy.base_ang_vel.history_length == 10
