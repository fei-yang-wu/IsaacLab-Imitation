from __future__ import annotations

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.surfaces import (
    ImitationG1SonicOfficialFSQSurfaceEnvCfg,
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
        ("policy", "expert_motion"),
        ("policy", "expert_anchor_pos_b"),
        ("policy", "expert_anchor_ori_b"),
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
    cfg = ImitationG1SonicOfficialFSQSurfaceEnvCfg()

    interface = cfg.command_interface
    # The command is one 64-value FSQ code the agent publishes, encoded from a
    # ten-frame reference window that advances every control step.
    assert interface.actor_kind() == "latent"
    assert interface.actor.dim == 64
    assert interface.encoder is not None
    assert interface.encoder.past_steps == 0
    assert interface.encoder.future_steps == 9
    selection = interface.reference.selection
    assert selection.random_step_min == 0
    assert selection.random_step_max == 200
    assert selection.full_trajectory is False
    assert selection.adaptive_failure_rate_max_over_mean == 50.0
    assert cfg.observations.policy.base_ang_vel.history_length == 10
