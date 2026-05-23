import gymnasium as gym

from . import (
    agents,
    imitation_g1_env_cfg,
    imitation_g1_latent_env_cfg,
    imitation_g1_latent_vqvae_env_cfg,
)

__all__ = [
    "imitation_g1_env_cfg",
    "imitation_g1_latent_env_cfg",
    "imitation_g1_latent_vqvae_env_cfg",
    "agents",
]

_VANILLA_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.imitation_g1_env_cfg:ImitationG1LafanTrackEnvCfg",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_ppo_cfg_entry_point": f"{agents.__name__}.rlopt_ppo_cfg:G1ImitationRLOptPPOConfig",
    "rlopt_sac_cfg_entry_point": f"{agents.__name__}.rlopt_sac_cfg:G1ImitationRLOptSACConfig",
    "rlopt_fastsac_cfg_entry_point": f"{agents.__name__}.rlopt_fastsac_cfg:G1ImitationRLOptFastSACConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationRLOptIPMDConfig",
    "rlopt_ipmd_sr_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_sr_cfg:G1ImitationRLOptIPMDSRConfig",
    "rlopt_ipmd_bilinear_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:G1ImitationRLOptIPMDBilinearConfig",
    "rlopt_gail_cfg_entry_point": f"{agents.__name__}.rlopt_gail_cfg:G1ImitationRLOptGAILConfig",
    "rlopt_amp_cfg_entry_point": f"{agents.__name__}.rlopt_amp_cfg:G1ImitationRLOptAMPConfig",
}

_LATENT_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentEnvCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ase_cfg:G1ImitationRLOptASEConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentRLOptIPMDConfig",
    "rlopt_ipmd_sr_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_sr_cfg:G1ImitationLatentRLOptIPMDSRConfig",
    "rlopt_ipmd_bilinear_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:G1ImitationLatentRLOptIPMDBilinearConfig",
    "rlopt_ase_cfg_entry_point": f"{agents.__name__}.rlopt_ase_cfg:G1ImitationRLOptASEConfig",
}

_LATENT_GOAL_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentGoalEnvCfg"
    ),
    "rlopt_ipmd_bilinear_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:"
        "G1ImitationLatentGoalRLOptIPMDBilinearConfig"
    ),
}

_LATENT_VQVAE_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_vqvae_env_cfg:ImitationG1LatentVQVAEEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_vqvae_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_vqvae_cfg:G1ImitationLatentRLOptIPMDVQVAEConfig"
    ),
    "rlopt_ipmd_bilinear_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:"
        "G1ImitationLatentRLOptIPMDBilinearVQVAEConfig"
    ),
}

gym.register(
    id="Isaac-Imitation-G1-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_VANILLA_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-LafanTrack-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_VANILLA_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-Goal-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_GOAL_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-VQVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_VQVAE_TASK_KWARGS,
)
