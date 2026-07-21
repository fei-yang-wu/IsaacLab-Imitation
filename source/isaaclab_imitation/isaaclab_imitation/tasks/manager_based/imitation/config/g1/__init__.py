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

# DEPRECATED (2026-07-19): the pre-migration beyondmimic-style latent surface
# (torso anchor, loose terminations, no proprio history, [0, 200] reset
# starts). Kept only for pre-migration checkpoints and frozen paper-protocol
# reproductions under `Isaac-Imitation-G1-Latent-Legacy-v0`. New work uses the
# SONIC surface, which is the `Isaac-Imitation-G1-Latent-v0` default below.
_LATENT_LEGACY_TASK_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentEnvCfg",
    "rlopt_cfg_entry_point": f"{agents.__name__}.rlopt_ase_cfg:G1ImitationRLOptASEConfig",
    "rlopt_ipmd_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentRLOptIPMDConfig",
    "rlopt_ipmd_sr_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_sr_cfg:G1ImitationLatentRLOptIPMDSRConfig",
    "rlopt_ipmd_bilinear_cfg_entry_point": f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:G1ImitationLatentRLOptIPMDBilinearConfig",
    "rlopt_ase_cfg_entry_point": f"{agents.__name__}.rlopt_ase_cfg:G1ImitationRLOptASEConfig",
}

# Default latent task surface: the SONIC release environment (pelvis anchor,
# strict adaptive terminations, adaptive failure sampling, SONIC actuators,
# rewards, and 10-step histories) with the SONIC release optimizer contract
# (actor lr 2e-5, joint grad clip 0.1, init std 0.05, 6-layer SiLU MLPs with
# running input normalization). Confirmed default (2026-07-20): single-GPU
# ICE H100 runs now target the release's own 100k-iteration convergence
# budget (~10B frames at 8192 envs x 12 steps), so the release contract is
# in scale rather than the flat regime seen at 50M-100M local scale. See
# wiki/isaaclab3-cu130-runtime-migration.md, "Training-gate resolution
# (2026-07-19)" and the 2026-07-20 follow-up.
_LATENT_SONIC_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentSonicEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicRLOptIPMDConfig"
    ),
    # Exact public-release optimizer contract; needs cluster-scale compute.
    "rlopt_ipmd_sonic_release_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentSonicReleaseRLOptIPMDConfig"
    ),
    "rlopt_ipmd_bilinear_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_bilinear_cfg:"
        "G1ImitationLatentRLOptIPMDBilinearConfig"
    ),
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

_LATENT_FUTURE_CVAE_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentFutureCVAEEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentFutureCVAERLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentFutureCVAERLOptIPMDConfig"
    ),
}

_LATENT_PER_STEP_VQ_TASK_KWARGS = {
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentPerStepVQEnvCfg"
    ),
    "rlopt_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
    ),
    "rlopt_ipmd_cfg_entry_point": (
        f"{agents.__name__}.rlopt_ipmd_cfg:G1ImitationLatentPerStepVQRLOptIPMDConfig"
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

# Default latent task: the SONIC surface.
gym.register(
    id="Isaac-Imitation-G1-Latent-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_SONIC_TASK_KWARGS,
)

# Back-compat alias for commands written while SONIC was opt-in; same kwargs
# as Isaac-Imitation-G1-Latent-v0.
gym.register(
    id="Isaac-Imitation-G1-Latent-Sonic-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_SONIC_TASK_KWARGS,
)

# DEPRECATED: pre-migration latent surface; see _LATENT_LEGACY_TASK_KWARGS.
gym.register(
    id="Isaac-Imitation-G1-Latent-Legacy-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_LEGACY_TASK_KWARGS,
)

# DEPRECATED (2026-07-20): pelvis-anchored legacy surface with annealed
# strict terminations. Was briefly floated as a candidate default while the
# full SONIC surface looked flat at local (50M-100M frame) scale; superseded
# once single-GPU ICE H100 runs adopted the release's own ~10B-frame /
# 100k-iteration budget, where the SONIC surface (`Isaac-Imitation-G1-Latent-v0`)
# is the confirmed default instead. Kept only for reproducing runs already
# started on this surface.
_LATENT_STRICT_TASK_KWARGS = {
    **_LATENT_LEGACY_TASK_KWARGS,
    "env_cfg_entry_point": (
        f"{__name__}.imitation_g1_latent_env_cfg:ImitationG1LatentStrictEnvCfg"
    ),
}

gym.register(
    id="Isaac-Imitation-G1-Latent-Strict-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_STRICT_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-Goal-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_GOAL_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-FutureCVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_FUTURE_CVAE_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-PerStepVQ-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_PER_STEP_VQ_TASK_KWARGS,
)

gym.register(
    id="Isaac-Imitation-G1-Latent-VQVAE-v0",
    entry_point="isaaclab_imitation.envs:ImitationRLEnv",
    disable_env_checker=True,
    kwargs=_LATENT_VQVAE_TASK_KWARGS,
)
