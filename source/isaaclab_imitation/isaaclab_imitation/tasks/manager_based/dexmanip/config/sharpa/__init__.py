"""Sharpa floating-hand PhysX task registration."""

import gymnasium as gym

gym.register(
    id="Isaac-Sharpa-V2D-PhysX-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_env_cfg:SharpaV2DEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.sharpa_rsl_rl_cfg:SharpaV2DPPORunnerCfg",
        "rlopt_cfg_entry_point": f"{__name__}.sharpa_agents:SharpaRLOptPPOConfig",
        "rlopt_ppo_cfg_entry_point": f"{__name__}.sharpa_agents:SharpaRLOptPPOConfig",
    },
)

gym.register(
    id="Isaac-Sharpa-V2D-Source-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.sharpa_source_env_cfg:SharpaV2DSourceParityEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{__name__}.sharpa_rsl_rl_cfg:SharpaV2DPPORunnerCfg",
        "rlopt_cfg_entry_point": f"{__name__}.sharpa_agents:SharpaRLOptPPOConfig",
        "rlopt_ppo_cfg_entry_point": f"{__name__}.sharpa_agents:SharpaRLOptPPOConfig",
    },
)

__all__ = ["SharpaV2DEnvCfg"]
