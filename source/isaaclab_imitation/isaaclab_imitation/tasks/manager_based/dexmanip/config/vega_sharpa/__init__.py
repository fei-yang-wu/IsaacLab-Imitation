"""Mounted Vega U / Sharpa task registration."""

import gymnasium as gym

gym.register(
    id="Isaac-Imitation-Vega-Sharpa-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vega_sharpa_env_cfg:VegaSharpaEnvCfg",
        "rlopt_cfg_entry_point": f"{__name__}.vega_sharpa_agents:VegaSharpaRLOptPPOConfig",
        "rlopt_ppo_cfg_entry_point": f"{__name__}.vega_sharpa_agents:VegaSharpaRLOptPPOConfig",
    },
)
