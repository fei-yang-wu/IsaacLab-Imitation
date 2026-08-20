"""Vega plus Wuji dexterous imitation task."""

import gymnasium as gym

from .vega_wuji_env_cfg import VegaWujiImitationEnvCfg

gym.register(
    id="Isaac-Imitation-Vega-Wuji-v0",
    entry_point="isaaclab_imitation.envs:VegaWujiImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vega_wuji_env_cfg:VegaWujiImitationEnvCfg",
        "rlopt_cfg_entry_point": (
            f"{__name__}.vega_wuji_agents:VegaWujiRLOptPPOConfig"
        ),
        "rlopt_ppo_cfg_entry_point": (
            f"{__name__}.vega_wuji_agents:VegaWujiRLOptPPOConfig"
        ),
    },
)

__all__ = ["VegaWujiImitationEnvCfg"]
