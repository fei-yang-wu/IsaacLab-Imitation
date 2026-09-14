# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Released video-to-data RSL-RL PPO settings."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SharpaMLPModelCfg:
    """RSL-RL 5 model fields without its incompatible legacy fields."""

    class_name: str = "MLPModel"
    hidden_dims: list[int] = None  # type: ignore[assignment]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: object | None = None


@configclass
class SharpaV2DPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "sharpa_v2d"
    empirical_normalization = True
    actor = SharpaMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.1,
            std_type="scalar",
        ),
    )
    critic = SharpaMLPModelCfg(
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )
    # Keep the deprecated policy block so older Isaac Lab/RSL-RL releases can
    # still read the same source hyperparameters.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
    logger = "wandb"
    wandb_project = "v2d_hands"


__all__ = ["SharpaV2DPPORunnerCfg"]
