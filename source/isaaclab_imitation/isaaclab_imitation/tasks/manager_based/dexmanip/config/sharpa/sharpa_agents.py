"""RLOpt PPO configuration for the Sharpa dual floating-hand task.

The released video-to-data recipe is an RSL-RL ``OnPolicyRunner`` config
(:class:`SharpaV2DPPORunnerCfg`). This module states the same recipe in RLOpt
fields so ``scripts/rlopt/train.py --algo PPO`` trains the task. The RSL-RL
config stays registered as the A/B baseline; the contract test asserts the
two agree field by field.
"""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass
from isaaclab_imitation.envs.rlopt import PPORLOptConfig


# Reference scale of the released recipe. RSL-RL states the rollout split and
# the checkpoint cadence as counts that hold at every environment count; RLOpt
# sizes both in frames. The declarative fields below let the launcher restate
# the counts at the live environment count (see scripts/rlopt/train_impl.py).
SHARPA_REFERENCE_NUM_ENVS = 4096
SHARPA_ROLLOUT_HORIZON = 24
SHARPA_MINI_BATCHES = 4
SHARPA_TRAIN_ITERATIONS = 20_000
SHARPA_SAVE_INTERVAL_ITERATIONS = 200
SHARPA_LOG_INTERVAL_ITERATIONS = 100
SHARPA_REFERENCE_BATCH = SHARPA_REFERENCE_NUM_ENVS * SHARPA_ROLLOUT_HORIZON

# The Sharpa curriculum converts PPO-update thresholds into control steps with
# this horizon (dexmanip/curriculum.py). It must stay 24 under every trainer.
SHARPA_CURRICULUM_NUM_STEPS_PER_ENV = 24


@configclass
class SharpaRLOptPPOConfig(PPORLOptConfig):
    """RLOpt PPO settings equal to the released Sharpa RSL-RL recipe."""

    reference_num_envs: int = SHARPA_REFERENCE_NUM_ENVS
    mini_batches_per_rollout: int = SHARPA_MINI_BATCHES
    save_interval_iterations: int = SHARPA_SAVE_INTERVAL_ITERATIONS
    log_interval_iterations: int = SHARPA_LOG_INTERVAL_ITERATIONS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.value_function is None:
            raise RuntimeError("Sharpa PPO requires a value function config.")
        if self.trainer is None:
            raise RuntimeError("Sharpa PPO requires a trainer config.")

        # The Sharpa observation group is one concatenated ``policy`` tensor
        # (SharpaObservationsCfg sets concatenate_terms=True), so both networks
        # read the default ``["policy"]`` key. The critic is symmetric, as in
        # the released recipe.
        self.policy.input_keys = None
        self.value_function.input_keys = None

        # Rollout: 24 steps per environment, no random warm-up frames.
        self.collector.init_random_frames = 0
        self.collector.frames_per_batch = SHARPA_ROLLOUT_HORIZON
        self.collector.no_cuda_sync = True
        self.collector.total_frames = SHARPA_TRAIN_ITERATIONS * SHARPA_REFERENCE_BATCH
        self.replay_buffer.size = SHARPA_REFERENCE_BATCH

        # Loss: 5 epochs, 4 minibatches, MSE value loss, clipped value.
        self.loss.epochs = 5
        self.loss.mini_batch_size = SHARPA_REFERENCE_BATCH // SHARPA_MINI_BATCHES
        self.loss.loss_critic_type = "l2"
        self.loss.gamma = 0.99

        self.ppo.gae_lambda = 0.95
        self.ppo.clip_epsilon = 0.1
        self.ppo.clip_value = True
        self.ppo.critic_coeff = 1.0
        self.ppo.entropy_coeff = 0.001
        # RSL-RL normalizes the advantage once over the whole rollout.
        self.ppo.normalize_advantage = True
        self.ppo.normalize_advantage_global = True
        # RSL-RL ``init_noise_std=0.1``; RLOpt parameterizes the log std.
        self.ppo.log_std_init = math.log(0.1)
        self.ppo.clip_log_std = True
        # RSL-RL adds gamma * V(s_t) at time-outs and treats them as terminal.
        self.ppo.truncation_bootstrap = "rsl_rl"

        # Optimizer: Adam, adaptive KL schedule with RSL-RL bounds and step.
        self.optim.optimizer = "adam"
        self.optim.lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"
        self.optim.desired_kl = 0.005
        self.optim.lr_adaptation_factor = 1.5
        self.optim.kl_adapt_step = "update"
        self.optim.min_lr = 1.0e-5
        self.optim.max_lr = 1.0e-2

        # Networks: ELU MLP [1024, 512, 256, 128] with input normalization.
        hidden_dims = [1024, 512, 256, 128]
        self.policy.num_cells = list(hidden_dims)
        self.policy.activation_fn = "elu"
        self.policy.normalize_input = True
        self.value_function.num_cells = list(hidden_dims)
        self.value_function.activation_fn = "elu"
        self.value_function.normalize_input = True

        # Cadence in frames at the reference scale; restated by the launcher.
        self.save_interval = SHARPA_SAVE_INTERVAL_ITERATIONS * SHARPA_REFERENCE_BATCH
        self.trainer.log_interval = (
            SHARPA_LOG_INTERVAL_ITERATIONS * SHARPA_REFERENCE_BATCH
        )
        self.trainer.progress_bar = True
        self.compile.compile = False

        # Local smokes run without credentials; campaigns opt into W&B.
        self.logger.backend = ""
        self.logger.project_name = "v2d_hands"


__all__ = [
    "SHARPA_CURRICULUM_NUM_STEPS_PER_ENV",
    "SHARPA_LOG_INTERVAL_ITERATIONS",
    "SHARPA_MINI_BATCHES",
    "SHARPA_REFERENCE_BATCH",
    "SHARPA_REFERENCE_NUM_ENVS",
    "SHARPA_ROLLOUT_HORIZON",
    "SHARPA_SAVE_INTERVAL_ITERATIONS",
    "SHARPA_TRAIN_ITERATIONS",
    "SharpaRLOptPPOConfig",
]
