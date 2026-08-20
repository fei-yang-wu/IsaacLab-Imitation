"""RLOpt PPO configuration for the Vega-Wuji dexterous task."""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass
from isaaclab_imitation.envs.rlopt import PPORLOptConfig


# Reference scale of the released recipe. The upstream RSL-RL recipe states its
# rollout split and checkpoint cadence as counts, not as absolute frames:
# num_mini_batches and save_interval hold at every environment count. RLOpt
# sizes both in frames, so the translation is only exact at the environment
# count it was written against. These constants record that reference scale,
# and the declarative fields below let the launcher restate the counts at the
# environment count actually used.
VEGA_WUJI_REFERENCE_NUM_ENVS = 4096
VEGA_WUJI_ROLLOUT_HORIZON = 24
VEGA_WUJI_MINI_BATCHES = 4
VEGA_WUJI_TRAIN_ITERATIONS = 20_000
VEGA_WUJI_SAVE_INTERVAL_ITERATIONS = 200
VEGA_WUJI_LOG_INTERVAL_ITERATIONS = 100
VEGA_WUJI_REFERENCE_BATCH = VEGA_WUJI_REFERENCE_NUM_ENVS * VEGA_WUJI_ROLLOUT_HORIZON


VEGA_WUJI_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "wrist_position"),
    ("policy", "wrist_orientation"),
    ("policy", "wrist_velocity"),
    ("policy", "finger_joint_pos"),
    ("policy", "finger_joint_vel"),
    ("policy", "object_position"),
    ("policy", "object_orientation"),
    ("policy", "command"),
    ("policy", "last_action"),
    ("policy", "processed_right_actions"),
    ("policy", "processed_left_actions"),
    ("policy", "contact_position_direction"),
    ("policy", "current_robot_state"),
    ("policy", "next_desired_robot_state"),
    ("policy", "next_desired_object_state"),
    ("policy", "current_object_state"),
]


@configclass
class VegaWujiRLOptPPOConfig(PPORLOptConfig):
    """RLOpt PPO settings for the Vega/Wuji observation surface."""

    # Portable form of the released recipe. The launcher restates these counts
    # at the live environment count; the frame-sized fields below stay the
    # exact reference-scale values, so a run at VEGA_WUJI_REFERENCE_NUM_ENVS is
    # unchanged. An explicit override of a frame-sized field is not re-derived.
    reference_num_envs: int = VEGA_WUJI_REFERENCE_NUM_ENVS
    mini_batches_per_rollout: int = VEGA_WUJI_MINI_BATCHES
    save_interval_iterations: int = VEGA_WUJI_SAVE_INTERVAL_ITERATIONS
    log_interval_iterations: int = VEGA_WUJI_LOG_INTERVAL_ITERATIONS

    def __post_init__(self) -> None:
        super().__post_init__()

        self.policy.input_keys = list(VEGA_WUJI_POLICY_INPUT_KEYS)
        if self.value_function is None:
            raise RuntimeError("Vega-Wuji PPO requires a value function config.")
        self.value_function.input_keys = list(VEGA_WUJI_POLICY_INPUT_KEYS)

        self.collector.init_random_frames = 0
        self.collector.frames_per_batch = VEGA_WUJI_ROLLOUT_HORIZON
        self.replay_buffer.size = VEGA_WUJI_REFERENCE_BATCH
        # Keep the internal pre-release recipe runnable without external
        # credentials. Campaigns may opt into a remote backend explicitly.
        self.logger.backend = ""

        self.loss.epochs = 5
        self.loss.mini_batch_size = VEGA_WUJI_REFERENCE_BATCH // VEGA_WUJI_MINI_BATCHES
        self.loss.loss_critic_type = "l2"

        # Translate the released CHORD RSL-RL PPO recipe to RLOpt fields.
        self.ppo.clip_epsilon = 0.1
        self.ppo.gae_lambda = 0.95
        self.ppo.entropy_coeff = 0.001
        self.ppo.critic_coeff = 1.0
        self.ppo.clip_value = True
        self.ppo.normalize_advantage = True
        self.ppo.clip_log_std = True
        self.ppo.log_std_init = math.log(0.1)

        self.optim.optimizer = "adam"
        self.optim.lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"
        self.optim.desired_kl = 0.005

        self.loss.gamma = 0.99

        hidden_dims = [1024, 512, 256, 128]
        self.policy.num_cells = list(hidden_dims)
        self.policy.activation_fn = "elu"
        self.policy.normalize_input = True
        self.value_function.num_cells = list(hidden_dims)
        self.value_function.activation_fn = "elu"
        self.value_function.normalize_input = True

        self.collector.total_frames = (
            VEGA_WUJI_TRAIN_ITERATIONS * VEGA_WUJI_REFERENCE_BATCH
        )
        self.save_interval = (
            VEGA_WUJI_SAVE_INTERVAL_ITERATIONS * VEGA_WUJI_REFERENCE_BATCH
        )
        self.compile.compile = False
        self.collector.no_cuda_sync = True
        if self.trainer is None:
            raise RuntimeError("Vega-Wuji PPO requires a trainer config.")
        self.trainer.progress_bar = True
        self.trainer.log_interval = (
            VEGA_WUJI_LOG_INTERVAL_ITERATIONS * VEGA_WUJI_REFERENCE_BATCH
        )


__all__ = [
    "VEGA_WUJI_MINI_BATCHES",
    "VEGA_WUJI_POLICY_INPUT_KEYS",
    "VEGA_WUJI_REFERENCE_BATCH",
    "VEGA_WUJI_REFERENCE_NUM_ENVS",
    "VEGA_WUJI_ROLLOUT_HORIZON",
    "VEGA_WUJI_SAVE_INTERVAL_ITERATIONS",
    "VEGA_WUJI_TRAIN_ITERATIONS",
    "VegaWujiRLOptPPOConfig",
]
