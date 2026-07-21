from isaaclab.utils.configclass import configclass

from isaaclab_imitation.envs.rlopt import FastSACRLOptConfig


VANILLA_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_ori_b"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

VANILLA_CRITIC_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "expert_motion"),
    ("critic", "expert_anchor_pos_b"),
    ("critic", "expert_anchor_ori_b"),
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "last_action"),
]


@configclass
class G1ImitationRLOptFastSACConfig(FastSACRLOptConfig):
    """RLOpt FastSAC configuration for vanilla G1 imitation.

    Uses distributional C51 critic with LayerNorm+SiLU networks,
    observation normalization, and batched sampling for efficiency.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.q_function is not None, "Q-function configuration must be provided."

        # Observation keys (same as SAC)
        self.policy.input_keys = list(VANILLA_POLICY_INPUT_KEYS)
        self.q_function.input_keys = list(VANILLA_CRITIC_INPUT_KEYS)

        # Collector / schedule.
        # frames_per_batch=1 here → train.py scales it to 1*num_envs,
        # so the collector returns exactly one step per env per call,
        # matching SimpleReplayBuffer.extend()'s expected [n_env, ...] shape.
        self.collector.frames_per_batch = 1
        self.collector.total_frames = 100_000_000
        self.collector.init_random_frames = 0
        self.collector.no_cuda_sync = True

        # Replay buffer
        self.replay_buffer.size = 200_000  # total transitions in buffer

        # FastSAC algorithm hyperparameters
        self.fastsac.batch_size = 8  # per-env; total = 8*128 = 1024
        self.fastsac.num_updates = 8
        self.fastsac.policy_frequency = 4
        self.fastsac.actor_hidden_dim = 512
        self.fastsac.critic_hidden_dim = 768
        self.fastsac.num_q_networks = 2
        self.fastsac.num_atoms = 101
        self.fastsac.v_min = -20.0
        self.fastsac.v_max = 20.0
        self.fastsac.tau = 0.125
        self.fastsac.gamma = 0.97
        self.fastsac.learning_starts = 10
        self.fastsac.alpha_init = 0.001
        self.fastsac.use_autotune = True
        self.fastsac.target_entropy_ratio = 0.0
        self.fastsac.log_std_min = -5.0
        self.fastsac.log_std_max = 0.0
        self.fastsac.use_layer_norm = True
        self.fastsac.norm_obs = True
        self.fastsac.max_grad_norm = 0.0
        self.fastsac.num_steps = 1

        # Optimizer
        self.optim.optimizer = "adamw"
        self.optim.lr = 3.0e-4
        self.optim.weight_decay = 0.001
        self.optim.target_update_polyak = 0.995  # unused by FastSAC (uses fastsac.tau)

        # Logging / saving  (both intervals are in *samples*, not iterations)
        self.save_interval = 5_000_000    # samples: save checkpoint every 5 M steps
        self.trainer.progress_bar = True
        self.trainer.log_interval = 10_000_000  # samples: print console summary every 1 M steps
        self.compile.compile = False

        self.logger.project_name = "G1-Imitation-RLOpt-FastSAC"
