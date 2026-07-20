from isaaclab.utils.configclass import configclass

from isaaclab_imitation.envs.rlopt import PPORLOptConfig


VANILLA_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_pos_b"),
    ("policy", "expert_anchor_ori_b"),
    ("policy", "base_lin_vel"),
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
class G1ImitationRLOptPPOConfig(PPORLOptConfig):
    """RLOpt PPO configuration for G1 imitation."""

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, PPORLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.policy.input_keys = list(VANILLA_POLICY_INPUT_KEYS)
        self.value_function.input_keys = list(VANILLA_CRITIC_INPUT_KEYS)

        self.collector.init_random_frames = 0
        self.collector.frames_per_batch = 24
        self.replay_buffer.size = 4096 * 24

        self.loss.epochs = 5
        self.loss.mini_batch_size = 4096 * 24 // 4
        self.loss.loss_critic_type = "l2"

        self.ppo.clip_epsilon = 0.2
        self.ppo.gae_lambda = 0.95
        self.ppo.entropy_coeff = 0.005
        self.ppo.critic_coeff = 1.0
        self.ppo.clip_value = True
        self.ppo.normalize_advantage = True
        self.ppo.clip_log_std = False
        self.ppo.log_std_init = 0.0

        self.optim.lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"
        self.optim.desired_kl = 0.01

        self.loss.gamma = 0.99

        self.policy.num_cells = [512, 256, 128]
        self.value_function.num_cells = [512, 256, 128]

        self.collector.total_frames = 30000 * 4096 * 24
        self.save_interval = 5_000_000  # samples
        self.compile.compile = False
        self.collector.no_cuda_sync = True
        self.trainer.progress_bar = True
        self.trainer.log_interval = 10_000_000  # samples
