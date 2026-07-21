from isaaclab.utils.configclass import configclass

from rlopt.agent import ASERLOptConfig


LATENT_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "latent_command"),
    ("policy", "projected_gravity"),
    ("policy", "body_pos"),
    ("policy", "body_ori"),
    ("policy", "base_lin_vel"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

LATENT_CRITIC_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "latent_command"),
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "projected_gravity"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "joint_pos"),
    ("critic", "joint_vel"),
    ("critic", "last_action"),
]

EXPERT_INPUT_KEYS: list[tuple[str, str]] = [
    ("expert_state", "joint_pos"),
    ("expert_state", "joint_vel"),
    ("expert_state", "root_pos"),
    ("expert_state", "root_quat"),
    ("expert_state", "root_lin_vel"),
    ("expert_state", "root_ang_vel"),
]


@configclass
class G1ImitationRLOptASEConfig(ASERLOptConfig):
    """RLOpt ASE configuration for G1 imitation."""

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, ASERLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.policy.input_keys = list(LATENT_POLICY_INPUT_KEYS)
        self.value_function.input_keys = list(LATENT_CRITIC_INPUT_KEYS)
        self.gail.discriminator_input_keys = list(EXPERT_INPUT_KEYS)

        self.collector.init_random_frames = 0
        self.collector.frames_per_batch = 24
        self.replay_buffer.size = 4096 * 24

        self.loss.epochs = 5
        self.loss.mini_batch_size = 4096 * 24 // 4
        self.loss.loss_critic_type = "l2"

        self.ppo.clip_epsilon = 0.2
        self.ppo.gae_lambda = 0.95
        self.ppo.entropy_coeff = 0.01
        self.ppo.critic_coeff = 1.0
        self.ppo.clip_value = True
        self.ppo.normalize_advantage = True
        self.ppo.clip_log_std = False
        self.ppo.log_std_init = 0.0

        self.optim.lr = 1.0e-4
        self.optim.max_grad_norm = 1.0
        self.optim.scheduler = "adaptive"
        self.optim.desired_kl = 0.01

        self.loss.gamma = 0.99

        self.policy.num_cells = [512, 256, 128]
        self.value_function.num_cells = [512, 256, 128]

        self.collector.total_frames = 30000 * 4096 * 24
        self.save_interval = 5_000_000   # samples

        self.gail.expert_batch_size = int(self.loss.mini_batch_size)
        self.gail.discriminator_updates_per_policy_update = 1
        self.gail.discriminator_batch_size = int(self.loss.mini_batch_size)

        self.gail.normalize_discriminator_input = True
        self.gail.discriminator_grad_penalty_coeff = 0.2
        self.gail.discriminator_logit_reg_coeff = 0.02
        self.gail.discriminator_weight_decay_coeff = 1.0e-5

        self.gail.discriminator_replay_size = 200000
        self.gail.discriminator_replay_ratio = 0.5
        self.gail.discriminator_replay_keep_prob = 0.25

        self.gail.use_gail_reward = True
        self.gail.normalize_discriminator_reward = False

        self.gail.amp_reward_clip = True
        self.gail.amp_reward_scale = 1.0

        self.ase.latent_dim = 64
        self.ase.latent_key = ("policy", "latent_command")
        self.ase.latent_steps_min = 30
        self.ase.latent_steps_max = 120
        self.ase.command_source = "random"
        self.ase.task_reward_w = 0.0
        self.ase.discriminator_reward_w = 0.5
        self.ase.mi_reward_w = 0.5
        self.ase.mi_enc_weight_decay = 1.0e-5
        self.ase.mi_enc_grad_penalty = 0.05
        self.ase.conditional_discriminator = False
        self.ase.mi_critic_hidden_dims = [256, 256]
        self.ase.mi_critic_activation = "elu"
        self.ase.mi_critic_lr = 3.0e-4
        self.ase.mi_critic_grad_clip_norm = 1.0
        self.ase.discriminator_critic_hidden_dims = [256, 256]
        self.ase.discriminator_critic_activation = "elu"
        self.ase.discriminator_critic_lr = 3.0e-4
        self.ase.discriminator_critic_grad_clip_norm = 1.0

        self.ase.diversity_bonus = 0.1
        self.ase.diversity_tar = 1.0
        self.ase.latent_uniformity_weight = 0.000
        self.ase.uniformity_kernel_scale = 2.0

        self.collector.no_cuda_sync = True
        self.trainer.log_interval = 10_000_000
        self.trainer.progress_bar = True
        self.log_level = "warning"
        self.compile.compile = False
