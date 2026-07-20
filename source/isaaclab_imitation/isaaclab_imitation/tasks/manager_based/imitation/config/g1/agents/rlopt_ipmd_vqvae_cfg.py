"""IPMD + VQ-VAE skill-codebook configuration for G1 imitation."""

from isaaclab.utils.configclass import configclass

from isaaclab_imitation.envs.rlopt import IPMDRLOptConfig


VANILLA_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_pos_b"),
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

LATENT_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "latent_command"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

# Posterior consumes the temporally-extended expert window already published by
# the env via the ``expert_window`` observation group; raising
# ``latent_patch_past_steps`` (set in __post_init__) widens the encoder context.
LATENT_POSTERIOR_INPUT_KEYS: list[tuple[str, str]] = [
    ("expert_window", "expert_motion"),
    ("expert_window", "expert_anchor_pos_b"),
    ("expert_window", "expert_anchor_ori_b"),
]

LATENT_PRIOR_INPUT_KEYS: list[tuple[str, str]] = []

LATENT_CRITIC_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "latent_command"),
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

REWARD_INPUT_KEYS: list[tuple[str, str]] = [
    ("reward_input", "expert_motion"),
    ("reward_input", "expert_anchor_pos_b"),
    ("reward_input", "expert_anchor_ori_b"),
]


@configclass
class _G1ImitationRLOptIPMDVQVAEBaseConfig(IPMDRLOptConfig):
    """Shared IPMD configuration with a VQ-VAE skill codebook for G1 imitation."""

    _default_use_latent_command: bool = True

    def sync_input_keys(self) -> None:
        use_latent_command = bool(self.ipmd.use_latent_command)
        self.policy.input_keys = (
            list(LATENT_POLICY_INPUT_KEYS)
            if use_latent_command
            else list(VANILLA_POLICY_INPUT_KEYS)
        )
        if self.value_function is not None:
            self.value_function.input_keys = (
                list(LATENT_CRITIC_INPUT_KEYS)
                if use_latent_command
                else list(VANILLA_CRITIC_INPUT_KEYS)
            )
        self.ipmd.reward_input_keys = list(REWARD_INPUT_KEYS)
        self.ipmd.latent_learning.posterior_input_keys = list(
            LATENT_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = list(LATENT_PRIOR_INPUT_KEYS)
        self.ipmd.latent_key = ("policy", "latent_command")
        self.ipmd.use_latent_command = use_latent_command

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, IPMDRLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.ipmd.use_latent_command = bool(self._default_use_latent_command)
        self.sync_input_keys()

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

        self.collector.total_frames = 5_000_000_000
        self.save_interval = 5_000_000  # samples

        # Skill chunking: patch_vqvae holds each posterior code for code_period
        # env steps, so the policy sees a skill token instead of per-step pose.
        self.ipmd.latent_dim = 64
        self.ipmd.latent_learning.code_latent_dim = 64
        self.ipmd.latent_learning.command_phase_mode = "none"
        self.ipmd.latent_steps_min = 30
        self.ipmd.latent_steps_max = 30

        self.ipmd.latent_learning.method = "patch_vqvae"
        # Defaults: FSQ (no codebook collapse, no aux tuning).
        self.ipmd.latent_learning.quantizer = "fsq"
        self.ipmd.latent_learning.fsq_levels = [8, 8, 8, 5, 5]
        # Used by ``vq_ema`` and ``gumbel`` quantizers.
        self.ipmd.latent_learning.codebook_size = 512
        self.ipmd.latent_learning.codebook_embed_dim = None
        self.ipmd.latent_learning.commitment_coeff = 0.25
        self.ipmd.latent_learning.ema_decay = 0.99
        self.ipmd.latent_learning.dead_code_reset_iters = 1000
        # Gumbel-Softmax fallback (lab familiarity, stochastic HL planner).
        self.ipmd.latent_learning.gumbel_tau_start = 1.0
        self.ipmd.latent_learning.gumbel_tau_end = 0.3
        self.ipmd.latent_learning.gumbel_tau_anneal_iters = 200_000
        self.ipmd.latent_learning.gumbel_hard = True
        self.ipmd.latent_learning.gumbel_kl_to_uniform_coeff = 0.01
        self.ipmd.latent_learning.code_usage_entropy_coeff = 0.01

        # Encoder eats a window of expert observations; bump past_steps to give
        # the codebook a temporally-extended view.  future_steps=0 keeps the
        # rollout-time encoder strictly causal.
        self.ipmd.latent_learning.patch_past_steps = 8
        self.ipmd.latent_learning.patch_future_steps = 0

        self.ipmd.latent_learning.encoder_hidden_dims = [512, 256]
        self.ipmd.latent_learning.encoder_activation = "elu"
        self.ipmd.latent_learning.decoder_hidden_dims = [256, 512]
        self.ipmd.latent_learning.decoder_activation = "elu"

        self.ipmd.latent_learning.lr = 3.0e-4
        self.ipmd.latent_learning.grad_clip_norm = 1.0
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.action_recon_coeff = 0.5
        self.ipmd.latent_learning.weight_decay_coeff = 0.0

        # Decouple PPO from encoder co-adaptation; let recon train the codebook.
        self.ipmd.latent_learning.train_posterior_through_policy = False
        self.ipmd.latent_learning.freeze_encoder = False

        # Mirrors latent_steps_min/max for readability. In posterior VQVAE mode,
        # the learner's collector hook owns the actual hold cadence.
        self.ipmd.latent_learning.code_period = 30
        self.ipmd.latent_learning.latent_dropout_to_random_code_prob = 0.0

        self.ipmd.latent_learning.probe_enabled = True
        self.ipmd.latent_learning.probe_condition_on_state = False
        self.ipmd.latent_learning.probe_target_keys = list(REWARD_INPUT_KEYS)
        self.ipmd.latent_learning.probe_hidden_dims = [256, 256]
        self.ipmd.latent_learning.probe_activation = "elu"
        self.ipmd.latent_learning.probe_lr = 3.0e-4
        self.ipmd.latent_learning.probe_grad_clip_norm = 1.0
        self.ipmd.latent_learning.probe_batch_size = 8192

        self.ipmd.diversity_bonus_coeff = 0.0
        self.ipmd.diversity_target = 0.0
        self.ipmd.latent_uniformity_temperature = 2.0

        self.ipmd.reward_input_type = "s"
        self.ipmd.use_estimated_rewards_for_ppo = False
        self.ipmd.expert_batch_size = int(self.loss.mini_batch_size)
        self.ipmd.bc_coef = 0.0
        self.compile.compile = False
        self.ipmd.reward_output_scale = 1.0
        self.ipmd.estimated_reward_clamp_min = -1.0
        self.ipmd.estimated_reward_clamp_max = 1.0
        self.ipmd.est_reward_weight = 1.0
        self.ipmd.env_reward_weight = 1.0
        self.ipmd.reward_loss_coeff = 1.0
        self.ipmd.reward_l2_coeff = 0.0
        self.ipmd.reward_grad_penalty_coeff = 0.0
        self.collector.no_cuda_sync = True

        if self._default_use_latent_command:
            self.ipmd.command_source = "posterior"


@configclass
class G1ImitationLatentRLOptIPMDVQVAEConfig(_G1ImitationRLOptIPMDVQVAEBaseConfig):
    """Latent-conditioned IPMD + VQ-VAE configuration for G1 imitation."""

    _default_use_latent_command: bool = True
