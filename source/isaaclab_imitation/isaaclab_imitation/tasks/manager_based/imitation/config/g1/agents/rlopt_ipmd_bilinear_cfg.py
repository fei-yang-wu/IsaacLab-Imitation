from isaaclab.utils import configclass

from isaaclab_imitation.envs.rlopt import IPMDBilinearRLOptConfig

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    LATENT_CRITIC_INPUT_KEYS,
    LATENT_POLICY_INPUT_KEYS,
    LATENT_POSTERIOR_INPUT_KEYS,
    LATENT_PRIOR_INPUT_KEYS,
    REWARD_INPUT_KEYS,
    VANILLA_CRITIC_INPUT_KEYS,
    VANILLA_POLICY_INPUT_KEYS,
)

UNITREE_G1_WBT_LEROBOT_REPO_IDS: list[str] = [
    "unitreerobotics/G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
    "unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
    "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
    "unitreerobotics/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
    "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow",
    "unitreerobotics/G1_WBT_Inspire_Put_Clothes_into_Washing_Machine",
    "unitreerobotics/G1_WBT_Brainco_Make_The_Bed",
    "unitreerobotics/G1_WBT_Inspire_Put_Clothes_Into_Basket",
    "unitreerobotics/G1_WBT_Dex1_Put_Clothes_into_Washing_Machine",
    "unitreerobotics/G1_WBT_Inspire_Put_Drinks_Into_Fridge",
    "unitreerobotics/G1_WBT_Inspire_Put_Vegetables_Into_Basket",
    "unitreerobotics/G1_WBT_Brainco_Pick_Up_Medicine",
    "unitreerobotics/G1_WBT_Inspire_Pick_Up_Drinks",
]

BILINEAR_OBS_KEYS: list[tuple[str, str]] = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

BILINEAR_NEXT_OBS_KEYS: list[tuple[str, str]] = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
]

VQVAE_LATENT_POSTERIOR_INPUT_KEYS: list[tuple[str, str]] = [
    ("expert_window", "expert_motion"),
    ("expert_window", "expert_anchor_pos_b"),
    ("expert_window", "expert_anchor_ori_b"),
]

GOAL_LATENT_POSTERIOR_INPUT_KEYS: list[tuple[str, str]] = [
    ("expert_goal", "expert_motion"),
    ("expert_goal", "expert_anchor_pos_b"),
    ("expert_goal", "expert_anchor_ori_b"),
]


@configclass
class _G1ImitationRLOptIPMDBilinearBaseConfig(IPMDBilinearRLOptConfig):
    """Shared RLOpt IPMD + Bilinear configuration for G1 imitation."""

    _default_use_latent_command: bool = False

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
        self.bilinear.obs_keys = list(BILINEAR_OBS_KEYS)
        self.bilinear.next_obs_keys = list(BILINEAR_NEXT_OBS_KEYS)

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, IPMDBilinearRLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.ipmd.use_latent_command = bool(self._default_use_latent_command)
        self.sync_input_keys()
        self.logger.project_name = "G1-Imitation-RLOpt-Pretrain"
        self.logger.group_name = ""

        # More initial exploration to improve policy-state coverage for inverse reward.
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
        self.save_interval = 100  # rollout iterations

        # Debug: latent posterior input mirrors the single-step vanilla tracker
        # policy reference payload directly: expert_motion (58) + anchor_ori (6).
        self.ipmd.latent_dim = 64
        self.ipmd.latent_steps_min = 1
        self.ipmd.latent_steps_max = 1
        self.ipmd.latent_learning.method = "patch_autoencoder"
        self.ipmd.latent_learning.encoder_hidden_dims = [256, 256]
        self.ipmd.latent_learning.encoder_activation = "elu"
        self.ipmd.latent_learning.prior_hidden_dims = [256, 256]
        self.ipmd.latent_learning.prior_activation = "elu"
        self.ipmd.latent_learning.patch_past_steps = 0
        self.ipmd.latent_learning.patch_future_steps = 0
        self.ipmd.latent_learning.lr = 3.0e-4
        self.ipmd.latent_learning.grad_clip_norm = 1.0
        self.ipmd.latent_learning.freeze_encoder = True
        self.ipmd.latent_learning.train_posterior_through_policy = True

        # Debug mode still trains the autoencoder on expert reference patches,
        # but the live latent_command path publishes the raw posterior features
        # directly so data flow can be checked independently of the encoder.
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.weight_decay_coeff = 0.0
        self.ipmd.latent_learning.kl_coeff = 0.0
        self.ipmd.latent_learning.probe_enabled = False
        self.ipmd.latent_learning.probe_condition_on_state = False
        self.ipmd.latent_learning.probe_target_keys = list(REWARD_INPUT_KEYS)
        self.ipmd.latent_learning.probe_hidden_dims = [256, 256]
        self.ipmd.latent_learning.probe_activation = "elu"
        self.ipmd.latent_learning.probe_lr = 3.0e-4
        self.ipmd.latent_learning.probe_grad_clip_norm = 1.0
        self.ipmd.latent_learning.probe_batch_size = 8192
        self.ipmd.env_reward_weight = 1.0

        # Keep the policy objective free of extra latent shaping.
        self.ipmd.diversity_bonus_coeff = 0.0
        self.ipmd.diversity_target = 0.0
        self.ipmd.latent_uniformity_temperature = 2.0

        self.ipmd.reward_input_type = "s"
        self.ipmd.use_estimated_rewards_for_ppo = False
        self.ipmd.expert_batch_size = int(self.loss.mini_batch_size)
        self.ipmd.bc_coef = 0.0
        self.compile.compile = False
        # self.trainer.progress_bar = False
        # self.trainer.log_interval = 10_000_000
        self.ipmd.reward_output_scale = 1.0
        self.ipmd.estimated_reward_clamp_min = -1.0
        self.ipmd.estimated_reward_clamp_max = 1.0
        self.ipmd.est_reward_weight = 1.0
        self.ipmd.reward_loss_coeff = 1.0
        self.ipmd.reward_l2_coeff = 0.0
        self.ipmd.reward_grad_penalty_coeff = 0.0
        self.collector.no_cuda_sync = True

        # Collector latents should consume the same observation-manager channel
        # stored in the rollout TensorDict. Expert data enters through
        # sample_expert_batch(...) during updates, not through live env getters.
        if self._default_use_latent_command:
            self.ipmd.command_source = "posterior"

        self.bilinear.feature_dim = self.ipmd.latent_dim
        self.bilinear.policy_include_raw_state = False
        self.bilinear.detach_features_for_policy = True
        self.bilinear.offline_pretrain.enabled = False
        self.bilinear.offline_pretrain.num_updates = 2000
        self.bilinear.offline_pretrain.batch_size = 8192
        self.bilinear.offline_pretrain.log_interval = 100
        self.bilinear.offline_pretrain.policy_bc_train_latent = False

        self.offline_dataset.source = "lerobot_stream"
        self.offline_dataset.repo_id = "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow"
        self.offline_dataset.repo_ids = list(UNITREE_G1_WBT_LEROBOT_REPO_IDS)
        self.offline_dataset.mapper = "unitree_g1_wbt_29dof"
        self.offline_dataset.cache_storage = "torchrl_memmap"
        self.offline_dataset.fps = 30.0
        self.offline_dataset.quat_order = "wxyz"


@configclass
class G1ImitationRLOptIPMDBilinearConfig(_G1ImitationRLOptIPMDBilinearBaseConfig):
    """Vanilla RLOpt IPMD + Bilinear configuration for G1 imitation."""

    _default_use_latent_command: bool = False


@configclass
class G1ImitationLatentRLOptIPMDBilinearConfig(_G1ImitationRLOptIPMDBilinearBaseConfig):
    """Latent-conditioned RLOpt IPMD + Bilinear configuration for G1 imitation."""

    _default_use_latent_command: bool = True


@configclass
class G1ImitationLatentRLOptIPMDBilinearVQVAEConfig(
    _G1ImitationRLOptIPMDBilinearBaseConfig
):
    """Latent VQ-VAE env with bilinear offline pretraining enabled."""

    _default_use_latent_command: bool = True

    def sync_input_keys(self) -> None:
        super().sync_input_keys()
        self.ipmd.latent_learning.posterior_input_keys = list(
            VQVAE_LATENT_POSTERIOR_INPUT_KEYS
        )

    def __post_init__(self):
        super().__post_init__()

        self.sync_input_keys()

        self.ipmd.latent_dim = 64
        self.ipmd.latent_steps_min = 10
        self.ipmd.latent_steps_max = 10
        self.ipmd.command_source = "posterior"

        self.ipmd.latent_learning.method = "patch_vqvae"
        self.ipmd.latent_learning.quantizer = "fsq"
        self.ipmd.latent_learning.fsq_levels = [8, 8, 8, 5, 5]
        self.ipmd.latent_learning.code_latent_dim = 64
        self.ipmd.latent_learning.codebook_size = 512
        self.ipmd.latent_learning.codebook_embed_dim = None
        self.ipmd.latent_learning.commitment_coeff = 0.25
        self.ipmd.latent_learning.ema_decay = 0.99
        self.ipmd.latent_learning.dead_code_reset_iters = 1000
        self.ipmd.latent_learning.gumbel_tau_start = 1.0
        self.ipmd.latent_learning.gumbel_tau_end = 0.3
        self.ipmd.latent_learning.gumbel_tau_anneal_iters = 200_000
        self.ipmd.latent_learning.gumbel_hard = True
        self.ipmd.latent_learning.gumbel_kl_to_uniform_coeff = 0.01
        self.ipmd.latent_learning.code_usage_entropy_coeff = 0.01
        self.ipmd.latent_learning.command_phase_mode = "none"
        self.ipmd.latent_learning.code_period = 10
        self.ipmd.latent_learning.latent_dropout_to_random_code_prob = 0.0

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
        self.ipmd.latent_learning.freeze_encoder = False
        self.ipmd.latent_learning.train_posterior_through_policy = True

        self.ipmd.latent_learning.probe_enabled = True
        self.ipmd.latent_learning.probe_condition_on_state = False
        self.ipmd.latent_learning.probe_target_keys = list(REWARD_INPUT_KEYS)
        self.ipmd.latent_learning.probe_hidden_dims = [256, 256]
        self.ipmd.latent_learning.probe_activation = "elu"
        self.ipmd.latent_learning.probe_lr = 3.0e-4
        self.ipmd.latent_learning.probe_grad_clip_norm = 1.0
        self.ipmd.latent_learning.probe_batch_size = 8192

        self.bilinear.feature_dim = self.ipmd.latent_dim
        self.bilinear.offline_pretrain.policy_bc_train_latent = True


@configclass
class G1ImitationLatentGoalRLOptIPMDBilinearConfig(
    _G1ImitationRLOptIPMDBilinearBaseConfig
):
    """Held future-goal latent command with a continuous AE embedding."""

    _default_use_latent_command: bool = True

    def sync_input_keys(self) -> None:
        super().sync_input_keys()
        self.ipmd.latent_learning.posterior_input_keys = list(
            GOAL_LATENT_POSTERIOR_INPUT_KEYS
        )

    def __post_init__(self):
        super().__post_init__()

        self.sync_input_keys()

        # The manager action is a 128D continuous goal embedding. The bilinear
        # low-level still uses a 64D spectral feature basis, so the policy head
        # learns a command projector from goal embedding -> spectral direction.
        self.ipmd.latent_dim = 128
        self.ipmd.latent_steps_min = 25
        self.ipmd.latent_steps_max = 25
        self.ipmd.command_source = "posterior"

        self.ipmd.latent_learning.method = "patch_autoencoder"
        self.ipmd.latent_learning.encoder_hidden_dims = [512, 256]
        self.ipmd.latent_learning.encoder_activation = "elu"
        self.ipmd.latent_learning.decoder_hidden_dims = [256, 512]
        self.ipmd.latent_learning.decoder_activation = "elu"
        self.ipmd.latent_learning.patch_past_steps = 0
        self.ipmd.latent_learning.patch_future_steps = 0
        self.ipmd.latent_learning.posterior_command_period = 25
        self.ipmd.latent_learning.lr = 3.0e-4
        self.ipmd.latent_learning.grad_clip_norm = 1.0
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.weight_decay_coeff = 0.0
        self.ipmd.latent_learning.freeze_encoder = False
        self.ipmd.latent_learning.train_posterior_through_policy = False

        self.bilinear.feature_dim = 64
        self.bilinear.offline_pretrain.policy_bc_train_latent = False
