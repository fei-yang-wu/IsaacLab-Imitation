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

PROPRIO_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

PRIVILEGED_CRITIC_STATE_KEYS: list[tuple[str, str]] = [
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "base_lin_vel"),
    ("critic", "base_ang_vel"),
    ("critic", "joint_pos_rel"),
    ("critic", "joint_vel_rel"),
    ("critic", "last_action"),
]

FULL_BODY_TRAJECTORY_COMMAND_KEYS: list[tuple[str, str]] = [
    ("expert_window", "expert_motion"),
    ("expert_window", "expert_anchor_pos_b"),
    ("expert_window", "expert_anchor_ori_b"),
]

EE_TRAJECTORY_COMMAND_KEYS: list[tuple[str, str]] = [
    ("expert_window", "expert_ee_pos_b"),
    ("expert_window", "expert_ee_ori_b"),
]

COMMAND_SPACE_ALIASES: dict[str, str] = {
    "single_frame_full_body": "single_frame_full_body",
    "single_frame": "single_frame_full_body",
    "vanilla": "single_frame_full_body",
    "full_state": "single_frame_full_body",
    "full_body": "single_frame_full_body",
    "full_body_trajectory": "full_body_trajectory",
    "full_state_trajectory": "full_body_trajectory",
    "whole_body_trajectory": "full_body_trajectory",
    "full_traj": "full_body_trajectory",
    "ee_trajectory": "ee_trajectory",
    "end_effector_trajectory": "ee_trajectory",
    "end_effector": "ee_trajectory",
    "ee_pose_trajectory": "ee_trajectory",
}

LATENT_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "latent_command"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

SONIC_LATENT_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "latent_command"),
    ("policy", "projected_gravity"),
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

LATENT_POSTERIOR_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_pos_b"),
    ("policy", "expert_anchor_ori_b"),
]

LATENT_PRIOR_INPUT_KEYS: list[tuple[str, str]] = []

FUTURE_CVAE_POSTERIOR_INPUT_KEYS: list[tuple[str, str]] = [
    ("expert_window", "expert_motion"),
    ("expert_window", "expert_anchor_pos_b"),
    ("expert_window", "expert_anchor_ori_b"),
]

FUTURE_CVAE_PRIOR_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_pos_b"),
    ("policy", "expert_anchor_ori_b"),
]

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

SONIC_LATENT_CRITIC_INPUT_KEYS: list[tuple[str, str]] = [
    ("critic", "latent_command"),
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


def normalize_command_space(command_space: str) -> str:
    normalized = str(command_space).strip().lower().replace("-", "_")
    try:
        return COMMAND_SPACE_ALIASES[normalized]
    except KeyError as err:
        raise ValueError(
            f"Unsupported command_space={command_space!r}. "
            f"Expected one of {sorted(set(COMMAND_SPACE_ALIASES.values()))}."
        ) from err


def command_space_policy_input_keys(command_space: str) -> list[tuple[str, str]]:
    command_space = normalize_command_space(command_space)
    if command_space == "single_frame_full_body":
        return list(VANILLA_POLICY_INPUT_KEYS)
    if command_space == "full_body_trajectory":
        return list(FULL_BODY_TRAJECTORY_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    if command_space == "ee_trajectory":
        return list(EE_TRAJECTORY_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    raise AssertionError(f"Unhandled command space: {command_space}")


def command_space_critic_input_keys(command_space: str) -> list[tuple[str, str]]:
    command_space = normalize_command_space(command_space)
    if command_space == "single_frame_full_body":
        return list(VANILLA_CRITIC_INPUT_KEYS)
    if command_space == "full_body_trajectory":
        return list(FULL_BODY_TRAJECTORY_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    if command_space == "ee_trajectory":
        return list(EE_TRAJECTORY_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    raise AssertionError(f"Unhandled command space: {command_space}")


@configclass
class _G1ImitationRLOptIPMDBaseConfig(IPMDRLOptConfig):
    """Shared RLOpt IPMD configuration for G1 imitation."""

    _default_use_latent_command: bool = False
    command_space: str = "single_frame_full_body"

    def sync_input_keys(self) -> None:
        use_latent_command = bool(self.ipmd.use_latent_command)
        self.command_space = normalize_command_space(self.command_space)
        self.policy.input_keys = (
            list(LATENT_POLICY_INPUT_KEYS)
            if use_latent_command
            else command_space_policy_input_keys(self.command_space)
        )
        if self.value_function is not None:
            self.value_function.input_keys = (
                list(LATENT_CRITIC_INPUT_KEYS)
                if use_latent_command
                else command_space_critic_input_keys(self.command_space)
            )
        self.ipmd.reward_input_keys = list(REWARD_INPUT_KEYS)
        self.ipmd.latent_learning.posterior_input_keys = list(
            LATENT_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = list(LATENT_PRIOR_INPUT_KEYS)
        self.ipmd.latent_key = ("policy", "latent_command")
        self.ipmd.use_latent_command = use_latent_command
        # If running input normalization is enabled on either network, the
        # pretrained latent command (skill code z + sin/cos phase) must pass
        # through untouched: its scale and geometry are part of the
        # encoder/policy contract.
        self.policy.normalize_input_exclude_keys = (
            [("policy", "latent_command")] if use_latent_command else []
        )
        if self.value_function is not None:
            self.value_function.normalize_input_exclude_keys = (
                [("critic", "latent_command")] if use_latent_command else []
            )

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self, IPMDRLOptConfig)
        assert self.value_function is not None, (
            "Value function configuration must be provided."
        )

        self.ipmd.use_latent_command = bool(self._default_use_latent_command)
        self.sync_input_keys()
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
        if self.ipmd.use_latent_command:
            self.value_function.num_cells = [768, 512, 256]

        self.collector.total_frames = 5_000_000_000
        # RLOpt interprets this field in collected samples, not rollout
        # iterations. At the default 4096 envs x 24 steps this is 100 rollouts
        # (about 9.83M frames) instead of writing a checkpoint every rollout.
        self.save_interval = 4096 * 24 * 100

        # Base ("posterior") latent width: the single-step reference payload
        # mirrored directly -- expert_motion (58) + anchor_ori (6) = 64. This is
        # the default for vanilla IPMD and for the posterior latent mode; the
        # latent-conditioned config overrides it below (hl_skill, 258).
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

        # Posterior mode trains the autoencoder on expert reference patches and
        # publishes the raw posterior features on the live latent_command path
        # (encoder-independent data flow). The default latent scheme below uses a
        # pretrained hl_skill encoder instead.
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

        # Default latent-conditioned scheme: consume a pretrained high-level
        # diffsr skill encoder as the latent command (256-d skill code z + 2-d
        # sin/cos phase = 258). This is the current production latent scheme.
        # `command_source="posterior"` (raw posterior features, latent_dim=64) is
        # still a valid mode -- select it via overrides if desired.
        #
        # NOTE: `hl_skill_checkpoint_path` MUST be provided per run (path to a
        # pretrained skill-encoder best.pt from train_hl_skill_diffsr.py); there
        # is no repo-default checkpoint.
        if self._default_use_latent_command:
            self.ipmd.latent_dim = 258
            self.ipmd.command_source = "hl_skill"
            self.ipmd.hl_skill_command_mode = "z"
            self.ipmd.hl_skill_horizon_steps = 25
            self.ipmd.hl_skill_finetune_enabled = False
            self.ipmd.hl_skill_pg_coeff = 0.05
            self.ipmd.hl_skill_anchor_coeff = 0.01
            self.ipmd.hl_skill_offline_diffsr_coeff = 1.0
            self.ipmd.hl_skill_lr = 3.0e-5
            self.ipmd.latent_steps_min = 25
            self.ipmd.latent_steps_max = 25
            self.ipmd.latent_learning.command_phase_mode = "sin_cos"
            self.ipmd.latent_learning.code_period = 25
            self.ipmd.latent_learning.code_latent_dim = 256
            # hl_skill drives the objective; disable the learned-reward terms.
            self.ipmd.reward_loss_coeff = 0.0
            self.ipmd.reward_l2_coeff = 0.0
            self.ipmd.reward_grad_penalty_coeff = 0.0
            self.ipmd.reward_logit_reg_coeff = 0.0
            self.ipmd.reward_param_weight_decay_coeff = 0.0


@configclass
class G1ImitationRLOptIPMDConfig(_G1ImitationRLOptIPMDBaseConfig):
    """Vanilla RLOpt IPMD configuration for G1 imitation."""

    _default_use_latent_command: bool = False


@configclass
class G1ImitationLatentRLOptIPMDConfig(_G1ImitationRLOptIPMDBaseConfig):
    """Latent-conditioned RLOpt IPMD configuration for G1 imitation.

    DEPRECATED as a task default (2026-07-19): this is the pre-migration
    surface, now reachable only via ``Isaac-Imitation-G1-Latent-Legacy-v0``.
    It remains the shared parent of the SONIC config and the latent variants.
    """

    _default_use_latent_command: bool = True


@configclass
class G1ImitationLatentSonicRLOptIPMDConfig(G1ImitationLatentRLOptIPMDConfig):
    """Latent IPMD policy with SONIC's proprioception contract.

    The environment side (pelvis anchor, rewards, strict adaptive
    terminations, adaptive failure sampling, domain randomization, actuators)
    always follows the public SONIC release. The optimizer contract is split:

    - Default (``sonic_release_optimizer=False``): the locally-validated
      RLOpt contract (512/256/128 ELU MLPs, actor lr 1e-3). Reverted back to
      the default on 2026-07-21: briefly flipped to the release contract on
      2026-07-20 on the theory that single-GPU ICE H100's ~10B-frame /
      100k-iteration budget would be in-scale for it, but W&B run bn931wny
      (Latent-Strict-v0 + this local contract, same 8192x12x12288 scale)
      reached episode/length=244 / episode/return=13.1 -- far above anything
      the release contract produced at matched scale in the concurrent VRAM
      ablation. See the CU130 migration wiki page, "Training-gate resolution
      (2026-07-19)" and the 2026-07-21 reversal.
    - ``sonic_release_optimizer=True``: the exact public-release contract
      (actor lr 2e-5 adaptive in [1e-5, 2e-4], joint grad clip 0.1, init std
      0.05 clamped to [0.001, 0.5], global per-rollout advantage
      normalization, 6-layer SiLU MLPs with running input normalization),
      unconfirmed at any tested scale so far.
    """

    sonic_release_optimizer: bool = False

    def sync_input_keys(self) -> None:
        super().sync_input_keys()
        self.policy.input_keys = list(SONIC_LATENT_POLICY_INPUT_KEYS)
        if self.value_function is not None:
            self.value_function.input_keys = list(SONIC_LATENT_CRITIC_INPUT_KEYS)

    def _apply_release_optimizer_contract(self) -> None:
        assert self.value_function is not None
        self.policy.num_cells = [2048, 2048, 1024, 1024, 512, 512]
        self.policy.activation_fn = "silu"
        self.policy.normalize_input = True
        self.value_function.num_cells = [2048, 2048, 1024, 1024, 512, 512]
        self.value_function.activation_fn = "silu"
        self.value_function.normalize_input = True
        self.ipmd.actor_learning_rate = 2.0e-5
        self.ipmd.critic_learning_rate = 1.0e-3
        self.optim.min_lr = 1.0e-5
        self.optim.max_lr = 2.0e-4
        self.optim.max_grad_norm = 0.1
        self.ppo.entropy_coeff = 0.01
        self.ppo.normalize_advantage_global = True
        self.ppo.clip_log_std = True
        self.ppo.log_std_init = -2.995732273553991  # log(0.05)
        self.ppo.log_std_min = -6.907755278982137  # log(0.001)
        self.ppo.log_std_max = -0.6931471805599453  # log(0.5)

    def _apply_local_optimizer_contract(self) -> None:
        assert self.value_function is not None
        self.policy.num_cells = [512, 256, 128]
        self.policy.activation_fn = "elu"
        self.policy.normalize_input = False
        self.value_function.num_cells = [768, 512, 256]
        self.value_function.activation_fn = "elu"
        self.value_function.normalize_input = False
        self.ipmd.actor_learning_rate = 1.0e-3
        self.ipmd.critic_learning_rate = 1.0e-3
        self.optim.min_lr = 1.0e-5
        self.optim.max_lr = 1.0e-3
        self.optim.max_grad_norm = 1.0
        self.ppo.entropy_coeff = 0.005
        self.ppo.normalize_advantage_global = False
        self.ppo.clip_log_std = False
        self.ppo.log_std_init = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.sonic_release_optimizer:
            self._apply_release_optimizer_contract()
        else:
            self._apply_local_optimizer_contract()
        # SONIC's g1_recon auxiliary reconstructs the future motion command
        # through its token autoencoder; it is not action behavior cloning.
        # DiffSR pretraining owns that reconstruction objective in this path.
        # Keep rollout action supervision available only as an explicit
        # diagnostic override rather than silently changing the PPO recipe.
        self.ipmd.rollout_bc_coef = 0.0
        self.sync_input_keys()


@configclass
class G1ImitationLatentSonicReleaseRLOptIPMDConfig(
    G1ImitationLatentSonicRLOptIPMDConfig
):
    """Exact public-SONIC-release optimizer contract for cluster-scale runs.

    Select with ``--agent rlopt_ipmd_sonic_release_cfg_entry_point``. Not the
    default as of 2026-07-21 -- the base
    ``G1ImitationLatentSonicRLOptIPMDConfig`` reverted to
    ``sonic_release_optimizer=False`` after underperforming the
    Latent-Strict-v0 + local-optimizer combination at matched scale (see that
    class's docstring). Kept as an explicit, override-proof alias for cluster
    submission scripts that need the release contract specifically.
    """

    sonic_release_optimizer: bool = True


@configclass
class G1ImitationLatentFutureCVAERLOptIPMDConfig(G1ImitationLatentRLOptIPMDConfig):
    """G1 latent policy whose oracle command compresses a 10-frame segment."""

    def sync_input_keys(self) -> None:
        super().sync_input_keys()
        self.ipmd.latent_learning.posterior_input_keys = list(
            FUTURE_CVAE_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = list(FUTURE_CVAE_PRIOR_INPUT_KEYS)
        self.ipmd.latent_learning.reconstruction_target_keys = list(
            FUTURE_CVAE_POSTERIOR_INPUT_KEYS
        )

    def __post_init__(self):
        super().__post_init__()
        self.ipmd.latent_dim = 256
        self.ipmd.latent_steps_min = 10
        self.ipmd.latent_steps_max = 10
        self.ipmd.latent_learning.method = "future_cvae"
        self.ipmd.latent_learning.code_latent_dim = 256
        self.ipmd.latent_learning.command_phase_mode = "none"
        self.sync_input_keys()
        self.ipmd.latent_learning.patch_past_steps = 0
        self.ipmd.latent_learning.patch_future_steps = 9
        self.ipmd.latent_learning.posterior_command_period = 10
        self.ipmd.latent_learning.freeze_encoder = False
        self.ipmd.latent_learning.train_posterior_through_policy = False
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.kl_coeff = 0.01


@configclass
class G1ImitationLatentPerStepVQRLOptIPMDConfig(G1ImitationLatentRLOptIPMDConfig):
    """G1 latent policy consuming one token from a ten-token packet per step."""

    def sync_input_keys(self) -> None:
        super().sync_input_keys()
        self.ipmd.latent_learning.posterior_input_keys = list(
            FUTURE_CVAE_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = []

    def __post_init__(self):
        super().__post_init__()
        self.ipmd.latent_dim = 64
        self.ipmd.latent_steps_min = 1
        self.ipmd.latent_steps_max = 1
        self.sync_input_keys()
        self.ipmd.latent_learning.method = "per_step_vq_sequence"
        self.ipmd.latent_learning.quantizer = "vq_ema"
        self.ipmd.latent_learning.codebook_size = 512
        self.ipmd.latent_learning.codebook_embed_dim = 64
        self.ipmd.latent_learning.code_latent_dim = 64
        self.ipmd.latent_learning.command_phase_mode = "none"
        self.ipmd.latent_learning.token_sequence_horizon = 10
        self.ipmd.latent_learning.patch_past_steps = 0
        self.ipmd.latent_learning.patch_future_steps = 9
        self.ipmd.latent_learning.freeze_encoder = False
        self.ipmd.latent_learning.train_posterior_through_policy = False
        self.ipmd.latent_learning.recon_coeff = 1.0
        self.ipmd.latent_learning.action_recon_coeff = 0.0
