from isaaclab.utils import configclass

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

LATENT_POSTERIOR_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion"),
    ("policy", "expert_anchor_pos_b"),
    ("policy", "expert_anchor_ori_b"),
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
        self.save_interval = 100  # rollout iterations

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
    """Latent-conditioned RLOpt IPMD configuration for G1 imitation."""

    _default_use_latent_command: bool = True
