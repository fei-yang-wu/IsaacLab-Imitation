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

# Heracles-style 38D: 29 joint positions + 3D root position + 6D root
# orientation, per frame. No joint velocities -- the controller is trained on
# this space, so they are absent rather than reconstructed.
ROOT_QPOS_COMMAND_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_motion_qpos"),
    ("policy", "expert_anchor_pos_b"),
    ("policy", "expert_anchor_ori_b"),
]

# HuMI-style sparse keypoints, 24D per frame: 5 keypoint positions (pelvis plus
# the four end-effectors) + 3D root position + 6D root orientation. The most
# compressed explicit interface in the study -- and the one that tests whether
# adding the root repairs the deficiency that made EE-only untrackable.
ROOT_POINTS5_COMMAND_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_keypoint_pos_b"),
    ("policy", "expert_anchor_pos_b"),
    ("policy", "expert_anchor_ori_b"),
]

# Five full keypoint poses plus root pose: 54D per frame, 540D per ten-frame
# packet. This remains a named preset for compatibility, but is assembled from
# the same independent components available through ``command_components``.
ROOT_POINTS5_POSE_COMMAND_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_keypoint_pos_b"),
    ("policy", "expert_keypoint_ori_b"),
    ("policy", "expert_anchor_pos_b"),
    ("policy", "expert_anchor_ori_b"),
]

SINGLE_FRAME_EE_COMMAND_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_ee_pos_b"),
    ("policy", "expert_ee_ori_b"),
]

# Ordered actor contract for the EE-chunk tracker (126 = 12 + 24 + 3 + 29 + 29
# + 29). The command halves come from the ``expert_window`` group rather than
# ``policy``; under ee_chunk_current_slot those terms return the phase-aligned
# slot of the held packet, so the tracker still sees a single 36-value frame.
EE_POLICY_INPUT_KEYS: list[tuple[str, str]] = [
    ("policy", "expert_ee_pos_b"),
    ("policy", "expert_ee_ori_b"),
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
    "single_frame_ee": "single_frame_ee",
    "root_qpos": "root_qpos",
    "root_points5": "root_points5",
    "root_points5_pose": "root_points5_pose",
    "root_keypoints5_pose": "root_points5_pose",
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

# Atomic command components are canonicalized in this order. The config selects
# a set, not a concatenation order, so spelling the same ablation in a different
# YAML order cannot silently change a checkpoint's actor contract.
COMMAND_COMPONENT_ORDER: tuple[str, ...] = (
    "joint_qpos_qvel",
    "joint_qpos",
    "keypoint_pos",
    "keypoint_ori",
    "ee_pos",
    "ee_ori",
    "root_pos",
    "root_ori",
)

COMMAND_COMPONENT_TERM_NAMES: dict[str, str] = {
    "joint_qpos_qvel": "expert_motion",
    "joint_qpos": "expert_motion_qpos",
    "keypoint_pos": "expert_keypoint_pos_b",
    "keypoint_ori": "expert_keypoint_ori_b",
    "ee_pos": "expert_ee_pos_b",
    "ee_ori": "expert_ee_ori_b",
    "root_pos": "expert_anchor_pos_b",
    "root_ori": "expert_anchor_ori_b",
}

COMMAND_COMPONENT_ALIASES: dict[str, str] = {
    **{name: name for name in COMMAND_COMPONENT_ORDER},
    "qpos_qvel": "joint_qpos_qvel",
    "full_joint_state": "joint_qpos_qvel",
    "qpos": "joint_qpos",
    "keypoint_position": "keypoint_pos",
    "keypoint_orientation": "keypoint_ori",
    "ee_position": "ee_pos",
    "ee_orientation": "ee_ori",
    "root_position": "root_pos",
    "root_orientation": "root_ori",
}

COMMAND_SPACE_COMPONENT_PRESETS: dict[str, tuple[str, ...]] = {
    "single_frame_full_body": ("joint_qpos_qvel", "root_pos", "root_ori"),
    "single_frame_ee": ("ee_pos", "ee_ori"),
    "root_qpos": ("joint_qpos", "root_pos", "root_ori"),
    "root_points5": ("keypoint_pos", "root_pos", "root_ori"),
    "root_points5_pose": (
        "keypoint_pos",
        "keypoint_ori",
        "root_pos",
        "root_ori",
    ),
}


def normalize_command_components(
    command_components: list[str] | tuple[str, ...] | str,
) -> tuple[str, ...]:
    """Validate and canonically order an explicit command component set.

    Accepts a ``"[a,b,c]"``/``"a,b,c"`` string form as well: Isaac Lab's
    strict config updater passes a Hydra CLI override for a ``None``-default
    field through as the raw string (same gotcha as
    ``env.expert_macro_state_terms``).
    """
    if isinstance(command_components, str):
        command_components = [
            part
            for part in command_components.strip().strip("[]").split(",")
            if part.strip()
        ]
    normalized: list[str] = []
    for raw_name in command_components:
        name = str(raw_name).strip().lower().replace("-", "_")
        try:
            normalized.append(COMMAND_COMPONENT_ALIASES[name])
        except KeyError as err:
            raise ValueError(
                f"Unsupported command component {raw_name!r}. Expected a subset "
                f"of {list(COMMAND_COMPONENT_ORDER)}."
            ) from err
    if not normalized:
        raise ValueError("command_components must select at least one component.")
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"command_components contains duplicates: {duplicates}.")
    if {"joint_qpos_qvel", "joint_qpos"}.issubset(normalized):
        raise ValueError(
            "joint_qpos_qvel and joint_qpos are mutually exclusive command components."
        )
    selected = set(normalized)
    return tuple(name for name in COMMAND_COMPONENT_ORDER if name in selected)


def command_component_input_keys(
    command_components: list[str] | tuple[str, ...],
    *,
    observation_group: str = "policy",
) -> list[tuple[str, str]]:
    """Build ordered observation keys for a composable explicit command."""
    return [
        (str(observation_group), COMMAND_COMPONENT_TERM_NAMES[name])
        for name in normalize_command_components(command_components)
    ]


def command_space_components(command_space: str) -> tuple[str, ...] | None:
    """Return the composable form of a legacy single-frame command preset."""
    normalized = normalize_command_space(command_space)
    return COMMAND_SPACE_COMPONENT_PRESETS.get(normalized)


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
    # Restored 2026-07-27. Dropping this left the critic without the expert's
    # 29 joint positions and velocities, reachable only through the 258-d
    # latent, while the actor-side SONIC contract only *adds* projected
    # gravity. ICE job 5541139 halved the per-step `motion_body_ori` and
    # `motion_body_ang_vel` reward against the Study B `deterministic` row and
    # left the anchor terms nearly intact -- the signature of a critic starved
    # of joint-configuration detail. SONIC's own release critic
    # (`privileged_mf_hist`) is likewise reference-aware, so the omission was
    # not release fidelity.
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


def apply_reward_estimation_switch(
    agent_cfg, reward_input_keys: list[tuple[str, str]] = REWARD_INPUT_KEYS
) -> None:
    """Apply the declarative ``reward_estimation`` switch (single authority).

    ``reward_estimation=False`` (the default) parks the IPMD IRL stack: all
    five reward-estimator loss/regularizer coefficients are zeroed and
    ``ipmd.reward_input_keys`` is cleared to ``None`` -- RLOpt then falls back
    to the value-function input keys for the constructed-but-never-updated
    estimator, so the run needs no ``reward_input`` observation group (the
    ``-G1-v2`` env default drops it). ``reward_estimation=True`` declares that
    the run trains the estimator: it selects ``reward_input_keys`` (so the env
    must expose the group; set ``env.enable_reward_input_observations=True``
    on tasks that default it off) and restores the historical vanilla
    coefficients (``reward_loss_coeff=1.0``, every regularizer 0.0).

    Called at the end of each IPMD-family ``sync_input_keys`` and
    ``__post_init__`` so it wins over every earlier branch (including the
    latent hl_skill zeroing) and over Hydra-applied ``agent.*`` overrides
    (the train entrypoint re-runs ``sync_input_keys`` after applying them).
    """
    reward_estimation = bool(agent_cfg.reward_estimation)
    ipmd = agent_cfg.ipmd
    ipmd.reward_input_keys = list(reward_input_keys) if reward_estimation else None
    ipmd.reward_loss_coeff = 1.0 if reward_estimation else 0.0
    ipmd.reward_l2_coeff = 0.0
    ipmd.reward_grad_penalty_coeff = 0.0
    ipmd.reward_logit_reg_coeff = 0.0
    ipmd.reward_param_weight_decay_coeff = 0.0


def normalize_command_space(command_space: str) -> str:
    normalized = str(command_space).strip().lower().replace("-", "_")
    try:
        return COMMAND_SPACE_ALIASES[normalized]
    except KeyError as err:
        raise ValueError(
            f"Unsupported command_space={command_space!r}. "
            f"Expected one of {sorted(set(COMMAND_SPACE_ALIASES.values()))}."
        ) from err


def command_space_policy_input_keys(
    command_space: str,
    command_components: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, str]]:
    if command_components is not None:
        return list(
            command_component_input_keys(command_components) + PROPRIO_POLICY_INPUT_KEYS
        )
    command_space = normalize_command_space(command_space)
    if command_space == "single_frame_full_body":
        return list(VANILLA_POLICY_INPUT_KEYS)
    if command_space == "full_body_trajectory":
        return list(FULL_BODY_TRAJECTORY_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    if command_space == "ee_trajectory":
        return list(EE_TRAJECTORY_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    if command_space == "single_frame_ee":
        # Chunked EE: the actor consumes one 36-value frame (the phase-aligned
        # slot of the held packet), exactly as during training.
        return list(SINGLE_FRAME_EE_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    if command_space == "root_qpos":
        return list(ROOT_QPOS_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    if command_space == "root_points5":
        return list(ROOT_POINTS5_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    if command_space == "root_points5_pose":
        return list(ROOT_POINTS5_POSE_COMMAND_KEYS + PROPRIO_POLICY_INPUT_KEYS)
    raise AssertionError(f"Unhandled command space: {command_space}")


def command_space_critic_input_keys(
    command_space: str,
    command_components: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, str]]:
    if command_components is not None:
        # Actor and critic command entries have the same numerical values but
        # remain separate observation groups; privileged state is critic-only.
        return list(
            command_component_input_keys(command_components, observation_group="critic")
            + PRIVILEGED_CRITIC_STATE_KEYS
        )
    command_space = normalize_command_space(command_space)
    if command_space == "single_frame_full_body":
        return list(VANILLA_CRITIC_INPUT_KEYS)
    if command_space == "full_body_trajectory":
        return list(FULL_BODY_TRAJECTORY_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    if command_space == "ee_trajectory":
        return list(EE_TRAJECTORY_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    if command_space == "single_frame_ee":
        return list(SINGLE_FRAME_EE_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    if command_space == "root_qpos":
        return list(ROOT_QPOS_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    if command_space == "root_points5":
        return list(ROOT_POINTS5_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    if command_space == "root_points5_pose":
        return list(ROOT_POINTS5_POSE_COMMAND_KEYS + PRIVILEGED_CRITIC_STATE_KEYS)
    raise AssertionError(f"Unhandled command space: {command_space}")


@configclass
class _G1ImitationRLOptIPMDBaseConfig(IPMDRLOptConfig):
    """Shared RLOpt IPMD configuration for G1 imitation."""

    _default_use_latent_command: bool = False
    command_space: str = "single_frame_full_body"
    # Optional atomic explicit-interface selection. When set, this is the actor
    # command contract and ``command_space`` is retained only as a run label for
    # backward-compatible metadata. Hydra can therefore express new ablations
    # without adding an enum arm in Python.
    command_components: list[str] | None = None
    command_spec_name: str = ""
    # Declares that this run trains the IPMD reward estimator (IRL) and
    # therefore requires the env's reward_input observation group. False (the
    # default) parks the stack: see `apply_reward_estimation_switch`.
    reward_estimation: bool = False

    def sync_input_keys(self) -> None:
        use_latent_command = bool(self.ipmd.use_latent_command)
        components: tuple[str, ...] | None = None
        if self.command_components is None:
            self.command_space = normalize_command_space(self.command_space)
            if not self.command_spec_name:
                self.command_spec_name = self.command_space
        else:
            if use_latent_command:
                raise ValueError(
                    "command_components configures an explicit tracker and cannot "
                    "be combined with ipmd.use_latent_command=true."
                )
            components = normalize_command_components(self.command_components)
            self.command_components = list(components)
            if not self.command_spec_name:
                self.command_spec_name = "composed__" + "__".join(components)
        self.policy.input_keys = (
            list(LATENT_POLICY_INPUT_KEYS)
            if use_latent_command
            else command_space_policy_input_keys(self.command_space, components)
        )
        if self.value_function is not None:
            self.value_function.input_keys = (
                list(LATENT_CRITIC_INPUT_KEYS)
                if use_latent_command
                else command_space_critic_input_keys(self.command_space, components)
            )
        apply_reward_estimation_switch(self)
        self.ipmd.latent_learning.posterior_input_keys = list(
            LATENT_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = list(LATENT_PRIOR_INPUT_KEYS)
        self.ipmd.latent_key = ("policy", "latent_command")
        self.ipmd.use_latent_command = use_latent_command
        if not use_latent_command and str(self.ipmd.command_source) in (
            "hl_skill",
            "skill_commander",
        ):
            # Explicit command mode consumes no latent command, but latent
            # task defaults still carry command_source='hl_skill', whose
            # validation demands an encoder checkpoint that an explicit
            # tracker does not have. Downgrade to the inert source unless the
            # user explicitly wired a checkpoint (e.g. packet-encoder eval).
            if not str(getattr(self.ipmd, "hl_skill_checkpoint_path", "") or ""):
                self.ipmd.command_source = "random"
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
        # Reward-estimator coefficients are owned by the declarative
        # `reward_estimation` switch applied at the end of this method.
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
            # hl_skill drives the objective; the learned-reward terms stay
            # disabled via the `reward_estimation` switch below.

        # Single authority for the parked-vs-active reward-estimation wiring;
        # runs after every branch above (including the latent one) so the
        # declarative `reward_estimation` field always wins.
        apply_reward_estimation_switch(self)


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
      (the strict surface, now Latent-v0, + this local contract, same 8192x12x12288 scale)
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
        if not bool(self.ipmd.use_latent_command):
            # Explicit command mode (env.command_mode=explicit): keep the base
            # class's command_space / command_components key selection instead
            # of forcing the SONIC latent contract. Latent runs are unchanged.
            return
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
    strict-surface (now Latent-v0) + local-optimizer combination at matched scale (see that
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


@configclass
class G1ImitationLatentSonicOfficialFSQRLOptIPMDConfig(
    G1ImitationLatentSonicRLOptIPMDConfig
):
    """Official-window SONIC FSQ recipe adapted to this repo's RLOpt stack.

    The entire current-plus-nine-future-frame reference window is encoded into
    one 64-value FSQ command and recomputed every 50 Hz control step. This
    replaces the removed cached-packet implementation, which encoded ten
    independent per-step tokens and consumed them without renewing the window.
    """

    sonic_release_optimizer: bool = True

    def sync_input_keys(self) -> None:
        super().sync_input_keys()
        self.ipmd.latent_learning.posterior_input_keys = list(
            FUTURE_CVAE_POSTERIOR_INPUT_KEYS
        )
        self.ipmd.latent_learning.prior_input_keys = []

    def __post_init__(self):
        super().__post_init__()
        # Public SONIC normalizes only the critic input; its actor declares
        # running_mean_std=false because the quantized token geometry is fixed.
        self.policy.normalize_input = False
        assert self.value_function is not None
        self.value_function.normalize_input = True
        self.ipmd.latent_dim = 64
        self.ipmd.command_source = "posterior"
        self.ipmd.latent_steps_min = 1
        self.ipmd.latent_steps_max = 1
        self.ipmd.latent_learning.method = "patch_vqvae"
        self.ipmd.latent_learning.quantizer = "fsq"
        # Public SONIC: max_num_tokens=2, fsq_level_list=32, hence 64 scalar
        # coordinates with 32 levels each.
        self.ipmd.latent_learning.fsq_levels = [32] * 64
        self.ipmd.latent_learning.fsq_normalize_codes = True
        self.ipmd.latent_learning.code_latent_dim = 64
        self.ipmd.latent_learning.command_phase_mode = "none"
        self.ipmd.latent_learning.patch_past_steps = 0
        self.ipmd.latent_learning.patch_future_steps = 9
        self.ipmd.latent_learning.code_period = 1
        self.ipmd.latent_learning.posterior_command_period = 1
        self.ipmd.latent_learning.encoder_hidden_dims = [2048, 1024, 512, 512]
        self.ipmd.latent_learning.encoder_activation = "silu"
        self.ipmd.latent_learning.decoder_hidden_dims = [2048, 1024, 512, 512]
        self.ipmd.latent_learning.decoder_activation = "silu"
        self.ipmd.latent_learning.lr = 2.0e-5
        self.ipmd.latent_learning.freeze_encoder = False
        self.ipmd.latent_learning.train_posterior_through_policy = True
        self.ipmd.latent_learning.recon_coeff = 0.01
        self.ipmd.latent_learning.action_recon_coeff = 0.0
        self.sync_input_keys()
