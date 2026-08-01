# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared G1 tracking env machinery (the base class, recipes, Hydra plumbing).

Architecture: three recipes x pure command configuration. A *recipe* is the
reward/termination/reset design and is the only axis with separate env
classes; the latent-vs-explicit command choice is configuration
(``command_mode`` + ``command_observation_terms`` on the env,
``ipmd.use_latent_command`` / ``command_components`` on the agent).

- **LafanTrack** (``imitation_g1_env_v0.ImitationG1LafanTrackEnvCfg``,
  ``Isaac-Imitation-G1-v0``): the original torso-anchored, loose-termination
  tracking recipe on the vanilla observation surface.
- **Strict** (``_apply_strict_recipe``): strict SONIC termination functions on
  the legacy scaffolding (pelvis anchor, [0, 200] reset starts, no
  curriculum). Pins live in ``variants/strict.py``.
- **Stable** (``imitation_g1_env_v1.ImitationG1EnvCfg``,
  ``Isaac-Imitation-G1-v1``): the SONIC release recipe with this repo's
  legacy reset distribution. The flagship class name follows the newest
  release.

``ImitationG1BaseTrackingEnvCfg`` carries everything every G1 task shares:
the vanilla component defaults, the LAFAN1 dataset/manifest resolution, the
table-driven anchor re-pointing (``_set_anchor_body``), and the command-term
pruning/restoration for ``command_mode``. Release files
(``imitation_g1_env_v*.py``) and variant files compose it with components
from ``common``; the recipe helpers (``_apply_pelvis_protocol``,
``_apply_strict_recipe``) keep the shared protocol deltas in one place.
"""

import copy
from collections.abc import Mapping
from pathlib import Path

from isaaclab.utils.configclass import configclass

from ....imitation_env_cfg import ImitationLearningEnvCfg
from ....motion_manifest import (
    build_lafan1_loader_kwargs,
    dataset_path_from_entries,
    infer_npz_manifest_control_freq,
    load_lafan1_manifest,
    load_lafan1_manifest_loader_options,
    load_manifest_family,
)
from .actions import G1ActionsCfg
from .constants import (
    G1_29DOF_DATASET_BODY_NAMES,
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from .events import G1EventCfg
from .observations import (
    _DEFAULT_EXPERT_MACRO_STATE_TERMS,
    _EXPERT_WINDOW_TERM_NAMES,
    _LATENT_MODE_DEFAULT_COMMAND_TERM_NAMES,
    _PRUNABLE_COMMAND_TERM_NAMES,
    _VANILLA_ANCHOR_TERM_NAMES_BY_GROUP,
    G1ObservationCfg,
)
from .presets import (
    G1ImitationContactSensorCfg,
    G1ImitationPhysicsCfg,
    G1ImitationRobotCfg,
    _set_contact_sensor_update_period,
)
from .rewards import G1RewardsCfg
from .terminations import G1TerminationsCfg


@configclass
class ImitationG1BaseTrackingEnvCfg(ImitationLearningEnvCfg):
    """Shared 29-DoF G1 tracking config aligned with Unitree mimic tracking settings."""

    actions = G1ActionsCfg()
    observations = G1ObservationCfg()
    rewards = G1RewardsCfg()  # type: ignore
    terminations = G1TerminationsCfg()  # type: ignore
    events = G1EventCfg()

    device: str = "cuda"
    replay_reference: bool = False
    replay_only: bool = False
    reference_start_frame: int = 0
    latent_command_dim: int = 64
    latent_patch_past_steps: int = 0
    latent_patch_future_steps: int = 0
    # Anchor used when constructing expert batches and high-level macro states.
    # Keep the historical torso convention by default; SONIC overrides this to
    # pelvis so offline skill pretraining and live policy commands agree.
    expert_anchor_body_name: str = "torso_link"
    # Hold command-window observations for N control steps between renewals
    # (VLA-style chunk consumption): the window is snapshotted every N steps in
    # the renewal-time anchor frame and consumed as a time-shifted view with
    # tail padding. 0 keeps the per-step sliding-window behavior. Requires
    # latent_patch_past_steps == 0 when enabled.
    command_hold_steps: int = 0
    random_reset_step_min: int = 0
    random_reset_step_max: int = 0
    # Starting-frame selection on reset. "auto" (default) keeps the legacy
    # behavior: uniform in [random_reset_step_min, random_reset_step_max] when
    # a range is configured, else the fixed reference_start_frame. Explicit
    # "fixed" / "random" / "adaptive" select the StartFrameSampler mode
    # directly; "adaptive" uses the SONIC failure-weight function (or a custom
    # callable attached as `cfg.adaptive_reset_weight_fn`, which must accept
    # (trajectory_ranks, frame_steps) tensors and return one non-negative
    # weight per pair). Only applies when random_reset_full_trajectory is
    # False -- full-trajectory resets keep the SONIC joint rank+frame sampler
    # unchanged.
    reset_start_mode: str = "auto"
    random_reset_full_trajectory: bool = False
    adaptive_failure_reset_bin_size: int = 50
    adaptive_failure_reset_sequence_length_agnostic: bool = True
    adaptive_failure_reset_init_num_failures: float = 1.0
    adaptive_failure_reset_uniform_ratio: float = 0.1
    adaptive_failure_reset_pre_failure_window: int = 200
    adaptive_failure_reset_failure_rate_max_over_mean: float = 50.0

    _debug_rewards: bool = False

    # Offscreen-video camera. Default: a static elevated bird view over the
    # env grid near the origin (set below via cfg.viewer), which shows a
    # couple dozen robots for generic motion-quality checks. The follow
    # camera remains available (video_follow_robot=True) for close-ups of a
    # single environment, e.g. with full-trajectory random starts where
    # robots wander far from their origins.
    video_follow_robot: bool = False
    video_follow_env_index: int = 0
    video_follow_eye_offset: tuple[float, float, float] = (3.5, 3.5, 2.0)
    video_follow_lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Master switch for all expensive visualizers/marker debug rendering.
    # Keep disabled by default for training/runtime performance.
    enable_visualizers: bool = False
    visualize_reference_arrows: bool = True
    print_reference_velocity: bool = False
    print_reference_velocity_every: int = 50

    # `target_joint_names` MUST be the robot articulation (USD) order because the
    # reference is written directly onto robot.data.joint_pos.torch. `reference_joint_names`
    # is the default order assumed for reference data; it is overridden at runtime
    # by the dataset's own `joint_names` when present (self-describing data), and the
    # reference->target remap converts to articulation order.
    reference_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    target_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    # Body order of the recorded NPZ body arrays (PhysX enumeration). Used by
    # the env instead of the live robot's body order, which is backend-specific.
    reference_body_names: list[str] = G1_29DOF_DATASET_BODY_NAMES.copy()
    # Same body set the closed-loop evaluators use, so the training curve and
    # the reported evaluation MPJPE are the same quantity.
    mpjpe_metric_body_names: list[str] = G1_TRACKED_BODY_NAMES.copy()
    command_ee_body_names: list[str] = G1_EE_BODY_NAMES.copy()
    command_keypoint_body_names: list[str] = G1_KEYPOINT5_BODY_NAMES.copy()
    command_observation_source: str = "reference"
    # The chunk adapter redirects only the three policy command tensors while
    # preserving the vanilla actor keys and 67-D command contract.
    policy_command_mode: str = "reference"
    # Optional whitelist of policy-group COMMAND terms to keep. The observation
    # manager computes every term in a group whether or not the actor reads it,
    # and each command term whose body set differs from the others forces its own
    # expert-window build. Measured at 1024 envs on Newton: keeping all four
    # reduced-interface command terms costs 39.5 ms/step versus 36.6 ms with them
    # dropped -- about 7% of wall clock, or ~3.5 h on a two-day job, paid by every
    # run including those that read none of them.
    #
    # None (the default) keeps every term, so existing runs are unaffected. Set it
    # to the command terms the active command space actually consumes, e.g.
    #   env.command_observation_terms='[expert_keypoint_pos_b,expert_anchor_pos_b,expert_anchor_ori_b]'
    # Pruning is numerically neutral for the actor: these terms carry no
    # observation noise, so removing them consumes no RNG and the retained terms
    # are byte-identical.
    command_observation_terms: list[str] | None = None
    # Which command family feeds the actor: this is pure configuration, not a
    # recipe axis. "explicit" prunes the agent-published `latent_command` term
    # (where the observation surface has one) and keeps the explicit command
    # terms selected by `command_observation_terms` (all of them when None).
    # "latent" keeps `latent_command` plus, by default, the historical explicit
    # baseline terms (expert_motion + anchors) that latent surfaces expose for
    # posterior-mode baselines; `command_observation_terms` overrides that set.
    # Pair with the matching agent config: `agent.ipmd.use_latent_command`
    # must agree with this mode (validated at training entry).
    command_mode: str = "explicit"
    # Expert-window terms making up one DiffSR macro-state frame. None keeps the
    # full-body default (expert_motion 58 + anchor_pos 3 + anchor_ori 6 = 67 ->
    # 670 per 10-frame window, byte-identical to the full-body packet). Set to
    # ["expert_motion_qpos", "expert_anchor_pos_b", "expert_anchor_ori_b"] for a
    # GR00T-style whole-body qpos+root latent space: 29+3+6 = 38 -> 380,
    # byte-identical to the root_qpos packet. The skill encoder's input width
    # follows from this, so changing it invalidates any existing encoder.
    expert_macro_state_terms: list[str] | None = None
    # Optional whitelist of expert_window group terms to keep. The observation
    # manager computes every window term each step whether or not anything
    # reads it, and each distinct body set forces its own expert-window build.
    # Measured at 1024 envs on PhysX (Isaac-Imitation-G1-v1, workstation):
    # keeping only the macro-state trio cut mean step time 72.2 -> 65.2 ms
    # (-9.6% wall clock). None (the default) keeps every term, so existing
    # runs are unaffected. The kept set must cover the active
    # `expert_macro_state_terms` (validated); runs whose planner/collection
    # stage reads other window terms (e.g. EE-chunk interfaces) must keep
    # those terms too.
    expert_window_observation_terms: list[str] | None = None
    # Master switch for the expert_goal observation group (where the surface
    # has one). The group is computed every step but consumed only by agents
    # whose input keys select it (goal-conditioned posteriors / hierarchical
    # skills). Measured at 1024 envs on PhysX (Isaac-Imitation-G1-v1):
    # disabling it cut mean step time 72.2 -> 66.7 ms (-7.6%); combined with
    # the macro-only window whitelist above, 72.2 -> 59.4 ms (-17.7%). True
    # (the default) keeps the historical layout.
    enable_expert_goal_observations: bool = True
    # Master switch for the reward_input observation group (where the surface
    # has one). The group feeds only the IPMD reward estimator (IRL
    # discriminator inputs, selected via `agent.ipmd.reward_input_keys`); no
    # current recipe consumes an estimated reward, yet the group's three terms
    # are computed every step and the env pre-materializes an expert-side
    # cache for them at construction. True (the default) keeps the historical
    # v0/v1 layouts; the v2 release parks the stack by defaulting this off.
    # Opting back in pairs with `agent.reward_estimation=true` so the agent
    # actually trains the estimator the group feeds.
    enable_reward_input_observations: bool = True

    def _anchor_term_names_by_group(self) -> dict[str, tuple[str, ...]]:
        """Anchor-relative observation terms per group for this surface.

        Recipe/observation variants override this to return their own table;
        the re-anchoring mechanism itself is shared (see ``_set_anchor_body``).
        """
        return _VANILLA_ANCHOR_TERM_NAMES_BY_GROUP

    def _set_anchor_body(self, anchor_body_name: str) -> None:
        """Point every anchor-relative observation term at one body."""
        for group_name, term_names in self._anchor_term_names_by_group().items():
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            for term_name in term_names:
                term = getattr(group, term_name)
                if term is None:
                    continue
                if "anchor_body_name" in term.params:
                    term.params["anchor_body_name"] = anchor_body_name

    def _set_reward_anchor_body(self, anchor_body_name: str) -> None:
        """Point the anchor-relative reward terms at one body."""
        for term_name in (
            "motion_global_anchor_pos",
            "motion_global_anchor_ori",
            "motion_body_pos",
            "motion_body_ori",
        ):
            getattr(self.rewards, term_name).params["anchor_body_name"] = (
                anchor_body_name
            )

    def _sync_expert_window_observation_params(self) -> None:
        past_steps = int(self.latent_patch_past_steps)
        future_steps = int(self.latent_patch_future_steps)
        # Every term in the expert_window group must appear here. A term left
        # out keeps its declaration-time past/future steps (0/0 -> a 1-step
        # request) while the rest follow the task's window, and the observation
        # manager evaluates the whole group regardless of which terms the macro
        # state selects -- so an unsynced term raises "Planner command window
        # mismatch" on any task with a multi-step window, even when nothing
        # reads it. (Terms pruned to None by `expert_window_observation_terms`
        # are skipped.)
        for term in (
            self.observations.expert_window.expert_motion,
            self.observations.expert_window.expert_motion_qpos,
            self.observations.expert_window.expert_anchor_pos_b,
            self.observations.expert_window.expert_anchor_ori_b,
        ):
            if term is None:
                continue
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
        for term in (
            self.observations.expert_window.expert_ee_pos_b,
            self.observations.expert_window.expert_ee_ori_b,
        ):
            if term is None:
                continue
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
            term.params["reference_body_names"] = tuple(self.command_ee_body_names)
        for term in (
            self.observations.expert_window.expert_keypoint_pos_b,
            self.observations.expert_window.expert_keypoint_ori_b,
        ):
            if term is None:
                continue
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
            term.params["reference_body_names"] = tuple(
                self.command_keypoint_body_names
            )

    def __post_init__(self) -> None:
        super().__post_init__()  # type: ignore

        robot_preset = G1ImitationRobotCfg()
        for variant in (
            robot_preset.default,
            robot_preset.physx,
            robot_preset.newton_mjwarp,
        ):
            variant.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot_preset  # type: ignore
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Isaac Lab 3.0: SimulationCfg.physx was replaced by the backend-selecting
        # SimulationCfg.physics field. The PresetCfg resolves to PhysX by default
        # and to Newton with the `physics=newton_mjwarp` CLI override.
        self.sim.physics = G1ImitationPhysicsCfg()

        if self.scene.contact_forces is not None:
            # Per-backend sensor implementations; keep every variant's runtime
            # settings in sync because preset resolution happens after this.
            contact_preset = G1ImitationContactSensorCfg()
            for variant in (
                contact_preset.default,
                contact_preset.physx,
                contact_preset.newton_mjwarp,
            ):
                variant.update_period = self.sim.dt
                variant.force_threshold = 10.0
                variant.debug_vis = bool(self.enable_visualizers)
            self.scene.contact_forces = contact_preset

        # Reference marker visualizers are also gated by the master toggle.
        self.visualize_reference_arrows = bool(
            self.enable_visualizers and self.visualize_reference_arrows
        )

        self.scene.height_scanner = None

        if int(self.latent_patch_past_steps) < 0:
            raise ValueError("latent_patch_past_steps must be >= 0.")
        if int(self.latent_patch_future_steps) < 0:
            raise ValueError("latent_patch_future_steps must be >= 0.")
        if int(self.command_hold_steps) < 0:
            raise ValueError("command_hold_steps must be >= 0.")
        if int(self.command_hold_steps) > 0 and int(self.latent_patch_past_steps) > 0:
            raise ValueError(
                "command_hold_steps requires latent_patch_past_steps == 0; "
                "held chunk consumption is only defined for future-only windows."
            )
        normalized_policy_mode = (
            str(self.policy_command_mode).strip().lower().replace("-", "_")
        )
        if normalized_policy_mode not in {
            "reference",
            "explicit_chunk_current_slot",
            "full_body_chunk_current_slot",
            "ee_chunk_current_slot",
        }:
            raise ValueError(
                "policy_command_mode must be reference or a supported "
                "*_chunk_current_slot adapter."
            )
        self.policy_command_mode = normalized_policy_mode
        if int(self.random_reset_step_min) < 0:
            raise ValueError("random_reset_step_min must be >= 0.")
        if int(self.random_reset_step_max) < int(self.random_reset_step_min):
            raise ValueError("random_reset_step_max must be >= random_reset_step_min.")
        if str(self.reset_start_mode).strip().lower() not in {
            "auto",
            "fixed",
            "random",
            "adaptive",
        }:
            raise ValueError(
                "reset_start_mode must be one of 'auto', 'fixed', 'random', "
                f"'adaptive'; got {self.reset_start_mode!r}."
            )
        self.reset_start_mode = str(self.reset_start_mode).strip().lower()
        if int(self.adaptive_failure_reset_bin_size) <= 0:
            raise ValueError("adaptive_failure_reset_bin_size must be positive.")
        if float(self.adaptive_failure_reset_init_num_failures) <= 0.0:
            raise ValueError(
                "adaptive_failure_reset_init_num_failures must be positive."
            )
        if not 0.0 <= float(self.adaptive_failure_reset_uniform_ratio) <= 1.0:
            raise ValueError("adaptive_failure_reset_uniform_ratio must be in [0, 1].")
        if int(self.adaptive_failure_reset_pre_failure_window) < 0:
            raise ValueError("adaptive_failure_reset_pre_failure_window must be >= 0.")
        if float(self.adaptive_failure_reset_failure_rate_max_over_mean) <= 0.0:
            raise ValueError(
                "adaptive_failure_reset_failure_rate_max_over_mean must be positive."
            )

        self._sync_expert_window_observation_params()

        self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
        self._normalize_sequence_overrides()
        self._validate_reset_schedule()
        self._resolve_manifest_config()
        self._prune_command_observation_terms()
        self._prune_expert_window_observation_terms()
        self._apply_expert_goal_observation_toggle()
        self._apply_reward_input_observation_toggle()

    # ------------------------------------------------------------------
    # LAFAN1 dataset/manifest machinery, inherited by every G1 task.
    # ------------------------------------------------------------------

    dataset_path: str | None = "data/lafan1/g1/"
    loader_type: str = "lafan1_csv"
    loader_kwargs: dict = {
        "dataset_name": "lafan1",
        "dataset": {"trajectories": {"lafan1_csv": []}},
        "control_freq": 50.0,
        "sim": {"dt": 0.005},
        "decimation": 4,
        "joint_names": G1_29DOF_ISAACLAB_JOINT_NAMES,
        "canonical_joint_names": G1_29DOF_ISAACLAB_JOINT_NAMES,
    }
    reset_schedule: str = "random"
    refresh_zarr_dataset: bool = False
    require_npz_body_states: bool = True
    lafan1_manifest_path: str | None = None
    motions: list[str] | None = None
    trajectories: list[str] | None = None
    wrap_steps: bool = False
    sync_control_rate_to_manifest: bool = True
    preferred_manifest_physics_fps: float = 240.0
    lafan1_loader_chunk_size: int | None = None
    lafan1_loader_shard_size: int | None = None

    def _apply_optional_hydra_overrides(self, data: Mapping) -> dict:
        """Apply optional top-level overrides before Isaac Lab's strict type updater.

        Isaac Lab updates config objects by comparing the incoming value type against the
        runtime type of the existing attribute. That rejects Hydra overrides such as
        `None -> str` for optional public fields like `lafan1_manifest_path`.
        """
        remaining = dict(data)

        if "lafan1_manifest_path" in remaining:
            value = remaining.pop("lafan1_manifest_path")
            self.lafan1_manifest_path = None if value is None else str(value)

        if "dataset_path" in remaining:
            value = remaining.pop("dataset_path")
            self.dataset_path = None if value is None else str(value)

        if "motions" in remaining:
            value = remaining.pop("motions")
            if value is None:
                self.motions = None
            elif isinstance(value, (list, tuple)):
                self.motions = [str(item) for item in value]
            else:
                raise ValueError("motions must be a list of motion names or null.")

        if "command_observation_terms" in remaining:
            # Optional None-default field: Isaac's strict updater rejects the
            # `None -> str` transition for the raw "[a,b,c]" Hydra CLI string.
            # Store it as-is; `_prune_command_observation_terms` parses both
            # forms.
            value = remaining.pop("command_observation_terms")
            if value is None or isinstance(value, str):
                self.command_observation_terms = value
            elif isinstance(value, (list, tuple)):
                self.command_observation_terms = [str(item) for item in value]
            else:
                raise ValueError(
                    "command_observation_terms must be a list of term names, "
                    "an '[a,b,c]' string, or null."
                )

        if "expert_window_observation_terms" in remaining:
            # Same None-default gotcha as `command_observation_terms`:
            # `_prune_expert_window_observation_terms` parses both forms.
            value = remaining.pop("expert_window_observation_terms")
            if value is None or isinstance(value, str):
                self.expert_window_observation_terms = value
            elif isinstance(value, (list, tuple)):
                self.expert_window_observation_terms = [str(item) for item in value]
            else:
                raise ValueError(
                    "expert_window_observation_terms must be a list of term "
                    "names, an '[a,b,c]' string, or null."
                )

        if "trajectories" in remaining:
            value = remaining.pop("trajectories")
            if value is None:
                self.trajectories = None
            elif isinstance(value, (list, tuple)):
                self.trajectories = [str(item) for item in value]
            else:
                raise ValueError(
                    "trajectories must be a list of trajectory names or null."
                )

        return remaining

    def _lafan_source_entries(self) -> list[dict[str, object]]:
        try:
            entries = self.loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]
        except Exception as err:
            raise ValueError(
                "loader_kwargs must define dataset.trajectories.lafan1_csv with at least one source entry."
            ) from err
        if not isinstance(entries, list) or len(entries) == 0:
            raise ValueError(
                "loader_kwargs.dataset.trajectories.lafan1_csv must be a non-empty list."
            )
        return entries

    def _validate_source_path(self, source_path: Path) -> None:
        if not source_path.is_file():
            raise FileNotFoundError(
                "LAFAN1 motion source is missing. "
                f"Expected: {source_path}. "
                "Set `lafan1_manifest_path` to a manifest that points at repo-local NPZ motions."
            )
        if self.require_npz_body_states and source_path.suffix.lower() != ".npz":
            raise ValueError(
                "This tracking env requires an npz source with body states "
                "(body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w). "
                f"Got: {source_path}. "
                "Generate repo-local NPZ files before loading this manifest."
            )

    def _normalize_sequence_overrides(self) -> None:
        if self.motions is not None:
            self.motions = list(self.motions)
        if self.trajectories is not None:
            self.trajectories = list(self.trajectories)

    def _validate_reset_schedule(self) -> None:
        allowed_reset_schedules = {"random", "sequential", "round_robin"}
        self.reset_schedule = self.reset_schedule.strip().lower()
        if self.reset_schedule not in allowed_reset_schedules:
            raise ValueError(
                f"Unsupported reset_schedule='{self.reset_schedule}'. "
                f"Allowed values: {sorted(allowed_reset_schedules)}."
            )

    def _validate_lafan_source_entries(
        self, source_entries: list[dict[str, object]]
    ) -> None:
        for source in source_entries:
            source_path = Path(str(source["path"])).expanduser().resolve()
            source["path"] = str(source_path)
            self._validate_source_path(source_path)

    def _set_control_frequency(self, control_freq: float) -> None:
        control_freq = float(control_freq)
        if control_freq <= 0.0:
            raise ValueError("control_freq must be positive.")

        def _integer_timing_for(
            physics_fps: float,
        ) -> tuple[float, int] | None:
            if physics_fps <= 0.0:
                return None
            decimation = max(int(round(physics_fps / control_freq)), 1)
            actual_control_freq = physics_fps / decimation
            if abs(actual_control_freq - control_freq) <= 1.0e-6:
                return 1.0 / physics_fps, decimation
            return None

        current_physics_fps = 1.0 / float(self.sim.dt)
        timing = _integer_timing_for(current_physics_fps)
        if timing is None:
            timing = _integer_timing_for(float(self.preferred_manifest_physics_fps))
        if timing is None:
            timing = (1.0 / control_freq, 1)

        self.sim.dt, self.decimation = timing
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            _set_contact_sensor_update_period(self.scene.contact_forces, self.sim.dt)

    def _sync_control_rate_to_manifest_entries(
        self,
        source_entries: list[dict[str, object]],
        *,
        timing_explicit: bool = False,
    ) -> None:
        if timing_explicit or not bool(self.sync_control_rate_to_manifest):
            return
        control_freq = infer_npz_manifest_control_freq(source_entries)
        if control_freq is None:
            return
        self._set_control_frequency(control_freq)

    def _resolve_manifest_config(
        self,
        *,
        dataset_path_explicit: bool = False,
        motions_explicit: bool = False,
        timing_explicit: bool = False,
    ) -> None:
        if self.lafan1_manifest_path is None:
            return

        _, manifest_entries = load_lafan1_manifest(self.lafan1_manifest_path)
        manifest_loader_options = load_lafan1_manifest_loader_options(
            self.lafan1_manifest_path
        )
        loader_chunk_size = self.lafan1_loader_chunk_size
        if loader_chunk_size is None:
            loader_chunk_size = manifest_loader_options.get("chunk_size")
        loader_shard_size = self.lafan1_loader_shard_size
        if loader_shard_size is None:
            loader_shard_size = manifest_loader_options.get("shard_size")
        self._sync_control_rate_to_manifest_entries(
            manifest_entries,
            timing_explicit=timing_explicit,
        )
        self.loader_type = "lafan1_csv"
        self.loader_kwargs = build_lafan1_loader_kwargs(
            entries=manifest_entries,
            sim_dt=float(self.sim.dt),
            decimation=int(self.decimation),
            joint_names=list(self.reference_joint_names),
            canonical_joint_names=list(self.target_joint_names),
        )

        if dataset_path_explicit and self.dataset_path is not None:
            self.dataset_path = str(Path(self.dataset_path).expanduser().resolve())
        else:
            self.dataset_path = dataset_path_from_entries(
                manifest_entries,
                manifest_path=self.lafan1_manifest_path,
                family=load_manifest_family(self.lafan1_manifest_path),
            )

        if motions_explicit and self.motions is not None:
            self.motions = list(self.motions)
        else:
            self.motions = [str(entry["name"]) for entry in manifest_entries]

        self._validate_lafan_source_entries(
            self.loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]
        )

    def _critic_prunable_command_term_names(self) -> tuple[str, ...]:
        """Critic-group command terms that command pruning may drop.

        Empty on the vanilla surface: its critic has always carried every
        explicit command term regardless of the policy selection, and that
        contract stays untouched. Latent surfaces override this with the
        supplemental explicit terms their critic gained for explicit
        command mode.
        """
        return ()

    def _normalized_command_mode(self) -> str:
        mode = str(self.command_mode).strip().lower()
        if mode not in {"latent", "explicit"}:
            raise ValueError(
                f"Unsupported command_mode={self.command_mode!r}; expected "
                "'latent' or 'explicit'."
            )
        return mode

    def _prune_command_observation_terms(self) -> None:
        """Select the active command-term set for ``command_mode``.

        - ``explicit``: prune ``latent_command`` (where the surface has one)
          and keep the policy command terms selected by
          ``command_observation_terms`` (all of them when None -- the
          historical vanilla default, so this cannot change an existing run).
          Command-side expert noise is disabled on the kept terms.
        - ``latent``: keep ``latent_command`` plus the selected explicit
          baseline terms (``expert_motion`` + anchors when None -- the
          historical latent default).
        """
        mode = self._normalized_command_mode()
        self.command_mode = mode
        policy = self.observations.policy
        critic = getattr(self.observations, "critic", None)
        if mode == "latent" and not hasattr(policy, "latent_command"):
            raise ValueError(
                "command_mode='latent' requires an observation surface with a "
                f"latent_command policy term; {type(self.observations).__name__} "
                "has none."
            )
        if isinstance(self.command_observation_terms, str):
            # Isaac Lab's strict updater passes a Hydra CLI override for this
            # None-default field through as the raw "[a,b,c]" string (same
            # gotcha as expert_macro_state_terms).
            self.command_observation_terms = [
                part
                for part in self.command_observation_terms.strip()
                .strip("[]")
                .split(",")
                if part.strip()
            ]
        if self.command_observation_terms is None:
            keep = (
                set(_PRUNABLE_COMMAND_TERM_NAMES)
                if mode == "explicit"
                else set(_LATENT_MODE_DEFAULT_COMMAND_TERM_NAMES)
            )
        else:
            keep = {str(name) for name in self.command_observation_terms}
            unknown = keep - set(_PRUNABLE_COMMAND_TERM_NAMES)
            if unknown:
                raise ValueError(
                    "command_observation_terms names terms that are not policy "
                    f"command terms: {sorted(unknown)}. Expected a subset of "
                    f"{sorted(_PRUNABLE_COMMAND_TERM_NAMES)}."
                )
            if not keep:
                raise ValueError(
                    "command_observation_terms is empty; the actor would receive "
                    "no command at all. Leave it None to keep every term."
                )
        for term_name in _PRUNABLE_COMMAND_TERM_NAMES:
            if term_name not in keep and hasattr(policy, term_name):
                setattr(policy, term_name, None)
        if critic is not None:
            for term_name in self._critic_prunable_command_term_names():
                if term_name not in keep and hasattr(critic, term_name):
                    setattr(critic, term_name, None)
        if mode == "explicit":
            for group in (policy, critic):
                if (
                    group is not None
                    and getattr(group, "latent_command", None) is not None
                ):
                    group.latent_command = None
            # Command-side expert noise stays disabled on explicit trackers
            # (frozen protocol); the vanilla terms declare none, so this only
            # affects latent surfaces switched to explicit commands.
            for term_name in keep:
                term = getattr(policy, term_name, None)
                if term is not None:
                    term.noise = None

    def _effective_expert_macro_state_terms(self) -> tuple[str, ...]:
        """The expert_window terms the DiffSR macro state currently selects."""
        terms = self.expert_macro_state_terms
        if terms is None:
            return _DEFAULT_EXPERT_MACRO_STATE_TERMS
        if isinstance(terms, str):
            # Raw "[a,b,c]" Hydra CLI form (same gotcha as
            # command_observation_terms); parse without mutating the field.
            terms = [
                part for part in terms.strip().strip("[]").split(",") if part.strip()
            ]
        return tuple(str(name) for name in terms)

    def _prune_expert_window_observation_terms(self) -> None:
        """Apply the ``expert_window_observation_terms`` whitelist.

        None (the default) keeps every window term -- the historical layout.
        A whitelist must cover the active macro-state terms, because the env
        builds the skill/planner macro state from this group; anything else a
        specific run reads from the window (e.g. EE-chunk planner packets) is
        the caller's responsibility to keep.
        """
        window = getattr(self.observations, "expert_window", None)
        if window is None:
            return
        if isinstance(self.expert_window_observation_terms, str):
            self.expert_window_observation_terms = [
                part
                for part in self.expert_window_observation_terms.strip()
                .strip("[]")
                .split(",")
                if part.strip()
            ]
        if self.expert_window_observation_terms is None:
            return
        keep = {str(name) for name in self.expert_window_observation_terms}
        unknown = keep - set(_EXPERT_WINDOW_TERM_NAMES)
        if unknown:
            raise ValueError(
                "expert_window_observation_terms names terms that are not "
                f"expert_window terms: {sorted(unknown)}. Expected a subset "
                f"of {sorted(_EXPERT_WINDOW_TERM_NAMES)}."
            )
        missing_macro = set(self._effective_expert_macro_state_terms()) - keep
        if missing_macro:
            raise ValueError(
                "expert_window_observation_terms must retain the active "
                f"expert_macro_state_terms; missing {sorted(missing_macro)}."
            )
        for term_name in _EXPERT_WINDOW_TERM_NAMES:
            if term_name not in keep and hasattr(window, term_name):
                setattr(window, term_name, None)

    def _apply_expert_goal_observation_toggle(self) -> None:
        """Drop the expert_goal group when ``enable_expert_goal_observations``
        is False; inert on surfaces without the group."""
        if bool(self.enable_expert_goal_observations):
            return
        if getattr(self.observations, "expert_goal", None) is not None:
            self.observations.expert_goal = None

    def _apply_reward_input_observation_toggle(self) -> None:
        """Drop the reward_input group when ``enable_reward_input_observations``
        is False; inert on surfaces without the group."""
        if bool(self.enable_reward_input_observations):
            return
        if getattr(self.observations, "reward_input", None) is not None:
            self.observations.reward_input = None

    def _refresh_command_observation_terms(self) -> None:
        """Re-derive every pruned observation selection from the field values.

        Isaac Lab 3.0's ``register_task`` applies plain ``env.*`` CLI
        overrides with a direct ``setattr`` on the config and only routes
        through ``from_dict`` when non-dotted Hydra args remain, so
        ``command_mode`` / ``command_observation_terms`` /
        ``expert_window_observation_terms`` / ``enable_expert_goal_observations``
        / ``enable_reward_input_observations`` overrides can arrive after
        ``__post_init__`` already pruned with the class defaults.
        ``ImitationRLEnv`` calls this at construction (and ``from_dict`` calls
        it after applying a dict) so the final field values always win.
        Idempotent: restore + re-anchor + re-sync + prune is a fixed point.
        """
        self._restore_pruned_command_observation_terms()
        self._restore_pruned_expert_window_and_goal_observations()
        self._set_anchor_body(self.expert_anchor_body_name)
        self._sync_expert_window_observation_params()
        sync_goal_params = getattr(self, "_sync_expert_goal_observation_params", None)
        if callable(sync_goal_params):
            sync_goal_params()
        self._prune_command_observation_terms()
        self._prune_expert_window_observation_terms()
        self._apply_expert_goal_observation_toggle()
        self._apply_reward_input_observation_toggle()

    def _restore_pruned_expert_window_and_goal_observations(self) -> None:
        """Re-instate pruned expert_window terms and the expert_goal /
        reward_input groups.

        Restored terms/groups carry declaration-time params, so callers must
        re-apply ``_set_anchor_body`` and the window/goal param syncs before
        pruning again (see ``_refresh_command_observation_terms``).
        """
        window = getattr(self.observations, "expert_window", None)
        if window is not None:
            defaults = None
            for term_name in _EXPERT_WINDOW_TERM_NAMES:
                if not hasattr(window, term_name):
                    continue
                if getattr(window, term_name) is not None:
                    continue
                if defaults is None:
                    defaults = type(window)()
                default_term = getattr(defaults, term_name, None)
                if default_term is not None:
                    setattr(window, term_name, default_term)
        if (
            hasattr(self.observations, "expert_goal")
            and self.observations.expert_goal is None
        ):
            observation_defaults = type(self.observations)()
            self.observations.expert_goal = observation_defaults.expert_goal
        if (
            hasattr(self.observations, "reward_input")
            and self.observations.reward_input is None
        ):
            observation_defaults = type(self.observations)()
            self.observations.reward_input = observation_defaults.reward_input

    def _restore_pruned_command_observation_terms(self) -> None:
        """Re-instate pruned command terms from the group-class declarations.

        ``from_dict`` may change ``command_mode`` or
        ``command_observation_terms`` after ``__post_init__`` already pruned
        with the class defaults, and pruning is destructive. Restored terms
        carry declaration-time params, so callers must re-apply
        ``_set_anchor_body`` before pruning again.
        """
        for group_name in ("policy", "critic"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            defaults = None
            for term_name in _PRUNABLE_COMMAND_TERM_NAMES + ("latent_command",):
                if not hasattr(group, term_name):
                    continue
                if getattr(group, term_name) is not None:
                    continue
                if defaults is None:
                    defaults = type(group)()
                default_term = getattr(defaults, term_name, None)
                if default_term is not None:
                    setattr(group, term_name, default_term)


def _apply_pelvis_protocol(
    cfg: ImitationG1BaseTrackingEnvCfg,
    *,
    reset_step_min: int = 0,
    reset_step_max: int = 200,
    random_reset_full_trajectory: bool = False,
    failure_rate_max_over_mean: float | None = None,
    set_anchor: bool = True,
) -> None:
    """Apply the shared strict/pelvis protocol block to a G1 tracking cfg.

    One authority for the copy-pasted protocol deltas: reset-start bounds,
    full-trajectory reset toggle, optional adaptive-failure
    ``failure_rate_max_over_mean``, and (unless ``set_anchor=False`` because a
    parent class already anchored the surface) the pelvis expert anchor plus
    the observation/reward re-anchoring that must follow it.
    """
    cfg.random_reset_step_min = reset_step_min
    cfg.random_reset_step_max = reset_step_max
    cfg.random_reset_full_trajectory = random_reset_full_trajectory
    if failure_rate_max_over_mean is not None:
        cfg.adaptive_failure_reset_failure_rate_max_over_mean = (
            failure_rate_max_over_mean
        )
    if set_anchor:
        cfg.expert_anchor_body_name = "pelvis"
        cfg._set_anchor_body("pelvis")
        cfg._set_reward_anchor_body("pelvis")


def _apply_strict_recipe(cfg: ImitationG1BaseTrackingEnvCfg) -> None:
    """The Strict recipe, shared by its explicit and latent command pins.

    Strict SONIC termination functions on the legacy scaffolding: pelvis
    anchor, [0, 200] reset starts, no full-trajectory resets, no curriculum.
    The pin classes (``ImitationG1StrictTrackEnvCfg`` and
    ``ImitationG1LatentStrictEnvCfg`` in ``variants/strict.py``) contribute
    only the observation surface and command configuration; the recipe itself
    lives here.
    """
    _apply_pelvis_protocol(cfg)


def _g1_lafan_track_env_cfg_from_dict(
    self: ImitationG1BaseTrackingEnvCfg, data: dict
) -> None:
    dataset_path_explicit = isinstance(data, Mapping) and "dataset_path" in data
    motions_explicit = isinstance(data, Mapping) and "motions" in data
    timing_explicit = isinstance(data, Mapping) and (
        "sim" in data or "decimation" in data
    )

    if isinstance(data, Mapping):
        data = self._apply_optional_hydra_overrides(data)

    ImitationG1BaseTrackingEnvCfg.from_dict(self, data)
    self._sync_expert_window_observation_params()
    sync_goal_params = getattr(self, "_sync_expert_goal_observation_params", None)
    if callable(sync_goal_params):
        sync_goal_params()

    self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
    self._normalize_sequence_overrides()
    self._validate_reset_schedule()
    self._resolve_manifest_config(
        dataset_path_explicit=dataset_path_explicit,
        motions_explicit=motions_explicit,
        timing_explicit=timing_explicit,
    )
    # Hydra-set `command_mode` / `command_observation_terms` must apply on this
    # path too, not only in `__post_init__`, or the override silently no-ops.
    # `__post_init__` already pruned with the class defaults and pruning is
    # destructive, so restore the declared terms and re-anchor them to the
    # surface's expert anchor before pruning with the overridden fields.
    self._refresh_command_observation_terms()


def _bind_lafan_track_from_dict(*cfg_classes: type) -> None:
    """Rebind ``from_dict`` so Hydra overrides go through the LAFAN1 resolver.

    Call this on every concrete (registered) env cfg class in the module that
    defines it. Never bind ``ImitationG1BaseTrackingEnvCfg`` itself: the
    resolver calls ``ImitationG1BaseTrackingEnvCfg.from_dict`` for the plain
    field update, so binding the base would recurse.
    """
    for cfg_class in cfg_classes:
        cfg_class.from_dict = _g1_lafan_track_env_cfg_from_dict
