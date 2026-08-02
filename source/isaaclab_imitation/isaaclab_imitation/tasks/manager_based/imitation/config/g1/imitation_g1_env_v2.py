# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v2``: ONE configurable G1 tracking environment.

Single env class (``ImitationG1V2EnvCfg``) that superseded the lean/full
split on 2026-08-02. Every command type is DECLARED -- the latent
``command`` term (serves the agent-published latent command, owns the
reset-start samplers and the tracking metrics), the explicit ``motion``
term (the 67-D explicit command behind ``command_manager.get_command("motion")``
and the explicit-interface rows), and the held ``chunk`` term (streams a
published packet slot-by-slot) -- and what is actually BUILT is selected by
configuration at construction:

- ``command_mode="latent"`` (default): the ``command`` term only.
- ``command_mode="explicit"``: the ``motion`` term (plus ``chunk`` under a
  ``*_chunk_current_slot`` ``policy_command_mode``).

The observation surface is also single: policy (latent command + the full
windowed explicit command superset + proprio) and critic (latent command +
single-frame expert command + privileged state). The historical
``expert_window`` group is GONE: windowed command data IS the explicit
motion command, so the window params (``latent_patch_past_steps`` /
``latent_patch_future_steps``) live on the policy command terms and the
encoder families consume ``("policy", ...)`` directly. With 0/0 the terms
are exactly the single-frame commands the explicit actor reads.

The environment runs a single-compute step (one observation compute per
control step, at the returned next-reference frame), so v2 is deliberately
NOT bit-equivalent to the legacy env -- the discarded mid-step compute also
drew observation noise, so v2 has its own fresh stochastic stream. Rewards,
terminations, events, and actions are the same frozen SONIC recipe
components as v1.

Flat design: ``ImitationG1V2EnvCfg`` inherits ONLY the generic
``ImitationLearningEnvCfg`` base -- not the legacy
``ImitationG1BaseTrackingEnvCfg`` machinery (deprecated, v0/v1-only, in
``common/tracking_env.py``). Every field this surface needs is declared
here, and every derived adjustment happens in ONE deterministic step,
``resolve_late_overrides()``, which the env constructor calls after all
Hydra / plain-setattr overrides have landed (dataset preparation, command
mode / whitelist derivation, group toggles, anchor + window syncs,
command-term build and width lockstep). No restore machinery, no from_dict
monkey-patching, no defensive fallbacks: construction fails loudly on a
missing manifest or an incoherent command selection.
"""

import copy
import pathlib
from collections.abc import Mapping

from isaaclab.utils.configclass import configclass

from ...imitation_env_cfg import ImitationLearningEnvCfg
from ...motion_manifest import (
    build_lafan1_loader_kwargs,
    dataset_path_from_entries,
    infer_npz_manifest_control_freq,
    load_lafan1_manifest,
    load_manifest_family,
)
from ...mdp.commands import (
    HeldChunkCommandCfg,
    MotionCommandCfg,
    ReferenceCommandCfg,
)
from .common.actions import G1SonicActionsCfg
from .common.constants import (
    G1_29DOF_DATASET_BODY_NAMES,
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from .common.events import G1SonicEventCfg
from .common.observations import (
    G1V2ObservationCfg,
    _LATENT_MODE_DEFAULT_COMMAND_TERM_NAMES,
    _PRUNABLE_COMMAND_TERM_NAMES,
    _V2_ANCHOR_TERM_NAMES_BY_GROUP,
)
from .common.presets import (
    G1ImitationContactSensorCfg,
    G1ImitationPhysicsCfg,
    G1SonicRobotCfg,
    _set_contact_sensor_update_period,
)
from .common.rewards import G1SonicRewardsCfg
from .common.terminations import G1SonicTerminationsCfg

# The `*_chunk_current_slot` policy command modes (the same set the base cfg's
# `__post_init__` validation accepts next to "reference"): only under one of
# these does the env stream the actor's command from a held packet, so only
# then does the `chunk` command term exist.
_POLICY_CHUNK_COMMAND_MODES = (
    "explicit_chunk_current_slot",
    "full_body_chunk_current_slot",
    "ee_chunk_current_slot",
)


@configclass
class G1V2CommandsCfg:
    """All v2 command terms declared; the env builds them on demand.

    Exactly one term owns the reset-start samplers and the tracking metrics
    (``owns_reset=True``): ``command`` in latent mode, ``motion`` in explicit
    mode. ``chunk`` exists only under a ``*_chunk_current_slot``
    ``policy_command_mode`` (the CommandManager skips None entries).
    """

    command: ReferenceCommandCfg | None = ReferenceCommandCfg(
        anchor_body_name="pelvis",
        mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
        # The term owns reference reset-start sampling (same sampler semantics
        # and cfg knobs as the explicit `motion` term and v0/v1's env-inline
        # path).
        owns_reset=True,
    )

    motion: MotionCommandCfg | None = MotionCommandCfg(
        anchor_body_name="pelvis",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
        mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
        owns_reset=True,
    )

    # Held explicit-chunk term. None unless `policy_command_mode` is a
    # `*_chunk_current_slot` adapter (see `_sync_chunk_command_cfg`).
    chunk: HeldChunkCommandCfg | None = None


@configclass
class ImitationG1V2EnvCfg(ImitationLearningEnvCfg):
    """The single configurable v2 G1 tracking environment.

    One env for every command surface: the latent default (policy + critic
    groups, one ``command`` term), the explicit interfaces (FB / root+qpos /
    EE / keypoint via ``command_mode=explicit`` +
    ``command_observation_terms``), the chunk/packet interfaces
    (``policy_command_mode=*_chunk_current_slot``), and the encoder surfaces
    (windowed policy command terms read by the vqvae / cvae / per-step-vq /
    sonic agents). What is built follows the knobs; nothing unused is
    instantiated.

    Inherits only the generic ``ImitationLearningEnvCfg``; every G1-specific
    field, validation, and derivation lives in this file. Construction-time
    overrides (Hydra ``env.*`` CLI args or plain ``setattr``) are applied by
    :meth:`resolve_late_overrides`, the single resolution step the env
    constructor calls.
    """

    # -- components (shared SONIC blocks from common) --
    actions = G1SonicActionsCfg()
    observations = G1V2ObservationCfg()  # type: ignore
    rewards = G1SonicRewardsCfg()  # type: ignore
    terminations = G1SonicTerminationsCfg()  # type: ignore
    events = G1SonicEventCfg()
    curriculum = None

    # pyrefly: ignore[bad-override-mutable-attribute]  # configclass override idiom
    commands: G1V2CommandsCfg = G1V2CommandsCfg()

    # -- command configuration --
    # Which command family feeds the actor: "explicit" prunes the
    # agent-published `latent_command` term and keeps the explicit command
    # terms selected by `command_observation_terms` (all of them when None);
    # "latent" keeps `latent_command` plus, by default, the historical
    # explicit baseline terms (expert_motion + anchors). Pair with the
    # matching agent config: `agent.ipmd.use_latent_command` must agree
    # (validated at training entry and by the command-matrix audit).
    command_mode: str = "latent"
    latent_command_dim: int = 258
    # Optional whitelist of policy-group COMMAND terms to keep (see
    # `_apply_command_mode`).
    command_observation_terms: list[str] | None = None
    command_observation_source: str = "reference"
    policy_command_mode: str = "reference"
    # Expert-window terms making up one DiffSR macro-state frame (consumed by
    # the env API `current_expert_macro_transition_batch`, not an obs group).
    expert_macro_state_terms: list[str] | None = None
    # Master switch for the reward_input observation group (parked IPMD
    # reward-estimation stack).
    enable_reward_input_observations: bool = False

    # -- reset / start-frame configuration (parsed once by the env) --
    reference_start_frame: int = 0
    random_reset_step_min: int = 0
    random_reset_step_max: int = 200
    random_reset_full_trajectory: bool = False
    reset_start_mode: str = "auto"
    adaptive_failure_reset_bin_size: int = 50
    adaptive_failure_reset_sequence_length_agnostic: bool = True
    adaptive_failure_reset_init_num_failures: float = 1.0
    adaptive_failure_reset_uniform_ratio: float = 0.1
    adaptive_failure_reset_pre_failure_window: int = 200
    adaptive_failure_reset_failure_rate_max_over_mean: float = 50.0

    # -- latent patch / held-command windows --
    latent_patch_past_steps: int = 0
    latent_patch_future_steps: int = 0
    command_hold_steps: int = 0

    # Anchor used when constructing expert batches and high-level macro states.
    # The SONIC protocol anchors at the pelvis (v2 default, declared here
    # rather than derived).
    expert_anchor_body_name: str = "pelvis"

    # -- body / joint name tables --
    reference_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    target_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    reference_body_names: list[str] = G1_29DOF_DATASET_BODY_NAMES.copy()
    mpjpe_metric_body_names: list[str] = G1_TRACKED_BODY_NAMES.copy()
    command_ee_body_names: list[str] = G1_EE_BODY_NAMES.copy()
    command_keypoint_body_names: list[str] = G1_KEYPOINT5_BODY_NAMES.copy()

    # -- LAFAN1 dataset / manifest machinery --
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

    # -- video / visualizers --
    video_follow_robot: bool = False
    video_follow_env_index: int = 0
    video_follow_eye_offset: tuple[float, float, float] = (3.5, 3.5, 2.0)
    video_follow_lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_visualizers: bool = False
    visualize_reference_arrows: bool = True
    print_reference_velocity: bool = False
    print_reference_velocity_every: int = 50

    _debug_rewards: bool = False

    # ------------------------------------------------------------------
    # Anchor / pruning tables (this surface).
    # ------------------------------------------------------------------

    def _anchor_term_names_by_group(self) -> dict[str, tuple[str, ...]]:
        return _V2_ANCHOR_TERM_NAMES_BY_GROUP

    def _critic_prunable_command_term_names(self) -> tuple[str, ...]:
        # The supplemental explicit terms the latent critic gained for
        # explicit command mode; the latent critic contract terms
        # (expert_motion + anchors) are never pruned.
        return (
            "expert_motion_qpos",
            "expert_ee_pos_b",
            "expert_ee_ori_b",
            "expert_keypoint_pos_b",
            "expert_keypoint_ori_b",
        )

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

    def _sync_command_window_params(self) -> None:
        """Sync the policy command terms' window to ``latent_patch_*``.

        Every windowed term in the policy group must appear here: the
        observation manager evaluates the whole group each step, and an
        unsynced term keeps its declaration-time 0/0 (single frame) while the
        encoder surfaces expect the task's window. (Terms pruned to None by
        `command_observation_terms` are skipped.)
        """
        past_steps = int(self.latent_patch_past_steps)
        future_steps = int(self.latent_patch_future_steps)
        policy = self.observations.policy
        for term in (
            policy.expert_motion,
            policy.expert_motion_qpos,
            policy.expert_anchor_pos_b,
            policy.expert_anchor_ori_b,
        ):
            if term is None:
                continue
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
        for term in (
            policy.expert_ee_pos_b,
            policy.expert_ee_ori_b,
        ):
            if term is None:
                continue
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
            term.params["reference_body_names"] = tuple(self.command_ee_body_names)
        for term in (
            policy.expert_keypoint_pos_b,
            policy.expert_keypoint_ori_b,
        ):
            if term is None:
                continue
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
            term.params["reference_body_names"] = tuple(
                self.command_keypoint_body_names
            )

    # ------------------------------------------------------------------
    # Input normalization (Hydra / from_dict path).
    # ------------------------------------------------------------------

    def _normalize_optional_none_overrides(self, data: Mapping) -> dict:
        """Normalize optional None-default fields before the strict updater.

        ``configclass.from_dict`` rejects `None -> str` transitions (the
        strict type check in isaaclab.utils.dict), and the raw "[a,b,c]"
        list-form Hydra strings for None-default list fields likewise fail.
        This hook converts exactly those fields so every other override can
        take the standard strict path.
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

        if "command_observation_terms" in remaining:
            # Store both the None and the raw "[a,b,c]" string forms;
            # `_apply_command_mode` parses both.
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

        return remaining

    # ------------------------------------------------------------------
    # Dataset / manifest preparation (fail-fast; idempotent).
    # ------------------------------------------------------------------

    def _prepare_dataset_config(
        self,
        *,
        dataset_path_explicit: bool = False,
        motions_explicit: bool = False,
        timing_explicit: bool = False,
    ) -> None:
        """Normalize the dataset fields, then resolve the manifest when set.

        Runs at construction (post-init and again at env construction, where
        late plain-setattr overrides land). Without a manifest it only
        normalizes; with one it rebuilds ``loader_kwargs`` / ``dataset_path``
        / ``motions`` from the manifest entries and validates every source
        file exists and is an NPZ with body states.
        """
        self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
        if self.motions is not None:
            self.motions = list(self.motions)
        if self.trajectories is not None:
            self.trajectories = list(self.trajectories)
        self.reset_schedule = self.reset_schedule.strip().lower()
        if self.reset_schedule not in {"random", "sequential", "round_robin"}:
            raise ValueError(
                f"Unsupported reset_schedule='{self.reset_schedule}'. "
                "Allowed values: ['random', 'sequential', 'round_robin']."
            )
        if self.lafan1_manifest_path is None:
            return

        _, manifest_entries = load_lafan1_manifest(self.lafan1_manifest_path)

        if not timing_explicit and self.sync_control_rate_to_manifest:
            control_freq = infer_npz_manifest_control_freq(manifest_entries)
            if control_freq is not None:
                if control_freq <= 0.0:
                    raise ValueError("control_freq must be positive.")

                # Keep sim.dt / decimation an integer fit to the manifest's
                # control rate, preferring the current physics rate, then the
                # preferred manifest physics rate, then a 1:1 fallback.
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
                    timing = _integer_timing_for(
                        float(self.preferred_manifest_physics_fps)
                    )
                if timing is None:
                    timing = (1.0 / control_freq, 1)
                self.sim.dt, self.decimation = timing
                self.sim.render_interval = self.decimation
                if self.scene.contact_forces is not None:
                    _set_contact_sensor_update_period(
                        self.scene.contact_forces, self.sim.dt
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
            self.dataset_path = str(
                pathlib.Path(self.dataset_path).expanduser().resolve()
            )
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

        for source in self.loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]:
            source_path = pathlib.Path(str(source["path"])).expanduser().resolve()
            source["path"] = str(source_path)
            if not source_path.is_file():
                raise FileNotFoundError(
                    "LAFAN1 motion source is missing. "
                    f"Expected: {source_path}. "
                    "Set `lafan1_manifest_path` to a manifest that points at "
                    "repo-local NPZ motions."
                )
            if self.require_npz_body_states and source_path.suffix.lower() != ".npz":
                raise ValueError(
                    "This tracking env requires an npz source with body states "
                    "(body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w). "
                    f"Got: {source_path}. "
                    "Generate repo-local NPZ files before loading this manifest."
                )

    # ------------------------------------------------------------------
    # Command-mode / whitelist derivation (single deterministic pass).
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_whitelist(value: list[str] | str | None) -> list[str] | None:
        """Parse the optional whitelist field (both None-default forms)."""
        if value is None:
            return None
        if isinstance(value, str):
            return [
                part for part in value.strip().strip("[]").split(",") if part.strip()
            ]
        return [str(item) for item in value]

    def _apply_command_mode(self) -> None:
        """Apply the final ``command_mode`` + whitelist to the observation groups.

        Single deterministic pass over the declared surface, run once per env
        construction (see :meth:`resolve_late_overrides`) after every override
        has landed. There is no restore step: the v2 surface is declared
        complete and this pass only ever narrows it.

        - ``explicit``: prune ``latent_command`` (where the surface has one)
          and keep the policy command terms selected by
          ``command_observation_terms`` (all of them when None). Command-side
          expert noise is disabled on the kept terms.
        - ``latent``: keep ``latent_command`` plus the selected explicit
          baseline terms (``expert_motion`` + anchors when None -- the
          historical latent default).
        """
        mode = str(self.command_mode).strip().lower()
        if mode not in {"latent", "explicit"}:
            raise ValueError(
                f"Unsupported command_mode={self.command_mode!r}; expected "
                "'latent' or 'explicit'."
            )
        self.command_mode = mode
        policy = self.observations.policy
        critic = getattr(self.observations, "critic", None)
        if mode == "latent" and not hasattr(policy, "latent_command"):
            raise ValueError(
                "command_mode='latent' requires an observation surface with a "
                f"latent_command policy term; {type(self.observations).__name__} "
                "has none."
            )
        whitelist = self._parse_whitelist(self.command_observation_terms)
        self.command_observation_terms = whitelist
        if whitelist is None:
            keep = (
                set(_PRUNABLE_COMMAND_TERM_NAMES)
                if mode == "explicit"
                else set(_LATENT_MODE_DEFAULT_COMMAND_TERM_NAMES)
            )
        else:
            keep = set(whitelist)
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
            # (frozen protocol).
            for term_name in keep:
                term = getattr(policy, term_name, None)
                if term is not None:
                    term.noise = None

    # ------------------------------------------------------------------
    # Command-term build / sync.
    # ------------------------------------------------------------------

    def _build_command_terms(self) -> None:
        """Build the command-term set the current configuration uses.

        Exactly one term owns the reset-start samplers: ``command`` in latent
        mode, ``motion`` in explicit mode (chunk modes keep ``motion`` as the
        reset owner and add the held ``chunk`` term). The CommandManager
        skips None entries, so unused terms cost nothing.
        """
        if self.command_mode == "latent":
            self.commands.motion = None
            self.commands.chunk = None
        else:
            self.commands.command = None
            self._sync_chunk_command_cfg()

    def _sync_command_cfg(self) -> None:
        """Wire the latent ``command`` term's width and anchor from the env fields.

        The ReferenceCommand serves the env's ``_agent_latent_command`` buffer
        (width ``latent_command_dim``) and computes its metrics against the
        env's expert anchor; both must match or construction fails loudly.
        Idempotent. Inert in explicit mode (no ``command`` term).
        """
        command = getattr(self.commands, "command", None)
        if command is None:
            return
        command.latent_command_dim = int(self.latent_command_dim)
        command.anchor_body_name = self.expert_anchor_body_name

    def _sync_motion_command_cfg(self) -> None:
        """Wire the explicit ``motion`` term's anchor from the env fields."""
        motion = getattr(self.commands, "motion", None)
        if motion is None:
            return
        motion.anchor_body_name = self.expert_anchor_body_name

    def _sync_chunk_command_cfg(self) -> None:
        """Instantiate/prune the ``chunk`` term from ``policy_command_mode``.

        Only a ``*_chunk_current_slot`` mode streams the actor's command from
        the env's held window, so only then does the adapter term exist. When
        present, its knobs are wired in lockstep with the env fields the
        held-window machinery reads: ``hold_steps`` from
        ``command_hold_steps`` (HeldChunkCommand fails loudly on a mismatch),
        the pinned 29-DoF joint order of the chunk-mode
        ``policy_expert_motion_command`` observation term, and the expert
        anchor body. Idempotent.
        """
        mode = str(self.policy_command_mode).strip().lower().replace("-", "_")
        if mode not in _POLICY_CHUNK_COMMAND_MODES:
            self.commands.chunk = None
            return
        if self.commands.chunk is None:
            self.commands.chunk = HeldChunkCommandCfg()
        self.commands.chunk.hold_steps = int(self.command_hold_steps)
        self.commands.chunk.joint_names = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
        self.commands.chunk.anchor_body_name = self.expert_anchor_body_name

    # ------------------------------------------------------------------
    # The single construction-time resolution step.
    # ------------------------------------------------------------------

    def resolve_late_overrides(
        self,
        *,
        dataset_path_explicit: bool = False,
        motions_explicit: bool = False,
        timing_explicit: bool = False,
    ) -> None:
        """Resolve every override-dependent field; called by the env constructor.

        Isaac Lab 3.0's Hydra integration applies ``env.*`` CLI overrides with
        plain ``setattr`` on the config (and ``from_dict`` for non-dotted
        args), so manifest paths, command modes, whitelists, and window knobs
        can all arrive after ``__post_init__``. This is the ONE place those
        final field values are turned into the derived layout: dataset
        preparation, command-mode / whitelist derivation, group toggles,
        anchor re-pointing, window syncs, and the command-term build with
        width lockstep. Idempotent: every step is deterministic and only
        narrows or re-syncs, so calling it once per env construction is a
        fixed point. Fails loudly on a missing manifest source or an
        incoherent selection.
        """
        self._prepare_dataset_config(
            dataset_path_explicit=dataset_path_explicit,
            motions_explicit=motions_explicit,
            timing_explicit=timing_explicit,
        )
        self._apply_command_mode()
        # Drop the reward_input group when its toggle is off (parked default).
        if not self.enable_reward_input_observations:
            if getattr(self.observations, "reward_input", None) is not None:
                self.observations.reward_input = None
        self._set_anchor_body(self.expert_anchor_body_name)
        self._sync_command_window_params()
        self._build_command_terms()
        self._sync_command_cfg()
        self._sync_motion_command_cfg()

    def __post_init__(self):
        super().__post_init__()

        # Single-frame skill command over a live sliding reference window.
        self.latent_patch_past_steps = 0
        self.latent_patch_future_steps = 0

        # SONIC robot asset (actuator contract matching G1SonicActionsCfg).
        robot_preset = G1SonicRobotCfg()
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

        # -- validations (fail fast, no defensive fallbacks) --
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

        # The SONIC pelvis protocol + legacy reset distribution: starts in
        # [0, 200], no full-trajectory adaptive-failure sampling,
        # failure_rate_max_over_mean=50 (declared defaults; re-anchor the
        # declared observation surface and the reward terms to the pelvis).
        self._set_anchor_body(self.expert_anchor_body_name)
        for term_name in (
            "motion_global_anchor_pos",
            "motion_global_anchor_ori",
            "motion_body_pos",
            "motion_body_ori",
        ):
            getattr(self.rewards, term_name).params["anchor_body_name"] = (
                self.expert_anchor_body_name
            )

        # Dataset preparation at construction time so the cfg is
        # self-consistent as soon as it exists (the env re-runs it later to
        # catch plain-setattr overrides; idempotent).
        self._prepare_dataset_config()

        self._sync_command_window_params()
        # NOTE: the command-term build (`_build_command_terms`) is NOT run
        # here: post-init runs before any overrides land, and the build is
        # destructive (it None's the inactive terms). Every command term stays
        # declared until the env-construction resolution selects the active
        # set from the final field values.
        self._sync_command_cfg()
        self._sync_motion_command_cfg()


def _imitation_g1_env_cfg_from_dict(self: ImitationG1V2EnvCfg, data: dict) -> None:
    """``from_dict`` for the flat v2 config: normalize then strict-update.

    Bound in place of the configclass default on the concrete class below.
    Normalizes the optional None-default fields (`_normalize_optional_none_overrides`)
    so Hydra ``env.*`` CLI overrides can reach them through the strict
    ``from_dict`` path, then delegates the rest of the update to the base
    implementation. Derived layout (manifest, command-mode pruning, syncs) is
    intentionally NOT applied here -- ``resolve_late_overrides`` owns that,
    once, at env construction.
    """
    if isinstance(data, Mapping):
        data = self._normalize_optional_none_overrides(data)

    ImitationLearningEnvCfg.from_dict(self, data)


# Bind the normalization `from_dict` on the concrete (registered) class.
ImitationG1V2EnvCfg.from_dict = _imitation_g1_env_cfg_from_dict  # type: ignore[assignment]

# Back-compat aliases: configs recorded against the pre-merge names
# (`imitation_g1_env_v2:ImitationG1EnvCfg` / `ImitationG1EnvV2Cfg` /
# `ImitationG1FullSurfaceEnvCfg`) keep resolving.
ImitationG1EnvCfg = ImitationG1V2EnvCfg
ImitationG1EnvV2Cfg = ImitationG1V2EnvCfg
ImitationG1FullSurfaceEnvCfg = ImitationG1V2EnvCfg

__all__ = [
    "G1V2CommandsCfg",
    "ImitationG1EnvCfg",
    "ImitationG1EnvV2Cfg",
    "ImitationG1FullSurfaceEnvCfg",
    "ImitationG1V2EnvCfg",
]
