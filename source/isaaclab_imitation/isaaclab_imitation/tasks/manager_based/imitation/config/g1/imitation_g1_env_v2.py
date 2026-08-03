# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v2``: ONE configurable G1 tracking environment.

Single env class (:class:`ImitationG1V2EnvCfg`) whose command surface is one
declared :class:`~...command_interface.CommandInterfaceCfg`: the always-present
dataset-backed ``reference`` channel (selection, reset-start sampling, tracking
metrics, the privileged critic view) plus exactly one ``actor`` channel --
latent, explicit, or chunk. Swapping the comparison row is swapping
``command_interface.actor``; nothing else about the environment changes.

The observation surface is single: policy (the actor's command, plus a latent
recipe's windowed encoder view, plus proprio) and critic (its command view plus
privileged state). Both groups DECLARE every command term and the interface
narrows and parameterizes them at construction, so the declaration is the
complete surface and resolution only ever removes.

The environment runs a single-compute step (one observation compute per control
step, at the returned next-reference frame). Rewards, terminations, events, and
actions are the same frozen SONIC recipe components as v1.

Flat design: :class:`ImitationG1V2EnvCfg` inherits ONLY the generic
``ImitationLearningEnvCfg`` -- not the legacy ``ImitationG1BaseTrackingEnvCfg``
machinery (deprecated, v0/v1-only, in ``common/tracking_env.py``). Every field
this surface needs is declared here, and every derived adjustment happens in ONE
deterministic step, :meth:`ImitationG1V2EnvCfg.resolve_late_overrides`, which
the env constructor calls after all Hydra / plain-setattr overrides have landed.
No restore machinery, no defensive fallbacks: construction fails loudly on a
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
from ...command_interface import (
    CommandInterfaceCfg,
    EncoderViewCfg,
    LatentCommandCfg,
    ReferenceChannelCfg,
    ReferenceSelectionCfg,
)
from ...mdp.commands import ACTOR_TERM_NAME, REFERENCE_TERM_NAME
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
    _V2_ANCHOR_TERM_NAMES_BY_GROUP,
)
from .common.presets import (
    G1ImitationContactSensorCfg,
    G1ImitationPhysicsCfg,
    G1SonicRobotCfg,
    _set_contact_sensor_update_period,
)
from .common.rewards import G1SonicRewardsCfg
from .common.terminations import (
    G1SonicTerminationCurriculumCfg,
    G1SonicTerminationsCfg,
)


@configclass
class G1V2CommandsCfg:
    """The two built command terms, derived from the declared interface.

    Both entries are the very objects on ``ImitationG1V2EnvCfg.command_interface``
    (assigned by :meth:`ImitationG1V2EnvCfg.resolve_late_overrides`), so there is
    one place a command is configured and no chance of the manager and the
    interface disagreeing.
    """

    reference = None
    actor = None


@configclass
class ImitationG1V2EnvCfg(ImitationLearningEnvCfg):
    """The single configurable v2 G1 tracking environment.

    One env for every command surface, selected by ``command_interface.actor``:
    the latent default (:class:`LatentCommandCfg`), the explicit interfaces
    (:class:`~...command_interface.ExplicitCommandCfg` over any component set --
    full body / root+qpos / EE / keypoints), and the planner packet
    (:class:`~...command_interface.ChunkCommandCfg`). The encoder surfaces
    (vqvae / cvae / per-step-vq / sonic) set ``command_interface.encoder``'s
    window. What is built follows the interface; nothing unused is
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

    # -- the declared command interface (the whole command surface) --
    # Default: the latent recipe. The reference channel is dataset-backed and
    # always built; the actor channel is what a comparison row swaps.
    command_interface: CommandInterfaceCfg = CommandInterfaceCfg(
        reference=ReferenceChannelCfg(
            anchor_body_name="pelvis",
            joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
            mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
            ee_body_names=G1_EE_BODY_NAMES.copy(),
            keypoint_body_names=G1_KEYPOINT5_BODY_NAMES.copy(),
            # The SONIC pelvis protocol's legacy reset distribution.
            selection=ReferenceSelectionCfg(
                schedule="random",
                start_mode="auto",
                random_step_min=0,
                random_step_max=200,
                full_trajectory=False,
                adaptive_failure_rate_max_over_mean=50.0,
            ),
        ),
        actor=LatentCommandCfg(dim=258),
        # The skill encoder's windowed reference view; 0/0 is the single-frame
        # view the default recipe's posterior reads.
        encoder=EncoderViewCfg(past_steps=0, future_steps=0),
        critic_channels=("actor", "reference"),
    )

    # Expert-window terms making up one DiffSR macro-state frame (consumed by
    # the env API `current_expert_macro_transition_batch`, not an obs group).
    expert_macro_state_terms: list[str] | None = None
    # Master switch for the reward_input observation group (parked IPMD
    # reward-estimation stack).
    enable_reward_input_observations: bool = False

    # Master switch for the SONIC termination-threshold anneal, off by default
    # so `curriculum = None` stays the surface's declared behaviour.
    #
    # Off means the strict release thresholds apply from frame 0, which is the
    # 2026-07-21 decision: a moving threshold makes early curves incomparable
    # across runs. That reasoning is about measurement, and it is why this stays
    # opt-in rather than becoming the default. It is worth having a switch
    # because the anneal's own start values were measured as the fastest local
    # learner (episode length 25.9 against 14.6 for strict-from-scratch over
    # 50M, migration wiki 2026-07-19), and because the anneal completes at 500M
    # -- 10% of a 5B budget -- so a long run ends on bit-identical strict SONIC
    # thresholds either way.
    enable_termination_curriculum: bool = False

    # Anneal window for that curriculum, in environment frames. Exposed as env
    # fields rather than left to `env.curriculum.<term>.params.*` overrides,
    # which cannot work: `curriculum` is still None when Hydra applies overrides
    # and is only installed during `resolve_late_overrides`, so a CLI override
    # would target a non-existent node. None keeps each term's own default
    # (50M -> 500M).
    termination_curriculum_start_frames: int | None = None
    termination_curriculum_end_frames: int | None = None

    # Anchor used when constructing expert batches and high-level macro states,
    # and by the anchor-relative reward terms. Synced into the reference
    # channel at resolution so there is one anchor body per environment.
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
    # Command interface (the whole command surface, in one place).
    # ------------------------------------------------------------------

    def _sync_command_interface(self) -> None:
        """Resolve the declared interface and apply it to the surface.

        Syncs the environment-level fields that must agree with the reference
        channel (the anchor body and the MPJPE metric bodies are also read by
        the reward terms and the data plane, so the environment cfg stays their
        single authority and the channel follows), then resolves the interface,
        narrows the observation groups to the declared command terms, and
        publishes the two built terms to the manager's command group.
        """
        reference = self.command_interface.reference
        reference.anchor_body_name = self.expert_anchor_body_name
        reference.mpjpe_body_names = list(self.mpjpe_metric_body_names)
        reference.joint_names = list(self.target_joint_names)
        self.command_interface.resolve()
        self.command_interface.apply_to_observations(self.observations)
        setattr(self.commands, REFERENCE_TERM_NAME, reference)
        setattr(self.commands, ACTOR_TERM_NAME, self.command_interface.actor)

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
        args), so manifest paths and command-interface fields can all arrive
        after ``__post_init__``. This is the ONE place those final field values
        become the derived layout: dataset preparation, the command interface
        (validation, observation narrowing, command-term build), the group
        toggles, and the anchor re-pointing. Idempotent: every step is
        deterministic and only narrows or re-syncs, so calling it once per env
        construction is a fixed point. Fails loudly on a missing manifest
        source or an incoherent command selection.
        """
        self._prepare_dataset_config(
            dataset_path_explicit=dataset_path_explicit,
            motions_explicit=motions_explicit,
            timing_explicit=timing_explicit,
        )
        # Drop the reward_input group when its toggle is off (parked default).
        if not self.enable_reward_input_observations:
            if getattr(self.observations, "reward_input", None) is not None:
                self.observations.reward_input = None
        # Installed here, not in __post_init__: the toggle is an `env.*` CLI
        # override and so only has its final value by this point. Guarded on
        # `is None` so an explicitly assigned curriculum is never replaced.
        if self.enable_termination_curriculum and self.curriculum is None:
            self.curriculum = G1SonicTerminationCurriculumCfg()
            for window_field, value in (
                ("start_frames", self.termination_curriculum_start_frames),
                ("end_frames", self.termination_curriculum_end_frames),
            ):
                if value is None:
                    continue
                for term_name in (
                    "anchor_pos_threshold",
                    "anchor_ori_threshold",
                    "ee_body_pos_threshold",
                    "foot_pos_xyz_threshold",
                ):
                    getattr(self.curriculum, term_name).params[window_field] = int(
                        value
                    )
        self._set_anchor_body(self.expert_anchor_body_name)
        self._sync_command_interface()

    def __post_init__(self):
        super().__post_init__()

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

        # Re-anchor the declared observation surface and the reward terms to
        # the environment's anchor body (the SONIC protocol's pelvis).
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

        # NOTE: the command interface is NOT resolved here. Post-init runs
        # before any override lands, and resolution is destructive (it narrows
        # the observation groups to the declared command terms). The whole
        # surface stays declared until `resolve_late_overrides` selects from
        # the final field values.


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
