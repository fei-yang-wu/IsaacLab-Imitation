# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Isaac-Imitation-G1-v2``: ONE configurable G1 tracking environment.

Every field on :class:`ImitationG1V2EnvCfg` is a *selection*: something a class
default, a preset, a Hydra ``env.*`` override, or a job script chose. Nothing
on it is a derived value, and no method writes one back. Everything that
follows from a selection -- the dataset layout, the control decimation, which
observation terms survive, which command terms the manager builds, where the
anchor-relative terms point -- is computed by :meth:`resolve`, once, from the
final field values, and stored where it cannot be mistaken for an input.

That single rule is what keeps this file short:

- ``__post_init__`` **declares**. It assembles components and states the
  protocol's fixed choices. It performs no IO, resolves no manifest, prunes
  no observation group. It cannot: Isaac Lab applies ``env.*`` CLI overrides
  and preset selections *after* it runs, so anything derived here would be
  derived from values the user is about to replace.
- :meth:`resolve` **derives**, and is called once by the environment
  constructor. It is idempotent because it recomputes from inputs it never
  modifies -- not because each step is individually careful.
- Nothing needs to know whether a field was "explicitly set". ``None`` means
  derive it; anything else is the answer. There is no comparing a live value
  against a class default to reconstruct user intent.

The command surface is one declared :class:`~...command_interface.CommandInterfaceCfg`:
the always-present dataset-backed ``reference`` channel (selection, reset-start
sampling, tracking metrics, the privileged critic view) plus exactly one
``actor`` channel. Both observation groups DECLARE every command term and the
interface narrows and parameterizes them, so the declaration is the complete
surface and resolution only ever removes. Swapping a comparison row is one
launch-time token::

    env.command_interface.actor=explicit
    env.command_interface.actor=chunk
    env.command_interface.encoder=causal9
    env.command_interface.reference.selection=sonic

The environment runs a single-compute step (one observation compute per control
step, at the returned next-reference frame). Rewards, terminations, events, and
actions are the frozen SONIC recipe components.
"""

from isaaclab.utils.configclass import configclass

from ...command_interface import (
    ActorCommandPreset,
    CommandInterfaceCfg,
    EncoderViewPreset,
    ReferenceChannelCfg,
    ReferenceSelectionPreset,
)
from ...mdp.commands import ACTOR_TERM_NAME, REFERENCE_TERM_NAME
from ...motion_data import MotionDataCfg, ResolvedMotionData
from ..._resolve import (
    coerce_declared_types,
    resolve_remaining_presets,
    stamp_anchor_body,
)
from ...imitation_env_cfg import ImitationLearningEnvCfg
from .common.actions import G1SonicActionsCfg
from .common.constants import (
    G1_29DOF_DATASET_BODY_NAMES,
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from .common.events import G1SonicEventCfg
from .common.observations import G1V2ObservationCfg
from .common.presets import (
    G1ImitationContactSensorCfg,
    G1ImitationPhysicsCfg,
    G1SonicRobotCfg,
)
from .common.rewards import G1V2TunedRewardsCfg
from .common.terminations import (
    G1SonicTerminationCurriculumCfg,
    G1SonicTerminationsCfg,
)

_CURRICULUM_THRESHOLD_TERMS = (
    "anchor_pos_threshold",
    "anchor_ori_threshold",
    "ee_body_pos_threshold",
    "foot_pos_xyz_threshold",
)

# Dataset fields this surface replaced, and what to use instead. Some are still
# declared by the shared `ImitationLearningEnvCfg` base (the legacy v0/v1
# environments read them), and Isaac Lab's CLI applies `env.<name>=...` with a
# plain setattr, so a stale launcher would otherwise set a field nothing reads
# and train on the wrong data in silence. Refusing to run is the only honest
# response to an override that cannot take effect.
_RETIRED_DATA_FIELDS: dict[str, str] = {
    "lafan1_manifest_path": "data.manifest",
    "motions": "data.clips",
    "trajectories": "data.clips",
    "dataset_path": "data.cache_dir",
    "refresh_zarr_dataset": "data.cache_refresh",
    "require_npz_body_states": "data.require_body_states",
    "wrap_steps": "data.wrap_steps",
    "dataset_keys": "data.keys",
    "dataset_storage_device": "data.storage_device",
    "dataset_storage_persist_dir": "data.persist_dir",
    "dataset_storage_persist_id": "data.persist_id",
    "dataset_storage_persist_rebuild": "data.persist_rebuild",
    "loader_type": "data.manifest (the loader is derived from it)",
    "loader_kwargs": "data.manifest (loader arguments are derived from it)",
    "sync_control_rate_to_manifest": (
        "nothing: sim.dt x decimation fix the control rate and clips are "
        "checked against it"
    ),
    "preferred_manifest_physics_fps": (
        "nothing: the physics rate is declared by sim.dt, never retuned to fit the data"
    ),
    "lafan1_loader_chunk_size": "the manifest's metadata.loader_kwargs.chunk_size",
    "lafan1_loader_shard_size": "the manifest's metadata.loader_kwargs.shard_size",
}


@configclass
class G1V2CommandsCfg:
    """The two built command terms, derived from the declared interface.

    Both entries are the very objects on ``ImitationG1V2EnvCfg.command_interface``
    (assigned by :meth:`ImitationG1V2EnvCfg.resolve`), so there is one place a
    command is configured and no chance of the manager and the interface
    disagreeing.
    """

    reference = None
    actor = None


# The v2 default DiffSR macro-state frame: qpos + root pose, no joint velocity.
_ROOT_QPOS_MACRO_STATE_TERMS: list[str] = [
    "expert_motion_qpos",
    "expert_anchor_pos_b",
    "expert_anchor_ori_b",
]


@configclass
class ImitationG1V2EnvCfg(ImitationLearningEnvCfg):
    """The single configurable v2 G1 tracking environment.

    One env for every command surface, selected by ``command_interface.actor``:
    the latent default, the explicit interfaces (over any component set --
    full body / root+qpos / EE / keypoints), and the planner packet. The
    encoder window is a second, independent selection. What is built follows
    the interface; nothing unused is instantiated.
    """

    # -- components (shared SONIC blocks from common) --
    actions = G1SonicActionsCfg()
    observations = G1V2ObservationCfg()  # type: ignore
    rewards = G1V2TunedRewardsCfg()  # type: ignore
    terminations = G1SonicTerminationsCfg()  # type: ignore
    events = G1SonicEventCfg()
    curriculum = None

    # pyrefly: ignore[bad-override-mutable-attribute]  # configclass override idiom
    commands: G1V2CommandsCfg = G1V2CommandsCfg()

    # -- the declared command interface (the whole command surface) --
    # The reference channel is dataset-backed and always built; the actor
    # channel and the encoder window are what a comparison row selects.
    command_interface: CommandInterfaceCfg = CommandInterfaceCfg(
        reference=ReferenceChannelCfg(
            anchor_body_name="pelvis",
            joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES.copy(),
            mpjpe_body_names=G1_TRACKED_BODY_NAMES.copy(),
            ee_body_names=G1_EE_BODY_NAMES.copy(),
            keypoint_body_names=G1_KEYPOINT5_BODY_NAMES.copy(),
            selection=ReferenceSelectionPreset(),
        ),
        actor=ActorCommandPreset(),
        encoder=EncoderViewPreset(),
        critic_channels=("actor", "reference"),
    )

    # -- the motion data this environment tracks --
    data: MotionDataCfg = MotionDataCfg()

    # -- body / joint name tables --
    # The anchor body and the EE / keypoint body sets live only on the
    # reference channel: it is their single home, and the command terms, the
    # data plane, and the observation surface all read them from there.
    #
    # These four cannot follow, because the shared `ImitationLearningEnvCfg`
    # base declares them and the legacy environments read them off the config.
    # They stay INPUTS here; resolution copies them onto the reference channel,
    # which makes the channel's `joint_names` / `mpjpe_body_names` outputs.
    # Configure the joint order and the metric bodies here, not there.
    reference_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    target_joint_names: list[str] = G1_29DOF_ISAACLAB_JOINT_NAMES.copy()
    reference_body_names: list[str] = G1_29DOF_DATASET_BODY_NAMES.copy()
    mpjpe_metric_body_names: list[str] = G1_TRACKED_BODY_NAMES.copy()

    # Expert-window terms making up one DiffSR macro-state frame (consumed by
    # the env API `current_expert_macro_transition_batch`, not an obs group).
    #
    # ROOT_QPOS: 29 joint qpos + 3 root pos + 6 root ori = 38 per frame, so a
    # 380-wide encoder input over the 10-frame window. The previous default was
    # the full-body frame (`expert_motion`, 58 = qpos + qvel), which is 67 per
    # frame and 670 over the window. Joint velocity is the only difference.
    #
    # Measured at 500M on LAFAN1 with matched rewards (2026-08-04): strict
    # MPJPE-G landed inside the full-body arm's three-seed range, so dropping
    # qvel costs nothing in precision, while survival was the highest measured
    # (445.3, above every control and full-body seed) and full-horizon MPJPE-G
    # the best produced by any arm (0.1503 against full-body's 0.1740-0.2303).
    # Single seed on a pass with ~28% training-seed spread.
    #
    # THE ENCODER MUST MATCH. An encoder is built for one input width, so this
    # default requires a `root_qpos` encoder; a full-body (670) one fails at the
    # first forward with `hl/state shape mismatch: expected (N, 38), got
    # (N, 67)`. Loud, never silent -- but it does mean a v2 checkpoint trained
    # against the old default now needs
    # `env.expert_macro_state_terms=[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]`
    # to reproduce. See the `g1-encoder-interface` skill.
    expert_macro_state_terms: list[str] | None = _ROOT_QPOS_MACRO_STATE_TERMS.copy()

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

    # Anneal window in environment frames; None keeps each term's own default
    # (50M -> 500M). Env-level fields rather than `env.curriculum.<term>.*`
    # overrides because `curriculum` is None until resolution installs it, so a
    # CLI override would target a node that does not exist yet.
    termination_curriculum_start_frames: int | None = None
    termination_curriculum_end_frames: int | None = None

    # -- video / visualizers --
    video_follow_robot: bool = False
    video_follow_env_index: int = 0
    video_follow_eye_offset: tuple[float, float, float] = (3.5, 3.5, 2.0)
    video_follow_lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_visualizers: bool = False
    visualize_reference_arrows: bool = True
    print_reference_velocity: bool = False
    print_reference_velocity_every: int = 50

    replay_reference: bool = False
    replay_only: bool = False

    _debug_rewards: bool = False

    # -- derived, written only by `resolve` --
    resolved_data: ResolvedMotionData | None = None
    """The concrete dataset layout. An OUTPUT: never configure this."""

    # ------------------------------------------------------------------
    # The single construction-time resolution step.
    # ------------------------------------------------------------------

    def resolve(self) -> None:
        """Derive every override-dependent field; called by the env constructor.

        The one place final field values become the derived layout. Fails
        loudly on a missing manifest source or an incoherent command selection.
        """
        # Presets first: everything below reads the selected alternatives, and
        # a config built without Hydra still has its presets standing here.
        resolve_remaining_presets(self)
        coerce_declared_types(self)
        self._reject_retired_data_fields()

        self._apply_timing()
        self.resolved_data = self.data.resolve(
            sim_dt=float(self.sim.dt),
            decimation=int(self.decimation),
            joint_names=list(self.reference_joint_names),
            canonical_joint_names=list(self.target_joint_names),
        )

        # Drop the reward_input group when its toggle is off (parked default).
        if not self.enable_reward_input_observations:
            self.observations.reward_input = None

        if self.enable_termination_curriculum and self.curriculum is None:
            self.curriculum = self._termination_curriculum()

        # The two base-declared tables the reference channel mirrors (see the
        # field comment above): the config is the input, the channel the output.
        interface = self.command_interface
        interface.reference.joint_names = list(self.target_joint_names)
        interface.reference.mpjpe_body_names = list(self.mpjpe_metric_body_names)

        # The interface decides which declared command terms survive, which
        # channel each reads, and at what window; then the two built terms are
        # published to the manager's command group.
        interface.resolve()
        interface.apply_to_observations(self.observations)
        setattr(self.commands, REFERENCE_TERM_NAME, interface.reference)
        setattr(self.commands, ACTOR_TERM_NAME, interface.actor)

        # After the interface has narrowed the groups, so the walk sees exactly
        # the surviving anchor-relative terms.
        stamp_anchor_body(
            interface.reference.anchor_body_name, self.observations, self.rewards
        )

    def _reject_retired_data_fields(self) -> None:
        """Refuse to run with a dataset override this surface no longer reads."""
        stale = []
        for name, replacement in _RETIRED_DATA_FIELDS.items():
            value = getattr(self, name, None)
            if value is None or value == [] or value == {} or value is False:
                continue
            stale.append(f"  env.{name}={value!r}  ->  env.{replacement}")
        if stale:
            raise ValueError(
                "These dataset settings are not read by this environment:\n"
                + "\n".join(stale)
                + "\n\nThe motion data is configured under `env.data.*` "
                "(see MotionDataCfg). Update the command or launcher rather "
                "than running against data it did not select."
            )

    def _apply_timing(self) -> None:
        """Re-sync what follows from the declared physics step and decimation.

        ``sim.dt`` and ``decimation`` are themselves declarations -- overridable
        on the command line, but never moved by resolution -- so this only
        propagates them to the render interval and the sensor update period,
        which would otherwise keep post-init's values after an override.
        """
        self.sim.render_interval = int(self.decimation)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

    def _termination_curriculum(self) -> G1SonicTerminationCurriculumCfg:
        curriculum = G1SonicTerminationCurriculumCfg()
        window = {
            "start_frames": self.termination_curriculum_start_frames,
            "end_frames": self.termination_curriculum_end_frames,
        }
        for field_name, value in window.items():
            if value is None:
                continue
            for term_name in _CURRICULUM_THRESHOLD_TERMS:
                getattr(curriculum, term_name).params[field_name] = int(value)
        return curriculum

    # ------------------------------------------------------------------
    # Declaration only. No IO, no derivation, no pruning.
    # ------------------------------------------------------------------

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
        self.scene.height_scanner = None

        self.episode_length_s = 10.0
        # The task's timing: 200 Hz physics, 50 Hz control. This is a protocol
        # decision, not something derived from whatever data is loaded -- every
        # reward, termination threshold, episode length, and recorded result is
        # defined at 50 Hz, and the conversion pipeline resamples clips to it.
        # Resolution checks the clips against this rate and refuses a mismatch;
        # it never moves the rate to fit the data.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.physics_material = self.scene.terrain.physics_material
        # Isaac Lab 3.0: SimulationCfg.physx was replaced by the backend-selecting
        # SimulationCfg.physics field. The PresetCfg resolves to PhysX by default
        # and to Newton with the `physics=newton_mjwarp` CLI override.
        self.sim.physics = G1ImitationPhysicsCfg()

        if self.scene.contact_forces is not None:
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


__all__ = [
    "G1V2CommandsCfg",
    "ImitationG1V2EnvCfg",
]
