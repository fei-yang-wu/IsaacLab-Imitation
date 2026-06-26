# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
from collections.abc import Mapping
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_imitation.assets.robots.unitree import (
    UNITREE_G1_29DOF_SDK_JOINT_NAMES,
    UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
    UNITREE_G1_29DOF_MIMIC_CFG,
)

from ... import mdp
from ...imitation_env_cfg import ImitationLearningEnvCfg
from ...lafan1_manifest import (
    build_lafan1_loader_kwargs,
    dataset_path_from_entries,
    infer_npz_manifest_control_freq,
    load_lafan1_manifest,
)


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

G1_29DOF_JOINT_NAMES: list[str] = list(UNITREE_G1_29DOF_SDK_JOINT_NAMES)

# Body tracking set aligned with the original Unitree G1 mimic tracking config.
G1_TRACKED_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_EE_BODY_NAMES: list[str] = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

G1_OBS_ANCHOR_BODY_NAME = "torso_link"


def _g1_tracked_body_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        body_names=G1_TRACKED_BODY_NAMES,
        preserve_order=True,
    )


def _g1_tracked_body_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": _g1_tracked_body_asset_cfg(),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
    }


def _g1_expert_motion_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_JOINT_NAMES,
        )
    }


def _g1_expert_anchor_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
    }


def _g1_expert_window_motion_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=G1_29DOF_JOINT_NAMES,
        ),
        "past_steps": 0,
        "future_steps": 0,
    }


def _g1_expert_window_anchor_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
        "past_steps": 0,
        "future_steps": 0,
    }


def _g1_expert_window_ee_obs_params() -> dict[str, object]:
    return {
        "asset_cfg": SceneEntityCfg("robot"),
        "reference_body_names": tuple(G1_EE_BODY_NAMES),
        "anchor_body_name": G1_OBS_ANCHOR_BODY_NAME,
        "past_steps": 0,
        "future_steps": 0,
    }


@configclass
class G1ActionsCfg:
    """Action settings for 29-DoF mimic G1."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class G1ObservationCfg:
    """Observation settings aligned with the 29-DoF tracking environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        expert_motion = ObsTerm(
            func=mdp.expert_motion_command,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        expert_motion = ObsTerm(
            func=mdp.expert_motion_command,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params=_g1_tracked_body_obs_params(),
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params=_g1_tracked_body_obs_params(),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class ExpertStateCfg(ObsGroup):
        """Single-frame expert observations exposed through the observation manager."""

        joint_pos = ObsTerm(
            func=mdp.expert_joint_pos,
            params=_g1_expert_motion_obs_params(),
        )
        joint_vel = ObsTerm(
            func=mdp.expert_joint_vel,
            params=_g1_expert_motion_obs_params(),
        )
        root_pos = ObsTerm(func=mdp.expert_root_pos)
        root_quat = ObsTerm(func=mdp.expert_root_quat)
        root_lin_vel = ObsTerm(func=mdp.expert_root_lin_vel)
        root_ang_vel = ObsTerm(func=mdp.expert_root_ang_vel)
        expert_motion = ObsTerm(
            func=mdp.expert_motion_command,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class ExpertWindowCfg(ObsGroup):
        """Temporal expert observations exposed through the observation manager."""

        expert_motion = ObsTerm(
            func=mdp.expert_window_motion,
            params=_g1_expert_window_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_window_anchor_pos_b,
            params=_g1_expert_window_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_window_anchor_ori_b,
            params=_g1_expert_window_anchor_obs_params(),
        )
        expert_ee_pos_b = ObsTerm(
            func=mdp.expert_window_ee_pos_b,
            params=_g1_expert_window_ee_obs_params(),
        )
        expert_ee_ori_b = ObsTerm(
            func=mdp.expert_window_ee_ori_b,
            params=_g1_expert_window_ee_obs_params(),
        )

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class RewardInputCfg(ObsGroup):
        """Inputs consumed by discriminator / reward estimator networks.

        On rollout, terms are computed from the robot's actual state; on the
        expert minibatch the env's expert-observation mapper returns the
        idealized-expert counterpart (reference motion, zero anchor error).
        """

        expert_motion = ObsTerm(
            func=mdp.robot_motion,
            params=_g1_expert_motion_obs_params(),
        )
        expert_anchor_pos_b = ObsTerm(
            func=mdp.expert_anchor_pos_b,
            params=_g1_expert_anchor_obs_params(),
        )
        expert_anchor_ori_b = ObsTerm(
            func=mdp.expert_anchor_ori_b,
            params=_g1_expert_anchor_obs_params(),
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    expert_state: ExpertStateCfg = ExpertStateCfg()
    expert_window: ExpertWindowCfg = ExpertWindowCfg()
    reward_input: RewardInputCfg = RewardInputCfg()


@configclass
class G1EventCfg:
    """Event settings aligned with the 29-DoF tracking environment."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    reset_reference_state = EventTerm(
        func=mdp.reset_root_and_joints_to_reference_with_randomization,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": VELOCITY_RANGE,
            "joint_position_range": (-0.1, 0.1),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class G1RewardsCfg:
    """Reward terms aligned to the 29-DoF tracking environment."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.reference_global_anchor_position_error_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "std": 0.3,
        },
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.reference_global_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "std": 0.4,
        },
    )
    motion_body_pos = RewTerm(
        func=mdp.reference_relative_body_position_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.3,
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.reference_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "anchor_body_name": "torso_link",
            "std": 0.4,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.reference_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 1.0,
        },
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.reference_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_TRACKED_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_TRACKED_BODY_NAMES,
            "std": 3.14,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    (
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    )
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class G1TerminationsCfg:
    """Termination terms aligned to the 29-DoF tracking environment."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reference_finished = DoneTerm(func=mdp.reference_trajectory_finished)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.25,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.8,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_reference_body_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_EE_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_EE_BODY_NAMES,
            "threshold": 0.25,
        },
    )
    # body too low
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.4,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )


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
    enable_latent_command: bool = False
    latent_command_dim: int = 64
    latent_patch_past_steps: int = 0
    latent_patch_future_steps: int = 0
    random_reset_step_min: int = 0
    random_reset_step_max: int = 0
    random_reset_full_trajectory: bool = False
    adaptive_failure_reset_uniform_ratio: float = 0.1
    adaptive_failure_reset_alpha: float = 0.001

    _debug_rewards: bool = False

    # Master switch for all expensive visualizers/marker debug rendering.
    # Keep disabled by default for training/runtime performance.
    enable_visualizers: bool = False
    visualize_reference_arrows: bool = True
    print_reference_velocity: bool = False
    print_reference_velocity_every: int = 50

    reference_joint_names: list[str] = G1_29DOF_JOINT_NAMES.copy()
    target_joint_names: list[str] = G1_29DOF_JOINT_NAMES.copy()
    command_ee_body_names: list[str] = G1_EE_BODY_NAMES.copy()
    command_observation_source: str = "reference"

    def _sync_expert_window_observation_params(self) -> None:
        past_steps = int(self.latent_patch_past_steps)
        future_steps = int(self.latent_patch_future_steps)
        for term in (
            self.observations.expert_window.expert_motion,
            self.observations.expert_window.expert_anchor_pos_b,
            self.observations.expert_window.expert_anchor_ori_b,
        ):
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
        for term in (
            self.observations.expert_window.expert_ee_pos_b,
            self.observations.expert_window.expert_ee_ori_b,
        ):
            term.params["past_steps"] = past_steps
            term.params["future_steps"] = future_steps
            term.params["reference_body_names"] = tuple(self.command_ee_body_names)

    def __post_init__(self) -> None:
        super().__post_init__()  # type: ignore

        self.scene.robot = UNITREE_G1_29DOF_MIMIC_CFG.replace(  # type: ignore
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            self.scene.contact_forces.force_threshold = 10.0
            self.scene.contact_forces.debug_vis = bool(self.enable_visualizers)

        # Reference marker visualizers are also gated by the master toggle.
        self.visualize_reference_arrows = bool(
            self.enable_visualizers and self.visualize_reference_arrows
        )

        self.scene.height_scanner = None

        if int(self.latent_patch_past_steps) < 0:
            raise ValueError("latent_patch_past_steps must be >= 0.")
        if int(self.latent_patch_future_steps) < 0:
            raise ValueError("latent_patch_future_steps must be >= 0.")
        if int(self.random_reset_step_min) < 0:
            raise ValueError("random_reset_step_min must be >= 0.")
        if int(self.random_reset_step_max) < int(self.random_reset_step_min):
            raise ValueError("random_reset_step_max must be >= random_reset_step_min.")
        if float(self.adaptive_failure_reset_uniform_ratio) < 0.0:
            raise ValueError("adaptive_failure_reset_uniform_ratio must be >= 0.")
        if not 0.0 <= float(self.adaptive_failure_reset_alpha) <= 1.0:
            raise ValueError("adaptive_failure_reset_alpha must be in [0, 1].")

        self._sync_expert_window_observation_params()


@configclass
class ImitationG1LafanTrackEnvCfg(ImitationG1BaseTrackingEnvCfg):
    """General 29-DoF motion-tracking env driven by a LAFAN1 manifest."""

    dataset_path: str | None = "data/lafan1/g1/"
    loader_type: str = "lafan1_csv"
    loader_kwargs: dict = {
        "dataset_name": "lafan1",
        "dataset": {"trajectories": {"lafan1_csv": []}},
        "control_freq": 50.0,
        "sim": {"dt": 0.005},
        "decimation": 4,
        "joint_names": G1_29DOF_JOINT_NAMES,
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
    reconstructed_reference_action: bool = True
    reconstructed_reference_action_mode = "next_pose"
    random_reset_full_trajectory: bool = True

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
            self.scene.contact_forces.update_period = self.sim.dt

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
        )

        if dataset_path_explicit and self.dataset_path is not None:
            self.dataset_path = str(Path(self.dataset_path).expanduser().resolve())
        else:
            self.dataset_path = dataset_path_from_entries(
                manifest_entries,
                manifest_path=self.lafan1_manifest_path,
            )

        if motions_explicit and self.motions is not None:
            self.motions = list(self.motions)
        else:
            self.motions = [str(entry["name"]) for entry in manifest_entries]

        self._validate_lafan_source_entries(
            self.loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]
        )

    def __post_init__(self) -> None:
        super().__post_init__()

        self.loader_kwargs = copy.deepcopy(self.loader_kwargs)
        self._normalize_sequence_overrides()
        self._validate_reset_schedule()
        self._resolve_manifest_config()


# Backward-compatible aliases.
ImitationG1EnvCfg = ImitationG1LafanTrackEnvCfg


def _g1_lafan_track_env_cfg_from_dict(
    self: ImitationG1LafanTrackEnvCfg, data: dict
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


ImitationG1LafanTrackEnvCfg.from_dict = _g1_lafan_track_env_cfg_from_dict
