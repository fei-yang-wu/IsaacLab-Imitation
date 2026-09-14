"""Fixed-base arm-and-hand tracking with the shared Sharpa contact rewards."""

from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path

from iltools.core import load_dexterous_reference_set, sha256_file
from isaaclab import sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_imitation.assets.sharpa import finger_actuators
from ...sharpa_object_action import SharpaVirtualObjectActionCfg
from ...vega_sharpa_actions import VegaSharpaJointResidualCfg
from ...vega_sharpa_command import (
    VegaSharpaReferenceCommandCfg,
    arm_state,
    box_center_position_tracking_exp,
)
from ..sharpa.sharpa_env_cfg import (
    SharpaSceneCfg,
    SharpaPhysicsCfg,
    _object_cfg,
    _support_cfg,
)
from ..sharpa.sharpa_source_env_cfg import (
    SharpaSourceObservationsCfg,
    SharpaSourceRewardsCfg,
    SharpaSourceTerminationsCfg,
    SharpaSourceEventsCfg,
    SharpaSourceCurriculumCfg,
)


@configclass
class VegaSharpaSceneCfg(SharpaSceneCfg):
    right_robot = None
    left_robot = None
    robot: ArticulationCfg = MISSING


@configclass
class VegaSharpaCommandsCfg:
    sharpa_reference = VegaSharpaReferenceCommandCfg()


@configclass
class VegaSharpaActionsCfg:
    right_hand = VegaSharpaJointResidualCfg(asset_name="robot", side="right")
    left_hand = VegaSharpaJointResidualCfg(asset_name="robot", side="left")
    virtual_object = SharpaVirtualObjectActionCfg(asset_name="object")


@configclass
class VegaSharpaObservationsCfg(SharpaSourceObservationsCfg):
    @configclass
    class PolicyCfg(SharpaSourceObservationsCfg.PolicyCfg):
        arm_state = ObsTerm(func=arm_state)

    policy = PolicyCfg()


@configclass
class VegaSharpaReferenceDataCfg:
    manifest: str = os.environ.get("VEGA_SHARPA_REFERENCE_MANIFEST", "")


@configclass
class VegaSharpaRewardsCfg(SharpaSourceRewardsCfg):
    object_center_tracking_exp = RewTerm(
        func=box_center_position_tracking_exp,
        weight=0.0,
        params={"command_name": "sharpa_reference", "position_scale_m": 0.05},
    )


@configclass
class VegaSharpaEnvCfg(ManagerBasedRLEnvCfg):
    scene = VegaSharpaSceneCfg(
        # Newton worlds isolate contacts by world ID. A common origin keeps
        # world-fixed geometry identical in the batched MuJoCo model.
        num_envs=64,
        env_spacing=0.0,
        replicate_physics=True,
        filter_collisions=True,
    )
    commands = VegaSharpaCommandsCfg()
    actions = VegaSharpaActionsCfg()
    observations = VegaSharpaObservationsCfg()
    rewards = VegaSharpaRewardsCfg()
    terminations = SharpaSourceTerminationsCfg()
    events = SharpaSourceEventsCfg()
    curriculum = SharpaSourceCurriculumCfg()
    reference = VegaSharpaReferenceDataCfg()

    def __post_init__(self):
        if not self.reference.manifest:
            raise ValueError(
                "Set VEGA_SHARPA_REFERENCE_MANIFEST to the mounted ILTools manifest."
            )
        references = load_dexterous_reference_set(self.reference.manifest)
        if len(references) != 1:
            raise ValueError("The initial mounted task requires one Reference.")
        reference = references[0]
        if (
            reference.robot_name != "vega_u_sharpa"
            or reference.robot_layout != "fixed_base"
        ):
            raise ValueError("Expected a fixed-base Vega U / Sharpa Reference.")
        if (
            reference.fps != 20
            or reference.contacts is None
            or reference.scene_physics is None
        ):
            raise ValueError(
                "Mounted data requires 20 Hz poses, contact geometry, and explicit scene physics."
            )
        if not reference.support_surface_names:
            raise ValueError("The box-on-stand task requires support geometry.")
        asset = reference.metadata["mounted_robot_asset"]
        usd = Path(asset["usd_path"])
        if sha256_file(usd) != asset["usd_sha256"]:
            raise ValueError("Mounted robot USD differs from the Reference asset.")
        if sha256_file(asset["model_path"]) != asset["model_sha256"]:
            raise ValueError("Mounted robot MJCF differs from the converted asset.")
        if (
            reference.metadata["mounted_retarget"]["model_sha256"]
            != asset["model_sha256"]
        ):
            raise ValueError("Retarget and runtime assembly model hashes differ.")
        for layer, digest in asset["usd_layer_sha256"].items():
            if sha256_file(layer) != digest:
                raise ValueError(f"Mounted robot USD layer changed: {layer}")
        root = reference.fixed_root_pose_w
        from imitation_experiments.data.vega_sharpa_model import base_ground_alignment

        if (
            base_ground_alignment(Path(asset["model_path"]), root)[
                "base_ground_clearance_m"
            ]
            < -1e-5
        ):
            raise ValueError(
                "The Reference puts the robot base below the floor; rerun mounted preparation with automatic base height."
            )
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # Ideal gravity compensation, matching the collaborator's
                    # fixed-base arm controller. The object retains gravity.
                    # Newton's per-body gravity exclusion is not implemented
                    # by this common flag. The asset authors mjc:gravcomp=1.
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                    max_contact_impulse=1e3,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(float(v) for v in root[:3]),
                rot=tuple(float(v) for v in (*root[4:7], root[3])),
                joint_pos={
                    n: float(q)
                    for n, q in zip(
                        reference.joint_names, reference.qpos[0], strict=True
                    )
                },
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=1.0,
            actuators={
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["[LR]_arm_j[1-7]"],
                    stiffness=400.0,
                    damping=40.0,
                    effort_limit_sim=300.0,
                    velocity_limit_sim=2.4,
                    armature=0.01,
                ),
                "fingers": finger_actuators.copy(),
            },
        )
        self.scene.object = _object_cfg(
            reference.object_asset_paths[0], reference.scene_physics
        )
        for i, name in enumerate(reference.support_surface_names):
            support = _support_cfg(
                name,
                reference.support_surface_asset_paths[i],
                reference.support_surface_poses_w[i],
                reference.support_surface_scales[i],
                reference.scene_physics,
            )
            setattr(self.scene, name, support)
        for side in ("left", "right"):
            sensor = getattr(self.scene, f"{side}_hand_object_contacts")
            sensor.newton_mjwarp.prim_path = "{ENV_REGEX_NS}/robot/.*"
            sensor.newton_mjwarp.filter_prim_paths_expr = ["{ENV_REGEX_NS}/object"]
            # PhysX uses one object sensor with one filter per hand collision
            # shape; conversion records these exact paths when available.
            filters = asset.get("physx_contact_filters", {}).get(side)
            if filters:
                for variant in (sensor.default, sensor.physx):
                    variant.prim_path = "{ENV_REGEX_NS}/object"
                    variant.filter_prim_paths_expr = filters["paths"]
                    variant.track_contact_points = True
                    variant.max_contact_data_count_per_prim = 128
                setattr(
                    self.commands.sharpa_reference,
                    f"{side}_contact_filter_link_index",
                    tuple(filters["link_indices"]),
                )
            event = getattr(self.events, f"{side}_physics_material")
            event.params["asset_cfg"] = SceneEntityCfg("robot", body_names=f"{side}_.*")
        self.commands.sharpa_reference.reference_manifest = self.reference.manifest
        self.decimation = 5
        self.sim.dt = 0.01
        self.sim.render_interval = 5
        self.sim.physics = SharpaPhysicsCfg()
        self.episode_length_s = (reference.frame_count + 20) * 0.05 + 1.0
        self.viewer.eye = (2.2, 2.2, 1.8)
        self.viewer.lookat = (0.0, 0.0, 1.0)
