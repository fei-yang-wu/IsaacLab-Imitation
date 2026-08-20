"""Newton dexterous-manipulation configuration for the Vega robot and two Wuji hands."""

from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path
from typing import Any, cast

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import MjcfFileCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from ...actions import VegaWujiReferenceResidualActionCfg
from ...command import VegaWujiReferenceCommandCfg
from ...curriculum import (
    FixedTimestepCurriculum,
    build_curriculum_params,
)
from ...newton_material import configure_wuji_hand_material
from ...newton_scene_material import configure_scene_material
from ...newton_scene_offsets import configure_scene_contact_offsets


def _default_mjcf_path() -> str:
    return os.environ.get(
        "DEXMANIP_VEGA_WUJI_MJCF_PATH",
        str(
            Path(__file__).resolve().parents[5]
            / "assets/vega_wuji/vega_u_wuji_v2_beta1_with_mount.xml"
        ),
    )


def _default_reference_path() -> str:
    return os.environ.get("DEXMANIP_VEGA_WUJI_REFERENCE_PATH", "")


_FINGER_STIFFNESS = {
    "[lr]_thumb_cmc_flex": 0.40844710645048043,
    "[lr]_thumb_cmc_abd": 0.6858601063643346,
    "[lr]_thumb_mcp": 0.2391112196099482,
    "[lr]_thumb_ip": 0.20736128319711747,
    "[lr]_index_finger_mcp_flex": 0.37352218155860073,
    "[lr]_index_finger_mcp_abd": 0.45592448909027794,
    "[lr]_index_finger_pip": 0.24368366649522863,
    "[lr]_index_finger_dip": 0.18026971340925335,
    "[lr]_middle_finger_mcp_flex": 0.3687093483646485,
    "[lr]_middle_finger_mcp_abd": 0.4164253443634641,
    "[lr]_middle_finger_pip": 0.22218607502059182,
    "[lr]_middle_finger_dip": 0.19427606072023446,
    "[lr]_ring_finger_mcp_flex": 0.35718151495111794,
    "[lr]_ring_finger_mcp_abd": 0.42977315313086895,
    "[lr]_ring_finger_pip": 0.24930151196247122,
    "[lr]_ring_finger_dip": 0.2285032688178066,
    "[lr]_pinky_mcp_flex": 0.3655325975433942,
    "[lr]_pinky_mcp_abd": 0.41393113081120425,
    "[lr]_pinky_pip": 0.22729367621965954,
    "[lr]_pinky_dip": 0.1964723550341816,
}

_FINGER_DAMPING = {
    "[lr]_thumb_cmc_flex": 0.020882010257063675,
    "[lr]_thumb_cmc_abd": 0.030610939996373314,
    "[lr]_thumb_mcp": 0.010181475050560962,
    "[lr]_thumb_ip": 0.00909698844675045,
    "[lr]_index_finger_mcp_flex": 0.01882274330029718,
    "[lr]_index_finger_mcp_abd": 0.019798167597016643,
    "[lr]_index_finger_pip": 0.010477031953162727,
    "[lr]_index_finger_dip": 0.008240212147903584,
    "[lr]_middle_finger_mcp_flex": 0.01848622487024593,
    "[lr]_middle_finger_mcp_abd": 0.018032947229953678,
    "[lr]_middle_finger_pip": 0.009592200014076666,
    "[lr]_middle_finger_dip": 0.009152994605972402,
    "[lr]_ring_finger_mcp_flex": 0.018376606800780005,
    "[lr]_ring_finger_mcp_abd": 0.01867700966212433,
    "[lr]_ring_finger_pip": 0.01059512121009555,
    "[lr]_ring_finger_dip": 0.009917602877441107,
    "[lr]_pinky_mcp_flex": 0.018616960278988272,
    "[lr]_pinky_mcp_abd": 0.018732177029667153,
    "[lr]_pinky_pip": 0.00951005441616486,
    "[lr]_pinky_dip": 0.009017295241756363,
}

_FINGER_EFFORT = {
    "[lr]_thumb_cmc_(flex|abd)": 0.6,
    "[lr]_thumb_(mcp|ip)": 0.3,
    "[lr]_(index_finger|middle_finger|ring_finger|pinky)_mcp_flex": 2.0,
    "[lr]_(index_finger|middle_finger|ring_finger|pinky)_mcp_abd": 0.2,
    "[lr]_(index_finger|middle_finger|ring_finger|pinky)_(pip|dip)": 0.3,
}


def _robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=MjcfFileCfg(
            asset_path=_default_mjcf_path(),
            fix_base=True,
            self_collision=True,
            robot_type="Humanoid",
            collision_from_visuals=False,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.19),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["Lift"],
                effort_limit_sim=1000.0,
                velocity_limit_sim=100.0,
                stiffness=2000.0,
                damping=100.0,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_flip"],
                effort_limit_sim=500.0,
                velocity_limit_sim=100.0,
                stiffness=1000.0,
                damping=50.0,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_j[1-3]"],
                effort_limit_sim=30.0,
                velocity_limit_sim=100.0,
                stiffness=120.0,
                damping=12.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["[LR]_arm_j[1-7]"],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=["[lr]_.*"],
                effort_limit_sim=_FINGER_EFFORT,
                velocity_limit_sim=100.0,
                stiffness=_FINGER_STIFFNESS,
                damping=_FINGER_DAMPING,
            ),
        },
    )


def _newton_cfg() -> NewtonCfg:
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=5000,
            nconmax=1000,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            use_mujoco_contacts=True,
            tolerance=1.0e-8,
        ),
        num_substeps=1,
        debug_mode=False,
        # Keep all imported shapes on the released zero contact/rest-offset
        # convention, including assets that omit Newton collision schemas.
        default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
    )


@configclass
class VegaWujiPhysicsCfg(PresetCfg):
    """The task supports only the Newton MJWarp backend."""

    default = _newton_cfg()
    newton_mjwarp = _newton_cfg()


@configclass
class VegaWujiRobotCfg(PresetCfg):
    """Use the same MJCF-backed articulation for the Newton presets."""

    default = _robot_cfg()
    newton_mjwarp = _robot_cfg()


@configclass
class VegaWujiSceneCfg(InteractiveSceneCfg):
    """Base scene. ILTools Reference data adds objects and support surfaces."""

    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=700.0,
            color=(0.9, 0.9, 0.9),
        ),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2500.0,
            angle=0.35,
            color=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.0, 0.3535534, 0.3535534, 0.8660254)
        ),
    )
    robot: ArticulationCfg = cast(ArticulationCfg, MISSING)


@configclass
class VegaWujiActionsCfg:
    """One residual per live actuator. The environment adds zero-size VOC terms."""

    joint_residual = VegaWujiReferenceResidualActionCfg(command_name="motion")


@configclass
class VegaWujiObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        wrist_position = ObsTerm(
            func=mdp.wrist_position_e,
            params={"command_name": "motion"},
        )
        wrist_orientation = ObsTerm(
            func=mdp.wrist_orientation_e,
            params={"command_name": "motion"},
        )
        wrist_velocity = ObsTerm(
            func=mdp.wrist_velocity_b,
            params={"command_name": "motion"},
        )
        finger_joint_pos = ObsTerm(
            func=mdp.finger_joint_pos,
            params={"command_name": "motion"},
        )
        finger_joint_vel = ObsTerm(
            func=mdp.finger_joint_vel,
            params={"command_name": "motion"},
        )
        object_position = ObsTerm(
            func=mdp.object_position_e,
            params={"command_name": "motion"},
        )
        object_orientation = ObsTerm(
            func=mdp.object_orientation_e,
            params={"command_name": "motion"},
        )
        command = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "motion"},
        )
        last_action = ObsTerm(func=isaac_mdp.last_action)
        processed_right_actions = ObsTerm(
            func=mdp.processed_action,
            params={"action_name": "joint_residual", "side": "right"},
        )
        processed_left_actions = ObsTerm(
            func=mdp.processed_action,
            params={"action_name": "joint_residual", "side": "left"},
        )
        contact_position_direction = ObsTerm(
            func=mdp.contact_position_direction_in_wrist,
            params={"command_name": "motion"},
        )
        # Additive transition-conditioning contract.  Keep every released
        # observation above intact for compatibility and expose the full
        # state/target tuple explicitly for the RLOpt policy.
        current_robot_state = ObsTerm(
            func=mdp.current_robot_state,
            params={"command_name": "motion"},
        )
        next_desired_robot_state = ObsTerm(
            func=mdp.next_desired_robot_state,
            params={"command_name": "motion"},
        )
        next_desired_object_state = ObsTerm(
            func=mdp.next_desired_object_state,
            params={"command_name": "motion"},
        )
        current_object_state = ObsTerm(
            func=mdp.current_object_state,
            params={"command_name": "motion"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class VegaWujiCommandsCfg:
    """ILTools motion command used by all dexmanip terms."""

    motion = VegaWujiReferenceCommandCfg()


@configclass
class VegaWujiEventsCfg:
    """Startup materials. The data contract owns reset state."""

    wuji_hand_material = EventTerm(
        func=configure_wuji_hand_material,
        mode="startup",
        params={
            "static_friction_range": (2.0, 2.01),
            "dynamic_friction_range": (2.0, 2.01),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    scene_material = EventTerm(
        func=configure_scene_material,
        mode="startup",
    )
    # Isaac Lab cannot reach an instanced collision prim, so the declared
    # contact offsets never arrive through the spawner. Restate the zero-offset
    # contract on the finalized Newton model.
    scene_contact_offsets = EventTerm(
        func=configure_scene_contact_offsets,
        mode="startup",
        params={
            "object_contact_margin": 0.0,
            "object_contact_gap": 0.0,
            "support_contact_margin": 0.0,
            "support_contact_gap": 0.0,
        },
    )


@configclass
class VegaWujiRewardsCfg:
    # ReconBody signal: the policy is rewarded for producing the complete
    # retargeted Vega/Wuji state, not only hand/object keypoints.
    robot_joint_position_tracking_exp = RewTerm(
        func=mdp.robot_joint_position_tracking_exp,
        weight=5.0,
        params={"command_name": "motion", "var": 1.0},
    )
    robot_joint_velocity_tracking_exp = RewTerm(
        func=mdp.robot_joint_velocity_tracking_exp,
        weight=0.5,
        params={"command_name": "motion", "var": 4.0},
    )
    object_position_tracking_exp = RewTerm(
        func=mdp.object_position_tracking_exp,
        weight=1.0,
        params={"command_name": "motion", "var": 0.04},
    )
    object_orientation_tracking_exp = RewTerm(
        func=mdp.object_orientation_tracking_exp,
        weight=1.0,
        params={"command_name": "motion", "var": 0.4},
    )
    object_twist_tracking_exp = RewTerm(
        func=mdp.object_twist_tracking_exp,
        weight=0.25,
        params={"command_name": "motion", "var": 1.0},
    )
    action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-5.0e-3)
    action_norm = RewTerm(
        func=mdp.action_norm,
        weight=-2.0e-3,
        params={"action_name": "joint_residual"},
    )
    object_keypoints_tracking_exp = RewTerm(
        func=mdp.object_keypoints_tracking_exp,
        weight=0.0,
        params={"command_name": "motion", "var": 0.1},
    )
    hand_keypoints_tracking_exp = RewTerm(
        func=mdp.hand_keypoints_tracking_exp,
        weight=0.0,
        params={"command_name": "motion", "var": 0.1},
    )
    hand_joint_pos_tracking_exp = RewTerm(
        func=mdp.hand_joint_pos_tracking_exp,
        weight=0.0,
        params={"command_name": "motion", "var": 1.0},
    )
    termination_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-100.0,
    )
    contact_wrench_support_reward = RewTerm(
        func=mdp.contact_wrench_support_reward,
        weight=10.0,
        params={"command_name": "motion", "tolerance": 0.1, "var": 0.1},
    )
    unintended_contact_penalty = RewTerm(
        func=mdp.unintended_contact_penalty,
        weight=-10.0,
        params={"command_name": "motion"},
    )
    missed_contact_penalty = RewTerm(
        func=mdp.missed_contact_penalty,
        weight=-1.0,
        params={"command_name": "motion"},
    )
    joint_position_limit = RewTerm(
        func=isaac_mdp.joint_pos_limits,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_velocity_l2 = RewTerm(
        func=isaac_mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_torque_l2 = RewTerm(
        func=isaac_mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    support_contact_force_penalty = RewTerm(
        func=mdp.support_contact_force_penalty,
        weight=-10.0,
        params={
            "sensor_name": "robot_support_contacts",
            "penalty_start_force": 5.0,
            "saturation_force": 50.0,
        },
    )


@configclass
class VegaWujiTerminationsCfg:
    time_out = DoneTerm(
        func=mdp.timestep_timeout,
        time_out=True,
        params={"command_name": "motion"},
    )
    wrist_away_from_trajectory = DoneTerm(
        func=mdp.hand_wrist_away_from_trajectory,
        params={"command_name": "motion", "threshold": 0.2},
    )
    robot_joint_away_from_trajectory = DoneTerm(
        func=mdp.robot_joint_away_from_trajectory,
        params={
            "command_name": "motion",
            "normalized_position_threshold": 0.35,
        },
    )
    object_away_from_trajectory = DoneTerm(
        func=mdp.object_away_from_trajectory,
        params={
            "command_name": "motion",
            "position_threshold": 0.2,
            "orientation_threshold": 0.7,
        },
    )
    excessive_support_contact_force = DoneTerm(
        func=mdp.excessive_support_contact_force,
        params={"sensor_name": "robot_support_contacts", "threshold": 75.0},
    )
    nonfinite_physics_state = DoneTerm(
        func=mdp.nonfinite_physics_state,
        params={"command_name": "motion"},
    )


@configclass
class VegaWujiCurriculumCfg:
    fixed_timestep = CurrTerm(
        func=FixedTimestepCurriculum,
        params=build_curriculum_params(
            command_name="motion",
            num_steps_per_env=24,
        ),
    )


@configclass
class VegaWujiImitationEnvCfg(ManagerBasedRLEnvCfg):
    """Object-and-hand tracking with Vega, Wuji, Newton, and RLOpt."""

    scene = VegaWujiSceneCfg(
        num_envs=4096,
        env_spacing=1.5,
        # The Reference contract requires the same robot and scene assets in each
        # environment. Newton physics replication keeps joint properties aligned
        # with the articulation instance count.
        replicate_physics=True,
        filter_collisions=False,
    )
    observations = VegaWujiObservationsCfg()
    actions = VegaWujiActionsCfg()
    commands = VegaWujiCommandsCfg()
    rewards = VegaWujiRewardsCfg()
    terminations = VegaWujiTerminationsCfg()
    events = VegaWujiEventsCfg()
    curriculum = VegaWujiCurriculumCfg()

    reference_path: str = _default_reference_path()
    reference_fps: float = 20.0
    reference_joint_names: list[str] = []
    randomize_reference: bool = True
    reference_motion_index: int = 0
    expected_robot_name: str = "vega_wuji"
    # PPO/evaluation fail closed on typed ILTools qualification and a hash-bound
    # JSON Manifest. Only the state-lock inspection script disables this gate.
    require_training_qualified_reference: bool = True

    def __post_init__(self) -> None:
        scene: Any = self.scene
        scene.robot = VegaWujiRobotCfg()
        self.commands.motion.randomize_reference = self.randomize_reference
        self.commands.motion.fixed_motion_index = self.reference_motion_index
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        simulation: Any = self.sim
        simulation.physics = VegaWujiPhysicsCfg()
        self.viewer.eye = (-0.7, 0.7, 1.4)
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.viewer.origin_type = "env"


__all__ = [
    "VegaWujiImitationEnvCfg",
    "VegaWujiPhysicsCfg",
    "VegaWujiRobotCfg",
]
