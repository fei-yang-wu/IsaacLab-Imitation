# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two free-floating Sharpa hands, one rigid object, and PhysX."""

from __future__ import annotations

from dataclasses import MISSING
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from iltools.core import load_dexterous_reference_set
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg, UsdFileCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_imitation.assets.sharpa import LEFT_SHARPA_WAVE_CFG, RIGHT_SHARPA_WAVE_CFG

from ...curriculum import FixedTimestepCurriculum, build_curriculum_params
from ...sharpa_actions import SharpaResidualActionCfg
from ...sharpa_command import SharpaReferenceCommandCfg
from ...sharpa_object_action import SharpaVirtualObjectActionCfg
from ... import sharpa_mdp


@configclass
class SharpaPhysicsCfg(PresetCfg):
    """Physics backends for Sharpa; PhysX is the default, Newton is opt-in."""

    default = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=5000,
            nconmax=1000,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=True,
            tolerance=1.0e-8,
        ),
        num_substeps=1,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
    )


@configclass
class SharpaContactSensorCfg(PresetCfg):
    """Contact sensor implementations matching the selected physics backend."""

    default = PhysXContactSensorCfg(history_length=1, track_air_time=False)
    physx = PhysXContactSensorCfg(history_length=1, track_air_time=False)
    newton_mjwarp = NewtonContactSensorCfg(history_length=1, track_air_time=False)


@configclass
class SharpaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    right_robot: ArticulationCfg = MISSING
    left_robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING
    right_hand_object_contacts = SharpaContactSensorCfg()
    left_hand_object_contacts = SharpaContactSensorCfg()


@configclass
class SharpaCommandsCfg:
    sharpa_reference = SharpaReferenceCommandCfg()


@configclass
class SharpaActionsCfg:
    right_hand = SharpaResidualActionCfg(asset_name="right_robot")
    left_hand = SharpaResidualActionCfg(asset_name="left_robot")
    virtual_object = SharpaVirtualObjectActionCfg(asset_name="object")


@configclass
class SharpaObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        wrist_position = ObsTerm(func=sharpa_mdp.wrist_position_e)
        wrist_orientation = ObsTerm(func=sharpa_mdp.wrist_orientation_e)
        wrist_velocity = ObsTerm(func=sharpa_mdp.wrist_velocity_b)
        finger_joint_pos = ObsTerm(func=sharpa_mdp.finger_joint_pos)
        finger_joint_vel = ObsTerm(func=sharpa_mdp.finger_joint_vel)
        object_pose = ObsTerm(func=sharpa_mdp.object_pose_e)
        command = ObsTerm(func=sharpa_mdp.generated_command)
        actions = ObsTerm(func=isaac_mdp.last_action)
        processed_actions = ObsTerm(func=sharpa_mdp.processed_actions)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SharpaRewardsCfg:
    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-5.0e-3)
    action_l1 = RewTerm(func=sharpa_mdp.action_l1, weight=-2.0e-3)
    object_keypoints_tracking_exp = RewTerm(
        func=sharpa_mdp.object_keypoints_tracking_exp, weight=0.0, params={"var": 0.1}
    )
    object_pose_tracking_exp = RewTerm(
        func=sharpa_mdp.object_pose_tracking_exp,
        weight=0.0,
        params={"position_var": 0.1, "orientation_var": 0.5},
    )
    object_goal_tracking_exp = RewTerm(
        func=sharpa_mdp.object_goal_tracking_exp,
        weight=0.0,
        params={"position_var": 0.1, "orientation_var": 0.5},
    )
    object_lift_progress = RewTerm(
        func=sharpa_mdp.object_lift_progress,
        weight=0.0,
        params={"target_height": 0.1},
    )
    object_task_success = RewTerm(
        func=sharpa_mdp.object_task_success,
        weight=0.0,
        params={
            "position_threshold": 0.05,
            "orientation_threshold": 0.35,
            "min_lift": 0.05,
        },
    )
    hand_keypoints_tracking_exp = RewTerm(
        func=sharpa_mdp.hand_keypoints_tracking_exp, weight=0.25, params={"var": 0.1}
    )
    hand_joint_pos_tracking_exp = RewTerm(
        func=sharpa_mdp.hand_joint_pos_tracking_exp, weight=0.25, params={"var": 1.0}
    )
    contact_wrench_support_reward = RewTerm(
        func=sharpa_mdp.contact_wrench_support_reward, weight=10.0
    )
    unintended_contact_penalty = RewTerm(
        func=sharpa_mdp.unintended_contact_penalty, weight=-10.0
    )
    missed_contact_penalty = RewTerm(
        func=sharpa_mdp.missed_contact_penalty, weight=-1.0
    )
    termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-100.0)


@configclass
class SharpaTerminationsCfg:
    time_out = DoneTerm(func=sharpa_mdp.reference_finished, time_out=True)
    wrist_away_from_trajectory = DoneTerm(
        func=sharpa_mdp.wrist_too_far, params={"threshold": 0.2}
    )
    object_away_from_trajectory = DoneTerm(
        func=sharpa_mdp.object_too_far, params={"threshold": 0.2}
    )


@configclass
class SharpaEventsCfg:
    """No external reset terms; the reference command resets all assets."""

    pass


@configclass
class SharpaCurriculumCfg:
    fixed_timestep = CurrTerm(
        func=FixedTimestepCurriculum,
        params=build_curriculum_params(
            command_name="sharpa_reference", num_steps_per_env=24
        ),
    )


@configclass
class SharpaReferenceDataCfg:
    """Reference dataset selected for this run."""

    manifest: str = os.environ.get("SHARPA_REFERENCE_MANIFEST", "")


def _object_cfg(asset_path: str, scene_physics: Any | None) -> RigidObjectCfg:
    suffix = Path(asset_path).suffix.lower()
    common: dict[str, Any] = {
        # The released rigid-object spawn reports contacts on the object so a
        # sensor can sit on it (source-parity sensor direction).
        "activate_contact_sensors": True,
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1.0,
            # Caps the impulse a penetrating reset may inject. The retarget
            # leaves the fingers inside the object, and without this cap the
            # first physics step throws both the hands and the object off the
            # Reference.
            max_contact_impulse=1e3,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.0,
            rest_offset=0.0,
        ),
    }
    if scene_physics is not None:
        from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg

        common["mass_props"] = MassPropertiesCfg(
            mass=float(scene_physics.object_mass_kg[0])
        )
        common["physics_material"] = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(scene_physics.object_static_friction[0]),
            dynamic_friction=float(scene_physics.object_dynamic_friction[0]),
            restitution=float(scene_physics.object_restitution[0]),
        )
    if suffix == ".urdf":
        spawn = UrdfFileCfg(
            asset_path=asset_path,
            fix_base=False,
            joint_drive=None,
            merge_fixed_joints=True,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
            # The released spawn decomposes the mesh. A convex hull, which is
            # the importer default, fills a container's cavity, so a hand
            # reaching into the box reads as centimetres of penetration and
            # the solver fights to push it back out.
            collision_type="Convex Decomposition",
            **common,
        )
    elif suffix in {".usd", ".usda", ".usdc"}:
        spawn = UsdFileCfg(usd_path=asset_path, **common)
    else:
        raise ValueError("Sharpa v1 object assets must be rigid URDF or USD files.")
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )


def _support_cfg(
    name: str,
    asset_path: str,
    pose_wxyz: np.ndarray,
    scale: np.ndarray,
    scene_physics: Any | None,
) -> AssetBaseCfg:
    """Build one static support collider from the reference scene contract."""

    def wxyz_to_xyzw(quaternion: np.ndarray) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in (*quaternion[1:4], quaternion[0]))

    physics_material = None
    if scene_physics is not None:
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(scene_physics.support_static_friction[0]),
            dynamic_friction=float(scene_physics.support_dynamic_friction[0]),
            restitution=float(scene_physics.support_restitution[0]),
        )
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=UsdFileCfg(
            usd_path=asset_path,
            scale=tuple(float(value) for value in scale),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=physics_material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in pose_wxyz[:3]),
            rot=wxyz_to_xyzw(pose_wxyz[3:7]),
        ),
    )


USD_SUFFIXES = (".usd", ".usda", ".usdc")


_USD_BODY_QUERY = """
import sys
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(sys.argv[1])
if stage is None:
    raise SystemExit("no-stage")
default_prim = stage.GetDefaultPrim()
if not default_prim:
    raise SystemExit("no-default-prim")
root = default_prim.GetPath()
bodies = [
    prim.GetPath()
    for prim in stage.Traverse()
    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
]
print("\\n".join(str(body) for body in bodies))
print(str(root))
"""


def _usd_paths_out_of_process(path: Path) -> tuple[list[str], str]:
    """Return ``(rigid body prim paths, default prim path)`` of one USD.

    The stage is opened in a child interpreter so this process never holds a
    USD stage of its own. Returned paths are absolute prim paths.
    """

    result = subprocess.run(
        [sys.executable, "-c", _USD_BODY_QUERY, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Cannot read the rigid bodies of {path}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"The USD {path} declares no rigid body.")
    return lines[:-1], lines[-1]


def _usd_rigid_body_relative_path(path: Path) -> str:
    """Return the object's rigid-body path relative to the USD default prim.

    Never assume the body sits at the asset root. A USD written by Isaac Lab's
    own URDF converter keeps the importer's ``Geometry/<link>`` nesting, while
    a hand-authored proxy may carry the body on the default prim itself. Read
    the stage and let it say which, so a wrong guess cannot silently produce a
    scene whose object has no rigid body.

    The read always happens in a child interpreter. Environment configs are
    resolved before Isaac Sim starts, and opening a USD stage in this process
    first makes Kit crash on startup.
    """

    body_paths, root_path = _usd_paths_out_of_process(path)
    if len(body_paths) != 1:
        raise ValueError(
            f"A Sharpa rigid object USD must declare exactly one rigid body; "
            f"{path.name} declares {body_paths}."
        )
    return _relative_prim_path(body_paths[0], root_path, path)


def _relative_prim_path(body: str, root: str, source: Path) -> str:
    """Return ``body`` expressed relative to ``root``, both absolute prim paths."""

    if body == root:
        return ""
    prefix = root.rstrip("/") + "/"
    if not body.startswith(prefix):
        raise ValueError(
            f"The rigid body {body} of {source.name} is outside its default "
            f"prim {root}."
        )
    return body[len(prefix) :]


def object_body_prim_path(asset_path: str) -> str:
    """Return the prim path of the object's rigid body inside one environment.

    Isaac Lab's URDF importer nests each link under ``<asset>/Geometry/<link>``
    and names the shape prims after the mesh files, not after the URDF, so the
    body path has to come from the URDF's own link name. A pre-converted USD is
    read directly, because the conversion keeps that same nesting.
    """

    path = Path(asset_path)
    suffix = path.suffix.lower()
    if suffix in USD_SUFFIXES:
        relative = _usd_rigid_body_relative_path(path)
        return "{ENV_REGEX_NS}/object" + (f"/{relative}" if relative else "")
    if suffix != ".urdf":
        return "{ENV_REGEX_NS}/object"
    root = ET.parse(path).getroot()
    links = [str(link.attrib["name"]) for link in root.findall("link")]
    if len(links) != 1:
        raise ValueError(
            f"A Sharpa rigid object URDF must declare exactly one link; "
            f"{path.name} declares {links}."
        )
    return "{ENV_REGEX_NS}/object/Geometry/" + links[0]


def _object_contact_filters(asset_path: str) -> list[str]:
    """Return the contact-filter expression for the rigid object."""

    return [object_body_prim_path(asset_path)]


@configclass
class SharpaV2DEnvCfg(ManagerBasedRLEnvCfg):
    scene = SharpaSceneCfg(
        num_envs=4096,
        env_spacing=1.5,
        replicate_physics=True,
        filter_collisions=False,
    )
    observations = SharpaObservationsCfg()
    actions = SharpaActionsCfg()
    commands = SharpaCommandsCfg()
    rewards = SharpaRewardsCfg()
    terminations = SharpaTerminationsCfg()
    events = SharpaEventsCfg()
    curriculum = SharpaCurriculumCfg()

    reference = SharpaReferenceDataCfg()
    # Compatibility alias for early Sharpa v1 commands and saved configs.
    reference_manifest: str = ""
    legacy_reset_hold: bool = False

    def __post_init__(self) -> None:
        manifest = self.reference.manifest or self.reference_manifest
        if not manifest:
            raise ValueError(
                "Set env.reference.manifest to an ILTools Sharpa NPZ or manifest."
            )
        self.reference.manifest = manifest
        self.reference_manifest = manifest
        references = load_dexterous_reference_set(manifest)
        first = references[0]
        physics_fields = (
            "object_mass_kg",
            "object_center_of_mass_m",
            "object_diagonal_inertia_kg_m2",
            "object_static_friction",
            "object_dynamic_friction",
            "object_restitution",
            "support_static_friction",
            "support_dynamic_friction",
            "support_restitution",
        )
        for reference in references:
            reference.verify_scene_assets(require_hashes=True)
            if reference.robot_layout != "dual_floating_hand":
                raise ValueError("Sharpa references require dual_floating_hand layout.")
            if reference.robot_name != "sharpa_wave":
                raise ValueError(
                    "Sharpa references must declare robot_name='sharpa_wave'."
                )
            if len(reference.object_names) != 1:
                raise ValueError("Sharpa v1 supports exactly one object.")
            if reference.metadata.get("object_kind", "rigid") != "rigid":
                raise ValueError("Sharpa v1 does not support articulated objects.")
            if reference.metadata.get("physics_backend", "physx") not in {
                "physx",
                "newton_mjwarp",
            }:
                raise ValueError(
                    "Sharpa references require PhysX or Newton MJWarp physics metadata."
                )
            if not np.isclose(float(reference.metadata.get("physics_dt", 0.01)), 0.01):
                raise ValueError("Sharpa reference physics_dt must be 0.01 seconds.")
            if not np.isclose(float(reference.metadata.get("control_dt", 0.05)), 0.05):
                raise ValueError("Sharpa reference control_dt must be 0.05 seconds.")
            if not np.isclose(reference.fps, 20.0):
                raise ValueError("Sharpa reference data must be sampled at 20 Hz.")
            if reference.object_asset_paths != first.object_asset_paths:
                raise ValueError(
                    "All references in one run must use the same object asset."
                )
            if (
                reference.support_surface_names != first.support_surface_names
                or reference.support_surface_asset_paths
                != first.support_surface_asset_paths
                or reference.support_surface_asset_sha256
                != first.support_surface_asset_sha256
            ):
                raise ValueError(
                    "All references in one run must use the same support assets."
                )
            if (reference.scene_physics is None) != (first.scene_physics is None):
                raise ValueError(
                    "Sharpa references have inconsistent physics metadata."
                )
            if reference.scene_physics is not None and first.scene_physics is not None:
                for field in physics_fields:
                    if not np.allclose(
                        getattr(reference.scene_physics, field),
                        getattr(first.scene_physics, field),
                        rtol=0.0,
                        atol=1.0e-8,
                    ):
                        raise ValueError(
                            "Sharpa references have inconsistent physics metadata."
                        )
        scene: Any = self.scene
        scene.right_robot = RIGHT_SHARPA_WAVE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/right_robot"
        )
        scene.left_robot = LEFT_SHARPA_WAVE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/left_robot"
        )
        scene.object = _object_cfg(first.object_asset_paths[0], first.scene_physics)
        object_filters = _object_contact_filters(first.object_asset_paths[0])
        for sensor, side in (
            (scene.right_hand_object_contacts, "right"),
            (scene.left_hand_object_contacts, "left"),
        ):
            for variant in (sensor.default, sensor.physx, sensor.newton_mjwarp):
                variant.prim_path = f"{{ENV_REGEX_NS}}/{side}_robot/.*"
                # PhysX consumes imported shape paths; Newton resolves body
                # labels and therefore needs the rigid-object body expression.
                variant.filter_prim_paths_expr = (
                    ["{ENV_REGEX_NS}/object"]
                    if variant is sensor.newton_mjwarp
                    else object_filters
                )
                # Newton resolves body labels, so the asset root is correct
                # there; PhysX needs the imported rigid-body path.
        for index, (name, asset_path) in enumerate(
            zip(
                first.support_surface_names,
                first.support_surface_asset_paths,
                strict=True,
            )
        ):
            if not name or not asset_path:
                raise ValueError("Sharpa support surfaces require names and assets.")
            setattr(
                scene,
                name,
                _support_cfg(
                    name,
                    asset_path,
                    first.support_surface_poses_w[index],
                    first.support_surface_scales[index],
                    first.scene_physics,
                ),
            )
        self.commands.sharpa_reference.reference_manifest = self.reference_manifest
        self.commands.sharpa_reference.legacy_reset_hold = self.legacy_reset_hold
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = SharpaPhysicsCfg()
        self.viewer.eye = (-0.5, 0.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.7)
        self.viewer.origin_type = "env"


__all__ = ["SharpaV2DEnvCfg"]
