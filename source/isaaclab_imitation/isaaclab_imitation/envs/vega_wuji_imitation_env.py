"""Newton dexterous-manipulation environment for Vega and two Wuji hands."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from iltools.core import (
    DexterousReference,
    load_dexterous_reference_set,
    verify_training_qualification,
)
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab_newton.sensors import ContactSensorCfg

from isaaclab_imitation.tasks.manager_based.dexmanip.command import (
    VegaWujiReferenceCommand,
    wxyz_to_xyzw_tensor,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.mdp import (
    root_link_twist_to_com_twist_w,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.newton_voc import (
    NewtonVirtualRigidObjectControlCfg,
)
from isaaclab_imitation.tasks.manager_based.dexmanip.scene import (
    ContactMaterialSpec,
    RigidObjectSpec,
    SceneSpec,
    SupportSurfaceSpec,
    apply_scene_assets,
    wxyz_to_xyzw,
)


_WRIST_FRAME_CONTRACT = {
    "right": ("right_palm", "r_mount"),
    "left": ("left_palm", "l_mount"),
}
_REQUIRED_ROBOT_BODIES = ("R_ee", "L_ee", "r_mount", "l_mount")
_REQUIRED_ROBOT_SITES = ("right_palm", "left_palm")
_FORBIDDEN_SCENE_BODIES = ("grasp_cube", "desk")
_SITE_ORIENTATION_ATTRIBUTES = ("axisangle", "euler", "xyaxes", "zaxis")
_REFERENCE_TIMING_ABS_TOLERANCE_S = 1.0e-9


def _xml_floats(
    element: ET.Element,
    attribute: str,
    *,
    default: tuple[float, ...],
) -> np.ndarray:
    """Read one fixed-width MJCF numeric attribute."""

    raw = element.attrib.get(attribute)
    if raw is None:
        return np.asarray(default, dtype=np.float64)
    try:
        value = np.asarray([float(item) for item in raw.split()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            f"MJCF {element.tag} {element.attrib.get('name', '<unnamed>')!r} has "
            f"an invalid {attribute} attribute."
        ) from exc
    if value.shape != (len(default),):
        raise ValueError(
            f"MJCF {element.tag} {element.attrib.get('name', '<unnamed>')!r} "
            f"{attribute} must have {len(default)} values."
        )
    return value


def _identity_site_issues(root: ET.Element) -> list[str]:
    """Check that each Reference palm site is its expected mount body frame."""

    issues: list[str] = []
    for side, (site_name, body_name) in _WRIST_FRAME_CONTRACT.items():
        bodies = [
            element
            for element in root.iter("body")
            if element.attrib.get("name") == body_name
        ]
        sites = [
            element
            for element in root.iter("site")
            if element.attrib.get("name") == site_name
        ]
        if len(bodies) != 1:
            issues.append(f"{body_name} must resolve exactly once")
            continue
        if len(sites) != 1:
            issues.append(f"{site_name} must resolve exactly once")
            continue
        body = bodies[0]
        site = sites[0]
        if site not in list(body):
            issues.append(f"{site_name} is not a direct child of {body_name}")
            continue
        position = _xml_floats(site, "pos", default=(0.0, 0.0, 0.0))
        quaternion = _xml_floats(site, "quat", default=(1.0, 0.0, 0.0, 0.0))
        identity_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
        has_other_orientation = any(
            attribute in site.attrib for attribute in _SITE_ORIENTATION_ATTRIBUTES
        )
        if (
            not np.allclose(position, 0.0, rtol=0.0, atol=1.0e-9)
            or not np.allclose(
                np.abs(quaternion), identity_quaternion, rtol=0.0, atol=1.0e-9
            )
            or has_other_orientation
            or "fromto" in site.attrib
        ):
            issues.append(
                f"{site_name} must be an identity site on {body_name} for the "
                f"{side} wrist frame"
            )
    return issues


def _validate_robot_only_mjcf(path: str | Path) -> None:
    """Reject a robot asset that contains a task scene."""

    asset_path = Path(path).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Vega-Wuji MJCF is missing: {asset_path}")
    try:
        root = ET.parse(asset_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Vega-Wuji MJCF is not valid XML: {asset_path}") from exc

    body_names = {
        element.attrib["name"]
        for element in root.iter("body")
        if "name" in element.attrib
    }
    site_names = {
        element.attrib["name"]
        for element in root.iter("site")
        if "name" in element.attrib
    }
    missing_bodies = [name for name in _REQUIRED_ROBOT_BODIES if name not in body_names]
    missing_sites = [name for name in _REQUIRED_ROBOT_SITES if name not in site_names]
    scene_bodies = [name for name in _FORBIDDEN_SCENE_BODIES if name in body_names]
    issues: list[str] = []
    if missing_bodies:
        issues.append("missing required bodies: " + ", ".join(missing_bodies))
    if missing_sites:
        issues.append("missing required sites: " + ", ".join(missing_sites))
    if scene_bodies:
        issues.append(
            "robot-only MJCF contains scene bodies: " + ", ".join(scene_bodies)
        )
    issues.extend(_identity_site_issues(root))
    if issues:
        raise ValueError(
            f"Invalid Vega-Wuji robot-only MJCF {asset_path}: {'; '.join(issues)}"
        )


def _validate_wrist_frame_cfg(cfg: Any) -> None:
    """Keep command and Reference wrist frames on one fixed contract.

    The default full-joint action has no wrist-frame parameter.  Validate the
    wrist body too when a legacy wrist-pose action is configured.
    """

    issues: list[str] = []
    legacy_action = getattr(cfg.actions, "wrist_pose_residual", None)
    for side, (reference_site, runtime_body) in _WRIST_FRAME_CONTRACT.items():
        command_body = str(getattr(cfg.commands.motion, f"{side}_wrist_body_name"))
        configured_site = str(
            getattr(cfg.commands.motion, f"{side}_wrist_reference_frame_name")
        )
        if command_body != runtime_body:
            issues.append(
                f"command {side} wrist body is {command_body!r}, expected "
                f"{runtime_body!r}"
            )
        if legacy_action is not None:
            action_body = str(getattr(legacy_action, f"{side}_wrist_body_name"))
            if action_body != runtime_body:
                issues.append(
                    f"action {side} wrist body is {action_body!r}, expected "
                    f"{runtime_body!r}"
                )
        if configured_site != reference_site:
            issues.append(
                f"Reference {side} wrist frame is {configured_site!r}, expected "
                f"{reference_site!r}"
            )
    if issues:
        raise ValueError("Invalid Vega-Wuji wrist-frame config: " + "; ".join(issues))


def _mjcf_actuator_joint_names(path: str | Path) -> tuple[str, ...]:
    """Return the canonical joint order from the MJCF actuators."""

    root = ET.parse(Path(path).expanduser().resolve()).getroot()
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("Vega-Wuji MJCF must contain an actuator section.")
    names = tuple(item.attrib["joint"] for item in actuator if item.attrib.get("joint"))
    if not names or len(names) != len(set(names)):
        raise ValueError("Vega-Wuji actuator joints must be non-empty and unique.")
    return names


def _side_contact(reference: DexterousReference, side: str) -> dict[str, Any]:
    """Return one hand contact group without unused padded slots."""

    contacts = reference.contacts
    if contacts is None:
        raise ValueError(
            f"Reference {reference.sequence_id!r} must contain reference contacts."
        )
    try:
        side_index = contacts.hand_sides.index(side)
    except ValueError as exc:
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no {side} contact group."
        ) from exc
    slot_indices = [
        index
        for index, name in enumerate(contacts.link_names[side_index].tolist())
        if str(name)
    ]
    if not slot_indices:
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no named {side} contact links."
        )
    names = tuple(str(contacts.link_names[side_index, index]) for index in slot_indices)
    index = np.asarray(slot_indices, dtype=np.int64)
    return {
        "link_names": names,
        "object_positions_w": contacts.object_positions_w[:, side_index, index],
        "link_normals_w": contacts.link_normals_w[:, side_index, index],
        "object_indices": contacts.object_indices[:, side_index, index],
        "active": contacts.active[:, side_index, index],
    }


def _validate_shared_scene(
    reference: DexterousReference,
    first: DexterousReference,
) -> None:
    """Require one scene layout for all motions in a vectorized scene."""

    exact_fields = (
        "object_names",
        "object_asset_paths",
        "object_asset_sha256",
        "left_wrist_frame_name",
        "right_wrist_frame_name",
        "left_hand_frame_names",
        "right_hand_frame_names",
        "support_surface_names",
        "support_surface_asset_paths",
        "support_surface_asset_sha256",
    )
    for field_name in exact_fields:
        if getattr(reference, field_name) != getattr(first, field_name):
            raise ValueError(
                f"Reference {reference.sequence_id!r} changes {field_name}. "
                "All motions in one task must use one scene contract."
            )
    array_fields = (
        "fixed_root_pose_w",
        "object_scales",
        "object_radii",
        "support_surface_scales",
        "support_surface_poses_w",
    )
    for field_name in array_fields:
        if not np.allclose(
            getattr(reference, field_name),
            getattr(first, field_name),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise ValueError(
                f"Reference {reference.sequence_id!r} changes {field_name}."
            )
    if (reference.scene_physics is None) != (first.scene_physics is None):
        raise ValueError(f"Reference {reference.sequence_id!r} changes scene_physics.")
    if reference.scene_physics is not None and first.scene_physics is not None:
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
        for field_name in physics_fields:
            if not np.allclose(
                getattr(reference.scene_physics, field_name),
                getattr(first.scene_physics, field_name),
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise ValueError(
                    f"Reference {reference.sequence_id!r} changes "
                    f"scene_physics.{field_name}."
                )


def _copy_with_tail(target: np.ndarray, motion_index: int, value: np.ndarray) -> None:
    """Copy one variable-length motion and hold its final frame in padding."""

    frame_count = len(value)
    target[motion_index, :frame_count] = value
    target[motion_index, frame_count:] = value[-1]


def _pack_references(
    references: Sequence[DexterousReference],
    *,
    expected_robot_name: str,
    expected_fps: float,
    joint_names: Sequence[str],
    fingertip_names: dict[str, Sequence[str]],
    wrist_frame_names: dict[str, str],
) -> dict[str, Any]:
    """Validate and pack ILTools References for one vectorized task."""

    references = tuple(references)
    if not references:
        raise ValueError("The ILTools Reference set must contain at least one motion.")
    first = references[0]
    target_joint_names = tuple(str(name) for name in joint_names)
    expected_wrist_frame_names = {
        side: str(wrist_frame_names[side]) for side in ("right", "left")
    }
    if not first.object_names:
        raise ValueError("The Reference must contain at least one rigid object.")
    if any(not path for path in first.object_asset_paths):
        raise ValueError("Each Reference object must have an asset path in ILTools.")
    if np.any(np.asarray(first.object_radii) <= 0.0):
        raise ValueError("Each Reference object radius must be positive.")
    if any(not path for path in first.support_surface_asset_paths):
        raise ValueError("Each support surface must have an asset path in ILTools.")

    max_frames = max(reference.frame_count for reference in references)
    motion_count = len(references)
    joint_count = len(target_joint_names)
    object_count = len(first.object_names)
    qpos = np.zeros((motion_count, max_frames, joint_count), dtype=np.float32)
    qvel = np.zeros_like(qpos)
    wrist_pose = {
        side: np.zeros((motion_count, max_frames, 7), dtype=np.float32)
        for side in ("right", "left")
    }
    object_pose = np.zeros(
        (motion_count, max_frames, object_count, 7), dtype=np.float32
    )
    object_twist = np.zeros(
        (motion_count, max_frames, object_count, 6), dtype=np.float32
    )
    hand_frame_pose = {
        side: np.zeros(
            (motion_count, max_frames, len(fingertip_names[side]), 7),
            dtype=np.float32,
        )
        for side in ("right", "left")
    }
    lengths = np.empty(motion_count, dtype=np.int64)

    first_contacts = {side: _side_contact(first, side) for side in ("right", "left")}
    contact_arrays: dict[str, dict[str, np.ndarray]] = {}
    for side in ("right", "left"):
        slot_count = len(first_contacts[side]["link_names"])
        contact_arrays[side] = {
            "object_positions_w": np.zeros(
                (motion_count, max_frames, slot_count, 3), dtype=np.float32
            ),
            "link_normals_w": np.zeros(
                (motion_count, max_frames, slot_count, 3), dtype=np.float32
            ),
            "object_indices": np.full(
                (motion_count, max_frames, slot_count), -1, dtype=np.int64
            ),
            "active": np.zeros((motion_count, max_frames, slot_count), dtype=np.bool_),
        }

    for motion_index, reference in enumerate(references):
        if reference.robot_name != expected_robot_name:
            raise ValueError(
                f"Reference {reference.sequence_id!r} targets robot "
                f"{reference.robot_name!r}, expected {expected_robot_name!r}."
            )
        if not np.isclose(reference.fps, expected_fps, rtol=0.0, atol=1.0e-5):
            raise ValueError(
                f"Reference {reference.sequence_id!r} has {reference.fps} fps, "
                f"expected {expected_fps}."
            )
        if set(reference.joint_names) != set(target_joint_names):
            missing = sorted(set(target_joint_names) - set(reference.joint_names))
            extra = sorted(set(reference.joint_names) - set(target_joint_names))
            raise ValueError(
                f"Reference {reference.sequence_id!r} joint set differs from the "
                f"Vega-Wuji MJCF. Missing={missing}; extra={extra}."
            )
        for side in ("right", "left"):
            actual_frame = str(getattr(reference, f"{side}_wrist_frame_name"))
            expected_frame = expected_wrist_frame_names[side]
            if actual_frame != expected_frame:
                raise ValueError(
                    f"Reference {reference.sequence_id!r} {side} wrist frame is "
                    f"{actual_frame!r}, expected {expected_frame!r}."
                )
        _validate_shared_scene(reference, first)
        source_indices = [
            reference.joint_names.index(name) for name in target_joint_names
        ]
        lengths[motion_index] = reference.frame_count
        _copy_with_tail(qpos, motion_index, reference.qpos[:, source_indices])
        _copy_with_tail(qvel, motion_index, reference.qvel[:, source_indices])
        _copy_with_tail(wrist_pose["right"], motion_index, reference.right_wrist_pose_w)
        _copy_with_tail(wrist_pose["left"], motion_index, reference.left_wrist_pose_w)
        _copy_with_tail(object_pose, motion_index, reference.object_poses_w)
        _copy_with_tail(object_twist, motion_index, reference.object_twists_w)

        for side in ("right", "left"):
            names = tuple(getattr(reference, f"{side}_hand_frame_names"))
            required_names = tuple(str(name) for name in fingertip_names[side])
            missing_frames = [name for name in required_names if name not in names]
            if missing_frames:
                raise ValueError(
                    f"Reference {reference.sequence_id!r} is missing {side} hand "
                    f"frames: {missing_frames}."
                )
            frame_indices = [names.index(name) for name in required_names]
            source_frames = getattr(reference, f"{side}_hand_frame_poses_w")
            _copy_with_tail(
                hand_frame_pose[side],
                motion_index,
                source_frames[:, frame_indices],
            )

            contacts = _side_contact(reference, side)
            if contacts["link_names"] != first_contacts[side]["link_names"]:
                raise ValueError(
                    f"Reference {reference.sequence_id!r} changes {side} contact links."
                )
            frame_count = reference.frame_count
            for field_name, destination in contact_arrays[side].items():
                destination[motion_index, :frame_count] = contacts[field_name]

    return {
        "first": first,
        "joint_names": target_joint_names,
        "lengths": lengths,
        "qpos": qpos,
        "qvel": qvel,
        "wrist_pose_wxyz": wrist_pose,
        "object_pose_wxyz": object_pose,
        "object_twist_w": object_twist,
        "hand_frame_pose_wxyz": hand_frame_pose,
        "contact_link_names": {
            side: tuple(dict.fromkeys(first_contacts[side]["link_names"]))
            for side in ("right", "left")
        },
        "contacts": contact_arrays,
    }


def _scene_spec(reference: DexterousReference) -> SceneSpec:
    """Create the rigid scene specification from one ILTools Reference."""

    scene_physics = reference.scene_physics
    objects = tuple(
        RigidObjectSpec(
            name=name,
            asset_path=asset_path,
            init_pos=tuple(
                float(value) for value in reference.object_poses_w[0, index, :3]
            ),
            init_quat_wxyz=tuple(
                float(value) for value in reference.object_poses_w[0, index, 3:7]
            ),
            scale=tuple(float(value) for value in reference.object_scales[index]),
            mass=(
                float(scene_physics.object_mass_kg[index])
                if scene_physics is not None
                else None
            ),
            material=(
                ContactMaterialSpec(
                    static_friction=float(scene_physics.object_static_friction[index]),
                    dynamic_friction=float(
                        scene_physics.object_dynamic_friction[index]
                    ),
                    restitution=float(scene_physics.object_restitution[index]),
                )
                if scene_physics is not None
                else None
            ),
        )
        for index, (name, asset_path) in enumerate(
            zip(reference.object_names, reference.object_asset_paths, strict=True)
        )
    )
    supports = tuple(
        SupportSurfaceSpec(
            name=name,
            asset_path=asset_path,
            init_pos=tuple(
                float(value) for value in reference.support_surface_poses_w[index, :3]
            ),
            init_quat_wxyz=tuple(
                float(value) for value in reference.support_surface_poses_w[index, 3:7]
            ),
            scale=tuple(
                float(value) for value in reference.support_surface_scales[index]
            ),
            material=(
                ContactMaterialSpec(
                    static_friction=float(scene_physics.support_static_friction[index]),
                    dynamic_friction=float(
                        scene_physics.support_dynamic_friction[index]
                    ),
                    restitution=float(scene_physics.support_restitution[index]),
                )
                if scene_physics is not None
                else ContactMaterialSpec()
            ),
        )
        for index, (name, asset_path) in enumerate(
            zip(
                reference.support_surface_names,
                reference.support_surface_asset_paths,
                strict=True,
            )
        )
    )
    return SceneSpec(objects=objects, support_surfaces=supports)


def _apply_scene_inertial_contract(
    objects: Sequence[RigidObject],
    scene_physics: Any | None,
    *,
    num_envs: int,
    device: str | torch.device,
) -> None:
    """Write ILTools mass, COM, and diagonal inertia through public Newton APIs."""

    if scene_physics is None:
        return
    if len(objects) != len(scene_physics.object_mass_kg):
        raise RuntimeError("Scene physics and runtime object counts differ.")
    for object_index, item in enumerate(objects):
        mass = torch.full(
            (num_envs, 1),
            float(scene_physics.object_mass_kg[object_index]),
            dtype=torch.float32,
            device=device,
        )
        center_of_mass = (
            torch.as_tensor(
                scene_physics.object_center_of_mass_m[object_index],
                dtype=torch.float32,
                device=device,
            )
            .view(1, 1, 3)
            .repeat(num_envs, 1, 1)
        )
        diagonal = torch.as_tensor(
            scene_physics.object_diagonal_inertia_kg_m2[object_index],
            dtype=torch.float32,
            device=device,
        )
        inertia = torch.diag(diagonal).reshape(1, 1, 9).repeat(num_envs, 1, 1)
        item.set_masses_index(masses=mass)
        item.set_coms_index(coms=center_of_mass)
        item.set_inertias_index(inertias=inertia)


def _set_robot_initial_pose(robot_cfg: Any, pose_wxyz: np.ndarray) -> None:
    """Set the fixed root pose on each unresolved robot preset variant."""

    variants = [
        getattr(robot_cfg, name)
        for name in ("default", "newton_mjwarp")
        if hasattr(robot_cfg, name)
    ]
    if not variants:
        variants = [robot_cfg]
    position = tuple(float(value) for value in pose_wxyz[:3])
    orientation = wxyz_to_xyzw(pose_wxyz[3:7])
    for variant in variants:
        variant.init_state.pos = position
        variant.init_state.rot = orientation


def _add_contact_sensors(
    cfg: Any,
    link_names: dict[str, tuple[str, ...]],
    object_names: tuple[str, ...],
    support_surface_names: tuple[str, ...] = (),
) -> None:
    """Add Newton hand-to-object force matrices before scene creation."""

    # Newton treats these expressions as shell globs after it replaces ``.*``
    # with ``*``. It does not support regular-expression alternation. Match the
    # complete hand subtree and let the command term select the Reference links.
    # Imported rigid-object bodies are below the scene object's root path.
    object_paths = [
        path
        for name in object_names
        for path in (
            f"{{ENV_REGEX_NS}}/{name}",
            f"{{ENV_REGEX_NS}}/{name}/.*",
        )
    ]
    for side in ("right", "left"):
        if not link_names[side]:
            raise ValueError(f"The {side} hand must have a named contact link.")
        wrist_name = str(getattr(cfg.commands.motion, f"{side}_wrist_body_name"))
        sensor_cfg = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/.*/{wrist_name}.*",
            update_period=float(cfg.sim.dt),
            history_length=3,
            force_threshold=float(cfg.commands.motion.contact_force_threshold),
            filter_prim_paths_expr=object_paths,
            track_contact_points=False,
            track_friction_forces=False,
        )
        setattr(cfg.scene, f"{side}_hand_object_contacts", sensor_cfg)
    if support_surface_names:
        support_paths = [
            path
            for name in support_surface_names
            for path in (
                f"{{ENV_REGEX_NS}}/{name}",
                f"{{ENV_REGEX_NS}}/{name}/.*",
            )
        ]
        cfg.scene.robot_support_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=float(cfg.sim.dt),
            history_length=1,
            force_threshold=float(cfg.commands.motion.contact_force_threshold),
            # Supports are static shapes without RigidBodyAPI, so Newton must
            # resolve the counterpart at shape rather than body level.
            filter_shape_prim_expr=support_paths,
            track_contact_points=False,
            track_friction_forces=False,
        )
    else:
        # The task also supports free-space handovers. Avoid a manager term
        # referring to a sensor with no filtered counterpart in that case.
        cfg.rewards.support_contact_force_penalty = None
        cfg.terminations.excessive_support_contact_force = None


def _sync_motion_command_cfg(cfg: Any) -> None:
    """Apply late Hydra selection overrides to the managed command config."""

    cfg.commands.motion.randomize_reference = bool(cfg.randomize_reference)
    cfg.commands.motion.fixed_motion_index = int(cfg.reference_motion_index)


def _validate_reference_timing(cfg: Any) -> float:
    """Require one Reference frame per policy transition after overrides.

    The command term advances exactly one frame after each control transition.
    Any mismatch between the ILTools Reference rate and the environment control
    rate would therefore hold or skip physical time while still looking index
    aligned.  Validate after Hydra has applied all overrides and fail closed.

    Returns:
        The validated control/Reference period in seconds.
    """

    simulation_dt = float(cfg.sim.dt)
    decimation = int(cfg.decimation)
    reference_fps = float(cfg.reference_fps)
    if not math.isfinite(simulation_dt) or simulation_dt <= 0.0:
        raise ValueError("sim.dt must be finite and positive.")
    if isinstance(cfg.decimation, bool) or decimation <= 0:
        raise ValueError("decimation must be a positive integer.")
    if not math.isfinite(reference_fps) or reference_fps <= 0.0:
        raise ValueError("reference_fps must be finite and positive.")
    control_period_s = simulation_dt * decimation
    reference_period_s = 1.0 / reference_fps
    if not math.isclose(
        control_period_s,
        reference_period_s,
        rel_tol=0.0,
        abs_tol=_REFERENCE_TIMING_ABS_TOLERANCE_S,
    ):
        raise ValueError(
            "Vega-Wuji requires one ILTools Reference frame per control "
            "transition; got control period "
            f"sim.dt * decimation = {control_period_s:.12g}s and Reference "
            f"period 1 / reference_fps = {reference_period_s:.12g}s."
        )
    return control_period_s


class VegaWujiImitationEnv(ManagerBasedRLEnv):
    """Track ILTools dexterous References with one fixed-base Vega articulation."""

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any) -> None:
        _sync_motion_command_cfg(cfg)
        _validate_reference_timing(cfg)
        _validate_wrist_frame_cfg(cfg)
        robot_cfg = getattr(cfg.scene.robot, "default", cfg.scene.robot)
        asset_path = Path(robot_cfg.spawn.asset_path).expanduser().resolve()
        _validate_robot_only_mjcf(asset_path)
        joint_names = _mjcf_actuator_joint_names(asset_path)
        if (
            cfg.reference_joint_names
            and tuple(cfg.reference_joint_names) != joint_names
        ):
            raise ValueError(
                "Configured reference_joint_names must match the MJCF actuator order."
            )

        reference_path = str(cfg.reference_path).strip()
        if not reference_path:
            raise ValueError(
                "Set env.reference_path or DEXMANIP_VEGA_WUJI_REFERENCE_PATH "
                "to an ILTools dexterous Reference NPZ or JSON Manifest."
            )
        reference_source = Path(reference_path).expanduser().resolve()
        is_manifest = reference_source.suffix.lower() == ".json"
        references = load_dexterous_reference_set(
            reference_source,
            runtime_model_path=asset_path if is_manifest else None,
            require_model_hash=is_manifest,
        )
        if bool(cfg.require_training_qualified_reference):
            if not is_manifest:
                raise ValueError(
                    "Training requires a JSON Reference Manifest so the runtime "
                    "Vega-Wuji MJCF and every motion are hash-bound. Direct NPZ "
                    "inputs are allowed only for explicit inspection replay."
                )
            for reference in references:
                verify_training_qualification(reference)
        for reference in references:
            reference.verify_scene_assets(require_hashes=is_manifest)
        fingertip_names = {
            side: tuple(getattr(cfg.commands.motion, f"{side}_fingertip_body_names"))
            for side in ("right", "left")
        }
        packed = _pack_references(
            references,
            expected_robot_name=str(cfg.expected_robot_name),
            expected_fps=float(cfg.reference_fps),
            joint_names=joint_names,
            fingertip_names=fingertip_names,
            wrist_frame_names={
                side: str(
                    getattr(
                        cfg.commands.motion,
                        f"{side}_wrist_reference_frame_name",
                    )
                )
                for side in ("right", "left")
            },
        )
        first: DexterousReference = packed["first"]
        apply_scene_assets(cfg.scene, _scene_spec(first))
        _set_robot_initial_pose(cfg.scene.robot, first.fixed_root_pose_w)
        _add_contact_sensors(
            cfg,
            packed["contact_link_names"],
            first.object_names,
            first.support_surface_names,
        )
        for object_name in first.object_names:
            setattr(
                cfg.actions,
                f"virtual_object_control_{object_name}",
                NewtonVirtualRigidObjectControlCfg(
                    asset_name=object_name,
                    command_name="motion",
                ),
            )

        device = cfg.sim.device
        self._reference_joint_names = packed["joint_names"]
        self._reference_qpos = torch.as_tensor(
            packed["qpos"], dtype=torch.float32, device=device
        )
        self._reference_qvel = torch.as_tensor(
            packed["qvel"], dtype=torch.float32, device=device
        )
        self._reference_lengths = torch.as_tensor(
            packed["lengths"], dtype=torch.long, device=device
        )
        self._reference_wrist_pose_wxyz = {
            side: torch.as_tensor(value, dtype=torch.float32, device=device)
            for side, value in packed["wrist_pose_wxyz"].items()
        }
        self._reference_object_pose_wxyz = torch.as_tensor(
            packed["object_pose_wxyz"], dtype=torch.float32, device=device
        )
        self._reference_object_twist_w = torch.as_tensor(
            packed["object_twist_w"], dtype=torch.float32, device=device
        )
        self._reference_hand_frame_pose_wxyz = {
            side: torch.as_tensor(value, dtype=torch.float32, device=device)
            for side, value in packed["hand_frame_pose_wxyz"].items()
        }
        self._reference_contacts = {
            side: {
                field_name: torch.as_tensor(value, device=device)
                for field_name, value in fields.items()
            }
            for side, fields in packed["contacts"].items()
        }
        self._reference_contact_link_names = packed["contact_link_names"]
        self._reference_object_names = tuple(first.object_names)
        self._reference_support_surface_names = tuple(first.support_surface_names)
        self._reference_object_radii = torch.as_tensor(
            first.object_radii, dtype=torch.float32, device=device
        )
        self._scene_physics = first.scene_physics

        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self._motion_command: VegaWujiReferenceCommand = self.command_manager.get_term(
            "motion"
        )
        live_joint_names = tuple(str(name) for name in self.robot.joint_names)
        if set(live_joint_names) != set(self._reference_joint_names):
            raise RuntimeError(
                "The imported Vega-Wuji joints differ from the MJCF actuator set. "
                f"Live={live_joint_names}; expected={self._reference_joint_names}."
            )
        if live_joint_names != self._reference_joint_names:
            source_indices = torch.as_tensor(
                [self._reference_joint_names.index(name) for name in live_joint_names],
                dtype=torch.long,
                device=self.device,
            )
            self._reference_qpos = torch.index_select(
                self._reference_qpos, dim=-1, index=source_indices
            )
            self._reference_qvel = torch.index_select(
                self._reference_qvel, dim=-1, index=source_indices
            )
            self._reference_joint_names = live_joint_names
            self._motion_command._reference_qpos = self._reference_qpos
            self._motion_command._reference_qvel = self._reference_qvel
        self._objects: list[RigidObject] = [
            self.scene[name] for name in self._reference_object_names
        ]
        _apply_scene_inertial_contract(
            self._objects,
            self._scene_physics,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._motion_command.refresh_object_com_poses()
        finger_ids: list[int] = []
        for side in ("right", "left"):
            ids, names = self.robot.find_joints(
                [getattr(cfg.commands.motion, f"{side}_finger_joint_expr")],
                preserve_order=True,
            )
            if len(ids) != 20:
                raise RuntimeError(f"{side} Wuji hand has {len(names)} joints, not 20.")
            finger_ids.extend(int(index) for index in ids)
        self._finger_joint_ids = torch.as_tensor(
            finger_ids, dtype=torch.long, device=self.device
        )

    @staticmethod
    def _env_ids_tensor(
        env_ids: Sequence[int] | torch.Tensor,
        device: str | torch.device,
    ) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=device, dtype=torch.long)
        return torch.as_tensor(tuple(env_ids), dtype=torch.long, device=device)

    def _write_object_reference(self, env_ids: torch.Tensor) -> None:
        motion_ids = self._motion_command.motion_index[env_ids]
        frame_ids = self._motion_command.timestep_counter[env_ids]
        object_pose_wxyz = self._reference_object_pose_wxyz[motion_ids, frame_ids]
        object_twist_w = self._reference_object_twist_w[motion_ids, frame_ids]
        for object_index, item in enumerate(self._objects):
            local_pose = object_pose_wxyz[:, object_index]
            link_twist_w = object_twist_w[:, object_index]
            com_twist_w = root_link_twist_to_com_twist_w(
                link_twist_w,
                local_pose[:, 3:7],
                self._motion_command._object_com_poses_b_wxyz[
                    env_ids, object_index, :3
                ],
            )
            world_pose = torch.cat(
                (
                    local_pose[:, :3] + self.scene.env_origins[env_ids],
                    wxyz_to_xyzw_tensor(local_pose[:, 3:7]),
                ),
                dim=-1,
            )
            item.write_root_pose_to_sim(world_pose, env_ids=env_ids)
            item.write_root_velocity_to_sim(
                com_twist_w,
                env_ids=env_ids,
            )

    def set_reference_motion(
        self, motion_index: int, env_ids: Sequence[int] | None = None
    ) -> None:
        """Select one ILTools motion for replay or a deterministic check."""

        motion_count = int(self._reference_qpos.shape[0])
        if not 0 <= int(motion_index) < motion_count:
            raise IndexError(
                f"Reference motion index {motion_index} is outside [0, {motion_count})."
            )
        if env_ids is None:
            env_ids = tuple(range(self.num_envs))
        ids = self._env_ids_tensor(env_ids, self.device)
        self._motion_command.motion_index[ids] = int(motion_index)
        self._motion_command.timestep_counter[ids] = 0
        self._motion_command.steps_since_last_reset[ids] = int(
            self.cfg.commands.motion.warmup_steps
        )
        self._motion_command.tracking_lengths[ids] = self._reference_lengths[
            int(motion_index)
        ]
        self.write_reference_state_to_sim(ids)

    def write_reference_state_to_sim(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> None:
        """Write the current robot and object Reference state to Newton."""

        if env_ids is None:
            env_ids = tuple(range(self.num_envs))
        ids = self._env_ids_tensor(env_ids, self.device)
        self.robot.write_joint_state_to_sim(
            self.current_reference_joint_pos[ids],
            self.current_reference_joint_vel[ids],
            env_ids=ids,
        )
        self._write_object_reference(ids)

    @property
    def current_reference_joint_pos(self) -> torch.Tensor:
        return self._motion_command.reference_joint_pos

    @property
    def current_reference_joint_vel(self) -> torch.Tensor:
        return self._motion_command.reference_joint_vel

    @property
    def reference_frame(self) -> torch.Tensor:
        return self._motion_command.timestep_counter

    @property
    def reference_motion(self) -> torch.Tensor:
        return self._motion_command.motion_index

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        ids = self._env_ids_tensor(env_ids, self.device)
        if len(ids) == 0:
            return
        joint_position = self._motion_command.reference_joint_pos[ids].clone()
        if bool(self.cfg.commands.motion.randomize_reset_finger_openness):
            openness = torch.rand(len(ids), 1, device=self.device) * float(
                self.cfg.commands.motion.reset_finger_openness
            )
            joint_position[:, self._finger_joint_ids] *= openness
        limits = self.robot.data.soft_joint_pos_limits.torch[ids]
        joint_position = torch.maximum(
            torch.minimum(joint_position, limits[..., 1]), limits[..., 0]
        )
        if int(self.cfg.commands.motion.warmup_steps) == 0:
            joint_velocity = self._motion_command.reference_joint_vel[ids]
        else:
            # A warmup freezes the selected Reference frame, so zero velocity
            # is the only state consistent with that held position target.
            joint_velocity = torch.zeros_like(joint_position)
        self.robot.write_joint_state_to_sim(
            joint_position,
            joint_velocity,
            env_ids=ids,
        )
        self._write_object_reference(ids)


__all__ = ["VegaWujiImitationEnv"]
