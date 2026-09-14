# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Source-parity variant of the dual floating-hand Sharpa task.

``Isaac-Sharpa-V2D-Source-v0`` reproduces the released video-to-data
``Sharpa-V2D-v0`` recipe: the twelve-term observation, the 65-value command,
the friction-cone contact-wrench rewards, the released terminations, the
hand-material randomization, the ``step`` virtual-object-control decay with
its settling hold, and the six-term reward curriculum. Data must be converted
with ``--motion-speed 0.5`` (the released half-speed reference playback).

Known departures from the released environment, each documented here and in
``wiki/sharpa-progress.md``:

- Support surfaces come from the reference USDA as static colliders, not as
  re-spawned kinematic cylinders, and hand-to-support collisions stay enabled
  (the released ``configure_collision_groups`` is not ported).
- On PhysX the contact positions are Isaac Lab's averaged contact points when
  tracked, else the link origins. Newton uses exact contact points.
- ``scene.replicate_physics`` stays ``True`` (released: ``False``) because the
  Newton contact-pair adapter requires it; the MDP is unchanged.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab_imitation.assets.sharpa import ASSET_DIR, HAND_CONTACT_BODIES

from ...curriculum import (
    SOURCE_REWARD_WEIGHT_SCHEDULES,
    FixedTimestepCurriculum,
    build_curriculum_params,
)
from ...sharpa_source_command import SharpaSourceReferenceCommandCfg
from ... import sharpa_source_mdp
from .sharpa_env_cfg import (
    SharpaActionsCfg,
    SharpaV2DEnvCfg,
    object_body_prim_path,
)

COMMAND = {"command_name": "sharpa_reference"}
SOURCE_MOTION_SPEED = 0.5
"""Released reference playback: 40 Hz data advanced once per 20 Hz control step."""

NOISE = Unoise(n_min=-0.01, n_max=0.01)


def _mesh_stem(link: ET.Element) -> list[str]:
    stems = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        if mesh is not None:
            stems.append(Path(mesh.attrib["filename"]).stem)
    return stems


def hand_link_contact_filters(
    side: str, link_patterns: tuple[str, ...] | list[str]
) -> tuple[list[str], list[int]]:
    """Return one PhysX filter pattern per collision shape of each contact link.

    Isaac Lab's URDF importer nests every non-merged link below its parent
    under ``<robot>/Geometry/<root>/...``, merges fixed-joint children into
    their parent body, and names each collision prim after its mesh file
    stem. PhysX requires one shape per filter pattern, so a link whose merged
    fixed children carry collision meshes (the fingertip elastomers) yields
    several patterns. The second list maps each pattern to its link index in
    ``link_patterns`` so the command term can reduce shapes to links.
    """

    urdf = ASSET_DIR / "urdfs" / "sharpawave" / f"{side}_sharpa_wave.urdf"
    root = ET.parse(urdf).getroot()
    links = {str(link.attrib["name"]): link for link in root.findall("link")}
    parent_of: dict[str, tuple[str, str]] = {}
    fixed_children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        child = joint.find("child").attrib["link"]  # type: ignore[union-attr]
        parent = joint.find("parent").attrib["link"]  # type: ignore[union-attr]
        joint_type = joint.attrib.get("type", "fixed")
        parent_of[child] = (parent, joint_type)
        if joint_type == "fixed":
            fixed_children.setdefault(parent, []).append(child)
    patterns: list[str] = []
    link_index: list[int] = []
    for index, pattern in enumerate(link_patterns):
        expression = pattern.replace(".*", side, 1)
        if expression not in links:
            raise ValueError(
                f"Contact link pattern {pattern!r} resolved no link in {urdf.name}."
            )
        chain = []
        current = expression
        while current in parent_of:
            parent, joint_type = parent_of[current]
            if joint_type != "fixed":
                chain.append(current)
            current = parent
        chain.append(current)
        chain.reverse()
        body = f"{{ENV_REGEX_NS}}/{side}_robot/Geometry/" + "/".join(chain)
        stems = list(_mesh_stem(links[expression]))
        queue = list(fixed_children.get(expression, []))
        while queue:
            merged = queue.pop(0)
            stems.extend(_mesh_stem(links[merged]))
            queue.extend(fixed_children.get(merged, []))
        if not stems:
            raise ValueError(f"Contact link {expression!r} has no collision mesh.")
        for stem in stems:
            patterns.append(f"{body}/{stem}")
            link_index.append(index)
    return patterns, link_index


def hand_link_prim_paths(
    side: str, link_patterns: tuple[str, ...] | list[str]
) -> list[str]:
    """Return the imported rigid-body prim path of each contact link."""

    patterns, link_index = hand_link_contact_filters(side, link_patterns)
    paths: list[str] = []
    for pattern, index in zip(patterns, link_index, strict=True):
        if len(paths) == index:
            paths.append(pattern.rsplit("/", 1)[0])
    return paths


@configclass
class SharpaSourceObservationsCfg:
    """The released twelve-term policy observation, in order."""

    @configclass
    class PolicyCfg(ObsGroup):
        wrist_position_e = ObsTerm(
            func=sharpa_source_mdp.wrist_position_e, params=COMMAND, noise=NOISE
        )
        wrist_orientation_e = ObsTerm(
            func=sharpa_source_mdp.wrist_orientation_e, params=COMMAND, noise=NOISE
        )
        wrist_velocity_b = ObsTerm(
            func=sharpa_source_mdp.wrist_velocity_b, params=COMMAND, noise=NOISE
        )
        finger_joint_pos = ObsTerm(
            func=sharpa_source_mdp.finger_joint_pos, params=COMMAND, noise=NOISE
        )
        finger_joint_vel = ObsTerm(
            func=sharpa_source_mdp.finger_joint_vel, params=COMMAND, noise=NOISE
        )
        object_position_e = ObsTerm(
            func=sharpa_source_mdp.object_position_e, params=COMMAND, noise=NOISE
        )
        object_orientation_e = ObsTerm(
            func=sharpa_source_mdp.object_orientation_e, params=COMMAND, noise=NOISE
        )
        command = ObsTerm(func=isaac_mdp.generated_commands, params=COMMAND)
        actions = ObsTerm(func=isaac_mdp.last_action)
        processed_right_actions = ObsTerm(
            func=sharpa_source_mdp.processed_action,
            params={"action_name": "right_hand"},
        )
        processed_left_actions = ObsTerm(
            func=sharpa_source_mdp.processed_action,
            params={"action_name": "left_hand"},
        )
        contact_position_direction_in_wrist = ObsTerm(
            func=sharpa_source_mdp.contact_position_direction_in_wrist, params=COMMAND
        )

        def __post_init__(self) -> None:
            # The released config declares the noise terms but disables
            # corruption, so the noise is inert. Keep both facts.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SharpaSourceCommandsCfg:
    sharpa_reference = SharpaSourceReferenceCommandCfg()


@configclass
class SharpaSourceRewardsCfg:
    """The released reward set and initial weights."""

    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-5.0e-3)
    action_l1 = RewTerm(
        func=sharpa_source_mdp.action_norm_sum,
        weight=-2.0e-3,
        params={"action_names": ["right_hand", "left_hand"]},
    )
    object_keypoints_tracking_exp = RewTerm(
        func=sharpa_source_mdp.object_keypoints_tracking_exp,
        weight=0.0,
        params={**COMMAND, "var": 0.1},
    )
    hand_keypoints_tracking_exp = RewTerm(
        func=sharpa_source_mdp.hand_keypoints_tracking_exp,
        weight=0.0,
        params={**COMMAND, "var": 0.1},
    )
    hand_joint_pos_tracking_exp = RewTerm(
        func=sharpa_source_mdp.hand_joint_pos_tracking_exp,
        weight=0.0,
        params={**COMMAND, "var": 1.0},
    )
    termination_penalty = RewTerm(
        func=sharpa_source_mdp.termination_penalty, weight=-100.0
    )
    contact_wrench_support_reward = RewTerm(
        func=sharpa_source_mdp.contact_wrench_support_reward,
        weight=10.0,
        params={**COMMAND, "tolerance": 0.1, "var": 0.1},
    )
    unintended_contact_penalty = RewTerm(
        func=sharpa_source_mdp.unintended_contact_penalty,
        weight=-10.0,
        params=COMMAND,
    )
    missed_contact_penalty = RewTerm(
        func=sharpa_source_mdp.missed_contact_penalty, weight=-1.0, params=COMMAND
    )


@configclass
class SharpaSourceTerminationsCfg:
    time_out = DoneTerm(
        func=sharpa_source_mdp.reference_timeout, params=COMMAND, time_out=True
    )
    hand_wrist_away_from_trajectory = DoneTerm(
        func=sharpa_source_mdp.hand_wrist_away_from_trajectory,
        params={**COMMAND, "threshold": 0.2},
    )
    object_away_from_trajectory = DoneTerm(
        func=sharpa_source_mdp.object_away_from_trajectory,
        params={**COMMAND, "position_threshold": 0.2, "orientation_threshold": 0.7},
    )


def _hand_material_event(side: str) -> EventTerm:
    return EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(f"{side}_robot", body_names=".*"),
            "static_friction_range": (2.0, 2.01),
            "dynamic_friction_range": (2.0, 2.01),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class SharpaSourceEventsCfg:
    """The released startup material randomization for both hands."""

    right_physics_material = _hand_material_event("right")
    left_physics_material = _hand_material_event("left")


@configclass
class SharpaSourceCurriculumCfg:
    fixed_timestep = CurrTerm(
        func=FixedTimestepCurriculum,
        params=build_curriculum_params(
            command_name="sharpa_reference",
            num_steps_per_env=24,
            reward_weight_schedules=SOURCE_REWARD_WEIGHT_SCHEDULES,
        ),
    )


@configclass
class SharpaV2DSourceParityEnvCfg(SharpaV2DEnvCfg):
    """Released Sharpa-V2D-v0 recipe on this repository's runtime."""

    observations = SharpaSourceObservationsCfg()
    actions = SharpaActionsCfg()
    commands = SharpaSourceCommandsCfg()
    rewards = SharpaSourceRewardsCfg()
    terminations = SharpaSourceTerminationsCfg()
    events = SharpaSourceEventsCfg()
    curriculum = SharpaSourceCurriculumCfg()

    require_source_motion_speed: bool = True
    """Refuse references not resampled with the released motion speed."""

    def __post_init__(self) -> None:
        super().__post_init__()
        from iltools.core import load_dexterous_reference_set

        references = load_dexterous_reference_set(self.reference.manifest)
        for reference in references:
            if reference.contacts is None:
                raise ValueError(
                    "Source-parity Sharpa references need contact geometry."
                )
            resample: dict[str, Any] = dict(reference.metadata.get("resample", {}))
            speed = resample.get("motion_speed")
            if self.require_source_motion_speed and (
                speed is None or abs(float(speed) - SOURCE_MOTION_SPEED) > 1.0e-9
            ):
                raise ValueError(
                    "Source-parity Sharpa references must be converted with "
                    f"--motion-speed {SOURCE_MOTION_SPEED}; this reference records "
                    f"motion_speed={speed!r}."
                )
        max_frames = max(reference.frame_count for reference in references)
        hold = int(self.commands.sharpa_reference.virtual_object_control_decay_steps)
        # Released rule: the trajectory duration at the played-back rate plus
        # the settling hold, with one extra second so timestep_timeout ends
        # the episode rather than the manager's episode clock.
        self.episode_length_s = (max_frames + hold) * 0.05 + 1.0
        # The released config sets replicate_physics=False for PhysX. Newton's
        # contact-pair adapter needs replicated physics to see one object body
        # per environment, and replication does not change the MDP, so both
        # backends keep the scene default (True).

        scene: Any = self.scene
        object_asset = references[0].object_asset_paths[0]
        for side in ("right", "left"):
            sensor = getattr(scene, f"{side}_hand_object_contacts")
            # Released direction: the object is sensed and the 17 hand links
            # are the filters, one shape each.
            #
            # PhysX's GPU pipeline rejects every one of these shape filters
            # ("GPU contact filter for collider ... is not supported", all 44
            # shapes) and then faults with a CUDA misaligned address inside
            # GpuRigidContactView. Filtering on the link's rigid-body prim
            # instead removes that warning but matches no rigid contact entry
            # at all, so the sensor cannot initialize. PhysX GPU contact
            # sensing on this asset is unresolved; see
            # wiki/sharpa-progress.md. Newton reads exact contact pairs
            # through NewtonContactPairAdapter and is unaffected.
            filters, link_index = hand_link_contact_filters(side, HAND_CONTACT_BODIES)
            setattr(
                self.commands.sharpa_reference,
                f"{side}_contact_filter_link_index",
                tuple(link_index),
            )
            physx = PhysXContactSensorCfg(
                prim_path=object_body_prim_path(object_asset),
                filter_prim_paths_expr=filters,
                history_length=3,
                track_air_time=True,
                track_contact_points=True,
                force_threshold=0.1,
                max_contact_data_count_per_prim=128,
            )
            sensor.default = physx
            sensor.physx = physx
            # Newton senses the hand links against the object; the command
            # term reads exact contact pairs through NewtonContactPairAdapter.
            sensor.newton_mjwarp.prim_path = f"{{ENV_REGEX_NS}}/{side}_robot/.*"
            sensor.newton_mjwarp.filter_prim_paths_expr = ["{ENV_REGEX_NS}/object"]


__all__ = [
    "SOURCE_MOTION_SPEED",
    "SharpaSourceObservationsCfg",
    "SharpaSourceRewardsCfg",
    "SharpaSourceTerminationsCfg",
    "SharpaSourceEventsCfg",
    "SharpaV2DSourceParityEnvCfg",
    "hand_link_contact_filters",
    "hand_link_prim_paths",
]
