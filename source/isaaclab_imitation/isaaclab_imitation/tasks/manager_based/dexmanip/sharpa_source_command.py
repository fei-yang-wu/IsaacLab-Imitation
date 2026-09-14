# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Source-parity reference command for the dual floating-hand Sharpa task.

This command reproduces the released ``DualHandsObjectTrackingCommand`` of
video-to-data on top of :class:`SharpaReferenceCommand`:

- the 65-value command layout (relative wrist poses, finger deltas, relative
  object pose),
- per-link contact tensors and friction-cone contact-wrench supports for the
  CHORD reward terms in :mod:`dexmanip.mdp`,
- the ``step`` virtual-object-control decay with a reference hold during the
  reset settling window,
- the released start-frame rule, and
- unique wrist quaternions and limit-scaled finger positions in the live
  joint order.

The task variant in :mod:`sharpa_command` keeps its own simpler contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from isaaclab.utils.configclass import configclass

from isaaclab_imitation.assets.sharpa import HAND_CONTACT_BODIES

from . import mdp
from .newton_contacts import NewtonContactPairAdapter
from .sharpa_command import (
    SharpaReferenceCommand,
    SharpaReferenceCommandCfg,
    _torch,
    xyzw_to_wxyz,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class SharpaSourceReferenceCommandCfg(SharpaReferenceCommandCfg):
    """Released command settings for the source-parity Sharpa task."""

    class_type: type = None  # type: ignore[assignment]
    contact_link_names: tuple[str, ...] = tuple(HAND_CONTACT_BODIES)
    """Hand links with contact sensing, in the released order (17 per hand)."""

    num_wrench_basis: int = 512
    num_friction_cone_edges: int = 8
    friction_coefficient: float = 0.1
    wrench_basis_seed: int = 0
    contact_force_threshold: float = 0.1
    """Force magnitude (N) above which a live link contact counts as active."""

    make_quat_unique: bool = True
    virtual_object_control_decay_mode: str = "step"

    right_contact_filter_link_index: tuple[int, ...] = ()
    left_contact_filter_link_index: tuple[int, ...] = ()
    """PhysX only: contact link index of each filter shape, set by the env cfg."""

    def __post_init__(self) -> None:
        self.class_type = SharpaSourceReferenceCommand


class SharpaSourceReferenceCommand(SharpaReferenceCommand):
    """Publish the released 65-value command with contact-wrench supports."""

    cfg: SharpaSourceReferenceCommandCfg

    def __init__(
        self, cfg: SharpaSourceReferenceCommandCfg, env: ManagerBasedRLEnv
    ) -> None:
        super().__init__(cfg, env)
        if cfg.virtual_object_control_decay_mode != "step":
            raise ValueError(
                "The source-parity command supports only the released 'step' "
                "virtual-object-control decay mode."
            )
        references = load_references(cfg.reference_manifest)

        # One rigid object shared by every reference: one radius.
        radii = [
            float(np.asarray(reference.object_radii)[0]) for reference in references
        ]
        if any(not np.isclose(radius, radii[0]) for radius in radii):
            raise ValueError("Sharpa references must share one object radius.")
        if radii[0] <= 0.0:
            raise ValueError("The Sharpa object radius must be positive.")
        self.object_radii = torch.tensor([radii[0]], device=self.device)

        # Reference contact geometry per hand side: (M, T, S, 3) padded.
        self._reference_contacts: dict[str, dict[str, torch.Tensor]] = {}
        self._reference_contact_link_names: dict[str, tuple[str, ...]] = {}
        max_length = int(self._lengths.max().item())
        for side in ("right", "left"):
            slot_names = None
            positions = []
            normals = []
            active = []
            for reference in references:
                contacts = reference.contacts
                if contacts is None:
                    raise ValueError(
                        "Source-parity Sharpa references need contact geometry."
                    )
                slots = [
                    slot
                    for slot, hand_side in enumerate(contacts.hand_sides)
                    if hand_side == side
                ]
                if len(slots) != 1:
                    raise ValueError(
                        f"Expected one {side} contact block; got {len(slots)}."
                    )
                slot = slots[0]
                names = tuple(str(name) for name in contacts.link_names[slot])
                if slot_names is None:
                    slot_names = names
                elif names != slot_names:
                    raise ValueError("Contact link names differ between references.")
                positions.append(np.asarray(contacts.object_positions_w[:, slot]))
                normals.append(np.asarray(contacts.link_normals_w[:, slot]))
                active.append(np.asarray(contacts.active[:, slot]))
            assert slot_names is not None
            self._reference_contact_link_names[side] = slot_names
            slot_count = len(slot_names)

            def pack(values: list[np.ndarray], shape: tuple[int, ...], dtype: Any):
                out = torch.zeros(
                    (len(values), max_length, *shape), device=self.device, dtype=dtype
                )
                for index, value in enumerate(values):
                    tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
                    out[index, : len(tensor)] = tensor
                    out[index, len(tensor) :] = tensor[-1]
                return out

            self._reference_contacts[side] = {
                "object_positions_w": pack(positions, (slot_count, 3), torch.float32),
                "link_normals_w": pack(normals, (slot_count, 3), torch.float32),
                "active": pack(active, (slot_count,), torch.bool),
            }

        # Live contact links, sensors, and backend adapters.
        self._contact_body_ids: dict[str, list[int]] = {}
        self._contact_body_names: dict[str, tuple[str, ...]] = {}
        self._contact_sensors: dict[str, Any] = {}
        self._contact_backend: dict[str, str] = {}
        self._contact_adapters: dict[str, NewtonContactPairAdapter | None] = {}
        for side, robot in (("right", self.right_robot), ("left", self.left_robot)):
            ids, names = robot.find_bodies(
                [name.replace(".*", side, 1) for name in cfg.contact_link_names], preserve_order=True
            )
            if len(ids) != len(cfg.contact_link_names):
                raise ValueError(
                    f"{side} contact links resolved {len(ids)} bodies for "
                    f"{len(cfg.contact_link_names)} expressions: {names}."
                )
            self._contact_body_ids[side] = list(ids)
            self._contact_body_names[side] = tuple(str(name) for name in names)
            sensor = env.scene.sensors.get(f"{side}_hand_object_contacts")
            if sensor is None:
                raise ValueError(
                    f"The scene has no {side}_hand_object_contacts sensor."
                )
            self._contact_sensors[side] = sensor
            backend = (
                "newton"
                if type(sensor).__module__.startswith("isaaclab_newton")
                else "physx"
            )
            self._contact_backend[side] = backend
            if backend == "newton":
                self._contact_adapters[side] = (
                    NewtonContactPairAdapter.from_contact_sensor(
                        sensor,
                        link_body_names=self._contact_body_names[side],
                        object_body_names=(cfg.object_name,),
                        force_threshold=float(cfg.contact_force_threshold),
                    )
                )
            else:
                self._contact_adapters[side] = None
                self._check_physx_sensor(side, sensor)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(cfg.wrench_basis_seed))
        self.wrench_space_bases = mdp.sample_wrench_space_basis_scaled(
            int(cfg.num_wrench_basis), 1.0, self.device, generator=generator
        ).unsqueeze(0)
        support_shape = (self.num_envs, self.num_bodies, int(cfg.num_wrench_basis))
        self._command_supports = {
            side: torch.zeros(support_shape, device=self.device)
            for side in ("right", "left")
        }
        self._current_supports = {
            side: torch.zeros(support_shape, device=self.device)
            for side in ("right", "left")
        }
        self._support_cache_step = -1
        self._support_cache_frames: torch.Tensor | None = None
        self._support_cache_motions: torch.Tensor | None = None
        self._live_contact_cache_step = -1
        self._live_contact_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._refresh_object_com_offset()
        for name in (
            "right_hand_wrist_wxyz_error",
            "left_hand_wrist_wxyz_error",
            "right_hand_finger_joints_error",
            "left_hand_finger_joints_error",
            "object_body_wxyz_error",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Sensors and contacts
    # ------------------------------------------------------------------

    def _check_physx_sensor(self, side: str, sensor: Any) -> None:
        """Require the released sensor direction: object sensed, links filtered."""

        expected = len(self._contact_body_ids[side])
        filters = list(getattr(sensor.cfg, "filter_prim_paths_expr", []) or [])
        link_index = tuple(getattr(self.cfg, f"{side}_contact_filter_link_index"))
        if len(link_index) != len(filters) or sorted(set(link_index)) != list(
            range(expected)
        ):
            raise ValueError(
                f"The {side} PhysX contact sensor must filter every collision shape "
                f"of the {expected} hand links; it filters {len(filters)} patterns "
                f"mapped to links {sorted(set(link_index))}."
            )
        self._shape_to_link = getattr(self, "_shape_to_link", {})
        self._shape_to_link[side] = torch.as_tensor(
            link_index, dtype=torch.long, device=self.device
        )
        body_names = tuple(str(name) for name in sensor.body_names)
        if len(body_names) != 1:
            raise ValueError(
                f"The {side} PhysX contact sensor must sense the rigid object only; "
                f"it senses {body_names}."
            )

    def _physx_live_contacts(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor = self._contact_sensors[side]
        forces = _torch(sensor.data.force_matrix_w)
        if forces is None:
            raise RuntimeError(f"The {side} PhysX contact sensor reports no forces.")
        forces = forces[:, 0]  # (N, F, 3): force on the object per filter shape
        shape_to_link = self._shape_to_link[side]
        expected = len(self._contact_body_ids[side])
        if forces.shape[1] != shape_to_link.numel():
            raise RuntimeError(
                f"The {side} PhysX contact sensor resolved {forces.shape[1]} filter "
                f"shapes for {shape_to_link.numel()} configured patterns."
            )
        shape_positions = sensor.data.contact_pos_w
        if shape_positions is not None:
            shape_positions = torch.nan_to_num(_torch(shape_positions)[:, 0], nan=0.0)
        else:
            # Isaac Lab did not track contact points: use the link origins.
            robot = self.right_robot if side == "right" else self.left_robot
            shape_positions = _torch(robot.data.body_link_pos_w)[
                :, self._contact_body_ids[side]
            ][:, shape_to_link]
        # Reduce collision shapes to links: sum forces, force-weighted position.
        magnitude = torch.linalg.vector_norm(forces, dim=-1, keepdim=True)
        link_forces = torch.zeros(
            forces.shape[0], expected, 3, device=forces.device, dtype=forces.dtype
        ).index_add_(1, shape_to_link, forces)
        weighted = torch.zeros_like(link_forces).index_add_(
            1, shape_to_link, shape_positions * magnitude
        )
        weight = torch.zeros(
            forces.shape[0], expected, 1, device=forces.device, dtype=forces.dtype
        ).index_add_(1, shape_to_link, magnitude)
        positions = weighted / weight.clamp_min(1.0e-9)
        forces = link_forces
        active = torch.linalg.vector_norm(forces, dim=-1) > float(
            self.cfg.contact_force_threshold
        )
        active &= self.steps_since_last_reset.unsqueeze(-1) > 0
        mask = active.unsqueeze(-1).to(forces.dtype)
        return (
            (positions * mask).unsqueeze(1),
            (forces * mask).unsqueeze(1),
            active.unsqueeze(1),
        )

    def _refresh_live_contact_cache(self) -> None:
        current_step = int(self._env.common_step_counter)
        if self._live_contact_cache_step == current_step:
            return
        valid = self.steps_since_last_reset > 0
        for side in ("right", "left"):
            adapter = self._contact_adapters[side]
            if adapter is None:
                self._live_contact_cache[side] = self._physx_live_contacts(side)
            else:
                pairs = adapter.refresh(valid_env_mask=valid)
                self._live_contact_cache[side] = (
                    pairs.positions_w,
                    pairs.forces_on_object_w,
                    pairs.active,
                )
        self._live_contact_cache_step = current_step

    def _live_contacts_grouped(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._refresh_live_contact_cache()
        return self._live_contact_cache[side]

    def _reference_contacts_grouped(
        self, side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._reference_contacts[side]
        positions = data["object_positions_w"][self.motion_index, self.timestep_counter]
        normals = data["link_normals_w"][self.motion_index, self.timestep_counter]
        active = data["active"][self.motion_index, self.timestep_counter]
        mask = active.unsqueeze(-1).to(positions.dtype)
        return (
            (positions * mask).unsqueeze(1),
            (normals * mask).unsqueeze(1),
            active.unsqueeze(1),
        )

    def _refresh_object_com_offset(self) -> None:
        com_b = _torch(self.object.data.body_com_pos_b)[:, 0]
        identity = torch.zeros((com_b.shape[0], 4), device=self.device)
        identity[:, 0] = 1.0
        self._object_com_pose_b = torch.cat((com_b, identity), dim=-1)

    @property
    def object_com_position_and_wxyz_w(self) -> torch.Tensor:
        pose = _torch(self.object.data.root_com_pose_w)
        return torch.cat((pose[:, :3], xyzw_to_wxyz(pose[:, 3:7])), dim=-1)[:, None]

    def _refresh_support_cache(self) -> None:
        current_step = int(self._env.common_step_counter)
        if (
            self._support_cache_step == current_step
            and self._support_cache_frames is not None
            and self._support_cache_motions is not None
            and torch.equal(self._support_cache_frames, self.timestep_counter)
            and torch.equal(self._support_cache_motions, self.motion_index)
        ):
            return
        live_com = self.object_com_position_and_wxyz_w
        reference_object = self._select(self._object_pose)
        reference_object_w = reference_object.clone()
        reference_object_w[:, :3] += self._env.scene.env_origins
        reference_com_position, reference_com_wxyz = mdp.compose_pose(
            reference_object_w[:, None, :3],
            reference_object_w[:, None, 3:7],
            self._object_com_pose_b[:, None, :3],
            self._object_com_pose_b[:, None, 3:7],
        )
        assert reference_com_wxyz is not None
        edges = int(self.cfg.num_friction_cone_edges)
        friction = float(self.cfg.friction_coefficient)
        for side in ("right", "left"):
            ref_position, ref_normal, ref_active = self._reference_contacts_grouped(
                side
            )
            ref_position = ref_position + self._env.scene.env_origins[:, None, None]
            self._command_supports[side] = mdp.compute_contact_wrench_supports(
                ref_position,
                ref_normal,
                reference_com_position,
                reference_com_wxyz,
                self.wrench_space_bases,
                self.object_radii,
                contact_active=ref_active,
                num_friction_cone_edges=edges,
                friction_coefficient=friction,
            )
            live_position, live_force, live_active = self._live_contacts_grouped(side)
            self._current_supports[side] = mdp.compute_contact_wrench_supports(
                live_position,
                live_force,
                live_com[..., :3],
                live_com[..., 3:7],
                self.wrench_space_bases,
                self.object_radii,
                contact_active=live_active,
                num_friction_cone_edges=edges,
                friction_coefficient=friction,
            )
        self._support_cache_step = current_step
        self._support_cache_frames = self.timestep_counter.clone()
        self._support_cache_motions = self.motion_index.clone()

    @property
    def right_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._command_supports["right"]

    @property
    def left_hand_contact_wrench_supports_command(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._command_supports["left"]

    @property
    def right_hand_contact_wrench_supports(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._current_supports["right"]

    @property
    def left_hand_contact_wrench_supports(self) -> torch.Tensor:
        self._refresh_support_cache()
        return self._current_supports["left"]

    @property
    def right_hand_object_contact_positions_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("right")[0]

    @property
    def left_hand_object_contact_positions_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("left")[0]

    @property
    def right_hand_object_contact_forces_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("right")[1]

    @property
    def left_hand_object_contact_forces_w(self) -> torch.Tensor:
        return self._live_contacts_grouped("left")[1]

    @property
    def right_hand_object_contact_active(self) -> torch.Tensor:
        return self._live_contacts_grouped("right")[2]

    @property
    def left_hand_object_contact_active(self) -> torch.Tensor:
        return self._live_contacts_grouped("left")[2]

    # ------------------------------------------------------------------
    # Released observation conventions
    # ------------------------------------------------------------------

    @property
    def retargeted_horizon(self) -> torch.Tensor:
        return self._lengths[self.motion_index]

    def _root_wxyz(self, robot: Any) -> torch.Tensor:
        wxyz = super()._root_wxyz(robot)
        return mdp.quat_unique(wxyz) if self.cfg.make_quat_unique else wxyz

    @property
    def right_hand_wrist_position_w(self) -> torch.Tensor:
        return _torch(self.right_robot.data.root_link_pos_w)

    @property
    def left_hand_wrist_position_w(self) -> torch.Tensor:
        return _torch(self.left_robot.data.root_link_pos_w)

    def _finger_joint_pos_scaled(self, side: str) -> torch.Tensor:
        robot = self.right_robot if side == "right" else self.left_robot
        ids = (
            self.right_finger_joint_ids
            if side == "right"
            else self.left_finger_joint_ids
        )
        limits = _torch(robot.data.joint_pos_limits)[:, ids]
        position = _torch(robot.data.joint_pos)[:, ids]
        lower = limits[..., 0]
        upper = limits[..., 1]
        return 2.0 * (position - lower) / (upper - lower).clamp_min(1.0e-6) - 1.0

    @property
    def right_hand_finger_joint_pos_scaled(self) -> torch.Tensor:
        return self._finger_joint_pos_scaled("right")

    @property
    def left_hand_finger_joint_pos_scaled(self) -> torch.Tensor:
        return self._finger_joint_pos_scaled("left")

    @property
    def command(self) -> torch.Tensor:
        """Released layout: relative wrist poses, finger deltas, relative object pose."""

        right_delta = mdp.pose_delta_command(
            self.right_hand_wrist_position_e,
            self.right_hand_wrist_wxyz_e,
            self.right_hand_wrist_pose_command_e[:, :3],
            self.right_hand_wrist_pose_command_e[:, 3:7],
            unique=False,
        )
        left_delta = mdp.pose_delta_command(
            self.left_hand_wrist_position_e,
            self.left_hand_wrist_wxyz_e,
            self.left_hand_wrist_pose_command_e[:, :3],
            self.left_hand_wrist_pose_command_e[:, 3:7],
            unique=False,
        )
        object_delta = mdp.pose_delta_command(
            self.object_position_e[:, 0],
            self.object_orientation_e[:, 0],
            self.object_body_position_command_e[:, 0],
            self.object_body_wxyz_command_e[:, 0],
            unique=False,
        )
        return torch.cat(
            (
                right_delta,
                left_delta,
                self.right_hand_finger_joint_pos_command
                - self.right_hand_finger_joint_pos,
                self.left_hand_finger_joint_pos_command
                - self.left_hand_finger_joint_pos,
                object_delta[:, :3],
                object_delta[:, 3:7],
            ),
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Released reset and update rules
    # ------------------------------------------------------------------

    def _sample_start_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        return mdp.sample_reference_start_frames(
            lengths,
            always_reset_to_first_frame=bool(self.cfg.always_reset_to_first_frame),
            reset_to_first_frame_prob=float(self.cfg.reset_to_first_frame_prob),
            virtual_object_control_scale=self.virtual_object_controller_curriculum_scale,
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        # Reset invalidates the cached contact and support tensors.
        self._live_contact_cache_step = -1
        self._support_cache_step = -1

    def _update_command(self) -> None:
        self.steps_since_last_reset += 1
        decay_steps = int(self.cfg.virtual_object_control_decay_steps)
        settled = self.steps_since_last_reset >= decay_steps
        scale = self.virtual_object_controller_curriculum_scale.to(torch.float32)
        self.virtual_object_controller_scale_factor_per_env[settled, 0] = scale
        horizon = self._lengths[self.motion_index]
        advance = settled & (self.timestep_counter < horizon - 1)
        self.trajectory_manager.advance_cursors(
            torch.nonzero(advance, as_tuple=False).flatten()
        )

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["right_hand_wrist_wxyz_error"] = mdp.quat_error_magnitude(
            self.right_hand_wrist_wxyz_e, self.right_hand_wrist_pose_command_e[:, 3:7]
        )
        self.metrics["left_hand_wrist_wxyz_error"] = mdp.quat_error_magnitude(
            self.left_hand_wrist_wxyz_e, self.left_hand_wrist_pose_command_e[:, 3:7]
        )
        self.metrics["right_hand_finger_joints_error"] = torch.linalg.vector_norm(
            self.right_hand_finger_joint_pos_command - self.right_hand_finger_joint_pos,
            dim=-1,
        )
        self.metrics["left_hand_finger_joints_error"] = torch.linalg.vector_norm(
            self.left_hand_finger_joint_pos_command - self.left_hand_finger_joint_pos,
            dim=-1,
        )
        self.metrics["object_body_wxyz_error"] = mdp.quat_error_magnitude(
            self.object_orientation_e[:, 0], self.object_body_wxyz_command_e[:, 0]
        )


def load_references(manifest: str) -> tuple[Any, ...]:
    """Load the reference set once more for the source-parity contact data."""

    from iltools.core import load_dexterous_reference_set

    return tuple(load_dexterous_reference_set(manifest))


SharpaSourceReferenceCommandCfg.class_type = SharpaSourceReferenceCommand

__all__ = ["SharpaSourceReferenceCommand", "SharpaSourceReferenceCommandCfg"]
