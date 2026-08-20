# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Newton contact-point and force adapter for the dexmanip task.

The Isaac Lab Newton contact sensor does not publish contact positions. Newton
1.2.1 keeps the required per-contact data in its public ``Contacts`` buffer.
This adapter reads that buffer after a physics step. It averages object-side
contact points and sums forces for each environment, object, and hand link.
This matches the aggregation used by the Isaac Lab PhysX contact sensor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager


@dataclass(frozen=True)
class NewtonContactPairTensors:
    """Aggregated contact data in environment, object, and link order."""

    positions_w: torch.Tensor
    """Average object-side contact positions, shape ``(N, O, L, 3)``."""

    forces_on_object_w: torch.Tensor
    """Sum of forces on each object, shape ``(N, O, L, 3)``."""

    active: torch.Tensor
    """Force-threshold contact mask, shape ``(N, O, L)``."""

    counts: torch.Tensor
    """Number of manifold points in each pair, shape ``(N, O, L)``."""


@wp.kernel(enable_backward=False)
def _aggregate_contact_pairs_kernel(
    rigid_contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0_b: wp.array[wp.vec3],
    contact_point1_b: wp.array[wp.vec3],
    contact_force_on_body0: wp.array[wp.spatial_vector],  # ty: ignore[invalid-type-form]
    shape_body: wp.array[wp.int32],
    body_q_w: wp.array[wp.transform],
    body_to_env: wp.array[wp.int32],
    hand_body_to_link: wp.array[wp.int32],
    object_body_to_index: wp.array[wp.int32],
    valid_env_mask: wp.array[wp.bool],
    num_objects: int,
    num_links: int,
    position_sums_w: wp.array[wp.vec3],
    force_sums_on_object_w: wp.array[wp.vec3],
    contact_counts: wp.array[wp.int32],
):
    """Collect active Newton contacts into flat pair buffers."""

    contact_index = wp.tid()
    if contact_index >= rigid_contact_count[0]:
        return

    shape0 = contact_shape0[contact_index]
    shape1 = contact_shape1[contact_index]
    if shape0 < 0 or shape1 < 0:
        return
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    if body0 < 0 or body1 < 0:
        return

    link0 = hand_body_to_link[body0]
    link1 = hand_body_to_link[body1]
    object0 = object_body_to_index[body0]
    object1 = object_body_to_index[body1]

    env = -1
    link = -1
    object_index = -1
    point_w = wp.vec3(0.0)
    force_on_object_w = wp.vec3(0.0)
    linear_force_on_body0 = wp.spatial_top(contact_force_on_body0[contact_index])

    if link0 >= 0 and object1 >= 0:
        env = body_to_env[body0]
        if env != body_to_env[body1]:
            return
        link = link0
        object_index = object1
        point_w = wp.transform_point(body_q_w[body1], contact_point1_b[contact_index])
        force_on_object_w = linear_force_on_body0 * -1.0  # ty: ignore[unsupported-operator]
    elif object0 >= 0 and link1 >= 0:
        env = body_to_env[body0]
        if env != body_to_env[body1]:
            return
        link = link1
        object_index = object0
        point_w = wp.transform_point(body_q_w[body0], contact_point0_b[contact_index])
        force_on_object_w = linear_force_on_body0
    else:
        return

    if env < 0 or not valid_env_mask[env]:
        return
    pair_index = (env * num_objects + object_index) * num_links + link
    wp.atomic_add(position_sums_w, pair_index, point_w)  # ty: ignore[invalid-argument-type]
    wp.atomic_add(
        force_sums_on_object_w,  # ty: ignore[invalid-argument-type]
        pair_index,
        force_on_object_w,
    )
    wp.atomic_add(contact_counts, pair_index, 1)  # ty: ignore[invalid-argument-type]


@wp.kernel(enable_backward=False)
def _finalize_contact_pairs_kernel(
    position_sums_w: wp.array[wp.vec3],
    force_sums_on_object_w: wp.array[wp.vec3],
    contact_counts: wp.array[wp.int32],
    force_threshold: wp.float32,
    positions_w: wp.array[wp.vec3],
    active: wp.array[wp.bool],
):
    """Average positions and set the force-threshold mask."""

    pair_index = wp.tid()
    count = contact_counts[pair_index]
    if count > 0:
        positions_w[pair_index] = (  # ty: ignore[invalid-assignment]
            position_sums_w[pair_index] / wp.float32(count)
        )
    else:
        positions_w[pair_index] = wp.vec3(0.0)  # ty: ignore[invalid-assignment]
    force = force_sums_on_object_w[pair_index]
    active[pair_index] = (  # ty: ignore[invalid-assignment]
        count > 0 and wp.length_sq(force) > force_threshold * force_threshold
    )


def _as_id_rows(
    value: Sequence[Sequence[int]] | torch.Tensor, name: str
) -> list[list[int]]:
    """Convert one body-ID matrix to a checked host list."""

    if isinstance(value, torch.Tensor):
        if value.ndim != 2:
            raise ValueError(f"{name} must have two dimensions.")
        rows = value.detach().cpu().tolist()
    else:
        rows = [list(row) for row in value]
    if not rows or not rows[0]:
        raise ValueError(f"{name} must not be empty.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} rows must have equal lengths.")
    return [[int(item) for item in row] for row in rows]


def build_contact_body_maps(
    body_count: int,
    link_body_ids: Sequence[Sequence[int]] | torch.Tensor,
    object_body_ids: Sequence[Sequence[int]] | torch.Tensor,
) -> tuple[list[int], list[int], list[int]]:
    """Build body-to-environment, body-to-link, and body-to-object maps."""

    links = _as_id_rows(link_body_ids, "link_body_ids")
    objects = _as_id_rows(object_body_ids, "object_body_ids")
    if len(links) != len(objects):
        raise ValueError(
            "Link and object body-ID matrices must have the same row count."
        )
    body_to_env = [-1] * body_count
    hand_body_to_link = [-1] * body_count
    object_body_to_index = [-1] * body_count

    def check_body(body_id: int) -> None:
        if body_id < 0 or body_id >= body_count:
            raise ValueError(
                f"Body ID {body_id} is outside the model range [0, {body_count})."
            )
        if body_to_env[body_id] >= 0:
            raise ValueError(f"Body ID {body_id} occurs more than once.")

    for env, row in enumerate(links):
        for link, body_id in enumerate(row):
            check_body(body_id)
            body_to_env[body_id] = env
            hand_body_to_link[body_id] = link
    for env, row in enumerate(objects):
        for object_index, body_id in enumerate(row):
            check_body(body_id)
            body_to_env[body_id] = env
            object_body_to_index[body_id] = object_index
    return body_to_env, hand_body_to_link, object_body_to_index


def _body_ids_from_names(
    labels: Sequence[str],
    worlds: Sequence[int],
    names: Sequence[str],
    num_envs: int,
    allowed_body_ids: set[int],
    kind: str,
) -> list[list[int]]:
    """Resolve one ordered body ID per environment and requested body name.

    Robot link names normally match the body-label leaf. A rigid object can
    have a generic leaf such as ``base`` below a scene-object prim. In that
    case, the scene-object name matches one complete path segment.
    """

    resolved = [[-1 for _ in names] for _ in range(num_envs)]
    name_to_slot = {name: slot for slot, name in enumerate(names)}
    if len(name_to_slot) != len(names):
        raise ValueError(f"{kind} names must be unique.")
    for body_id in sorted(allowed_body_ids):
        world = int(worlds[body_id])
        if world < 0 or world >= num_envs:
            continue
        label = str(labels[body_id])
        leaf = label.rsplit("/", 1)[-1]
        slot = name_to_slot.get(label, name_to_slot.get(leaf))
        if slot is None:
            segment_slots = {
                name_to_slot[segment]
                for segment in label.split("/")
                if segment in name_to_slot
            }
            if len(segment_slots) > 1:
                raise ValueError(
                    f"Newton body label {label!r} matches more than one requested "
                    f"{kind} name."
                )
            if segment_slots:
                slot = segment_slots.pop()
        if slot is None:
            continue
        if resolved[world][slot] >= 0:
            raise ValueError(
                f"Newton world {world} contains more than one {kind} body named "
                f"{names[slot]!r}."
            )
        resolved[world][slot] = body_id
    for world, row in enumerate(resolved):
        missing = [names[index] for index, body_id in enumerate(row) if body_id < 0]
        if missing:
            available = sorted(
                str(labels[body_id])
                for body_id in allowed_body_ids
                if int(worlds[body_id]) == world
            )
            raise ValueError(
                f"Newton world {world} is missing {kind} bodies: {missing}. "
                f"Available labels: {available}."
            )
    return resolved


def _rows_from_environment_major_ids(
    body_ids: Sequence[int],
    *,
    num_envs: int,
    row_width: int,
    name: str,
) -> list[list[int]]:
    """Split one environment-major body-ID sequence into checked rows."""

    expected = num_envs * row_width
    if len(body_ids) != expected:
        raise ValueError(f"{name} has {len(body_ids)} entries; expected {expected}.")
    return [
        [int(body_id) for body_id in body_ids[start : start + row_width]]
        for start in range(0, expected, row_width)
    ]


class NewtonContactPairAdapter:
    """Read exact Newton contacts into the per-link tensor layout.

    Construct this adapter after an Isaac Lab Newton contact sensor. The sensor
    requests the optional Newton ``force`` contact attribute. Call ``refresh``
    only after a physics step. For post-reset observations, pass a false entry
    in ``valid_env_mask`` until that environment completes a physics step.
    """

    def __init__(
        self,
        link_body_ids: Sequence[Sequence[int]] | torch.Tensor,
        object_body_ids: Sequence[Sequence[int]] | torch.Tensor,
        *,
        force_threshold: float = 0.1,
    ) -> None:
        """Create body maps and persistent aggregation buffers."""

        if force_threshold < 0.0:
            raise ValueError("force_threshold must be zero or positive.")
        model = NewtonManager.get_model()
        contacts = NewtonManager.get_contacts()
        state = NewtonManager.get_state_0()
        if contacts is None:
            raise RuntimeError("The active Newton solver does not provide contacts.")
        if contacts.force is None:
            raise RuntimeError(
                "Newton contact force storage is not active. Construct an Isaac Lab "
                "Newton contact sensor before this adapter."
            )
        if model.shape_body is None or state.body_q is None:
            raise RuntimeError(
                "The Newton model does not provide rigid-body contact data."
            )

        links = _as_id_rows(link_body_ids, "link_body_ids")
        objects = _as_id_rows(object_body_ids, "object_body_ids")
        self.num_envs = len(links)
        self.num_links = len(links[0])
        self.num_objects = len(objects[0])
        manager_num_envs = NewtonManager.get_num_envs()
        if manager_num_envs is not None and self.num_envs != int(manager_num_envs):
            raise ValueError(
                f"Body maps contain {self.num_envs} environments, but Newton has "
                f"{manager_num_envs}."
            )
        body_to_env, hand_body_to_link, object_body_to_index = build_contact_body_maps(
            int(model.body_count), links, objects
        )
        self._device = model.device
        self._force_threshold = float(force_threshold)
        self._body_to_env = wp.array(body_to_env, dtype=wp.int32, device=self._device)
        self._hand_body_to_link = wp.array(
            hand_body_to_link, dtype=wp.int32, device=self._device
        )
        self._object_body_to_index = wp.array(
            object_body_to_index, dtype=wp.int32, device=self._device
        )
        self._all_envs_valid = wp.full(
            self.num_envs, True, dtype=wp.bool, device=self._device
        )
        self._valid_env_mask_torch: torch.Tensor | None = None

        pair_count = self.num_envs * self.num_objects * self.num_links
        self._position_sums_w = wp.zeros(pair_count, dtype=wp.vec3, device=self._device)
        self._positions_w = wp.zeros(pair_count, dtype=wp.vec3, device=self._device)
        self._force_sums_on_object_w = wp.zeros(
            pair_count, dtype=wp.vec3, device=self._device
        )
        self._contact_counts = wp.zeros(pair_count, dtype=wp.int32, device=self._device)
        self._active = wp.zeros(pair_count, dtype=wp.bool, device=self._device)

        self._positions_torch = wp.to_torch(self._positions_w).reshape(
            self.num_envs, self.num_objects, self.num_links, 3
        )
        self._forces_torch = wp.to_torch(self._force_sums_on_object_w).reshape(
            self.num_envs, self.num_objects, self.num_links, 3
        )
        self._active_torch = wp.to_torch(self._active).reshape(
            self.num_envs, self.num_objects, self.num_links
        )
        self._counts_torch = wp.to_torch(self._contact_counts).reshape(
            self.num_envs, self.num_objects, self.num_links
        )

    @classmethod
    def from_contact_sensor(
        cls,
        sensor: Any,
        link_body_names: Sequence[str],
        object_body_names: Sequence[str],
        *,
        force_threshold: float = 0.1,
    ) -> NewtonContactPairAdapter:
        """Resolve ordered global body IDs from one Isaac Lab Newton sensor."""

        view = sensor.contact_view
        model = NewtonManager.get_model()
        num_sensors = int(sensor.num_sensors)
        if num_sensors <= 0:
            raise ValueError("The Newton contact sensor has no sensing bodies.")
        total_sensors = int(view.total_force.shape[0])
        if total_sensors % num_sensors:
            raise ValueError(
                f"Newton has {total_sensors} sensing bodies, which is not divisible "
                f"by the per-environment count {num_sensors}."
            )
        num_envs = total_sensors // num_sensors
        shape_body_array = model.shape_body
        if shape_body_array is None:
            raise RuntimeError("The Newton model does not provide shape-to-body maps.")
        shape_body = shape_body_array.numpy()
        if view.sensing_obj_type == "body":
            sensing_body_sequence = [int(index) for index in view.sensing_obj_idx]
        elif view.sensing_obj_type == "shape":
            sensing_body_sequence = [
                int(shape_body[index]) for index in view.sensing_obj_idx
            ]
        else:
            raise ValueError(
                f"Unsupported sensing object type: {view.sensing_obj_type!r}."
            )
        sensing_rows = _rows_from_environment_major_ids(
            sensing_body_sequence,
            num_envs=num_envs,
            row_width=num_sensors,
            name="Newton sensing-body sequence",
        )
        sensor_names = tuple(str(name) for name in sensor.sensor_names or ())
        if len(sensor_names) != num_sensors or len(set(sensor_names)) != num_sensors:
            raise ValueError("Newton contact sensor names must be unique and complete.")
        missing_links = [name for name in link_body_names if name not in sensor_names]
        if missing_links:
            raise ValueError(
                f"Newton contact sensor is missing hand links: {missing_links}."
            )
        link_columns = [sensor_names.index(name) for name in link_body_names]
        links = [[row[column] for column in link_columns] for row in sensing_rows]

        raw_counterpart_rows = view.counterpart_indices
        if len(raw_counterpart_rows) != total_sensors:
            raise ValueError(
                "Newton contact counterpart rows must align with sensing bodies."
            )
        if view.counterpart_type == "body":
            counterpart_rows = [
                [int(index) for index in row] for row in raw_counterpart_rows
            ]
        elif view.counterpart_type == "shape":
            counterpart_rows = [
                [int(shape_body[index]) for index in row]
                for row in raw_counterpart_rows
            ]
        else:
            raise ValueError("The Newton contact sensor must have object counterparts.")

        object_worlds = [-1] * int(model.body_count)
        object_body_ids: set[int] = set()
        for env, start in enumerate(range(0, total_sensors, num_sensors)):
            env_rows = counterpart_rows[start : start + num_sensors]
            expected_ids = set(env_rows[0])
            if any(set(row) != expected_ids for row in env_rows[1:]):
                raise ValueError(
                    f"Newton environment {env} has inconsistent object counterparts."
                )
            for body_id in expected_ids:
                if body_id < 0 or body_id >= int(model.body_count):
                    raise ValueError(
                        f"Newton object body ID {body_id} is out of range."
                    )
                if object_worlds[body_id] not in {-1, env}:
                    raise ValueError(
                        f"Newton object body ID {body_id} occurs in multiple environments."
                    )
                object_worlds[body_id] = env
                object_body_ids.add(body_id)
        objects = _body_ids_from_names(
            model.body_label,
            object_worlds,
            object_body_names,
            num_envs,
            object_body_ids,
            "object",
        )
        return cls(links, objects, force_threshold=force_threshold)

    def refresh(
        self, valid_env_mask: torch.Tensor | None = None
    ) -> NewtonContactPairTensors:
        """Aggregate the latest post-step contact buffer without a host copy."""

        contacts = NewtonManager.get_contacts()
        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        if contacts is None or contacts.force is None:
            raise RuntimeError("Newton contact force storage is not available.")
        if state.body_q is None or model.shape_body is None:
            raise RuntimeError("Newton rigid-body contact data is not available.")

        if valid_env_mask is None:
            valid_mask_wp = self._all_envs_valid
            self._valid_env_mask_torch = None
        else:
            if valid_env_mask.shape != (self.num_envs,):
                raise ValueError(f"valid_env_mask must have shape ({self.num_envs},).")
            if valid_env_mask.device != self._positions_torch.device:
                raise ValueError("valid_env_mask must use the Newton device.")
            self._valid_env_mask_torch = valid_env_mask.to(
                dtype=torch.bool
            ).contiguous()
            valid_mask_wp = wp.from_torch(self._valid_env_mask_torch, dtype=wp.bool)

        self._position_sums_w.zero_()
        self._positions_w.zero_()
        self._force_sums_on_object_w.zero_()
        self._contact_counts.zero_()
        self._active.zero_()
        wp.launch(
            _aggregate_contact_pairs_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.force,
                model.shape_body,
                state.body_q,
                self._body_to_env,
                self._hand_body_to_link,
                self._object_body_to_index,
                valid_mask_wp,
                self.num_objects,
                self.num_links,
            ],
            outputs=[
                self._position_sums_w,
                self._force_sums_on_object_w,
                self._contact_counts,
            ],
            device=self._device,
        )
        wp.launch(
            _finalize_contact_pairs_kernel,
            dim=self._contact_counts.shape[0],
            inputs=[
                self._position_sums_w,
                self._force_sums_on_object_w,
                self._contact_counts,
                self._force_threshold,
            ],
            outputs=[self._positions_w, self._active],
            device=self._device,
        )
        return NewtonContactPairTensors(
            positions_w=self._positions_torch,
            forces_on_object_w=self._forces_torch,
            active=self._active_torch,
            counts=self._counts_torch,
        )


__all__ = [
    "NewtonContactPairAdapter",
    "NewtonContactPairTensors",
    "build_contact_body_maps",
]
