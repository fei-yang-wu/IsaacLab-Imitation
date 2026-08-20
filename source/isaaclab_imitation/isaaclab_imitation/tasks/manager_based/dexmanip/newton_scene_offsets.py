"""Fail-closed Newton contact-offset override for ILTools scene assets.

Isaac Lab applies ``NewtonCollisionPropertiesCfg`` through ``apply_nested``,
which skips every USD prim that is an instance.  A URDF-converted object keeps
its geometry under instanced meshes, so the declared contact margin and gap
never reach those prims and Isaac Lab only logs a warning.

Measured on the corn-can scene (2026-08-20): the shapes the spawner could not
reach carry ``COLLIDE_SHAPES = False``.  They are visual meshes, so their
``shape_gap`` of 0.1 m has no effect on contact.  Every colliding scene shape
already arrives with zero margin and zero gap, because the MJWarp contact path
clamps them.  The warning is therefore not a live defect today.

This module keeps that agreement true by construction instead of by luck.  It
restates the zero-offset contract on Newton's public per-shape arrays after
the model is finalized, in the same way ``newton_scene_material`` restates
friction and restitution, and verifies the write by readback.  A collision
pipeline that stops clamping, or an asset whose collider inherits a non-zero
gap, then fails loudly instead of adding a silent contact shell.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

from .newton_scene_material import (
    NewtonSceneShapeSelection,
    select_scene_collision_shape_ids,
)


def build_scene_contact_offset_assignments(
    selection: NewtonSceneShapeSelection,
    *,
    object_contact_margin: float,
    object_contact_gap: float,
    support_contact_margin: float,
    support_contact_gap: float,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    """Expand the declared offsets into deterministic per-shape writes."""

    values = {
        "object_contact_margin": float(object_contact_margin),
        "object_contact_gap": float(object_contact_gap),
        "support_contact_margin": float(support_contact_margin),
        "support_contact_gap": float(support_contact_gap),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    assignments: list[tuple[int, float, float]] = []

    def append_kind(
        shape_ids_by_env: tuple[tuple[tuple[int, ...], ...], ...],
        margin: float,
        gap: float,
    ) -> None:
        for env_shapes in shape_ids_by_env:
            for shape_ids in env_shapes:
                assignments.extend(
                    (int(shape_id), margin, gap) for shape_id in shape_ids
                )

    append_kind(
        selection.object_shape_ids_by_env,
        values["object_contact_margin"],
        values["object_contact_gap"],
    )
    append_kind(
        selection.support_shape_ids_by_env,
        values["support_contact_margin"],
        values["support_contact_gap"],
    )
    shape_ids = tuple(item[0] for item in assignments)
    if len(set(shape_ids)) != len(shape_ids):
        raise ValueError("A Newton shape matched more than one scene offset asset.")
    return (
        shape_ids,
        tuple(item[1] for item in assignments),
        tuple(item[2] for item in assignments),
    )


def configure_scene_contact_offsets(
    env: Any,
    env_ids: Any | None,
    object_contact_margin: float = 0.0,
    object_contact_gap: float = 0.0,
    support_contact_margin: float = 0.0,
    support_contact_gap: float = 0.0,
) -> None:
    """Apply the scene contact-offset contract to the live Newton model."""

    if env_ids is not None:
        raise ValueError("The scene offset event supports startup mode only.")

    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverNotifyFlags

    object_names = tuple(str(name) for name in env._reference_object_names)
    support_names = tuple(str(name) for name in env._reference_support_surface_names)

    model = NewtonManager.get_model()
    if model is None:
        raise RuntimeError("The Newton model is not ready for the scene offset event.")
    required_fields = ("shape_label", "shape_flags", "shape_margin", "shape_gap")
    missing_fields = [
        name for name in required_fields if getattr(model, name, None) is None
    ]
    if missing_fields:
        raise RuntimeError(
            "The Newton model is missing scene offset fields: "
            + ", ".join(missing_fields)
        )

    shape_flags = wp.to_torch(model.shape_flags)
    shape_labels = tuple(str(label) for label in model.shape_label)
    if shape_flags.ndim != 1 or shape_flags.numel() != len(shape_labels):
        raise RuntimeError("Newton shape flags and labels have incompatible shapes.")
    selection = select_scene_collision_shape_ids(
        shape_labels,
        shape_flags.detach().cpu().tolist(),
        num_envs=int(env.num_envs),
        object_names=object_names,
        support_surface_names=support_names,
    )
    shape_ids, margins, gaps = build_scene_contact_offset_assignments(
        selection,
        object_contact_margin=object_contact_margin,
        object_contact_gap=object_contact_gap,
        support_contact_margin=support_contact_margin,
        support_contact_gap=support_contact_gap,
    )

    shape_margin = wp.to_torch(model.shape_margin)
    shape_gap = wp.to_torch(model.shape_gap)
    for name, tensor in (("margin", shape_margin), ("gap", shape_gap)):
        if tensor.ndim != 1 or tensor.numel() != len(shape_labels):
            raise RuntimeError(f"Newton {name} and shape-label counts differ.")
    indices = torch.as_tensor(shape_ids, dtype=torch.long, device=shape_margin.device)
    expected_margin = torch.as_tensor(
        margins, dtype=shape_margin.dtype, device=shape_margin.device
    )
    expected_gap = torch.as_tensor(gaps, dtype=shape_gap.dtype, device=shape_gap.device)
    shape_margin[indices] = expected_margin
    shape_gap[indices] = expected_gap
    NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)

    if not torch.equal(shape_margin.index_select(0, indices), expected_margin):
        raise RuntimeError("Newton scene contact margin readback differs.")
    if not torch.equal(shape_gap.index_select(0, indices), expected_gap):
        raise RuntimeError("Newton scene contact gap readback differs.")


def scene_shape_offsets(
    shape_labels: Sequence[str],
    shape_margin: Sequence[float],
    shape_gap: Sequence[float],
) -> dict[str, tuple[float, float]]:
    """Return one ``label -> (margin, gap)`` record for inspection and tests."""

    if not (len(shape_labels) == len(shape_margin) == len(shape_gap)):
        raise ValueError("shape labels, margins, and gaps must have equal lengths.")
    return {
        str(label): (float(margin), float(gap))
        for label, margin, gap in zip(shape_labels, shape_margin, shape_gap)
    }


__all__ = [
    "build_scene_contact_offset_assignments",
    "configure_scene_contact_offsets",
    "scene_shape_offsets",
]
