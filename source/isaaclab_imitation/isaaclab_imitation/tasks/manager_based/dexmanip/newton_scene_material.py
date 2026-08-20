"""Fail-closed Newton material override for ILTools scene assets.

USD material binding is not reliable for instanced meshes in the Newton import
path.  The Vega-Wuji task therefore reapplies the ILTools ``ScenePhysics``
contract to Newton's public per-shape arrays after the model is finalized.

Newton/MJWarp exposes one sliding-friction coefficient per shape, whereas the
ILTools contract records both static and dynamic friction.  This adapter uses
the declared *dynamic* friction for ``shape_material_mu`` and preserves the
declared restitution exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any


NEWTON_COLLIDE_SHAPES_FLAG = 1 << 1
"""Public ``newton.ShapeFlags.COLLIDE_SHAPES`` value."""


@dataclass(frozen=True)
class NewtonSceneShapeSelection:
    """Colliding shape IDs grouped by environment and declared asset order."""

    object_shape_ids_by_env: tuple[tuple[tuple[int, ...], ...], ...]
    support_shape_ids_by_env: tuple[tuple[tuple[int, ...], ...], ...]


def _asset_name(value: str, *, field_name: str) -> str:
    name = str(value)
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError(f"{field_name} entries must be non-empty path segments.")
    return name


def _path_is_root_or_descendant(label: str, root: str) -> bool:
    normalized = str(label).rstrip("/")
    return normalized == root or normalized.startswith(root + "/")


def select_scene_collision_shape_ids(
    shape_labels: Sequence[str],
    shape_flags: Sequence[int],
    *,
    num_envs: int,
    object_names: Sequence[str],
    support_surface_names: Sequence[str],
    collide_shapes_flag: int = NEWTON_COLLIDE_SHAPES_FLAG,
    env_root_path: str = "/World/envs",
) -> NewtonSceneShapeSelection:
    """Match every replicated scene asset by its exact full-path subtree.

    A requested asset is accepted only when every environment has at least one
    colliding shape whose label equals ``<env root>/<asset>`` or is below that
    root.  Leaf-name and substring fallbacks are deliberately forbidden: they
    could silently change a similarly named robot or scene asset.
    """

    if len(shape_labels) != len(shape_flags):
        raise ValueError("shape_labels and shape_flags must have equal lengths.")
    if int(num_envs) <= 0:
        raise ValueError("num_envs must be positive.")
    if int(collide_shapes_flag) <= 0:
        raise ValueError("collide_shapes_flag must be positive.")
    env_root = str(env_root_path).rstrip("/")
    if not env_root.startswith("/") or not env_root:
        raise ValueError("env_root_path must be a non-empty absolute path.")

    objects = tuple(
        _asset_name(name, field_name="object_names") for name in object_names
    )
    supports = tuple(
        _asset_name(name, field_name="support_surface_names")
        for name in support_surface_names
    )
    requested_names = objects + supports
    if not requested_names:
        raise ValueError("At least one scene object or support surface is required.")
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("Scene object and support names must be globally unique.")

    colliding_ids = tuple(
        shape_id
        for shape_id, flags in enumerate(shape_flags)
        if int(flags) & int(collide_shapes_flag)
    )

    def select_kind(names: tuple[str, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
        by_env: list[tuple[tuple[int, ...], ...]] = []
        for env_id in range(int(num_envs)):
            env_path = f"{env_root}/env_{env_id}"
            per_asset: list[tuple[int, ...]] = []
            for name in names:
                asset_root = f"{env_path}/{name}"
                matches = tuple(
                    shape_id
                    for shape_id in colliding_ids
                    if _path_is_root_or_descendant(
                        str(shape_labels[shape_id]), asset_root
                    )
                )
                if not matches:
                    raise ValueError(
                        "Newton scene material selection found no colliding shape "
                        f"at {asset_root!r} or below it."
                    )
                per_asset.append(matches)
            by_env.append(tuple(per_asset))
        return tuple(by_env)

    return NewtonSceneShapeSelection(
        object_shape_ids_by_env=select_kind(objects),
        support_shape_ids_by_env=select_kind(supports),
    )


def build_scene_material_assignments(
    selection: NewtonSceneShapeSelection,
    *,
    object_dynamic_friction: Sequence[float],
    object_restitution: Sequence[float],
    support_dynamic_friction: Sequence[float],
    support_restitution: Sequence[float],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    """Expand per-asset ILTools values into deterministic per-shape writes."""

    object_mu = tuple(float(value) for value in object_dynamic_friction)
    object_bounce = tuple(float(value) for value in object_restitution)
    support_mu = tuple(float(value) for value in support_dynamic_friction)
    support_bounce = tuple(float(value) for value in support_restitution)
    if len(object_mu) != len(object_bounce):
        raise ValueError("Object friction and restitution counts differ.")
    if len(support_mu) != len(support_bounce):
        raise ValueError("Support friction and restitution counts differ.")

    for name, values in (
        ("object_dynamic_friction", object_mu),
        ("support_dynamic_friction", support_mu),
    ):
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"{name} values must be finite and non-negative.")
    for name, values in (
        ("object_restitution", object_bounce),
        ("support_restitution", support_bounce),
    ):
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{name} values must be finite and in [0, 1].")

    assignments: list[tuple[int, float, float]] = []

    def append_kind(
        shape_ids_by_env: tuple[tuple[tuple[int, ...], ...], ...],
        friction: tuple[float, ...],
        restitution: tuple[float, ...],
        *,
        kind: str,
    ) -> None:
        for env_shapes in shape_ids_by_env:
            if len(env_shapes) != len(friction):
                raise ValueError(
                    f"{kind} shape groups do not align with material values."
                )
            for asset_index, shape_ids in enumerate(env_shapes):
                assignments.extend(
                    (int(shape_id), friction[asset_index], restitution[asset_index])
                    for shape_id in shape_ids
                )

    append_kind(
        selection.object_shape_ids_by_env,
        object_mu,
        object_bounce,
        kind="Object",
    )
    append_kind(
        selection.support_shape_ids_by_env,
        support_mu,
        support_bounce,
        kind="Support",
    )
    shape_ids = tuple(item[0] for item in assignments)
    if len(set(shape_ids)) != len(shape_ids):
        raise ValueError("A Newton shape matched more than one scene material asset.")
    return (
        shape_ids,
        tuple(item[1] for item in assignments),
        tuple(item[2] for item in assignments),
    )


def configure_scene_material(env: Any, env_ids: Any | None) -> None:
    """Apply the ILTools scene material contract to the live Newton model."""

    if env_ids is not None:
        raise ValueError("The scene material event supports startup mode only.")

    scene_physics = getattr(env, "_scene_physics", None)
    if scene_physics is None:
        if bool(getattr(env.cfg, "require_training_qualified_reference", False)):
            raise RuntimeError(
                "Training-qualified Vega-Wuji references must declare ScenePhysics."
            )
        # Explicit inspection mode remains backward-compatible with legacy
        # files which have no typed physics contract.
        return

    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverNotifyFlags

    object_names = tuple(str(name) for name in env._reference_object_names)
    support_names = tuple(str(name) for name in env._reference_support_surface_names)
    scene_physics.validate(
        object_count=len(object_names), support_count=len(support_names)
    )

    model = NewtonManager.get_model()
    if model is None:
        raise RuntimeError(
            "The Newton model is not ready for the scene material event."
        )
    required_fields = (
        "shape_label",
        "shape_flags",
        "shape_material_mu",
        "shape_material_restitution",
    )
    missing_fields = [
        name for name in required_fields if getattr(model, name, None) is None
    ]
    if missing_fields:
        raise RuntimeError(
            "The Newton model is missing scene material fields: "
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
    shape_ids, friction, restitution = build_scene_material_assignments(
        selection,
        object_dynamic_friction=scene_physics.object_dynamic_friction,
        object_restitution=scene_physics.object_restitution,
        support_dynamic_friction=scene_physics.support_dynamic_friction,
        support_restitution=scene_physics.support_restitution,
    )

    material_mu = wp.to_torch(model.shape_material_mu)
    material_restitution = wp.to_torch(model.shape_material_restitution)
    if material_mu.ndim != 1 or material_mu.numel() != len(shape_labels):
        raise RuntimeError("Newton friction and shape-label counts differ.")
    if material_restitution.ndim != 1 or material_restitution.numel() != len(
        shape_labels
    ):
        raise RuntimeError("Newton restitution and shape-label counts differ.")
    indices = torch.as_tensor(shape_ids, dtype=torch.long, device=material_mu.device)
    expected_mu = torch.as_tensor(
        friction, dtype=material_mu.dtype, device=material_mu.device
    )
    expected_restitution = torch.as_tensor(
        restitution,
        dtype=material_restitution.dtype,
        device=material_restitution.device,
    )
    material_mu[indices] = expected_mu
    material_restitution[indices] = expected_restitution
    NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)

    if not torch.equal(material_mu.index_select(0, indices), expected_mu):
        raise RuntimeError("Newton scene friction readback differs from ScenePhysics.")
    if not torch.equal(
        material_restitution.index_select(0, indices), expected_restitution
    ):
        raise RuntimeError(
            "Newton scene restitution readback differs from ScenePhysics."
        )


__all__ = [
    "NEWTON_COLLIDE_SHAPES_FLAG",
    "NewtonSceneShapeSelection",
    "build_scene_material_assignments",
    "configure_scene_material",
    "select_scene_collision_shape_ids",
]
