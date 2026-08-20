"""Newton contact material override for the integrated Wuji hands.

The released CHORD task assigns one material to each complete hand robot at
startup. Vega and both Wuji hands are one articulation in this task. This
module therefore selects Newton collision shapes by their parent body names so
that Vega arm and scene-object materials stay unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import random
from typing import Any


NEWTON_COLLIDE_SHAPES_FLAG = 1 << 1
"""Public ``newton.ShapeFlags.COLLIDE_SHAPES`` value."""

WUJI_HAND_BODY_PREFIXES = ("l_", "r_")
"""Lower-case prefixes that distinguish Wuji bodies from Vega arm bodies."""

WUJI_HAND_BODY_NAMES = tuple(
    f"{side}_{body_name}"
    for side in ("l", "r")
    for body_name in (
        "mount",
        "wrist",
        "thumb_proximal",
        "thumb_proximal_abd",
        "thumb_middle",
        "thumb_distal",
        "index_finger_proximal",
        "index_finger_proximal_abd",
        "index_finger_middle",
        "index_finger_distal",
        "middle_finger_proximal",
        "middle_finger_proximal_abd",
        "middle_finger_middle",
        "middle_finger_distal",
        "ring_finger_proximal",
        "ring_finger_proximal_abd",
        "ring_finger_middle",
        "ring_finger_distal",
        "pinky_proximal",
        "pinky_proximal_abd",
        "pinky_middle",
        "pinky_distal",
    )
)
"""Exact integrated Wuji body names from the validated robot MJCF."""


@dataclass(frozen=True)
class WujiHandMaterialSpec:
    """Released startup material distribution for Newton MJWarp."""

    static_friction_range: tuple[float, float] = (2.0, 2.01)
    dynamic_friction_range: tuple[float, float] = (2.0, 2.01)
    restitution_range: tuple[float, float] = (0.0, 0.0)
    num_buckets: int = 64

    def __post_init__(self) -> None:
        ranges = {
            "static_friction_range": self.static_friction_range,
            "dynamic_friction_range": self.dynamic_friction_range,
            "restitution_range": self.restitution_range,
        }
        for name, value_range in ranges.items():
            if len(value_range) != 2:
                raise ValueError(f"{name} must contain two values.")
            low, high = (float(value) for value in value_range)
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(f"{name} values must be finite.")
            if low > high:
                raise ValueError(f"{name} must be in low-to-high order.")
        if self.static_friction_range[0] < 0.0:
            raise ValueError("Wuji friction must be zero or positive.")
        if not 0.0 <= self.restitution_range[0] <= self.restitution_range[1] <= 1.0:
            raise ValueError("Wuji restitution must stay in [0, 1].")
        if self.static_friction_range != self.dynamic_friction_range:
            raise ValueError(
                "Newton MJWarp has one sliding-friction value per shape. "
                "Wuji static and dynamic friction ranges must be equal."
            )
        if self.num_buckets <= 0:
            raise ValueError("Wuji material num_buckets must be positive.")


def sample_bucketed_wuji_materials(
    shape_count: int,
    *,
    spec: WujiHandMaterialSpec = WujiHandMaterialSpec(),
    seed: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Sample one released material bucket per selected Newton shape."""

    if shape_count <= 0:
        raise ValueError("shape_count must be positive.")
    generator = random.Random(int(seed))
    buckets = tuple(
        (
            generator.uniform(*spec.static_friction_range),
            # Consume the dynamic-friction draw from the released bucket triple.
            # MJWarp has no separate field for this value.
            generator.uniform(*spec.dynamic_friction_range),
            generator.uniform(*spec.restitution_range),
        )
        for _ in range(spec.num_buckets)
    )
    bucket_ids = tuple(
        generator.randrange(spec.num_buckets) for _ in range(shape_count)
    )
    return (
        tuple(buckets[bucket_id][0] for bucket_id in bucket_ids),
        tuple(buckets[bucket_id][2] for bucket_id in bucket_ids),
    )


def _body_leaf(label: str) -> str:
    """Return the final body path segment from one Newton label."""

    return str(label).rstrip("/").rsplit("/", 1)[-1]


def select_wuji_collision_shape_ids(
    body_labels: Sequence[str],
    shape_body_ids: Sequence[int],
    shape_flags: Sequence[int],
    *,
    collide_shapes_flag: int = NEWTON_COLLIDE_SHAPES_FLAG,
    robot_path_segment: str = "Robot",
    required_body_names: Sequence[str] = WUJI_HAND_BODY_NAMES,
) -> tuple[int, ...]:
    """Select colliding shapes on Wuji bodies and fail on an incomplete match.

    Full Newton paths must contain the ``Robot`` articulation segment. If a
    backend publishes only leaf labels, exact MJCF body names are required.
    The exact-name fallback is the strongest possible scope without paths.
    """

    if collide_shapes_flag <= 0:
        raise ValueError("collide_shapes_flag must be positive.")
    if len(shape_body_ids) != len(shape_flags):
        raise ValueError("shape_body_ids and shape_flags must have equal lengths.")
    if not body_labels:
        raise ValueError("Newton body labels must not be empty.")
    if not robot_path_segment or "/" in robot_path_segment:
        raise ValueError("robot_path_segment must be one path segment.")

    required_names = set(required_body_names)
    body_matches: list[tuple[str, str] | None] = []
    declared_body_names_by_robot: dict[str, set[str]] = {}
    for raw_label in body_labels:
        label = str(raw_label).rstrip("/")
        path_parts = tuple(part for part in label.split("/") if part)
        body_name = _body_leaf(label)
        if body_name not in required_names:
            body_matches.append(None)
            continue
        if len(path_parts) > 1:
            try:
                robot_segment_index = path_parts.index(robot_path_segment)
            except ValueError:
                body_matches.append(None)
                continue
            robot_scope = "/".join(path_parts[: robot_segment_index + 1])
        else:
            robot_scope = "<leaf-labels>"
        body_matches.append((robot_scope, body_name))
        declared_body_names_by_robot.setdefault(robot_scope, set()).add(body_name)

    if not declared_body_names_by_robot:
        raise ValueError("No Newton body labels matched the Wuji hand bodies.")
    for robot_scope, declared_body_names in declared_body_names_by_robot.items():
        missing = sorted(required_names - declared_body_names)
        if missing:
            raise ValueError(
                f"Newton Wuji body labels for {robot_scope!r} are incomplete. "
                "Missing bodies: " + ", ".join(missing)
            )

    selected: list[int] = []
    selected_body_names_by_robot: dict[str, set[str]] = {
        robot_scope: set() for robot_scope in declared_body_names_by_robot
    }
    for shape_id, (body_id_raw, flags_raw) in enumerate(
        zip(shape_body_ids, shape_flags, strict=True)
    ):
        body_id = int(body_id_raw)
        flags = int(flags_raw)
        if body_id < 0:
            continue
        if body_id >= len(body_labels):
            raise ValueError(
                f"Newton shape {shape_id} uses body ID {body_id}, but the model "
                f"has {len(body_labels)} body labels."
            )
        if flags & collide_shapes_flag == 0:
            continue
        body_match = body_matches[body_id]
        if body_match is None:
            continue
        robot_scope, body_name = body_match
        selected.append(shape_id)
        selected_body_names_by_robot[robot_scope].add(body_name)

    if not selected:
        raise ValueError("No Newton collision shapes matched the Wuji hand bodies.")
    for robot_scope, selected_body_names in selected_body_names_by_robot.items():
        missing = sorted(required_names - selected_body_names)
        if missing:
            raise ValueError(
                f"Newton Wuji collision-shape selection for {robot_scope!r} is "
                "incomplete. Missing bodies: " + ", ".join(missing)
            )
    return tuple(selected)


def configure_wuji_hand_material(
    env: Any,
    env_ids: Any | None,
    *,
    static_friction_range: tuple[float, float] = (2.0, 2.01),
    dynamic_friction_range: tuple[float, float] = (2.0, 2.01),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    num_buckets: int = 64,
) -> None:
    """Apply the released hand material through Newton public arrays.

    This function is an Isaac Lab startup event. It runs after Newton has
    finalized its model and before the first policy step. The solver change
    notification makes MJWarp copy the new per-shape friction before it steps.
    """

    if env_ids is not None:
        raise ValueError("The Wuji material event supports startup mode only.")

    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverNotifyFlags

    spec = WujiHandMaterialSpec(
        static_friction_range=tuple(float(value) for value in static_friction_range),
        dynamic_friction_range=tuple(float(value) for value in dynamic_friction_range),
        restitution_range=tuple(float(value) for value in restitution_range),
        num_buckets=int(num_buckets),
    )
    run_seed = getattr(env.cfg, "seed", None)
    if run_seed is None:
        raise ValueError("Set the environment seed before the Wuji material event.")
    model = NewtonManager.get_model()
    if model is None:
        raise RuntimeError("The Newton model is not ready for the Wuji material event.")
    required_fields = (
        "body_label",
        "shape_body",
        "shape_flags",
        "shape_material_mu",
        "shape_material_restitution",
    )
    missing_fields = [
        name for name in required_fields if getattr(model, name, None) is None
    ]
    if missing_fields:
        raise RuntimeError(
            "The Newton model is missing Wuji material fields: "
            + ", ".join(missing_fields)
        )

    shape_body = wp.to_torch(model.shape_body)
    shape_flags = wp.to_torch(model.shape_flags)
    shape_ids = select_wuji_collision_shape_ids(
        tuple(str(label) for label in model.body_label),
        shape_body.detach().cpu().tolist(),
        shape_flags.detach().cpu().tolist(),
    )
    indices = torch.as_tensor(shape_ids, dtype=torch.long, device=shape_body.device)
    friction_values, restitution_values = sample_bucketed_wuji_materials(
        len(shape_ids), spec=spec, seed=int(run_seed)
    )
    material_mu = wp.to_torch(model.shape_material_mu)
    material_restitution = wp.to_torch(model.shape_material_restitution)
    material_mu[indices] = torch.as_tensor(
        friction_values, dtype=material_mu.dtype, device=material_mu.device
    )
    material_restitution[indices] = torch.as_tensor(
        restitution_values,
        dtype=material_restitution.dtype,
        device=material_restitution.device,
    )
    NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)


__all__ = [
    "NEWTON_COLLIDE_SHAPES_FLAG",
    "WUJI_HAND_BODY_PREFIXES",
    "WUJI_HAND_BODY_NAMES",
    "WujiHandMaterialSpec",
    "configure_wuji_hand_material",
    "sample_bucketed_wuji_materials",
    "select_wuji_collision_shape_ids",
]
