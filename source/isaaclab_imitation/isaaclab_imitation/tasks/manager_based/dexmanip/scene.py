"""Newton-safe scene specifications and Isaac Lab asset builders.

This module is importable without Isaac Lab.  The data layer can therefore
validate a scene before it starts Isaac Sim.  Isaac Lab imports occur only in
``build_scene_asset_configs``.

Reference motion data stores quaternions as WXYZ.  Isaac Lab 3.0 asset initial
states in this workspace use XYZW.  The builder converts the order once at
that boundary.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse


_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
_RIGID_OBJECT_SUFFIXES = _USD_SUFFIXES | {".urdf"}
_USD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
CollisionApproximation = Literal["convex_hull", "convex_decomposition"]


def _finite_tuple(
    value: Sequence[float], size: int, field_name: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{field_name} must contain {size} numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain only finite numbers")
    return result


def _validate_name(name: str, field_name: str = "name") -> None:
    if not _USD_NAME.fullmatch(name):
        raise ValueError(
            f"{field_name} must be a Python attribute and one USD path segment; got {name!r}"
        )


def _asset_suffix(asset_path: str) -> str:
    parsed_path = urlparse(asset_path).path
    return Path(parsed_path).suffix.lower()


def _is_remote_asset(asset_path: str) -> bool:
    return bool(urlparse(asset_path).scheme and "://" in asset_path)


def _validate_asset_path(
    asset_path: str,
    *,
    allowed_suffixes: frozenset[str],
    role: str,
    require_exists: bool,
) -> None:
    if not isinstance(asset_path, str) or not asset_path.strip():
        raise ValueError(f"{role} asset_path must be a non-empty string")
    suffix = _asset_suffix(asset_path)
    if suffix not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ValueError(
            f"{role} asset must use one of [{allowed}], got {asset_path!r}"
        )
    if not require_exists or _is_remote_asset(asset_path):
        return
    path = Path(asset_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{role} asset does not exist or is not a file: {path}")


def _validate_rigid_urdf(asset_path: str) -> None:
    """Require a URDF that converts to one rigid body after fixed-joint merge."""
    try:
        root = ET.parse(Path(asset_path).expanduser()).getroot()
    except ET.ParseError as error:
        raise ValueError(f"rigid object URDF is not valid XML: {asset_path}") from error
    if root.tag != "robot":
        raise ValueError(f"rigid object URDF root must be <robot>: {asset_path}")

    link_names = [link.get("name") for link in root.findall("link")]
    if not link_names or any(not name for name in link_names):
        raise ValueError(f"rigid object URDF must define named links: {asset_path}")
    if len(set(link_names)) != len(link_names):
        raise ValueError(
            f"rigid object URDF contains duplicate link names: {asset_path}"
        )

    parent = {name: name for name in link_names if name is not None}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type != "fixed":
            joint_name = joint.get("name", "<unnamed>")
            raise ValueError(
                f"rigid object URDF contains non-fixed joint {joint_name!r} "
                f"of type {joint_type!r}: {asset_path}"
            )
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        parent_name = parent_element.get("link") if parent_element is not None else None
        child_name = child_element.get("link") if child_element is not None else None
        if parent_name not in parent or child_name not in parent:
            raise ValueError(
                f"rigid object URDF joint references an unknown link: {asset_path}"
            )
        parent_root = find(parent_name)
        child_root = find(child_name)
        if parent_root == child_root:
            raise ValueError(
                f"rigid object URDF fixed-joint graph contains a cycle: {asset_path}"
            )
        parent[child_root] = parent_root

    components = {find(name) for name in parent}
    if len(components) != 1:
        raise ValueError(
            "rigid object URDF must become one body after fixed-joint merge; "
            f"found {len(components)} disconnected components in {asset_path}"
        )


@dataclass(frozen=True, slots=True)
class ContactMaterialSpec:
    """Solver material values shared by a rigid object or support surface."""

    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    torsional_friction: float | None = None
    rolling_friction: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "static_friction",
            "dynamic_friction",
            "restitution",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if self.static_friction < 0.0 or self.dynamic_friction < 0.0:
            raise ValueError("friction values must be non-negative")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        for field_name in ("torsional_friction", "rolling_friction"):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class RigidObjectSpec:
    """One tracked, dynamic, single-body reference object."""

    name: str
    asset_path: str
    init_pos: Vec3 = (0.0, 0.0, 0.0)
    init_quat_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    mass: float | None = None
    collision_approximation: CollisionApproximation = "convex_decomposition"
    # The upstream recipe zeroes contact/rest offsets.  Spell the Newton equivalents
    # out instead of inheriting ``NewtonShapeCfg.gap`` (currently 1 cm).  The
    # active MJWarp MuJoCo-contact path also clamps its effective geom gap and
    # margin to zero, but explicit values keep the scene contract invariant if
    # the collision pipeline changes.
    contact_margin: float | None = 0.0
    contact_gap: float | None = 0.0
    material: ContactMaterialSpec | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name)
        _validate_asset_path(
            self.asset_path,
            allowed_suffixes=_RIGID_OBJECT_SUFFIXES,
            role=f"object {self.name!r}",
            require_exists=False,
        )
        pos = _finite_tuple(self.init_pos, 3, "init_pos")
        quat = _finite_tuple(self.init_quat_wxyz, 4, "init_quat_wxyz")
        scale = _finite_tuple(self.scale, 3, "scale")
        if not math.isclose(sum(value * value for value in quat), 1.0, abs_tol=1.0e-4):
            raise ValueError("init_quat_wxyz must be a unit quaternion")
        if any(value <= 0.0 for value in scale):
            raise ValueError("scale values must be positive")
        object.__setattr__(self, "init_pos", pos)
        object.__setattr__(self, "init_quat_wxyz", quat)
        object.__setattr__(self, "scale", scale)
        if self.mass is not None:
            mass = float(self.mass)
            if not math.isfinite(mass) or mass <= 0.0:
                raise ValueError("mass must be finite and positive")
            object.__setattr__(self, "mass", mass)
        if self.collision_approximation not in {
            "convex_hull",
            "convex_decomposition",
        }:
            raise ValueError(
                "collision_approximation must be 'convex_hull' or "
                "'convex_decomposition'"
            )
        for field_name in ("contact_margin", "contact_gap"):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)

    def validate_asset(self) -> None:
        _validate_asset_path(
            self.asset_path,
            allowed_suffixes=_RIGID_OBJECT_SUFFIXES,
            role=f"object {self.name!r}",
            require_exists=True,
        )
        if _asset_suffix(self.asset_path) == ".urdf" and not _is_remote_asset(
            self.asset_path
        ):
            _validate_rigid_urdf(self.asset_path)


@dataclass(frozen=True, slots=True)
class SupportSurfaceSpec:
    """One static support-surface USD loaded below each environment."""

    name: str
    asset_path: str
    init_pos: Vec3 = (0.0, 0.0, 0.0)
    init_quat_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    # Keep support contacts on the same zero-offset convention as the dynamic
    # objects and the released scene assets.
    contact_margin: float | None = 0.0
    contact_gap: float | None = 0.0
    material: ContactMaterialSpec = field(default_factory=ContactMaterialSpec)

    def __post_init__(self) -> None:
        _validate_name(self.name)
        _validate_asset_path(
            self.asset_path,
            allowed_suffixes=_USD_SUFFIXES,
            role=f"support surface {self.name!r}",
            require_exists=False,
        )
        pos = _finite_tuple(self.init_pos, 3, "init_pos")
        quat = _finite_tuple(self.init_quat_wxyz, 4, "init_quat_wxyz")
        scale = _finite_tuple(self.scale, 3, "scale")
        if not math.isclose(sum(value * value for value in quat), 1.0, abs_tol=1.0e-4):
            raise ValueError("init_quat_wxyz must be a unit quaternion")
        if any(value <= 0.0 for value in scale):
            raise ValueError("scale values must be positive")
        object.__setattr__(self, "init_pos", pos)
        object.__setattr__(self, "init_quat_wxyz", quat)
        object.__setattr__(self, "scale", scale)
        for field_name in ("contact_margin", "contact_gap"):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
            object.__setattr__(self, field_name, value)

    def validate_asset(self) -> None:
        _validate_asset_path(
            self.asset_path,
            allowed_suffixes=_USD_SUFFIXES,
            role=f"support surface {self.name!r}",
            require_exists=True,
        )


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """A complete rigid-object scene that can be applied before construction."""

    objects: tuple[RigidObjectSpec, ...]
    support_surfaces: tuple[SupportSurfaceSpec, ...] = ()
    primary_object_name: str | None = None

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        support_surfaces = tuple(self.support_surfaces)
        if not objects:
            raise ValueError("a scene spec must contain at least one rigid object")
        if not all(isinstance(item, RigidObjectSpec) for item in objects):
            raise TypeError("objects must contain only RigidObjectSpec values")
        if not all(isinstance(item, SupportSurfaceSpec) for item in support_surfaces):
            raise TypeError(
                "support_surfaces must contain only SupportSurfaceSpec values"
            )
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "support_surfaces", support_surfaces)

        names = [item.name for item in (*objects, *support_surfaces)]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"scene asset names must be unique: {duplicate_names}")

        primary = self.primary_object_name or objects[0].name
        if primary not in {item.name for item in objects}:
            raise ValueError(
                f"primary_object_name {primary!r} does not name a rigid object"
            )
        object.__setattr__(self, "primary_object_name", primary)

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.objects)

    @property
    def support_surface_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.support_surfaces)

    def validate_assets(self) -> None:
        for item in (*self.objects, *self.support_surfaces):
            item.validate_asset()


@dataclass(frozen=True, slots=True)
class SceneAssetConfigs:
    """Isaac Lab configurations generated for one validated scene."""

    objects: Mapping[str, Any]
    support_surfaces: Mapping[str, Any]
    primary_object_name: str

    @property
    def all_assets(self) -> dict[str, Any]:
        return {**self.objects, **self.support_surfaces}


def wxyz_to_xyzw(quaternion: Sequence[float]) -> Quat:
    """Convert one unit quaternion from WXYZ to Isaac Lab XYZW."""
    w, x, y, z = _finite_tuple(quaternion, 4, "quaternion")
    if not math.isclose(w * w + x * x + y * y + z * z, 1.0, abs_tol=1.0e-4):
        raise ValueError("quaternion must be a unit quaternion")
    return (x, y, z, w)


def _material_cfg(spec: ContactMaterialSpec) -> Any:
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    return NewtonMaterialPropertiesCfg(
        static_friction=spec.static_friction,
        dynamic_friction=spec.dynamic_friction,
        restitution=spec.restitution,
        torsional_friction=spec.torsional_friction,
        rolling_friction=spec.rolling_friction,
    )


def _collision_cfg(contact_margin: float | None, contact_gap: float | None) -> Any:
    from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg

    return NewtonCollisionPropertiesCfg(
        collision_enabled=True,
        contact_margin=contact_margin,
        contact_gap=contact_gap,
    )


def build_scene_asset_configs(
    scene_spec: SceneSpec,
    *,
    env_namespace: str = "{ENV_REGEX_NS}",
    validate_asset_files: bool = True,
) -> SceneAssetConfigs:
    """Generate backend-compatible Isaac Lab configurations before scene build.

    Dynamic objects use ``RigidObjectCfg``.  Support USDs use ``AssetBaseCfg``
    and keep their collision geometry static.  No stage traversal, live scene
    mutation, or backend-specific contact reporter is used here.
    """
    if not isinstance(scene_spec, SceneSpec):
        raise TypeError("scene_spec must be a SceneSpec")
    if not env_namespace or not env_namespace.startswith("{"):
        raise ValueError("env_namespace must be an Isaac Lab namespace token")
    if validate_asset_files:
        scene_spec.validate_assets()

    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyBaseCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg, UsdFileCfg

    objects: dict[str, Any] = {}
    for item in scene_spec.objects:
        spawn_common: dict[str, Any] = {
            "scale": item.scale,
            "rigid_props": RigidBodyBaseCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            "collision_props": _collision_cfg(
                item.contact_margin,
                item.contact_gap,
            ),
            "mass_props": MassPropertiesCfg(mass=item.mass),
        }
        if item.material is not None:
            spawn_common["physics_material"] = _material_cfg(item.material)
        suffix = _asset_suffix(item.asset_path)
        if suffix == ".urdf":
            collision_type = {
                "convex_hull": "Convex Hull",
                "convex_decomposition": "Convex Decomposition",
            }[item.collision_approximation]
            spawn = UrdfFileCfg(
                asset_path=item.asset_path,
                fix_base=False,
                joint_drive=None,
                merge_fixed_joints=True,
                collision_type=collision_type,
                run_asset_transformer=True,
                run_multi_physics_conversion=True,
                **spawn_common,
            )
        else:
            spawn = UsdFileCfg(usd_path=item.asset_path, **spawn_common)
        objects[item.name] = RigidObjectCfg(
            prim_path=f"{env_namespace}/{item.name}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=item.init_pos,
                rot=wxyz_to_xyzw(item.init_quat_wxyz),
            ),
            collision_group=0,
        )

    support_surfaces: dict[str, Any] = {}
    for item in scene_spec.support_surfaces:
        support_surfaces[item.name] = AssetBaseCfg(
            prim_path=f"{env_namespace}/{item.name}",
            spawn=UsdFileCfg(
                usd_path=item.asset_path,
                scale=item.scale,
                # The support stays a static collider. Isaac Lab warns that
                # 'modify_rigid_body_properties' reached no prim, because a
                # support USD carries no RigidBodyAPI. That warning is benign:
                # the declaration keeps the static intent explicit.
                rigid_props=RigidBodyBaseCfg(
                    rigid_body_enabled=False,
                    kinematic_enabled=False,
                ),
                collision_props=_collision_cfg(
                    item.contact_margin,
                    item.contact_gap,
                ),
                physics_material=_material_cfg(item.material),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=item.init_pos,
                rot=wxyz_to_xyzw(item.init_quat_wxyz),
            ),
            collision_group=0,
        )

    return SceneAssetConfigs(
        objects=objects,
        support_surfaces=support_surfaces,
        primary_object_name=str(scene_spec.primary_object_name),
    )


def apply_scene_assets(
    scene_cfg: Any,
    scene_spec: SceneSpec,
    *,
    validate_asset_files: bool = True,
) -> SceneAssetConfigs:
    """Attach generated assets to an ``InteractiveSceneCfg`` before env build."""
    bundle = build_scene_asset_configs(
        scene_spec,
        validate_asset_files=validate_asset_files,
    )
    for name, asset_cfg in bundle.all_assets.items():
        if hasattr(scene_cfg, name):
            raise ValueError(f"scene already defines an asset named {name!r}")
        setattr(scene_cfg, name, asset_cfg)
    return bundle


__all__ = [
    "ContactMaterialSpec",
    "RigidObjectSpec",
    "SceneAssetConfigs",
    "SceneSpec",
    "SupportSurfaceSpec",
    "apply_scene_assets",
    "build_scene_asset_configs",
    "wxyz_to_xyzw",
]
