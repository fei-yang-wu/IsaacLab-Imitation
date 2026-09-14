"""Explicit collision assets shared by mounted-retarget audits and Isaac."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from iltools.core import sha256_file
from iltools.retarget.convex_parts import convex_part_paths


def build_scene_assets(
    mesh_path: Path,
    initial_object_pose: np.ndarray,
    output: Path,
    *,
    stand_radius_m: float = 0.055,
) -> dict:
    """Write explicit convex pieces and a fixed stand for the local prototype.

    Stand radius is an explicit approximation, not a measured ARCTIC asset.
    Stand top and center are inferred from the first source object pose. The
    Object mass retains the proxy recipe's value; COM and diagonal inertia
    follow a uniform-density approximation of the watertight bottom mesh.
    """
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh_copy = output / mesh_path.name
    shutil.copy2(mesh_path, mesh_copy)
    # Reuse an already-computed decomposition without writing through the
    # source dataset's read-only input symlink.
    for cache in mesh_path.parent.glob(".convex_parts_*"):
        if cache.is_dir():
            shutil.copytree(cache, output / cache.name, dirs_exist_ok=True)
    parts = convex_part_paths(mesh_copy)
    source_mesh = trimesh.load(mesh_copy, force="mesh")
    if not source_mesh.is_watertight or source_mesh.mass <= 0:
        raise ValueError("Object inertial estimation requires a watertight mesh.")
    object_com = np.asarray(source_mesh.center_mass)
    object_inertia = np.diag(source_mesh.moment_inertia * 0.3 / source_mesh.mass).copy()
    object_path = output / "object.usda"
    stage = Usd.Stage.CreateNew(str(object_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/object")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(0.3)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*object_com.tolist()))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*object_inertia.tolist()))
    for i, part in enumerate(parts):
        mesh = trimesh.load(part, force="mesh")
        geom = UsdGeom.Mesh.Define(stage, f"/object/part_{i:03d}")
        geom.CreatePointsAttr(
            [Gf.Vec3f(*p) for p in np.asarray(mesh.vertices).tolist()]
        )
        geom.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        geom.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).reshape(-1).tolist())
        geom.CreateSubdivisionSchemeAttr("none")
        geom.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.35, 0.12)])
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr(
            "convexHull"
        )
    stage.GetRootLayer().Save()
    mesh = trimesh.load(mesh_copy, force="mesh")
    pose = np.asarray(initial_object_pose)
    vertices = (
        Rotation.from_quat(pose[3:], scalar_first=True).apply(mesh.vertices) + pose[:3]
    )
    low, high = vertices.min(0), vertices.max(0)
    center = (low + high) / 2
    height = float(low[2])
    if height <= 0 or stand_radius_m <= 0:
        raise ValueError("Stand dimensions must be positive.")
    stand_pose = [float(center[0]), float(center[1]), height / 2, 1, 0, 0, 0]
    support_path = output / "stand.usda"
    stage = Usd.Stage.CreateNew(str(support_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stand_root = UsdGeom.Xform.Define(stage, "/stand")
    stand = UsdGeom.Cylinder.Define(stage, "/stand/collider")
    stand.CreateRadiusAttr(stand_radius_m)
    stand.CreateHeightAttr(height)
    stand.CreateAxisAttr("Z")
    stand.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.28, 0.3)])
    UsdPhysics.CollisionAPI.Apply(stand.GetPrim())
    stage.SetDefaultPrim(stand_root.GetPrim())
    stage.GetRootLayer().Save()
    metadata = {
        "object_usd": str(object_path),
        "object_usd_sha256": sha256_file(object_path),
        "object_parts": [str(p) for p in parts],
        "object_part_sha256": [sha256_file(p) for p in parts],
        "source_mesh_sha256": sha256_file(mesh_path),
        "source_initial_object_pose_w": pose.tolist(),
        "object_mass_kg": 0.3,
        "object_center_of_mass_m": object_com.tolist(),
        "object_diagonal_inertia_kg_m2": object_inertia.tolist(),
        "stand_usd": str(support_path),
        "stand_usd_sha256": sha256_file(support_path),
        "stand_pose_w": stand_pose,
        "stand_radius_m": stand_radius_m,
        "stand_height_m": height,
        "stand_provenance": "prototype cylinder; radius explicit, center/top inferred from first source object mesh pose",
        "stand_runtime_constraint": "static collision geometry; zero movable joints",
        "object_physics_provenance": "0.3 kg from existing rigid proxy; uniform-density watertight-mesh COM and diagonal inertia approximation (cross terms omitted)",
    }
    (output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def build_audit_model(model_path: Path, root_pose: np.ndarray, scene: dict):
    """Use the same convex pieces, body frames, exclusions, and stand as Isaac."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(model_path.resolve()))
    robot = spec.worldbody.first_body()
    robot.pos = root_pose[:3]
    robot.quat = root_pose[3:]
    obj = spec.worldbody.add_body(name="object", mocap=True)
    for i, path in enumerate(scene["object_parts"]):
        spec.add_mesh(name=f"object_part_{i}", file=path)
        obj.add_geom(
            name=f"object_part_{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"object_part_{i}",
        )
    stand = spec.worldbody.add_body(name="stand", pos=scene["stand_pose_w"][:3])
    stand.add_geom(
        name="stand",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[scene["stand_radius_m"], scene["stand_height_m"] / 2, 0],
    )
    model = spec.compile()
    return model
