"""Build the mounted Vega U / Sharpa model from the vendored robot assets."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from iltools.core import sha256_file

HAND_CONTACT_SUFFIXES = (
    "hand_C_MC",
    "thumb_MC",
    "thumb_PP",
    "thumb_DP",
    "index_PP",
    "index_MP",
    "index_DP",
    "middle_PP",
    "middle_MP",
    "middle_DP",
    "ring_PP",
    "ring_MP",
    "ring_DP",
    "pinky_MC",
    "pinky_PP",
    "pinky_MP",
    "pinky_DP",
)


def base_ground_alignment(model_path: Path, root_pose) -> dict:
    """Measure the fixed base's collision-mesh bottom against the z=0 floor.

    The Vega base frame is above the physical mounting surface. Its origin
    therefore cannot be used as the floor contact point.
    """
    import mujoco
    import numpy as np

    pose = np.asarray(root_pose, dtype=np.float64)
    spec = mujoco.MjSpec.from_file(str(model_path.resolve()))
    base = spec.worldbody.first_body()
    base.pos, base.quat = pose[:3], pose[3:]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    base_id = model.body("base_link").id
    bottoms = []
    for geom in np.flatnonzero(model.geom_bodyid == base_id):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Vega base alignment expects mesh collision geometry.")
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        bottoms.append(float(world[:, 2].min()))
    if not bottoms:
        raise ValueError("Vega base has no collision mesh to align with the floor.")
    minimum = min(bottoms)
    return {
        "ground_height_m": 0.0,
        "base_collision_minimum_z_m": minimum,
        "base_ground_clearance_m": minimum,
        "ground_aligned_root_z_m": float(pose[2] - minimum),
    }


def author_robot_gravity_compensation(usd_path: Path) -> int:
    """Author the Newton/MuJoCo per-body setting outside backend variants."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.CreateAttribute(
                "mjc:gravcomp", Sdf.ValueTypeNames.Float, custom=True
            ).Set(1.0)
            count += 1
    if not count:
        raise ValueError("Cannot author gravity compensation without robot bodies.")
    stage.GetRootLayer().Save()
    return count


def author_robot_shape_collision_exclusions(usd_path: Path) -> int:
    """Expand body filters to collider filters for Newton 1.2.1's USD reader.

    The converter writes standard body-level filtered pairs. The pinned Newton
    reader only consumes this relationship on collision shapes. Each shape is
    assigned to its nearest rigid body, so nested articulation links do not
    acquire their parent's exclusions.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    # Collider paths inside instances cannot carry per-link relationships.
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    body_shapes = {}
    body_filters = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_shapes.setdefault(prim.GetPath(), [])
            for target in prim.GetRelationship("physics:filteredPairs").GetTargets():
                body_filters.append((prim.GetPath(), target))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            parent = prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            if parent:
                body_shapes.setdefault(parent.GetPath(), []).append(prim.GetPath())
    pairs = set()
    for body, other in body_filters:
        if other not in body_shapes:
            raise ValueError(f"Unknown excluded rigid body: {other}")
        for shape in body_shapes[body]:
            relationship = UsdPhysics.FilteredPairsAPI.Apply(
                stage.GetPrimAtPath(shape)
            ).CreateFilteredPairsRel()
            for target in body_shapes[other]:
                relationship.AddTarget(target)
                pairs.add(tuple(sorted((str(shape), str(target)))))
    stage.GetRootLayer().Save()
    return len(pairs)


def describe_robot_asset(usd_path: Path, model_path: Path) -> dict:
    """Record imported bodies, per-shape contact paths, and all USD layer hashes."""
    from pxr import Usd, UsdPhysics

    usd_path, model_path = usd_path.resolve(), model_path.resolve()
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim().GetPath()
    filters = {side: {"paths": [], "link_indices": []} for side in ("left", "right")}
    bodies, colliders = [], []
    gravity_compensation = {}
    shape_collision_exclusions = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(str(prim.GetPath()))
            gravity_compensation[str(prim.GetPath())] = float(
                prim.GetAttribute("mjc:gravcomp").Get() or 0.0
            )
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        colliders.append(str(prim.GetPath()))
        for target in prim.GetRelationship("physics:filteredPairs").GetTargets():
            shape_collision_exclusions.append([str(prim.GetPath()), str(target)])
        parent = prim
        while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
            parent = parent.GetParent()
        if not parent:
            continue
        for side in ("left", "right"):
            names = [f"{side}_{suffix}" for suffix in HAND_CONTACT_SUFFIXES]
            if parent.GetName() in names:
                path = str(prim.GetPath().MakeRelativePath(root))
                filters[side]["paths"].append("{ENV_REGEX_NS}/robot/" + path)
                filters[side]["link_indices"].append(names.index(parent.GetName()))
    if not colliders:
        raise ValueError("The converted robot has no collision shapes.")
    if not gravity_compensation or any(
        value != 1.0 for value in gravity_compensation.values()
    ):
        raise ValueError("Every converted robot body must declare mjc:gravcomp=1.")
    return {
        "usd_path": str(usd_path),
        "usd_sha256": sha256_file(usd_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "usd_layer_sha256": {
            str(p.resolve()): sha256_file(p)
            for p in usd_path.parent.rglob("*")
            if p.suffix in (".usd", ".usda", ".usdc")
        },
        "rigid_body_paths": bodies,
        "collision_shape_paths": colliders,
        "physx_contact_filters": filters,
        "gravity_compensation": "mjc:gravcomp=1 on each robot rigid body",
        "body_gravity_compensation": gravity_compensation,
        "shape_collision_exclusions": shape_collision_exclusions,
    }


def compose_model(assets: Path, output: Path) -> Path:
    """Attach Sharpa roots at the arm flanges and lock the five setup joints.

    The identity flange mounts follow the collaborator's ``compose_urdf.py``.
    Their physical adapter thickness and clocking still require measurement.
    No Wuji adapter geometry is retained.
    """
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vega = assets / "vega_wuji/vega_u_wuji_v2_beta1_with_mount.xml"
    root = ET.parse(vega).getroot()
    root.set("model", "vega_u_sharpa")
    # The older Wuji retarget asset disabled backbone collisions. The
    # mounted task keeps arm/torso contact active as well as hand contact.
    for geom in root.findall(".//default[@class='collision']/geom"):
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
    compiler = root.find("compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.set("fusestatic", "false")
    sources = {str(vega.resolve()): sha256_file(vega)}
    asset = root.find("asset")
    for item in asset:
        if "file" in item.attrib:
            item.set(
                "file", os.path.relpath(vega.parent / item.get("file"), output.parent)
            )
    for tag in ("custom", "keyframe"):
        for element in root.findall(tag):
            root.remove(element)
    setup_joints = {"Lift", "torso_flip", "head_j1", "head_j2", "head_j3"}
    for body in root.findall(".//worldbody//body"):
        body.set("gravcomp", "1")
        for joint in list(body.findall("joint")):
            if joint.get("name") in setup_joints:
                if float(joint.get("ref", "0")) != 0:
                    raise ValueError(
                        "Nonzero setup reference requires an explicit fixed transform."
                    )
                body.remove(joint)
    actuator = root.find("actuator")
    for side, prefix in (("left", "L"), ("right", "R")):
        flange = root.find(f".//body[@name='{prefix}_arm_l8']")
        if flange is None:
            raise ValueError(f"Missing {prefix} flange.")
        for child in list(flange.findall("body")):
            flange.remove(child)
        hand_path = assets / f"sharpa_wave/xmls/sharpawave/{side}_sharpawave.xml"
        sources[str(hand_path.resolve())] = sha256_file(hand_path)
        hand = ET.parse(hand_path).getroot()
        mesh_dir = hand_path.parent / hand.find("compiler").get("meshdir", "")
        names = {e.get("name"): f"{side}__{e.get('name')}" for e in hand.find("asset")}
        classes = {
            e.get("class"): f"{side}__{e.get('class')}"
            for e in hand.findall(".//default[@class]")
        }
        for element in hand.iter():
            for attribute in ("mesh", "material", "texture"):
                if element.get(attribute) in names:
                    element.set(attribute, names[element.get(attribute)])
            for attribute in ("class", "childclass"):
                if element.get(attribute) in classes:
                    element.set(attribute, classes[element.get(attribute)])
        for item in hand.find("asset"):
            item.set("name", names[item.get("name")])
            if "file" in item.attrib:
                item.set(
                    "file", os.path.relpath(mesh_dir / item.get("file"), output.parent)
                )
            asset.append(copy.deepcopy(item))
        defaults = hand.find("default")
        defaults.set("class", f"{side}__hand")
        root.find("default").append(copy.deepcopy(defaults))
        hand_body = hand.find("worldbody/body")
        for body in hand_body.iter("body"):
            body.set("gravcomp", "1")
        hand_body.set("childclass", f"{side}__hand")
        ET.SubElement(
            hand_body, "site", name=f"{side}_mounted_wrist", pos="0 0 0", quat="1 0 0 0"
        )
        flange.append(copy.deepcopy(hand_body))
        for item in hand.find("actuator"):
            actuator.append(copy.deepcopy(item))
        for item in hand.findall("contact/*"):
            root.find("contact").append(copy.deepcopy(item))
    joint_names = {e.get("name") for e in root.findall(".//worldbody//joint")}
    for item in list(actuator):
        if item.get("joint") not in joint_names:
            actuator.remove(item)
    body_names = {e.get("name") for e in root.findall(".//worldbody//body")}
    contact = root.find("contact")
    for item in list(contact):
        if item.get("body1") not in body_names or item.get("body2") not in body_names:
            contact.remove(item)
    # Locking the setup joints welds the torso into the world. MuJoCo does
    # not auto-filter a world/first-joint pair, so keep the adjacent shoulder
    # housings excluded explicitly (arm_center is only a geometry-free frame).
    for prefix in ("L", "R"):
        ET.SubElement(
            contact, "exclude", body1="torso_flip_link", body2=f"{prefix}_arm_l1"
        )
    used_meshes = {e.get("mesh") for e in root.iter() if e.get("mesh")}
    for item in list(asset):
        if item.tag == "mesh" and item.get("name") not in used_meshes:
            asset.remove(item)
    if len(joint_names) != 58 or len(actuator) != 58:
        raise ValueError(
            "Expected fourteen arm and forty-four finger joints/actuators."
        )
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
    dependencies = {
        str((output.parent / e.get("file")).resolve()): sha256_file(
            output.parent / e.get("file")
        )
        for e in asset
        if e.get("file")
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "model_sha256": sha256_file(output),
                "sources": sources,
                "assets": dependencies,
                "mounts": {
                    side: {
                        "parent": f"{prefix}_arm_l8",
                        "xyz_wxyz": [0, 0, 0, 1, 0, 0, 0],
                    }
                    for side, prefix in (("left", "L"), ("right", "R"))
                },
                "mount_status": "collaborator assembly convention; physical adapters unmeasured",
                "fixed_setup_joints": {name: 0.0 for name in sorted(setup_joints)},
                "gravity_compensation": "MuJoCo gravcomp=1 on every robot body",
                "backbone_collision_masks": {"contype": 1, "conaffinity": 1},
                "additional_adjacent_exclusions": [
                    ["torso_flip_link", f"{p}_arm_l1"] for p in ("L", "R")
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return output
