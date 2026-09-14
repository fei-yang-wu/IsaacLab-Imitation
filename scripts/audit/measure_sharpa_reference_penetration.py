#!/usr/bin/env python3
"""Measure how far a retargeted Sharpa reference puts the hands inside the object.

The retarget solver places the fingers by inverse kinematics on fingertip
targets. Nothing in that step knows about the object's surface, so a frame can
end with a finger centimetres inside the mesh. Isaac then has to resolve that
on the first physics step: Newton answers with a large impulse that throws the
hands and the object off the Reference, and the episode terminates before the
policy acts.

This tool turns that into numbers, without Isaac. It rebuilds the reference in
MuJoCo, runs collision detection only, and reports per frame how deep the
worst hand-object contact is. It also compares geometry against the
reference's own ``contact_active`` flags, so a frame that claims contact while
the hand is nowhere near the object shows up as a miss.

Only kinematics and collision run, never the solver: a penetrating pose makes
the constraint factorization rank deficient.

Example:

    pixi run python scripts/audit/measure_sharpa_reference_penetration.py \\
        --manifest data/dexmanip/sharpa/arctic_box_grab_speed05/manifest.json \\
        --object-mesh data/dexmanip/sharpa/assets/arctic_box_bottom/bottom.obj

Exit codes: 0 within tolerance, 1 penetration above ``--max-penetration``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = (
    REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets/sharpa_wave"
)
DEFAULT_MJCF = {
    side: ASSET_ROOT / "xmls" / "sharpawave" / f"{side}_sharpawave.xml"
    for side in ("left", "right")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--object-mesh", type=Path, default=None)
    parser.add_argument("--object-half-size", type=float, default=0.05)
    parser.add_argument("--left-mjcf", type=Path, default=DEFAULT_MJCF["left"])
    parser.add_argument("--right-mjcf", type=Path, default=DEFAULT_MJCF["right"])
    parser.add_argument(
        "--max-penetration",
        type=float,
        default=0.002,
        help="Deepest penetration, in metres, this reference may carry.",
    )
    parser.add_argument(
        "--convex-hull",
        action="store_true",
        help=(
            "Collide against the mesh's convex hull instead of its convex "
            "decomposition. This is what the released penetration filter "
            "measures, and it is wrong for a container."
        ),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _load_reference(manifest: Path, index: int) -> dict[str, Any]:
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    motion = spec["motions"][index]
    path = manifest.parent / motion["path"]
    with np.load(path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def _build_scene(args: argparse.Namespace) -> tuple[Any, Any]:
    import mujoco

    spec = mujoco.MjSpec()
    spec.option.gravity = [0.0, 0.0, 0.0]
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_free")
    if args.object_mesh is None:
        body.add_geom(
            name="object_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[args.object_half_size] * 3,
        )
    elif args.convex_hull:
        spec.add_mesh(name="object_mesh", file=str(args.object_mesh.resolve()))
        body.add_geom(
            name="object_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="object_mesh",
        )
    else:
        # One geom per convex part, matching what Isaac simulates. A single
        # mesh geom is its convex hull, which fills a container's cavity and
        # reports a hand reaching inside as deep penetration that is not real.
        from iltools.retarget.convex_parts import convex_part_paths

        for index, part in enumerate(convex_part_paths(args.object_mesh)):
            name = f"object_part_{index:03d}"
            spec.add_mesh(name=name, file=str(part))
            body.add_geom(
                name=f"object_geom_{index:03d}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
            )
    for side, path in (("left", args.left_mjcf), ("right", args.right_mjcf)):
        child = mujoco.MjSpec.from_file(str(path.resolve()))
        root = spec.worldbody.add_body(name=f"{side}_root")
        root.add_freejoint(name=f"{side}_free")
        # Both hand MJCFs share mesh names, so every attached name is prefixed.
        root.add_frame().attach_body(child.worldbody.first_body(), f"{side}_", "")
    model = spec.compile()
    return model, mujoco.MjData(model)


def main() -> int:
    import mujoco

    args = parse_args()
    reference = _load_reference(args.manifest, args.motion_index)
    model, data = _build_scene(args)

    object_body = model.body("object").id
    object_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom(geom_id).bodyid[0]) == object_body
    }
    hand_geoms = {
        geom_id for geom_id in range(model.ngeom) if geom_id not in object_geoms
    }

    qpos = np.asarray(reference["qpos"], dtype=np.float64)
    joint_names = [str(name) for name in reference["joint_names"]]
    left_names = [str(name) for name in reference["left_joint_names"]]
    right_names = [str(name) for name in reference["right_joint_names"]]
    columns = {name: index for index, name in enumerate(joint_names)}
    object_poses = np.asarray(reference["object_poses_w"], dtype=np.float64)
    left_wrist = np.asarray(reference["left_wrist_pose_w"], dtype=np.float64)
    right_wrist = np.asarray(reference["right_wrist_pose_w"], dtype=np.float64)
    contact_active = np.asarray(reference["contact_active"])

    frames = range(0, int(qpos.shape[0]), max(1, args.stride))
    depths: list[float] = []
    claimed_but_apart: list[int] = []
    contact_counts: list[int] = []

    for frame in frames:
        address = model.joint("object_free").qposadr[0]
        data.qpos[address : address + 7] = object_poses[frame, 0, :7]
        for side, wrist, names in (
            ("left", left_wrist, left_names),
            ("right", right_wrist, right_names),
        ):
            base = model.joint(f"{side}_free").qposadr[0]
            data.qpos[base : base + 7] = wrist[frame, :7]
            for name in names:
                value = qpos[frame, columns[name]]
                data.qpos[model.joint(f"{side}_{name}").qposadr[0]] = value
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)

        worst = 0.0
        touching = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & object_geoms) or not (pair & hand_geoms):
                continue
            touching += 1
            worst = min(worst, float(contact.dist))
        depths.append(worst)
        contact_counts.append(touching)
        claims = bool(np.any(contact_active[frame])) if contact_active.size else False
        if claims and touching == 0:
            claimed_but_apart.append(int(frame))

    depth_array = np.asarray(depths)
    penetration = -depth_array  # positive metres of overlap
    total = len(depths)
    deepest = float(penetration.max()) if total else 0.0
    over = int((penetration > args.max_penetration).sum())

    print(f"frames measured            : {total}")
    print(f"frames with any contact    : {sum(1 for c in contact_counts if c)}")
    print(f"deepest penetration        : {deepest * 1000:.2f} mm")
    print(f"mean penetration           : {float(penetration.mean()) * 1000:.2f} mm")
    print(
        f"frames over {args.max_penetration * 1000:.1f} mm       : "
        f"{over} ({100.0 * over / total:.1f}%)"
        if total
        else "no frames"
    )
    print(
        f"frames claiming contact    : "
        f"{int(np.any(contact_active, axis=tuple(range(1, contact_active.ndim))).sum())}"
        if contact_active.size
        else "frames claiming contact    : n/a"
    )
    print(f"claimed but geometry apart : {len(claimed_but_apart)}")

    record = {
        "manifest": str(args.manifest),
        "frames": total,
        "deepest_penetration_m": deepest,
        "mean_penetration_m": float(penetration.mean()) if total else 0.0,
        "frames_over_limit": over,
        "max_penetration_m": args.max_penetration,
        "claimed_but_apart_frames": claimed_but_apart[:50],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if deepest > args.max_penetration:
        print(
            f"\nFAILED: the reference penetrates the object by "
            f"{deepest * 1000:.2f} mm, above the "
            f"{args.max_penetration * 1000:.1f} mm limit."
        )
        return 1
    print("\nPASSED: the reference rests on the object surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
