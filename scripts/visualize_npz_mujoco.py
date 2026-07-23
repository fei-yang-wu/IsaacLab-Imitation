#!/usr/bin/env python3
"""Visualize a G1 reference NPZ with MuJoCo — interactive viewer or mesh MP4.

MuJoCo is the right tool for eyeballing these motions: its renderer is reliable
headless (unlike the Isaac TiledCamera + Fabric teleport path, which renders a
frozen pose), it shows the full robot mesh, and the same model doubles as an
interactive viewer you can scrub locally.

The NPZ ``qpos`` is ``[root_pos(3), root_quat(4, wxyz), joint_pos(29)]`` = 36,
which is exactly MuJoCo's ``qpos`` for the G1 MJCF (free base + 29 hinges). Joint
values are mapped by NAME (the NPZ ``joint_names`` are ``*_joint`` and match the
MJCF), so this is robust to any joint-order differences.

Offscreen mesh MP4 (headless, needs a GPU; sets MUJOCO_GL=egl automatically):

    pixi run -e isaaclab python scripts/visualize_npz_mujoco.py \
        --npz ~/Storage/bones_seed_full/npz/g1/Loop_Forward_Walk_001__A018.npz \
        --out ~/Storage/bones_seed_full/quality_videos/walk_mujoco.mp4

Interactive viewer (run at the workstation with a display; drag to orbit, space
to pause):

    pixi run -e isaaclab python scripts/visualize_npz_mujoco.py \
        --npz .../Loop_Forward_Walk_001__A018.npz --viewer
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    REPO_ROOT
    / "source"
    / "isaaclab_imitation"
    / "isaaclab_imitation"
    / "assets"
    / "unitree"
    / "g1_description"
    / "g1_29dof_rev_1_0.xml"
)


def _load_frames(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as data:
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)  # (T, 3)
        root_quat = np.asarray(data["root_quat"], dtype=np.float64)  # (T, 4) wxyz
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)  # (T, 29)
        joint_names = [str(x) for x in data["joint_names"].tolist()]
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
    return root_pos, root_quat, joint_pos, joint_names, fps


def _build_qpos_indexer(model, joint_names):
    import mujoco

    # Map each NPZ joint name -> its qpos address in the MuJoCo model.
    name_to_adr = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name_to_adr[name] = int(model.jnt_qposadr[i])
    missing = [n for n in joint_names if n not in name_to_adr]
    if missing:
        raise SystemExit(f"[viz] NPZ joints not found in MJCF: {missing[:5]}")
    free_adr = int(model.jnt_qposadr[0])  # free base is the first joint
    return free_adr, [name_to_adr[n] for n in joint_names]


def _set_frame(model, data, t, root_pos, root_quat, joint_pos, free_adr, joint_adrs):
    data.qpos[free_adr : free_adr + 3] = root_pos[t]
    data.qpos[free_adr + 3 : free_adr + 7] = root_quat[t]  # wxyz, matches MuJoCo
    for adr, value in zip(joint_adrs, joint_pos[t]):
        data.qpos[adr] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Output MP4 (offscreen).")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--distance", type=float, default=3.0)
    parser.add_argument("--elevation", type=float, default=-15.0)
    parser.add_argument("--azimuth", type=float, default=90.0)
    args = parser.parse_args()

    if not args.viewer and args.out is None:
        parser.error("Provide --out for an MP4, or --viewer for the interactive viewer.")
    # Offscreen rendering must use a headless GL backend; set before importing mujoco.
    if not args.viewer and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    import mujoco

    npz_path = args.npz.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    root_pos, root_quat, joint_pos, joint_names, fps = _load_frames(npz_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    free_adr, joint_adrs = _build_qpos_indexer(model, joint_names)
    total = root_pos.shape[0]

    if args.viewer:
        import mujoco.viewer

        print(f"[viz] interactive viewer: {npz_path.name} ({total} frames @ {fps:.0f} fps)")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            dt = 1.0 / fps
            t = 0
            while viewer.is_running():
                start = time.perf_counter()
                _set_frame(model, data, t, root_pos, root_quat, joint_pos, free_adr, joint_adrs)
                mujoco.mj_forward(model, data)
                viewer.sync()
                t = (t + 1) % total
                sleep = dt - (time.perf_counter() - start)
                if sleep > 0:
                    time.sleep(sleep)
        return

    import imageio.v2 as imageio

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The MJCF's default offscreen framebuffer is 640x480; grow it to fit.
    model.vis.global_.offwidth = max(args.width, int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(args.height, int(model.vis.global_.offheight))
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance = args.distance
    cam.elevation = args.elevation
    cam.azimuth = args.azimuth
    writer = imageio.get_writer(
        str(out_path), fps=max(1, int(round(fps))), codec="libx264", macro_block_size=None
    )
    try:
        for t in range(0, total, max(1, args.stride)):
            _set_frame(model, data, t, root_pos, root_quat, joint_pos, free_adr, joint_adrs)
            mujoco.mj_forward(model, data)
            # Track the pelvis so the robot stays framed as it translates.
            cam.lookat[:] = data.qpos[free_adr : free_adr + 3]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    finally:
        writer.close()
    print(f"[viz] wrote {out_path} ({total // max(1, args.stride)} frames @ {fps:.0f} fps)")
    print(str(out_path))


if __name__ == "__main__":
    sys.exit(main())
