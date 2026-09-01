"""Render gallery filmstrips as one-scene keyframe strips.

This tool re-renders the latent-semantics gallery filmstrips in a
publication style: for each cluster member it places eight copies of the
G1 robot in one MuJoCo scene, one copy for each sampled reference frame,
on a neutral gray floor with a shadow-casting light. The output is one
wide image for each member, written next to the original Isaac
filmstrips.

The frame selection comes from the gallery's own ``gallery_index.json``
(``sampled_frames`` for each member), so the strips show the exact same
reference-motion frames as the original Isaac renders. Poses come from
the dataset NPZ files (``qpos``: root pose plus 29 joint angles), remapped
by joint name into the repo-owned G1 MJCF.

Run in the default Pixi environment (no Isaac Sim needed):

    MUJOCO_GL=egl pixi run python \
        experiments/qualitative/src/qualitative_filmstrip_scene.py \
        --gallery_dir outputs/.../latent_semantics_gallery
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
G1_MJCF = (
    REPO_ROOT
    / "source/isaaclab_imitation/isaaclab_imitation/assets/unitree/g1_description/g1_29dof_rev_1_0.xml"
)
DEFAULT_NPZ_DIR = REPO_ROOT / "data/bones_seed_sonic_129k_50hz/npz/g1"

FOG_GRAY = "0.825 0.818 0.805"

SCENE_XML = f"""
<mujoco model="filmstrip_scene">
  <statistic extent="10" center="{{cx}} 0 0.8"/>
  <visual>
    <headlight ambient="0.52 0.52 0.52" diffuse="0.14 0.14 0.14" specular="0 0 0"/>
    <quality shadowsize="8192" offsamples="8"/>
    <global offwidth="{{width}}" offheight="{{height}}" fovy="{{fovy}}"/>
    <map fogstart="0.9" fogend="2.5" znear="0.05"/>
    <rgba fog="{FOG_GRAY} 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="{FOG_GRAY}" rgb2="{FOG_GRAY}" width="64" height="64"/>
    <material name="floor_mat" rgba="0.78 0.773 0.76 1" specular="0" shininess="0" reflectance="0"/>
  </asset>
  <worldbody>
    <light directional="true" castshadow="true" pos="{{cx}} -3 6" dir="-0.20 0.30 -1"
           diffuse="0.50 0.50 0.50" specular="0.05 0.05 0.05"/>
    <geom name="floor" type="plane" size="80 80 0.1" pos="{{cx}} 0 0" material="floor_mat"
          group="1" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def _quat_yaw(q: np.ndarray) -> float:
    """Return the yaw angle of a wxyz quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_z(yaw: float, q: np.ndarray) -> np.ndarray:
    """Left-multiply the wxyz quaternion ``q`` by a rotation about z."""
    half = 0.5 * yaw
    rw, rz = math.cos(half), math.sin(half)
    w, x, y, z = q
    return np.array(
        [
            rw * w - rz * z,
            rw * x - rz * y,
            rw * y + rz * x,
            rw * z + rz * w,
        ],
        dtype=np.float64,
    )


class StripRenderer:
    """One compiled multi-robot scene reused for every strip."""

    def __init__(
        self,
        num_slots: int,
        spacing: float,
        width: int,
        height: int,
        fovy: float,
        cam_distance: float,
        cam_elevation: float,
        face_yaw_deg: float,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.num_slots = num_slots
        self.spacing = spacing
        self.face_yaw = math.radians(face_yaw_deg)
        self.cx = 0.5 * (num_slots - 1) * spacing

        scene = mujoco.MjSpec.from_string(
            SCENE_XML.format(cx=self.cx, width=width, height=height, fovy=fovy)
        )
        for i in range(num_slots):
            robot = mujoco.MjSpec.from_file(str(G1_MJCF))
            # The robot file ships its own scene: a keyframe, a checker
            # floor, a black skybox, and a directional light. Eight
            # attached copies would add eight lights and overexpose
            # everything, so strip the robot spec down to the robot itself.
            for obj in (
                list(robot.keys)
                + list(robot.lights)
                + list(robot.worldbody.geoms)
                + list(robot.textures)
                + [m for m in robot.materials if m.name == "groundplane"]
            ):
                robot.delete(obj)
            frame = scene.worldbody.add_frame()
            scene.attach(robot, frame=frame, prefix=f"t{i}_")
        self.model = scene.compile()
        self.data = mujoco.MjData(self.model)

        # Joint order of the npz files, resolved once by name.
        probe = np.load(next(DEFAULT_NPZ_DIR.glob("*.npz")))
        self.npz_joint_names = [str(n) for n in probe["joint_names"]]

        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = [self.cx, 0.0, 0.72]
        self.cam.azimuth = 90.0
        self.cam.elevation = cam_elevation
        self.cam.distance = cam_distance
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = 1  # visual meshes and the floor only

    def render(self, qpos_frames: np.ndarray) -> np.ndarray:
        """Render one strip from ``(num_slots, 36)`` npz qpos rows."""
        mujoco = self._mujoco
        assert qpos_frames.shape == (self.num_slots, 36)

        # Face every copy the same way: cancel the yaw of the first frame
        # so each member reads left to right regardless of how the motion
        # happens to be oriented in the world.
        yaw_fix = self.face_yaw - _quat_yaw(qpos_frames[0, 3:7])

        for i in range(self.num_slots):
            root = qpos_frames[i]
            quat = _quat_mul_z(yaw_fix, root[3:7])
            free = self.data.joint(f"t{i}_floating_base_joint")
            free.qpos[:3] = [i * self.spacing, 0.0, root[2]]
            free.qpos[3:] = quat
            for name, angle in zip(self.npz_joint_names, root[7:]):
                self.data.joint(f"t{i}_{name}").qpos[0] = angle
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        return self.renderer.render()


def _load_name_map(npz_dir: Path) -> dict[str, str]:
    """Map manifest motion names to NPZ basenames.

    The dataset manifest is the authority on how a motion name maps to a
    file (names collapse the double underscore before the actor tag), so
    read the mapping from it instead of guessing.
    """
    manifest = next(npz_dir.parent.parent.glob("*_manifest.json"))
    entries = json.loads(manifest.read_text())["dataset"]["trajectories"]["lafan1_csv"]
    return {str(e["name"]): Path(str(e["path"])).name for e in entries}


def _npz_path(npz_dir: Path, name_map: dict[str, str], motion: str) -> Path:
    basename = name_map.get(motion, f"{motion}.npz")
    path = npz_dir / basename
    if not path.exists():
        raise FileNotFoundError(f"no NPZ for motion '{motion}' in {npz_dir}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery_dir", type=Path, required=True)
    parser.add_argument("--npz_dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--out_subdir", default="filmstrips_scene")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help=(
            "Timesteps per strip. 0 uses the gallery's own sampled_frames; "
            "any other value resamples that many evenly spaced frames from "
            "the same window, so no Isaac re-run is needed to change the "
            "count."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Image width in px; 0 scales with the frame count (300 px each).",
    )
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--fovy", type=float, default=9.0)
    parser.add_argument("--cam_distance", type=float, default=11.5)
    parser.add_argument("--cam_elevation", type=float, default=-7.0)
    parser.add_argument("--spacing", type=float, default=0.95)
    parser.add_argument("--face_yaw_deg", type=float, default=-15.0)
    parser.add_argument(
        "--only", default="", help="Comma list like cluster_000_member_0."
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image

    index = json.loads((args.gallery_dir / "gallery_index.json").read_text())
    out_dir = args.gallery_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    renderer: StripRenderer | None = None
    name_map = _load_name_map(args.npz_dir)
    record: list[dict[str, object]] = []
    cluster_hints: dict[str, list[str]] = {}
    for cluster in index["clusters"]:
        term_hint = [str(t) for t in cluster.get("term_hint") or []]
        cluster_hints[f"cluster_{cluster['cluster']:03d}"] = term_hint
        goals = {
            int(r["env_index"]): str(r.get("language_goal", ""))
            for r in cluster.get("robots") or []
        }
        strips = cluster.get("filmstrips") or []
        for strip in strips:
            name = f"cluster_{cluster['cluster']:03d}_member_{strip['env_index']}"
            if only and name not in only:
                continue
            frames = [int(f) for f in strip["sampled_frames"]]
            if int(args.num_frames) > 0:
                frames = np.unique(
                    np.linspace(min(frames), max(frames), int(args.num_frames))
                    .round()
                    .astype(np.int64)
                ).tolist()
            motion = str(strip["motion"])
            data = np.load(_npz_path(args.npz_dir, name_map, motion))
            qpos = data["qpos"]
            rows = np.stack([qpos[min(f, len(qpos) - 1)] for f in frames]).astype(
                np.float64
            )
            if renderer is None:
                renderer = StripRenderer(
                    num_slots=len(frames),
                    spacing=args.spacing,
                    width=int(args.width) or 300 * len(frames),
                    height=args.height,
                    fovy=args.fovy,
                    cam_distance=args.cam_distance,
                    cam_elevation=args.cam_elevation,
                    face_yaw_deg=args.face_yaw_deg,
                )
            image = renderer.render(rows)
            path = out_dir / f"{name}.png"
            Image.fromarray(image).save(path)
            print(f"[STRIP] {path}")
            record.append(
                {
                    "name": name,
                    "motion": motion,
                    "sampled_frames": frames,
                    "term_hint": term_hint,
                    "language_goal": goals.get(int(strip["env_index"]), ""),
                    "path": str(path),
                }
            )

    (out_dir / "scene_strips.json").write_text(
        json.dumps(
            {
                "gallery_index": str(args.gallery_dir / "gallery_index.json"),
                "mjcf": str(G1_MJCF),
                "npz_dir": str(args.npz_dir),
                "spacing": args.spacing,
                "face_yaw_deg": args.face_yaw_deg,
                "num_frames": int(args.num_frames),
                "cluster_hints": cluster_hints,
                "strips": record,
            },
            indent=2,
        )
    )
    print(f"[DONE] {len(record)} strips -> {out_dir}")


if __name__ == "__main__":
    main()
