#!/usr/bin/env python3
# ruff: noqa: E402

"""Replay a Unitree LeRobot episode segment on the Isaac G1 model."""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def _append_workspace_sources() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    workspace_root = repo_root.parent
    candidate_paths = [
        repo_root / "IsaacLab" / "source" / "isaaclab",
        repo_root / "IsaacLab" / "source" / "isaaclab_tasks",
        repo_root / "source" / "isaaclab_imitation",
        workspace_root / "IsaacLab" / "source" / "isaaclab",
        workspace_root / "IsaacLab" / "source" / "isaaclab_tasks",
    ]
    for candidate in candidate_paths:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)


_append_workspace_sources()

from isaaclab.app import AppLauncher


def _default_video_output(repo_id: str, episode_index: int) -> str:
    repo_slug = repo_id.replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(
        Path("logs")
        / "unitree_lerobot_replay"
        / f"{repo_slug}_ep{episode_index}_{stamp}.mp4"
    )


parser = argparse.ArgumentParser(
    description=(
        "Stream a Unitree WBT LeRobot episode from Hugging Face and replay "
        "a 36-wide root+joint configuration field on the Isaac G1 model."
    )
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="unitreerobotics/G1_WBT_Brainco_Pickup_Pillow",
    help="Hugging Face dataset repo id.",
)
parser.add_argument("--split", type=str, default="train", help="Dataset split.")
parser.add_argument(
    "--episode_index", type=int, default=0, help="Episode index to replay."
)
parser.add_argument(
    "--state_field",
    type=str,
    default="action.robot_q_desired",
    help=(
        "36-wide Unitree configuration field to replay. Use action.robot_q_desired "
        "to visualize the command/label sequence, or observation.state.robot_q_current "
        "to inspect measured robot tracking."
    ),
)
parser.add_argument(
    "--quat_order",
    type=str,
    choices=("wxyz", "xyzw"),
    default="wxyz",
    help="Root quaternion order in --state_field.",
)
parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="Input LeRobot frame rate. Unitree WBT rows are currently 30 Hz.",
)
parser.add_argument(
    "--output_fps",
    type=float,
    default=None,
    help=(
        "Output replay/export frame rate after interpolation. Defaults to --fps. "
        "Use 30 for native Unitree WBT control-rate playback."
    ),
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=180,
    help="Maximum number of episode frames to stream and replay.",
)
parser.add_argument(
    "--max_scan_rows",
    type=int,
    default=100_000,
    help="Maximum streamed rows to scan while looking for --episode_index.",
)
parser.add_argument(
    "--video_output",
    type=str,
    default=None,
    help="Output MP4 path. Defaults under logs/unitree_lerobot_replay/.",
)
parser.add_argument(
    "--overwrite_video",
    action="store_true",
    default=False,
    help="Overwrite an existing --video_output.",
)
parser.add_argument(
    "--npz_output",
    type=str,
    default=None,
    help=(
        "Optional LAFAN-style NPZ output with qpos/qvel, joint states, and "
        "Isaac body states at --output_fps."
    ),
)
parser.add_argument(
    "--overwrite_npz",
    action="store_true",
    default=False,
    help="Overwrite an existing --npz_output.",
)
parser.add_argument(
    "--no_video",
    action="store_true",
    default=False,
    help="Replay without recording an MP4.",
)
parser.add_argument(
    "--root_z_alignment",
    type=str,
    choices=("first_frame_to_default", "none"),
    default="none",
    help=(
        "How to align LeRobot root height before replay/export. The default preserves "
        "the recorded root trajectory. first_frame_to_default adds a constant z offset "
        "so frame 0 matches the robot default root z."
    ),
)
parser.add_argument(
    "--print_joint_debug",
    action="store_true",
    default=False,
    help="Print full dataset->Unitree SDK->Isaac articulation joint mapping.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.fps <= 0:
    raise ValueError(f"--fps must be positive, got {args_cli.fps}")
if args_cli.output_fps is None:
    args_cli.output_fps = float(args_cli.fps)
if args_cli.output_fps <= 0:
    raise ValueError(f"--output_fps must be positive, got {args_cli.output_fps}")
if args_cli.max_frames < 2:
    raise ValueError(f"--max_frames must be at least 2, got {args_cli.max_frames}")
if args_cli.max_scan_rows < args_cli.max_frames:
    raise ValueError(
        f"--max_scan_rows must be >= --max_frames, got {args_cli.max_scan_rows}"
    )

if not args_cli.no_video:
    args_cli.enable_cameras = True
    if args_cli.video_output is None:
        args_cli.video_output = _default_video_output(
            args_cli.repo_id, args_cli.episode_index
        )
    video_output_path = Path(args_cli.video_output).expanduser().resolve()
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    if video_output_path.exists() and not args_cli.overwrite_video:
        raise FileExistsError(
            f"Video output exists: {video_output_path}. "
            "Use --overwrite_video to replace it."
        )
    args_cli.video_output = str(video_output_path)
if args_cli.npz_output is not None:
    npz_output_path = Path(args_cli.npz_output).expanduser().resolve()
    npz_output_path.parent.mkdir(parents=True, exist_ok=True)
    if npz_output_path.exists() and not args_cli.overwrite_npz:
        raise FileExistsError(
            f"NPZ output exists: {npz_output_path}. Use --overwrite_npz to replace it."
        )
    args_cli.npz_output = str(npz_output_path)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from datasets import load_dataset

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_slerp
import omni.kit.app

from isaaclab_imitation.assets.robots.unitree import (
    UNITREE_G1_29DOF_JOINT_ORDER_SOURCE,
    UNITREE_G1_29DOF_URDF_REVOLUTE_JOINT_NAMES,
    UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES,
    UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES,
    UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG,
)

if not args_cli.no_video:
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("omni.kit.capture.viewport", True)
    extension_manager.set_extension_enabled_immediate("omni.videoencoding", True)

    import omni.kit.viewport.utility as vp_utils
    from omni.kit.capture.viewport import (
        CaptureExtension,
        CaptureOptions,
        CaptureRangeType,
    )
else:
    vp_utils = None
    CaptureExtension = None
    CaptureOptions = None
    CaptureRangeType = None


@dataclass(frozen=True)
class UnitreeLeRobotMotion:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    fps: float
    repo_id: str
    split: str
    episode_index: int
    state_field: str

    @property
    def frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


def _format_indexed_names(names: list[str] | tuple[str, ...]) -> str:
    return ", ".join(f"{index}:{name}" for index, name in enumerate(names))


def _default_robot_root_z() -> float:
    return float(ROBOT_CFG.init_state.pos[2])


def _align_root_z_to_default(root_pos: torch.Tensor) -> tuple[torch.Tensor, float]:
    if args_cli.root_z_alignment == "none":
        return root_pos, 0.0
    aligned_root_pos = root_pos.clone()
    root_z_offset = _default_robot_root_z() - float(aligned_root_pos[0, 2].item())
    aligned_root_pos[:, 2] += root_z_offset
    return aligned_root_pos, root_z_offset


def _nested_row_get(row: dict, key: str):
    if key in row:
        return row[key]
    value = row
    for part in key.split("."):
        value = value[part]
    return value


def _row_episode_index(row: dict) -> int:
    return int(torch.as_tensor(_nested_row_get(row, "episode_index")).item())


def _row_frame_index(row: dict) -> int:
    return int(torch.as_tensor(_nested_row_get(row, "frame_index")).item())


def _normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat_norm = quat.norm(dim=-1, keepdim=True)
    if torch.any(quat_norm == 0):
        raise ValueError("Root quaternion contains a zero-norm sample.")
    return quat / quat_norm


def _make_quat_continuous_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.clone()
    for index in range(1, quat.shape[0]):
        if torch.dot(quat[index - 1], quat[index]) < 0:
            quat[index] = -quat[index]
    return quat


def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1.0 - blend) + b * blend


def _slerp_quat_wxyz(
    a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
    out = torch.zeros_like(a)
    for index in range(a.shape[0]):
        out[index] = quat_slerp(a[index], b[index], blend[index])
    return out


def _interpolate_unitree_motion(
    *,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    input_fps: float,
    output_fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        root_pos.shape[0] != root_quat.shape[0]
        or root_pos.shape[0] != joint_pos.shape[0]
    ):
        raise ValueError("root_pos, root_quat, and joint_pos frame counts must match.")
    if root_pos.shape[0] < 2:
        raise ValueError("Need at least two frames to interpolate a motion.")
    if np.isclose(float(input_fps), float(output_fps)):
        return root_pos, root_quat, joint_pos

    input_dt = 1.0 / float(input_fps)
    output_dt = 1.0 / float(output_fps)
    duration = (int(root_pos.shape[0]) - 1) * input_dt
    times = torch.arange(
        0.0,
        duration,
        output_dt,
        device=root_pos.device,
        dtype=root_pos.dtype,
    )
    if times.numel() == 0:
        times = torch.zeros((1,), device=root_pos.device, dtype=root_pos.dtype)
    phase = times / duration
    index_0 = torch.floor(phase * (root_pos.shape[0] - 1)).to(dtype=torch.int64)
    index_1 = torch.minimum(
        index_0 + 1,
        torch.full_like(index_0, root_pos.shape[0] - 1),
    )
    blend = phase * (root_pos.shape[0] - 1) - index_0.to(dtype=root_pos.dtype)
    return (
        _lerp(root_pos[index_0], root_pos[index_1], blend.unsqueeze(1)),
        _make_quat_continuous_wxyz(
            _normalize_quat_wxyz(
                _slerp_quat_wxyz(root_quat[index_0], root_quat[index_1], blend)
            )
        ),
        _lerp(joint_pos[index_0], joint_pos[index_1], blend.unsqueeze(1)),
    )


def _load_unitree_episode() -> UnitreeLeRobotMotion:
    dataset = load_dataset(args_cli.repo_id, split=args_cli.split, streaming=True)
    rows = []
    scanned_rows = 0
    for row in dataset:
        scanned_rows += 1
        episode_index = _row_episode_index(row)
        if episode_index == args_cli.episode_index:
            rows.append(row)
            if len(rows) >= args_cli.max_frames:
                break
        elif rows:
            break
        if scanned_rows >= args_cli.max_scan_rows:
            break

    if len(rows) < 2:
        raise RuntimeError(
            f"Found {len(rows)} rows for episode {args_cli.episode_index} "
            f"after scanning {scanned_rows} rows from {args_cli.repo_id}/{args_cli.split}."
        )

    rows.sort(key=_row_frame_index)
    q_current = torch.stack(
        [
            torch.as_tensor(
                _nested_row_get(row, args_cli.state_field), dtype=torch.float32
            )
            for row in rows
        ],
        dim=0,
    )
    if q_current.ndim != 2 or q_current.shape[1] != 36:
        raise ValueError(
            f"{args_cli.state_field} must have shape [T, 36], got "
            f"{tuple(q_current.shape)}"
        )

    root_pos = q_current[:, 0:3]
    root_quat = q_current[:, 3:7]
    first_raw_quat = root_quat[0].detach().cpu()
    first_quat_dominant_slot = int(first_raw_quat.abs().argmax().item())
    if args_cli.print_joint_debug:
        print("[DEBUG]: First raw root quaternion q[3:7]:", first_raw_quat.tolist())
        print(
            "[DEBUG]: First raw root quaternion dominant slot:",
            first_quat_dominant_slot,
        )
    if args_cli.quat_order == "wxyz" and first_quat_dominant_slot == 3:
        print(
            "[WARN]: First root quaternion looks xyzw-like, but --quat_order=wxyz "
            "was selected."
        )
    elif args_cli.quat_order == "xyzw" and first_quat_dominant_slot == 0:
        print(
            "[WARN]: First root quaternion looks wxyz-like, but --quat_order=xyzw "
            "was selected."
        )
    if args_cli.quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    root_quat = _make_quat_continuous_wxyz(_normalize_quat_wxyz(root_quat))
    joint_pos = q_current[:, 7:]
    if joint_pos.shape[1] != 29:
        raise ValueError(
            f"G1 replay expects 29 joint positions, got {joint_pos.shape[1]}"
        )
    root_pos, root_quat, joint_pos = _interpolate_unitree_motion(
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=joint_pos,
        input_fps=float(args_cli.fps),
        output_fps=float(args_cli.output_fps),
    )
    raw_root_pos = root_pos
    root_pos, root_z_offset = _align_root_z_to_default(root_pos)

    print(
        "[INFO]: Loaded Unitree LeRobot episode:",
        f"repo={args_cli.repo_id}",
        f"split={args_cli.split}",
        f"episode={args_cli.episode_index}",
        f"field={args_cli.state_field}",
        f"input_frames={q_current.shape[0]}",
        f"output_frames={root_pos.shape[0]}",
        f"input_fps={args_cli.fps}",
        f"output_fps={args_cli.output_fps}",
        f"q_shape={tuple(q_current.shape)}",
    )
    print(
        "[INFO]: Root z alignment:",
        f"mode={args_cli.root_z_alignment}",
        f"default_root_z={_default_robot_root_z():.4f}",
        f"raw_first_z={float(raw_root_pos[0, 2].item()):.4f}",
        f"offset={root_z_offset:.4f}",
    )
    print(
        "[INFO]: Root xyz first/mid/last:",
        root_pos[0].tolist(),
        root_pos[root_pos.shape[0] // 2].tolist(),
        root_pos[-1].tolist(),
    )
    root_z = root_pos[:, 2]
    print(
        "[INFO]: Root z min/max:",
        f"{float(root_z.min().item()):.4f}",
        f"{float(root_z.max().item()):.4f}",
    )
    if float(root_z.min().item()) < 0.5:
        print(
            "[WARN]: This segment's recorded root z drops below 0.50 m. "
            "A low or prone-looking replay may be present in the source motion, "
            "not necessarily caused by joint reordering."
        )

    return UnitreeLeRobotMotion(
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=joint_pos,
        fps=float(args_cli.output_fps),
        repo_id=args_cli.repo_id,
        split=args_cli.split,
        episode_index=int(args_cli.episode_index),
        state_field=args_cli.state_field,
    )


unitree_motion = _load_unitree_episode()


@configclass
class ReplayUnitreeLeRobotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _axis_angle_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat_wxyz(quat)
    vector = quat[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quat[..., :1].clamp(-1.0, 1.0))
    axis = vector / vector_norm.clamp_min(1.0e-8)
    return axis * angle


def _save_npz_output(log: dict[str, list[np.ndarray] | float | list[str]]) -> None:
    if args_cli.npz_output is None:
        return
    root_pos = np.stack(log["root_pos"], axis=0).astype(np.float32)  # type: ignore[arg-type]
    root_quat = np.stack(log["root_quat"], axis=0).astype(np.float32)  # type: ignore[arg-type]
    root_lin_vel = np.stack(log["root_lin_vel"], axis=0).astype(np.float32)  # type: ignore[arg-type]
    root_ang_vel = np.stack(log["root_ang_vel"], axis=0).astype(np.float32)  # type: ignore[arg-type]
    joint_pos = np.stack(log["joint_pos"], axis=0).astype(np.float32)  # type: ignore[arg-type]
    joint_vel = np.stack(log["joint_vel"], axis=0).astype(np.float32)  # type: ignore[arg-type]
    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=-1).astype(np.float32)
    qvel = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=-1).astype(
        np.float32
    )
    np.savez(
        args_cli.npz_output,
        fps=np.asarray([float(args_cli.output_fps)], dtype=np.float32),
        qpos=qpos,
        qvel=qvel,
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=np.stack(log["body_pos_w"], axis=0).astype(np.float32),  # type: ignore[arg-type]
        body_quat_w=np.stack(log["body_quat_w"], axis=0).astype(np.float32),  # type: ignore[arg-type]
        body_lin_vel_w=np.stack(log["body_lin_vel_w"], axis=0).astype(np.float32),  # type: ignore[arg-type]
        body_ang_vel_w=np.stack(log["body_ang_vel_w"], axis=0).astype(np.float32),  # type: ignore[arg-type]
        joint_names=np.asarray(log["joint_names"], dtype=np.str_),  # type: ignore[arg-type]
    )
    print("[INFO]: Motion NPZ saved to", args_cli.npz_output)


class IsaacReplayMotion:
    def __init__(self, motion: UnitreeLeRobotMotion, device: torch.device):
        self.device = device
        self.fps = motion.fps
        self.dt = motion.dt
        self.root_pos = motion.root_pos.to(device)
        self.root_quat = motion.root_quat.to(device)
        self.joint_pos = motion.joint_pos.to(device)
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.dt, dim=0)[0]
        self.joint_vel = torch.gradient(self.joint_pos, spacing=self.dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_quat, self.dt)
        self.current_index = 0

    @property
    def frames(self) -> int:
        return int(self.root_pos.shape[0])

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        if rotations.shape[0] < 3:
            return torch.zeros(
                (rotations.shape[0], 3),
                dtype=rotations.dtype,
                device=rotations.device,
            )
        q_prev = rotations[:-2]
        q_next = rotations[2:]
        q_rel = _quat_mul_wxyz(q_next, _quat_conjugate_wxyz(q_prev))
        omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        index = self.current_index
        self.current_index += 1
        reset = self.current_index >= self.frames
        if reset:
            self.current_index = 0
        state = (
            self.root_pos[index : index + 1],
            self.root_quat[index : index + 1],
            self.root_lin_vel[index : index + 1],
            self.root_ang_vel[index : index + 1],
            self.joint_pos[index : index + 1],
            self.joint_vel[index : index + 1],
        )
        return state, reset


def _pump_app_once() -> None:
    if hasattr(simulation_app, "update"):
        simulation_app.update()
        return
    omni.kit.app.get_app().update()


def _start_video_capture(output_frames: int):
    if args_cli.no_video:
        return None
    if (
        CaptureExtension is None
        or CaptureOptions is None
        or CaptureRangeType is None
        or vp_utils is None
    ):
        raise RuntimeError(
            "Video capture requested, but viewport capture extensions are unavailable."
        )

    viewport = vp_utils.get_active_viewport()
    if viewport is None:
        raise RuntimeError(
            "Video capture requested, but no active viewport is available."
        )

    output_path = Path(args_cli.video_output).expanduser().resolve()
    capture = CaptureExtension.get_instance()
    options = CaptureOptions()
    options.camera = viewport.camera_path.pathString
    options.output_folder = str(output_path.parent)
    options.file_name = output_path.stem
    options.file_type = output_path.suffix or ".mp4"
    options.range_type = CaptureRangeType.FRAMES
    options.start_frame = 1
    options.end_frame = int(output_frames)
    options.capture_every_Nth_frames = 1
    options.fps = int(round(float(args_cli.output_fps)))
    options.overwrite_existing_frames = True
    capture.options = options

    if not capture.start():
        raise RuntimeError(f"Failed to start video capture for: {output_path}")

    print("[INFO]: Recording replay video to", output_path)
    return capture


def _wait_for_capture(capture) -> None:
    if capture is None:
        return
    updates = 0
    max_updates = 900
    while not capture.done and simulation_app.is_running() and updates < max_updates:
        _pump_app_once()
        updates += 1
    if capture.done:
        print("[INFO]: Video capture completed:", capture.get_outputs())
    else:
        raise TimeoutError("Timed out waiting for video capture to finish.")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    motion = IsaacReplayMotion(unitree_motion, device=sim.device)
    robot = scene["robot"]
    dataset_joint_names = list(UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES)
    target_joint_names = list(scene.cfg.robot.joint_sdk_names)
    if dataset_joint_names != list(UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES):
        raise RuntimeError(
            "Unitree LeRobot dataset joint order must match the vendored G1 XML "
            "motor order."
        )
    if target_joint_names != list(UNITREE_G1_29DOF_URDF_REVOLUTE_JOINT_NAMES):
        raise RuntimeError(
            "Replay target joint order must match the vendored G1 URDF revolute "
            "joint order."
        )
    dataset_name_to_index = {
        joint_name: index for index, joint_name in enumerate(dataset_joint_names)
    }
    dataset_to_target_indexes = torch.tensor(
        [dataset_name_to_index[joint_name] for joint_name in target_joint_names],
        device=sim.device,
        dtype=torch.int64,
    )
    robot_joint_indexes = robot.find_joints(target_joint_names, preserve_order=True)[0]
    robot_joint_names = list(robot.joint_names)
    resolved_robot_joint_names = [
        robot_joint_names[int(index)] for index in robot_joint_indexes
    ]
    if resolved_robot_joint_names != target_joint_names:
        raise RuntimeError(
            "Isaac robot.find_joints(..., preserve_order=True) did not preserve "
            "Unitree target order. Resolved order: "
            f"{resolved_robot_joint_names}"
        )
    print(
        "[INFO]: Unitree LeRobot joint order:",
        "robot_q[7:] uses Unitree SDK order; replay target uses",
        "scene.cfg.robot.joint_sdk_names.",
        f"source={UNITREE_G1_29DOF_JOINT_ORDER_SOURCE}",
    )
    if args_cli.print_joint_debug:
        print(
            "[DEBUG]: Dataset robot_q[7:] order:",
            _format_indexed_names(dataset_joint_names),
        )
        print("[DEBUG]: Target SDK order:", _format_indexed_names(target_joint_names))
        print(
            "[DEBUG]: Isaac articulation order:",
            _format_indexed_names(robot_joint_names),
        )
        print(
            "[DEBUG]: Isaac target joint indexes:",
            [int(index) for index in robot_joint_indexes],
        )
        print(
            "[DEBUG]: Dataset->target index map:",
            [int(index) for index in dataset_to_target_indexes],
        )
    log: dict[str, list[np.ndarray] | float | list[str]] | None = None
    if args_cli.npz_output is not None:
        log = {
            "root_pos": [],
            "root_quat": [],
            "root_lin_vel": [],
            "root_ang_vel": [],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
            "joint_names": target_joint_names,
        }

    first_lookat = motion.root_pos[0].cpu().numpy()
    sim.set_camera_view(first_lookat + np.array([2.0, 2.0, 0.7]), first_lookat)
    capture = _start_video_capture(motion.frames)
    printed_first_write_debug = False

    while simulation_app.is_running():
        (
            (
                root_pos,
                root_quat,
                root_lin_vel,
                root_ang_vel,
                joint_pos_input,
                joint_vel_input,
            ),
            reset,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.torch.clone()
        root_states[:, :3] = root_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        # This script's motion pipeline is scalar-first (w, x, y, z); Isaac Lab
        # 3.0 sim state is scalar-last (x, y, z, w).
        root_states[:, 3:7] = root_quat[:, [1, 2, 3, 0]]
        root_states[:, 7:10] = root_lin_vel
        root_states[:, 10:] = root_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.torch.clone()
        joint_vel = robot.data.default_joint_vel.torch.clone()
        joint_pos_target = joint_pos_input.index_select(1, dataset_to_target_indexes)
        joint_vel_target = joint_vel_input.index_select(1, dataset_to_target_indexes)
        joint_pos[:, robot_joint_indexes] = joint_pos_target
        joint_vel[:, robot_joint_indexes] = joint_vel_target
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        if args_cli.print_joint_debug and not printed_first_write_debug:
            write_error = (
                (joint_pos[:, robot_joint_indexes] - joint_pos_target).abs().max()
            )
            print(
                "[DEBUG]: First-frame dataset joint vector:",
                joint_pos_input[0].detach().cpu().tolist(),
            )
            print(
                "[DEBUG]: First-frame target joint vector:",
                joint_pos_target[0].detach().cpu().tolist(),
            )
            print(
                "[DEBUG]: First-frame Isaac write max_abs_error:",
                f"{float(write_error.item()):.6g}",
            )
            printed_first_write_debug = True

        lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(lookat + np.array([2.0, 2.0, 0.7]), lookat)
        sim.render()
        scene.update(sim.get_physics_dt())

        if log is not None:
            log["root_pos"].append(root_pos[0].cpu().numpy().copy())
            log["root_quat"].append(root_quat[0].cpu().numpy().copy())
            log["root_lin_vel"].append(root_lin_vel[0].cpu().numpy().copy())
            log["root_ang_vel"].append(root_ang_vel[0].cpu().numpy().copy())
            log["joint_pos"].append(joint_pos_target[0].cpu().numpy().copy())
            log["joint_vel"].append(joint_vel_target[0].cpu().numpy().copy())
            log["body_pos_w"].append(
                robot.data.body_pos_w.torch[0].cpu().numpy().copy()
            )
            log["body_quat_w"].append(
                robot.data.body_quat_w.torch[0].cpu().numpy().copy()
            )
            log["body_lin_vel_w"].append(
                robot.data.body_lin_vel_w.torch[0].cpu().numpy().copy()
            )
            log["body_ang_vel_w"].append(
                robot.data.body_ang_vel_w.torch[0].cpu().numpy().copy()
            )

        if reset:
            break

    if log is not None:
        _save_npz_output(log)
    _wait_for_capture(capture)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.output_fps)
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayUnitreeLeRobotSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    # Isaac Sim can hang during shutdown after offline replay/export. This script is
    # a one-shot converter, so exit once requested files are written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
