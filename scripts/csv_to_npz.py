#!/usr/bin/env python3
# ruff: noqa: E402
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a motion CSV, export an NPZ, and optionally capture an MP4 video.

.. code-block:: bash

    # Usage
    python scripts/csv_to_npz.py -f path_to_input.csv --input_fps 60
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _append_workspace_sources() -> None:
    """Best-effort source path setup for local mono-workspace usage."""
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

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Replay motion from csv file and output to npz file."
)
parser.add_argument(
    "--input_file",
    "-f",
    type=str,
    required=True,
    help="The path to the input motion csv file.",
)
parser.add_argument(
    "--input_fps", type=float, default=60.0, help="The fps of the input motion."
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, help="The name of the motion npz file.")
parser.add_argument(
    "--output_fps", type=float, default=50.0, help="The fps of the output motion."
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Capture an MP4 video during replay.",
)
parser.add_argument(
    "--video_output",
    type=str,
    default=None,
    help="Output MP4 path for the optional replay video.",
)
parser.add_argument(
    "--overwrite_video",
    action="store_true",
    default=False,
    help="Overwrite an existing video at --video_output.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if not args_cli.output_name:
    # generate at the same location as input file
    args_cli.output_name = (
        "/".join(args_cli.input_file.split("/")[:-1])
        + "/"
        + args_cli.input_file.split("/")[-1].replace(".csv", ".npz")
    )
if args_cli.video and args_cli.video_output is None:
    args_cli.video_output = str(Path(args_cli.output_name).with_suffix(".mp4"))
if args_cli.video:
    args_cli.enable_cameras = True

output_name_path = Path(args_cli.output_name).expanduser().resolve()
output_name_path.parent.mkdir(parents=True, exist_ok=True)
args_cli.output_name = str(output_name_path)
if args_cli.video_output is not None:
    video_output_path = Path(args_cli.video_output).expanduser().resolve()
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    if video_output_path.exists() and not args_cli.overwrite_video:
        raise FileExistsError(
            f"Video output exists: {video_output_path}. Use --overwrite_video to replace it."
        )
    args_cli.video_output = str(video_output_path)


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
import omni.kit.app

##
# Pre-defined configs
##
from isaaclab_imitation.assets.robots.unitree import (
    UNITREE_G1_29DOF_CFG as ROBOT_CFG,
)  # Currently only support G1-29dof

if args_cli.video:
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("omni.kit.capture.viewport", True)
    extension_manager.set_extension_enabled_immediate("omni.videoencoding", True)

    import omni.kit.viewport.utility as vp_utils
    from omni.kit.capture.viewport import (
        CaptureExtension,
        CaptureOptions,
        CaptureRangeType,
        CaptureRenderPreset,
    )
else:  # pragma: no cover - exercised only when video capture is disabled
    vp_utils = None
    CaptureExtension = None
    CaptureOptions = None
    CaptureRangeType = None
    CaptureRenderPreset = None


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: float,
        output_fps: float,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        # CSV stores quaternions scalar-last (x, y, z, w), which matches the
        # Isaac Lab 3.0 convention, so no reordering is needed anymore.
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self._make_quat_continuous(
            self._normalize_quat(self.motion_base_rots_input)
        )
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}"
        )

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        if self.input_frames < 2:
            raise ValueError("Need at least two frames to interpolate a motion.")
        if np.isclose(float(self.input_fps), float(self.output_fps)):
            self.motion_base_poss = self.motion_base_poss_input
            self.motion_base_rots = self.motion_base_rots_input
            self.motion_dof_poss = self.motion_dof_poss_input
            self.output_frames = int(self.input_frames)
            print(
                f"Motion kept at native fps, frames: {self.output_frames}, fps: {self.output_fps}"
            )
            return

        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(
        self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _normalize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """Normalize WXYZ quaternions."""
        return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _make_quat_continuous(self, quat: torch.Tensor) -> torch.Tensor:
        """Choose quaternion signs consistently over time."""
        quat = quat.clone()
        for index in range(1, quat.shape[0]):
            if torch.dot(quat[index - 1], quat[index]) < 0:
                quat[index] = -quat[index]
        return quat

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots, self.output_dt
        )

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        if rotations.shape[0] < 3:
            return torch.zeros(
                (rotations.shape[0], 3),
                dtype=rotations.dtype,
                device=rotations.device,
            )
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat(
            [omega[:1], omega, omega[-1:]], dim=0
        )  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def _pump_app_once() -> None:
    if hasattr(simulation_app, "update"):
        simulation_app.update()
        return
    omni.kit.app.get_app().update()


def _start_video_capture(output_frames: int):
    if not args_cli.video:
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
    options.fps = int(args_cli.output_fps)
    options.overwrite_existing_frames = True
    if CaptureRenderPreset is not None and hasattr(
        CaptureRenderPreset, "REAL_TIME_PATHTRACING"
    ):
        options.render_preset = CaptureRenderPreset.REAL_TIME_PATHTRACING
    capture.options = options

    if not capture.start():
        raise RuntimeError(f"Failed to start video capture for: {output_path}")

    print("[INFO]: Recording replay video to", output_path)
    return capture


def _wait_for_capture(capture) -> None:
    if capture is None:
        return
    max_updates = 600
    updates = 0
    while not capture.done and simulation_app.is_running() and updates < max_updates:
        _pump_app_once()
        updates += 1
    if capture.done:
        print("[INFO]: Video capture completed:", capture.get_outputs())
    else:
        print("[WARN]: Timed out waiting for video capture to finish.")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=float(args_cli.input_fps),
        output_fps=float(args_cli.output_fps),
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(
        scene.cfg.robot.joint_sdk_names, preserve_order=True
    )[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    capture = _start_video_capture(motion.output_frames)
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.torch.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.torch.clone()
        joint_vel = robot.data.default_joint_vel.torch.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # Ensure link transforms correspond to the state written above. A render
        # call does not reliably run articulation forward kinematics.
        sim.forward()
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        if not file_saved:
            log["body_pos_w"].append(
                robot.data.body_pos_w.torch[0, :].cpu().numpy().copy()
            )
            log["body_quat_w"].append(
                robot.data.body_quat_w.torch[0, :].cpu().numpy().copy()
            )
            log["body_lin_vel_w"].append(
                robot.data.body_lin_vel_w.torch[0, :].cpu().numpy().copy()
            )
            log["body_ang_vel_w"].append(
                robot.data.body_ang_vel_w.torch[0, :].cpu().numpy().copy()
            )

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
            # The NPZ dataset format stores quaternions scalar-first (w, x, y, z),
            # while Isaac Lab 3.0 sim state and this script's internal math are
            # scalar-last (x, y, z, w). Convert at the write boundary.
            log["body_quat_w"] = log["body_quat_w"][..., [3, 0, 1, 2]]

            root_pos = motion.motion_base_poss.cpu().numpy().astype(np.float32)
            root_quat = (
                motion.motion_base_rots[:, [3, 0, 1, 2]]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            root_lin_vel = motion.motion_base_lin_vels.cpu().numpy().astype(np.float32)
            root_ang_vel = motion.motion_base_ang_vels.cpu().numpy().astype(np.float32)
            # Revert of dd1db87: store joint states in the robot's articulation
            # (USD) joint order rather than the SDK/source order. The env applies
            # the reference directly to robot.data.joint_pos.torch (articulation order)
            # with an identity reference->target remap, so saving articulation
            # order is what actually matches the robot. `robot_joint_indexes`
            # maps SDK column k -> articulation slot, mirroring the in-loop
            # scatter used to drive the robot above.
            _dof_pos_art = torch.zeros_like(motion.motion_dof_poss)
            _dof_vel_art = torch.zeros_like(motion.motion_dof_vels)
            _dof_pos_art[:, robot_joint_indexes] = motion.motion_dof_poss
            _dof_vel_art[:, robot_joint_indexes] = motion.motion_dof_vels
            joint_pos_target = _dof_pos_art.cpu().numpy().astype(np.float32)
            joint_vel_target = _dof_vel_art.cpu().numpy().astype(np.float32)
            log["root_pos"] = root_pos
            log["root_quat"] = root_quat
            log["root_lin_vel"] = root_lin_vel
            log["root_ang_vel"] = root_ang_vel
            log["joint_pos"] = joint_pos_target
            log["joint_vel"] = joint_vel_target
            log["qpos"] = np.concatenate(
                [root_pos, root_quat, joint_pos_target], axis=-1
            ).astype(np.float32)
            log["qvel"] = np.concatenate(
                [root_lin_vel, root_ang_vel, joint_vel_target], axis=-1
            ).astype(np.float32)
            # Self-describing: save the honest joint order the arrays are in,
            # i.e. the robot's articulation (USD) joint order. Downstream the
            # loader records this as reference_joint_names and remaps it onto
            # target_joint_names (the robot articulation order), so the data is
            # correct regardless of the source ordering.
            log["joint_names"] = np.asarray(robot.joint_names, dtype=np.str_)

            np.savez(args_cli.output_name, **log)
            print("[INFO]: Motion npz file saved to", args_cli.output_name)
            break

    _wait_for_capture(capture)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # Isaac Sim shutdown can hang after one-shot offline conversion.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
