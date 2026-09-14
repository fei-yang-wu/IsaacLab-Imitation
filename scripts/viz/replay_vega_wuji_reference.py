#!/usr/bin/env python3
# ruff: noqa: E402
"""Validate a retargeted Vega-Wuji Reference in Isaac Lab.

The script accepts one reference NPZ or a hashed Vega-Wuji manifest. A manifest
motion is selected by index, the articulation is locked to the dataset qpos at
each control step by default. ``--unassisted-dynamics`` instead applies zero
policy residuals, so the normal joint-position actuators track the Reference
while the virtual object controller stays exactly off. This is a Reference and
oracle-dynamics check, not a learned-policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--motion-index", type=int, default=0)
parser.add_argument("--task", default="Isaac-Imitation-Vega-Wuji-v0")
parser.add_argument(
    "--physics", choices=("default", "newton_mjwarp"), default="newton_mjwarp"
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video-length", type=int, default=None)
parser.add_argument(
    "--unassisted-dynamics",
    action="store_true",
    help=(
        "Do not teleport state after each step. Apply zero Reference-residual "
        "actions, force virtual object control to zero from reset, retain the "
        "normal safety terminations, and report dynamic tracking errors."
    ),
)
parser.add_argument(
    "--allow-unqualified-inspection",
    action="store_true",
    help=(
        "Allow replay of a Reference whose metadata explicitly marks it as "
        "not qualified for Isaac runtime training, provided its source, emitted, "
        "dense, rendered, and can-support scene-geometry audits all pass with at "
        "least 5 mm robot clearance. This is visual inspection only and never a "
        "motion/contact certificate."
    ),
)
parser.add_argument(
    "--output-dir", type=Path, default=Path("logs/reference_replay/vega_wuji")
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--debug-vis",
    action="store_true",
    default=False,
    help="Draw command markers: live and commanded wrist, object, and "
    "fingertip frames. Contact markers appear only when the Reference "
    "carries active contact geometry.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

STATE_LOCK_JOINT_POSITION_TOLERANCE_RAD = 1.0e-5
STATE_LOCK_JOINT_VELOCITY_TOLERANCE_RAD_S = 1.0e-5
STATE_LOCK_OBJECT_POSITION_TOLERANCE_M = 1.0e-5
STATE_LOCK_OBJECT_ORIENTATION_TOLERANCE_RAD = 1.0e-5
STATE_LOCK_OBJECT_TWIST_TOLERANCE = 1.0e-5

if args_cli.video:
    args_cli.enable_cameras = True

reference_path = args_cli.reference.expanduser().resolve()
if not reference_path.is_file():
    raise FileNotFoundError(f"Reference file not found: {reference_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_motion(
    path: Path, index: int
) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    if path.suffix.lower() != ".json":
        return path, {"name": path.stem}, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("motions") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Reference manifest has no motions: {path}")
    if not 0 <= index < len(entries):
        raise IndexError(f"motion-index {index} is outside [0, {len(entries)})")
    entry = entries[index]
    metadata = dict(entry) if isinstance(entry, dict) else {"path": entry}
    motion_path = Path(str(metadata["path"]))
    if not motion_path.is_absolute():
        motion_path = path.parent / motion_path
    motion_path = motion_path.resolve()
    expected_hash = metadata.get("sha256")
    if expected_hash is not None and str(expected_hash).lower() != _sha256(motion_path):
        raise ValueError(f"Manifest hash mismatch: {motion_path}")
    return motion_path, metadata, payload if isinstance(payload, dict) else None


motion_path, motion_metadata, manifest_payload = _manifest_motion(
    reference_path, args_cli.motion_index
)
manifest_sha256 = _sha256(reference_path) if manifest_payload is not None else None
manifest_model_path: Path | None = None
manifest_model_sha256: str | None = None
if manifest_payload is not None:
    model_value = manifest_payload.get("model")
    manifest_model_sha256 = manifest_payload.get("model_sha256")
    if not isinstance(model_value, str) or not isinstance(manifest_model_sha256, str):
        raise ValueError("Reference manifest must bind model and model_sha256.")
    manifest_model_path = Path(model_value)
    if not manifest_model_path.is_absolute():
        manifest_model_path = reference_path.parent / manifest_model_path
    manifest_model_path = manifest_model_path.resolve()
    if _sha256(manifest_model_path) != manifest_model_sha256.lower():
        raise ValueError(f"Manifest model hash mismatch: {manifest_model_path}")
if args_cli.unassisted_dynamics:
    if manifest_payload is None:
        raise ValueError(
            "Unassisted validation requires a hash-bound JSON manifest; direct "
            "NPZ input is inspection-only."
        )
    if not isinstance(motion_metadata.get("sha256"), str):
        raise ValueError("Unassisted validation requires a manifest motion hash.")
with np.load(motion_path, allow_pickle=False) as data:
    if "qpos" not in data:
        raise ValueError(f"Reference NPZ must contain qpos: {motion_path}")
    reference_frames = int(np.asarray(data["qpos"]).shape[0])
    reference_metadata: dict[str, Any] = {}
    if "metadata_json" in data:
        metadata_value = np.asarray(data["metadata_json"]).item()
        if isinstance(metadata_value, bytes):
            metadata_value = metadata_value.decode("utf-8")
        reference_metadata = json.loads(str(metadata_value))
        if not isinstance(reference_metadata, dict):
            raise ValueError("Reference metadata_json must decode to an object.")
is_unqualified_inspection = (
    reference_metadata.get("isaac_runtime_qualified") is False
    or reference_metadata.get("inspection_only") is True
)
is_geometry_qualified_inspection = False
if reference_metadata.get("inspection_only") is True:
    geometry_audit = reference_metadata.get("geometry_clearance_audit")
    is_geometry_qualified_inspection = bool(
        isinstance(geometry_audit, dict)
        and reference_metadata.get("inspection_scene_geometry_qualified") is True
        and geometry_audit.get("qualified") is True
        and float(geometry_audit.get("required_clearance_m", 0.0)) >= 0.005
        and all(
            isinstance(geometry_audit.get(rate), dict)
            and geometry_audit[rate].get("qualified") is True
            for rate in (
                "source_200hz",
                "emitted_20hz",
                "source_path_dense_1khz",
                "emitted_path_dense_1khz",
                "rendered_geometry_20hz",
                "source_path_rendered_dense_1khz",
                "emitted_path_rendered_dense_1khz",
                "source_path_can_support_dense_1khz",
                "emitted_path_can_support_dense_1khz",
            )
        )
    )
    if not is_geometry_qualified_inspection:
        raise ValueError(
            "Inspection-only Reference is missing one or more required source, "
            "emitted, dense, rendered, or can-support scene-geometry audits. "
            "Refusing to render a potentially penetrative trajectory."
        )
if is_unqualified_inspection and not args_cli.allow_unqualified_inspection:
    raise ValueError(
        "Reference metadata marks this motion as not qualified for Isaac "
        "runtime training. Pass --allow-unqualified-inspection only for a "
        "clearly labeled visual inspection."
    )
if reference_frames < 2:
    raise ValueError(f"Reference must contain at least two frames: {motion_path}")
full_horizon_steps = (
    reference_frames if args_cli.unassisted_dynamics else reference_frames - 1
)
replay_steps = args_cli.video_length or full_horizon_steps
if replay_steps <= 0 or replay_steps > full_horizon_steps:
    raise ValueError(
        f"video length must be in [1, {full_horizon_steps}], got {replay_steps}"
    )
if args_cli.unassisted_dynamics and replay_steps != full_horizon_steps:
    raise ValueError("Unassisted validation must run the full Reference horizon.")

# Keep a JSON Manifest intact so the environment binds its model hash to the
# runtime MJCF. The fixed index keeps replay deterministic.
hydra_args.extend(
    [
        f"env.reference_path={reference_path}",
        "env.randomize_reference=false",
        "env.commands.motion.always_reset_to_first_frame=true",
        f"env.reference_motion_index={args_cli.motion_index}",
        f"physics={args_cli.physics}",
    ]
)
if args_cli.unassisted_dynamics:
    hydra_args.extend(
        [
            "env.commands.motion.warmup_steps=0",
            "env.commands.motion.force_unassisted_object_control=true",
        ]
    )
if is_unqualified_inspection:
    # This script teleports the Reference after each step and is not a training
    # or policy-evaluation path. Keep the escape local and explicit.
    hydra_args.append("env.require_training_qualified_reference=false")
    os.environ["ISAACLAB_IMITATION_VEGA_WUJI_INSPECTION_REPLAY"] = "1"
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config


class _ReferenceStateLock(gym.Wrapper):
    """Keep the visualized articulation on the selected reference frame."""

    def __init__(self, env):
        super().__init__(env)
        self.max_joint_position_error = 0.0
        self.max_joint_velocity_error = 0.0
        self.max_object_position_error = 0.0
        self.max_object_orientation_error = 0.0
        self.max_object_twist_error = 0.0

    def _apply(self) -> None:
        base_env = self.env.unwrapped
        base_env.write_reference_state_to_sim((0,))
        robot = base_env.scene["robot"]
        self.max_joint_position_error = max(
            self.max_joint_position_error,
            float(
                torch.max(
                    torch.abs(
                        robot.data.joint_pos.torch[0]
                        - base_env.current_reference_joint_pos[0]
                    )
                ).item()
            ),
        )
        self.max_joint_velocity_error = max(
            self.max_joint_velocity_error,
            float(
                torch.max(
                    torch.abs(
                        robot.data.joint_vel.torch[0]
                        - base_env.current_reference_joint_vel[0]
                    )
                ).item()
            ),
        )
        motion_command = base_env.command_manager.get_term("motion")
        self.max_object_position_error = max(
            self.max_object_position_error,
            float(
                torch.max(
                    torch.abs(
                        motion_command.object_position_e[0]
                        - motion_command.object_body_position_command_e[0]
                    )
                ).item()
            ),
        )
        orientation_dot = torch.sum(
            motion_command.object_orientation_e[0]
            * motion_command.object_body_wxyz_command_e[0],
            dim=-1,
        )
        orientation_error = 2.0 * torch.acos(orientation_dot.abs().clamp(0.0, 1.0))
        self.max_object_orientation_error = max(
            self.max_object_orientation_error,
            float(torch.max(orientation_error).item()),
        )
        self.max_object_twist_error = max(
            self.max_object_twist_error,
            float(
                torch.max(
                    torch.abs(
                        motion_command.object_twist_w[0]
                        - motion_command.object_body_twist_command_w[0]
                    )
                ).item()
            ),
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.max_joint_position_error <= STATE_LOCK_JOINT_POSITION_TOLERANCE_RAD
            and self.max_joint_velocity_error
            <= STATE_LOCK_JOINT_VELOCITY_TOLERANCE_RAD_S
            and self.max_object_position_error <= STATE_LOCK_OBJECT_POSITION_TOLERANCE_M
            and self.max_object_orientation_error
            <= STATE_LOCK_OBJECT_ORIENTATION_TOLERANCE_RAD
            and self.max_object_twist_error <= STATE_LOCK_OBJECT_TWIST_TOLERANCE
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        base_env = self.env.unwrapped
        base_env.set_reference_motion(args_cli.motion_index, (0,))
        self._apply()
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        self._apply()
        return result


@hydra_task_config(args_cli.task, "rlopt_ppo_cfg_entry_point")
def main(env_cfg, agent_cfg):  # noqa: ARG001
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.commands.motion.debug_vis = args_cli.debug_vis
    env_cfg.episode_length_s = 1.0e9
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.log_dir = str(args_cli.output_dir.expanduser().resolve())
    terminations = getattr(env_cfg, "terminations", None)
    if args_cli.unassisted_dynamics:
        # The fixed replay horizon owns timeout. Keep every physical/tracking
        # termination so an oracle-dynamics failure remains visible.
        if terminations is not None and hasattr(terminations, "time_out"):
            terminations.time_out = None
        env_cfg.curriculum = None
    else:
        if terminations is not None:
            for name in getattr(terminations, "__dataclass_fields__", {}):
                if getattr(terminations, name, None) is not None:
                    setattr(terminations, name, None)
        rewards = getattr(env_cfg, "rewards", None)
        if rewards is not None:
            for name in getattr(rewards, "__dataclass_fields__", {}):
                if getattr(rewards, name, None) is not None:
                    setattr(rewards, name, None)
        env_cfg.curriculum = None

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    state_lock: _ReferenceStateLock | None = None
    if not args_cli.unassisted_dynamics:
        state_lock = _ReferenceStateLock(env)
        env = state_lock
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(output_dir / "videos"),
            # RecordVideo captures the reset frame only for episode-triggered
            # recordings.  A step trigger at zero starts after the first step
            # and silently drops Reference frame zero.
            "episode_trigger": lambda episode: episode == 0,
            "video_length": replay_steps,
            "disable_logger": True,
        }
        print("[INFO] Recording reference replay video")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    action = torch.zeros(
        env.unwrapped.action_manager.total_action_dim,
        dtype=torch.float32,
        device=env.unwrapped.device,
    ).reshape(1, -1)
    env.reset()
    task_env = env.unwrapped
    motion_command = task_env.command_manager.get_term("motion")

    def virtual_object_controller_scale() -> float:
        scale = motion_command.virtual_object_controller_scale
        return float(torch.abs(scale).max().item())

    reset_virtual_object_controller_scale = virtual_object_controller_scale()
    assistance_violation_step: int | None = None
    if args_cli.unassisted_dynamics and reset_virtual_object_controller_scale != 0.0:
        assistance_violation_step = 0
    try:
        support_contact_sensor = task_env.scene["robot_support_contacts"]
    except KeyError:
        support_contact_sensor = None
    dynamics_metrics: list[dict[str, Any]] = []
    robot_joint_names = tuple(task_env.robot.joint_names)
    termination_names = (
        tuple(task_env.termination_manager.active_terms)
        if args_cli.unassisted_dynamics
        else ()
    )
    fired_termination_terms: list[str] = []
    completed_steps = 0
    terminated_early = False
    termination_step: int | None = None
    for step_index in range(replay_steps):
        if assistance_violation_step is not None:
            break
        _, _, terminated, truncated, _ = env.step(action)
        completed_steps = step_index + 1
        done = bool(torch.as_tensor(terminated).any().item())
        timed_out = bool(torch.as_tensor(truncated).any().item())
        if args_cli.unassisted_dynamics:
            reference_pos = task_env.current_reference_joint_pos[0]
            reference_vel = task_env.current_reference_joint_vel[0]
            joint_pos = task_env.robot.data.joint_pos.torch[0]
            joint_vel = task_env.robot.data.joint_vel.torch[0]
            object_position_error = torch.linalg.vector_norm(
                motion_command.object_position_e
                - motion_command.object_body_position_command_e,
                dim=-1,
            )
            object_quaternion_dot = (
                motion_command.object_orientation_e
                * motion_command.object_body_wxyz_command_e
            ).sum(dim=-1)
            object_orientation_error = 2.0 * torch.acos(
                object_quaternion_dot.abs().clamp(0.0, 1.0)
            )
            support_force = 0.0
            if support_contact_sensor is not None:
                force_matrix = support_contact_sensor.data.force_matrix_w
                if force_matrix is None:
                    raise RuntimeError(
                        "robot_support_contacts did not publish force_matrix_w."
                    )
                force_matrix = getattr(force_matrix, "torch", force_matrix)
                support_force = float(
                    torch.linalg.vector_norm(force_matrix[0], dim=-1).max().item()
                )
            pos_error = torch.abs(joint_pos - reference_pos)
            vel_error = torch.abs(joint_vel - reference_vel)
            joint_ranges = (
                task_env.robot.data.soft_joint_pos_limits.torch[0, :, 1]
                - task_env.robot.data.soft_joint_pos_limits.torch[0, :, 0]
            )
            normalized_pos_error = pos_error / joint_ranges
            max_error_joint = int(torch.argmax(pos_error).item())
            max_normalized_error_joint = int(torch.argmax(normalized_pos_error).item())
            current_virtual_object_controller_scale = virtual_object_controller_scale()
            dynamics_metrics.append(
                {
                    "step": step_index + 1,
                    "reference_frame": int(task_env.reference_frame[0].item()),
                    "joint_position_mae_rad": float(pos_error.mean().item()),
                    "joint_position_max_error_rad": float(pos_error.max().item()),
                    "joint_position_max_error_joint": robot_joint_names[
                        max_error_joint
                    ],
                    "joint_position_max_normalized_error": float(
                        normalized_pos_error[max_normalized_error_joint].item()
                    ),
                    "joint_position_max_normalized_error_joint": robot_joint_names[
                        max_normalized_error_joint
                    ],
                    "joint_position_max_normalized_error_target_rad": float(
                        reference_pos[max_normalized_error_joint].item()
                    ),
                    "joint_position_max_normalized_error_live_rad": float(
                        joint_pos[max_normalized_error_joint].item()
                    ),
                    "joint_velocity_mae_rad_s": float(vel_error.mean().item()),
                    "object_position_error_m": float(
                        object_position_error.max().item()
                    ),
                    "object_orientation_error_rad": float(
                        object_orientation_error.max().item()
                    ),
                    "support_contact_force_max_n": support_force,
                    "virtual_object_controller_scale": (
                        current_virtual_object_controller_scale
                    ),
                }
            )
            if current_virtual_object_controller_scale != 0.0:
                assistance_violation_step = step_index + 1
        if done or timed_out:
            terminated_early = step_index + 1 < replay_steps
            termination_step = step_index + 1
            fired_termination_terms = [
                name
                for name in termination_names
                if bool(task_env.termination_manager.get_term(name).any().item())
            ]
            break

    env.close()
    # Keep metadata scoped to this recorder instead of picking up unrelated
    # MP4 files from nested inspection directories under output_dir.
    videos = sorted((output_dir / "videos").glob("*.mp4"))
    metadata = {
        "task": args_cli.task,
        "mode": (
            "unassisted_zero_residual_dynamics"
            if args_cli.unassisted_dynamics
            else "reference_state_lock"
        ),
        "physics": args_cli.physics,
        "reference_manifest_or_npz": str(reference_path),
        "selected_motion": str(motion_path),
        "selected_motion_metadata": motion_metadata,
        "reference_metadata": reference_metadata,
        "is_unqualified_inspection": is_unqualified_inspection,
        "is_geometry_qualified_inspection": is_geometry_qualified_inspection,
        "unqualified_inspection_allowed": bool(args_cli.allow_unqualified_inspection),
        "selected_motion_sha256": _sha256(motion_path),
        "reference_manifest_sha256": manifest_sha256,
        "manifest_model_path": (
            None if manifest_model_path is None else str(manifest_model_path)
        ),
        "manifest_model_sha256": manifest_model_sha256,
        "reference_frames": reference_frames,
        # Keep the original key for existing result readers.
        "replay_steps": replay_steps,
        "requested_replay_steps": replay_steps,
        "completed_replay_steps": completed_steps,
        "completed_without_termination": termination_step is None,
        "terminated_early": terminated_early,
        "termination_step": termination_step,
        "fired_termination_terms": fired_termination_terms,
        # State-lock needs T-1 transitions because reset supplies frame zero
        # and the wrapper teleports to each newly advanced frame. Unassisted
        # dynamics needs T: action processing targets the current command
        # before the manager advances it, so transition T is what actually
        # actuator-targets and checks final frame T-1.
        "full_reference_horizon_requested": replay_steps == full_horizon_steps,
        "full_reference_horizon_completed": bool(
            completed_steps == full_horizon_steps
            and termination_step is None
            and assistance_violation_step is None
            and (state_lock is None or state_lock.passed)
        ),
        "reset_virtual_object_controller_scale": (
            reset_virtual_object_controller_scale
        ),
        "zero_virtual_object_controller_verified": bool(
            reset_virtual_object_controller_scale == 0.0
            and assistance_violation_step is None
        ),
        "assistance_violation_step": assistance_violation_step,
        "max_joint_position_error": (
            None if state_lock is None else state_lock.max_joint_position_error
        ),
        "max_joint_velocity_error": (
            None if state_lock is None else state_lock.max_joint_velocity_error
        ),
        "state_lock": (
            None
            if state_lock is None
            else {
                "passed": state_lock.passed,
                "joint_position_tolerance_rad": (
                    STATE_LOCK_JOINT_POSITION_TOLERANCE_RAD
                ),
                "joint_velocity_tolerance_rad_s": (
                    STATE_LOCK_JOINT_VELOCITY_TOLERANCE_RAD_S
                ),
                "object_position_tolerance_m": (STATE_LOCK_OBJECT_POSITION_TOLERANCE_M),
                "object_orientation_tolerance_rad": (
                    STATE_LOCK_OBJECT_ORIENTATION_TOLERANCE_RAD
                ),
                "object_twist_tolerance": STATE_LOCK_OBJECT_TWIST_TOLERANCE,
                "max_joint_position_error_rad": (state_lock.max_joint_position_error),
                "max_joint_velocity_error_rad_s": (state_lock.max_joint_velocity_error),
                "max_object_position_error_m": (state_lock.max_object_position_error),
                "max_object_orientation_error_rad": (
                    state_lock.max_object_orientation_error
                ),
                "max_object_twist_error": state_lock.max_object_twist_error,
            }
        ),
        "video_paths": [str(path.resolve()) for path in videos],
    }
    if dynamics_metrics:
        metadata["unassisted_dynamics"] = {
            "mean_joint_position_mae_rad": float(
                np.mean([row["joint_position_mae_rad"] for row in dynamics_metrics])
            ),
            "max_joint_position_error_rad": float(
                np.max(
                    [row["joint_position_max_error_rad"] for row in dynamics_metrics]
                )
            ),
            "max_joint_position_normalized_error": float(
                np.max(
                    [
                        row["joint_position_max_normalized_error"]
                        for row in dynamics_metrics
                    ]
                )
            ),
            "mean_joint_velocity_mae_rad_s": float(
                np.mean([row["joint_velocity_mae_rad_s"] for row in dynamics_metrics])
            ),
            "mean_object_position_error_m": float(
                np.mean([row["object_position_error_m"] for row in dynamics_metrics])
            ),
            "max_object_position_error_m": float(
                np.max([row["object_position_error_m"] for row in dynamics_metrics])
            ),
            "mean_object_orientation_error_rad": float(
                np.mean(
                    [row["object_orientation_error_rad"] for row in dynamics_metrics]
                )
            ),
            "max_object_orientation_error_rad": float(
                np.max(
                    [row["object_orientation_error_rad"] for row in dynamics_metrics]
                )
            ),
            "max_support_contact_force_n": float(
                np.max([row["support_contact_force_max_n"] for row in dynamics_metrics])
            ),
            "max_virtual_object_controller_scale": float(
                np.max(
                    [row["virtual_object_controller_scale"] for row in dynamics_metrics]
                )
            ),
            "per_step": dynamics_metrics,
        }
    (output_dir / "replay_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for video in videos:
        print(f"[VIDEO] {video.resolve()}")
    print(f"[INFO] Replay metadata: {(output_dir / 'replay_metadata.json').resolve()}")
    if args_cli.unassisted_dynamics and assistance_violation_step is not None:
        raise RuntimeError(
            "Unassisted Reference dynamics observed nonzero virtual object "
            f"controller scale at step {assistance_violation_step}."
        )
    if args_cli.unassisted_dynamics and termination_step is not None:
        raise RuntimeError(
            "Unassisted Reference dynamics fired a normal termination at "
            f"step {termination_step}: {fired_termination_terms}."
        )
    if state_lock is not None and not state_lock.passed:
        raise RuntimeError(
            "Reference state-lock validation exceeded a robot or object "
            "state tolerance. See replay_metadata.json for exact errors."
        )


if __name__ == "__main__":
    failed = False
    try:
        main()
    except Exception:  # noqa: BLE001 - preserve a nonzero CLI status after Kit closes
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        raise SystemExit(1)
