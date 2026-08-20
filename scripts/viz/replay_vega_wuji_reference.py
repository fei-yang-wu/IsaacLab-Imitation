#!/usr/bin/env python3
# ruff: noqa: E402
"""Replay a retargeted Vega-Wuji reference motion in Isaac Lab.

The script accepts one reference NPZ or a hashed Vega-Wuji manifest. A manifest
motion is selected by index, the articulation is locked to the dataset qpos at
each control step, and the result can be recorded as an MP4. This is a
reference-data check, not a policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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


def _manifest_motion(path: Path, index: int) -> tuple[Path, dict[str, Any]]:
    if path.suffix.lower() != ".json":
        return path, {"name": path.stem}
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
    return motion_path, metadata


motion_path, motion_metadata = _manifest_motion(reference_path, args_cli.motion_index)
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
replay_steps = args_cli.video_length or reference_frames - 1
if replay_steps <= 0:
    raise ValueError("video length must be positive")

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
if is_unqualified_inspection:
    # This script teleports the Reference after each step and is not a training
    # or policy-evaluation path. Keep the escape local and explicit.
    hydra_args.append("env.require_training_qualified_reference=false")
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
    env_cfg.episode_length_s = 1.0e9
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    env_cfg.log_dir = str(args_cli.output_dir.expanduser().resolve())
    terminations = getattr(env_cfg, "terminations", None)
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
    env = _ReferenceStateLock(env)
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
    for _ in range(replay_steps):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                "Reference replay terminated before the requested horizon"
            )

    env.close()
    # Keep metadata scoped to this recorder instead of picking up unrelated
    # MP4 files from nested inspection directories under output_dir.
    videos = sorted((output_dir / "videos").glob("*.mp4"))
    metadata = {
        "task": args_cli.task,
        "physics": args_cli.physics,
        "reference_manifest_or_npz": str(reference_path),
        "selected_motion": str(motion_path),
        "selected_motion_metadata": motion_metadata,
        "reference_metadata": reference_metadata,
        "is_unqualified_inspection": is_unqualified_inspection,
        "is_geometry_qualified_inspection": is_geometry_qualified_inspection,
        "unqualified_inspection_allowed": bool(args_cli.allow_unqualified_inspection),
        "selected_motion_sha256": _sha256(motion_path),
        "reference_frames": reference_frames,
        "replay_steps": replay_steps,
        "max_joint_position_error": env.env.max_joint_position_error
        if isinstance(env, gym.wrappers.RecordVideo)
        else env.max_joint_position_error,
        "max_joint_velocity_error": env.env.max_joint_velocity_error
        if isinstance(env, gym.wrappers.RecordVideo)
        else env.max_joint_velocity_error,
        "video_paths": [str(path.resolve()) for path in videos],
    }
    (output_dir / "replay_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for video in videos:
        print(f"[VIDEO] {video.resolve()}")
    print(f"[INFO] Replay metadata: {(output_dir / 'replay_metadata.json').resolve()}")


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
