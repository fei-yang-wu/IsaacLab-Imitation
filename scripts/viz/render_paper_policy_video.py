# Copyright (c) 2026, IsaacLab-Imitation Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E402  # Isaac entrypoint: imports must follow AppLauncher.

"""Render paper-ready, policy-only videos of a low-level tracking checkpoint.

One video per requested trajectory rank, in one process. Unlike
``compare_policy_reference.py`` this script draws NO reference robot and NO
marker overlays: the frame contains only the policy-driven robot on a clean
studio floor, so the video argues quality, smoothness, and expressivity
directly, the way the SONIC paper's clips present a single performing robot.

Presentation choices baked in:

- **Studio scene.** The training env's Nucleus marble/grid floor is replaced
  with a plain matte ground (``--style light`` or ``--style dark``) plus the
  sky dome light and one angled key light, so the robot reads with soft
  shadows and no texture noise. No dependency on a Nucleus MDL download.
- **Follow camera.** The Kit recording camera chases the robot root with an
  exponentially smoothed pursuit (``--camera_tau``), from a fixed azimuth and
  distance (``--camera_azimuth_deg``, ``--camera_distance``,
  ``--camera_height``), optionally orbiting slowly
  (``--orbit_deg_per_s``). A static camera loses a traveling clip; a raw
  hard-locked camera transmits every impact to the frame.
- **Full clips.** Every rank plays from frame 0 to its reference's final
  frame. All termination and reward terms are disabled; a stumble stays in
  frame instead of resetting.
- **Deterministic.** Domain randomization and pushes are disabled and actions
  use the policy mode.

The Kit RTX camera is the only backend with real lighting, and it exists only
under PhysX, so this script REFUSES a Newton physics selection. Paper numbers
still come from the Newton evaluator; these videos are presentation renders
and the physics backend is recorded in the summary JSON.

Example (latent hold-1 recipe; everything after the flags is Hydra):

.. code-block:: bash

    pixi run -e isaaclab python scripts/viz/render_paper_policy_video.py \\
        --checkpoint <model_step_N.pt> \\
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \\
        --ranks 606 467 551 \\
        --output_dir logs/showcase_videos/paper_reel \\
        --style light --headless \\
        physics=physx env.data.reference_arrays_dir=... <latent overrides>
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Render paper-ready policy-only videos for a tracking checkpoint."
)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v2")
parser.add_argument("--algo", type=str, default="IPMD", choices=["PPO", "SAC", "IPMD"])
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)."
)
parser.add_argument(
    "--agent_entry_point",
    type=str,
    required=True,
    help=(
        "Gym registry agent-config entry point. Required: the tuned checkpoints "
        "do not load under the default architecture."
    ),
)
parser.add_argument(
    "--ranks",
    type=int,
    nargs="+",
    required=True,
    help="Trajectory ranks to render, one video each, in order.",
)
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Videos and the summary JSON land here.",
)
parser.add_argument(
    "--style",
    type=str,
    default="light",
    choices=["light", "dark"],
    help="Studio floor/lighting palette.",
)
parser.add_argument(
    "--backdrop",
    type=str,
    default="infinite",
    choices=["infinite", "sky"],
    help=(
        "'infinite' replaces the sky HDR with a uniform dome, so the floor "
        "fades into the background with no horizon line -- the slab edge "
        "otherwise draws a dark line across a paper figure. 'sky' keeps the "
        "training env's HDR."
    ),
)
parser.add_argument(
    "--camera_distance",
    type=float,
    default=3.4,
    help="Horizontal chase distance from the robot root (m).",
)
parser.add_argument(
    "--camera_height",
    type=float,
    default=1.5,
    help="Camera height above the ground (m).",
)
parser.add_argument(
    "--camera_azimuth_deg",
    type=float,
    default=215.0,
    help="World-frame azimuth of the camera around the robot (deg).",
)
parser.add_argument(
    "--orbit_deg_per_s",
    type=float,
    default=0.0,
    help="Slow orbit rate; 0 keeps a fixed azimuth.",
)
parser.add_argument(
    "--camera_tau",
    type=float,
    default=0.4,
    help="Pursuit smoothing time constant (s); larger = calmer camera.",
)
parser.add_argument(
    "--lookat_height",
    type=float,
    default=0.75,
    help="Height of the camera target above the smoothed root (m).",
)
parser.add_argument("--video_width", type=int, default=1920)
parser.add_argument("--video_height", type=int, default=1080)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# The recorder needs the offscreen camera pipeline.
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import os
import tempfile
from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMD, PPO, SAC
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
    bind_command_interface,
)

from imitation_experiments.provenance.paper_protocol_metadata import (
    disable_domain_randomization,
)

ALGORITHM_CLASS_MAP = {"PPO": PPO, "SAC": SAC, "IPMD": IPMD}

# Studio palettes: (floor RGB, dome intensity, key-light intensity).
_STYLES = {
    "light": {
        # Mid-gray: a white robot must separate from the floor, and near-white
        # blows out under the sky dome.
        "floor_color": (0.55, 0.56, 0.60),
        "floor_roughness": 0.85,
        # Matched to the floor's luminance at the horizon so the floor fades
        # into the backdrop without a visible seam.
        "backdrop_color": (0.87, 0.88, 0.90),
        "dome_intensity": 900.0,
        "key_intensity": 4000.0,
        "key_color": (1.0, 0.98, 0.94),
    },
    "dark": {
        "floor_color": (0.16, 0.17, 0.19),
        "floor_roughness": 0.7,
        "backdrop_color": (0.09, 0.10, 0.12),
        "dome_intensity": 450.0,
        "key_intensity": 3500.0,
        "key_color": (1.0, 0.97, 0.9),
    },
}


def _require_kit_camera_physics(env_cfg) -> str:
    """Refuse a physics selection whose recorder is the Newton GL viewer."""
    physics_name = type(env_cfg.sim.physics).__name__
    if "newton" in physics_name.lower():
        raise SystemExit(
            "render_paper_policy_video.py records through the Kit RTX camera, "
            "which only exists under PhysX. Pass `physics=physx`. "
            f"(got physics={physics_name})"
        )
    return physics_name


def _apply_studio_scene(env_cfg, style: dict, backdrop: str) -> None:
    """Plain matte floor + sky dome + one angled key light, no Nucleus MDL.

    The plane terrain type always spawns Isaac's grid ``default_environment``
    USD -- ``visual_material`` only tints it. So the physics ground stays (and
    is hidden after creation, see ``_hide_grid_ground``) while a visual-only
    matte slab with its top face exactly at z=0 provides the studio floor.
    """
    from isaaclab.assets import AssetBaseCfg

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.studio_floor = AssetBaseCfg(
        prim_path="/World/studioFloor",
        spawn=sim_utils.CuboidCfg(
            # Large enough that the slab's dark side face sits at the horizon
            # blur instead of drawing a hard line across the frame.
            size=(20000.0, 20000.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=style["floor_color"],
                roughness=style["floor_roughness"],
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )
    sky = getattr(env_cfg.scene, "sky_light", None)
    if sky is not None:
        sky.spawn.intensity = style["dome_intensity"]
        if backdrop == "infinite":
            # A uniform dome IS the background: no sky texture, no horizon,
            # and the studio slab fades into it like a cyclorama.
            sky.spawn.texture_file = None
            sky.spawn.color = style["backdrop_color"]
    # A key light gives the robot a defined shadow; the dome alone is flat.
    env_cfg.scene.key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=style["key_intensity"],
            color=style["key_color"],
            angle=1.5,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.933, 0.25, 0.25, 0.067),  # tilted key from the camera side
        ),
    )


def _hide_grid_ground() -> None:
    """Make the physics grid plane invisible; the studio slab is the visual."""
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom

    prim = get_current_stage().GetPrimAtPath("/World/ground")
    if prim is not None and prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()


def _disable_all_terminations(env_cfg) -> None:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return
    for name in list(vars(terminations)):
        if name.startswith("_"):
            continue
        setattr(terminations, name, None)


def _disable_all_rewards(env_cfg) -> None:
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return
    for name in list(vars(rewards)):
        if name.startswith("_"):
            continue
        term = getattr(rewards, name)
        if term is not None and hasattr(term, "weight"):
            term.weight = 0.0


def _unwrap_imitation_env(env):
    inner = env
    while hasattr(inner, "env") or hasattr(inner, "unwrapped"):
        candidate = getattr(inner, "unwrapped", None)
        if candidate is not None and candidate is not inner:
            inner = candidate
            continue
        nxt = getattr(inner, "env", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    return inner


def _force_trajectory_on_reset(base_env, *, rank: int, start_step: int) -> None:
    """Pin every reset to one rank/frame (v2 command-term aware).

    Mirrors ``compare_policy_reference._force_policy_trajectory_on_reset``:
    the v2 reference command term owns start selection and would otherwise
    bypass the trajectory manager's custom rank callback.
    """
    tm = base_env.trajectory_manager

    def _custom_reset_fn(env_ids: torch.Tensor, _num: int) -> torch.Tensor:
        return torch.full(
            (int(env_ids.numel()),), int(rank), dtype=torch.long, device=env_ids.device
        )

    tm.reset_schedule = "custom"
    tm.custom_reset_fn = _custom_reset_fn
    tm.reset_start_step = int(start_step)

    reference_term = getattr(base_env, "reference_command", None)
    selection = getattr(getattr(reference_term, "cfg", None), "selection", None)
    if selection is not None:
        selection.schedule = "custom"
        selection.full_trajectory = False
        selection.start_mode = "fixed"
        selection.start_frame = int(start_step)
        selection.random_step_min = int(start_step)
        selection.random_step_max = int(start_step)
        selection.adaptive_weight_fn = None
        reference_term._adaptive_failure_reset_sampler = None
        reference_term._build_reset_samplers()

    if hasattr(base_env, "_random_reset_full_trajectory"):
        base_env._random_reset_full_trajectory = False
    if hasattr(base_env, "_random_reset_step_min"):
        base_env._random_reset_step_min = 0
    if hasattr(base_env, "_random_reset_step_max"):
        base_env._random_reset_step_max = 0


class _FollowCamera:
    """Exponentially smoothed chase camera driving the Kit recording prim.

    The Kit video capture samples ``/OmniverseKit_Persp`` every frame but only
    positions it once at construction, so re-aiming the prim each step is the
    supported way to get a moving recorded camera.
    """

    def __init__(self, base_env, args) -> None:
        self._env = base_env
        self._tau = max(1.0e-3, float(args.camera_tau))
        self._distance = float(args.camera_distance)
        self._height = float(args.camera_height)
        self._azimuth = math.radians(float(args.camera_azimuth_deg))
        self._orbit_rate = math.radians(float(args.orbit_deg_per_s))
        self._lookat_height = float(args.lookat_height)
        self._smoothed: torch.Tensor | None = None
        from isaacsim.core.rendering_manager import ViewportManager

        self._viewport_manager = ViewportManager

    def reset(self) -> None:
        self._smoothed = None
        self.update(snap=True)

    def update(self, snap: bool = False) -> None:
        root = self._env.robot.data.root_pos_w
        root = (root.torch if hasattr(root, "torch") else root)[0].detach().cpu()
        target = root.clone()
        target[2] = 0.0
        if self._smoothed is None or snap:
            self._smoothed = target
        else:
            dt = float(self._env.step_dt)
            alpha = 1.0 - math.exp(-dt / self._tau)
            self._smoothed = self._smoothed + alpha * (target - self._smoothed)
            self._azimuth += self._orbit_rate * dt
        center = self._smoothed
        eye = [
            float(center[0]) + self._distance * math.cos(self._azimuth),
            float(center[1]) + self._distance * math.sin(self._azimuth),
            self._height,
        ]
        lookat = [float(center[0]), float(center[1]), self._lookat_height]
        self._viewport_manager.set_camera_view(
            "/OmniverseKit_Persp", eye=eye, target=lookat
        )


def _video_stem(rank: int, motion: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in motion
    ).strip("_")
    return f"rank-{rank:06d}-{safe or 'motion'}"


@hydra_task_config(args_cli.task, args_cli.agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    if bind_command_interface(agent_cfg, env_cfg) is None:
        sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
        if callable(sync_input_keys):
            sync_input_keys()

    physics_name = _require_kit_camera_physics(env_cfg)
    style = _STYLES[args_cli.style]

    env_cfg.scene.num_envs = 1
    agent_cfg.env.num_envs = 1
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    env_cfg.seed = int(args_cli.seed)

    logger_cfg = getattr(agent_cfg, "logger", None)
    if logger_cfg is not None:
        logger_cfg.backend = ""
        logger_cfg.video = False

    _apply_studio_scene(env_cfg, style, str(args_cli.backdrop))
    _disable_all_terminations(env_cfg)
    _disable_all_rewards(env_cfg)
    disable_domain_randomization(env_cfg)
    if hasattr(env_cfg, "video_recorder") and env_cfg.video_recorder is not None:
        env_cfg.video_recorder.window_width = int(args_cli.video_width)
        env_cfg.video_recorder.window_height = int(args_cli.video_height)
    # A long ceiling; each clip is stopped manually at its reference's end.
    env_cfg.episode_length_s = 1.0e9

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base_env = _unwrap_imitation_env(env)
    _hide_grid_ground()

    num_trajectories = int(base_env.trajectory_manager._length.shape[0])
    invalid = [r for r in args_cli.ranks if not 0 <= r < num_trajectories]
    if invalid:
        raise SystemExit(
            f"Ranks {invalid} outside [0, {num_trajectories - 1}] for this source."
        )
    longest = max(
        int(base_env.trajectory_manager._length[int(r)].item()) for r in args_cli.ranks
    )

    video_recorder = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir / "videos"),
        step_trigger=lambda _step: False,  # every clip is started manually
        video_length=longest + 2,
        disable_logger=True,
    )
    env = video_recorder

    env = IsaacLabWrapper(env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(longest + 2)),
    )

    agent = ALGORITHM_CLASS_MAP[args_cli.algo](env=env, config=agent_cfg)

    # Inference-only: strip optimizer state so param-group layout mismatches
    # from differently-configured training runs cannot block the restore.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and (
        "optimizer_state_dict" in payload or "reward_optimizer_state_dict" in payload
    ):
        stripped = {
            key: value
            for key, value in payload.items()
            if key not in ("optimizer_state_dict", "reward_optimizer_state_dict")
        }
        tmp = tempfile.NamedTemporaryFile(
            prefix="paper_video_weights_", suffix=".pt", delete=False
        )
        tmp.close()
        torch.save(stripped, tmp.name)
        agent.load_model(tmp.name)
        os.unlink(tmp.name)
    else:
        agent.load_model(checkpoint_path)

    policy = agent.collector_policy
    policy.eval()

    camera = _FollowCamera(base_env, args_cli)
    results = []

    for index, rank in enumerate(args_cli.ranks):
        _force_trajectory_on_reset(
            base_env, rank=int(rank), start_step=int(args_cli.start_step)
        )
        with torch.inference_mode():
            td = env.reset()
        camera.reset()

        dataset, motion, trajectory = base_env.trajectory_manager.get_env_traj_info(0)
        clip_steps = int(base_env.trajectory_manager._length[int(rank)].item())
        stem = _video_stem(int(rank), motion)
        print(
            f"[RENDER] {index + 1}/{len(args_cli.ranks)} rank={rank} "
            f"motion={motion!r} steps={clip_steps}"
        )
        video_recorder.start_recording(stem)

        timestep = 0
        while simulation_app.is_running():
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                td = policy(td)
                td = env.step(td)
                camera.update()
                td = step_mdp(
                    td, exclude_reward=True, exclude_done=False, exclude_action=True
                )
            timestep += 1
            if bool(base_env.current_reference_is_final_frame()[0].item()):
                break
            if timestep >= clip_steps + 2:
                break

        if video_recorder.recording:
            video_recorder.stop_recording()
        video_path = output_dir / "videos" / f"{stem}.mp4"
        print(f"[RENDER] wrote {video_path}")
        results.append(
            {
                "trajectory_rank": int(rank),
                "dataset": dataset,
                "motion": motion,
                "trajectory": trajectory,
                "steps": timestep,
                "video": str(video_path),
            }
        )

    summary = {
        "checkpoint": checkpoint_path,
        "task": args_cli.task,
        "physics_cfg": physics_name,
        "style": args_cli.style,
        "backdrop": args_cli.backdrop,
        "seed": int(args_cli.seed),
        "resolution": [int(args_cli.video_width), int(args_cli.video_height)],
        "camera": {
            "distance": float(args_cli.camera_distance),
            "height": float(args_cli.camera_height),
            "azimuth_deg": float(args_cli.camera_azimuth_deg),
            "orbit_deg_per_s": float(args_cli.orbit_deg_per_s),
            "tau": float(args_cli.camera_tau),
        },
        "clips": results,
    }
    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[RENDER] summary: {summary_path}")
    for clip in results:
        print(f"[VIDEO] {clip['video']}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
