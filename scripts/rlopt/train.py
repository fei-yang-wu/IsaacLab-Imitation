# Feiyang Wu (feiyangwu@gatech.edu)
"""Runtime-aware RLOpt training entrypoint for Isaac Lab 3."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from runtime_bootstrap import (
    assert_kit_not_loaded,
    config_contains_type_name,
    install_kit_import_guard,
)

logger = logging.getLogger(__name__)

ALGORITHMS = (
    "PPO",
    "SAC",
    "FASTSAC",
    "IPMD",
    "IPMD_SR",
    "IPMD_BILINEAR",
    "GAIL",
    "AMP",
    "ASE",
)
ENTRY_POINT_ALGORITHM_MAP = {
    "rlopt_ppo_cfg_entry_point": "PPO",
    "rlopt_sac_cfg_entry_point": "SAC",
    "rlopt_fastsac_cfg_entry_point": "FASTSAC",
    "rlopt_ipmd_cfg_entry_point": "IPMD",
    "rlopt_ipmd_sr_cfg_entry_point": "IPMD_SR",
    "rlopt_ipmd_bilinear_cfg_entry_point": "IPMD_BILINEAR",
    "rlopt_gail_cfg_entry_point": "GAIL",
    "rlopt_amp_cfg_entry_point": "AMP",
    "rlopt_ase_cfg_entry_point": "ASE",
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI after an optional strict no-Kit guard is installed."""
    from isaaclab_tasks.utils import add_launcher_args

    parser = argparse.ArgumentParser(description="Train an RLOpt agent with Isaac Lab.")
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos during training.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Video length in environment steps.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="Steps between video recordings.",
    )
    parser.add_argument(
        "--video_width", type=int, default=None, help="Optional render width override."
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=None,
        help="Optional render height override.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of simulated environments."
    )
    parser.add_argument(
        "--task", type=str, default=None, help="Registered Isaac Lab task name."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rlopt_cfg_entry_point",
        help="RLOpt agent configuration entry point.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Environment and agent seed."
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=None,
        help="Metric cadence in environment steps.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint to resume."
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Training rollout iterations."
    )
    parser.add_argument("--export_io_descriptors", action="store_true", default=False)
    parser.add_argument(
        "--algo",
        "--algorithm",
        dest="algorithm",
        type=str.upper,
        default="PPO",
        choices=ALGORITHMS,
        help="RLOpt algorithm; it must match a task config entry point.",
    )
    parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
    parser.add_argument(
        "--assert-kitless",
        action="store_true",
        help="Require Newton and fail if Isaac Sim or Omniverse Kit is imported.",
    )
    parser.add_argument(
        "--match-sonic-release-overrides",
        action="store_true",
        help=(
            "Apply SONIC release-only overrides that are easy to miss in its "
            "composed config: mass scale [0.8, 2.5] and max grad norm 0.1."
        ),
    )
    add_launcher_args(parser)
    return parser


def _parse_args(argv: list[str]) -> argparse.Namespace:
    from isaaclab_tasks.utils import setup_preset_cli

    parser = _build_parser()
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    if args_cli.video:
        args_cli.enable_cameras = True
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli


def _resolve_agent_cfg_entry_point(
    task_name: str, agent_entry_point: str, algorithm: str
) -> str:
    """Resolve the algorithm-specific RLOpt task entry point."""
    import gymnasium as gym

    if agent_entry_point != "rlopt_cfg_entry_point":
        return agent_entry_point
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    try:
        spec = gym.spec(task_id)
    except Exception as exc:
        raise ValueError(
            f"Could not resolve task {task_id!r} from the Gym registry."
        ) from exc

    if spec.kwargs.get(algo_entry_point) is not None:
        print(f"[INFO] Using agent config entry point: {algo_entry_point}")
        return algo_entry_point

    supported_algorithms = sorted(
        ENTRY_POINT_ALGORITHM_MAP[key]
        for key in ENTRY_POINT_ALGORITHM_MAP
        if spec.kwargs.get(key) is not None
    )
    raise ValueError(
        f"Task {task_id!r} does not expose an RLOpt config for {algorithm!r}. "
        f"Supported algorithms: {supported_algorithms}."
    )


def _validate_newton_robot_asset(env_cfg: object) -> None:
    """Fail before launch when a kit-less Newton robot USD is unavailable."""
    if not config_contains_type_name(env_cfg, "NewtonCfg"):
        return
    scene = getattr(env_cfg, "scene", None)
    robot = getattr(scene, "robot", None)
    spawn = getattr(robot, "spawn", None)
    usd_path = getattr(spawn, "usd_path", None)
    if not isinstance(usd_path, str) or "://" in usd_path:
        return
    if not Path(usd_path).is_file():
        raise FileNotFoundError(
            "The kit-less Newton path requires the G1 USD root layer, which "
            f"does not exist: {usd_path}. The repo packages the official "
            "Unitree asset via git-lfs; run `git lfs pull` to materialize it."
        )


def _enable_bird_video_visualizer(
    env_cfg: object,
    *,
    width: int | None,
    height: int | None,
) -> None:
    """Attach a headless Newton visualizer for cheap bird-view video capture.

    The video recorder prefers a live Newton visualizer as its capture
    backend and follows that visualizer's camera. Capping the visualizer to a
    small block of environments near the grid center keeps the GL viewer's
    per-frame instance submission cheap at any training scale (the full-scene
    viewer costs ~18 s/frame at 8192 environments), while still showing
    several robots plus the ground for motion-quality checks.
    """
    import math

    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    num_envs = int(getattr(env_cfg.scene, "num_envs", 16) or 16)
    grid_cols = max(int(math.ceil(math.sqrt(num_envs))), 1)
    center = (grid_cols // 2) * grid_cols + grid_cols // 2
    block: list[int] = []
    for row in range(4):
        for col in range(4):
            idx = center + row * grid_cols + col
            if 0 <= idx < num_envs:
                block.append(idx)
    if not block:
        block = list(range(min(16, num_envs)))

    visualizer = NewtonVisualizerCfg(
        headless=True,
        window_width=int(width or 1280),
        window_height=int(height or 720),
        eye=(11.0, 11.0, 7.0),
        lookat=(0.0, 0.0, 0.8),
        visible_env_indices=block,
        # Keep shadows enabled: newton's RendererGL only initializes its
        # light-space matrix when shadows are on, and _render_scene
        # references it unconditionally (AttributeError otherwise).
    )
    existing = list(getattr(env_cfg.sim, "visualizer_cfgs", None) or [])
    env_cfg.sim.visualizer_cfgs = existing + [visualizer]
    print(
        "[INFO] Headless Newton bird-view visualizer enabled "
        f"(envs {block[0]}..{block[-1]} of {num_envs})."
    )


def _apply_sonic_release_overrides(
    env_cfg: object, agent_cfg: object, *, task_name: str
) -> None:
    """Apply values from SONIC's final release layer after config composition."""
    if "Sonic" not in task_name:
        raise ValueError(
            f"--match-sonic-release-overrides requires a SONIC task; got {task_name!r}."
        )
    events = getattr(env_cfg, "events", None)
    mass_event = getattr(events, "randomize_rigid_body_mass", None)
    if mass_event is None:
        raise RuntimeError("The resolved SONIC task has no rigid-body mass event.")
    mass_event.params["mass_distribution_params"] = (0.8, 2.5)

    optim = getattr(agent_cfg, "optim", None)
    if optim is None or not hasattr(optim, "max_grad_norm"):
        raise RuntimeError("The resolved SONIC agent has no optim.max_grad_norm field.")
    optim.max_grad_norm = 0.1
    print("[INFO] SONIC release overrides: mass scale [0.8, 2.5], max grad norm 0.1")


def run(argv: list[str] | None = None, *, require_running_kit: bool = False) -> int:
    """Resolve configuration, own the simulation context, and run training."""
    if argv is None:
        argv = sys.argv[1:]
    strict_kitless = "--assert-kitless" in argv
    if strict_kitless:
        install_kit_import_guard()

    args_cli = _parse_args(argv)
    if args_cli.task is None:
        raise ValueError("--task is required for RLOpt training.")

    # Task registration is intentionally after the optional Kit-first bootstrap.
    import isaaclab_imitation.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils import has_kit
    from isaaclab_tasks.utils import (
        compute_kit_requirements,
        launch_simulation,
        resolve_task_config,
    )

    args_cli.agent = _resolve_agent_cfg_entry_point(
        args_cli.task, args_cli.agent, args_cli.algorithm
    )
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    if args_cli.video and config_contains_type_name(env_cfg, "NewtonCfg"):
        _enable_bird_video_visualizer(
            env_cfg, width=args_cli.video_width, height=args_cli.video_height
        )
    if args_cli.match_sonic_release_overrides:
        _apply_sonic_release_overrides(env_cfg, agent_cfg, task_name=args_cli.task)
    needs_kit, _, _ = compute_kit_requirements(env_cfg, args_cli)
    _validate_newton_robot_asset(env_cfg)

    if strict_kitless:
        if needs_kit or not config_contains_type_name(env_cfg, "NewtonCfg"):
            raise RuntimeError(
                "--assert-kitless requires a resolved NewtonCfg with no Kit cameras or Kit visualizer. "
                "Pass physics=newton_mjwarp and a kit-less renderer."
            )
        assert_kit_not_loaded()
        print(
            "[INFO] Strict kit-less Newton runtime validated before simulation startup."
        )

    if require_running_kit and not has_kit():
        raise RuntimeError(
            "The PhysX bootstrap did not start SimulationApp before RLOpt configuration loading."
        )
    if os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1" and needs_kit and not has_kit():
        raise RuntimeError(
            "The split runtime requires PhysX to start through scripts/rlopt/train_physx.py "
            "with /isaac-sim/python.sh."
        )

    with launch_simulation(env_cfg, args_cli):
        from train_impl import train

        train(env_cfg, agent_cfg, args_cli)

    if strict_kitless:
        assert_kit_not_loaded()
        print("[INFO] Strict kit-less invariant held through RLOpt shutdown.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper that preserves a nonzero exit code through Isaac shutdown."""
    try:
        return run(argv)
    except Exception:
        logger.exception("Unhandled exception during RLOpt training.")
        logging.shutdown()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
