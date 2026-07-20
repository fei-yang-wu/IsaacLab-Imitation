#!/usr/bin/env python3
"""Compare G1 reference-action tracking across Isaac Lab physics backends.

The probe deliberately avoids a learned policy.  At every control step it sends
the transition-aligned reconstructed reference action already exposed by the G1
imitation environment, then records joint/body tracking error and the exact
termination causes.  This isolates robot, actuator, contact, and termination
behavior from PPO/IPMD optimization.

Examples (run from the repository root):

.. code-block:: bash

    pixi run -e isaaclab python scripts/diagnose_g1_dynamics.py \
        --task Isaac-Imitation-G1-Latent-v0 --num_envs 128 --steps 500 \
        --assert-kitless physics=newton_mjwarp \
        env.lafan1_manifest_path=data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json

    pixi run -e isaaclab python scripts/diagnose_g1_dynamics.py \
        --task Isaac-Imitation-G1-Latent-v0 --num_envs 128 --steps 500 \
        physics=physx \
        env.lafan1_manifest_path=data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_HELPER_DIR = SCRIPT_DIR / "rlopt"
if str(RUNTIME_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_HELPER_DIR))

from runtime_bootstrap import (  # noqa: E402
    assert_kit_not_loaded,
    config_contains_type_name,
    detect_gpu_names,
    install_kit_import_guard,
    requested_backend,
    validate_gpu_policy,
)


logger = logging.getLogger(__name__)
DEFAULT_TASK = "Isaac-Imitation-G1-Latent-v0"
DEFAULT_OUTPUT_ROOT = Path("logs/dynamics_diagnostics")
EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def _build_parser() -> argparse.ArgumentParser:
    from isaaclab_tasks.utils import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("reference", "zero"),
        default="reference",
        help="Reference is the transition-aligned next-pose action; zero is a control.",
    )
    parser.add_argument(
        "--training-randomization",
        action="store_true",
        help="Deprecated alias for --randomization-profile all.",
    )
    parser.add_argument(
        "--randomization-profile",
        choices=("none", "startup", "reset", "all"),
        default="none",
        help=(
            "Select SONIC randomization for attribution: startup keeps asset/domain "
            "randomization, reset keeps reference-state perturbations, and all also "
            "keeps interval pushes."
        ),
    )
    parser.add_argument(
        "--match-sonic-release-overrides",
        action="store_true",
        help="Use SONIC's final [0.8, 2.5] rigid-body mass scale range.",
    )
    parser.add_argument(
        "--full-trajectory-random-starts",
        action="store_true",
        help="Preserve the task's full-trajectory reset sampler instead of pinning frame zero.",
    )
    parser.add_argument(
        "--assert-kitless",
        action="store_true",
        help="Require a strict no-Kit Newton runtime.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: logs/dynamics_diagnostics/<backend>_<timestamp>.json).",
    )
    add_launcher_args(parser)
    return parser


def _parse_args(argv: list[str]) -> argparse.Namespace:
    from isaaclab_tasks.utils import setup_preset_cli

    parser = _build_parser()
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *hydra_args]
    return args_cli


def _disable_training_randomization(env_cfg: object) -> None:
    """Keep reference reset wiring but remove its stochastic perturbations."""
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    for name in (
        "physics_material",
        "add_joint_default_pos",
        "base_com",
        "push_robot",
        "randomize_rigid_body_mass",
    ):
        if hasattr(events, name):
            setattr(events, name, None)

    reset_term = getattr(events, "reset_reference_state", None)
    if reset_term is None:
        return
    reset_term.params["pose_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_term.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_term.params["joint_position_range"] = (0.0, 0.0)


def _configure_randomization_profile(env_cfg: object, profile: str) -> None:
    """Select startup and reset randomization independently for attribution."""
    if profile == "all":
        return
    if profile == "none":
        _disable_training_randomization(env_cfg)
        return

    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    if hasattr(events, "push_robot"):
        events.push_robot = None

    if profile == "reset":
        for name in (
            "physics_material",
            "add_joint_default_pos",
            "base_com",
            "randomize_rigid_body_mass",
        ):
            if hasattr(events, name):
                setattr(events, name, None)
        return

    reset_term = getattr(events, "reset_reference_state", None)
    if reset_term is None:
        return
    reset_term.params["pose_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_term.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    reset_term.params["joint_position_range"] = (0.0, 0.0)


def _apply_sonic_release_overrides(env_cfg: object, *, task_name: str) -> None:
    if "Sonic" not in task_name:
        raise ValueError(
            f"--match-sonic-release-overrides requires a SONIC task; got {task_name!r}."
        )
    events = getattr(env_cfg, "events", None)
    mass_event = getattr(events, "randomize_rigid_body_mass", None)
    if mass_event is None:
        raise RuntimeError("The resolved SONIC task has no rigid-body mass event.")
    mass_event.params["mass_distribution_params"] = (0.8, 2.5)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _named_means(
    names: list[str] | tuple[str, ...], totals: list[float], samples: int
) -> dict:
    if samples == 0:
        return {name: None for name in names}
    return {name: total / samples for name, total in zip(names, totals, strict=True)}


def _named_values(names: list[str] | tuple[str, ...], values: list[float]) -> dict:
    return dict(zip(names, values, strict=True))


def _run_probe(env_cfg: object, args_cli: argparse.Namespace, backend: str) -> dict:
    import gymnasium as gym
    import torch

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.match_sonic_release_overrides:
        _apply_sonic_release_overrides(env_cfg, task_name=args_cli.task)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.reconstructed_reference_action = True
    if not args_cli.full_trajectory_random_starts:
        env_cfg.random_reset_step_min = 0
        env_cfg.random_reset_step_max = 0
        env_cfg.random_reset_full_trajectory = False
    env_cfg.observations.policy.enable_corruption = False
    randomization_profile = args_cli.randomization_profile
    if args_cli.training_randomization:
        if randomization_profile != "none":
            raise ValueError(
                "Use either --training-randomization or --randomization-profile, not both."
            )
        randomization_profile = "all"
    _configure_randomization_profile(env_cfg, randomization_profile)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = torch.device(str(unwrapped.device))
    env.reset(seed=int(args_cli.seed))
    initial_local_steps = (
        unwrapped.trajectory_manager.env_step.detach().to(device="cpu").clone()
    )
    robot_joint_names = list(robot.joint_names)
    robot_body_names = list(robot.body_names)
    action_joint_names = list(
        unwrapped.action_manager.get_term("joint_pos")._joint_names
    )
    robot_spawn_cfg = getattr(env_cfg.scene.robot, "spawn", None)
    robot_usd_path = getattr(robot_spawn_cfg, "usd_path", None)
    mass_event = getattr(
        getattr(env_cfg, "events", None), "randomize_rigid_body_mass", None
    )
    mass_distribution_params = None
    if mass_event is not None:
        mass_distribution_params = mass_event.params.get("mass_distribution_params")

    ee_ids_list, ee_names = robot.find_bodies(EE_BODY_NAMES, preserve_order=True)
    if list(ee_names) != list(EE_BODY_NAMES):
        raise RuntimeError(
            f"Could not resolve ordered G1 end effectors: expected={EE_BODY_NAMES}, got={ee_names}"
        )
    ee_ids = torch.tensor(ee_ids_list, dtype=torch.long, device=device)
    termination_names = tuple(unwrapped.termination_manager.active_terms)
    reward_names = tuple(unwrapped.reward_manager.active_terms)
    termination_counts: dict[str, int] = defaultdict(int)
    episode_ages = torch.zeros(int(args_cli.num_envs), dtype=torch.long, device=device)
    completed_episode_lengths: list[float] = []

    joint_pos_mae: list[float] = []
    joint_pos_max: list[float] = []
    ee_z_mae: list[float] = []
    ee_z_max: list[float] = []
    ee_xyz_mae: list[float] = []
    action_abs_mean: list[float] = []
    applied_torque_abs_mean: list[float] = []
    joint_error_sum = torch.zeros(robot.num_joints, dtype=torch.float64, device=device)
    joint_error_max = torch.zeros(robot.num_joints, dtype=torch.float32, device=device)
    joint_torque_sum = torch.zeros(robot.num_joints, dtype=torch.float64, device=device)
    joint_torque_max = torch.zeros(robot.num_joints, dtype=torch.float32, device=device)
    reward_term_sum = torch.zeros(len(reward_names), dtype=torch.float64, device=device)
    ee_z_error_sum = torch.zeros(len(EE_BODY_NAMES), dtype=torch.float64, device=device)
    ee_z_error_max = torch.zeros(len(EE_BODY_NAMES), dtype=torch.float32, device=device)
    ee_xyz_error_sum = torch.zeros(
        len(EE_BODY_NAMES), dtype=torch.float64, device=device
    )
    ee_xyz_error_max = torch.zeros(
        len(EE_BODY_NAMES), dtype=torch.float32, device=device
    )
    vector_sample_count = 0

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for _ in range(int(args_cli.steps)):
                ref_joint_pos = unwrapped.current_expert_frame["joint_pos"]
                joint_error = (robot.data.joint_pos.torch - ref_joint_pos).abs()
                joint_pos_mae.append(float(joint_error.mean().item()))
                joint_pos_max.append(float(joint_error.max().item()))
                joint_error_sum.add_(joint_error.double().sum(dim=0))
                joint_error_max = torch.maximum(
                    joint_error_max, joint_error.max(dim=0).values
                )

                ref_ee_pos = unwrapped._get_reference_body_pose_w_fast(EE_BODY_NAMES)[0]
                robot_ee_pos = unwrapped._get_robot_body_pose_w_fast(ee_ids)[0]
                ee_delta = (robot_ee_pos - ref_ee_pos).abs()
                ee_z_mae.append(float(ee_delta[..., 2].mean().item()))
                ee_z_max.append(float(ee_delta[..., 2].max().item()))
                ee_xyz_mae.append(
                    float(torch.linalg.vector_norm(ee_delta, dim=-1).mean().item())
                )
                ee_xyz_error = torch.linalg.vector_norm(ee_delta, dim=-1)
                ee_z_error_sum.add_(ee_delta[..., 2].double().sum(dim=0))
                ee_z_error_max = torch.maximum(
                    ee_z_error_max, ee_delta[..., 2].max(dim=0).values
                )
                ee_xyz_error_sum.add_(ee_xyz_error.double().sum(dim=0))
                ee_xyz_error_max = torch.maximum(
                    ee_xyz_error_max, ee_xyz_error.max(dim=0).values
                )
                vector_sample_count += int(args_cli.num_envs)

                if args_cli.action_mode == "reference":
                    action = unwrapped.current_reconstructed_reference_action().clone()
                else:
                    action = torch.zeros(
                        (int(args_cli.num_envs), int(robot.num_joints)),
                        dtype=robot.data.joint_pos.torch.dtype,
                        device=device,
                    )
                action_abs_mean.append(float(action.abs().mean().item()))

                _, _, terminated, truncated, _ = env.step(action)
                reward_term_sum.add_(
                    unwrapped.reward_manager._step_reward.double().sum(dim=0)
                )
                done = torch.logical_or(terminated, truncated)
                episode_ages.add_(1)

                for name in termination_names:
                    term = unwrapped.termination_manager.get_term(name)
                    termination_counts[name] += int(term.count_nonzero().item())

                if bool(done.any()):
                    completed_episode_lengths.extend(
                        float(value)
                        for value in episode_ages[done].detach().cpu().tolist()
                    )
                    episode_ages[done] = 0

                applied_torque = robot.data.applied_torque.torch.abs()
                applied_torque_abs_mean.append(float(applied_torque.mean().item()))
                joint_torque_sum.add_(applied_torque.double().sum(dim=0))
                joint_torque_max = torch.maximum(
                    joint_torque_max, applied_torque.max(dim=0).values
                )
        torch.cuda.synchronize(device)
    finally:
        elapsed_s = time.perf_counter() - start
        env.close()

    total_env_steps = int(args_cli.num_envs) * int(args_cli.steps)
    total_terminations = sum(termination_counts.values())
    result = {
        "task": args_cli.task,
        "backend": backend,
        "physics_cfg": type(env_cfg.sim.physics).__name__,
        "match_sonic_release_overrides": bool(args_cli.match_sonic_release_overrides),
        "robot_spawn_cfg": type(robot_spawn_cfg).__name__,
        "robot_usd_path": robot_usd_path,
        "mass_distribution_params": mass_distribution_params,
        "action_mode": args_cli.action_mode,
        "training_randomization": randomization_profile == "all",
        "randomization_profile": randomization_profile,
        "full_trajectory_random_starts": bool(args_cli.full_trajectory_random_starts),
        "initial_reference_step_min": int(initial_local_steps.min().item()),
        "initial_reference_step_max": int(initial_local_steps.max().item()),
        "initial_reference_step_unique": int(initial_local_steps.unique().numel()),
        "initial_reference_step_nonzero_fraction": float(
            initial_local_steps.ne(0).float().mean().item()
        ),
        "seed": int(args_cli.seed),
        "num_envs": int(args_cli.num_envs),
        "steps": int(args_cli.steps),
        "total_env_steps": total_env_steps,
        "elapsed_s": elapsed_s,
        "env_steps_per_second": total_env_steps / elapsed_s,
        "completed_episodes": len(completed_episode_lengths),
        "completed_episode_length_mean": _mean(completed_episode_lengths),
        "completed_episode_length_max": _max(completed_episode_lengths),
        "unfinished_episode_age_mean": float(episode_ages.float().mean().item()),
        "unfinished_episode_age_max": int(episode_ages.max().item()),
        "termination_counts": dict(termination_counts),
        "termination_fractions": {
            name: count / total_terminations if total_terminations else 0.0
            for name, count in termination_counts.items()
        },
        "reward_term_mean_per_second": _named_means(
            reward_names,
            reward_term_sum.cpu().tolist(),
            total_env_steps,
        ),
        "joint_pos_mae_mean_rad": _mean(joint_pos_mae),
        "joint_pos_error_max_rad": _max(joint_pos_max),
        "ee_z_mae_mean_m": _mean(ee_z_mae),
        "joint_pos_mae_by_joint_rad": _named_means(
            robot_joint_names, joint_error_sum.cpu().tolist(), vector_sample_count
        ),
        "joint_pos_error_max_by_joint_rad": _named_values(
            robot_joint_names, joint_error_max.cpu().tolist()
        ),
        "ee_z_error_max_m": _max(ee_z_max),
        "ee_xyz_error_mean_m": _mean(ee_xyz_mae),
        "ee_z_mae_by_body_m": _named_means(
            EE_BODY_NAMES,
            ee_z_error_sum.cpu().tolist(),
            vector_sample_count,
        ),
        "ee_z_error_max_by_body_m": _named_values(
            EE_BODY_NAMES,
            ee_z_error_max.cpu().tolist(),
        ),
        "ee_xyz_error_mean_by_body_m": _named_means(
            EE_BODY_NAMES,
            ee_xyz_error_sum.cpu().tolist(),
            vector_sample_count,
        ),
        "ee_xyz_error_max_by_body_m": _named_values(
            EE_BODY_NAMES,
            ee_xyz_error_max.cpu().tolist(),
        ),
        "action_abs_mean": _mean(action_abs_mean),
        "applied_torque_abs_mean_nm": _mean(applied_torque_abs_mean),
        "robot_joint_names": robot_joint_names,
        "action_joint_names": action_joint_names,
        "robot_body_names": robot_body_names,
    }
    return result


def run(argv: list[str], *, require_running_kit: bool = False) -> int:
    strict_kitless = "--assert-kitless" in argv
    if strict_kitless:
        install_kit_import_guard()

    args_cli = _parse_args(argv)
    backend = requested_backend(argv)
    if args_cli.steps < 1 or args_cli.num_envs < 1:
        raise ValueError("--steps and --num_envs must both be positive.")

    import isaaclab_imitation.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils import has_kit
    from isaaclab_tasks.utils import launch_simulation, resolve_task_config

    env_cfg, _ = resolve_task_config(args_cli.task, "rlopt_ipmd_cfg_entry_point")
    if strict_kitless:
        if not config_contains_type_name(env_cfg, "NewtonCfg"):
            raise RuntimeError("--assert-kitless requires physics=newton_mjwarp.")
        assert_kit_not_loaded()
    if require_running_kit and not has_kit():
        raise RuntimeError("PhysX diagnosis requires a running Kit application.")

    with launch_simulation(env_cfg, args_cli):
        result = _run_probe(env_cfg, args_cli, backend)

    if strict_kitless:
        assert_kit_not_loaded()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output = args_cli.output or (DEFAULT_OUTPUT_ROOT / f"{backend}_{timestamp}.json")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DYNAMICS_DIAGNOSTIC " + json.dumps(result, sort_keys=True), flush=True)
    print(f"[INFO] Dynamics diagnostic written to {output}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    backend = requested_backend(argv)
    if backend == "newton":
        if "--assert-kitless" not in argv:
            argv.append("--assert-kitless")
        return run(argv)

    if backend != "physx":
        raise ValueError(f"Unsupported dynamics diagnostic backend: {backend!r}")
    validate_gpu_policy("physx", detect_gpu_names())

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args(argv)
    app_launcher = AppLauncher(launcher_args)
    try:
        return run(argv, require_running_kit=True)
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("G1 dynamics diagnosis failed.")
        raise
