#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate a learned command-interface planner in closed loop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v0")
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD",
    choices=[
        "PPO",
        "SAC",
        "FASTSAC",
        "IPMD",
        "IPMD_SR",
        "IPMD_BILINEAR",
        "GAIL",
        "AMP",
        "ASE",
    ],
)
parser.add_argument(
    "--checkpoint", type=Path, required=True, help="Low-level checkpoint."
)
parser.add_argument("--planner_checkpoint", type=Path, required=True)
parser.add_argument("--output_json", type=Path, default=None)
parser.add_argument("--output_csv", type=Path, default=None)
parser.add_argument("--append_csv", action="store_true", default=False)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Recorded video length in steps. <=0 uses --steps.",
)
parser.add_argument("--motion_manifest", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--state_history_steps", type=int, default=0)
parser.add_argument("--command_past_steps", type=int, default=0)
parser.add_argument("--command_future_steps", type=int, default=25)
parser.add_argument("--flow_num_inference_steps", type=int, default=16)
parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
parser.add_argument("--reset_schedule", type=str, default="sequential")
parser.add_argument("--reference_start_frame", type=int, default=0)
parser.add_argument("--refresh_zarr_dataset", action="store_true", default=False)
parser.add_argument("--keep_after_done", action="store_true", default=False)
parser.add_argument(
    "--disable_env_terminations",
    action="store_true",
    default=False,
    help="Disable env termination terms before env creation; outer --steps controls rollout length.",
)
parser.add_argument(
    "--enable_observation_corruption", action="store_true", default=False
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils import math as math_utils
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_EE_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import (  # noqa: E402
    flatten_command_terms,
    load_planner_checkpoint,
    planner_state_from_batch,
    rmse_per_row,
    unflatten_command_target,
)


ALGORITHM_CLASS_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "FASTSAC": FastSAC,
    "IPMD": IPMD,
    "IPMD_SR": IPMDSR,
    "IPMD_BILINEAR": IPMDBilinear,
    "GAIL": GAIL,
    "AMP": AMP,
    "ASE": ASE,
}

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


def resolve_agent_cfg_entry_point(task_name: str | None, algorithm: str) -> str:
    if task_name is None:
        return f"rlopt_{algorithm.lower()}_cfg_entry_point"
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    spec = gym.spec(task_id)
    if spec.kwargs.get(algo_entry_point) is not None:
        return algo_entry_point
    supported_algorithms = sorted(
        ENTRY_POINT_ALGORITHM_MAP[key]
        for key in ENTRY_POINT_ALGORITHM_MAP
        if spec.kwargs.get(key) is not None
    )
    raise ValueError(
        f"Task {task_id!r} does not expose {algorithm}; supported={supported_algorithms}."
    )


def _unwrap_imitation_env(env: object) -> ImitationRLEnv:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ImitationRLEnv):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if isinstance(unwrapped, ImitationRLEnv):
            return unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError("Could not unwrap an ImitationRLEnv.")


def _disable_observation_corruption(env_cfg: object) -> None:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return
    for group_name in (
        "policy",
        "critic",
        "expert_state",
        "expert_window",
        "reward_input",
    ):
        group = getattr(observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False


def _sync_env_window_params(env_cfg: object) -> None:
    for method_name in (
        "_sync_expert_window_observation_params",
        "_sync_expert_goal_observation_params",
    ):
        method = getattr(env_cfg, method_name, None)
        if callable(method):
            method()


def _disable_env_terminations(env_cfg: object) -> list[str]:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return []
    disabled: list[str] = []
    for name in dir(terminations):
        if name.startswith("_"):
            continue
        value = getattr(terminations, name, None)
        if value is None or callable(value):
            continue
        setattr(terminations, name, None)
        disabled.append(str(name))
    return disabled


def _configured_step_dt(env_cfg: object) -> float | None:
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 0.0) or 0.0)
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    if sim_dt > 0.0 and decimation > 0:
        return sim_dt * decimation
    return None


def _get_optional(
    td: TensorDictBase, key: str | tuple[str, ...]
) -> torch.Tensor | None:
    try:
        value = td.get(key)
    except KeyError:
        return None
    return value if isinstance(value, torch.Tensor) else None


def _optional_flat_tensor(
    td: TensorDictBase,
    key: str | tuple[str, ...],
    *,
    num_envs: int,
    default: float | bool,
) -> torch.Tensor:
    value = _get_optional(td, key)
    if value is None:
        return torch.full((num_envs,), default)
    flat = value.detach().reshape(-1).cpu()
    if flat.numel() == 1 and num_envs > 1:
        flat = flat.expand(num_envs)
    if flat.numel() < num_envs:
        raise RuntimeError(
            f"Expected at least {num_envs} values for {key}, got {flat.numel()}."
        )
    return flat[:num_envs]


def _termination_term_masks(
    base_env: ImitationRLEnv, num_envs: int
) -> dict[str, torch.Tensor]:
    manager = getattr(base_env, "termination_manager", None)
    active_terms = tuple(getattr(manager, "active_terms", ()) or ())
    masks: dict[str, torch.Tensor] = {}
    for term_name in active_terms:
        try:
            value = manager.get_term(term_name)
        except Exception:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        flat = value.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
        if flat.numel() == 1 and num_envs > 1:
            flat = flat.expand(num_envs)
        if flat.numel() < num_envs:
            continue
        masks[str(term_name)] = flat[:num_envs]
    return masks


def _resolve_existing_body_names(
    base_env: ImitationRLEnv, requested_names: list[str]
) -> list[str]:
    names: list[str] = []
    for name in requested_names:
        try:
            base_env._get_robot_anchor_body_id_fast(name)
            base_env._get_reference_body_ids_fast((name,))
        except Exception as exc:
            print(f"[WARNING] Skipping unavailable body metric target {name!r}: {exc}")
            continue
        names.append(str(name))
    return names


def _mean_body_pose_errors(
    base_env: ImitationRLEnv,
    names: list[str],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if len(names) == 0:
        return None
    body_ids = [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]
    actual_pos, actual_quat = base_env._get_robot_body_pose_w_fast(body_ids)
    ref_pos, ref_quat = base_env._get_reference_body_pose_w_fast(tuple(names))
    pos_error = torch.linalg.vector_norm(actual_pos - ref_pos, dim=-1).mean(dim=-1)
    ori_error = math_utils.quat_error_magnitude(
        actual_quat.reshape(-1, 4),
        ref_quat.reshape(-1, 4),
    ).reshape(actual_quat.shape[0], -1)
    return pos_error, ori_error.mean(dim=-1)


def _tracking_metrics(
    base_env: ImitationRLEnv,
    *,
    tracked_body_names: list[str],
    ee_body_names: list[str],
) -> dict[str, torch.Tensor]:
    robot_data = base_env.robot.data
    root_pos_ref, root_quat_ref, root_lin_vel_ref, root_ang_vel_ref = (
        base_env._get_reference_root_state_w_fast()
    )
    joint_pos_ref = base_env.current_expert_frame["joint_pos"]
    joint_vel_ref = base_env.current_expert_frame["joint_vel"]
    root_pos_error = robot_data.root_pos_w - root_pos_ref
    metrics = {
        "root_pos_xy_error_m": torch.linalg.vector_norm(root_pos_error[:, :2], dim=-1),
        "root_ori_error_rad": math_utils.quat_error_magnitude(
            robot_data.root_quat_w, root_quat_ref
        ),
        "joint_pos_rmse_rad": torch.sqrt(
            torch.mean((robot_data.joint_pos - joint_pos_ref).square(), dim=-1)
        ),
        "joint_vel_rmse_radps": torch.sqrt(
            torch.mean((robot_data.joint_vel - joint_vel_ref).square(), dim=-1)
        ),
        "root_lin_vel_rmse_mps": torch.sqrt(
            torch.mean((robot_data.root_lin_vel_w - root_lin_vel_ref).square(), dim=-1)
        ),
        "root_ang_vel_rmse_radps": torch.sqrt(
            torch.mean((robot_data.root_ang_vel_w - root_ang_vel_ref).square(), dim=-1)
        ),
    }
    tracked_errors = _mean_body_pose_errors(base_env, tracked_body_names)
    if tracked_errors is not None:
        metrics["tracked_body_pos_error_m"] = tracked_errors[0]
        metrics["tracked_body_ori_error_rad"] = tracked_errors[1]
    ee_errors = _mean_body_pose_errors(base_env, ee_body_names)
    if ee_errors is not None:
        metrics["ee_pos_error_m"] = ee_errors[0]
        metrics["ee_ori_error_rad"] = ee_errors[1]
    return metrics


def _refresh_tensordict_observations(
    td: TensorDictBase, base_env: ImitationRLEnv
) -> TensorDictBase:
    observations = base_env.observation_manager.compute(update_history=False)
    for group_name, group_obs in observations.items():
        if isinstance(group_obs, dict):
            group_td = td.get(group_name)
            if not isinstance(group_td, TensorDictBase):
                group_td = TensorDict(
                    {}, batch_size=[base_env.num_envs], device=base_env.device
                )
                td.set(group_name, group_td)
            for term_name, value in group_obs.items():
                td.set((group_name, term_name), value)
            continue
        td.set(group_name, group_obs)
    return td


def _command_reference_kwargs(
    interface: str, *, ee_body_names: list[str]
) -> dict[str, object]:
    if interface == "ee_trajectory":
        return {"reference_body_names": tuple(ee_body_names)}
    return {}


def _current_reference_command_terms(
    base_env: ImitationRLEnv,
    *,
    interface: str,
    ee_body_names: list[str],
) -> dict[str, torch.Tensor]:
    ref_kwargs = _command_reference_kwargs(interface, ee_body_names=ee_body_names)
    term_names = (
        ("expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b")
        if interface == "full_body_trajectory"
        else ("expert_ee_pos_b", "expert_ee_ori_b")
    )
    return {
        term_name: base_env.get_current_expert_window_term(
            term_name=term_name,
            past_steps=int(args_cli.command_past_steps),
            future_steps=int(args_cli.command_future_steps),
            **ref_kwargs,
        )
        for term_name in term_names
    }


def _accumulate_metric(
    stats: dict[str, list[torch.Tensor]],
    metric_name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    selected = values.detach().cpu()[mask.cpu()]
    if selected.numel() == 0:
        return
    stats.setdefault(metric_name, []).append(selected.float())


def _finalize_metric_stats(
    stats: dict[str, list[torch.Tensor]],
) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for name, chunks in sorted(stats.items()):
        values = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        finalized[name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item())
            if values.numel() > 1
            else 0.0,
            "count": int(values.numel()),
        }
    return finalized


def _tensor_mean_std(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    selected = values[mask]
    if selected.numel() == 0:
        return float("nan"), float("nan")
    return (
        float(selected.mean().item()),
        float(selected.std(unbiased=False).item()) if selected.numel() > 1 else 0.0,
    )


def _write_csv(summary: dict[str, Any], output_csv: Path, *, append: bool) -> None:
    row: dict[str, Any] = {}
    row.update(summary["metadata"])
    row.update(summary["aggregate"])
    for metric_name, metric_values in summary["metrics"].items():
        for stat_name, value in metric_values.items():
            row[f"{metric_name}_{stat_name}"] = value
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and output_csv.is_file() else "w"
    with output_csv.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg
) -> None:
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    planner_checkpoint = args_cli.planner_checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Low-level checkpoint not found: {checkpoint_path}")
    if not planner_checkpoint.is_file():
        raise FileNotFoundError(f"Planner checkpoint not found: {planner_checkpoint}")

    planner, target_spec, planner_metadata = load_planner_checkpoint(
        planner_checkpoint,
        map_location=args_cli.device or "cpu",
    )
    planner = planner.to(args_cli.device or "cpu")
    planner.eval()
    interface = target_spec.interface

    agent_cfg.command_space = interface
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()
    env_cfg.latent_patch_past_steps = int(args_cli.command_past_steps)
    env_cfg.latent_patch_future_steps = int(args_cli.command_future_steps)
    env_cfg.command_observation_source = "planner"
    _sync_env_window_params(env_cfg)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = args_cli.seed if args_cli.seed != -1 else random.randint(0, 10000)
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    motion_manifest = (
        args_cli.motion_manifest.expanduser().resolve()
        if args_cli.motion_manifest is not None
        else None
    )
    if motion_manifest is not None:
        env_cfg.lafan1_manifest_path = str(motion_manifest)
        resolve_manifest_config = getattr(env_cfg, "_resolve_manifest_config", None)
        if callable(resolve_manifest_config):
            resolve_manifest_config()
    if hasattr(env_cfg, "refresh_zarr_dataset"):
        env_cfg.refresh_zarr_dataset = bool(args_cli.refresh_zarr_dataset)
    disabled_termination_terms: list[str] = []
    if args_cli.disable_env_terminations:
        disabled_termination_terms = _disable_env_terminations(env_cfg)
    if hasattr(env_cfg, "reference_start_frame"):
        env_cfg.reference_start_frame = int(args_cli.reference_start_frame)
    if hasattr(env_cfg, "random_reset_full_trajectory"):
        env_cfg.random_reset_full_trajectory = False
    if hasattr(env_cfg, "reset_schedule"):
        env_cfg.reset_schedule = str(args_cli.reset_schedule)
    if hasattr(env_cfg, "wrap_steps"):
        env_cfg.wrap_steps = False
    if not args_cli.enable_observation_corruption:
        _disable_observation_corruption(env_cfg)
    step_dt = _configured_step_dt(env_cfg)
    if step_dt is not None and hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s), float(args_cli.steps + 2) * step_dt
        )

    output_root = (
        args_cli.output_json.expanduser().resolve().parent
        if args_cli.output_json is not None
        else planner_checkpoint.parent / "closed_loop_eval"
    )
    env_cfg.log_dir = str(output_root)
    agent_cfg.env.num_envs = int(args_cli.num_envs)
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = env_cfg.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    if hasattr(agent_cfg, "logger"):
        agent_cfg.logger.backend = ""
        agent_cfg.logger.log_dir = str(output_root / "agent_logs")
    if hasattr(agent_cfg, "device"):
        agent_cfg.device = env_cfg.sim.device

    render_mode = "rgb_array" if args_cli.video else None
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    video_folder: Path | None = None
    video_length = 0
    if args_cli.video:
        video_folder = output_root / "videos" / "play"
        video_length = max(
            1,
            int(args_cli.video_length)
            if int(args_cli.video_length) > 0
            else int(args_cli.steps),
        )
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording interface planner video.")
        print(f"[INFO] Video options: {video_kwargs}")
        raw_env = gym.wrappers.RecordVideo(raw_env, **video_kwargs)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")
    env = IsaacLabWrapper(raw_env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env, transform=Compose(RewardSum(), StepCounter(args_cli.steps + 2))
    )
    base_env = _unwrap_imitation_env(env)
    tracked_body_names = _resolve_existing_body_names(
        base_env, list(G1_TRACKED_BODY_NAMES)
    )
    ee_body_names = _resolve_existing_body_names(
        base_env,
        list(getattr(env_cfg, "command_ee_body_names", G1_EE_BODY_NAMES)),
    )

    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    print(f"[INFO] Loading low-level checkpoint: {checkpoint_path}")
    agent.load_model(str(checkpoint_path))
    policy = agent.collector_policy
    policy.eval()

    num_envs = int(args_cli.num_envs)
    active = torch.ones(num_envs, dtype=torch.bool)
    survival_steps = torch.zeros(num_envs, dtype=torch.float32)
    return_sum = torch.zeros(num_envs, dtype=torch.float32)
    done_events = torch.zeros(num_envs, dtype=torch.float32)
    terminated_events = torch.zeros(num_envs, dtype=torch.float32)
    truncated_events = torch.zeros(num_envs, dtype=torch.float32)
    done_first_step = torch.full((num_envs,), -1, dtype=torch.int64)
    termination_counts: dict[str, int] = {}
    termination_first_step: dict[str, int] = {}
    metric_stats: dict[str, list[torch.Tensor]] = {}
    previous_action: torch.Tensor | None = None
    valid_transition_count = 0
    planner_publish_count = 0

    td = env.reset()
    for step_idx in range(int(args_cli.steps)):
        step_active = active.clone()
        if not bool(step_active.any()):
            break
        achieved_batch = base_env.current_achieved_macro_transition_batch(
            horizon_steps=int(args_cli.command_future_steps),
            state_history_steps=int(args_cli.state_history_steps),
        )
        planner_state = planner_state_from_batch(
            achieved_batch,
            state_history_steps=int(args_cli.state_history_steps),
        ).to(device=next(planner.parameters()).device, dtype=torch.float32)
        with torch.inference_mode():
            predicted_target = planner(
                planner_state,
                num_inference_steps=int(args_cli.flow_num_inference_steps),
                inference_noise_std=float(args_cli.flow_inference_noise_std),
            )
        command_terms = unflatten_command_target(
            predicted_target.to(device=base_env.device),
            target_spec,
        )
        base_env.set_agent_trajectory_command(command_terms)
        planner_publish_count += 1
        td = _refresh_tensordict_observations(td, base_env)

        reference_target, _ = flatten_command_terms(
            interface,
            _current_reference_command_terms(
                base_env,
                interface=interface,
                ee_body_names=ee_body_names,
            ),
        )
        target_rmse = rmse_per_row(
            predicted_target.to(reference_target.device), reference_target
        )
        _accumulate_metric(
            metric_stats, "planner_target_rmse", target_rmse.cpu(), step_active
        )

        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = policy(td)
        action = td.get("action")
        if action is None:
            raise RuntimeError("Policy did not write an action tensor.")
        action_2d = action.detach().reshape(num_envs, -1)
        _accumulate_metric(
            metric_stats,
            "action_l2",
            torch.linalg.vector_norm(action_2d, dim=-1).cpu(),
            step_active,
        )
        if previous_action is not None:
            action_delta_l2 = torch.linalg.vector_norm(
                action_2d.cpu() - previous_action, dim=-1
            )
            _accumulate_metric(
                metric_stats, "action_delta_l2", action_delta_l2, step_active
            )
        previous_action = action_2d.cpu()

        with torch.inference_mode():
            td_step = env.step(td)
        rewards = _optional_flat_tensor(
            td_step, ("next", "reward"), num_envs=num_envs, default=0.0
        )
        dones = _optional_flat_tensor(
            td_step, ("next", "done"), num_envs=num_envs, default=False
        ).bool()
        terminateds = _optional_flat_tensor(
            td_step, ("next", "terminated"), num_envs=num_envs, default=False
        ).bool()
        truncateds = _optional_flat_tensor(
            td_step, ("next", "truncated"), num_envs=num_envs, default=False
        ).bool()
        done_any = dones | terminateds | truncateds
        return_sum += rewards.float() * step_active.float()
        survival_steps += step_active.float()
        newly_done = done_any & step_active & (done_events <= 0)
        done_events += (done_any & step_active).float()
        terminated_events += (terminateds & step_active).float()
        truncated_events += (truncateds & step_active).float()
        if bool(newly_done.any()):
            done_first_step[newly_done] = int(step_idx + 1)
            for term_name, term_mask in _termination_term_masks(
                base_env, num_envs
            ).items():
                hit_count = int((term_mask & newly_done.cpu()).sum().item())
                if hit_count <= 0:
                    continue
                termination_counts[term_name] = (
                    termination_counts.get(term_name, 0) + hit_count
                )
                termination_first_step.setdefault(term_name, int(step_idx + 1))

        metric_mask = (
            step_active if args_cli.keep_after_done else step_active & ~done_any
        )
        valid_transition_count += int(metric_mask.sum().item())
        for metric_name, values in _tracking_metrics(
            base_env,
            tracked_body_names=tracked_body_names,
            ee_body_names=ee_body_names,
        ).items():
            _accumulate_metric(metric_stats, metric_name, values.cpu(), metric_mask)
        if not args_cli.keep_after_done:
            active &= ~done_any
        td = step_mdp(
            td_step, exclude_reward=True, exclude_done=False, exclude_action=True
        )

    active_mask = survival_steps > 0
    return_mean, return_std = _tensor_mean_std(return_sum, active_mask)
    survival_mean, survival_std = _tensor_mean_std(survival_steps, active_mask)
    aggregate = {
        "return_sum_mean": return_mean,
        "return_sum_std": return_std,
        "survival_steps_mean": survival_mean,
        "survival_steps_std": survival_std,
        "done_rate": float((done_events[active_mask] > 0).float().mean().item())
        if bool(active_mask.any())
        else float("nan"),
        "terminated_rate": float(
            (terminated_events[active_mask] > 0).float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "truncated_rate": float(
            (truncated_events[active_mask] > 0).float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "valid_transition_count": int(valid_transition_count),
        "planner_publish_count": int(planner_publish_count),
    }
    summary = {
        "metadata": {
            "label": args_cli.label,
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "checkpoint": str(checkpoint_path),
            "planner_checkpoint": str(planner_checkpoint),
            "interface": interface,
            "state_history_steps": int(args_cli.state_history_steps),
            "command_past_steps": int(args_cli.command_past_steps),
            "command_future_steps": int(args_cli.command_future_steps),
            "flow_num_inference_steps": int(args_cli.flow_num_inference_steps),
            "flow_inference_noise_std": float(args_cli.flow_inference_noise_std),
            "video_folder": str(video_folder) if video_folder is not None else None,
            "video_length": int(video_length) if args_cli.video else None,
            "disabled_termination_terms": disabled_termination_terms,
            "planner_target_dim": int(target_spec.target_dim),
            "planner_metadata": planner_metadata,
            "num_envs": int(num_envs),
            "seed": int(env_cfg.seed),
            "motion_manifest": str(motion_manifest)
            if motion_manifest is not None
            else None,
        },
        "aggregate": aggregate,
        "termination": {
            "counts": termination_counts,
            "first_step": termination_first_step,
            "done_first_step": [
                int(value)
                for value in done_first_step[active_mask].detach().cpu().tolist()
            ],
        },
        "metrics": _finalize_metric_stats(metric_stats),
    }
    output_json = args_cli.output_json
    if output_json is None:
        output_json = (
            planner_checkpoint.parent / "closed_loop_eval" / f"{interface}_eval.json"
        )
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if args_cli.output_csv is not None:
        _write_csv(summary, args_cli.output_csv, append=bool(args_cli.append_csv))
    print(
        "[RESULT] "
        f"interface={interface} return={aggregate['return_sum_mean']:.4f} "
        f"survival={aggregate['survival_steps_mean']:.1f} "
        f"done_rate={aggregate['done_rate']:.3f} "
        f"planner_rmse={summary['metrics'].get('planner_target_rmse', {}).get('mean', float('nan')):.4f}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
