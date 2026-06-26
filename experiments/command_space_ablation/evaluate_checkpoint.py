#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate a deterministic RLOpt checkpoint against the G1 reference motion."""

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
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_json", type=Path, default=None)
parser.add_argument("--output_csv", type=Path, default=None)
parser.add_argument(
    "--append_csv",
    action="store_true",
    default=False,
    help="Append to --output_csv instead of replacing it.",
)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--command_space", type=str, default=None)
parser.add_argument("--command_past_steps", type=int, default=None)
parser.add_argument("--command_future_steps", type=int, default=None)
parser.add_argument(
    "--command_observation_source",
    choices=["reference", "planner_oracle", "planner"],
    default=None,
)
parser.add_argument(
    "--planner_mode",
    choices=["none", "reference", "hold_current", "noisy_reference", "zero"],
    default="none",
    help=(
        "External planner publisher used with command_observation_source=planner. "
        "'reference' publishes the exact oracle command through the planner API; "
        "'hold_current' repeats the current command frame over the horizon; "
        "'noisy_reference' adds Gaussian noise to the oracle command; "
        "'zero' publishes zero commands."
    ),
)
parser.add_argument(
    "--planner_update_interval",
    type=int,
    default=1,
    help="Publish a new planner command every N policy steps; larger values hold the previous command.",
)
parser.add_argument(
    "--planner_noise_std",
    type=float,
    default=0.0,
    help="Gaussian noise std for --planner_mode noisy_reference.",
)
parser.add_argument(
    "--motion_manifest",
    type=Path,
    default=None,
    help="Optional manifest used to condition evaluation on a motion set.",
)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--reset_schedule", type=str, default="sequential")
parser.add_argument("--reference_start_frame", type=int, default=0)
parser.add_argument(
    "--debug_compare_command_sources",
    action="store_true",
    default=False,
    help="Compare reference, planner_oracle, and planner observation tensors, then exit.",
)
parser.add_argument("--refresh_zarr_dataset", action="store_true", default=False)
parser.add_argument(
    "--keep_after_done",
    action="store_true",
    default=False,
    help="Keep collecting after envs report done/truncated; otherwise ignore later steps.",
)
parser.add_argument(
    "--enable_observation_corruption",
    action="store_true",
    default=False,
    help="Leave policy observation corruption enabled during evaluation.",
)
parser.add_argument(
    "--preserve_episode_length",
    action="store_true",
    default=False,
    help="Do not extend env.episode_length_s to cover --steps.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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
    """Resolve the agent config entry point based on algorithm and task registry."""
    if task_name is None:
        return f"rlopt_{algorithm.lower()}_cfg_entry_point"
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    try:
        spec = gym.spec(task_id)
    except Exception as exc:
        msg = f"Could not resolve task '{task_id}' from registry."
        raise ValueError(msg) from exc

    if spec.kwargs.get(algo_entry_point) is not None:
        print(f"[INFO] Using agent config entry point: {algo_entry_point}")
        return algo_entry_point

    supported_algorithms = sorted(
        ENTRY_POINT_ALGORITHM_MAP[key]
        for key in ENTRY_POINT_ALGORITHM_MAP
        if spec.kwargs.get(key) is not None
    )
    msg = (
        "Unsupported task/algo combination: "
        f"task '{task_id}' does not expose an RLOpt config for '{algorithm}'. "
        f"Supported RLOpt algorithms for this task: {supported_algorithms}."
    )
    raise ValueError(msg)


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
            f"Expected at least {num_envs} values for tensordict key {key}, got {flat.numel()}."
        )
    return flat[:num_envs]


def _resolve_existing_body_names(
    base_env: ImitationRLEnv,
    requested_names: list[str] | tuple[str, ...],
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


def _body_ids_for_names(base_env: ImitationRLEnv, names: list[str]) -> list[int]:
    return [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]


def _mean_body_pose_errors(
    base_env: ImitationRLEnv,
    names: list[str],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if len(names) == 0:
        return None
    body_ids = _body_ids_for_names(base_env, names)
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
    root_lin_vel_error = robot_data.root_lin_vel_w - root_lin_vel_ref
    root_ang_vel_error = robot_data.root_ang_vel_w - root_ang_vel_ref

    metrics = {
        "root_pos_xy_error_m": torch.linalg.vector_norm(root_pos_error[:, :2], dim=-1),
        "root_pos_z_abs_error_m": root_pos_error[:, 2].abs(),
        "root_ori_error_rad": math_utils.quat_error_magnitude(
            robot_data.root_quat_w,
            root_quat_ref,
        ),
        "root_lin_vel_rmse_mps": torch.sqrt(
            torch.mean(root_lin_vel_error.square(), dim=-1)
        ),
        "root_ang_vel_rmse_radps": torch.sqrt(
            torch.mean(root_ang_vel_error.square(), dim=-1)
        ),
        "joint_pos_rmse_rad": torch.sqrt(
            torch.mean((robot_data.joint_pos - joint_pos_ref).square(), dim=-1)
        ),
        "joint_vel_rmse_radps": torch.sqrt(
            torch.mean((robot_data.joint_vel - joint_vel_ref).square(), dim=-1)
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


def _trajectory_command_terms(command_space: str) -> tuple[str, ...]:
    if command_space == "full_body_trajectory":
        return ("expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b")
    if command_space == "ee_trajectory":
        return ("expert_ee_pos_b", "expert_ee_ori_b")
    return ()


def _command_reference_kwargs(
    command_space: str,
    *,
    ee_body_names: list[str],
) -> dict[str, object]:
    if command_space == "ee_trajectory":
        return {"reference_body_names": tuple(ee_body_names)}
    return {}


def _refresh_tensordict_observations(
    td: TensorDictBase,
    base_env: ImitationRLEnv,
) -> TensorDictBase:
    observations = base_env.observation_manager.compute(update_history=False)
    for group_name, group_obs in observations.items():
        if isinstance(group_obs, dict):
            group_td = td.get(group_name)
            if not isinstance(group_td, TensorDictBase):
                group_td = TensorDict(
                    {},
                    batch_size=[base_env.num_envs],
                    device=base_env.device,
                )
                td.set(group_name, group_td)
            for term_name, value in group_obs.items():
                td.set((group_name, term_name), value)
            continue
        td.set(group_name, group_obs)
    return td


def _planner_command_terms(command_space: str) -> tuple[str, ...]:
    return _trajectory_command_terms(command_space)


def _current_reference_command_terms(
    base_env: ImitationRLEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
) -> dict[str, torch.Tensor]:
    past_steps = int(getattr(base_env, "_latent_patch_past_steps", 0))
    future_steps = int(getattr(base_env, "_latent_patch_future_steps", 0))
    ref_kwargs = _command_reference_kwargs(command_space, ee_body_names=ee_body_names)
    return {
        term_name: base_env.get_current_expert_window_term(
            term_name=term_name,
            past_steps=past_steps,
            future_steps=future_steps,
            **ref_kwargs,
        )
        for term_name in _planner_command_terms(command_space)
    }


def _hold_current_command_window(
    command_terms: dict[str, torch.Tensor],
    *,
    past_steps: int,
    future_steps: int,
) -> dict[str, torch.Tensor]:
    window_steps = int(past_steps) + int(future_steps) + 1
    if window_steps <= 0:
        raise ValueError("Planner command window must contain at least one step.")
    held_terms: dict[str, torch.Tensor] = {}
    center_index = int(past_steps)
    for term_name, value in command_terms.items():
        if int(value.shape[1]) % window_steps != 0:
            raise ValueError(
                f"Planner command term {term_name!r} width {value.shape[1]} is not divisible "
                f"by window_steps={window_steps}."
            )
        per_step_width = int(value.shape[1]) // window_steps
        sequence = value.reshape(value.shape[0], window_steps, per_step_width)
        held_terms[term_name] = (
            sequence[:, center_index : center_index + 1, :]
            .expand(-1, window_steps, -1)
            .reshape(value.shape[0], -1)
            .contiguous()
        )
    return held_terms


def _build_planner_command_terms(
    base_env: ImitationRLEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
    planner_mode: str,
    planner_noise_std: float,
) -> dict[str, torch.Tensor]:
    terms = _planner_command_terms(command_space)
    if len(terms) == 0:
        return {}
    if planner_mode == "zero":
        return {
            term_name: torch.zeros_like(
                base_env.get_agent_trajectory_command_term(term_name)
            )
            for term_name in terms
        }

    command_terms = _current_reference_command_terms(
        base_env,
        command_space=command_space,
        ee_body_names=ee_body_names,
    )
    if planner_mode == "hold_current":
        command_terms = _hold_current_command_window(
            command_terms,
            past_steps=int(getattr(base_env, "_latent_patch_past_steps", 0)),
            future_steps=int(getattr(base_env, "_latent_patch_future_steps", 0)),
        )
    elif planner_mode == "noisy_reference":
        noise_std = float(planner_noise_std)
        if noise_std > 0.0:
            command_terms = {
                term_name: value + torch.randn_like(value) * noise_std
                for term_name, value in command_terms.items()
            }
    elif planner_mode != "reference":
        raise ValueError(f"Unsupported planner_mode={planner_mode!r}.")
    return command_terms


def _maybe_publish_planner_command(
    base_env: ImitationRLEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
    planner_mode: str,
    planner_update_interval: int,
    planner_noise_std: float,
    step_index: int,
) -> bool:
    if planner_mode == "none":
        return False
    update_interval = max(1, int(planner_update_interval))
    if int(step_index) % update_interval != 0:
        return False
    command_terms = _build_planner_command_terms(
        base_env,
        command_space=command_space,
        ee_body_names=ee_body_names,
        planner_mode=planner_mode,
        planner_noise_std=planner_noise_std,
    )
    if len(command_terms) == 0:
        return False
    base_env.set_agent_trajectory_command(command_terms)
    return True


def _command_metrics(
    base_env: ImitationRLEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
) -> dict[str, torch.Tensor]:
    if getattr(base_env, "_command_observation_source", "reference") == "reference":
        return {}

    terms = _trajectory_command_terms(command_space)
    if len(terms) == 0:
        return {}

    past_steps = int(getattr(base_env, "_latent_patch_past_steps", 0))
    future_steps = int(getattr(base_env, "_latent_patch_future_steps", 0))
    ref_kwargs = _command_reference_kwargs(command_space, ee_body_names=ee_body_names)
    metrics: dict[str, torch.Tensor] = {}
    invalid_mask = torch.zeros(
        base_env.num_envs, device=base_env.device, dtype=torch.bool
    )
    for term_name in terms:
        command = base_env.get_agent_trajectory_command_term(term_name)
        reference = base_env.get_current_expert_window_term(
            term_name=term_name,
            past_steps=past_steps,
            future_steps=future_steps,
            **ref_kwargs,
        )
        invalid_mask |= ~torch.isfinite(command).all(dim=-1)
        metrics[f"command_{term_name}_rmse"] = torch.sqrt(
            torch.mean((command - reference).square(), dim=-1)
        )
    metrics["command_invalid"] = invalid_mask.to(dtype=torch.float32)
    return metrics


def _clone_observation_terms(
    observations: object,
    *,
    group_name: str,
    term_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    group = observations[group_name]  # type: ignore[index]
    return {term_name: group[term_name].detach().clone() for term_name in term_names}


def _debug_compare_command_sources(
    env: TransformedEnv,
    base_env: ImitationRLEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
) -> None:
    term_names = _planner_command_terms(command_space)
    if len(term_names) == 0:
        raise ValueError(
            "--debug_compare_command_sources requires a trajectory command space."
        )

    env.reset()
    original_source = getattr(base_env, "_command_observation_source", "reference")
    try:
        base_env._command_observation_source = "reference"
        reference_obs = _clone_observation_terms(
            base_env.observation_manager.compute(update_history=False),
            group_name="expert_window",
            term_names=term_names,
        )

        base_env.reset_agent_trajectory_command()
        base_env._command_observation_source = "planner_oracle"
        planner_oracle_obs = _clone_observation_terms(
            base_env.observation_manager.compute(update_history=False),
            group_name="expert_window",
            term_names=term_names,
        )

        base_env.reset_agent_trajectory_command()
        command_terms = _current_reference_command_terms(
            base_env,
            command_space=command_space,
            ee_body_names=ee_body_names,
        )
        print("[DEBUG] source=direct_reference_terms")
        for term_name in term_names:
            reference = reference_obs[term_name]
            value = command_terms[term_name].detach()
            diff = value - reference
            rmse = torch.sqrt(torch.mean(diff.square())).item()
            max_abs = torch.max(torch.abs(diff)).item()
            print(
                "[DEBUG] "
                f"term={term_name} shape={tuple(value.shape)} "
                f"rmse={rmse:.8f} max_abs={max_abs:.8f} "
                f"reference_mean={reference.mean().item():.8f} "
                f"value_mean={value.mean().item():.8f}"
            )
        base_env.set_agent_trajectory_command(command_terms)
        planner_buffer = {
            term_name: base_env.get_agent_trajectory_command_term(term_name)
            .detach()
            .clone()
            for term_name in term_names
        }
        base_env._command_observation_source = "planner"
        planner_obs = _clone_observation_terms(
            base_env.observation_manager.compute(update_history=False),
            group_name="expert_window",
            term_names=term_names,
        )

        for source_name, source_obs in (
            ("planner_oracle", planner_oracle_obs),
            ("planner", planner_obs),
        ):
            print(f"[DEBUG] source={source_name}")
            for term_name in term_names:
                reference = reference_obs[term_name]
                value = source_obs[term_name]
                diff = value - reference
                rmse = torch.sqrt(torch.mean(diff.square())).item()
                max_abs = torch.max(torch.abs(diff)).item()
                print(
                    "[DEBUG] "
                    f"term={term_name} shape={tuple(value.shape)} "
                    f"rmse={rmse:.8f} max_abs={max_abs:.8f} "
                    f"reference_mean={reference.mean().item():.8f} "
                    f"value_mean={value.mean().item():.8f}"
                )
        print("[DEBUG] source=planner_buffer")
        for term_name in term_names:
            reference = reference_obs[term_name]
            value = planner_buffer[term_name]
            diff = value - reference
            rmse = torch.sqrt(torch.mean(diff.square())).item()
            max_abs = torch.max(torch.abs(diff)).item()
            print(
                "[DEBUG] "
                f"term={term_name} shape={tuple(value.shape)} "
                f"rmse={rmse:.8f} max_abs={max_abs:.8f} "
                f"reference_mean={reference.mean().item():.8f} "
                f"value_mean={value.mean().item():.8f}"
            )
    finally:
        base_env._command_observation_source = original_source


def _accumulate_metric(
    stats: dict[str, dict[str, float]],
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    values = values.detach()
    if values.ndim != 1:
        values = values.reshape(values.shape[0], -1).mean(dim=-1)
    selected = values[mask]
    if selected.numel() == 0:
        return
    selected_cpu = selected.float().cpu()
    item = stats.setdefault(name, {"sum": 0.0, "sumsq": 0.0, "count": 0.0})
    item["sum"] += float(selected_cpu.sum().item())
    item["sumsq"] += float(selected_cpu.square().sum().item())
    item["count"] += float(selected_cpu.numel())


def _finalize_metric_stats(
    stats: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    finalized: dict[str, dict[str, float | int]] = {}
    for name, item in sorted(stats.items()):
        count = int(item["count"])
        if count <= 0:
            continue
        mean = item["sum"] / count
        variance = max(0.0, item["sumsq"] / count - mean * mean)
        finalized[name] = {
            "mean": mean,
            "std": variance**0.5,
            "count": count,
        }
    return finalized


def _tensor_mean_std(
    values: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[float, float]:
    values = values.detach().float().cpu()
    if mask is not None:
        values = values[mask.detach().cpu()]
    if values.numel() == 0:
        return float("nan"), float("nan")
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
    return mean, std


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _flatten_summary(summary: dict[str, Any]) -> dict[str, object]:
    metadata = summary["metadata"]
    aggregate = summary["aggregate"]
    metrics = summary["metrics"]
    row: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    for key, value in aggregate.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    for metric_name, metric_stats in metrics.items():
        row[f"{metric_name}_mean"] = metric_stats["mean"]
        row[f"{metric_name}_std"] = metric_stats["std"]
        row[f"{metric_name}_count"] = metric_stats["count"]
    return row


def _write_csv(summary: dict[str, Any], output_csv: Path, *, append: bool) -> None:
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row = _flatten_summary(summary)
    fieldnames = sorted(row)

    file_exists = output_csv.is_file()
    mode = "a" if append and file_exists else "w"
    if mode == "a":
        with output_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            existing_header = next(reader, None)
        if existing_header != fieldnames:
            raise ValueError(
                f"CSV header mismatch for append: {output_csv}. "
                "Use a new --output_csv or delete the old file."
            )

    with output_csv.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerow(row)
    print(f"[INFO] Wrote CSV row: {output_csv}")


def _sync_env_window_params(env_cfg: object) -> None:
    for method_name in (
        "_sync_expert_window_observation_params",
        "_sync_expert_goal_observation_params",
    ):
        sync_method = getattr(env_cfg, method_name, None)
        if callable(sync_method):
            sync_method()


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    motion_manifest = (
        args_cli.motion_manifest.expanduser().resolve()
        if args_cli.motion_manifest is not None
        else None
    )
    if motion_manifest is not None and not motion_manifest.is_file():
        raise FileNotFoundError(f"Motion manifest not found: {motion_manifest}")

    if args_cli.command_space is not None:
        if not hasattr(agent_cfg, "command_space"):
            raise TypeError(
                f"Agent config for {args_cli.algorithm} has no command_space field."
            )
        agent_cfg.command_space = args_cli.command_space
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.command_past_steps is not None:
        env_cfg.latent_patch_past_steps = int(args_cli.command_past_steps)
    if args_cli.command_future_steps is not None:
        env_cfg.latent_patch_future_steps = int(args_cli.command_future_steps)
    if args_cli.command_observation_source is not None:
        env_cfg.command_observation_source = args_cli.command_observation_source
    elif args_cli.planner_mode != "none":
        env_cfg.command_observation_source = "planner"
    _sync_env_window_params(env_cfg)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = (
        args_cli.seed if args_cli.seed is not None else getattr(agent_cfg, "seed", None)
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    if motion_manifest is not None:
        if not hasattr(env_cfg, "lafan1_manifest_path"):
            raise TypeError(f"Task {args_cli.task} does not support --motion_manifest.")
        env_cfg.lafan1_manifest_path = str(motion_manifest)
        resolve_manifest_config = getattr(env_cfg, "_resolve_manifest_config", None)
        if callable(resolve_manifest_config):
            resolve_manifest_config()
    if hasattr(env_cfg, "refresh_zarr_dataset"):
        env_cfg.refresh_zarr_dataset = bool(args_cli.refresh_zarr_dataset)
    if hasattr(env_cfg, "reference_start_frame"):
        env_cfg.reference_start_frame = int(args_cli.reference_start_frame)
    if hasattr(env_cfg, "random_reset_full_trajectory"):
        env_cfg.random_reset_full_trajectory = False
    if hasattr(env_cfg, "random_reset_step_min"):
        env_cfg.random_reset_step_min = int(args_cli.reference_start_frame)
    if hasattr(env_cfg, "random_reset_step_max"):
        env_cfg.random_reset_step_max = int(args_cli.reference_start_frame)
    if hasattr(env_cfg, "reset_schedule"):
        env_cfg.reset_schedule = str(args_cli.reset_schedule)
    if hasattr(env_cfg, "wrap_steps"):
        env_cfg.wrap_steps = False
    if not args_cli.enable_observation_corruption:
        _disable_observation_corruption(env_cfg)

    step_dt = _configured_step_dt(env_cfg)
    if (
        step_dt is not None
        and hasattr(env_cfg, "episode_length_s")
        and not args_cli.preserve_episode_length
    ):
        current_episode_length_s = float(getattr(env_cfg, "episode_length_s"))
        required_episode_length_s = float(args_cli.steps + 2) * step_dt
        if current_episode_length_s < required_episode_length_s:
            env_cfg.episode_length_s = required_episode_length_s
            print(
                "[INFO] Extended env.episode_length_s for evaluation: "
                f"{current_episode_length_s:.3f} -> {required_episode_length_s:.3f}"
            )

    output_root = (
        args_cli.output_json.expanduser().resolve().parent
        if args_cli.output_json is not None
        else checkpoint_path.parent / "evaluation"
    )
    env_cfg.log_dir = str(output_root)

    agent_cfg.env.num_envs = int(args_cli.num_envs)
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    if hasattr(agent_cfg, "logger"):
        agent_cfg.logger.backend = ""
        agent_cfg.logger.log_dir = str(output_root / "agent_logs")
    if hasattr(agent_cfg, "device"):
        agent_cfg.device = env_cfg.sim.device
    if args_cli.planner_mode != "none" and int(args_cli.planner_update_interval) <= 0:
        raise ValueError("--planner_update_interval must be positive.")

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError(
            "DirectMARLEnv is not supported for RLOpt evaluation."
        )

    env = IsaacLabWrapper(raw_env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(args_cli.steps + 2)),
    )
    base_env = _unwrap_imitation_env(env)

    command_space = str(getattr(agent_cfg, "command_space", "unknown"))
    if (
        args_cli.planner_mode != "none"
        and len(_planner_command_terms(command_space)) == 0
    ):
        raise ValueError(
            "--planner_mode requires command_space to be full_body_trajectory or ee_trajectory."
        )
    tracked_body_names = _resolve_existing_body_names(base_env, G1_TRACKED_BODY_NAMES)
    ee_body_names = _resolve_existing_body_names(
        base_env,
        list(getattr(env_cfg, "command_ee_body_names", G1_EE_BODY_NAMES)),
    )
    if args_cli.debug_compare_command_sources:
        _debug_compare_command_sources(
            env,
            base_env,
            command_space=command_space,
            ee_body_names=ee_body_names,
        )
        env.close()
        return

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(str(checkpoint_path))
    collector_policy = agent.collector_policy
    collector_policy.eval()

    num_envs = int(args_cli.num_envs)
    active = torch.ones(num_envs, dtype=torch.bool)
    survival_steps = torch.zeros(num_envs, dtype=torch.float32)
    return_sum = torch.zeros(num_envs, dtype=torch.float32)
    done_events = torch.zeros(num_envs, dtype=torch.float32)
    terminated_events = torch.zeros(num_envs, dtype=torch.float32)
    truncated_events = torch.zeros(num_envs, dtype=torch.float32)
    metric_stats: dict[str, dict[str, float]] = {}
    previous_action: torch.Tensor | None = None
    steps_executed = 0
    valid_transition_count = 0
    planner_publish_count = 0
    dt = float(getattr(base_env, "step_dt", 0.0) or 0.0)

    td = env.reset()
    print(
        "[INFO] Starting deterministic evaluation: "
        f"num_envs={num_envs}, steps={args_cli.steps}, command_space={command_space}"
    )
    for step_idx in range(int(args_cli.steps)):
        step_active = active.clone()
        if not bool(step_active.any()):
            break

        published = _maybe_publish_planner_command(
            base_env,
            command_space=command_space,
            ee_body_names=ee_body_names,
            planner_mode=args_cli.planner_mode,
            planner_update_interval=int(args_cli.planner_update_interval),
            planner_noise_std=float(args_cli.planner_noise_std),
            step_index=step_idx,
        )
        if published:
            planner_publish_count += 1
        if args_cli.planner_mode != "none":
            td = _refresh_tensordict_observations(td, base_env)

        for metric_name, metric_values in _command_metrics(
            base_env,
            command_space=command_space,
            ee_body_names=ee_body_names,
        ).items():
            _accumulate_metric(
                metric_stats,
                metric_name,
                metric_values.cpu(),
                step_active,
            )

        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = collector_policy(td)

        action = td.get("action")
        if action is None:
            raise RuntimeError("Policy did not write an 'action' tensor.")
        action_2d = action.detach().reshape(num_envs, -1)
        action_l2 = torch.linalg.vector_norm(action_2d, dim=-1).cpu()
        _accumulate_metric(metric_stats, "action_l2", action_l2, step_active)
        if previous_action is not None:
            action_delta_l2 = torch.linalg.vector_norm(
                action_2d.cpu() - previous_action, dim=-1
            )
            _accumulate_metric(
                metric_stats, "action_delta_l2", action_delta_l2, step_active
            )
            if dt > 0.0:
                _accumulate_metric(
                    metric_stats,
                    "action_rate_l2",
                    action_delta_l2 / dt,
                    step_active,
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
            td_step,
            ("next", "terminated"),
            num_envs=num_envs,
            default=False,
        ).bool()
        truncateds = _optional_flat_tensor(
            td_step,
            ("next", "truncated"),
            num_envs=num_envs,
            default=False,
        ).bool()
        done_any = dones | terminateds | truncateds
        return_sum += rewards.float() * step_active.float()
        survival_steps += step_active.float()
        done_events += (done_any & step_active).float()
        terminated_events += (terminateds & step_active).float()
        truncated_events += (truncateds & step_active).float()

        metric_mask = (
            step_active if args_cli.keep_after_done else step_active & ~done_any
        )
        valid_transition_count += int(metric_mask.sum().item())
        tracking = _tracking_metrics(
            base_env,
            tracked_body_names=tracked_body_names,
            ee_body_names=ee_body_names,
        )
        for metric_name, metric_values in tracking.items():
            _accumulate_metric(
                metric_stats, metric_name, metric_values.cpu(), metric_mask
            )

        if not args_cli.keep_after_done:
            active &= ~done_any

        td = step_mdp(
            td_step, exclude_reward=True, exclude_done=False, exclude_action=True
        )
        steps_executed = step_idx + 1

    active_mask = survival_steps > 0
    return_mean, return_std = _tensor_mean_std(return_sum, active_mask)
    survival_mean, survival_std = _tensor_mean_std(survival_steps, active_mask)
    num_evaluated_envs = int(active_mask.sum().item())

    aggregate = {
        "return_sum_mean": return_mean,
        "return_sum_std": return_std,
        "survival_steps_mean": survival_mean,
        "survival_steps_std": survival_std,
        "done_rate": float((done_events[active_mask] > 0).float().mean().item())
        if num_evaluated_envs > 0
        else float("nan"),
        "terminated_rate": float(
            (terminated_events[active_mask] > 0).float().mean().item()
        )
        if num_evaluated_envs > 0
        else float("nan"),
        "truncated_rate": float(
            (truncated_events[active_mask] > 0).float().mean().item()
        )
        if num_evaluated_envs > 0
        else float("nan"),
        "done_events_per_env": float(done_events[active_mask].mean().item())
        if num_evaluated_envs > 0
        else float("nan"),
        "steps_executed": int(steps_executed),
        "valid_transition_count": int(valid_transition_count),
        "num_evaluated_envs": int(num_evaluated_envs),
        "planner_publish_count": int(planner_publish_count),
    }
    summary = {
        "metadata": {
            "label": args_cli.label,
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "checkpoint": str(checkpoint_path),
            "motion_manifest": str(motion_manifest)
            if motion_manifest is not None
            else None,
            "command_space": command_space,
            "command_observation_source": str(
                getattr(base_env, "_command_observation_source", "unknown")
            ),
            "command_past_steps": int(getattr(base_env, "_latent_patch_past_steps", 0)),
            "command_future_steps": int(
                getattr(base_env, "_latent_patch_future_steps", 0)
            ),
            "num_envs": int(num_envs),
            "steps_requested": int(args_cli.steps),
            "seed": agent_cfg.seed,
            "reset_schedule": str(args_cli.reset_schedule),
            "reference_start_frame": int(args_cli.reference_start_frame),
            "keep_after_done": bool(args_cli.keep_after_done),
            "observation_corruption_enabled": bool(
                args_cli.enable_observation_corruption
            ),
            "planner_mode": args_cli.planner_mode,
            "planner_update_interval": int(args_cli.planner_update_interval),
            "planner_noise_std": float(args_cli.planner_noise_std),
            "tracked_body_names": tracked_body_names,
            "ee_body_names": ee_body_names,
        },
        "aggregate": aggregate,
        "metrics": _finalize_metric_stats(metric_stats),
    }

    output_json = args_cli.output_json
    if output_json is None:
        label = args_cli.label or command_space
        output_json = checkpoint_path.parent / "evaluation" / f"{label}_eval.json"
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    print(f"[INFO] Wrote JSON summary: {output_json}")
    if args_cli.output_csv is not None:
        _write_csv(summary, args_cli.output_csv, append=args_cli.append_csv)

    metrics = summary["metrics"]
    print(
        "[RESULT] "
        f"command_space={command_space} "
        f"return_sum_mean={aggregate['return_sum_mean']:.4f} "
        f"survival_steps_mean={aggregate['survival_steps_mean']:.1f} "
        f"done_rate={aggregate['done_rate']:.3f} "
        f"joint_pos_rmse_rad={metrics.get('joint_pos_rmse_rad', {}).get('mean', float('nan')):.4f} "
        f"ee_pos_error_m={metrics.get('ee_pos_error_m', {}).get('mean', float('nan')):.4f}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
