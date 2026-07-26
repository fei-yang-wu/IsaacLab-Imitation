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
parser.add_argument(
    "--low_level_command_mode",
    choices=("native", "streamed_vanilla"),
    default="native",
    help=(
        "native constructs the low-level policy for the planner target interface; "
        "streamed_vanilla sends each full-body chunk through the unchanged "
        "single-frame vanilla tracker."
    ),
)
parser.add_argument("--planner_checkpoint", type=Path, required=True)
parser.add_argument("--output_json", type=Path, default=None)
parser.add_argument("--output_csv", type=Path, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Recorded control steps. <=0 records the full requested evaluation.",
)
parser.add_argument("--append_csv", action="store_true", default=False)
parser.add_argument(
    "--save_rollout_training_samples",
    action="store_true",
    default=False,
    help="Save planner-visited causal states with expert targets for DAgger.",
)
parser.add_argument("--samples_output_dir", type=Path, default=None)
parser.add_argument(
    "--sample_rows_per_file",
    type=int,
    default=1,
    help="Buffer this many planner rows per sample file.",
)
parser.add_argument(
    "--balanced_rows_per_motion",
    type=int,
    default=0,
    help=(
        "When positive, save exactly this many planner rows for the selected "
        "--motion_name and stop collection once the budget is complete."
    ),
)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--motion_manifest", type=Path, default=None)
parser.add_argument(
    "--dataset_path",
    type=Path,
    default=None,
    help="Existing trajectory cache matching --motion_manifest.",
)
parser.add_argument(
    "--motion_name",
    type=str,
    default="",
    help="Explicitly restrict the reference to one named motion.",
)
parser.add_argument(
    "--language_embeddings",
    type=Path,
    default=None,
    help="Language embedding table for a language-conditioned shared planner.",
)
parser.add_argument(
    "--language_goal_name",
    type=str,
    default="",
    help="Explicit deployable language goal; never inferred from the reference cursor.",
)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--state_history_steps", type=int, default=9)
parser.add_argument("--command_past_steps", type=int, default=0)
parser.add_argument("--command_future_steps", type=int, default=25)
parser.add_argument(
    "--planner_update_interval",
    type=int,
    default=1,
    help=(
        "Query the planner every N control steps. 1 replans per step "
        "(receding horizon). N>1 holds each published chunk for N steps, "
        "consumed VLA-style via env.command_hold_steps=N, so the planner "
        "runs at (control rate / N)."
    ),
)
parser.add_argument("--flow_num_inference_steps", type=int, default=16)
parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
parser.add_argument(
    "--tracking_success_root_height_threshold",
    type=float,
    default=0.25,
    help=(
        "Tracking failure threshold for absolute root-height deviation from "
        "the reference. Set <=0 to disable this criterion."
    ),
)
parser.add_argument(
    "--tracking_success_root_ori_threshold",
    type=float,
    default=1.0,
    help=(
        "Tracking failure threshold for root orientation error in radians. "
        "Set <=0 to disable this criterion."
    ),
)
parser.add_argument("--reset_schedule", type=str, default="sequential")
parser.add_argument("--reference_start_frame", type=int, default=0)
parser.add_argument("--refresh_zarr_dataset", action="store_true", default=False)
parser.add_argument("--keep_after_done", action="store_true", default=False)
parser.add_argument(
    "--keep_configured_episode_length",
    action="store_true",
    default=False,
    help=(
        "Keep the task's configured timeout instead of extending it to cover "
        "--steps. Use this for M3 so each rollout matches the low-level "
        "training episode duration."
    ),
)
parser.add_argument(
    "--disable_tracking_terminations",
    action="store_true",
    default=False,
    help=(
        "Treat anchor position/orientation and end-effector tracking errors as "
        "metrics instead of termination conditions. The base-too-low fall "
        "termination remains active."
    ),
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
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    VANILLA_POLICY_INPUT_KEYS,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from low_level_tracker import load_frozen_low_level_tracker  # noqa: E402
from balanced_motion_rows import BalancedMotionRowSelector  # noqa: E402
from paper_protocol_metadata import interval_event_metadata  # noqa: E402
from planner_latency import PlannerForwardTimer  # noqa: E402


TRACKING_TERMINATION_NAMES = ("anchor_pos", "anchor_ori", "ee_body_pos")
FALL_TERMINATION_NAME = "base_too_low"


def _disable_tracking_terminations(terminations: Any) -> list[str]:
    disabled: list[str] = []
    for name in TRACKING_TERMINATION_NAMES:
        if hasattr(terminations, name) and getattr(terminations, name) is not None:
            setattr(terminations, name, None)
            disabled.append(name)
    return disabled


from planner_publish_schedule import planner_renew_env_ids  # noqa: E402

from interface_planner_common import (  # noqa: E402
    flatten_command_terms,
    load_language_goal_embedding,
    load_planner_checkpoint,
    planner_state_from_batch,
    rmse_per_row,
    unflatten_command_target,
)
from planner_sample_schema import (  # noqa: E402
    PlannerSampleWriter,
    add_sample_format_metadata,
    build_planner_sample,
)


def _trajectory_metadata(raw_env: Any) -> dict[str, Any]:
    """Record the active motion and reference frame for every environment."""
    trajectory_manager = getattr(raw_env, "trajectory_manager", None)
    try:
        names = [str(name) for name in raw_env.expert_trajectory_motion_names()]
    except Exception:
        names = []
    if trajectory_manager is None:
        return {"trajectory_ranks": [], "motion_names": [], "local_steps": []}
    ranks = trajectory_manager.env_traj_rank.detach().cpu().reshape(-1).tolist()
    local_steps = trajectory_manager.env_step.detach().cpu().reshape(-1).tolist()
    rank_tensor = trajectory_manager.env_traj_rank.reshape(-1).to(
        device=trajectory_manager._state_device, dtype=torch.long
    )
    lengths = trajectory_manager._length.index_select(0, rank_tensor)
    return {
        "trajectory_ranks": [int(rank) for rank in ranks],
        "motion_names": [
            names[int(rank)] if 0 <= int(rank) < len(names) else str(rank)
            for rank in ranks
        ],
        "local_steps": [int(step) for step in local_steps],
        "trajectory_lengths": [
            int(length) for length in lengths.detach().cpu().tolist()
        ],
    }


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


def _body_tracking_tensors(
    base_env: ImitationRLEnv,
    names: list[str],
) -> dict[str, torch.Tensor] | None:
    if len(names) == 0:
        return None
    body_ids = [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]
    actual_pos, actual_quat = base_env._get_robot_body_pose_w_fast(body_ids)
    ref_pos, ref_quat = base_env._get_reference_body_pose_w_fast(tuple(names))
    actual_ang_vel, actual_lin_vel = base_env._get_robot_body_velocity_w_fast(body_ids)
    ref_ang_vel, ref_lin_vel = base_env._get_reference_body_velocity_w_fast(
        tuple(names)
    )
    return {
        "actual_pos": actual_pos,
        "actual_quat": actual_quat,
        "actual_ang_vel": actual_ang_vel,
        "actual_lin_vel": actual_lin_vel,
        "ref_pos": ref_pos,
        "ref_quat": ref_quat,
        "ref_ang_vel": ref_ang_vel,
        "ref_lin_vel": ref_lin_vel,
    }


def _tracking_metrics(
    base_env: ImitationRLEnv,
    *,
    tracked_body_names: list[str],
    ee_body_names: list[str],
    tracking_success_root_height_threshold: float,
    tracking_success_root_ori_threshold: float,
) -> tuple[
    dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor
]:
    robot_data = base_env.robot.data
    root_pos_ref, root_quat_ref, root_lin_vel_ref, root_ang_vel_ref = (
        base_env._get_reference_root_state_w_fast()
    )
    joint_pos_ref = base_env.current_expert_frame["joint_pos"]
    joint_vel_ref = base_env.current_expert_frame["joint_vel"]
    root_pos_error = robot_data.root_pos_w - root_pos_ref
    root_ori_error = math_utils.quat_error_magnitude(
        robot_data.root_quat_w, root_quat_ref
    )
    root_height_error = torch.abs(root_pos_error[:, 2])
    tracking_failure = torch.zeros_like(root_height_error, dtype=torch.bool)
    if float(tracking_success_root_height_threshold) > 0.0:
        tracking_failure |= root_height_error > float(
            tracking_success_root_height_threshold
        )
    if float(tracking_success_root_ori_threshold) > 0.0:
        tracking_failure |= root_ori_error > float(tracking_success_root_ori_threshold)
    metrics = {
        "tracking_failure": tracking_failure.float(),
        "root_pos_xyz_error_m": torch.linalg.vector_norm(root_pos_error, dim=-1),
        "root_pos_xy_error_m": torch.linalg.vector_norm(root_pos_error[:, :2], dim=-1),
        "root_height_error_m": root_height_error,
        "root_ori_error_rad": root_ori_error,
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
    tracked_body_lin_vel: tuple[torch.Tensor, torch.Tensor] | None = None
    tracked_tensors = _body_tracking_tensors(base_env, tracked_body_names)
    if tracked_tensors is not None:
        tracked_pos_error = torch.linalg.vector_norm(
            tracked_tensors["actual_pos"] - tracked_tensors["ref_pos"], dim=-1
        )
        tracked_ori_error = math_utils.quat_error_magnitude(
            tracked_tensors["actual_quat"].reshape(-1, 4),
            tracked_tensors["ref_quat"].reshape(-1, 4),
        ).reshape(tracked_tensors["actual_quat"].shape[0], -1)
        actual_root_rel = (
            tracked_tensors["actual_pos"] - robot_data.root_pos_w[:, None, :]
        )
        ref_root_rel = tracked_tensors["ref_pos"] - root_pos_ref[:, None, :]
        tracking_mpjpe_m = torch.linalg.vector_norm(
            actual_root_rel - ref_root_rel, dim=-1
        ).mean(dim=-1)
        body_lin_vel_error = torch.linalg.vector_norm(
            tracked_tensors["actual_lin_vel"] - tracked_tensors["ref_lin_vel"], dim=-1
        ).mean(dim=-1)
        body_ang_vel_error = torch.linalg.vector_norm(
            tracked_tensors["actual_ang_vel"] - tracked_tensors["ref_ang_vel"], dim=-1
        ).mean(dim=-1)
        metrics["tracked_body_pos_error_m"] = tracked_pos_error.mean(dim=-1)
        metrics["tracked_body_ori_error_rad"] = tracked_ori_error.mean(dim=-1)
        metrics["tracked_body_lin_vel_error_mps"] = body_lin_vel_error
        metrics["tracked_body_ang_vel_error_radps"] = body_ang_vel_error
        metrics["tracking_mpjpe_m"] = tracking_mpjpe_m
        metrics["tracking_mpjpe_mm"] = tracking_mpjpe_m * 1000.0
        metrics["tracking_velocity_distance_mps"] = body_lin_vel_error
        tracked_body_lin_vel = (
            tracked_tensors["actual_lin_vel"].detach(),
            tracked_tensors["ref_lin_vel"].detach(),
        )
    ee_errors = _mean_body_pose_errors(base_env, ee_body_names)
    if ee_errors is not None:
        metrics["ee_pos_error_m"] = ee_errors[0]
        metrics["ee_ori_error_rad"] = ee_errors[1]
    return metrics, tracked_body_lin_vel, tracking_failure


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
    env_ids: torch.Tensor | None = None,
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
            env_ids=env_ids,
            **ref_kwargs,
        )
        for term_name in term_names
    }


def _current_demonstration_command_terms(
    base_env: ImitationRLEnv,
    *,
    interface: str,
    ee_body_names: list[str],
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ref_kwargs = _command_reference_kwargs(interface, ee_body_names=ee_body_names)
    return base_env.current_offline_demo_command_terms(
        past_steps=int(args_cli.command_past_steps),
        future_steps=int(args_cli.command_future_steps),
        env_ids=env_ids,
        **ref_kwargs,
    )


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


def _require_streamed_tracker_checkpoint_contract(
    planner_metadata: dict[str, Any],
    tracker_provenance: dict[str, Any],
    *,
    command_future_steps: int,
    planner_interval_steps: int,
    seed: int,
) -> None:
    """Reject planners trained for a different explicit low-level interface."""
    sample_metadata = planner_metadata.get("sample_metadata")
    if not isinstance(sample_metadata, dict):
        raise ValueError(
            "Streamed-vanilla planner checkpoint has no sample_metadata; "
            "retrain it from provenance-bound planner samples."
        )
    expected_values = {
        "interface": "full_body_trajectory",
        "low_level_command_mode": "streamed_vanilla",
        "low_level_command_space": "single_frame_full_body",
        "policy_command_mode": "full_body_chunk_current_slot",
        "command_past_steps": 0,
        "command_future_steps": int(command_future_steps),
        "planner_interval_steps": int(planner_interval_steps),
        "seed": int(seed),
    }
    mismatches = {
        key: {"checkpoint": sample_metadata.get(key), "runtime": expected}
        for key, expected in expected_values.items()
        if sample_metadata.get(key) != expected
    }
    source_tracker = sample_metadata.get("low_level_tracker")
    if not isinstance(source_tracker, dict):
        provenance = sample_metadata.get("provenance")
        if isinstance(provenance, dict):
            source_tracker = provenance.get("low_level_tracker")
    if not isinstance(source_tracker, dict):
        raise ValueError(
            "Streamed-vanilla planner samples have no frozen-tracker provenance."
        )
    for key in (
        "checkpoint_sha256",
        "policy_input_keys",
        "strict_policy_restore",
        "policy_frozen",
    ):
        if source_tracker.get(key) != tracker_provenance.get(key):
            mismatches[f"low_level_tracker.{key}"] = {
                "checkpoint": source_tracker.get(key),
                "runtime": tracker_provenance.get(key),
            }
    if mismatches:
        raise ValueError(
            "Planner checkpoint is incompatible with the runtime streamed-vanilla "
            f"contract: {mismatches}."
        )


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg
) -> None:
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.sample_rows_per_file <= 0:
        raise ValueError("--sample_rows_per_file must be positive.")
    if args_cli.balanced_rows_per_motion < 0:
        raise ValueError("--balanced_rows_per_motion must be non-negative.")
    if args_cli.balanced_rows_per_motion > 0 and not bool(
        args_cli.save_rollout_training_samples
    ):
        raise ValueError(
            "--balanced_rows_per_motion requires --save_rollout_training_samples."
        )
    if args_cli.balanced_rows_per_motion > 0 and not str(args_cli.motion_name).strip():
        raise ValueError(
            "--balanced_rows_per_motion requires one explicit --motion_name."
        )
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
    planner_latency_timer = PlannerForwardTimer(planner)
    planner_language_dim = int(getattr(planner, "language_dim", 0))
    forced_language: torch.Tensor | None = None
    language_metadata: dict[str, Any] = {
        "enabled": False,
        "embedding_dim": 0,
    }
    if planner_language_dim > 0:
        if args_cli.language_embeddings is None:
            raise ValueError(
                "Language-conditioned planners require --language_embeddings."
            )
        if not str(args_cli.language_goal_name).strip():
            raise ValueError(
                "Language-conditioned deployable evaluation requires an explicit "
                "--language_goal_name."
            )
        if (
            str(args_cli.motion_name).strip()
            and str(args_cli.motion_name).strip()
            != str(args_cli.language_goal_name).strip()
        ):
            raise ValueError(
                "--motion_name must match --language_goal_name for a "
                "language-conditioned evaluation."
            )
        forced_language, language_metadata = load_language_goal_embedding(
            args_cli.language_embeddings,
            goal_name=args_cli.language_goal_name,
            device=next(planner.parameters()).device,
        )
        if int(forced_language.shape[-1]) != planner_language_dim:
            raise ValueError(
                "Language table width does not match planner checkpoint: "
                f"{forced_language.shape[-1]} != {planner_language_dim}."
            )
    elif (
        args_cli.language_embeddings is not None
        or str(args_cli.language_goal_name).strip()
    ):
        raise ValueError(
            "State-only planner checkpoint does not accept language input."
        )
    interface = target_spec.interface
    low_level_command_mode = str(args_cli.low_level_command_mode)
    low_level_command_space = interface
    if low_level_command_mode == "streamed_vanilla":
        if interface != "full_body_trajectory":
            raise ValueError(
                "streamed_vanilla requires a full_body_trajectory planner target."
            )
        low_level_command_space = "single_frame_full_body"
        env_cfg.policy_command_mode = "full_body_chunk_current_slot"
    else:
        env_cfg.policy_command_mode = "reference"

    agent_cfg.command_space = low_level_command_space
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()
    env_cfg.latent_patch_past_steps = int(args_cli.command_past_steps)
    env_cfg.latent_patch_future_steps = int(args_cli.command_future_steps)
    env_cfg.command_observation_source = "planner"
    planner_update_interval = int(args_cli.planner_update_interval)
    if planner_update_interval < 1:
        raise ValueError("--planner_update_interval must be >= 1.")
    if planner_update_interval > 1 and int(args_cli.command_past_steps) != 0:
        raise ValueError(
            "--planner_update_interval > 1 requires --command_past_steps 0."
        )
    if low_level_command_mode == "streamed_vanilla":
        if int(args_cli.command_past_steps) != 0:
            raise ValueError("streamed_vanilla requires --command_past_steps 0.")
        if int(args_cli.command_future_steps) + 1 < planner_update_interval:
            raise ValueError(
                "streamed_vanilla requires command_future_steps + 1 >= "
                "planner_update_interval so every held control step has a slot."
            )
        env_cfg.command_hold_steps = planner_update_interval
    elif planner_update_interval > 1:
        env_cfg.command_hold_steps = planner_update_interval
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
    if args_cli.dataset_path is not None:
        env_cfg.dataset_path = str(args_cli.dataset_path.expanduser().resolve())
    if motion_manifest is not None:
        env_cfg.lafan1_manifest_path = str(motion_manifest)
        resolve_manifest_config = getattr(env_cfg, "_resolve_manifest_config", None)
        if callable(resolve_manifest_config):
            resolve_manifest_config(
                dataset_path_explicit=args_cli.dataset_path is not None
            )
    if str(args_cli.motion_name).strip():
        env_cfg.motions = [str(args_cli.motion_name).strip()]
    if hasattr(env_cfg, "refresh_zarr_dataset"):
        env_cfg.refresh_zarr_dataset = bool(args_cli.refresh_zarr_dataset)
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
    disabled_tracking_termination_terms: list[str] = []
    if args_cli.disable_tracking_terminations:
        if not hasattr(env_cfg, "random_reset_step_min") or not hasattr(
            env_cfg, "random_reset_step_max"
        ):
            raise ValueError("M3 evaluation requires configurable random reset steps.")
        env_cfg.random_reset_step_min = 0
        env_cfg.random_reset_step_max = 200
        if hasattr(env_cfg, "random_reset_full_trajectory"):
            env_cfg.random_reset_full_trajectory = False
        terminations = getattr(env_cfg, "terminations", None)
        if terminations is None:
            raise ValueError(
                "--disable_tracking_terminations requires an environment "
                "termination configuration."
            )
        disabled_tracking_termination_terms = _disable_tracking_terminations(
            terminations
        )
        missing = sorted(
            set(TRACKING_TERMINATION_NAMES) - set(disabled_tracking_termination_terms)
        )
        if missing:
            raise ValueError(
                "M3 tracking termination terms were missing or already disabled: "
                f"{missing}."
            )
        if (
            not hasattr(terminations, FALL_TERMINATION_NAME)
            or getattr(terminations, FALL_TERMINATION_NAME) is None
        ):
            raise ValueError(
                "M3 metrics-only evaluation requires the base_too_low fall "
                "termination to remain active."
            )
    step_dt = _configured_step_dt(env_cfg)
    episode_length_extension_enabled = bool(
        not args_cli.keep_configured_episode_length
        and step_dt is not None
        and hasattr(env_cfg, "episode_length_s")
    )
    if episode_length_extension_enabled:
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
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")
    video_dir: Path | None = None
    if args_cli.video:
        video_dir = output_root / "videos" / "play"
        video_length = (
            int(args_cli.video_length)
            if int(args_cli.video_length) > 0
            else int(args_cli.steps)
        )
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=max(1, video_length),
            disable_logger=True,
        )
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
    runtime_planner_observation_spec = base_env.causal_planner_observation_spec(
        history_steps=int(args_cli.state_history_steps)
    )
    checkpoint_planner_observation_spec = planner_metadata.get(
        "planner_observation_spec"
    )
    if checkpoint_planner_observation_spec is None:
        sample_metadata = planner_metadata.get("sample_metadata", {})
        if isinstance(sample_metadata, dict):
            checkpoint_planner_observation_spec = sample_metadata.get(
                "planner_observation_spec"
            )
    if not isinstance(checkpoint_planner_observation_spec, dict):
        raise ValueError(
            "Planner checkpoint has no causal planner_observation_spec. "
            "Retrain it from robot-only planner samples."
        )
    if checkpoint_planner_observation_spec != runtime_planner_observation_spec:
        raise ValueError(
            "Planner observation specification mismatch between checkpoint and "
            f"environment: {checkpoint_planner_observation_spec} != "
            f"{runtime_planner_observation_spec}."
        )
    if int(planner.state_dim) != int(runtime_planner_observation_spec["flat_dim"]):
        raise ValueError(
            f"Planner state_dim={planner.state_dim} does not match causal input "
            f"width {runtime_planner_observation_spec['flat_dim']}."
        )
    sample_metadata = add_sample_format_metadata(
        {
            "interface": interface,
            "low_level_command_mode": low_level_command_mode,
            "low_level_command_space": low_level_command_space,
            "policy_command_mode": str(env_cfg.policy_command_mode),
            "target_spec": target_spec.to_dict(),
            "state_history_steps": int(args_cli.state_history_steps),
            "command_past_steps": int(args_cli.command_past_steps),
            "command_future_steps": int(args_cli.command_future_steps),
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "seed": int(env_cfg.seed),
            "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
            "motion_name": str(args_cli.motion_name).strip() or None,
            "balanced_collection": (
                {
                    "motion_names": [str(args_cli.motion_name).strip()],
                    "rows_per_motion": int(args_cli.balanced_rows_per_motion),
                }
                if int(args_cli.balanced_rows_per_motion) > 0
                else None
            ),
            "planner_observation_spec": runtime_planner_observation_spec,
            "reset_schedule": str(getattr(env_cfg, "reset_schedule", "unknown")),
            "random_reset_step_min": int(getattr(env_cfg, "random_reset_step_min", -1)),
            "random_reset_step_max": int(getattr(env_cfg, "random_reset_step_max", -1)),
            "wrap_steps": bool(getattr(env_cfg, "wrap_steps", False)),
            "policy_observation_corruption_enabled": bool(
                getattr(
                    getattr(getattr(env_cfg, "observations", None), "policy", None),
                    "enable_corruption",
                    False,
                )
            ),
            "early_terminations_enabled": True,
            "tracking_terminations_enabled": not bool(
                args_cli.disable_tracking_terminations
            ),
            "disabled_tracking_termination_terms": (
                disabled_tracking_termination_terms
            ),
            "survival_definition": "no_base_too_low_termination",
            "time_out_enabled": True,
            "episode_length_extension_enabled": episode_length_extension_enabled,
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
            "reward_clipping_enabled": False,
            "push_perturbation": interval_event_metadata(env_cfg, "push_robot"),
            "language_conditioning": language_metadata,
            "provenance": {
                "low_level_checkpoint": str(checkpoint_path),
                "planner_checkpoint": str(planner_checkpoint),
                "motion_manifest": str(motion_manifest)
                if motion_manifest is not None
                else None,
                "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
            },
        },
        collection_stage="planner_rollout",
        planner_interval_steps=planner_update_interval,
        control_rate_hz=(1.0 / step_dt) if step_dt else 50.0,
    )
    samples_dir = (
        args_cli.samples_output_dir.expanduser().resolve()
        if args_cli.samples_output_dir is not None
        else output_root / "rollout_training_samples"
    )
    if args_cli.save_rollout_training_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)
    sample_writer = PlannerSampleWriter(
        samples_dir,
        rows_per_file=int(args_cli.sample_rows_per_file),
    )
    tracked_body_names = _resolve_existing_body_names(
        base_env, list(G1_TRACKED_BODY_NAMES)
    )
    ee_body_names = _resolve_existing_body_names(
        base_env,
        list(getattr(env_cfg, "command_ee_body_names", G1_EE_BODY_NAMES)),
    )

    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    print(f"[INFO] Loading low-level checkpoint: {checkpoint_path}")
    tracker_provenance: dict[str, Any] | None = None
    if low_level_command_mode == "streamed_vanilla":
        frozen_tracker = load_frozen_low_level_tracker(
            agent,
            checkpoint_path,
            expected_input_keys=VANILLA_POLICY_INPUT_KEYS,
            map_location=env_cfg.sim.device,
        )
        policy = frozen_tracker.policy
        tracker_provenance = frozen_tracker.provenance
        sample_metadata["low_level_tracker"] = tracker_provenance
        provenance = sample_metadata.get("provenance")
        if isinstance(provenance, dict):
            provenance["low_level_tracker"] = tracker_provenance
        _require_streamed_tracker_checkpoint_contract(
            planner_metadata,
            tracker_provenance,
            command_future_steps=int(args_cli.command_future_steps),
            planner_interval_steps=planner_update_interval,
            seed=int(env_cfg.seed),
        )
    else:
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
    termination_term_names = list(base_env.termination_manager.active_terms)
    termination_hits = {
        term_name: torch.zeros(num_envs, dtype=torch.bool)
        for term_name in termination_term_names
    }
    strict_failure_term_names = [
        term_name
        for term_name in termination_term_names
        if not base_env.termination_manager.get_term_cfg(term_name).time_out
        and term_name != "reference_finished"
    ]
    strict_tracking_failure_events = torch.zeros(num_envs, dtype=torch.float32)
    metric_stats: dict[str, list[torch.Tensor]] = {}
    previous_action: torch.Tensor | None = None
    previous_body_lin_vel: tuple[torch.Tensor, torch.Tensor] | None = None
    previous_velocity_valid = torch.zeros(num_envs, dtype=torch.bool)
    tracking_failure_events = torch.zeros(num_envs, dtype=torch.float32)
    valid_transition_count = 0
    planner_publish_count = 0
    saved_sample_files = 0
    saved_sample_rows = 0
    steps_run = 0
    episode_ids = torch.zeros(num_envs, dtype=torch.long)
    motion_name_table = [
        str(name) for name in base_env.expert_trajectory_motion_names()
    ]
    balanced_selector = (
        BalancedMotionRowSelector(
            [str(args_cli.motion_name).strip()],
            rows_per_motion=int(args_cli.balanced_rows_per_motion),
        )
        if int(args_cli.balanced_rows_per_motion) > 0
        else None
    )

    td = env.reset()
    start_trajectories = _trajectory_metadata(base_env)
    trajectory_manager = base_env.trajectory_manager
    start_trajectory_ranks = (
        trajectory_manager.env_traj_rank.detach().cpu().reshape(-1).to(torch.long)
    )
    start_motion_names = [
        motion_name_table[int(rank)]
        if 0 <= int(rank) < len(motion_name_table)
        else str(int(rank))
        for rank in start_trajectory_ranks.tolist()
    ]
    stop_reason = "max_steps"
    for step_idx in range(int(args_cli.steps)):
        step_active = active.clone()
        if not bool(step_active.any()):
            break
        renew_env_ids = planner_renew_env_ids(
            base_env.episode_length_buf,
            planner_update_interval,
            initial_publication=step_idx == 0,
        )
        if int(renew_env_ids.numel()) > 0:
            active_on_device = step_active.to(device=renew_env_ids.device)
            renew_env_ids = renew_env_ids[
                active_on_device.index_select(0, renew_env_ids)
            ]
        if int(renew_env_ids.numel()) > 0:
            achieved_batch = base_env.current_causal_planner_observation(
                env_ids=renew_env_ids,
                history_steps=int(args_cli.state_history_steps),
            )
            planner_state = planner_state_from_batch(
                achieved_batch,
                state_history_steps=int(args_cli.state_history_steps),
            ).to(device=next(planner.parameters()).device, dtype=torch.float32)
            language = (
                None
                if forced_language is None
                else forced_language.expand(int(planner_state.shape[0]), -1)
            )
            with torch.inference_mode(), planner_latency_timer.enabled():
                predicted_target = planner(
                    planner_state,
                    num_inference_steps=int(args_cli.flow_num_inference_steps),
                    inference_noise_std=float(args_cli.flow_inference_noise_std),
                    language=language,
                )
            command_terms = unflatten_command_target(
                predicted_target.to(device=base_env.device),
                target_spec,
            )
            base_env.set_agent_trajectory_command(
                command_terms,
                env_ids=renew_env_ids,
            )
            planner_publish_count += int(renew_env_ids.numel())

            reference_target, _ = flatten_command_terms(
                interface,
                _current_reference_command_terms(
                    base_env,
                    interface=interface,
                    ee_body_names=ee_body_names,
                    env_ids=renew_env_ids,
                ),
            )
            if args_cli.save_rollout_training_samples:
                expert_batch = base_env.current_expert_macro_transition_batch(
                    horizon_steps=int(args_cli.command_future_steps),
                    env_ids=renew_env_ids,
                    state_history_steps=int(args_cli.state_history_steps),
                )
                all_traj_rank = (
                    expert_batch.get(("hl", "traj_rank")).detach().cpu().reshape(-1)
                )
                all_local_step = (
                    expert_batch.get(("hl", "local_step")).detach().cpu().reshape(-1)
                )
                candidate_motion_names = [
                    motion_name_table[int(rank)]
                    if 0 <= int(rank) < len(motion_name_table)
                    else str(int(rank))
                    for rank in all_traj_rank.tolist()
                ]
                if balanced_selector is not None and set(candidate_motion_names) != set(
                    balanced_selector.motion_names
                ):
                    raise RuntimeError(
                        "Parallel planner collection lost its explicit "
                        "goal-to-reference binding: "
                        f"expected {list(balanced_selector.motion_names)}, "
                        f"observed {sorted(set(candidate_motion_names))}."
                    )
                selected_positions_cpu = torch.tensor(
                    (
                        balanced_selector.select(candidate_motion_names)
                        if balanced_selector is not None
                        else tuple(range(len(candidate_motion_names)))
                    ),
                    dtype=torch.long,
                )
                selected_positions = selected_positions_cpu.to(
                    device=renew_env_ids.device
                )
                sample_env_ids = renew_env_ids.index_select(0, selected_positions)
                sample_planner_state = planner_state.index_select(
                    0,
                    selected_positions_cpu.to(device=planner_state.device),
                )
                sample_reference_target = reference_target.index_select(
                    0,
                    selected_positions_cpu.to(device=reference_target.device),
                )
                sample_language = (
                    None
                    if language is None
                    else language.index_select(
                        0,
                        selected_positions_cpu.to(device=language.device),
                    )
                )
                demonstration_target, _ = flatten_command_terms(
                    interface,
                    _current_demonstration_command_terms(
                        base_env,
                        interface=interface,
                        ee_body_names=ee_body_names,
                        env_ids=sample_env_ids,
                    ),
                )
                demonstration_batch = base_env.current_offline_demo_planner_observation(
                    env_ids=sample_env_ids,
                    history_steps=int(args_cli.state_history_steps),
                )
                traj_rank = all_traj_rank.index_select(0, selected_positions_cpu)
                local_step = all_local_step.index_select(0, selected_positions_cpu)
                motion_names = [
                    candidate_motion_names[int(index)]
                    for index in selected_positions_cpu.tolist()
                ]
                sample_env_ids_cpu = sample_env_ids.detach().cpu()
                sample = build_planner_sample(
                    causal_state_history=sample_planner_state,
                    demonstration_state_history=planner_state_from_batch(
                        demonstration_batch,
                        state_history_steps=int(args_cli.state_history_steps),
                    ),
                    causal_target=sample_reference_target,
                    demonstration_target=demonstration_target,
                    trajectory_rank=traj_rank,
                    episode_id=episode_ids.index_select(0, sample_env_ids_cpu),
                    control_step=local_step,
                    planner_step=torch.div(
                        local_step,
                        planner_update_interval,
                        rounding_mode="floor",
                    ),
                    motion_names=motion_names,
                    metadata=sample_metadata,
                    language_embedding=sample_language,
                )
                sample_writer.add(sample)
                saved_sample_rows += int(sample_reference_target.shape[0])
            target_rmse = rmse_per_row(
                predicted_target.to(reference_target.device),
                reference_target,
            )
            _accumulate_metric(
                metric_stats,
                "planner_target_rmse",
                target_rmse.cpu(),
                torch.ones(target_rmse.shape[0], dtype=torch.bool),
            )
            if balanced_selector is not None and balanced_selector.complete:
                stop_reason = "balanced_rows_complete"
                break
        td = _refresh_tensordict_observations(td, base_env)

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
        steps_run += 1
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
        episode_ids += done_any.to(dtype=torch.long)
        return_sum += rewards.float() * step_active.float()
        survival_steps += step_active.float()
        done_events += (done_any & step_active).float()
        terminated_events += (terminateds & step_active).float()
        truncated_events += (truncateds & step_active).float()
        current_termination_terms: dict[str, torch.Tensor] = {}
        for term_name in termination_term_names:
            term_values = (
                base_env.termination_manager.get_term(term_name)
                .detach()
                .reshape(-1)[:num_envs]
                .to(device="cpu", dtype=torch.bool)
            )
            current_termination_terms[term_name] = term_values
            termination_hits[term_name] |= term_values & step_active
        strict_failure = torch.zeros(num_envs, dtype=torch.bool)
        for term_name in strict_failure_term_names:
            strict_failure |= current_termination_terms[term_name]
        strict_tracking_failure_events += (strict_failure & step_active).float()

        metric_mask = (
            step_active if args_cli.keep_after_done else step_active & ~done_any
        )
        valid_transition_count += int(metric_mask.sum().item())
        tracking_metrics, body_lin_vel, tracking_failure = _tracking_metrics(
            base_env,
            tracked_body_names=tracked_body_names,
            ee_body_names=ee_body_names,
            tracking_success_root_height_threshold=float(
                args_cli.tracking_success_root_height_threshold
            ),
            tracking_success_root_ori_threshold=float(
                args_cli.tracking_success_root_ori_threshold
            ),
        )
        tracking_failure_events += (tracking_failure.cpu() & step_active).float()
        for metric_name, values in tracking_metrics.items():
            _accumulate_metric(metric_stats, metric_name, values.cpu(), metric_mask)
        if body_lin_vel is not None and step_dt is not None:
            if previous_body_lin_vel is not None:
                actual_lin_vel, ref_lin_vel = body_lin_vel
                prev_actual_lin_vel, prev_ref_lin_vel = previous_body_lin_vel
                actual_acc = (actual_lin_vel - prev_actual_lin_vel) / float(step_dt)
                ref_acc = (ref_lin_vel - prev_ref_lin_vel) / float(step_dt)
                acceleration_distance = torch.linalg.vector_norm(
                    actual_acc - ref_acc, dim=-1
                ).mean(dim=-1)
                acceleration_mask = metric_mask & previous_velocity_valid
                _accumulate_metric(
                    metric_stats,
                    "tracking_acceleration_distance_mps2",
                    acceleration_distance.cpu(),
                    acceleration_mask,
                )
            previous_body_lin_vel = (body_lin_vel[0].clone(), body_lin_vel[1].clone())
            previous_velocity_valid = step_active & ~done_any
        if not args_cli.keep_after_done:
            active &= ~done_any
        td = step_mdp(
            td_step, exclude_reward=True, exclude_done=False, exclude_action=True
        )

    sample_writer.flush()
    saved_sample_files = sample_writer.file_count
    if saved_sample_rows != sample_writer.row_count:
        raise RuntimeError(
            "Planner sample writer row accounting differs from collection: "
            f"collected={saved_sample_rows}, written={sample_writer.row_count}."
        )
    active_mask = survival_steps > 0
    return_mean, return_std = _tensor_mean_std(return_sum, active_mask)
    survival_mean, survival_std = _tensor_mean_std(survival_steps, active_mask)
    fall_events = termination_hits.get(
        FALL_TERMINATION_NAME, torch.zeros(num_envs, dtype=torch.bool)
    )
    fall_free = ~fall_events
    aggregate = {
        "return_sum_mean": return_mean,
        "return_sum_std": return_std,
        "survival_steps_mean": survival_mean,
        "survival_steps_std": survival_std,
        "survival_rate": float(fall_free[active_mask].float().mean().item())
        if bool(active_mask.any())
        else float("nan"),
        "fall_free_rate": float(fall_free[active_mask].float().mean().item())
        if bool(active_mask.any())
        else float("nan"),
        "fall_rate": float(fall_events[active_mask].float().mean().item())
        if bool(active_mask.any())
        else float("nan"),
        "fallen_env_count": int(fall_events[active_mask].sum().item())
        if bool(active_mask.any())
        else 0,
        "done_rate": float((done_events[active_mask] > 0).float().mean().item())
        if bool(active_mask.any())
        else float("nan"),
        "tracking_success_rate": float(
            (strict_tracking_failure_events[active_mask] == 0).float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "tracking_failure_rate": float(
            (strict_tracking_failure_events[active_mask] > 0).float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "tracking_failed_env_count": int(
            (strict_tracking_failure_events[active_mask] > 0).sum().item()
        )
        if bool(active_mask.any())
        else 0,
        "threshold_tracking_success_rate": float(
            (tracking_failure_events[active_mask] == 0).float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "tracking_success_root_height_threshold": float(
            args_cli.tracking_success_root_height_threshold
        ),
        "tracking_success_root_ori_threshold": float(
            args_cli.tracking_success_root_ori_threshold
        ),
        "valid_transition_count": int(valid_transition_count),
        "planner_publish_count": int(planner_publish_count),
        "termination_cause_env_counts": {
            term_name: int(values[active_mask].sum().item())
            for term_name, values in termination_hits.items()
        },
    }
    summary = {
        "metadata": {
            "label": args_cli.label,
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "checkpoint": str(checkpoint_path),
            "planner_checkpoint": str(planner_checkpoint),
            "interface": interface,
            "low_level_command_mode": low_level_command_mode,
            "low_level_command_space": low_level_command_space,
            "policy_command_mode": str(env_cfg.policy_command_mode),
            "state_history_steps": int(args_cli.state_history_steps),
            "command_past_steps": int(args_cli.command_past_steps),
            "command_future_steps": int(args_cli.command_future_steps),
            "planner_update_interval": planner_update_interval,
            "flow_num_inference_steps": int(args_cli.flow_num_inference_steps),
            "flow_inference_noise_std": float(args_cli.flow_inference_noise_std),
            "planner_target_dim": int(target_spec.target_dim),
            "planner_metadata": planner_metadata,
            "planner_observation_spec": runtime_planner_observation_spec,
            "low_level_tracker": tracker_provenance,
            "num_envs": int(num_envs),
            "seed": int(env_cfg.seed),
            "motion_manifest": str(motion_manifest)
            if motion_manifest is not None
            else None,
            "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
            "motion_name": str(args_cli.motion_name).strip() or None,
            "reset_schedule": str(getattr(env_cfg, "reset_schedule", "unknown")),
            "random_reset_step_min": int(getattr(env_cfg, "random_reset_step_min", -1)),
            "random_reset_step_max": int(getattr(env_cfg, "random_reset_step_max", -1)),
            "wrap_steps": bool(getattr(env_cfg, "wrap_steps", False)),
            "policy_observation_corruption_enabled": bool(
                getattr(
                    getattr(getattr(env_cfg, "observations", None), "policy", None),
                    "enable_corruption",
                    False,
                )
            ),
            "early_terminations_enabled": True,
            "tracking_terminations_enabled": not bool(
                args_cli.disable_tracking_terminations
            ),
            "disabled_tracking_termination_terms": (
                disabled_tracking_termination_terms
            ),
            "survival_definition": "no_base_too_low_termination",
            "time_out_enabled": True,
            "episode_length_extension_enabled": episode_length_extension_enabled,
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
            "reward_clipping_enabled": False,
            "push_perturbation": interval_event_metadata(env_cfg, "push_robot"),
            "language_conditioning": language_metadata,
        },
        "aggregate": aggregate,
        "metrics": _finalize_metric_stats(metric_stats),
        "start_trajectories": start_trajectories,
        "final_trajectories": _trajectory_metadata(base_env),
        "planner_inference_latency_ms": planner_latency_timer.summary(warmup_calls=1),
        "video_dir": str(video_dir) if video_dir is not None else None,
        "save_rollout_training_samples": bool(args_cli.save_rollout_training_samples),
        "samples_output_dir": str(samples_dir)
        if args_cli.save_rollout_training_samples
        else None,
        "sample_file_count": int(saved_sample_files),
        "sample_rows_per_file": int(args_cli.sample_rows_per_file),
        "balanced_collection": (
            {
                "motion_names": list(balanced_selector.motion_names),
                "rows_per_motion": balanced_selector.rows_per_motion,
                "counts": balanced_selector.counts(),
                "complete": balanced_selector.complete,
                "missing": balanced_selector.missing(),
            }
            if balanced_selector is not None
            else None
        ),
        "saved_rows": int(saved_sample_rows),
        "max_steps": int(args_cli.steps),
        "steps_run": int(steps_run),
        "stop_reason": (
            stop_reason
            if stop_reason == "balanced_rows_complete"
            else (
                "max_steps"
                if int(steps_run) == int(args_cli.steps)
                else "all_envs_done"
            )
        ),
        "per_environment": [
            {
                "env_id": env_id,
                "trajectory_rank": int(start_trajectory_ranks[env_id].item()),
                "motion_name": start_motion_names[env_id],
                "return_sum": float(return_sum[env_id].item()),
                "survival_steps": int(survival_steps[env_id].item()),
                "survived_without_fall": bool(fall_free[env_id].item()),
                "fell": bool(fall_events[env_id].item()),
                "done": bool(done_events[env_id].item() > 0),
                "terminated": bool(terminated_events[env_id].item() > 0),
                "truncated": bool(truncated_events[env_id].item() > 0),
                "tracking_success": bool(
                    strict_tracking_failure_events[env_id].item() == 0
                ),
                "termination_terms": [
                    term_name
                    for term_name in termination_term_names
                    if bool(termination_hits[term_name][env_id].item())
                ],
            }
            for env_id in range(num_envs)
            if bool(active_mask[env_id].item())
        ],
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
    if balanced_selector is not None and not balanced_selector.complete:
        raise RuntimeError(
            "Balanced collection ended before the selected motion reached its "
            f"row budget: {balanced_selector.missing()}."
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
    planner_latency_timer.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
