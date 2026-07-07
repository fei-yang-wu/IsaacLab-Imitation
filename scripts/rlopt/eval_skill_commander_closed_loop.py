#!/usr/bin/env python3
# ruff: noqa: E402
"""Closed-loop SkillCommander eval with achieved-state diagnostics.

This script runs a trained low-level controller in Isaac Lab, optionally records
video, and scores a loaded SkillCommander at the live rollout cursor. Unlike the
M1 expert-state diagnostic, it also feeds the commander the robot's achieved
macro state so we can measure the M3 failure mode directly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Run closed-loop low-level eval and log SkillCommander M3 metrics."
)
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Recorded video length in steps. <=0 uses --max_steps / reference end.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Rollout steps. <=0 runs until the active reference trajectory ends.",
)
parser.add_argument(
    "--metric_interval",
    type=int,
    default=1,
    help="Log M3 diagnostics every N simulation steps.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Imitation-G1-Latent-v0",
    help="Isaac Lab task.",
)
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD_BILINEAR",
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
    help="RLOpt low-level algorithm.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Low-level controller checkpoint (.pt).",
)
parser.add_argument(
    "--planner_checkpoint",
    type=str,
    required=True,
    help="SkillCommander checkpoint to score.",
)
parser.add_argument(
    "--skill_checkpoint",
    type=str,
    default=None,
    help="Override frozen high-level skill checkpoint from planner checkpoint.",
)
parser.add_argument(
    "--language_embeddings",
    type=str,
    default=None,
    help="Override language embedding table from planner checkpoint.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory. Defaults to logs/skill_commander_closed_loop_eval/<timestamp>.",
)
parser.add_argument("--label", type=str, default="", help="Optional summary label.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--motion_name",
    type=str,
    default=None,
    help="Restrict env.motions to this motion before env creation.",
)
parser.add_argument(
    "--trajectory_name",
    type=str,
    default=None,
    help="Restrict env.trajectories to this trajectory before env creation.",
)
parser.add_argument(
    "--allow_random_reset",
    action="store_true",
    default=False,
    help="Preserve env random reset offsets instead of forcing frame-0 eval.",
)
parser.add_argument(
    "--keep_time_out",
    action="store_true",
    default=False,
    help="Keep the task timeout termination. By default only reference end stops eval.",
)
parser.add_argument(
    "--keep_early_terminations",
    action="store_true",
    default=False,
    help=(
        "Keep non-reference failure terminations. By default only reference end "
        "stops eval."
    ),
)
parser.add_argument(
    "--continue_after_reset",
    action="store_true",
    default=False,
    help="Continue after env done/reset events instead of stopping at first done.",
)
parser.add_argument(
    "--save_rollout_training_samples",
    action="store_true",
    default=False,
    help="Save achieved-state planner inputs and target z tensors for finetuning.",
)
parser.add_argument(
    "--flow_num_inference_steps",
    type=int,
    default=None,
    help="Override flow-matching inference steps for metric-side scoring.",
)
parser.add_argument(
    "--flow_inference_noise_std",
    type=float,
    default=0.0,
    help="Override flow-matching inference noise std for metric-side scoring.",
)
parser.add_argument(
    "--diffusion_num_inference_steps",
    type=int,
    default=None,
    help="Override diffusion-policy inference steps for metric-side scoring.",
)
parser.add_argument(
    "--diffusion_inference_scheduler",
    type=str,
    default=None,
    choices=("ddpm", "ddim"),
    help="Override diffusion-policy inference scheduler for metric-side scoring.",
)
parser.add_argument(
    "--diffusion_ddim_eta",
    type=float,
    default=None,
    help="Override diffusion-policy DDIM eta for metric-side scoring.",
)
parser.add_argument(
    "--diffusion_inference_noise_std",
    type=float,
    default=None,
    help="Override diffusion-policy inference noise std for metric-side scoring.",
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
import torch.nn.functional as F
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils import math as math_utils
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_EE_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import (
    AMP,
    ASE,
    GAIL,
    IPMD,
    IPMDBilinear,
    IPMDSR,
    PPO,
    SAC,
    FastSAC,
    SkillCommanderConfig,
    SkillCommanderTrainer,
)
from tensordict import TensorDictBase
from tensordict.nn import InteractionType
from torch import Tensor
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
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


def resolve_agent_cfg_entry_point(task_name: str, algorithm: str) -> str:
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


def _run_dir() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs", "skill_commander_closed_loop_eval", timestamp).resolve()


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameter_count": int(
            sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
        ),
    }


def _skill_commander_planner_metadata(
    checkpoint: dict[str, Any],
    *,
    generator: torch.nn.Module,
    trainer_config: SkillCommanderConfig,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    checkpoint_metadata = checkpoint.get("metadata")
    if isinstance(checkpoint_metadata, dict):
        metadata.update(checkpoint_metadata)

    config = checkpoint.get("config")
    config_values = config if isinstance(config, dict) else trainer_config.to_dict()
    metadata.setdefault("interface", "latent_skill")
    metadata.setdefault(
        "planner_type",
        config_values.get("planner_type", generator.__class__.__name__),
    )
    for key in (
        "flow_num_inference_steps",
        "diffusion_num_inference_steps",
        "flow_inference_noise_std",
        "diffusion_inference_noise_std",
    ):
        value = config_values.get(key)
        if value not in (None, ""):
            metadata.setdefault(key, value)

    rollout_finetune = checkpoint.get("rollout_finetune")
    finetune_num_updates: int | None = None
    if isinstance(rollout_finetune, dict):
        sample_count = rollout_finetune.get("num_samples")
        if sample_count not in (None, ""):
            metadata.setdefault("source_sample_count", int(sample_count))
            metadata.setdefault("num_samples", int(sample_count))
            metadata.setdefault("selected_sample_count", int(sample_count))
            metadata.setdefault("heldout_sample_count", 0)
        for key in ("batch_size", "state_dim", "lang_embed_dim", "z_dim"):
            value = rollout_finetune.get(key)
            if value not in (None, ""):
                metadata.setdefault(key, value)
        args_payload = rollout_finetune.get("args")
        if isinstance(args_payload, dict):
            for key in (
                "num_updates",
                "lr",
                "weight_decay",
                "flow_inference_noise_std",
            ):
                value = args_payload.get(key)
                if value not in (None, ""):
                    metadata.setdefault(key, value)
            value = args_payload.get("num_updates")
            if value not in (None, ""):
                finetune_num_updates = int(value)
                metadata.setdefault("finetune_num_updates", finetune_num_updates)

    checkpoint_update = checkpoint.get("update")
    if metadata.get("pretrain_num_updates") in (None, "") and checkpoint_update not in (
        None,
        "",
    ):
        pretrain_update = int(checkpoint_update)
        if finetune_num_updates is not None:
            pretrain_update = max(0, pretrain_update - int(finetune_num_updates))
        metadata.setdefault("pretrain_num_updates", pretrain_update)

    metadata.update(_parameter_counts(generator))
    return metadata


def _mean_dict(rows: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}


def _any_done(td: Any) -> bool:
    for key in (
        ("next", "done"),
        ("next", "terminated"),
        ("next", "truncated"),
        "done",
        "terminated",
        "truncated",
    ):
        value = td.get(key, None)
        if isinstance(value, Tensor) and bool(value.any().detach().cpu().item()):
            return True
    return False


def _get_optional(td: TensorDictBase, key: str | tuple[str, ...]) -> Tensor | None:
    try:
        value = td.get(key)
    except KeyError:
        return None
    return value if isinstance(value, Tensor) else None


def _optional_flat_tensor(
    td: TensorDictBase,
    key: str | tuple[str, ...],
    *,
    num_envs: int,
    default: float | bool,
) -> Tensor:
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
) -> tuple[Tensor, Tensor] | None:
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
) -> dict[str, Tensor]:
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


def _accumulate_metric(
    stats: dict[str, list[Tensor]],
    metric_name: str,
    values: Tensor,
    mask: Tensor,
) -> None:
    selected = values.detach().cpu()[mask.cpu()]
    if selected.numel() == 0:
        return
    stats.setdefault(metric_name, []).append(selected.float())


def _finalize_metric_stats(
    stats: dict[str, list[Tensor]],
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


def _tensor_mean_std(values: Tensor, mask: Tensor) -> tuple[float, float]:
    selected = values[mask]
    if selected.numel() == 0:
        return float("nan"), float("nan")
    return (
        float(selected.mean().item()),
        float(selected.std(unbiased=False).item()) if selected.numel() > 1 else 0.0,
    )


def _auto_reference_steps(raw_env: Any) -> int:
    tm = getattr(raw_env, "trajectory_manager", None)
    if tm is None:
        return 500
    ranks = tm.env_traj_rank.reshape(-1).to(device=tm._state_device, dtype=torch.long)
    lengths = tm._length.index_select(0, ranks).to(dtype=torch.long)
    local_steps = tm.env_step.reshape(-1).to(device=tm._state_device, dtype=torch.long)
    remaining = (lengths - local_steps).clamp(min=1)
    return int(remaining.min().item())


def _trajectory_metadata(raw_env: Any) -> dict[str, Any]:
    tm = getattr(raw_env, "trajectory_manager", None)
    names: list[str] = []
    try:
        names = [str(name) for name in raw_env.expert_trajectory_motion_names()]
    except Exception:
        names = []
    if tm is None:
        return {"trajectory_ranks": [], "motion_names": [], "local_steps": []}
    ranks = tm.env_traj_rank.detach().cpu().reshape(-1).tolist()
    local_steps = tm.env_step.detach().cpu().reshape(-1).tolist()
    lengths = tm._length.index_select(
        0, tm.env_traj_rank.reshape(-1).to(device=tm._state_device, dtype=torch.long)
    )
    motion_names = [
        names[int(rank)] if 0 <= int(rank) < len(names) else str(rank) for rank in ranks
    ]
    return {
        "trajectory_ranks": [int(rank) for rank in ranks],
        "motion_names": motion_names,
        "local_steps": [int(step) for step in local_steps],
        "trajectory_lengths": [int(item) for item in lengths.detach().cpu().tolist()],
    }


def _trainer_config_from_checkpoint(
    checkpoint: dict[str, Any],
) -> SkillCommanderConfig:
    values = dict(checkpoint.get("config", {}))
    values.setdefault(
        "skill_checkpoint_path", checkpoint.get("skill_checkpoint_path", "")
    )
    values.setdefault(
        "language_embeddings_path", checkpoint.get("language_embeddings_path", "")
    )
    if args_cli.skill_checkpoint is not None:
        values["skill_checkpoint_path"] = str(
            Path(args_cli.skill_checkpoint).expanduser()
        )
    if args_cli.language_embeddings is not None:
        values["language_embeddings_path"] = str(
            Path(args_cli.language_embeddings).expanduser()
        )
    if args_cli.flow_num_inference_steps is not None:
        values["flow_num_inference_steps"] = int(args_cli.flow_num_inference_steps)
    if args_cli.flow_inference_noise_std is not None:
        values["flow_inference_noise_std"] = float(args_cli.flow_inference_noise_std)
    if args_cli.diffusion_num_inference_steps is not None:
        values["diffusion_num_inference_steps"] = int(
            args_cli.diffusion_num_inference_steps
        )
    if args_cli.diffusion_inference_scheduler is not None:
        values["diffusion_inference_scheduler"] = str(
            args_cli.diffusion_inference_scheduler
        )
    if args_cli.diffusion_ddim_eta is not None:
        values["diffusion_ddim_eta"] = float(args_cli.diffusion_ddim_eta)
    if args_cli.diffusion_inference_noise_std is not None:
        values["diffusion_inference_noise_std"] = float(
            args_cli.diffusion_inference_noise_std
        )
    values["batch_size"] = 1
    values["num_updates"] = 1
    values["eval_batches"] = 1
    values["eval_batch_size"] = 1
    return SkillCommanderConfig.from_dict(values)


def _disable_non_reference_terminations(terminations: Any) -> None:
    names = set(getattr(terminations, "__dict__", {}).keys())
    names.update(("anchor_pos", "anchor_ori", "ee_body_pos", "base_too_low"))
    for name in sorted(names):
        if name.startswith("_") or name == "reference_finished":
            continue
        if hasattr(terminations, name):
            setattr(terminations, name, None)


def _planner_state(batch: Any, state_history_steps: int) -> Tensor:
    if int(state_history_steps) > 0:
        state_history = batch.get(("hl", "state_history"))
        if state_history is None:
            msg = "Expected hl/state_history for state-history planner checkpoint."
            raise ValueError(msg)
        return state_history.reshape(int(state_history.shape[0]), -1).contiguous()
    return batch.get(("hl", "state"))


def _cosine_mean(lhs: Tensor, rhs: Tensor) -> float:
    return float(F.cosine_similarity(lhs, rhs, dim=-1).mean().detach().item())


def _mse_mean(lhs: Tensor, rhs: Tensor) -> float:
    return float((lhs - rhs).pow(2).mean().detach().item())


def _diff_stats(prefix: str, lhs: Tensor, rhs: Tensor) -> dict[str, float]:
    diff = lhs - rhs
    return {
        f"{prefix}/mae": float(diff.abs().mean().detach().item()),
        f"{prefix}/max_abs": float(diff.abs().amax().detach().item()),
        f"{prefix}/rmse": float(diff.pow(2).mean().sqrt().detach().item()),
    }


@torch.no_grad()
def _measure_commander(
    *,
    trainer: SkillCommanderTrainer,
    wrapped_env: IsaacLabWrapper,
    env_ids: Tensor,
    sample_path: Path | None = None,
    sample_step: int | None = None,
) -> dict[str, float]:
    horizon_steps = int(trainer.horizon_steps)
    state_history_steps = int(trainer.config.state_history_steps)
    expert_batch = wrapped_env.current_expert_macro_transition_batch(
        horizon_steps=horizon_steps,
        env_ids=env_ids,
        state_history_steps=state_history_steps,
    )
    achieved_batch = wrapped_env.current_achieved_macro_transition_batch(
        horizon_steps=horizon_steps,
        env_ids=env_ids,
        state_history_steps=state_history_steps,
    )

    expert_state = expert_batch.get(("hl", "state")).to(
        device=trainer.device, dtype=torch.float32
    )
    achieved_state = achieved_batch.get(("hl", "state")).to(
        device=trainer.device, dtype=torch.float32
    )
    future_window = expert_batch.get(("hl", "future_window")).to(
        device=trainer.device, dtype=torch.float32
    )
    traj_rank = (
        expert_batch.get(("hl", "traj_rank"))
        .reshape(-1)
        .to(device=trainer.device, dtype=torch.long)
    )
    expert_planner_state = _planner_state(expert_batch, state_history_steps).to(
        device=trainer.device, dtype=torch.float32
    )
    achieved_planner_state = _planner_state(achieved_batch, state_history_steps).to(
        device=trainer.device, dtype=torch.float32
    )

    z_target = trainer._target_z(expert_state, future_window)
    lang = trainer._lang_for_ranks(traj_rank)
    trainer.generator.eval()
    z_m1 = trainer.generator(expert_planner_state, lang)
    z_m3 = trainer.generator(achieved_planner_state, lang)

    if sample_path is not None:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": None if sample_step is None else int(sample_step),
                "planner_state": achieved_planner_state.detach().cpu(),
                "expert_planner_state": expert_planner_state.detach().cpu(),
                "lang": lang.detach().cpu(),
                "z_target": z_target.detach().cpu(),
                "traj_rank": traj_rank.detach().cpu(),
            },
            sample_path,
        )

    metrics = {
        "m1/z_cosine": _cosine_mean(z_m1, z_target),
        "m1/z_mse": _mse_mean(z_m1, z_target),
        "m3/z_cosine": _cosine_mean(z_m3, z_target),
        "m3/z_mse": _mse_mean(z_m3, z_target),
        "m3_vs_m1/z_cosine": _cosine_mean(z_m3, z_m1),
        "m3_vs_m1/z_mse": _mse_mean(z_m3, z_m1),
    }
    metrics.update(
        _diff_stats("state/achieved_vs_expert", achieved_state, expert_state)
    )

    slices = wrapped_env.expert_macro_feature_slices(horizon_steps=horizon_steps)
    for name, (start, end) in sorted(slices.items()):
        metrics.update(
            _diff_stats(
                f"state/{name}/achieved_vs_expert",
                achieved_state[:, int(start) : int(end)],
                expert_state[:, int(start) : int(end)],
            )
        )

    published = wrapped_env.get_agent_latent_command(env_ids=env_ids).to(
        device=trainer.device, dtype=torch.float32
    )
    z_dim = int(trainer.z_dim)
    if published.ndim == 2 and int(published.shape[-1]) >= z_dim:
        published_z = published[:, :z_dim]
        metrics["published_z_vs_m3/z_cosine"] = _cosine_mean(published_z, z_m3)
        metrics["published_z_vs_m3/z_mse"] = _mse_mean(published_z, z_m3)
        metrics["published_z_vs_target/z_cosine"] = _cosine_mean(published_z, z_target)
        metrics["published_z_vs_target/z_mse"] = _mse_mean(published_z, z_target)
    return metrics


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: Any,
) -> None:
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(agent_cfg.seed)

    if args_cli.motion_name is not None:
        if not hasattr(env_cfg, "motions"):
            raise TypeError(f"Task {args_cli.task} does not support --motion_name.")
        env_cfg.motions = [str(args_cli.motion_name)]
    if args_cli.trajectory_name is not None:
        if not hasattr(env_cfg, "trajectories"):
            raise TypeError(f"Task {args_cli.task} does not support --trajectory_name.")
        env_cfg.trajectories = [str(args_cli.trajectory_name)]
    if not args_cli.allow_random_reset:
        for name, value in (
            ("random_reset_step_min", 0),
            ("random_reset_step_max", 0),
            ("random_reset_full_trajectory", False),
        ):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, value)
    terminations = getattr(env_cfg, "terminations", None)
    if not args_cli.keep_time_out:
        if terminations is not None and hasattr(terminations, "time_out"):
            terminations.time_out = None
    if not args_cli.keep_early_terminations:
        if terminations is not None:
            _disable_non_reference_terminations(terminations)

    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    planner_checkpoint_path = Path(args_cli.planner_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Low-level checkpoint not found: {checkpoint_path}")
    if not planner_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SkillCommander checkpoint not found: {planner_checkpoint_path}"
        )

    log_dir = _run_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    summary_path = log_dir / "summary.json"
    env_cfg.log_dir = str(log_dir)
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)

    config_payload = {
        "task": args_cli.task,
        "algorithm": args_cli.algorithm,
        "num_envs": int(env_cfg.scene.num_envs),
        "seed": int(agent_cfg.seed),
        "low_level_checkpoint": str(checkpoint_path),
        "planner_checkpoint": str(planner_checkpoint_path),
        "skill_checkpoint_override": args_cli.skill_checkpoint,
        "language_embeddings_override": args_cli.language_embeddings,
        "motion_name": args_cli.motion_name,
        "trajectory_name": args_cli.trajectory_name,
        "allow_random_reset": bool(args_cli.allow_random_reset),
        "keep_time_out": bool(args_cli.keep_time_out),
        "keep_early_terminations": bool(args_cli.keep_early_terminations),
        "continue_after_reset": bool(args_cli.continue_after_reset),
        "save_rollout_training_samples": bool(args_cli.save_rollout_training_samples),
        "command": " ".join(sys.orig_argv),
    }
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True), encoding="utf-8"
    )
    print(f"[INFO] Logging closed-loop SkillCommander eval to: {log_dir}")

    render_mode = "rgb_array" if args_cli.video else None
    raw_gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(raw_gym_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    raw_isaac_env = raw_gym_env.unwrapped
    auto_steps = _auto_reference_steps(raw_isaac_env)
    max_steps = int(args_cli.max_steps) if int(args_cli.max_steps) > 0 else auto_steps
    max_steps = max(1, max_steps)
    video_length = (
        int(args_cli.video_length) if int(args_cli.video_length) > 0 else max_steps
    )
    video_length = max(1, video_length)

    gym_env: Any = raw_gym_env
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(log_dir / "videos" / "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during closed-loop eval.")
        print_dict(video_kwargs, nesting=4)
        gym_env = gym.wrappers.RecordVideo(gym_env, **video_kwargs)

    try:
        wrapped_env = IsaacLabWrapper(gym_env)
        wrapped_env = wrapped_env.set_info_dict_reader(
            IsaacLabTerminalObsReader(
                observation_spec=wrapped_env.observation_spec, backend="gymnasium"
            )
        )
        env = TransformedEnv(
            base_env=wrapped_env,
            transform=Compose(
                RewardSum(),
                StepCounter(max_steps + 1),
                RewardClipping(-10.0, 5.0),
            ),
        )
        if not isinstance(raw_isaac_env, ImitationRLEnv):
            raise TypeError("Expected the unwrapped gym env to be an ImitationRLEnv.")
        base_env = raw_isaac_env
        tracked_body_names = _resolve_existing_body_names(
            base_env, list(G1_TRACKED_BODY_NAMES)
        )
        ee_body_names = _resolve_existing_body_names(
            base_env,
            list(getattr(env_cfg, "command_ee_body_names", G1_EE_BODY_NAMES)),
        )

        planner_checkpoint = torch.load(
            planner_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        trainer_config = _trainer_config_from_checkpoint(planner_checkpoint)
        trainer = SkillCommanderTrainer(config=trainer_config, env=wrapped_env)
        trainer.generator.load_state_dict(planner_checkpoint["generator_state_dict"])
        trainer.update = int(planner_checkpoint.get("update", 0))
        trainer.generator.eval()

        agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
        agent = agent_class(env=env, config=agent_cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
    print(f"[INFO] Loading low-level checkpoint: {checkpoint_path}")
    agent.load_model(str(checkpoint_path))
    collector_policy = agent.collector_policy
    collector_policy.eval()

    dt = getattr(env, "step_dt", None)
    env_ids = torch.arange(
        int(env_cfg.scene.num_envs),
        device=torch.device(getattr(raw_isaac_env, "device", env_cfg.sim.device)),
        dtype=torch.long,
    )
    td = env.reset()
    start_metadata = _trajectory_metadata(raw_isaac_env)
    language_mode = (
        "motion-name embedding" if bool(trainer.condition_on_language) else "none"
    )
    print(
        "[INFO] Conditioning: "
        f"language={language_mode} trajectories={start_metadata['motion_names']}"
    )
    print(f"[INFO] Rollout steps: {max_steps} (auto_reference_steps={auto_steps})")

    num_envs = int(env_cfg.scene.num_envs)
    active = torch.ones(num_envs, dtype=torch.bool)
    survival_steps = torch.zeros(num_envs, dtype=torch.float32)
    return_sum = torch.zeros(num_envs, dtype=torch.float32)
    done_events = torch.zeros(num_envs, dtype=torch.float32)
    rollout_metric_stats: dict[str, list[Tensor]] = {}
    previous_action: Tensor | None = None
    valid_transition_count = 0
    rows: list[dict[str, Any]] = []
    samples_dir = log_dir / "rollout_training_samples"
    saved_sample_files = 0
    saved_sample_rows = 0
    timestep = 0
    stop_reason = "max_steps"
    if int(args_cli.metric_interval) <= 0:
        raise ValueError("--metric_interval must be > 0.")
    while simulation_app.is_running() and timestep < max_steps:
        start_time = time.time()
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            step_active = active.clone()
            should_measure = timestep % int(args_cli.metric_interval) == 0
            metric_row: dict[str, Any] = {}
            td = collector_policy(td)
            action = td.get("action", None)
            if isinstance(action, Tensor):
                action_2d = action.detach().reshape(num_envs, -1).cpu()
                _accumulate_metric(
                    rollout_metric_stats,
                    "action_l2",
                    torch.linalg.vector_norm(action_2d, dim=-1),
                    step_active,
                )
                if previous_action is not None:
                    action_delta_l2 = torch.linalg.vector_norm(
                        action_2d - previous_action, dim=-1
                    )
                    _accumulate_metric(
                        rollout_metric_stats,
                        "action_delta_l2",
                        action_delta_l2,
                        step_active,
                    )
                previous_action = action_2d
            if should_measure:
                # Measure after policy injection so published_z_* reflects the
                # command actually sent to System 0 on this step, while the env
                # state is still the pre-step state used to form the command.
                metric_row.update(
                    _measure_commander(
                        trainer=trainer,
                        wrapped_env=wrapped_env,
                        env_ids=env_ids,
                        sample_path=(
                            samples_dir / f"sample_step_{timestep:06d}.pt"
                            if args_cli.save_rollout_training_samples
                            else None
                        ),
                        sample_step=timestep,
                    )
                )
                row = {
                    "step": int(timestep),
                    **_trajectory_metadata(raw_isaac_env),
                    **metric_row,
                }
                _write_jsonl(metrics_path, row)
                rows.append(row)
                if args_cli.save_rollout_training_samples:
                    saved_sample_files += 1
                    saved_sample_rows += int(env_ids.numel())
            stepped_td = env.step(td)
            rewards = _optional_flat_tensor(
                stepped_td, ("next", "reward"), num_envs=num_envs, default=0.0
            )
            dones = _optional_flat_tensor(
                stepped_td, ("next", "done"), num_envs=num_envs, default=False
            ).bool()
            terminateds = _optional_flat_tensor(
                stepped_td,
                ("next", "terminated"),
                num_envs=num_envs,
                default=False,
            ).bool()
            truncateds = _optional_flat_tensor(
                stepped_td,
                ("next", "truncated"),
                num_envs=num_envs,
                default=False,
            ).bool()
            done_any = dones | terminateds | truncateds
            return_sum += rewards.float() * step_active.float()
            survival_steps += step_active.float()
            done_events += (done_any & step_active).float()
            metric_mask = (
                step_active
                if args_cli.continue_after_reset
                else step_active & ~done_any
            )
            valid_transition_count += int(metric_mask.sum().item())
            for metric_name, values in _tracking_metrics(
                base_env,
                tracked_body_names=tracked_body_names,
                ee_body_names=ee_body_names,
            ).items():
                _accumulate_metric(
                    rollout_metric_stats, metric_name, values.cpu(), metric_mask
                )
            if not args_cli.continue_after_reset:
                active &= ~done_any
            done = _any_done(stepped_td)
            td = step_mdp(
                stepped_td,
                exclude_reward=True,
                exclude_done=False,
                exclude_action=True,
            )

        timestep += 1
        if done and not args_cli.continue_after_reset:
            stop_reason = "env_done"
            print(f"[INFO] Stopping at step {timestep}: env emitted done.")
            break
        if args_cli.real_time and dt is not None:
            sleep_time = float(dt) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    if not simulation_app.is_running():
        stop_reason = "simulation_app_stopped"

    final_metadata = _trajectory_metadata(raw_isaac_env)
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
        "valid_transition_count": int(valid_transition_count),
    }
    metric_means = _mean_dict(rows)
    rollout_metrics = _finalize_metric_stats(rollout_metric_stats)
    if "m3/z_mse" in metric_means:
        rollout_metrics["planner_target_rmse"] = {
            "mean": float(max(metric_means["m3/z_mse"], 0.0) ** 0.5),
            "std": 0.0,
            "count": int(len(rows)),
        }
    planner_metadata = _skill_commander_planner_metadata(
        planner_checkpoint,
        generator=trainer.generator,
        trainer_config=trainer_config,
    )
    summary = {
        **config_payload,
        "metadata": {
            "label": args_cli.label,
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "checkpoint": str(checkpoint_path),
            "planner_checkpoint": str(planner_checkpoint_path),
            "interface": "latent_skill",
            "planner_target_dim": int(trainer.z_dim),
            "planner_metadata": planner_metadata,
            "num_envs": int(num_envs),
            "seed": int(agent_cfg.seed),
            "motion_name": args_cli.motion_name,
            "trajectory_name": args_cli.trajectory_name,
        },
        "aggregate": aggregate,
        "metrics": rollout_metrics,
        "output_dir": str(log_dir),
        "video_dir": str(log_dir / "videos" / "play") if args_cli.video else None,
        "planner_config": trainer_config.to_dict(),
        "planner_update": int(trainer.update),
        "planner_target_dim": int(trainer.z_dim),
        "auto_reference_steps": int(auto_steps),
        "max_steps": int(max_steps),
        "steps_run": int(timestep),
        "stop_reason": stop_reason,
        "metric_interval": int(args_cli.metric_interval),
        "start_trajectories": start_metadata,
        "final_trajectories": final_metadata,
        "metric_means": metric_means,
        "num_metric_rows": len(rows),
        "saved_rows": int(saved_sample_rows),
        "saved_steps": int(saved_sample_files),
        "sample_file_count": int(saved_sample_files),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
