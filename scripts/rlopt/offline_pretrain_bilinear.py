# Feiyang Wu (feiyangwu@gatech.edu)
# ruff: noqa: E402

"""Run IPMD_BILINEAR offline pretraining only, then evaluate the BC policy.

This is intentionally separate from ``train.py`` because the normal training
loop always enters online collection after offline pretraining.  This script
stops after the offline bilinear/BC phase, saves a checkpoint, and optionally
rolls the resulting policy out for a quick local sanity check.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Run offline IPMD_BILINEAR pretraining only and evaluate the BC policy."
)
parser.add_argument("--video", action="store_true", default=False, help="Record an eval video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded eval video.")
parser.add_argument("--video_width", type=int, default=None, help="Optional render width override.")
parser.add_argument("--video_height", type=int, default=None, help="Optional render height override.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint to initialize before offline pretraining.",
)
parser.add_argument(
    "--output_checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint path. Defaults to <log_dir>/models/model_step_0.pt.",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=400,
    help="Deterministic rollout steps after offline pretraining. Set 0 to skip.",
)
parser.add_argument(
    "--disable_terminations",
    action="store_true",
    default=False,
    help="Disable environment termination terms during eval so the rollout runs open-loop to eval_steps.",
)
parser.add_argument(
    "--bc_updates",
    type=int,
    default=0,
    help=(
        "Additional policy-BC updates run in chunks after built-in offline pretrain. "
        "Use this with agent.bilinear.offline_pretrain.policy_bc_updates=0 for convergence runs."
    ),
)
parser.add_argument(
    "--bc_chunk_updates",
    type=int,
    default=2000,
    help="Number of additional BC updates between validation checks.",
)
parser.add_argument(
    "--bc_eval_batches",
    type=int,
    default=8,
    help="Number of held-out expert batches used for BC validation.",
)
parser.add_argument(
    "--bc_eval_batch_size",
    type=int,
    default=2048,
    help="Batch size for held-out BC validation.",
)
parser.add_argument(
    "--bc_patience",
    type=int,
    default=5,
    help="Stop after this many validation checks without deterministic action-MAE improvement.",
)
parser.add_argument(
    "--bc_min_delta",
    type=float,
    default=1.0e-3,
    help="Minimum action-MAE improvement required to reset BC convergence patience.",
)
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD_BILINEAR",
    choices=["IPMD_BILINEAR"],
    help="Only IPMD_BILINEAR is supported by this offline pretrain helper.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
    if not os.environ.get("DISPLAY"):
        args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import numpy as np
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMDBilinear
from rlopt.config_base import TrainerConfig
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

logger = logging.getLogger(__name__)

ENTRY_POINT_ALGORITHM_MAP = {
    "rlopt_ipmd_bilinear_cfg_entry_point": "IPMD_BILINEAR",
}


def resolve_agent_cfg_entry_point(task_name: str | None) -> str:
    """Resolve the task-provided RLOpt bilinear config entry point."""
    if task_name is None:
        return "rlopt_ipmd_bilinear_cfg_entry_point"
    task_id = task_name.split(":")[-1]
    entry_point = "rlopt_ipmd_bilinear_cfg_entry_point"
    try:
        spec = gym.spec(task_id)
    except Exception as exc:
        msg = f"Could not resolve task '{task_id}' from registry."
        raise ValueError(msg) from exc

    if spec.kwargs.get(entry_point) is not None:
        print(f"[INFO] Using agent config entry point: {entry_point}")
        return entry_point

    supported_algorithms = sorted(
        ENTRY_POINT_ALGORITHM_MAP[key]
        for key in ENTRY_POINT_ALGORITHM_MAP
        if spec.kwargs.get(key) is not None
    )
    msg = (
        "Unsupported task/algo combination: "
        f"task '{task_id}' does not expose IPMD_BILINEAR. "
        f"Supported here: {supported_algorithms}."
    )
    raise ValueError(msg)


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task)


def _coerce_bool_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.reshape(value.shape[0], -1)
    return value.any(dim=-1).to(torch.bool)


def _get_done(td) -> torch.Tensor | None:
    for key in (("next", "done"), ("next", "terminated"), ("next", "truncated")):
        try:
            value = td.get(key)
        except Exception:
            value = None
        if isinstance(value, torch.Tensor):
            done = _coerce_bool_tensor(value)
            if done.any() or key == ("next", "done"):
                return done
    return None


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(arr.max()),
    }


def _find_raw_isaac_env(env: object) -> object | None:
    stack: list[object] = [env]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if hasattr(current, "reward_manager") and hasattr(current, "termination_manager"):
            return current
        for attr_name in ("base_env", "env", "_env", "unwrapped"):
            try:
                child = getattr(current, attr_name, None)
            except Exception:
                continue
            if child is not None:
                stack.append(child)
    return None


def _reward_term_contrib(raw_env: object, *, num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    reward_manager = getattr(raw_env, "reward_manager", None)
    if reward_manager is None:
        return {}
    step_reward = getattr(reward_manager, "_step_reward", None)
    active_terms = getattr(reward_manager, "active_terms", None)
    step_dt = float(getattr(raw_env, "step_dt", 1.0))
    if not isinstance(step_reward, torch.Tensor) or active_terms is None:
        return {}
    contrib = step_reward.detach().to(device=device) * step_dt
    if contrib.ndim != 2 or contrib.shape[0] != num_envs:
        return {}
    return {
        str(name): contrib[:, idx].clone()
        for idx, name in enumerate(list(active_terms))
        if idx < contrib.shape[1]
    }


def _termination_term_flags(raw_env: object, *, num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    termination_manager = getattr(raw_env, "termination_manager", None)
    if termination_manager is None:
        return {}
    active_terms = getattr(termination_manager, "active_terms", None)
    term_dones = getattr(termination_manager, "_last_episode_dones", None)
    if not isinstance(term_dones, torch.Tensor) or active_terms is None:
        return {}
    flags = term_dones.detach().to(device=device)
    if flags.ndim != 2 or flags.shape[0] != num_envs:
        return {}
    return {
        str(name): flags[:, idx].to(torch.bool).clone()
        for idx, name in enumerate(list(active_terms))
        if idx < flags.shape[1]
    }


def _build_expert_obs_td(agent: IPMDBilinear, expert_batch, *, train_latent: bool = False):
    expert_batch = expert_batch.to(agent.device)
    expert_action = cast(torch.Tensor, expert_batch.get("expert_action"))
    expert_obs_td = expert_batch.clone(False)
    if getattr(agent, "_use_latent_command", False) and getattr(agent, "_latent_key", None) not in expert_obs_td.keys(True):
        latent = agent._expert_latents_from_td(
            expert_batch,
            detach=not train_latent,
        ).reshape(*expert_batch.batch_size, agent._latent_dim)
        expert_obs_td.set(agent._latent_key, latent)
    return expert_obs_td.select(*agent._policy_operator.in_keys), expert_action


def _dist_mean_scale(dist: object) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mean = getattr(dist, "mean", None)
    scale = getattr(dist, "stddev", None)
    base_dist = getattr(dist, "base_dist", None)
    if not isinstance(mean, torch.Tensor) and base_dist is not None:
        mean = getattr(base_dist, "loc", None)
    if not isinstance(scale, torch.Tensor) and base_dist is not None:
        scale = getattr(base_dist, "scale", None)
    return (
        mean if isinstance(mean, torch.Tensor) else None,
        scale if isinstance(scale, torch.Tensor) else None,
    )


@torch.no_grad()
def _evaluate_bc_fit(
    agent: IPMDBilinear,
    *,
    batches: int,
    batch_size: int,
) -> dict[str, float]:
    required_keys = agent._offline_policy_bc_required_keys()
    nll_values: list[float] = []
    mae_values: list[float] = []
    rmse_values: list[float] = []
    max_abs_values: list[float] = []
    scale_values: list[float] = []
    scale_min_values: list[float] = []
    scale_max_values: list[float] = []
    agent.actor_critic.eval()
    for _ in range(max(1, int(batches))):
        expert_batch = agent._next_offline_expert_batch(
            batch_size=max(1, int(batch_size)),
            required_keys=required_keys,
        )
        obs_td, expert_action = _build_expert_obs_td(agent, expert_batch, train_latent=False)
        dist = agent._policy_operator.get_dist(obs_td)
        log_prob = dist.log_prob(expert_action)
        log_prob = agent._reduce_log_prob(log_prob, expert_action)
        mean, scale = _dist_mean_scale(dist)
        nll_values.append(float((-log_prob.mean()).item()))
        if mean is not None:
            diff = mean - expert_action
            mae_values.append(float(diff.abs().mean().item()))
            rmse_values.append(float(diff.pow(2).mean().sqrt().item()))
            max_abs_values.append(float(diff.abs().max().item()))
        if scale is not None:
            scale_values.append(float(scale.mean().item()))
            scale_min_values.append(float(scale.min().item()))
            scale_max_values.append(float(scale.max().item()))
    agent.actor_critic.train()
    metrics = {
        "bc_val_nll": float(np.mean(nll_values)),
    }
    if mae_values:
        metrics.update(
            {
                "bc_val_action_mae": float(np.mean(mae_values)),
                "bc_val_action_rmse": float(np.mean(rmse_values)),
                "bc_val_action_max_abs": float(np.max(max_abs_values)),
            }
        )
    if scale_values:
        metrics.update(
            {
                "bc_val_scale_mean": float(np.mean(scale_values)),
                "bc_val_scale_min": float(np.min(scale_min_values)),
                "bc_val_scale_max": float(np.max(scale_max_values)),
            }
        )
    return metrics


def _best_checkpoint_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_best{path.suffix or '.pt'}")


def _run_chunked_bc_until_converged(
    agent: IPMDBilinear,
    *,
    output_checkpoint: Path,
    max_updates: int,
    chunk_updates: int,
    eval_batches: int,
    eval_batch_size: int,
    patience: int,
    min_delta: float,
) -> dict[str, Any]:
    if max_updates <= 0:
        return {"enabled": False}

    offline_cfg = agent.config.bilinear.offline_pretrain
    required_keys = agent._offline_policy_bc_required_keys()
    batch_size = int(offline_cfg.policy_bc_batch_size)
    chunk_updates = max(1, int(chunk_updates))
    best_mae = float("inf")
    best_update = 0
    stale_checks = 0
    total_updates = 0
    history: list[dict[str, float | int]] = []
    best_path = _best_checkpoint_path(output_checkpoint)
    start_time = time.time()
    agent.actor_critic.train()

    while total_updates < int(max_updates):
        updates_this_chunk = min(chunk_updates, int(max_updates) - total_updates)
        last_train_metrics: dict[str, float] = {}
        for _ in range(updates_this_chunk):
            expert_batch = agent._next_offline_expert_batch(
                batch_size=batch_size,
                required_keys=required_keys,
            )
            last_train_metrics = agent._offline_policy_bc_step(expert_batch)
        total_updates += updates_this_chunk

        val_metrics = _evaluate_bc_fit(
            agent,
            batches=eval_batches,
            batch_size=eval_batch_size,
        )
        current_mae = float(val_metrics.get("bc_val_action_mae", float("inf")))
        improved = current_mae < best_mae - float(min_delta)
        if improved:
            best_mae = current_mae
            best_update = total_updates
            stale_checks = 0
            best_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save_model(best_path)
        else:
            stale_checks += 1

        row: dict[str, float | int] = {
            "updates": total_updates,
            "elapsed_seconds": float(time.time() - start_time),
            "stale_checks": stale_checks,
            **{f"train_{k}": float(v) for k, v in last_train_metrics.items()},
            **{k: float(v) for k, v in val_metrics.items()},
        }
        history.append(row)
        print("[BC_CONVERGENCE] " + json.dumps(row, sort_keys=True))

        if stale_checks >= max(1, int(patience)):
            break

    return {
        "enabled": True,
        "requested_updates": int(max_updates),
        "completed_updates": int(total_updates),
        "chunk_updates": int(chunk_updates),
        "best_update": int(best_update),
        "best_action_mae": float(best_mae),
        "best_checkpoint": str(best_path.resolve()) if best_path.exists() else None,
        "stopped_by_patience": bool(total_updates < int(max_updates)),
        "history": history,
    }


def _disable_env_terminations(env_cfg: object) -> list[str]:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return []
    disabled: list[str] = []
    for name, value in list(vars(terminations).items()):
        if name.startswith("_") or value is None:
            continue
        setattr(terminations, name, None)
        disabled.append(str(name))
    return disabled


def _evaluate_policy(env, agent: IPMDBilinear, *, steps: int) -> dict[str, object]:
    if steps <= 0:
        return {"eval_steps": 0, "completed_episodes": 0}

    collector_policy = agent.collector_policy
    collector_policy.eval()
    td = env.reset()
    num_envs = int(td.batch_size[0]) if len(td.batch_size) > 0 else 1
    raw_env = _find_raw_isaac_env(env)
    returns = torch.zeros(num_envs, device=agent.device)
    lengths = torch.zeros(num_envs, device=agent.device)
    reward_term_returns: dict[str, torch.Tensor] = {}
    completed_reward_terms: dict[str, list[float]] = {}
    termination_counts: dict[str, int] = {}
    completed_returns: list[float] = []
    completed_lengths: list[float] = []

    for _ in range(int(steps)):
        with torch.inference_mode(), set_exploration_type(InteractionType.DETERMINISTIC):
            td = collector_policy(td)
            td = env.step(td)

        reward = td.get(("next", "reward"))
        if isinstance(reward, torch.Tensor):
            returns += reward.reshape(num_envs, -1).sum(dim=-1).to(agent.device)
        if raw_env is not None:
            for name, contrib in _reward_term_contrib(raw_env, num_envs=num_envs, device=agent.device).items():
                if name not in reward_term_returns:
                    reward_term_returns[name] = torch.zeros(num_envs, device=agent.device)
                reward_term_returns[name] += contrib
        lengths += 1.0

        done = _get_done(td)
        if done is not None:
            done = done.to(agent.device)
            if done.any():
                if raw_env is not None:
                    for name, flags in _termination_term_flags(raw_env, num_envs=num_envs, device=agent.device).items():
                        termination_counts[name] = termination_counts.get(name, 0) + int(flags[done].sum().item())
                completed_returns.extend(returns[done].detach().cpu().tolist())
                completed_lengths.extend(lengths[done].detach().cpu().tolist())
                for name, term_returns in reward_term_returns.items():
                    completed_reward_terms.setdefault(name, []).extend(
                        term_returns[done].detach().cpu().tolist()
                    )
                    term_returns[done] = 0.0
                returns[done] = 0.0
                lengths[done] = 0.0

        td = step_mdp(td, exclude_reward=True, exclude_done=False, exclude_action=True)

    summary: dict[str, object] = {
        "eval_steps": int(steps),
        "completed_episodes": len(completed_returns),
        "partial_envs": num_envs,
        "partial_return_mean": float(returns.mean().item()),
        "partial_length_mean": float(lengths.mean().item()),
    }
    if completed_returns:
        summary.update(
            {
                "completed_return": _summary_stats(completed_returns),
                "completed_length": _summary_stats(completed_lengths),
                "completed_return_mean": _summary_stats(completed_returns)["mean"],
                "completed_length_mean": _summary_stats(completed_lengths)["mean"],
            }
        )
    if completed_reward_terms:
        summary["completed_reward_terms"] = {
            name: _summary_stats(values)
            for name, values in sorted(completed_reward_terms.items())
        }
    if reward_term_returns:
        summary["partial_reward_terms"] = {
            name: _summary_stats(term_returns.detach().cpu().tolist())
            for name, term_returns in sorted(reward_term_returns.items())
        }
    if termination_counts:
        total_terms = max(1, sum(termination_counts.values()))
        summary["termination_counts"] = dict(sorted(termination_counts.items()))
        summary["termination_fractions"] = {
            name: count / total_terms
            for name, count in sorted(termination_counts.items())
        }
    return summary


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
):
    """Run offline pretraining and deterministic evaluation."""
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if agent_cfg.trainer is None:
        agent_cfg.trainer = TrainerConfig()
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    disabled_terminations: list[str] = []
    if args_cli.disable_terminations:
        disabled_terminations = _disable_env_terminations(env_cfg)
        print(f"[INFO] Disabled termination terms for open-loop eval: {disabled_terminations}")

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(
        os.path.join("logs", "rlopt", args_cli.algorithm.lower(), args_cli.task)
    )
    log_dir = os.path.join(log_root_path, run_info)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    agent_cfg.logger.log_dir = log_dir
    Path(log_dir, "command.txt").write_text(" ".join(sys.orig_argv))
    env_cfg.log_dir = log_dir

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported for RLOpt pretraining.")

    video_folder = None
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "eval")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during BC-policy evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    isaac_wrapper = IsaacLabWrapper(env)
    isaac_wrapper.log_infos = deque(maxlen=max(1, int(args_cli.eval_steps) + 1024))
    env = isaac_wrapper.set_info_dict_reader(
        IsaacLabTerminalObsReader(observation_spec=isaac_wrapper.observation_spec, backend="gymnasium")
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(1000), RewardClipping(-10.0, 5.0)),
    )

    agent = IPMDBilinear(env=env, config=agent_cfg)
    if args_cli.checkpoint is not None:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint before offline pretraining: {checkpoint_path}")
        agent.load_model(checkpoint_path)

    try:
        start_time = time.time()
        agent.validate_training()
        agent._offline_pretrain_spectral_representation()

        output_checkpoint = (
            Path(args_cli.output_checkpoint).expanduser()
            if args_cli.output_checkpoint is not None
            else Path(log_dir) / agent_cfg.logger.save_path / "model_step_0.pt"
        )
        bc_convergence_summary = _run_chunked_bc_until_converged(
            agent,
            output_checkpoint=output_checkpoint,
            max_updates=int(args_cli.bc_updates),
            chunk_updates=int(args_cli.bc_chunk_updates),
            eval_batches=int(args_cli.bc_eval_batches),
            eval_batch_size=int(args_cli.bc_eval_batch_size),
            patience=int(args_cli.bc_patience),
            min_delta=float(args_cli.bc_min_delta),
        )
        best_checkpoint = bc_convergence_summary.get("best_checkpoint")
        if isinstance(best_checkpoint, str) and Path(best_checkpoint).is_file():
            print(f"[INFO] Loading best BC checkpoint for final eval/save: {best_checkpoint}")
            agent.load_model(best_checkpoint)
        pretrain_seconds = time.time() - start_time

        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        agent.save_model(output_checkpoint)
        print(f"[INFO] Saved offline-pretrained checkpoint: {output_checkpoint.resolve()}")

        eval_summary = _evaluate_policy(env, agent, steps=int(args_cli.eval_steps))
        eval_summary["pretrain_seconds"] = float(pretrain_seconds)
        eval_summary["bc_convergence"] = bc_convergence_summary
        eval_summary["checkpoint"] = str(output_checkpoint.resolve())
        if disabled_terminations:
            eval_summary["disabled_terminations"] = disabled_terminations
        if video_folder is not None:
            videos = sorted(str(p.resolve()) for p in Path(video_folder).glob("*.mp4"))
            eval_summary["videos"] = videos

        summary_path = Path(log_dir) / "offline_pretrain_eval_summary.json"
        summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True))
        print("[INFO] Offline pretrain/eval summary:")
        print(json.dumps(eval_summary, indent=2, sort_keys=True))
        print(f"[INFO] Wrote summary: {summary_path.resolve()}")
    finally:
        shutdown = getattr(getattr(agent, "collector", None), "shutdown", None)
        if callable(shutdown):
            shutdown()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
