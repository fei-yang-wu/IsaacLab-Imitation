#!/usr/bin/env python3
# ruff: noqa: E402
"""Collect oracle-drive achieved-state samples for command-interface planners."""

from __future__ import annotations

import argparse
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
parser.add_argument(
    "--interface", choices=("full_body_trajectory", "ee_trajectory"), required=True
)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--motion_manifest", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--state_history_steps", type=int, default=0)
parser.add_argument("--command_past_steps", type=int, default=0)
parser.add_argument("--command_future_steps", type=int, default=25)
parser.add_argument("--reset_schedule", type=str, default="sequential")
parser.add_argument("--reference_start_frame", type=int, default=0)
parser.add_argument("--refresh_zarr_dataset", action="store_true", default=False)
parser.add_argument(
    "--enable_observation_corruption", action="store_true", default=False
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
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_EE_BODY_NAMES,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import (  # noqa: E402
    flatten_command_terms,
    planner_state_from_batch,
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


def _configured_step_dt(env_cfg: object) -> float | None:
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 0.0) or 0.0)
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    if sim_dt > 0.0 and decimation > 0:
        return sim_dt * decimation
    return None


def _resolve_existing_body_names(
    base_env: ImitationRLEnv, requested_names: list[str]
) -> list[str]:
    names: list[str] = []
    for name in requested_names:
        try:
            base_env._get_robot_anchor_body_id_fast(name)
            base_env._get_reference_body_ids_fast((name,))
        except Exception as exc:
            print(f"[WARNING] Skipping unavailable EE target {name!r}: {exc}")
            continue
        names.append(str(name))
    return names


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
    return {
        term_name: base_env.get_current_expert_window_term(
            term_name=term_name,
            past_steps=int(args_cli.command_past_steps),
            future_steps=int(args_cli.command_future_steps),
            **ref_kwargs,
        )
        for term_name in (
            ("expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b")
            if interface == "full_body_trajectory"
            else ("expert_ee_pos_b", "expert_ee_ori_b")
        )
    }


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
    if args_cli.state_history_steps < 0:
        raise ValueError("--state_history_steps must be >= 0.")

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

    agent_cfg.command_space = args_cli.interface
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()
    env_cfg.latent_patch_past_steps = int(args_cli.command_past_steps)
    env_cfg.latent_patch_future_steps = int(args_cli.command_future_steps)
    env_cfg.command_observation_source = "reference"
    _sync_env_window_params(env_cfg)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    if motion_manifest is not None:
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

    output_dir = args_cli.output_dir.expanduser().resolve()
    samples_dir = output_dir / "rollout_training_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output_dir)

    agent_cfg.env.num_envs = int(args_cli.num_envs)
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    if hasattr(agent_cfg, "logger"):
        agent_cfg.logger.backend = ""
        agent_cfg.logger.log_dir = str(output_dir / "agent_logs")
    if hasattr(agent_cfg, "device"):
        agent_cfg.device = env_cfg.sim.device

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")
    env = IsaacLabWrapper(raw_env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(int(args_cli.steps) + 2)),
    )
    base_env = _unwrap_imitation_env(env)
    ee_body_names = _resolve_existing_body_names(
        base_env,
        list(getattr(env_cfg, "command_ee_body_names", G1_EE_BODY_NAMES)),
    )

    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    print(f"[INFO] Loading low-level checkpoint: {checkpoint_path}")
    agent.load_model(str(checkpoint_path))
    policy = agent.collector_policy
    policy.eval()

    metadata: dict[str, Any] | None = None
    td = env.reset()
    saved_steps = 0
    saved_rows = 0
    with torch.inference_mode(), set_exploration_type(InteractionType.DETERMINISTIC):
        for step_idx in range(int(args_cli.steps)):
            expert_batch = base_env.current_expert_macro_transition_batch(
                horizon_steps=int(args_cli.command_future_steps),
                state_history_steps=int(args_cli.state_history_steps),
            )
            achieved_batch = base_env.current_achieved_macro_transition_batch(
                horizon_steps=int(args_cli.command_future_steps),
                state_history_steps=int(args_cli.state_history_steps),
            )
            command_terms = _current_reference_command_terms(
                base_env,
                interface=args_cli.interface,
                ee_body_names=ee_body_names,
            )
            target, target_spec = flatten_command_terms(
                args_cli.interface, command_terms
            )
            if metadata is None:
                metadata = {
                    "interface": args_cli.interface,
                    "target_spec": target_spec.to_dict(),
                    "state_history_steps": int(args_cli.state_history_steps),
                    "command_past_steps": int(args_cli.command_past_steps),
                    "command_future_steps": int(args_cli.command_future_steps),
                    "task": args_cli.task,
                    "algorithm": args_cli.algorithm,
                    "checkpoint": str(checkpoint_path),
                    "motion_manifest": str(motion_manifest)
                    if motion_manifest is not None
                    else None,
                    "num_envs": int(args_cli.num_envs),
                    "seed": int(args_cli.seed),
                }
            sample = {
                "step": int(step_idx),
                "planner_state": planner_state_from_batch(
                    achieved_batch,
                    state_history_steps=int(args_cli.state_history_steps),
                )
                .detach()
                .cpu(),
                "expert_planner_state": planner_state_from_batch(
                    expert_batch, state_history_steps=int(args_cli.state_history_steps)
                )
                .detach()
                .cpu(),
                "target": target.detach().cpu(),
                "traj_rank": expert_batch.get(("hl", "traj_rank")).detach().cpu(),
                "metadata": metadata,
            }
            torch.save(sample, samples_dir / f"sample_step_{step_idx:06d}.pt")
            saved_steps += 1
            saved_rows += int(target.shape[0])

            td = policy(td)
            td_step = env.step(td)
            td = step_mdp(
                td_step, exclude_reward=True, exclude_done=False, exclude_action=True
            )

    summary = {
        "metadata": metadata or {},
        "saved_steps": saved_steps,
        "saved_rows": saved_rows,
        "sample_file_count": saved_steps,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(
        f"[INFO] Wrote {saved_rows} sample rows "
        f"across {saved_steps} files to: {samples_dir}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
