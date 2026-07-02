#!/usr/bin/env python3
# ruff: noqa: E402
"""Minimal closed-loop SkillCommander rollout.

This is intentionally much smaller than the diagnostic eval scripts: no metric
trainer, no target-z computation, no tracking errors, no rollout sample export,
and no summary files. It only builds the task/policy shell required for tensor
shapes, points the agent at a SkillCommander checkpoint, loads the low-level
policy weights, and steps the deterministic collector policy.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Bare minimal M3 SkillCommander rollout: load planner + policy and step."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Low-level policy checkpoint (.pt).",
)
parser.add_argument(
    "--planner_checkpoint",
    type=str,
    required=True,
    help="SkillCommander checkpoint used by the live policy command source.",
)
parser.add_argument(
    "--language_embeddings",
    type=str,
    default="",
    help="Optional language embedding table override for language-conditioned planners.",
)
parser.add_argument(
    "--allow_language_conditioning",
    action="store_true",
    default=False,
    help=(
        "Allow language-conditioned planners. Disabled by default because language "
        "lookup uses trajectory rank/motion-name metadata."
    ),
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=1000,
    help="Number of simulation steps to run.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
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
    "--real-time",
    action="store_true",
    default=False,
    help="Throttle to env step_dt when available.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
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
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from tensordict import TensorDictBase
from tensordict.nn import InteractionType
from torch import Tensor
from torchrl.envs import Compose, TransformedEnv
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


def _checkpoint_condition_on_language(path: Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if isinstance(config, dict) and "condition_on_language" in config:
        return bool(config["condition_on_language"])
    return bool(checkpoint.get("condition_on_language", True))


def _policy_input_keys(agent: Any) -> list[Any]:
    keys = getattr(agent, "_policy_obs_keys", None)
    return list(keys) if keys is not None else []


def _key_to_str(key: Any) -> str:
    if isinstance(key, tuple | list):
        return "/".join(str(part) for part in key)
    return str(key)


def _forbidden_policy_keys(keys: list[Any]) -> list[str]:
    forbidden: list[str] = []
    for key in keys:
        text = _key_to_str(key).lower()
        if (
            "expert" in text
            or "reference" in text
            or text.startswith("critic/")
            or text.startswith("reward_input/")
        ):
            forbidden.append(_key_to_str(key))
    return forbidden


def _any_done(td: TensorDictBase) -> bool:
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


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: Any,
) -> None:
    if int(args_cli.max_steps) <= 0:
        raise ValueError("--max_steps must be > 0.")

    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    planner_checkpoint_path = Path(args_cli.planner_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Low-level checkpoint not found: {checkpoint_path}")
    if not planner_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SkillCommander checkpoint not found: {planner_checkpoint_path}"
        )

    if (
        _checkpoint_condition_on_language(planner_checkpoint_path)
        and not args_cli.allow_language_conditioning
    ):
        raise ValueError(
            "Planner checkpoint is language-conditioned. Re-run with "
            "--allow_language_conditioning only if trajectory rank/motion-name "
            "goal metadata is intended for this eval."
        )

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.log_dir = str(checkpoint_path.parent)

    ipmd_cfg = getattr(agent_cfg, "ipmd", None)
    if ipmd_cfg is None:
        raise ValueError("Minimal SkillCommander eval requires an IPMD-style config.")
    ipmd_cfg.use_latent_command = True
    ipmd_cfg.command_source = "skill_commander"
    ipmd_cfg.skill_commander_checkpoint_path = str(planner_checkpoint_path)
    ipmd_cfg.skill_commander_use_achieved_state = True
    if str(args_cli.language_embeddings).strip():
        ipmd_cfg.skill_commander_embeddings_path = str(
            Path(args_cli.language_embeddings).expanduser().resolve()
        )

    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs

    torch.manual_seed(int(args_cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args_cli.seed))

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    wrapped_env = IsaacLabWrapper(raw_env)
    wrapped_env = wrapped_env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=wrapped_env.observation_spec,
            backend="gymnasium",
        )
    )
    env = TransformedEnv(base_env=wrapped_env, transform=Compose())

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)
    agent.load_model(str(checkpoint_path))
    policy_keys = _policy_input_keys(agent)
    forbidden = _forbidden_policy_keys(policy_keys)
    if forbidden:
        raise ValueError(
            "Policy input keys include expert/reference fields during minimal M3 "
            f"eval: {forbidden}."
        )
    sampler = getattr(agent, "_hl_skill_command_sampler", None)
    if sampler is None or sampler.__class__.__name__ != "FrozenSkillCommanderSampler":
        raise RuntimeError("Live SkillCommander sampler was not constructed.")
    if not bool(getattr(sampler, "use_achieved_state", False)):
        raise RuntimeError("Live SkillCommander sampler is not using achieved state.")

    collector_policy = agent.collector_policy
    collector_policy.eval()

    td = env.reset()
    dt = getattr(env, "step_dt", None)
    print("[INFO] Minimal SkillCommander rollout")
    print(f"[INFO] policy={checkpoint_path}")
    print(f"[INFO] planner={planner_checkpoint_path}")
    print(f"[INFO] policy_input_keys={[_key_to_str(key) for key in policy_keys]}")

    for step in range(int(args_cli.max_steps)):
        start_time = time.time()
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = collector_policy(td)
            stepped_td = env.step(td)
            done = _any_done(stepped_td)
            td = step_mdp(
                stepped_td,
                exclude_reward=True,
                exclude_done=False,
                exclude_action=True,
            )
        if done:
            print(f"[INFO] env emitted done at step {step + 1}; continuing.")
        if args_cli.real_time and dt is not None:
            sleep_time = float(dt) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    print(f"[INFO] finished {int(args_cli.max_steps)} steps")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
