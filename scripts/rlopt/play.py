# Feiyang Wu (feiyangwu@gatech.edu)
# ruff: noqa: E402

"""Script to play a checkpoint of an RLOpt agent (PPO, IPMD, ASE, etc.)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RLOpt agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD",
    choices=["PPO", "SAC", "FASTSAC", "IPMD", "IPMD_SR", "IPMD_BILINEAR", "GAIL", "AMP", "ASE"],
    help="RLOpt algorithm (must match the checkpoint).",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
parser.add_argument("--output_dir", type=str, default=None, help="Optional log/video output directory for this play run.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp
from tensordict.nn import InteractionType

import isaaclab_tasks  # noqa: F401
import isaaclab_imitation.tasks  # noqa: F401

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


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
):
    """Play with an RLOpt agent."""
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # validate checkpoint
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required for play.")
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args_cli.output_dir is None:
        log_dir = os.path.dirname(checkpoint_path)
    else:
        log_dir = os.path.abspath(args_cli.output_dir)
        os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported for RLOpt play.")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = IsaacLabWrapper(env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(
            RewardSum(),
            StepCounter(1000),
            RewardClipping(-10.0, 5.0),
        ),
    )

    # create agent and load checkpoint
    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(checkpoint_path)

    # switch to eval / inference mode
    collector_policy = agent.collector_policy
    collector_policy.eval()

    dt = getattr(env, "step_dt", None)
    if dt is None:
        # fall back: try to get it from the unwrapped env
        base = env
        while hasattr(base, "env") or hasattr(base, "base_env"):
            base = getattr(base, "env", None) or getattr(base, "base_env", None)
            if base is None:
                break
            if hasattr(base, "step_dt"):
                dt = base.step_dt
                break

    # reset environment
    td = env.reset()
    timestep = 0
    print("[INFO] Starting inference loop. Press Ctrl+C to stop.")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode(), set_exploration_type(InteractionType.DETERMINISTIC):
            td = collector_policy(td)
            td = env.step(td)
            td = step_mdp(td, exclude_reward=True, exclude_done=False, exclude_action=True)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        if args_cli.real_time and dt is not None:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
