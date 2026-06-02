# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""
# ruff: noqa: E402

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import statistics
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--perf_report",
    action="store_true",
    default=False,
    help="Record per-iteration performance metrics and print a summary at the end.",
)
parser.add_argument(
    "--perf_warmup_iterations",
    type=int,
    default=1,
    help="Number of initial logged iterations to exclude from the final perf summary.",
)
parser.add_argument(
    "--perf_json",
    type=str,
    default=None,
    help="Optional path to write the final structured performance summary as JSON.",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--ray-proc-id",
    "-rid",
    type=int,
    default=None,
    help="Automatically configured by Ray integration, otherwise None.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = [
        "pixi",
        "add",
        "--feature",
        "isaaclab",
        "--pypi",
        f"rsl-rl-lib=={RSL_RL_VERSION}",
    ]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# import logger
logger = logging.getLogger(__name__)

import isaaclab_imitation.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _install_perf_report_hook(runner, enabled: bool) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []
    if not enabled:
        return metrics

    original_log = runner.log

    def wrapped_log(locs: dict, *args, **kwargs):
        collection_time = float(locs["collection_time"])
        learn_time = float(locs["learn_time"])
        iteration_time = collection_time + learn_time
        collection_size = (
            runner.num_steps_per_env
            * runner.env.num_envs
            * getattr(runner, "gpu_world_size", 1)
        )
        fps = (
            float(collection_size / iteration_time)
            if iteration_time > 0.0
            else float("inf")
        )
        metrics.append(
            {
                "iteration": int(locs["it"]),
                "collection_size": int(collection_size),
                "fps": fps,
                "collection_time_s": collection_time,
                "learn_time_s": learn_time,
                "iteration_time_s": iteration_time,
            }
        )
        return original_log(locs, *args, **kwargs)

    runner.log = wrapped_log
    return metrics


def _build_perf_summary(
    metrics: list[dict[str, float | int]], warmup_iterations: int
) -> dict[str, object] | None:
    if len(metrics) == 0:
        return None

    warmup_iterations = max(int(warmup_iterations), 0)
    summary_metrics = (
        metrics[warmup_iterations:] if len(metrics) > warmup_iterations else metrics
    )
    fps_values = [float(metric["fps"]) for metric in summary_metrics]
    collection_values = [
        float(metric["collection_time_s"]) for metric in summary_metrics
    ]
    learn_values = [float(metric["learn_time_s"]) for metric in summary_metrics]
    iteration_values = [float(metric["iteration_time_s"]) for metric in summary_metrics]

    return {
        "logged_iterations": len(metrics),
        "warmup_iterations_skipped": min(warmup_iterations, len(metrics)),
        "used_iterations": len(summary_metrics),
        "collection_size": int(summary_metrics[0]["collection_size"]),
        "fps_mean": statistics.fmean(fps_values),
        "fps_median": statistics.median(fps_values),
        "fps_min": min(fps_values),
        "fps_max": max(fps_values),
        "collection_time_mean_s": statistics.fmean(collection_values),
        "learn_time_mean_s": statistics.fmean(learn_values),
        "iteration_time_mean_s": statistics.fmean(iteration_values),
        "per_iteration": metrics,
    }


def _print_perf_summary(summary: dict[str, object]) -> None:
    print("[PERF] Summary")
    print(
        "[PERF] "
        f"iterations={summary['used_iterations']} "
        f"(logged={summary['logged_iterations']}, warmup_skipped={summary['warmup_iterations_skipped']})"
    )
    print(
        "[PERF] "
        f"fps_mean={summary['fps_mean']:.2f} "
        f"fps_median={summary['fps_median']:.2f} "
        f"fps_min={summary['fps_min']:.2f} "
        f"fps_max={summary['fps_max']:.2f}"
    )
    print(
        "[PERF] "
        f"collection_time_mean_s={summary['collection_time_mean_s']:.4f} "
        f"learn_time_mean_s={summary['learn_time_mean_s']:.4f} "
        f"iteration_time_mean_s={summary['iteration_time_mean_s']:.4f}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # check for invalid combination of CPU device with distributed training
    if (
        args_cli.distributed
        and args_cli.device is not None
        and "cpu" in args_cli.device
    ):
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    perf_metrics = _install_perf_report_hook(runner, args_cli.perf_report)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )
    perf_summary = _build_perf_summary(perf_metrics, args_cli.perf_warmup_iterations)
    if perf_summary is not None:
        perf_summary.update(
            {
                "task": args_cli.task,
                "num_envs": int(env_cfg.scene.num_envs),
                "max_iterations": int(agent_cfg.max_iterations),
                "device": str(agent_cfg.device),
                "log_dir": log_dir,
            }
        )
        _print_perf_summary(perf_summary)
        if args_cli.perf_json is not None:
            with open(args_cli.perf_json, "w", encoding="utf-8") as perf_file:
                json.dump(perf_summary, perf_file, indent=2)
            print(f"[PERF] Wrote summary JSON to {args_cli.perf_json}")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
