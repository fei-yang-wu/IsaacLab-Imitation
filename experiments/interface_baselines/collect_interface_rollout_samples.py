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
    "--planner_checkpoint",
    type=Path,
    default=None,
    help=(
        "Planner checkpoint for planner-driven token or Future-CVAE rollouts. "
        "Omit it to collect posterior-oracle samples."
    ),
)
parser.add_argument(
    "--interface",
    choices=(
        "single_frame_full_body",
        "full_body_trajectory",
        "ee_trajectory",
        "future_cvae",
        "per_step_token_sequence",
    ),
    required=True,
)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--motion_manifest", type=Path, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Recorded control steps. <=0 records the full requested rollout.",
)
parser.add_argument(
    "--motion_name",
    type=str,
    default="",
    help="Explicitly restrict collection to one named motion.",
)
parser.add_argument(
    "--motion_names",
    nargs="+",
    default=None,
    help="Explicitly restrict collection to the listed named motions.",
)
parser.add_argument(
    "--balanced_rows_per_motion",
    type=int,
    default=0,
    help=(
        "When positive, save exactly this many rows for every balanced motion "
        "and stop once all per-motion budgets are full."
    ),
)
parser.add_argument(
    "--balanced_motion_names",
    nargs="+",
    default=None,
    help=(
        "Motion names covered by --balanced_rows_per_motion. Defaults to "
        "--motion_names, or to every motion loaded by the environment."
    ),
)
parser.add_argument(
    "--language_embeddings",
    type=Path,
    default=None,
    help="Optional rank-aligned language embedding table for shared planners.",
)
parser.add_argument(
    "--dataset_path",
    type=Path,
    default=None,
    help=(
        "Existing trajectory dataset to use. Pass this explicitly when a manifest "
        "is also provided so manifest resolution does not select a temporary cache."
    ),
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--sample_rows_per_file",
    type=int,
    default=1,
    help="Buffer this many planner rows per sample file.",
)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument(
    "--control_steps",
    type=int,
    default=0,
    help=(
        "Exact low-level evaluation length. A positive value overrides "
        "--steps * --planner_interval_steps."
    ),
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--state_history_steps", type=int, default=9)
parser.add_argument(
    "--planner_interval_steps",
    type=int,
    default=10,
    help="Low-level control steps between saved 5 Hz planner decisions.",
)
parser.add_argument("--command_past_steps", type=int, default=0)
parser.add_argument("--command_future_steps", type=int, default=25)
parser.add_argument("--reset_schedule", type=str, default="sequential")
parser.add_argument("--reference_start_frame", type=int, default=0)
parser.add_argument("--refresh_zarr_dataset", action="store_true", default=False)
parser.add_argument(
    "--enable_observation_corruption", action="store_true", default=False
)
parser.add_argument(
    "--evaluation_only",
    action="store_true",
    default=False,
    help="Measure closed-loop behavior without saving planner-training samples.",
)
parser.add_argument(
    "--stop_after_done",
    action="store_true",
    default=False,
    help="Stop measuring each environment after its first done event.",
)
parser.add_argument(
    "--keep_configured_episode_length",
    action="store_true",
    default=False,
    help="Keep the task's configured timeout while collecting across resets.",
)
parser.add_argument(
    "--disable_tracking_terminations",
    action="store_true",
    default=False,
    help=(
        "Use the M3 fall-only termination policy and the 0-200 random reset "
        "range while collecting demonstration rows."
    ),
)
parser.add_argument(
    "--low_level_command_mode",
    choices=("native", "streamed_vanilla"),
    default="native",
    help=(
        "For full-body targets, streamed_vanilla consumes the held chunk through "
        "the unchanged single-frame vanilla tracker."
    ),
)
parser.add_argument(
    "--low_level_action_source",
    choices=("policy", "reconstructed_reference"),
    default="policy",
    help=(
        "Action source used for the environment step. The reconstructed-reference "
        "option is a training-target diagnostic, not a deployable policy result."
    ),
)
parser.add_argument(
    "--tracking_success_root_height_threshold", type=float, default=0.25
)
parser.add_argument("--tracking_success_root_ori_threshold", type=float, default=1.0)
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
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from balanced_motion_rows import BalancedMotionRowSelector  # noqa: E402
from low_level_tracker import load_frozen_low_level_tracker  # noqa: E402
from planner_publish_schedule import planner_renew_env_ids  # noqa: E402

from interface_planner_common import (  # noqa: E402
    InterfaceTargetSpec,
    flatten_command_terms,
    load_rank_language_embeddings,
    planner_state_from_batch,
)
from closed_loop_metrics import (  # noqa: E402
    accumulate_metric,
    finalize_metric_stats,
    optional_flat_tensor,
    resolve_existing_body_names,
    tensor_mean_std,
    tracking_metrics,
)
from planner_sample_schema import (  # noqa: E402
    PlannerSampleWriter,
    add_sample_format_metadata,
    build_planner_sample,
)


def trajectory_metadata(raw_env: Any) -> dict[str, Any]:
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


TRACKING_TERMINATION_NAMES = ("anchor_pos", "anchor_ori", "ee_body_pos")
FALL_TERMINATION_NAME = "base_too_low"


def _disable_tracking_terminations(terminations: Any) -> list[str]:
    disabled: list[str] = []
    for name in TRACKING_TERMINATION_NAMES:
        if hasattr(terminations, name) and getattr(terminations, name) is not None:
            setattr(terminations, name, None)
            disabled.append(name)
    return disabled


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
    sync_derived_fields = getattr(env_cfg, "sync_derived_fields", None)
    if callable(sync_derived_fields):
        sync_derived_fields()
        return
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


def _current_demonstration_command_terms(
    base_env: ImitationRLEnv,
    *,
    interface: str,
    ee_body_names: list[str],
) -> dict[str, torch.Tensor]:
    ref_kwargs = _command_reference_kwargs(interface, ee_body_names=ee_body_names)
    return base_env.current_offline_demo_command_terms(
        past_steps=int(args_cli.command_past_steps),
        future_steps=int(args_cli.command_future_steps),
        **ref_kwargs,
    )


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
    if args_cli.sample_rows_per_file <= 0:
        raise ValueError("--sample_rows_per_file must be positive.")
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args_cli.control_steps < 0:
        raise ValueError("--control_steps must be >= 0.")
    if args_cli.state_history_steps < 0:
        raise ValueError("--state_history_steps must be >= 0.")
    if args_cli.planner_interval_steps <= 0:
        raise ValueError("--planner_interval_steps must be > 0.")
    selected_motion_name = str(args_cli.motion_name).strip()
    selected_motion_names = (
        [str(name).strip() for name in args_cli.motion_names]
        if args_cli.motion_names is not None
        else None
    )
    if selected_motion_name and selected_motion_names is not None:
        raise ValueError("--motion_name and --motion_names are mutually exclusive.")
    if selected_motion_names is not None and (
        not selected_motion_names or any(not name for name in selected_motion_names)
    ):
        raise ValueError("--motion_names must contain non-empty names.")
    if int(args_cli.balanced_rows_per_motion) < 0:
        raise ValueError("--balanced_rows_per_motion must be >= 0.")
    if args_cli.balanced_motion_names and int(args_cli.balanced_rows_per_motion) <= 0:
        raise ValueError(
            "--balanced_motion_names requires positive --balanced_rows_per_motion."
        )
    if int(args_cli.balanced_rows_per_motion) > 0 and args_cli.evaluation_only:
        raise ValueError(
            "Balanced row collection cannot be used with --evaluation_only."
        )

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    planner_checkpoint_path = (
        args_cli.planner_checkpoint.expanduser().resolve()
        if args_cli.planner_checkpoint is not None
        else None
    )
    if planner_checkpoint_path is not None and not planner_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Planner checkpoint not found: {planner_checkpoint_path}"
        )
    if planner_checkpoint_path is not None and args_cli.interface not in {
        "per_step_token_sequence",
        "future_cvae",
    }:
        raise ValueError(
            "--planner_checkpoint is only valid for token or Future-CVAE targets."
        )
    motion_manifest = (
        args_cli.motion_manifest.expanduser().resolve()
        if args_cli.motion_manifest is not None
        else None
    )
    dataset_path = (
        args_cli.dataset_path.expanduser().resolve()
        if args_cli.dataset_path is not None
        else None
    )
    if dataset_path is not None and not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    low_level_command_mode = str(args_cli.low_level_command_mode)
    low_level_command_space = str(getattr(agent_cfg, "command_space", "unknown"))
    if args_cli.interface in {"full_body_trajectory", "ee_trajectory"}:
        low_level_command_space = args_cli.interface
    if low_level_command_mode == "streamed_vanilla":
        if args_cli.interface != "full_body_trajectory":
            raise ValueError(
                "streamed_vanilla requires --interface full_body_trajectory."
            )
        if int(args_cli.command_past_steps) != 0:
            raise ValueError("streamed_vanilla requires --command_past_steps 0.")
        if int(args_cli.command_future_steps) + 1 < int(
            args_cli.planner_interval_steps
        ):
            raise ValueError(
                "streamed_vanilla requires command_future_steps + 1 >= "
                "planner_interval_steps."
            )
        low_level_command_space = "single_frame_full_body"
        env_cfg.policy_command_mode = "full_body_chunk_current_slot"
    else:
        env_cfg.policy_command_mode = "reference"
    if args_cli.interface in {"full_body_trajectory", "ee_trajectory"}:
        agent_cfg.command_space = low_level_command_space
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()
    if args_cli.interface == "per_step_token_sequence":
        agent_cfg.ipmd.command_source = (
            "token_planner" if planner_checkpoint_path is not None else "posterior"
        )
        agent_cfg.ipmd.token_planner_checkpoint_path = (
            str(planner_checkpoint_path) if planner_checkpoint_path is not None else ""
        )
    elif args_cli.interface == "future_cvae":
        agent_cfg.ipmd.command_source = (
            "continuous_planner" if planner_checkpoint_path is not None else "posterior"
        )
        agent_cfg.ipmd.continuous_planner_checkpoint_path = (
            str(planner_checkpoint_path) if planner_checkpoint_path is not None else ""
        )
    env_cfg.latent_patch_past_steps = int(args_cli.command_past_steps)
    env_cfg.latent_patch_future_steps = int(args_cli.command_future_steps)
    if low_level_command_mode == "streamed_vanilla":
        env_cfg.command_observation_source = "planner_oracle"
        env_cfg.command_hold_steps = int(args_cli.planner_interval_steps)
    else:
        env_cfg.command_observation_source = "reference"
    _sync_env_window_params(env_cfg)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    if dataset_path is not None:
        env_cfg.dataset_path = str(dataset_path)
    if motion_manifest is not None:
        env_cfg.lafan1_manifest_path = str(motion_manifest)
        resolve_manifest_config = getattr(env_cfg, "_resolve_manifest_config", None)
        if callable(resolve_manifest_config):
            resolve_manifest_config(dataset_path_explicit=dataset_path is not None)
    if selected_motion_name:
        env_cfg.motions = [selected_motion_name]
    elif selected_motion_names is not None:
        env_cfg.motions = selected_motion_names
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
            raise ValueError("M3 collection requires configurable random reset steps.")
        env_cfg.random_reset_step_min = 0
        env_cfg.random_reset_step_max = 200
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
            raise ValueError("M3 collection requires base_too_low to remain active.")
    step_dt = _configured_step_dt(env_cfg)
    total_control_steps = (
        int(args_cli.control_steps)
        if int(args_cli.control_steps) > 0
        else int(args_cli.steps) * int(args_cli.planner_interval_steps)
    )
    episode_length_extension_enabled = bool(
        not args_cli.keep_configured_episode_length
        and step_dt is not None
        and hasattr(env_cfg, "episode_length_s")
    )
    if episode_length_extension_enabled:
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s), float(total_control_steps + 2) * step_dt
        )

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "rollout_training_samples"
    if not args_cli.evaluation_only:
        samples_dir.mkdir(parents=True, exist_ok=True)
    sample_writer = PlannerSampleWriter(
        samples_dir,
        rows_per_file=int(args_cli.sample_rows_per_file),
    )
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

    render_mode = "rgb_array" if args_cli.video else None
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")
    if args_cli.video:
        video_length = (
            int(args_cli.video_length)
            if int(args_cli.video_length) > 0
            else total_control_steps
        )
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(output_dir / "videos" / "play"),
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
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(total_control_steps + 2)),
    )
    base_env = _unwrap_imitation_env(env)
    tracked_body_names = resolve_existing_body_names(
        base_env, list(G1_TRACKED_BODY_NAMES)
    )
    ee_body_names = resolve_existing_body_names(
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
    else:
        agent.load_model(str(checkpoint_path))
        policy = agent.collector_policy
        policy.eval()
    token_learner = None
    future_cvae_learner = None
    if args_cli.interface == "per_step_token_sequence":
        token_learner = getattr(agent, "_latent_learner", None)
        if token_learner is None or not callable(
            getattr(token_learner, "infer_token_packet", None)
        ):
            raise ValueError(
                "per_step_token_sequence collection requires a low-level "
                "per_step_vq_sequence checkpoint."
            )
    elif args_cli.interface == "future_cvae":
        future_cvae_learner = getattr(agent, "_latent_learner", None)
        if future_cvae_learner is None or not callable(
            getattr(future_cvae_learner, "infer_expert_latents", None)
        ):
            raise ValueError(
                "future_cvae collection requires a low-level future_cvae checkpoint."
            )

    metadata: dict[str, Any] | None = None
    td = env.reset()
    num_envs = int(args_cli.num_envs)
    start_trajectories = trajectory_metadata(base_env)
    episode_ids = torch.zeros(num_envs, dtype=torch.long)
    motion_name_table = [
        str(name) for name in base_env.expert_trajectory_motion_names()
    ]
    balanced_selector: BalancedMotionRowSelector | None = None
    if int(args_cli.balanced_rows_per_motion) > 0:
        balanced_motion_names = (
            [str(name).strip() for name in args_cli.balanced_motion_names]
            if args_cli.balanced_motion_names is not None
            else (
                selected_motion_names
                if selected_motion_names is not None
                else list(motion_name_table)
            )
        )
        missing_motion_names = sorted(
            set(balanced_motion_names).difference(motion_name_table)
        )
        if missing_motion_names:
            raise ValueError(
                "Balanced motions are not loaded by the environment: "
                f"{missing_motion_names}."
            )
        balanced_selector = BalancedMotionRowSelector(
            balanced_motion_names,
            rows_per_motion=int(args_cli.balanced_rows_per_motion),
        )
    language_lookup: torch.Tensor | None = None
    language_metadata: dict[str, Any] = {
        "enabled": False,
        "embedding_dim": 0,
    }
    if args_cli.language_embeddings is not None:
        language_lookup, language_metadata = load_rank_language_embeddings(
            args_cli.language_embeddings,
            motion_names=motion_name_table,
            device="cpu",
        )
    saved_steps = 0
    saved_rows = 0
    steps_run = 0
    stop_reason = "max_steps"
    active = torch.ones(num_envs, dtype=torch.bool)
    survival_steps = torch.zeros(num_envs, dtype=torch.float32)
    return_sum = torch.zeros(num_envs, dtype=torch.float32)
    done_events = torch.zeros(num_envs, dtype=torch.float32)
    tracking_failure_events = torch.zeros(num_envs, dtype=torch.float32)
    termination_event_counts: dict[str, int] = {}
    metric_stats: dict[str, list[torch.Tensor]] = {}
    previous_action: torch.Tensor | None = None
    previous_body_lin_vel: tuple[torch.Tensor, torch.Tensor] | None = None
    previous_velocity_valid = torch.zeros(num_envs, dtype=torch.bool)
    valid_transition_count = 0
    planner_publish_count = 0
    planner_sampler = getattr(agent, "_hl_skill_command_sampler", None)
    command_window_steps = (
        int(args_cli.command_past_steps) + int(args_cli.command_future_steps) + 1
    )
    with torch.inference_mode(), set_exploration_type(InteractionType.DETERMINISTIC):
        for step_idx in range(total_control_steps):
            step_active = active.clone()
            if not bool(step_active.any()):
                stop_reason = "all_envs_done"
                break
            expert_batch = base_env.current_expert_macro_transition_batch(
                horizon_steps=command_window_steps,
                state_history_steps=int(args_cli.state_history_steps),
            )
            achieved_batch = base_env.current_causal_planner_observation(
                history_steps=int(args_cli.state_history_steps),
            )
            expert_planner_batch = base_env.current_offline_demo_planner_observation(
                history_steps=int(args_cli.state_history_steps),
            )
            if token_learner is not None:
                target, _ = token_learner.infer_token_packet(td, detach=True)
                target_spec = InterfaceTargetSpec(
                    interface="per_step_token_sequence",
                    term_names=("token_ids",),
                    term_widths=(int(target.shape[-1]),),
                )
            elif future_cvae_learner is not None:
                target = future_cvae_learner.infer_expert_latents(td, detach=True)
                target_spec = InterfaceTargetSpec(
                    interface="future_cvae",
                    term_names=("z",),
                    term_widths=(int(target.shape[-1]),),
                )
            elif args_cli.interface == "single_frame_full_body":
                target_terms = {
                    name: base_env.get_current_expert_window_term(
                        term_name=name,
                        past_steps=0,
                        future_steps=0,
                    )
                    for name in (
                        "expert_motion",
                        "expert_anchor_pos_b",
                        "expert_anchor_ori_b",
                    )
                }
                target = torch.cat(list(target_terms.values()), dim=-1)
                target_spec = InterfaceTargetSpec(
                    interface="single_frame_full_body",
                    term_names=tuple(target_terms),
                    term_widths=tuple(
                        int(value.shape[-1]) for value in target_terms.values()
                    ),
                )
                demonstration_terms = base_env.current_offline_demo_command_terms(
                    past_steps=0,
                    future_steps=0,
                )
                demonstration_target = torch.cat(
                    [demonstration_terms[name] for name in target_terms], dim=-1
                )
            else:
                command_terms = _current_reference_command_terms(
                    base_env,
                    interface=args_cli.interface,
                    ee_body_names=ee_body_names,
                )
                target, target_spec = flatten_command_terms(
                    args_cli.interface, command_terms
                )
                demonstration_target, _ = flatten_command_terms(
                    args_cli.interface,
                    _current_demonstration_command_terms(
                        base_env,
                        interface=args_cli.interface,
                        ee_body_names=ee_body_names,
                    ),
                )
            if token_learner is not None or future_cvae_learner is not None:
                demonstration_target = target
            if metadata is None:
                metadata = add_sample_format_metadata(
                    {
                        "interface": args_cli.interface,
                        "low_level_command_mode": low_level_command_mode,
                        "low_level_command_space": low_level_command_space,
                        "policy_command_mode": str(env_cfg.policy_command_mode),
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
                        "motion_name": selected_motion_name or None,
                        "motion_names": selected_motion_names,
                        "balanced_collection": (
                            {
                                "motion_names": list(balanced_selector.motion_names),
                                "rows_per_motion": balanced_selector.rows_per_motion,
                            }
                            if balanced_selector is not None
                            else None
                        ),
                        "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
                        "num_envs": int(args_cli.num_envs),
                        "seed": int(args_cli.seed),
                        "low_level_action_source": args_cli.low_level_action_source,
                        "low_level_tracker": tracker_provenance,
                        "planner_observation_spec": base_env.causal_planner_observation_spec(
                            history_steps=int(args_cli.state_history_steps)
                        ),
                        "reset_schedule": str(
                            getattr(env_cfg, "reset_schedule", "unknown")
                        ),
                        "random_reset_step_min": int(
                            getattr(env_cfg, "random_reset_step_min", -1)
                        ),
                        "random_reset_step_max": int(
                            getattr(env_cfg, "random_reset_step_max", -1)
                        ),
                        "wrap_steps": bool(getattr(env_cfg, "wrap_steps", False)),
                        "policy_observation_corruption_enabled": bool(
                            getattr(
                                getattr(
                                    getattr(env_cfg, "observations", None),
                                    "policy",
                                    None,
                                ),
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
                        "episode_length_extension_enabled": (
                            episode_length_extension_enabled
                        ),
                        "episode_length_s": float(
                            getattr(env_cfg, "episode_length_s", -1.0)
                        ),
                        "reward_clipping_enabled": False,
                        "language_conditioning": language_metadata,
                        **(
                            {
                                "target_encoding": {
                                    "kind": "categorical_sequence",
                                    "horizon": int(target.shape[-1]),
                                    "codebook_size": int(
                                        token_learner._codebook_size()
                                    ),
                                }
                            }
                            if token_learner is not None
                            else {}
                        ),
                        "provenance": {
                            "low_level_checkpoint": str(checkpoint_path),
                            "low_level_tracker": tracker_provenance,
                            "planner_checkpoint": (
                                str(planner_checkpoint_path)
                                if planner_checkpoint_path is not None
                                else None
                            ),
                            "motion_manifest": str(motion_manifest)
                            if motion_manifest is not None
                            else None,
                            "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
                        },
                    },
                    collection_stage=(
                        "reference_action_diagnostic"
                        if args_cli.low_level_action_source == "reconstructed_reference"
                        else (
                            "planner_rollout"
                            if planner_checkpoint_path is not None
                            else "oracle_rollout"
                        )
                    ),
                    planner_interval_steps=int(args_cli.planner_interval_steps),
                    control_rate_hz=(1.0 / step_dt) if step_dt else 50.0,
                )
            renew_env_ids = planner_renew_env_ids(
                base_env.episode_length_buf,
                int(args_cli.planner_interval_steps),
                initial_publication=step_idx == 0,
            )
            if int(renew_env_ids.numel()) > 0:
                active_on_device = step_active.to(device=renew_env_ids.device)
                renew_env_ids = renew_env_ids[
                    active_on_device.index_select(0, renew_env_ids)
                ]
            if int(renew_env_ids.numel()) > 0:
                renew_env_ids_cpu = renew_env_ids.detach().cpu()
                target_at_renew = target.index_select(
                    0, renew_env_ids.to(device=target.device)
                )
                demonstration_target_at_renew = demonstration_target.index_select(
                    0, renew_env_ids.to(device=demonstration_target.device)
                )
                planner_state_all = planner_state_from_batch(
                    achieved_batch,
                    state_history_steps=int(args_cli.state_history_steps),
                )
                planner_state = planner_state_all.index_select(
                    0, renew_env_ids.to(device=planner_state_all.device)
                )
                demonstration_state_all = planner_state_from_batch(
                    expert_planner_batch,
                    state_history_steps=int(args_cli.state_history_steps),
                )
                demonstration_state = demonstration_state_all.index_select(
                    0, renew_env_ids.to(device=demonstration_state_all.device)
                )
                traj_rank_all = (
                    expert_batch.get(("hl", "traj_rank")).detach().cpu().reshape(-1)
                )
                local_step_all = (
                    expert_batch.get(("hl", "local_step")).detach().cpu().reshape(-1)
                )
                traj_rank = traj_rank_all.index_select(0, renew_env_ids_cpu)
                local_step = local_step_all.index_select(0, renew_env_ids_cpu)
                motion_names = [
                    motion_name_table[int(rank)]
                    if 0 <= int(rank) < len(motion_name_table)
                    else str(int(rank))
                    for rank in traj_rank.tolist()
                ]
                language_at_renew = (
                    None
                    if language_lookup is None
                    else language_lookup.index_select(0, traj_rank)
                )
                save_row_indices: torch.Tensor | None = None
                if balanced_selector is not None:
                    selected_indices = balanced_selector.select(motion_names)
                    save_row_indices = torch.tensor(
                        selected_indices, dtype=torch.long, device="cpu"
                    )
                renew_mask = torch.ones(int(renew_env_ids.numel()), dtype=torch.bool)
                planner_publish_count += int(renew_env_ids.numel())
                if planner_checkpoint_path is not None:
                    if planner_sampler is None:
                        raise RuntimeError(
                            "Planner command sampler was not initialized."
                        )
                    planner = getattr(planner_sampler, "planner", None)
                    planner_device = getattr(planner_sampler, "device", base_env.device)
                    if not callable(planner):
                        raise RuntimeError("Planner command sampler has no planner.")
                    planner_input = planner_state.to(
                        device=planner_device, dtype=torch.float32
                    ).reshape(int(renew_env_ids.numel()), -1)
                    if token_learner is not None:
                        predicted_target = planner(planner_input).to(
                            target_at_renew.device
                        )
                        token_accuracy = (
                            (predicted_target == target_at_renew).float().mean(dim=-1)
                        )
                        packet_accuracy = (
                            (predicted_target == target_at_renew).all(dim=-1).float()
                        )
                        accumulate_metric(
                            metric_stats,
                            "planner_token_accuracy",
                            token_accuracy.cpu(),
                            renew_mask,
                        )
                        accumulate_metric(
                            metric_stats,
                            "planner_packet_accuracy",
                            packet_accuracy.cpu(),
                            renew_mask,
                        )
                    else:
                        predicted_target = planner(
                            planner_input,
                            num_inference_steps=int(
                                getattr(planner_sampler, "num_inference_steps", 16)
                            ),
                            inference_noise_std=float(
                                getattr(planner_sampler, "inference_noise_std", 0.0)
                            ),
                        ).to(target_at_renew.device)
                        target_rmse = torch.sqrt(
                            torch.mean(
                                (predicted_target - target_at_renew).square(), dim=-1
                            )
                        )
                        accumulate_metric(
                            metric_stats,
                            "planner_target_rmse",
                            target_rmse.cpu(),
                            renew_mask,
                        )
                if not args_cli.evaluation_only and (
                    save_row_indices is None or int(save_row_indices.numel()) > 0
                ):
                    if save_row_indices is None:
                        sample_planner_state = planner_state
                        sample_demonstration_state = demonstration_state
                        sample_target = target_at_renew
                        sample_demonstration_target = demonstration_target_at_renew
                        sample_traj_rank = traj_rank
                        sample_local_step = local_step
                        sample_episode_id = episode_ids.index_select(
                            0, renew_env_ids_cpu
                        )
                        sample_motion_names = motion_names
                        sample_language = language_at_renew
                    else:
                        sample_planner_state = planner_state.index_select(
                            0, save_row_indices.to(device=planner_state.device)
                        )
                        sample_demonstration_state = demonstration_state.index_select(
                            0, save_row_indices.to(device=demonstration_state.device)
                        )
                        sample_target = target_at_renew.index_select(
                            0, save_row_indices.to(device=target_at_renew.device)
                        )
                        sample_demonstration_target = (
                            demonstration_target_at_renew.index_select(
                                0,
                                save_row_indices.to(
                                    device=demonstration_target_at_renew.device
                                ),
                            )
                        )
                        sample_traj_rank = traj_rank.index_select(0, save_row_indices)
                        sample_local_step = local_step.index_select(0, save_row_indices)
                        sample_episode_id = episode_ids.index_select(
                            0,
                            renew_env_ids_cpu.index_select(0, save_row_indices),
                        )
                        sample_motion_names = [
                            motion_names[int(index)]
                            for index in save_row_indices.tolist()
                        ]
                        sample_language = (
                            None
                            if language_at_renew is None
                            else language_at_renew.index_select(0, save_row_indices)
                        )
                    sample = build_planner_sample(
                        causal_state_history=sample_planner_state,
                        demonstration_state_history=sample_demonstration_state,
                        causal_target=sample_target,
                        demonstration_target=sample_demonstration_target,
                        trajectory_rank=sample_traj_rank,
                        episode_id=sample_episode_id,
                        control_step=sample_local_step,
                        planner_step=torch.div(
                            sample_local_step,
                            int(args_cli.planner_interval_steps),
                            rounding_mode="floor",
                        ),
                        motion_names=sample_motion_names,
                        metadata=metadata,
                        language_embedding=sample_language,
                    )
                    sample_writer.add(sample)
                    saved_rows += int(sample_target.shape[0])

                if balanced_selector is not None and balanced_selector.complete:
                    stop_reason = "balanced_rows_complete"
                    break

            td = policy(td)
            if args_cli.low_level_action_source == "reconstructed_reference":
                td.set("action", base_env.current_reconstructed_reference_action())
            action = td.get("action")
            if isinstance(action, torch.Tensor):
                action_2d = action.detach().reshape(num_envs, -1).cpu()
                accumulate_metric(
                    metric_stats,
                    "action_l2",
                    torch.linalg.vector_norm(action_2d, dim=-1),
                    step_active,
                )
                if previous_action is not None:
                    accumulate_metric(
                        metric_stats,
                        "action_delta_l2",
                        torch.linalg.vector_norm(action_2d - previous_action, dim=-1),
                        step_active,
                    )
                previous_action = action_2d
            td_step = env.step(td)
            steps_run += 1
            rewards = optional_flat_tensor(
                td_step, ("next", "reward"), num_envs=num_envs, default=0.0
            )
            dones = optional_flat_tensor(
                td_step, ("next", "done"), num_envs=num_envs, default=False
            ).bool()
            terminateds = optional_flat_tensor(
                td_step, ("next", "terminated"), num_envs=num_envs, default=False
            ).bool()
            truncateds = optional_flat_tensor(
                td_step, ("next", "truncated"), num_envs=num_envs, default=False
            ).bool()
            done_any = dones | terminateds | truncateds
            termination_manager = getattr(base_env, "termination_manager", None)
            if termination_manager is not None:
                for term_name in termination_manager.active_terms:
                    term_done = termination_manager.get_term(term_name).detach().cpu()
                    termination_event_counts[term_name] = termination_event_counts.get(
                        term_name, 0
                    ) + int((term_done.bool() & step_active).sum().item())
            episode_ids += done_any.to(dtype=torch.long)
            return_sum += rewards.float() * step_active.float()
            survival_steps += step_active.float()
            done_events += (done_any & step_active).float()
            metric_mask = (
                step_active if not args_cli.stop_after_done else step_active & ~done_any
            )
            valid_transition_count += int(metric_mask.sum().item())
            step_metrics, body_lin_vel, tracking_failure = tracking_metrics(
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
            for metric_name, values in step_metrics.items():
                accumulate_metric(metric_stats, metric_name, values.cpu(), metric_mask)
            if body_lin_vel is not None and step_dt is not None:
                if previous_body_lin_vel is not None:
                    actual_lin_vel, ref_lin_vel = body_lin_vel
                    prev_actual_lin_vel, prev_ref_lin_vel = previous_body_lin_vel
                    acceleration_distance = torch.linalg.vector_norm(
                        (actual_lin_vel - prev_actual_lin_vel) / float(step_dt)
                        - (ref_lin_vel - prev_ref_lin_vel) / float(step_dt),
                        dim=-1,
                    ).mean(dim=-1)
                    accumulate_metric(
                        metric_stats,
                        "tracking_acceleration_distance_mps2",
                        acceleration_distance.cpu(),
                        metric_mask & previous_velocity_valid,
                    )
                previous_body_lin_vel = (
                    body_lin_vel[0].clone(),
                    body_lin_vel[1].clone(),
                )
                previous_velocity_valid = step_active & ~done_any
            if args_cli.stop_after_done:
                active &= ~done_any
            td = step_mdp(
                td_step, exclude_reward=True, exclude_done=False, exclude_action=True
            )

    sample_writer.flush()
    saved_steps = sample_writer.file_count
    if saved_rows != sample_writer.row_count:
        raise RuntimeError(
            "Planner sample writer row accounting differs from collection: "
            f"collected={saved_rows}, written={sample_writer.row_count}."
        )
    active_mask = survival_steps > 0
    return_mean, return_std = tensor_mean_std(return_sum, active_mask)
    survival_mean, survival_std = tensor_mean_std(survival_steps, active_mask)
    completed_horizon = survival_steps >= float(total_control_steps)
    avoided_tracking_failure = tracking_failure_events == 0
    aggregate = {
        "return_sum_mean": return_mean,
        "return_sum_std": return_std,
        "survival_steps_mean": survival_mean,
        "survival_steps_std": survival_std,
        "survival_fraction_mean": survival_mean / float(total_control_steps),
        "horizon_completion_rate": float(
            completed_horizon[active_mask].float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "done_rate": float((done_events[active_mask] > 0).float().mean().item())
        if bool(active_mask.any())
        else float("nan"),
        "tracking_success_rate": float(
            (tracking_failure_events[active_mask] == 0).float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "tracking_failure_rate": float(
            (tracking_failure_events[active_mask] > 0).float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "tracking_failed_env_count": int(
            (tracking_failure_events[active_mask] > 0).sum().item()
        )
        if bool(active_mask.any())
        else 0,
        "closed_loop_success_rate": float(
            (completed_horizon & avoided_tracking_failure)[active_mask]
            .float()
            .mean()
            .item()
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
        "termination_event_counts": dict(sorted(termination_event_counts.items())),
        "termination_event_rates": {
            name: float(count) / float(num_envs)
            for name, count in sorted(termination_event_counts.items())
        },
    }
    summary = {
        "metadata": metadata or {},
        "aggregate": aggregate,
        "metrics": finalize_metric_stats(metric_stats),
        "start_trajectories": start_trajectories,
        "final_trajectories": trajectory_metadata(base_env),
        "saved_steps": saved_steps,
        "saved_rows": saved_rows,
        "sample_file_count": saved_steps,
        "sample_rows_per_file": int(args_cli.sample_rows_per_file),
        "control_steps": total_control_steps,
        "max_steps": total_control_steps,
        "steps_run": steps_run,
        "stop_reason": stop_reason,
        "evaluation_only": bool(args_cli.evaluation_only),
        "stop_after_done": bool(args_cli.stop_after_done),
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
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if args_cli.evaluation_only:
        print(
            f"[RESULT] interface={args_cli.interface} "
            f"survival={aggregate['survival_steps_mean']:.1f} "
            f"done_rate={aggregate['done_rate']:.3f} "
            f"tracking_success={aggregate['tracking_success_rate']:.3f}"
        )
    else:
        print(
            f"[INFO] Wrote {saved_rows} sample rows "
            f"across {saved_steps} files to: {samples_dir}"
        )
    env.close()
    if balanced_selector is not None and not balanced_selector.complete:
        raise RuntimeError(
            "Balanced collection ended before every motion reached its row budget: "
            f"{balanced_selector.missing()}."
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
