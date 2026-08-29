#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate a deterministic RLOpt checkpoint against the G1 reference motion."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

from isaaclab.app import AppLauncher
from imitation_experiments.evaluation.push_attribution import (
    attach_push_tracker,
)
from imitation_experiments.audit.backend_determinism import (
    RANDOMIZATION_PROFILES,
    apply_randomization_profile,
    pin_reference_start,
)
from imitation_experiments.lowlevel.motion_candidate_screen import (
    build_env_rank_assignment,
)


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
        "IPMD",
    ],
)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--output_json", type=Path, default=None)
# Score several checkpoints of ONE arm in a single process. The interface is
# fixed by the arm, so only the policy weights change between cells, and one
# Isaac Sim start then serves the whole budget axis instead of one start per
# checkpoint. `--trajectory_ranks` pins env i to rank_table[i], a pure function
# of env id, so every cell resets onto the same clips and the rows stay
# comparable; the loop asserts that. `--checkpoints` and `--output_jsons` are
# parallel lists of equal length.
parser.add_argument("--checkpoints", type=Path, nargs="+", default=None)
parser.add_argument("--output_jsons", type=Path, nargs="+", default=None)
parser.add_argument("--output_csv", type=Path, default=None)
parser.add_argument(
    "--append_csv",
    action="store_true",
    default=False,
    help="Append to --output_csv instead of replacing it.",
)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--command_space", type=str, default=None)
parser.add_argument(
    "--skill_encoder_source",
    type=str,
    default="auto",
    choices=("auto", "checkpoint", "pretrained"),
    help=(
        "Which skill encoder drives the rollout. 'auto' prefers the encoder "
        "embedded in the tracker checkpoint when one is present -- the only "
        "correct choice for an arm trained with hl_skill_finetune_enabled=true "
        "-- and falls back to the pretrained file otherwise. 'pretrained' "
        "forces the file named by agent.ipmd.hl_skill_checkpoint_path and will "
        "MIS-SCORE a co-trained arm. The resolved choice is recorded in the "
        "summary."
    ),
)
parser.add_argument(
    "--policy_only_checkpoint",
    action="store_true",
    default=False,
    help=(
        "Strict-load only policy_state_dict. Required when evaluating a frozen "
        "vanilla actor independently of training-only critic/optimizer shapes."
    ),
)
parser.add_argument(
    "--ipmd_l2t_policy_role",
    choices=("teacher", "student"),
    default=None,
    help=(
        "Evaluate one role from a full IPMD-L2T checkpoint. 'teacher' uses the "
        "privileged teacher policy; 'student' uses the deployable policy. This "
        "requires --algo IPMD and the IPMD-L2T agent entry point."
    ),
)
parser.add_argument(
    "--low_level_command_mode",
    choices=("native", "streamed_vanilla"),
    default="native",
    help=(
        "streamed_vanilla treats --command_space as a full-body planner target "
        "but executes its current slot with the unchanged vanilla tracker."
    ),
)
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
parser.add_argument(
    "--motion_name",
    type=str,
    default=None,
    help="Optional single motion name to evaluate from the manifest.",
)
parser.add_argument(
    "--trajectory_rank",
    type=int,
    default=None,
    help=(
        "Optional exact trajectory rank in the loaded dataset. Unlike "
        "--motion_name, this preserves the full-store data path and is useful "
        "for reproducing one row from a large parallel evaluation."
    ),
)
parser.add_argument(
    "--trajectory_ranks",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Exact trajectory ranks to evaluate in one process. num_envs must be a "
        "multiple of the rank count; each rank receives an equal, stable block "
        "of environments across asynchronous resets."
    ),
)
parser.add_argument(
    "--dataset_path",
    type=Path,
    default=None,
    help="Optional zarr dataset/cache path. Useful on clusters where /tmp is small.",
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
parser.add_argument(
    "--certify_streamed_vanilla_equivalence",
    action="store_true",
    default=False,
    help=(
        "Verify that oracle full-body chunk slots and deterministic actions match "
        "direct vanilla commands across every held-command phase, then exit."
    ),
)
parser.add_argument("--equivalence_steps", type=int, default=20)
parser.add_argument("--equivalence_atol", type=float, default=5.0e-5)
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
parser.add_argument(
    "--agent_entry_point",
    type=str,
    default=None,
    help=(
        "Agent config entry point to build the network from, e.g. "
        "rlopt_ipmd_tuned_cfg_entry_point. Must match what the checkpoint was "
        "TRAINED with: the tuned recipe changes widths and activations, so "
        "loading a tuned checkpoint under the default contract fails on a "
        "state-dict mismatch. Defaults to the task's algorithm entry point."
    ),
)
parser.add_argument(
    "--randomization",
    type=str,
    default="none",
    choices=RANDOMIZATION_PROFILES,
    help=(
        "Which domain-randomization families stay live. Default 'none' is the "
        "TRACKING-FIDELITY protocol: MPJPE should measure how well the policy "
        "tracks, not how hard it was pushed, and reset randomization alone "
        "contributes ~39 mm at frame 0 on the G1's 14-body set. 'no_push' keeps "
        "startup and reset randomization but removes only the interval push. "
        "Use 'all' for the ROBUSTNESS protocol, which keeps the interval push -- "
        "that is what the paper comparison requires, since both interfaces must "
        "eat the same perturbation. Whichever is chosen is recorded in the output."
    ),
)
parser.add_argument(
    "--action_sampling",
    type=str,
    default="mode",
    choices=("mode", "stochastic"),
    help=(
        "How actions are drawn during the rollout. Default 'mode' is the "
        "protocol: a checkpoint is judged on the policy it would deploy. "
        "'stochastic' samples from the action distribution the way TRAINING "
        "does, and exists only to explain the train/eval metric gap -- it is "
        "NOT a protocol option and must not be used to report a score. The "
        "equivalence certificate stays deterministic either way, since its "
        "whole purpose is a bit-comparison. Recorded in the output."
    ),
)
parser.add_argument(
    "--disable_early_terminations",
    action="store_true",
    default=False,
    help=(
        "Full-horizon diagnostic: disable every early termination, including "
        "base_too_low, so MPJPE is scored over the whole horizon instead of a "
        "termination-truncated rollout. Diagnostic only -- this cannot satisfy "
        "oracle qualification, which requires the strict protocol."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help=(
        "Render the rollout to MP4. Implies --enable_cameras. Rendering is far "
        "slower than headless evaluation, so pair it with a small --num_envs "
        "and pinned --trajectory_ranks; the metrics it writes are still the "
        "protocol's, so a rendered run stays comparable."
    ),
)
parser.add_argument("--video_dir", type=Path, default=None)
parser.add_argument("--video_length", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    # Rendering needs the camera pipeline; asking for a video without it
    # produces a silently empty file.
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_imitation.contracts import mpjpe_local_global
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils import math as math_utils
from isaaclab_imitation.envs import ImitationRLEnv
from isaaclab_imitation.envs.imitation_rl_env_legacy import ImitationRLEnvLegacy
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_EE_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    VANILLA_POLICY_INPUT_KEYS,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMD, PPO, SAC
from rlopt.agent.ipmd.ipmd_l2t import IPMDL2T
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp


from imitation_experiments.lowlevel.low_level_tracker import (
    load_frozen_low_level_tracker,
)
from imitation_experiments.provenance.paper_protocol_metadata import (
    interval_event_metadata,
)
from isaaclab_imitation.contracts.planner_publish_schedule import planner_renew_env_ids
from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
    bind_command_interface,
)
from isaaclab_imitation.tasks.manager_based.imitation.motion_data import (
    apply_motion_data,
)


ALGORITHM_CLASS_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "IPMD": IPMD,
}

ENTRY_POINT_ALGORITHM_MAP = {
    "rlopt_ppo_cfg_entry_point": "PPO",
    "rlopt_sac_cfg_entry_point": "SAC",
    "rlopt_ipmd_cfg_entry_point": "IPMD",
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


#: Both imitation env classes. They are SIBLINGS -- each subclasses
#: ``ManagerBasedRLEnv`` directly -- so accepting only one silently excludes the
#: other. This script matched ``ImitationRLEnvLegacy`` alone, which is the
#: deprecated v0/v1 backing env, so it could not evaluate a ``-G1-v2``
#: checkpoint at all: it raised here and the traceback was then swallowed by
#: ``simulation_app.close()``, leaving exit code 0 and no output.
_IMITATION_ENV_CLASSES = (ImitationRLEnv, ImitationRLEnvLegacy)

ImitationEnv = ImitationRLEnv | ImitationRLEnvLegacy


def _unwrap_imitation_env(env: object) -> ImitationEnv:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, _IMITATION_ENV_CLASSES):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if isinstance(unwrapped, _IMITATION_ENV_CLASSES):
            return unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError(
        "Could not unwrap an imitation env; expected one of "
        f"{[cls.__name__ for cls in _IMITATION_ENV_CLASSES]}."
    )


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
    base_env: ImitationEnv,
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


def _body_ids_for_names(base_env: ImitationEnv, names: list[str]) -> list[int]:
    return [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]


def _mean_body_pose_errors(
    base_env: ImitationEnv,
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


def _body_tracking_tensors(
    base_env: ImitationEnv,
    names: list[str],
) -> dict[str, torch.Tensor] | None:
    if len(names) == 0:
        return None
    body_ids = _body_ids_for_names(base_env, names)
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


def _as_torch(value: Any) -> Any:
    """Cross the Newton/Isaac Lab 3.0 array boundary.

    Under the Newton backend ``robot.data.*`` returns a Warp-backed
    ``ProxyArray``. Arithmetic on it happens to work, so most expressions here
    were silently fine, but a ``torch.jit`` function such as
    ``quat_error_magnitude`` rejects it outright. The repo convention is the
    ``.torch`` view; PhysX already hands back real tensors, where this is a
    no-op.
    """
    return value.torch if hasattr(value, "torch") else value


def _skill_encoder_provenance(
    agent: Any,
    checkpoint_path: Any,
    *,
    source: str,
    map_location: Any,
) -> dict[str, Any] | None:
    """Say which skill encoder is actually driving the rollout, and pin it.

    An arm trained with `hl_skill_finetune_enabled=true` keeps updating its
    encoder during RL, so the encoder inside the tracker checkpoint is NOT the
    pretrained file the CLI points at -- 0.739 max abs divergence was measured
    on `fsq64_hold10_dyn`. Scoring such an arm against the pretrained file pairs
    a tracker with an encoder it never saw, which is why every `*_dyn` arm was
    excluded from every board.

    This reports the live encoder's distance to both candidates, so the answer
    is recorded rather than assumed, and lets the caller pin the choice.
    """
    import torch

    sampler = getattr(agent, "_hl_skill_command_sampler", None)
    encoder = getattr(sampler, "skill_encoder", None)
    if encoder is None:
        return None

    blob = torch.load(
        str(checkpoint_path), map_location=map_location, weights_only=False
    )
    embedded = None
    sampler_state = blob.get("hl_skill_command_sampler_state_dict")
    if isinstance(sampler_state, dict):
        embedded = sampler_state.get("skill_encoder_state_dict")

    def _max_abs_delta(state: dict[str, Any] | None) -> float | None:
        if not isinstance(state, dict):
            return None
        live = encoder.state_dict()
        worst = 0.0
        for key, value in state.items():
            other = live.get(key)
            if (
                other is None
                or not hasattr(value, "shape")
                or value.shape != other.shape
            ):
                return None
            worst = max(worst, float((other.detach().cpu() - value.cpu()).abs().max()))
        return worst

    provenance: dict[str, Any] = {
        "requested_source": source,
        "checkpoint_carries_encoder": embedded is not None,
        "live_vs_checkpoint_encoder_max_abs": _max_abs_delta(embedded),
    }

    if source == "checkpoint" or (source == "auto" and embedded is not None):
        if embedded is None:
            raise ValueError(
                f"{checkpoint_path} carries no embedded skill encoder; "
                "--skill_encoder_source checkpoint cannot be satisfied."
            )
        encoder.load_state_dict(embedded)
        provenance["resolved_source"] = "checkpoint"
    else:
        provenance["resolved_source"] = "pretrained_file"
    provenance["post_load_vs_checkpoint_max_abs"] = _max_abs_delta(embedded)
    return provenance


def _tracking_metrics(
    base_env: ImitationEnv,
    *,
    tracked_body_names: list[str],
    ee_body_names: list[str],
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None]:
    robot_data = base_env.robot.data
    root_pos_ref, root_quat_ref, root_lin_vel_ref, root_ang_vel_ref = (
        base_env._get_reference_root_state_w_fast()
    )
    joint_pos_ref = base_env.current_expert_frame["joint_pos"]
    joint_vel_ref = base_env.current_expert_frame["joint_vel"]

    root_pos_error = _as_torch(robot_data.root_pos_w) - root_pos_ref
    root_lin_vel_error = _as_torch(robot_data.root_lin_vel_w) - root_lin_vel_ref
    root_ang_vel_error = _as_torch(robot_data.root_ang_vel_w) - root_ang_vel_ref
    root_ori_error = math_utils.quat_error_magnitude(
        _as_torch(robot_data.root_quat_w),
        root_quat_ref,
    )
    root_height_error = root_pos_error[:, 2].abs()

    metrics = {
        "root_pos_xy_error_m": torch.linalg.vector_norm(root_pos_error[:, :2], dim=-1),
        "root_pos_xyz_error_m": torch.linalg.vector_norm(root_pos_error, dim=-1),
        "root_height_error_m": root_height_error,
        "root_ori_error_rad": root_ori_error,
        "root_lin_vel_rmse_mps": torch.sqrt(
            torch.mean(root_lin_vel_error.square(), dim=-1)
        ),
        "root_ang_vel_rmse_radps": torch.sqrt(
            torch.mean(root_ang_vel_error.square(), dim=-1)
        ),
        "joint_pos_rmse_rad": torch.sqrt(
            torch.mean(
                (_as_torch(robot_data.joint_pos) - joint_pos_ref).square(), dim=-1
            )
        ),
        "joint_vel_rmse_radps": torch.sqrt(
            torch.mean(
                (_as_torch(robot_data.joint_vel) - joint_vel_ref).square(), dim=-1
            )
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
        tracking_mpjpe_m, tracking_mpjpe_g_m = mpjpe_local_global(
            tracked_tensors["actual_pos"],
            _as_torch(robot_data.root_pos_w),
            tracked_tensors["ref_pos"],
            root_pos_ref,
        )
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
        metrics["tracking_mpjpe_g_m"] = tracking_mpjpe_g_m
        metrics["tracking_mpjpe_g_mm"] = tracking_mpjpe_g_m * 1000.0
        metrics["tracking_velocity_distance_mps"] = body_lin_vel_error
        tracked_body_lin_vel = (
            tracked_tensors["actual_lin_vel"].detach(),
            tracked_tensors["ref_lin_vel"].detach(),
        )

    ee_errors = _mean_body_pose_errors(base_env, ee_body_names)
    if ee_errors is not None:
        metrics["ee_pos_error_m"] = ee_errors[0]
        metrics["ee_ori_error_rad"] = ee_errors[1]
        # Root-relative EE, the counterpart of MPJPE-L. `ee_pos_error_m` is
        # WORLD frame, and measurement shows it tracks root position almost
        # 1:1 -- so it answers "how far is the hand from where it should be in
        # the world", which is mostly a drift question. This answers the
        # different question "is the hand in the right place relative to the
        # body", which is what a wrist-specific reward could actually move.
        # Without the split, a drift fix and an articulation fix are
        # indistinguishable in the reported number.
        ee_ids = _body_ids_for_names(base_env, ee_body_names)
        actual_ee = _as_torch(base_env._get_robot_body_pose_w_fast(ee_ids)[0])
        ref_ee = _as_torch(
            base_env._get_reference_body_pose_w_fast(tuple(ee_body_names))[0]
        )
        metrics["ee_pos_error_local_m"] = torch.linalg.vector_norm(
            (actual_ee - _as_torch(robot_data.root_pos_w)[:, None, :])
            - (ref_ee - root_pos_ref[:, None, :]),
            dim=-1,
        ).mean(dim=-1)

    return metrics, tracked_body_lin_vel


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
    base_env: ImitationEnv,
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


def _certify_streamed_vanilla_equivalence(
    env: TransformedEnv,
    base_env: ImitationEnv,
    collector_policy: torch.nn.Module,
    *,
    num_steps: int,
    atol: float,
) -> dict[str, Any]:
    """Certify oracle chunk-current-slot and direct vanilla equivalence."""
    if base_env.policy_command_mode != "full_body_chunk_current_slot":
        raise ValueError(
            "Equivalence certification requires streamed_vanilla command mode."
        )
    if getattr(base_env, "_command_observation_source", None) != "planner_oracle":
        raise ValueError(
            "Equivalence certification requires planner_mode=none so the exact "
            "planner_oracle packet drives the adapter."
        )
    if int(num_steps) <= 0:
        raise ValueError("--equivalence_steps must be positive.")
    if float(atol) < 0.0:
        raise ValueError("--equivalence_atol must be non-negative.")

    command_keys = (
        ("policy", "expert_motion"),
        ("policy", "expert_anchor_pos_b"),
        ("policy", "expert_anchor_ori_b"),
    )
    policy_keys = tuple(VANILLA_POLICY_INPUT_KEYS)
    expected_policy_widths = {
        ("policy", "expert_motion"): 58,
        ("policy", "expert_anchor_pos_b"): 3,
        ("policy", "expert_anchor_ori_b"): 6,
        ("policy", "base_ang_vel"): 3,
        ("policy", "joint_pos_rel"): 29,
        ("policy", "joint_vel_rel"): 29,
        ("policy", "last_action"): 29,
    }
    if set(policy_keys) != set(expected_policy_widths):
        raise RuntimeError(
            "Vanilla policy-key contract changed without updating equivalence widths."
        )
    state_before = {
        key: value.detach().clone()
        for key, value in collector_policy.state_dict().items()
    }
    max_abs_by_key = {"/".join(key): 0.0 for key in policy_keys}
    max_action_abs = 0.0
    observed_phases: set[int] = set()
    hold_steps = int(getattr(base_env, "_command_hold_steps", 0))
    async_rephase_required = (
        int(base_env.num_envs) > 1 and int(num_steps) > hold_steps + 1
    )
    async_rephase_exercised = False
    td = env.reset()

    try:
        for step_index in range(int(num_steps)):
            if async_rephase_required and step_index == hold_steps + 1:
                # Desynchronize one environment's publication phase without
                # perturbing its robot/reference state. This exercises the same
                # row-wise renewal path used immediately after an async reset.
                base_env.episode_length_buf[0] = 0
                async_rephase_exercised = True
            phases = base_env._command_hold_phase().detach().cpu().tolist()
            observed_phases.update(int(phase) for phase in phases)

            base_env._policy_command_mode = "full_body_chunk_current_slot"
            adapter_td = _refresh_tensordict_observations(td.clone(), base_env)
            base_env._policy_command_mode = "reference"
            direct_td = _refresh_tensordict_observations(td.clone(), base_env)
            base_env._policy_command_mode = "full_body_chunk_current_slot"

            for key in policy_keys:
                adapter_value = adapter_td.get(key)
                direct_value = direct_td.get(key)
                if not isinstance(adapter_value, torch.Tensor) or not isinstance(
                    direct_value, torch.Tensor
                ):
                    raise KeyError(f"Missing vanilla tracker input key {key!r}.")
                expected_width = expected_policy_widths[key]
                if (
                    adapter_value.ndim < 2
                    or direct_value.shape != adapter_value.shape
                    or int(adapter_value.shape[-1]) != expected_width
                ):
                    raise RuntimeError(
                        f"Vanilla input {key!r} shape mismatch: "
                        f"adapter={tuple(adapter_value.shape)}, "
                        f"direct={tuple(direct_value.shape)}, "
                        f"expected_width={expected_width}."
                    )
                max_abs = float(
                    torch.max(torch.abs(adapter_value - direct_value)).item()
                )
                max_abs_by_key["/".join(key)] = max(
                    max_abs_by_key["/".join(key)], max_abs
                )

            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                adapter_action_td = collector_policy(adapter_td.clone())
                direct_action_td = collector_policy(direct_td.clone())
            adapter_action = adapter_action_td.get("action")
            direct_action = direct_action_td.get("action")
            if not isinstance(adapter_action, torch.Tensor) or not isinstance(
                direct_action, torch.Tensor
            ):
                raise RuntimeError("Vanilla tracker did not produce an action.")
            if (
                adapter_action.ndim < 2
                or direct_action.shape != adapter_action.shape
                or int(adapter_action.shape[-1]) != 29
            ):
                raise RuntimeError(
                    "Vanilla tracker action shape mismatch: "
                    f"adapter={tuple(adapter_action.shape)}, "
                    f"direct={tuple(direct_action.shape)}, expected_width=29."
                )
            max_action_abs = max(
                max_action_abs,
                float(torch.max(torch.abs(adapter_action - direct_action)).item()),
            )
            adapter_action_td.set("action", adapter_action)
            with torch.inference_mode():
                td_step = env.step(adapter_action_td)
            td = step_mdp(
                td_step,
                exclude_reward=True,
                exclude_done=False,
                exclude_action=True,
            )
    finally:
        base_env._policy_command_mode = "full_body_chunk_current_slot"

    for key, value in collector_policy.state_dict().items():
        if not torch.equal(value, state_before[key]):
            raise RuntimeError(f"Frozen tracker state changed during rollout: {key}.")

    expected_phases = set(range(hold_steps))
    missing_phases = sorted(expected_phases - observed_phases)
    command_max = max(max_abs_by_key["/".join(key)] for key in command_keys)
    all_input_max = max(max_abs_by_key.values())
    passed = (
        not missing_phases
        and (not async_rephase_required or async_rephase_exercised)
        and all_input_max <= float(atol)
        and max_action_abs <= float(atol)
    )
    result = {
        "passed": bool(passed),
        "steps": int(num_steps),
        "atol": float(atol),
        "hold_steps": hold_steps,
        "command_future_steps": int(base_env._latent_patch_future_steps),
        "window_steps": int(base_env._latent_patch_future_steps) + 1,
        "observed_phases": sorted(observed_phases),
        "missing_phases": missing_phases,
        "asynchronous_rephase_required": async_rephase_required,
        "asynchronous_rephase_exercised": async_rephase_exercised,
        "expected_policy_widths": {
            "/".join(key): width for key, width in expected_policy_widths.items()
        },
        "expected_action_width": 29,
        "max_abs_by_policy_input": max_abs_by_key,
        "max_all_policy_input_abs": all_input_max,
        "max_command_abs": command_max,
        "max_action_abs": max_action_abs,
        "policy_state_unchanged": True,
    }
    if not passed:
        raise RuntimeError(
            f"Streamed-vanilla equivalence certification failed: {result}."
        )
    return result


def _planner_command_terms(command_space: str) -> tuple[str, ...]:
    return _trajectory_command_terms(command_space)


def _current_reference_command_terms(
    base_env: ImitationEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
    env_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    past_steps = int(getattr(base_env, "_latent_patch_past_steps", 0))
    future_steps = int(getattr(base_env, "_latent_patch_future_steps", 0))
    ref_kwargs = _command_reference_kwargs(command_space, ee_body_names=ee_body_names)
    return {
        term_name: base_env.get_current_expert_window_term(
            term_name=term_name,
            past_steps=past_steps,
            future_steps=future_steps,
            env_ids=env_ids,
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
    base_env: ImitationEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
    planner_mode: str,
    planner_noise_std: float,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    terms = _planner_command_terms(command_space)
    if len(terms) == 0:
        return {}
    if planner_mode == "zero":
        return {
            term_name: torch.zeros_like(
                base_env.get_agent_trajectory_command_term(
                    term_name,
                    env_ids=env_ids,
                )
            )
            for term_name in terms
        }

    command_terms = _current_reference_command_terms(
        base_env,
        command_space=command_space,
        ee_body_names=ee_body_names,
        env_ids=env_ids,
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
    base_env: ImitationEnv,
    *,
    command_space: str,
    ee_body_names: list[str],
    planner_mode: str,
    planner_update_interval: int,
    planner_noise_std: float,
    active_mask: torch.Tensor,
    initial_publication: bool,
) -> int:
    if planner_mode == "none":
        return 0
    renew_env_ids = planner_renew_env_ids(
        base_env.episode_length_buf,
        planner_update_interval,
        initial_publication=initial_publication,
    )
    if int(renew_env_ids.numel()) == 0:
        return 0
    active_on_device = active_mask.to(device=renew_env_ids.device)
    renew_env_ids = renew_env_ids[active_on_device.index_select(0, renew_env_ids)]
    if int(renew_env_ids.numel()) == 0:
        return 0
    command_terms = _build_planner_command_terms(
        base_env,
        command_space=command_space,
        ee_body_names=ee_body_names,
        planner_mode=planner_mode,
        planner_noise_std=planner_noise_std,
        env_ids=renew_env_ids,
    )
    if len(command_terms) == 0:
        return 0
    base_env.set_agent_trajectory_command(command_terms, env_ids=renew_env_ids)
    return int(renew_env_ids.numel())


def _command_metrics(
    base_env: ImitationEnv,
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
    window_steps = past_steps + future_steps + 1
    for term_name in terms:
        command = base_env.get_current_command_window_term(
            term_name=term_name,
            past_steps=past_steps,
            future_steps=future_steps,
            **ref_kwargs,
        )
        reference = base_env.get_current_expert_window_term(
            term_name=term_name,
            past_steps=past_steps,
            future_steps=future_steps,
            **ref_kwargs,
        )
        if command.shape != reference.shape or command.shape[-1] % window_steps != 0:
            raise RuntimeError(
                f"Effective command shape mismatch for {term_name!r}: "
                f"command={tuple(command.shape)}, reference={tuple(reference.shape)}, "
                f"window_steps={window_steps}."
            )
        invalid_mask |= ~torch.isfinite(command).all(dim=-1)
        frame_width = int(command.shape[-1]) // window_steps
        command_frames = command.reshape(base_env.num_envs, window_steps, frame_width)
        reference_frames = reference.reshape(
            base_env.num_envs, window_steps, frame_width
        )
        # The paper-facing command error is the frame actually consumed by the
        # 50 Hz tracker. The full effective-window error remains diagnostic:
        # held packets necessarily have a stale/tail-padded lookahead between
        # 5 Hz publications even when their consumed current slot is exact.
        current_command = command_frames[:, past_steps, :]
        current_reference = reference_frames[:, past_steps, :]
        metrics[f"command_{term_name}_rmse"] = torch.sqrt(
            torch.mean((current_command - current_reference).square(), dim=-1)
        )
        metrics[f"command_{term_name}_effective_window_rmse"] = torch.sqrt(
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
    base_env: ImitationEnv,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


agent_entry_point = args_cli.agent_entry_point or resolve_agent_cfg_entry_point(
    args_cli.task, args_cli.algorithm
)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # One cell = one (checkpoint, output json) pair. `--checkpoint` keeps the
    # single-cell contract every existing launcher uses; `--checkpoints` adds
    # the budget axis in one process.
    if (args_cli.checkpoint is None) == (args_cli.checkpoints is None):
        raise ValueError("Pass exactly one of --checkpoint or --checkpoints.")
    if args_cli.checkpoints is not None:
        if args_cli.output_jsons is None or len(args_cli.output_jsons) != len(
            args_cli.checkpoints
        ):
            raise ValueError(
                "--checkpoints requires --output_jsons of the same length "
                f"({len(args_cli.checkpoints)} checkpoints)."
            )
        cells = [
            (path.expanduser().resolve(), out.expanduser().resolve())
            for path, out in zip(args_cli.checkpoints, args_cli.output_jsons)
        ]
    else:
        cells = [(args_cli.checkpoint.expanduser().resolve(), args_cli.output_json)]
    for path, _ in cells:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    if len(cells) > 1:
        # Both paths write one summary and stop, so they have no meaning for a
        # list of checkpoints; fail here rather than silently scoring only the
        # first cell.
        if args_cli.certify_streamed_vanilla_equivalence:
            raise ValueError(
                "--certify_streamed_vanilla_equivalence takes a single --checkpoint."
            )
        if args_cli.debug_compare_command_sources:
            raise ValueError(
                "--debug_compare_command_sources takes a single --checkpoint."
            )
        if args_cli.video:
            raise ValueError("--video takes a single --checkpoint.")
    checkpoint_path = cells[0][0]

    motion_manifest = (
        args_cli.motion_manifest.expanduser().resolve()
        if args_cli.motion_manifest is not None
        else None
    )
    if motion_manifest is not None and not motion_manifest.is_file():
        raise FileNotFoundError(f"Motion manifest not found: {motion_manifest}")

    if not hasattr(agent_cfg, "command_space"):
        raise TypeError(
            f"Agent config for {args_cli.algorithm} has no command_space field."
        )
    target_command_space = str(
        args_cli.command_space
        if args_cli.command_space is not None
        else getattr(agent_cfg, "command_space", "unknown")
    )
    low_level_command_mode = str(args_cli.low_level_command_mode)
    low_level_command_space = target_command_space
    if low_level_command_mode == "streamed_vanilla":
        if target_command_space != "full_body_trajectory":
            raise ValueError(
                "streamed_vanilla requires --command_space full_body_trajectory."
            )
        if int(args_cli.planner_update_interval) <= 0:
            raise ValueError("--planner_update_interval must be positive.")
        low_level_command_space = "single_frame_full_body"
        env_cfg.policy_command_mode = "full_body_chunk_current_slot"
    else:
        env_cfg.policy_command_mode = "reference"
    if args_cli.policy_only_checkpoint and (
        low_level_command_space != "single_frame_full_body"
    ):
        raise ValueError(
            "--policy_only_checkpoint currently requires the vanilla "
            "single_frame_full_body actor contract."
        )
    agent_cfg.command_space = low_level_command_space
    # The environment's declared command interface is the authority on the
    # actor and critic input keys, exactly as in the training entry point.
    # Without this binding an env-side `command_interface.critic_channels`
    # override (for example the critic-no-latent ablation) leaves the agent
    # asking for a critic observation the environment no longer publishes.
    if bind_command_interface(agent_cfg, env_cfg) is None:
        sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
        if callable(sync_input_keys):
            sync_input_keys()

    if args_cli.command_past_steps is not None:
        env_cfg.latent_patch_past_steps = int(args_cli.command_past_steps)
    if args_cli.command_future_steps is not None:
        env_cfg.latent_patch_future_steps = int(args_cli.command_future_steps)
    if low_level_command_mode == "streamed_vanilla":
        if int(getattr(env_cfg, "latent_patch_past_steps", 0)) != 0:
            raise ValueError("streamed_vanilla requires command_past_steps=0.")
        if int(getattr(env_cfg, "latent_patch_future_steps", 0)) + 1 < int(
            args_cli.planner_update_interval
        ):
            raise ValueError(
                "streamed_vanilla requires command_future_steps + 1 >= "
                "planner_update_interval."
            )
        env_cfg.command_hold_steps = int(args_cli.planner_update_interval)
        env_cfg.command_observation_source = (
            "planner" if args_cli.planner_mode != "none" else "planner_oracle"
        )
    elif args_cli.command_observation_source is not None:
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
    apply_motion_data(
        env_cfg,
        manifest=motion_manifest,
        cache_dir=(
            args_cli.dataset_path.expanduser().resolve()
            if args_cli.dataset_path is not None
            else None
        ),
        clips=(
            [str(args_cli.motion_name)] if args_cli.motion_name is not None else None
        ),
        cache_refresh=bool(args_cli.refresh_zarr_dataset),
        wrap_steps=False,
    )
    if hasattr(env_cfg, "reference_start_frame"):
        env_cfg.reference_start_frame = int(args_cli.reference_start_frame)
    # Pin the reference start through the shared helper rather than by writing
    # `random_reset_step_min/max` directly. Those attributes exist only on the
    # frozen v0/v1 lineage; v2 moved reset sampling to
    # `command_interface.reference.selection`, so the old `hasattr` guards were
    # silently False there and `--reference_start_frame` did nothing at all --
    # every v2 evaluation ran from a random 0-200 start while reporting a pinned
    # one. `pin_reference_start` raises on a config it cannot pin, so an
    # unrecognized surface is an error instead of a no-op.
    reference_selection_surface = pin_reference_start(
        env_cfg, start_frame=int(args_cli.reference_start_frame)
    )
    reference_channel = getattr(
        getattr(env_cfg, "command_interface", None), "reference", None
    )
    reference_selection = getattr(reference_channel, "selection", None)
    if reference_selection is not None:
        # v2 owns trajectory assignment on the reference command channel. The
        # old flat field is absent (or ignored) there, which previously made
        # --reset_schedule metadata disagree with the actual ranks.
        reference_selection.schedule = str(args_cli.reset_schedule)
    elif hasattr(env_cfg, "reset_schedule"):
        # Frozen v0/v1 surfaces retain the legacy flat field.
        env_cfg.reset_schedule = str(args_cli.reset_schedule)
    if args_cli.trajectory_rank is not None and args_cli.trajectory_ranks is not None:
        raise ValueError(
            "--trajectory_rank and --trajectory_ranks are mutually exclusive."
        )
    if args_cli.trajectory_rank is not None:
        if args_cli.motion_name is not None:
            raise ValueError(
                "--trajectory_rank and --motion_name are mutually exclusive."
            )
        trajectory_rank = int(args_cli.trajectory_rank)
        if trajectory_rank < 0:
            raise ValueError("--trajectory_rank must be non-negative.")

        def _fixed_trajectory_rank(
            env_ids: torch.Tensor, num_trajectories: int
        ) -> torch.Tensor:
            if trajectory_rank >= int(num_trajectories):
                raise ValueError(
                    f"--trajectory_rank={trajectory_rank} is outside the loaded "
                    f"dataset with {num_trajectories} trajectories."
                )
            return torch.full(
                (int(env_ids.numel()),),
                trajectory_rank,
                dtype=torch.long,
                device=env_ids.device,
            )

        if reference_selection is not None:
            reference_selection.schedule = "custom"
            reference_selection.custom_fn = _fixed_trajectory_rank
        elif hasattr(env_cfg, "reset_schedule"):
            env_cfg.reset_schedule = "custom"
            env_cfg.custom_reset_fn = _fixed_trajectory_rank
        else:
            raise RuntimeError("The environment has no trajectory selection surface.")
    elif args_cli.trajectory_ranks is not None:
        if args_cli.motion_name is not None:
            raise ValueError(
                "--trajectory_ranks and --motion_name are mutually exclusive."
            )
        env_rank_assignment = build_env_rank_assignment(
            args_cli.trajectory_ranks, int(args_cli.num_envs)
        )

        def _fixed_trajectory_ranks(
            env_ids: torch.Tensor, num_trajectories: int
        ) -> torch.Tensor:
            if max(env_rank_assignment) >= int(num_trajectories):
                raise ValueError(
                    "--trajectory_ranks contains a rank outside the loaded dataset "
                    f"with {num_trajectories} trajectories."
                )
            rank_table = torch.as_tensor(
                env_rank_assignment, dtype=torch.long, device=env_ids.device
            )
            return rank_table[env_ids.long()]

        if reference_selection is not None:
            reference_selection.schedule = "custom"
            reference_selection.custom_fn = _fixed_trajectory_ranks
        elif hasattr(env_cfg, "reset_schedule"):
            env_cfg.reset_schedule = "custom"
            env_cfg.custom_reset_fn = _fixed_trajectory_ranks
        else:
            raise RuntimeError("The environment has no trajectory selection surface.")
    if not args_cli.enable_observation_corruption:
        _disable_observation_corruption(env_cfg)
    # Domain randomization was previously left entirely live here while
    # `sim2sim_backend_eval.py` disabled it by default, so the two tools
    # reported MPJPE in different regimes and neither number said which.
    randomization_kept = apply_randomization_profile(env_cfg, args_cli.randomization)
    if args_cli.disable_early_terminations:
        # The AGENTS.md full-horizon diagnostic: every early termination off,
        # including `base_too_low`, so MPJPE is measured over the intended
        # horizon rather than a termination-truncated rollout. Diagnostic only
        # -- it cannot satisfy oracle qualification, which needs the strict
        # protocol with every term active.
        terminations = getattr(env_cfg, "terminations", None)
        for term_name in (
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "foot_pos_xyz",
            "base_too_low",
        ):
            if terminations is not None and getattr(terminations, term_name, None):
                setattr(terminations, term_name, None)
    if args_cli.certify_streamed_vanilla_equivalence:
        # Certification isolates command/action equivalence from controller
        # quality. Normal evaluation keeps the strict tracking terminations.
        terminations = getattr(env_cfg, "terminations", None)
        for term_name in (
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "base_too_low",
        ):
            if terminations is not None and hasattr(terminations, term_name):
                setattr(terminations, term_name, None)

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

    raw_env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError(
            "DirectMARLEnv is not supported for RLOpt evaluation."
        )
    video_dir: Path | None = None
    if args_cli.video:
        video_dir = (
            (
                args_cli.video_dir
                or Path("logs/checkpoint_eval/videos") / (args_cli.label or "eval")
            )
            .expanduser()
            .resolve()
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=int(args_cli.video_length),
            name_prefix=str(args_cli.label or "eval"),
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
        transform=Compose(RewardSum(), StepCounter(args_cli.steps + 2)),
    )
    base_env = _unwrap_imitation_env(env)

    command_space = target_command_space
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

    l2t_policy_role = args_cli.ipmd_l2t_policy_role
    if l2t_policy_role is not None:
        if args_cli.algorithm != "IPMD":
            raise ValueError("--ipmd_l2t_policy_role requires --algo IPMD.")
        if not hasattr(agent_cfg, "ipmd_l2t"):
            raise ValueError(
                "--ipmd_l2t_policy_role requires an IPMD-L2T agent entry point."
            )
        if args_cli.policy_only_checkpoint:
            raise ValueError(
                "--ipmd_l2t_policy_role cannot be combined with "
                "--policy_only_checkpoint."
            )
        agent_class = IPMDL2T
    else:
        agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)
    # One Isaac Sim start serves every cell: the arm fixes the interface, so
    # only the policy weights change between checkpoints.
    first_cell_ranks: torch.Tensor | None = None
    for cell_index, (checkpoint_path, output_json_target) in enumerate(cells):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        # A list of cells shares one `--label`, so the row is named after the
        # output file it lands in, the way the campaign launchers name theirs.
        cell_label = args_cli.label
        if len(cells) > 1:
            cell_label = (
                output_json_target.stem
                if output_json_target is not None
                else f"{args_cli.label or 'eval'}_cell{cell_index}"
            )
        tracker_provenance: dict[str, Any] | None = None
        skill_encoder_provenance: dict[str, Any] | None = None
        policy_only_checkpoint = bool(
            args_cli.policy_only_checkpoint
            or low_level_command_mode == "streamed_vanilla"
        )
        if policy_only_checkpoint:
            frozen_tracker = load_frozen_low_level_tracker(
                agent,
                checkpoint_path,
                expected_input_keys=VANILLA_POLICY_INPUT_KEYS,
                map_location=env_cfg.sim.device,
            )
            collector_policy = frozen_tracker.policy
            tracker_provenance = frozen_tracker.provenance
        else:
            agent.load_model(str(checkpoint_path))
            skill_encoder_provenance = _skill_encoder_provenance(
                agent,
                checkpoint_path,
                source=str(args_cli.skill_encoder_source),
                map_location=env_cfg.sim.device,
            )
            if skill_encoder_provenance is not None:
                print(f"[INFO] skill encoder: {skill_encoder_provenance}")
            if l2t_policy_role == "teacher":
                collector_policy = agent.teacher_policy
            elif l2t_policy_role == "student":
                collector_policy = agent.deployment_policy
            else:
                collector_policy = agent.collector_policy
            collector_policy.eval()

        if args_cli.certify_streamed_vanilla_equivalence:
            result = _certify_streamed_vanilla_equivalence(
                env,
                base_env,
                collector_policy,
                num_steps=int(args_cli.equivalence_steps),
                atol=float(args_cli.equivalence_atol),
            )
            result["low_level_tracker"] = tracker_provenance
            result["checkpoint"] = str(checkpoint_path)
            result["checkpoint_sha256"] = _file_sha256(checkpoint_path)
            result["motion_manifest"] = (
                str(args_cli.motion_manifest.expanduser().resolve())
                if args_cli.motion_manifest is not None
                else None
            )
            result["motion_manifest_sha256"] = (
                _file_sha256(args_cli.motion_manifest.expanduser().resolve())
                if args_cli.motion_manifest is not None
                else None
            )
            result["dataset_path"] = (
                str(args_cli.dataset_path.expanduser().resolve())
                if args_cli.dataset_path is not None
                else str(getattr(env_cfg, "dataset_path", ""))
            )
            if output_json_target is not None:
                output_json = output_json_target.expanduser().resolve()
                output_json.parent.mkdir(parents=True, exist_ok=True)
                output_json.write_text(
                    json.dumps(result, indent=2, default=_json_default) + "\n",
                    encoding="utf-8",
                )
            print(
                "[PASS] Streamed vanilla equivalence: "
                f"command_max={result['max_command_abs']:.3e} "
                f"action_max={result['max_action_abs']:.3e}."
            )
            env.close()
            return

        num_envs = int(args_cli.num_envs)
        active = torch.ones(num_envs, dtype=torch.bool)
        survival_steps = torch.zeros(num_envs, dtype=torch.float32)
        return_sum = torch.zeros(num_envs, dtype=torch.float32)
        done_events = torch.zeros(num_envs, dtype=torch.float32)
        terminated_events = torch.zeros(num_envs, dtype=torch.float32)
        truncated_events = torch.zeros(num_envs, dtype=torch.float32)
        trajectory_ranks = torch.full((num_envs,), -1, dtype=torch.long)
        motion_names = [f"env_{env_id}" for env_id in range(num_envs)]
        termination_term_names = list(base_env.termination_manager.active_terms)
        termination_hits = {
            term_name: torch.zeros(num_envs, dtype=torch.bool)
            for term_name in termination_term_names
        }
        # Push-to-termination attribution: live only when the profile keeps the
        # interval push (``--randomization all``); attach returns None otherwise.
        push_tracker = attach_push_tracker(
            getattr(base_env, "event_manager", None), num_envs
        )
        tracking_failure_term_names: list[str] = []
        for term_name in termination_term_names:
            term_cfg = base_env.termination_manager.get_term_cfg(term_name)
            if not term_cfg.time_out and term_name != "reference_finished":
                tracking_failure_term_names.append(term_name)
        metric_stats: dict[str, dict[str, float]] = {}
        per_env_tracking_metric_sums: dict[str, torch.Tensor] = {}
        per_env_tracking_metric_counts: dict[str, torch.Tensor] = {}
        previous_action: torch.Tensor | None = None
        previous_previous_action: torch.Tensor | None = None
        previous_body_lin_vel: tuple[torch.Tensor, torch.Tensor] | None = None
        previous_velocity_valid = torch.zeros(num_envs, dtype=torch.bool)
        previous_actual_acc: torch.Tensor | None = None
        previous_acc_valid = torch.zeros(num_envs, dtype=torch.bool)
        # Last active-step root drift per env: with the mean this gives the
        # drift ACCUMULATION ratio (final/mean ~1 = bounded servo, ~2 =
        # uncorrected linear drift). SONIC's rows carry the analogous
        # anchor_pos_err_final_m from the command term.
        final_root_drift = torch.zeros(num_envs, dtype=torch.float64)
        steps_executed = 0
        valid_transition_count = 0
        planner_publish_count = 0
        dt = float(getattr(base_env, "step_dt", 0.0) or 0.0)

        # The rollout steps the environment inside `torch.inference_mode()`, so
        # the expert data plane's reference-row buffers are allocated as
        # inference tensors during the FIRST cell. A later cell's reset writes
        # those same buffers with `index_copy_`, which torch refuses outside
        # inference mode. The first cell keeps today's exact reset so the
        # single-cell path is untouched.
        with contextlib.nullcontext() if cell_index == 0 else torch.inference_mode():
            td = env.reset()
        trajectory_manager = base_env.trajectory_manager
        env_traj_rank = getattr(trajectory_manager, "env_traj_rank", None)
        if (
            isinstance(env_traj_rank, torch.Tensor)
            and env_traj_rank.numel() >= num_envs
        ):
            trajectory_ranks = env_traj_rank[:num_envs].detach().cpu().long().clone()
            ordered_motion_names = base_env.expert_trajectory_motion_names()
            if first_cell_ranks is None:
                first_cell_ranks = trajectory_ranks.clone()
            elif not torch.equal(trajectory_ranks, first_cell_ranks):
                raise RuntimeError(
                    "Trajectory ranks changed between cells: cell "
                    f"{cell_index} reset onto a different population, so its "
                    "row is not comparable with the first cell's."
                )
            motion_names = [
                ordered_motion_names[int(rank)]
                if 0 <= int(rank) < len(ordered_motion_names)
                else f"trajectory_rank_{int(rank)}"
                for rank in trajectory_ranks.tolist()
            ]
        # DETERMINISTIC is the protocol. RANDOM exists only to reproduce the way
        # training draws actions, so the train/eval metric gap can be attributed
        # rather than argued about; see --action_sampling.
        action_sampling = str(args_cli.action_sampling)
        rollout_exploration_type = (
            InteractionType.RANDOM
            if action_sampling == "stochastic"
            else InteractionType.DETERMINISTIC
        )
        print(
            f"[INFO] Starting {action_sampling} evaluation: "
            f"num_envs={num_envs}, steps={args_cli.steps}, command_space={command_space}"
        )

        def _accumulate_per_env(
            metric_name: str, metric_values_cpu: torch.Tensor, mask: torch.Tensor
        ) -> None:
            values = metric_values_cpu.reshape(num_envs, -1).mean(dim=-1).double()
            valid = mask & torch.isfinite(values)
            if metric_name not in per_env_tracking_metric_sums:
                per_env_tracking_metric_sums[metric_name] = torch.zeros(
                    num_envs, dtype=torch.float64
                )
                per_env_tracking_metric_counts[metric_name] = torch.zeros(
                    num_envs, dtype=torch.int64
                )
            per_env_tracking_metric_sums[metric_name] += torch.where(
                valid, values, torch.zeros_like(values)
            )
            per_env_tracking_metric_counts[metric_name] += valid.to(torch.int64)

        for step_idx in range(int(args_cli.steps)):
            step_active = active.clone()
            if not bool(step_active.any()):
                break

            published_rows = _maybe_publish_planner_command(
                base_env,
                command_space=command_space,
                ee_body_names=ee_body_names,
                planner_mode=args_cli.planner_mode,
                planner_update_interval=int(args_cli.planner_update_interval),
                planner_noise_std=float(args_cli.planner_noise_std),
                active_mask=step_active,
                initial_publication=step_idx == 0,
            )
            planner_publish_count += published_rows
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
                set_exploration_type(rollout_exploration_type),
            ):
                td = collector_policy(td)

            action = td.get("action")
            if action is None:
                raise RuntimeError("Policy did not write an 'action' tensor.")
            action_2d = action.detach().reshape(num_envs, -1)
            action_l2 = torch.linalg.vector_norm(action_2d, dim=-1).cpu()
            _accumulate_metric(metric_stats, "action_l2", action_l2, step_active)
            # Per-environment too (2026-08-29): without it the action-space
            # measures exist only as board-wide all-transition means and can
            # never join the success-only row every paper metric reports on.
            _accumulate_per_env("action_l2", action_l2, step_active)
            action_cpu = action_2d.cpu()
            if previous_action is not None:
                action_delta_l2 = torch.linalg.vector_norm(
                    action_cpu - previous_action, dim=-1
                )
                _accumulate_metric(
                    metric_stats, "action_delta_l2", action_delta_l2, step_active
                )
                _accumulate_per_env("action_delta_l2", action_delta_l2, step_active)
                if dt > 0.0:
                    _accumulate_metric(
                        metric_stats,
                        "action_rate_l2",
                        action_delta_l2 / dt,
                        step_active,
                    )
                if previous_previous_action is not None:
                    # Reference-free second difference: buzz, not authority.
                    # A fast sustained move has a large first difference and a
                    # small second difference; per-step trembling has both.
                    action_jerk_l2 = torch.linalg.vector_norm(
                        action_cpu - 2.0 * previous_action + previous_previous_action,
                        dim=-1,
                    )
                    _accumulate_metric(
                        metric_stats, "action_jerk_l2", action_jerk_l2, step_active
                    )
                    _accumulate_per_env("action_jerk_l2", action_jerk_l2, step_active)
            previous_previous_action = previous_action
            previous_action = action_cpu

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
            terminal_step_mask = done_any & step_active
            step_term_masks: dict[str, torch.Tensor] = {}
            for term_name in termination_term_names:
                term_values = (
                    base_env.termination_manager.get_term(term_name)
                    .detach()
                    .reshape(-1)[:num_envs]
                    .to(device="cpu", dtype=torch.bool)
                )
                step_term_masks[term_name] = term_values & terminal_step_mask
                termination_hits[term_name] |= step_term_masks[term_name]
            if push_tracker is not None:
                push_tracker.on_terminal(
                    terminateds & step_active,
                    done_any,
                    int(base_env.common_step_counter),
                    step_term_masks,
                )
            return_sum += rewards.float() * step_active.float()
            survival_steps += step_active.float()
            done_events += (done_any & step_active).float()
            terminated_events += (terminateds & step_active).float()
            truncated_events += (truncateds & step_active).float()

            metric_mask = (
                step_active if args_cli.keep_after_done else step_active & ~done_any
            )
            valid_transition_count += int(metric_mask.sum().item())
            tracking, body_lin_vel = _tracking_metrics(
                base_env,
                tracked_body_names=tracked_body_names,
                ee_body_names=ee_body_names,
            )
            for metric_name, metric_values in tracking.items():
                metric_values_cpu = metric_values.cpu()
                _accumulate_metric(
                    metric_stats, metric_name, metric_values_cpu, metric_mask
                )
                _accumulate_per_env(metric_name, metric_values_cpu, metric_mask)
                if metric_name == "root_pos_xyz_error_m":
                    drift_now = metric_values_cpu.reshape(num_envs, -1)[:, 0].double()
                    final_root_drift = torch.where(
                        metric_mask & torch.isfinite(drift_now),
                        drift_now,
                        final_root_drift,
                    )
            if body_lin_vel is not None and dt > 0.0:
                if previous_body_lin_vel is not None:
                    actual_lin_vel, ref_lin_vel = body_lin_vel
                    prev_actual_lin_vel, prev_ref_lin_vel = previous_body_lin_vel
                    actual_acc = (actual_lin_vel - prev_actual_lin_vel) / float(dt)
                    ref_acc = (ref_lin_vel - prev_ref_lin_vel) / float(dt)
                    acceleration_distance = torch.linalg.vector_norm(
                        actual_acc - ref_acc, dim=-1
                    ).mean(dim=-1)
                    acceleration_mask = metric_mask & previous_velocity_valid
                    acceleration_distance_cpu = acceleration_distance.cpu()
                    _accumulate_metric(
                        metric_stats,
                        "tracking_acceleration_distance_mps2",
                        acceleration_distance_cpu,
                        acceleration_mask,
                    )
                    # Per environment too: without this the metric exists only as a
                    # board-wide, all-transition mean, so it cannot join the
                    # success-only row that every other paper metric is reported on.
                    _accumulate_per_env(
                        "tracking_acceleration_distance_mps2",
                        acceleration_distance_cpu,
                        acceleration_mask,
                    )
                    # Reference-FREE smoothness (2026-08-29). The distance
                    # metric above is an error against the reference: copying
                    # a jerky clip scores 0 on it. `body_acc_mps2` is the
                    # robot's own acceleration magnitude, and `body_jerk_mps3`
                    # its finite difference — what "smooth" means physically,
                    # independent of what the clip does.
                    body_acc = torch.linalg.vector_norm(actual_acc, dim=-1).mean(
                        dim=-1
                    )
                    body_acc_cpu = body_acc.cpu()
                    _accumulate_metric(
                        metric_stats, "body_acc_mps2", body_acc_cpu, acceleration_mask
                    )
                    _accumulate_per_env(
                        "body_acc_mps2", body_acc_cpu, acceleration_mask
                    )
                    if previous_actual_acc is not None:
                        body_jerk = torch.linalg.vector_norm(
                            (actual_acc - previous_actual_acc) / float(dt), dim=-1
                        ).mean(dim=-1)
                        jerk_mask = acceleration_mask & previous_acc_valid
                        body_jerk_cpu = body_jerk.cpu()
                        _accumulate_metric(
                            metric_stats, "body_jerk_mps3", body_jerk_cpu, jerk_mask
                        )
                        _accumulate_per_env(
                            "body_jerk_mps3", body_jerk_cpu, jerk_mask
                        )
                    previous_actual_acc = actual_acc.clone()
                    # Acc at this step is usable as a jerk operand next step
                    # only if the velocity pair behind it was valid and the
                    # episode continues.
                    previous_acc_valid = (
                        previous_velocity_valid & step_active & ~done_any
                    )
                previous_body_lin_vel = (
                    body_lin_vel[0].clone(),
                    body_lin_vel[1].clone(),
                )
                previous_velocity_valid = step_active & ~done_any

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
        tracking_failure = torch.zeros(num_envs, dtype=torch.bool)
        for term_name in tracking_failure_term_names:
            tracking_failure |= termination_hits[term_name]
        tracking_success = active_mask & ~tracking_failure
        reference_finished = termination_hits.get(
            "reference_finished", torch.zeros(num_envs, dtype=torch.bool)
        )
        completed_tracking_success = tracking_success & reference_finished
        successful_metrics: dict[str, dict[str, float | int]] = {}
        for metric_name, per_env_sums in per_env_tracking_metric_sums.items():
            per_env_counts = per_env_tracking_metric_counts[metric_name]
            selected = completed_tracking_success & (per_env_counts > 0)
            selected_count = int(per_env_counts[selected].sum().item())
            successful_metrics[metric_name] = {
                "mean": (
                    float(per_env_sums[selected].sum().item() / selected_count)
                    if selected_count > 0
                    else float("nan")
                ),
                "count": selected_count,
                "num_successful_envs": int(selected.sum().item()),
            }

        def _term_rate(term_name: str) -> float:
            if num_evaluated_envs == 0 or term_name not in termination_hits:
                return float("nan")
            return float(termination_hits[term_name][active_mask].float().mean().item())

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
            "tracking_success_rate": float(
                tracking_success[active_mask].float().mean().item()
            )
            if num_evaluated_envs > 0
            else float("nan"),
            "tracking_failure_rate": float(
                tracking_failure[active_mask].float().mean().item()
            )
            if num_evaluated_envs > 0
            else float("nan"),
            "completed_tracking_success_rate": float(
                completed_tracking_success[active_mask].float().mean().item()
            )
            if num_evaluated_envs > 0
            else float("nan"),
            "completed_requested_horizon_rate": float(
                (done_events[active_mask] == 0).float().mean().item()
            )
            if num_evaluated_envs > 0
            else float("nan"),
            "reference_finished_rate": _term_rate("reference_finished"),
            "time_out_rate": _term_rate("time_out"),
            "termination_term_rates": {
                term_name: _term_rate(term_name) for term_name in termination_term_names
            },
            "termination_cause_env_counts": {
                term_name: int(termination_hits[term_name][active_mask].sum().item())
                for term_name in termination_term_names
            },
            "steps_executed": int(steps_executed),
            "valid_transition_count": int(valid_transition_count),
            "num_evaluated_envs": int(num_evaluated_envs),
            "planner_publish_count": int(planner_publish_count),
        }
        if push_tracker is not None:
            aggregate["push_attribution"] = push_tracker.summary()
        summary = {
            "metadata": {
                "label": cell_label,
                "video_dir": str(video_dir) if video_dir is not None else None,
                "task": args_cli.task,
                "algorithm": args_cli.algorithm,
                "ipmd_l2t_policy_role": l2t_policy_role,
                "checkpoint": str(checkpoint_path),
                "motion_manifest": str(motion_manifest)
                if motion_manifest is not None
                else None,
                "motion_name": args_cli.motion_name,
                "trajectory_rank": args_cli.trajectory_rank,
                "trajectory_ranks": args_cli.trajectory_ranks,
                "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
                "command_space": command_space,
                "low_level_command_mode": low_level_command_mode,
                "low_level_command_space": low_level_command_space,
                "policy_only_checkpoint": policy_only_checkpoint,
                "policy_command_mode": str(
                    getattr(base_env, "policy_command_mode", "unknown")
                ),
                "low_level_tracker": tracker_provenance,
                # Which skill encoder actually drove the rollout. Mandatory for a
                # co-trained arm, whose encoder lives inside the tracker checkpoint.
                "skill_encoder": skill_encoder_provenance,
                "command_observation_source": str(
                    getattr(base_env, "_command_observation_source", "unknown")
                ),
                "command_past_steps": int(
                    getattr(base_env, "_latent_patch_past_steps", 0)
                ),
                "command_future_steps": int(
                    getattr(base_env, "_latent_patch_future_steps", 0)
                ),
                "num_envs": int(num_envs),
                "steps_requested": int(args_cli.steps),
                "seed": agent_cfg.seed,
                "reset_schedule": str(
                    getattr(
                        reference_selection,
                        "schedule",
                        getattr(env_cfg, "reset_schedule", args_cli.reset_schedule),
                    )
                ),
                "reference_start_frame": int(args_cli.reference_start_frame),
                "keep_after_done": bool(args_cli.keep_after_done),
                "observation_corruption_enabled": bool(
                    args_cli.enable_observation_corruption
                ),
                "policy_observation_corruption_enabled": bool(
                    getattr(
                        getattr(getattr(env_cfg, "observations", None), "policy", None),
                        "enable_corruption",
                        False,
                    )
                ),
                "wrap_steps": bool(getattr(env_cfg, "wrap_steps", False)),
                "early_terminations_enabled": not bool(
                    args_cli.disable_early_terminations
                ),
                # Which reset-sampling surface was actually pinned. Recorded because
                # v2 and the frozen v0/v1 lineage expose different ones, and writing
                # the wrong one used to fail silently.
                "reference_selection_surface": reference_selection_surface,
                "randomization_profile": args_cli.randomization,
                "randomization_kept": randomization_kept,
                "action_sampling": action_sampling,
                "qualification_eligible": not bool(args_cli.disable_early_terminations),
                "time_out_enabled": True,
                "episode_length_extension_enabled": not bool(
                    args_cli.preserve_episode_length
                ),
                "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
                "reward_clipping_enabled": False,
                "push_perturbation": interval_event_metadata(env_cfg, "push_robot"),
                "planner_mode": args_cli.planner_mode,
                "planner_update_interval": int(args_cli.planner_update_interval),
                "planner_noise_std": float(args_cli.planner_noise_std),
                "tracked_body_names": tracked_body_names,
                "ee_body_names": ee_body_names,
            },
            "aggregate": aggregate,
            "metrics": _finalize_metric_stats(metric_stats),
            "successful_metrics": successful_metrics,
            "max_steps": int(args_cli.steps),
            "steps_run": int(steps_executed),
            "stop_reason": (
                "max_steps"
                if int(steps_executed) == int(args_cli.steps)
                else "all_envs_done"
            ),
            "per_environment": [
                {
                    "env_id": env_id,
                    "trajectory_rank": int(trajectory_ranks[env_id].item()),
                    "motion_name": motion_names[env_id],
                    "return_sum": float(return_sum[env_id].item()),
                    "survival_steps": int(survival_steps[env_id].item()),
                    "done": bool(done_events[env_id].item() > 0),
                    "terminated": bool(terminated_events[env_id].item() > 0),
                    "truncated": bool(truncated_events[env_id].item() > 0),
                    "tracking_success": bool(tracking_success[env_id].item()),
                    "completed_tracking_success": bool(
                        completed_tracking_success[env_id].item()
                    ),
                    "termination_terms": [
                        term_name
                        for term_name in termination_term_names
                        if bool(termination_hits[term_name][env_id].item())
                    ],
                    "tracking_metrics": {
                        metric_name: (
                            float(
                                per_env_tracking_metric_sums[metric_name][env_id].item()
                            )
                            / int(
                                per_env_tracking_metric_counts[metric_name][
                                    env_id
                                ].item()
                            )
                        )
                        for metric_name in per_env_tracking_metric_sums
                        if int(
                            per_env_tracking_metric_counts[metric_name][env_id].item()
                        )
                        > 0
                    }
                    | {
                        # Root drift at the LAST active step; with the mean
                        # above this gives the accumulation ratio.
                        "root_pos_xyz_error_final_m": float(
                            final_root_drift[env_id].item()
                        )
                    },
                    "tracking_metric_counts": {
                        metric_name: int(
                            per_env_tracking_metric_counts[metric_name][env_id].item()
                        )
                        for metric_name in per_env_tracking_metric_counts
                        if int(
                            per_env_tracking_metric_counts[metric_name][env_id].item()
                        )
                        > 0
                    },
                }
                for env_id in range(num_envs)
                if bool(active_mask[env_id].item())
            ],
        }

        output_json = output_json_target
        if output_json is None:
            label = cell_label or command_space
            output_json = checkpoint_path.parent / "evaluation" / f"{label}_eval.json"
        output_json = output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Wrote JSON summary: {output_json}")
        if args_cli.output_csv is not None:
            _write_csv(summary, args_cli.output_csv, append=args_cli.append_csv)

        metrics = summary["metrics"]
        successful_mpjpe_l_mm = (
            summary["successful_metrics"]
            .get("tracking_mpjpe_mm", {})
            .get("mean", float("nan"))
        )
        print(
            "[RESULT] "
            f"command_space={command_space} "
            f"return_sum_mean={aggregate['return_sum_mean']:.4f} "
            f"survival_steps_mean={aggregate['survival_steps_mean']:.1f} "
            f"done_rate={aggregate['done_rate']:.3f} "
            f"completed_tracking_success_rate="
            f"{aggregate['completed_tracking_success_rate']:.3f} "
            f"successful_mpjpe_l_mm={successful_mpjpe_l_mm:.3f} "
            f"joint_pos_rmse_rad={metrics.get('joint_pos_rmse_rad', {}).get('mean', float('nan')):.4f} "
            f"ee_pos_error_m={metrics.get('ee_pos_error_m', {}).get('mean', float('nan')):.4f}"
        )
    env.close()


if __name__ == "__main__":
    # Print and flush BEFORE `simulation_app.close()`. Kit's close() terminates
    # the process itself, discarding any pending traceback and returning 0, so
    # without this an evaluation that crashed was indistinguishable from one
    # that passed -- including to any gate that shells out and checks $?.
    _exit_code = 0
    try:
        main()
    except BaseException:
        import traceback as _traceback

        _traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        _exit_code = 1
    finally:
        simulation_app.close()
    sys.exit(_exit_code)
