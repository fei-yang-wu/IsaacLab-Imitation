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

# CU130 split-runtime bootstrap (ICE only): Kit's bundled CPython stdlib shadows
# the runtime python's on sys.path, and its platform.py cannot parse the
# conda-forge sys.version banner. Scrub Kit's stdlib and install the kit-import
# guard before importing isaaclab. Guarded by ISAACLAB_SPLIT_RUNTIME, so local
# runs are completely unaffected.
import os as _os

_actor_trace: list = []
_target_probe: list = []
_actor_trace_n = [0]

if _os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1":
    # Drop Kit's stdlib dir (its platform.py can't parse the runtime python's
    # conda-forge sys.version banner) but KEEP its site-packages (lazy_loader/
    # hydra/omegaconf, absent from the runtime env). No kit-import guard: this
    # script uses IsaacLab's AppLauncher (which references isaacsim) and runs
    # headless Newton without the RT renderer on compute-only GPUs.
    _KIT_PY = "/isaac-sim/kit/python"
    sys.path[:] = [
        _p
        for _p in sys.path
        if not (
            _os.path.realpath(_p or ".").startswith(_KIT_PY)
            and "site-packages" not in _os.path.realpath(_p or ".")
        )
    ]

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
        "IPMD",
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
    "--packet_prediction_horizon_steps",
    type=int,
    default=0,
    help=(
        "Planner packet prediction horizon. Zero uses the execution window. "
        "For a long-horizon explicit planner, the packet is reduced to the "
        "execution window by first-window slicing or temporal ensembling."
    ),
)
parser.add_argument(
    "--packet_temporal_ensemble",
    choices=("none", "exponential"),
    default="none",
    help=(
        "Reduce a long-horizon explicit packet to the execution window by "
        "discarding future slots or by anchor-aware exponential overlap "
        "ensembling."
    ),
)
parser.add_argument(
    "--packet_temporal_ensemble_decay",
    type=float,
    default=0.5,
    help="Age decay for --packet_temporal_ensemble=exponential.",
)
parser.add_argument(
    "--packet_anchor_conversion",
    choices=("none", "torso_to_pelvis"),
    default="none",
    help=(
        "Convert a torso-relative root_qpos packet into the pelvis-relative "
        "Strict tracker contract using predicted waist-qpos forward kinematics."
    ),
)
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
    "--in_step_publication",
    action="store_true",
    default=False,
    help=(
        "DIAGNOSTIC. Publish planner packets from inside the observation pass "
        "instead of from the outer loop. Measured no benefit (156.58 vs 156.85 "
        "mm) and it BYPASSES the loop's rollout-sample collection, so it must "
        "stay off for any run using --save_rollout_training_samples."
    ),
)
parser.add_argument(
    "--no_in_step_publication",
    dest="in_step_publication",
    action="store_false",
    help="Publish from the outer loop (historical behaviour).",
)
parser.add_argument(
    "--use_command_publisher",
    action="store_true",
    default=False,
    help=(
        "Route publication through the shared CommandPublisher (option b). It "
        "owns joint-order pinning and the renewal schedule, so both interfaces "
        "share one control plane. Must reproduce the legacy path exactly."
    ),
)
parser.add_argument(
    "--publish_phase_offset",
    type=int,
    default=0,
    help=(
        "Shift the publication schedule by this many control steps. The loop "
        "publishes before env.step(), so it reads episode_length_buf BEFORE the "
        "increment while the env computes the hold phase AFTER it; offset=1 "
        "aligns the publication with the env's phase-zero observation."
    ),
)
parser.add_argument(
    "--atomic_command_anchor",
    action="store_true",
    default=False,
    help=(
        "Capture the held command anchor pose at the instant the packet is "
        "published, matching the env-filled oracle path. Without this the "
        "published chunk is re-expressed against a stale anchor and the root "
        "command is systematically wrong."
    ),
)
parser.add_argument(
    "--no_atomic_command_anchor",
    dest="atomic_command_anchor",
    action="store_false",
    help="Disable atomic anchor capture (reproduces the historical behaviour).",
)
parser.add_argument(
    "--pin_command_joint_order",
    type=str,
    choices=("auto", "on", "off"),
    default="auto",
    help=(
        "Re-index published full-body command packets from the live "
        "articulation order into the pinned G1_29DOF_ISAACLAB_JOINT_NAMES order "
        "the env consumes. Without this the tracker receives every joint target "
        "assigned to the wrong joint. 'auto' enables it for full_body_trajectory "
        "and is the correct setting; 'off' reproduces the historical "
        "unpinned-publication behaviour for A/B diagnosis."
    ),
)
parser.add_argument(
    "--oracle_substitute",
    type=str,
    default="none",
    help=(
        "DIAGNOSTIC ONLY. Comma-separated channel groups of the published "
        "command packet to overwrite with expert (ground-truth) values at each "
        "publication: 'qpos', 'qvel', 'root_anchor', or 'all'; 'none' disables "
        "substitution. This leaks oracle information into the planner row, so a "
        "substituted run is an UPPER BOUND on that interface and must never be "
        "reported as a planner result. The chosen mask is recorded in "
        "summary.json under 'oracle_substitution'."
    ),
)
parser.add_argument(
    "--no_per_step_metrics",
    action="store_true",
    help=(
        "Skip the per-step metric .npz written next to summary.json. Retained "
        "by default so fall time and alternative failure thresholds can be "
        "re-derived without re-running; it is a few hundred KB per eval."
    ),
)
parser.add_argument(
    "--allow_shorter_planner_interval",
    action="store_true",
    default=False,
    help=(
        "C3 freshness studies only: permit a runtime planner_interval_steps "
        "SHORTER than the checkpoint's, consuming only the freshest slots of "
        "each packet. Records the deviation in the summary; every other "
        "streamed-vanilla contract term stays hard, and a LONGER interval is "
        "still refused. A run using this is not matched-contract."
    ),
)
parser.add_argument(
    "--allow_cross_tracker_planner",
    action="store_true",
    default=False,
    help=(
        "Diagnostic only: deploy an explicit packet planner whose collection "
        "provenance names a different frozen tracker. Interface, causal-input, "
        "horizon, seed, and publication-rate checks remain strict; every waived "
        "tracker-contract field is recorded in the summary."
    ),
)
parser.add_argument(
    "--fall_height_m",
    type=float,
    default=0.4,
    help=(
        "Absolute torso height below which an environment counts as fallen. "
        "Detected from raw body height, independently of the termination "
        "manager, so it stays valid when tracking terminations are disabled. "
        "Default matches the injected base_too_low term."
    ),
)
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
    "--base_only_termination",
    action="store_true",
    default=False,
    help=(
        "Keep only base_too_low (plus reference end), disabling tracking-error "
        "and timeout terms. Metrics therefore contain only pre-fall transitions."
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
from isaaclab_imitation.envs.imitation_rl_env_legacy import ImitationRLEnvLegacy
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_EE_BODY_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    command_space_policy_input_keys,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMD, PPO, SAC
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

from imitation_experiments.lowlevel.low_level_tracker import (
    load_frozen_low_level_tracker,
)  # noqa: E402
from imitation_experiments.data.balanced_motion_rows import BalancedMotionRowSelector  # noqa: E402
from imitation_experiments.provenance.paper_protocol_metadata import (
    interval_event_metadata,
)  # noqa: E402
from imitation_experiments.planner.planner_latency import PlannerForwardTimer  # noqa: E402


TRACKING_TERMINATION_NAMES = ("anchor_pos", "anchor_ori", "ee_body_pos")
FALL_TERMINATION_NAME = "base_too_low"
# Body whose world height defines a fall. Kept identical to the asset_cfg of the
# injected base_too_low term so the raw-height detector and the termination term
# cannot disagree about what "fallen" means.
FALL_TERMINATION_BODY_NAME = "torso_link"
# Tracking metrics that are also reported truncated at the first fall, as
# `<name>_prefall`. Restricted to the headline tracking errors: every metric
# would double the summary for no interpretive gain.
FALL_TRUNCATED_METRIC_NAMES = (
    "tracking_mpjpe_mm",
    "tracked_body_pos_error_m",
    "root_pos_xyz_error_m",
    "root_ori_error_rad",
    "joint_pos_rmse_rad",
    "tracking_failure",
)
# Metrics written per step to the retained `.npz`. Wider than the truncated set
# because the threshold sweep needs root height *and* orientation error, which
# are what the failure definition is built from.
PER_STEP_RETAINED_METRIC_NAMES = (
    "tracking_mpjpe_mm",
    "tracked_body_pos_error_m",
    "root_pos_xyz_error_m",
    "root_height_error_m",
    "root_ori_error_rad",
    "joint_pos_rmse_rad",
    "tracking_failure",
)


def _disable_tracking_terminations(terminations: Any) -> list[str]:
    disabled: list[str] = []
    for name in TRACKING_TERMINATION_NAMES:
        if hasattr(terminations, name) and getattr(terminations, name) is not None:
            setattr(terminations, name, None)
            disabled.append(name)
    return disabled


def _configure_base_only_termination(
    terminations: Any, *, minimum_height: float
) -> list[str]:
    if minimum_height <= 0.0:
        raise ValueError("--fall_height_m must be positive.")

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as _DoneTerm
    from isaaclab_imitation.tasks.manager_based.imitation import mdp as _imitation_mdp

    terminations.base_too_low = _DoneTerm(
        func=_imitation_mdp.root_height_below_minimum,
        params={
            "minimum_height": float(minimum_height),
            "asset_cfg": SceneEntityCfg("robot", body_names=FALL_TERMINATION_BODY_NAME),
        },
    )
    names = set(getattr(terminations, "__dict__", {}).keys())
    names.update(
        (
            "time_out",
            "reference_finished",
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "foot_pos_xyz",
            "base_too_low",
        )
    )
    disabled: list[str] = []
    for name in sorted(names):
        if name.startswith("_") or name in {"reference_finished", "base_too_low"}:
            continue
        if hasattr(terminations, name) and getattr(terminations, name) is not None:
            setattr(terminations, name, None)
            disabled.append(name)
    return disabled


from isaaclab_imitation.contracts.command_publisher import (  # noqa: E402
    ChunkCommandPublisher,
    renewal_env_ids as publisher_renewal_env_ids,
)
from isaaclab_imitation.contracts.planner_publish_schedule import planner_renew_env_ids  # noqa: E402
from isaaclab_imitation.tasks.manager_based.imitation.motion_data import (
    apply_motion_data,
)

from imitation_experiments.evaluation.closed_loop_metrics import FallTracker  # noqa: E402
from imitation_experiments.planner.interface_planner_common import (  # noqa: E402
    INTERFACE_TERMS,
    InterfaceTargetSpec,
    flatten_command_terms,
    load_language_goal_embedding,
    load_planner_checkpoint,
    planner_state_from_batch,
    rmse_per_row,
    unflatten_command_target,
)
from imitation_experiments.capacity.packet_to_latent_command import (  # noqa: E402
    OverlappingPacketEnsembler,
    PacketLayout,
    convert_root_qpos_torso_to_pelvis,
    first_packet_window,
)
from imitation_experiments.data.planner_sample_schema import (  # noqa: E402
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
    "IPMD": IPMD,
}

ENTRY_POINT_ALGORITHM_MAP = {
    "rlopt_ppo_cfg_entry_point": "PPO",
    "rlopt_sac_cfg_entry_point": "SAC",
    "rlopt_ipmd_cfg_entry_point": "IPMD",
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


def _unwrap_imitation_env(env: object) -> ImitationRLEnvLegacy:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ImitationRLEnvLegacy):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if isinstance(unwrapped, ImitationRLEnvLegacy):
            return unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError("Could not unwrap an ImitationRLEnvLegacy.")


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
    base_env: ImitationRLEnvLegacy, requested_names: list[str]
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
    base_env: ImitationRLEnvLegacy,
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
    base_env: ImitationRLEnvLegacy,
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
    base_env: ImitationRLEnvLegacy,
    *,
    tracked_body_names: list[str],
    ee_body_names: list[str],
    tracking_success_root_height_threshold: float,
    tracking_success_root_ori_threshold: float,
) -> tuple[
    dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor
]:
    robot_data = base_env.robot.data

    # Newton backend exposes robot_data.* as warp ProxyArrays; torch/jit ops
    # (quat_error_magnitude) need real tensors. `.torch` bridges them (no-op on
    # PhysX where they are already tensors).
    def _as_tensor(value: Any) -> torch.Tensor:
        return value.torch if hasattr(value, "torch") else value

    root_pos_w = _as_tensor(robot_data.root_pos_w)
    root_quat_w = _as_tensor(robot_data.root_quat_w)
    joint_pos_w = _as_tensor(robot_data.joint_pos)
    joint_vel_w = _as_tensor(robot_data.joint_vel)
    root_lin_vel_w = _as_tensor(robot_data.root_lin_vel_w)
    root_ang_vel_w = _as_tensor(robot_data.root_ang_vel_w)
    root_pos_ref, root_quat_ref, root_lin_vel_ref, root_ang_vel_ref = (
        base_env._get_reference_root_state_w_fast()
    )
    joint_pos_ref = base_env.current_expert_frame["joint_pos"]
    joint_vel_ref = base_env.current_expert_frame["joint_vel"]
    root_pos_error = root_pos_w - root_pos_ref
    root_ori_error = math_utils.quat_error_magnitude(root_quat_w, root_quat_ref)
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
            torch.mean((joint_pos_w - joint_pos_ref).square(), dim=-1)
        ),
        "joint_vel_rmse_radps": torch.sqrt(
            torch.mean((joint_vel_w - joint_vel_ref).square(), dim=-1)
        ),
        "root_lin_vel_rmse_mps": torch.sqrt(
            torch.mean((root_lin_vel_w - root_lin_vel_ref).square(), dim=-1)
        ),
        "root_ang_vel_rmse_radps": torch.sqrt(
            torch.mean((root_ang_vel_w - root_ang_vel_ref).square(), dim=-1)
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
        actual_root_rel = tracked_tensors["actual_pos"] - root_pos_w[:, None, :]
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
    td: TensorDictBase, base_env: ImitationRLEnvLegacy
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
    interface: str,
    *,
    ee_body_names: list[str],
    keypoint_body_names: list[str] | None = None,
) -> dict[str, object]:
    """Body list for interfaces whose packet carries body-set-valued terms.

    Joint- and anchor-valued terms take no body list; the two body interfaces
    differ in WHICH set they carry.
    """
    if interface == "ee_trajectory":
        return {"reference_body_names": tuple(ee_body_names)}
    if interface in {"root_points5", "root_points5_pose"}:
        return {
            "reference_body_names": tuple(
                keypoint_body_names or G1_KEYPOINT5_BODY_NAMES
            )
        }
    return {}


def resolve_pinned_command_joint_ids(base_env: ImitationRLEnvLegacy) -> torch.Tensor:
    """Return live-articulation indices ordered by the pinned joint convention.

    The env delivers full-body command observations through an observation term
    pinned to ``G1_29DOF_ISAACLAB_JOINT_NAMES`` with ``preserve_order=True``
    (see ``_g1_expert_motion_obs_params``), because the expert frame is stored
    in the *live* articulation order, which is backend-specific. Planner packets
    are produced in that live order, so publishing them without re-indexing
    delivers every joint target to the wrong joint. Indexing live-ordered data
    with the returned tensor converts it into the pinned order the env consumes.
    """
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (  # noqa: E501
        G1_29DOF_ISAACLAB_JOINT_NAMES,
    )

    asset_cfg = SceneEntityCfg(
        "robot",
        joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES,
        preserve_order=True,
    )
    asset_cfg.resolve(base_env.scene)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        raise ValueError(
            "Pinned command joint order resolved to a slice; expected explicit "
            "indices. The robot articulation may not expose the pinned joints."
        )
    pinned_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=base_env.device)

    # Determine the re-index direction empirically instead of assuming it: ask
    # the env for the same expert window in both bases and keep whichever
    # mapping actually reproduces the pinned window from the live one. Guessing
    # here is silent and catastrophic -- the wrong direction still yields a
    # well-formed packet that drives every joint from the wrong target.
    probe = {"term_name": "expert_motion", "past_steps": 0, "future_steps": 9}
    live = base_env.get_current_expert_window_term(**probe)
    pinned = base_env.get_current_expert_window_term(**probe, joint_ids=pinned_ids)
    steps = 10
    per_frame = int(live.shape[-1]) // steps
    half = per_frame // 2
    live_q = live.view(-1, steps, per_frame)[..., :half]
    pinned_q = pinned.view(-1, steps, per_frame)[..., :half]
    gather = pinned_ids
    scatter = torch.argsort(pinned_ids)
    for name, index in (("gather", gather), ("scatter", scatter)):
        if torch.allclose(live_q.index_select(-1, index), pinned_q, atol=1e-5):
            print(f"[COMMAND] live->pinned joint re-index resolved as {name}.")
            return index
    raise RuntimeError(
        "Could not reproduce the pinned expert joint window from the live one "
        "with either candidate re-index. The command joint-order contract has "
        "changed; refusing to publish a possibly permuted command."
    )


# Reduced explicit interfaces: single-frame policy-group command spaces whose
# name IS the command space, unlike full_body_trajectory / ee_trajectory which
# map onto a separate "single_frame_*" alias.
_REDUCED_EXPLICIT_INTERFACES = frozenset(
    {"root_qpos", "root_points5", "root_points5_pose"}
)

# Interface -> the command space its frozen tracker was trained on.
_INTERFACE_COMMAND_SPACE: dict[str, str] = {
    "full_body_trajectory": "single_frame_full_body",
    "ee_trajectory": "single_frame_ee",
    "root_qpos": "root_qpos",
    "root_points5": "root_points5",
    "root_points5_pose": "root_points5_pose",
}

# Proprioception appears in every command space and is not part of the packet.
_PROPRIO_TERM_NAMES = frozenset(
    {"base_ang_vel", "joint_pos_rel", "joint_vel_rel", "last_action"}
)


def _interface_command_term_names(interface: str) -> tuple[str, ...]:
    """Packet term names for an interface, in the actor's ordered contract.

    Derived from the command space rather than listed per interface, so adding a
    command space cannot leave a stale list behind here. Cross-checked against
    the planner's own target registry: those two describe the same packet from
    opposite ends, and a disagreement means the planner would be trained to
    predict one layout while the tracker consumes another -- silent, and exactly
    what the joint-order bug was.
    """
    try:
        command_space = _INTERFACE_COMMAND_SPACE[interface]
    except KeyError as err:
        raise ValueError(
            f"Interface {interface!r} has no command-space mapping; expected one "
            f"of {sorted(_INTERFACE_COMMAND_SPACE)}."
        ) from err
    derived = tuple(
        key[1]
        for key in command_space_policy_input_keys(command_space)
        if key[1] not in _PROPRIO_TERM_NAMES
    )
    declared = INTERFACE_TERMS.get(interface)
    if declared is not None and tuple(declared) != derived:
        raise ValueError(
            f"Interface {interface!r} packet layout disagrees between the planner "
            f"target registry {tuple(declared)!r} and the tracker's command-space "
            f"contract {derived!r}."
        )
    return derived


# Command terms carried in JOINT order, and how many joint-width blocks each one
# holds per frame. These are the only terms pinning applies to; anchor, EE and
# keypoint terms are body/root quantities with no joint indexing.
#   expert_motion       2 blocks: cat(joint_pos, joint_vel)
#   expert_motion_qpos  1 block:  positions only (root_qpos drops velocities)
_JOINT_INDEXED_COMMAND_TERMS: dict[str, int] = {
    "expert_motion": 2,
    "expert_motion_qpos": 1,
}


def pin_command_joint_order(
    command_terms: dict[str, torch.Tensor],
    *,
    pinned_joint_ids: torch.Tensor,
    window_steps: int,
) -> dict[str, torch.Tensor]:
    """Re-index a live-order command packet into the env's pinned joint order.

    Every joint-indexed term present is permuted; each of its joint-width blocks
    gets the same permutation. Terms with no joint indexing are returned
    untouched. Publishing without this delivers every joint target to the wrong
    joint -- the defect that invalidated the full-body baseline, and it applies
    equally to any packet carrying a joint half.
    """
    steps = int(window_steps)
    n_joints = int(pinned_joint_ids.numel())
    result = dict(command_terms)
    for name, blocks in _JOINT_INDEXED_COMMAND_TERMS.items():
        if name not in command_terms:
            continue
        value = command_terms[name]
        width = int(value.shape[-1])
        if width % steps != 0:
            raise ValueError(
                f"{name} width {width} is not divisible by window_steps {steps}."
            )
        per_frame = width // steps
        if per_frame != blocks * n_joints:
            raise ValueError(
                f"{name} per-frame width {per_frame} is not {blocks} x "
                f"{n_joints} joints; refusing to publish a mismatched command."
            )
        index = pinned_joint_ids.to(value.device)
        frames = value.view(-1, steps, blocks, n_joints)
        result[name] = frames.index_select(-1, index).reshape(-1, width).contiguous()
    return result


ORACLE_SUBSTITUTION_GROUPS = ("qpos", "qvel", "root_anchor")

# Groups whose substituted values are known to land in the basis the tracker
# actually consumes. ``qpos``/``qvel`` live inside ``expert_motion``, which the
# command trace verifies is reproduced exactly (consumed == expert, 0.0000 at
# every hold phase) once --pin_command_joint_order is on.
#
# ``root_anchor`` is NOT verified and is refused below. Substituting the anchor
# terms with a freshly fetched expert window makes closed-loop tracking ~5x
# WORSE than the planner's own prediction (82.93 mm -> 431.38 mm on an otherwise
# identical config), and the command trace shows the consumed anchors never
# match the expert at publication (pos ~0.004-0.015, ori ~0.06-0.17) while
# expert_motion is exact. The anchors are re-expressed from the held anchor
# frame on every consumption step, so a re-fetched window is not in the basis
# the buffer expects. Until that convention is resolved, an anchor-substituted
# row is meaningless -- it measures the substitution bug, not the interface.
ORACLE_SUBSTITUTION_VERIFIED_GROUPS = ("qpos", "qvel")


def parse_oracle_substitution(spec: str) -> tuple[str, ...]:
    """Parse --oracle_substitute into a canonical, validated group tuple."""
    raw = [token.strip().lower() for token in str(spec).split(",") if token.strip()]
    if not raw or raw == ["none"]:
        return ()
    if raw == ["self"]:
        # Null test: run the substitution machinery with the planner's own
        # prediction as the "expert" source. Must be a no-op; if the measured
        # result differs from --oracle_substitute none, the machinery itself
        # corrupts the packet and every substituted row is invalid.
        return ("self",)
    if "none" in raw:
        raise ValueError("--oracle_substitute cannot combine 'none' with groups.")
    if "all" in raw:
        if len(raw) != 1:
            raise ValueError("--oracle_substitute='all' cannot be combined.")
        raw = list(ORACLE_SUBSTITUTION_GROUPS)
    unknown = sorted(set(raw) - set(ORACLE_SUBSTITUTION_GROUPS))
    if unknown:
        raise ValueError(
            f"Unknown --oracle_substitute group(s) {unknown}; expected a subset of "
            f"{list(ORACLE_SUBSTITUTION_GROUPS)}, or 'all'/'none'."
        )
    unverified = sorted(set(raw) - set(ORACLE_SUBSTITUTION_VERIFIED_GROUPS))
    if unverified:
        raise ValueError(
            f"--oracle_substitute group(s) {unverified} are not supported: their "
            "substituted values do not land in the basis the tracker consumes, so "
            "the resulting row measures the substitution bug rather than the "
            "interface. Verified groups are "
            f"{list(ORACLE_SUBSTITUTION_VERIFIED_GROUPS)}. See the note above "
            "ORACLE_SUBSTITUTION_VERIFIED_GROUPS for the evidence."
        )
    return tuple(g for g in ORACLE_SUBSTITUTION_GROUPS if g in set(raw))


def verify_oracle_substitution(
    substituted: dict[str, torch.Tensor],
    expert_terms: dict[str, torch.Tensor],
    *,
    groups: tuple[str, ...],
    window_steps: int,
    atol: float = 1e-4,
) -> None:
    """Assert the substituted channels really carry the expert values.

    Guards the ladder against the failure that invalidated its first run: a
    substituted packet that is well formed but expressed in the wrong basis
    still produces plausible-looking numbers, so the corruption is invisible
    unless it is checked directly.
    """
    if not groups or groups == ("self",):
        return
    steps = int(window_steps)
    name = "expert_motion"
    if {"qpos", "qvel"} & set(groups) and name in substituted:
        width = int(substituted[name].shape[-1])
        per_frame = width // steps
        half = per_frame // 2
        got = substituted[name].view(-1, steps, per_frame)
        ref = expert_terms[name].to(got.device, got.dtype).view(-1, steps, per_frame)
        for group, sl in (("qpos", slice(0, half)), ("qvel", slice(half, per_frame))):
            if group not in groups:
                continue
            delta = (got[..., sl] - ref[..., sl]).abs().max().item()
            if delta > atol:
                raise RuntimeError(
                    f"Oracle substitution for {group!r} did not take: max|diff| "
                    f"vs the expert window is {delta:.6g} (tolerance {atol}). "
                    "Refusing to report a row whose substituted channels are not "
                    "the expert values."
                )


def apply_oracle_substitution(
    command_terms: dict[str, torch.Tensor],
    expert_terms: dict[str, torch.Tensor],
    *,
    groups: tuple[str, ...],
    window_steps: int,
) -> dict[str, torch.Tensor]:
    """Overwrite selected channel groups of a predicted packet with expert values.

    DIAGNOSTIC ONLY -- this injects ground-truth future information into a
    planner row, so the result is an upper bound, never a planner result.

    ``expert_motion`` is laid out time-major as ``[B, T*D]`` with per-frame
    features ``cat(joint_pos, joint_vel)``, so qpos is the first half of each
    frame and qvel the second. The anchor terms carry the root command.
    """
    if not groups:
        return command_terms
    if groups == ("self",):
        groups = ("qpos", "qvel")
        expert_terms = command_terms
    steps = int(window_steps)
    if steps <= 0:
        raise ValueError(f"window_steps must be positive, got {steps}.")
    out = {name: value.clone() for name, value in command_terms.items()}
    if {"qpos", "qvel"} & set(groups):
        name = "expert_motion"
        if name not in out or name not in expert_terms:
            raise ValueError(
                f"Oracle substitution needs {name!r} in both predicted and expert "
                f"terms; got predicted={sorted(out)} expert={sorted(expert_terms)}."
            )
        width = int(out[name].shape[-1])
        if width % steps != 0:
            raise ValueError(
                f"{name} width {width} is not divisible by window_steps {steps}."
            )
        per_frame = width // steps
        if per_frame % 2 != 0:
            raise ValueError(
                f"{name} per-frame width {per_frame} is not an even qpos/qvel split."
            )
        half = per_frame // 2
        pred = out[name].view(-1, steps, per_frame)
        ref = expert_terms[name].to(pred.device, pred.dtype).view(-1, steps, per_frame)
        if "qpos" in groups:
            pred[..., :half] = ref[..., :half]
        if "qvel" in groups:
            pred[..., half:] = ref[..., half:]
        out[name] = pred.reshape(-1, width)
    if "root_anchor" in groups:
        for name in ("expert_anchor_pos_b", "expert_anchor_ori_b"):
            if name not in out or name not in expert_terms:
                raise ValueError(
                    f"Oracle substitution needs {name!r} in both predicted and "
                    "expert terms."
                )
            out[name] = expert_terms[name].to(out[name].device, out[name].dtype).clone()
    return out


def _current_reference_command_terms(
    base_env: ImitationRLEnvLegacy,
    *,
    interface: str,
    ee_body_names: list[str],
    env_ids: torch.Tensor | None = None,
    past_steps: int | None = None,
    future_steps: int | None = None,
    anchor_body_name: str | None = None,
) -> dict[str, torch.Tensor]:
    ref_kwargs = _command_reference_kwargs(interface, ee_body_names=ee_body_names)
    if anchor_body_name is not None:
        ref_kwargs["anchor_body_name"] = str(anchor_body_name)
    term_names = _interface_command_term_names(interface)
    resolved_past_steps = (
        int(args_cli.command_past_steps) if past_steps is None else int(past_steps)
    )
    resolved_future_steps = (
        int(args_cli.command_future_steps)
        if future_steps is None
        else int(future_steps)
    )
    return {
        term_name: base_env.get_current_expert_window_term(
            term_name=term_name,
            past_steps=resolved_past_steps,
            future_steps=resolved_future_steps,
            env_ids=env_ids,
            **ref_kwargs,
        )
        for term_name in term_names
    }


def _current_demonstration_command_terms(
    base_env: ImitationRLEnvLegacy,
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
    interface: str,
    low_level_command_space: str,
    policy_command_mode: str,
    command_future_steps: int,
    planner_interval_steps: int,
    seed: int,
    allow_shorter_planner_interval: bool = False,
    allow_cross_tracker_planner: bool = False,
    long_horizon_prediction_steps: int | None = None,
) -> dict[str, Any]:
    """Reject planners trained for a different explicit low-level interface.

    The runtime side is passed in rather than assumed: it used to hardcode the
    full-body triple, which made every non-full-body interface look like a
    checkpoint/runtime mismatch even when the planner and tracker agreed. The
    check itself is the valuable part -- it is what catches a planner trained
    against one interface being evaluated against another -- so it must compare
    against what the runtime actually resolved.
    """
    sample_metadata = planner_metadata.get("sample_metadata")
    if not isinstance(sample_metadata, dict):
        raise ValueError(
            "Streamed-vanilla planner checkpoint has no sample_metadata; "
            "retrain it from provenance-bound planner samples."
        )
    expected_values = {
        "interface": str(interface),
        "low_level_command_mode": "streamed_vanilla",
        "low_level_command_space": str(low_level_command_space),
        "policy_command_mode": str(policy_command_mode),
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
    waived: dict[str, Any] = {}
    # `explicit_chunk_current_slot` is the descriptive replacement for the
    # historical generic name. Both names dispatch to the same env adapter.
    # Preserve provenance from already-collected reduced-interface samples
    # without rewriting their tensors or pretending this was a runtime change.
    if (
        interface in _REDUCED_EXPLICIT_INTERFACES
        and "policy_command_mode" in mismatches
        and mismatches["policy_command_mode"]
        == {
            "checkpoint": "full_body_chunk_current_slot",
            "runtime": "explicit_chunk_current_slot",
        }
    ):
        mismatches.pop("policy_command_mode")
        print(
            "[INFO] Normalized historical full_body_chunk_current_slot metadata "
            "to its exact explicit_chunk_current_slot alias.",
            flush=True,
        )

    # A long-horizon explicit planner is trained to predict (for example) H30,
    # while the streamed tracker still consumes one H10 execution window at
    # each 5 Hz renewal. This is an explicit, recorded deployment reduction:
    # the planner target horizon must match the checkpoint, but the runtime
    # command window is intentionally shorter and is produced by the packet
    # reducer/temporal ensembler.
    if (
        long_horizon_prediction_steps is not None
        and int(long_horizon_prediction_steps) > int(command_future_steps) + 1
        and "command_future_steps" in mismatches
    ):
        entry = mismatches["command_future_steps"]
        checkpoint_future_steps = entry.get("checkpoint")
        expected_checkpoint_future_steps = int(long_horizon_prediction_steps) - 1
        if checkpoint_future_steps == expected_checkpoint_future_steps:
            waived["command_future_steps"] = mismatches.pop("command_future_steps")

    if allow_cross_tracker_planner:
        cross_tracker_keys = {
            "low_level_command_mode",
            "low_level_command_space",
            "policy_command_mode",
            "low_level_tracker.checkpoint_sha256",
            "low_level_tracker.policy_input_keys",
            "low_level_tracker.strict_policy_restore",
            "low_level_tracker.policy_frozen",
        }
        cross_tracker = {
            key: mismatches.pop(key)
            for key in tuple(mismatches)
            if key in cross_tracker_keys
        }
        if cross_tracker:
            waived["cross_tracker_plug_in"] = cross_tracker

    # C3 freshness studies deliberately republish more often than the planner
    # was trained to (interval 2 or 5 against a 10-frame packet), consuming only
    # the freshest slots. That IS a train/deploy deviation and the guard is right
    # to catch it -- but it is a study axis, not a defect, so it gets a narrow,
    # explicit, RECORDED exemption rather than a relaxed check. Only
    # planner_interval_steps may be waived, and only when the runtime interval is
    # shorter (using fewer, fresher slots of a packet the planner did produce).
    # A longer interval would consume slots past the trained horizon, which is a
    # genuine contract violation and stays refused.
    if allow_shorter_planner_interval and "planner_interval_steps" in mismatches:
        entry = mismatches["planner_interval_steps"]
        checkpoint_interval = entry.get("checkpoint")
        runtime_interval = entry.get("runtime")
        if (
            isinstance(checkpoint_interval, int)
            and isinstance(runtime_interval, int)
            and 0 < runtime_interval < checkpoint_interval
        ):
            waived["planner_interval_steps"] = mismatches.pop("planner_interval_steps")
    if mismatches:
        raise ValueError(
            "Planner checkpoint is incompatible with the runtime streamed-vanilla "
            f"contract: {mismatches}."
            + ("" if not waived else f" (recorded diagnostic waivers: {waived})")
        )
    if waived:
        print(
            "[WARN] Explicit streamed contract deviation permitted: "
            f"{waived}. This result is a cross-contract diagnostic and must not "
            "be presented as a matched training/deployment row.",
            flush=True,
        )
    return waived


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
    execution_frames = (
        int(args_cli.command_past_steps) + 1 + int(args_cli.command_future_steps)
    )
    packet_prediction_frames = int(args_cli.packet_prediction_horizon_steps)
    if packet_prediction_frames <= 0:
        packet_prediction_frames = execution_frames
    if packet_prediction_frames < execution_frames:
        raise ValueError(
            "--packet_prediction_horizon_steps cannot be shorter than the "
            f"execution window ({packet_prediction_frames} < {execution_frames})."
        )
    packet_layout = PacketLayout.from_target_spec(
        target_spec, packet_frames=packet_prediction_frames
    )
    packet_anchor_conversion = str(args_cli.packet_anchor_conversion)
    if packet_anchor_conversion != "none" and interface != "root_qpos":
        raise ValueError(
            "--packet_anchor_conversion is defined only for root_qpos packets; "
            f"got interface={interface!r}."
        )
    if args_cli.packet_temporal_ensemble != "none" and interface != "root_qpos":
        raise ValueError(
            "Packet temporal ensembling in the direct streamed evaluator is "
            "currently defined for root_qpos packets only."
        )
    if (
        args_cli.packet_temporal_ensemble == "exponential"
        and packet_prediction_frames == execution_frames
    ):
        raise ValueError(
            "Temporal ensembling requires a prediction horizon longer than the "
            "execution window."
        )
    execution_target_spec = InterfaceTargetSpec(
        interface=target_spec.interface,
        term_names=target_spec.term_names,
        term_widths=tuple(
            int(width) // packet_prediction_frames * execution_frames
            for width in target_spec.term_widths
        ),
    )
    pin_mode = str(args_cli.pin_command_joint_order)
    # Pinning applies to any packet carrying a joint half, not to one named
    # interface: root_qpos publishes 29 joint targets and needs it exactly as
    # much as full_body_trajectory. Packets made only of body/root quantities
    # (root_points5, ee_trajectory) have nothing to permute.
    packet_terms = tuple(target_spec.term_names)
    has_joint_term = any(name in _JOINT_INDEXED_COMMAND_TERMS for name in packet_terms)
    pin_command_joints = pin_mode == "on" or (pin_mode == "auto" and has_joint_term)
    if pin_command_joints and not has_joint_term:
        raise ValueError(
            "--pin_command_joint_order=on requires a packet with a joint-indexed "
            f"term ({sorted(_JOINT_INDEXED_COMMAND_TERMS)}); interface="
            f"{interface!r} carries {list(packet_terms)}, which has none. Use "
            "auto (the default), which pins exactly when there are joints."
        )
    atomic_command_anchor = bool(args_cli.atomic_command_anchor)
    in_step_publication = bool(args_cli.in_step_publication) and (
        interface == "full_body_trajectory"
    )
    if packet_anchor_conversion != "none" and in_step_publication:
        raise ValueError(
            "Packet anchor conversion currently requires outer-loop publication."
        )
    if in_step_publication and bool(args_cli.save_rollout_training_samples):
        raise ValueError(
            "--in_step_publication bypasses the outer loop's rollout-sample "
            "collection, so it cannot be combined with "
            "--save_rollout_training_samples: the run would complete having "
            "written zero rows."
        )
    oracle_substitution_groups = parse_oracle_substitution(args_cli.oracle_substitute)
    if oracle_substitution_groups:
        if interface != "full_body_trajectory":
            raise ValueError(
                "--oracle_substitute is only defined for the full_body_trajectory "
                f"packet; got interface={interface!r}."
            )
        print(
            "[DIAGNOSTIC] oracle substitution active for channel groups "
            f"{list(oracle_substitution_groups)}. This run leaks ground-truth "
            "future command data and is an UPPER BOUND, not a planner result."
        )
    low_level_command_mode = str(args_cli.low_level_command_mode)
    low_level_command_space = interface
    if low_level_command_mode == "streamed_vanilla":
        # Both chunk trackers are SINGLE-FRAME consumers -- full-body 157 =
        # 90 proprioception + 67 command, EE 126 = 90 + 36 (4 bodies x 9). So a
        # published 10-frame packet drives either one slot-by-slot: the window is
        # phase-shifted each control step and the time-aligned frame is taken.
        if interface == "full_body_trajectory":
            low_level_command_space = "single_frame_full_body"
            env_cfg.policy_command_mode = "full_body_chunk_current_slot"
        elif interface == "ee_trajectory":
            low_level_command_space = "single_frame_ee"
            env_cfg.policy_command_mode = "ee_chunk_current_slot"
        elif interface in _REDUCED_EXPLICIT_INTERFACES:
            # root_qpos / root_points5 are already single-frame policy-group
            # spaces, so the interface name IS the command space -- there is no
            # separate "single_frame_*" alias to map onto. They share the generic
            # chunk-slot adapter (see _POLICY_COMMAND_MODES on why the mode name
            # is historical rather than per-space).
            low_level_command_space = interface
            env_cfg.policy_command_mode = "explicit_chunk_current_slot"
        else:
            raise ValueError(
                "streamed_vanilla supports explicit command interfaces "
                f"{sorted({'full_body_trajectory', 'ee_trajectory'} | _REDUCED_EXPLICIT_INTERFACES)}; "
                f"got {interface!r}."
            )
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
    if args_cli.base_only_termination and args_cli.disable_tracking_terminations:
        raise ValueError(
            "--base_only_termination and --disable_tracking_terminations are "
            "mutually exclusive."
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
    apply_motion_data(
        env_cfg,
        manifest=motion_manifest,
        cache_dir=(
            args_cli.dataset_path.expanduser().resolve()
            if args_cli.dataset_path is not None
            else None
        ),
        clips=(
            [str(args_cli.motion_name).strip()]
            if str(args_cli.motion_name).strip()
            else None
        ),
        cache_refresh=bool(args_cli.refresh_zarr_dataset),
        wrap_steps=False,
    )
    if hasattr(env_cfg, "reference_start_frame"):
        env_cfg.reference_start_frame = int(args_cli.reference_start_frame)
    if hasattr(env_cfg, "random_reset_full_trajectory"):
        env_cfg.random_reset_full_trajectory = False
    if hasattr(env_cfg, "reset_schedule"):
        env_cfg.reset_schedule = str(args_cli.reset_schedule)
    if not args_cli.enable_observation_corruption:
        _disable_observation_corruption(env_cfg)
    disabled_tracking_termination_terms: list[str] = []
    if args_cli.base_only_termination:
        terminations = getattr(env_cfg, "terminations", None)
        if terminations is None:
            raise ValueError(
                "--base_only_termination requires an environment termination "
                "configuration."
            )
        disabled_tracking_termination_terms = _configure_base_only_termination(
            terminations, minimum_height=float(args_cli.fall_height_m)
        )
        print(
            "[INFO] Base-only evaluation: torso height "
            f"< {float(args_cli.fall_height_m):.2f} m terminates; disabled "
            f"{disabled_tracking_termination_terms}.",
            flush=True,
        )
    elif args_cli.disable_tracking_terminations:
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
        # SONIC termination configs (Strict-v0 chunk trackers) null out
        # base_too_low and fold fall detection into anchor_pos, which M3
        # disables. Inject the standard fall term the latent surface
        # (Latent-v0 / G1TerminationsCfg) uses so M3 survival is defined
        # identically across interfaces: torso root height < 0.4 m.
        if getattr(terminations, FALL_TERMINATION_NAME, None) is None:
            from isaaclab.managers import SceneEntityCfg
            from isaaclab.managers import TerminationTermCfg as _DoneTerm
            from isaaclab_imitation.tasks.manager_based.imitation import (
                mdp as _imitation_mdp,
            )

            setattr(
                terminations,
                FALL_TERMINATION_NAME,
                _DoneTerm(
                    func=_imitation_mdp.root_height_below_minimum,
                    params={
                        "minimum_height": float(args_cli.fall_height_m),
                        "asset_cfg": SceneEntityCfg(
                            "robot", body_names=FALL_TERMINATION_BODY_NAME
                        ),
                    },
                ),
            )
            print(
                f"[INFO] Injected {FALL_TERMINATION_NAME} "
                f"({FALL_TERMINATION_BODY_NAME} height < "
                f"{float(args_cli.fall_height_m):.2f} m) for M3 survival on a "
                "SONIC env that nulls it.",
                flush=True,
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
    command_anchor_body_name = (
        "pelvis"
        if packet_anchor_conversion == "torso_to_pelvis"
        else str(getattr(base_env, "_expert_anchor_body_name", "torso_link"))
    )
    packet_joint_names = tuple(str(name) for name in base_env.robot.joint_names)
    anchor_conversion_stats: dict[str, Any] = {
        "mode": packet_anchor_conversion,
        "source_anchor": (
            "torso_link" if packet_anchor_conversion == "torso_to_pelvis" else None
        ),
        "target_anchor": (
            "pelvis" if packet_anchor_conversion == "torso_to_pelvis" else None
        ),
        "oracle_validation_max_abs": None,
        "oracle_validation_passed": None,
        "converted_publications": 0,
    }
    packet_ensembler: OverlappingPacketEnsembler | None = None
    if args_cli.packet_temporal_ensemble == "exponential":
        packet_ensembler = OverlappingPacketEnsembler(
            num_envs=int(base_env.num_envs),
            prediction_layout=packet_layout,
            execution_frames=execution_frames,
            decay=float(args_cli.packet_temporal_ensemble_decay),
            device=base_env.device,
            dtype=torch.float32,
        )
    pinned_command_joint_ids = (
        resolve_pinned_command_joint_ids(base_env) if pin_command_joints else None
    )
    # Option (b): one shared control plane. The publisher owns joint-order
    # pinning and the renewal schedule; the env remains the buffer owner, so this
    # is behaviour-preserving by construction and can be A/B verified.
    command_publisher = None
    if bool(args_cli.use_command_publisher) and interface == "full_body_trajectory":
        command_publisher = ChunkCommandPublisher(
            num_envs=int(base_env.num_envs),
            term_widths=dict(
                zip(target_spec.term_names, [int(w) for w in target_spec.term_widths])
            ),
            hold_steps=int(args_cli.planner_update_interval),
            window_steps=(
                int(args_cli.command_past_steps)
                + 1
                + int(args_cli.command_future_steps)
            ),
            device=base_env.device,
            joint_reindex=pinned_command_joint_ids,
        )
        print("[COMMAND] publication routed through the shared CommandPublisher.")
    if pinned_command_joint_ids is not None:
        print(
            "[COMMAND] publishing full-body packets in the pinned joint order "
            f"({int(pinned_command_joint_ids.numel())} joints re-indexed from the "
            "live articulation order)."
        )
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
            "execution_window_steps": int(execution_frames),
            "packet_prediction_horizon_steps": int(packet_prediction_frames),
            "packet_temporal_ensemble": str(args_cli.packet_temporal_ensemble),
            "packet_temporal_ensemble_decay": float(
                args_cli.packet_temporal_ensemble_decay
            ),
            "packet_anchor_conversion": packet_anchor_conversion,
            "command_anchor_body_name": command_anchor_body_name,
            "cross_tracker_planner_diagnostic": bool(
                args_cli.allow_cross_tracker_planner
            ),
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
                args_cli.disable_tracking_terminations or args_cli.base_only_termination
            ),
            "base_only_termination": bool(args_cli.base_only_termination),
            "fall_height_m": float(args_cli.fall_height_m),
            "disabled_tracking_termination_terms": (
                disabled_tracking_termination_terms
            ),
            "survival_definition": "no_base_too_low_termination",
            "time_out_enabled": bool(
                getattr(getattr(env_cfg, "terminations", None), "time_out", None)
                is not None
            ),
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

    def _convert_packet_anchor_if_requested(
        packet: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        if packet_anchor_conversion == "none":
            return packet
        torso_pos_w, torso_quat_w = base_env._get_robot_anchor_state_w_fast(
            "torso_link"
        )
        pelvis_pos_w, pelvis_quat_w = base_env._get_robot_anchor_state_w_fast("pelvis")
        selected = env_ids.to(device=torso_pos_w.device, dtype=torch.long)
        kwargs = {
            "layout": packet_layout,
            "actual_torso_pos_w": torso_pos_w.index_select(0, selected),
            "actual_torso_quat_w": torso_quat_w.index_select(0, selected),
            "actual_pelvis_pos_w": pelvis_pos_w.index_select(0, selected),
            "actual_pelvis_quat_w": pelvis_quat_w.index_select(0, selected),
            "joint_names": packet_joint_names,
        }
        converted = convert_root_qpos_torso_to_pelvis(packet, **kwargs)
        anchor_conversion_stats["converted_publications"] += int(env_ids.numel())

        if anchor_conversion_stats["oracle_validation_passed"] is None:
            source_terms = _current_reference_command_terms(
                base_env,
                interface=interface,
                ee_body_names=ee_body_names,
                env_ids=env_ids,
                past_steps=0,
                future_steps=packet_prediction_frames - 1,
                anchor_body_name="torso_link",
            )
            target_terms = _current_reference_command_terms(
                base_env,
                interface=interface,
                ee_body_names=ee_body_names,
                env_ids=env_ids,
                past_steps=0,
                future_steps=packet_prediction_frames - 1,
                anchor_body_name="pelvis",
            )
            source_target, _ = flatten_command_terms(interface, source_terms)
            target_target, _ = flatten_command_terms(interface, target_terms)
            converted_oracle = convert_root_qpos_torso_to_pelvis(
                source_target.to(device=packet.device, dtype=packet.dtype), **kwargs
            )
            absolute_error = (
                converted_oracle - target_target.to(converted_oracle)
            ).abs()
            max_abs = float(absolute_error.max())
            term_errors: dict[str, float] = {}
            cursor = 0
            for term_name, term_width in zip(
                target_spec.term_names, target_spec.term_widths
            ):
                term_errors[str(term_name)] = float(
                    absolute_error[:, cursor : cursor + int(term_width)].max()
                )
                cursor += int(term_width)
            tolerance = 2.0e-5
            anchor_conversion_stats["oracle_validation_max_abs"] = max_abs
            anchor_conversion_stats["oracle_validation_term_max_abs"] = term_errors
            anchor_conversion_stats["oracle_validation_tolerance"] = tolerance
            anchor_conversion_stats["oracle_validation_passed"] = max_abs <= tolerance
            if max_abs > tolerance:
                raise RuntimeError(
                    "Torso-to-pelvis FK conversion failed its oracle equivalence "
                    f"gate: max_abs={max_abs:.9g} > {tolerance:.9g}; "
                    f"term_max_abs={term_errors}."
                )
            print(
                "[COMMAND] torso->pelvis FK oracle equivalence passed: "
                f"max_abs={max_abs:.3e}.",
                flush=True,
            )
        return converted

    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    print(f"[INFO] Loading low-level checkpoint: {checkpoint_path}")
    tracker_provenance: dict[str, Any] | None = None
    if low_level_command_mode == "streamed_vanilla":
        frozen_tracker = load_frozen_low_level_tracker(
            agent,
            checkpoint_path,
            # Derived from the command space -- the single source of truth the
            # tracker is actually built from -- so a new command space cannot
            # drift out of sync with this check.
            expected_input_keys=command_space_policy_input_keys(
                low_level_command_space
            ),
            map_location=env_cfg.sim.device,
        )
        policy = frozen_tracker.policy
        tracker_provenance = frozen_tracker.provenance
        sample_metadata["low_level_tracker"] = tracker_provenance
        provenance = sample_metadata.get("provenance")
        if isinstance(provenance, dict):
            provenance["low_level_tracker"] = tracker_provenance
        contract_waivers = _require_streamed_tracker_checkpoint_contract(
            planner_metadata,
            tracker_provenance,
            interface=interface,
            low_level_command_space=low_level_command_space,
            policy_command_mode=str(env_cfg.policy_command_mode),
            command_future_steps=int(args_cli.command_future_steps),
            planner_interval_steps=planner_update_interval,
            seed=int(env_cfg.seed),
            allow_shorter_planner_interval=bool(
                args_cli.allow_shorter_planner_interval
            ),
            allow_cross_tracker_planner=bool(args_cli.allow_cross_tracker_planner),
            long_horizon_prediction_steps=(
                int(packet_prediction_frames)
                if int(packet_prediction_frames) > int(execution_frames)
                else None
            ),
        )
    else:
        agent.load_model(str(checkpoint_path))
        policy = agent.collector_policy
        policy.eval()

    num_envs = int(args_cli.num_envs)
    # Fall detection reads raw torso height, not the termination manager: under
    # the full-horizon protocol no fall term is registered at all, so anything
    # keyed on terminations reports zero falls whether or not the robot fell.
    # Body and threshold match the injected `base_too_low` term above
    # (torso_link height < 0.4 m) so survival is defined identically either way.
    # Resolve once so a missing body fails before the rollout, not at step 0.
    base_env._get_robot_anchor_body_id_fast(FALL_TERMINATION_BODY_NAME)
    fall_tracker = FallTracker(
        num_envs,
        fall_height_m=float(args_cli.fall_height_m),
        step_dt=step_dt,
    )
    per_step_series: dict[str, list[torch.Tensor]] | None = (
        None if args_cli.no_per_step_metrics else {}
    )
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
    # In-step publication. Publishing between env.step() calls fetches the
    # body-frame anchor terms one physics step before the step that consumes
    # them, which biases the root command; planner_oracle avoids this by filling
    # inside the observation pass. Registering a provider gives the planner the
    # same contract. The packet it produces is stashed so the loop below can
    # still report planner_target_rmse against the matching expert window.
    _publish_stash: dict[str, object] = {}
    _window_steps = (
        int(args_cli.command_past_steps) + 1 + int(args_cli.command_future_steps)
    )

    def _planner_command_provider(env_ids):
        achieved = base_env.current_causal_planner_observation(
            env_ids=env_ids, history_steps=int(args_cli.state_history_steps)
        )
        state = planner_state_from_batch(
            achieved, state_history_steps=int(args_cli.state_history_steps)
        ).to(device=next(planner.parameters()).device, dtype=torch.float32)
        lang = (
            None
            if forced_language is None
            else forced_language.expand(int(state.shape[0]), -1)
        )
        with torch.inference_mode(), planner_latency_timer.enabled():
            predicted = planner(
                state,
                num_inference_steps=int(args_cli.flow_num_inference_steps),
                inference_noise_std=float(args_cli.flow_inference_noise_std),
                language=lang,
            )
        terms = unflatten_command_target(
            predicted.to(device=base_env.device), target_spec
        )
        reference = _current_reference_command_terms(
            base_env,
            interface=interface,
            ee_body_names=ee_body_names,
            env_ids=env_ids,
        )
        if oracle_substitution_groups:
            terms = apply_oracle_substitution(
                terms,
                reference,
                groups=oracle_substitution_groups,
                window_steps=_window_steps,
            )
            verify_oracle_substitution(
                terms,
                reference,
                groups=oracle_substitution_groups,
                window_steps=_window_steps,
            )
        if pinned_command_joint_ids is not None:
            terms = pin_command_joint_order(
                terms,
                pinned_joint_ids=pinned_command_joint_ids,
                window_steps=_window_steps,
            )
        _publish_stash["predicted"] = predicted
        _publish_stash["reference_terms"] = reference
        _publish_stash["count"] = int(env_ids.numel())
        return terms

    if in_step_publication:
        base_env.set_planner_command_provider(_planner_command_provider)
        print("[COMMAND] planner publishes in-step (matches the oracle contract).")

    stop_reason = "max_steps"
    for step_idx in range(int(args_cli.steps)):
        step_active = active.clone()
        if not bool(step_active.any()):
            break
        if command_publisher is not None:
            renew_env_ids = publisher_renewal_env_ids(
                base_env.episode_length_buf + int(args_cli.publish_phase_offset),
                planner_update_interval,
                initial=step_idx == 0,
            )
        else:
            renew_env_ids = planner_renew_env_ids(
                base_env.episode_length_buf + int(args_cli.publish_phase_offset),
                planner_update_interval,
                initial_publication=step_idx == 0,
            )
        if int(renew_env_ids.numel()) > 0:
            active_on_device = step_active.to(device=renew_env_ids.device)
            renew_env_ids = renew_env_ids[
                active_on_device.index_select(0, renew_env_ids)
            ]
        if in_step_publication and int(renew_env_ids.numel()) > 0:
            # The provider published this packet inside env.step(); reuse its
            # stashed tensors for metrics instead of re-running the planner.
            predicted_target = _publish_stash.get("predicted")
            reference_terms = _publish_stash.get("reference_terms")
            if predicted_target is not None and reference_terms is not None:
                planner_publish_count += int(_publish_stash.get("count", 0))
                reference_target, _ = flatten_command_terms(interface, reference_terms)
        elif int(renew_env_ids.numel()) > 0:
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
            predicted_target = predicted_target.to(device=base_env.device)
            execution_prediction = _convert_packet_anchor_if_requested(
                predicted_target, renew_env_ids
            )
            if int(packet_prediction_frames) > int(execution_frames):
                if packet_ensembler is None:
                    reduced_target, _ = first_packet_window(
                        execution_prediction,
                        prediction_layout=packet_layout,
                        execution_frames=execution_frames,
                    )
                else:
                    anchor_pos, anchor_quat = base_env._get_robot_anchor_state_w_fast(
                        command_anchor_body_name
                    )
                    reduced_target = packet_ensembler.update(
                        env_ids=renew_env_ids,
                        packet=execution_prediction,
                        anchor_pos=anchor_pos.index_select(0, renew_env_ids),
                        anchor_quat=anchor_quat.index_select(0, renew_env_ids),
                        episode_steps=base_env.episode_length_buf.index_select(
                            0, renew_env_ids
                        ),
                    )
                command_terms = unflatten_command_target(
                    reduced_target,
                    execution_target_spec,
                )
            else:
                command_terms = unflatten_command_target(
                    execution_prediction,
                    target_spec,
                )
            # The expert window for exactly the frames this packet covers. It is
            # needed for the planner_target_rmse metric below, and -- under the
            # diagnostic oracle-substitution mask -- as the replacement source.
            reference_terms = _current_reference_command_terms(
                base_env,
                interface=interface,
                ee_body_names=ee_body_names,
                env_ids=renew_env_ids,
            )
            prediction_reference_terms = (
                reference_terms
                if int(packet_prediction_frames) == int(execution_frames)
                else _current_reference_command_terms(
                    base_env,
                    interface=interface,
                    ee_body_names=ee_body_names,
                    env_ids=renew_env_ids,
                    future_steps=int(packet_prediction_frames) - 1,
                )
            )
            if oracle_substitution_groups:
                command_terms = apply_oracle_substitution(
                    command_terms,
                    reference_terms,
                    groups=oracle_substitution_groups,
                    window_steps=(
                        int(args_cli.command_past_steps)
                        + 1
                        + int(args_cli.command_future_steps)
                    ),
                )
                verify_oracle_substitution(
                    command_terms,
                    reference_terms,
                    groups=oracle_substitution_groups,
                    window_steps=(
                        int(args_cli.command_past_steps)
                        + 1
                        + int(args_cli.command_future_steps)
                    ),
                )
            # Planner packets (and the expert window used for substitution above)
            # are produced in the live articulation order; the env consumes this
            # buffer through a term pinned to G1_29DOF_ISAACLAB_JOINT_NAMES.
            # Re-index here so every joint target reaches its own joint.
            if command_publisher is not None:
                command_terms = command_publisher.pin_joint_order(command_terms)
            elif pinned_command_joint_ids is not None:
                command_terms = pin_command_joint_order(
                    command_terms,
                    pinned_joint_ids=pinned_command_joint_ids,
                    window_steps=(
                        int(args_cli.command_past_steps)
                        + 1
                        + int(args_cli.command_future_steps)
                    ),
                )
            base_env.set_agent_trajectory_command(
                command_terms,
                env_ids=renew_env_ids,
            )
            # The chunk is stored in the anchor frame at publish time and
            # re-expressed on every consumption step. The env-filled oracle path
            # writes and captures that reference pose atomically; an external
            # publisher must do the same or its root command is interpreted
            # against a stale anchor.
            if atomic_command_anchor:
                base_env.capture_held_command_anchor(
                    anchor_body_name=command_anchor_body_name,
                    env_ids=renew_env_ids,
                )
            planner_publish_count += int(renew_env_ids.numel())

            reference_target, _ = flatten_command_terms(interface, reference_terms)
            prediction_reference_target, _ = flatten_command_terms(
                interface, prediction_reference_terms
            )
            if _os.environ.get("ISAACLAB_TARGET_PROBE") and len(_target_probe) < 20:
                # Diagnostic: the planner is trained to regress reference_target,
                # so a perfect planner would emit exactly this. Dump both at the
                # same publication to measure any temporal offset between what
                # the planner learned and what a fresh fetch returns.
                _target_probe.append(
                    {
                        "step": int(step_idx),
                        "predicted": predicted_target.detach().cpu().clone(),
                        "reference": reference_target.detach().cpu().clone(),
                        "episode_length_buf": base_env.episode_length_buf.detach()
                        .cpu()
                        .clone(),
                    }
                )
                torch.save(_target_probe, _os.environ["ISAACLAB_TARGET_PROBE"])
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
                    env_id=sample_env_ids_cpu,
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
                predicted_target.to(prediction_reference_target.device),
                prediction_reference_target,
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
            if _os.environ.get("ISAACLAB_ACTOR_TRACE") and _actor_trace_n[0] < 12:
                _rec = {"step": _actor_trace_n[0]}
                for _k in td.keys(True, True):
                    try:
                        _v = td.get(_k)
                        if hasattr(_v, "detach"):
                            _rec[str(_k)] = _v.detach().cpu().clone()
                    except Exception:
                        pass
                _actor_trace.append(_rec)
                _actor_trace_n[0] += 1
                torch.save(_actor_trace, _os.environ["ISAACLAB_ACTOR_TRACE"])
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
        # Raw-height fall detection, and pre-fall truncation of the tracking
        # metrics. A fallen robot keeps accruing error for the rest of the
        # rollout while the reference walks away, so the full-horizon mean below
        # conflates "tracks badly" with "fell at step N"; the *_prefall values
        # answer only the first question.
        fall_body_pos_w, _ = base_env._get_robot_anchor_state_w_fast(
            FALL_TERMINATION_BODY_NAME
        )
        fall_body_height = fall_body_pos_w[:num_envs, 2]
        fall_tracker.update(
            fall_body_height,
            {
                name: values[:num_envs]
                for name, values in tracking_metrics.items()
                if name in FALL_TRUNCATED_METRIC_NAMES
            },
        )
        # A3: retain the per-step series so fall time, pre-fall windows and
        # alternative failure thresholds can be re-derived post hoc. The
        # aggregate above collapses to {mean, std, count} and throws away
        # exactly the time structure those questions need. Rollouts are
        # deterministic, so this is the only copy that matters.
        if per_step_series is not None:
            per_step_series.setdefault(
                f"{FALL_TERMINATION_BODY_NAME}_height_m", []
            ).append(fall_body_height.detach().float().cpu().clone())
            per_step_series.setdefault("step_active", []).append(
                step_active.detach().cpu().clone()
            )
            for name in PER_STEP_RETAINED_METRIC_NAMES:
                values = tracking_metrics.get(name)
                if values is not None:
                    per_step_series.setdefault(name, []).append(
                        values[:num_envs].detach().float().cpu().clone()
                    )
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
    # Falls come from raw torso height, never from the termination manager.
    # `G1SonicTerminationsCfg` sets base_too_low=None and the full-horizon
    # protocol nulls the remaining tracking terms, so the previous
    # `termination_hits.get(FALL_TERMINATION_NAME, zeros)` default made "no
    # detector registered" indistinguishable from "no falls" -- every run
    # reported fall_rate 0.00 while robots lay on the floor for hundreds of
    # steps. FallTracker does not depend on which terms are active.
    fall_summary = fall_tracker.summary()
    fall_events = fall_tracker.fall_step >= 0
    fall_free = ~fall_events
    # Report the termination-derived count alongside, but do not require the two
    # to match and do not use it as the headline. When the term is registered it
    # *resets* the environment on the step it fires, so the state this loop
    # reads afterwards is the post-reset upright pose -- the height detector
    # legitimately sees fewer breaches than the term counts. The two are
    # different quantities, not a consistency check.
    fall_summary["fall_detection_source"] = "raw_body_height"
    fall_summary["fall_body_name"] = FALL_TERMINATION_BODY_NAME
    termination_fall_events = termination_hits.get(FALL_TERMINATION_NAME)
    fall_summary["termination_registered"] = termination_fall_events is not None
    fall_summary["termination_fallen_env_count"] = (
        None
        if termination_fall_events is None
        else int(termination_fall_events.sum().item())
    )
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
        # Raw-height fall detection: fall_step / fall_time_s per environment,
        # plus every FALL_TRUNCATED_METRIC_NAMES entry averaged over pre-fall
        # steps only, as `<name>_prefall`. The full-horizon means above keep
        # accruing error after a fall, so quote *_prefall when the question is
        # "how well does it track" and fall_step when it is "did it stay up".
        "fall_detection": fall_summary,
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
            "execution_window_steps": int(execution_frames),
            "packet_prediction_horizon_steps": int(packet_prediction_frames),
            "packet_temporal_ensemble": str(args_cli.packet_temporal_ensemble),
            "packet_temporal_ensemble_decay": float(
                args_cli.packet_temporal_ensemble_decay
            ),
            "packet_anchor_conversion": packet_anchor_conversion,
            "command_anchor_body_name": command_anchor_body_name,
            "cross_tracker_planner_diagnostic": bool(
                args_cli.allow_cross_tracker_planner
            ),
            "planner_metadata": planner_metadata,
            "planner_observation_spec": runtime_planner_observation_spec,
            "low_level_tracker": tracker_provenance,
            # Non-empty only when --allow_shorter_planner_interval waived a
            # streamed-vanilla contract term. Present so a C3 freshness result
            # can never be mistaken downstream for a matched-contract one.
            "streamed_contract_waivers": contract_waivers,
            "num_envs": int(num_envs),
            "seed": int(env_cfg.seed),
            # The physics backend the rollout actually ran on. The frozen
            # low-level checkpoints are Newton-trained, so evaluating them under
            # PhysX would be a silent plant mismatch -- and nothing else in this
            # summary would show it. command.txt carries the override, but the
            # aggregators read this file, so record it where they can see it.
            "physics_backend": type(getattr(env_cfg.sim, "physics", None)).__name__,
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
                args_cli.disable_tracking_terminations or args_cli.base_only_termination
            ),
            "base_only_termination": bool(args_cli.base_only_termination),
            "fall_height_m": float(args_cli.fall_height_m),
            "disabled_tracking_termination_terms": (
                disabled_tracking_termination_terms
            ),
            "survival_definition": "no_base_too_low_termination",
            "time_out_enabled": bool(
                getattr(getattr(env_cfg, "terminations", None), "time_out", None)
                is not None
            ),
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
        # Provenance for the diagnostic oracle-substitution ladder. A run with a
        # non-empty mask received ground-truth future command channels and is an
        # upper bound on the interface, never a reportable planner result.
        "oracle_substitution": {
            "groups": list(oracle_substitution_groups),
            "enabled": bool(oracle_substitution_groups),
            "is_upper_bound_not_planner_result": bool(oracle_substitution_groups),
        },
        # Joint-order basis the published command packet was written in. Results
        # produced with this disabled deliver each joint target to the wrong
        # joint and are not comparable to the env-filled oracle.
        "command_joint_order_pinned": bool(pin_command_joints),
        "atomic_command_anchor": bool(atomic_command_anchor),
        "use_command_publisher": bool(command_publisher is not None),
        "in_step_publication": bool(in_step_publication),
        "packet_temporal_ensemble_stats": (
            packet_ensembler.stats() if packet_ensembler is not None else None
        ),
        "packet_anchor_conversion_stats": anchor_conversion_stats,
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
                "fall_step": fall_summary["fall_step"][env_id],
                "fall_time_s": fall_summary["fall_time_s"][env_id],
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
    if per_step_series:
        import numpy as _np

        per_step_path = output_json.with_name(f"{output_json.stem}_per_step.npz")
        # [steps, num_envs] per key, so a reader can slice either axis without
        # knowing the schema.
        _np.savez_compressed(
            per_step_path,
            **{
                name: torch.stack(chunks).numpy()
                for name, chunks in per_step_series.items()
            },
            fall_step=fall_tracker.fall_step.numpy(),
            step_dt=_np.asarray(
                float("nan") if step_dt is None else float(step_dt), dtype=_np.float64
            ),
            fall_height_m=_np.asarray(float(args_cli.fall_height_m)),
        )
        summary["metadata"]["per_step_metrics_path"] = str(per_step_path)
        output_json.write_text(
            json.dumps(summary, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Wrote per-step metrics: {per_step_path}", flush=True)
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
    except BaseException:  # surface the traceback before Kit's close() fast-exits 0
        import traceback as _tb

        _tb.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
