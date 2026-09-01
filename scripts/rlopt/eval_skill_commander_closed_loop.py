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
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# CU130 split-runtime bootstrap (ICE only): scrub Kit's bundled CPython stdlib off
# sys.path so the runtime python's own platform.py is used (Kit's cannot parse the
# conda-forge sys.version banner). No-op outside the split runtime.
if os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1":
    _KIT_PY = "/isaac-sim/kit/python"
    # Drop Kit's stdlib dir (holds the broken platform.py) but KEEP its
    # site-packages (lazy_loader/hydra/omegaconf, absent from the runtime env).
    sys.path[:] = [
        _p
        for _p in sys.path
        if not (
            os.path.realpath(_p or ".").startswith(_KIT_PY)
            and "site-packages" not in os.path.realpath(_p or ".")
        )
    ]

from runtime_bootstrap import (
    assert_kit_not_loaded,
    config_contains_type_name,
    install_kit_import_guard,
)


STRICT_KITLESS = "--assert-kitless" in sys.argv[1:]
if STRICT_KITLESS:
    install_kit_import_guard()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    "--video_track_env",
    type=int,
    default=-1,
    help=(
        "Environment index whose robot the recording camera follows. <0 keeps the "
        "static world camera built from the viewer eye/lookat."
    ),
)
parser.add_argument(
    "--video_track_offset",
    type=float,
    nargs=3,
    default=(3.0, 3.0, 1.8),
    help="Camera position offset from the tracked robot root, in world axes (m).",
)
parser.add_argument(
    "--video_track_height",
    type=float,
    default=0.8,
    help="Height above the tracked robot root that the camera looks at (m).",
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
    "--tracking_success_root_height_threshold",
    type=float,
    default=0.25,
    help=(
        "Tracking failure threshold for absolute root-height "
        "deviation from the reference. Set <=0 to disable this criterion."
    ),
)
parser.add_argument(
    "--tracking_success_root_ori_threshold",
    type=float,
    default=1.0,
    help=(
        "Tracking failure threshold for root orientation error in "
        "radians. Set <=0 to disable this criterion."
    ),
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
    "--agent_entry_point",
    type=str,
    default=None,
    help="Override the task's RLOpt agent config entry point.",
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
    default=None,
    help=(
        "Planner checkpoint to score or deploy. Required for skill_commander "
        "control; optional for hl_skill oracle collection."
    ),
)
parser.add_argument(
    "--skill_checkpoint",
    type=str,
    default=None,
    help="Override frozen high-level skill checkpoint from planner checkpoint.",
)
parser.add_argument(
    "--state_history_steps",
    type=int,
    default=9,
    help=(
        "Causal history steps used when oracle collection has no planner "
        "checkpoint. Nine past frames plus current is the paper contract."
    ),
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
    "--motion_names",
    nargs="+",
    default=None,
    help="Restrict env.motions to the listed motions before env creation.",
)
parser.add_argument(
    "--trajectory_ranks",
    nargs="+",
    type=int,
    default=None,
    help=(
        "Pin equal environment blocks to these exact trajectory ranks. This is "
        "the scalable selection path for prebuilt reference arrays, where a "
        "small manifest or env.data.clips must not be paired with the full store."
    ),
)
parser.add_argument(
    "--require_goal_motion_match",
    action="store_true",
    default=False,
    help=(
        "Fail if any live environment motion differs from the single explicit "
        "--motion_name. Use this for deployable language-goal collection."
    ),
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
    "--balanced_trajectories_per_motion",
    type=int,
    default=0,
    help=(
        "When positive, collect exactly this many completed variable-length "
        "trajectory segments for every balanced motion. Rows from an episode "
        "are buffered until reset, so the final cutoff contains no partial "
        "trajectories."
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
    "--trajectory_name",
    type=str,
    default=None,
    help="Restrict env.trajectories to this trajectory before env creation.",
)
parser.add_argument(
    "--packet_planner_checkpoint",
    type=Path,
    default=None,
    help=(
        "BB1 shared-tracker mode: drive this latent tracker from an explicit "
        "packet planner routed through the frozen skill encoder, instead of "
        "from the expert window. Requires command_source=hl_skill."
    ),
)
parser.add_argument(
    "--gr00t_checkpoint",
    type=Path,
    default=None,
    help=(
        "Publish latent commands from a trained GR00T action head instead of "
        "the expert window. Reads only the causal robot history plus the "
        "explicit language goal. Requires command_source=hl_skill."
    ),
)
parser.add_argument(
    "--gr00t_goal_features",
    type=Path,
    default=None,
    help="Cached Cosmos goal-feature table for --gr00t_checkpoint.",
)
parser.add_argument(
    "--gr00t_goal",
    type=str,
    default="",
    help=(
        "Explicit language goal name published to every environment. Never "
        "inferred from the reference cursor or trajectory rank."
    ),
)
parser.add_argument(
    "--gr00t_goals_per_env",
    nargs="+",
    default=None,
    help=(
        "Explicit per-environment goal names, one per environment (cycled if "
        "shorter). Evaluates every goal in ONE process instead of paying the "
        "simulator start-up per goal. The assignment is fixed at start-up; a "
        "later goal/reference divergence is a hard error, never a silent "
        "re-derivation from the trajectory rank."
    ),
)
parser.add_argument(
    "--gr00t_consumption",
    choices=("open_loop", "fresh"),
    default="open_loop",
    help=(
        "open_loop consumes the head's consecutive predicted latents one per "
        "publication; fresh re-runs the head at every publication."
    ),
)
parser.add_argument(
    "--gr00t_temporal_ensemble",
    choices=("none", "exponential"),
    default="none",
    help=(
        "Blend the head's overlapping predictions across publications. Each "
        "publication is covered by the current prediction and by the earlier "
        "ones whose horizon still reaches it; exponential weights them by "
        "--gr00t_temporal_ensemble_decay ** age. Distinct from sample "
        "averaging: the estimates come from different states, not redraws."
    ),
)
parser.add_argument("--gr00t_temporal_ensemble_decay", type=float, default=0.5)
parser.add_argument(
    "--gr00t_service",
    type=str,
    default=None,
    help=(
        "ZeroMQ endpoint of a running gr00t_batch_service. When set, the "
        "head runs in that separate process and the sampler follows the "
        "asynchronous lead-time/deadline-miss protocol; the summary is "
        "labelled planner_execution=async_service and is never poolable "
        "with a sync row."
    ),
)
parser.add_argument(
    "--gr00t_lead_steps",
    type=int,
    default=5,
    help="Control steps before a needed renewal at which the async request fires.",
)
parser.add_argument(
    "--gr00t_packet_frame",
    choices=("auto", "anchor", "heading"),
    default="auto",
    help=(
        "Frame the chunk head's predictions live in: 'anchor' is the full "
        "anchor pose, 'heading' the yaw-only anchor frame of a robot_heading "
        "collection. 'auto' resolves from env.expert_macro_anchor_mode. The "
        "native route pins this frame on the chunk term so a heading-frame "
        "head can drive a robot-frame explicit tracker; the encoded route "
        "re-expresses cached packet windows in it."
    ),
)
parser.add_argument(
    "--gr00t_packet_consume_frames",
    type=int,
    default=None,
    help=(
        "chunk_encoded only: re-plan the packet every N control steps and "
        "serve the intermediate publications from the cached packet at its "
        "current age (receding-horizon cursor), instead of re-running the "
        "head at every publication. Matches the latent hold-1 arm's re-plan "
        "cadence; required when chunk_encoded runs against --gr00t_service."
    ),
)
parser.add_argument(
    "--gr00t_consume_slots",
    type=int,
    default=None,
    help=(
        "Slots consumed before the head re-plans. Defaults to the full action "
        "horizon. Setting it below the horizon is a receding-horizon "
        "schedule: a hold-1 head predicting 30 latents with "
        "--gr00t_consume_slots 10 republishes every 10 control steps and "
        "discards the tail."
    ),
)
parser.add_argument(
    "--sample_every_control_step",
    action="store_true",
    default=False,
    help=(
        "Write a training row at every control step instead of only at "
        "planner publication boundaries. Required for a hold-1 collection, "
        "where the downstream join needs a latent at `control_step + k` for "
        "every k. Costs one row per environment per step."
    ),
)
parser.add_argument(
    "--gr00t_inference_steps",
    type=int,
    default=4,
    help=(
        "Euler steps for the head's flow-matching integration. GR00T's "
        "default is 4; more steps trade inference time for a tighter solve of "
        "the same learned velocity field, with no retraining."
    ),
)
parser.add_argument(
    "--gr00t_samples_per_publication",
    type=int,
    default=1,
    help=(
        "Average this many independent flow samples before publishing. Flow "
        "matching starts from a fresh noise draw, so this cuts the sampler's "
        "own variance at inference cost only. Averaging happens on the "
        "continuous value, before any FSQ snap."
    ),
)
parser.add_argument(
    "--fall_only_success",
    action="store_true",
    default=False,
    help=(
        "Success means only 'did not fall'. Also disables the foot_pos_xyz "
        "tracking termination, which otherwise ends a large share of episodes "
        "that never fell and truncates the horizon tracking error averages "
        "over. Pair with --disable_tracking_terminations."
    ),
)
parser.add_argument(
    "--gr00t_route",
    choices=("latent", "chunk_native", "chunk_encoded"),
    default="latent",
    help=(
        "How a GR00T head's prediction reaches the tracker. 'latent': the head "
        "predicts latents published directly (latent arm). 'chunk_native': a "
        "chunk head's explicit frames are published into the chunk actor term "
        "and consumed slot-by-slot by an explicit tracker. 'chunk_encoded': the "
        "SAME chunk head's window is routed through the frozen skill encoder "
        "onto a LATENT tracker -- the tracker-matched chunk-vs-latent row."
    ),
)
parser.add_argument(
    "--packet_source",
    choices=("planner", "expert"),
    default="planner",
    help=(
        "'expert' is the pin test: route the environment's own expert window "
        "through the identical reorder/split/encode path instead of the "
        "planner's prediction. It must reproduce the latent oracle exactly; if "
        "it does not, the bug is in the packet plumbing, not the interface."
    ),
)
parser.add_argument(
    "--packet_noise_alpha",
    type=float,
    default=0.0,
    help=(
        "BB3: noise added to the packet BEFORE the encoder, in per-dimension "
        "std units. Mutually exclusive with --z_noise_alpha."
    ),
)
parser.add_argument(
    "--z_noise_alpha",
    type=float,
    default=0.0,
    help="BB3: noise added to z AFTER the encoder, in per-dimension std units.",
)
parser.add_argument(
    "--noise_reference_samples",
    type=Path,
    default=None,
    help=(
        "Oracle sample .pt used to calibrate both BB3 noise scales, so an alpha "
        "means the same relative perturbation on either side of the encoder."
    ),
)
parser.add_argument("--noise_seed", type=int, default=0)
parser.add_argument(
    "--packet_interface",
    type=str,
    default="full_body_trajectory",
    help="Interface the --packet_planner_checkpoint must declare.",
)
parser.add_argument(
    "--packet_prediction_horizon_steps",
    type=int,
    default=0,
    help=(
        "Explicit planner prediction horizon. Zero uses the frozen encoder's "
        "native H10 input. Longer horizons still execute H10 per renewal."
    ),
)
parser.add_argument(
    "--packet_temporal_ensemble",
    choices=("none", "exponential"),
    default="none",
    help=(
        "For a prediction horizon longer than H10, either execute the first ten "
        "and discard the rest, or ensemble aligned overlapping predictions."
    ),
)
parser.add_argument(
    "--packet_temporal_ensemble_decay",
    type=float,
    default=0.5,
    help="Non-negative exponential age decay for overlapping packet predictions.",
)
parser.add_argument(
    "--latent_temporal_ensemble",
    choices=("first", "exponential", "clipped_gated"),
    default="first",
    help=(
        "Execution rule for an ordered H3 latent planner. 'first' discards the "
        "two forecasts; the other modes align overlapping forecasts of the "
        "current H10 z across publications."
    ),
)
parser.add_argument(
    "--latent_temporal_ensemble_decay",
    type=float,
    default=0.5,
    help="Exponential age decay for overlapping H3 latent forecasts.",
)
parser.add_argument(
    "--latent_temporal_clip_std",
    type=float,
    default=1.0,
    help="Per-dimension training-std clip for stale candidates in clipped_gated mode.",
)
parser.add_argument(
    "--latent_temporal_gate_distance",
    type=float,
    default=2.0,
    help="Reject an old forecast above this normalized RMS distance from fresh z.",
)
parser.add_argument(
    "--latent_temporal_gate_cosine",
    type=float,
    default=0.5,
    help="Reject an old forecast below this cosine agreement with fresh z.",
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
    "--extend_episode_length_for_max_steps",
    action="store_true",
    default=False,
    help=(
        "Extend env.episode_length_s to cover --max_steps plus two control steps. "
        "This matches the focused explicit-interface evaluator timeout protocol."
    ),
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
    "--disable_tracking_terminations",
    action="store_true",
    default=False,
    help=(
        "Keep fall termination active but treat anchor position/orientation and "
        "end-effector tracking errors as metrics instead of terminations."
    ),
)
parser.add_argument(
    "--base_only_termination",
    action="store_true",
    default=False,
    help=(
        "Keep only the base-too-low fall termination (plus reference end). "
        "All tracking-error and timeout terms are disabled, so reported "
        "tracking metrics are truncated only by a physical fall."
    ),
)
parser.add_argument(
    "--fall_height_m",
    type=float,
    default=0.4,
    help=(
        "Absolute torso height used by --base_only_termination. The default "
        "matches the repository's standard G1 base_too_low detector."
    ),
)
parser.add_argument(
    "--disable_reward_clipping",
    action="store_true",
    default=False,
    help=(
        "Disable the legacy [-10, 5] reward transform. Use this for focused "
        "comparisons whose peer evaluators report unclipped environment rewards."
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
    "--sample_target_interface",
    type=str,
    default="latent_skill",
    help=(
        "Target written with saved rows. latent_skill writes z; any other name "
        "writes active expert_macro_state_terms as a term-major packet."
    ),
)
parser.add_argument(
    "--sample_rows_per_file",
    type=int,
    default=1,
    help="Buffer this many planner rows per sample file.",
)
parser.add_argument(
    "--sample_future_window_frames",
    type=int,
    default=10,
    help=(
        "Number of current-plus-future root_qpos frames retained with every "
        "planner publication. Thirty supports later temporal-ensemble and "
        "direct-root_qpos interface studies without recollecting rollouts."
    ),
)
parser.add_argument(
    "--require_root_qpos_samples",
    action="store_true",
    default=False,
    help=(
        "Require the active expert macro interface to be 38-D root_qpos and "
        "write achieved/current/future root_qpos tensors into every sample."
    ),
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
parser.add_argument(
    "--deterministic_tracking",
    action="store_true",
    default=False,
    help=(
        "Measure tracking fidelity without perturbation: start exactly on the "
        "reference and disable interval pushes and domain randomization. Use "
        "for an absolute MPJPE claim or a comparison against externally "
        "published numbers, which are measured on unperturbed rollouts. This "
        "is NOT the paired interface comparison protocol, which keeps "
        "perturbations on and identical across rows; metric keys are prefixed "
        "so the two can never be pooled by accident."
    ),
)
parser.add_argument(
    "--sonic_success_terminations",
    action="store_true",
    default=False,
    help=(
        "Collect/evaluate with the official SONIC completion criterion: "
        "0.25-m pelvis/EE Z, 1-rad pelvis orientation, foot XYZ disabled, "
        "and base-too-low disabled."
    ),
)
parser.add_argument(
    "--disable_push_event",
    action="store_true",
    default=False,
    help="Disable only the interval push event while keeping other randomization.",
)
parser.add_argument(
    "--assert-kitless",
    action="store_true",
    help="Require Newton and fail if Isaac Sim or Omniverse Kit is imported.",
)
from isaaclab_tasks.utils import add_launcher_args

add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

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
from isaaclab_imitation.envs.imitation_rl_env_legacy import ImitationRLEnvLegacy
from isaaclab_imitation.envs.imitation_rl_env_v2 import ImitationRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_EE_BODY_NAMES,
    G1_TRACKED_BODY_NAMES,
    G1TerminationsCfg,
)
from isaaclab_imitation.tasks.manager_based.imitation.motion_data import (
    apply_motion_data,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_cfg import (
    LATENT_POLICY_INPUT_KEYS,
    SONIC_LATENT_POLICY_INPUT_KEYS,
)
from isaaclab_tasks.utils import (
    compute_kit_requirements,
    launch_simulation,
    resolve_task_config,
)
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
from rlopt.agent.skill_commander import FrozenSkillCommanderSampler
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType
from torch import Tensor
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

from imitation_experiments.data.balanced_motion_rows import BalancedMotionRowSelector
from imitation_experiments.lowlevel.motion_candidate_screen import (
    build_env_rank_assignment,
)
from imitation_experiments.planner.interface_planner_common import (
    load_planner_checkpoint,
)
from imitation_experiments.planner.latent_receding_horizon import (
    install_latent_receding_horizon,
)

from imitation_experiments.capacity.packet_to_latent_command import (
    build_noise_reference,
    frames_to_term_major,
    install_packet_encoder_command_source,
    PacketLayout,
)
from imitation_experiments.lowlevel.low_level_tracker import (
    load_frozen_low_level_tracker,
)
from imitation_experiments.provenance.paper_protocol_metadata import (
    disable_domain_randomization,
    interval_event_metadata,
)
from imitation_experiments.audit.backend_determinism import pin_reference_start

# Prefix applied to every metric produced by a --deterministic_tracking run, so
# an unperturbed number can never be pooled with a perturbed one: the paper
# aggregators look up bare names such as "tracking_mpjpe_mm" and will fail
# loudly rather than silently mix protocols.
DETERMINISTIC_METRIC_PREFIX = "deterministic_tracking/"
from imitation_experiments.planner.planner_latency import PlannerForwardTimer
from isaaclab_imitation.contracts import mpjpe_local_global
from isaaclab_imitation.contracts.planner_publish_schedule import planner_renew_env_ids
from imitation_experiments.data.planner_sample_schema import (
    CompletedTrajectorySampleWriter,
    PlannerSampleWriter,
    add_sample_format_metadata,
    build_planner_sample,
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
    if (
        metadata.get("training_stage") != "oracle"
        and metadata.get("pretrain_num_updates") in (None, "")
        and checkpoint_update not in (None, "")
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


def _as_torch_tensor(value: Any, *, label: str) -> Tensor:
    """Normalize Isaac Lab tensors and Newton ProxyArrays at metric boundaries."""
    if isinstance(value, Tensor):
        return value
    torch_value = getattr(value, "torch", None)
    if isinstance(torch_value, Tensor):
        return torch_value
    raise TypeError(
        f"Expected Tensor or Newton ProxyArray for {label}, got {type(value).__name__}."
    )


def _mean_body_pose_errors(
    base_env: ImitationRLEnvLegacy,
    names: list[str],
) -> tuple[Tensor, Tensor] | None:
    if len(names) == 0:
        return None
    body_ids = [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]
    actual_pos, actual_quat = base_env._get_robot_body_pose_w_fast(body_ids)
    ref_pos, ref_quat = base_env._get_reference_body_pose_w_fast(tuple(names))
    actual_pos = _as_torch_tensor(actual_pos, label="robot body positions")
    actual_quat = _as_torch_tensor(actual_quat, label="robot body orientations")
    ref_pos = _as_torch_tensor(ref_pos, label="reference body positions")
    ref_quat = _as_torch_tensor(ref_quat, label="reference body orientations")
    pos_error = torch.linalg.vector_norm(actual_pos - ref_pos, dim=-1).mean(dim=-1)
    ori_error = math_utils.quat_error_magnitude(
        actual_quat.reshape(-1, 4),
        ref_quat.reshape(-1, 4),
    ).reshape(actual_quat.shape[0], -1)
    return pos_error, ori_error.mean(dim=-1)


def _body_tracking_tensors(
    base_env: ImitationRLEnvLegacy,
    names: list[str],
) -> dict[str, Tensor] | None:
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
        "actual_pos": _as_torch_tensor(actual_pos, label="robot body positions"),
        "actual_quat": _as_torch_tensor(actual_quat, label="robot body orientations"),
        "actual_ang_vel": _as_torch_tensor(
            actual_ang_vel, label="robot body angular velocities"
        ),
        "actual_lin_vel": _as_torch_tensor(
            actual_lin_vel, label="robot body linear velocities"
        ),
        "ref_pos": _as_torch_tensor(ref_pos, label="reference body positions"),
        "ref_quat": _as_torch_tensor(ref_quat, label="reference body orientations"),
        "ref_ang_vel": _as_torch_tensor(
            ref_ang_vel, label="reference body angular velocities"
        ),
        "ref_lin_vel": _as_torch_tensor(
            ref_lin_vel, label="reference body linear velocities"
        ),
    }


def _tracking_metrics(
    base_env: ImitationRLEnvLegacy,
    *,
    tracked_body_names: list[str],
    ee_body_names: list[str],
    tracking_success_root_height_threshold: float,
    tracking_success_root_ori_threshold: float,
) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor] | None, Tensor]:
    robot_data = base_env.robot.data
    root_pos_ref, root_quat_ref, root_lin_vel_ref, root_ang_vel_ref = (
        base_env._get_reference_root_state_w_fast()
    )
    root_pos_ref = _as_torch_tensor(root_pos_ref, label="reference root position")
    root_quat_ref = _as_torch_tensor(root_quat_ref, label="reference root orientation")
    root_lin_vel_ref = _as_torch_tensor(
        root_lin_vel_ref, label="reference root linear velocity"
    )
    root_ang_vel_ref = _as_torch_tensor(
        root_ang_vel_ref, label="reference root angular velocity"
    )
    joint_pos_ref = _as_torch_tensor(
        base_env.current_expert_frame["joint_pos"], label="reference joint position"
    )
    joint_vel_ref = _as_torch_tensor(
        base_env.current_expert_frame["joint_vel"], label="reference joint velocity"
    )
    root_pos_w = _as_torch_tensor(robot_data.root_pos_w, label="robot root position")
    root_quat_w = _as_torch_tensor(
        robot_data.root_quat_w, label="robot root orientation"
    )
    joint_pos = _as_torch_tensor(robot_data.joint_pos, label="robot joint position")
    joint_vel = _as_torch_tensor(robot_data.joint_vel, label="robot joint velocity")
    root_lin_vel_w = _as_torch_tensor(
        robot_data.root_lin_vel_w, label="robot root linear velocity"
    )
    root_ang_vel_w = _as_torch_tensor(
        robot_data.root_ang_vel_w, label="robot root angular velocity"
    )
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
            torch.mean((joint_pos - joint_pos_ref).square(), dim=-1)
        ),
        "joint_vel_rmse_radps": torch.sqrt(
            torch.mean((joint_vel - joint_vel_ref).square(), dim=-1)
        ),
        "root_lin_vel_rmse_mps": torch.sqrt(
            torch.mean((root_lin_vel_w - root_lin_vel_ref).square(), dim=-1)
        ),
        "root_ang_vel_rmse_radps": torch.sqrt(
            torch.mean((root_ang_vel_w - root_ang_vel_ref).square(), dim=-1)
        ),
    }
    tracked_body_lin_vel: tuple[Tensor, Tensor] | None = None
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


class _FollowCameraWrapper(gym.Wrapper):
    """Move the recording camera with one environment's robot.

    The Isaac Lab video recorder captures a camera that is static in world space,
    so a walking robot leaves the frame. This wrapper repositions that camera
    before every rendered frame, which keeps a single robot centred for figure
    clips. It touches only the recorder camera: no simulation state, no metrics.
    """

    def __init__(
        self,
        env: Any,
        env_index: int,
        offset: tuple[float, float, float],
        look_height: float,
    ) -> None:
        super().__init__(env)
        self._env_index = env_index
        self._offset = offset
        self._look_height = look_height

    def render(self, *args: Any, **kwargs: Any) -> Any:
        self._follow()
        return self.env.render(*args, **kwargs)

    def _follow(self) -> None:
        recorder = getattr(self.env.unwrapped, "video_recorder", None)
        # A live Newton viewer overwrites the camera from its own state every
        # frame, so following would fight it; headless runs have no viewer.
        if (
            recorder is None
            or getattr(recorder, "_matched_visualizer", None) is not None
        ):
            return
        capture = getattr(recorder, "_capture", None)
        update_camera = getattr(capture, "update_camera", None)
        if update_camera is None:
            return
        robot = self.env.unwrapped.scene["robot"]
        root = robot.data.root_pos_w[self._env_index].detach().float().cpu().tolist()
        eye = (
            root[0] + self._offset[0],
            root[1] + self._offset[1],
            root[2] + self._offset[2],
        )
        target = (root[0], root[1], root[2] + self._look_height)
        update_camera(eye, target)


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
    *,
    device: str,
) -> SkillCommanderConfig:
    if "planner_config" in checkpoint and "planner_state_dict" in checkpoint:
        planner_config = checkpoint.get("planner_config", {})
        metadata = checkpoint.get("metadata", {})
        sample_metadata = (
            metadata.get("sample_metadata", {}) if isinstance(metadata, dict) else {}
        )
        provenance = (
            sample_metadata.get("provenance", {})
            if isinstance(sample_metadata, dict)
            else {}
        )
        skill_checkpoint_path = args_cli.skill_checkpoint or (
            provenance.get("skill_checkpoint", "")
            if isinstance(provenance, dict)
            else ""
        )
        if not skill_checkpoint_path:
            raise ValueError(
                "Shared latent planner checkpoint is missing source skill provenance."
            )
        language_metadata = (
            sample_metadata.get("language_conditioning", {})
            if isinstance(sample_metadata, dict)
            else {}
        )
        language_path = args_cli.language_embeddings or (
            language_metadata.get("embedding_path", "")
            if isinstance(language_metadata, dict)
            else ""
        )
        condition_on_language = (
            int(
                planner_config.get("language_dim", 0)
                if isinstance(planner_config, dict)
                else 0
            )
            > 0
        )
        return SkillCommanderConfig(
            skill_checkpoint_path=str(skill_checkpoint_path),
            condition_on_language=condition_on_language,
            language_embeddings_path=str(language_path),
            state_history_steps=int(sample_metadata.get("state_history_steps", 0)),
            planner_type="flow_matching",
            batch_size=1,
            num_updates=1,
            eval_batches=1,
            eval_batch_size=1,
            device=str(device),
        )
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
    # Evaluation must place the planner on the same accelerator as its inputs.
    # Checkpoints often retain device="cpu" from a portable training/config
    # serialization; honoring that here makes only the direct latent route run
    # on CPU while packet planners are explicitly moved to the simulator GPU.
    values["device"] = str(device)
    values["batch_size"] = 1
    values["num_updates"] = 1
    values["eval_batches"] = 1
    values["eval_batch_size"] = 1
    return SkillCommanderConfig.from_dict(values)


def _configured_step_dt(env_cfg: object) -> float | None:
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 0.0) or 0.0)
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    if sim_dt > 0.0 and decimation > 0:
        return sim_dt * decimation
    return None


TRACKING_TERMINATION_NAMES = ("anchor_pos", "anchor_ori", "ee_body_pos")
# `foot_pos_xyz` is a tracking termination too, but it is NOT in the M3 set
# above. Under a fall-only success definition it must also be disabled, or a
# large share of episodes end on it while never falling -- which truncates the
# horizon that tracking error is averaged over.
FOOT_TRACKING_TERMINATION_NAME = "foot_pos_xyz"
FALL_TERMINATION_NAME = "base_too_low"


def _tracking_mpjpe_pair_mm(base_env: Any) -> tuple[Tensor, Tensor] | None:
    """Per-env root-relative and world-frame MPJPE in mm.

    Same computation as the 4,096-motion tracker scoreboard: subtract each
    side's own root position before comparing the tracked bodies, so the metric
    is blind to global drift and is directly comparable to that board's
    successful MPJPE-L.

    Written against the environment's accessors rather than importing the
    scoreboard module: `evaluate_checkpoint` is a SCRIPT with module-level
    argparse, so importing it mid-run re-parses `sys.argv` and aborts the
    evaluation.
    """
    names = list(getattr(base_env.cfg.data, "runtime_cache_body_names", []) or [])
    if not names:
        return None
    body_ids = [int(base_env._get_robot_anchor_body_id_fast(name)) for name in names]
    actual_pos, _ = base_env._get_robot_body_pose_w_fast(body_ids)
    reference_pos, _ = base_env._get_reference_body_pose_w_fast(tuple(names))
    robot_root = base_env.robot.data.root_pos_w
    robot_root = getattr(robot_root, "torch", robot_root)
    reference_root, _, _, _ = base_env._get_reference_root_state_w_fast()
    local_m, global_m = mpjpe_local_global(
        actual_pos,
        robot_root,
        reference_pos,
        reference_root,
    )
    return local_m * 1000.0, global_m * 1000.0


def _disable_tracking_terminations(
    terminations: Any, *, include_foot: bool = False
) -> list[str]:
    disabled: list[str] = []
    names = TRACKING_TERMINATION_NAMES + (
        (FOOT_TRACKING_TERMINATION_NAME,) if include_foot else ()
    )
    for name in names:
        if hasattr(terminations, name) and getattr(terminations, name) is not None:
            setattr(terminations, name, None)
            disabled.append(name)
    return disabled


def _disable_non_reference_terminations(terminations: Any) -> None:
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
    for name in sorted(names):
        if name.startswith("_") or name == "reference_finished":
            continue
        if hasattr(terminations, name):
            setattr(terminations, name, None)


def _configure_sonic_success_terminations(terminations: Any) -> dict[str, Any]:
    """Apply the paper-compatible SONIC completion boundary in place."""
    if terminations is None:
        raise ValueError(
            "--sonic_success_terminations requires an environment termination config."
        )
    required = ("anchor_pos", "anchor_ori", "ee_body_pos")
    missing = [name for name in required if getattr(terminations, name, None) is None]
    if missing:
        raise ValueError(f"SONIC termination profile is missing terms: {missing}.")
    for name in ("anchor_pos", "ee_body_pos"):
        params = getattr(terminations, name).params
        params["threshold"] = 0.25
        if "down_threshold" in params:
            params["down_threshold"] = 0.25
    getattr(terminations, "anchor_ori").params["threshold"] = 1.0
    if hasattr(terminations, "foot_pos_xyz"):
        terminations.foot_pos_xyz = None
    if hasattr(terminations, "base_too_low"):
        terminations.base_too_low = None
    return {
        "name": "sonic_success_foot_xyz_disabled",
        "anchor_pos_threshold_m": 0.25,
        "anchor_ori_threshold_rad": 1.0,
        "ee_body_pos_threshold_m": 0.25,
        "foot_pos_xyz_enabled": False,
        "base_too_low_enabled": False,
    }


def _disable_push_event(env_cfg: Any) -> dict[str, Any]:
    events = getattr(env_cfg, "events", None)
    previously_enabled = bool(
        events is not None and getattr(events, "push_robot", None) is not None
    )
    if events is not None and hasattr(events, "push_robot"):
        events.push_robot = None
    return {
        "enabled_before_override": previously_enabled,
        "enabled": False,
        "other_randomization_kept": True,
    }


def _reference_selection_cfg(env_cfg: Any) -> Any | None:
    """Return the concrete v2 reference-selection config, if present."""
    selection = getattr(
        getattr(getattr(env_cfg, "command_interface", None), "reference", None),
        "selection",
        None,
    )
    seen = 0
    while selection is not None and not hasattr(selection, "start_mode"):
        selection = getattr(selection, "default", None)
        seen += 1
        if seen > 8:
            return None
    return selection


def _install_skill_commander_preflight_sampler(environment: Any) -> None:
    """Provide constructor-only widths without fabricating offline robot data.

    V2 intentionally has no offline causal-robot sampler: causal planner inputs
    exist only during a policy rollout. ``SkillCommanderTrainer`` nevertheless
    asks for one batch at construction solely to discover widths. Expert macro
    frames are valid for the frozen encoder; zero planner histories provide the
    declared shape only and are never trained on or saved. Every collected row
    below still comes from ``current_causal_planner_observation``.
    """

    def _preflight_batch(
        *,
        batch_size: int,
        horizon_steps: int,
        split: str | None,
        eval_fraction: float,
        split_seed: int,
        history_steps: int,
    ) -> TensorDictBase:
        batch = environment.sample_expert_macro_transition_batch(
            batch_size=int(batch_size),
            horizon_steps=int(horizon_steps),
            split=split,
            eval_fraction=float(eval_fraction),
            split_seed=int(split_seed),
        )
        spec = environment.causal_planner_observation_spec(
            history_steps=int(history_steps)
        )
        history = torch.zeros(
            (
                int(batch_size),
                int(spec["history_frames"]),
                int(spec["frame_dim"]),
            ),
            device=environment.device,
            dtype=torch.float32,
        )
        batch.set(
            "planner",
            TensorDict(
                {"state": history[:, -1], "state_history": history},
                batch_size=[int(batch_size)],
                device=environment.device,
            ),
        )
        return batch

    environment.sample_causal_planner_training_batch = _preflight_batch


def _configure_base_only_termination(
    terminations: Any, *, minimum_height: float
) -> list[str]:
    if minimum_height <= 0.0:
        raise ValueError("--fall_height_m must be positive.")
    base_term = G1TerminationsCfg().base_too_low
    base_term.params["minimum_height"] = float(minimum_height)
    terminations.base_too_low = base_term

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


def _planner_state(batch: Any, state_history_steps: int) -> Tensor:
    group = "planner" if batch.get(("planner", "state")) is not None else "hl"
    if int(state_history_steps) > 0:
        state_history = batch.get((group, "state_history"))
        if state_history is None:
            msg = (
                f"Expected {group}/state_history for state-history planner checkpoint."
            )
            raise ValueError(msg)
        return state_history.reshape(int(state_history.shape[0]), -1).contiguous()
    return batch.get((group, "state"))


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


def _macro_packet_layout(wrapped_env: Any, *, horizon_steps: int) -> PacketLayout:
    """Derive a term-major packet layout from the active encoder macro terms."""
    raw_slices = wrapped_env.expert_macro_feature_slices(
        horizon_steps=int(horizon_steps)
    )
    ordered = sorted(
        ((str(name), int(span[0]), int(span[1])) for name, span in raw_slices.items()),
        key=lambda item: item[1],
    )
    cursor = 0
    term_widths: list[tuple[str, int]] = []
    for name, start, stop in ordered:
        if start != cursor or stop <= start:
            raise ValueError(
                f"Invalid expert macro feature slice {name!r}=[{start}, {stop}); "
                f"expected a contiguous slice beginning at {cursor}."
            )
        term_widths.append((name, stop - start))
        cursor = stop
    return PacketLayout(tuple(term_widths), packet_frames=int(horizon_steps))


@torch.no_grad()
def _measure_commander(
    *,
    trainer: SkillCommanderTrainer,
    wrapped_env: Any,
    env_ids: Tensor,
    sample_path: Path | None = None,
    sample_writer: PlannerSampleWriter | None = None,
    sample_step: int | None = None,
    sample_metadata: dict[str, Any] | None = None,
    episode_ids: Tensor | None = None,
    sample_motion_names: list[str] | None = None,
    sample_target_interface: str = "latent_skill",
    sample_future_window_frames: int = 10,
    require_root_qpos_samples: bool = False,
    compute_metrics: bool = True,
) -> dict[str, float]:
    if sample_path is not None and sample_writer is not None:
        raise ValueError("Provide sample_path or sample_writer, not both.")
    horizon_steps = int(trainer.horizon_steps)
    state_history_steps = int(trainer.config.state_history_steps)
    retained_frames = int(sample_future_window_frames)
    if retained_frames <= 0:
        raise ValueError("sample_future_window_frames must be positive.")
    query_horizon = max(horizon_steps, retained_frames - 1)
    expert_batch = wrapped_env.current_expert_macro_transition_batch(
        horizon_steps=query_horizon,
        env_ids=env_ids,
        state_history_steps=state_history_steps,
    )
    causal_planner_batch = wrapped_env.current_causal_planner_observation(
        env_ids=env_ids,
        history_steps=state_history_steps,
    )

    expert_state = expert_batch.get(("hl", "state")).to(
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
    achieved_planner_state = _planner_state(
        causal_planner_batch, state_history_steps
    ).to(device=trainer.device, dtype=torch.float32)

    skill_future_window = future_window[:, :horizon_steps]
    z_target = trainer._target_z(expert_state, skill_future_window)
    packet_layout = _macro_packet_layout(wrapped_env, horizon_steps=horizon_steps)
    required_future = packet_layout.packet_frames - 1
    if int(future_window.shape[1]) < required_future:
        raise ValueError(
            f"Expert future window has {int(future_window.shape[1])} frames; "
            f"packet target {sample_target_interface!r} needs {required_future}."
        )
    packet_frames = torch.cat(
        [expert_state.unsqueeze(1), future_window[:, :required_future]], dim=1
    )
    packet_target = frames_to_term_major(packet_frames, packet_layout)
    sample_target = (
        z_target if str(sample_target_interface) == "latent_skill" else packet_target
    )
    lang = trainer._lang_for_ranks(traj_rank)

    if sample_path is not None or sample_writer is not None:
        if (
            sample_metadata is None
            or episode_ids is None
            or sample_motion_names is None
        ):
            raise ValueError(
                "Saving rollout samples requires metadata, episode IDs, and motion names."
            )
        if sample_path is not None:
            sample_path.parent.mkdir(parents=True, exist_ok=True)
        local_step = expert_batch.get(("hl", "local_step")).detach().cpu().reshape(-1)
        trajectory_length = (
            expert_batch.get(("hl", "trajectory_length")).detach().cpu().reshape(-1)
        )
        sample = build_planner_sample(
            causal_state_history=achieved_planner_state,
            # Oracle-policy collection observes the real robot, not a synthetic
            # expert-state rollout. Keep the paired compatibility key identical
            # and mark the true semantics explicitly below.
            demonstration_state_history=achieved_planner_state,
            causal_target=sample_target,
            demonstration_target=sample_target,
            trajectory_rank=traj_rank,
            episode_id=episode_ids,
            # `env_ids` may be a balanced subset of the live vectorized envs.
            # Preserve the actual simulator IDs: together with the per-env
            # episode counter they uniquely identify a trajectory segment.
            env_id=env_ids.detach().cpu().reshape(-1),
            control_step=local_step,
            planner_step=torch.div(local_step, horizon_steps, rounding_mode="floor"),
            motion_names=sample_motion_names,
            metadata=sample_metadata,
            language_embedding=lang if trainer.condition_on_language else None,
        )
        # Both planner routes receive targets from the exact same simulator rows.
        sample["latent_skill_target"] = z_target.detach().cpu().float().contiguous()
        sample["encoder_input_packet_target"] = (
            packet_target.detach().cpu().float().contiguous()
        )
        # Keep the old latent target alias during migration of analysis tools.
        sample["z_target"] = sample["latent_skill_target"]
        sample["oracle_rollout_state_history"] = (
            achieved_planner_state.detach().cpu().float().contiguous()
        )
        sample["reference_local_step"] = local_step.to(dtype=torch.long)
        sample["reference_trajectory_length"] = trajectory_length.to(dtype=torch.long)
        if require_root_qpos_samples:
            if int(expert_state.shape[-1]) != 38:
                raise ValueError(
                    "--require_root_qpos_samples needs a 38-D root_qpos macro "
                    f"state, got {int(expert_state.shape[-1])}."
                )
            achieved_batch = wrapped_env.current_achieved_macro_transition_batch(
                horizon_steps=horizon_steps,
                env_ids=env_ids,
                state_history_steps=state_history_steps,
            )
            achieved_root_qpos = achieved_batch.get(("hl", "state")).to(
                device=trainer.device, dtype=torch.float32
            )
            if tuple(achieved_root_qpos.shape) != tuple(expert_state.shape):
                raise ValueError(
                    "Achieved root_qpos shape does not match expert root_qpos: "
                    f"{tuple(achieved_root_qpos.shape)} != {tuple(expert_state.shape)}."
                )
            future_needed = retained_frames - 1
            root_qpos_window = torch.cat(
                [expert_state.unsqueeze(1), future_window[:, :future_needed]], dim=1
            )
            offsets = torch.arange(
                retained_frames, device=trainer.device, dtype=torch.long
            ).unsqueeze(0)
            valid = expert_batch.get(("hl", "local_step")).to(
                device=trainer.device, dtype=torch.long
            ).reshape(-1, 1) + offsets < expert_batch.get(
                ("hl", "trajectory_length")
            ).to(device=trainer.device, dtype=torch.long).reshape(-1, 1)
            sample["expert_root_qpos"] = (
                expert_state.detach().cpu().float().contiguous()
            )
            sample["achieved_root_qpos"] = (
                achieved_root_qpos.detach().cpu().float().contiguous()
            )
            sample["expert_root_qpos_future"] = (
                root_qpos_window.detach().cpu().float().contiguous()
            )
            sample["expert_root_qpos_future_valid"] = (
                valid.detach().cpu().bool().contiguous()
            )
        sample["step"] = None if sample_step is None else int(sample_step)
        if sample_writer is not None:
            sample_writer.add(sample)
        else:
            torch.save(sample, sample_path)

    if not compute_metrics:
        return {}

    achieved_batch = wrapped_env.current_achieved_macro_transition_batch(
        horizon_steps=horizon_steps,
        env_ids=env_ids,
        state_history_steps=state_history_steps,
    )
    achieved_state = achieved_batch.get(("hl", "state")).to(
        device=trainer.device, dtype=torch.float32
    )
    trainer.generator.eval()
    if bool(getattr(trainer, "_uses_shared_interface_planner", False)):
        flow_steps = int(getattr(trainer, "shared_flow_num_inference_steps", 16))
        flow_noise = float(getattr(trainer, "shared_flow_inference_noise_std", 0.0))
        z_m3 = trainer.generator(
            achieved_planner_state,
            num_inference_steps=flow_steps,
            inference_noise_std=flow_noise,
            language=lang,
        )
    else:
        z_m3 = trainer.generator(achieved_planner_state, lang)
    if int(z_m3.shape[-1]) != int(z_target.shape[-1]):
        if int(z_m3.shape[-1]) % int(z_target.shape[-1]) != 0:
            raise ValueError(
                "Planner metric prediction cannot be split into H10 latent tokens: "
                f"{tuple(z_m3.shape)} vs {tuple(z_target.shape)}."
            )
        # Metric-side forward calls do not mutate the deployed overlap history.
        # Score the fresh token here; the actually fused command is measured by
        # published_z_vs_target below.
        z_m3 = z_m3[:, : int(z_target.shape[-1])]

    metrics = {
        "m3/z_cosine": _cosine_mean(z_m3, z_target),
        "m3/z_mse": _mse_mean(z_m3, z_target),
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

    published_provider = getattr(wrapped_env, "get_agent_latent_command", None)
    if callable(published_provider):
        published = published_provider(env_ids=env_ids)
    else:
        actor_command = getattr(
            getattr(wrapped_env, "actor_command", None), "command", None
        )
        if not isinstance(actor_command, Tensor):
            raise RuntimeError("The active latent actor command is unavailable.")
        published = actor_command.index_select(
            0, env_ids.to(device=actor_command.device, dtype=torch.long)
        )
    published = published.to(device=trainer.device, dtype=torch.float32)
    z_dim = int(trainer.z_dim)
    if published.ndim == 2 and int(published.shape[-1]) >= z_dim:
        published_z = published[:, :z_dim]
        metrics["published_z_vs_m3/z_cosine"] = _cosine_mean(published_z, z_m3)
        metrics["published_z_vs_m3/z_mse"] = _mse_mean(published_z, z_m3)
        metrics["published_z_vs_target/z_cosine"] = _cosine_mean(published_z, z_target)
        metrics["published_z_vs_target/z_mse"] = _mse_mean(published_z, z_target)
    return metrics


agent_entry_point = args_cli.agent_entry_point or resolve_agent_cfg_entry_point(
    args_cli.task, args_cli.algorithm
)


def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: Any,
) -> None:
    print(
        "[INFO] Entered closed-loop evaluator main: "
        f"physics={type(getattr(env_cfg.sim, 'physics', None)).__name__}",
        flush=True,
    )
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    selected_motion_name = (
        str(args_cli.motion_name).strip() if args_cli.motion_name is not None else ""
    )
    selected_motion_names = (
        [str(name).strip() for name in args_cli.motion_names]
        if args_cli.motion_names is not None
        else None
    )
    if selected_motion_name and selected_motion_names is not None:
        raise ValueError("--motion_name and --motion_names are mutually exclusive.")
    if args_cli.require_goal_motion_match and not selected_motion_name:
        raise ValueError("--require_goal_motion_match requires --motion_name.")
    if selected_motion_names is not None and (
        not selected_motion_names or any(not name for name in selected_motion_names)
    ):
        raise ValueError("--motion_names must contain non-empty names.")
    if int(args_cli.balanced_rows_per_motion) < 0:
        raise ValueError("--balanced_rows_per_motion must be >= 0.")
    if int(args_cli.balanced_trajectories_per_motion) < 0:
        raise ValueError("--balanced_trajectories_per_motion must be >= 0.")
    if (
        int(args_cli.balanced_rows_per_motion) > 0
        and int(args_cli.balanced_trajectories_per_motion) > 0
    ):
        raise ValueError(
            "Row-balanced and completed-trajectory-balanced collection are "
            "mutually exclusive."
        )
    if int(args_cli.sample_rows_per_file) <= 0:
        raise ValueError("--sample_rows_per_file must be positive.")
    if int(args_cli.sample_future_window_frames) <= 0:
        raise ValueError("--sample_future_window_frames must be positive.")
    termination_profiles = sum(
        bool(value)
        for value in (
            args_cli.base_only_termination,
            args_cli.disable_tracking_terminations,
            args_cli.sonic_success_terminations,
        )
    )
    if termination_profiles > 1:
        raise ValueError(
            "--base_only_termination, --disable_tracking_terminations, and "
            "--sonic_success_terminations are mutually exclusive."
        )
    trajectory_ranks = (
        [int(rank) for rank in args_cli.trajectory_ranks]
        if args_cli.trajectory_ranks is not None
        else None
    )
    if trajectory_ranks is not None:
        if any(rank < 0 for rank in trajectory_ranks):
            raise ValueError("--trajectory_ranks must be non-negative.")
        if len(set(trajectory_ranks)) != len(trajectory_ranks):
            raise ValueError("--trajectory_ranks must be unique.")
    if args_cli.balanced_motion_names and not (
        int(args_cli.balanced_rows_per_motion) > 0
        or int(args_cli.balanced_trajectories_per_motion) > 0
    ):
        raise ValueError(
            "--balanced_motion_names requires a positive balanced row or "
            "trajectory budget."
        )
    if (
        int(args_cli.balanced_rows_per_motion) > 0
        or int(args_cli.balanced_trajectories_per_motion) > 0
    ) and not bool(args_cli.save_rollout_training_samples):
        raise ValueError(
            "Balanced collection requires --save_rollout_training_samples."
        )
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    if str(args_cli.gr00t_route) == "chunk_native":
        # The environment config is the authority on what the actor reads. The
        # latent routes hard-code that contract, but a chunk actor driving an
        # explicit tracker must derive it -- otherwise the agent either builds
        # a latent controller the env does not publish, or falls back to a
        # proprio list missing `projected_gravity` and mismatches the trained
        # 131-wide policy by exactly 3. Same binding the training and
        # evaluate_checkpoint entry points perform.
        from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (  # noqa: PLC0415
            bind_command_interface,
        )

        if bind_command_interface(agent_cfg, env_cfg) is None:
            sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
            if callable(sync_input_keys):
                sync_input_keys()
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

    selected_clips = (
        [selected_motion_name] if selected_motion_name else selected_motion_names
    )
    if selected_clips is not None:
        apply_motion_data(env_cfg, clips=selected_clips)
    if args_cli.trajectory_name is not None:
        if not hasattr(env_cfg, "trajectories"):
            raise TypeError(f"Task {args_cli.task} does not support --trajectory_name.")
        env_cfg.trajectories = [str(args_cli.trajectory_name)]
    reference_selection_surface = "unmodified"
    if not args_cli.allow_random_reset:
        reference_selection_surface = pin_reference_start(env_cfg, start_frame=0)
    if trajectory_ranks is not None:
        env_rank_assignment = build_env_rank_assignment(
            trajectory_ranks, int(env_cfg.scene.num_envs)
        )

        def _fixed_trajectory_ranks(
            env_ids: torch.Tensor, num_trajectories: int
        ) -> torch.Tensor:
            if max(env_rank_assignment) >= int(num_trajectories):
                raise ValueError(
                    "--trajectory_ranks contains a rank outside the loaded "
                    f"dataset with {num_trajectories} trajectories."
                )
            rank_table = torch.as_tensor(
                env_rank_assignment, dtype=torch.long, device=env_ids.device
            )
            return rank_table[env_ids.long()]

        reference_selection = _reference_selection_cfg(env_cfg)
        if reference_selection is not None:
            reference_selection.schedule = "custom"
            reference_selection.custom_fn = _fixed_trajectory_ranks
            env_cfg.command_interface.reference.selection = reference_selection
        elif hasattr(env_cfg, "reset_schedule"):
            env_cfg.reset_schedule = "custom"
            env_cfg.custom_reset_fn = _fixed_trajectory_ranks
        else:
            raise RuntimeError("The environment has no trajectory selection surface.")
    terminations = getattr(env_cfg, "terminations", None)
    if not args_cli.keep_time_out:
        if terminations is not None and hasattr(terminations, "time_out"):
            terminations.time_out = None
    deterministic_tracking_record: dict[str, Any] = {"enabled": False}
    if args_cli.deterministic_tracking:
        deterministic_tracking_record = disable_domain_randomization(env_cfg)
    push_event_record = interval_event_metadata(env_cfg, "push_robot")
    if args_cli.disable_push_event:
        push_event_record = _disable_push_event(env_cfg)
    disabled_tracking_termination_terms: list[str] = []
    sonic_termination_record: dict[str, Any] | None = None
    if args_cli.sonic_success_terminations:
        sonic_termination_record = _configure_sonic_success_terminations(terminations)
        print(
            "[INFO] SONIC success evaluation: anchor/EE=0.25 m, "
            "anchor orientation=1 rad, foot XYZ and base-too-low disabled.",
            flush=True,
        )
    elif args_cli.base_only_termination:
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
        if terminations is None:
            raise ValueError(
                "--disable_tracking_terminations requires an environment "
                "termination configuration."
            )
        disabled_tracking_termination_terms = _disable_tracking_terminations(
            terminations, include_foot=bool(args_cli.fall_only_success)
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
            # The strict SONIC-derived low-level task disables this term during
            # controller training. Phase-5 planner evaluation defines survival
            # by this fall event, so restore the shared G1 fall detector before
            # environment construction instead of silently changing survival.
            terminations.base_too_low = G1TerminationsCfg().base_too_low
            print(
                "[INFO] Restored base_too_low for M3 survival evaluation.",
                flush=True,
            )
        if (
            not hasattr(terminations, FALL_TERMINATION_NAME)
            or getattr(terminations, FALL_TERMINATION_NAME) is None
        ):
            raise ValueError(
                "M3 metrics-only evaluation requires the base_too_low fall "
                "termination to remain active."
            )
    elif not args_cli.keep_early_terminations:
        if terminations is not None:
            _disable_non_reference_terminations(terminations)
    if args_cli.extend_episode_length_for_max_steps:
        if int(args_cli.max_steps) <= 0:
            raise ValueError(
                "--extend_episode_length_for_max_steps requires positive --max_steps."
            )
        step_dt = _configured_step_dt(env_cfg)
        if step_dt is None or not hasattr(env_cfg, "episode_length_s"):
            raise ValueError(
                "Cannot extend episode length because the configured control step "
                "duration or env.episode_length_s is unavailable."
            )
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s),
            float(int(args_cli.max_steps) + 2) * step_dt,
        )

    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    planner_checkpoint_path = (
        Path(args_cli.planner_checkpoint).expanduser().resolve()
        if args_cli.planner_checkpoint is not None
        else None
    )
    command_source = str(getattr(agent_cfg.ipmd, "command_source", "unknown"))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Low-level checkpoint not found: {checkpoint_path}")
    if planner_checkpoint_path is not None and not planner_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SkillCommander checkpoint not found: {planner_checkpoint_path}"
        )
    if command_source == "skill_commander" and planner_checkpoint_path is None:
        raise ValueError(
            "--planner_checkpoint is required when agent.ipmd.command_source="
            "skill_commander."
        )
    if planner_checkpoint_path is None and args_cli.skill_checkpoint is None:
        raise ValueError(
            "Oracle collection without --planner_checkpoint requires "
            "--skill_checkpoint."
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
        "planner_checkpoint": (
            str(planner_checkpoint_path)
            if planner_checkpoint_path is not None
            else None
        ),
        "skill_checkpoint_override": args_cli.skill_checkpoint,
        "language_embeddings_override": args_cli.language_embeddings,
        "motion_name": selected_motion_name or None,
        "motion_names": selected_motion_names,
        "trajectory_ranks": trajectory_ranks,
        "reference_selection_surface": reference_selection_surface,
        "goal_motion_match_required": bool(args_cli.require_goal_motion_match),
        "balanced_rows_per_motion": int(args_cli.balanced_rows_per_motion),
        "balanced_trajectories_per_motion": int(
            args_cli.balanced_trajectories_per_motion
        ),
        "balanced_motion_names": args_cli.balanced_motion_names,
        "trajectory_name": args_cli.trajectory_name,
        "allow_random_reset": bool(args_cli.allow_random_reset),
        "random_reset_step_min": int(getattr(env_cfg, "random_reset_step_min", -1)),
        "random_reset_step_max": int(getattr(env_cfg, "random_reset_step_max", -1)),
        "keep_time_out": bool(args_cli.keep_time_out),
        "episode_length_extension_enabled": bool(
            args_cli.extend_episode_length_for_max_steps
        ),
        "keep_early_terminations": bool(args_cli.keep_early_terminations),
        "sonic_success_terminations": bool(args_cli.sonic_success_terminations),
        "sonic_termination_profile": sonic_termination_record,
        "disable_push_event": bool(args_cli.disable_push_event),
        "push_perturbation": push_event_record,
        "base_only_termination": bool(args_cli.base_only_termination),
        "fall_height_m": float(args_cli.fall_height_m),
        "tracking_terminations_enabled": not bool(
            args_cli.disable_tracking_terminations or args_cli.base_only_termination
        ),
        "disabled_tracking_termination_terms": disabled_tracking_termination_terms,
        "survival_definition": (
            "reference_finished_without_sonic_tracking_failure"
            if args_cli.sonic_success_terminations
            else "no_base_too_low_termination"
        ),
        "reward_clipping_enabled": not bool(args_cli.disable_reward_clipping),
        "continue_after_reset": bool(args_cli.continue_after_reset),
        "save_rollout_training_samples": bool(args_cli.save_rollout_training_samples),
        "sample_future_window_frames": int(args_cli.sample_future_window_frames),
        "require_root_qpos_samples": bool(args_cli.require_root_qpos_samples),
        "tracking_success_root_height_threshold": float(
            args_cli.tracking_success_root_height_threshold
        ),
        "tracking_success_root_ori_threshold": float(
            args_cli.tracking_success_root_ori_threshold
        ),
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
    if args_cli.video and int(args_cli.video_track_env) >= 0:
        gym_env = _FollowCameraWrapper(
            gym_env,
            env_index=int(args_cli.video_track_env),
            offset=tuple(float(x) for x in args_cli.video_track_offset),
            look_height=float(args_cli.video_track_height),
        )
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

    wrapped_env = IsaacLabWrapper(gym_env)
    wrapped_env = wrapped_env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=wrapped_env.observation_spec, backend="gymnasium"
        )
    )
    transforms = [RewardSum(), StepCounter(max_steps + 1)]
    if not args_cli.disable_reward_clipping:
        transforms.append(RewardClipping(-10.0, 5.0))
    env = TransformedEnv(base_env=wrapped_env, transform=Compose(*transforms))
    if not isinstance(raw_isaac_env, (ImitationRLEnv, ImitationRLEnvLegacy)):
        raise TypeError(
            "Expected the unwrapped gym env to be an imitation environment, got "
            f"{type(raw_isaac_env).__name__}."
        )
    base_env = raw_isaac_env
    loaded_motion_names = [
        str(name) for name in base_env.expert_trajectory_motion_names()
    ]
    balanced_selector: BalancedMotionRowSelector | None = None
    balanced_trajectory_motion_names: list[str] | None = None
    if int(args_cli.balanced_rows_per_motion) > 0:
        balanced_motion_names = (
            [str(name).strip() for name in args_cli.balanced_motion_names]
            if args_cli.balanced_motion_names is not None
            else (
                selected_motion_names
                if selected_motion_names is not None
                else list(loaded_motion_names)
            )
        )
        missing_motion_names = sorted(
            set(balanced_motion_names).difference(loaded_motion_names)
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
    elif int(args_cli.balanced_trajectories_per_motion) > 0:
        balanced_trajectory_motion_names = (
            [str(name).strip() for name in args_cli.balanced_motion_names]
            if args_cli.balanced_motion_names is not None
            else (
                selected_motion_names
                if selected_motion_names is not None
                else list(loaded_motion_names)
            )
        )
        missing_motion_names = sorted(
            set(balanced_trajectory_motion_names).difference(loaded_motion_names)
        )
        if missing_motion_names:
            raise ValueError(
                "Balanced trajectory motions are not loaded by the environment: "
                f"{missing_motion_names}."
            )
    tracked_body_names = _resolve_existing_body_names(
        base_env, list(G1_TRACKED_BODY_NAMES)
    )
    ee_body_names = _resolve_existing_body_names(
        base_env,
        list(getattr(env_cfg, "command_ee_body_names", G1_EE_BODY_NAMES)),
    )

    shared_target_spec = None
    if planner_checkpoint_path is None:
        planner_checkpoint: dict[str, Any] = {}
        language_path = str(args_cli.language_embeddings or "").strip()
        trainer_config = SkillCommanderConfig(
            skill_checkpoint_path=str(Path(args_cli.skill_checkpoint).expanduser()),
            condition_on_language=bool(language_path),
            language_embeddings_path=language_path,
            state_history_steps=int(args_cli.state_history_steps),
            planner_type="flow_matching",
            batch_size=1,
            num_updates=1,
            eval_batches=1,
            eval_batch_size=1,
            device="cpu",
        )
    else:
        planner_checkpoint = torch.load(
            planner_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        trainer_config = _trainer_config_from_checkpoint(
            planner_checkpoint, device=str(env_cfg.sim.device)
        )
    _install_skill_commander_preflight_sampler(raw_isaac_env)
    trainer = SkillCommanderTrainer(config=trainer_config, env=raw_isaac_env)
    if "planner_config" in planner_checkpoint:
        assert planner_checkpoint_path is not None
        shared_generator, shared_target_spec, _ = load_planner_checkpoint(
            planner_checkpoint_path,
            map_location=trainer.device,
        )
        if shared_target_spec.interface != "latent_skill":
            raise ValueError(
                "Shared SkillCommander planner must target latent_skill, got "
                f"{shared_target_spec.interface!r}."
            )
        shared_generator = shared_generator.to(trainer.device)
        trainer.generator = shared_generator
        trainer._uses_shared_interface_planner = True
        checkpoint_metadata = planner_checkpoint.get("metadata", {})
        trainer.shared_flow_num_inference_steps = int(
            args_cli.flow_num_inference_steps
            or (
                checkpoint_metadata.get("flow_num_inference_steps", 16)
                if isinstance(checkpoint_metadata, dict)
                else 16
            )
        )
        trainer.shared_flow_inference_noise_std = float(
            args_cli.flow_inference_noise_std
        )
    elif "generator_state_dict" in planner_checkpoint:
        trainer.generator.load_state_dict(planner_checkpoint["generator_state_dict"])
    trainer.update = int(
        planner_checkpoint.get(
            "update",
            planner_checkpoint.get("metadata", {}).get("num_updates", 0),
        )
    )
    trainer.generator.eval()

    h3_latent_planner = False
    if command_source == "skill_commander" and shared_target_spec is not None:
        planner_target_dim = int(shared_target_spec.target_dim)
        if planner_target_dim != int(trainer.z_dim):
            expected_widths = (int(trainer.z_dim),) * 3
            if tuple(shared_target_spec.term_widths) != expected_widths:
                raise ValueError(
                    "A long-horizon latent planner must expose three ordered H10 "
                    f"tokens {expected_widths}, got {shared_target_spec.term_widths}."
                )
            h3_latent_planner = True

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    if h3_latent_planner:
        # TorchRL probes the collector policy during agent construction.  An H3
        # commander emits 3*z_dim and therefore cannot be installed until its
        # receding-horizon reducer is present.  Bootstrap that unscored probe
        # with the frozen oracle sampler, then replace it below before reset or
        # any evaluated simulator step.
        agent_cfg.ipmd.command_source = "hl_skill"
    try:
        agent = agent_class(env=env, config=agent_cfg)
    finally:
        agent_cfg.ipmd.command_source = command_source
    if h3_latent_planner:
        commander_overrides: dict[str, object] = {}
        if agent_cfg.ipmd.skill_commander_flow_num_inference_steps > 0:
            commander_overrides["flow_num_inference_steps"] = int(
                agent_cfg.ipmd.skill_commander_flow_num_inference_steps
            )
        if agent_cfg.ipmd.skill_commander_flow_inference_noise_std >= 0.0:
            commander_overrides["flow_inference_noise_std"] = float(
                agent_cfg.ipmd.skill_commander_flow_inference_noise_std
            )
        if agent_cfg.ipmd.skill_commander_diffusion_num_inference_steps > 0:
            commander_overrides["diffusion_num_inference_steps"] = int(
                agent_cfg.ipmd.skill_commander_diffusion_num_inference_steps
            )
        if agent_cfg.ipmd.skill_commander_diffusion_inference_scheduler:
            commander_overrides["diffusion_inference_scheduler"] = str(
                agent_cfg.ipmd.skill_commander_diffusion_inference_scheduler
            )
        if agent_cfg.ipmd.skill_commander_diffusion_ddim_eta >= 0.0:
            commander_overrides["diffusion_ddim_eta"] = float(
                agent_cfg.ipmd.skill_commander_diffusion_ddim_eta
            )
        if agent_cfg.ipmd.skill_commander_diffusion_inference_noise_std >= 0.0:
            commander_overrides["diffusion_inference_noise_std"] = float(
                agent_cfg.ipmd.skill_commander_diffusion_inference_noise_std
            )
        assert planner_checkpoint_path is not None
        agent._hl_skill_command_sampler = FrozenSkillCommanderSampler(
            env=agent.env,
            checkpoint_path=planner_checkpoint_path,
            language_embeddings_path=str(
                agent_cfg.ipmd.skill_commander_embeddings_path
            ),
            latent_dim=int(agent_cfg.ipmd.latent_dim),
            latent_steps_min=int(agent_cfg.ipmd.latent_steps_min),
            latent_steps_max=int(agent_cfg.ipmd.latent_steps_max),
            generator_config_overrides=commander_overrides,
            horizon_steps=(
                int(agent_cfg.ipmd.hl_skill_horizon_steps)
                if int(agent_cfg.ipmd.hl_skill_horizon_steps) > 0
                else None
            ),
            command_phase_mode=str(agent_cfg.ipmd.latent_learning.command_phase_mode),
            code_latent_dim=int(agent_cfg.ipmd.latent_learning.code_latent_dim),
            phase_period=int(agent_cfg.ipmd.latent_learning.code_period),
            command_mode=str(agent_cfg.ipmd.hl_skill_command_mode),
            use_achieved_state=bool(agent_cfg.ipmd.skill_commander_use_achieved_state),
            goal_name=str(agent_cfg.ipmd.skill_commander_goal_name),
            goal_rank=int(agent_cfg.ipmd.skill_commander_goal_rank),
            discover_env_method=agent._discover_env_method,
            device=agent._get_device(agent.config.device),
        )
        agent._command_source = command_source
    gr00t_sampler = None
    gr00t_chunk_publisher = None
    gr00t_packet_planner = None
    # Shared by the BB1 packet route below and by a GR00T chunk_encoded run;
    # declared here so whichever installs the packet->encoder source owns them.
    packet_encoder_stats: Any = None
    packet_encoder_provenance: dict[str, Any] | None = None
    # One goal for all environments, or an explicit per-environment assignment
    # that lets a single process cover every goal.
    if args_cli.gr00t_goals_per_env:
        _requested = [str(name) for name in args_cli.gr00t_goals_per_env]
        gr00t_goal_spec: Any = [
            _requested[index % len(_requested)]
            for index in range(int(env_cfg.scene.num_envs))
        ]
    else:
        gr00t_goal_spec = str(args_cli.gr00t_goal)
    if args_cli.gr00t_checkpoint is not None and args_cli.gr00t_route != "latent":
        # Chunk-target head. Both routes share the head and the causal input;
        # they differ only in which channel carries the prediction to the
        # tracker, which is exactly the variable the comparison isolates.
        from imitation_experiments.planner.gr00t_chunk_publisher import (  # noqa: PLC0415
            Gr00tChunkPublisher,
            Gr00tPacketPlanner,
        )

        # Only the ENCODED route needs the oracle sampler: it borrows that
        # sampler's frozen skill encoder to turn the packet into a latent. The
        # native route publishes the packet straight into the chunk actor term
        # and touches no encoder, so it must not require a latent agent.
        oracle_sampler = getattr(agent, "_hl_skill_command_sampler", None)
        if args_cli.gr00t_route == "chunk_encoded" and oracle_sampler is None:
            raise ValueError(
                "--gr00t_route=chunk_encoded requires agent.ipmd.command_source="
                "hl_skill so the frozen sampler and its skill encoder exist."
            )
        if args_cli.gr00t_goal_features is None or not (
            args_cli.gr00t_goal or args_cli.gr00t_goals_per_env
        ):
            raise ValueError(
                "--gr00t_checkpoint requires --gr00t_goal_features plus either "
                "--gr00t_goal (one goal for every environment) or "
                "--gr00t_goals_per_env (an explicit per-environment assignment)."
            )
        causal_fn = agent._discover_env_method(
            agent.env, "current_causal_planner_observation"
        )
        if causal_fn is None:
            raise ValueError(
                "the environment exposes no current_causal_planner_observation; "
                "the GR00T planner input is causal-only by contract."
            )
        chunk_term = None
        if args_cli.gr00t_route == "chunk_native":
            chunk_term = getattr(raw_isaac_env, "actor_command", None)
            if chunk_term is None or not hasattr(chunk_term, "window_steps"):
                raise ValueError(
                    "--gr00t_route=chunk_native needs a chunk actor term: run "
                    "with env.command_interface.actor=chunk and "
                    "env.command_interface.actor.source=external."
                )
        gr00t_chunk_publisher = Gr00tChunkPublisher(
            chunk_term=chunk_term,
            causal_observation_fn=causal_fn,
            state_history_steps=int(args_cli.state_history_steps),
            gr00t_checkpoint=args_cli.gr00t_checkpoint,
            goal_features_path=args_cli.gr00t_goal_features,
            goal_name=gr00t_goal_spec,
            num_envs=int(env_cfg.scene.num_envs),
            device=agent._get_device(agent.config.device),
        )

        # Shared accessors for the cursor/async execution modes: the anchor
        # frame a prediction is expressed in, and the episode-local clock the
        # deadline protocol counts in. 'auto' takes the frame from the env's
        # macro anchor mode, which is the frame the training collection's
        # chunk targets were expressed in.
        gr00t_packet_frame = str(args_cli.gr00t_packet_frame)
        if gr00t_packet_frame == "auto":
            macro_mode = str(getattr(env_cfg, "expert_macro_anchor_mode", "robot"))
            if macro_mode == "robot_heading":
                gr00t_packet_frame = "heading"
            elif macro_mode == "robot":
                gr00t_packet_frame = "anchor"
            else:
                raise ValueError(
                    "--gr00t_packet_frame auto cannot resolve "
                    f"expert_macro_anchor_mode={macro_mode!r}; pass the frame "
                    "explicitly."
                )
        print(f"[INFO] GR00T packet frame: {gr00t_packet_frame}")

        def _chunk_anchor_state() -> tuple[Tensor, Tensor]:
            name = str(getattr(raw_isaac_env, "_expert_anchor_body_name", "pelvis"))
            pos_w, quat_w = raw_isaac_env._get_robot_anchor_state_w_fast(name)
            pos_w = pos_w.reshape(-1, 3)
            quat_w = quat_w.reshape(-1, 4)
            if gr00t_packet_frame == "heading":
                from isaaclab_imitation.tasks.manager_based.imitation.mdp._compiled import (  # noqa: PLC0415
                    heading_anchor_frame,
                )

                return heading_anchor_frame(pos_w, quat_w)
            return pos_w, quat_w

        def _chunk_episode_steps() -> Tensor:
            return raw_isaac_env.episode_length_buf

        if args_cli.gr00t_route == "chunk_native" and gr00t_packet_frame == "heading":
            # The term's publish-time capture assumes the packet lives in the
            # full anchor pose; a heading-frame head needs its frame pinned on
            # every publish, sync or async.
            gr00t_chunk_publisher._pin_anchor_state_fn = _chunk_anchor_state
        sync_chunk_publisher = gr00t_chunk_publisher
        if (
            args_cli.gr00t_route == "chunk_native"
            and args_cli.gr00t_service is not None
        ):
            from imitation_experiments.planner.gr00t_async_chunk import (  # noqa: PLC0415
                Gr00tAsyncChunkPublisher,
            )

            gr00t_chunk_publisher = Gr00tAsyncChunkPublisher(
                publisher=sync_chunk_publisher,
                chunk_term=chunk_term,
                hold_steps=int(trainer.horizon_steps),
                service_endpoint=str(args_cli.gr00t_service),
                lead_steps=int(args_cli.gr00t_lead_steps),
                anchor_state_fn=_chunk_anchor_state,
                episode_step_fn=_chunk_episode_steps,
                num_envs=int(env_cfg.scene.num_envs),
            )
        if args_cli.gr00t_route == "chunk_encoded":
            # Tracker-matched row: the identical chunk head drives the SAME
            # latent tracker the latent arm uses, through the frozen encoder.
            def _gr00t_causal_state(env_ids: Tensor) -> Tensor:
                # BB1 calls this immediately before the planner, and the
                # planner signature carries no env ids -- record them here so
                # each row is conditioned on ITS environment's goal.
                if gr00t_packet_planner is not None:
                    gr00t_packet_planner.note_env_ids(env_ids)
                return _planner_state(
                    raw_isaac_env.current_causal_planner_observation(
                        env_ids=env_ids,
                        history_steps=int(args_cli.state_history_steps),
                    ),
                    int(args_cli.state_history_steps),
                )

            # The layout verifier reads `env._expert_macro_feature_slices`, but
            # in the v2 env that cache lives on the composed ExpertDataPlane and
            # is filled lazily. Resolve it through the public accessor and mirror
            # it where the verifier looks, so the first publish is checked
            # instead of refused.
            raw_isaac_env._expert_macro_feature_slices = (
                raw_isaac_env.expert_macro_feature_slices(
                    horizon_steps=int(trainer.horizon_steps)
                )
            )
            # PacketLayout carries (term_name, width) pairs, in packet order,
            # and is verified against the ENV's expert-macro frame layout --
            # so it must use the macro term names, not the command-space names
            # the chunk actor term uses for the same 38 values.
            from imitation_experiments.planner.gr00t_chunk_publisher import (  # noqa: PLC0415
                ROOT_QPOS_MACRO_TERMS,
            )

            gr00t_packet_layout = PacketLayout(
                ROOT_QPOS_MACRO_TERMS,
                packet_frames=int(sync_chunk_publisher._gr00t_horizon),
            )
            if args_cli.gr00t_packet_consume_frames is not None:
                # Receding-horizon cursor: re-plan every N steps, serve the
                # intermediate publications from the cached packet at its age.
                # The installer's own ensembler assumes a fresh head call per
                # publication, so the two modes are mutually exclusive.
                if str(args_cli.gr00t_temporal_ensemble) != "none":
                    raise ValueError(
                        "--gr00t_packet_consume_frames requires "
                        "--gr00t_temporal_ensemble none."
                    )
                assert oracle_sampler is not None  # checked above for this route
                encoder_frames = int(oracle_sampler.skill_encoder.window_steps) + 1
                if args_cli.gr00t_service is not None:
                    from imitation_experiments.planner.gr00t_async_chunk import (  # noqa: PLC0415
                        Gr00tAsyncPacketPlanner,
                    )

                    gr00t_packet_planner = Gr00tAsyncPacketPlanner(
                        sync_chunk_publisher,
                        service_endpoint=str(args_cli.gr00t_service),
                        lead_steps=int(args_cli.gr00t_lead_steps),
                        consume_frames=int(args_cli.gr00t_packet_consume_frames),
                        encoder_frames=encoder_frames,
                        anchor_state_fn=_chunk_anchor_state,
                        episode_step_fn=_chunk_episode_steps,
                        num_envs=int(env_cfg.scene.num_envs),
                    )
                else:
                    from imitation_experiments.planner.gr00t_async_chunk import (  # noqa: PLC0415
                        Gr00tCursorPacketPlanner,
                    )

                    gr00t_packet_planner = Gr00tCursorPacketPlanner(
                        sync_chunk_publisher,
                        consume_frames=int(args_cli.gr00t_packet_consume_frames),
                        encoder_frames=encoder_frames,
                        anchor_state_fn=_chunk_anchor_state,
                        episode_step_fn=_chunk_episode_steps,
                        num_envs=int(env_cfg.scene.num_envs),
                    )
            elif args_cli.gr00t_service is not None:
                raise ValueError(
                    "--gr00t_route=chunk_encoded with --gr00t_service needs "
                    "--gr00t_packet_consume_frames: the async protocol is "
                    "defined on the receding-horizon cursor."
                )
            else:
                gr00t_packet_planner = Gr00tPacketPlanner(sync_chunk_publisher)
            packet_encoder_stats = install_packet_encoder_command_source(
                oracle_sampler,
                planner=gr00t_packet_planner,
                causal_state_provider=_gr00t_causal_state,
                env=raw_isaac_env,
                packet_layout=gr00t_packet_layout,
                packet_source=str(args_cli.packet_source),
            )
            packet_encoder_provenance = {
                "packet_source": (
                    "gr00t_chunk_head"
                    if str(args_cli.packet_source) == "planner"
                    else "expert_pin"
                ),
                "packet_planner_checkpoint": str(args_cli.gr00t_checkpoint),
                "packet_interface": "root_qpos",
            }
        print(
            f"[INFO] GR00T {args_cli.gr00t_route} route: "
            f"{gr00t_chunk_publisher.provenance}"
        )
    elif args_cli.gr00t_checkpoint is not None:
        # Swap the oracle latent source for the GR00T head, keeping the frozen
        # sampler's hold/phase/renewal machinery and its skill encoder (the
        # oracle latent stays available as a diagnostic and never reaches the
        # policy).
        from imitation_experiments.planner.gr00t_latent_sampler import (  # noqa: PLC0415
            Gr00tLatentCommandSampler,
        )

        gr00t_sampler_cls = Gr00tLatentCommandSampler
        gr00t_async_kwargs: dict = {}
        if args_cli.gr00t_service is not None:
            from imitation_experiments.planner.gr00t_async_sampler import (  # noqa: PLC0415
                Gr00tAsyncLatentCommandSampler,
            )

            gr00t_sampler_cls = Gr00tAsyncLatentCommandSampler
            gr00t_async_kwargs = {
                "service_endpoint": str(args_cli.gr00t_service),
                "lead_steps": int(args_cli.gr00t_lead_steps),
            }

        oracle_sampler = getattr(agent, "_hl_skill_command_sampler", None)
        if oracle_sampler is None:
            raise ValueError(
                "--gr00t_checkpoint requires agent.ipmd.command_source=hl_skill "
                "so the frozen sampler (and its skill encoder) exists."
            )
        if args_cli.gr00t_goal_features is None or not (
            args_cli.gr00t_goal or args_cli.gr00t_goals_per_env
        ):
            raise ValueError(
                "--gr00t_checkpoint requires --gr00t_goal_features plus either "
                "--gr00t_goal (one goal for every environment) or "
                "--gr00t_goals_per_env (an explicit per-environment assignment)."
            )
        causal_fn = agent._discover_env_method(
            agent.env, "current_causal_planner_observation"
        )
        if causal_fn is None:
            raise ValueError(
                "the environment exposes no current_causal_planner_observation; "
                "the GR00T planner input is causal-only by contract."
            )
        gr00t_sampler = gr00t_sampler_cls(
            **gr00t_async_kwargs,
            causal_observation_fn=causal_fn,
            state_history_steps=int(args_cli.state_history_steps),
            gr00t_checkpoint=args_cli.gr00t_checkpoint,
            goal_features_path=args_cli.gr00t_goal_features,
            goal_name=gr00t_goal_spec,
            num_envs=int(env_cfg.scene.num_envs),
            consumption=str(args_cli.gr00t_consumption),
            num_inference_timesteps=int(args_cli.gr00t_inference_steps),
            samples_per_publication=int(args_cli.gr00t_samples_per_publication),
            consume_slots=(
                None
                if args_cli.gr00t_consume_slots is None
                else int(args_cli.gr00t_consume_slots)
            ),
            temporal_ensemble=str(args_cli.gr00t_temporal_ensemble),
            temporal_ensemble_decay=float(args_cli.gr00t_temporal_ensemble_decay),
            env=agent.env,
            checkpoint_path=str(agent_cfg.ipmd.hl_skill_checkpoint_path),
            latent_dim=int(agent_cfg.ipmd.latent_dim),
            latent_steps_min=int(agent_cfg.ipmd.latent_steps_min),
            latent_steps_max=int(agent_cfg.ipmd.latent_steps_max),
            horizon_steps=(
                int(agent_cfg.ipmd.hl_skill_horizon_steps)
                if int(agent_cfg.ipmd.hl_skill_horizon_steps) > 0
                else None
            ),
            command_phase_mode=str(agent_cfg.ipmd.latent_learning.command_phase_mode),
            code_latent_dim=int(agent_cfg.ipmd.latent_learning.code_latent_dim),
            phase_period=int(agent_cfg.ipmd.latent_learning.code_period),
            command_mode=str(agent_cfg.ipmd.hl_skill_command_mode),
            discover_env_method=agent._discover_env_method,
            device=agent._get_device(agent.config.device),
        )
        agent._hl_skill_command_sampler = gr00t_sampler
        print(f"[INFO] GR00T latent command source: {gr00t_sampler.gr00t_provenance}")
    print(f"[INFO] Loading low-level checkpoint: {checkpoint_path}")
    resolved_policy_input_keys = tuple(agent_cfg.policy.get_input_keys())
    supported_latent_policy_input_keys = {
        tuple(LATENT_POLICY_INPUT_KEYS),
        tuple(SONIC_LATENT_POLICY_INPUT_KEYS),
    }
    if (
        resolved_policy_input_keys not in supported_latent_policy_input_keys
        and str(args_cli.gr00t_route) != "chunk_native"
    ):
        raise ValueError(
            "Closed-loop latent evaluation requires either the legacy/Strict "
            "or SONIC/Stable ordered actor-input contract, got "
            f"{resolved_policy_input_keys!r}."
        )
    if str(args_cli.gr00t_route) == "chunk_native":
        # A genuinely explicit tracker reads command terms instead of a
        # latent, so the latent contract above does not apply. The tracker's
        # own restore below is still strict, which is what actually protects
        # against a mismatched policy.
        print(
            f"[INFO] chunk_native actor-input contract: {resolved_policy_input_keys!r}"
        )
    frozen_tracker = load_frozen_low_level_tracker(
        agent,
        checkpoint_path,
        expected_input_keys=resolved_policy_input_keys,
        map_location=env_cfg.sim.device,
    )
    # Keep the strictly restored frozen policy module, but execute it through
    # the agent-owned collector wrapper so hl_skill / skill_commander commands
    # are published before the policy reads the latent observation.
    collector_policy = agent.collector_policy
    collector_policy.eval()
    tracker_provenance = frozen_tracker.provenance
    latent_receding_stats: Any = None
    latent_receding_provenance: dict[str, Any] | None = None
    if command_source == "skill_commander" and shared_target_spec is not None:
        planner_target_dim = int(shared_target_spec.target_dim)
        if planner_target_dim != int(trainer.z_dim):
            command_sampler = getattr(agent, "_hl_skill_command_sampler", None)
            if command_sampler is None:
                raise RuntimeError("The deployed H3 SkillCommander sampler is missing.")
            latent_receding_stats = install_latent_receding_horizon(
                command_sampler,
                env=raw_isaac_env,
                token_count=3,
                token_width=int(trainer.z_dim),
                mode=str(args_cli.latent_temporal_ensemble),
                decay=float(args_cli.latent_temporal_ensemble_decay),
                clip_std=float(args_cli.latent_temporal_clip_std),
                gate_distance=float(args_cli.latent_temporal_gate_distance),
                gate_cosine=float(args_cli.latent_temporal_gate_cosine),
            )
            checkpoint_metadata = planner_checkpoint.get("metadata", {})
            sample_metadata = (
                checkpoint_metadata.get("sample_metadata", {})
                if isinstance(checkpoint_metadata, dict)
                else {}
            )
            horizon_metadata = (
                sample_metadata.get("latent_receding_horizon", {})
                if isinstance(sample_metadata, dict)
                else {}
            )
            latent_receding_provenance = {
                "prediction_tokens": 3,
                "token_width": int(trainer.z_dim),
                "prediction_horizon_steps": 30,
                "execution_horizon_steps": int(trainer.horizon_steps),
                "target_frame": horizon_metadata.get("target_frame"),
                "skill_checkpoint_sha256": horizon_metadata.get(
                    "skill_checkpoint_sha256"
                ),
                "execution_mode": str(args_cli.latent_temporal_ensemble),
            }
            print(
                "[INFO] H3 latent receding horizon: "
                f"target_frame={latent_receding_provenance['target_frame']} "
                f"mode={args_cli.latent_temporal_ensemble}.",
                flush=True,
            )
        elif str(args_cli.latent_temporal_ensemble) != "first":
            raise ValueError(
                "Latent temporal ensembling requires an ordered H3 checkpoint; "
                f"this planner predicts only {planner_target_dim} values."
            )
    # BB1 shared-tracker comparison: drive THIS latent tracker from an explicit
    # packet planner routed through the frozen skill encoder, so the only thing
    # that differs from the latent row is the planner's output space. Requires
    # the oracle sampler, because that is the one holding the encoder.
    # NOTE: `packet_encoder_stats` / `packet_encoder_provenance` are initialized
    # before the GR00T routes above, because a chunk_encoded run installs the
    # packet->encoder source there and must not have it cleared here.
    packet_planner_metadata: dict[str, Any] | None = None
    packet_planner_target_dim: int | None = None
    packet_planner_path: Path | None = None
    planner_latency_timer: PlannerForwardTimer | None = None
    if args_cli.packet_planner_checkpoint is not None:
        if command_source != "hl_skill":
            raise ValueError(
                "--packet_planner_checkpoint requires "
                "agent.ipmd.command_source=hl_skill (the oracle sampler owns the "
                f"frozen skill encoder); got command_source={command_source!r}."
            )
        packet_sampler = getattr(agent, "_hl_skill_command_sampler", None)
        if packet_sampler is None:
            raise RuntimeError("No high-level command sampler was constructed.")
        packet_planner_path = (
            Path(args_cli.packet_planner_checkpoint).expanduser().resolve()
        )
        packet_planner, packet_spec, _packet_meta = load_planner_checkpoint(
            packet_planner_path, map_location=env_cfg.sim.device
        )
        # map_location only controls where the *tensors* are unpickled; the
        # reconstructed module is still built on CPU, so its parameters and
        # normalization buffers must be moved explicitly or the first forward
        # dies on a cuda/cpu addmm mismatch.
        packet_planner = packet_planner.to(device=env_cfg.sim.device)
        packet_planner_metadata = {
            **(_packet_meta if isinstance(_packet_meta, dict) else {}),
            **_parameter_counts(packet_planner),
        }
        packet_planner_target_dim = int(packet_spec.target_dim)
        if str(args_cli.packet_source) == "planner":
            # Use the same root-module CUDA hook as the direct latent route.
            # The timer therefore includes only packet_planner.forward and
            # excludes packet conversion, the frozen encoder, tracker, and sim.
            planner_latency_timer = PlannerForwardTimer(packet_planner)
        if str(packet_spec.interface) != str(args_cli.packet_interface):
            raise ValueError(
                f"Packet planner targets {packet_spec.interface!r} but "
                f"--packet_interface is {args_cli.packet_interface!r}."
            )
        packet_prediction_horizon = int(args_cli.packet_prediction_horizon_steps)
        if packet_prediction_horizon <= 0:
            packet_prediction_horizon = int(trainer.horizon_steps)
        packet_layout = PacketLayout.from_target_spec(
            packet_spec,
            packet_frames=packet_prediction_horizon,
        )

        def _packet_causal_state(env_ids: Tensor) -> Tensor:
            return _planner_state(
                raw_isaac_env.current_causal_planner_observation(
                    env_ids=env_ids,
                    history_steps=int(args_cli.state_history_steps),
                ),
                int(args_cli.state_history_steps),
            )

        packet_noise_reference = None
        if (
            float(args_cli.packet_noise_alpha) > 0.0
            or float(args_cli.z_noise_alpha) > 0.0
        ):
            if args_cli.noise_reference_samples is None:
                raise ValueError(
                    "BB3 noise requires --noise_reference_samples so both "
                    "alphas are calibrated against the same oracle packets."
                )
            ref = torch.load(
                Path(args_cli.noise_reference_samples).expanduser().resolve(),
                map_location="cpu",
                weights_only=False,
            )
            key = "causal_target" if "causal_target" in ref else "demonstration_target"
            packet_noise_reference = build_noise_reference(
                packet_sampler.skill_encoder, ref[key], packet_layout=packet_layout
            )
        packet_encoder_stats = install_packet_encoder_command_source(
            packet_sampler,
            planner=packet_planner,
            causal_state_provider=_packet_causal_state,
            env=raw_isaac_env,
            packet_layout=packet_layout,
            flow_num_inference_steps=int(args_cli.flow_num_inference_steps),
            flow_inference_noise_std=float(args_cli.flow_inference_noise_std),
            packet_source=str(args_cli.packet_source),
            packet_noise_alpha=float(args_cli.packet_noise_alpha),
            z_noise_alpha=float(args_cli.z_noise_alpha),
            noise_seed=int(args_cli.noise_seed),
            noise_reference=packet_noise_reference,
            temporal_ensemble_mode=str(args_cli.packet_temporal_ensemble),
            temporal_ensemble_decay=float(args_cli.packet_temporal_ensemble_decay),
        )
        packet_encoder_provenance = {
            "packet_source": str(args_cli.packet_source),
            "packet_planner_checkpoint": str(packet_planner_path),
            "packet_planner_sha256": _file_sha256(packet_planner_path),
            "packet_interface": str(packet_spec.interface),
            "packet_target_dim": int(packet_spec.target_dim),
            "encoder_input_width": int(trainer.horizon_steps)
            * packet_layout.frame_width,
            "packet_frames": int(trainer.horizon_steps),
            "planner_prediction_frames": packet_layout.packet_frames,
            "planner_prediction_width": packet_layout.packet_width,
            "packet_term_widths": list(packet_layout.term_widths),
            "packet_temporal_ensemble": str(args_cli.packet_temporal_ensemble),
            "packet_temporal_ensemble_decay": float(
                args_cli.packet_temporal_ensemble_decay
            ),
        }
        print(
            f"[INFO] BB1: {packet_spec.interface} planner "
            f"({packet_spec.target_dim}) -> frozen skill encoder -> z -> latent "
            "tracker.",
            flush=True,
        )
    if command_source == "skill_commander":
        command_sampler = getattr(agent, "_hl_skill_command_sampler", None)
        deployed_generator = getattr(command_sampler, "generator", None)
        if not isinstance(deployed_generator, torch.nn.Module):
            raise RuntimeError(
                "The deployed SkillCommander generator is unavailable for "
                "planner-only latency measurement."
            )
        planner_latency_timer = PlannerForwardTimer(deployed_generator)

    dt = getattr(env, "step_dt", None)
    if dt is None:
        dt = getattr(raw_isaac_env, "step_dt", None)
    planner_observation_spec = trainer.planner_observation_spec
    if not isinstance(planner_observation_spec, dict):
        if args_cli.save_rollout_training_samples:
            raise ValueError(
                "Planner checkpoint lacks the causal observation specification "
                "required to save Phase 2 samples."
            )
        planner_observation_spec = {}
    collection_stage = (
        "planner_rollout"
        if command_source == "skill_commander" or args_cli.packet_planner_checkpoint
        else "oracle_rollout"
    )
    language_metadata: dict[str, Any] = {
        "enabled": bool(trainer.condition_on_language),
        "embedding_dim": int(trainer.lang_embed_dim),
    }
    if trainer.condition_on_language:
        language_path = (
            Path(trainer.config.language_embeddings_path).expanduser().resolve()
        )
        language_metadata.update(
            {
                "embedding_path": str(language_path),
                "embedding_sha256": _file_sha256(language_path),
                "backend": trainer.language_table.get("backend"),
                "model": trainer.language_table.get("model"),
                "motion_count": len(trainer.motion_names),
            }
        )
        if command_source == "skill_commander":
            goal_name = str(agent_cfg.ipmd.skill_commander_goal_name).strip()
            goal_rank = int(agent_cfg.ipmd.skill_commander_goal_rank)
            if goal_name:
                language_metadata["goal_name"] = goal_name
                names = [str(name) for name in trainer.language_table.get("names", [])]
                phrases = [
                    str(phrase) for phrase in trainer.language_table.get("phrases", [])
                ]
                if goal_name in names:
                    goal_index = names.index(goal_name)
                    if goal_index < len(phrases):
                        language_metadata["goal_phrase"] = phrases[goal_index]
            elif goal_rank >= 0:
                language_metadata["goal_rank"] = goal_rank
    sample_target_interface = str(args_cli.sample_target_interface).strip()
    if not sample_target_interface:
        raise ValueError("--sample_target_interface must not be empty.")
    if sample_target_interface == "latent_skill":
        sample_target_spec = {
            "interface": "latent_skill",
            "term_names": ["z"],
            "term_widths": [int(trainer.z_dim)],
            "target_dim": int(trainer.z_dim),
        }
        sample_command_future_steps = int(trainer.horizon_steps)
    else:
        sample_packet_layout = _macro_packet_layout(
            raw_isaac_env, horizon_steps=int(trainer.horizon_steps)
        )
        sample_target_spec = {
            "interface": sample_target_interface,
            "term_names": [name for name, _ in sample_packet_layout.term_widths],
            "term_widths": [
                width * sample_packet_layout.packet_frames
                for _, width in sample_packet_layout.term_widths
            ],
            "target_dim": sample_packet_layout.packet_width,
        }
        sample_command_future_steps = sample_packet_layout.packet_frames - 1
    latent_target_spec = {
        "interface": "latent_skill",
        "term_names": ["z"],
        "term_widths": [int(trainer.z_dim)],
        "target_dim": int(trainer.z_dim),
    }
    if sample_target_interface == "latent_skill":
        sample_packet_layout = _macro_packet_layout(
            raw_isaac_env, horizon_steps=int(trainer.horizon_steps)
        )
        packet_target_spec = {
            "interface": "encoder_input_packet",
            "term_names": [name for name, _ in sample_packet_layout.term_widths],
            "term_widths": [
                width * sample_packet_layout.packet_frames
                for _, width in sample_packet_layout.term_widths
            ],
            "target_dim": sample_packet_layout.packet_width,
        }
    else:
        packet_target_spec = dict(sample_target_spec)

    sample_metadata = add_sample_format_metadata(
        {
            "interface": sample_target_interface,
            "target_spec": sample_target_spec,
            "paired_interface_target_specs": {
                "latent_skill_target": latent_target_spec,
                "encoder_input_packet_target": packet_target_spec,
            },
            "state_history_steps": int(trainer.config.state_history_steps),
            "command_past_steps": 0,
            "command_future_steps": sample_command_future_steps,
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "seed": int(agent_cfg.seed),
            "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
            "motion_name": selected_motion_name or None,
            "motion_names": selected_motion_names,
            "trajectory_ranks": trajectory_ranks,
            "goal_motion_match_required": bool(args_cli.require_goal_motion_match),
            "balanced_collection": (
                {
                    "motion_names": list(balanced_selector.motion_names),
                    "rows_per_motion": balanced_selector.rows_per_motion,
                }
                if balanced_selector is not None
                else None
            ),
            "balanced_trajectory_collection": (
                {
                    "motion_names": list(balanced_trajectory_motion_names),
                    "trajectories_per_motion": int(
                        args_cli.balanced_trajectories_per_motion
                    ),
                }
                if balanced_trajectory_motion_names is not None
                else None
            ),
            "collection_unit": (
                "completed_trajectory"
                if balanced_trajectory_motion_names is not None
                else "planner_publication_row"
            ),
            "planner_state_source": "causal_robot_history_during_oracle_policy_rollout",
            "auxiliary_targets": {
                "root_qpos": {
                    "enabled": bool(args_cli.require_root_qpos_samples),
                    "frame_dim": 38 if args_cli.require_root_qpos_samples else None,
                    "window_frames": int(args_cli.sample_future_window_frames),
                    "window_key": "expert_root_qpos_future",
                    "validity_key": "expert_root_qpos_future_valid",
                    "achieved_key": "achieved_root_qpos",
                }
            },
            "planner_observation_spec": dict(planner_observation_spec),
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
            "policy_action_selection": "deterministic",
            "early_terminations_enabled": bool(
                args_cli.keep_early_terminations
                or args_cli.disable_tracking_terminations
                or args_cli.base_only_termination
                or args_cli.sonic_success_terminations
            ),
            "tracking_terminations_enabled": not bool(
                args_cli.disable_tracking_terminations or args_cli.base_only_termination
            ),
            "base_only_termination": bool(args_cli.base_only_termination),
            "sonic_success_terminations": bool(args_cli.sonic_success_terminations),
            "sonic_termination_profile": sonic_termination_record,
            "fall_height_m": float(args_cli.fall_height_m),
            "disabled_tracking_termination_terms": (
                disabled_tracking_termination_terms
            ),
            "survival_definition": (
                "reference_finished_without_sonic_tracking_failure"
                if args_cli.sonic_success_terminations
                else "no_base_too_low_termination"
            ),
            "time_out_enabled": bool(args_cli.keep_time_out),
            "episode_length_extension_enabled": bool(
                args_cli.extend_episode_length_for_max_steps
            ),
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
            "reward_clipping_enabled": not bool(args_cli.disable_reward_clipping),
            "push_perturbation": push_event_record,
            "deterministic_tracking": deterministic_tracking_record,
            "language_conditioning": language_metadata,
            "provenance": {
                "low_level_checkpoint": str(checkpoint_path),
                "low_level_tracker": tracker_provenance,
                # BB1: present only when the command came from an explicit packet
                # planner through the frozen encoder. `publishes` counts encoder
                # calls so a summary cannot silently report the latent oracle path.
                "packet_encoder_command": (
                    None
                    if packet_encoder_provenance is None
                    else {
                        **packet_encoder_provenance,
                        **(packet_encoder_stats() if packet_encoder_stats else {}),
                    }
                ),
                "planner_checkpoint": (
                    str(planner_checkpoint_path)
                    if planner_checkpoint_path is not None
                    else ""
                ),
                # Present only when a GR00T action head published the latents.
                # `head_calls` counts head forwards, so a summary cannot
                # silently report the oracle path as a planner result.
                "gr00t_planner": (
                    gr00t_sampler.gr00t_report()
                    if gr00t_sampler is not None
                    else (
                        {
                            "route": str(args_cli.gr00t_route),
                            **gr00t_chunk_publisher.report(),
                            **(
                                gr00t_packet_planner.report()
                                if gr00t_packet_planner is not None
                                and hasattr(gr00t_packet_planner, "report")
                                else {}
                            ),
                        }
                        if gr00t_chunk_publisher is not None
                        else None
                    )
                ),
                "skill_checkpoint": str(
                    args_cli.skill_checkpoint
                    or planner_checkpoint.get("skill_checkpoint_path", "")
                    or trainer_config.skill_checkpoint_path
                ),
                "motion_manifest": str(getattr(env_cfg, "lafan1_manifest_path", "")),
            },
        },
        collection_stage=collection_stage,
        planner_interval_steps=int(trainer.horizon_steps),
        control_rate_hz=(1.0 / float(dt)) if dt else 50.0,
    )
    env_ids = torch.arange(
        int(env_cfg.scene.num_envs),
        device=torch.device(getattr(raw_isaac_env, "device", env_cfg.sim.device)),
        dtype=torch.long,
    )
    td = env.reset()
    start_metadata = _trajectory_metadata(raw_isaac_env)
    if args_cli.require_goal_motion_match and set(start_metadata["motion_names"]) != {
        selected_motion_name
    }:
        raise RuntimeError(
            "Explicit goal-to-reference binding failed at reset: "
            f"expected {selected_motion_name!r}, observed "
            f"{sorted(set(start_metadata['motion_names']))}."
        )
    language_mode = (
        "motion-name embedding" if bool(trainer.condition_on_language) else "none"
    )
    print(
        "[INFO] Conditioning: "
        f"language={language_mode} trajectories={start_metadata['motion_names']}"
    )
    print(f"[INFO] Rollout steps: {max_steps} (auto_reference_steps={auto_steps})")

    num_envs = int(env_cfg.scene.num_envs)
    episode_ids = torch.zeros(num_envs, dtype=torch.long)
    active = torch.ones(num_envs, dtype=torch.bool)
    survival_steps = torch.zeros(num_envs, dtype=torch.float32)
    return_sum = torch.zeros(num_envs, dtype=torch.float32)
    done_events = torch.zeros(num_envs, dtype=torch.float32)
    terminated_events = torch.zeros(num_envs, dtype=torch.float32)
    truncated_events = torch.zeros(num_envs, dtype=torch.float32)
    rollout_metric_stats: dict[str, list[Tensor]] = {}
    per_env_metric_sums: dict[str, Tensor] = {}
    per_env_metric_counts: dict[str, Tensor] = {}
    strict_tracking_failure_events = torch.zeros(num_envs, dtype=torch.float32)
    termination_term_names = list(raw_isaac_env.termination_manager.active_terms)
    termination_hits = {
        term_name: torch.zeros(num_envs, dtype=torch.bool)
        for term_name in termination_term_names
    }
    strict_failure_term_names: list[str] = []
    if (
        args_cli.keep_early_terminations
        or args_cli.disable_tracking_terminations
        or args_cli.sonic_success_terminations
    ):
        for term_name in raw_isaac_env.termination_manager.active_terms:
            term_cfg = raw_isaac_env.termination_manager.get_term_cfg(term_name)
            if not term_cfg.time_out and term_name != "reference_finished":
                strict_failure_term_names.append(term_name)
    previous_action: Tensor | None = None
    previous_body_lin_vel: tuple[Tensor, Tensor] | None = None
    previous_velocity_valid = torch.zeros(num_envs, dtype=torch.bool)
    tracking_failure_events = torch.zeros(num_envs, dtype=torch.float32)
    valid_transition_count = 0
    rows: list[dict[str, Any]] = []
    samples_dir = log_dir / "rollout_training_samples"
    trajectory_sample_writer: CompletedTrajectorySampleWriter | None = None
    if balanced_trajectory_motion_names is not None:
        trajectory_sample_writer = CompletedTrajectorySampleWriter(
            samples_dir,
            motion_names=balanced_trajectory_motion_names,
            trajectories_per_motion=int(args_cli.balanced_trajectories_per_motion),
            rows_per_file=int(args_cli.sample_rows_per_file),
        )
        sample_writer: PlannerSampleWriter | CompletedTrajectorySampleWriter = (
            trajectory_sample_writer
        )
    else:
        sample_writer = PlannerSampleWriter(
            samples_dir,
            rows_per_file=int(args_cli.sample_rows_per_file),
        )
    saved_sample_files = 0
    saved_sample_rows = 0
    timestep = 0
    stop_reason = "max_steps"
    trajectory_budget_complete = False
    if int(args_cli.metric_interval) <= 0:
        raise ValueError("--metric_interval must be > 0.")
    while timestep < max_steps:
        start_time = time.time()
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            step_active = active.clone()
            should_measure = timestep % int(args_cli.metric_interval) == 0
            renew_env_ids = planner_renew_env_ids(
                base_env.episode_length_buf,
                int(trainer.horizon_steps),
                initial_publication=timestep == 0,
            )
            if gr00t_chunk_publisher is not None and hasattr(
                gr00t_chunk_publisher, "note_step"
            ):
                # Async native route: drain service replies and fire lead-time
                # requests once per control step, renewal or not.
                gr00t_chunk_publisher.note_step()
            if (
                gr00t_chunk_publisher is not None
                and str(args_cli.gr00t_route) == "chunk_native"
                and int(renew_env_ids.numel()) > 0
            ):
                # Native chunk route: publish the head's explicit packet into
                # the chunk actor term on the same per-environment renewal
                # schedule the latent routes use. Asynchronous resets make a
                # global timestep-modulo publication invalid, which is why this
                # rides `planner_renew_env_ids`.
                gr00t_chunk_publisher.publish(renew_env_ids)
            if int(renew_env_ids.numel()) > 0:
                active_on_device = step_active.to(device=renew_env_ids.device)
                renew_env_ids = renew_env_ids[
                    active_on_device.index_select(0, renew_env_ids)
                ]
            sample_env_ids = renew_env_ids
            if bool(args_cli.sample_every_control_step):
                # Hold-1 collection: the planner must supply one latent per
                # control step, so `_join_slots(hold_steps=1)` needs a row at
                # every step rather than only at the publication boundaries the
                # renewal mask selects. Publication itself is untouched — the
                # tracker still consumes on its own schedule; only the sampling
                # rate changes.
                sample_env_ids = step_active.nonzero(as_tuple=False).reshape(-1)
            sample_motion_names: list[str] = []
            current_motion_names: list[str] = (
                _trajectory_metadata(raw_isaac_env)["motion_names"]
                if trajectory_sample_writer is not None
                or bool(args_cli.sample_every_control_step)
                else []
            )
            if bool(args_cli.sample_every_control_step):
                # The balanced writer is the only other producer of per-row
                # motion names, and a hold-1 collection does not use it: rows
                # are written every step rather than per completed trajectory.
                # Without this the sample builder receives an empty name list.
                sample_motion_names = [
                    current_motion_names[int(env_id)]
                    for env_id in sample_env_ids.detach().cpu().tolist()
                ]
            gr00t_goal_owner = gr00t_sampler or gr00t_chunk_publisher
            if (
                gr00t_goal_owner is not None
                and args_cli.gr00t_goals_per_env
                and int(renew_env_ids.numel()) > 0
            ):
                # Per-environment goals are assigned once at start-up; the
                # reference channel can reassign a trajectory on reset. Verify
                # every publication rather than let language and motion drift
                # apart silently.
                if not current_motion_names:
                    current_motion_names = _trajectory_metadata(raw_isaac_env)[
                        "motion_names"
                    ]
                renew_cpu = renew_env_ids.detach().cpu()
                gr00t_goal_owner.gr00t_assert_goal_matches(
                    renew_cpu,
                    [current_motion_names[int(i)] for i in renew_cpu.tolist()],
                )
            if (
                bool(args_cli.save_rollout_training_samples)
                and int(renew_env_ids.numel()) > 0
            ):
                renew_env_ids_cpu = renew_env_ids.detach().cpu()
                if not current_motion_names:
                    current_motion_names = _trajectory_metadata(raw_isaac_env)[
                        "motion_names"
                    ]
                candidate_motion_names = [
                    current_motion_names[int(index)]
                    for index in renew_env_ids_cpu.tolist()
                ]
                if args_cli.require_goal_motion_match and set(
                    candidate_motion_names
                ) != {selected_motion_name}:
                    raise RuntimeError(
                        "Explicit goal-to-reference binding changed during "
                        "collection: "
                        f"expected {selected_motion_name!r}, observed "
                        f"{sorted(set(candidate_motion_names))}."
                    )
                if balanced_selector is not None:
                    selected_indices = torch.tensor(
                        balanced_selector.select(candidate_motion_names),
                        dtype=torch.long,
                        device=renew_env_ids.device,
                    )
                    sample_env_ids = renew_env_ids.index_select(0, selected_indices)
                    sample_motion_names = [
                        candidate_motion_names[int(index)]
                        for index in selected_indices.detach().cpu().tolist()
                    ]
                else:
                    sample_motion_names = candidate_motion_names
            should_save = (
                bool(args_cli.save_rollout_training_samples)
                and int(sample_env_ids.numel()) > 0
            )
            metric_row: dict[str, Any] = {}
            if planner_latency_timer is None:
                td = collector_policy(td)
            else:
                with planner_latency_timer.enabled():
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
                        wrapped_env=raw_isaac_env,
                        env_ids=env_ids,
                    )
                )
                # Root-relative MPJPE, the repo's headline tracking metric,
                # computed exactly as the 4,096-motion tracker scoreboard does
                # so planner numbers are comparable to the oracle ceilings.
                mpjpe_pair_mm = _tracking_mpjpe_pair_mm(raw_isaac_env)
                if mpjpe_pair_mm is not None:
                    mpjpe_mm, mpjpe_g_mm = mpjpe_pair_mm
                    active_mask = step_active.to(mpjpe_mm.device)
                    if bool(active_mask.any()):
                        metric_row["tracking_mpjpe_mm"] = float(
                            mpjpe_mm[active_mask].mean()
                        )
                    # World-frame counterpart, which keeps the global drift the
                    # root-relative metric is blind to. Reported alongside rather
                    # than instead: an arm can track posture well while drifting,
                    # and the two numbers separate those failures.
                    active_mask = step_active.to(mpjpe_g_mm.device)
                    if bool(active_mask.any()):
                        metric_row["tracking_mpjpe_g_mm"] = float(
                            mpjpe_g_mm[active_mask].mean()
                        )
            if should_save:
                sample_env_ids_cpu = sample_env_ids.detach().cpu()
                _measure_commander(
                    trainer=trainer,
                    wrapped_env=raw_isaac_env,
                    env_ids=sample_env_ids,
                    sample_writer=sample_writer,
                    sample_step=timestep,
                    sample_metadata=sample_metadata,
                    episode_ids=episode_ids.index_select(0, sample_env_ids_cpu),
                    sample_motion_names=sample_motion_names,
                    sample_target_interface=sample_target_interface,
                    sample_future_window_frames=int(
                        args_cli.sample_future_window_frames
                    ),
                    require_root_qpos_samples=bool(args_cli.require_root_qpos_samples),
                    compute_metrics=False,
                )
            if should_measure:
                row = {
                    "step": int(timestep),
                    **_trajectory_metadata(raw_isaac_env),
                    **metric_row,
                }
                _write_jsonl(metrics_path, row)
                rows.append(row)
            if should_save:
                saved_sample_rows += int(sample_env_ids.numel())
            if balanced_selector is not None and balanced_selector.complete:
                stop_reason = "balanced_rows_complete"
                break
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
            current_termination_terms: dict[str, Tensor] = {}
            for term_name in termination_term_names:
                term_values = (
                    raw_isaac_env.termination_manager.get_term(term_name)
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
            completed_now = done_any & step_active
            if trajectory_sample_writer is not None and bool(completed_now.any()):
                done_env_ids = completed_now.nonzero(as_tuple=True)[0]
                trajectory_sample_writer.complete(
                    env_ids=done_env_ids.tolist(),
                    episode_ids=episode_ids.index_select(0, done_env_ids).tolist(),
                    motion_names=[
                        current_motion_names[int(env_id)]
                        for env_id in done_env_ids.tolist()
                    ],
                    termination_reasons=[
                        [
                            term_name
                            for term_name, values in current_termination_terms.items()
                            if bool(values[int(env_id)].item())
                        ]
                        for env_id in done_env_ids.tolist()
                    ],
                    tracking_success=[
                        not bool(strict_failure[int(env_id)].item())
                        for env_id in done_env_ids.tolist()
                    ],
                )
                trajectory_budget_complete = trajectory_sample_writer.complete_budget
            episode_ids += done_any.to(dtype=torch.long)
            return_sum += rewards.float() * step_active.float()
            survival_steps += step_active.float()
            done_events += (done_any & step_active).float()
            terminated_events += (terminateds & step_active).float()
            truncated_events += (truncateds & step_active).float()
            metric_mask = (
                step_active
                if args_cli.continue_after_reset
                else step_active & ~done_any
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
                values_cpu = values.detach().cpu().float()
                _accumulate_metric(
                    rollout_metric_stats, metric_name, values_cpu, metric_mask
                )
                per_env_metric_sums.setdefault(
                    metric_name, torch.zeros(num_envs, dtype=torch.float32)
                ).add_(values_cpu * metric_mask.float())
                per_env_metric_counts.setdefault(
                    metric_name, torch.zeros(num_envs, dtype=torch.long)
                ).add_(metric_mask.to(dtype=torch.long))
            if body_lin_vel is not None:
                if previous_body_lin_vel is not None and dt is not None:
                    actual_lin_vel, ref_lin_vel = body_lin_vel
                    prev_actual_lin_vel, prev_ref_lin_vel = previous_body_lin_vel
                    actual_acc = (actual_lin_vel - prev_actual_lin_vel) / float(dt)
                    ref_acc = (ref_lin_vel - prev_ref_lin_vel) / float(dt)
                    acceleration_distance = torch.linalg.vector_norm(
                        actual_acc - ref_acc, dim=-1
                    ).mean(dim=-1)
                    acceleration_mask = metric_mask & previous_velocity_valid
                    _accumulate_metric(
                        rollout_metric_stats,
                        "tracking_acceleration_distance_mps2",
                        acceleration_distance.cpu(),
                        acceleration_mask,
                    )
                previous_body_lin_vel = (
                    body_lin_vel[0].clone(),
                    body_lin_vel[1].clone(),
                )
                previous_velocity_valid = step_active & ~done_any
            if not args_cli.continue_after_reset:
                active &= ~done_any
            td = step_mdp(
                stepped_td,
                exclude_reward=True,
                exclude_done=False,
                exclude_action=True,
            )

        timestep += 1
        if trajectory_budget_complete:
            stop_reason = "balanced_trajectories_complete"
            break
        if not args_cli.continue_after_reset and not bool(active.any()):
            stop_reason = "all_envs_done"
            print(f"[INFO] Stopping at step {timestep}: all environments are done.")
            break
        if args_cli.real_time and dt is not None:
            sleep_time = float(dt) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    sample_writer.flush()
    saved_sample_files = sample_writer.file_count
    if trajectory_sample_writer is not None:
        saved_sample_rows = sample_writer.row_count
    if saved_sample_rows != sample_writer.row_count:
        raise RuntimeError(
            "Planner sample writer row accounting differs from collection: "
            f"collected={saved_sample_rows}, written={sample_writer.row_count}."
        )
    final_metadata = _trajectory_metadata(raw_isaac_env)
    active_mask = survival_steps > 0
    return_mean, return_std = _tensor_mean_std(return_sum, active_mask)
    survival_mean, survival_std = _tensor_mean_std(survival_steps, active_mask)
    fall_events = termination_hits.get(
        FALL_TERMINATION_NAME, torch.zeros(num_envs, dtype=torch.bool)
    )
    fall_free = ~fall_events
    reference_finished_events = termination_hits.get(
        "reference_finished", torch.zeros(num_envs, dtype=torch.bool)
    )
    completed_tracking_success = reference_finished_events & (
        strict_tracking_failure_events == 0
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
        "reference_finished_rate": float(
            reference_finished_events[active_mask].float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "completed_tracking_success_rate": float(
            completed_tracking_success[active_mask].float().mean().item()
        )
        if bool(active_mask.any())
        else float("nan"),
        "completed_tracking_success_count": int(
            completed_tracking_success[active_mask].sum().item()
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
        "termination_cause_env_counts": {
            term_name: int(values[active_mask].sum().item())
            for term_name, values in termination_hits.items()
        },
    }
    metric_means = _mean_dict(rows)
    rollout_metrics = _finalize_metric_stats(rollout_metric_stats)
    successful_trajectory_metrics: dict[str, dict[str, float | int]] = {}
    for metric_name in sorted(per_env_metric_sums):
        success_counts = per_env_metric_counts[metric_name][completed_tracking_success]
        total_count = int(success_counts.sum().item())
        if total_count <= 0:
            continue
        total = float(
            per_env_metric_sums[metric_name][completed_tracking_success].sum().item()
        )
        successful_trajectory_metrics[metric_name] = {
            "mean": total / total_count,
            "count": total_count,
            "trajectory_count": int(completed_tracking_success.sum().item()),
        }
    if "m3/z_mse" in metric_means:
        rollout_metrics["planner_target_rmse"] = {
            "mean": float(max(metric_means["m3/z_mse"], 0.0) ** 0.5),
            "std": 0.0,
            "count": int(len(rows)),
        }
    if args_cli.deterministic_tracking:
        # Rename before anything consumes these. The paper aggregators look up
        # bare names such as "tracking_mpjpe_mm", so an unperturbed result file
        # makes them fail loudly instead of silently pooling two protocols.
        rollout_metrics = {
            f"{DETERMINISTIC_METRIC_PREFIX}{name}": value
            for name, value in rollout_metrics.items()
        }
    planner_metadata = (
        packet_planner_metadata
        if packet_planner_metadata is not None
        else _skill_commander_planner_metadata(
            planner_checkpoint,
            generator=trainer.generator,
            trainer_config=trainer_config,
        )
    )
    summary = {
        **config_payload,
        "metadata": {
            "label": args_cli.label,
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "checkpoint": str(checkpoint_path),
            "low_level_tracker": tracker_provenance,
            # Present only when a GR00T action head produced the commands.
            # `head_calls` / `publications` count real head forwards, so an
            # oracle run can never be read as a planner result.
            "gr00t_planner": (
                gr00t_sampler.gr00t_report()
                if gr00t_sampler is not None
                else (
                    {
                        "route": str(args_cli.gr00t_route),
                        **gr00t_chunk_publisher.report(),
                        **(
                            gr00t_packet_planner.report()
                            if gr00t_packet_planner is not None
                            and hasattr(gr00t_packet_planner, "report")
                            else {}
                        ),
                    }
                    if gr00t_chunk_publisher is not None
                    else None
                )
            ),
            # BB1: present only when the command came from an explicit packet
            # planner through the frozen encoder. `publishes` counts encoder
            # calls so a summary cannot silently report the latent oracle path.
            "packet_encoder_command": (
                None
                if packet_encoder_provenance is None
                else {
                    **packet_encoder_provenance,
                    **(packet_encoder_stats() if packet_encoder_stats else {}),
                }
            ),
            "latent_receding_horizon": (
                None
                if latent_receding_provenance is None
                else {
                    **latent_receding_provenance,
                    **(latent_receding_stats() if latent_receding_stats else {}),
                }
            ),
            "planner_checkpoint": (
                str(packet_planner_path)
                if packet_planner_path is not None
                else (
                    str(planner_checkpoint_path)
                    if planner_checkpoint_path is not None
                    else None
                )
            ),
            "interface": (
                str(args_cli.packet_interface)
                if packet_planner_path is not None
                else "latent_skill"
            ),
            "planner_target_dim": (
                int(packet_planner_target_dim)
                if packet_planner_target_dim is not None
                else (
                    int(shared_target_spec.target_dim)
                    if shared_target_spec is not None
                    else int(trainer.z_dim)
                )
            ),
            "planner_metadata": planner_metadata,
            "num_envs": int(num_envs),
            "seed": int(agent_cfg.seed),
            "motion_name": selected_motion_name or None,
            "motion_names": selected_motion_names,
            "goal_motion_match_required": bool(args_cli.require_goal_motion_match),
            "trajectory_name": args_cli.trajectory_name,
            "motion_manifest": str(getattr(env_cfg, "lafan1_manifest_path", "")),
            "dataset_path": str(getattr(env_cfg, "dataset_path", "")),
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
            "policy_action_selection": "deterministic",
            "early_terminations_enabled": bool(
                args_cli.keep_early_terminations
                or args_cli.disable_tracking_terminations
                or args_cli.base_only_termination
                or args_cli.sonic_success_terminations
            ),
            "tracking_terminations_enabled": not bool(
                args_cli.disable_tracking_terminations or args_cli.base_only_termination
            ),
            "base_only_termination": bool(args_cli.base_only_termination),
            "sonic_success_terminations": bool(args_cli.sonic_success_terminations),
            "sonic_termination_profile": sonic_termination_record,
            "fall_height_m": float(args_cli.fall_height_m),
            "disabled_tracking_termination_terms": (
                disabled_tracking_termination_terms
            ),
            "survival_definition": (
                "reference_finished_without_sonic_tracking_failure"
                if args_cli.sonic_success_terminations
                else "no_base_too_low_termination"
            ),
            "time_out_enabled": bool(args_cli.keep_time_out),
            "episode_length_extension_enabled": bool(
                args_cli.extend_episode_length_for_max_steps
            ),
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
            "reward_clipping_enabled": not bool(args_cli.disable_reward_clipping),
            "push_perturbation": push_event_record,
            "deterministic_tracking": deterministic_tracking_record,
            "language_conditioning": language_metadata,
        },
        "aggregate": aggregate,
        "metrics": rollout_metrics,
        "successful_trajectory_metrics": successful_trajectory_metrics,
        "planner_inference_latency_ms": (
            planner_latency_timer.summary(warmup_calls=1)
            if planner_latency_timer is not None
            else None
        ),
        "output_dir": str(log_dir),
        "video_dir": str(log_dir / "videos" / "play") if args_cli.video else None,
        "planner_config": trainer_config.to_dict(),
        "planner_update": int(trainer.update),
        "planner_target_dim": (
            int(packet_planner_target_dim)
            if packet_planner_target_dim is not None
            else (
                int(shared_target_spec.target_dim)
                if shared_target_spec is not None
                else int(trainer.z_dim)
            )
        ),
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
        "balanced_trajectory_collection": (
            {
                "motion_names": list(trajectory_sample_writer.motion_names),
                "trajectories_per_motion": (
                    trajectory_sample_writer.trajectories_per_motion
                ),
                "counts": trajectory_sample_writer.counts(),
                "completed_trajectory_count": (
                    trajectory_sample_writer.completed_trajectory_count
                ),
                "records": trajectory_sample_writer.records(),
                "complete": trajectory_sample_writer.complete_budget,
                "missing": trajectory_sample_writer.missing(),
                "discarded_incomplete_trajectory_count": (
                    trajectory_sample_writer.buffered_trajectory_count
                ),
            }
            if trajectory_sample_writer is not None
            else None
        ),
        "per_environment": [
            {
                "env_id": env_id,
                "trajectory_rank": int(start_metadata["trajectory_ranks"][env_id]),
                "motion_name": str(start_metadata["motion_names"][env_id]),
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
                "reference_finished": bool(reference_finished_events[env_id].item()),
                "completed_tracking_success": bool(
                    completed_tracking_success[env_id].item()
                ),
                "tracking_metrics": {
                    metric_name: (
                        float(per_env_metric_sums[metric_name][env_id].item())
                        / int(per_env_metric_counts[metric_name][env_id].item())
                    )
                    for metric_name in sorted(per_env_metric_sums)
                    if int(per_env_metric_counts[metric_name][env_id].item()) > 0
                },
                "tracking_metric_counts": {
                    metric_name: int(per_env_metric_counts[metric_name][env_id].item())
                    for metric_name in sorted(per_env_metric_counts)
                    if int(per_env_metric_counts[metric_name][env_id].item()) > 0
                },
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if planner_latency_timer is not None:
        planner_latency_timer.close()
    env.close()
    if balanced_selector is not None and not balanced_selector.complete:
        raise RuntimeError(
            "Balanced collection ended before every motion reached its row budget: "
            f"{balanced_selector.missing()}."
        )
    if (
        trajectory_sample_writer is not None
        and not trajectory_sample_writer.complete_budget
    ):
        raise RuntimeError(
            "Balanced collection ended before every motion reached its completed "
            f"trajectory budget: {trajectory_sample_writer.missing()}."
        )


if __name__ == "__main__":
    resolved_env_cfg, resolved_agent_cfg = resolve_task_config(
        args_cli.task,
        agent_entry_point,
    )
    print("[INFO] Closed-loop evaluator task configuration resolved.", flush=True)
    needs_kit, _, _ = compute_kit_requirements(resolved_env_cfg, args_cli)
    if args_cli.assert_kitless:
        if needs_kit or not config_contains_type_name(resolved_env_cfg, "NewtonCfg"):
            raise RuntimeError(
                "--assert-kitless requires a resolved NewtonCfg with no Kit cameras "
                "or Kit visualizer. Pass physics=newton_mjwarp."
            )
        assert_kit_not_loaded()
        print(
            "[INFO] Strict kit-less Newton evaluator runtime validated.",
            flush=True,
        )
    if os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1" and needs_kit:
        raise RuntimeError(
            "The split runtime cannot launch Kit from this evaluator. "
            "Use physics=newton_mjwarp with --assert-kitless on compute-only GPUs."
        )
    with launch_simulation(resolved_env_cfg, args_cli):
        main(resolved_env_cfg, resolved_agent_cfg)
    if args_cli.assert_kitless:
        assert_kit_not_loaded()
        print("[INFO] Strict kit-less evaluator invariant held through shutdown.")
