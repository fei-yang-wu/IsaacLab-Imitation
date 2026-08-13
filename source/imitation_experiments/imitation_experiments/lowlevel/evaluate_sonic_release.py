#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate NVIDIA's released SONIC G1 tracker inside our environment.

This drives `sonic_release/last.pt` (HF ``nvidia/GEAR-SONIC``) directly: no
RLOpt agent, no `gear_sonic` import, evaluation only. The released actor is a
plain feed-forward stack, so a small adapter is enough:

    reference window (640) -> encoder -> FSQ -> token (64)
    token + proprioception (930) -> decoder -> joint targets (29)

Everything physical already matches (see
``experiments/campaigns/2026-08-07-sonic-release-tier2/README.md``): identical
G1 joint order, identical actuator gains and action scale via
``UNITREE_G1_29DOF_SONIC_CFG``, and identical 50 Hz control. What this script
supplies is the observation plumbing, and it asserts the two conventions that
would otherwise fail silently rather than loudly.

This is a native reproduction attempt, not a "SONIC baseline": every deviation
we could not reproduce is recorded in the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from isaaclab.app import AppLauncher

from imitation_experiments.audit.backend_determinism import (
    RANDOMIZATION_PROFILES,
    apply_randomization_profile,
    pin_reference_start,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v2")
parser.add_argument(
    "--sonic_checkpoint",
    type=Path,
    required=True,
    help="Released SONIC PPO snapshot (sonic_release/last.pt).",
)
parser.add_argument(
    "--sonic_version",
    choices=("release", "v1_1", "auto"),
    default="release",
    help=(
        "Checkpoint observation contract. 'release' uses full anchor "
        "orientation; 'v1_1' derives SONIC v1.1 heading-only anchor "
        "orientation from the same reference window. 'auto' infers this from "
        "the decoder shape."
    ),
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--label", type=str, default="sonic_release")
parser.add_argument("--output_json", type=Path, default=None)
parser.add_argument(
    "--randomization", type=str, default="no_push", choices=RANDOMIZATION_PROFILES
)
parser.add_argument("--reference_start_frame", type=int, default=0)
parser.add_argument(
    "--trajectory_ranks",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Exact trajectory ranks to evaluate. The 'sequential' schedule advances "
        "a global cursor, so two runs can silently score different motion "
        "blocks; pin the ranks when comparing against another checkpoint."
    ),
)
parser.add_argument(
    "--reset_schedule",
    type=str,
    default="sequential",
    help=(
        "Reference schedule after the start frame is pinned. The default matches "
        "evaluate_checkpoint's ordered motion assignment so the two tools score "
        "the same motion block unless --trajectory_ranks pins an exact set."
    ),
)
parser.add_argument(
    "--termination_contract",
    choices=("sonic", "task", "fall_only"),
    default="sonic",
    help=(
        "Termination contract for reportable release scores. 'sonic' applies "
        "SONIC's released thresholds and disables foot_pos_xyz/base_too_low. "
        "'fall_only' is the M3 planner contract: every tracking termination "
        "off, `base_too_low` the only failure, so surviving means not falling "
        "and tracking error stays a continuous metric. 'task' keeps the task "
        "config and is for diagnostics only."
    ),
)
parser.add_argument(
    "--disable_early_terminations",
    action="store_true",
    default=False,
    help="Full-horizon diagnostic pass: no termination truncates the rollout.",
)
parser.add_argument(
    "--preserve_episode_length",
    action="store_true",
    default=False,
    help="Do not extend env.episode_length_s to cover --steps.",
)
parser.add_argument(
    "--allow_incomplete_release",
    action="store_true",
    default=False,
    help=(
        "Allow writing a reportable release JSON even when not all environments "
        "reach reference_finished before the step cap. Intended for debugging."
    ),
)
parser.add_argument(
    "--proprioception_order",
    choices=("gravity_last", "gravity_first"),
    default="gravity_last",
    help=(
        "gravity_last follows SONIC's PolicyCfg field declaration order, which "
        "is what Isaac Lab concatenates by. gravity_first follows the training "
        "YAML's key order. They disagree; this switch decides it empirically."
    ),
)
parser.add_argument(
    "--history_order",
    choices=("oldest_first", "newest_first"),
    default="oldest_first",
    help="Frame order within each history term (Isaac Lab buffers oldest first).",
)
parser.add_argument(
    "--action_clip",
    type=float,
    default=20.0,
    help="SONIC's env config sets action_clip_value=20.0.",
)
parser.add_argument(
    "--action_source",
    choices=("policy", "zeros"),
    default="policy",
    help=(
        "Diagnostic control. 'zeros' holds the default pose instead of the "
        "released decoder's output; if the rollout is unchanged, our action "
        "tensor is not reaching the articulation and no number here is real."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Render the rollout. Implies --enable_cameras.",
)
parser.add_argument("--video_dir", type=Path, default=None)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument(
    "--diagnose_steps",
    type=int,
    default=0,
    help=(
        "Print per-step checksums of the encoder window, proprioception, "
        "action, and the resulting joint state for this many steps."
    ),
)
parser.add_argument(
    "--save_rollout_training_samples",
    action="store_true",
    default=False,
    help=(
        "Write planner training rows to <output_dir>/rollout_training_samples. "
        "One row per environment per CONTROL STEP, not per publication: the "
        "released tracker re-encodes its token every 50 Hz step, so a planner "
        "that replaces the encoder has to supply one latent per step too."
    ),
)
parser.add_argument(
    "--sample_output_dir",
    type=Path,
    default=None,
    help="Collection root. Required with --save_rollout_training_samples.",
)
parser.add_argument("--sample_rows_per_file", type=int, default=8192)
parser.add_argument(
    "--sample_future_window_frames",
    type=int,
    default=0,
    help=(
        "Store this many frames of expert `root_qpos` lookahead per row. Zero "
        "omits it, which is right for a SONIC-latent target. Set 30 to make "
        "the collection re-encodable through OUR encoders."
    ),
)
parser.add_argument(
    "--sample_target",
    choices=("pre_quantization", "post_quantization"),
    default="pre_quantization",
    help=(
        "Which SONIC latent becomes `z_target`. 'pre_quantization' is the "
        "continuous encoder output before the FSQ lattice snap; the consumer "
        "snaps at publication, so the deployed value is identical while the "
        "regression target stays continuous. 'post_quantization' regresses the "
        "snapped lattice value directly."
    ),
)
parser.add_argument("--state_history_steps", type=int, default=9)
parser.add_argument(
    "--gr00t_checkpoint",
    type=Path,
    default=None,
    help=(
        "Latent GR00T action head that REPLACES the SONIC encoder. The tracker "
        "decoder is unchanged; only the source of its 64-D token changes, from "
        "the reference-driven encoder to the planner's causal prediction."
    ),
)
parser.add_argument("--gr00t_goal_features", type=Path, default=None)
parser.add_argument(
    "--gr00t_goals_per_env",
    type=str,
    nargs="+",
    default=None,
    help=(
        "One language goal per environment, in environment order. The rank "
        "assignment blocks environments by motion, so this list must be blocked "
        "the same way; the binding is asserted against the live motion names."
    ),
)
parser.add_argument(
    "--gr00t_publish_interval",
    type=int,
    default=10,
    help=(
        "Control steps between planner publications. The head predicts a "
        "30-latent horizon; only the first `interval` slots are consumed."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.save_rollout_training_samples and args_cli.sample_output_dir is None:
    parser.error("--save_rollout_training_samples requires --sample_output_dir")
if args_cli.video:
    # Rendering needs the camera pipeline; asking for a video without it
    # produces blank frames rather than an error.
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils.hydra import hydra_task_config

from imitation_experiments.data.planner_sample_schema import (
    PAIRED_TARGET_CONTRACT,
    PLANNER_SAMPLE_FORMAT,
    PLANNER_SAMPLE_VERSION,
    PlannerSampleWriter,
    build_planner_sample,
)
from imitation_experiments.lowlevel.sonic_release_actor import (
    ENCODER_FRAMES,
    heading_relative_rot6d_from_full_relative,
    load_sonic_release_actor,
    pack_encoder_window,
)


# The G1 29-DoF articulation order. SONIC's ``G1_ISAACLAB_JOINTS`` reduces to
# exactly this sequence, so no permutation is needed - but a silent reordering
# in either asset would invalidate every number, so it is asserted, not assumed.
EXPECTED_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# SONIC's PolicyCfg declares these fields in this order, and Isaac Lab
# concatenates a group by field declaration order - not by the order keys happen
# to appear in a YAML override. The training YAML lists gravity first; the class
# does not, and the class is what built the weights.
SONIC_PROPRIOCEPTION_ORDER: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
    ("last_action", 29),
    ("projected_gravity", 3),
)

# The training YAML's key order, kept only so the two readings can be compared.
SONIC_PROPRIOCEPTION_ORDER_YAML: tuple[tuple[str, int], ...] = (
    ("projected_gravity", 3),
    ("base_ang_vel", 3),
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
    ("last_action", 29),
)


def _proprioception_layout(name: str) -> tuple[tuple[str, int], ...]:
    return (
        SONIC_PROPRIOCEPTION_ORDER
        if name == "gravity_last"
        else SONIC_PROPRIOCEPTION_ORDER_YAML
    )


JOINT_QPOS_QVEL_WIDTH = 58
ANCHOR_ORI_WIDTH = 6
SONIC_SUCCESS_TERMINATION_THRESHOLDS: dict[str, dict[str, float]] = {
    "anchor_pos": {"threshold": 0.25, "down_threshold": 0.25},
    "anchor_ori": {"threshold": 1.0},
    "ee_body_pos": {"threshold": 0.25, "down_threshold": 0.25},
}
SONIC_SUCCESS_DISABLED_TERMINATIONS = ("foot_pos_xyz", "base_too_low")


def _configured_step_dt(env_cfg: object) -> float | None:
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 0.0) or 0.0)
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    if sim_dt > 0.0 and decimation > 0:
        return sim_dt * decimation
    return None


def _extend_episode_length_for_steps(env_cfg: Any, steps: int) -> dict[str, Any]:
    """Make the env timeout longer than the requested evaluation cap."""
    step_dt = _configured_step_dt(env_cfg)
    if step_dt is None or not hasattr(env_cfg, "episode_length_s"):
        return {
            "extended": False,
            "reason": "step_dt_or_episode_length_unavailable",
        }
    current = float(getattr(env_cfg, "episode_length_s"))
    required = float(int(steps) + 2) * step_dt
    if current >= required:
        return {
            "extended": False,
            "before_s": current,
            "after_s": current,
            "required_s": required,
        }
    env_cfg.episode_length_s = required
    print(
        "[INFO] Extended env.episode_length_s for release evaluation: "
        f"{current:.3f} -> {required:.3f}",
        flush=True,
    )
    return {
        "extended": True,
        "before_s": current,
        "after_s": required,
        "required_s": required,
    }


def _apply_sonic_success_terminations(env_cfg: Any) -> dict[str, Any]:
    """Bake SONIC's released success criterion into the env config."""
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        raise RuntimeError("env_cfg has no terminations group.")
    applied: dict[str, Any] = {"mode": "sonic", "thresholds": {}, "disabled": []}
    for term_name, params in SONIC_SUCCESS_TERMINATION_THRESHOLDS.items():
        term = getattr(terminations, term_name, None)
        if term is None:
            raise RuntimeError(
                f"SONIC release evaluation requires termination {term_name!r}."
            )
        term_params = getattr(term, "params", None)
        if term_params is None:
            raise RuntimeError(f"Termination {term_name!r} has no params.")
        for key, value in params.items():
            term_params[key] = float(value)
        applied["thresholds"][term_name] = dict(params)
    for term_name in SONIC_SUCCESS_DISABLED_TERMINATIONS:
        if hasattr(terminations, term_name):
            setattr(terminations, term_name, None)
        applied["disabled"].append(term_name)
    return applied


FALL_ONLY_DISABLED_TERMINATIONS = (
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
    "foot_pos_xyz",
)


def _apply_fall_only_terminations(env_cfg: Any) -> dict[str, Any]:
    """M3 planner contract: falling is the only failure.

    Every tracking-error termination is removed and `base_too_low` is required
    to be present, so `success` means "finished the reference without falling"
    and tracking error is reported as a continuous metric rather than deciding
    the episode. This is the contract the 30-motion planner arms are scored
    under; SONIC's own contract does the opposite (tracking thresholds on,
    `base_too_low` off), so the two are not comparable without this mode.
    """
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        raise RuntimeError("env_cfg has no terminations group.")
    disabled: list[str] = []
    for term_name in FALL_ONLY_DISABLED_TERMINATIONS:
        if getattr(terminations, term_name, None) is not None:
            setattr(terminations, term_name, None)
            disabled.append(term_name)
    if getattr(terminations, "base_too_low", None) is None:
        # The v2 task ships with `base_too_low = None` (a fall is not a
        # transient there), so the M3 contract restores the repository's
        # standard detector rather than leaving the run with no failure term.
        from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.terminations import (  # noqa: PLC0415,E501
            G1TerminationsCfg,
        )

        terminations.base_too_low = G1TerminationsCfg().base_too_low
    return {
        "mode": "fall_only",
        "disabled": disabled,
        "failure_terms": ["base_too_low"],
    }


def _describe_task_termination_contract(env_cfg: Any) -> dict[str, Any]:
    terminations = getattr(env_cfg, "terminations", None)
    active_terms = []
    if terminations is not None:
        active_terms = [
            name
            for name, value in vars(terminations).items()
            if not name.startswith("_") and value is not None
        ]
    return {"mode": "task", "active_terms": sorted(active_terms)}


def _with_tracking_mpjpe_alias(
    metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if "tracking_mpjpe_mm" not in metrics and "mpjpe_l_mm" in metrics:
        metrics = dict(metrics)
        metrics["tracking_mpjpe_mm"] = metrics["mpjpe_l_mm"]
    return metrics


def _require_reportable_release(summary: dict[str, Any]) -> None:
    aggregate = summary.get("aggregate", {})
    problems = []
    if summary.get("stop_reason") != "all_envs_done":
        problems.append(f"stop_reason={summary.get('stop_reason')!r}")
    if float(aggregate.get("done_rate", 0.0)) != 1.0:
        problems.append(f"done_rate={aggregate.get('done_rate')!r}")
    if float(aggregate.get("time_out_rate", 1.0)) != 0.0:
        problems.append(f"time_out_rate={aggregate.get('time_out_rate')!r}")
    if problems:
        raise RuntimeError(
            "Refusing to write a reportable SONIC release score from an "
            "incomplete run: "
            + ", ".join(problems)
            + ". Increase --steps or pass --allow_incomplete_release for a "
            "debug artifact."
        )


def _configure_sonic_contract(env_cfg: Any) -> dict[str, Any]:
    """Force the environment onto the released checkpoint's exact contract."""
    from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
        EncoderViewCfg,
    )

    from isaaclab_imitation.assets.robots import UNITREE_G1_29DOF_SONIC_ACTION_SCALE
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.presets import (
        G1SonicRobotCfg,
    )

    # SONIC's actuator contract: their hip pitch is the larger 7520-22 (139 Nm),
    # where our default recipe keeps 7520-14. The action scale is induced from
    # the actuators (0.25 * effort_limit / stiffness), so it moves with them.
    #
    # Take the actuators from the *preset*, not from the bare
    # ``UNITREE_G1_29DOF_SONIC_CFG``: that one spawns from URDF, and a
    # URDF-imported articulation reaches Newton with no actuator gains, so every
    # joint position target becomes a no-op and the robot merely falls. The
    # preset spawns the same robot from the preconverted USD, which carries the
    # gains, and it keeps the per-backend variants the task's physics override
    # selects between.
    robot_preset = G1SonicRobotCfg()
    existing = env_cfg.scene.robot
    prim_path = getattr(existing, "prim_path", None) or existing.default.prim_path
    for variant in (
        robot_preset.default,
        robot_preset.physx,
        robot_preset.newton_mjwarp,
    ):
        variant.prim_path = prim_path
    env_cfg.scene.robot = robot_preset
    env_cfg.actions.joint_pos.scale = UNITREE_G1_29DOF_SONIC_ACTION_SCALE

    interface = env_cfg.command_interface
    # 10 reference frames spaced 5 apart at 50 Hz: offsets [0, 5, ... 45],
    # i.e. SONIC's dt_future_ref_frames=0.1 over a 0.9 s lookahead.
    interface.encoder = EncoderViewCfg(
        components=("joint_qpos_qvel", "root_ori"),
        past_steps=0,
        future_steps=ENCODER_FRAMES - 1,
        frame_stride=5,
    )
    env_cfg.expert_macro_state_terms = ["expert_motion", "expert_anchor_ori_b"]
    # Deterministic evaluation: the released actor must read the true state, not
    # a noise-corrupted one. Our groups apply SONIC's own training noise
    # (gravity and anchor +/-0.05, joint velocity +/-0.5), which is correct for
    # training and wrong for scoring a frozen checkpoint.
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False
    # SONIC's decoder reads 10 past frames of each proprioception term. The task
    # default is single-frame, so the history is requested here rather than
    # assumed - a 3-wide base_ang_vel would otherwise silently misalign the
    # whole 930-wide vector.
    missing = []
    for term_name, _ in SONIC_PROPRIOCEPTION_ORDER:
        term = getattr(env_cfg.observations.policy, term_name, None)  # noqa: E501
        if term is None:
            missing.append(term_name)
            continue
        term.history_length = ENCODER_FRAMES
        term.flatten_history_dim = True
    if missing:
        raise RuntimeError(
            f"The policy group has no {missing} term(s); the released SONIC "
            "decoder cannot be fed without them."
        )
    return {
        "encoder_components": list(interface.encoder.components),
        "encoder_future_steps": int(interface.encoder.future_steps),
        "encoder_frame_stride": int(interface.encoder.frame_stride),
        "encoder_orientation_source": "expert_anchor_ori_b",
        "expert_macro_state_terms": list(env_cfg.expert_macro_state_terms),
        "observation_corruption_enabled": False,
        "robot_cfg": "UNITREE_G1_29DOF_SONIC_CFG",
        "action_scale": "UNITREE_G1_29DOF_SONIC_ACTION_SCALE",
    }


def _action_joint_names(base_env: Any) -> list[str]:
    """Joint order the action vector is indexed in.

    Our action term is configured with ``preserve_order=True`` over
    ``G1_29DOF_ISAACLAB_JOINT_NAMES``, which is the breadth-first order - the
    same order SONIC uses - while the articulation itself enumerates joints in
    Unitree's SDK order. The two differ, so the action order is read from the
    live term rather than assumed to match either one.
    """
    term = base_env.action_manager.get_term("joint_pos")
    for attribute in ("_joint_names", "joint_names"):
        names = getattr(term, attribute, None)
        if names:
            return [str(name) for name in names]
    raise RuntimeError(
        "Could not read the action term's joint order; refusing to guess which "
        "order the released decoder's 29 outputs should be written in."
    )


def _gather_indices(source: list[str], target: list[str], device: Any) -> torch.Tensor:
    """Indices such that ``x_target = x_source[..., result]``."""
    position = {name: index for index, name in enumerate(source)}
    return torch.tensor(
        [position[name] for name in target], dtype=torch.long, device=device
    )


class JointOrders:
    """The three joint orders in play, and the gathers between them.

    This environment uses two orders at once, which is the trap:

    * the **articulation** enumerates joints in Unitree's SDK serial order
      (left leg, right leg, waist, left arm, right arm), and every observation
      read off ``robot.data`` - ``joint_pos_rel``, ``joint_vel_rel``, and the
      reference ``expert_motion`` - is in that order;
    * the **action term** is configured ``preserve_order=True`` over
      ``G1_29DOF_ISAACLAB_JOINT_NAMES``, the breadth-first order, so the action
      vector and IsaacLab's ``last_action`` observation are in *that* order.

    SONIC's order is breadth-first, so it happens to coincide with our action
    order and to differ from our observation order. Both are read from the live
    environment rather than assumed.
    """

    def __init__(self, base_env: Any) -> None:
        device = base_env.device
        self.articulation = [str(name) for name in base_env.robot.data.joint_names]
        self.action = _action_joint_names(base_env)
        for label, names in (
            ("articulation", self.articulation),
            ("action term", self.action),
        ):
            if sorted(names) != sorted(EXPECTED_JOINT_ORDER):
                raise RuntimeError(
                    f"The {label} joint set differs from SONIC's, so no "
                    "permutation exists."
                )
        sonic = list(EXPECTED_JOINT_ORDER)
        # Observations (articulation order) -> SONIC order.
        self.obs_to_sonic = _gather_indices(self.articulation, sonic, device)
        # ``last_action`` is the raw action buffer, i.e. action-term order.
        self.action_to_sonic = _gather_indices(self.action, sonic, device)
        # The decoder's 29 outputs -> the order the action term expects.
        self.sonic_to_action = _gather_indices(sonic, self.action, device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "articulation": self.articulation,
            "action_term": self.action,
            "sonic": list(EXPECTED_JOINT_ORDER),
            "articulation_matches_sonic": self.articulation
            == list(EXPECTED_JOINT_ORDER),
            "action_term_matches_sonic": self.action == list(EXPECTED_JOINT_ORDER),
        }


def _assert_gravity_convention(base_env: Any, observed: torch.Tensor) -> float:
    """Our ``projected_gravity`` must equal SONIC's ``gravity_dir``.

    SONIC computes ``quat_apply(quat_inv(robot_anchor_quat_w), (0, 0, -1))``.
    For the G1 the pelvis anchor is the root link, so the two are the same
    quantity - but a differing sign or frame would be invisible in the metrics
    and catastrophic for the actions.
    """
    root_quat = base_env.robot.data.root_quat_w
    if not torch.is_tensor(root_quat):
        root_quat = torch.as_tensor(root_quat)
    down = torch.zeros_like(root_quat[:, :3])
    down[:, 2] = -1.0
    expected = math_utils.quat_apply(math_utils.quat_inv(root_quat), down)
    # The term carries a 10-step history, oldest first, so the live frame is the
    # last three values.
    error = float((observed[:, -3:] - expected).abs().max().item())
    if error > 1.0e-4:
        raise RuntimeError(
            "projected_gravity does not match SONIC's gravity_dir "
            f"(max abs diff {error:.3e}); the released decoder would read a "
            "different quantity than it was trained on."
        )
    return error


def _policy_terms(observations: Any) -> dict[str, torch.Tensor]:
    policy = observations["policy"]
    if not isinstance(policy, dict):
        raise TypeError(
            "This adapter needs the policy group as a term dict "
            "(concatenate_terms=False), got a flat tensor."
        )
    return policy


def _encoder_window(
    policy: dict[str, torch.Tensor],
    *,
    sonic_version: str,
    robot_anchor_quat_w: torch.Tensor | None = None,
) -> torch.Tensor:  # noqa: D401
    """Assemble the released encoder's 640-wide input from the strided view.

    ``expert_motion`` is **not** permuted here. The reference channel indexes the
    articulation through ``find_joints(target_joint_names, preserve_order=True)``,
    so it already emits the interleaved order that SONIC uses - unlike
    ``joint_pos_rel``/``joint_vel_rel``, which come straight off the Newton
    articulation buffer in the grouped SDK order. Permuting it a second time
    scatters the reference across the wrong joints, which reads as a policy that
    cannot track rather than as a wiring fault. ``_assert_reference_joint_order``
    checks this against the live reset state instead of trusting the reasoning.
    """
    motion = policy["expert_motion"]
    num_envs = motion.shape[0]
    motion = motion.reshape(num_envs, ENCODER_FRAMES, JOINT_QPOS_QVEL_WIDTH)
    anchor_ori = policy["expert_anchor_ori_b"].reshape(
        num_envs, ENCODER_FRAMES, ANCHOR_ORI_WIDTH
    )
    if sonic_version == "v1_1":
        if robot_anchor_quat_w is None:
            raise ValueError("v1_1 encoder window needs robot_anchor_quat_w.")
        anchor_ori = heading_relative_rot6d_from_full_relative(
            anchor_ori, robot_anchor_quat_w
        )
    elif sonic_version != "release":
        raise ValueError(f"Unsupported SONIC version: {sonic_version!r}.")
    return pack_encoder_window(
        motion[..., :29].contiguous(),
        motion[..., 29:].contiguous(),
        anchor_ori.contiguous(),
    )


def _assert_reference_joint_order(
    base_env: Any, policy: dict[str, torch.Tensor], orders: "JointOrders"
) -> dict[str, float]:
    """Decide empirically which joint order ``expert_motion`` is published in.

    The reset places the robot on its reference frame, so the first reference
    frame and the live joint state describe the same pose. Comparing both
    candidate alignments turns a silent 29-way scramble into a startup failure,
    and tolerates the reset noise that an absolute threshold could not.
    """
    reference = policy["expert_motion"].reshape(
        policy["expert_motion"].shape[0], ENCODER_FRAMES, JOINT_QPOS_QVEL_WIDTH
    )[:, 0, :29]
    live = base_env.robot.data.joint_pos
    if not torch.is_tensor(live):
        live = torch.as_tensor(live)
    live_sonic = live.index_select(-1, orders.obs_to_sonic)
    errors = {
        "reference_already_sonic_ordered": float(
            (reference - live_sonic).abs().mean().item()
        ),
        "reference_needs_permutation": float(
            (reference.index_select(-1, orders.obs_to_sonic) - live_sonic)
            .abs()
            .mean()
            .item()
        ),
    }
    if (
        errors["reference_already_sonic_ordered"]
        > errors["reference_needs_permutation"]
    ):
        raise RuntimeError(
            "expert_motion matches the live pose better after permutation "
            f"({errors}); the reference channel is not publishing SONIC's joint "
            "order and the encoder window would be scrambled."
        )
    return errors


def _assert_joint_observation_order(
    base_env: Any, policy: dict[str, torch.Tensor], orders: "JointOrders"
) -> dict[str, float]:
    """Check that ``joint_pos_rel`` is published in SONIC's joint order.

    Reading ``robot.data.joint_pos`` directly would give the grouped Newton SDK
    order, but this task's ``joint_pos_rel`` term is declared over
    ``G1_29DOF_ISAACLAB_JOINT_NAMES`` with ``preserve_order=True``, so the
    observation manager has already gathered it into the interleaved order SONIC
    uses. Permuting it again scatters the robot's own state across the wrong
    joints - the same 29-way scramble as on the reference side, and just as
    invisible in the summary. The live state settles it.
    """
    observed = policy["joint_pos_rel"].reshape(
        policy["joint_pos_rel"].shape[0], ENCODER_FRAMES, 29
    )[:, -1]
    live = base_env.robot.data.joint_pos - base_env.robot.data.default_joint_pos
    if not torch.is_tensor(live):
        live = torch.as_tensor(live)
    errors = {
        "joint_pos_rel_already_sonic_ordered": float(
            (observed - live.index_select(-1, orders.obs_to_sonic)).abs().mean().item()
        ),
        "joint_pos_rel_in_articulation_order": float(
            (observed - live).abs().mean().item()
        ),
    }
    if (
        errors["joint_pos_rel_already_sonic_ordered"]
        > errors["joint_pos_rel_in_articulation_order"]
    ):
        raise RuntimeError(
            "joint_pos_rel matches the raw articulation buffer better than the "
            f"canonical order ({errors}); the proprioception fed to the released "
            "decoder would be permuted."
        )
    return errors


def _proprioception(
    policy: dict[str, torch.Tensor],
    orders: "JointOrders",
    *,
    layout: tuple[tuple[str, int], ...],
    newest_first: bool,
) -> torch.Tensor:
    """Concatenate the five terms in SONIC's field-declaration order.

    No joint permutation is applied. Every 29-wide term this group publishes is
    already in the interleaved order SONIC uses: ``joint_pos_rel`` and
    ``joint_vel_rel`` are declared with ``preserve_order=True`` over the
    canonical joint list, and ``last_action`` is the raw action buffer, whose
    order is the action term's. Only a term read straight off ``robot.data``
    would arrive in the grouped Newton order.
    """
    del orders  # kept in the signature so callers pass the checked orders object
    parts: list[torch.Tensor] = []
    for term_name, width in layout:
        term = policy[term_name]
        expected = width * ENCODER_FRAMES
        if term.shape[-1] != expected:
            raise ValueError(
                f"{term_name} must carry a {ENCODER_FRAMES}-step history "
                f"({expected} values), got {term.shape[-1]}."
            )
        frames = term.reshape(term.shape[0], ENCODER_FRAMES, width)
        if newest_first:
            frames = frames.flip(1)
        parts.append(frames.reshape(term.shape[0], expected))
    return torch.cat(parts, dim=-1)


def _episode_metrics(base_env: Any) -> dict[str, torch.Tensor]:
    """Snapshot the reference channel's running metrics as ordinary tensors.

    The round trip through numpy is deliberate: a tensor cloned inside
    ``inference_mode`` stays an inference tensor, and the accumulators built
    from it could never be written to afterwards.
    """
    metrics = base_env.reference_command.metrics
    return _with_tracking_mpjpe_alias(
        {
            name: torch.from_numpy(value.detach().float().cpu().numpy().copy())
            for name, value in metrics.items()
        }
    )


@hydra_task_config(args_cli.task, "rlopt_ipmd_tuned_cfg_entry_point")
def main(env_cfg, agent_cfg):
    del agent_cfg  # The released SONIC actor replaces the RLOpt agent entirely.
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num_envs and --steps must be positive.")
    checkpoint = args_cli.sonic_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SONIC checkpoint not found: {checkpoint}")

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    contract = _configure_sonic_contract(env_cfg)
    if args_cli.disable_early_terminations:
        termination_contract = {"mode": "disabled"}
    elif args_cli.termination_contract == "sonic":
        termination_contract = _apply_sonic_success_terminations(env_cfg)
    elif args_cli.termination_contract == "fall_only":
        termination_contract = _apply_fall_only_terminations(env_cfg)
    else:
        termination_contract = _describe_task_termination_contract(env_cfg)
        print(
            "[WARNING] --termination_contract task is diagnostic only; do not "
            "report its JSON as a SONIC success score.",
            flush=True,
        )
    episode_length_contract = (
        {
            "extended": False,
            "reason": "preserve_episode_length",
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
        }
        if args_cli.preserve_episode_length
        else _extend_episode_length_for_steps(env_cfg, int(args_cli.steps))
    )
    randomization_kept = apply_randomization_profile(env_cfg, args_cli.randomization)
    reference_surface = pin_reference_start(
        env_cfg, start_frame=int(args_cli.reference_start_frame)
    )
    if args_cli.trajectory_ranks is not None:
        from imitation_experiments.lowlevel.motion_candidate_screen import (
            build_env_rank_assignment,
        )

        env_rank_assignment = build_env_rank_assignment(
            args_cli.trajectory_ranks, int(args_cli.num_envs)
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

        selection = env_cfg.command_interface.reference.selection
        selection.schedule = "custom"
        selection.custom_fn = _fixed_trajectory_ranks
    elif args_cli.reset_schedule is not None:
        # ``pin_reference_start`` pins the schedule to round_robin. Comparing
        # against an ``evaluate_checkpoint`` run means scoring the same motions
        # in the same order, so the schedule has to be settable, and the ranks
        # recorded so the two runs can be hash-matched rather than assumed equal.
        env_cfg.command_interface.reference.selection.schedule = str(
            args_cli.reset_schedule
        )
    if args_cli.disable_early_terminations:
        for name in list(vars(env_cfg.terminations)):
            if name not in ("time_out", "reference_finished"):
                setattr(env_cfg.terminations, name, None)

    video_dir: Path | None = None
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if args_cli.video:
        video_dir = (
            (
                args_cli.video_dir
                or Path("logs/sonic_release_eval/videos") / str(args_cli.label)
            )
            .expanduser()
            .resolve()
        )
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=int(args_cli.video_length),
            name_prefix=str(args_cli.label),
            disable_logger=True,
        )
    base_env = env.unwrapped
    device = base_env.device

    actor = load_sonic_release_actor(checkpoint, version=args_cli.sonic_version).to(
        device
    )
    actor.eval()
    sonic_version = actor.spec.version
    contract["sonic_version"] = sonic_version
    contract["checkpoint_orientation_contract"] = actor.spec.orientation_contract
    if sonic_version == "v1_1":
        contract["encoder_orientation_source"] = (
            "expert_anchor_ori_b plus current robot root heading"
        )

    orders = JointOrders(base_env)
    layout = _proprioception_layout(args_cli.proprioception_order)

    # A URDF-imported articulation reaches Newton with no actuator gains, so
    # every joint position target silently becomes a no-op: the robot falls, the
    # metrics look like a bad policy, and nothing in the summary says the actions
    # never landed. ``robot.data.joint_stiffness`` does not catch it - that
    # buffer echoes the config whether or not the simulator honours it - so the
    # spawn source itself is the check.
    spawn_cfg = type(base_env.robot.cfg.spawn).__name__
    if spawn_cfg != "UsdFileCfg":
        raise RuntimeError(
            f"The robot spawns from {spawn_cfg}; the released SONIC tracker must "
            "run on the preconverted USD asset, because a URDF import carries no "
            "actuator gains into Newton and every joint target would be ignored."
        )
    joint_stiffness = base_env.robot.data.joint_stiffness
    joint_damping = base_env.robot.data.joint_damping
    actuator_gains = {
        "robot_spawn_cfg": spawn_cfg,
        "joint_stiffness_min": float(joint_stiffness.min().item()),
        "joint_stiffness_max": float(joint_stiffness.max().item()),
        "joint_damping_min": float(joint_damping.min().item()),
        "joint_damping_max": float(joint_damping.max().item()),
    }
    print(f"[DIAG] simulated actuator gains: {actuator_gains}", flush=True)

    observations, _ = env.reset()
    trajectory_ranks: list[int] = []
    env_traj_rank = getattr(base_env.trajectory_manager, "env_traj_rank", None)
    if torch.is_tensor(env_traj_rank) and env_traj_rank.numel() >= base_env.num_envs:
        trajectory_ranks = (
            env_traj_rank[: base_env.num_envs].detach().cpu().long().tolist()
        )
    policy = _policy_terms(observations)
    gravity_error = _assert_gravity_convention(base_env, policy["projected_gravity"])
    reference_order_errors = _assert_reference_joint_order(base_env, policy, orders)
    joint_order_errors = _assert_joint_observation_order(base_env, policy, orders)
    print(
        f"[DIAG] joint-order checks: {reference_order_errors} {joint_order_errors}",
        flush=True,
    )

    sample_writer: PlannerSampleWriter | None = None
    sample_metadata: dict[str, Any] = {}
    sample_motion_names: list[str] = []
    state_history_steps = int(args_cli.state_history_steps)
    sample_future_frames = int(args_cli.sample_future_window_frames)
    episode_ids = torch.zeros(base_env.num_envs, dtype=torch.long)
    if bool(args_cli.save_rollout_training_samples):
        collection_root = Path(args_cli.sample_output_dir)
        if collection_root.exists():
            raise FileExistsError(f"Refusing to overwrite {collection_root}.")
        sample_writer = PlannerSampleWriter(
            collection_root / "rollout_training_samples",
            rows_per_file=int(args_cli.sample_rows_per_file),
        )
        # Rank order is pinned for the whole run by `--trajectory_ranks`, so the
        # per-environment motion name is a constant. It is still read from the
        # live manager rather than assumed, because without pinned ranks the
        # sequential schedule reassigns a motion on every reset.
        rank_names = [str(name) for name in base_env.expert_trajectory_motion_names()]
        sample_motion_names = [
            rank_names[rank] if 0 <= rank < len(rank_names) else str(rank)
            for rank in trajectory_ranks
        ]
        sample_metadata = {
            "sample_format": {
                "name": PLANNER_SAMPLE_FORMAT,
                "version": PLANNER_SAMPLE_VERSION,
            },
            "paired_target_contract": PAIRED_TARGET_CONTRACT,
            "planner_observation_spec": base_env.causal_planner_observation_spec(
                history_steps=state_history_steps
            ),
            "state_history_steps": state_history_steps,
            # One row per control step, so a downstream `hold_steps: 1` join
            # yields consecutive 50 Hz latents rather than 5 Hz publications.
            "collection_unit": "control_step_row",
            "planner_rate_hz": 50.0,
            "planner_state_source": (
                "causal_robot_history_during_sonic_release_oracle_rollout"
            ),
            "target_encoding": {"kind": "continuous"},
            "latent_interface": {
                "producer": "sonic_release_g1_encoder",
                "sonic_version": str(sonic_version),
                "token_dim": int(actor.spec.token_dim),
                "target": str(args_cli.sample_target),
                "fsq_levels": 32,
                "max_num_tokens": 2,
                "hold_steps": 1,
                "encoder_frames": int(actor.spec.encoder_frames),
                "encoder_frame_stride": 5,
            },
            "sonic_checkpoint": str(checkpoint),
            "sonic_checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "reference_start_frame": int(args_cli.reference_start_frame),
            "randomization": str(args_cli.randomization),
            "seed": int(args_cli.seed),
        }

    planner: Any | None = None
    planner_record: dict[str, Any] = {}
    if args_cli.gr00t_checkpoint is not None:
        if args_cli.gr00t_goal_features is None or args_cli.gr00t_goals_per_env is None:
            raise ValueError(
                "--gr00t_checkpoint needs --gr00t_goal_features and "
                "--gr00t_goals_per_env."
            )
        from imitation_experiments.planner.gr00t_isaac_sampler import (  # noqa: PLC0415
            Gr00tSkillCommandSampler,
        )

        class _StandalonePlanner(Gr00tSkillCommandSampler):
            """The mixin's `gr00t_*` methods need no sampler base state."""

        planner = _StandalonePlanner()
        planner_record = planner.configure_gr00t(
            checkpoint_path=args_cli.gr00t_checkpoint,
            goal_features_path=args_cli.gr00t_goal_features,
            goal_name=list(args_cli.gr00t_goals_per_env),
            num_envs=int(base_env.num_envs),
            consumption="open_loop",
            # SONIC's lattice: 32 levels per coordinate, half width 16. The head
            # regresses the continuous pre-quantization value, so the snap here
            # is what makes the published token identical to one the encoder
            # could have produced.
            fsq_half_levels=torch.tensor(16.0),
            device=device,
            expected_target_mode="latent",
            consume_slots=int(args_cli.gr00t_publish_interval),
        )
        rank_names = [str(name) for name in base_env.expert_trajectory_motion_names()]
        planner.gr00t_assert_goal_matches(
            torch.arange(base_env.num_envs, device=device),
            [
                rank_names[rank] if 0 <= rank < len(rank_names) else str(rank)
                for rank in trajectory_ranks
            ],
        )
        print(f"[PLANNER] {planner_record}", flush=True)

    survived = torch.zeros(base_env.num_envs, dtype=torch.long, device=device)
    done_once = torch.zeros(base_env.num_envs, dtype=torch.bool, device=device)
    termination_causes: dict[str, int] = {}
    # Per environment, not per term: several terms can fire on the same step,
    # so summing term counts would over-count failures (and push the success
    # rate below zero).
    failed = torch.zeros(base_env.num_envs, dtype=torch.bool, device=device)
    # SONIC scores a motion as successful only when it reaches the end of its
    # reference without a failure term. Surviving a step cap is not success.
    finished = torch.zeros(base_env.num_envs, dtype=torch.bool, device=device)
    # Allocated before the rollout so the accumulators are ordinary tensors.
    final_metrics: dict[str, torch.Tensor] = {
        name: torch.zeros_like(value)
        for name, value in _episode_metrics(base_env).items()
    }
    final_metrics = _with_tracking_mpjpe_alias(final_metrics)
    # The last reading taken while an environment was still running its episode.
    carried_metrics: dict[str, torch.Tensor] = {
        name: value.clone() for name, value in _episode_metrics(base_env).items()
    }
    carried_metrics = _with_tracking_mpjpe_alias(carried_metrics)
    steps_run = 0
    _previous_slot0: torch.Tensor | None = None

    # ``no_grad`` rather than ``inference_mode``: the metric accumulators are
    # written across the loop boundary, which inference tensors forbid.
    with torch.no_grad():
        for _ in range(int(args_cli.steps)):
            policy = _policy_terms(observations)
            robot_anchor_quat_w = base_env.robot.data.root_quat_w
            if not torch.is_tensor(robot_anchor_quat_w):
                robot_anchor_quat_w = torch.as_tensor(
                    robot_anchor_quat_w, device=device
                )
            window = _encoder_window(
                policy,
                sonic_version=sonic_version,
                robot_anchor_quat_w=robot_anchor_quat_w,
            )
            proprioception = _proprioception(
                policy,
                orders,
                layout=layout,
                newest_first=args_cli.history_order == "newest_first",
            )
            # The decoder emits joint targets in SONIC's order; the action term
            # indexes them in its own.
            if planner is None:
                token = actor.encode(window)
            else:
                # Same decoder, same proprioception: the ONLY difference from
                # the oracle row above is where the 64-D token comes from, so a
                # score gap is attributable to the planner and nothing else.
                token = planner.gr00t_z(
                    base_env.current_causal_planner_observation(
                        history_steps=state_history_steps
                    )
                    .get(("planner", "state_history"))
                    .flatten(1),
                    torch.arange(base_env.num_envs, device=device),
                )
            action = actor.decode(token, proprioception).index_select(
                -1, orders.sonic_to_action
            )
            action = action.clamp(
                -float(args_cli.action_clip), float(args_cli.action_clip)
            )
            if args_cli.action_source == "zeros":
                action = torch.zeros_like(action)

            if steps_run < int(args_cli.diagnose_steps):
                # Window cadence: at stride 5 the gap between adjacent window
                # slots is five reference frames, so it must be about five times
                # the gap one control step opens between successive slot 0s.
                motion_window = policy["expert_motion"].reshape(
                    base_env.num_envs, ENCODER_FRAMES, JOINT_QPOS_QVEL_WIDTH
                )[..., :29]
                slot_gap = float(
                    (motion_window[:, 1] - motion_window[:, 0]).abs().mean().item()
                )
                step_gap = (
                    float((motion_window[:, 0] - _previous_slot0).abs().mean().item())
                    if _previous_slot0 is not None
                    else float("nan")
                )
                _previous_slot0 = motion_window[:, 0].clone()
                print(
                    f"[DIAG] window cadence step={steps_run} "
                    f"slot0_to_slot1={slot_gap:.6f} "
                    f"one_control_step={step_gap:.6f}",
                    flush=True,
                )
                term = base_env.action_manager.get_term("joint_pos")
                processed = getattr(term, "processed_actions", None)
                target = base_env.robot.data.joint_pos_target
                print(
                    f"[DIAG] step={steps_run} "
                    f"window={float(window.double().sum()):.10f} "
                    f"proprioception={float(proprioception.double().sum()):.10f} "
                    f"action={float(action.double().sum()):.10f} "
                    f"processed={float(processed.double().sum()) if processed is not None else float('nan'):.10f} "
                    f"target={float(torch.as_tensor(target).double().sum()):.10f} "
                    f"joint_pos={float(base_env.robot.data.joint_pos.double().sum()):.10f}",
                    flush=True,
                )

            if sample_writer is not None:
                # Written from the PRE-step state, so the row's history and its
                # latent describe the same instant the tracker acted on.
                if args_cli.sample_target == "pre_quantization":
                    # The BOUNDED, lattice-scaled value, not the raw encoder
                    # output: `FSQ.snap` expects an already-normalized code, so
                    # this is the continuous quantity whose only difference
                    # from the deployed token is the rounding. The raw output
                    # is unbounded and saturates through `tanh`, which would
                    # put the regression target on a different scale.
                    quantizer = actor.quantizer
                    latent = (
                        quantizer.bound(actor.encode_pre_quantization(window))
                        / quantizer.half_width
                    )
                else:
                    latent = actor.encode(window)
                latent = latent.to(dtype=torch.float32)
                causal_history = base_env.current_causal_planner_observation(
                    history_steps=state_history_steps
                ).get(("planner", "state_history"))
                # `episode_length_buf` is read before `step`, so it is the
                # episode-local index of the state just observed and advances by
                # exactly one per row. `_join_slots` needs that to find slot k at
                # `control_step + k`, and needs the key to stay unique: the reset
                # inside `step` returns the counter to 0 while `episode_id`
                # increments below, so the pair never repeats.
                control_step = base_env.episode_length_buf.detach().cpu().reshape(-1)
                row = build_planner_sample(
                    causal_state_history=causal_history,
                    demonstration_state_history=causal_history,
                    causal_target=latent,
                    demonstration_target=latent,
                    trajectory_rank=torch.as_tensor(trajectory_ranks),
                    # Cloned, not passed by reference: the writer buffers rows
                    # until a flush, and the in-place `episode_ids += done`
                    # below would otherwise rewrite the episode number of every
                    # row still in the buffer.
                    episode_id=episode_ids.clone(),
                    env_id=torch.arange(base_env.num_envs),
                    control_step=control_step,
                    planner_step=control_step,
                    motion_names=sample_motion_names,
                    metadata=sample_metadata,
                )
                row["z_target"] = row["causal_target"]
                row["latent_skill_target"] = row["causal_target"]
                row["oracle_rollout_state_history"] = row["planner_state"]
                if sample_future_frames > 0:
                    # The 30-frame expert `root_qpos` lookahead, so OUR
                    # encoders can be applied to a SONIC-driven rollout
                    # offline. Without it the collection can only ever train a
                    # SONIC-latent head.
                    expert = base_env.current_expert_macro_transition_batch(
                        horizon_steps=sample_future_frames
                    )
                    expert_state = expert.get(("hl", "state")).float()
                    if int(expert_state.shape[-1]) != 38:
                        raise ValueError(
                            "--sample_future_window_frames needs a 38-D "
                            f"root_qpos macro state, got {int(expert_state.shape[-1])}."
                        )
                    window = expert.get(("hl", "future_window")).float()
                    row["expert_root_qpos_future"] = (
                        torch.cat(
                            [
                                expert_state.unsqueeze(1),
                                window[:, : sample_future_frames - 1],
                            ],
                            dim=1,
                        )
                        .detach()
                        .cpu()
                        .contiguous()
                    )
                    offsets = torch.arange(
                        sample_future_frames, device=device, dtype=torch.long
                    ).unsqueeze(0)
                    local = expert.get(("hl", "local_step")).long().reshape(-1, 1)
                    length = (
                        expert.get(("hl", "trajectory_length")).long().reshape(-1, 1)
                    )
                    row["expert_root_qpos_future_valid"] = (
                        (local + offsets < length).detach().cpu().contiguous()
                    )
                sample_writer.add(row)

            observations, _, terminated, truncated, _ = env.step(action)
            steps_run += 1
            done = terminated | truncated
            episode_ids += done.detach().cpu().long()
            if planner is not None and bool(done.any()):
                # A reset environment must re-plan from its own fresh state
                # rather than keep consuming slots predicted for the pose it
                # had before the reset.
                planner.gr00t_reset(done.nonzero(as_tuple=False).reshape(-1))
            survived += (~done_once).long()

            newly_done = done & ~done_once
            # Isaac Lab resets a finished environment *inside* ``step``, and the
            # command manager then recomputes its metrics on the fresh post-reset
            # state - where the robot sits on its new reference and the tracking
            # error is the reset placement noise, not what the episode achieved.
            # So a finishing environment must commit the value carried over from
            # the previous step, and only a still-running one may take the fresh
            # reading. Reading the live metrics here instead silently reports the
            # reset noise as MPJPE for every episode that ends.
            snapshot = _episode_metrics(base_env)
            if bool(newly_done.any()):
                mask = newly_done.cpu()
                for name, value in carried_metrics.items():
                    final_metrics[name][mask] = value[mask]
                for term_name in getattr(
                    base_env.termination_manager, "active_terms", []
                ):
                    fired = base_env.termination_manager.get_term(term_name)
                    hit_mask = fired & newly_done
                    hit = int(hit_mask.sum().item())
                    if hit:
                        termination_causes[term_name] = (
                            termination_causes.get(term_name, 0) + hit
                        )
                    if term_name == "reference_finished":
                        finished |= hit_mask
                    elif term_name != "time_out":
                        failed |= hit_mask
            still_running = (~done).cpu()
            for name, value in snapshot.items():
                carried_metrics[name][still_running] = value[still_running]
            done_once |= done
            if bool(done_once.all()):
                break

    collection: dict[str, Any] = {}
    if sample_writer is not None:
        sample_writer.flush()
        collection = {
            "root": str(Path(args_cli.sample_output_dir)),
            "rows": int(sample_writer.row_count),
            "files": int(sample_writer.file_count),
            "rows_per_file": int(args_cli.sample_rows_per_file),
            "target": str(args_cli.sample_target),
            "collection_unit": "control_step_row",
        }
        print(f"[COLLECT] {collection}", flush=True)

    unfinished = (~done_once).cpu()
    for name, value in carried_metrics.items():
        final_metrics[name][unfinished] = value[unfinished]

    failures = int(failed.sum().item())
    success_rate = 1.0 - failures / float(base_env.num_envs)

    # SONIC's released callback micro-averages the tracking metrics over the
    # successful trajectories: it sums each frame's value and divides by the
    # total frame count, so a long motion weighs more than a short one. Our
    # per-environment metric is already an episode mean over ``survived`` steps,
    # so re-weighting by that step count reproduces the same quantity.
    completed = (finished & ~failed).cpu()

    # Per-environment rows so a comparison can be restricted to the motions two
    # controllers both completed. Success-only means across different success
    # rates are not comparable: the controller that fails the hard motions is
    # scored on an easier subset, which flatters it.
    failed_cpu = failed.cpu()
    finished_cpu = finished.cpu()
    survived_cpu = survived.cpu()
    per_environment = [
        {
            "env_id": env_id,
            "trajectory_rank": trajectory_ranks[env_id]
            if env_id < len(trajectory_ranks)
            else None,
            "completed_tracking_success": bool(
                finished_cpu[env_id] and not failed_cpu[env_id]
            ),
            "failed": bool(failed_cpu[env_id]),
            "reference_finished": bool(finished_cpu[env_id]),
            "survival_steps": int(survived_cpu[env_id]),
            "metrics": {
                name: float(value[env_id]) for name, value in final_metrics.items()
            },
        }
        for env_id in range(base_env.num_envs)
    ]
    completed_count = int(completed.sum().item())
    frames = survived.float().cpu()
    completed_frames = float((frames * completed.float()).sum().item())
    successful_metrics = {
        name: {
            "mean": (
                float((value * frames * completed.float()).sum().item())
                / completed_frames
            )
            if completed_frames > 0.0
            else None,
            "count": int(completed_frames),
            "micro_mean": (
                float((value * frames * completed.float()).sum().item())
                / completed_frames
            )
            if completed_frames > 0.0
            else None,
            "macro_mean": float(value[completed].mean().item())
            if completed_count
            else None,
            "num_successful_envs": completed_count,
        }
        for name, value in sorted(final_metrics.items())
    }
    time_out_count = int(termination_causes.get("time_out", 0))
    stop_reason = "all_envs_done" if bool(done_once.all()) else "max_steps"

    summary: dict[str, Any] = {
        "schema": "sonic_release_native_evaluation_v1",
        "label": str(args_cli.label),
        "task": str(args_cli.task),
        "checkpoint": str(checkpoint),
        "sonic_version": sonic_version,
        "sonic_version_requested": str(args_cli.sonic_version),
        "num_envs": int(base_env.num_envs),
        "steps_requested": int(args_cli.steps),
        "steps_run": int(steps_run),
        "seed": int(args_cli.seed),
        "randomization_profile": str(args_cli.randomization),
        "randomization_kept": randomization_kept,
        "collection": collection,
        "planner": (
            {**planner_record, **planner.gr00t_stats()} if planner is not None else {}
        ),
        "metadata": {
            "label": str(args_cli.label),
            "task": str(args_cli.task),
            "checkpoint": str(checkpoint),
            "sonic_version": sonic_version,
            "sonic_version_requested": str(args_cli.sonic_version),
            "num_envs": int(base_env.num_envs),
            "seed": int(args_cli.seed),
            "action_sampling": "mode",
            "randomization_profile": str(args_cli.randomization),
            "randomization_kept": randomization_kept,
            "push_perturbation": {
                "enabled": bool(randomization_kept.get("push", True))
            },
            "reference_start_frame": int(args_cli.reference_start_frame),
            "reset_schedule": args_cli.reset_schedule,
            "early_terminations_enabled": not bool(args_cli.disable_early_terminations),
            "termination_contract": termination_contract,
            "episode_length_contract": episode_length_contract,
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
        },
        "reference_selection_surface": reference_surface,
        "reference_start_frame": int(args_cli.reference_start_frame),
        "reset_schedule": args_cli.reset_schedule,
        # Matches `jq -c '[.per_environment[].trajectory_rank]' | sha256sum`,
        # the repo's recorded comparison key: compact separators, trailing
        # newline. A differently formatted hash cannot be compared with the
        # existing evaluation records.
        "trajectory_ranks_sha256": hashlib.sha256(
            (json.dumps(trajectory_ranks, separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest()
        if trajectory_ranks
        else None,
        "trajectory_ranks": trajectory_ranks,
        "early_terminations_enabled": not bool(args_cli.disable_early_terminations),
        "termination_contract": termination_contract,
        "episode_length_contract": episode_length_contract,
        "episode_length_s": float(getattr(env_cfg, "episode_length_s", -1.0)),
        "action_clip_value": float(args_cli.action_clip),
        "action_source": str(args_cli.action_source),
        "actor_spec": actor.spec.to_dict(),
        "sonic_contract": contract,
        "proprioception_order": [name for name, _ in layout],
        "proprioception_order_choice": str(args_cli.proprioception_order),
        "history_order": str(args_cli.history_order),
        "gravity_convention_max_abs_diff": gravity_error,
        "reference_joint_order_check": reference_order_errors,
        "proprioception_joint_order_check": joint_order_errors,
        "joint_orders": orders.to_dict(),
        "actuator_gains": actuator_gains,
        "aggregate": {
            # No failure term fired. Includes environments still running at the
            # step cap, so it is an upper bound, not SONIC's success rate.
            "tracking_success_rate": success_rate,
            # SONIC's definition: the motion ran to its end without failing.
            "completed_tracking_success_rate": completed_count
            / float(base_env.num_envs),
            "completed_env_count": completed_count,
            "failed_env_count": failures,
            "num_evaluated_envs": int(base_env.num_envs),
            "done_rate": float(done_once.float().mean().item()),
            "time_out_rate": time_out_count / float(base_env.num_envs),
            "termination_cause_env_counts": termination_causes,
            "survival_steps_mean": float(survived.float().mean().item()),
        },
        "successful_metrics": successful_metrics,
        "max_steps": int(args_cli.steps),
        "stop_reason": stop_reason,
        "per_environment": per_environment,
        "metrics": {
            name: {
                "mean": float(value.mean().item()),
                "std": float(value.std(unbiased=False).item()),
                "count": int(value.numel()),
            }
            for name, value in sorted(final_metrics.items())
        },
        "deviations_from_sonic_env": [
            "Our reward and reset protocol, not SONIC's.",
            "Our BONES-SEED reference arrays and retargeting pipeline.",
            "Newton/MJWarp physics backend rather than SONIC's PhysX setup.",
        ],
    }

    env.close()
    if (
        not args_cli.allow_incomplete_release
        and not args_cli.disable_early_terminations
        and args_cli.termination_contract == "sonic"
    ):
        _require_reportable_release(summary)

    print(json.dumps(summary, indent=2))
    if args_cli.output_json is not None:
        output = args_cli.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[PASS] Wrote {output}")
    if video_dir is not None:
        for clip in sorted(video_dir.glob("*.mp4")):
            print(f"[VIDEO] {clip}")


if __name__ == "__main__":
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
