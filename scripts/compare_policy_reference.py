#!/usr/bin/env python3
# ruff: noqa: E402

"""Side-by-side reference/policy playback for an RLOpt checkpoint."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Compare a policy-controlled robot against expert reference replay."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during play."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help=(
        "Optional rollout/video step limit. By default the run continues until "
        "the selected reference trajectory ends."
    ),
)
parser.add_argument(
    "--video_seconds",
    type=float,
    default=None,
    help="Optional rollout/video duration in seconds; converted to env steps after env creation.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
    help="RLOpt algorithm (must match the checkpoint).",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional exact output directory for this comparison run.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment."
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--enable_wandb",
    action="store_true",
    default=False,
    help="Enable RLOpt W&B logging during comparison eval. Disabled by default.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional rollout step limit. Default is to stop when the reference ends.",
)
parser.add_argument(
    "--keep_terminations",
    action="store_true",
    default=False,
    help=(
        "Keep env termination terms enabled. By default comparison playback "
        "disables them so the reference clip is not interrupted by RL resets."
    ),
)
parser.add_argument(
    "--keep_rewards",
    action="store_true",
    default=False,
    help="Keep env reward terms enabled. By default comparison playback disables them.",
)
parser.add_argument(
    "--policy_trajectory_rank",
    type=int,
    default=None,
    help="Trajectory rank used by the policy env and therefore the language planner.",
)
parser.add_argument(
    "--policy_motion",
    type=str,
    default=None,
    help="Motion name used by the policy env and language planner, e.g. dance1_subject1.",
)
parser.add_argument(
    "--policy_dataset",
    type=str,
    default=None,
    help="Optional dataset filter when resolving --policy_motion.",
)
parser.add_argument(
    "--policy_trajectory",
    type=str,
    default=None,
    help="Optional trajectory-name filter when resolving --policy_motion.",
)
parser.add_argument(
    "--policy_start_step",
    type=int,
    default=0,
    help="Local trajectory step used when resetting the policy/reference envs.",
)
parser.add_argument(
    "--list_trajectories",
    action="store_true",
    default=False,
    help="Print rank, dataset, motion, and trajectory names, then exit.",
)
parser.add_argument(
    "--reference_visualization",
    type=str,
    default="body_markers",
    choices=["body_markers", "robot", "both"],
    help=(
        "How to visualize the expert reference. body_markers draws the body "
        "state tensors used by training; robot uses the qpos articulation replay."
    ),
)
parser.add_argument(
    "--metrics_json",
    type=str,
    default=None,
    help=(
        "Write per-step policy tracking metrics (root height, joint MAE, "
        "end-effector error) plus a summary to this JSON path. Makes two "
        "playback runs quantitatively comparable instead of visually."
    ),
)
parser.add_argument(
    "--fall_height",
    type=float,
    default=0.4,
    help="Pelvis height (m) below which the policy env counts as fallen.",
)
parser.add_argument(
    "--emulate_joint_order_from",
    type=str,
    default=None,
    help=(
        "DIAGNOSTIC ONLY. Path to a scripts/dump_backend_index_contract.py JSON "
        "whose articulation joint order should be emulated. Permutes the "
        "backend-order-dependent expert command terms and the action offset so a "
        "checkpoint trained under that backend sees the joint ordering it was "
        "actually trained on. Use only to attribute a cross-backend failure; "
        "never for a paper number."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.dict import print_dict
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_EE_BODY_NAMES,
)
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

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

REFERENCE_ENV_ID = 0
POLICY_ENV_ID = 1
REFERENCE_MARKER_COLOR = (0.0, 0.75, 1.0)
POLICY_MARKER_COLOR = (1.0, 0.1, 0.0)
REFERENCE_BODY_MARKER_RADIUS = 0.06
MARKER_HEIGHT_OFFSET = 1.35


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


def _unwrap_imitation_env(env) -> ImitationRLEnv:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ImitationRLEnv):
            return current
        current_unwrapped = getattr(current, "unwrapped", None)
        if isinstance(current_unwrapped, ImitationRLEnv):
            return current_unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError("Could not unwrap an ImitationRLEnv from the provided environment.")


def _create_role_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/reference_policy_role_markers",
        markers={
            "reference": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=REFERENCE_MARKER_COLOR
                ),
            ),
            "policy": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=POLICY_MARKER_COLOR
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _update_role_markers(
    base_env: ImitationRLEnv,
    role_markers: VisualizationMarkers,
    *,
    reference_root_pos_w: torch.Tensor | None = None,
) -> None:
    root_pos = base_env.robot.data.root_pos_w.torch[
        [REFERENCE_ENV_ID, POLICY_ENV_ID]
    ].clone()
    if reference_root_pos_w is not None:
        root_pos[0] = reference_root_pos_w.to(device=base_env.device).reshape(3)
    root_pos[:, 2] += MARKER_HEIGHT_OFFSET
    marker_indices = torch.tensor([0, 1], dtype=torch.long, device=base_env.device)
    role_markers.visualize(translations=root_pos, marker_indices=marker_indices)


def _create_reference_body_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/reference_body_state_markers",
        markers={
            "reference_body": sim_utils.SphereCfg(
                radius=REFERENCE_BODY_MARKER_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=REFERENCE_MARKER_COLOR
                ),
            ),
        },
    )
    marker = VisualizationMarkers(marker_cfg)
    marker.set_visibility(True)
    return marker


def _reference_body_pose_keys(reference) -> tuple[str, str | None]:
    # Mirror ImitationRLEnv._initialize_mdp_fast_paths/_expert_body_pose_fields.
    pos_key = "xpos" if "xpos" in reference else "body_pos_w"
    quat_key = "xquat" if "xquat" in reference else "body_quat_w"
    if reference.get(pos_key) is None:
        raise KeyError(
            "Expert frame is missing body position tensors: expected xpos or body_pos_w."
        )
    if reference.get(quat_key) is None:
        quat_key = None
    return pos_key, quat_key


def _reference_body_positions_w(
    base_env: ImitationRLEnv,
    *,
    source_env_id: int,
    target_env_id: int,
) -> tuple[torch.Tensor, str]:
    reference = base_env.current_expert_frame
    pos_key, quat_key = _reference_body_pose_keys(reference)
    ref_pos = reference[pos_key][source_env_id : source_env_id + 1]
    ref_quat = (
        reference[quat_key][source_env_id : source_env_id + 1]
        if quat_key is not None
        else None
    )
    target_env_ids = torch.tensor(
        [target_env_id], dtype=torch.long, device=base_env.device
    )
    pos_w, _ = base_env._transform_reference_body_pose_to_init_alignment(
        ref_pos, ref_quat, env_ids=target_env_ids
    )
    return pos_w.squeeze(0), pos_key


def _update_reference_body_markers(
    base_env: ImitationRLEnv, reference_body_markers: VisualizationMarkers
) -> tuple[torch.Tensor | None, str, int, int]:
    positions_w, pos_key = _reference_body_positions_w(
        base_env, source_env_id=POLICY_ENV_ID, target_env_id=REFERENCE_ENV_ID
    )
    finite_mask = torch.isfinite(positions_w).all(dim=-1)
    num_total = int(positions_w.shape[0])
    num_rendered = int(finite_mask.sum().item())
    if num_rendered == 0:
        reference_body_markers.set_visibility(False)
        return None, pos_key, 0, num_total

    visible_positions = positions_w[finite_mask].contiguous()
    reference_body_markers.set_visibility(True)
    reference_body_markers.visualize(translations=visible_positions)
    root_pos_w = positions_w[0] if bool(finite_mask[0].item()) else visible_positions[0]
    return root_pos_w, pos_key, num_rendered, num_total


def _set_comparison_camera(
    base_env: ImitationRLEnv,
    *,
    reference_root_pos_w: torch.Tensor | None = None,
) -> None:
    policy_root = base_env.robot.data.root_pos_w.torch[POLICY_ENV_ID].detach()
    if reference_root_pos_w is None:
        origins = base_env.scene.env_origins[[REFERENCE_ENV_ID, POLICY_ENV_ID]]
        lookat = origins.mean(dim=0).detach().clone()
        lookat[2] = 0.9
    else:
        reference_root = reference_root_pos_w.to(device=base_env.device).reshape(3)
        lookat = 0.5 * (reference_root.detach() + policy_root)
        lookat = lookat.clone()
        lookat[2] = max(float(lookat[2].item()), 0.9)

    eye = lookat + torch.tensor([3.0, -5.0, 2.0], device=base_env.device)
    base_env.sim.set_camera_view(
        eye.detach().cpu().tolist(), lookat.detach().cpu().tolist()
    )


def _disable_termination_terms(env_cfg) -> None:
    """Disable termination terms so visual comparison runs until our explicit stop."""
    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is None:
        return

    disabled_terms: list[str] = []
    for name in getattr(terminations_cfg, "__dataclass_fields__", {}):
        if getattr(terminations_cfg, name, None) is None:
            continue
        setattr(terminations_cfg, name, None)
        disabled_terms.append(name)

    # Termination curricula mutate their target termination configurations on
    # reset. They cannot remain active after the corresponding terms above
    # have been removed for a full-horizon diagnostic playback.
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    disabled_curricula: list[str] = []
    if curriculum_cfg is not None:
        for name in getattr(curriculum_cfg, "__dataclass_fields__", {}):
            if getattr(curriculum_cfg, name, None) is None:
                continue
            setattr(curriculum_cfg, name, None)
            disabled_curricula.append(name)

    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 1.0e9

    if len(disabled_terms) > 0:
        print(
            "[INFO] Disabled comparison termination terms: "
            + ", ".join(sorted(disabled_terms))
        )
    if len(disabled_curricula) > 0:
        print(
            "[INFO] Disabled comparison termination curricula: "
            + ", ".join(sorted(disabled_curricula))
        )


def _disable_reward_terms(env_cfg) -> None:
    """Disable reward terms; this script is visual/evaluation playback only."""
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return

    disabled_terms: list[str] = []
    for name in getattr(rewards_cfg, "__dataclass_fields__", {}):
        if getattr(rewards_cfg, name, None) is None:
            continue
        setattr(rewards_cfg, name, None)
        disabled_terms.append(name)

    if len(disabled_terms) > 0:
        print(
            "[INFO] Disabled comparison reward terms: "
            + ", ".join(sorted(disabled_terms))
        )


def _ordered_trajectories(base_env: ImitationRLEnv) -> list[tuple[str, str, str]]:
    ordered = getattr(base_env.trajectory_manager, "_ordered_traj_list", None)
    if not ordered:
        raise RuntimeError("The trajectory manager does not expose trajectories.")
    return [(str(dataset), str(motion), str(traj)) for dataset, motion, traj in ordered]


def _print_trajectories(base_env: ImitationRLEnv) -> None:
    print("[INFO] Available trajectories:")
    for rank, (dataset, motion, trajectory) in enumerate(
        _ordered_trajectories(base_env)
    ):
        print(f"{rank:04d}\t{dataset}\t{motion}\t{trajectory}")


def _resolve_policy_trajectory_rank(base_env: ImitationRLEnv) -> int | None:
    if args_cli.policy_trajectory_rank is not None:
        rank = int(args_cli.policy_trajectory_rank)
        num_trajectories = len(_ordered_trajectories(base_env))
        if not 0 <= rank < num_trajectories:
            raise ValueError(
                f"--policy_trajectory_rank must be in [0, {num_trajectories - 1}], "
                f"got {rank}."
            )
        return rank

    if args_cli.policy_motion is None:
        return None

    matches: list[tuple[int, tuple[str, str, str]]] = []
    for rank, info in enumerate(_ordered_trajectories(base_env)):
        dataset, motion, trajectory = info
        if motion != args_cli.policy_motion:
            continue
        if args_cli.policy_dataset is not None and dataset != args_cli.policy_dataset:
            continue
        if (
            args_cli.policy_trajectory is not None
            and trajectory != args_cli.policy_trajectory
        ):
            continue
        matches.append((rank, info))

    if not matches:
        filters = {
            "dataset": args_cli.policy_dataset,
            "motion": args_cli.policy_motion,
            "trajectory": args_cli.policy_trajectory,
        }
        raise ValueError(f"No trajectory matched {filters}. Use --list_trajectories.")
    if len(matches) > 1:
        options = ", ".join(
            f"{rank}:{dataset}/{motion}/{trajectory}"
            for rank, (dataset, motion, trajectory) in matches
        )
        raise ValueError(
            "Motion selection is ambiguous; add --policy_dataset, "
            f"--policy_trajectory, or use --policy_trajectory_rank. Matches: {options}"
        )
    return matches[0][0]


def _force_policy_trajectory_on_reset(
    base_env: ImitationRLEnv,
    *,
    rank: int,
    start_step: int,
) -> None:
    if start_step < 0:
        raise ValueError("--policy_start_step must be >= 0.")
    tm = base_env.trajectory_manager

    def _custom_reset_fn(env_ids: torch.Tensor, _num_trajectories: int) -> torch.Tensor:
        return torch.full(
            (int(env_ids.numel()),),
            int(rank),
            dtype=torch.long,
            device=env_ids.device,
        )

    tm.reset_schedule = "custom"
    tm.custom_reset_fn = _custom_reset_fn
    tm.reset_start_step = int(start_step)

    # The G1 env can otherwise replace reset_start_step with its adaptive
    # full-trajectory sampler during _reset_idx. For explicit eval trajectory
    # selection, the CLI start step should be literal.
    if hasattr(base_env, "_random_reset_full_trajectory"):
        base_env._random_reset_full_trajectory = False
    if hasattr(base_env, "_random_reset_step_min"):
        base_env._random_reset_step_min = 0
    if hasattr(base_env, "_random_reset_step_max"):
        base_env._random_reset_step_max = 0

    dataset, motion, trajectory = _ordered_trajectories(base_env)[rank]
    print(
        "[INFO] Policy/language trajectory fixed to "
        f"rank={rank} dataset={dataset!r} motion={motion!r} "
        f"trajectory={trajectory!r} start_step={start_step}."
    )


class _PolicyTrackingMetrics:
    """Record per-step tracking quality for the policy environment.

    All quantities are read from the env's own live-order buffers, so they stay
    comparable across backends and are unaffected by any observation-side
    joint-order emulation.
    """

    def __init__(self, base_env, env_id: int, fall_height: float):
        self._env = base_env
        self._env_id = int(env_id)
        self._fall_height = float(fall_height)
        self._robot = base_env.scene["robot"]
        ee_ids, ee_names = self._robot.find_bodies(
            G1_EE_BODY_NAMES, preserve_order=True
        )
        if list(ee_names) != list(G1_EE_BODY_NAMES):
            raise RuntimeError(
                f"Could not resolve ordered G1 end effectors: got {ee_names}."
            )
        self._ee_ids = torch.tensor(
            ee_ids, dtype=torch.long, device=torch.device(str(base_env.device))
        )
        self.root_height: list[float] = []
        self.joint_pos_mae: list[float] = []
        self.ee_xyz_error: list[float] = []

    def record(self) -> None:
        env_id = self._env_id
        self.root_height.append(
            float(self._robot.data.root_pos_w.torch[env_id, 2].item())
        )
        expert_joint_pos = self._env.current_expert_frame["joint_pos"][env_id]
        live_joint_pos = self._robot.data.joint_pos.torch[env_id]
        self.joint_pos_mae.append(
            float((live_joint_pos - expert_joint_pos).abs().mean().item())
        )
        reference_ee = self._env._get_reference_body_pose_w_fast(G1_EE_BODY_NAMES)[0]
        robot_ee = self._env._get_robot_body_pose_w_fast(self._ee_ids)[0]
        delta = (robot_ee - reference_ee)[env_id]
        self.ee_xyz_error.append(
            float(torch.linalg.vector_norm(delta, dim=-1).mean().item())
        )

    def summary(self, step_dt: float | None) -> dict:
        fallen_at = next(
            (i for i, h in enumerate(self.root_height) if h < self._fall_height),
            None,
        )
        steps = len(self.root_height)

        def _mean_upto(values: list[float]) -> float | None:
            window = values[:fallen_at] if fallen_at is not None else values
            return sum(window) / len(window) if window else None

        return {
            "steps": steps,
            "fall_height_threshold_m": self._fall_height,
            "fell": fallen_at is not None,
            "fall_step": fallen_at,
            "fall_time_s": (
                None if fallen_at is None or step_dt is None else fallen_at * step_dt
            ),
            "survived_steps": steps if fallen_at is None else fallen_at,
            "survived_fraction": (steps if fallen_at is None else fallen_at)
            / max(steps, 1),
            "min_root_height_m": min(self.root_height) if self.root_height else None,
            "final_root_height_m": self.root_height[-1] if self.root_height else None,
            # Averaged before the fall, so a collapsed robot cannot flatter or
            # inflate the tracking numbers.
            "joint_pos_mae_rad_prefall": _mean_upto(self.joint_pos_mae),
            "ee_xyz_error_m_prefall": _mean_upto(self.ee_xyz_error),
        }


class _JointOrderEmulator:
    """Replay a legacy checkpoint's joint ordering.

    Checkpoints trained before the joint-order fix encode the *live*
    articulation order of the backend they were trained on, because the expert
    command terms and the action offset were resolved from the live enumeration
    rather than the pinned canonical list. Isaac Lab backends enumerate the G1
    differently (PhysX breadth-first, Newton depth-first), so such a checkpoint
    is only self-consistent on its original backend.

    The env now emits these terms in the pinned canonical order on every
    backend, so this shim permutes pinned -> foreign to make a legacy
    checkpoint runnable again, on either backend. It is a diagnostic and
    salvage aid, not a correctness fix: retraining is the real remedy.

    Only the genuinely order-dependent quantities are touched: the
    ``expert_motion`` blocks (29 positions followed by 29 velocities) and the
    action term's offset. Proprioception, body observations, and the action
    targets were always pinned and are left alone.
    """

    # Groups whose expert command reaches the actor or the latent posterior.
    _PERMUTED_GROUPS = ("policy", "critic")
    _JOINT_COUNT = 29

    def __init__(self, contract_path: str, base_env, action_term, device):
        import json

        contract = json.loads(Path(contract_path).expanduser().read_text("utf-8"))
        foreign_order = list(contract["robot_joint_names"])
        robot = base_env.scene["robot"]
        # Post-fix the expert command terms are pinned to the action term's
        # joint order on every backend, so that is the source ordering here.
        source_order = list(action_term._joint_names)
        if sorted(foreign_order) != sorted(source_order):
            raise ValueError(
                "The contract's joint names are a different set from the action "
                "term's joints; contracts must come from the same robot."
            )
        self.is_noop = foreign_order == source_order
        self.foreign_backend = contract.get("physics_cfg", "<unknown>")

        # expert_motion arrives in pinned order; deliver it in foreign order.
        self._perm = torch.tensor(
            [source_order.index(name) for name in foreign_order],
            dtype=torch.long,
            device=device,
        )

        # Reproduce the offset the foreign backend would have written: pinned
        # slot j receives the rest pose of foreign_order[j]. Built by name, so
        # it does not depend on the live enumeration.
        default_joint_pos = robot.data.default_joint_pos.torch
        default_by_name = {
            name: default_joint_pos[:, index]
            for index, name in enumerate(robot.joint_names)
        }
        self._action_term = action_term
        self._foreign_offset = torch.stack(
            [default_by_name[name] for name in foreign_order], dim=1
        )

    def apply_action_offset(self) -> None:
        """Restore the foreign-order offset (the reset event rewrites it)."""
        self._action_term._offset.copy_(self._foreign_offset)

    def permute_observations(self, td) -> None:
        """Reorder every ``expert_motion`` block in place."""
        for group in self._PERMUTED_GROUPS:
            key = (group, "expert_motion")
            try:
                tensor = td.get(key)
            except KeyError:
                continue
            if tensor is None:
                continue
            blocks = tensor.shape[-1] // self._JOINT_COUNT
            if blocks * self._JOINT_COUNT != tensor.shape[-1]:
                raise ValueError(
                    f"{key} width {tensor.shape[-1]} is not a multiple of "
                    f"{self._JOINT_COUNT}; cannot reorder it by joint."
                )
            reshaped = tensor.reshape(*tensor.shape[:-1], blocks, self._JOINT_COUNT)
            td.set(key, reshaped.index_select(-1, self._perm).reshape(tensor.shape))


def _skill_commander_embeddings_path(agent_cfg) -> str | None:
    ipmd_cfg = getattr(agent_cfg, "ipmd", None)
    if ipmd_cfg is None:
        return None
    path_value = str(getattr(ipmd_cfg, "skill_commander_embeddings_path", "")).strip()
    return path_value or None


def _language_phrase_for_motion(
    motion_name: str, embeddings_path: str | None
) -> tuple[str | None, str | None]:
    if embeddings_path is None:
        return None, None
    table_path = Path(embeddings_path).expanduser()
    if not table_path.is_file():
        return None, str(table_path)
    table = torch.load(table_path, map_location="cpu", weights_only=False)
    name_to_index = table.get("name_to_index", {})
    index = name_to_index.get(str(motion_name))
    if index is None:
        return None, str(table_path)
    phrases = table.get("phrases")
    if isinstance(phrases, list) and 0 <= int(index) < len(phrases):
        return str(phrases[int(index)]), str(table_path)
    names = table.get("names")
    if isinstance(names, list) and 0 <= int(index) < len(names):
        return str(names[int(index)]), str(table_path)
    return None, str(table_path)


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
):
    """Play an RLOpt policy next to the expert reference motion."""
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.max_steps is not None and int(args_cli.max_steps) <= 0:
        raise ValueError("--max_steps must be > 0 when provided.")
    if args_cli.video_length is not None and int(args_cli.video_length) <= 0:
        raise ValueError("--video_length must be > 0 when provided.")
    if args_cli.video_seconds is not None and float(args_cli.video_seconds) <= 0.0:
        raise ValueError("--video_seconds must be > 0 when provided.")

    env_cfg.scene.num_envs = 2
    agent_cfg.env.num_envs = 2
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    logger_cfg = getattr(agent_cfg, "logger", None)
    if logger_cfg is not None and not args_cli.enable_wandb:
        logger_cfg.backend = ""
        logger_cfg.video = False
        print("[INFO] Disabled RLOpt W&B logging for comparison eval.")

    if args_cli.keep_terminations:
        print("[INFO] Keeping comparison termination terms enabled.")
    else:
        _disable_termination_terms(env_cfg)
    if args_cli.keep_rewards:
        print("[INFO] Keeping comparison reward terms enabled.")
    else:
        _disable_reward_terms(env_cfg)

    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required for compare_policy_reference.py.")
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args_cli.output_dir is not None:
        log_dir = os.path.abspath(os.path.expanduser(args_cli.output_dir))
    else:
        task_name = (
            args_cli.task.split(":")[-1]
            if args_cli.task is not None
            else "unknown_task"
        )
        log_root_path = os.path.abspath(
            os.path.join("logs", "rlopt_eval", "compare_policy_reference", task_name)
        )
        log_dir = os.path.join(
            log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging comparison eval in directory: {log_dir}")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported for RLOpt play.")

    raw_base_env = _unwrap_imitation_env(env)
    if args_cli.list_trajectories:
        _print_trajectories(raw_base_env)
        env.close()
        return

    policy_rank = _resolve_policy_trajectory_rank(raw_base_env)
    if policy_rank is not None:
        _force_policy_trajectory_on_reset(
            raw_base_env,
            rank=policy_rank,
            start_step=int(args_cli.policy_start_step),
        )

    step_limits: list[int] = []
    if args_cli.max_steps is not None:
        step_limits.append(int(args_cli.max_steps))
    if args_cli.video_length is not None:
        step_limits.append(int(args_cli.video_length))
    if args_cli.video_seconds is not None:
        step_dt = float(getattr(raw_base_env, "step_dt", 0.0) or 0.0)
        if step_dt <= 0.0:
            raise ValueError("Could not infer env step_dt for --video_seconds.")
        step_limits.append(max(1, int(round(float(args_cli.video_seconds) / step_dt))))
    rollout_step_limit = min(step_limits) if len(step_limits) > 0 else None

    tm = raw_base_env.trajectory_manager
    if policy_rank is not None:
        selected_reference_steps = int(tm._length[int(policy_rank)].item())
        selected_start_step = min(
            int(args_cli.policy_start_step), max(selected_reference_steps - 1, 0)
        )
        default_run_steps = max(1, selected_reference_steps - selected_start_step)
    else:
        default_run_steps = max(1, int(tm._length.max().item()))
    video_length = (
        rollout_step_limit if rollout_step_limit is not None else default_run_steps
    )
    step_counter_limit = max(1, video_length + 1)

    _set_comparison_camera(raw_base_env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "compare_policy_reference"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during reference/policy comparison.")
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
            StepCounter(step_counter_limit),
            RewardClipping(-10.0, 5.0),
        ),
    )

    base_env = _unwrap_imitation_env(env)
    use_reference_robot_replay = args_cli.reference_visualization in ("robot", "both")
    use_reference_body_markers = args_cli.reference_visualization in (
        "body_markers",
        "both",
    )
    base_env.configure_reference_replay_targets(
        source_env_ids=[POLICY_ENV_ID],
        target_env_ids=[REFERENCE_ENV_ID],
    )

    reference_body_markers = (
        _create_reference_body_markers() if use_reference_body_markers else None
    )
    role_markers = _create_role_markers()

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(checkpoint_path)

    collector_policy = agent.collector_policy
    collector_policy.eval()

    dt = getattr(base_env, "step_dt", None)

    joint_order_emulator = None
    if args_cli.emulate_joint_order_from is not None:
        joint_order_emulator = _JointOrderEmulator(
            args_cli.emulate_joint_order_from,
            base_env,
            base_env.action_manager.get_term("joint_pos"),
            torch.device(str(base_env.device)),
        )
        if joint_order_emulator.is_noop:
            raise ValueError(
                "--emulate_joint_order_from names a contract whose joint order "
                "already matches this backend; the emulation would do nothing. "
                "Pass the contract from the OTHER backend."
            )
        print(
            "[WARN] DIAGNOSTIC joint-order emulation is ACTIVE: expert_motion "
            f"and the action offset are permuted into {joint_order_emulator.foreign_backend} "
            "order. This run is not a valid performance measurement."
        )

    tracking_metrics = None
    if args_cli.metrics_json is not None:
        tracking_metrics = _PolicyTrackingMetrics(
            base_env, POLICY_ENV_ID, args_cli.fall_height
        )

    td = env.reset()
    base_env.apply_reference_replay_targets()

    reference_root_pos_w = None
    reference_marker_stats = None
    if reference_body_markers is not None:
        reference_root_pos_w, reference_pos_key, rendered_bodies, total_bodies = (
            _update_reference_body_markers(base_env, reference_body_markers)
        )
        reference_marker_stats = (reference_pos_key, rendered_bodies, total_bodies)
    _set_comparison_camera(base_env, reference_root_pos_w=reference_root_pos_w)
    _update_role_markers(
        base_env, role_markers, reference_root_pos_w=reference_root_pos_w
    )
    timestep = 0
    if args_cli.reference_visualization == "body_markers":
        print(
            "[INFO] Starting comparison loop. env 0 shows reference body-state markers "
            "with expert qpos robot replay; env 1 runs policy."
        )
    elif args_cli.reference_visualization == "both":
        print(
            "[INFO] Starting comparison loop. env 0 shows reference body-state markers "
            "plus qpos robot replay; env 1 runs policy."
        )
    else:
        print(
            "[INFO] Starting comparison loop. env 0 replays reference qpos robot; "
            "env 1 runs policy."
        )
    print("[INFO] Visual markers: blue = REFERENCE body state/role, red = POLICY role.")
    dataset, motion, trajectory = base_env.trajectory_manager.get_env_traj_info(
        POLICY_ENV_ID
    )
    tm = base_env.trajectory_manager
    loaded_rank = int(tm.env_traj_rank[POLICY_ENV_ID].item())
    loaded_step = int(tm.env_step[POLICY_ENV_ID].item())
    embeddings_path = _skill_commander_embeddings_path(agent_cfg)
    language_phrase, resolved_embeddings_path = _language_phrase_for_motion(
        motion, embeddings_path
    )
    print(
        "[INFO] Loaded env 1 trajectory for policy/language conditioning: "
        f"rank={loaded_rank}, local_step={loaded_step}, dataset={dataset!r}, "
        f"motion={motion!r}, trajectory={trajectory!r}."
    )
    if language_phrase is None:
        print(
            "[INFO] Language conditioning: "
            f"motion_name={motion!r}, phrase=<unresolved>, "
            f"embeddings={resolved_embeddings_path!r}."
        )
    else:
        print(
            "[INFO] Language conditioning: "
            f"motion_name={motion!r}, phrase={language_phrase!r}, "
            f"embeddings={resolved_embeddings_path!r}."
        )

    if reference_marker_stats is not None:
        reference_pos_key, rendered_bodies, total_bodies = reference_marker_stats
        print(
            "[INFO] Reference visualization source: "
            f"env={POLICY_ENV_ID} current_expert_frame[{reference_pos_key!r}] -> "
            f"env={REFERENCE_ENV_ID} marker lane, "
            f"rendered_bodies={rendered_bodies}/{total_bodies}."
        )
    if use_reference_robot_replay:
        print(
            "[INFO] qpos robot replay is enabled for env 0. This is diagnostic; "
            "training losses/observations use the body-state tensors above."
        )

    while simulation_app.is_running():
        start_time = time.time()
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            if joint_order_emulator is not None:
                # The reset event rewrites the offset, so reassert it every step.
                joint_order_emulator.apply_action_offset()
                joint_order_emulator.permute_observations(td)
            td = collector_policy(td)
            action = td.get("action")
            if action is None:
                raise KeyError(
                    "Collector output is missing the top-level 'action' tensor."
                )
            action[REFERENCE_ENV_ID].zero_()
            td = env.step(td)
            if tracking_metrics is not None:
                tracking_metrics.record()
            reference_root_pos_w = None
            if reference_body_markers is not None:
                reference_root_pos_w, _, _, _ = _update_reference_body_markers(
                    base_env, reference_body_markers
                )
            _set_comparison_camera(base_env, reference_root_pos_w=reference_root_pos_w)
            _update_role_markers(
                base_env, role_markers, reference_root_pos_w=reference_root_pos_w
            )
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )

        timestep += 1
        if rollout_step_limit is not None and timestep >= rollout_step_limit:
            print(f"[INFO] Stopping comparison after step limit: {rollout_step_limit}.")
            break

        reference_done = base_env.current_reference_is_final_frame()[POLICY_ENV_ID]
        if bool(reference_done.item()):
            print(
                f"[INFO] Stopping comparison because env 1 reference ended at step {timestep}."
            )
            break

        if args_cli.real_time and dt is not None:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if tracking_metrics is not None:
        import json

        summary = tracking_metrics.summary(dt)
        payload = {
            "checkpoint": checkpoint_path,
            "task": args_cli.task,
            "physics_cfg": type(env_cfg.sim.physics).__name__,
            "emulated_joint_order_from": args_cli.emulate_joint_order_from,
            "motion": motion,
            "seed": args_cli.seed,
            "summary": summary,
            "per_step": {
                "root_height_m": tracking_metrics.root_height,
                "joint_pos_mae_rad": tracking_metrics.joint_pos_mae,
                "ee_xyz_error_m": tracking_metrics.ee_xyz_error,
            },
        }
        metrics_path = Path(args_cli.metrics_json).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("POLICY_TRACKING_SUMMARY " + json.dumps(summary, sort_keys=True))
        print(f"[INFO] Policy tracking metrics written to {metrics_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
