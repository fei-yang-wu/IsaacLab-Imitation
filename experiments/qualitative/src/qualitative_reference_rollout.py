#!/usr/bin/env python3
# ruff: noqa: E402
"""Reference motion vs. encoder-driven policy rollout, side by side.

Task 1 of the qualitative analysis. For each of N motions, two robots run in one
scene: env 0 replays the expert reference articulation and env 1 is the
low-level tracker, driven by skill latents the frozen encoder produces from the
reference itself. One MP4 per motion, with a camera that keeps both lanes in
frame.

The window-by-window behaviour is the agent's own, not a reimplementation. With
``agent.ipmd.command_source=hl_skill`` the IPMD agent builds a
``FrozenHighLevelSkillCommandSampler`` that holds each ``z`` for
``latent_steps`` control steps and then re-encodes the next window from
``current_expert_macro_transition_batch`` at the live reference cursor. That is
exactly "roll out for a window, then take the next window's latent", and using
the production path means this playback cannot drift from how the tracker was
trained. The script asserts the checkpoint horizon and logs each renewal step so
the boundaries are auditable in the transcript.

Terminations are disabled by default, so MPJPE is measured over the intended
horizon rather than a termination-truncated rollout, and the retained video
shows the same non-terminating pass.

Example::

    pixi run -e isaaclab python experiments/qualitative/src/qualitative_reference_rollout.py \\
        --encoder_checkpoint logs/.../encoder/checkpoints/latest.pt \\
        --policy_checkpoint logs/.../models/model_step_4325179392.pt \\
        --num_motions 8 --seed 0 --video --output_dir outputs/.../reference_rollout \\
        <shared hydra overrides>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# `qualitative_common` imports torch, and torch must not initialize CUDA before
# AppLauncher has chosen the device -- doing so makes AppLauncher's deferred
# `torch.cuda.set_device` fail with a device-index assert. So it is imported
# below, after the app is launched, and the parser uses literals here.
from isaaclab.app import AppLauncher

REFERENCE_ENV_ID = 0
POLICY_ENV_ID = 1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v2")
parser.add_argument(
    "--algo", "--algorithm", dest="algorithm", type=str.upper, default="IPMD"
)
parser.add_argument("--encoder_checkpoint", type=str, required=True)
parser.add_argument("--policy_checkpoint", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument(
    "--reference_arrays_dir",
    type=str,
    default=None,
    help="Reference arrays directory. Defaults to the repo-local 129k arrays.",
)
parser.add_argument("--num_motions", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--ranks", type=str, default=None, help="Comma-separated ranks; overrides the draw."
)
parser.add_argument(
    "--motions", type=str, default=None, help="Comma-separated motion names."
)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument(
    "--switch_at_step",
    type=int,
    default=None,
    help=(
        "Retarget the reference to a second motion at this control step, without "
        "resetting the robot. Off by default."
    ),
)
parser.add_argument(
    "--switch_motion",
    type=str,
    default=None,
    help=(
        "Motion to switch to. Default: the next motion in the selected list, so "
        "N motions give N A->B pairs."
    ),
)
parser.add_argument(
    "--switch_rank", type=int, default=None, help="Switch target by rank instead."
)
parser.add_argument(
    "--switch_start_frame",
    type=int,
    default=0,
    help="Frame of the second motion to continue from.",
)
parser.add_argument(
    "--switch_command_frame",
    type=str,
    default="reference",
    choices=["reference", "robot"],
    help=(
        "After the switch, which frame the published command is encoded in. "
        "reference: the second motion in ITS OWN frame, transplanted onto the "
        "robot wherever it happens to be -- the command then says 'do this "
        "motion' and carries no correction for the robot's pose. robot: the "
        "ordinary deployment path, the second motion seen from the robot."
    ),
)
parser.add_argument(
    "--switch_align",
    type=str,
    default="xy",
    choices=["xy", "none"],
    help=(
        "xy: move the second motion's placement so it starts at the robot's "
        "current ground position, which keeps the switch a change of MOTION. "
        "none: leave it at the dataset placement, so the target also jumps by "
        "however far the robot travelled. Both record the measured jump."
    ),
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Cap per motion. Default: run until the reference clip ends.",
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--keep_terminations",
    action="store_true",
    default=False,
    help=(
        "Keep early terminations. Off by default: the retained pass is the "
        "full-horizon diagnostic AGENTS.md requires."
    ),
)
parser.add_argument("--fall_height", type=float, default=0.4)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below needs the simulation app running."""

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qualitative_common as qc
import qualitative_rollout as qr

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMD
from rlopt.env_interface import resolve_imitation_interface, supports
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

import isaaclab_tasks  # noqa: F401
import isaaclab_imitation.tasks  # noqa: F401

from imitation_experiments.provenance.paper_protocol_metadata import (
    disable_domain_randomization,
)

ALGORITHM_CLASS_MAP = {"IPMD": IPMD}

REFERENCE_MARKER_COLOR = (0.0, 0.75, 1.0)
POLICY_MARKER_COLOR = (1.0, 0.1, 0.0)
MARKER_HEIGHT_OFFSET = 1.35
G1_EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def _create_role_markers() -> VisualizationMarkers:
    """A blue dot over the reference lane and a red one over the policy lane."""
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/World/Visuals/qualitative_role_markers",
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
    )


def _update_role_markers(base_env, markers: VisualizationMarkers) -> None:
    positions = base_env.robot.data.root_pos_w.torch[
        [REFERENCE_ENV_ID, POLICY_ENV_ID]
    ].clone()
    positions[:, 2] += MARKER_HEIGHT_OFFSET
    markers.visualize(
        translations=positions,
        marker_indices=torch.tensor([0, 1], dtype=torch.long, device=base_env.device),
    )


def _set_comparison_camera(base_env) -> None:
    """Follow the midpoint of the two lanes, close enough to read joint angles."""
    reference_root = base_env.robot.data.root_pos_w.torch[REFERENCE_ENV_ID].detach()
    policy_root = base_env.robot.data.root_pos_w.torch[POLICY_ENV_ID].detach()
    lookat = (0.5 * (reference_root + policy_root)).clone()
    lookat[2] = max(float(lookat[2].item()), 0.9)
    eye = lookat + torch.tensor([2.0, -3.5, 1.2], device=base_env.device)
    base_env.sim.set_camera_view(
        eye.detach().cpu().tolist(), lookat.detach().cpu().tolist()
    )


def _force_trajectory_on_reset(base_env, *, rank: int, start_step: int) -> None:
    """Pin both lanes to one (rank, frame).

    Both halves are required: the trajectory manager's callback picks the rank,
    and the v2 reference command term's own selection sampler must be rebuilt as
    fixed, or the SONIC training default resamples rank and frame jointly and
    silently overrides the callback.
    """
    tm = base_env.trajectory_manager

    def _custom_reset_fn(env_ids: torch.Tensor, _num_trajectories: int) -> torch.Tensor:
        return torch.full(
            (int(env_ids.numel()),), int(rank), dtype=torch.long, device=env_ids.device
        )

    tm.reset_schedule = "custom"
    tm.custom_reset_fn = _custom_reset_fn
    tm.reset_start_step = int(start_step)

    reference_term = getattr(base_env, "reference_command", None)
    selection = getattr(getattr(reference_term, "cfg", None), "selection", None)
    if selection is not None:
        selection.schedule = "custom"
        selection.full_trajectory = False
        selection.start_mode = "fixed"
        selection.start_frame = int(start_step)
        selection.random_step_min = int(start_step)
        selection.random_step_max = int(start_step)
        selection.adaptive_weight_fn = None
        reference_term._adaptive_failure_reset_sampler = None
        reference_term._build_reset_samplers()

    for attribute, value in (
        ("_random_reset_full_trajectory", False),
        ("_random_reset_step_min", 0),
        ("_random_reset_step_max", 0),
    ):
        if hasattr(base_env, attribute):
            setattr(base_env, attribute, value)


class _TrackingMetrics:
    """Per-step tracking quality for the policy lane, from the env's own buffers."""

    def __init__(self, base_env, env_id: int, fall_height: float) -> None:
        self._env = base_env
        self._env_id = int(env_id)
        self._fall_height = float(fall_height)
        self._robot = base_env.scene["robot"]
        ee_ids, ee_names = self._robot.find_bodies(
            list(G1_EE_BODY_NAMES), preserve_order=True
        )
        if list(ee_names) != list(G1_EE_BODY_NAMES):
            msg = f"Could not resolve ordered G1 end effectors: got {ee_names}."
            raise RuntimeError(msg)
        self._ee_ids = torch.tensor(
            ee_ids, dtype=torch.long, device=torch.device(str(base_env.device))
        )
        self.root_height: list[float] = []
        self.joint_pos_mae: list[float] = []
        self.ee_xyz_error: list[float] = []
        self.mpjpe_local_m: list[float] = []
        self.mpjpe_global_m: list[float] = []

    def record(self) -> None:
        """Read the env's own live-order buffers, as compare_policy_reference does."""
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
        mpjpe = self._env._compute_mpjpe_metrics()
        if mpjpe is None:
            raise RuntimeError("The environment did not expose MPJPE metric bodies.")
        local, glob = mpjpe
        self.mpjpe_local_m.append(float(local[env_id].item()))
        self.mpjpe_global_m.append(float(glob[env_id].item()))

    def summary(self) -> dict[str, object]:
        fell_at = next(
            (
                index
                for index, height in enumerate(self.root_height)
                if height < self._fall_height
            ),
            None,
        )
        steps = len(self.root_height)

        def _mean_prefall(values: list[float]) -> float | None:
            # Average before the fall, so a collapsed robot cannot flatter or
            # inflate the tracking numbers.
            window = values[:fell_at] if fell_at is not None else values
            return float(sum(window) / len(window)) if window else None

        return {
            "steps": steps,
            # v2 uses G1SonicTerminationsCfg, which sets base_too_low=None, so a
            # fall never terminates. Survival is measured from root height here.
            "fall_height_threshold_m": self._fall_height,
            "fell": fell_at is not None,
            "fall_step": fell_at,
            "survived_steps": steps if fell_at is None else fell_at,
            "min_root_height_m": min(self.root_height) if self.root_height else None,
            "joint_pos_mae_rad_prefall": _mean_prefall(self.joint_pos_mae),
            "ee_xyz_error_m_prefall": _mean_prefall(self.ee_xyz_error),
            "mpjpe_local_mm_prefall": (
                None
                if _mean_prefall(self.mpjpe_local_m) is None
                else 1000.0 * _mean_prefall(self.mpjpe_local_m)
            ),
            "mpjpe_global_mm_prefall": (
                None
                if _mean_prefall(self.mpjpe_global_m) is None
                else 1000.0 * _mean_prefall(self.mpjpe_global_m)
            ),
        }


#: A replay lane further than this from the reference it is supposed to be
#: replaying is not a reference frame at all, and its code would be neither the
#: deployed one nor the motion's own.
REFERENCE_LANE_TOLERANCE_M = 0.05


@torch.no_grad()
def _encode_macro_windows(bundle, base_env):
    """Encode the macro window of BOTH lanes at the current reference cursor.

    Called immediately after the collector step and before ``env.step``, so the
    cursor is still the one the frozen sampler just encoded from.

    One forward covers both lanes, and they mean different things, because the
    macro state's anchor terms are the expert anchor expressed in the LIVE
    robot's frame:

    * the policy lane (env 1) gives the command the tracker was actually given;
    * the reference lane (env 0) replays the expert articulation, so its robot
      anchor IS the reference anchor and its window is the one expressed in the
      reference's own frame -- the same view DiffSR pretraining samples through
      ``sample_expert_macro_transition_batch``, whose "expert" context anchors a
      window on its own first frame.
    """
    batch = base_env.current_expert_macro_transition_batch(
        horizon_steps=bundle.horizon_steps
    )["hl"]
    state = batch["state"].to(device=bundle.device, dtype=torch.float32)
    future_window = batch["future_window"].to(device=bundle.device, dtype=torch.float32)
    encoded = qc.encode_windows(bundle, state, future_window)
    categories = encoded.get("categories")
    return (
        # None for a continuous latent: there is no code to record, and the
        # caller omits the code columns rather than storing a stand-in.
        categories.detach().cpu() if categories is not None else None,
        encoded["z"].detach().cpu(),
        state.detach().cpu(),
    )


def _retarget_reference(
    base_env,
    *,
    env_ids: list[int],
    rank: int,
    start_frame: int,
    align: str,
    anchor_slice: tuple[int, int],
    horizon_steps: int,
) -> dict[str, object]:
    """Point the reference at another motion mid-rollout, without a reset.

    The robot keeps its physical state; only the thing it is being asked to
    follow changes. The trajectory manager's own ``set_env_cursor`` does the
    retarget, so the cursor stays inside the manager's clamping rules.

    Placement matters here. A reference is laid into the world by a FIXED
    transform -- identity rotation, translation equal to the env origin -- so a
    motion always replays from the same spot. If the robot walked three metres
    while tracking the first motion, the second motion still starts back at the
    origin, and the command would be dominated by that gap rather than by the
    change of motion. ``align="xy"`` therefore shifts the env's reference
    placement so the new motion begins at the robot's current ground position.
    Heading is NOT aligned: this environment builds the alignment rotation as
    identity, so turning the reference would need a different transform API.
    The measured jump is returned either way.
    """
    from isaaclab.utils import math as math_utils

    tm = base_env.trajectory_manager
    plane = base_env.expert_data_plane
    policy_id = int(env_ids[0])
    all_ids = torch.as_tensor(env_ids, dtype=torch.long, device=tm.env_step.device)

    # Only the policy lane is retargeted. The replay lane is synchronized from
    # it by the env's own hook below, which also rewrites that lane's
    # articulation, so both lanes stay on one cursor by construction.
    policy_ids = torch.as_tensor(
        [policy_id], dtype=torch.long, device=tm.env_step.device
    )
    tm.set_env_cursor(
        env_ids=policy_ids,
        ranks=torch.full_like(policy_ids, int(rank)),
        steps=int(start_frame),
    )

    def _gap() -> torch.Tensor:
        """The world vector from the robot's anchor to the expert's anchor.

        The per-step MDP cache holds the previous alignment and the previous
        reference reads; without invalidating it the next macro window would be
        built from the motion we just left.
        """
        plane._mdp_cache_step = -1
        batch = base_env.current_expert_macro_transition_batch(
            horizon_steps=horizon_steps
        )["hl"]
        start, end = int(anchor_slice[0]), int(anchor_slice[1])
        offset_b = batch["state"][policy_id, start:end].detach()
        quat = base_env.robot.data.root_quat_w.torch.detach()[policy_id].reshape(1, 4)
        return math_utils.quat_apply(
            quat, offset_b.reshape(1, 3).to(quat.device)
        ).reshape(3)

    offset_w = _gap()
    gap_before = float(offset_w.norm())

    if align == "xy":
        origins = plane._expert_env_origins
        # MINUS the offset: the expert has to come to the robot. Adding it would
        # push the reference further away by the same amount, which is exactly
        # what the before/after gap catches.
        shift = (-offset_w).to(device=origins.device, dtype=origins.dtype).clone()
        # Ground plane only: a vertical shift would sink the reference into the
        # floor or float it, and a height mismatch is real information.
        shift[2] = 0.0
        # Both lanes move by the SAME amount, each about its own origin, so the
        # two lanes stay comparable side by side.
        origins.index_add_(
            0,
            all_ids.to(origins.device),
            shift.reshape(1, 3).expand(int(all_ids.numel()), 3),
        )
        offset_w = _gap()
        horizontal = float(offset_w[:2].norm())
        if horizontal > 1.0e-3:
            msg = (
                f"Aligning the switch target left {horizontal * 1000:.1f} mm of "
                "horizontal gap; it should be zero by construction. The "
                "placement shift did not land where it was computed."
            )
            raise RuntimeError(msg)

    gap_after = float(offset_w.norm())

    # Rewrite the replay lane from the new cursor, so its articulation is the
    # new motion before anything reads its macro window.
    base_env.apply_reference_replay_targets()
    plane._mdp_cache_step = -1

    return {
        "rank": int(rank),
        "start_frame": int(start_frame),
        "align": align,
        "anchor_gap_before_m": round(gap_before, 6),
        "anchor_gap_after_m": round(gap_after, 6),
    }


def _video_stem(rank: int, motion: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in motion
    ).strip("_")
    return f"rank-{rank:06d}-{safe or 'motion'}"


@hydra_task_config(args_cli.task, qc.AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
) -> None:
    reference_arrays_dir = Path(
        args_cli.reference_arrays_dir
        or (qc.repo_root() / qc.DEFAULT_REFERENCE_ARRAYS_DIRNAME)
    ).resolve()

    device = torch.device(args_cli.device or "cuda:0")
    bundle = qc.load_skill_encoder(args_cli.encoder_checkpoint, device)
    binding = qc.assert_encoder_binding(
        bundle.checkpoint_path, args_cli.policy_checkpoint
    )
    print(
        f"[PASS] encoder binding: {binding['skill_checkpoint_sha256'][:16]}... "
        f"embedded in {Path(args_cli.policy_checkpoint).name}"
    )

    # The agent's frozen sampler renews z every `latent_steps` steps; that hold
    # must be the encoder's horizon or the published command stops matching the
    # window it was encoded from.
    hold_steps = int(agent_cfg.ipmd.latent_steps_min)
    if hold_steps != int(agent_cfg.ipmd.latent_steps_max):
        msg = (
            "This playback needs a fixed command hold: "
            f"latent_steps_min={hold_steps} != latent_steps_max="
            f"{int(agent_cfg.ipmd.latent_steps_max)}."
        )
        raise ValueError(msg)
    if hold_steps != bundle.horizon_steps:
        msg = (
            f"Command hold ({hold_steps} steps) does not match the encoder horizon "
            f"({bundle.horizon_steps}). Pass agent.ipmd.latent_steps_min/max="
            f"{bundle.horizon_steps}."
        )
        raise ValueError(msg)
    if str(agent_cfg.ipmd.command_source) != "hl_skill":
        msg = (
            "This script drives the command from the frozen encoder; pass "
            "agent.ipmd.command_source=hl_skill."
        )
        raise ValueError(msg)
    print(
        f"[INFO] Command: {bundle.z_dim} "
        f"{'code' if bundle.is_discrete else 'continuous'} + 2 phase = "
        f"{bundle.latent_command_dim}, re-encoded every {hold_steps} steps."
    )

    catalog = qc.MotionCatalog.from_reference_arrays(reference_arrays_dir)
    selected = catalog.select(
        count=int(args_cli.num_motions),
        seed=int(args_cli.seed),
        min_length=bundle.horizon_steps + 1,
        ranks=qr.parse_int_list(args_cli.ranks),
        motions=qr.parse_str_list(args_cli.motions),
    )
    print(f"[INFO] {len(selected)} motions from {catalog.manifest_path}:")
    for entry in selected:
        print(f"       rank={entry.rank:6d} frames={entry.length:5d} {entry.motion}")

    output_dir = qc.prepare_output_dir(
        args_cli.output_dir, overwrite=args_cli.overwrite
    )
    video_folder = output_dir / "videos"

    env_cfg.scene.num_envs = 2
    agent_cfg.env.num_envs = 2
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.collector.frames_per_batch *= 2
    logger_cfg = getattr(agent_cfg, "logger", None)
    if logger_cfg is not None:
        logger_cfg.backend = ""
        logger_cfg.video = False

    if args_cli.keep_terminations:
        disabled_terminations: list[str] = []
        print("[INFO] Keeping termination terms enabled.")
    else:
        disabled_terminations = qr.disable_all_terminations(env_cfg)
        print(
            "[INFO] Full-horizon diagnostic pass; disabled terminations: "
            f"{sorted(disabled_terminations)}"
        )
    dr_record = disable_domain_randomization(env_cfg)
    env_cfg.log_dir = str(output_dir)

    longest = max(entry.length for entry in selected)
    step_cap = int(args_cli.max_steps) if args_cli.max_steps is not None else longest
    if step_cap <= 0:
        raise ValueError("--max_steps must be > 0.")

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")
    raw_base_env = qr.unwrap_imitation_env(env)

    video_recorder = None
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            # Playlist clips are started and stopped by hand, one per motion.
            step_trigger=lambda _step: False,
            video_length=step_cap + 2,
            disable_logger=True,
        )
        video_recorder = env

    env = IsaacLabWrapper(env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(
            RewardSum(), StepCounter(step_cap + 1), RewardClipping(-10.0, 5.0)
        ),
    )
    base_env = qr.unwrap_imitation_env(env)
    base_env.configure_reference_replay_targets(
        source_env_ids=[POLICY_ENV_ID], target_env_ids=[REFERENCE_ENV_ID]
    )
    role_markers = _create_role_markers()

    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    loaded = qr.load_policy_weights(
        agent, Path(args_cli.policy_checkpoint).expanduser().resolve(), device
    )
    print(f"[INFO] Loaded network weights: {loaded}")
    collector_policy = agent.collector_policy
    collector_policy.eval()
    # The frozen encoder sampler the collector wrapper drives. Read only, to
    # record when it actually renews the command.
    sampler = getattr(agent, "_hl_skill_command_sampler", None)
    if sampler is None:
        msg = (
            "The agent did not build a frozen high-level skill sampler. "
            "agent.ipmd.command_source=hl_skill must reach the agent config."
        )
        raise RuntimeError(msg)
    if int(sampler.phase_period) != hold_steps:
        msg = (
            f"Sampler phase period {int(sampler.phase_period)} != command hold "
            f"{hold_steps}; the published phase would not match the window."
        )
        raise ValueError(msg)

    # Where `expert_anchor_pos_b` sits inside one 38-value macro row. Asked of
    # the env rather than hard-coded, so a macro-term change cannot silently
    # move the check onto joint angles.
    anchor_slice = base_env.expert_macro_feature_slices(bundle.horizon_steps)[
        "expert_anchor_pos_b"
    ]
    anchor_start, anchor_end = int(anchor_slice[0]), int(anchor_slice[1])
    print(
        f"[INFO] expert_anchor_pos_b occupies macro values "
        f"[{anchor_start}, {anchor_end}) of {bundle.state_dim}."
    )

    # --- the optional mid-rollout switch --------------------------------- #
    switch_step = args_cli.switch_at_step
    switch_targets: list[qc.MotionEntry] | None = None
    if switch_step is not None:
        if int(switch_step) <= 0:
            raise ValueError(
                "--switch_at_step must be > 0; 0 would never play the first motion."
            )
        if args_cli.switch_rank is not None and args_cli.switch_motion is not None:
            raise ValueError("Pass --switch_rank or --switch_motion, not both.")
        if args_cli.switch_rank is not None:
            target = catalog.by_rank(int(args_cli.switch_rank))
            switch_targets = [target] * len(selected)
        elif args_cli.switch_motion is not None:
            target = catalog.by_rank(catalog.rank_for_motion(args_cli.switch_motion))
            switch_targets = [target] * len(selected)
        else:
            if len(selected) < 2:
                raise ValueError(
                    "Pairing motions round-robin needs at least 2 selections; pass "
                    "--switch_motion / --switch_rank, or raise --num_motions."
                )
            # Round robin: N motions give N A->B pairs and every motion appears
            # on both sides once.
            switch_targets = [
                selected[(i + 1) % len(selected)] for i in range(len(selected))
            ]
        short = [
            target
            for target in switch_targets
            if target.length
            < int(args_cli.switch_start_frame) + bundle.horizon_steps + 1
        ]
        if short:
            names = ", ".join(f"{e.motion} ({e.length} frames)" for e in short[:5])
            raise ValueError(
                f"Switch targets have no complete window at frame "
                f"{int(args_cli.switch_start_frame)}: {names}"
            )
        print(
            f"[INFO] Switch at step {int(switch_step)} to frame "
            f"{int(args_cli.switch_start_frame)}, align={args_cli.switch_align}:"
        )
        for source, target in zip(selected, switch_targets, strict=True):
            print(f"       {source.motion} -> {target.motion}")

    # After the switch the command can be transplanted: encoded from the second
    # motion in ITS OWN frame and handed to the tracker as-is. The frozen
    # sampler cannot produce that -- it always reads the window relative to the
    # live robot -- so those steps are published by hand, through the same
    # capability surface RLOpt's own controller uses.
    transplant = (
        switch_step is not None and str(args_cli.switch_command_frame) == "reference"
    )
    publish_actor_command = None
    policy_operator = None
    latent_key = ("policy", "latent_command")
    sim_device = torch.device(str(base_env.device))
    if transplant:
        interface = resolve_imitation_interface(base_env)
        if not supports(interface, "publish_actor_command"):
            msg = (
                "The environment cannot accept an agent-published actor command, "
                "so a transplanted command cannot be delivered. Use "
                "--switch_command_frame robot."
            )
            raise RuntimeError(msg)
        publish_actor_command = interface.publish_actor_command
        policy_operator = agent.actor_critic.get_policy_operator()
        policy_operator.eval()
        latent_key = getattr(agent, "_latent_key", ("policy", "latent_command"))
        print(
            "[INFO] After the switch the command is the second motion encoded in "
            "its own frame, published directly. It carries no correction for the "
            "robot's pose."
        )

    # `--switch_align xy` moves the env's reference placement, and that write
    # persists: without restoring it, clip 2 would start with clip 1's shift
    # still applied and its reference would sit metres from the robot.
    canonical_expert_origins = base_env.expert_data_plane._expert_env_origins.clone()

    results: list[dict[str, object]] = []
    for index, entry in enumerate(selected):
        _force_trajectory_on_reset(
            raw_base_env, rank=entry.rank, start_step=int(args_cli.start_frame)
        )
        # Reset inside inference_mode. The env's buffers were tagged as
        # inference tensors while the previous motion rolled out under
        # inference_mode, and resetting outside that context tries to mutate
        # them in place -- which torch refuses. Only bites from the second
        # motion onward, so a single-motion smoke run never sees it.
        with torch.inference_mode():
            # Before the reset, so the clip is laid out at the canonical
            # placement rather than inheriting the previous clip's alignment
            # shift. Inside inference_mode for the same reason as the reset
            # itself: the tensor was tagged while the previous clip rolled out.
            base_env.expert_data_plane._expert_env_origins.copy_(
                canonical_expert_origins
            )
            base_env.expert_data_plane._mdp_cache_step = -1
            td = env.reset()
            # Inside inference_mode with the reset, and for the same reason:
            # the countdown was tagged an inference tensor while the previous
            # clip rolled out, and torch refuses an in-place write to one from
            # outside. Only bites from the second clip onward.
            counter = getattr(sampler, "_latent_steps", None)
            if counter is not None:
                counter.zero_()
        # Force the sampler to re-encode on this clip's FIRST step. Its per-env
        # countdown survives a reset, so without this the opening `hold_steps`
        # of every clip run on a command encoded somewhere else entirely -- the
        # previous motion's last window, or, for the first clip, whatever the
        # reference happened to be during agent construction. Measured: the
        # stale command differed from the correct one by 1.06 in z, and the
        # renewals landed at steps 9/19/29 instead of 0/10/20. Zeroing the
        # counter makes `renew_mask` fire immediately, so the clip starts at
        # phase 0 with a command encoded from its own first window.
        _set_comparison_camera(base_env)

        loaded_rank = int(base_env.trajectory_manager.env_traj_rank[POLICY_ENV_ID])
        if loaded_rank != entry.rank:
            msg = (
                f"Environment loaded rank {loaded_rank}, expected {entry.rank} "
                f"({entry.motion})."
            )
            raise RuntimeError(msg)
        print(
            f"[INFO] Comparison {index + 1}/{len(selected)}: rank={entry.rank} "
            f"motion={entry.motion!r} frames={entry.length}"
        )

        switch_target = switch_targets[index] if switch_targets is not None else None
        stem = _video_stem(entry.rank, entry.motion)
        if switch_target is not None:
            stem = (
                f"{stem}__to__{_video_stem(switch_target.rank, switch_target.motion)}"
            )
        switch_record: dict[str, object] | None = None
        window_after_switch: list[bool] = []
        if video_recorder is not None:
            video_recorder.start_recording(stem)

        metrics = _TrackingMetrics(base_env, POLICY_ENV_ID, args_cli.fall_height)
        renewal_steps: list[int] = []
        # One row per published command, so a later composite can put a cursor
        # on the exact code that was driving the robot at a given video frame.
        renewal_cursors: list[int] = []
        # The deployed command (policy lane, robot frame) and the motion's own
        # code (reference lane, reference frame). Same window, two frames.
        renewal_codes: list[object] = []
        renewal_latents: list[object] = []
        reference_codes: list[object] = []
        reference_latents: list[object] = []
        reference_offsets: list[float] = []
        # The ordinary deployment view, kept even when it is not what was sent,
        # so "what we published" and "what deployment would have published" can
        # be compared directly.
        robot_codes: list[object] = []
        robot_latents: list[object] = []
        window_command_frame: list[str] = []
        # Own countdown for the transplanted phase; the sampler's is unused then.
        held = 0
        transplant_z = None
        stopped_because = "step_cap"
        timestep = 0

        def _record_command(step: int, *, source_env: int, expected_z) -> None:
            """Record the command currently published, with its window.

            Called immediately after the policy call and before ``env.step``, so
            the reference cursor is still the one the command was encoded from.

            ``source_env`` says which lane the PUBLISHED command came from: the
            policy lane for the ordinary robot-frame command, the replay lane
            for a transplanted one. Both encodings are stored either way. The
            recorded latent is checked against ``expected_z`` -- the frozen
            sampler's own buffer, or the tensor this script published -- so a
            recorded code is never a plausible re-derivation of something else.
            """
            renewal_steps.append(int(step))
            renewal_cursors.append(
                int(base_env.trajectory_manager.env_step[POLICY_ENV_ID])
            )
            codes, latents, state = _encode_macro_windows(bundle, base_env)

            encoded_z = latents[source_env]
            if expected_z is not None:
                published_z = expected_z.detach().cpu().to(dtype=encoded_z.dtype)
                difference = float((encoded_z - published_z).abs().max())
                if difference > 1.0e-5:
                    msg = (
                        f"Recorded code does not reproduce the published command "
                        f"at step {step} of {entry.motion}: max difference "
                        f"{difference:.3e}. Refusing to record a code the robot "
                        "was not driven by."
                    )
                    raise RuntimeError(msg)
            if codes is not None:
                renewal_codes.append(codes[source_env].numpy())
                robot_codes.append(codes[POLICY_ENV_ID].numpy())
            renewal_latents.append(encoded_z.numpy())
            robot_latents.append(latents[POLICY_ENV_ID].numpy())
            window_command_frame.append(
                "reference" if source_env == REFERENCE_ENV_ID else "robot"
            )
            window_after_switch.append(
                switch_target is not None and int(step) >= int(switch_step)
            )

            # 2. The motion's own code. The replay lane's robot is the expert
            # articulation, so the window's first frame sits at its origin: the
            # anchor offset must be ~0, or this lane is not on the reference and
            # its "reference frame" would be a fiction.
            offset = float(state[REFERENCE_ENV_ID, anchor_start:anchor_end].norm())
            if offset > REFERENCE_LANE_TOLERANCE_M:
                msg = (
                    f"Reference lane is {offset * 1000:.1f} mm from the expert "
                    f"anchor at step {step} of {entry.motion}, above the "
                    f"{REFERENCE_LANE_TOLERANCE_M * 1000:.0f} mm limit. Its window "
                    "is not expressed in the reference's own frame."
                )
                raise RuntimeError(msg)
            reference_offsets.append(offset)
            if codes is not None:
                reference_codes.append(codes[REFERENCE_ENV_ID].numpy())
            reference_latents.append(latents[REFERENCE_ENV_ID].numpy())

        while simulation_app.is_running() and timestep < step_cap:
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                if switch_target is not None and timestep == int(switch_step):
                    # Before the policy call, so the command published on this
                    # step is already encoded from the new motion.
                    switch_record = _retarget_reference(
                        base_env,
                        env_ids=[POLICY_ENV_ID, REFERENCE_ENV_ID],
                        rank=switch_target.rank,
                        start_frame=int(args_cli.switch_start_frame),
                        align=str(args_cli.switch_align),
                        anchor_slice=(anchor_start, anchor_end),
                        horizon_steps=bundle.horizon_steps,
                    )
                    # Force a renewal now, or the tracker would hold the old
                    # motion's command for up to `hold_steps` more steps and the
                    # switch would be smeared across a window boundary.
                    countdown = getattr(sampler, "_latent_steps", None)
                    if countdown is not None:
                        countdown.zero_()
                    loaded = int(
                        base_env.trajectory_manager.env_traj_rank[POLICY_ENV_ID]
                    )
                    if loaded != switch_target.rank:
                        msg = (
                            f"Retarget failed: lane is on rank {loaded}, expected "
                            f"{switch_target.rank} ({switch_target.motion})."
                        )
                        raise RuntimeError(msg)
                    print(
                        f"[INFO] step {timestep}: switched to {switch_target.motion!r} "
                        f"(rank {switch_target.rank}, frame "
                        f"{int(args_cli.switch_start_frame)}); anchor gap "
                        f"{switch_record['anchor_gap_before_m']:.3f} m -> "
                        f"{switch_record['anchor_gap_after_m']:.3f} m"
                    )
                transplanting = (
                    transplant
                    and switch_target is not None
                    and timestep >= int(switch_step)
                )
                if transplanting:
                    # The second motion in its own frame, held for `hold_steps`
                    # and renewed on the same cadence the tracker was trained
                    # with. Only the phase moves in between.
                    phase_index = held % hold_steps
                    if phase_index == 0:
                        _, fresh, _ = _encode_macro_windows(bundle, base_env)
                        transplant_z = (
                            fresh[REFERENCE_ENV_ID]
                            .to(device=sim_device, dtype=torch.float32)
                            .reshape(1, -1)
                        )
                    phase = torch.full(
                        (2,),
                        phase_index / float(hold_steps),
                        device=sim_device,
                        dtype=torch.float32,
                    )
                    command = qc.append_sin_cos_phase(
                        transplant_z.expand(2, -1).contiguous(), phase
                    )
                    latent_dim = bundle.latent_command_dim
                    td.set(latent_key, command.reshape(*td.batch_size, latent_dim))
                    publish_actor_command(command.reshape(-1, latent_dim))
                    td = policy_operator(td)
                    if phase_index == 0:
                        _record_command(
                            timestep,
                            source_env=REFERENCE_ENV_ID,
                            expected_z=transplant_z.reshape(-1),
                        )
                    held += 1
                else:
                    td = collector_policy(td)
                    # Observe the frozen sampler's own per-env countdown instead
                    # of assuming a cadence. It reloads the counter to
                    # `phase_period` on a renewal and decrements once per call,
                    # so reading exactly `phase_period - 1` right after the call
                    # means this step re-encoded the next window. Unambiguous,
                    # and it does not depend on when the sampler was first
                    # called.
                    counter = getattr(sampler, "_latent_steps", None)
                    renewed = (
                        counter is not None
                        and int(counter[POLICY_ENV_ID]) == hold_steps - 1
                    )
                    expected = (
                        sampler._codes[POLICY_ENV_ID] if sampler is not None else None
                    )
                    if timestep == 0 and not renewed:
                        # Unreachable while the countdown is zeroed above, and
                        # kept as a tripwire: it records whatever command is
                        # actually in force at step 0, and _record_command
                        # aborts if that is not an encoding of the clip's own
                        # first window. Silently leaving the opening steps out of
                        # the grid would put the cursor on the wrong column for
                        # all of them.
                        _record_command(
                            0, source_env=POLICY_ENV_ID, expected_z=expected
                        )
                    if renewed:
                        _record_command(
                            timestep, source_env=POLICY_ENV_ID, expected_z=expected
                        )
                action = td.get("action")
                if action is None:
                    raise KeyError("Collector output is missing the 'action' tensor.")
                # The reference lane is a replay, not a controlled robot.
                action[REFERENCE_ENV_ID].zero_()
                td = env.step(td)
                metrics.record()
                _set_comparison_camera(base_env)
                _update_role_markers(base_env, role_markers)
                td = step_mdp(
                    td, exclude_reward=True, exclude_done=False, exclude_action=True
                )
            timestep += 1
            if bool(base_env.current_reference_is_final_frame()[POLICY_ENV_ID].item()):
                stopped_because = "reference_finished"
                break

        if video_recorder is not None and getattr(video_recorder, "recording", False):
            video_recorder.stop_recording()

        codes_path = None
        if renewal_latents:
            codes_dir = output_dir / "codes"
            codes_dir.mkdir(parents=True, exist_ok=True)
            codes_path = codes_dir / f"{stem}.npz"
            columns = {
                "motion": np.array([entry.motion]),
                "trajectory_rank": np.array([entry.rank]),
                "video_stem": np.array([stem]),
                "latent_mode": np.array([bundle.latent_mode]),
                "hold_steps": np.array([hold_steps]),
                # Control step at which each command was published, and the
                # reference frame it was encoded from.
                "renewal_step": np.asarray(renewal_steps, dtype=np.int64),
                "local_step": np.asarray(renewal_cursors, dtype=np.int64),
                # What the tracker was actually given. Before a switch, and in
                # robot-frame mode throughout, this is the robot-frame command;
                # after a transplanted switch it is the reference-frame one.
                "latent_published": np.stack(renewal_latents).astype(np.float32),
                # The ordinary deployment view, recorded even when it was not
                # what was sent: the expert window in the ROBOT's frame, which
                # carries the tracking error.
                "latent_robot_frame": np.stack(robot_latents).astype(np.float32),
                "window_command_frame": np.array(window_command_frame),
                # The same window in the REFERENCE's own frame: a property of
                # the motion alone, and the view DiffSR pretraining samples.
                "latent_reference": np.stack(reference_latents).astype(np.float32),
                "reference_lane_offset_m": np.asarray(
                    reference_offsets, dtype=np.float32
                ),
                # Mid-rollout reference switch, if any. `switch_step` is -1 when
                # the clip ran on one motion, so a composer can test one field.
                "switch_step": np.array(
                    [int(switch_step) if switch_record is not None else -1],
                    dtype=np.int64,
                ),
                "switch_motion": np.array(
                    [switch_target.motion if switch_target is not None else ""]
                ),
                "switch_rank": np.array(
                    [switch_target.rank if switch_target is not None else -1],
                    dtype=np.int64,
                ),
                "switch_command_frame": np.array(
                    [
                        str(args_cli.switch_command_frame)
                        if switch_record is not None
                        else ""
                    ]
                ),
                "window_after_switch": np.asarray(window_after_switch, dtype=bool),
                # Steps actually rolled out, for the frame -> step mapping.
                "total_steps": np.array([timestep], dtype=np.int64),
            }
            # The three `category_*` columns and the code-space shape exist only
            # for a discrete code space. A continuous latent has no code, and the
            # `latent_*` columns above already carry what was published.
            if renewal_codes:
                columns.update(
                    groups=np.array([bundle.groups]),
                    categories=np.array([bundle.categories]),
                    category_published=np.stack(renewal_codes).astype(np.int64),
                    category_robot_frame=np.stack(robot_codes).astype(np.int64),
                    category_reference=np.stack(reference_codes).astype(np.int64),
                )
            np.savez_compressed(codes_path, **columns)

        summary = metrics.summary()
        summary.update(
            {
                "trajectory_rank": entry.rank,
                "motion": entry.motion,
                "frames": entry.length,
                "stopped_because": stopped_because,
                "published_commands": len(renewal_steps),
                "first_command_steps": renewal_steps[:6],
                "switch_step": int(switch_step) if switch_record is not None else None,
                "switch_motion": (
                    switch_target.motion if switch_target is not None else None
                ),
                "switch_command_frame": (
                    str(args_cli.switch_command_frame)
                    if switch_record is not None
                    else None
                ),
                "switch_anchor_gap_before_m": (
                    switch_record["anchor_gap_before_m"]
                    if switch_record is not None
                    else None
                ),
                "switch_anchor_gap_after_m": (
                    switch_record["anchor_gap_after_m"]
                    if switch_record is not None
                    else None
                ),
                "max_reference_lane_offset_mm": (
                    round(max(reference_offsets) * 1000.0, 3)
                    if reference_offsets
                    else None
                ),
                # Fraction of code positions that differ between the two
                # lanes. Discrete code spaces only -- None for a continuous
                # latent, which has no positions to disagree on.
                "robot_vs_reference_code_disagreement": (
                    round(
                        float(
                            (np.stack(robot_codes) != np.stack(reference_codes)).mean()
                        ),
                        4,
                    )
                    if reference_codes
                    else None
                ),
                # The same question asked continuously, and therefore reported
                # for every code space: how far the robot's own view of the
                # window sits from the motion's own view. Zero means the tracker
                # is exactly on the reference; it grows with tracking error.
                # Absolute, so it is comparable across arms only alongside the
                # latent scale recorded in provenance.
                "robot_vs_reference_latent_distance": (
                    round(
                        float(
                            np.linalg.norm(
                                np.stack(robot_latents) - np.stack(reference_latents),
                                axis=-1,
                            ).mean()
                        ),
                        5,
                    )
                    if reference_latents
                    else None
                ),
                "codes": str(codes_path) if codes_path is not None else None,
            }
        )
        results.append(summary)
        print(
            f"[INFO] {entry.motion}: {summary['steps']} steps "
            f"({stopped_because}), fell={summary['fell']}, "
            f"global MPJPE={summary['mpjpe_global_mm_prefall']} mm, "
            f"{len(renewal_steps)} commands held {hold_steps} steps each"
        )

    env.close()

    video_paths = (
        sorted(str(path.resolve()) for path in video_folder.glob("*.mp4"))
        if args_cli.video
        else []
    )
    for path in video_paths:
        qc.announce_video(path)
    if args_cli.video and not video_paths:
        print(f"[WARN] No MP4 was written under {video_folder}.")

    if results:
        fieldnames = list(results[0])
        with (output_dir / "tracking_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({key: row.get(key) for key in fieldnames})

    provenance = qc.write_provenance(
        output_dir,
        mode="reference_rollout",
        task=args_cli.task,
        seed=int(args_cli.seed),
        start_frame=int(args_cli.start_frame),
        max_steps=args_cli.max_steps,
        command_source="hl_skill",
        command_hold_steps=hold_steps,
        latent_command_dim=bundle.latent_command_dim,
        z_dim=bundle.z_dim,
        latent_mode=bundle.latent_mode,
        is_discrete=bool(bundle.is_discrete),
        groups=bundle.groups if bundle.is_discrete else None,
        categories_per_group=bundle.categories if bundle.is_discrete else None,
        code_diagnostics=qc.code_diagnostics_meaning(bundle),
        skill_checkpoint=binding["skill_checkpoint"],
        skill_checkpoint_sha256=binding["skill_checkpoint_sha256"],
        policy_checkpoint=binding["low_level_checkpoint"],
        policy_checkpoint_sha256=binding["low_level_checkpoint_sha256"],
        policy_checkpoint_step=qc.policy_checkpoint_step(args_cli.policy_checkpoint),
        switch={
            "at_step": int(switch_step) if switch_step is not None else None,
            "start_frame": int(args_cli.switch_start_frame),
            "command_frame": str(args_cli.switch_command_frame),
            "command_frame_meaning": (
                "reference: after the switch the command is the second motion "
                "encoded in its own frame and published directly, so it carries "
                "no correction for the robot's pose. robot: the ordinary "
                "deployment path, the second motion seen from the robot."
            ),
            "align": str(args_cli.switch_align),
            "align_meaning": (
                "xy: the second motion is placed so it starts at the robot's "
                "ground position; heading is not aligned. none: dataset placement."
            ),
            "pairs": (
                [
                    {"from": source.motion, "to": target.motion}
                    for source, target in zip(selected, switch_targets, strict=True)
                ]
                if switch_targets is not None
                else None
            ),
        },
        encoder_binding=binding,
        encoder_config=bundle.config.to_dict()
        if hasattr(bundle.config, "to_dict")
        else None,
        reference_arrays_dir=str(reference_arrays_dir),
        reference_arrays_manifest_sha256=qc.sha256(catalog.manifest_path),
        persist_id=qc.PERSIST_ID,
        macro_state_terms=qc.MACRO_STATE_TERMS,
        motions=[
            {
                "rank": entry.rank,
                "dataset": entry.dataset,
                "motion": entry.motion,
                "frames": entry.length,
            }
            for entry in selected
        ],
        protocol={
            "reference_env_id": REFERENCE_ENV_ID,
            "policy_env_id": POLICY_ENV_ID,
            "reference_lane": "articulation replay of the policy lane's cursor",
            "disabled_terminations": sorted(disabled_terminations),
            "full_horizon_diagnostic": not args_cli.keep_terminations,
            "domain_randomization": dr_record,
            "fall_height_m": float(args_cli.fall_height),
        },
        results=results,
        video_paths=video_paths,
    )
    print(f"[INFO] Wrote {provenance}")
    print(f"[INFO] Output root: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
