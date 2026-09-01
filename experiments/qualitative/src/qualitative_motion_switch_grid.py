#!/usr/bin/env python3
# ruff: noqa: E402
"""Eight robots, eight motions, one shared switch, one video.

Each robot in the grid tracks a DIFFERENT reference motion. At
``--switch_at_step`` every robot's reference is retargeted to the SAME motion --
by default a backward jump -- without resetting anything, so all eight carry
their own physical state into the same new intent. One MP4 of the whole grid.

The frozen encoder needs no special handling: it reads each environment's own
window, so eight environments on eight motions already produce eight different
commands, and retargeting the cursor is what changes them.

This is the grid counterpart of the two-lane
``qualitative_reference_rollout.py --switch_at_step``. It has no reference
replay lane: every environment here is a controlled robot, which is what makes
one frame hold eight of them.

Example::

    pixi run -e isaaclab python experiments/qualitative/src/qualitative_motion_switch_grid.py \\
        --encoder_checkpoint logs/.../encoder/checkpoints/latest.pt \\
        --policy_checkpoint logs/.../models/model_step_4750049280.pt \\
        --switch_motion jump_backward_004_A044 --video \\
        --output_dir outputs/.../motion_switch_grid <shared hydra overrides>
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
parser.add_argument(
    "--num_robots", type=int, default=8, help="Robots in the grid, one motion each."
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--motions",
    type=str,
    default=None,
    help="Comma-separated motion names for the robots; overrides the seeded draw.",
)
parser.add_argument(
    "--ranks", type=str, default=None, help="Comma-separated ranks; same idea."
)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument(
    "--switch_at_step",
    type=int,
    default=200,
    help="Control step at which every robot is retargeted to the shared motion.",
)
parser.add_argument(
    "--switch_motion",
    type=str,
    default="jump_backward_004_A044",
    help="The motion every robot switches to.",
)
parser.add_argument(
    "--switch_rank", type=int, default=None, help="Switch target by rank instead."
)
parser.add_argument("--switch_start_frame", type=int, default=0)
parser.add_argument(
    "--switch_align",
    type=str,
    default="xy",
    choices=["xy", "none"],
    help=(
        "xy: place the shared motion so it starts at each robot's own ground "
        "position, so the switch is a change of motion rather than a teleport. "
        "none: leave the dataset placement. Both record the measured jump."
    ),
)
parser.add_argument(
    "--switch_command_frame",
    type=str,
    default="reference",
    choices=["reference", "robot"],
    help=(
        "After the switch, which frame the published command is encoded in. "
        "reference: the shared motion in ITS OWN frame, so every robot receives "
        "the SAME command and it carries no correction for where that robot is. "
        "robot: the ordinary deployment path, each robot's own view."
    ),
)
parser.add_argument(
    "--after_steps",
    type=int,
    default=150,
    help="Control steps to run after the switch.",
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Metres between environments. Wider keeps walking robots apart on screen.",
)
parser.add_argument("--njmax", type=int, default=320)
parser.add_argument("--nconmax", type=int, default=40)
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
import numpy as np
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils import math as math_utils
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


def _pin_ranks_on_reset(base_env, ranks: list[int], start_step: int) -> None:
    """Give env ``i`` motion ``ranks[i]``, always starting at ``start_step``.

    Both halves are required: the trajectory manager's callback picks the rank,
    and the v2 reference command term's own selection sampler must be rebuilt as
    fixed, or the SONIC training default resamples rank and frame jointly and
    silently overrides the callback.
    """
    rank_tensor = torch.as_tensor(ranks, dtype=torch.long)
    tm = base_env.trajectory_manager

    def _custom_reset_fn(env_ids: torch.Tensor, _num_trajectories: int) -> torch.Tensor:
        return rank_tensor.to(device=env_ids.device).index_select(
            0, env_ids.to(dtype=torch.long)
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


@torch.no_grad()
def _anchor_gaps(base_env, bundle, anchor_slice: tuple[int, int], num_envs: int):
    """Per-env world vector from each robot's anchor to its expert anchor."""
    plane = base_env.expert_data_plane
    plane._mdp_cache_step = -1
    batch = base_env.current_expert_macro_transition_batch(
        horizon_steps=bundle.horizon_steps
    )["hl"]
    start, end = anchor_slice
    offset_b = batch["state"][:num_envs, start:end].detach()
    quat = base_env.robot.data.root_quat_w.torch.detach()[:num_envs]
    return math_utils.quat_apply(quat, offset_b.to(quat.device))


@hydra_task_config(args_cli.task, qc.AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
) -> None:
    reference_arrays_dir = Path(
        args_cli.reference_arrays_dir
        or (qc.repo_root() / qc.DEFAULT_REFERENCE_ARRAYS_DIRNAME)
    ).resolve()
    num_robots = int(args_cli.num_robots)
    if num_robots < 2:
        raise ValueError("--num_robots must be >= 2 to be a grid.")
    switch_step = int(args_cli.switch_at_step)
    if switch_step <= 0:
        raise ValueError("--switch_at_step must be > 0; the first motion must play.")
    total_steps = switch_step + int(args_cli.after_steps)

    device = torch.device(args_cli.device or "cuda:0")
    bundle = qc.load_skill_encoder(args_cli.encoder_checkpoint, device)
    binding = qc.assert_encoder_binding(
        bundle.checkpoint_path, args_cli.policy_checkpoint
    )
    print(
        f"[PASS] encoder binding: {binding['skill_checkpoint_sha256'][:16]}... "
        f"embedded in {Path(args_cli.policy_checkpoint).name}"
    )

    hold_steps = int(agent_cfg.ipmd.latent_steps_min)
    if hold_steps != int(agent_cfg.ipmd.latent_steps_max):
        raise ValueError("This playback needs a fixed command hold.")
    if hold_steps != bundle.horizon_steps:
        msg = (
            f"Command hold ({hold_steps}) does not match the encoder horizon "
            f"({bundle.horizon_steps})."
        )
        raise ValueError(msg)
    if str(agent_cfg.ipmd.command_source) != "hl_skill":
        raise ValueError("Pass agent.ipmd.command_source=hl_skill.")

    # --- motions ---------------------------------------------------------- #
    catalog = qc.MotionCatalog.from_reference_arrays(reference_arrays_dir)
    # Every robot must still be inside its own motion when the switch lands, or
    # its reference would already be clamped at the final frame and the "before"
    # half would not be the motion it is labelled with.
    min_length = switch_step + 1
    selected = catalog.select(
        count=num_robots,
        seed=int(args_cli.seed),
        min_length=min_length,
        ranks=qr.parse_int_list(args_cli.ranks),
        motions=qr.parse_str_list(args_cli.motions),
    )
    if len(selected) != num_robots:
        msg = f"Selected {len(selected)} motions for {num_robots} robots."
        raise ValueError(msg)

    if args_cli.switch_rank is not None:
        target = catalog.by_rank(int(args_cli.switch_rank))
    else:
        target = catalog.by_rank(catalog.rank_for_motion(args_cli.switch_motion))
    if target.length < int(args_cli.switch_start_frame) + bundle.horizon_steps + 1:
        msg = (
            f"Switch target {target.motion} has no complete window at frame "
            f"{int(args_cli.switch_start_frame)} ({target.length} frames)."
        )
        raise ValueError(msg)

    ranks = [entry.rank for entry in selected]
    print(f"[INFO] {num_robots} robots, one motion each (seed {args_cli.seed}):")
    for slot, entry in enumerate(selected):
        print(
            f"       env {slot}: rank={entry.rank:6d} frames={entry.length:5d} {entry.motion}"
        )
    print(
        f"[INFO] At step {switch_step} every robot switches to "
        f"{target.motion!r} (rank {target.rank}, {target.length} frames) at frame "
        f"{int(args_cli.switch_start_frame)}, align={args_cli.switch_align}."
    )
    print(
        f"[INFO] Rollout: {switch_step} + {int(args_cli.after_steps)} = {total_steps} steps."
    )

    output_dir = qc.prepare_output_dir(
        args_cli.output_dir, overwrite=args_cli.overwrite
    )
    video_folder = output_dir / "videos"

    # --- env / agent ------------------------------------------------------- #
    env_cfg.scene.num_envs = num_robots
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
    agent_cfg.env.num_envs = num_robots
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.collector.frames_per_batch *= num_robots
    logger_cfg = getattr(agent_cfg, "logger", None)
    if logger_cfg is not None:
        logger_cfg.backend = ""
        logger_cfg.video = False

    solver_cfg = getattr(getattr(env_cfg.sim, "physics", None), "solver_cfg", None)
    if solver_cfg is not None:
        # A multi-robot grid overruns the training contact limits.
        solver_cfg.njmax = int(args_cli.njmax)
        solver_cfg.nconmax = int(args_cli.nconmax)

    disabled_terminations = qr.disable_all_terminations(env_cfg)
    dr_record = disable_domain_randomization(env_cfg)
    print(f"[INFO] Disabled terminations: {sorted(disabled_terminations)}")
    env_cfg.log_dir = str(output_dir)

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")

    video_recorder = None
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            # One clip, started and stopped by hand.
            step_trigger=lambda _step: False,
            video_length=total_steps + 2,
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
            RewardSum(), StepCounter(total_steps + 1), RewardClipping(-10.0, 5.0)
        ),
    )
    base_env = qr.unwrap_imitation_env(env)

    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    loaded = qr.load_policy_weights(
        agent, Path(args_cli.policy_checkpoint).expanduser().resolve(), device
    )
    print(f"[INFO] Loaded network weights: {loaded}")
    collector_policy = agent.collector_policy
    collector_policy.eval()
    sampler = getattr(agent, "_hl_skill_command_sampler", None)
    if sampler is None:
        raise RuntimeError("The agent did not build a frozen skill sampler.")

    slices = base_env.expert_macro_feature_slices(bundle.horizon_steps)
    anchor_slice = (
        int(slices["expert_anchor_pos_b"][0]),
        int(slices["expert_anchor_pos_b"][1]),
    )
    ori_slice = (
        int(slices["expert_anchor_ori_b"][0]),
        int(slices["expert_anchor_ori_b"][1]),
    )
    print(
        f"[INFO] macro row: anchor_pos_b {anchor_slice}, anchor_ori_b {ori_slice} "
        f"of {bundle.state_dim}."
    )

    # After the switch the command can be encoded in the shared motion's OWN
    # frame. The frozen sampler cannot produce that -- it always reads the
    # window relative to the live robot -- so those steps are published by hand,
    # through the same capability surface RLOpt's own controller uses.
    transplant = str(args_cli.switch_command_frame) == "reference"
    publish_actor_command = None
    policy_operator = None
    latent_key = ("policy", "latent_command")
    sim_device = torch.device(str(base_env.device))
    if transplant:
        interface = resolve_imitation_interface(base_env)
        if not supports(interface, "publish_actor_command"):
            msg = (
                "The environment cannot accept an agent-published actor command; "
                "use --switch_command_frame robot."
            )
            raise RuntimeError(msg)
        publish_actor_command = interface.publish_actor_command
        policy_operator = agent.actor_critic.get_policy_operator()
        policy_operator.eval()
        latent_key = getattr(agent, "_latent_key", ("policy", "latent_command"))
        print(
            "[INFO] After the switch every robot receives the SAME command: the "
            "shared motion encoded in its own frame, carrying no correction for "
            "the robot's pose."
        )

    _pin_ranks_on_reset(base_env, ranks, int(args_cli.start_frame))
    with torch.inference_mode():
        td = env.reset()
        # The sampler's countdown survives a reset, so without this the opening
        # steps would run on a command encoded during agent construction.
        countdown = getattr(sampler, "_latent_steps", None)
        if countdown is not None:
            countdown.zero_()
    loaded_ranks = [
        int(rank) for rank in base_env.trajectory_manager.env_traj_rank[:num_robots]
    ]
    if loaded_ranks != ranks:
        msg = f"Environments loaded {loaded_ranks}, expected {ranks}."
        raise RuntimeError(msg)
    qr.set_grid_camera(base_env, num_robots)

    if video_recorder is not None:
        video_recorder.start_recording(f"motion_switch_grid_{num_robots}robots")

    # --- rollout ----------------------------------------------------------- #
    plane = base_env.expert_data_plane
    switch_record: dict[str, object] | None = None
    code_steps: list[int] = []
    # Latents are recorded for every code space; the two `*_values` code lists
    # stay empty for a continuous latent, and their output columns are omitted
    # rather than filled with a placeholder.
    latent_values: list[object] = []
    published_latents: list[object] = []
    code_values: list[object] = []
    published_values: list[object] = []
    code_frames: list[str] = []
    root_track: list[object] = []
    held = 0
    transplant_z = None
    z_spread_across_robots: list[float] = []

    @torch.no_grad()
    def _encode_reference_frame_window():
        """Encode the live window re-expressed in the MOTION's own frame.

        The environment hands the window over relative to each robot; anchoring
        it on its own first frame cancels that transform exactly, so what the
        encoder sees is a property of the motion. Every robot shares one cursor
        after the switch, so all rows should come out identical -- the spread is
        measured rather than assumed.
        """
        batch = base_env.current_expert_macro_transition_batch(
            horizon_steps=bundle.horizon_steps
        )["hl"]
        state = batch["state"].to(device=bundle.device, dtype=torch.float32)
        future = batch["future_window"].to(device=bundle.device, dtype=torch.float32)
        window = torch.cat((state.unsqueeze(1), future), dim=1)
        anchored = qc.reanchor_window_to_first_frame(
            window, pos_slice=anchor_slice, ori_slice=ori_slice
        )
        residual = float(anchored[:, 0, anchor_slice[0] : anchor_slice[1]].abs().max())
        if residual > 1.0e-5:
            msg = (
                f"Re-anchored window still offsets its own first frame by "
                f"{residual:.3e}; the anchor slice is wrong."
            )
            raise RuntimeError(msg)
        encoded = qc.encode_windows(bundle, anchored[:, 0], anchored[:, 1:])
        # `categories` is None for a continuous latent; the caller records the
        # code column only when there is one.
        return encoded["z"], encoded.get("categories")

    for timestep in range(total_steps):
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            if timestep == switch_step:
                gaps_before = _anchor_gaps(base_env, bundle, anchor_slice, num_robots)
                base_env.trajectory_manager.set_env_cursor(
                    env_ids=torch.arange(num_robots, dtype=torch.long),
                    ranks=torch.full((num_robots,), int(target.rank), dtype=torch.long),
                    steps=int(args_cli.switch_start_frame),
                )
                plane._mdp_cache_step = -1
                gaps = _anchor_gaps(base_env, bundle, anchor_slice, num_robots)
                if args_cli.switch_align == "xy":
                    # Each robot gets its own shift: after the first motion they
                    # stand in different places, so one shared offset would put
                    # the new motion beside most of them.
                    origins = plane._expert_env_origins
                    shift = (-gaps).to(device=origins.device, dtype=origins.dtype)
                    shift[:, 2] = 0.0
                    origins[:num_robots] += shift
                    plane._mdp_cache_step = -1
                    gaps = _anchor_gaps(base_env, bundle, anchor_slice, num_robots)
                    horizontal = float(gaps[:, :2].norm(dim=-1).max())
                    if horizontal > 1.0e-3:
                        msg = (
                            f"Aligning left {horizontal * 1000:.1f} mm of horizontal "
                            "gap; it should be zero by construction."
                        )
                        raise RuntimeError(msg)
                # A renewal now, or the robots would hold the previous motion's
                # command for up to `hold_steps` more steps.
                countdown = getattr(sampler, "_latent_steps", None)
                if countdown is not None:
                    countdown.zero_()
                switch_record = {
                    "step": switch_step,
                    "rank": int(target.rank),
                    "motion": target.motion,
                    "start_frame": int(args_cli.switch_start_frame),
                    "align": str(args_cli.switch_align),
                    "anchor_gap_before_m": [
                        round(float(value), 6) for value in gaps_before.norm(dim=-1)
                    ],
                    "anchor_gap_after_m": [
                        round(float(value), 6) for value in gaps.norm(dim=-1)
                    ],
                }
                print(
                    f"[INFO] step {timestep}: all {num_robots} robots -> "
                    f"{target.motion!r}; anchor gap max "
                    f"{max(switch_record['anchor_gap_before_m']):.3f} m -> "
                    f"{max(switch_record['anchor_gap_after_m']):.3f} m"
                )

            transplanting = transplant and timestep >= switch_step
            if transplanting:
                phase_index = held % hold_steps
                if phase_index == 0:
                    latents, categories = _encode_reference_frame_window()
                    transplant_z = latents[:num_robots].to(
                        device=sim_device, dtype=torch.float32
                    )
                    # One cursor, one motion, one frame of reference: the robots
                    # should all be handed the same numbers.
                    z_spread_across_robots.append(
                        float((transplant_z - transplant_z[0:1]).abs().max())
                    )
                    code_steps.append(timestep)
                    published_latents.append(
                        transplant_z[:num_robots].cpu().numpy().astype(np.float32)
                    )
                    if categories is not None:
                        published_values.append(categories[:num_robots].cpu().numpy())
                    code_frames.append("reference")
                    # The deployment view is recorded alongside, so the two can
                    # be compared afterwards.
                    robot_batch = base_env.current_expert_macro_transition_batch(
                        horizon_steps=bundle.horizon_steps
                    )["hl"]
                    robot_encoded = qc.encode_windows(
                        bundle,
                        robot_batch["state"].to(
                            device=bundle.device, dtype=torch.float32
                        ),
                        robot_batch["future_window"].to(
                            device=bundle.device, dtype=torch.float32
                        ),
                    )
                    latent_values.append(
                        robot_encoded["z"][:num_robots].cpu().numpy().astype(np.float32)
                    )
                    robot_categories = robot_encoded.get("categories")
                    if robot_categories is not None:
                        code_values.append(robot_categories[:num_robots].cpu().numpy())
                phase = torch.full(
                    (num_robots,),
                    phase_index / float(hold_steps),
                    device=sim_device,
                    dtype=torch.float32,
                )
                latent_dim = bundle.latent_command_dim
                command = qc.append_sin_cos_phase(transplant_z, phase)
                td.set(latent_key, command.reshape(*td.batch_size, latent_dim))
                publish_actor_command(command.reshape(-1, latent_dim))
                td = policy_operator(td)
                held += 1
            else:
                td = collector_policy(td)
                counter = getattr(sampler, "_latent_steps", None)
                if counter is not None and int(counter[0]) == hold_steps - 1:
                    batch = base_env.current_expert_macro_transition_batch(
                        horizon_steps=bundle.horizon_steps
                    )["hl"]
                    encoded = qc.encode_windows(
                        bundle,
                        batch["state"].to(device=bundle.device, dtype=torch.float32),
                        batch["future_window"].to(
                            device=bundle.device, dtype=torch.float32
                        ),
                    )
                    code_steps.append(timestep)
                    renewal_z = (
                        encoded["z"][:num_robots].cpu().numpy().astype(np.float32)
                    )
                    latent_values.append(renewal_z)
                    published_latents.append(renewal_z)
                    renewal_categories = encoded.get("categories")
                    if renewal_categories is not None:
                        code = renewal_categories[:num_robots].cpu().numpy()
                        code_values.append(code)
                        published_values.append(code)
                    code_frames.append("robot")
            td = env.step(td)
            qr.set_grid_camera(base_env, num_robots)
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )
        root_track.append(
            base_env.robot.data.root_pos_w.torch[:num_robots].detach().cpu().numpy()
        )

    if video_recorder is not None and getattr(video_recorder, "recording", False):
        video_recorder.stop_recording()
    env.close()

    # --- artifacts --------------------------------------------------------- #
    heights = np.stack(root_track).astype(np.float32)[..., 2]
    with (output_dir / "robots.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "env_id",
                "rank",
                "motion",
                "frames",
                "min_root_height_m",
                "final_root_height_m",
            ]
        )
        for slot, entry in enumerate(selected):
            writer.writerow(
                [
                    slot,
                    entry.rank,
                    entry.motion,
                    entry.length,
                    f"{float(heights[:, slot].min()):.4f}",
                    f"{float(heights[-1, slot]):.4f}",
                ]
            )

    if latent_values:
        # [renewals, robots, z_dim]. `*_robot_frame` is the ordinary deployment
        # view -- each robot's own window seen from itself, so the rows differ
        # across robots. `*_published` is what was actually sent, which after a
        # reference-frame switch is the motion's own command and identical for
        # every robot.
        #
        # The latent columns are always written; the `category_*` columns exist
        # only for a discrete code space. That is what makes this analysis read
        # the same way on a continuous latent: the experiment is about what the
        # robots were commanded with, and the latent IS the command.
        columns = {
            "renewal_step": np.asarray(code_steps, dtype=np.int64),
            "latent_robot_frame": np.stack(latent_values).astype(np.float32),
            "latent_published": np.stack(published_latents).astype(np.float32),
            "window_command_frame": np.array(code_frames),
            "motions": np.array([entry.motion for entry in selected]),
            "switch_step": np.array([switch_step], dtype=np.int64),
            "latent_mode": np.array([bundle.latent_mode]),
        }
        if code_values:
            columns["category_robot_frame"] = np.stack(code_values).astype(np.int64)
            columns["category_published"] = np.stack(published_values).astype(np.int64)
        np.savez_compressed(output_dir / "codes.npz", **columns)

    video_paths = (
        sorted(str(path.resolve()) for path in video_folder.glob("*.mp4"))
        if args_cli.video
        else []
    )
    for path in video_paths:
        qc.announce_video(path)
    if args_cli.video and not video_paths:
        print(f"[WARN] No MP4 was written under {video_folder}.")

    provenance = qc.write_provenance(
        output_dir,
        mode="motion_switch_grid",
        task=args_cli.task,
        seed=int(args_cli.seed),
        num_robots=num_robots,
        start_frame=int(args_cli.start_frame),
        switch=switch_record,
        total_steps=total_steps,
        steps_before_switch=switch_step,
        steps_after_switch=int(args_cli.after_steps),
        command_hold_steps=hold_steps,
        command_frame_before_switch="robot",
        command_frame_after_switch=str(args_cli.switch_command_frame),
        command_frame_meaning=(
            "reference: after the switch the window is re-anchored on its own "
            "first frame, which cancels the robot transform exactly, so every "
            "robot receives the same motion-intrinsic command. robot: the "
            "ordinary deployment path, each robot's own view."
        ),
        max_command_spread_across_robots=(
            round(max(z_spread_across_robots), 9) if z_spread_across_robots else None
        ),
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
        encoder_binding=binding,
        reference_arrays_dir=str(reference_arrays_dir),
        reference_arrays_manifest_sha256=qc.sha256(catalog.manifest_path),
        persist_id=qc.PERSIST_ID,
        macro_state_terms=qc.MACRO_STATE_TERMS,
        robots=[
            {
                "env_id": slot,
                "rank": entry.rank,
                "motion": entry.motion,
                "frames": entry.length,
            }
            for slot, entry in enumerate(selected)
        ],
        protocol={
            "disabled_terminations": sorted(disabled_terminations),
            "domain_randomization": dr_record,
            "njmax": int(args_cli.njmax),
            "nconmax": int(args_cli.nconmax),
            "camera": "wide shot reframed each step on the live robot positions",
        },
        video_paths=video_paths,
    )
    if z_spread_across_robots:
        print(
            "[INFO] Transplanted command spread across robots: max "
            f"{max(z_spread_across_robots):.3e} (0 means every robot got the "
            "identical command)."
        )
    print(f"[INFO] Wrote {provenance}")
    print(f"[INFO] Output root: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
