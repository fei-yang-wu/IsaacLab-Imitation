#!/usr/bin/env python3
# ruff: noqa: E402
"""Skill composability: eight robots chaining random codes, one after another.

Every 100 steps (2 s) each robot independently draws a fresh uniform code from
the prior -- for ``sonic_fsq`` that is one of the 32 lattice levels for each of
the 64 coordinates -- and holds it until the next boundary. Ten such segments
make 20 s of continuously changing intent per robot, and the eight robots draw
independent sequences, so one MP4 holds eight independent samples of the same
question: can the tracker absorb an arbitrary new skill from whatever body
state the previous one left it in?

That is the property a downstream planner depends on. A planner publishes a new
``z`` every window and never gets to choose the state it inherits, so a tracker
that only works from a rest pose is not usable even when every individual code
looks fine in isolation.

The rollout opens with ``--warmup_seconds`` of ordinary encoder-driven tracking
on one real motion, rounded down to whole command windows. Robots therefore
enter the first random code upright and moving, and a later fall is
attributable to the code sequence rather than to the spawn pose.

``z`` is frozen inside a segment; the sin/cos phase keeps cycling with period
``horizon_steps`` exactly as in training, so a segment is a whole number of
ordinary command windows and no window ever mixes two codes.

A uniformly random code is almost certainly out of distribution, so robots do
fall. With ``--reset_fallen`` (the default) a robot below ``--fall_height`` at a
segment boundary is put back on its reference pose at that boundary, which
keeps all eight lanes alive for the whole clip; every such reset is recorded
per robot and per segment so a large behavioural jump beside a reset is never
read as the code alone.

``fsq64`` only. A ``deterministic`` encoder has no code alphabet to draw
uniformly from, and no continuous analogue is invented here: the encoder is
refused at load.

Example::

    pixi run -e isaaclab python experiments/qualitative/src/qualitative_skill_composability.py \\
        --encoder_checkpoint logs/.../encoder/checkpoints/latest.pt \\
        --policy_checkpoint logs/.../models/model_step_4325179392.pt \\
        --video --output_dir outputs/.../skill_composability <shared hydra overrides>
"""

from __future__ import annotations

import argparse
import csv
import math
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
    "--num_robots",
    type=int,
    default=8,
    help="Robots in the grid. Each draws its own code sequence.",
)
parser.add_argument(
    "--segment_steps",
    type=int,
    default=100,
    help=(
        "Steps one code is held. 100 is 2 s at 50 Hz. Must be a whole multiple "
        "of the encoder's horizon_steps so a switch lands on a command-window "
        "boundary."
    ),
)
parser.add_argument(
    "--num_segments",
    type=int,
    default=10,
    help="Codes drawn per robot. 10 x 100 steps is 20 s of random codes.",
)
parser.add_argument(
    "--warmup_seconds",
    type=float,
    default=1.0,
    help=(
        "Ordinary encoder-driven tracking before the first random code, rounded "
        "DOWN to whole command windows. 0 disables the prefix, which starts "
        "every robot from the reset pose instead."
    ),
)
parser.add_argument(
    "--fall_height",
    type=float,
    default=0.4,
    help="Base height (metres) at or below which a robot counts as fallen.",
)
reset_group = parser.add_mutually_exclusive_group()
reset_group.add_argument(
    "--reset_fallen",
    dest="reset_fallen",
    action="store_true",
    default=True,
    help="Put a fallen robot back on its reference pose at the next boundary.",
)
reset_group.add_argument(
    "--no_reset_fallen",
    dest="reset_fallen",
    action="store_false",
    help="Leave a fallen robot down; its remaining segments are commanded anyway.",
)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument(
    "--motion",
    type=str,
    default=None,
    help="Warmup motion every robot starts from. Defaults to a seeded draw.",
)
parser.add_argument(
    "--rank", type=int, default=None, help="Trajectory rank; overrides --motion."
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Metres between robots on screen. Defaults to the task's own spacing.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--njmax", type=int, default=320, help="Newton contact limit for the grid."
)
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


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Heading angle from a ``[N, 4]`` ``(w, x, y, z)`` quaternion."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _refresh_observations(base_env, td) -> list[str]:
    """Overwrite the tensordict's observations with the env's current ones.

    Needed after a hand-driven partial reset. ``env.step`` recomputes the
    observation at the END of a step, so without this the next
    ``policy_operator(td)`` would act on the pre-reset body state of a robot
    that has just been put back on its feet.

    Only keys the tensordict already carries are written, so an observation
    group the policy never reads cannot be introduced here by accident.
    """
    plane = getattr(base_env, "expert_data_plane", None)
    if plane is not None and hasattr(plane, "_mdp_cache_step"):
        # The cache is keyed on the step counter, which the reset did not
        # advance; stale reference-derived terms would survive it.
        plane._mdp_cache_step = -1
    observations = base_env.observation_manager.compute()
    existing = set(td.keys(include_nested=True, leaves_only=True))
    refreshed: list[str] = []
    for group_name in observations.keys():
        group = observations[group_name]
        if isinstance(group, torch.Tensor):
            if group_name in existing:
                td.set(group_name, group.reshape(td.get(group_name).shape))
                refreshed.append(str(group_name))
            continue
        for key in group.keys():
            destination = (group_name, key)
            if destination in existing:
                td.set(destination, group[key].reshape(td.get(destination).shape))
                refreshed.append(f"{group_name}/{key}")
    return refreshed


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
        raise ValueError("--num_robots must be >= 2; the grid is the comparison.")
    num_segments = int(args_cli.num_segments)
    if num_segments < 2:
        raise ValueError(
            "--num_segments must be >= 2; a single code is the intervention "
            "experiment, not a composability one."
        )
    segment_steps = int(args_cli.segment_steps)
    if segment_steps < 1:
        raise ValueError("--segment_steps must be >= 1.")

    device = torch.device(args_cli.device or "cuda:0")
    # Pinned: drawing "one category per group" needs a code alphabet. A
    # deterministic encoder is refused at load rather than part-way through a
    # render, and no continuous analogue is invented for it.
    bundle = qc.load_skill_encoder(
        args_cli.encoder_checkpoint,
        device,
        require_latent_mode=qc.DISCRETE_LATENT_MODES,
    )
    binding = qc.assert_encoder_binding(
        bundle.checkpoint_path, args_cli.policy_checkpoint
    )
    print(
        f"[PASS] encoder binding: {binding['skill_checkpoint_sha256'][:16]}... "
        f"embedded in {Path(args_cli.policy_checkpoint).name}"
    )

    phase_period = bundle.horizon_steps
    latent_dim = bundle.latent_command_dim
    # Validated here, before Isaac builds a scene: a bad --segment_steps is
    # knowable from the encoder alone.
    qc.plan_code_schedule(
        warmup_seconds=float(args_cli.warmup_seconds),
        step_dt=0.02,
        phase_period=phase_period,
        segment_steps=segment_steps,
        num_segments=num_segments,
    )

    # --- draw every code up front ----------------------------------------- #
    # One draw for the whole run, so the sequence is a property of --seed and
    # not of how the rollout happened to unfold. Rows are independent: robot i
    # and robot j never share a segment's code except by chance.
    generator = qc.make_generator(int(args_cli.seed))
    codes = qc.sample_random_codes(
        bundle, num_robots * num_segments, generator
    ).reshape(num_robots, num_segments, bundle.groups)
    jumps = torch.zeros(num_robots, num_segments, dtype=torch.float32)
    jumps_max = torch.zeros(num_robots, num_segments, dtype=torch.float32)
    changed_fraction = torch.zeros(num_robots, num_segments, dtype=torch.float32)
    for robot in range(num_robots):
        distance = qc.lattice_distance(codes[robot])
        jumps[robot, 1:] = distance["mean"]
        jumps_max[robot, 1:] = distance["max"]
        changed_fraction[robot, 1:] = distance["changed"]
    print(
        f"[INFO] Drew {num_robots} x {num_segments} codes over "
        f"{bundle.groups} {bundle.group_noun}s x {bundle.categories} "
        f"{bundle.category_noun}s. Mean move between consecutive codes: "
        f"{float(jumps[:, 1:].mean()):.2f} {bundle.category_noun}s."
    )

    # --- pick the shared warmup motion ------------------------------------- #
    catalog = qc.MotionCatalog.from_reference_arrays(reference_arrays_dir)
    step_dt_guess = 0.02
    warmup_steps_guess = int(round(float(args_cli.warmup_seconds) / step_dt_guess))
    min_length = int(args_cli.start_frame) + warmup_steps_guess + phase_period + 1
    if args_cli.rank is not None:
        entry = catalog.by_rank(int(args_cli.rank))
    elif args_cli.motion is not None:
        entry = catalog.by_rank(catalog.rank_for_motion(args_cli.motion))
    else:
        entry = catalog.select(count=1, seed=int(args_cli.seed), min_length=min_length)[
            0
        ]
    print(
        f"[INFO] Warmup motion: rank={entry.rank} motion={entry.motion!r} "
        f"frames={entry.length} start_frame={args_cli.start_frame}"
    )

    output_root = qc.prepare_output_dir(
        args_cli.output_dir, overwrite=args_cli.overwrite
    )

    # --- env / agent config ------------------------------------------------ #
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

    # `random` keeps the agent from constructing a frozen encoder sampler; we
    # publish the command ourselves and never call the random sampler.
    agent_cfg.ipmd.use_latent_command = True
    agent_cfg.ipmd.command_source = "random"
    agent_cfg.ipmd.latent_dim = latent_dim
    agent_cfg.ipmd.latent_steps_min = phase_period
    agent_cfg.ipmd.latent_steps_max = phase_period
    agent_cfg.ipmd.latent_learning.command_phase_mode = "sin_cos"
    agent_cfg.ipmd.latent_learning.code_period = phase_period
    agent_cfg.ipmd.latent_learning.code_latent_dim = bundle.z_dim
    if hasattr(env_cfg, "latent_command_dim"):
        env_cfg.latent_command_dim = latent_dim
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

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
    print(
        "[INFO] Disabled domain randomization: "
        f"events={dr_record.get('events_disabled', [])}, "
        f"reset_ranges_zeroed={list(dr_record.get('reset_ranges_zeroed', {}))}"
    )

    video_folder = output_root / "videos"
    env_cfg.log_dir = str(output_root)
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")

    # Provisional; the real total needs step_dt, which needs the built env.
    provisional_total = warmup_steps_guess + num_segments * segment_steps
    video_recorder = None
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            # One clip for the whole run, started and stopped by hand.
            step_trigger=lambda _step: False,
            video_length=provisional_total + 2,
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
            RewardSum(),
            StepCounter(provisional_total + 1),
            RewardClipping(-10.0, 5.0),
        ),
    )
    base_env = qr.unwrap_imitation_env(env)

    # The environment is the single source of truth for where rollout tensors
    # live. `args_cli.device` is what we ASKED for; under a restricted
    # CUDA_VISIBLE_DEVICES the env can land elsewhere, and mixing the two
    # raises "Expected all tensors to be on the same device".
    sim_device = torch.device(str(base_env.device))
    if sim_device != device:
        print(
            f"[INFO] Encoder loaded on {device}; environment is on {sim_device}. "
            "Rollout tensors follow the environment."
        )

    qr.pin_single_rank_on_reset(base_env, entry.rank, int(args_cli.start_frame))

    # Publish through the capability surface, not a wrapper attribute. v2's
    # ImitationRLEnv has no `set_agent_latent_command` -- that is the legacy
    # env's name -- and TransformedEnv does not forward unknown attributes.
    interface = resolve_imitation_interface(base_env)
    if not supports(interface, "publish_actor_command"):
        msg = (
            "The environment cannot accept an agent-published actor command. "
            "This mode requires the latent actor channel "
            f"(env.command_interface.actor.dim={latent_dim})."
        )
        raise RuntimeError(msg)
    publish_actor_command = interface.publish_actor_command

    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    loaded = qr.load_policy_weights(
        agent, Path(args_cli.policy_checkpoint).expanduser().resolve(), device
    )
    print(f"[INFO] Loaded network weights: {loaded}")
    policy_operator = agent.actor_critic.get_policy_operator()
    policy_operator.eval()
    latent_key = getattr(agent, "_latent_key", ("policy", "latent_command"))

    step_dt = float(getattr(base_env, "step_dt", 0.0) or 0.0)
    if step_dt <= 0.0:
        raise RuntimeError("Could not read env step_dt; cannot size the schedule.")
    schedule = qc.plan_code_schedule(
        warmup_seconds=float(args_cli.warmup_seconds),
        step_dt=step_dt,
        phase_period=phase_period,
        segment_steps=segment_steps,
        num_segments=num_segments,
    )
    warmup_steps = int(schedule["warmup_steps"])
    total_steps = int(schedule["total_steps"])
    switch_steps = list(schedule["switch_steps"])
    print(
        f"[INFO] Schedule at dt={step_dt:.4f}s: {warmup_steps} warmup steps "
        f"({warmup_steps * step_dt:.2f}s), then {num_segments} x {segment_steps} "
        f"steps ({num_segments * segment_steps * step_dt:.2f}s of random codes), "
        f"{total_steps} steps total. Switches at {switch_steps}."
    )
    if entry.length < total_steps:
        print(
            f"[INFO] The warmup motion is {entry.length} frames, shorter than the "
            f"{total_steps}-step rollout. That is expected: after the warmup the "
            "actor is driven only by the published latent command, and the "
            "expert data plane clamps the reference cursor at the trajectory "
            "boundary."
        )

    latents = (
        qc.code_to_z(bundle, codes.reshape(-1, bundle.groups))
        .reshape(num_robots, num_segments, bundle.z_dim)
        .to(device=sim_device, dtype=torch.float32)
    )

    # --- rollout ----------------------------------------------------------- #
    with torch.inference_mode():
        td = env.reset()
    qr.set_grid_camera(base_env, num_robots)
    clip_stem = "skill_composability"
    if video_recorder is not None:
        # The warmup is recorded too: the clip then opens with eight robots
        # tracking one motion together, and the moment they fan out.
        video_recorder.start_recording(clip_stem)

    # --- warmup: ordinary encoder-driven tracking -------------------------- #
    warm_z = None
    for step in range(warmup_steps):
        phase_index = step % phase_period
        if phase_index == 0:
            warm_z = qr.encode_live_window(bundle, base_env, sim_device)
        phase = torch.full(
            (num_robots,),
            phase_index / float(phase_period),
            device=sim_device,
            dtype=torch.float32,
        )
        command = qc.append_sin_cos_phase(warm_z, phase)
        td.set(latent_key, command.reshape(*td.batch_size, latent_dim))
        publish_actor_command(command.reshape(-1, latent_dim))
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = policy_operator(td)
            td = env.step(td)
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )
        qr.set_grid_camera(base_env, num_robots)

    # --- the segments ------------------------------------------------------ #
    rows: list[dict[str, object]] = []
    reset_at_boundary = torch.zeros(num_robots, num_segments, dtype=torch.bool)
    root_track: list[np.ndarray] = []
    fall_height = float(args_cli.fall_height)

    for segment in range(num_segments):
        # 1. Recover anyone who is down, at the boundary and nowhere else.
        # Read inside inference_mode: from the first env.step onward the robot
        # buffers are inference tensors, and torch refuses to mix them with
        # ordinary ones outside that context.
        refreshed: list[str] = []
        with torch.inference_mode():
            heights = base_env.robot.data.root_pos_w.torch[:num_robots, 2]
            fallen = [int(i) for i in (heights <= fall_height).nonzero().reshape(-1)]
            if args_cli.reset_fallen and fallen:
                env_ids = torch.tensor(fallen, dtype=torch.long, device=sim_device)
                # TorchRL cannot do this: IsaacLabWrapper derives from
                # GymWrapper, whose _reset performs a FULL env.reset() and
                # ignores a partial `_reset` mask, and the imitation interface
                # exposes no reset capability. The env's own per-env path is
                # the only route, and `qr.pin_single_rank_on_reset` makes it
                # re-pin the same motion at the same frame.
                base_env._reset_idx(env_ids)
                refreshed = _refresh_observations(base_env, td)
        if args_cli.reset_fallen and fallen:
            reset_at_boundary[fallen, segment] = True
            print(
                f"[INFO] Segment {segment}: reset {len(fallen)} fallen robot(s) "
                f"{fallen} onto the reference pose "
                f"({len(refreshed)} observation keys refreshed)."
            )
        elif fallen:
            print(
                f"[INFO] Segment {segment}: {len(fallen)} robot(s) are down and "
                f"stay down (--no_reset_fallen): {fallen}"
            )

        # 2. Hold this segment's code for the whole segment.
        segment_z = latents[:, segment]
        with torch.inference_mode():
            start_root = base_env.robot.data.root_pos_w.torch[:num_robots].clone()
            start_yaw = _yaw_from_quat(
                base_env.robot.data.root_quat_w.torch[:num_robots]
            ).clone()
            start_joints = base_env.robot.data.joint_pos.torch[:num_robots].clone()
            previous_joints = start_joints.clone()
            min_height = start_root[:, 2].clone()
        action_norms = torch.zeros(num_robots, device=sim_device)
        joint_path = torch.zeros(num_robots, device=sim_device)

        for step in range(segment_steps):
            phase_index = step % phase_period
            phase = torch.full(
                (num_robots,),
                phase_index / float(phase_period),
                device=sim_device,
                dtype=torch.float32,
            )
            command = qc.append_sin_cos_phase(segment_z, phase)
            td.set(latent_key, command.reshape(*td.batch_size, latent_dim))
            publish_actor_command(command.reshape(-1, latent_dim))
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                td = policy_operator(td)
                action = td.get("action")
                action_norms += action.reshape(num_robots, -1).norm(dim=-1)
                td = env.step(td)
                td = step_mdp(
                    td, exclude_reward=True, exclude_done=False, exclude_action=True
                )
                # Accumulated inside the same context the step ran in: the env
                # buffers are inference tensors from here on, and mixing them
                # with ordinary tensors outside it is what makes torch refuse.
                joints = base_env.robot.data.joint_pos.torch[:num_robots]
                joint_path += (joints - previous_joints).norm(dim=-1)
                previous_joints = joints.clone()
                root = base_env.robot.data.root_pos_w.torch[:num_robots]
                min_height = torch.minimum(min_height, root[:, 2])
                root_track.append(root.detach().cpu().numpy().copy())
            qr.set_grid_camera(base_env, num_robots)

        with torch.inference_mode():
            end_root = base_env.robot.data.root_pos_w.torch[:num_robots].clone()
            end_yaw = _yaw_from_quat(base_env.robot.data.root_quat_w.torch[:num_robots])
            end_joints = base_env.robot.data.joint_pos.torch[:num_robots]
            delta_root = (end_root - start_root).detach().cpu()
            delta_yaw = _wrap_to_pi(end_yaw - start_yaw).detach().cpu()
            joint_delta = (end_joints - start_joints).norm(dim=-1).detach().cpu()
            final_height = end_root[:, 2].detach().cpu()
            min_height_cpu = min_height.detach().cpu()
            mean_action = (action_norms / float(segment_steps)).detach().cpu()
            joint_path_cpu = joint_path.detach().cpu()
        for robot in range(num_robots):
            rows.append(
                {
                    "robot": robot,
                    "segment": segment,
                    "start_step": switch_steps[segment],
                    "reset_at_boundary": bool(reset_at_boundary[robot, segment]),
                    "code_move_mean": float(jumps[robot, segment]),
                    "code_move_max": float(jumps_max[robot, segment]),
                    "code_changed_fraction": float(changed_fraction[robot, segment]),
                    "dx_m": float(delta_root[robot, 0]),
                    "dy_m": float(delta_root[robot, 1]),
                    "dz_m": float(delta_root[robot, 2]),
                    "dyaw_rad": float(delta_yaw[robot]),
                    "joint_delta_norm_rad": float(joint_delta[robot]),
                    "joint_path_rad": float(joint_path_cpu[robot]),
                    "mean_action_norm": float(mean_action[robot]),
                    "min_base_height_m": float(min_height_cpu[robot]),
                    "final_base_height_m": float(final_height[robot]),
                    "upright": bool(float(final_height[robot]) > fall_height),
                }
            )

    if video_recorder is not None and getattr(video_recorder, "recording", False):
        video_recorder.stop_recording()

    # --- artifacts --------------------------------------------------------- #
    fieldnames = list(rows[0].keys())
    with (output_root / "segment_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    np.savez_compressed(
        output_root / "codes.npz",
        level=codes.cpu().numpy().astype(np.int64),
        latent=latents.detach().cpu().numpy().astype(np.float32),
        switch_step=np.asarray(switch_steps, dtype=np.int64),
        code_move_mean=jumps.numpy().astype(np.float32),
        code_move_max=jumps_max.numpy().astype(np.float32),
        code_changed_fraction=changed_fraction.numpy().astype(np.float32),
        reset_at_boundary=reset_at_boundary.numpy(),
        root_pos_w=np.stack(root_track).astype(np.float32),
    )

    # Colour by the OUTCOME, annotate with the command. Two independent uniform
    # draws differ by (32^2 - 1) / (3 * 32) = 10.66 levels on average with very
    # little spread, so a strip coloured by code distance would be one flat
    # colour; what varies is what the robot did with each code.
    displacement = torch.zeros(num_robots, num_segments, dtype=torch.float32)
    for row in rows:
        displacement[int(row["robot"]), int(row["segment"])] = float(
            (float(row["dx_m"]) ** 2 + float(row["dy_m"]) ** 2) ** 0.5
        )
    timeline = qc.plot_code_timeline(
        values=displacement,
        annotations=jumps,
        switch_steps=switch_steps,
        output_path=output_root / "code_timeline.png",
        value_label="horizontal root displacement over the segment (m)",
        annotation_label=f"mean |d {bundle.category_noun}| from the previous code",
        title=(
            f"skill composability: {num_robots} robots x {num_segments} codes, "
            f"{segment_steps} steps each"
        ),
        fell=reset_at_boundary,
    )
    print(f"[PLOT] {timeline}")
    for robot in range(num_robots):
        path = qc.plot_codebook_selection(
            categories=codes[robot],
            local_steps=switch_steps,
            motion=f"robot {robot}: {num_segments} random codes",
            num_categories=bundle.categories,
            output_path=output_root / f"codes_robot_{robot}.png",
            group_noun=bundle.group_noun,
            category_noun=bundle.category_noun,
            ordered_categories=bundle.is_fsq,
        )
        print(f"[PLOT] {path}")

    video_paths = sorted(
        str(path.resolve()) for path in video_folder.glob(f"{clip_stem}*.mp4")
    )
    for path in video_paths:
        qc.announce_video(path)
    if args_cli.video and not video_paths:
        print(f"[WARN] No MP4 matching {clip_stem}*.mp4 under {video_folder}.")

    upright_final = sum(
        1 for row in rows if row["segment"] == num_segments - 1 and row["upright"]
    )
    total_resets = int(reset_at_boundary.sum())
    print(
        f"[INFO] {upright_final}/{num_robots} robots upright at the end; "
        f"{total_resets} boundary resets over {num_robots * num_segments} segments."
    )

    qc.write_provenance(
        output_root,
        mode="skill_composability",
        task=args_cli.task,
        seed=int(args_cli.seed),
        num_robots=num_robots,
        num_segments=num_segments,
        segment_steps=segment_steps,
        segment_seconds=segment_steps * step_dt,
        warmup_command="encoder on the live macro window, renewed per window",
        schedule=schedule,
        code_source="uniform over the product code space, independently per robot",
        code_is_constant_within_segment=True,
        code_renewal_period_steps=phase_period,
        phase_mode="sin_cos",
        code_move_units=f"{bundle.category_noun}s on the ordered lattice",
        mean_code_move=float(jumps[:, 1:].mean()),
        fall_height_m=fall_height,
        reset_fallen_at_boundary=bool(args_cli.reset_fallen),
        boundary_resets=total_resets,
        upright_at_end=upright_final,
        start_frame=int(args_cli.start_frame),
        warmup_motion=entry.motion,
        warmup_trajectory_rank=entry.rank,
        warmup_motion_frames=entry.length,
        reference_cursor_outlives_motion=bool(entry.length < total_steps),
        skill_checkpoint=str(bundle.checkpoint_path),
        skill_checkpoint_sha256=binding["skill_checkpoint_sha256"],
        policy_checkpoint=binding["low_level_checkpoint"],
        policy_checkpoint_sha256=binding["low_level_checkpoint_sha256"],
        policy_checkpoint_step=qc.policy_checkpoint_step(args_cli.policy_checkpoint),
        encoder_binding=binding,
        encoder_config=bundle.config.to_dict()
        if hasattr(bundle.config, "to_dict")
        else None,
        latent_mode=bundle.latent_mode,
        is_discrete=bundle.is_discrete,
        fsq_levels=list(bundle.levels) if bundle.levels is not None else None,
        code_diagnostics=qc.code_diagnostics_meaning(bundle),
        groups=bundle.groups,
        categories_per_group=bundle.categories,
        code_dim=bundle.code_dim,
        z_dim=bundle.z_dim,
        latent_command_dim=latent_dim,
        reference_arrays_dir=str(reference_arrays_dir),
        reference_arrays_manifest_sha256=qc.sha256(catalog.manifest_path),
        persist_id=qc.PERSIST_ID,
        macro_state_terms=qc.MACRO_STATE_TERMS,
        video_paths=video_paths,
        protocol={
            "disabled_terminations": sorted(disabled_terminations),
            "domain_randomization": dr_record,
            "njmax": int(args_cli.njmax),
            "nconmax": int(args_cli.nconmax),
            "env_spacing": args_cli.env_spacing,
            "command_injection": "raw policy operator + publish_actor_command",
            "recovery": (
                "base_env._reset_idx at a segment boundary, observations "
                "refreshed into the tensordict"
                if args_cli.reset_fallen
                else "none; a fallen robot stays down"
            ),
        },
    )

    env.close()
    print(f"[INFO] Output root: {output_root}")


if __name__ == "__main__":
    main()
    simulation_app.close()
