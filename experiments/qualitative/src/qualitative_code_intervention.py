#!/usr/bin/env python3
# ruff: noqa: E402
"""Controlled code interventions on a discrete skill latent, 32 robots at a time.

Tasks 3 and 4 of the qualitative analysis, selected with ``--mode``:

``one_group``
    Sample a random 64-group product code, pick a group, and decode 32
    variants that differ from the base only in that group's category. The 32
    latents are then identical outside the 4 z values that group owns.

``half_groups``
    Sample a random base code and build 32 variants, each independently
    choosing 32 of the 64 groups and a fresh category for every selected
    group. Each variant differs from the base in exactly 128 of the 256 z
    values.

In both modes all 32 environments start from the same frame of the same motion
and hold their assigned ``z`` for the entire rollout. The sine/cosine phase
keeps cycling with period ``horizon_steps`` because the tracker was trained
against a cycling phase -- only ``z`` is frozen. The 32 robots in the scene grid
are the side-by-side comparison; one MP4 per run.

The command is injected directly rather than through ``agent.collector_policy``:
that wrapper carries its own random latent sampler, which would overwrite the
code under test on every step. This runs the raw policy operator and publishes
the command to both the policy tensordict and the environment.

Category ids are nominal and local to each group, and a perturbed product code
may never have occurred in training, so a behavioural difference here is
evidence about local controller response, not a semantic label.

Example::

    pixi run -e isaaclab python experiments/qualitative/src/qualitative_code_intervention.py \\
        --mode one_group --group_ids 0,8,56 \\
        --encoder_checkpoint logs/.../encoder/checkpoints/latest.pt \\
        --policy_checkpoint logs/.../models/model_step_4325179392.pt \\
        --video --output_dir outputs/.../category_intervention <shared hydra overrides>
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
parser.add_argument(
    "--mode",
    type=str,
    default="one_group",
    choices=["n_groups", "one_group", "half_groups", "across_groups"],
    help=(
        "n_groups: THE UNIFIED MODE -- one shared set of --num_groups groups, "
        "resampled to a distinct category per robot. num_groups=1 is one_group "
        "and num_groups=32 is half_groups, so a sweep over it is a single axis. "
        "one_group: one group, 32 categories across the robots. "
        "half_groups: half the groups perturbed per robot. "
        "across_groups: a different group per robot, each moved to one random "
        "other category -- one grid covering 32 groups instead of 32 categories."
    ),
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
    "--variants",
    type=int,
    default=32,
    help="Robots per video. Variant 0 is always the unperturbed base code.",
)
parser.add_argument(
    "--group_ids",
    type=str,
    default=None,
    help=(
        "one_group only: comma-separated groups to sweep, or 'all'. Defaults to "
        "--num_groups seeded picks."
    ),
)
parser.add_argument(
    "--num_sweep_groups",
    type=int,
    default=3,
    help="one_group only: how many groups to draw when --group_ids is absent.",
)
parser.add_argument(
    "--num_groups",
    type=int,
    default=1,
    help=(
        "n_groups only: how many groups are resampled, shared by every robot. "
        "The single axis of the unified sweep."
    ),
)
parser.add_argument(
    "--groups_per_robot",
    type=int,
    default=1,
    help=(
        "across_groups only: how many groups each robot perturbs. Sets are kept "
        "disjoint across robots while variants * groups_per_robot <= the group "
        "count; past that they are sampled independently and may overlap, which "
        "is logged and recorded in provenance.json."
    ),
)
parser.add_argument(
    "--groups_per_variant",
    type=int,
    default=None,
    help="half_groups only: groups perturbed per variant. Defaults to half of them.",
)
parser.add_argument("--rollout_steps", type=int, default=100)
parser.add_argument(
    "--warmup_seconds",
    type=float,
    default=1.0,
    help=(
        "Execute the motion under normal encoder control for this long before "
        "switching to the perturbed code. Puts every robot in the same real, "
        "moving state, and makes the base code an encoding of the live window "
        "rather than an out-of-distribution draw. 0 disables the prefix."
    ),
)
parser.add_argument(
    "--max_warmup_drift",
    type=float,
    default=0.5,
    help=(
        "Maximum root-position spread (metres) permitted between environments "
        "at the switch. Parallel envs are NOT bit-deterministic and humanoid "
        "contact dynamics are chaotic, so a simulated prefix always spreads them "
        "somewhat -- about 0.1 m after 1 s on the turn-walk. This is a catch for "
        "gross divergence (a robot that fell), not a determinism check; the "
        "measured spread is printed and recorded in provenance.json either way."
    ),
)
parser.add_argument(
    "--base_code",
    type=str,
    default="encoded",
    choices=["encoded", "random"],
    help=(
        "Where the unperturbed code comes from at the switch. 'encoded' reads "
        "the next macro window at the switch point, so the base is a code the "
        "encoder actually produces. 'random' draws uniformly over the product "
        "codebook, which is almost certainly out of distribution."
    ),
)
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument(
    "--motion",
    type=str,
    default=None,
    help="Motion name every robot starts from. Defaults to a seeded draw.",
)
parser.add_argument(
    "--rank", type=int, default=None, help="Trajectory rank; overrides --motion."
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--njmax", type=int, default=320, help="Newton contact limit for the 32-robot grid."
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


def _parse_group_ids(value: str | None, *, groups: int) -> list[int] | None:
    if value is None:
        return None
    if value.strip().lower() == "all":
        return list(range(groups))
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError(
            "--group_ids must be a nonempty comma-separated list or 'all'."
        )
    invalid = [g for g in parsed if not 0 <= g < groups]
    if invalid:
        raise ValueError(f"--group_ids out of range [0, {groups}): {invalid}.")
    return parsed


@hydra_task_config(args_cli.task, qc.AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
) -> None:
    reference_arrays_dir = Path(
        args_cli.reference_arrays_dir
        or (qc.repo_root() / qc.DEFAULT_REFERENCE_ARRAYS_DIRNAME)
    ).resolve()
    variants = int(args_cli.variants)
    if variants < 2:
        raise ValueError("--variants must be >= 2 to compare anything.")

    device = torch.device(args_cli.device or "cuda:0")
    # Pinned, unlike the other entrypoints: a per-group code edit has no
    # meaning without a code. A deterministic encoder is refused here, at load,
    # rather than part-way through a render.
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

    # --- build the codes -------------------------------------------------- #
    generator = qc.make_generator(int(args_cli.seed))
    # The codes cannot be built yet: under the default `encoded` base they
    # depend on the window the encoder sees at the switch point, which only
    # exists once the env has run the warmup prefix. Fix the job list here and
    # fill in each job's codes inside the rollout loop.
    if args_cli.mode == "n_groups":
        num_groups = int(args_cli.num_groups)
        jobs = [{"name": f"n{num_groups:02d}_groups", "group": None}]
        groups_per_variant = None
    elif args_cli.mode == "one_group":
        group_ids = _parse_group_ids(args_cli.group_ids, groups=bundle.groups)
        if group_ids is None:
            order = torch.randperm(bundle.groups, generator=generator)
            group_ids = sorted(int(g) for g in order[: int(args_cli.num_sweep_groups)])
        jobs = [{"name": f"group_{group:02d}", "group": group} for group in group_ids]
        groups_per_variant = None
    elif args_cli.mode == "across_groups":
        jobs = [
            {
                "name": f"across_{variants:02d}x{int(args_cli.groups_per_robot):02d}",
                "group": None,
            }
        ]
        groups_per_variant = None
    else:
        groups_per_variant = int(
            args_cli.groups_per_variant
            if args_cli.groups_per_variant is not None
            else bundle.groups // 2
        )
        jobs = [
            {
                "name": f"half_{groups_per_variant:02d}_of_{bundle.groups}",
                "group": None,
                "groups_per_variant": groups_per_variant,
            }
        ]

    # --- pick the shared start motion ------------------------------------- #
    catalog = qc.MotionCatalog.from_reference_arrays(reference_arrays_dir)
    if args_cli.rank is not None:
        entry = catalog.by_rank(int(args_cli.rank))
    elif args_cli.motion is not None:
        entry = catalog.by_rank(catalog.rank_for_motion(args_cli.motion))
    else:
        entry = catalog.select(
            count=1, seed=int(args_cli.seed), min_length=bundle.horizon_steps + 1
        )[0]
    print(
        f"[INFO] Shared start: rank={entry.rank} motion={entry.motion!r} "
        f"frames={entry.length} start_frame={args_cli.start_frame}"
    )

    output_root = qc.prepare_output_dir(
        args_cli.output_dir, overwrite=args_cli.overwrite
    )

    # --- env / agent config ----------------------------------------------- #
    env_cfg.scene.num_envs = variants
    agent_cfg.env.num_envs = variants
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.collector.frames_per_batch *= variants

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
        # A 32-robot grid overruns the training contact limits.
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

    rollout_steps = int(args_cli.rollout_steps)
    results: list[dict[str, object]] = []

    # One env and one agent for the whole sweep. Isaac Sim does not reliably
    # rebuild a scene inside a live process, and `GROUP_IDS=all` is 64 jobs, so
    # the jobs share the env and each gets its own clip via manual video
    # start/stop -- the same playlist pattern scripts/viz/compare_policy_reference.py
    # uses.
    video_folder = output_root / "videos"
    env_cfg.log_dir = str(output_root)
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
            # Clips are started and stopped by hand, one per job.
            step_trigger=lambda _step: False,
            video_length=rollout_steps + 2,
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
            StepCounter(rollout_steps + 1),
            RewardClipping(-10.0, 5.0),
        ),
    )
    base_env = qr.unwrap_imitation_env(env)

    # The environment is the single source of truth for where rollout tensors
    # live. `args_cli.device` is what we ASKED for; under a restricted
    # CUDA_VISIBLE_DEVICES the env can land elsewhere, and mixing the two
    # raises "Expected all tensors to be on the same device". Everything the
    # rollout touches is allocated on `sim_device` from here on.
    sim_device = torch.device(str(base_env.device))
    if sim_device != device:
        print(
            f"[INFO] Encoder loaded on {device}; environment is on {sim_device}. "
            "Rollout tensors follow the environment."
        )

    qr.pin_single_rank_on_reset(base_env, entry.rank, int(args_cli.start_frame))

    # Publish through the capability surface, not a wrapper attribute. v2's
    # ImitationRLEnv has no `set_agent_latent_command` -- that is the legacy
    # env's name -- and TransformedEnv does not forward unknown attributes
    # anyway. `resolve_imitation_interface` is what RLOpt's own
    # LatentCommandController uses, so this works on both env generations.
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

    # 1 second of prefix, rounded to a whole number of command windows so the
    # switch lands on a window boundary -- the encoder publishes a new z every
    # `phase_period` steps, and switching mid-window would mix two commands.
    step_dt = float(getattr(base_env, "step_dt", 0.0) or 0.0)
    if step_dt <= 0.0:
        raise RuntimeError("Could not read env step_dt; cannot size the warmup.")
    warmup_steps = int(round(float(args_cli.warmup_seconds) / step_dt))
    warmup_steps = (warmup_steps // phase_period) * phase_period
    print(
        f"[INFO] Warmup: {args_cli.warmup_seconds}s -> {warmup_steps} steps "
        f"({warmup_steps // phase_period} windows at dt={step_dt:.4f}s), then "
        f"{rollout_steps} steps holding the {args_cli.base_code} base code."
    )

    for job_index, job in enumerate(jobs):
        job_dir = output_root / str(job["name"])
        job_dir.mkdir(parents=True, exist_ok=True)

        # Reset inside inference_mode. The env's buffers were tagged as
        # inference tensors while the previous job rolled out under
        # inference_mode, and resetting outside that context tries to mutate
        # them in place -- which torch refuses. Only bites from the second job
        # onward, so a single-group smoke run never sees it.
        with torch.inference_mode():
            td = env.reset()
        qr.set_grid_camera(base_env, framing="static")
        clip_stem = f"{args_cli.mode}-{job['name']}"
        if video_recorder is not None:
            # Record the prefix too: the clip then shows 32 identical robots
            # tracking the motion, and the moment they fan out.
            video_recorder.start_recording(clip_stem)

        # --- warmup: ordinary encoder-driven tracking --------------------- #
        # Every environment is pinned to the same (rank, frame) with domain
        # randomization off, so all 32 stay identical through the prefix. That
        # is what makes the perturbed code the only difference afterwards.
        warm_z = None
        for step in range(warmup_steps):
            phase_index = step % phase_period
            if phase_index == 0:
                warm_z, _ = qr.encode_live_window(
                    bundle, base_env, sim_device, return_categories=True
                )
            phase = torch.full(
                (variants,),
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

        # --- the switch: encode the next window, then perturb it ---------- #
        _, live_categories = qr.encode_live_window(
            bundle, base_env, sim_device, return_categories=True
        )
        # Parallel environments do NOT stay bit-identical through the prefix
        # even with identical states, identical commands and randomization
        # off -- the batched solver is not deterministic across env slots, and
        # contact-rich motion amplifies that. So quantify the drift instead of
        # asserting it away: the comparison is only honest if the robots are
        # still physically together at the switch. Root position is taken
        # relative to each env's grid origin.
        origins = base_env.scene.env_origins[:variants]
        rel_root = base_env.robot.data.root_pos_w.torch[:variants] - origins
        root_spread_m = float((rel_root - rel_root[0:1]).norm(dim=-1).max())
        joint_spread_rad = float(
            (
                base_env.robot.data.joint_pos.torch[:variants]
                - base_env.robot.data.joint_pos.torch[0:1]
            )
            .abs()
            .max()
        )
        code_agreement = float(
            (live_categories == live_categories[0:1]).to(torch.float32).mean()
        )
        print(
            f"[INFO] At switch: root spread {root_spread_m * 1000:.2f} mm, "
            f"joint spread {joint_spread_rad:.4f} rad, "
            f"encoded-code agreement {code_agreement:.3f}"
        )
        if warmup_steps > 0 and root_spread_m > float(args_cli.max_warmup_drift):
            msg = (
                f"Environments drifted {root_spread_m * 1000:.1f} mm apart during "
                f"the warmup prefix, above the {float(args_cli.max_warmup_drift) * 1000:.0f} mm "
                "limit. The perturbation would no longer be the only difference "
                "between robots. Shorten --warmup_seconds or raise "
                "--max_warmup_drift deliberately."
            )
            raise RuntimeError(msg)
        if args_cli.base_code == "encoded":
            base_code = live_categories[0].clone()
        else:
            base_code = qc.sample_base_code(bundle, generator).cpu()

        if args_cli.mode == "n_groups":
            num_groups = int(args_cli.num_groups)
            codes, shared_groups = qc.perturb_shared_groups(
                bundle,
                base_code,
                variants=variants,
                num_groups=num_groups,
                generator=generator,
            )
            mask = None
            expected_changed = num_groups * bundle.code_dim
            job.update(
                {
                    "baseline_category": None,
                    "categories": None,
                    "num_groups": num_groups,
                    "shared_groups": [int(g) for g in shared_groups],
                }
            )
        elif job["group"] is not None:
            baseline = int(base_code[job["group"]])
            categories = qc.distinct_categories(
                bundle, baseline=baseline, count=variants, generator=generator
            )
            codes = qc.perturb_one_group(bundle, base_code, job["group"], categories)
            mask = None
            expected_changed = bundle.code_dim
            job.update({"baseline_category": baseline, "categories": categories})
        elif args_cli.mode == "across_groups":
            per_robot = int(args_cli.groups_per_robot)
            codes, swept_groups, disjoint = qc.perturb_distinct_groups(
                bundle,
                base_code,
                variants=variants,
                groups_per_robot=per_robot,
                generator=generator,
            )
            mask = None
            expected_changed = per_robot * bundle.code_dim
            if not disjoint:
                print(
                    f"[WARN] {variants} robots x {per_robot} groups exceeds "
                    f"{bundle.groups} groups, so the per-robot sets OVERLAP: two "
                    "robots may perturb the same group. Differences are no longer "
                    "purely differences of group identity."
                )
            job.update(
                {
                    "baseline_category": None,
                    "categories": None,
                    "groups_per_robot": per_robot,
                    "disjoint_groups": bool(disjoint),
                    "swept_groups": [
                        [int(g) for g in row] for row in swept_groups.tolist()
                    ],
                }
            )
        else:
            codes, mask = qc.perturb_group_subset(
                bundle,
                base_code,
                variants=variants,
                groups_per_variant=int(job["groups_per_variant"]),
                generator=generator,
            )
            expected_changed = int(job["groups_per_variant"]) * bundle.code_dim
            job.update({"baseline_category": None, "categories": None})
        job["codes"] = codes
        job["mask"] = mask

        latents_z = qc.code_to_z(bundle, codes).to(
            device=sim_device, dtype=torch.float32
        )

        # The interpretability claim of these modes is that only the edited
        # groups move. Assert it against the BASE latent rather than variant 0:
        # in across_groups every robot is perturbed, so variant 0 is not the
        # baseline and comparing to it would measure the wrong thing.
        base_latent = qc.code_to_z(bundle, base_code.reshape(1, -1)).to(
            device=sim_device, dtype=torch.float32
        )
        changed = (latents_z != base_latent).sum(dim=-1)
        for row in range(variants):
            if int(codes[row].ne(base_code).sum()) == 0:
                continue
            if int(changed[row]) != expected_changed:
                msg = (
                    f"Variant {row} of {job['name']} changes {int(changed[row])} of "
                    f"{bundle.z_dim} latent values, expected {expected_changed}. The "
                    "code->z map is not group-local; refusing to render a "
                    "misleading comparison."
                )
                raise RuntimeError(msg)

        # Effects are measured from the switch, not from the episode start, so
        # the shared prefix does not inflate every variant equally.
        initial_root = base_env.robot.data.root_pos_w.torch[:variants].clone()
        initial_joints = base_env.robot.data.joint_pos.torch[:variants].clone()
        action_norms = torch.zeros(variants, device=sim_device)
        root_track = []
        renewal_steps: list[int] = []

        for step in range(rollout_steps):
            # z is constant from the switch onward; only the phase advances,
            # restarting at each window boundary exactly as in training.
            phase_index = step % phase_period
            if phase_index == 0:
                renewal_steps.append(warmup_steps + step)
            phase = torch.full(
                (variants,),
                phase_index / float(phase_period),
                device=sim_device,
                dtype=torch.float32,
            )
            command = qc.append_sin_cos_phase(latents_z, phase)
            td.set(latent_key, command.reshape(*td.batch_size, latent_dim))
            publish_actor_command(command.reshape(-1, latent_dim))

            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                td = policy_operator(td)
                action = td.get("action")
                action_norms += action.reshape(variants, -1).norm(dim=-1)
                td = env.step(td)
                td = step_mdp(
                    td, exclude_reward=True, exclude_done=False, exclude_action=True
                )
            root_track.append(
                base_env.robot.data.root_pos_w.torch[:variants].detach().cpu().numpy()
            )

        if video_recorder is not None and getattr(video_recorder, "recording", False):
            video_recorder.stop_recording()

        final_root = base_env.robot.data.root_pos_w.torch[:variants].clone()
        final_joints = base_env.robot.data.joint_pos.torch[:variants].clone()
        root_delta = (final_root - initial_root).detach().cpu()
        joint_delta_norm = (final_joints - initial_joints).norm(dim=-1).detach().cpu()
        mean_action_norm = (action_norms / float(rollout_steps)).detach().cpu()

        # --- artifacts ---------------------------------------------------- #
        effects = torch.stack(
            [
                root_delta[:, 0],
                root_delta[:, 1],
                root_delta[:, 2],
                joint_delta_norm,
                mean_action_norm,
            ],
            dim=-1,
        )
        labels = []
        with (job_dir / "category_layout.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            if args_cli.mode == "n_groups":
                shared = job["shared_groups"]
                writer.writerow(
                    ["variant", "env_id", "is_baseline", "group_ids", "categories"]
                )
                for row in range(variants):
                    writer.writerow(
                        [
                            row,
                            row,
                            row == 0,
                            " ".join(str(g) for g in shared),
                            " ".join(str(int(codes[row, g])) for g in shared),
                        ]
                    )
                    labels.append("base" if row == 0 else f"v{row}")
            elif job["group"] is not None:
                writer.writerow(
                    ["variant", "env_id", "group", "category", "is_baseline"]
                )
                for row, category in enumerate(job["categories"]):
                    writer.writerow([row, row, job["group"], int(category), row == 0])
                    labels.append(f"c{int(category)}")
            elif args_cli.mode == "across_groups":
                writer.writerow(
                    [
                        "variant",
                        "env_id",
                        "num_groups",
                        "group_ids",
                        "base_categories",
                        "new_categories",
                    ]
                )
                for row, group_list in enumerate(job["swept_groups"]):
                    writer.writerow(
                        [
                            row,
                            row,
                            len(group_list),
                            " ".join(str(g) for g in group_list),
                            " ".join(str(int(base_code[g])) for g in group_list),
                            " ".join(str(int(codes[row, g])) for g in group_list),
                        ]
                    )
                    labels.append(
                        f"g{group_list[0]}"
                        if len(group_list) == 1
                        else f"v{row}({len(group_list)}g)"
                    )
            else:
                writer.writerow(
                    ["variant", "env_id", "changed_groups", "changed_group_ids"]
                )
                mask = job["mask"]
                for row in range(variants):
                    ids = mask[row].nonzero().reshape(-1).tolist()
                    writer.writerow([row, row, len(ids), " ".join(str(i) for i in ids)])
                    labels.append("base" if row == 0 else f"v{row}")

        with (job_dir / "effects.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "variant",
                    "label",
                    "final_dx_m",
                    "final_dy_m",
                    "final_dz_m",
                    "final_joint_delta_norm_rad",
                    "mean_action_norm",
                ]
            )
            for row in range(variants):
                writer.writerow(
                    [row, labels[row], *[f"{float(v):.6f}" for v in effects[row]]]
                )

        np.savez_compressed(
            job_dir / "rollouts.npz",
            codes=codes.cpu().numpy(),
            latents=latents_z.cpu().numpy(),
            root_pos_w=np.stack(root_track).astype(np.float32),
            effects=effects.numpy().astype(np.float32),
            labels=np.array(labels),
        )

        scatter = qc.plot_effect_scatter(
            labels=labels,
            values=effects,
            value_names=["dx", "dy", "dz", "|dq|", "|a|"],
            title=f"{args_cli.mode} / {job['name']} on {entry.motion}",
            output_path=job_dir / "effect_pca.png",
        )
        print(f"[PLOT] {scatter}")

        # All jobs share one recorder directory, so match this job's clip stem
        # rather than globbing everything written so far.
        video_paths = sorted(
            str(path.resolve()) for path in video_folder.glob(f"{clip_stem}*.mp4")
        )
        for path in video_paths:
            qc.announce_video(path)
        if args_cli.video and not video_paths:
            print(f"[WARN] No MP4 matching {clip_stem}*.mp4 under {video_folder}.")

        spans = {
            "final_dx_m": [float(effects[:, 0].min()), float(effects[:, 0].max())],
            "final_dy_m": [float(effects[:, 1].min()), float(effects[:, 1].max())],
            "final_dz_m": [float(effects[:, 2].min()), float(effects[:, 2].max())],
            "final_joint_delta_norm_rad": [
                float(effects[:, 3].min()),
                float(effects[:, 3].max()),
            ],
        }
        print(f"[INFO] {job['name']} effect spans: {spans}")

        results.append(
            {
                "name": job["name"],
                "group": job["group"],
                "baseline_category": job["baseline_category"],
                "categories": (
                    [int(c) for c in job["categories"]]
                    if job["categories"] is not None
                    else None
                ),
                "groups_per_variant": job.get("groups_per_variant"),
                "swept_groups": job.get("swept_groups"),
                "num_groups": job.get("num_groups"),
                "shared_groups": job.get("shared_groups"),
                "groups_per_robot": job.get("groups_per_robot"),
                "disjoint_groups": job.get("disjoint_groups"),
                "changed_latent_values": expected_changed,
                "video_paths": video_paths,
                "effect_spans": spans,
                "output_dir": str(job_dir),
            }
        )

        qc.write_provenance(
            job_dir,
            mode=args_cli.mode,
            task=args_cli.task,
            seed=int(args_cli.seed),
            base_code_categories=[int(c) for c in base_code.tolist()],
            base_code_source=args_cli.base_code,
            random_base_code=args_cli.base_code == "random",
            warmup_seconds=float(args_cli.warmup_seconds),
            warmup_steps=warmup_steps,
            warmup_command="encoder on the live macro window, renewed per window",
            switch_step=warmup_steps,
            warmup_root_spread_m=root_spread_m,
            warmup_joint_spread_rad=joint_spread_rad,
            warmup_code_agreement=code_agreement,
            effects_measured_from="switch_step",
            code_is_constant=True,
            code_renewal_period_steps=phase_period,
            code_renewal_steps=renewal_steps,
            phase_mode="sin_cos",
            rollout_steps=rollout_steps,
            variants=variants,
            start_frame=int(args_cli.start_frame),
            motion=entry.motion,
            trajectory_rank=entry.rank,
            skill_checkpoint=str(bundle.checkpoint_path),
            skill_checkpoint_sha256=binding["skill_checkpoint_sha256"],
            policy_checkpoint=binding["low_level_checkpoint"],
            policy_checkpoint_sha256=binding["low_level_checkpoint_sha256"],
            policy_checkpoint_step=qc.policy_checkpoint_step(
                args_cli.policy_checkpoint
            ),
            encoder_binding=binding,
            encoder_config=bundle.config.to_dict()
            if hasattr(bundle.config, "to_dict")
            else None,
            latent_mode=bundle.latent_mode,
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
            protocol={
                "disabled_terminations": sorted(disabled_terminations),
                "domain_randomization": dr_record,
                "njmax": int(args_cli.njmax),
                "nconmax": int(args_cli.nconmax),
                "command_injection": "raw policy operator + set_agent_latent_command",
            },
            result=results[-1],
        )
        print(
            f"[INFO] Finished {job_index + 1}/{len(jobs)}: {job['name']} -> {job_dir}"
        )

    env.close()
    print(f"[INFO] Output root: {output_root}")


if __name__ == "__main__":
    main()
    simulation_app.close()
