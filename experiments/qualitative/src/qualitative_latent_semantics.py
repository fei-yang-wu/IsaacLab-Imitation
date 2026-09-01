#!/usr/bin/env python3
# ruff: noqa: E402
"""Encode many motions' windows to latents, in each motion's OWN frame.

Stage 1 of the latent-semantics analysis. Draws N motions from the full
BONES-SEED catalog, picks ``--windows_per_motion`` evenly spaced windows inside
each one, encodes every window with the frozen skill encoder, and writes ONE
consolidated ``latents.npz``. Stage 2
(``qualitative_latent_semantics_cluster.py``) clusters those rows and names the
clusters from the dataset's ``language_goal`` annotations.

Two properties make these rows comparable across motions:

**Windows come from the environment, not from a hand-rolled reader.** Every row
is a ``current_expert_macro_transition_batch(horizon_steps=10)`` call -- the
exact call the frozen tracker makes at command-publication time -- so there is
no parallel implementation of the 38-value macro row to certify.

**Every window is re-expressed in its own motion's frame.** The environment
hands the window over relative to the live robot, so the same motion would
encode differently depending on where the robot happens to stand. Anchoring on
the window's own first frame cancels that transform exactly (see
``qualitative_common.reanchor_window_to_first_frame``), leaving a latent that is
a property of the MOTION. Without this the clusters would partly encode robot
placement, which is meaningless here. This is the same "reference frame" the
motion-switch analyses publish after a switch.

Because the frame is cancelled, the robot's physical state is irrelevant and the
simulation is never stepped: each batch re-points the reference cursors with
``set_env_cursor`` and reads the macro window straight out. One environment
therefore serves many motions, and the sweep costs encoder forward passes rather
than physics.

Example::

    pixi run -e isaaclab python experiments/qualitative/src/qualitative_latent_semantics.py \\
        --encoder_checkpoint logs/.../encoder/checkpoints/latest.pt \\
        --num_motions 4000 --windows_per_motion 8 --num_envs 512 \\
        --headless --output_dir outputs/.../latent_semantics_encode \\
        <shared hydra overrides>
"""

from __future__ import annotations

import argparse
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
    "--encoder_checkpoint",
    type=str,
    required=True,
    help="Frozen gumbel_multicat or sonic_fsq skill encoder (.pt).",
)
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    default=None,
    help=(
        "Optional low-level checkpoint. Not used for encoding; when given, its "
        "encoder binding is asserted and recorded so the clusters can be tied "
        "to the tracker they will be interpreted alongside."
    ),
)
parser.add_argument(
    "--reference_arrays_dir",
    type=str,
    default=None,
    help="Reference arrays directory. Defaults to the repo-local 129k arrays.",
)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument("--num_motions", type=int, default=4000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--windows_per_motion",
    type=int,
    default=8,
    help=(
        "Windows sampled per motion, spread evenly over the whole trajectory. "
        "Evenly spaced rather than the first K: the opening of most motions is "
        "a neutral standing pose, and taking only the first K would bias the "
        "whole analysis toward it."
    ),
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=512,
    help=(
        "Environments used as a batch. The sweep re-points cursors instead of "
        "stepping, so this is a throughput knob, not the motion count."
    ),
)
parser.add_argument(
    "--ranks",
    type=str,
    default=None,
    help="Comma-separated trajectory ranks; overrides the seeded draw.",
)
parser.add_argument(
    "--motions",
    type=str,
    default=None,
    help="Comma-separated motion names; overrides the seeded draw.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import isaaclab_imitation.tasks  # noqa: F401

from imitation_experiments.provenance.paper_protocol_metadata import (
    disable_domain_randomization,
)

#: The re-anchored window's own first frame must sit at the origin. Anything
#: above this means the anchor slice is wrong and the "motion frame" is fiction.
REANCHOR_TOLERANCE_M = 1.0e-5


@hydra_task_config(args_cli.task, qc.AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
) -> None:
    reference_arrays_dir = Path(
        args_cli.reference_arrays_dir
        or (qc.repo_root() / qc.DEFAULT_REFERENCE_ARRAYS_DIRNAME)
    ).resolve()

    bundle = qc.load_skill_encoder(
        args_cli.encoder_checkpoint,
        args_cli.device or "cuda:0",
    )
    print(
        f"[INFO] Encoder: {bundle.checkpoint_path}\n"
        f"       latent_mode={bundle.config.latent_mode} "
        + (
            f"groups={bundle.groups} categories={bundle.categories} "
            f"code_dim={bundle.code_dim} "
            if bundle.is_discrete
            else "continuous (no discrete code) "
        )
        + f"z_dim={bundle.z_dim}\n"
        f"       horizon={bundle.horizon_steps} window_steps={bundle.window_steps} "
        f"state_dim={bundle.state_dim}"
    )

    binding_record = None
    if args_cli.policy_checkpoint is not None:
        binding_record = qc.assert_encoder_binding(
            bundle.checkpoint_path, args_cli.policy_checkpoint
        )
        print(f"[PASS] encoder binding: {args_cli.policy_checkpoint}")

    catalog = qc.MotionCatalog.from_reference_arrays(reference_arrays_dir)
    selected = catalog.select(
        count=int(args_cli.num_motions),
        seed=int(args_cli.seed),
        # A window at local step t reads frames [t, t + horizon]; the shortest
        # usable trajectory therefore has horizon + 1 frames.
        min_length=bundle.horizon_steps + 1,
        ranks=qr.parse_int_list(args_cli.ranks),
        motions=qr.parse_str_list(args_cli.motions),
    )
    print(f"[INFO] {len(selected)} motions from {catalog.manifest_path}")

    # One flat job list of (motion, start frame). Building it up front means the
    # batch loop below is a pure map over jobs and the row count is known before
    # Isaac is touched.
    job_rank: list[int] = []
    job_start: list[int] = []
    job_motion: list[str] = []
    job_length: list[int] = []
    for entry in selected:
        starts = qc.plan_window_starts(
            entry.length, bundle.horizon_steps, int(args_cli.windows_per_motion)
        )
        for start in starts:
            job_rank.append(entry.rank)
            job_start.append(start)
            job_motion.append(entry.motion)
            job_length.append(entry.length)
    if not job_rank:
        raise RuntimeError(
            "No motion yielded a complete encoder window; nothing to encode."
        )
    print(
        f"[INFO] {len(job_rank)} windows over {len(selected)} motions "
        f"({len(job_rank) / len(selected):.1f} per motion, "
        f"{int(args_cli.windows_per_motion)} requested)."
    )

    num_envs = max(1, min(int(args_cli.num_envs), len(job_rank)))
    env_cfg.scene.num_envs = num_envs
    agent_cfg.env.num_envs = num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    logger_cfg = getattr(agent_cfg, "logger", None)
    if logger_cfg is not None:
        logger_cfg.backend = ""
        logger_cfg.video = False

    disabled_terminations = qr.disable_all_terminations(env_cfg)
    print(f"[INFO] Disabled terminations: {sorted(disabled_terminations)}")
    dr_record = disable_domain_randomization(env_cfg)

    output_dir = qc.prepare_output_dir(
        args_cli.output_dir, overwrite=args_cli.overwrite
    )
    env_cfg.log_dir = str(output_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")
    base_env = qr.unwrap_imitation_env(env)
    env.reset()

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
        f"[INFO] macro row: anchor_pos_b {anchor_slice}, anchor_ori_b "
        f"{ori_slice} of {bundle.state_dim}."
    )

    trajectory_manager = base_env.trajectory_manager
    plane = base_env.expert_data_plane
    all_env_ids = torch.arange(num_envs, dtype=torch.long)

    latents: list[np.ndarray] = []
    # Absent for a continuous latent: there is no code to record. Stage 2 reads
    # only `latent`, so the clustering is identical either way.
    levels: list[np.ndarray] = []
    max_residual = 0.0

    with torch.inference_mode():
        for begin in range(0, len(job_rank), num_envs):
            end = min(begin + num_envs, len(job_rank))
            size = end - begin
            env_ids = all_env_ids[:size]

            trajectory_manager.set_env_cursor(
                env_ids=env_ids,
                ranks=torch.as_tensor(job_rank[begin:end], dtype=torch.long),
                steps=torch.as_tensor(job_start[begin:end], dtype=torch.long),
            )
            # The macro row is cached per control step; nothing is stepped here,
            # so without this every batch would re-read the first batch's window.
            plane._mdp_cache_step = -1

            cursor = trajectory_manager.env_step[:size].detach().cpu().numpy()
            expected = np.asarray(job_start[begin:end], dtype=np.int64)
            if not np.array_equal(cursor, expected):
                bad = int(np.argmax(cursor != expected))
                msg = (
                    f"Cursor did not take the requested frame for job "
                    f"{begin + bad} ({job_motion[begin + bad]}): asked "
                    f"{expected[bad]}, got {cursor[bad]}. A clamped cursor "
                    "would encode a different window than the one recorded."
                )
                raise RuntimeError(msg)

            batch = base_env.current_expert_macro_transition_batch(
                horizon_steps=bundle.horizon_steps
            )["hl"]
            state = batch["state"].to(device=bundle.device, dtype=torch.float32)
            future = batch["future_window"].to(
                device=bundle.device, dtype=torch.float32
            )
            window = torch.cat((state.unsqueeze(1), future), dim=1)[:size]

            anchored = qc.reanchor_window_to_first_frame(
                window, pos_slice=anchor_slice, ori_slice=ori_slice
            )
            residual = float(
                anchored[:, 0, anchor_slice[0] : anchor_slice[1]].abs().max()
            )
            max_residual = max(max_residual, residual)
            if residual > REANCHOR_TOLERANCE_M:
                msg = (
                    f"Re-anchored window still offsets its own first frame by "
                    f"{residual:.3e}; the anchor slice is wrong and these "
                    "latents would not be in the motion's frame."
                )
                raise RuntimeError(msg)

            encoded = qc.encode_windows(bundle, anchored[:, 0], anchored[:, 1:])
            latents.append(encoded["z"].cpu().numpy().astype(np.float32))
            categories = encoded.get("categories")
            if categories is not None:
                levels.append(categories.cpu().numpy().astype(np.int64))

            done = end
            if done % (num_envs * 10) == 0 or done == len(job_rank):
                print(f"[INFO] encoded {done}/{len(job_rank)} windows")

    env.close()

    latent_array = np.concatenate(latents, axis=0)
    if latent_array.shape[0] != len(job_rank):
        msg = (
            f"Encoded {latent_array.shape[0]} rows for {len(job_rank)} jobs; "
            "the row/job alignment is broken."
        )
        raise RuntimeError(msg)

    columns = {
        "latent": latent_array,
        "motion": np.asarray(job_motion),
        "rank": np.asarray(job_rank, dtype=np.int64),
        "local_step": np.asarray(job_start, dtype=np.int64),
        "motion_length": np.asarray(job_length, dtype=np.int64),
    }
    if levels:
        columns["level"] = np.concatenate(levels, axis=0)
    np.savez_compressed(output_dir / "latents.npz", **columns)
    print(
        f"[INFO] Wrote {output_dir / 'latents.npz'}: "
        f"{latent_array.shape[0]} rows x {latent_array.shape[1]} latent values"
        + ("." if levels else ", no `level` column (continuous latent).")
    )

    # A deterministic z is unbounded -- only the pretrain L2 penalty limits it --
    # so distances in this space carry whatever per-dimension scale the encoder
    # happened to learn. Stage 2 clusters the raw published vector on purpose, so
    # record the spread here: a clustering that merely followed the widest few
    # dimensions is then visible as such rather than read as structure.
    latent_scale = qc.latent_scale_summary(torch.from_numpy(latent_array))
    print(
        "[INFO] Latent scale: per-dimension std "
        f"min={latent_scale['z_std_min']:.4f} "
        f"mean={latent_scale['z_std_mean']:.4f} "
        f"max={latent_scale['z_std_max']:.4f} "
        f"(max/median {latent_scale['z_std_ratio_max_over_median']:.2f}x), "
        f"effective rank {latent_scale['effective_rank']:.2f} of "
        f"{latent_array.shape[1]}"
    )

    qc.write_provenance(
        output_dir,
        mode="latent_semantics_encode",
        task=args_cli.task,
        encoder_checkpoint=str(bundle.checkpoint_path),
        encoder_sha256=qc.sha256(bundle.checkpoint_path),
        policy_checkpoint=(
            str(Path(args_cli.policy_checkpoint).resolve())
            if args_cli.policy_checkpoint
            else None
        ),
        encoder_binding=binding_record,
        latent_mode=bundle.latent_mode,
        z_dim=int(bundle.z_dim),
        groups=int(bundle.groups),
        horizon_steps=int(bundle.horizon_steps),
        reference_arrays_dir=str(reference_arrays_dir),
        reference_arrays_manifest=str(catalog.manifest_path),
        seed=int(args_cli.seed),
        num_motions=len(selected),
        windows_per_motion=int(args_cli.windows_per_motion),
        num_rows=int(latent_array.shape[0]),
        num_envs=num_envs,
        command_frame="reference",
        reanchor_max_residual_m=max_residual,
        reanchor_tolerance_m=REANCHOR_TOLERANCE_M,
        disabled_terminations=sorted(disabled_terminations),
        domain_randomization=dr_record,
        code_diagnostics=qc.code_diagnostics_meaning(bundle),
        is_discrete=bool(bundle.is_discrete),
        latent_scale=latent_scale,
    )
    print(f"[INFO] Output root: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
