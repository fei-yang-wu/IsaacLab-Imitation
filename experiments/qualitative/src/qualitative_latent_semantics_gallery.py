#!/usr/bin/env python3
# ruff: noqa: E402
"""One video per latent cluster: its member motions, side by side.

Stage 3 of the latent-semantics analysis, and the one that produces the actual
evidence. For each cluster in the stage-2 manifest it puts that cluster's member
motions in a single scene -- one robot each -- replays every robot's own window
at the same time, and records ONE clip.

That is the whole idea. If the cluster is real, eight robots drawn from eight
DIFFERENT motions visibly do the same thing, and you name the cluster from what
you watched. If they do eight unrelated things, the cluster has no topic, and
that is a finding rather than something to tune away.

The robots are not controlled here: each one is driven directly onto its
reference pose every step (``configure_reference_replay_targets`` with each env
as its own source), so what you see is the motion data the encoder read, not a
tracker's imitation of it.

**A window is 10 frames -- 0.2 s.** That is a blink: shown alone, from a shot
wide enough to hold eight robots, it is a few pixels of change and no one can
name it however many times it loops. So by default each clip surrounds the
window with ``--context_frames`` of reference either side (0.5 s, giving 1.2 s
total), holds every frame for ``--slowdown`` steps, and repeats the span
``--loops`` times.

Those context frames are **not** what the latent encodes. They are there so a
person can recognise the action at all; the cluster still groups the window
alone. Pass ``--context_frames 0`` to show only the encoded window and judge
that by itself. Provenance records ``shows_only_encoded_frames`` either way, so
a clip can never be mistaken for the stricter one.

Example::

    pixi run -e isaaclab python experiments/qualitative/src/qualitative_latent_semantics_gallery.py \\
        --clusters_json outputs/.../latent_semantics/clusters.json \\
        --encoder_checkpoint logs/.../encoder/checkpoints/latest.pt \\
        --output_dir outputs/.../latent_semantics_gallery --video --headless \\
        <shared hydra overrides>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v2")
parser.add_argument(
    "--clusters_json",
    type=str,
    required=True,
    help="clusters.json written by qualitative_latent_semantics_cluster.py.",
)
parser.add_argument(
    "--encoder_checkpoint",
    type=str,
    required=True,
    help="Recorded for provenance so a gallery is tied to the code space it shows.",
)
parser.add_argument(
    "--reference_arrays_dir",
    type=str,
    default=None,
    help="Reference arrays directory. Defaults to the repo-local 129k arrays.",
)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--overwrite", action="store_true", default=False)
parser.add_argument(
    "--clusters",
    type=str,
    default=None,
    help="Comma-separated cluster ids to render. Default: every cluster.",
)
parser.add_argument(
    "--loops", type=int, default=2, help="Times each span is replayed per clip."
)
parser.add_argument(
    "--slowdown",
    type=int,
    default=2,
    help="Control steps held per reference frame. 2 halves playback speed.",
)
parser.add_argument(
    "--context_frames",
    type=int,
    default=25,
    help=(
        "Reference frames shown before and after the encoded window. The window "
        "itself is 10 frames -- 0.2 s -- which a person cannot read as an "
        "action however many times it loops, so the default surrounds it with "
        "0.5 s each side. These extra frames are NOT what the latent encodes; "
        "pass 0 to show only the encoded window and judge that alone. "
        "`shows_only_encoded_frames` in provenance records which you did."
    ),
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument(
    "--filmstrip_members",
    type=int,
    default=3,
    help=(
        "Members per cluster that also get a still-frame strip: one small "
        "horizontal image of close-up robot frames sampled across that "
        "member's shown span. Member order is centroid-closeness, so these "
        "are the cluster's most typical members. 0 disables. Needs --video "
        "(rendering); without it the strips are skipped with a note."
    ),
)
parser.add_argument(
    "--filmstrip_frames",
    type=int,
    default=8,
    help="Frames per strip, evenly spaced over the member's shown span.",
)
parser.add_argument(
    "--filmstrip_px",
    type=int,
    default=300,
    help="Side of each square strip tile in pixels.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Metres between robots. Wider keeps travelling motions apart on screen.",
)
parser.add_argument("--njmax", type=int, default=320)
parser.add_argument("--nconmax", type=int, default=40)
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


def frame_schedule(
    start: int,
    horizon: int,
    length: int,
    *,
    loops: int,
    slowdown: int,
    context: int,
) -> list[int]:
    """Reference frames to show, in order, for one member window.

    The window itself is ``[start, start + horizon]``. ``context`` widens it on
    both sides, clamped to the motion. Each frame is repeated ``slowdown``
    times and the whole span is repeated ``loops`` times, because 10 frames at
    50 Hz is 0.2 s and is otherwise impossible to see.
    """
    if loops < 1 or slowdown < 1:
        raise ValueError("loops and slowdown must both be >= 1.")
    first = max(0, int(start) - int(context))
    last = min(int(length) - 1, int(start) + int(horizon) + int(context))
    span = [frame for frame in range(first, last + 1) for _ in range(int(slowdown))]
    return span * int(loops)


@hydra_task_config(args_cli.task, qc.AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
) -> None:
    manifest = json.loads(
        Path(args_cli.clusters_json).expanduser().resolve().read_text()
    )
    clusters = manifest["clusters"]
    wanted = qr.parse_int_list(args_cli.clusters, option="--clusters")
    if wanted is not None:
        known = {int(entry["cluster"]) for entry in clusters}
        missing = sorted(set(wanted) - known)
        if missing:
            raise ValueError(f"Clusters not in the manifest: {missing}.")
        clusters = [entry for entry in clusters if int(entry["cluster"]) in set(wanted)]
    if not clusters:
        raise ValueError("No clusters selected.")

    bundle = qc.load_skill_encoder(
        args_cli.encoder_checkpoint, args_cli.device or "cuda:0"
    )
    horizon = int(bundle.horizon_steps)

    # Every clip uses the same scene, so it must hold the largest cluster.
    num_envs = max(len(entry["members"]) for entry in clusters)
    print(
        f"[INFO] {len(clusters)} clusters, up to {num_envs} motions each, "
        f"window {horizon} frames, loops {args_cli.loops}, "
        f"slowdown {args_cli.slowdown}, context {args_cli.context_frames}."
    )

    env_cfg.scene.num_envs = num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = float(args_cli.env_spacing)
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
    solver_cfg = getattr(getattr(env_cfg, "sim", None), "newton", None)
    if solver_cfg is not None:
        solver_cfg.njmax = int(args_cli.njmax)
        solver_cfg.nconmax = int(args_cli.nconmax)

    disabled_terminations = qr.disable_all_terminations(env_cfg)
    dr_record = disable_domain_randomization(env_cfg)

    output_dir = qc.prepare_output_dir(
        args_cli.output_dir, overwrite=args_cli.overwrite
    )
    video_folder = output_dir / "videos"
    env_cfg.log_dir = str(output_dir)

    # The longest clip decides the recorder's frame budget.
    longest = max(
        len(
            frame_schedule(
                int(member["start_frame"]),
                horizon,
                int(member["motion_length"]),
                loops=int(args_cli.loops),
                slowdown=int(args_cli.slowdown),
                context=int(args_cli.context_frames),
            )
        )
        for entry in clusters
        for member in entry["members"]
    )

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
            # One clip per cluster, each started and stopped by hand.
            step_trigger=lambda _step: False,
            video_length=longest + 2,
            disable_logger=True,
        )
        video_recorder = env

    base_env = qr.unwrap_imitation_env(env)
    env.reset()

    # Each env replays its OWN cursor: pairing every env with itself makes the
    # cursor sync a no-op and leaves the replay, which is what puts the
    # reference motion on screen instead of a tracker's imitation of it.
    all_ids = list(range(num_envs))
    base_env.configure_reference_replay_targets(
        source_env_ids=all_ids, target_env_ids=all_ids
    )

    # Static per-environment world offsets: the grid the robots stand on. Used
    # only to label which robot in the frame is which motion.
    env_origins = base_env.scene.env_origins[:num_envs].detach().cpu()

    trajectory_manager = base_env.trajectory_manager
    env_ids = torch.arange(num_envs, dtype=torch.long)
    zero_action = torch.zeros(
        num_envs,
        int(base_env.action_manager.total_action_dim),
        device=torch.device(str(base_env.device)),
    )

    rendered: list[dict[str, object]] = []
    for entry in clusters:
        cluster = int(entry["cluster"])
        members = entry["members"]
        schedules = [
            frame_schedule(
                int(member["start_frame"]),
                horizon,
                int(member["motion_length"]),
                loops=int(args_cli.loops),
                slowdown=int(args_cli.slowdown),
                context=int(args_cli.context_frames),
            )
            for member in members
        ]
        steps = max(len(schedule) for schedule in schedules)
        # Short members hold their last frame rather than looping out of sync,
        # so every robot is still showing its own window when the clip ends.
        padded = [
            schedule + [schedule[-1]] * (steps - len(schedule))
            for schedule in schedules
        ]
        # Spare envs, if this cluster has fewer members than the largest one,
        # repeat the first member instead of showing a stale pose from the
        # previous cluster.
        while len(padded) < num_envs:
            padded.append(padded[0])
            members = list(members) + [members[0]]
        ranks = torch.as_tensor(
            [int(member["rank"]) for member in members[:num_envs]], dtype=torch.long
        )

        hint = " ".join(entry.get("term_hint", []))
        print(
            f"\n[INFO] cluster {cluster}: {len(entry['members'])} motions, "
            f"{steps} steps. hint: {hint}"
        )
        for member in entry["members"]:
            print(
                f"       rank={member['rank']:6d} frame={member['start_frame']:5d} "
                f"{member['motion']}  ({member['language_goal']})"
            )

        def _place(step: int) -> None:
            trajectory_manager.set_env_cursor(
                env_ids=env_ids,
                ranks=ranks,
                steps=torch.as_tensor(
                    [schedule[step] for schedule in padded[:num_envs]],
                    dtype=torch.long,
                ),
            )
            base_env.expert_data_plane._mdp_cache_step = -1
            base_env.apply_reference_replay_targets()

        # Warm up on the first frame BEFORE recording starts. Two reasons: the
        # camera must be framed on where these robots actually are, and
        # `root_pos_w` only reports that after a step has flushed the replay --
        # framing beforehand aims at the reset grid and sits too far back to
        # read a limb. The camera is also set before `start_recording` because
        # setting it mid-recording does not reach the recorded frames.
        with torch.inference_mode():
            _place(0)
            env.step(zero_action)
            qr.set_grid_camera(base_env, num_envs, framing="close", announce=True)
            env.step(zero_action)

        if video_recorder is not None:
            video_recorder.start_recording(f"cluster_{cluster:03d}")

        with torch.inference_mode():
            for step in range(steps):
                _place(step)
                env.step(zero_action)

        if video_recorder is not None and getattr(video_recorder, "recording", False):
            video_recorder.stop_recording()

        # Still-frame strips for the most typical members, AFTER the recorder
        # has stopped so no still leaks into the clip. On this backend
        # `env.render()` returns the frame the viewer LAST DREW, and the
        # viewer only draws during `env.step` -- a placement or camera change
        # alone never reaches the framebuffer (verified: capturing without
        # stepping returns the clip's stale final frame). So each still takes
        # two flush steps, mirroring the recording warmup above: place + step
        # so the pose reaches the viewer and `root_pos_w`, aim a third-person
        # camera behind the member's robot along its heading (re-aimed per
        # frame, so travelling motions stay centered), then re-place + step so
        # the drawn frame carries both. To show exactly ONE robot per strip,
        # the reference replay pairing is narrowed to the member for the
        # duration of its strip and every other robot is dropped far below
        # the floor once -- `step()` re-applies the replay every call, so a
        # teleport alone is overwritten before the draw, but a narrowed
        # pairing leaves the hidden robots where they fell. The full pairing
        # is restored afterwards and the next `_place` replays every robot
        # back onto its reference. The still therefore shows the same
        # one-control-step-from-reference look as the video.
        filmstrips: list[dict[str, object]] = []
        strip_count = min(int(args_cli.filmstrip_members), len(entry["members"]))
        if strip_count > 0 and args_cli.video:
            strip_dir = output_dir / "filmstrips"

            def _hide_other_robots(keep_index: int) -> None:
                robot = base_env.robot
                pose = torch.cat(
                    [
                        robot.data.root_pos_w.torch.clone(),
                        robot.data.root_quat_w.torch.clone(),
                    ],
                    dim=-1,
                )
                hide = torch.as_tensor(
                    [i for i in range(num_envs) if i != keep_index],
                    dtype=torch.long,
                    device=pose.device,
                )
                pose[hide, 2] -= 100.0
                robot.write_root_link_pose_to_sim(pose[hide], env_ids=hide)
                robot.write_data_to_sim()

            with torch.inference_mode():
                for member_index in range(strip_count):
                    member = members[member_index]
                    start = int(member["start_frame"])
                    length = int(member["motion_length"])
                    context = int(args_cli.context_frames)
                    slowdown = max(1, int(args_cli.slowdown))
                    first = max(0, start - context)
                    last = min(length - 1, start + horizon + context)
                    sampled = np.unique(
                        np.linspace(
                            first, last, num=max(1, int(args_cli.filmstrip_frames))
                        )
                        .round()
                        .astype(np.int64)
                    )
                    base_env.configure_reference_replay_targets(
                        source_env_ids=[member_index],
                        target_env_ids=[member_index],
                    )
                    _hide_other_robots(member_index)
                    stills = []
                    for frame in sampled.tolist():
                        step_index = (int(frame) - first) * slowdown
                        _place(step_index)
                        env.step(zero_action)
                        qr.set_third_person_camera(base_env, member_index)
                        _place(step_index)
                        env.step(zero_action)
                        stills.append(env.render())
                    strip = qc.compose_filmstrip(
                        stills, tile_px=int(args_cli.filmstrip_px)
                    )
                    strip_path = qc.save_image(
                        strip,
                        strip_dir / f"cluster_{cluster:03d}_member_{member_index}.png",
                    )
                    print(f"[FILMSTRIP] {strip_path}")
                    filmstrips.append(
                        {
                            "env_index": member_index,
                            "motion": str(member["motion"]),
                            "rank": int(member["rank"]),
                            "sampled_frames": [int(f) for f in sampled.tolist()],
                            "window_frames": [start, start + horizon],
                            "tile_px": int(args_cli.filmstrip_px),
                            "path": str(strip_path),
                        }
                    )
                base_env.configure_reference_replay_targets(
                    source_env_ids=all_ids,
                    target_env_ids=all_ids,
                )
        elif strip_count > 0:
            print("[NOTE] filmstrips skipped: rendering needs --video.")

        # Which robot on screen is which motion. Without this a clip shows eight
        # anonymous figures and a coherent-looking cluster cannot be checked
        # against the source clips, nor an odd one out identified. `env_origin`
        # is where that robot's environment sits in the world, which is what
        # maps a position in the frame back to a row here.
        video_name = f"cluster_{cluster:03d}"
        labels = {
            "cluster": cluster,
            "video": f"{video_name}.mp4",
            "term_hint": entry.get("term_hint", []),
            "steps": steps,
            "playback": {
                "horizon_frames": horizon,
                "loops": int(args_cli.loops),
                "slowdown": int(args_cli.slowdown),
                "context_frames": int(args_cli.context_frames),
                "shows_only_encoded_frames": int(args_cli.context_frames) == 0,
            },
            "robots": [
                {
                    "env_index": index,
                    "motion": str(member["motion"]),
                    "rank": int(member["rank"]),
                    "start_frame": int(member["start_frame"]),
                    "window_frames": [
                        int(member["start_frame"]),
                        int(member["start_frame"]) + horizon,
                    ],
                    "motion_length": int(member["motion_length"]),
                    "language_goal": str(member["language_goal"]),
                    # True when this cluster had fewer members than the scene has
                    # robots, so an earlier member is shown twice. Recorded so a
                    # duplicate on screen is never read as agreement.
                    "duplicate_filler": index >= len(entry["members"]),
                    "env_origin": [
                        round(float(value), 4)
                        for value in env_origins[index][:2].tolist()
                    ],
                }
                for index, member in enumerate(members[:num_envs])
            ],
            "filmstrips": filmstrips,
        }
        if video_recorder is not None:
            (video_folder / f"{video_name}.json").write_text(
                json.dumps(labels, indent=2)
            )

        rendered.append(labels)

    env.close()

    qc.write_provenance(
        output_dir,
        mode="latent_semantics_gallery",
        task=args_cli.task,
        clusters_json=str(Path(args_cli.clusters_json).resolve()),
        encoder_checkpoint=str(bundle.checkpoint_path),
        encoder_sha256=qc.sha256(bundle.checkpoint_path),
        latent_mode=bundle.latent_mode,
        horizon_steps=horizon,
        loops=int(args_cli.loops),
        slowdown=int(args_cli.slowdown),
        context_frames=int(args_cli.context_frames),
        shows_only_encoded_frames=int(args_cli.context_frames) == 0,
        num_envs=num_envs,
        filmstrip_members=int(args_cli.filmstrip_members),
        filmstrip_frames=int(args_cli.filmstrip_frames),
        filmstrip_px=int(args_cli.filmstrip_px),
        clusters_rendered=[entry["cluster"] for entry in rendered],
        disabled_terminations=sorted(disabled_terminations),
        domain_randomization=dr_record,
        gallery=rendered,
    )

    index_path = output_dir / "gallery_index.json"
    index_path.write_text(
        json.dumps(
            {
                "clusters_json": str(Path(args_cli.clusters_json).resolve()),
                "encoder_sha256": qc.sha256(bundle.checkpoint_path),
                "member_selection": manifest.get("member_selection"),
                "clusters": rendered,
            },
            indent=2,
        )
    )
    print(f"[INFO] Wrote {index_path}")

    if args_cli.video:
        print(
            f"\n[INFO] {len(rendered)} cluster videos, each with a .json of "
            "which robot is which motion:"
        )
        for path in sorted(video_folder.glob("*.mp4")):
            qc.announce_video(path)
    print(f"\n[INFO] Output root: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
