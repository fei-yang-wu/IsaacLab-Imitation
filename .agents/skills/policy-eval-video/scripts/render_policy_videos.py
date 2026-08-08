#!/usr/bin/env python3
"""Render one policy-vs-reference video per motion for a low-level checkpoint.

Wraps ``scripts/viz/compare_policy_reference.py``, which already does the hard
part: it plays a checkpoint against the expert reference, draws both, disables
tracking terminations unless asked not to, and runs until the selected
trajectory ends. What this adds is the loop over motions, the argument
validation that turns three silent-wrong-answer traps into startup errors, and a
summary that prints every retained video's ABSOLUTE path -- which the repo
requires because remote sessions cannot pass video files back.

The traps, all of which cost real time on 2026-08-05:

1. The agent entry point must match the one the checkpoint was TRAINED with.
   `record_policy_rollout.py` has no such flag at all and would silently build
   the default architecture; a checkpoint trained under
   `rlopt_ipmd_tuned_cfg_entry_point` ([1024,1024,512], silu, input
   normalization) does not match it. This script requires the flag explicitly.
2. `--reference_visualization` takes a VALUE (body_markers|robot|both), not a
   bare flag. As a flag it swallows the next argument and argparse fails.
3. Passing `--video_length` caps every motion at the same number of frames,
   which silently truncates the long ones. Omit it and each motion runs to its
   own end -- that is the default and it is what you want per motion.

Usage is one line per data source. Reference arrays:

    python render_policy_videos.py \\
      --checkpoint /path/model_step_N.pt \\
      --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \\
      --output_root logs/videos/my_run \\
      --reference_arrays /path/arrays --persist_id my@id \\
      -- <extra hydra overrides>

Manifest + Zarr:

    python render_policy_videos.py ... \\
      --motion_manifest /path/manifest.json --dataset_path /path/zarr -- ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


REPO_MARKERS = ("pixi.toml", "source")
COMPARE_SCRIPT = "scripts/viz/compare_policy_reference.py"


def find_repo_root(start: Path | None = None) -> Path:
    origin = (start or Path(__file__)).resolve()
    for candidate in [origin, *origin.parents]:
        if all((candidate / marker).exists() for marker in REPO_MARKERS):
            return candidate
    raise SystemExit(f"Could not locate the repository root above {origin}")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--planner_checkpoint",
        type=Path,
        default=None,
        help="Optional SkillCommander checkpoint for an M3 planner diagnostic.",
    )
    parser.add_argument(
        "--skill_checkpoint",
        type=Path,
        default=None,
        help="Frozen skill encoder paired with --planner_checkpoint.",
    )
    parser.add_argument(
        "--language_embeddings",
        type=Path,
        default=None,
        help="Goal table paired with --planner_checkpoint.",
    )
    parser.add_argument(
        "--agent_entry_point",
        required=True,
        help=(
            "MUST match the entry point the checkpoint was trained with. Getting "
            "this wrong builds a different network and either fails to load or "
            "loads something that is not the trained policy."
        ),
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--task", default="Isaac-Imitation-G1-v2")
    parser.add_argument("--algo", default="IPMD")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="Trajectory ranks to render. Default: every trajectory in the source.",
    )
    parser.add_argument(
        "--max_motions",
        type=int,
        default=None,
        help="Render at most this many ranks. Guards against a 129k-motion source.",
    )
    parser.add_argument(
        "--reference_visualization",
        default="both",
        choices=["body_markers", "robot", "both"],
        help="Takes a value, not a flag. 'both' draws markers and the qpos replay.",
    )
    parser.add_argument(
        "--keep_terminations",
        action="store_true",
        default=False,
        help=(
            "Leave tracking terminations enabled. Off by default because the "
            "diagnostic pass this repo mandates is non-terminating: a fall must "
            "stay in frame rather than reset."
        ),
    )
    parser.add_argument(
        "--randomized_no_push",
        action="store_true",
        default=False,
        help=(
            "Keep startup/reset domain randomization but disable only the "
            "interval push event."
        ),
    )
    parser.add_argument(
        "--latent_temporal_ensemble",
        choices=("first", "exponential", "clipped_gated"),
        default="first",
        help="Execution rule for an ordered H30 latent SkillCommander planner.",
    )
    parser.add_argument("--latent_temporal_ensemble_decay", type=float, default=0.5)
    parser.add_argument("--latent_temporal_clip_std", type=float, default=1.0)
    parser.add_argument("--latent_temporal_gate_distance", type=float, default=2.0)
    parser.add_argument("--latent_temporal_gate_cosine", type=float, default=0.5)
    parser.add_argument(
        "--video_seconds",
        type=float,
        default=None,
        help=(
            "Cap every clip at this duration. Omit to run each motion to its own "
            "end, which is almost always what you want."
        ),
    )
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)

    source = parser.add_argument_group("data source (choose one)")
    source.add_argument("--reference_arrays", type=Path, default=None)
    source.add_argument("--persist_id", default=None)
    source.add_argument("--motion_manifest", type=Path, default=None)
    source.add_argument("--dataset_path", type=Path, default=None)

    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def source_overrides(args: argparse.Namespace) -> list[str]:
    """Hydra overrides selecting the reference data, with the modes kept apart."""
    if args.reference_arrays is not None:
        if args.motion_manifest is not None:
            raise SystemExit(
                "--reference_arrays and --motion_manifest are different sources; "
                "pass one. The environment refuses both together anyway."
            )
        arrays = args.reference_arrays.expanduser().resolve()
        sidecar = arrays / "reference_arrays_manifest.json"
        if not sidecar.is_file():
            raise SystemExit(
                f"No reference_arrays_manifest.json in {arrays}. An interrupted "
                "build writes arrays but no sidecar; quarantine it."
            )
        overrides = [
            "env.data.manifest=null",
            f"env.data.reference_arrays_dir={arrays}",
        ]
        if args.persist_id:
            overrides.append(f"env.data.persist_id={args.persist_id}")
        return overrides
    if args.motion_manifest is not None:
        manifest = args.motion_manifest.expanduser().resolve()
        if not manifest.is_file():
            raise SystemExit(f"Motion manifest not found: {manifest}")
        overrides = [f"env.data.manifest={manifest}"]
        if args.dataset_path is not None:
            overrides.append(f"env.data.cache_dir={args.dataset_path.resolve()}")
        return overrides
    # Neither given: let the task's own default data configuration stand.
    return []


def discover_ranks(args: argparse.Namespace) -> list[int]:
    """Ranks from the reference-array sidecar, without starting Isaac."""
    if args.ranks is not None:
        return list(args.ranks)
    if args.reference_arrays is None:
        raise SystemExit(
            "--ranks is required unless --reference_arrays is given, because the "
            "trajectory count cannot be read without starting Isaac. Use "
            "compare_policy_reference.py --list_trajectories to see them."
        )
    sidecar = json.loads(
        (args.reference_arrays / "reference_arrays_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    count = len(sidecar["traj_info"]["ordered_traj_list"])
    ranks = list(range(count))
    if args.max_motions is not None:
        ranks = ranks[: args.max_motions]
    elif count > 32:
        raise SystemExit(
            f"The source holds {count} trajectories. Renders are minutes each, so "
            "pass --ranks or --max_motions rather than starting a run that would "
            "not finish."
        )
    return ranks


def motion_names(args: argparse.Namespace) -> dict[int, str]:
    if args.reference_arrays is None:
        return {}
    sidecar = json.loads(
        (args.reference_arrays / "reference_arrays_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        rank: str(entry[1])
        for rank, entry in enumerate(sidecar["traj_info"]["ordered_traj_list"])
    }


def video_paths(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.mp4"))


def probe(path: Path) -> str:
    """Frames and duration, when ffprobe is available. Never fatal."""
    if shutil.which("ffprobe") is None:
        return ""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout.split()
        if len(out) >= 2:
            return f"{out[0]} frames, {float(out[1]):.1f}s"
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""
    return ""


def main(argv: list[str] | None = None) -> int:
    args, extra = parse_args(argv)
    repo = find_repo_root()
    compare = repo / COMPARE_SCRIPT
    if not compare.is_file():
        raise SystemExit(f"Missing {compare}")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    planner_inputs = (
        args.planner_checkpoint,
        args.skill_checkpoint,
        args.language_embeddings,
    )
    if any(value is not None for value in planner_inputs) and not all(
        value is not None for value in planner_inputs
    ):
        raise SystemExit(
            "M3 rendering requires --planner_checkpoint, --skill_checkpoint, "
            "and --language_embeddings together."
        )
    planner_checkpoint = (
        args.planner_checkpoint.expanduser().resolve()
        if args.planner_checkpoint is not None
        else None
    )
    skill_checkpoint = (
        args.skill_checkpoint.expanduser().resolve()
        if args.skill_checkpoint is not None
        else None
    )
    language_embeddings = (
        args.language_embeddings.expanduser().resolve()
        if args.language_embeddings is not None
        else None
    )
    for label, artifact in (
        ("planner checkpoint", planner_checkpoint),
        ("skill checkpoint", skill_checkpoint),
        ("language embeddings", language_embeddings),
    ):
        if artifact is not None and not artifact.is_file():
            raise SystemExit(f"{label.title()} not found: {artifact}")

    # Validate the source BEFORE reading anything out of it, so a missing
    # sidecar reports the actionable message instead of a raw FileNotFoundError.
    overrides = source_overrides(args) + list(extra)
    ranks = discover_ranks(args)
    names = motion_names(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TERM": os.environ.get("TERM", "xterm"),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }

    print(f"[INFO] checkpoint : {checkpoint}")
    print(f"[INFO] agent cfg  : {args.agent_entry_point}")
    if planner_checkpoint is not None:
        print(f"[INFO] planner    : {planner_checkpoint}")
        print(f"[INFO] skill      : {skill_checkpoint}")
        print(f"[INFO] language   : {language_embeddings}")
    print(f"[INFO] ranks      : {ranks}")
    print(
        f"[INFO] horizon    : "
        f"{'to trajectory end' if args.video_seconds is None else f'{args.video_seconds}s cap'}"
    )
    print(
        f"[INFO] terminations: {'kept' if args.keep_terminations else 'disabled (diagnostic pass)'}"
    )
    print(
        "[INFO] randomization: "
        + (
            "startup/reset kept; interval push disabled"
            if args.randomized_no_push
            else "disabled (deterministic environment)"
        )
    )
    print()

    results: list[tuple[int, str, Path | None, str]] = []
    for rank in ranks:
        out = args.output_root / f"rank{rank}"
        label = names.get(rank, f"rank{rank}")
        if args.skip_existing and video_paths(out):
            print(f"[SKIP] rank {rank} ({label}) already rendered")
            results.append((rank, label, video_paths(out)[0], "skipped"))
            continue

        cmd = [
            "pixi",
            "run",
            "-e",
            "isaaclab",
            "python",
            "-u",
            str(compare),
            "--task",
            args.task,
            "--algo",
            args.algo,
            "--checkpoint",
            str(checkpoint),
            "--agent_entry_point",
            args.agent_entry_point,
            "--policy_trajectory_rank",
            str(rank),
            "--policy_start_step",
            "0",
            # A value, not a flag -- as a flag it eats the next argument.
            "--reference_visualization",
            args.reference_visualization,
            "--video",
            "--seed",
            str(args.seed),
            "--headless",
            "--output_dir",
            str(out),
            "--metrics_json",
            str(out / "metrics.json"),
            "--kit_args=--/app/extensions/fsWatcherEnabled=false",
        ]
        if args.keep_terminations:
            cmd.append("--keep_terminations")
        if args.randomized_no_push:
            cmd += ["--keep_domain_randomization", "--disable_push_event"]
        # Deliberately no --video_length: each motion runs to its own end.
        if args.video_seconds is not None:
            cmd += ["--video_seconds", str(args.video_seconds)]
        if planner_checkpoint is not None:
            goal_name = names.get(rank)
            if not goal_name:
                raise SystemExit(
                    "M3 rendering needs a reference-array sidecar so every rank "
                    "can be bound to its explicit language goal."
                )
            cmd += [
                "--latent_temporal_ensemble",
                str(args.latent_temporal_ensemble),
                "--latent_temporal_ensemble_decay",
                str(args.latent_temporal_ensemble_decay),
                "--latent_temporal_clip_std",
                str(args.latent_temporal_clip_std),
                "--latent_temporal_gate_distance",
                str(args.latent_temporal_gate_distance),
                "--latent_temporal_gate_cosine",
                str(args.latent_temporal_gate_cosine),
                "agent.ipmd.command_source=skill_commander",
                f"agent.ipmd.hl_skill_checkpoint_path={skill_checkpoint}",
                f"agent.ipmd.skill_commander_checkpoint_path={planner_checkpoint}",
                f"agent.ipmd.skill_commander_embeddings_path={language_embeddings}",
                f"agent.ipmd.skill_commander_goal_name={goal_name}",
                "agent.ipmd.skill_commander_use_achieved_state=true",
                "agent.ipmd.skill_commander_flow_num_inference_steps=16",
                "agent.ipmd.skill_commander_flow_inference_noise_std=0.0",
            ]
        # Keep all argparse flags before Hydra's unknown key=value tokens.
        # argparse.parse_known_args may stop recognizing options after the first
        # unknown positional token, which would silently restore the default
        # H30 execution rule.
        cmd += overrides

        print(f"[RENDER] rank {rank} ({label})")
        if args.dry_run:
            print("         " + " ".join(cmd))
            results.append((rank, label, None, "dry-run"))
            continue

        out.mkdir(parents=True, exist_ok=True)
        log = out / "render.log"
        with log.open("w", encoding="utf-8") as handle:
            code = subprocess.run(
                cmd, cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT
            ).returncode
        found = video_paths(out)
        if code != 0 or not found:
            print(f"         FAILED (exit {code}); see {log}")
            results.append((rank, label, None, f"failed exit={code}"))
            continue
        print(f"         ok  {probe(found[0])}")
        results.append((rank, label, found[0], "ok"))

    print()
    print("=" * 72)
    successful_statuses = ("ok", "skipped", "dry-run")
    ok = sum(1 for *_rest, status in results if status in successful_statuses)
    print(f"{ok}/{len(results)} rendered")
    # Absolute paths, because a remote session cannot hand the file back.
    for rank, label, path, status in results:
        if path is not None:
            print(f"  rank {rank:<4} {label:<44} {path.resolve()}")
        else:
            print(f"  rank {rank:<4} {label:<44} <{status}>")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
