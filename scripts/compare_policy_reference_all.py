#!/usr/bin/env python3
"""Run reference/policy comparison once per expert trajectory.

This is a thin orchestrator around ``scripts/compare_policy_reference.py``.  It
launches one Isaac process per trajectory rank so Gym's video recorder creates a
separate MP4 for every expert trajectory, while the single-trajectory script
remains the source of truth for controller/planner/replay behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "scripts" / "compare_policy_reference.py"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate all expert trajectories with one video per trajectory."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Imitation-G1-Latent-v0",
        help="Isaac Lab task forwarded to compare_policy_reference.py.",
    )
    parser.add_argument(
        "--algo",
        "--algorithm",
        dest="algorithm",
        type=str.upper,
        default="IPMD",
        help="RLOpt algorithm forwarded to compare_policy_reference.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Low-level controller checkpoint forwarded to compare_policy_reference.py.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "LAFAN manifest used to infer trajectory count/names. Defaults to "
            "env.lafan1_manifest_path from forwarded Hydra overrides."
        ),
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Fallback trajectory count when no manifest is available.",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        default="all",
        help="Ranks to evaluate, e.g. 'all', '0,3,7', or '0-9,15'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of selected ranks, useful for smoke tests.",
    )
    parser.add_argument(
        "--policy_start_step",
        type=int,
        default=0,
        help="Start step forwarded for every trajectory.",
    )
    parser.add_argument(
        "--video_seconds",
        type=float,
        default=None,
        help="Optional per-trajectory video duration in seconds.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Optional per-trajectory rollout/video step limit.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional per-trajectory rollout step limit.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for all trajectory eval logs/videos.",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default=str(DEFAULT_EVAL_SCRIPT),
        help="Path to the single-trajectory compare script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help=(
            "Python executable used for child evals. Defaults to this wrapper's "
            "interpreter; normally run the wrapper with `pixi run -e isaaclab python`."
        ),
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        default=False,
        help="Do not request videos from the child evals.",
    )
    parser.add_argument(
        "--no_headless",
        action="store_true",
        default=False,
        help="Do not add --headless to child evals.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Forward --real-time to child evals.",
    )
    parser.add_argument(
        "--keep_terminations",
        action="store_true",
        default=False,
        help="Forward --keep_terminations to child evals.",
    )
    parser.add_argument(
        "--keep_rewards",
        action="store_true",
        default=False,
        help="Forward --keep_rewards to child evals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed forwarded to child evals.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip a rank if its expected MP4 already exists under output_root.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume an interrupted output_root by skipping ranks that already have "
            "a successful runs.jsonl entry and an existing video, without appending "
            "duplicate skipped rows."
        ),
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Continue evaluating later ranks after a child process fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print child commands and write no videos.",
    )
    return parser.parse_known_args()


def _hydra_override(extra_args: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for token in extra_args:
        if token.startswith(prefix):
            value = token[len(prefix) :]
            return value.strip().strip("'\"")
    return None


def _extract_manifest_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        entries = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
        if entries is None:
            entries = data.get("lafan1_csv", data.get("motions"))
    else:
        entries = data
    if not isinstance(entries, list):
        raise ValueError("Manifest does not contain a trajectory list.")
    return [entry for entry in entries if isinstance(entry, dict)]


def _load_manifest(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    manifest_path = Path(path).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _extract_manifest_entries(data)


def _parse_rank_spec(spec: str, *, count: int) -> list[int]:
    spec = str(spec).strip().lower()
    if spec in ("", "all"):
        return list(range(count))
    ranks: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending rank range: {chunk!r}.")
            ranks.extend(range(start, end + 1))
        else:
            ranks.append(int(chunk))
    unique = list(dict.fromkeys(ranks))
    bad = [rank for rank in unique if rank < 0 or rank >= count]
    if bad:
        raise ValueError(f"Ranks out of range [0, {count - 1}]: {bad}.")
    return unique


def _sanitize(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return clean.strip("._-") or "trajectory"


def _motion_name_for_rank(rank: int, entries: list[dict[str, Any]]) -> str:
    if 0 <= rank < len(entries):
        entry = entries[rank]
        if entry.get("name"):
            return str(entry["name"])
        path_value = entry.get("path") or entry.get("file")
        if path_value:
            return Path(str(path_value)).stem
    return f"rank_{rank:04d}"


def _load_phrase_map(language_embeddings_path: str | None) -> dict[str, str]:
    if not language_embeddings_path:
        return {}
    table_path = Path(language_embeddings_path).expanduser()
    if not table_path.is_file():
        return {}
    try:
        import torch
    except ImportError:
        return {}
    table = torch.load(table_path, map_location="cpu", weights_only=False)
    names = table.get("names")
    phrases = table.get("phrases")
    if not isinstance(names, list) or not isinstance(phrases, list):
        return {}
    return {str(name): str(phrase) for name, phrase in zip(names, phrases)}


def _output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return Path(args.output_root).expanduser().resolve()
    task_name = args.task.split(":")[-1]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        REPO_ROOT
        / "logs"
        / "rlopt_eval"
        / "compare_policy_reference_all"
        / task_name
        / timestamp
    ).resolve()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _completed_ranks_from_runs(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.is_file():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}.") from exc
        if row.get("status") != "ok" or not row.get("video_exists", True):
            continue
        video_path = Path(str(row.get("video_path", ""))).expanduser()
        if video_path.is_file():
            completed.add(int(row["rank"]))
    return completed


def _build_child_cmd(
    args: argparse.Namespace,
    extra_args: list[str],
    *,
    rank: int,
    run_dir: Path,
) -> list[str]:
    cmd = [
        str(Path(args.python).expanduser()),
        str(Path(args.eval_script).expanduser().resolve()),
        "--task",
        args.task,
        "--algo",
        args.algorithm,
        "--checkpoint",
        args.checkpoint,
        "--policy_trajectory_rank",
        str(rank),
        "--policy_start_step",
        str(args.policy_start_step),
        "--output_dir",
        str(run_dir),
    ]
    if not args.no_video:
        cmd.append("--video")
    if not args.no_headless:
        cmd.append("--headless")
    if args.video_seconds is not None:
        cmd.extend(["--video_seconds", str(args.video_seconds)])
    if args.video_length is not None:
        cmd.extend(["--video_length", str(args.video_length)])
    if args.max_steps is not None:
        cmd.extend(["--max_steps", str(args.max_steps)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.real_time:
        cmd.append("--real-time")
    if args.keep_terminations:
        cmd.append("--keep_terminations")
    if args.keep_rewards:
        cmd.append("--keep_rewards")
    cmd.extend(extra_args)
    return cmd


def main() -> int:
    args, extra_args = _parse_args()
    manifest_path = args.manifest or _hydra_override(
        extra_args, "env.lafan1_manifest_path"
    )
    entries = _load_manifest(manifest_path)
    if entries:
        count = len(entries)
    elif args.num_trajectories is not None:
        count = int(args.num_trajectories)
    else:
        raise SystemExit(
            "Could not infer trajectory count. Pass --manifest, forward "
            "env.lafan1_manifest_path=..., or set --num_trajectories."
        )
    if count <= 0:
        raise SystemExit("Trajectory count must be positive.")

    ranks = _parse_rank_spec(args.ranks, count=count)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be > 0 when provided.")
        ranks = ranks[: int(args.limit)]

    root = _output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    runs_path = root / "runs.jsonl"
    language_path = _hydra_override(
        extra_args, "agent.ipmd.skill_commander_embeddings_path"
    )
    phrase_map = _load_phrase_map(language_path)

    summary = {
        "task": args.task,
        "algorithm": args.algorithm,
        "checkpoint": args.checkpoint,
        "manifest": manifest_path,
        "language_embeddings": language_path,
        "ranks": ranks,
        "num_selected": len(ranks),
        "num_trajectories": count,
        "extra_args": extra_args,
        "output_root": str(root),
    }
    (root / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    resume_completed: set[int] = set()
    if args.resume:
        try:
            resume_completed = _completed_ranks_from_runs(runs_path)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    print(f"[INFO] Writing all-trajectory eval to: {root}")
    print(f"[INFO] Selected ranks: {ranks}")
    if resume_completed:
        visible = sorted(rank for rank in ranks if rank in resume_completed)
        print(f"[INFO] Resume will skip completed ranks: {visible}")
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    failures = 0
    for index, rank in enumerate(ranks, start=1):
        motion = _motion_name_for_rank(rank, entries)
        phrase = phrase_map.get(motion)
        label = f"rank_{rank:04d}_{_sanitize(motion)}"
        run_dir = root / label
        video_path = (
            run_dir / "videos" / "compare_policy_reference" / "rl-video-step-0.mp4"
        )
        cmd = _build_child_cmd(args, extra_args, rank=rank, run_dir=run_dir)
        row: dict[str, Any] = {
            "rank": rank,
            "motion": motion,
            "language_phrase": phrase,
            "run_dir": str(run_dir),
            "video_path": str(video_path),
            "command": cmd,
        }

        print(
            f"[INFO] ({index}/{len(ranks)}) rank={rank} motion={motion!r} "
            f"language={phrase!r}"
        )
        if args.resume and rank in resume_completed and video_path.is_file():
            print(f"[INFO] Resume skipping completed rank with existing video: {video_path}")
            continue
        if args.skip_existing and video_path.is_file():
            print(f"[INFO] Skipping existing video: {video_path}")
            row.update({"status": "skipped", "returncode": 0})
            _append_jsonl(runs_path, row)
            continue
        if args.dry_run:
            print("[DRY-RUN] " + " ".join(cmd))
            row.update({"status": "dry_run", "returncode": 0})
            _append_jsonl(runs_path, row)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        row.update(
            {
                "status": "ok" if completed.returncode == 0 else "failed",
                "returncode": int(completed.returncode),
                "video_exists": video_path.is_file(),
            }
        )
        _append_jsonl(runs_path, row)
        if completed.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                print(f"[ERROR] Child eval failed for rank {rank}.")
                return completed.returncode

    if failures > 0:
        print(f"[WARN] Completed with {failures} failed trajectory evals.")
        return 1
    print(f"[INFO] Completed {len(ranks)} trajectory evals. Run manifest: {runs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
