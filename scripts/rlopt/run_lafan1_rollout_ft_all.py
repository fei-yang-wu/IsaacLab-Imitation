#!/usr/bin/env python3
"""Run oracle-rollout finetune and closed-loop eval for LAFAN1 trajectories."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each selected LAFAN1 trajectory: collect oracle-z achieved-state "
            "samples, finetune a SkillCommander, then run planner-driven eval."
        )
    )
    parser.add_argument("--task", default="Isaac-Imitation-G1-Latent-v0")
    parser.add_argument("--algorithm", default="IPMD")
    parser.add_argument(
        "--checkpoint",
        default=(
            "logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/"
            "2026-06-11_23-21-31/models/model_step_4600037376.pt"
        ),
        help="Low-level controller checkpoint.",
    )
    parser.add_argument(
        "--planner_checkpoint",
        default=(
            "logs/language_skill_generator/"
            "2026-06-18_flow_matching_lafan1_w25_z256_full/checkpoints/latest.pt"
        ),
        help="Base language-conditioned SkillCommander checkpoint.",
    )
    parser.add_argument(
        "--skill_checkpoint",
        default=(
            "logs/hl_skill_diffsr/"
            "lafan1_w25_z256_seed0_intermediate_pipeline_20260611_161049/"
            "checkpoints/latest.pt"
        ),
        help="Matching frozen skill encoder/DiffSR checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data/lafan1/manifests/g1_lafan1_manifest.json"),
        help="LAFAN1 manifest used for trajectory names and env loading.",
    )
    parser.add_argument(
        "--dataset_path",
        default=str(REPO_ROOT / "data/lafan1/g1"),
        help="Existing cached LAFAN1 Zarr dataset path.",
    )
    parser.add_argument(
        "--language_embeddings",
        default=str(REPO_ROOT / "data/lafan1/language/g1_lafan1_name_embeddings.pt"),
        help="Motion-name language embedding table.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Output root. Defaults to "
            "logs/planner_robustness/<timestamp>_lafan1_rollout_ft_all."
        ),
    )
    parser.add_argument(
        "--ranks",
        default="all",
        help="Trajectory ranks: all, comma list, or ranges like 0,3-5.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--metric_interval", type=int, default=1)
    parser.add_argument(
        "--video_length",
        type=int,
        default=500,
        help="Video length in env steps. 500 steps is about 10 seconds.",
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--no_video", action="store_true", default=False)
    parser.add_argument("--no_headless", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--pixi", default="pixi")
    parser.add_argument("--isaaclab_env", default="isaaclab")
    parser.add_argument("--latent_dim", type=int, default=386)
    parser.add_argument("--code_latent_dim", type=int, default=384)
    parser.add_argument("--latent_steps", type=int, default=25)
    parser.add_argument("--horizon_steps", type=int, default=25)
    parser.add_argument("--command_mode", default="z_phi")
    parser.add_argument("--command_phase_mode", default="sin_cos")
    parser.add_argument("--finetune_updates", type=int, default=2000)
    parser.add_argument("--finetune_batch_size", type=int, default=256)
    parser.add_argument("--finetune_lr", type=float, default=1.0e-4)
    parser.add_argument("--finetune_flow_loss_coeff", type=float, default=1.0)
    parser.add_argument("--finetune_endpoint_loss_coeff", type=float, default=1.0)
    parser.add_argument("--flow_num_inference_steps", type=int, default=16)
    parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (REPO_ROOT / value).resolve()


def _output_root(path: str | None) -> Path:
    if path:
        return _resolve(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        REPO_ROOT / "logs" / "planner_robustness" / f"{timestamp}_lafan1_rollout_ft_all"
    ).resolve()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if entries is None:
        entries = data.get("lafan1_csv", data.get("motions"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no LAFAN1 entries: {path}")
    return entries


def _motion_name(entry: dict[str, Any], rank: int) -> str:
    if entry.get("name"):
        return str(entry["name"])
    path = entry.get("path") or entry.get("file")
    if path:
        return Path(str(path)).stem
    return f"rank_{rank:04d}"


def _parse_ranks(spec: str, count: int) -> list[int]:
    raw = str(spec).strip().lower()
    if raw == "all":
        return list(range(count))
    ranks: list[int] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid rank range: {item}")
            ranks.extend(range(start, end + 1))
        else:
            ranks.append(int(item))
    unique = list(dict.fromkeys(ranks))
    bad = [rank for rank in unique if rank < 0 or rank >= count]
    if bad:
        raise ValueError(f"Ranks out of range [0, {count - 1}]: {bad}")
    return unique


def _sanitize(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return clean.strip("._-") or "trajectory"


def _load_phrase_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        import torch
    except ImportError:
        return {}
    table = torch.load(path, map_location="cpu", weights_only=False)
    names = table.get("names")
    phrases = table.get("phrases")
    if not isinstance(names, list) or not isinstance(phrases, list):
        return {}
    return {str(name): str(phrase) for name, phrase in zip(names, phrases)}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _common_eval_args(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        "--task",
        str(args.task),
        "--algorithm",
        str(args.algorithm),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--metric_interval",
        str(args.metric_interval),
        "--motion_name",
        str(paths["motion_name"]),
        "--skill_checkpoint",
        str(paths["skill_checkpoint"]),
        "--language_embeddings",
        str(paths["language_embeddings"]),
        "--flow_num_inference_steps",
        str(args.flow_num_inference_steps),
        "--flow_inference_noise_std",
        str(args.flow_inference_noise_std),
    ]


def _common_hydra_args(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        f"env.lafan1_manifest_path={paths['manifest']}",
        f"env.dataset_path={paths['dataset_path']}",
        "env.refresh_zarr_dataset=False",
        f"env.latent_command_dim={args.latent_dim}",
        f"agent.ipmd.latent_dim={args.latent_dim}",
        f"agent.ipmd.latent_steps_min={args.latent_steps}",
        f"agent.ipmd.latent_steps_max={args.latent_steps}",
        f"agent.ipmd.hl_skill_horizon_steps={args.horizon_steps}",
        f"agent.ipmd.hl_skill_command_mode={args.command_mode}",
        f"agent.ipmd.latent_learning.command_phase_mode={args.command_phase_mode}",
        f"agent.ipmd.latent_learning.code_latent_dim={args.code_latent_dim}",
        f"agent.ipmd.latent_learning.code_period={args.latent_steps}",
    ]


def _collect_cmd(
    args: argparse.Namespace,
    paths: dict[str, Path],
    output_dir: Path,
) -> list[str]:
    cmd = [
        str(args.pixi),
        "run",
        "-e",
        str(args.isaaclab_env),
        "python",
        "scripts/rlopt/eval_skill_commander_closed_loop.py",
        *(_common_eval_args(args, paths)),
        "--planner_checkpoint",
        str(paths["planner_checkpoint"]),
        "--output_dir",
        str(output_dir),
        "--save_rollout_training_samples",
    ]
    if not args.no_headless:
        cmd.insert(6, "--headless")
    if args.max_steps > 0:
        cmd.extend(["--max_steps", str(args.max_steps)])
    cmd.extend(
        [
            *_common_hydra_args(args, paths),
            "agent.ipmd.command_source=hl_skill",
            f"agent.ipmd.hl_skill_checkpoint_path={paths['skill_checkpoint']}",
        ]
    )
    return cmd


def _finetune_cmd(
    args: argparse.Namespace,
    paths: dict[str, Path],
    samples_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(args.pixi),
        "run",
        "python",
        "scripts/rlopt/finetune_skill_commander_rollout.py",
        "--checkpoint",
        str(paths["planner_checkpoint"]),
        "--samples_dir",
        str(samples_dir),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--num_updates",
        str(args.finetune_updates),
        "--batch_size",
        str(args.finetune_batch_size),
        "--lr",
        str(args.finetune_lr),
        "--flow_loss_coeff",
        str(args.finetune_flow_loss_coeff),
        "--endpoint_loss_coeff",
        str(args.finetune_endpoint_loss_coeff),
        "--flow_num_inference_steps",
        str(args.flow_num_inference_steps),
        "--flow_inference_noise_std",
        str(args.flow_inference_noise_std),
    ]


def _eval_cmd(
    args: argparse.Namespace,
    paths: dict[str, Path],
    planner_checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    cmd = [
        str(args.pixi),
        "run",
        "-e",
        str(args.isaaclab_env),
        "python",
        "scripts/rlopt/eval_skill_commander_closed_loop.py",
        *(_common_eval_args(args, paths)),
        "--planner_checkpoint",
        str(planner_checkpoint),
        "--output_dir",
        str(output_dir),
    ]
    insert_at = 6
    if not args.no_headless:
        cmd.insert(insert_at, "--headless")
        insert_at += 1
    if not args.no_video:
        cmd.insert(insert_at, "--video")
    if not args.no_video and args.video_length > 0:
        cmd.extend(["--video_length", str(args.video_length)])
    if args.max_steps > 0:
        cmd.extend(["--max_steps", str(args.max_steps)])
    cmd.extend(
        [
            *_common_hydra_args(args, paths),
            "agent.ipmd.command_source=skill_commander",
            f"agent.ipmd.skill_commander_checkpoint_path={planner_checkpoint}",
            f"agent.ipmd.skill_commander_embeddings_path={paths['language_embeddings']}",
            "agent.ipmd.skill_commander_use_achieved_state=true",
            f"agent.ipmd.skill_commander_flow_num_inference_steps={args.flow_num_inference_steps}",
            f"agent.ipmd.skill_commander_flow_inference_noise_std={args.flow_inference_noise_std}",
        ]
    )
    return cmd


def _run_stage(cmd: list[str], log_path: Path, *, cwd: Path, dry_run: bool) -> int:
    print("[CMD] " + " ".join(cmd))
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("WANDB_MODE", "offline")
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("[CMD] " + " ".join(cmd) + "\n")
        stream.flush()
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _stage_complete(path: Path, required_file: str) -> bool:
    return (path / required_file).is_file()


def main() -> int:
    args = _parse_args()
    manifest = _resolve(args.manifest)
    dataset_path = _resolve(args.dataset_path)
    language_embeddings = _resolve(args.language_embeddings)
    checkpoint = _resolve(args.checkpoint)
    planner_checkpoint = _resolve(args.planner_checkpoint)
    skill_checkpoint = _resolve(args.skill_checkpoint)
    root = _output_root(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    for label, path in (
        ("manifest", manifest),
        ("dataset_path", dataset_path),
        ("language_embeddings", language_embeddings),
        ("checkpoint", checkpoint),
        ("planner_checkpoint", planner_checkpoint),
        ("skill_checkpoint", skill_checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    entries = _load_manifest(manifest)
    ranks = _parse_ranks(args.ranks, count=len(entries))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0.")
        ranks = ranks[: int(args.limit)]
    phrase_map = _load_phrase_map(language_embeddings)
    runs_path = root / "runs.jsonl"
    manifest_payload = {
        "task": args.task,
        "algorithm": args.algorithm,
        "checkpoint": str(checkpoint),
        "planner_checkpoint": str(planner_checkpoint),
        "skill_checkpoint": str(skill_checkpoint),
        "manifest": str(manifest),
        "dataset_path": str(dataset_path),
        "language_embeddings": str(language_embeddings),
        "ranks": ranks,
        "num_selected": len(ranks),
        "num_trajectories": len(entries),
        "args": vars(args),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Output root: {root}")
    print(f"[INFO] Selected ranks: {ranks}")

    failures = 0
    for index, rank in enumerate(ranks, start=1):
        motion = _motion_name(entries[rank], rank)
        phrase = phrase_map.get(motion)
        label = f"rank_{rank:04d}_{_sanitize(motion)}"
        run_dir = root / label
        collect_dir = run_dir / "oracle_collect"
        finetune_dir = run_dir / "finetune"
        eval_dir = run_dir / "planner_eval"
        finetuned_checkpoint = finetune_dir / "checkpoints" / "latest.pt"
        video_path = eval_dir / "videos" / "play" / "rl-video-step-0.mp4"
        paths = {
            "manifest": manifest,
            "dataset_path": dataset_path,
            "language_embeddings": language_embeddings,
            "checkpoint": checkpoint,
            "planner_checkpoint": planner_checkpoint,
            "skill_checkpoint": skill_checkpoint,
            "motion_name": Path(motion),
        }
        row: dict[str, Any] = {
            "rank": int(rank),
            "motion": motion,
            "language_phrase": phrase,
            "run_dir": str(run_dir),
            "collect_dir": str(collect_dir),
            "finetune_dir": str(finetune_dir),
            "eval_dir": str(eval_dir),
            "video_path": str(video_path),
        }
        print(
            f"[INFO] ({index}/{len(ranks)}) rank={rank} "
            f"motion={motion!r} language={phrase!r}"
        )

        if (
            args.resume
            and _stage_complete(eval_dir, "summary.json")
            and (args.no_video or video_path.is_file())
        ):
            row.update({"status": "skipped_resume"})
            _append_jsonl(runs_path, row)
            print(f"[INFO] Resume skip: {eval_dir}")
            continue

        collect_cmd = _collect_cmd(args, paths, collect_dir)
        finetune_cmd = _finetune_cmd(
            args,
            paths,
            collect_dir / "rollout_training_samples",
            finetune_dir,
        )
        eval_cmd = _eval_cmd(args, paths, finetuned_checkpoint, eval_dir)
        row["commands"] = {
            "collect": collect_cmd,
            "finetune": finetune_cmd,
            "eval": eval_cmd,
        }

        status = "ok"
        for stage, cmd, stage_dir, required in (
            ("collect", collect_cmd, collect_dir, "summary.json"),
            ("finetune", finetune_cmd, finetune_dir, "summary.json"),
            ("eval", eval_cmd, eval_dir, "summary.json"),
        ):
            if args.resume and _stage_complete(stage_dir, required):
                print(f"[INFO] Resume skip stage {stage}: {stage_dir}")
                row[f"{stage}_returncode"] = 0
                row[f"{stage}_skipped"] = True
                continue
            rc = _run_stage(
                cmd,
                run_dir / "logs" / f"{stage}.log",
                cwd=REPO_ROOT,
                dry_run=bool(args.dry_run),
            )
            row[f"{stage}_returncode"] = int(rc)
            if rc != 0:
                status = "failed"
                failures += 1
                print(
                    f"[ERROR] Stage {stage} failed for rank {rank}. "
                    f"Log: {run_dir / 'logs' / f'{stage}.log'}"
                )
                break

        row.update(
            {
                "status": "dry_run" if args.dry_run else status,
                "video_exists": bool(video_path.is_file()),
            }
        )
        _append_jsonl(runs_path, row)
        if status != "ok" and not args.continue_on_error:
            return 1

    if failures > 0:
        print(f"[WARN] Completed with {failures} failed trajectories.")
        return 1
    print(f"[INFO] Completed selected trajectories. Runs: {runs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
