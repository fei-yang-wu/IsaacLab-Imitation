#!/usr/bin/env python3
"""Collect multi-motion oracle rollouts and finetune one SkillCommander."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python_bin", default=os.environ.get("PYTHON_BIN", "python"))
    parser.add_argument("--task", default="Isaac-Imitation-G1-Latent-v0")
    parser.add_argument("--algorithm", default="IPMD")
    parser.add_argument(
        "--manifest",
        default="data/lafan1/manifests/g1_lafan1_manifest.json",
        help="LAFAN1 manifest used for motion names and env loading.",
    )
    parser.add_argument(
        "--dataset_path",
        default="data/lafan1/g1_hl_diffsr",
        help="Cached LAFAN1 zarr dataset path.",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Low-level policy checkpoint."
    )
    parser.add_argument(
        "--planner_checkpoint", required=True, help="Base planner checkpoint."
    )
    parser.add_argument(
        "--skill_checkpoint", required=True, help="Frozen skill encoder checkpoint."
    )
    parser.add_argument(
        "--language_embeddings",
        default="",
        help=(
            "Optional language embedding table. Leave empty for no-language "
            "planner checkpoints."
        ),
    )
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--ranks", default="all", help="all, comma list, or ranges like 0,3-5"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seeds", default="0,1,2", help="Comma-separated rollout seeds."
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric_interval", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--chunk_rows", type=int, default=8192)
    parser.add_argument(
        "--delete_raw_samples_after_merge",
        action="store_true",
        default=False,
        help="Remove per-rollout sample_step files after they are merged into chunks.",
    )
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--skip_collect", action="store_true", default=False)
    parser.add_argument("--skip_finetune", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--finetune_updates", type=int, default=20000)
    parser.add_argument("--finetune_batch_size", type=int, default=1024)
    parser.add_argument("--finetune_lr", type=float, default=1.0e-4)
    parser.add_argument("--finetune_flow_loss_coeff", type=float, default=1.0)
    parser.add_argument("--finetune_endpoint_loss_coeff", type=float, default=1.0)
    parser.add_argument("--flow_num_inference_steps", type=int, default=16)
    parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
    parser.add_argument("--eval_ranks", default="all")
    parser.add_argument(
        "--eval_video_ranks",
        default="",
        help="Optional ranks to record video for during per-motion eval.",
    )
    parser.add_argument("--eval_video_length", type=int, default=500)
    parser.add_argument("--eval_metric_interval", type=int, default=1)
    parser.add_argument("--eval_max_steps", type=int, default=0)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument(
        "--command_mode",
        choices=("z", "phi", "z_phi"),
        default="z",
        help="Low-level skill command mode used by the trained IPMD policy.",
    )
    parser.add_argument(
        "--command_code_dim",
        type=int,
        default=None,
        help="Pre-phase command width. Defaults to --z_dim for backward compatibility.",
    )
    parser.add_argument("--horizon_steps", type=int, default=10)
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
        REPO_ROOT
        / "logs"
        / "lafan1_no_language_rollout_ft_merged"
        / f"{timestamp}_lafan1_merged_rollout_ft"
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


def _parse_int_list(spec: str, count: int | None = None) -> list[int]:
    raw = str(spec).strip().lower()
    if not raw:
        return []
    if raw == "all":
        if count is None:
            raise ValueError("'all' requires count.")
        return list(range(count))
    values: list[int] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid range: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    unique = list(dict.fromkeys(values))
    if count is not None:
        bad = [value for value in unique if value < 0 or value >= count]
        if bad:
            raise ValueError(f"Values out of range [0, {count - 1}]: {bad}")
    return unique


def _sanitize(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return clean.strip("._-") or "motion"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _run_cmd(cmd: list[str], log_path: Path, *, dry_run: bool) -> int:
    print("[CMD] " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("ACCEPT_EULA", "Y")
    env.setdefault("PRIVACY_CONSENT", "Y")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("[CMD] " + " ".join(cmd) + "\n")
        stream.flush()
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _sample_paths(sample_dir: Path) -> list[Path]:
    return sorted(sample_dir.glob("sample_step_*.pt")) + sorted(
        sample_dir.glob("sample_chunk_*.pt")
    )


def _latent_overrides(
    args: argparse.Namespace, manifest: Path, dataset_path: Path
) -> list[str]:
    command_code_dim = int(args.command_code_dim or args.z_dim)
    latent_dim = command_code_dim + 2
    return [
        f"env.lafan1_manifest_path={manifest}",
        f"env.dataset_path={dataset_path}",
        "env.refresh_zarr_dataset=false",
        f"env.latent_command_dim={latent_dim}",
        f"agent.ipmd.latent_dim={latent_dim}",
        f"agent.ipmd.hl_skill_horizon_steps={args.horizon_steps}",
        f"agent.ipmd.hl_skill_command_mode={args.command_mode}",
        f"agent.ipmd.latent_steps_min={args.horizon_steps}",
        f"agent.ipmd.latent_steps_max={args.horizon_steps}",
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
        f"agent.ipmd.latent_learning.code_latent_dim={command_code_dim}",
        f"agent.ipmd.latent_learning.code_period={args.horizon_steps}",
        "agent.ipmd.reward_loss_coeff=0.0",
        "agent.ipmd.reward_l2_coeff=0.0",
        "agent.ipmd.reward_grad_penalty_coeff=0.0",
        "agent.ipmd.reward_logit_reg_coeff=0.0",
        "agent.ipmd.reward_param_weight_decay_coeff=0.0",
    ]


def _collect_cmd(
    args: argparse.Namespace,
    *,
    manifest: Path,
    dataset_path: Path,
    checkpoint: Path,
    planner_checkpoint: Path,
    skill_checkpoint: Path,
    language_embeddings: Path | None,
    motion: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    cmd = [
        str(args.python_bin),
        "scripts/rlopt/eval_skill_commander_closed_loop.py",
        "--headless",
        "--device",
        str(args.device),
        "--num_envs",
        str(args.num_envs),
        "--task",
        str(args.task),
        "--algorithm",
        str(args.algorithm),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--skill_checkpoint",
        str(skill_checkpoint),
        "--metric_interval",
        str(args.metric_interval),
        "--flow_num_inference_steps",
        str(args.flow_num_inference_steps),
        "--flow_inference_noise_std",
        str(args.flow_inference_noise_std),
        "--planner_checkpoint",
        str(planner_checkpoint),
        "--output_dir",
        str(output_dir),
        "--motion_name",
        str(motion),
        "--save_rollout_training_samples",
        "--rollout_sample_chunk_rows",
        str(args.chunk_rows),
    ]
    if language_embeddings is not None:
        cmd.extend(["--language_embeddings", str(language_embeddings)])
    if int(args.max_steps) > 0:
        cmd.extend(["--max_steps", str(args.max_steps)])
    cmd.extend(
        [
            "agent.ipmd.command_source=hl_skill",
            f"agent.ipmd.hl_skill_checkpoint_path={skill_checkpoint}",
            "agent.ipmd.hl_skill_finetune_enabled=false",
            *_latent_overrides(args, manifest, dataset_path),
        ]
    )
    return cmd


def _finetune_cmd(
    args: argparse.Namespace,
    *,
    planner_checkpoint: Path,
    samples_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(args.python_bin),
        "scripts/rlopt/finetune_skill_commander_rollout.py",
        "--checkpoint",
        str(planner_checkpoint),
        "--samples_dir",
        str(samples_dir),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(args.seeds.split(",")[0].strip() or 0),
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
    *,
    manifest: Path,
    dataset_path: Path,
    checkpoint: Path,
    skill_checkpoint: Path,
    planner_checkpoint: Path,
    language_embeddings: Path | None,
    motion: str,
    seed: int,
    output_dir: Path,
    video: bool,
) -> list[str]:
    cmd = [
        str(args.python_bin),
        "scripts/rlopt/eval_skill_commander_closed_loop.py",
        "--headless",
        "--device",
        str(args.device),
        "--num_envs",
        "1",
        "--task",
        str(args.task),
        "--algorithm",
        str(args.algorithm),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--skill_checkpoint",
        str(skill_checkpoint),
        "--metric_interval",
        str(args.eval_metric_interval),
        "--flow_num_inference_steps",
        str(args.flow_num_inference_steps),
        "--flow_inference_noise_std",
        str(args.flow_inference_noise_std),
        "--planner_checkpoint",
        str(planner_checkpoint),
        "--output_dir",
        str(output_dir),
        "--motion_name",
        str(motion),
    ]
    if language_embeddings is not None:
        cmd.extend(["--language_embeddings", str(language_embeddings)])
    if int(args.eval_max_steps) > 0:
        cmd.extend(["--max_steps", str(args.eval_max_steps)])
    if video:
        cmd.extend(["--video", "--video_length", str(args.eval_video_length)])
    embeddings_override = (
        f"agent.ipmd.skill_commander_embeddings_path={language_embeddings}"
        if language_embeddings is not None
        else "agent.ipmd.skill_commander_embeddings_path="
    )
    cmd.extend(
        [
            "agent.ipmd.command_source=skill_commander",
            f"agent.ipmd.skill_commander_checkpoint_path={planner_checkpoint}",
            embeddings_override,
            f"agent.ipmd.skill_commander_flow_num_inference_steps={args.flow_num_inference_steps}",
            f"agent.ipmd.skill_commander_flow_inference_noise_std={args.flow_inference_noise_std}",
            "agent.ipmd.skill_commander_use_achieved_state=true",
            "agent.ipmd.hl_skill_finetune_enabled=false",
            *_latent_overrides(args, manifest, dataset_path),
        ]
    )
    return cmd


def _flush_chunk(buffers: dict[str, list[Any]], output_path: Path) -> int:
    import torch

    if not buffers["planner_state"]:
        return 0
    payload: dict[str, torch.Tensor] = {}
    for key, values in buffers.items():
        payload[key] = torch.cat(values, dim=0).contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    row_count = int(payload["planner_state"].shape[0])
    for values in buffers.values():
        values.clear()
    return row_count


def _merge_samples(
    source_dir: Path,
    merged_dir: Path,
    *,
    next_chunk_index: int,
    chunk_rows: int,
) -> tuple[int, int, int]:
    import torch

    paths = _sample_paths(source_dir)
    if not paths:
        return next_chunk_index, 0, 0

    buffers: dict[str, list[torch.Tensor]] = {
        "planner_state": [],
        "expert_planner_state": [],
        "lang": [],
        "z_target": [],
        "traj_rank": [],
        "step": [],
    }
    buffered_rows = 0
    written_rows = 0
    written_chunks = 0

    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        rows: dict[str, torch.Tensor] = {}
        for key in ("planner_state", "expert_planner_state", "lang", "z_target"):
            value = sample[key]
            if value.ndim == 1:
                value = value.unsqueeze(0)
            rows[key] = value.to(dtype=torch.float32).contiguous()
        row_count = int(rows["planner_state"].shape[0])
        traj_rank = sample["traj_rank"]
        if traj_rank.ndim == 0:
            traj_rank = traj_rank.reshape(1)
        traj_rank = traj_rank.reshape(-1).to(dtype=torch.long)
        if int(traj_rank.numel()) == 1 and row_count != 1:
            traj_rank = traj_rank.repeat(row_count)
        if int(traj_rank.numel()) != row_count:
            raise ValueError(f"{path} traj_rank rows do not match sample rows.")
        step = sample["step"]
        if isinstance(step, torch.Tensor):
            step = step.reshape(-1).to(dtype=torch.long)
            if int(step.numel()) == 1 and row_count != 1:
                step = step.repeat(row_count)
            if int(step.numel()) != row_count:
                raise ValueError(f"{path} step rows do not match sample rows.")
        else:
            step = torch.full((row_count,), int(step), dtype=torch.long)
        rows["traj_rank"] = traj_rank
        rows["step"] = step

        for key, value in rows.items():
            buffers[key].append(value)
        buffered_rows += row_count
        while buffered_rows >= int(chunk_rows):
            concat = {key: torch.cat(value, dim=0) for key, value in buffers.items()}
            head = {
                key: value[: int(chunk_rows)].contiguous()
                for key, value in concat.items()
            }
            tail = {
                key: value[int(chunk_rows) :].contiguous()
                for key, value in concat.items()
            }
            for key, value in buffers.items():
                value.clear()
                if int(tail[key].shape[0]) > 0:
                    value.append(tail[key])
            output_path = merged_dir / f"sample_chunk_{next_chunk_index:06d}.pt"
            torch.save(head, output_path)
            next_chunk_index += 1
            written_chunks += 1
            written_rows += int(head["planner_state"].shape[0])
            buffered_rows = int(tail["planner_state"].shape[0])

    if buffered_rows > 0:
        output_path = merged_dir / f"sample_chunk_{next_chunk_index:06d}.pt"
        written_rows += _flush_chunk(buffers, output_path)
        written_chunks += 1
        next_chunk_index += 1
    return next_chunk_index, written_chunks, written_rows


def main() -> int:
    args = _parse_args()
    manifest = _resolve(args.manifest)
    dataset_path = _resolve(args.dataset_path)
    checkpoint = _resolve(args.checkpoint)
    planner_checkpoint = _resolve(args.planner_checkpoint)
    skill_checkpoint = _resolve(args.skill_checkpoint)
    language_embeddings = (
        _resolve(args.language_embeddings)
        if str(args.language_embeddings).strip()
        else None
    )
    root = _output_root(args.output_root)
    merged_dir = root / "merged_rollout_training_samples"
    collect_root = root / "collect"
    finetune_dir = root / "planner_rollout_ft_merged"
    eval_root = root / "eval_finetuned_per_motion"
    runs_path = root / "runs.jsonl"

    for label, path in (
        ("manifest", manifest),
        ("dataset_path", dataset_path),
        ("checkpoint", checkpoint),
        ("planner_checkpoint", planner_checkpoint),
        ("skill_checkpoint", skill_checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if language_embeddings is not None and not language_embeddings.exists():
        raise FileNotFoundError(
            f"language_embeddings does not exist: {language_embeddings}"
        )
    if int(args.chunk_rows) <= 0:
        raise ValueError("--chunk_rows must be > 0.")

    entries = _load_manifest(manifest)
    ranks = _parse_int_list(args.ranks, count=len(entries))
    if args.limit is not None:
        ranks = ranks[: int(args.limit)]
    seeds = _parse_int_list(args.seeds)
    eval_ranks = _parse_int_list(args.eval_ranks, count=len(entries))
    eval_video_ranks = set(_parse_int_list(args.eval_video_ranks, count=len(entries)))
    root.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT", str(root / "unitree_usd_cache")
    )

    manifest_payload = {
        "task": args.task,
        "algorithm": args.algorithm,
        "manifest": str(manifest),
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint),
        "planner_checkpoint": str(planner_checkpoint),
        "skill_checkpoint": str(skill_checkpoint),
        "language_embeddings": (
            str(language_embeddings) if language_embeddings is not None else ""
        ),
        "ranks": ranks,
        "seeds": seeds,
        "eval_ranks": eval_ranks,
        "eval_video_ranks": sorted(eval_video_ranks),
        "args": vars(args),
    }
    _write_json(root / "manifest.json", manifest_payload)
    print(f"[INFO] Output root: {root}", flush=True)
    print(f"[INFO] Collect ranks={ranks} seeds={seeds}", flush=True)

    next_chunk_index = 0
    existing_chunks = sorted(merged_dir.glob("sample_chunk_*.pt"))
    if existing_chunks:
        last = existing_chunks[-1].stem.rsplit("_", 1)[-1]
        next_chunk_index = int(last) + 1

    failures = 0
    if not args.skip_collect:
        for rank in ranks:
            motion = _motion_name(entries[rank], rank)
            label = f"rank_{rank:04d}_{_sanitize(motion)}"
            for seed in seeds:
                run_dir = collect_root / label / f"seed_{seed}"
                collect_dir = run_dir / "oracle_collect"
                sample_dir = collect_dir / "rollout_training_samples"
                merge_marker = run_dir / "merge_complete.json"
                row: dict[str, Any] = {
                    "stage": "collect",
                    "rank": int(rank),
                    "motion": motion,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "collect_dir": str(collect_dir),
                }
                print(
                    f"[INFO] Collect rank={rank} motion={motion!r} seed={seed}",
                    flush=True,
                )
                if args.resume and merge_marker.is_file():
                    row["status"] = "skipped_resume"
                    _append_jsonl(runs_path, row)
                    print(f"[INFO] Resume skip merged run: {run_dir}", flush=True)
                    continue
                has_reusable_samples = bool(args.resume and _sample_paths(sample_dir))
                if not has_reusable_samples:
                    cmd = _collect_cmd(
                        args,
                        manifest=manifest,
                        dataset_path=dataset_path,
                        checkpoint=checkpoint,
                        planner_checkpoint=planner_checkpoint,
                        skill_checkpoint=skill_checkpoint,
                        language_embeddings=language_embeddings,
                        motion=motion,
                        seed=int(seed),
                        output_dir=collect_dir,
                    )
                    rc = _run_cmd(
                        cmd,
                        run_dir / "logs" / "collect.log",
                        dry_run=bool(args.dry_run),
                    )
                    row["collect_returncode"] = int(rc)
                    if rc != 0:
                        row["status"] = "failed_collect"
                        failures += 1
                        _append_jsonl(runs_path, row)
                        if not args.continue_on_error:
                            return 1
                        continue
                elif (collect_dir / "summary.json").is_file():
                    print(
                        f"[INFO] Resume reuse collected samples: {sample_dir}",
                        flush=True,
                    )
                if args.dry_run:
                    row["status"] = "dry_run"
                    _append_jsonl(runs_path, row)
                    continue
                next_chunk_index, chunks, rows = _merge_samples(
                    sample_dir,
                    merged_dir,
                    next_chunk_index=next_chunk_index,
                    chunk_rows=int(args.chunk_rows),
                )
                if rows <= 0:
                    row["status"] = "failed_no_samples"
                    row["merged_chunks"] = int(chunks)
                    row["merged_rows"] = int(rows)
                    failures += 1
                    _append_jsonl(runs_path, row)
                    if not args.continue_on_error:
                        return 1
                    continue
                merge_payload = {
                    "rank": int(rank),
                    "motion": motion,
                    "seed": int(seed),
                    "source_dir": str(sample_dir),
                    "merged_dir": str(merged_dir),
                    "chunks": int(chunks),
                    "rows": int(rows),
                }
                _write_json(merge_marker, merge_payload)
                if args.delete_raw_samples_after_merge:
                    shutil.rmtree(sample_dir)
                row.update(
                    {"status": "ok", "merged_chunks": chunks, "merged_rows": rows}
                )
                _append_jsonl(runs_path, row)

    if args.dry_run:
        return 0 if failures == 0 else 1

    finetuned_checkpoint = finetune_dir / "checkpoints" / "latest.pt"
    if not args.skip_finetune:
        if not any(merged_dir.glob("sample_chunk_*.pt")):
            raise FileNotFoundError(
                f"No merged rollout sample chunks found in {merged_dir}"
            )
        if (
            args.resume
            and (finetune_dir / "summary.json").is_file()
            and finetuned_checkpoint.is_file()
        ):
            print(f"[INFO] Resume skip finetune: {finetune_dir}", flush=True)
        else:
            cmd = _finetune_cmd(
                args,
                planner_checkpoint=planner_checkpoint,
                samples_dir=merged_dir,
                output_dir=finetune_dir,
            )
            rc = _run_cmd(
                cmd, root / "logs" / "finetune.log", dry_run=bool(args.dry_run)
            )
            _append_jsonl(runs_path, {"stage": "finetune", "returncode": int(rc)})
            if rc != 0:
                return 1

    if args.skip_eval:
        return 0 if failures == 0 else 1
    if not finetuned_checkpoint.is_file():
        raise FileNotFoundError(
            f"Finetuned checkpoint not found: {finetuned_checkpoint}"
        )

    for rank in eval_ranks:
        motion = _motion_name(entries[rank], rank)
        label = f"rank_{rank:04d}_{_sanitize(motion)}"
        eval_dir = eval_root / label
        if args.resume and (eval_dir / "summary.json").is_file():
            print(f"[INFO] Resume skip eval: {eval_dir}", flush=True)
            continue
        cmd = _eval_cmd(
            args,
            manifest=manifest,
            dataset_path=dataset_path,
            checkpoint=checkpoint,
            skill_checkpoint=skill_checkpoint,
            planner_checkpoint=finetuned_checkpoint,
            language_embeddings=language_embeddings,
            motion=motion,
            seed=seeds[0] if seeds else 0,
            output_dir=eval_dir,
            video=rank in eval_video_ranks,
        )
        rc = _run_cmd(cmd, eval_dir / "eval.log", dry_run=False)
        _append_jsonl(
            runs_path,
            {
                "stage": "eval",
                "rank": int(rank),
                "motion": motion,
                "returncode": int(rc),
                "eval_dir": str(eval_dir),
            },
        )
        if rc != 0:
            failures += 1
            if not args.continue_on_error:
                return 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
