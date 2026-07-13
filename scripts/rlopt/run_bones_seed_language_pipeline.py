#!/usr/bin/env python3
"""Prepare a curated BONES-SEED subset and run the language-planner pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "bones-studio/seed"
PAPER24_FILENAMES = (
    "Neutral_stoop_down_001__A057",
    "big_heavy_one_hand_front_high_to_front_low_R_001__A524",
    "big_heavy_one_hand_front_low_to_front_high_R_001__A524",
    "big_light_two_hands_pick_up_front_medium_R_001__A509",
    "drinking_standing_mug_R_001__A282",
    "inside_door_handle_left_side_open_walk_close_behind_R_001__A513",
    "inside_door_handle_right_side_open_walk_turn_close_R_001__A514",
    "read_book_both_hands_sitting_R_001__A456",
    "outside_door_knob_left_side_open_walk_switch_hand_close_R_002__A515",
    "outside_door_knob_left_side_open_walk_turn_switch_hand_close_R_001__A513",
    "outside_door_knob_right_side_open_walk_turn_close_R_001__A513",
    "outside_door_knob_right_side_open_walk_switch_hand_close_behind_R_001__A513",
    "inside_door_handle_left_side_open_walk_switch_hand_close_behind_R_001__A515",
    "inside_door_knob_left_side_open_walk_switch_hand_close_behind_R_001__A513",
    "outside_door_handle_left_side_open_walk_turn_close_R_001__A514",
    "outside_door_handle_left_side_open_walk_turn_switch_hand_close_R_001__A515",
    "mid_button_over_push_001__A486",
    "crouch_cupboard_low_out_mid_loot_R_001__A288",
    "heavy_item_cupboard_high_in_R_001__A285",
    "lift_crate_walk_ff_stop_315_003__A143",
    "push_obstacle_180_003__A342",
    "bump_into_small_obstacle_kick_walk_ff_180_R_001__A463",
    "cellphone_selfie_sequence_R_001__A423",
    "Neutral_kick_trash_001__A057",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("paper24",), default="paper24")
    parser.add_argument("--data-root", default="/data/bones_seed_paper24")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="Isaac-Imitation-G1-Latent-v0")
    parser.add_argument("--algorithm", default="IPMD_BILINEAR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-cap", type=int, default=2_000_000_000)
    parser.add_argument("--train-num-envs", type=int, default=4096)
    parser.add_argument("--frames-per-env-batch", type=int, default=24)
    parser.add_argument("--horizon-steps", type=int, default=25)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--diffsr-feature-dim", type=int, default=256)
    parser.add_argument("--pretrain-updates", type=int, default=50_000)
    parser.add_argument("--pretrain-batch-size", type=int, default=8192)
    parser.add_argument("--commander-updates", type=int, default=20_000)
    parser.add_argument("--commander-batch-size", type=int, default=4096)
    parser.add_argument("--finetune-updates", type=int, default=20_000)
    parser.add_argument("--finetune-batch-size", type=int, default=1024)
    parser.add_argument("--rollout-seeds", default="0,1,2")
    parser.add_argument("--eval-video-ranks", default="0,1,2,3,4,5,6,7,16,18,19,20")
    parser.add_argument("--logger-backend", default="wandb")
    parser.add_argument("--wandb-project", default="G1-Imitation-BONES-SEED")
    parser.add_argument("--wandb-group", default="bones_seed_paper24_2b")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--video-interval", type=int, default=20_000)
    parser.add_argument("--save-interval", type=int, default=100_000_000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--skip-skill", action="store_true")
    parser.add_argument("--skip-commander", action="store_true")
    parser.add_argument("--skip-low-level", action="store_true")
    parser.add_argument("--skip-rollout-ft", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (REPO_ROOT / value).resolve()


def _motion_name(filename: str) -> str:
    return filename.replace("__", "_")


def _hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _token() -> str | None:
    value = os.environ.get("HF_TOKEN")
    if value:
        return value.strip()
    for path in (Path.home() / ".hf_token", Path.home() / ".cache/huggingface/token"):
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def _download(url: str, output: Path, *, token: str | None, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0 and not force:
        print(f"[INFO] Reusing download: {output} ({output.stat().st_size} bytes)")
        return
    if force:
        output.unlink(missing_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    mode = "wb"
    if part.is_file() and part.stat().st_size > 0:
        headers["Range"] = f"bytes={part.stat().st_size}-"
        mode = "ab"
    print(f"[INFO] Downloading {url} -> {output}")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = int(getattr(response, "status", 200))
            if status != 206:
                mode = "wb"
            started = time.time()
            last_report = time.time()
            written = part.stat().st_size if mode == "ab" and part.exists() else 0
            with part.open(mode + "") as stream:
                while True:
                    chunk = response.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    if now - last_report > 60:
                        gib = written / (1024**3)
                        rate = written / max(now - started, 1.0) / (1024**2)
                        print(f"[INFO] Download progress {gib:.2f} GiB at {rate:.1f} MiB/s")
                        last_report = now
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed for {url}: HTTP {exc.code}") from exc
    part.replace(output)


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["filename"]: row for row in csv.DictReader(stream)}


def _load_events(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    events: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            filename = str(row.get("filename", "")).removesuffix(".csv")
            if filename:
                events[filename] = row
    return events


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _write_shortlist(
    *,
    filenames: tuple[str, ...],
    metadata_csv: Path,
    events_jsonl: Path,
    output: Path,
) -> None:
    metadata = _load_metadata(metadata_csv)
    events = _load_events(events_jsonl)
    rows: list[dict[str, Any]] = []
    missing = [name for name in filenames if name not in metadata]
    if missing:
        raise ValueError(f"Preset filenames missing from metadata: {missing}")
    for filename in filenames:
        item = metadata[filename]
        event_payload = events.get(filename, {})
        event_rows = event_payload.get("events", [])
        overview = _first_text(
            event_payload.get("overview_description"),
            item.get("content_natural_desc_4"),
            item.get("content_natural_desc_1"),
            item.get("content_short_description"),
            item.get("content_technical_description"),
        )
        rows.append(
            {
                "filename": filename,
                "overview_description": overview,
                "num_events": int(event_payload.get("num_events", len(event_rows))),
                "events": event_rows,
                "propagated_from_filename": event_payload.get("propagated_from_filename"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote shortlist: {output} ({len(rows)} motions)")


def _run(cmd: list[str], log_path: Path, *, dry_run: bool) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("[CMD] " + " ".join(cmd) + "\n")
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("TERM", "xterm")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def _latest_low_level_checkpoint(run_root: Path) -> Path | None:
    model_dir = run_root / "low_level_ipmd_bilinear_2b" / "models"
    if not model_dir.is_dir():
        return None
    paths = sorted(model_dir.glob("model_step_*.pt"))
    return paths[-1] if paths else None


def main() -> None:
    args = _parse_args()
    data_root = _resolve(args.data_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = _resolve(
        args.run_root
        or REPO_ROOT / "logs" / "bones_seed_language" / f"paper24_2b_{timestamp}"
    )
    raw_dir = data_root / "raw"
    metadata_csv = raw_dir / "metadata" / "seed_metadata_v004.csv"
    events_jsonl = raw_dir / "metadata" / "seed_metadata_v002_temporal_labels.jsonl"
    archive = raw_dir / "g1.tar.gz"
    shortlist = data_root / "curated" / "bones_seed_paper24.timeline.json"
    csv_dir = raw_dir / "g1"
    npz_dir = data_root / "npz" / "g1"
    manifest = data_root / "manifests" / "g1_bones_seed_paper24_manifest.json"
    language = data_root / "language" / "g1_bones_seed_paper24_language.json"
    embeddings = data_root / "language" / "g1_bones_seed_paper24_minilm_goal_embeddings.pt"
    dataset_path = data_root / "g1_hl_diffsr"
    skill_dir = run_root / "skill_encoder_h25_z256"
    commander_dir = run_root / "commander_contrastive_20000"
    low_dir = run_root / "low_level_ipmd_bilinear_2b"
    rollout_dir = run_root / "m3_rollout_ft_merged"
    skill_ckpt = skill_dir / "checkpoints" / "latest.pt"
    commander_ckpt = commander_dir / "checkpoints" / "latest.pt"

    token = _token()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "metadata.json").write_text(
        json.dumps(
            {
                "preset": args.preset,
                "data_root": str(data_root),
                "run_root": str(run_root),
                "manifest": str(manifest),
                "dataset_path": str(dataset_path),
                "frame_cap": args.frame_cap,
                "train_num_envs": args.train_num_envs,
                "filenames": PAPER24_FILENAMES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.skip_download:
        _download(
            _hf_url(HF_REPO, "metadata/seed_metadata_v004.csv"),
            metadata_csv,
            token=token,
            force=args.force_download,
        )
        _download(
            _hf_url(HF_REPO, "metadata/seed_metadata_v002_temporal_labels.jsonl"),
            events_jsonl,
            token=token,
            force=args.force_download,
        )
        _download(
            _hf_url(HF_REPO, "g1.tar.gz"),
            archive,
            token=token,
            force=args.force_download,
        )
    _write_shortlist(
        filenames=PAPER24_FILENAMES,
        metadata_csv=metadata_csv,
        events_jsonl=events_jsonl,
        output=shortlist,
    )

    common_env_overrides = [
        f"env.lafan1_manifest_path={manifest}",
        f"env.dataset_path={dataset_path}",
        "env.refresh_zarr_dataset=false",
    ]
    latent_overrides = [
        "env.latent_command_dim=258",
        "agent.ipmd.latent_dim=258",
        "agent.ipmd.hl_skill_horizon_steps=25",
        "agent.ipmd.hl_skill_command_mode=z",
        "agent.ipmd.latent_steps_min=25",
        "agent.ipmd.latent_steps_max=25",
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
        "agent.ipmd.latent_learning.code_latent_dim=256",
        "agent.ipmd.latent_learning.code_period=25",
        "agent.ipmd.reward_loss_coeff=0.0",
        "agent.ipmd.reward_l2_coeff=0.0",
        "agent.ipmd.reward_grad_penalty_coeff=0.0",
        "agent.ipmd.reward_logit_reg_coeff=0.0",
        "agent.ipmd.reward_param_weight_decay_coeff=0.0",
    ]

    if not args.skip_data_prep and not manifest.is_file():
        _run(
            [
                sys.executable,
                "scripts/prepare_bones_seed_subset.py",
                "--shortlist",
                str(shortlist),
                "--archive",
                str(archive),
                "--csv_dir",
                str(csv_dir),
                "--npz_dir",
                str(npz_dir),
                "--manifest_path",
                str(manifest),
                "--language_path",
                str(language),
                "--metadata_csv",
                str(metadata_csv),
                "--headless",
                "--device",
                args.device,
                "--skip_existing",
            ],
            run_root / "logs" / "prepare_data.log",
            dry_run=args.dry_run,
        )

    if not embeddings.is_file():
        _run(
            [
                sys.executable,
                "scripts/rlopt/build_language_goal_embeddings.py",
                "--manifest",
                str(manifest),
                "--language_sidecar",
                str(language),
                "--require_language_sidecar_matches",
                "--backend",
                "sentence-transformer",
                "--output",
                str(embeddings),
            ],
            run_root / "logs" / "build_embeddings.log",
            dry_run=args.dry_run,
        )

    if not args.skip_skill and not skill_ckpt.is_file():
        _run(
            [
                sys.executable,
                "scripts/rlopt/train_hl_skill_diffsr.py",
                "--headless",
                "--device",
                args.device,
                "--task",
                args.task,
                "--num_envs",
                "16",
                "--seed",
                str(args.seed),
                "--output_dir",
                str(skill_dir),
                "--horizon_steps",
                str(args.horizon_steps),
                "--encoder_window_mode",
                "intermediate",
                "--z_dim",
                str(args.z_dim),
                "--latent_mode",
                "deterministic",
                "--reg_coeff",
                "1.0e-3",
                "--diffsr_feature_dim",
                str(args.diffsr_feature_dim),
                "--diffsr_embed_dim",
                "512",
                "--batch_size",
                str(args.pretrain_batch_size),
                "--num_updates",
                str(args.pretrain_updates),
                "--log_interval",
                "100",
                "--eval_batches",
                "4",
                "--eval_batch_size",
                str(args.pretrain_batch_size),
                "--train_split",
                "all",
                "--eval_split",
                "all",
                "--logger_backend",
                args.logger_backend,
                "--wandb_project",
                args.wandb_project,
                "--wandb_group",
                args.wandb_group,
                "--wandb_run_name",
                f"{run_root.name}_skill_encoder",
                *common_env_overrides,
            ],
            skill_dir / "train.log",
            dry_run=args.dry_run,
        )

    if not args.skip_commander and not commander_ckpt.is_file():
        _run(
            [
                sys.executable,
                "scripts/rlopt/train_skill_commander.py",
                "--headless",
                "--device",
                args.device,
                "--task",
                args.task,
                "--num_envs",
                "16",
                "--seed",
                str(args.seed),
                "--output_dir",
                str(commander_dir),
                "--skill_checkpoint",
                str(skill_ckpt),
                "--language_embeddings",
                str(embeddings),
                "--planner_type",
                "mlp",
                "--generator_hidden_dims",
                "1024",
                "1024",
                "512",
                "--batch_size",
                str(args.commander_batch_size),
                "--num_updates",
                str(args.commander_updates),
                "--log_interval",
                "100",
                "--eval_batches",
                "4",
                "--eval_batch_size",
                str(args.commander_batch_size),
                "--train_split",
                "all",
                "--eval_split",
                "all",
                "--lr",
                "5.0e-4",
                "--cosine_loss_coeff",
                "1.0",
                "--z_norm_coeff",
                "1.0e-4",
                "--state_feature_dropout_prob",
                "0.7",
                "--state_feature_dropout_terms",
                "expert_motion",
                "--state_feature_dropout_mode",
                "shuffle",
                "--language_contrastive_coeff",
                "5.0",
                "--language_contrastive_margin",
                "0.1",
                *common_env_overrides,
            ],
            commander_dir / "train.log",
            dry_run=args.dry_run,
        )

    frames_per_batch = int(args.train_num_envs) * int(args.frames_per_env_batch)
    max_iterations = max(1, int(args.frame_cap) // frames_per_batch)
    low_ckpt = _latest_low_level_checkpoint(run_root)
    if not args.skip_low_level and low_ckpt is None:
        _run(
            [
                sys.executable,
                "scripts/rlopt/train.py",
                "--headless",
                "--video",
                "--video_length",
                str(args.video_length),
                "--video_interval",
                str(args.video_interval),
                "--device",
                args.device,
                "--num_envs",
                str(args.train_num_envs),
                "--task",
                args.task,
                "--algo",
                args.algorithm,
                "--seed",
                str(args.seed),
                "--max_iterations",
                str(max_iterations),
                "--log_interval",
                "1000000",
                f"agent.logger.backend={args.logger_backend}",
                f"agent.logger.project_name={args.wandb_project}",
                f"agent.logger.exp_name={args.exp_name or run_root.name}",
                f"agent.logger.group_name={args.wandb_group}",
                "agent.logger.video=true",
                f"agent.save_interval={args.save_interval}",
                "agent.ipmd.command_source=hl_skill",
                f"agent.ipmd.hl_skill_checkpoint_path={skill_ckpt}",
                "agent.ipmd.hl_skill_finetune_enabled=false",
                *latent_overrides,
                *common_env_overrides,
            ],
            low_dir / "train.log",
            dry_run=args.dry_run,
        )
        low_ckpt = _latest_low_level_checkpoint(run_root)
    if args.dry_run and low_ckpt is None:
        low_ckpt = low_dir / "models" / "model_step_DRYRUN.pt"

    if args.skip_rollout_ft:
        print("[INFO] Skipping rollout finetune/eval.")
        return
    if low_ckpt is None:
        raise FileNotFoundError(f"No low-level checkpoint found under {low_dir / 'models'}")
    _run(
        [
            sys.executable,
            "scripts/rlopt/run_lafan1_no_language_rollout_ft_merged.py",
            "--python_bin",
            sys.executable,
            "--task",
            args.task,
            "--algorithm",
            args.algorithm,
            "--manifest",
            str(manifest),
            "--dataset_path",
            str(dataset_path),
            "--checkpoint",
            str(low_ckpt),
            "--planner_checkpoint",
            str(commander_ckpt),
            "--skill_checkpoint",
            str(skill_ckpt),
            "--output_root",
            str(rollout_dir),
            "--ranks",
            "all",
            "--seeds",
            args.rollout_seeds,
            "--num_envs",
            "1",
            "--device",
            args.device,
            "--metric_interval",
            "1",
            "--max_steps",
            "0",
            "--chunk_rows",
            "8192",
            "--finetune_updates",
            str(args.finetune_updates),
            "--finetune_batch_size",
            str(args.finetune_batch_size),
            "--finetune_lr",
            "1.0e-4",
            "--finetune_flow_loss_coeff",
            "1.0",
            "--finetune_endpoint_loss_coeff",
            "1.0",
            "--flow_num_inference_steps",
            "16",
            "--flow_inference_noise_std",
            "0.0",
            "--eval_ranks",
            "all",
            "--eval_video_ranks",
            args.eval_video_ranks,
            "--eval_video_length",
            "500",
            "--eval_metric_interval",
            "1",
            "--eval_max_steps",
            "0",
            "--z_dim",
            str(args.z_dim),
            "--horizon_steps",
            str(args.horizon_steps),
            "--resume",
            "--continue_on_error",
        ],
        rollout_dir / "pipeline.log",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
