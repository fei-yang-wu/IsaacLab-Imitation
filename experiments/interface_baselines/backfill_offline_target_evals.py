#!/usr/bin/env python3
"""Backfill offline planner target-eval summaries for existing result roots."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


HAND_DESIGNED_INTERFACES = ("full_body_trajectory", "ee_trajectory")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_roots", type=Path, nargs="*", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--flow_num_inference_steps", type=int, default=16)
    parser.add_argument("--flow_inference_noise_std", type=float, default=0.0)
    parser.add_argument("--refresh", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def _discover_roots(args: argparse.Namespace) -> list[Path]:
    roots = [root.expanduser().resolve() for root in args.result_roots]
    for pattern in args.glob:
        roots.extend(Path(path).expanduser().resolve() for path in glob.glob(pattern))
    roots = sorted({root for root in roots if root.is_dir()})
    if not roots:
        raise ValueError("No result roots found.")
    return roots


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("[CMD] " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _summary_checkpoint(path: Path) -> Path | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    checkpoint = metadata.get("planner_checkpoint") or metadata.get(
        "skill_commander_checkpoint_path"
    )
    if checkpoint in (None, ""):
        return None
    checkpoint_path = Path(str(checkpoint)).expanduser()
    return checkpoint_path if checkpoint_path.is_file() else None


def _latent_pretrained_checkpoint(interface_root: Path) -> Path | None:
    for summary_path in (
        interface_root / "eval_pretrained_closed_loop" / "summary.json",
        interface_root / "oracle_low_level" / "summary.json",
    ):
        checkpoint = _summary_checkpoint(summary_path)
        if checkpoint is not None:
            return checkpoint
    return None


def _append_common_args(
    cmd: list[str],
    *,
    args: argparse.Namespace,
    output_json: Path,
    output_csv: Path,
    state_key: str,
    setting: str,
    label: str,
) -> list[str]:
    cmd.extend(
        [
            "--output_json",
            str(output_json),
            "--output_csv",
            str(output_csv),
            "--state_key",
            state_key,
            "--setting",
            setting,
            "--label",
            label,
            "--batch_size",
            str(args.batch_size),
            "--max_samples",
            str(args.max_samples),
            "--seed",
            str(args.seed),
            "--flow_num_inference_steps",
            str(args.flow_num_inference_steps),
            "--flow_inference_noise_std",
            str(args.flow_inference_noise_std),
        ]
    )
    return cmd


def _maybe_backfill_latent(root: Path, args: argparse.Namespace) -> int:
    interface_root = root / "latent_skill"
    samples_dir = interface_root / "oracle_drive_samples" / "rollout_training_samples"
    if not samples_dir.is_dir():
        return 0
    script = Path(__file__).resolve().parent / "eval_latent_skill_planner_offline.py"
    commands: list[list[str]] = []
    pretrained_checkpoint = _latent_pretrained_checkpoint(interface_root)
    if pretrained_checkpoint is not None:
        output_dir = interface_root / "eval_pretrained_expert_state"
        output_json = output_dir / "summary.json"
        if args.refresh or not output_json.is_file():
            commands.append(
                _append_common_args(
                    [
                        sys.executable,
                        str(script),
                        "--samples_dir",
                        str(samples_dir),
                        "--planner_checkpoint",
                        str(pretrained_checkpoint),
                    ],
                    args=args,
                    output_json=output_json,
                    output_csv=output_dir / "summary.csv",
                    state_key="expert_planner_state",
                    setting="eval_pretrained_expert_state",
                    label="latent_skill_pretrained_expert_state",
                )
            )
    finetuned_checkpoint = (
        interface_root / "planner_finetune_achieved_state" / "checkpoints" / "latest.pt"
    )
    if finetuned_checkpoint.is_file():
        output_dir = interface_root / "eval_finetuned_achieved_state"
        output_json = output_dir / "summary.json"
        if args.refresh or not output_json.is_file():
            commands.append(
                _append_common_args(
                    [
                        sys.executable,
                        str(script),
                        "--samples_dir",
                        str(samples_dir),
                        "--planner_checkpoint",
                        str(finetuned_checkpoint),
                    ],
                    args=args,
                    output_json=output_json,
                    output_csv=output_dir / "summary.csv",
                    state_key="planner_state",
                    setting="eval_finetuned_achieved_state",
                    label="latent_skill_finetuned_achieved_state",
                )
            )
    for cmd in commands:
        _run(cmd, dry_run=bool(args.dry_run))
    return len(commands)


def _variant_roots(interface_root: Path) -> list[Path]:
    variants = [
        path
        for path in interface_root.iterdir()
        if path.is_dir() and (path / "planner_pretrain_expert_state").is_dir()
    ]
    if variants:
        return sorted(variants)
    return [interface_root]


def _maybe_backfill_hand_designed(root: Path, args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parent / "eval_interface_planner_offline.py"
    command_count = 0
    for interface in HAND_DESIGNED_INTERFACES:
        interface_root = root / interface
        samples_dir = (
            interface_root / "oracle_drive_samples" / "rollout_training_samples"
        )
        if not samples_dir.is_dir():
            continue
        for variant_root in _variant_roots(interface_root):
            pretrain_checkpoint = (
                variant_root
                / "planner_pretrain_expert_state"
                / "checkpoints"
                / "latest.pt"
            )
            finetune_checkpoint = (
                variant_root
                / "planner_finetune_achieved_state"
                / "checkpoints"
                / "latest.pt"
            )
            for checkpoint, state_key, setting, suffix in (
                (
                    pretrain_checkpoint,
                    "expert_planner_state",
                    "eval_pretrained_expert_state",
                    "pretrained_expert_state",
                ),
                (
                    finetune_checkpoint,
                    "planner_state",
                    "eval_finetuned_achieved_state",
                    "finetuned_achieved_state",
                ),
            ):
                if not checkpoint.is_file():
                    continue
                output_dir = variant_root / setting
                output_json = output_dir / "summary.json"
                if not args.refresh and output_json.is_file():
                    continue
                cmd = _append_common_args(
                    [
                        sys.executable,
                        str(script),
                        "--samples_dir",
                        str(samples_dir),
                        "--planner_checkpoint",
                        str(checkpoint),
                        "--interface",
                        interface,
                    ],
                    args=args,
                    output_json=output_json,
                    output_csv=output_dir / "summary.csv",
                    state_key=state_key,
                    setting=setting,
                    label=f"{interface}_{suffix}",
                )
                _run(cmd, dry_run=bool(args.dry_run))
                command_count += 1
    return command_count


def main() -> None:
    args = _parse_args()
    total = 0
    for root in _discover_roots(args):
        total += _maybe_backfill_latent(root, args)
        total += _maybe_backfill_hand_designed(root, args)
    print(f"[INFO] Backfill commands {'planned' if args.dry_run else 'run'}: {total}")


if __name__ == "__main__":
    main()
