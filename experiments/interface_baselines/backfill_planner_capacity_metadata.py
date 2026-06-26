#!/usr/bin/env python3
"""Backfill planner capacity metadata into existing result summaries."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml


CHECKPOINT_METADATA_KEYS = (
    "interface",
    "planner_type",
    "state_key",
    "source_sample_count",
    "num_samples",
    "selected_sample_count",
    "heldout_sample_count",
    "max_samples",
    "batch_size",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "num_updates",
    "pretrain_num_updates",
    "finetune_num_updates",
    "lr",
    "weight_decay",
    "flow_num_inference_steps",
    "flow_inference_noise_std",
    "endpoint_num_inference_steps",
    "state_dim",
    "target_dim",
    "model_size",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_roots", type=Path, nargs="*", default=[])
    parser.add_argument("--glob", action="append", default=[])
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


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_structured(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            payload = yaml.safe_load(raw)
        except yaml.YAMLError:
            return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_from_summary(payload: dict[str, Any]) -> Path | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    checkpoint = metadata.get("planner_checkpoint") or metadata.get(
        "skill_commander_checkpoint_path"
    )
    if checkpoint in (None, ""):
        return None
    path = Path(str(checkpoint)).expanduser()
    return path if path.is_file() else None


def _state_dict_parameter_count(state_dict: dict[str, Any]) -> int:
    return int(
        sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))
    )


def _trainable_parameter_count(checkpoint: dict[str, Any], parameter_count: int) -> int:
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        trainable = metadata.get("trainable_parameter_count")
        if trainable not in (None, ""):
            return int(trainable)
    return int(parameter_count)


def _checkpoint_capacity(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path.expanduser(), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dict: {checkpoint_path}")
    state_dict = checkpoint.get("planner_state_dict") or checkpoint.get(
        "generator_state_dict"
    )
    if not isinstance(state_dict, dict):
        raise KeyError(
            f"Checkpoint has neither planner_state_dict nor generator_state_dict: {checkpoint_path}"
        )
    parameter_count = _state_dict_parameter_count(state_dict)
    payload: dict[str, Any] = {
        "parameter_count": parameter_count,
        "trainable_parameter_count": _trainable_parameter_count(
            checkpoint, parameter_count
        ),
    }
    planner_config = checkpoint.get("planner_config")
    if isinstance(planner_config, dict):
        payload["planner_config"] = planner_config
        planner_type = planner_config.get("planner_type")
        if planner_type not in (None, ""):
            payload["planner_type"] = planner_type
    config = checkpoint.get("config")
    if isinstance(config, dict):
        payload["planner_config"] = config
        planner_type = config.get("planner_type")
        if planner_type not in (None, ""):
            payload["planner_type"] = planner_type
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, dict):
        for key in CHECKPOINT_METADATA_KEYS:
            value = metadata.get(key)
            if value not in (None, ""):
                payload[key] = value
        checkpoint_metadata = metadata.get("checkpoint_metadata")
        if payload.get("pretrain_num_updates") in (None, "") and isinstance(
            checkpoint_metadata, dict
        ):
            value = checkpoint_metadata.get(
                "pretrain_num_updates", checkpoint_metadata.get("num_updates")
            )
            if value not in (None, ""):
                payload["pretrain_num_updates"] = value
    return payload


def _latent_rollout_finetune_metadata(summary_path: Path) -> dict[str, Any]:
    if summary_path.parent.name not in {
        "eval_finetuned_closed_loop",
        "eval_finetuned_achieved_state",
    }:
        return {}
    interface_dir = summary_path.parent.parent
    if interface_dir.name != "latent_skill":
        return {}

    finetune_dir = interface_dir / "planner_finetune_achieved_state"
    payload: dict[str, Any] | None = None
    for candidate in (finetune_dir / "summary.json", finetune_dir / "config.yaml"):
        if candidate.is_file():
            payload = _load_structured(candidate)
            if payload is not None:
                break
    if payload is None:
        return {}

    args = payload.get("args")
    if not isinstance(args, dict):
        args = {}
    metadata: dict[str, Any] = {
        "interface": "latent_skill",
        "state_key": "planner_state",
        "planner_type": "skill_commander",
        "heldout_sample_count": 0,
    }
    for output_key, source_key in (
        ("source_sample_count", "num_samples"),
        ("num_samples", "num_samples"),
        ("selected_sample_count", "num_samples"),
        ("state_dim", "state_dim"),
        ("target_dim", "z_dim"),
    ):
        value = payload.get(source_key)
        if value not in (None, ""):
            metadata[output_key] = int(value)
    for output_key, source_key in (
        ("batch_size", "batch_size"),
        ("num_updates", "num_updates"),
        ("finetune_num_updates", "num_updates"),
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
        ("flow_num_inference_steps", "flow_num_inference_steps"),
        ("flow_inference_noise_std", "flow_inference_noise_std"),
    ):
        value = args.get(source_key)
        if value not in (None, ""):
            metadata[output_key] = value

    checkpoint = payload.get("checkpoint")
    if checkpoint not in (None, ""):
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if checkpoint_path.is_file():
            try:
                checkpoint_payload = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            except (OSError, RuntimeError, ValueError):
                checkpoint_payload = None
            if isinstance(checkpoint_payload, dict):
                checkpoint_metadata = checkpoint_payload.get("metadata")
                value = None
                if isinstance(checkpoint_metadata, dict):
                    value = checkpoint_metadata.get(
                        "pretrain_num_updates", checkpoint_metadata.get("num_updates")
                    )
                if value in (None, ""):
                    value = checkpoint_payload.get("update")
                config = checkpoint_payload.get("config")
                if value in (None, "") and isinstance(config, dict):
                    value = config.get("num_updates")
                if value not in (None, ""):
                    metadata["pretrain_num_updates"] = int(value)
    return metadata


def _variant_metadata(summary_path: Path) -> dict[str, Any]:
    variant = _variant_from_path(summary_path)
    model_size, sample_budget = _parse_variant(variant)
    payload: dict[str, Any] = {}
    if variant:
        payload["planner_variant"] = variant
    if model_size:
        payload["model_size"] = model_size
    if sample_budget:
        payload["sample_budget"] = sample_budget
    return payload


def _variant_from_path(summary_path: Path) -> str:
    parts = summary_path.parts
    for part in parts:
        if part.startswith("chunked_transformer_"):
            return part
    return ""


def _parse_variant(variant: str) -> tuple[str, str]:
    match = re.fullmatch(r"chunked_transformer_([^_]+)_(.+)", variant)
    if match is None:
        return "", ""
    return match.group(1), match.group(2)


def _merge_capacity_metadata(
    payload: dict[str, Any],
    capacity: dict[str, Any],
    variant_metadata: dict[str, Any],
) -> bool:
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return False
    planner_metadata = metadata.setdefault("planner_metadata", {})
    if not isinstance(planner_metadata, dict):
        planner_metadata = {}
        metadata["planner_metadata"] = planner_metadata

    changed = False
    for key, value in {**variant_metadata, **capacity}.items():
        if value in (None, ""):
            continue
        if planner_metadata.get(key) != value:
            planner_metadata[key] = value
            changed = True
    return changed


def _backfill_root(root: Path, *, dry_run: bool) -> int:
    changed_count = 0
    for summary_path in sorted(root.rglob("summary.json")):
        payload = _load_json(summary_path)
        if payload is None:
            continue
        checkpoint_path = _checkpoint_from_summary(payload)
        if checkpoint_path is None:
            continue
        capacity = _checkpoint_capacity(checkpoint_path)
        capacity = {**capacity, **_latent_rollout_finetune_metadata(summary_path)}
        if not _merge_capacity_metadata(
            payload,
            capacity,
            _variant_metadata(summary_path),
        ):
            continue
        changed_count += 1
        print(f"[INFO] Backfilled capacity metadata: {summary_path}")
        if not dry_run:
            summary_path.write_text(
                json.dumps(payload, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    return changed_count


def main() -> None:
    args = _parse_args()
    total = 0
    for root in _discover_roots(args):
        total += _backfill_root(root, dry_run=bool(args.dry_run))
    action = "would update" if args.dry_run else "updated"
    print(f"[INFO] Capacity metadata summaries {action}: {total}")


if __name__ == "__main__":
    main()
