#!/usr/bin/env python3
"""Verify that a latent low-level checkpoint embeds the selected skill encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict(value: Any, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} is missing or empty.")
    invalid = [
        key for key, tensor in value.items() if not isinstance(tensor, torch.Tensor)
    ]
    if invalid:
        raise TypeError(f"{label} contains non-tensor entries: {invalid[:5]}")
    return value


def validate_binding(
    low_level_checkpoint: Path,
    skill_checkpoint: Path,
) -> dict[str, Any]:
    low_level_checkpoint = low_level_checkpoint.expanduser().resolve()
    skill_checkpoint = skill_checkpoint.expanduser().resolve()
    low_level = torch.load(low_level_checkpoint, map_location="cpu", weights_only=False)
    skill = torch.load(skill_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(low_level, dict):
        raise TypeError("Low-level checkpoint is not a mapping.")
    if not isinstance(skill, dict):
        raise TypeError("Skill checkpoint is not a mapping.")

    sampler = low_level.get("hl_skill_command_sampler_state_dict")
    if not isinstance(sampler, dict):
        raise ValueError(
            "Low-level checkpoint has no hl_skill_command_sampler_state_dict."
        )
    embedded = _state_dict(
        sampler.get("skill_encoder_state_dict"),
        label="embedded skill_encoder_state_dict",
    )
    selected = _state_dict(
        skill.get("skill_encoder_state_dict"),
        label="selected skill_encoder_state_dict",
    )

    missing = sorted(set(embedded).difference(selected))
    unexpected = sorted(set(selected).difference(embedded))
    mismatched: list[str] = []
    for key in sorted(set(embedded).intersection(selected)):
        left = embedded[key]
        right = selected[key]
        if (
            tuple(left.shape) != tuple(right.shape)
            or left.dtype != right.dtype
            or not torch.equal(left, right)
        ):
            mismatched.append(key)
    passed = not missing and not unexpected and not mismatched
    return {
        "passed": passed,
        "low_level_checkpoint": str(low_level_checkpoint),
        "low_level_checkpoint_sha256": _sha256(low_level_checkpoint),
        "skill_checkpoint": str(skill_checkpoint),
        "skill_checkpoint_sha256": _sha256(skill_checkpoint),
        "embedded_key_count": len(embedded),
        "selected_key_count": len(selected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low_level_checkpoint", type=Path, required=True)
    parser.add_argument("--skill_checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate_binding(args.low_level_checkpoint, args.skill_checkpoint)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit(
            "Latent low-level checkpoint was trained with a different skill encoder: "
            f"missing={result['missing_keys'][:5]}, "
            f"unexpected={result['unexpected_keys'][:5]}, "
            f"mismatched={result['mismatched_keys'][:5]}"
        )
    print(
        "[PASS] Latent low-level checkpoint embeds the selected skill encoder: "
        f"{result['embedded_key_count']} tensors."
    )


if __name__ == "__main__":
    main()
