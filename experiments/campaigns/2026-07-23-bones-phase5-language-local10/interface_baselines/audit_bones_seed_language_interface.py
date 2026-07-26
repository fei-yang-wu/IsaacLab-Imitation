#!/usr/bin/env python3
"""Audit a BONES-SEED language-conditioned latent versus explicit smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from interface_planner_common import load_rollout_samples  # noqa: E402
from planner_sample_schema import (  # noqa: E402
    PAIRED_TARGET_CONTRACT,
    PLANNER_SAMPLE_FORMAT,
    PLANNER_SAMPLE_VERSION,
)


BACKBONE_KEYS = (
    "planner_type",
    "state_dim",
    "d_model",
    "num_layers",
    "num_heads",
    "feedforward_dim",
    "patch_dim",
    "num_state_tokens",
    "language_dim",
    "num_language_tokens",
    "dropout",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--latent_samples", type=Path, required=True)
    parser.add_argument("--full_body_samples", type=Path, required=True)
    parser.add_argument("--latent_checkpoint", type=Path, required=True)
    parser.add_argument("--full_body_checkpoint", type=Path, required=True)
    parser.add_argument("--latent_summary", type=Path, required=True)
    parser.add_argument("--full_body_summary", type=Path, required=True)
    parser.add_argument("--single_motion_manifest", type=Path, required=True)
    parser.add_argument("--expected_goal_name", required=True)
    parser.add_argument("--expected_language_dim", type=int, default=384)
    parser.add_argument("--expected_state_dim", type=int, default=930)
    parser.add_argument("--expected_rows", type=int, default=2)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(
        path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected a checkpoint mapping: {path}")
    return value


def main() -> None:
    args = _parse_args()
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    preflight = _json(args.preflight)
    require(preflight.get("passed") is True, "BONES-SEED preflight did not pass")

    sample_records: dict[str, tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}
    for interface, path, target_dim in (
        ("latent_skill", args.latent_samples, 256),
        ("full_body_trajectory", args.full_body_samples, 670),
    ):
        tensors, metadata = load_rollout_samples(path.expanduser().resolve())
        sample_records[interface] = (tensors, metadata)
        require(
            int(tensors["planner_state"].shape[0]) == args.expected_rows,
            f"{interface}: expected {args.expected_rows} sample rows",
        )
        require(
            int(tensors["planner_state"].shape[-1]) == args.expected_state_dim,
            f"{interface}: causal planner state width mismatch",
        )
        require(
            int(tensors["causal_target"].shape[-1]) == target_dim,
            f"{interface}: target width mismatch",
        )
        language = tensors.get("language_embedding")
        require(language is not None, f"{interface}: missing language_embedding")
        if language is not None:
            require(
                int(language.shape[-1]) == args.expected_language_dim,
                f"{interface}: language width mismatch",
            )
        require(
            metadata.get("sample_format")
            == {"name": PLANNER_SAMPLE_FORMAT, "version": PLANNER_SAMPLE_VERSION},
            f"{interface}: planner sample schema mismatch",
        )
        require(
            metadata.get("paired_target_contract") == PAIRED_TARGET_CONTRACT,
            f"{interface}: paired target contract mismatch",
        )
        observation_spec = metadata.get("planner_observation_spec", {})
        require(
            observation_spec.get("reference_features") == [],
            f"{interface}: causal observation contains reference features",
        )

    checkpoints = {
        "latent_skill": _checkpoint(args.latent_checkpoint),
        "full_body_trajectory": _checkpoint(args.full_body_checkpoint),
    }
    configs = {
        name: value.get("planner_config", {}) for name, value in checkpoints.items()
    }
    for interface, config in configs.items():
        require(
            int(config.get("state_dim", -1)) == args.expected_state_dim,
            f"{interface}: checkpoint state_dim mismatch",
        )
        require(
            int(config.get("language_dim", -1)) == args.expected_language_dim,
            f"{interface}: checkpoint language_dim mismatch",
        )
    for key in BACKBONE_KEYS:
        require(
            configs["latent_skill"].get(key)
            == configs["full_body_trajectory"].get(key),
            f"shared planner backbone mismatch for {key}",
        )

    language_hashes: list[str] = []
    for interface, checkpoint in checkpoints.items():
        metadata = checkpoint.get("metadata", {}).get("sample_metadata", {})
        language_metadata = metadata.get("language_conditioning", {})
        require(
            language_metadata.get("enabled") is True,
            f"{interface}: checkpoint language conditioning is disabled",
        )
        language_hashes.append(str(language_metadata.get("embedding_sha256", "")))
    require(
        len(set(language_hashes)) == 1 and len(language_hashes[0]) == 64,
        "planner rows do not use the same recorded language table",
    )

    summaries = {
        "latent_skill": _json(args.latent_summary),
        "full_body_trajectory": _json(args.full_body_summary),
    }
    for interface, summary in summaries.items():
        language_metadata = summary.get("metadata", {}).get("language_conditioning", {})
        require(
            language_metadata.get("goal_name") == args.expected_goal_name,
            f"{interface}: explicit deployment goal was not recorded",
        )
        require(
            language_metadata.get("embedding_sha256") == language_hashes[0],
            f"{interface}: evaluation language table differs from training",
        )

    one_motion = _json(args.single_motion_manifest)
    entries = (
        one_motion.get("dataset", {}).get("trajectories", {}).get("lafan1_csv", [])
    )
    require(len(entries) == 1, "closed-loop explicit manifest is not one motion")
    if len(entries) == 1:
        require(
            entries[0].get("name") == args.expected_goal_name,
            "closed-loop explicit reference does not match the requested goal",
        )
    require(
        summaries["latent_skill"].get("motion_name") == args.expected_goal_name,
        "closed-loop latent reference does not match the requested goal",
    )
    require(
        summaries["full_body_trajectory"].get("metadata", {}).get("motion_name")
        == args.expected_goal_name,
        "closed-loop explicit reference does not match the requested goal",
    )

    report = {
        "passed": not errors,
        "errors": errors,
        "goal_name": args.expected_goal_name,
        "language_dim": args.expected_language_dim,
        "state_dim": args.expected_state_dim,
        "sample_rows_per_interface": args.expected_rows,
        "target_dims": {"latent_skill": 256, "full_body_trajectory": 670},
        "language_embedding_sha256": language_hashes[0] if language_hashes else "",
        "shared_backbone_keys": list(BACKBONE_KEYS),
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(f"[FAIL] {error}" for error in errors))
    print(f"[PASS] BONES-SEED language interface audit: {output}")


if __name__ == "__main__":
    main()
