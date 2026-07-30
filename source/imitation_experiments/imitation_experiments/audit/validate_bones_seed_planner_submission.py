#!/usr/bin/env python3
"""Fail closed unless BONES-SEED planner submission gates all match."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("language", type=Path)
    parser.add_argument("preparation", type=Path)
    parser.add_argument("vanilla_checkpoint", type=Path)
    parser.add_argument("latent_checkpoint", type=Path)
    parser.add_argument("skill_checkpoint", type=Path)
    parser.add_argument("vanilla_audit", type=Path)
    parser.add_argument("latent_audit", type=Path)
    parser.add_argument("equivalence", type=Path)
    parser.add_argument("expected_latent_dataset_path")
    parser.add_argument("expected_vanilla_dataset_path")
    parser.add_argument("expected_manifest_sha256")
    parser.add_argument("expected_language_sha256")
    parser.add_argument("expected_preparation_sha256")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def validate(args: argparse.Namespace) -> None:
    required = (
        args.manifest,
        args.language,
        args.preparation,
        args.vanilla_checkpoint,
        args.latent_checkpoint,
        args.skill_checkpoint,
        args.vanilla_audit,
        args.latent_audit,
        args.equivalence,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"Planner submission remains blocked; missing files: {missing}"
        )

    manifest_sha = _sha256(args.manifest)
    language_sha = _sha256(args.language)
    preparation_sha = _sha256(args.preparation)
    if manifest_sha != args.expected_manifest_sha256:
        raise ValueError("Planner submission remains blocked; manifest hash changed.")
    if language_sha != args.expected_language_sha256:
        raise ValueError(
            "Planner submission remains blocked; language table hash changed."
        )
    if preparation_sha != args.expected_preparation_sha256:
        raise ValueError(
            "Planner submission remains blocked; preparation record hash changed."
        )

    preparation = _json_object(args.preparation)
    vanilla = _json_object(args.vanilla_audit)
    latent = _json_object(args.latent_audit)
    equivalence = _json_object(args.equivalence)

    if preparation.get("status") != "complete":
        raise ValueError(
            "Planner submission remains blocked; preparation is incomplete."
        )
    if preparation.get("artifacts", {}).get("manifest_sha256") != manifest_sha:
        raise ValueError(
            "Planner submission remains blocked; preparation and manifest differ."
        )

    vanilla_sha = _sha256(args.vanilla_checkpoint)
    latent_sha = _sha256(args.latent_checkpoint)
    skill_sha = _sha256(args.skill_checkpoint)
    skill_binding = latent.get("low_level_skill_binding", {})
    if not (
        vanilla.get("protocol_passed") is True
        and vanilla.get("oracle_passed") is True
        and float(vanilla.get("success_rate", -1.0)) >= 0.8
        and vanilla.get("checkpoint_sha256") == vanilla_sha
        and vanilla.get("manifest_sha256") == manifest_sha
        and vanilla.get("dataset_path") == args.expected_vanilla_dataset_path
    ):
        raise ValueError(
            "Planner submission remains blocked; vanilla qualification failed."
        )
    if not (
        latent.get("protocol_passed") is True
        and latent.get("oracle_passed") is True
        and float(latent.get("tracking_success_rate", -1.0)) >= 0.8
        and latent.get("low_level_checkpoint_sha256") == latent_sha
        and latent.get("skill_checkpoint_sha256") == skill_sha
        and skill_binding.get("passed") is True
        and skill_binding.get("low_level_checkpoint_sha256") == latent_sha
        and skill_binding.get("skill_checkpoint_sha256") == skill_sha
        and latent.get("manifest_sha256") == manifest_sha
        and latent.get("dataset_path") == args.expected_latent_dataset_path
    ):
        raise ValueError(
            "Planner submission remains blocked; latent qualification failed."
        )
    tracker = equivalence.get("low_level_tracker", {})
    if not (
        equivalence.get("passed") is True
        and equivalence.get("observed_phases") == list(range(10))
        and equivalence.get("missing_phases") == []
        and equivalence.get("asynchronous_rephase_exercised") is True
        and equivalence.get("policy_state_unchanged") is True
        and tracker.get("checkpoint_sha256") == vanilla_sha
        and equivalence.get("checkpoint_sha256") == vanilla_sha
        and equivalence.get("motion_manifest_sha256") == manifest_sha
        and equivalence.get("dataset_path") == args.expected_vanilla_dataset_path
    ):
        raise ValueError(
            "Planner submission remains blocked; equivalence certificate failed."
        )


def main() -> None:
    args = _parse_args()
    validate(args)
    print("[PASS] Fresh data, both oracle audits, and streamed equivalence are valid.")


if __name__ == "__main__":
    main()
