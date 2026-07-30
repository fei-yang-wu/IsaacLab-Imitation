#!/usr/bin/env python3
"""Fail closed unless the no-language paper sweep uses qualified artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _motion_names(manifest: Path) -> list[str]:
    payload = _json(manifest)
    entries = payload.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Phase-4 manifest contains no trajectories.")
    names = [str(entry.get("name", "")).strip() for entry in entries]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Phase-4 motion names must be nonempty and unique.")
    return names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("vanilla_checkpoint", type=Path)
    parser.add_argument("latent_checkpoint", type=Path)
    parser.add_argument("skill_checkpoint", type=Path)
    parser.add_argument("vanilla_audit", type=Path)
    parser.add_argument("latent_audit", type=Path)
    parser.add_argument("equivalence", type=Path)
    parser.add_argument("expected_latent_dataset_path")
    parser.add_argument("expected_vanilla_dataset_path")
    parser.add_argument("expected_manifest_sha256")
    parser.add_argument("--expected_motion_count", type=int, default=40)
    parser.add_argument("--minimum_oracle_success", type=float, default=0.8)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict:
    required = (
        args.manifest,
        args.vanilla_checkpoint,
        args.latent_checkpoint,
        args.skill_checkpoint,
        args.vanilla_audit,
        args.latent_audit,
        args.equivalence,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Phase-4 submission remains blocked; missing: {missing}")
    manifest_sha = _sha256(args.manifest)
    if manifest_sha != args.expected_manifest_sha256:
        raise ValueError("Phase-4 submission remains blocked; manifest hash changed.")
    if len(_motion_names(args.manifest)) != int(args.expected_motion_count):
        raise ValueError("Phase-4 submission remains blocked; motion count changed.")

    vanilla = _json(args.vanilla_audit)
    latent = _json(args.latent_audit)
    equivalence = _json(args.equivalence)
    vanilla_sha = _sha256(args.vanilla_checkpoint)
    latent_sha = _sha256(args.latent_checkpoint)
    skill_sha = _sha256(args.skill_checkpoint)
    skill_binding = latent.get("low_level_skill_binding", {})
    threshold = float(args.minimum_oracle_success)
    if not (
        vanilla.get("protocol_passed") is True
        and vanilla.get("oracle_passed") is True
        and float(vanilla.get("success_rate", -1.0)) >= threshold
        and vanilla.get("checkpoint_sha256") == vanilla_sha
        and vanilla.get("manifest_sha256") == manifest_sha
        and vanilla.get("dataset_path") == args.expected_vanilla_dataset_path
    ):
        raise ValueError("Phase-4 submission remains blocked; vanilla gate failed.")
    if not (
        latent.get("protocol_passed") is True
        and latent.get("oracle_passed") is True
        and float(latent.get("tracking_success_rate", -1.0)) >= threshold
        and latent.get("low_level_checkpoint_sha256") == latent_sha
        and latent.get("skill_checkpoint_sha256") == skill_sha
        and skill_binding.get("passed") is True
        and skill_binding.get("low_level_checkpoint_sha256") == latent_sha
        and skill_binding.get("skill_checkpoint_sha256") == skill_sha
        and latent.get("manifest_sha256") == manifest_sha
        and latent.get("dataset_path") == args.expected_latent_dataset_path
    ):
        raise ValueError("Phase-4 submission remains blocked; latent gate failed.")
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
            "Phase-4 submission remains blocked; equivalence certificate failed."
        )
    return {
        "passed": True,
        "manifest_sha256": manifest_sha,
        "motion_count": int(args.expected_motion_count),
        "minimum_oracle_success": threshold,
        "vanilla_checkpoint_sha256": vanilla_sha,
        "latent_checkpoint_sha256": latent_sha,
        "skill_checkpoint_sha256": skill_sha,
        "latent_dataset_path": args.expected_latent_dataset_path,
        "vanilla_dataset_path": args.expected_vanilla_dataset_path,
        "vanilla_audit_sha256": _sha256(args.vanilla_audit),
        "latent_audit_sha256": _sha256(args.latent_audit),
        "equivalence_sha256": _sha256(args.equivalence),
        "vanilla_oracle_success": float(vanilla["success_rate"]),
        "latent_oracle_success": float(latent["tracking_success_rate"]),
    }


def main() -> None:
    args = _parse_args()
    result = validate(args)
    if args.output_json is not None:
        output = args.output_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("[PASS] Phase-4 data, oracle gates, and equivalence certificate match.")


if __name__ == "__main__":
    main()
