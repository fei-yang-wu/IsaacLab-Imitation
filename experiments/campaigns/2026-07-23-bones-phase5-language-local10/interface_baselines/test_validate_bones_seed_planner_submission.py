"""Tests for the fail-closed BONES-SEED planner submission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from validate_bones_seed_planner_submission import validate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _valid_args(tmp_path: Path) -> argparse.Namespace:
    manifest = tmp_path / "manifest.json"
    language = tmp_path / "language.pt"
    vanilla_checkpoint = tmp_path / "vanilla.pt"
    latent_checkpoint = tmp_path / "latent.pt"
    skill_checkpoint = tmp_path / "skill.pt"
    for path, value in (
        (manifest, b"manifest"),
        (language, b"language"),
        (vanilla_checkpoint, b"vanilla"),
        (latent_checkpoint, b"latent"),
        (skill_checkpoint, b"skill"),
    ):
        path.write_bytes(value)

    preparation = tmp_path / "preparation.json"
    vanilla_audit = tmp_path / "vanilla_audit.json"
    latent_audit = tmp_path / "latent_audit.json"
    equivalence = tmp_path / "equivalence.json"
    _write_json(
        preparation,
        {"status": "complete", "artifacts": {"manifest_sha256": _sha(manifest)}},
    )
    _write_json(
        vanilla_audit,
        {
            "protocol_passed": True,
            "oracle_passed": True,
            "success_rate": 0.8,
            "checkpoint_sha256": _sha(vanilla_checkpoint),
            "manifest_sha256": _sha(manifest),
            "dataset_path": "/data/vanilla",
        },
    )
    _write_json(
        latent_audit,
        {
            "protocol_passed": True,
            "oracle_passed": True,
            "tracking_success_rate": 0.9,
            "low_level_checkpoint_sha256": _sha(latent_checkpoint),
            "skill_checkpoint_sha256": _sha(skill_checkpoint),
            "low_level_skill_binding": {
                "passed": True,
                "low_level_checkpoint_sha256": _sha(latent_checkpoint),
                "skill_checkpoint_sha256": _sha(skill_checkpoint),
            },
            "manifest_sha256": _sha(manifest),
            "dataset_path": "/data/latent",
        },
    )
    _write_json(
        equivalence,
        {
            "passed": True,
            "observed_phases": list(range(10)),
            "missing_phases": [],
            "asynchronous_rephase_exercised": True,
            "policy_state_unchanged": True,
            "checkpoint_sha256": _sha(vanilla_checkpoint),
            "motion_manifest_sha256": _sha(manifest),
            "dataset_path": "/data/vanilla",
            "low_level_tracker": {"checkpoint_sha256": _sha(vanilla_checkpoint)},
        },
    )
    return argparse.Namespace(
        manifest=manifest,
        language=language,
        preparation=preparation,
        vanilla_checkpoint=vanilla_checkpoint,
        latent_checkpoint=latent_checkpoint,
        skill_checkpoint=skill_checkpoint,
        vanilla_audit=vanilla_audit,
        latent_audit=latent_audit,
        equivalence=equivalence,
        expected_latent_dataset_path="/data/latent",
        expected_vanilla_dataset_path="/data/vanilla",
        expected_manifest_sha256=_sha(manifest),
        expected_language_sha256=_sha(language),
        expected_preparation_sha256=_sha(preparation),
    )


def test_submission_gate_accepts_exact_passing_artifacts(tmp_path: Path) -> None:
    validate(_valid_args(tmp_path))


def test_submission_gate_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    args = _valid_args(tmp_path)
    args.vanilla_checkpoint.write_bytes(b"changed")

    with pytest.raises(ValueError, match="vanilla qualification failed"):
        validate(args)


def test_submission_gate_rejects_incomplete_equivalence(tmp_path: Path) -> None:
    args = _valid_args(tmp_path)
    payload = json.loads(args.equivalence.read_text(encoding="utf-8"))
    payload["missing_phases"] = [9]
    _write_json(args.equivalence, payload)

    with pytest.raises(ValueError, match="equivalence certificate failed"):
        validate(args)
