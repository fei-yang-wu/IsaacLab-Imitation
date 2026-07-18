from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from build_paper_release_bundle import (
    PHASE4_PROTOCOL,
    PHASE5_PROTOCOL,
    build_release,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _text(path: Path, value: str = "artifact\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _interface_specs() -> dict[str, dict[str, float | int]]:
    return {
        "latent_skill": {"target_dim": 256, "planner_rate_hz": 5.0},
        "full_body_trajectory": {"target_dim": 670, "planner_rate_hz": 5.0},
    }


def _phase4(root: Path) -> Path:
    aggregate = root / "phase4_aggregate"
    run_root = root / "phase4_run"
    cluster = _json(run_root / "cluster_submission.json", {"job_id": 123})
    audit_hashes: dict[str, str] = {}
    for index in range(360):
        relative = f"audits/audit_{index:03d}.json"
        audit = _json(run_root / relative, {"passed": True, "index": index})
        audit_hashes[relative] = _sha256(audit)
    source_hash = "a" * 64
    results = _json(
        aggregate / "phase4_no_language_results.json",
        {
            "protocol": PHASE4_PROTOCOL,
            "paper_protocol_complete": True,
            "aggregation_source_sha256": source_hash,
            "seeds": [0, 1, 2],
            "sample_budgets": [1000, 10000, 50000],
            "motion_count": 40,
            "interface_specs": _interface_specs(),
            "cluster_submission": _record(cluster),
            "audit_sha256": audit_hashes,
        },
    )
    csv = _text(aggregate / "phase4_no_language_per_motion.csv")
    markdown = _text(aggregate / "phase4_no_language_results.md")
    _json(
        aggregate / "aggregation_manifest.json",
        {
            "schema_version": 1,
            "protocol": PHASE4_PROTOCOL,
            "aggregation_source_sha256": source_hash,
            "outputs": {
                "results_json": _record(results),
                "per_motion_csv": _record(csv),
                "paper_markdown": _record(markdown),
            },
        },
    )
    return aggregate


def _phase5(root: Path) -> Path:
    aggregate = root / "phase5_aggregate"
    source_hash = "b" * 64
    results = _json(
        aggregate / "multiseed_results.json",
        {
            "protocol": PHASE5_PROTOCOL,
            "paper_protocol_complete": True,
            "aggregation_source_sha256": source_hash,
            "seeds": [0, 1, 2],
            "seed_count": 3,
            "goal_count": 100,
            "interface_specs": _interface_specs(),
        },
    )
    outputs = {
        "results_json": results,
        "per_goal_csv": _text(aggregate / "multiseed_per_goal.csv"),
        "paired_differences_csv": _text(
            aggregate / "multiseed_paired_differences.csv"
        ),
        "before_after_csv": _text(aggregate / "multiseed_before_after.csv"),
        "paper_markdown": _text(aggregate / "multiseed_results.md"),
    }
    source_runs = []
    for seed in range(3):
        run_root = root / f"phase5_seed_{seed}"
        artifacts = {
            name: _record(_text(run_root / f"{name}.json", f"{name} {seed}\n"))
            for name in (
                "run_config",
                "comparison_manifest",
                "final_summary",
                "protocol_audit",
                "summarize_stage",
                "cluster_submission",
            )
        }
        source_runs.append(
            {"seed": seed, "run_root": str(run_root), "artifacts": artifacts}
        )
    _json(
        aggregate / "aggregation_manifest.json",
        {
            "schema_version": 1,
            "protocol": PHASE5_PROTOCOL,
            "aggregation_source_sha256": source_hash,
            "source_run_artifacts": source_runs,
            "outputs": {name: _record(path) for name, path in outputs.items()},
        },
    )
    return aggregate


def test_builds_release_only_from_complete_hash_verified_aggregates(
    tmp_path: Path,
) -> None:
    outputs = build_release(
        phase4_aggregate=_phase4(tmp_path),
        phase5_aggregate=_phase5(tmp_path),
        output_dir=tmp_path / "release",
        command="test release",
    )
    payload = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert payload["paper_protocol_complete"] is True
    assert set(payload["studies"]) == {
        "phase4_no_language",
        "phase5_bones_language",
    }
    assert (
        payload["studies"]["phase4_no_language"]["source_evidence"][
            "budget_audit_count"
        ]
        == 360
    )
    assert outputs["index"].is_file()


def test_rejects_changed_aggregate_output(tmp_path: Path) -> None:
    phase4 = _phase4(tmp_path)
    phase5 = _phase5(tmp_path)
    (phase5 / "multiseed_results.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        build_release(
            phase4_aggregate=phase4,
            phase5_aggregate=phase5,
            output_dir=tmp_path / "release",
            command="test release",
        )


def test_refuses_to_overwrite_release(tmp_path: Path) -> None:
    output = tmp_path / "release"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_release(
            phase4_aggregate=tmp_path / "missing_phase4",
            phase5_aggregate=tmp_path / "missing_phase5",
            output_dir=output,
            command="test release",
        )
