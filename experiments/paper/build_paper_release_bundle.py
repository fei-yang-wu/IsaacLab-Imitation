#!/usr/bin/env python3
"""Build one fail-closed reproducibility index for the final paper results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


PHASE4_PROTOCOL = "phase4_no_language_sample_efficiency_v1"
PHASE5_PROTOCOL = "bones_seed_shared_multigoal_language_v1"
EXPECTED_INTERFACES = {
    "latent_skill": {"target_dim": 256, "planner_rate_hz": 5.0},
    "full_body_trajectory": {"target_dim": 670, "planner_rate_hz": 5.0},
}
EXPECTED_OUTPUTS = {
    "phase4_no_language": {"results_json", "per_motion_csv", "paper_markdown"},
    "phase5_bones_language": {
        "results_json",
        "per_goal_csv",
        "paired_differences_csv",
        "before_after_csv",
        "paper_markdown",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4_aggregate", type=Path, required=True)
    parser.add_argument("--phase5_aggregate", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hex_digest(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest.")
    return text


def _verify_artifact(record: dict[str, Any], *, root: Path, label: str) -> dict[str, str]:
    recorded_path = Path(str(record.get("path", ""))).expanduser()
    path = recorded_path.resolve()
    if not path.is_file():
        relocated = (root / recorded_path.name).resolve()
        if relocated.is_file():
            path = relocated
        else:
            raise FileNotFoundError(f"{label} is missing: {recorded_path}")
    expected = _require_hex_digest(record.get("sha256"), label=f"{label} hash")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash changed: {path}")
    return {"path": str(path), "sha256": actual}


def _verify_interfaces(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(EXPECTED_INTERFACES):
        raise ValueError(f"{label} interface set changed.")
    for interface, expected in EXPECTED_INTERFACES.items():
        actual = value[interface]
        if not isinstance(actual, dict):
            raise ValueError(f"{label} {interface} specification is invalid.")
        for key, expected_value in expected.items():
            if float(actual.get(key, -1)) != float(expected_value):
                raise ValueError(f"{label} {interface} {key} changed.")
    return value


def _verify_phase4_sources(results: dict[str, Any]) -> dict[str, Any]:
    cluster = results.get("cluster_submission", {})
    if not isinstance(cluster, dict):
        raise ValueError("Phase 4 cluster submission record is missing.")
    verified_cluster = _verify_artifact(
        cluster,
        root=Path(str(cluster.get("path", "."))).expanduser().resolve().parent,
        label="Phase 4 cluster submission",
    )
    run_root = Path(verified_cluster["path"]).parent
    audits = results.get("audit_sha256")
    if not isinstance(audits, dict) or len(audits) != 3 * 40 * 3:
        raise ValueError("Phase 4 must contain exactly 360 budget audit hashes.")
    for relative, expected in audits.items():
        path = (run_root / str(relative)).resolve()
        if not path.is_file() or _sha256(path) != _require_hex_digest(
            expected, label=f"Phase 4 audit {relative} hash"
        ):
            raise ValueError(f"Phase 4 source audit changed: {path}")
    return {
        "cluster_submission": verified_cluster,
        "budget_audit_count": len(audits),
    }


def _verify_phase5_sources(manifest: dict[str, Any], *, root: Path) -> list[dict[str, Any]]:
    sources = manifest.get("source_run_artifacts")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Phase 5 must contain exactly three source training seeds.")
    verified: list[dict[str, Any]] = []
    required = {
        "run_config",
        "comparison_manifest",
        "final_summary",
        "protocol_audit",
        "summarize_stage",
        "cluster_submission",
    }
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Phase 5 source artifact record is invalid.")
        artifacts = source.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != required:
            raise ValueError("Phase 5 source run is missing required artifacts.")
        verified.append(
            {
                "seed": int(source.get("seed", -1)),
                "run_root": str(source.get("run_root", "")),
                "artifacts": {
                    name: _verify_artifact(
                        record,
                        root=root,
                        label=f"Phase 5 seed {source.get('seed')} {name}",
                    )
                    for name, record in sorted(artifacts.items())
                },
            }
        )
    if sorted(item["seed"] for item in verified) != [0, 1, 2]:
        raise ValueError("Phase 5 source seeds must be exactly 0, 1, and 2.")
    return sorted(verified, key=lambda item: item["seed"])


def _verify_aggregate(root: Path, *, study: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest_path = root / "aggregation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {study} aggregation manifest: {manifest_path}")
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{study} aggregation manifest schema changed.")
    expected_protocol = PHASE4_PROTOCOL if study == "phase4_no_language" else PHASE5_PROTOCOL
    if manifest.get("protocol") != expected_protocol:
        raise ValueError(f"{study} aggregation protocol changed.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != EXPECTED_OUTPUTS[study]:
        raise ValueError(f"{study} aggregate output set is incomplete.")
    verified_outputs = {
        name: _verify_artifact(record, root=root, label=f"{study} {name}")
        for name, record in sorted(outputs.items())
    }
    results = _json(Path(verified_outputs["results_json"]["path"]))
    if (
        results.get("protocol") != expected_protocol
        or results.get("paper_protocol_complete") is not True
        or results.get("aggregation_source_sha256")
        != manifest.get("aggregation_source_sha256")
    ):
        raise ValueError(f"{study} results are not paper-protocol complete.")
    _verify_interfaces(results.get("interface_specs"), label=study)

    if study == "phase4_no_language":
        if (
            results.get("seeds") != [0, 1, 2]
            or results.get("sample_budgets") != [1000, 10000, 50000]
            or int(results.get("motion_count", -1)) != 40
        ):
            raise ValueError("Phase 4 result grid changed.")
        source_evidence: Any = _verify_phase4_sources(results)
    else:
        if (
            results.get("seeds") != [0, 1, 2]
            or int(results.get("seed_count", -1)) != 3
            or int(results.get("goal_count", -1)) != 100
        ):
            raise ValueError("Phase 5 result grid changed.")
        source_evidence = _verify_phase5_sources(manifest, root=root)

    return {
        "protocol": expected_protocol,
        "aggregate_root": str(root),
        "aggregation_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "outputs": verified_outputs,
        "source_evidence": source_evidence,
    }


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def build_release(
    *, phase4_aggregate: Path, phase5_aggregate: Path, output_dir: Path, command: str
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite paper release: {output_dir}")
    studies = {
        "phase4_no_language": _verify_aggregate(
            phase4_aggregate, study="phase4_no_language"
        ),
        "phase5_bones_language": _verify_aggregate(
            phase5_aggregate, study="phase5_bones_language"
        ),
    }
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "paper_release_manifest.json"
    markdown_path = output_dir / "paper_release_index.md"
    payload = {
        "schema_version": 1,
        "paper_protocol_complete": True,
        "release_builder_sha256": _sha256(Path(__file__).resolve()),
        "release_command": command,
        "repository": {
            "root": _git_value("rev-parse", "--show-toplevel"),
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "studies": studies,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Paper reproducibility index",
        "",
        "This index was generated only after both aggregate manifests and their source evidence passed hash verification.",
        "",
    ]
    for label, study in studies.items():
        lines.extend(
            [
                f"## {label}",
                "",
                f"Protocol: `{study['protocol']}`",
                "",
                f"Aggregate manifest: `{study['aggregation_manifest']['path']}`",
                "",
            ]
        )
        for name, artifact in study["outputs"].items():
            lines.append(f"- {name}: `{artifact['path']}` (`{artifact['sha256']}`)")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"manifest": manifest_path, "index": markdown_path}


def main() -> None:
    args = _parse_args()
    outputs = build_release(
        phase4_aggregate=args.phase4_aggregate,
        phase5_aggregate=args.phase5_aggregate,
        output_dir=args.output_dir,
        command=" ".join(shlex.quote(value) for value in sys.argv),
    )
    print(f"[PASS] Paper release manifest: {outputs['manifest']}")
    print(f"[PASS] Paper release index: {outputs['index']}")


if __name__ == "__main__":
    main()
