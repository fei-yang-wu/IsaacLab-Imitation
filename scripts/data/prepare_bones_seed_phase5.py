#!/usr/bin/env python3
"""Build a fresh, audited BONES-SEED Phase-5 reference tree from raw CSVs.

This wrapper never repairs or overwrites an existing dataset. It validates that
the raw CSV filenames and language sidecar have exact motion coverage, exports
all motions through ``batch_csv_to_npz.py`` into a new output root, writes a
self-contained manifest/language pair, and runs the Phase-5 preflight before
marking the preparation record complete.

Run this script through the Isaac Lab Pixi environment. Use ``--dry-run`` from
the default environment when only validating inputs and inspecting the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_NAME = "bones_seed_phase5_preparation"
SCHEMA_VERSION = 1
SOURCE_DATASET = "bones-studio/seed"


def _default_batch_converter() -> Path:
    return Path(__file__).resolve().with_name("batch_csv_to_npz.py")


def _default_preflight() -> Path:
    return Path(__file__).resolve().with_name("audit_bones_seed_phase5.py")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--language-sidecar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-converter-script",
        type=Path,
        default=_default_batch_converter(),
    )
    parser.add_argument(
        "--preflight-script",
        type=Path,
        default=_default_preflight(),
    )
    parser.add_argument(
        "--require-temporal-events",
        action="store_true",
        help="Reject a language sidecar if any motion has no temporal events.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the complete plan without writing anything.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _motion_name(csv_path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", csv_path.stem)
    name = name.replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError(f"CSV filename does not produce a motion name: {csv_path}")
    return name


def _dataset_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    slug = slug.replace("-", "_")
    slug = re.sub(r"_+", "_", slug).strip("_").lower()
    if not slug:
        raise ValueError("--dataset-name must contain at least one letter or number.")
    return slug


def _read_language_sidecar(
    path: Path,
    *,
    expected_names: list[str],
    require_temporal_events: bool,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read language sidecar {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Language sidecar root must be a JSON object.")
    motions = payload.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("Language sidecar must contain a non-empty motions list.")

    names: list[str] = []
    seen: set[str] = set()
    missing_goals: list[str] = []
    missing_events: list[str] = []
    for index, motion in enumerate(motions):
        if not isinstance(motion, Mapping):
            raise ValueError(f"Language motion #{index} must be a JSON object.")
        name = str(motion.get("name", "")).strip()
        if not name:
            raise ValueError(f"Language motion #{index} has no name.")
        if name in seen:
            raise ValueError(f"Language sidecar contains duplicate motion {name!r}.")
        seen.add(name)
        names.append(name)
        if not str(motion.get("language_goal", "")).strip():
            missing_goals.append(name)
        events = motion.get("events", [])
        if require_temporal_events and (not isinstance(events, list) or not events):
            missing_events.append(name)

    expected = set(expected_names)
    actual = set(names)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "CSV/language coverage differs: "
            f"missing_language={missing}, extra_language={extra}."
        )
    if missing_goals:
        raise ValueError(f"Motions without language_goal: {sorted(missing_goals)}")
    if missing_events:
        raise ValueError(f"Motions without temporal events: {sorted(missing_events)}")
    return dict(payload)


def _validate_output_root(output_root: Path, *, csv_dir: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}. Choose a new directory; "
            "this wrapper never overwrites or resumes a dataset tree."
        )
    if output_root == csv_dir:
        raise ValueError("Output root must differ from the raw CSV directory.")
    if output_root.is_relative_to(csv_dir) or csv_dir.is_relative_to(output_root):
        raise ValueError(
            "Output root and raw CSV directory must not contain one another. "
            "Use a separate sibling tree."
        )


def _validate_args(args: argparse.Namespace) -> dict[str, Any]:
    csv_dir = args.csv_dir.expanduser().resolve()
    language_source = args.language_sidecar.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    batch_converter = args.batch_converter_script.expanduser().resolve()
    preflight_script = args.preflight_script.expanduser().resolve()
    dataset_name = str(args.dataset_name).strip()
    dataset_slug = _dataset_slug(dataset_name)
    input_fps = float(args.input_fps)
    output_fps = float(args.output_fps)

    if not csv_dir.is_dir():
        raise NotADirectoryError(f"Raw CSV directory does not exist: {csv_dir}")
    if not language_source.is_file():
        raise FileNotFoundError(f"Language sidecar does not exist: {language_source}")
    if not batch_converter.is_file():
        raise FileNotFoundError(f"Batch converter does not exist: {batch_converter}")
    if not preflight_script.is_file():
        raise FileNotFoundError(f"Preflight script does not exist: {preflight_script}")
    if not math.isfinite(input_fps) or input_fps <= 0.0:
        raise ValueError("--input-fps must be positive and finite.")
    if not math.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError("--output-fps must be positive and finite.")
    _validate_output_root(output_root, csv_dir=csv_dir)

    csv_files = sorted(path for path in csv_dir.glob("*.csv") if path.is_file())
    if not csv_files:
        raise ValueError(f"No CSV files found directly under {csv_dir}.")
    nested_csvs = sorted(
        path for path in csv_dir.glob("**/*.csv") if path.parent != csv_dir
    )
    if nested_csvs:
        raise ValueError(
            "Nested CSVs are not supported by this subset wrapper. Stage the selected "
            f"clips into one flat source directory first; found {nested_csvs[0]}."
        )

    motion_names = [_motion_name(path) for path in csv_files]
    if len(set(motion_names)) != len(motion_names):
        duplicates = sorted(
            name for name in set(motion_names) if motion_names.count(name) > 1
        )
        raise ValueError(
            f"CSV filenames collapse to duplicate motion names: {duplicates}"
        )
    language_payload = _read_language_sidecar(
        language_source,
        expected_names=motion_names,
        require_temporal_events=bool(args.require_temporal_events),
    )

    manifest_path = output_root / "manifests" / f"g1_{dataset_slug}_manifest.json"
    language_path = output_root / "language" / language_source.name
    jobs_path = output_root / "preparation" / "export_jobs.json"
    preparation_path = output_root / "preparation" / "preparation.json"
    report_path = output_root / "reports" / "phase5_preflight.json"
    npz_dir = output_root / "npz" / "g1"
    jobs: list[dict[str, str]] = []
    motion_inputs: list[dict[str, Any]] = []
    for csv_file, motion_name in zip(csv_files, motion_names, strict=True):
        npz_path = npz_dir / f"{csv_file.stem}.npz"
        jobs.append(
            {
                "source_type": "csv",
                "input_file": str(csv_file),
                "output_name": str(npz_path),
            }
        )
        motion_inputs.append(
            {
                "name": motion_name,
                "csv_path": str(csv_file),
                "csv_sha256": _sha256(csv_file),
                "npz_path": str(npz_path),
            }
        )

    export_command = [
        sys.executable,
        str(batch_converter),
        "--jobs_json",
        str(jobs_path),
        "--input_fps",
        str(input_fps),
        "--output_fps",
        str(output_fps),
        "--headless",
    ]
    if args.device:
        export_command.extend(["--device", str(args.device)])
    preflight_command = [
        sys.executable,
        str(preflight_script),
        "--manifest",
        str(manifest_path),
        "--report",
        str(report_path),
        "--require-body-names",
    ]
    if args.require_temporal_events:
        preflight_command.append("--require-temporal-events")

    return {
        "csv_dir": csv_dir,
        "language_source": language_source,
        "language_payload": language_payload,
        "output_root": output_root,
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "batch_converter": batch_converter,
        "preflight_script": preflight_script,
        "manifest_path": manifest_path,
        "language_path": language_path,
        "jobs_path": jobs_path,
        "preparation_path": preparation_path,
        "report_path": report_path,
        "npz_dir": npz_dir,
        "jobs": jobs,
        "motion_inputs": motion_inputs,
        "export_command": export_command,
        "preflight_command": preflight_command,
    }


def _public_plan(
    plan: Mapping[str, Any], *, require_temporal_events: bool
) -> dict[str, Any]:
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "dataset_name": plan["dataset_name"],
        "source_dataset": SOURCE_DATASET,
        "motion_count": len(plan["jobs"]),
        "inputs": {
            "csv_dir": str(plan["csv_dir"]),
            "language_sidecar": str(plan["language_source"]),
            "language_sidecar_sha256": _sha256(plan["language_source"]),
            "input_fps": plan["input_fps"],
        },
        "outputs": {
            "root": str(plan["output_root"]),
            "manifest": str(plan["manifest_path"]),
            "language_sidecar": str(plan["language_path"]),
            "npz_dir": str(plan["npz_dir"]),
            "preflight_report": str(plan["report_path"]),
            "preparation_record": str(plan["preparation_path"]),
            "output_fps": plan["output_fps"],
        },
        "tools": {
            "wrapper": str(Path(__file__).resolve()),
            "wrapper_sha256": _sha256(Path(__file__).resolve()),
            "batch_converter": str(plan["batch_converter"]),
            "batch_converter_sha256": _sha256(plan["batch_converter"]),
            "preflight": str(plan["preflight_script"]),
            "preflight_sha256": _sha256(plan["preflight_script"]),
        },
        "requirements": {
            "fresh_output_root": True,
            "require_body_names": True,
            "require_temporal_events": require_temporal_events,
            "exact_language_coverage": True,
        },
        "export_command": list(plan["export_command"]),
        "preflight_command": list(plan["preflight_command"]),
        "motions": list(plan["motion_inputs"]),
    }


def _write_staged_inputs(
    plan: Mapping[str, Any], *, require_temporal_events: bool
) -> dict[str, Any]:
    output_root = plan["output_root"]
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(plan["jobs_path"], plan["jobs"])

    language_payload = dict(plan["language_payload"])
    language_payload["dataset_name"] = plan["dataset_name"]
    language_payload["source"] = SOURCE_DATASET
    language_payload["manifest"] = os.path.relpath(
        plan["manifest_path"], plan["language_path"].parent
    )
    language_payload["input_fps"] = plan["input_fps"]
    language_payload["output_fps"] = plan["output_fps"]
    motions_by_name = {
        str(motion["name"]): motion for motion in language_payload["motions"]
    }
    language_payload["motions"] = [
        motions_by_name[str(motion["name"])] for motion in plan["motion_inputs"]
    ]
    _write_json(plan["language_path"], language_payload)

    manifest_entries = []
    for motion, job in zip(plan["motion_inputs"], plan["jobs"], strict=True):
        npz_path = Path(job["output_name"])
        manifest_entries.append(
            {
                "name": motion["name"],
                "path": os.path.relpath(npz_path, plan["manifest_path"].parent),
                "input_fps": plan["output_fps"],
            }
        )
    manifest = {
        "dataset_name": plan["dataset_name"],
        "dataset": {"trajectories": {"lafan1_csv": manifest_entries}},
        "metadata": {
            "source_dataset": SOURCE_DATASET,
            "source_csv_dir": str(plan["csv_dir"]),
            "source_language_sidecar": str(plan["language_source"]),
            "language_annotations_path": os.path.relpath(
                plan["language_path"], plan["manifest_path"].parent
            ),
            "preparation_record": os.path.relpath(
                plan["preparation_path"], plan["manifest_path"].parent
            ),
            "num_motions": len(manifest_entries),
            "input_fps": plan["input_fps"],
            "output_fps": plan["output_fps"],
            "paths_are_relative_to_manifest": True,
            "body_position_frame": (
                "motion-local; batch_csv_to_npz.py removes each Isaac scene-grid "
                "origin before saving body_pos_w"
            ),
            "body_names_required": True,
        },
    }
    _write_json(plan["manifest_path"], manifest)

    record = _public_plan(plan, require_temporal_events=require_temporal_events)
    record["status"] = "staged"
    _write_json(plan["preparation_path"], record)
    return record


def _run(command: list[str], *, label: str) -> None:
    print(f"[INFO] {label}")
    print(f"[CMD]  {' '.join(shlex.quote(item) for item in command)}")
    subprocess.run(command, check=True)


def _finalize_record(plan: Mapping[str, Any], record: dict[str, Any]) -> None:
    outputs = []
    for motion, job in zip(plan["motion_inputs"], plan["jobs"], strict=True):
        npz_path = Path(job["output_name"])
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"Exporter did not produce expected NPZ: {npz_path}"
            )
        outputs.append(
            {
                "name": motion["name"],
                "npz_path": str(npz_path),
                "npz_sha256": _sha256(npz_path),
            }
        )
    record["status"] = "complete"
    record["artifacts"] = {
        "manifest_sha256": _sha256(plan["manifest_path"]),
        "language_sidecar_sha256": _sha256(plan["language_path"]),
        "preflight_report_sha256": _sha256(plan["report_path"]),
        "npz_files": outputs,
    }
    _write_json(plan["preparation_path"], record)


def main() -> None:
    args = _parse_args()
    plan = _validate_args(args)
    public_plan = _public_plan(
        plan, require_temporal_events=bool(args.require_temporal_events)
    )
    if args.dry_run:
        print(json.dumps(public_plan, indent=2))
        return

    record = _write_staged_inputs(
        plan, require_temporal_events=bool(args.require_temporal_events)
    )
    try:
        _run(plan["export_command"], label="Exporting corrected BONES-SEED NPZs")
    except subprocess.CalledProcessError as exc:
        record["status"] = "export_failed"
        record["returncode"] = int(exc.returncode)
        _write_json(plan["preparation_path"], record)
        raise

    try:
        _run(plan["preflight_command"], label="Auditing the generated Phase-5 tree")
    except subprocess.CalledProcessError as exc:
        record["status"] = "preflight_failed"
        record["returncode"] = int(exc.returncode)
        _write_json(plan["preparation_path"], record)
        raise

    _finalize_record(plan, record)
    print(f"[PASS] Corrected BONES-SEED tree: {plan['output_root']}")
    print(f"[INFO] Manifest: {plan['manifest_path']}")
    print(f"[INFO] Preflight: {plan['report_path']}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"[FAIL] Command exited with status {exc.returncode}: "
            f"{' '.join(shlex.quote(str(item)) for item in exc.cmd)}"
        ) from exc
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
