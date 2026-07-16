#!/usr/bin/env python3
"""Validate BONES-SEED motion and language inputs before a Phase-5 run.

The preflight is intentionally read-only with respect to the dataset. It checks
that every manifest motion has exactly one usable language annotation and that
the root body in every NPZ shares the same coordinate frame as ``root_pos``.
The latter catches NPZs exported with Isaac's per-environment scene-grid origin
left in ``body_pos_w``.

The requested JSON report is written on both success and validation failure.
The process exits nonzero whenever the report contains a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPORT_NAME = "bones_seed_phase5_preflight"
REPORT_VERSION = 1
ROOT_BODY_NAMES = ("pelvis", "pelvis_link", "base_link", "root")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--language-sidecar",
        type=Path,
        default=None,
        help=(
            "Language JSON. When omitted, metadata.language_annotations_path "
            "is resolved relative to the manifest."
        ),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--body-offset-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--body-offset-drift-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--require-temporal-events",
        action="store_true",
        help="Require at least one valid temporal event for every motion.",
    )
    parser.add_argument(
        "--require-body-names",
        action="store_true",
        help="Reject repaired legacy NPZs that do not record body_names.",
    )
    return parser.parse_args()


def _issue(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"code": str(code), "message": str(message), **context}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(
    path: Path,
    *,
    label: str,
    failures: list[dict[str, Any]],
) -> Any | None:
    if not path.is_file():
        failures.append(
            _issue(
                f"{label}_missing",
                f"{label.replace('_', ' ').title()} not found: {path}",
                path=str(path),
            )
        )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(
            _issue(
                f"{label}_invalid_json",
                f"Could not read {label.replace('_', ' ')} JSON: {exc}",
                path=str(path),
            )
        )
        return None


def _manifest_entries(
    payload: Any,
    *,
    failures: list[dict[str, Any]],
) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        failures.append(
            _issue("manifest_schema", "Manifest root must be a JSON object.")
        )
        return []
    dataset = payload.get("dataset")
    trajectories = dataset.get("trajectories") if isinstance(dataset, Mapping) else None
    entries = (
        trajectories.get("lafan1_csv") if isinstance(trajectories, Mapping) else None
    )
    if not isinstance(entries, list) or not entries:
        failures.append(
            _issue(
                "manifest_schema",
                "Manifest must define a non-empty dataset.trajectories.lafan1_csv list.",
            )
        )
        return []
    valid: list[Mapping[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            failures.append(
                _issue(
                    "manifest_entry_schema",
                    f"Manifest entry #{index} must be a JSON object.",
                    manifest_index=index,
                )
            )
            continue
        valid.append(entry)
    return valid


def _resolve_language_sidecar(
    *,
    manifest: Path,
    manifest_payload: Any,
    override: Path | None,
    failures: list[dict[str, Any]],
) -> tuple[Path | None, str | None]:
    declared: str | None = None
    if isinstance(manifest_payload, Mapping):
        metadata = manifest_payload.get("metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get("language_annotations_path")
            if value is not None and str(value).strip():
                declared = str(value).strip()

    if override is not None:
        path = override.expanduser()
        if not path.is_absolute():
            path = path.resolve()
        return path, declared
    if declared is None:
        failures.append(
            _issue(
                "language_sidecar_unspecified",
                "Pass --language-sidecar or set metadata.language_annotations_path.",
            )
        )
        return None, None
    path = Path(declared).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve(), declared


def _root_body_index(
    body_names: np.ndarray | None,
    *,
    body_count: int,
) -> tuple[int, str | None]:
    if body_names is None:
        return 0, None
    flattened = np.asarray(body_names).reshape(-1)
    if int(flattened.size) != int(body_count):
        raise ValueError(
            f"body_names has {flattened.size} entries for {body_count} bodies."
        )
    names = [str(item) for item in flattened.tolist()]
    normalized = [name.strip().lower() for name in names]
    for candidate in ROOT_BODY_NAMES:
        if candidate in normalized:
            index = normalized.index(candidate)
            return index, names[index]
    return 0, names[0]


def _audit_npz(
    *,
    name: str,
    source: Path,
    input_fps: float | None,
    offset_tolerance: float,
    drift_tolerance: float,
    require_body_names: bool,
    failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "source": str(source),
        "passed": False,
    }
    if not source.is_file():
        failures.append(
            _issue(
                "motion_source_missing",
                f"Motion source is missing for {name}: {source}",
                motion_name=name,
                source=str(source),
            )
        )
        return record
    if source.suffix.lower() != ".npz":
        failures.append(
            _issue(
                "motion_source_not_npz",
                f"Phase-5 body-frame validation requires NPZ input for {name}.",
                motion_name=name,
                source=str(source),
            )
        )
        return record

    try:
        with np.load(source, allow_pickle=False) as data:
            required = ("root_pos", "body_pos_w")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(f"missing arrays: {missing}")
            root_pos = np.asarray(data["root_pos"])
            body_pos = np.asarray(data["body_pos_w"])
            body_names = (
                np.asarray(data["body_names"]) if "body_names" in data.files else None
            )
            npz_fps = (
                float(np.asarray(data["fps"]).reshape(-1)[0])
                if "fps" in data.files and np.asarray(data["fps"]).size
                else None
            )
    except (OSError, ValueError, KeyError) as exc:
        failures.append(
            _issue(
                "motion_source_invalid",
                f"Could not validate NPZ for {name}: {exc}",
                motion_name=name,
                source=str(source),
            )
        )
        return record

    if (
        root_pos.ndim != 2
        or tuple(root_pos.shape[1:]) != (3,)
        or body_pos.ndim != 3
        or int(body_pos.shape[0]) != int(root_pos.shape[0])
        or int(body_pos.shape[-1]) != 3
        or int(root_pos.shape[0]) <= 0
        or int(body_pos.shape[1]) <= 0
    ):
        failures.append(
            _issue(
                "motion_shape_invalid",
                f"Incompatible root/body shapes for {name}: "
                f"root={root_pos.shape}, body={body_pos.shape}.",
                motion_name=name,
                source=str(source),
                root_pos_shape=list(root_pos.shape),
                body_pos_w_shape=list(body_pos.shape),
            )
        )
        return record
    if not np.isfinite(root_pos).all() or not np.isfinite(body_pos).all():
        failures.append(
            _issue(
                "motion_nonfinite",
                f"Non-finite root or body positions found for {name}.",
                motion_name=name,
                source=str(source),
            )
        )
        return record

    try:
        root_body_index, root_body_name = _root_body_index(
            body_names, body_count=int(body_pos.shape[1])
        )
    except ValueError as exc:
        failures.append(
            _issue(
                "body_names_invalid",
                f"Invalid body_names for {name}: {exc}",
                motion_name=name,
                source=str(source),
            )
        )
        return record

    has_body_names = body_names is not None
    if not has_body_names:
        issue = _issue(
            "body_names_missing",
            f"NPZ for {name} has no body_names; assuming body index 0 is the root.",
            motion_name=name,
            source=str(source),
        )
        if require_body_names:
            failures.append(issue)
        else:
            warnings.append(issue)

    offsets = body_pos[:, root_body_index, :] - root_pos
    median_offset = np.median(offsets, axis=0)
    max_abs_offset = float(np.max(np.abs(offsets)))
    max_offset_drift = float(np.max(np.abs(offsets - median_offset)))
    scene_grid_like = bool(
        max_abs_offset > offset_tolerance and max_offset_drift <= drift_tolerance
    )
    record.update(
        {
            "frame_count": int(root_pos.shape[0]),
            "body_count": int(body_pos.shape[1]),
            "has_body_names": has_body_names,
            "root_body_index": int(root_body_index),
            "root_body_name": root_body_name,
            "manifest_input_fps": input_fps,
            "npz_fps": npz_fps,
            "median_offset_xyz": [float(value) for value in median_offset],
            "max_abs_offset": max_abs_offset,
            "max_offset_drift": max_offset_drift,
            "scene_grid_like_offset": scene_grid_like,
        }
    )

    if (
        input_fps is not None
        and npz_fps is not None
        and not math.isclose(
            float(input_fps), float(npz_fps), rel_tol=0.0, abs_tol=1.0e-6
        )
    ):
        failures.append(
            _issue(
                "motion_fps_mismatch",
                f"Manifest/NPZ FPS mismatch for {name}: {input_fps} != {npz_fps}.",
                motion_name=name,
                source=str(source),
                manifest_input_fps=float(input_fps),
                npz_fps=float(npz_fps),
            )
        )
    if max_offset_drift > drift_tolerance:
        failures.append(
            _issue(
                "body_root_offset_drift",
                f"Root-body offset drifts for {name}: {max_offset_drift:.6g} exceeds "
                f"{drift_tolerance:.6g}.",
                motion_name=name,
                source=str(source),
                max_offset_drift=max_offset_drift,
            )
        )
    if max_abs_offset > offset_tolerance:
        failures.append(
            _issue(
                "body_frame_offset",
                f"Root-body offset for {name} is {max_abs_offset:.6g}, exceeding "
                f"{offset_tolerance:.6g}.",
                motion_name=name,
                source=str(source),
                median_offset_xyz=[float(value) for value in median_offset],
                max_abs_offset=max_abs_offset,
                scene_grid_like_offset=scene_grid_like,
            )
        )

    record["passed"] = bool(
        max_abs_offset <= offset_tolerance
        and max_offset_drift <= drift_tolerance
        and (has_body_names or not require_body_names)
        and (
            input_fps is None
            or npz_fps is None
            or math.isclose(
                float(input_fps), float(npz_fps), rel_tol=0.0, abs_tol=1.0e-6
            )
        )
    )
    return record


def _validate_event(
    event: Any,
    *,
    motion_name: str,
    event_index: int,
    failures: list[dict[str, Any]],
) -> bool:
    if not isinstance(event, Mapping):
        failures.append(
            _issue(
                "language_event_schema",
                f"Event #{event_index} for {motion_name} must be a JSON object.",
                motion_name=motion_name,
                event_index=event_index,
            )
        )
        return False
    try:
        start = float(event["start_time"])
        end = float(event["end_time"])
    except (KeyError, TypeError, ValueError):
        failures.append(
            _issue(
                "language_event_schema",
                f"Event #{event_index} for {motion_name} needs numeric start/end times.",
                motion_name=motion_name,
                event_index=event_index,
            )
        )
        return False
    description = str(event.get("description", "")).strip()
    valid = math.isfinite(start) and math.isfinite(end) and start >= 0.0 and end > start
    if not valid or not description:
        failures.append(
            _issue(
                "language_event_schema",
                f"Event #{event_index} for {motion_name} has invalid timing or description.",
                motion_name=motion_name,
                event_index=event_index,
                start_time=start,
                end_time=end,
            )
        )
        return False
    return True


def _audit_language(
    payload: Any,
    *,
    manifest_names: list[str],
    require_temporal_events: bool,
    failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "motion_count": 0,
        "unique_motion_count": 0,
        "language_goal_count": 0,
        "temporal_event_motion_count": 0,
        "temporal_event_count": 0,
        "missing_manifest_names": [],
        "extra_sidecar_names": [],
        "order_matches_manifest": False,
        "motions": [],
    }
    if not isinstance(payload, Mapping):
        failures.append(
            _issue(
                "language_sidecar_schema", "Language sidecar root must be an object."
            )
        )
        return result
    motions = payload.get("motions")
    if not isinstance(motions, list) or not motions:
        failures.append(
            _issue(
                "language_sidecar_schema",
                "Language sidecar must define a non-empty motions list.",
            )
        )
        return result

    sidecar_names: list[str] = []
    seen: set[str] = set()
    motion_records: list[dict[str, Any]] = []
    language_goal_count = 0
    event_motion_count = 0
    event_count = 0
    for index, motion in enumerate(motions):
        if not isinstance(motion, Mapping):
            failures.append(
                _issue(
                    "language_motion_schema",
                    f"Language motion #{index} must be a JSON object.",
                    language_index=index,
                )
            )
            continue
        name = str(motion.get("name", "")).strip()
        if not name:
            failures.append(
                _issue(
                    "language_motion_name_missing",
                    f"Language motion #{index} has no name.",
                    language_index=index,
                )
            )
            continue
        sidecar_names.append(name)
        if name in seen:
            failures.append(
                _issue(
                    "language_motion_duplicate",
                    f"Language sidecar contains duplicate motion {name!r}.",
                    motion_name=name,
                    language_index=index,
                )
            )
        seen.add(name)

        goal = str(motion.get("language_goal", "")).strip()
        if goal:
            language_goal_count += 1
        else:
            failures.append(
                _issue(
                    "language_goal_missing",
                    f"Language sidecar has no language_goal for {name}.",
                    motion_name=name,
                )
            )

        events = motion.get("events", [])
        if not isinstance(events, list):
            failures.append(
                _issue(
                    "language_events_schema",
                    f"events for {name} must be a list.",
                    motion_name=name,
                )
            )
            events = []
        declared_event_count = motion.get("num_events", len(events))
        try:
            declared_event_count = int(declared_event_count)
        except (TypeError, ValueError):
            declared_event_count = -1
        if declared_event_count != len(events):
            failures.append(
                _issue(
                    "language_event_count_mismatch",
                    f"num_events for {name} is {declared_event_count}, but events has "
                    f"{len(events)} rows.",
                    motion_name=name,
                    declared_num_events=declared_event_count,
                    actual_num_events=len(events),
                )
            )
        valid_events = sum(
            _validate_event(
                event,
                motion_name=name,
                event_index=event_index,
                failures=failures,
            )
            for event_index, event in enumerate(events)
        )
        if valid_events:
            event_motion_count += 1
            event_count += valid_events
        elif require_temporal_events:
            failures.append(
                _issue(
                    "temporal_events_missing",
                    f"Phase-5 temporal events are required for {name}.",
                    motion_name=name,
                )
            )
        motion_records.append(
            {
                "name": name,
                "has_language_goal": bool(goal),
                "natural_description_count": len(motion.get("natural_descriptions", []))
                if isinstance(motion.get("natural_descriptions", []), list)
                else 0,
                "short_description_count": len(motion.get("short_descriptions", []))
                if isinstance(motion.get("short_descriptions", []), list)
                else 0,
                "event_count": len(events),
                "valid_event_count": valid_events,
            }
        )

    manifest_unique = list(dict.fromkeys(manifest_names))
    sidecar_unique = list(dict.fromkeys(sidecar_names))
    missing = sorted(set(manifest_unique) - set(sidecar_unique))
    extra = sorted(set(sidecar_unique) - set(manifest_unique))
    if missing:
        failures.append(
            _issue(
                "language_coverage_missing",
                f"Language sidecar is missing {len(missing)} manifest motions.",
                motion_names=missing,
            )
        )
    if extra:
        failures.append(
            _issue(
                "language_coverage_extra",
                f"Language sidecar contains {len(extra)} motions absent from the manifest.",
                motion_names=extra,
            )
        )
    order_matches = manifest_names == sidecar_names
    if not missing and not extra and not order_matches:
        warnings.append(
            _issue(
                "language_order_differs",
                "Language coverage is exact, but sidecar order differs from manifest order.",
            )
        )
    result.update(
        {
            "motion_count": len(sidecar_names),
            "unique_motion_count": len(sidecar_unique),
            "language_goal_count": language_goal_count,
            "temporal_event_motion_count": event_motion_count,
            "temporal_event_count": event_count,
            "missing_manifest_names": missing,
            "extra_sidecar_names": extra,
            "order_matches_manifest": order_matches,
            "motions": motion_records,
        }
    )
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    offset_tolerance = float(args.body_offset_tolerance)
    drift_tolerance = float(args.body_offset_drift_tolerance)
    if offset_tolerance < 0.0 or drift_tolerance < 0.0:
        failures.append(
            _issue(
                "invalid_tolerance",
                "Body offset tolerances must be non-negative.",
                body_offset_tolerance=offset_tolerance,
                body_offset_drift_tolerance=drift_tolerance,
            )
        )
        offset_tolerance = max(0.0, offset_tolerance)
        drift_tolerance = max(0.0, drift_tolerance)

    manifest_payload = _read_json(manifest, label="manifest", failures=failures)
    entries = _manifest_entries(manifest_payload, failures=failures)
    manifest_names: list[str] = []
    motion_records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(entries):
        name = str(entry.get("name", "")).strip()
        if not name:
            failures.append(
                _issue(
                    "manifest_motion_name_missing",
                    f"Manifest entry #{index} has no motion name.",
                    manifest_index=index,
                )
            )
            continue
        manifest_names.append(name)
        if name in seen_names:
            failures.append(
                _issue(
                    "manifest_motion_duplicate",
                    f"Manifest contains duplicate motion {name!r}.",
                    motion_name=name,
                    manifest_index=index,
                )
            )
        seen_names.add(name)
        path_value = entry.get("path") or entry.get("file")
        if path_value is None or not str(path_value).strip():
            failures.append(
                _issue(
                    "manifest_motion_path_missing",
                    f"Manifest motion {name} has no path.",
                    motion_name=name,
                    manifest_index=index,
                )
            )
            continue
        source = Path(str(path_value)).expanduser()
        if not source.is_absolute():
            source = manifest.parent / source
        source = source.resolve()
        raw_fps = entry.get("input_fps")
        input_fps: float | None
        try:
            input_fps = float(raw_fps)
            if not math.isfinite(input_fps) or input_fps <= 0.0:
                raise ValueError
        except (TypeError, ValueError):
            input_fps = None
            failures.append(
                _issue(
                    "manifest_input_fps_invalid",
                    f"Manifest motion {name} needs a positive input_fps.",
                    motion_name=name,
                    manifest_index=index,
                )
            )
        motion_records.append(
            _audit_npz(
                name=name,
                source=source,
                input_fps=input_fps,
                offset_tolerance=offset_tolerance,
                drift_tolerance=drift_tolerance,
                require_body_names=bool(args.require_body_names),
                failures=failures,
                warnings=warnings,
            )
        )

    language_path, declared_language_path = _resolve_language_sidecar(
        manifest=manifest,
        manifest_payload=manifest_payload,
        override=args.language_sidecar,
        failures=failures,
    )
    language_payload = None
    if language_path is not None:
        language_payload = _read_json(
            language_path, label="language_sidecar", failures=failures
        )
    language_report = _audit_language(
        language_payload,
        manifest_names=manifest_names,
        require_temporal_events=bool(args.require_temporal_events),
        failures=failures,
        warnings=warnings,
    )

    manifest_hash = _sha256(manifest) if manifest.is_file() else None
    language_hash = (
        _sha256(language_path)
        if language_path is not None and language_path.is_file()
        else None
    )
    passed_motion_count = sum(bool(record.get("passed")) for record in motion_records)
    report = {
        "schema": {"name": REPORT_NAME, "version": REPORT_VERSION},
        "passed": not failures,
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_hash,
            "motion_count": len(manifest_names),
            "unique_motion_count": len(set(manifest_names)),
        },
        "language_sidecar": {
            "path": str(language_path) if language_path is not None else None,
            "declared_path": declared_language_path,
            "sha256": language_hash,
            **language_report,
        },
        "body_frames": {
            "offset_tolerance": float(offset_tolerance),
            "drift_tolerance": float(drift_tolerance),
            "require_body_names": bool(args.require_body_names),
            "checked_motion_count": len(motion_records),
            "passed_motion_count": passed_motion_count,
            "failed_motion_count": len(motion_records) - passed_motion_count,
            "motions": motion_records,
        },
        "requirements": {
            "require_temporal_events": bool(args.require_temporal_events),
        },
        "summary": {
            "manifest_motion_count": len(manifest_names),
            "language_motion_count": int(language_report["motion_count"]),
            "language_goal_count": int(language_report["language_goal_count"]),
            "temporal_event_motion_count": int(
                language_report["temporal_event_motion_count"]
            ),
            "body_frame_passed_motion_count": passed_motion_count,
            "failure_count": len(failures),
            "warning_count": len(warnings),
        },
        "failures": failures,
        "warnings": warnings,
        "report_path": str(report_path),
    }
    return report


def main() -> None:
    args = _parse_args()
    report_path = args.report.expanduser().resolve()
    try:
        report = audit(args)
    except Exception as exc:  # pragma: no cover - final report safety net
        report = {
            "schema": {"name": REPORT_NAME, "version": REPORT_VERSION},
            "passed": False,
            "failures": [
                _issue(
                    "preflight_internal_error",
                    f"Unexpected preflight error: {type(exc).__name__}: {exc}",
                )
            ],
            "warnings": [],
            "report_path": str(report_path),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not bool(report.get("passed")):
        failure_count = len(report.get("failures", []))
        print(
            f"[FAIL] BONES-SEED Phase-5 preflight ({failure_count} failures): {report_path}"
        )
        raise SystemExit(1)
    print(f"[PASS] BONES-SEED Phase-5 preflight: {report_path}")


if __name__ == "__main__":
    main()
