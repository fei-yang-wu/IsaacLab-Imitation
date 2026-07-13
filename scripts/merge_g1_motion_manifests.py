#!/usr/bin/env python3
"""Merge G1 LAFAN1-style NPZ manifests into one language-ready dataset.

The output manifest keeps the existing ``dataset.trajectories.lafan1_csv``
schema so IsaacLab-Imitation can load it without a new dataset loader. Motion
names are source-prefixed by default, and a compact language sidecar is written
alongside the manifest for building language-goal embedding tables.

Example:
    pixi run python scripts/merge_g1_motion_manifests.py

    pixi run python scripts/rlopt/build_language_goal_embeddings.py \
        --manifest data/unified/manifests/g1_lafan1_dance102_bones_seed_manifest.json \
        --language_sidecar data/unified/language/g1_lafan1_dance102_bones_seed_language.json \
        --output data/unified/language/g1_lafan1_dance102_bones_seed_name_embeddings.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LAFAN1_MANIFEST = REPO_ROOT / "data/lafan1/manifests/g1_lafan1_manifest.json"
DEFAULT_LAFAN1_LANGUAGE = (
    REPO_ROOT
    / "data/lafan1/language/g1_lafan1_manifest.with_codex_storyboard_language_v1.json"
)
DEFAULT_DANCE102_MANIFEST = (
    REPO_ROOT / "data/unitree/manifests/g1_unitree_dance102_manifest.json"
)
DEFAULT_BONES_MANIFEST = (
    REPO_ROOT / "data/bones_seed/manifests/g1_bones_seed_10_manifest.json"
)
DEFAULT_BONES_LANGUAGE = (
    REPO_ROOT / "data/bones_seed/language/g1_bones_seed_10_language.json"
)
DEFAULT_OUTPUT_MANIFEST = (
    REPO_ROOT / "data/unified/manifests/g1_lafan1_dance102_bones_seed_manifest.json"
)
DEFAULT_OUTPUT_LANGUAGE = (
    REPO_ROOT / "data/unified/language/g1_lafan1_dance102_bones_seed_language.json"
)

LANGUAGE_FIELD_PRIORITY = (
    "language_goal",
    "robot_instruction",
    "kinematic_description",
    "short_caption",
    "event_level",
    "technical_description",
    "fallback_category_prompt",
)


@dataclass(frozen=True)
class SourceSpec:
    label: str
    prefix: str
    manifest_path: Path
    language_path: Path | None = None
    fallback_goal: str | None = None


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def _sanitize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-/]+", "_", str(value))
    name = name.replace("/", "__").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "motion"


def _humanize_motion_name(name: str) -> str:
    base = re.sub(r"^[A-Za-z0-9]+__", "", str(name))
    base = re.sub(r"_subject\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", base)
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip().lower()
    return base or str(name).strip().lower()


def _load_manifest_entries(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {manifest_path}")
    entries = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Manifest must define a non-empty dataset.trajectories.lafan1_csv: {manifest_path}"
        )
    return data, entries


def _resolve_motion_path(entry: dict[str, Any], manifest_path: Path) -> Path:
    path_value = entry.get("path") or entry.get("file")
    if path_value is None:
        raise ValueError(f"Manifest entry is missing path/file: {entry}")
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dedupe_text(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _language_goal_from_payload(payload: dict[str, Any]) -> str | None:
    values = [payload.get(field) for field in LANGUAGE_FIELD_PRIORITY]
    return _first_text(*values)


def _compact_language_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "category",
        "language_goal",
        "short_caption",
        "robot_instruction",
        "kinematic_description",
        "event_level",
        "attribute_text",
        "technical_description",
        "fallback_category_prompt",
        "source",
        "review_status",
        "needs_human_review",
        "confidence",
        "distinguishing_features",
        "short_descriptions",
        "natural_descriptions",
        "events",
        "num_events",
    ):
        if key in payload:
            compact[key] = payload[key]
    return compact


def _load_language_by_name(language_path: Path | None) -> dict[str, dict[str, Any]]:
    if language_path is None:
        return {}
    if not language_path.is_file():
        raise FileNotFoundError(f"Language sidecar not found: {language_path}")
    data = json.loads(language_path.read_text(encoding="utf-8"))
    by_name: dict[str, dict[str, Any]] = {}

    def add(name: Any, payload: Any) -> None:
        if not name or not isinstance(payload, dict):
            return
        by_name[str(name)] = _compact_language_payload(payload)

    if isinstance(data, dict):
        motions = data.get("motions")
        if isinstance(motions, list):
            for item in motions:
                if isinstance(item, dict):
                    add(item.get("name"), item)

        prompts = data.get("prompts")
        if isinstance(prompts, dict):
            for name, payload in prompts.items():
                add(name, payload)

        entries = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                payload = entry.get("language")
                add(entry.get("name"), payload if isinstance(payload, dict) else entry)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                add(item.get("name"), item)

    return by_name


def _source_specs_from_args(args: argparse.Namespace) -> list[SourceSpec]:
    specs = [
        SourceSpec(
            label="lafan1",
            prefix=args.lafan1_prefix,
            manifest_path=_resolve_path(args.lafan1_manifest),
            language_path=_resolve_path(args.lafan1_language)
            if args.lafan1_language
            else None,
        ),
        SourceSpec(
            label="unitree_dance102",
            prefix=args.dance102_prefix,
            manifest_path=_resolve_path(args.dance102_manifest),
            fallback_goal=(
                "Perform a full-body humanoid dance routine with rhythmic steps, "
                "turns, and coordinated arm motion."
            ),
        ),
        SourceSpec(
            label="bones_seed",
            prefix=args.bones_prefix,
            manifest_path=_resolve_path(args.bones_manifest),
            language_path=_resolve_path(args.bones_language)
            if args.bones_language
            else None,
        ),
    ]
    if args.exclude_lafan1:
        specs = [spec for spec in specs if spec.label != "lafan1"]
    if args.exclude_dance102:
        specs = [spec for spec in specs if spec.label != "unitree_dance102"]
    if args.exclude_bones:
        specs = [spec for spec in specs if spec.label != "bones_seed"]
    return specs


def _merge_sources(
    sources: list[SourceSpec],
    *,
    output_manifest: Path,
    output_language: Path,
    dataset_name: str,
    prefix_names: bool,
    allow_missing_motion_files: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_entries: list[dict[str, Any]] = []
    language_motions: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    names_seen: set[str] = set()
    fps_values: list[float] = []

    for source in sources:
        source_data, source_entries = _load_manifest_entries(source.manifest_path)
        language_by_name = _load_language_by_name(source.language_path)
        matched_language = 0
        source_manifest_entries = 0

        for entry in source_entries:
            source_manifest_entries += 1
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{source.manifest_path} contains a non-mapping entry: {entry!r}"
                )
            original_name = str(entry.get("name") or "")
            motion_path = _resolve_motion_path(entry, source.manifest_path)
            if not motion_path.is_file() and not allow_missing_motion_files:
                raise FileNotFoundError(
                    f"Motion file for {source.label}:{original_name} does not exist: {motion_path}"
                )

            safe_original_name = _sanitize_name(original_name or motion_path.stem)
            merged_name = (
                f"{source.prefix}{safe_original_name}" if prefix_names else safe_original_name
            )
            if merged_name in names_seen:
                raise ValueError(f"Duplicate merged motion name: {merged_name}")
            names_seen.add(merged_name)

            input_fps = float(entry.get("input_fps", 50.0))
            fps_values.append(input_fps)
            merged_entry: dict[str, Any] = {
                "name": merged_name,
                "path": _relpath(motion_path, output_manifest.parent),
                "input_fps": input_fps,
                "source_dataset": source.label,
                "source_motion_name": original_name or motion_path.stem,
            }
            if "frame_range" in entry:
                merged_entry["frame_range"] = entry["frame_range"]
            manifest_entries.append(merged_entry)

            source_language = language_by_name.get(original_name, {})
            if source_language:
                matched_language += 1
            language_goal = _language_goal_from_payload(source_language)
            if language_goal is None:
                language_goal = source.fallback_goal or _humanize_motion_name(
                    original_name or motion_path.stem
                )
            natural_descriptions = _dedupe_text(
                [
                    language_goal,
                    *[
                        str(item)
                        for item in source_language.get("natural_descriptions", [])
                        if isinstance(item, str)
                    ],
                    *[
                        str(item)
                        for item in source_language.get("short_descriptions", [])
                        if isinstance(item, str)
                    ],
                    *[
                        str(source_language[field])
                        for field in (
                            "robot_instruction",
                            "kinematic_description",
                            "event_level",
                            "short_caption",
                            "technical_description",
                        )
                        if isinstance(source_language.get(field), str)
                    ],
                ]
            )
            language_entry: dict[str, Any] = {
                "name": merged_name,
                "source_dataset": source.label,
                "original_name": original_name or motion_path.stem,
                "path": _relpath(motion_path, output_language.parent),
                "language_goal": language_goal,
                "natural_descriptions": natural_descriptions,
            }
            for key in (
                "category",
                "short_caption",
                "robot_instruction",
                "kinematic_description",
                "event_level",
                "attribute_text",
                "source",
                "review_status",
                "needs_human_review",
                "confidence",
                "distinguishing_features",
                "events",
                "num_events",
            ):
                if key in source_language:
                    language_entry[key] = source_language[key]
            language_motions.append(language_entry)

        source_summaries.append(
            {
                "label": source.label,
                "prefix": source.prefix if prefix_names else "",
                "manifest": _relpath(source.manifest_path, output_manifest.parent),
                "language": _relpath(source.language_path, output_manifest.parent)
                if source.language_path is not None
                else None,
                "source_dataset_name": source_data.get("dataset_name"),
                "num_motions": source_manifest_entries,
                "language_matches": matched_language,
            }
        )

    control_freq = None
    if fps_values and all(abs(value - fps_values[0]) <= 1.0e-6 for value in fps_values):
        control_freq = fps_values[0]

    manifest = {
        "dataset_name": dataset_name,
        "dataset": {"trajectories": {"lafan1_csv": manifest_entries}},
        "metadata": {
            "num_motions": len(manifest_entries),
            "sources": source_summaries,
            "paths_are_relative_to_manifest": True,
            "generated_from_existing_npz": True,
            "fps_values": sorted(set(fps_values)),
            "control_freq": control_freq,
            "loader_kwargs": {"chunk_size": 1, "shard_size": 512},
            "language_annotations_path": _relpath(
                output_language, output_manifest.parent
            ),
            "name_prefixes_enabled": bool(prefix_names),
        },
    }
    language = {
        "dataset_name": dataset_name,
        "manifest": _relpath(output_manifest, output_language.parent),
        "source_manifests": source_summaries,
        "input_fps": control_freq,
        "output_fps": control_freq,
        "motions": language_motions,
    }
    return manifest, language


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge G1 LAFAN1-style manifests and language sidecars."
    )
    parser.add_argument("--lafan1_manifest", default=str(DEFAULT_LAFAN1_MANIFEST))
    parser.add_argument("--lafan1_language", default=str(DEFAULT_LAFAN1_LANGUAGE))
    parser.add_argument("--dance102_manifest", default=str(DEFAULT_DANCE102_MANIFEST))
    parser.add_argument("--bones_manifest", default=str(DEFAULT_BONES_MANIFEST))
    parser.add_argument("--bones_language", default=str(DEFAULT_BONES_LANGUAGE))
    parser.add_argument("--output_manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--output_language", default=str(DEFAULT_OUTPUT_LANGUAGE))
    parser.add_argument(
        "--dataset_name",
        default="g1_lafan1_dance102_bones_seed",
        help="Dataset name stored in the merged manifest.",
    )
    parser.add_argument("--lafan1_prefix", default="lafan1__")
    parser.add_argument("--dance102_prefix", default="unitree_dance102__")
    parser.add_argument("--bones_prefix", default="bones_seed__")
    parser.add_argument(
        "--no_prefix_names",
        action="store_true",
        default=False,
        help="Keep original motion names instead of prefixing by source.",
    )
    parser.add_argument("--exclude_lafan1", action="store_true", default=False)
    parser.add_argument("--exclude_dance102", action="store_true", default=False)
    parser.add_argument("--exclude_bones", action="store_true", default=False)
    parser.add_argument(
        "--allow_missing_motion_files",
        action="store_true",
        default=False,
        help="Write the manifest even if a referenced NPZ is missing.",
    )
    args = parser.parse_args()

    output_manifest = _resolve_path(args.output_manifest)
    output_language = _resolve_path(args.output_language)
    sources = _source_specs_from_args(args)
    if not sources:
        raise SystemExit("No sources selected.")

    manifest, language = _merge_sources(
        sources,
        output_manifest=output_manifest,
        output_language=output_language,
        dataset_name=args.dataset_name,
        prefix_names=not args.no_prefix_names,
        allow_missing_motion_files=bool(args.allow_missing_motion_files),
    )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_language.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output_language.write_text(json.dumps(language, indent=2) + "\n", encoding="utf-8")

    counts = {
        summary["label"]: summary["num_motions"]
        for summary in manifest["metadata"]["sources"]
    }
    print(f"[INFO] Wrote manifest: {output_manifest}")
    print(f"[INFO] Wrote language: {output_language}")
    print(f"[INFO] Motion count: {manifest['metadata']['num_motions']} {counts}")
    print(f"[INFO] FPS values: {manifest['metadata']['fps_values']}")


if __name__ == "__main__":
    main()
