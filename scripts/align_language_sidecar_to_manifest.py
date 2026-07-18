#!/usr/bin/env python3
"""Write language annotations in the exact order of a motion manifest.

The input may be an existing ``motions`` sidecar (used by BONES-SEED) or the
older LAFAN1 Codex storyboard ``prompts`` mapping.  The LAFAN1 conversion uses
``robot_instruction`` as the deployable ``language_goal`` and deliberately
omits workstation-local paths and diagnostic motion statistics.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--format",
        choices=("motions", "lafan_prompts"),
        required=True,
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_names(payload: Any) -> list[str]:
    entries = payload["dataset"]["trajectories"]["lafan1_csv"]
    names = [str(entry["name"]) for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Manifest contains duplicate motion names.")
    return names


def _ordered_records(
    names: list[str], records: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_name = {str(record["name"]): dict(record) for record in records}
    if len(by_name) != len(records):
        raise ValueError("Language sidecar contains duplicate motion names.")
    missing = sorted(set(names) - set(by_name))
    extra = sorted(set(by_name) - set(names))
    if missing or extra:
        raise ValueError(
            f"Manifest/language coverage differs: missing={missing}, extra={extra}."
        )
    return [by_name[name] for name in names]


def _motions_payload(
    *,
    source: Mapping[str, Any],
    names: list[str],
    manifest_path: Path,
    output_path: Path,
    dataset_name: str,
) -> dict[str, Any]:
    motions = source.get("motions")
    if not isinstance(motions, list):
        raise ValueError("Expected a motions list in the language sidecar.")
    ordered = _ordered_records(names, motions)
    for record in ordered:
        if not str(record.get("language_goal", "")).strip():
            raise ValueError(f"Missing language_goal for {record['name']!r}.")
    result = dict(source)
    result["dataset_name"] = dataset_name
    result["manifest"] = os.path.relpath(manifest_path, output_path.parent)
    result["motions"] = ordered
    return result


def _lafan_payload(
    *,
    source: Mapping[str, Any],
    names: list[str],
    manifest_path: Path,
    output_path: Path,
    dataset_name: str,
) -> dict[str, Any]:
    prompts = source.get("prompts")
    if not isinstance(prompts, Mapping):
        raise ValueError("Expected a prompts mapping in the LAFAN1 sidecar.")
    missing = sorted(set(names) - set(prompts))
    extra = sorted(set(prompts) - set(names))
    if missing or extra:
        raise ValueError(
            f"Manifest/prompt coverage differs: missing={missing}, extra={extra}."
        )
    motions: list[dict[str, Any]] = []
    for name in names:
        prompt = prompts[name]
        if not isinstance(prompt, Mapping):
            raise ValueError(f"Prompt for {name!r} must be an object.")
        language_goal = str(prompt.get("robot_instruction", "")).strip()
        if not language_goal:
            raise ValueError(f"Missing robot_instruction for {name!r}.")
        short_caption = str(prompt.get("short_caption", "")).strip()
        kinematic = str(prompt.get("kinematic_description", "")).strip()
        natural = [text for text in (short_caption, language_goal, kinematic) if text]
        motions.append(
            {
                "name": name,
                "category": str(prompt.get("category", "")).strip(),
                "language_goal": language_goal,
                "natural_descriptions": natural,
                "short_descriptions": [short_caption] if short_caption else [],
                "technical_description": kinematic,
                "event_level_description": str(
                    prompt.get("event_level", "")
                ).strip(),
                "attribute_text": str(prompt.get("attribute_text", "")).strip(),
                "distinguishing_features": list(
                    prompt.get("distinguishing_features", [])
                ),
                "source": str(prompt.get("source", source.get("source", ""))),
                "review_status": str(prompt.get("review_status", "")),
                "confidence": str(prompt.get("confidence", "")),
                "events": [],
                "num_events": 0,
            }
        )
    return {
        "schema_version": "g1_motion_language_commands_v1",
        "dataset_name": dataset_name,
        "source": str(source.get("source", "codex_storyboard_caption_v1")),
        "manifest": os.path.relpath(manifest_path, output_path.parent),
        "language_goal_field": "robot_instruction",
        "review_status": str(source.get("review_status", "")),
        "motions": motions,
    }


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    names = _manifest_names(_read_json(manifest_path))
    source = _read_json(input_path)
    if not isinstance(source, Mapping):
        raise ValueError("Language sidecar root must be an object.")
    if args.format == "motions":
        result = _motions_payload(
            source=source,
            names=names,
            manifest_path=manifest_path,
            output_path=output_path,
            dataset_name=args.dataset_name,
        )
    else:
        result = _lafan_payload(
            source=source,
            names=names,
            manifest_path=manifest_path,
            output_path=output_path,
            dataset_name=args.dataset_name,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] Wrote {len(names)} ordered language records: {output_path}")


if __name__ == "__main__":
    main()
