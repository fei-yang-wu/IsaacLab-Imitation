#!/usr/bin/env python3
"""Add explicit upstream BONES-SEED language fields to a language sidecar."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LANGUAGE_FIELDS = (
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--language-sidecar", type=Path, required=True)
    return parser.parse_args()


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split())


def main() -> None:
    args = _parse_args()
    with args.metadata_csv.open(newline="", encoding="utf-8") as stream:
        metadata = {row["filename"]: row for row in csv.DictReader(stream)}

    payload = json.loads(args.language_sidecar.read_text(encoding="utf-8"))
    motions = payload.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("Language sidecar must contain a non-empty motions list.")

    for motion in motions:
        filename = motion.get("bones_seed_filename")
        if filename not in metadata:
            raise ValueError(f"No BONES metadata row for {filename!r}.")
        row = metadata[filename]
        options = {
            field: str(row.get(field) or "").strip() or None for field in LANGUAGE_FIELDS
        }

        expected_natural = [options[field] for field in LANGUAGE_FIELDS[:4] if options[field]]
        expected_short = [options[field] for field in LANGUAGE_FIELDS[5:] if options[field]]
        if [_normalized(value) for value in motion.get("natural_descriptions", [])] != [
            _normalized(value) for value in expected_natural
        ]:
            raise ValueError(f"Natural descriptions differ from metadata for {filename!r}.")
        if [_normalized(value) for value in motion.get("short_descriptions", [])] != [
            _normalized(value) for value in expected_short
        ]:
            raise ValueError(f"Short descriptions differ from metadata for {filename!r}.")
        if _normalized(motion.get("technical_description")) != options[
            "content_technical_description"
        ]:
            raise ValueError(f"Technical description differs from metadata for {filename!r}.")

        goal = _normalized(motion.get("language_goal"))
        goal_sources = [
            field for field, value in options.items() if _normalized(value) == goal
        ]
        if not goal_sources:
            raise ValueError(f"language_goal does not match official metadata for {filename!r}.")

        motion["language_goal_source"] = goal_sources[0]
        motion["language_goal_sources"] = goal_sources
        motion["language_options"] = options

    payload["language_schema"] = {
        "version": 1,
        "language_goal": "Command selected for the current benchmark configuration.",
        "language_goal_source": "Official BONES-SEED column used for language_goal.",
        "language_options": "All official full-motion language columns, preserved by name.",
        "events": "Official temporal segment descriptions when available.",
    }
    args.language_sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Enriched {len(motions)} motions in {args.language_sidecar}")


if __name__ == "__main__":
    main()
