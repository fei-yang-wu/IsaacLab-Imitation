#!/usr/bin/env python3
"""Build frozen 50 Hz phase annotations from BONES temporal event labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--language_sidecar", type=Path, required=True)
    parser.add_argument("--trait_overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video_root", type=Path)
    parser.add_argument("--output_fps", type=float, default=50.0)
    return parser.parse_args()


def _slug(description: str, index: int) -> str:
    words = re.findall(r"[a-z0-9]+", description.lower())
    ignored = {"a", "an", "the", "person", "is", "their", "while", "and"}
    kept = [word for word in words if word not in ignored][:7]
    return "-".join(kept) if kept else f"phase-{index:02d}"


def _event_end_steps(
    events: list[dict[str, Any]], *, reference_frames: int, fps: float
) -> list[int]:
    """Convert event endpoints while guaranteeing positive contiguous phases."""
    if reference_frames < len(events):
        raise ValueError("Reference has fewer frames than temporal events.")
    result: list[int] = []
    previous = 0
    for index, event in enumerate(events):
        remaining = len(events) - index - 1
        if index == len(events) - 1:
            end = reference_frames
        else:
            end = int(round(float(event["end_time"]) * fps))
            end = max(previous + 1, min(end, reference_frames - remaining))
        result.append(end)
        previous = end
    return result


def build_annotations(
    *,
    selection_payload: dict[str, Any],
    language_payload: dict[str, Any],
    trait_payload: dict[str, Any],
    output_fps: float,
    video_root: Path | None,
) -> dict[str, Any]:
    """Join human temporal descriptions with a complete manual trait table."""
    selected = selection_payload.get("motions")
    language_rows = language_payload.get("motions")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Selection has no motions list.")
    if not isinstance(language_rows, list):
        raise ValueError("Language sidecar has no motions list.")
    if output_fps <= 0:
        raise ValueError("output_fps must be positive.")
    axes = trait_payload.get("semantic_axes")
    traits = trait_payload.get("motions")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("Trait table has no semantic_axes object.")
    if not isinstance(traits, dict):
        raise ValueError("Trait table has no motions object.")
    language_by_name = {str(row["name"]): row for row in language_rows}
    axis_names = tuple(str(name) for name in axes)
    motions: list[dict[str, Any]] = []
    for selection_rank, selected_row in enumerate(selected):
        name = str(selected_row["motion_name"])
        reference_frames = int(selected_row["reference_frames"])
        language_row = language_by_name.get(name)
        if language_row is None:
            raise ValueError(f"No temporal language row for {name!r}.")
        events = language_row.get("events")
        overrides = traits.get(name)
        if not isinstance(events, list) or not events:
            raise ValueError(f"{name} has no temporal events.")
        if not isinstance(overrides, list) or len(overrides) != len(events):
            raise ValueError(
                f"{name} needs exactly {len(events)} trait rows, got "
                f"{len(overrides) if isinstance(overrides, list) else None}."
            )
        ends = _event_end_steps(
            events, reference_frames=reference_frames, fps=output_fps
        )
        phases: list[dict[str, Any]] = []
        start = 0
        for index, (event, override, end) in enumerate(
            zip(events, overrides, ends, strict=True)
        ):
            if not isinstance(override, dict):
                raise ValueError(f"{name} trait row {index} is not an object.")
            activity = str(override.get("activity", "")).strip()
            true_axes = override.get("true_axes", [])
            if not activity or not isinstance(true_axes, list):
                raise ValueError(f"{name} trait row {index} lacks activity/true_axes.")
            unknown = sorted(set(str(axis) for axis in true_axes) - set(axis_names))
            if unknown:
                raise ValueError(
                    f"{name} trait row {index} has unknown axes {unknown}."
                )
            description = str(event["description"]).strip()
            phases.append(
                {
                    "start_step": start,
                    "end_step": end,
                    "label": str(override.get("label", _slug(description, index))),
                    "activity": activity,
                    "source_description": description,
                    "source_event_interval_s": [
                        float(event["start_time"]),
                        float(event["end_time"]),
                    ],
                    "semantics": {axis: axis in true_axes for axis in axis_names},
                }
            )
            start = end
        motion: dict[str, Any] = {
            "rank": selection_rank,
            "motion_name": name,
            "trajectory_rank": int(selected_row["trajectory_rank"]),
            "reference_frames": reference_frames,
            "category": str(selected_row.get("category", "unknown")),
            "language_goal": str(selected_row.get("language_goal", name)),
            "phases": phases,
        }
        if video_root is not None:
            motion["video_path"] = str(
                (
                    video_root
                    / f"rank{int(selected_row['trajectory_rank'])}"
                    / "videos"
                    / "compare_policy_reference"
                    / "rl-video-step-0.mp4"
                ).resolve()
            )
        motions.append(motion)
    return {
        "schema": "semantic_phase_annotations_v1",
        "phase_definition": (
            "A contiguous human-described BONES temporal event, converted from "
            f"seconds to exact zero-based, end-exclusive {output_fps:g} Hz steps."
        ),
        "annotation_method": (
            "Phase descriptions and times come from the BONES-SEED temporal language "
            "sidecar. Boolean traits were manually curated for this analysis; event "
            "endpoints were rounded to the prepared 50 Hz reference and the final "
            "endpoint was clamped to the exact reference length."
        ),
        "semantic_axes": axes,
        "activity_definition": trait_payload.get("activity_definition", {}),
        "output_fps": float(output_fps),
        "motions": motions,
    }


def main() -> None:
    args = _parse_args()
    selection = args.selection.expanduser().resolve()
    language = args.language_sidecar.expanduser().resolve()
    traits = args.trait_overrides.expanduser().resolve()
    output = args.output.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve() if args.video_root else None
    payload = build_annotations(
        selection_payload=json.loads(selection.read_text(encoding="utf-8")),
        language_payload=json.loads(language.read_text(encoding="utf-8")),
        trait_payload=json.loads(traits.read_text(encoding="utf-8")),
        output_fps=float(args.output_fps),
        video_root=video_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[PASS] Wrote {len(payload['motions'])} motions and "
        f"{sum(len(row['phases']) for row in payload['motions'])} phases."
    )
    print(f"[PASS] {output}")


if __name__ == "__main__":
    main()
