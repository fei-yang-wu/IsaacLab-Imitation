#!/usr/bin/env python3
"""Cut full-horizon evaluation videos into annotated semantic phase clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from imitation_experiments.evaluation.analyze_collected_latent_space import (
    load_semantic_phase_annotations,
)
from imitation_experiments.paths import REPO_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "phase"


def build_ffmpeg_command(
    *, input_path: Path, output_path: Path, start_step: int, end_step: int
) -> list[str]:
    """Build a frame-accurate, re-encoding ffmpeg phase-cut command."""
    if start_step < 0 or end_step <= start_step:
        raise ValueError(f"Invalid phase interval [{start_step}, {end_step}).")
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"trim=start_frame={start_step}:end_frame={end_step},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def _probe(path: Path) -> dict[str, float | int | None]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
    fps = float(numerator) / float(denominator)
    frames = stream.get("nb_frames")
    return {
        "frames": int(frames) if frames not in (None, "N/A") else None,
        "duration_s": float(payload["format"]["duration"]),
        "fps": fps,
    }


def segment_annotations(
    annotations_path: Path, output_dir: Path, *, overwrite: bool
) -> dict[str, Any]:
    """Render all annotated phase intervals and return their output manifest."""
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is unavailable: {executable}")
    annotations_path = annotations_path.expanduser().resolve()
    annotations = load_semantic_phase_annotations(annotations_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    for motion in annotations["motions"]:
        rank = int(motion["rank"])
        video_path = Path(str(motion["video_path"]))
        if not video_path.is_absolute():
            video_path = REPO_ROOT / video_path
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Annotated source video is missing: {video_path}")
        for phase_index, phase in enumerate(motion["phases"]):
            start = int(phase["start_step"])
            end = int(phase["end_step"])
            filename = (
                f"rank{rank:02d}_phase{phase_index:02d}_{start:04d}-{end:04d}_"
                f"{_slug(str(phase['label']))}.mp4"
            )
            output_path = output_dir / filename
            if overwrite or not output_path.is_file():
                subprocess.run(
                    build_ffmpeg_command(
                        input_path=video_path,
                        output_path=output_path,
                        start_step=start,
                        end_step=end,
                    ),
                    check=True,
                )
            clips.append(
                {
                    "rank": rank,
                    "motion_name": str(motion["motion_name"]),
                    "phase_index": phase_index,
                    "label": str(phase["label"]),
                    "activity": str(phase["activity"]),
                    "semantics": dict(phase["semantics"]),
                    "start_step": start,
                    "end_step": end,
                    "source_video": str(video_path),
                    "output_video": str(output_path),
                    "probe": _probe(output_path),
                }
            )
    manifest = {
        "schema": "semantic_phase_clip_manifest_v1",
        "annotations": str(annotations_path),
        "phase_definition": annotations["phase_definition"],
        "annotation_method": annotations.get("annotation_method"),
        "implicit_prop_note": annotations.get("implicit_prop_note"),
        "frame_convention": (
            "Reference intervals use the full reference length; comparison videos "
            "contain one frame per simulated transition and are therefore one "
            "frame shorter. Only each motion's final phase inherits that shortfall."
        ),
        "clip_count": len(clips),
        "clips": clips,
    }
    manifest_path = output_dir / "phase_clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = segment_annotations(
        args.annotations, args.output_dir, overwrite=bool(args.overwrite)
    )
    print(f"[PASS] {Path(args.output_dir).expanduser().resolve() / 'phase_clip_manifest.json'}")
    for clip in manifest["clips"]:
        print(f"[VIDEO] {clip['output_video']}")


if __name__ == "__main__":
    main()
