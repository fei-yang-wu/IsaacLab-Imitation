#!/usr/bin/env python3
"""Create uniformly sampled contact-sheet storyboards from motion videos.

This helper is deliberately video-only: it does not launch Isaac or replay
motions. First render/reference-record videos with the existing repo tools,
then run this script to make compact storyboard images for visual language
caption review.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "lafan1" / "manifests" / "g1_lafan1_manifest.json"
)


def _extract_manifest_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        entries = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
        if entries is None:
            entries = data.get("lafan1_csv", data.get("motions"))
    else:
        entries = data
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Manifest must define a non-empty 'dataset.trajectories.lafan1_csv' list."
        )
    return entries


def _motion_names(manifest_path: Path) -> list[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for index, entry in enumerate(_extract_manifest_entries(data)):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry #{index} must be a mapping.")
        name = entry.get("name")
        if not name:
            raw_path = entry.get("path") or entry.get("file")
            if raw_path is None:
                raise ValueError(f"Manifest entry #{index} needs 'name' or 'path'.")
            name = Path(str(raw_path)).stem
        names.append(str(name))
    return list(dict.fromkeys(names))


def _find_video(video_dir: Path, pattern: str, name: str) -> Path | None:
    candidate = (video_dir / pattern.format(name=name)).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    for suffix in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
        fallback = video_dir / f"{name}{suffix}"
        if fallback.is_file():
            return fallback.resolve()
    return None


def _ffprobe_frame_count(video_path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe found no video stream: {video_path}")
    stream = streams[0]
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "N/A"):
            count = int(value)
            if count > 0:
                return count
    raise ValueError(f"Could not infer frame count with ffprobe: {video_path}")


def _uniform_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    sample_count = max(1, min(int(sample_count), frame_count))
    if sample_count == 1:
        return [0]
    values = [
        int(round(index * (frame_count - 1) / (sample_count - 1)))
        for index in range(sample_count)
    ]
    return list(dict.fromkeys(values))


def _select_expression(indices: list[int]) -> str:
    return "+".join(f"eq(n\\,{index})" for index in indices)


def _make_storyboard(
    *,
    video_path: Path,
    output_path: Path,
    frame_count: int,
    sample_count: int,
    columns: int,
    tile_width: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[storyboard] skip existing: {output_path}")
        return
    indices = _uniform_indices(frame_count, sample_count)
    if not indices:
        raise ValueError(f"No frames selected for {video_path}")
    columns = max(1, int(columns))
    rows = int(math.ceil(len(indices) / columns))
    filter_graph = (
        f"select='{_select_expression(indices)}',"
        f"scale={int(tile_width)}:-1:flags=lanczos,"
        f"tile={columns}x{rows}"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        filter_graph,
        "-frames:v",
        "1",
        str(output_path),
    ]
    print(
        f"[storyboard] {video_path.name}: {len(indices)} frames -> {output_path.name}"
    )
    if dry_run:
        print(" ".join(command))
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniformly sample motion videos into contact-sheet storyboards."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="Manifest whose motion names are used to locate videos.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing rendered videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where storyboard images are written.",
    )
    parser.add_argument(
        "--video_pattern",
        type=str,
        default="{name}.mp4",
        help="Video filename pattern under --video_dir; may use {name}.",
    )
    parser.add_argument(
        "--output_pattern",
        type=str,
        default="{name}.jpg",
        help="Storyboard filename pattern under --output_dir; may use {name}.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=16,
        help="Number of uniformly spaced frames to sample from each video.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns in the output contact sheet.",
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=320,
        help="Width in pixels for each sampled frame before tiling.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=None,
        help="Optional raw motion names to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing storyboard images.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print ffmpeg commands without running them.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    video_dir = Path(args.video_dir).expanduser().resolve()
    if not video_dir.is_dir():
        raise SystemExit(f"Video directory not found: {video_dir}")
    output_dir = Path(args.output_dir).expanduser().resolve()

    names = _motion_names(manifest_path)
    if args.select:
        wanted = set(args.select)
        names = [name for name in names if name in wanted]
        missing = sorted(wanted - set(names))
        if missing:
            raise SystemExit(f"Selected motion videos not in manifest: {missing}")

    processed = 0
    missing_videos: list[str] = []
    for name in names:
        video_path = _find_video(video_dir, args.video_pattern, name)
        if video_path is None:
            missing_videos.append(name)
            continue
        frame_count = _ffprobe_frame_count(video_path)
        output_path = output_dir / args.output_pattern.format(name=name)
        _make_storyboard(
            video_path=video_path,
            output_path=output_path,
            frame_count=frame_count,
            sample_count=args.sample_count,
            columns=args.columns,
            tile_width=args.tile_width,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        processed += 1

    print(f"[storyboard] processed: {processed}")
    if missing_videos:
        preview = ", ".join(missing_videos[:10])
        suffix = " ..." if len(missing_videos) > 10 else ""
        print(f"[storyboard] missing videos: {len(missing_videos)} ({preview}{suffix})")


if __name__ == "__main__":
    main()
