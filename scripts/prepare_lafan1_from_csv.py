#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch-convert CSV motions to NPZ and emit a LAFAN1 manifest JSON.

Workflow:
1) Find CSV files under a folder.
2) Convert CSV -> NPZ in one batched Isaac Sim session.
3) Optionally record one MP4 per motion using a per-env camera in that same session.
4) Write one manifest JSON consumable by ImitationG1LafanTrackEnvCfg.

Example:
    conda run -n SL python scripts/prepare_lafan1_from_csv.py \
      --csv_dir /abs/path/csv_motions \
      --npz_dir /abs/path/npz_motions \
      --manifest_path /abs/path/g1_lafan1_manifest.json \
      --recursive --headless

    conda run -n SL python scripts/prepare_lafan1_from_csv.py \
      --csv_dir /abs/path/csv_motions \
      --npz_dir /abs/path/npz_motions \
      --manifest_path /abs/path/g1_lafan1_manifest.json \
      --recursive --auto_trim_mode g1_shoulder_roll --overwrite --headless
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


G1_CSV_JOINT_OFFSET = 7
G1_LEFT_SHOULDER_ROLL_INDEX = 16
G1_RIGHT_SHOULDER_ROLL_INDEX = 23


def _resolve_default_converter() -> Path:
    return Path(__file__).resolve().with_name("csv_to_npz.py")


def _resolve_default_batch_converter() -> Path:
    return Path(__file__).resolve().with_name("batch_csv_to_npz.py")


def _sanitize_motion_name(path_without_suffix: Path) -> str:
    name = path_without_suffix.as_posix()
    name = re.sub(r"[^A-Za-z0-9_\-/]+", "_", name)
    name = name.replace("/", "__").replace("-", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "motion"


def _discover_csv_files(csv_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in csv_dir.glob(pattern) if path.is_file())


def _build_npz_path(csv_file: Path, csv_root: Path, npz_root: Path) -> Path:
    relative = csv_file.relative_to(csv_root)
    return (npz_root / relative).with_suffix(".npz")


def _run_conversion(
    *,
    csv_file: Path,
    npz_file: Path,
    converter_script: Path,
    python_exe: str,
    input_fps: float,
    output_fps: float,
    frame_range: tuple[int, int] | None,
    headless: bool,
    device: str | None,
    video_output: Path | None,
    overwrite_video: bool,
) -> None:
    cmd = [
        python_exe,
        str(converter_script),
        "-f",
        str(csv_file),
        "--input_fps",
        str(input_fps),
        "--output_name",
        str(npz_file),
        "--output_fps",
        str(output_fps),
    ]
    if frame_range is not None:
        cmd.extend(["--frame_range", str(frame_range[0]), str(frame_range[1])])
    if device is not None:
        cmd.extend(["--device", device])
    if headless:
        cmd.append("--headless")
    if video_output is not None:
        cmd.extend(["--video", "--video_output", str(video_output)])
        if overwrite_video:
            cmd.append("--overwrite_video")

    print(f"[INFO] Converting: {csv_file} -> {npz_file}")
    print(f"[CMD]  {' '.join(shlex.quote(x) for x in cmd)}")
    subprocess.run(cmd, check=True)


def _run_batch_conversion(
    *,
    jobs: list[dict[str, object]],
    batch_converter_script: Path,
    python_exe: str,
    input_fps: float,
    output_fps: float,
    frame_range: tuple[int, int] | None,
    headless: bool,
    device: str | None,
    record_videos: bool,
    overwrite_videos: bool,
    video_width: int,
    video_height: int,
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="lafan1_batch_jobs_",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(jobs, handle)
        temp_jobs_path = Path(handle.name)

    cmd = [
        python_exe,
        str(batch_converter_script),
        "--jobs_json",
        str(temp_jobs_path),
        "--input_fps",
        str(input_fps),
        "--output_fps",
        str(output_fps),
    ]
    if frame_range is not None:
        cmd.extend(["--frame_range", str(frame_range[0]), str(frame_range[1])])
    if device is not None:
        cmd.extend(["--device", device])
    if headless:
        cmd.append("--headless")
    if record_videos:
        cmd.extend(
            [
                "--video",
                "--video_width",
                str(video_width),
                "--video_height",
                str(video_height),
            ]
        )
        if overwrite_videos:
            cmd.append("--overwrite_video")

    print(f"[INFO] Batch converting {len(jobs)} motion(s) in one Isaac Sim session.")
    print(f"[CMD]  {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        subprocess.run(cmd, check=True)
    finally:
        temp_jobs_path.unlink(missing_ok=True)


def _default_video_dir(npz_dir: Path) -> Path:
    if npz_dir.parent != npz_dir:
        return npz_dir.parent / "videos"
    return npz_dir / "videos"


def _build_video_output(csv_file: Path, csv_root: Path, video_root: Path) -> Path:
    relative = csv_file.relative_to(csv_root)
    return (video_root / relative).with_suffix(".mp4")


def _count_csv_frames(csv_file: Path) -> int:
    with csv_file.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _read_npz_fps(npz_file: Path, fallback_fps: float) -> float:
    try:
        with np.load(npz_file) as npz_data:
            if "fps" in npz_data.files:
                return float(np.asarray(npz_data["fps"]).reshape(-1)[0])
    except Exception:
        pass
    return float(fallback_fps)


def _validate_frame_range(
    frame_range: tuple[int, int], total_frames: int, *, csv_file: Path
) -> tuple[int, int]:
    start, end = frame_range
    if start < 1:
        raise ValueError(f"frame_range start must be >= 1 for {csv_file}.")
    if end < start:
        raise ValueError(
            f"frame_range end must be >= start for {csv_file}: {frame_range}."
        )
    if end > total_frames:
        raise ValueError(
            f"frame_range {frame_range} exceeds csv length {total_frames} for {csv_file}."
        )
    return (start, end)


def _find_first_consecutive_true(values: np.ndarray, *, hold_frames: int) -> int | None:
    run_length = 0
    for index, flag in enumerate(values):
        run_length = run_length + 1 if bool(flag) else 0
        if run_length >= hold_frames:
            return index - hold_frames + 2
    return None


def _infer_g1_shoulder_roll_trim_start(
    csv_file: Path,
    *,
    search_frames: int,
    shoulder_roll_threshold: float,
    hold_frames: int,
    pre_roll_frames: int,
) -> tuple[int, int, dict[str, object]]:
    total_frames = _count_csv_frames(csv_file)
    max_rows = min(int(search_frames), total_frames)
    usecols = (
        G1_CSV_JOINT_OFFSET + G1_LEFT_SHOULDER_ROLL_INDEX,
        G1_CSV_JOINT_OFFSET + G1_RIGHT_SHOULDER_ROLL_INDEX,
    )
    shoulder_roll = np.loadtxt(
        csv_file,
        delimiter=",",
        dtype=np.float32,
        usecols=usecols,
        max_rows=max_rows,
    )
    shoulder_roll = np.atleast_2d(shoulder_roll)
    shoulder_roll = shoulder_roll.reshape(-1, 2)
    below_threshold = np.logical_and(
        np.abs(shoulder_roll[:, 0]) <= shoulder_roll_threshold,
        np.abs(shoulder_roll[:, 1]) <= shoulder_roll_threshold,
    )
    detected_start = _find_first_consecutive_true(
        below_threshold, hold_frames=hold_frames
    )
    if detected_start is None:
        return (
            1,
            total_frames,
            {
                "detected": False,
                "detected_start_frame": None,
                "applied_start_frame": 1,
                "total_frames": total_frames,
                "search_frames": max_rows,
            },
        )

    applied_start = max(1, int(detected_start) - int(pre_roll_frames))
    applied_start = min(applied_start, total_frames)
    return (
        applied_start,
        total_frames,
        {
            "detected": True,
            "detected_start_frame": int(detected_start),
            "applied_start_frame": int(applied_start),
            "total_frames": total_frames,
            "search_frames": max_rows,
        },
    )


def _resolve_motion_frame_ranges(
    *,
    csv_files: list[Path],
    args: argparse.Namespace,
) -> tuple[dict[Path, tuple[int, int] | None], dict[Path, dict[str, object]]]:
    if args.frame_range is not None and args.auto_trim_mode != "none":
        raise ValueError("Use either --frame_range or --auto_trim_mode, not both.")
    if args.auto_trim_search_frames <= 0:
        raise ValueError("--auto_trim_search_frames must be > 0.")
    if args.auto_trim_hold_frames <= 0:
        raise ValueError("--auto_trim_hold_frames must be > 0.")
    if args.auto_trim_pre_roll < 0:
        raise ValueError("--auto_trim_pre_roll must be >= 0.")
    if args.auto_trim_shoulder_roll_threshold <= 0.0:
        raise ValueError("--auto_trim_shoulder_roll_threshold must be > 0.")

    frame_ranges: dict[Path, tuple[int, int] | None] = {}
    trim_details: dict[Path, dict[str, object]] = {}

    requested_frame_range = (
        tuple(int(value) for value in args.frame_range)
        if args.frame_range is not None
        else None
    )

    for csv_file in csv_files:
        if requested_frame_range is not None:
            total_frames = _count_csv_frames(csv_file)
            resolved = _validate_frame_range(
                requested_frame_range, total_frames, csv_file=csv_file
            )
            source_frame_range = (
                [int(resolved[0]), int(resolved[1])]
                if resolved != (1, total_frames)
                else None
            )
            frame_ranges[csv_file] = (
                resolved if source_frame_range is not None else None
            )
            trim_details[csv_file] = {
                "mode": "manual",
                "detected": True,
                "detected_start_frame": int(resolved[0]),
                "applied_start_frame": int(resolved[0]),
                "total_frames": int(total_frames),
                "source_frame_range": source_frame_range,
            }
            continue

        if args.auto_trim_mode == "g1_shoulder_roll":
            start_frame, total_frames, detail = _infer_g1_shoulder_roll_trim_start(
                csv_file,
                search_frames=int(args.auto_trim_search_frames),
                shoulder_roll_threshold=float(args.auto_trim_shoulder_roll_threshold),
                hold_frames=int(args.auto_trim_hold_frames),
                pre_roll_frames=int(args.auto_trim_pre_roll),
            )
            source_frame_range = (
                [int(start_frame), int(total_frames)] if start_frame > 1 else None
            )
            if start_frame <= 1:
                frame_ranges[csv_file] = None
            else:
                frame_ranges[csv_file] = (int(start_frame), int(total_frames))
            trim_details[csv_file] = {
                "mode": "g1_shoulder_roll",
                **detail,
                "source_frame_range": source_frame_range,
                "threshold": float(args.auto_trim_shoulder_roll_threshold),
                "hold_frames": int(args.auto_trim_hold_frames),
                "pre_roll": int(args.auto_trim_pre_roll),
            }
            continue

        frame_ranges[csv_file] = None
        trim_details[csv_file] = {
            "mode": "none",
            "detected": False,
            "detected_start_frame": None,
            "applied_start_frame": 1,
            "total_frames": int(_count_csv_frames(csv_file)),
            "source_frame_range": None,
        }

    return frame_ranges, trim_details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert CSV motions to NPZ and generate manifest JSON."
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Input folder containing CSV motion files.",
    )
    parser.add_argument(
        "--npz_dir", type=str, required=True, help="Output folder for NPZ files."
    )
    parser.add_argument(
        "--manifest_path", type=str, required=True, help="Output manifest JSON path."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan csv_dir.",
    )
    parser.add_argument(
        "--converter_script",
        type=str,
        default=str(_resolve_default_converter()),
        help="Path to csv_to_npz.py converter script.",
    )
    parser.add_argument(
        "--batch_converter_script",
        type=str,
        default=str(_resolve_default_batch_converter()),
        help="Path to batch_csv_to_npz.py used for one-shot batched conversion.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run converter.",
    )
    parser.add_argument(
        "--input_fps",
        type=float,
        default=60.0,
        help="Input CSV FPS passed to converter.",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=50.0,
        help="Output NPZ FPS passed to converter.",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional frame range passed to converter (1-indexed inclusive).",
    )
    parser.add_argument(
        "--auto_trim_mode",
        type=str,
        choices=("none", "g1_shoulder_roll"),
        default="none",
        help=(
            "Optional per-motion trim heuristic. "
            "`g1_shoulder_roll` trims the alignment pose by finding when both "
            "G1 shoulder-roll joints drop below a threshold."
        ),
    )
    parser.add_argument(
        "--auto_trim_search_frames",
        type=int,
        default=800,
        help="Maximum number of source CSV frames to inspect when auto trim is enabled.",
    )
    parser.add_argument(
        "--auto_trim_shoulder_roll_threshold",
        type=float,
        default=1.0,
        help=(
            "Absolute shoulder-roll threshold in radians used by "
            "`--auto_trim_mode g1_shoulder_roll`."
        ),
    )
    parser.add_argument(
        "--auto_trim_hold_frames",
        type=int,
        default=5,
        help=(
            "Require the auto-trim condition to hold for this many consecutive "
            "frames before accepting the detected start."
        ),
    )
    parser.add_argument(
        "--auto_trim_pre_roll",
        type=int,
        default=5,
        help="Keep this many frames before the detected auto-trim start.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Pass --headless to converter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sim device for converter (e.g. cuda:0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing NPZ files.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip conversion when target NPZ already exists.",
    )
    parser.add_argument(
        "--assume_npz_exists",
        action="store_true",
        default=False,
        help="Do not run converter; require NPZ files to already exist in npz_dir.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lafan1",
        help="Dataset name written to manifest.",
    )
    parser.add_argument(
        "--motion_name_prefix",
        type=str,
        default="",
        help="Optional prefix for generated motion names in manifest.",
    )
    parser.add_argument(
        "--record_videos",
        action="store_true",
        default=False,
        help="Record one MP4 per motion during the batched conversion session.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Output folder for recorded videos (defaults to a sibling 'videos' folder next to npz_dir).",
    )
    parser.add_argument(
        "--overwrite_videos",
        action="store_true",
        default=False,
        help="Overwrite existing videos with the same output path.",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=640,
        help="Per-env video width in pixels when --record_videos is enabled.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=480,
        help="Per-env video height in pixels when --record_videos is enabled.",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).expanduser().resolve()
    npz_dir = Path(args.npz_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    converter_script = Path(args.converter_script).expanduser().resolve()
    batch_converter_script = Path(args.batch_converter_script).expanduser().resolve()

    if not csv_dir.is_dir():
        raise NotADirectoryError(f"csv_dir does not exist: {csv_dir}")
    if not args.assume_npz_exists and not converter_script.is_file():
        raise FileNotFoundError(f"converter_script not found: {converter_script}")
    if not args.assume_npz_exists and not batch_converter_script.is_file():
        raise FileNotFoundError(
            f"batch_converter_script not found: {batch_converter_script}"
        )
    if args.assume_npz_exists and args.record_videos:
        raise ValueError(
            "--record_videos requires conversion, so it cannot be combined with --assume_npz_exists."
        )

    csv_files = _discover_csv_files(csv_dir, recursive=args.recursive)
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found under: {csv_dir}")

    npz_dir.mkdir(parents=True, exist_ok=True)
    video_dir = (
        Path(args.video_dir).expanduser().resolve()
        if args.video_dir is not None
        else _default_video_dir(npz_dir)
    )

    resolved_frame_ranges, trim_details = _resolve_motion_frame_ranges(
        csv_files=csv_files, args=args
    )
    trimmed_motion_count = sum(
        1 for frame_range in resolved_frame_ranges.values() if frame_range is not None
    )

    if args.frame_range is not None or args.auto_trim_mode != "none":
        for csv_file in csv_files:
            detail = trim_details[csv_file]
            source_range = detail.get("source_frame_range")
            if source_range is None:
                print(
                    f"[INFO] Full-range source: {csv_file} | total_frames={detail['total_frames']}"
                )
                continue
            detected_start = detail.get("detected_start_frame")
            print(
                f"[INFO] Source trim for {csv_file} | mode={detail['mode']} "
                f"| source_frame_range={source_range[0]}-{source_range[1]} "
                f"| detected_start={detected_start}"
            )

    batch_jobs: list[dict[str, object]] = []

    for csv_file in csv_files:
        npz_file = _build_npz_path(
            csv_file=csv_file, csv_root=csv_dir, npz_root=npz_dir
        )
        npz_file.parent.mkdir(parents=True, exist_ok=True)
        resolved_frame_range = resolved_frame_ranges[csv_file]

        if args.assume_npz_exists:
            if not npz_file.is_file():
                raise FileNotFoundError(
                    f"assume_npz_exists=True but target missing: {npz_file}. "
                    "Either disable --assume_npz_exists or pre-generate all NPZ files."
                )
        else:
            if npz_file.exists() and not args.overwrite:
                if args.skip_existing:
                    if resolved_frame_range is not None:
                        raise FileExistsError(
                            "A per-motion source trim was requested, but the target NPZ already "
                            f"exists: {npz_file}. Use --overwrite to rebuild trimmed NPZs or "
                            "--assume_npz_exists to leave NPZs untouched and encode the trim in the manifest."
                        )
                    print(f"[INFO] Skipping existing NPZ: {npz_file}")
                else:
                    raise FileExistsError(
                        f"Target NPZ exists: {npz_file}. Use --overwrite or --skip_existing."
                    )
            else:
                job: dict[str, object] = {
                    "input_file": str(csv_file),
                    "output_name": str(npz_file),
                }
                if resolved_frame_range is not None:
                    job["frame_range"] = [
                        int(resolved_frame_range[0]),
                        int(resolved_frame_range[1]),
                    ]
                if args.record_videos:
                    job["video_output"] = str(
                        _build_video_output(
                            csv_file=csv_file, csv_root=csv_dir, video_root=video_dir
                        )
                    )
                batch_jobs.append(job)

    if len(batch_jobs) > 0:
        _run_batch_conversion(
            jobs=batch_jobs,
            batch_converter_script=batch_converter_script,
            python_exe=args.python,
            input_fps=float(args.input_fps),
            output_fps=float(args.output_fps),
            frame_range=None,
            headless=bool(args.headless),
            device=args.device,
            record_videos=bool(args.record_videos),
            overwrite_videos=bool(args.overwrite_videos),
            video_width=int(args.video_width),
            video_height=int(args.video_height),
        )

    manifest_entries: list[dict[str, object]] = []

    for csv_file in csv_files:
        npz_file = _build_npz_path(
            csv_file=csv_file, csv_root=csv_dir, npz_root=npz_dir
        )
        if not npz_file.is_file():
            raise FileNotFoundError(f"Expected converted NPZ not found: {npz_file}")

        relative_no_suffix = npz_file.relative_to(npz_dir).with_suffix("")
        motion_name = _sanitize_motion_name(relative_no_suffix)
        if args.motion_name_prefix:
            motion_name = f"{args.motion_name_prefix}{motion_name}"

        entry: dict[str, object] = {
            "name": motion_name,
            "path": os.path.relpath(npz_file, manifest_path.parent),
            "input_fps": _read_npz_fps(npz_file, fallback_fps=float(args.output_fps)),
        }
        source_frame_range = resolved_frame_ranges[csv_file]
        trim_detail = trim_details[csv_file]
        if args.assume_npz_exists and source_frame_range is not None:
            entry["frame_range"] = [
                int(source_frame_range[0]),
                int(source_frame_range[1]),
            ]
        elif source_frame_range is not None:
            entry["source_frame_range"] = [
                int(source_frame_range[0]),
                int(source_frame_range[1]),
            ]
            entry["trim_mode"] = str(trim_detail["mode"])
        manifest_entries.append(entry)

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset": {
            "trajectories": {
                "lafan1_csv": manifest_entries,
            }
        },
        "metadata": {
            "csv_dir": str(csv_dir),
            "npz_dir": str(npz_dir),
            "num_motions": len(manifest_entries),
            "input_fps": float(args.input_fps),
            "output_fps": float(args.output_fps),
            "converter_script": str(converter_script),
            "assume_npz_exists": bool(args.assume_npz_exists),
            "trimmed_motion_count": int(trimmed_motion_count),
            "source_trim_applied_in_manifest": bool(
                args.assume_npz_exists and trimmed_motion_count > 0
            ),
            "source_trim_baked_into_npz": bool(
                (not args.assume_npz_exists) and trimmed_motion_count > 0
            ),
            "requested_frame_range": (
                [int(args.frame_range[0]), int(args.frame_range[1])]
                if args.frame_range is not None
                else None
            ),
            "auto_trim": (
                {
                    "mode": args.auto_trim_mode,
                    "search_frames": int(args.auto_trim_search_frames),
                    "shoulder_roll_threshold": float(
                        args.auto_trim_shoulder_roll_threshold
                    ),
                    "hold_frames": int(args.auto_trim_hold_frames),
                    "pre_roll": int(args.auto_trim_pre_roll),
                }
                if args.auto_trim_mode != "none"
                else None
            ),
            "record_videos": bool(args.record_videos),
            "video_dir": str(video_dir) if args.record_videos else None,
            "paths_are_relative_to_manifest": True,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[INFO] Wrote manifest: {manifest_path}")
    print(f"[INFO] Motion count: {len(manifest_entries)}")
    print(f"[INFO] Source-trimmed motions: {trimmed_motion_count}")
    if args.record_videos:
        print(f"[INFO] Video root: {video_dir}")

    cfg_snippet = (
        "\n"
        "# Paste into ImitationG1LafanTrackEnvCfg\n"
        f'lafan1_manifest_path = "{manifest_path}"\n'
    )
    print(cfg_snippet)


if __name__ == "__main__":
    main()
