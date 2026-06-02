#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the Hugging Face LAFAN1 retargeting dataset into ./data.

By default this script downloads the Unitree G1 subset from:
https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

It can also optionally convert the downloaded CSV files into NPZ motions and
generate a manifest JSON by delegating to ``scripts/prepare_lafan1_from_csv.py``.

Examples:
    pixi run python scripts/setup_lafan1_dataset.py

    pixi run -e isaaclab python scripts/setup_lafan1_dataset.py \
        --prepare-npz --headless

    pixi run -e isaaclab python scripts/setup_lafan1_dataset.py \
        --prepare-npz --headless --auto_trim_mode g1_shoulder_roll
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - import guard for misconfigured envs
    raise ImportError(
        "huggingface_hub is required. Run this script through the Pixi "
        "default or isaaclab environment."
    ) from exc


DATASET_REPO_ID = "lvhaidong/LAFAN1_Retargeting_Dataset"
DATASET_REPO_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
ROOT_ALLOW_PATTERNS = ["README.md", "LICENSE", "dataset_infos.json", "meta_data/*"]
ROBOT_CHOICES = ("g1", "h1", "h1_2", "all")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    return _repo_root() / "data" / "lafan1"


def _prepare_script_path() -> Path:
    return Path(__file__).resolve().with_name("prepare_lafan1_from_csv.py")


def _resolve_converter_script(explicit_path: str | None) -> Path | None:
    if explicit_path is not None:
        converter_script = Path(explicit_path).expanduser().resolve()
        if not converter_script.is_file():
            raise FileNotFoundError(f"converter_script not found: {converter_script}")
        return converter_script

    repo_root = _repo_root()
    candidate_paths = [
        repo_root / "scripts" / "csv_to_npz.py",
    ]
    for path in candidate_paths:
        if path.is_file():
            return path.resolve()
    return None


def _robot_subdirs(robot_type: str) -> list[str]:
    if robot_type == "all":
        return ["g1", "h1", "h1_2"]
    return [robot_type]


def _download_patterns(
    robot_type: str, include_robot_description: bool, include_visualizer: bool
) -> list[str]:
    patterns = list(ROOT_ALLOW_PATTERNS)
    patterns.extend(f"{robot_subdir}/*" for robot_subdir in _robot_subdirs(robot_type))
    if include_robot_description:
        patterns.append("robot_description/*")
    if include_visualizer:
        patterns.append("rerun_visualize.py")
    return patterns


def _csv_dir_for_robot(raw_root: Path, robot_type: str) -> Path:
    if robot_type == "all":
        return raw_root
    return raw_root / robot_type


def _npz_dir_for_robot(data_root: Path, robot_type: str) -> Path:
    return data_root / "npz" / robot_type


def _manifest_path_for_robot(data_root: Path, robot_type: str) -> Path:
    return data_root / "manifests" / f"{robot_type}_lafan1_manifest.json"


def _count_csv_files(csv_dir: Path, recursive: bool) -> int:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sum(1 for path in csv_dir.glob(pattern) if path.is_file())


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_prepare_pipeline(
    *,
    args: argparse.Namespace,
    csv_dir: Path,
    npz_dir: Path,
    manifest_path: Path,
    converter_script: Path,
) -> None:
    prepare_script = _prepare_script_path()
    if not prepare_script.is_file():
        raise FileNotFoundError(f"prepare script not found: {prepare_script}")

    cmd = [
        args.python,
        str(prepare_script),
        "--csv_dir",
        str(csv_dir),
        "--npz_dir",
        str(npz_dir),
        "--manifest_path",
        str(manifest_path),
        "--recursive",
        "--converter_script",
        str(converter_script),
        "--python",
        args.python,
        "--input_fps",
        str(args.input_fps),
        "--output_fps",
        str(args.output_fps),
        "--dataset_name",
        args.dataset_name,
    ]

    if args.motion_name_prefix:
        cmd.extend(["--motion_name_prefix", args.motion_name_prefix])
    if args.frame_range is not None:
        cmd.extend(
            ["--frame_range", str(args.frame_range[0]), str(args.frame_range[1])]
        )
    if args.auto_trim_mode != "none":
        cmd.extend(["--auto_trim_mode", args.auto_trim_mode])
        cmd.extend(["--auto_trim_search_frames", str(args.auto_trim_search_frames)])
        cmd.extend(
            [
                "--auto_trim_shoulder_roll_threshold",
                str(args.auto_trim_shoulder_roll_threshold),
            ]
        )
        cmd.extend(["--auto_trim_hold_frames", str(args.auto_trim_hold_frames)])
        cmd.extend(["--auto_trim_pre_roll", str(args.auto_trim_pre_roll)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.headless:
        cmd.append("--headless")
    if args.overwrite_npz:
        cmd.append("--overwrite")
    else:
        cmd.append("--skip_existing")

    print(f"[INFO] Preparing NPZ motions and manifest from: {csv_dir}")
    print(f"[CMD]  {_format_command(cmd)}")
    subprocess.run(cmd, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the Hugging Face LAFAN1 retargeting dataset into ./data and optionally prepare NPZ files."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(_default_data_root()),
        help="Root folder for raw CSV files, converted NPZ files, and manifest output.",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        choices=ROBOT_CHOICES,
        default="g1",
        help="Robot subset to download from the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset repo revision to download from Hugging Face.",
    )
    parser.add_argument(
        "--prepare-npz",
        action="store_true",
        default=False,
        help="Convert downloaded CSV files to NPZ and write a manifest JSON.",
    )
    parser.add_argument(
        "--converter_script",
        type=str,
        default=None,
        help="Optional path to the CSV-to-NPZ converter script.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for the downstream CSV-to-NPZ conversion step.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lafan1",
        help="Dataset name written into the generated manifest when --prepare-npz is used.",
    )
    parser.add_argument(
        "--motion_name_prefix",
        type=str,
        default="",
        help="Optional prefix for generated motion names in the manifest.",
    )
    parser.add_argument(
        "--input_fps",
        type=float,
        default=30.0,
        help="Input CSV FPS for the retargeted Hugging Face dataset during conversion.",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=50.0,
        help="Output NPZ FPS for the conversion step.",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional frame range passed through to prepare_lafan1_from_csv.py.",
    )
    parser.add_argument(
        "--auto_trim_mode",
        type=str,
        choices=("none", "g1_shoulder_roll"),
        default="none",
        help=(
            "Optional per-motion trim heuristic forwarded to "
            "prepare_lafan1_from_csv.py."
        ),
    )
    parser.add_argument(
        "--auto_trim_search_frames",
        type=int,
        default=800,
        help="Maximum number of CSV frames inspected by the auto-trim heuristic.",
    )
    parser.add_argument(
        "--auto_trim_shoulder_roll_threshold",
        type=float,
        default=1.0,
        help="Absolute shoulder-roll threshold for --auto_trim_mode g1_shoulder_roll.",
    )
    parser.add_argument(
        "--auto_trim_hold_frames",
        type=int,
        default=5,
        help="Require the auto-trim condition to hold for this many frames.",
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
        help="Pass --headless to the downstream CSV-to-NPZ converter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sim device for conversion, for example cuda:0.",
    )
    parser.add_argument(
        "--include-robot-description",
        action="store_true",
        default=False,
        help="Also download robot_description assets from the Hugging Face repo.",
    )
    parser.add_argument(
        "--include-visualizer",
        action="store_true",
        default=False,
        help="Also download the dataset rerun_visualize.py helper from the Hugging Face repo.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force Hugging Face to re-download files even if the local metadata cache is up to date.",
    )
    parser.add_argument(
        "--overwrite-npz",
        action="store_true",
        default=False,
        help="Overwrite existing NPZ files during the optional conversion step.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    raw_root = data_root / "raw"
    csv_dir = _csv_dir_for_robot(raw_root=raw_root, robot_type=args.robot_type)
    npz_dir = _npz_dir_for_robot(data_root=data_root, robot_type=args.robot_type)
    manifest_path = _manifest_path_for_robot(
        data_root=data_root, robot_type=args.robot_type
    )

    data_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    allow_patterns = _download_patterns(
        robot_type=args.robot_type,
        include_robot_description=bool(args.include_robot_description),
        include_visualizer=bool(args.include_visualizer),
    )

    print(f"[INFO] Downloading LAFAN1 dataset from: {DATASET_REPO_URL}")
    print(f"[INFO] Revision: {args.revision}")
    print(f"[INFO] Local raw dataset root: {raw_root}")
    print(f"[INFO] Allow patterns: {allow_patterns}")

    snapshot_path = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(raw_root),
        allow_patterns=allow_patterns,
        force_download=bool(args.force_download),
    )

    if not csv_dir.is_dir():
        raise FileNotFoundError(
            f"Expected downloaded CSV directory does not exist: {csv_dir}. "
            f"Downloaded snapshot root: {snapshot_path}"
        )

    csv_count = _count_csv_files(
        csv_dir=csv_dir, recursive=bool(args.robot_type == "all")
    )
    if csv_count == 0:
        raise RuntimeError(f"No CSV files found after download under: {csv_dir}")

    print(f"[INFO] Download completed into: {snapshot_path}")
    print(f"[INFO] CSV files ready under {csv_dir}: {csv_count}")

    if not args.prepare_npz:
        preview_cmd = [
            args.python,
            str(_prepare_script_path()),
            "--csv_dir",
            str(csv_dir),
            "--npz_dir",
            str(npz_dir),
            "--manifest_path",
            str(manifest_path),
            "--recursive",
            "--input_fps",
            str(args.input_fps),
            "--output_fps",
            str(args.output_fps),
        ]
        if args.frame_range is not None:
            preview_cmd.extend(
                ["--frame_range", str(args.frame_range[0]), str(args.frame_range[1])]
            )
        if args.auto_trim_mode != "none":
            preview_cmd.extend(["--auto_trim_mode", args.auto_trim_mode])
            preview_cmd.extend(
                ["--auto_trim_search_frames", str(args.auto_trim_search_frames)]
            )
            preview_cmd.extend(
                [
                    "--auto_trim_shoulder_roll_threshold",
                    str(args.auto_trim_shoulder_roll_threshold),
                ]
            )
            preview_cmd.extend(
                ["--auto_trim_hold_frames", str(args.auto_trim_hold_frames)]
            )
            preview_cmd.extend(["--auto_trim_pre_roll", str(args.auto_trim_pre_roll)])
        print("[INFO] Raw download is ready.")
        print("[INFO] To generate NPZ files and a manifest later, run:")
        print(f"[CMD]  {_format_command(preview_cmd)}")
        return

    converter_script = _resolve_converter_script(args.converter_script)
    if converter_script is None:
        raise FileNotFoundError(
            "Could not locate a CSV-to-NPZ converter script. Pass --converter_script explicitly."
        )

    npz_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    _run_prepare_pipeline(
        args=args,
        csv_dir=csv_dir,
        npz_dir=npz_dir,
        manifest_path=manifest_path,
        converter_script=converter_script,
    )

    print("[INFO] Dataset setup completed.")
    print(f"[INFO] Raw CSV root: {csv_dir}")
    print(f"[INFO] NPZ root: {npz_dir}")
    print(f"[INFO] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
