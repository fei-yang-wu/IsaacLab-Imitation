# Copyright (c) 2026, IsaacLab-Imitation Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lay a clip out frame by frame so a human can pick the poses for a figure.

The stroboscopic figure draws a handful of frames out of several hundred.
Sampling them evenly along the walked path is a reasonable default and a poor
editor: it cannot dwell on the part of a motion that carries the meaning -- the
reach down at floor level, the turn away from the door. This writes numbered
thumbnails of the whole clip so the frames can be chosen by eye, and the
numbers feed straight back as `--sequence_pose_steps`.

Reads the MP4 a render already wrote, so choosing frames costs no rendering.

Example:

.. code-block:: bash

    pixi run -e isaaclab python scripts/viz/frame_contact_sheet.py \\
        --video logs/.../rank-000013-inside_door....mp4 \\
        --every 5 --columns 10 \\
        --output outputs/paper/figures/contact/door.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument(
        "--every", type=int, default=5, help="Thumbnail every Nth frame."
    )
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument(
        "--thumb_width", type=int, default=320, help="Thumbnail width in pixels."
    )
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        default=(0.28, 0.10, 0.78, 0.95),
        help=(
            "Fractional left/top/right/bottom crop applied to every frame "
            "before scaling, so the subject fills the thumbnail instead of the "
            "backdrop around it."
        ),
    )
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _frame_count(video: Path) -> int:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(probe.stdout.strip())


def _read_frames(video: Path, every: int) -> tuple[list[int], list[np.ndarray]]:
    """Decode the clip once and keep every Nth frame.

    One pass with a select filter beats one ffmpeg call per frame by a wide
    margin on a 500-frame clip, and the frame numbering stays exact because the
    filter counts source frames.
    """
    from PIL import Image

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = (int(value) for value in probe.stdout.strip().split(","))
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=not(mod(n\\,{int(every)}))",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frame_bytes = width * height * 3
    kept = len(raw) // frame_bytes
    frames = [
        np.frombuffer(raw[i * frame_bytes : (i + 1) * frame_bytes], dtype=np.uint8)
        .reshape(height, width, 3)
        .copy()
        for i in range(kept)
    ]
    indices = [i * int(every) for i in range(kept)]
    _ = Image  # imported here so a missing Pillow fails before the decode
    return indices, frames


def main() -> None:
    args = _parse_args()
    from PIL import Image, ImageDraw

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    total = _frame_count(video)
    indices, frames = _read_frames(video, int(args.every))
    if not frames:
        raise SystemExit("decoded no frames")

    height, width, _ = frames[0].shape
    left, top, right, bottom = args.crop
    box = (
        int(left * width),
        int(top * height),
        int(right * width),
        int(bottom * height),
    )
    thumb_w = int(args.thumb_width)
    thumb_h = max(1, round(thumb_w * (box[3] - box[1]) / (box[2] - box[0])))
    label_h = 22
    columns = int(args.columns)
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumb_w, rows * (thumb_h + label_h)), (250, 250, 250)
    )
    draw = ImageDraw.Draw(sheet)
    for position, (index, frame) in enumerate(zip(indices, frames)):
        column, row = position % columns, position // columns
        x, y = column * thumb_w, row * (thumb_h + label_h)
        thumb = Image.fromarray(frame).crop(box).resize((thumb_w, thumb_h))
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 6, y + 5), str(index), fill=(20, 20, 20))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(
        f"[SHEET] {video.name}: {total} frames, {len(frames)} thumbnails "
        f"every {args.every} -> {output}"
    )


if __name__ == "__main__":
    main()
