#!/usr/bin/env python3
"""Render a G1 reference NPZ as a 3D skeleton animation MP4 — no Isaac Sim.

This visualizes ``body_pos_w`` (the actual per-body world positions stored in the
NPZ, which are the training signal) as an animated stick figure. Because it reads
the data directly, it is the most faithful quality check: what you see is exactly
what a tracking policy would be asked to match. It needs no GPU and no Isaac Sim
boot, so it never contends with a running conversion.

Run in the isaaclab Pixi env (it has matplotlib + imageio):

    pixi run -e isaaclab python scripts/render_npz_skeleton.py \
        --npz ~/Storage/bones_seed_full/quality_videos/npz/Loop_Forward_Walk_001__A018.npz \
        --out ~/Storage/bones_seed_full/quality_videos/Loop_Forward_Walk_skeleton.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# G1 kinematic chains expressed as ordered body-name paths. Consecutive names in
# each chain are connected by a bone. Resolved against the NPZ's body_names.
G1_CHAINS: tuple[tuple[str, ...], ...] = (
    (
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
    ),
    (
        "pelvis",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ),
    ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"),
    (
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    ),
    (
        "torso_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ),
)


def _resolve_edges(body_names: list[str]) -> list[tuple[int, int]]:
    index = {name: i for i, name in enumerate(body_names)}
    edges: list[tuple[int, int]] = []
    for chain in G1_CHAINS:
        present = [name for name in chain if name in index]
        for a, b in zip(present[:-1], present[1:]):
            edges.append((index[a], index[b]))
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS.")
    parser.add_argument(
        "--stride", type=int, default=1, help="Render every Nth frame for speed."
    )
    parser.add_argument("--elev", type=float, default=12.0)
    parser.add_argument("--azim", type=float, default=-70.0)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument(
        "--follow",
        action="store_true",
        default=False,
        help="Follow-cam: re-center X/Y on the pelvis each frame so the figure "
        "fills the view (best for spotting foot-skate / jitter / pose issues). "
        "Global translation is then not visible; use the default fixed view for that.",
    )
    parser.add_argument(
        "--follow_span",
        type=float,
        default=1.1,
        help="Half-extent (m) of the follow-cam window around the pelvis.",
    )
    args = parser.parse_args()

    npz_path = args.npz.expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as data:
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)  # (T, B, 3)
        body_names = [str(x) for x in data["body_names"].tolist()]
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
    out_fps = float(args.fps) if args.fps else fps

    edges = _resolve_edges(body_names)
    frames_idx = range(0, body_pos.shape[0], max(1, args.stride))

    # Fixed world-frame limits over the whole clip so translation is visible,
    # with equal aspect so proportions are honest.
    lo = body_pos.reshape(-1, 3).min(axis=0)
    hi = body_pos.reshape(-1, 3).max(axis=0)
    center = (lo + hi) / 2.0
    span = float((hi - lo).max()) * 0.55 + 0.2

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path), fps=max(1, int(round(out_fps))), codec="libx264",
        macro_block_size=None,
    )
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    try:
        for t in frames_idx:
            ax.cla()
            pts = body_pos[t]
            if args.follow:
                cx, cy = float(pts[0, 0]), float(pts[0, 1])
                cur_span = float(args.follow_span)
            else:
                cx, cy = float(center[0]), float(center[1])
                cur_span = span
            for a, b in edges:
                ax.plot(
                    [pts[a, 0], pts[b, 0]],
                    [pts[a, 1], pts[b, 1]],
                    [pts[a, 2], pts[b, 2]],
                    color="tab:blue",
                    linewidth=2.0,
                )
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, color="tab:red")
            # Ground plane reference at z=0.
            ax.plot(
                [cx - cur_span, cx + cur_span],
                [cy, cy],
                [0, 0],
                color="0.8",
                linewidth=0.8,
            )
            ax.set_xlim(cx - cur_span, cx + cur_span)
            ax.set_ylim(cy - cur_span, cy + cur_span)
            ax.set_zlim(0, max(2 * cur_span, 2.0))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_title(f"{npz_path.stem}  frame {t}/{body_pos.shape[0]}", fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[..., :3].copy())
    finally:
        writer.close()
        plt.close(fig)
    print(f"[skeleton] wrote {out_path} ({len(list(frames_idx))} frames @ {out_fps:.0f} fps)")
    print(str(out_path))


if __name__ == "__main__":
    main()
