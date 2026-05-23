#!/usr/bin/env python3

"""Create a renderer-free preview of a Unitree LeRobot episode segment."""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from datasets import load_dataset


def _default_output(repo_id: str, episode_index: int) -> str:
    repo_slug = repo_id.replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(
        Path("logs")
        / "unitree_lerobot_preview"
        / f"{repo_slug}_ep{episode_index}_{stamp}.png"
    )


parser = argparse.ArgumentParser(
    description="Preview Unitree WBT LeRobot low-dimensional episode fields."
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="unitreerobotics/G1_WBT_Brainco_Pickup_Pillow",
    help="Hugging Face dataset repo id.",
)
parser.add_argument("--split", type=str, default="train", help="Dataset split.")
parser.add_argument(
    "--episode_index", type=int, default=0, help="Episode index to preview."
)
parser.add_argument(
    "--state_field",
    type=str,
    default="observation.state.robot_q_current",
    help="36-wide current robot configuration field.",
)
parser.add_argument(
    "--action_field",
    type=str,
    default="action.robot_q_desired",
    help="36-wide desired robot configuration field.",
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=180,
    help="Maximum number of episode frames to stream and preview.",
)
parser.add_argument(
    "--max_scan_rows",
    type=int,
    default=100_000,
    help="Maximum streamed rows to scan while looking for --episode_index.",
)
parser.add_argument("--fps", type=float, default=30.0, help="Episode frame rate.")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output PNG path. Defaults under logs/unitree_lerobot_preview/.",
)
parser.add_argument(
    "--npz_output",
    type=str,
    default=None,
    help="Optional NPZ path for the pulled tensors.",
)
parser.add_argument(
    "--gif_output",
    type=str,
    default=None,
    help="Optional GIF path for an animated data replay.",
)
parser.add_argument(
    "--gif_stride",
    type=int,
    default=2,
    help="Use every Nth frame when writing --gif_output.",
)
parser.add_argument(
    "--gif_fps",
    type=int,
    default=12,
    help="Playback FPS for --gif_output.",
)


def _nested_row_get(row: dict, key: str):
    if key in row:
        return row[key]
    value = row
    for part in key.split("."):
        value = value[part]
    return value


def _row_episode_index(row: dict) -> int:
    return int(torch.as_tensor(_nested_row_get(row, "episode_index")).item())


def _row_frame_index(row: dict) -> int:
    return int(torch.as_tensor(_nested_row_get(row, "frame_index")).item())


def _load_episode_rows(args: argparse.Namespace) -> list[dict]:
    dataset = load_dataset(args.repo_id, split=args.split, streaming=True)
    rows = []
    scanned_rows = 0
    for row in dataset:
        scanned_rows += 1
        episode_index = _row_episode_index(row)
        if episode_index == args.episode_index:
            rows.append(row)
            if len(rows) >= args.max_frames:
                break
        elif rows:
            break
        if scanned_rows >= args.max_scan_rows:
            break

    if len(rows) < 2:
        raise RuntimeError(
            f"Found {len(rows)} rows for episode {args.episode_index} "
            f"after scanning {scanned_rows} rows from {args.repo_id}/{args.split}."
        )
    rows.sort(key=_row_frame_index)
    return rows


def _stack_field(rows: list[dict], field: str) -> torch.Tensor:
    tensor = torch.stack(
        [
            torch.as_tensor(_nested_row_get(row, field), dtype=torch.float32)
            for row in rows
        ],
        dim=0,
    )
    if tensor.ndim != 2 or tensor.shape[1] != 36:
        raise ValueError(f"{field} must have shape [T, 36], got {tuple(tensor.shape)}")
    return tensor


def _plot_preview(
    q_current: torch.Tensor,
    q_desired: torch.Tensor,
    fps: float,
    output_path: Path,
    title: str,
) -> None:
    frames = q_current.shape[0]
    time_s = np.arange(frames, dtype=np.float32) / fps
    root_pos = q_current[:, :3].numpy()
    joint_pos = q_current[:, 7:].numpy()
    joint_delta = (q_desired[:, 7:] - q_current[:, 7:]).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    ax_xy = axes[0, 0]
    ax_xy.plot(root_pos[:, 0], root_pos[:, 1], color="#2f6f8f", linewidth=2.0)
    ax_xy.scatter(root_pos[0, 0], root_pos[0, 1], color="#1f9d55", label="start")
    ax_xy.scatter(root_pos[-1, 0], root_pos[-1, 1], color="#d64545", label="end")
    ax_xy.set_title("Root XY Trajectory")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend(loc="best")

    ax_height = axes[0, 1]
    ax_height.plot(time_s, root_pos[:, 0], label="root x", linewidth=1.5)
    ax_height.plot(time_s, root_pos[:, 1], label="root y", linewidth=1.5)
    ax_height.plot(time_s, root_pos[:, 2], label="root z", linewidth=2.0)
    ax_height.set_title("Root Position Over Time")
    ax_height.set_xlabel("time [s]")
    ax_height.set_ylabel("position [m]")
    ax_height.grid(True, alpha=0.25)
    ax_height.legend(loc="best")

    ax_joint = axes[1, 0]
    im_joint = ax_joint.imshow(
        joint_pos.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_s[0], time_s[-1], 0, joint_pos.shape[1] - 1],
        cmap="viridis",
    )
    ax_joint.set_title("Current Joint Positions")
    ax_joint.set_xlabel("time [s]")
    ax_joint.set_ylabel("G1 SDK joint index")
    fig.colorbar(im_joint, ax=ax_joint, label="rad")

    ax_delta = axes[1, 1]
    delta_abs = np.max(np.abs(joint_delta))
    im_delta = ax_delta.imshow(
        joint_delta.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[time_s[0], time_s[-1], 0, joint_delta.shape[1] - 1],
        cmap="coolwarm",
        vmin=-delta_abs,
        vmax=delta_abs,
    )
    ax_delta.set_title("Desired - Current Joint Position")
    ax_delta.set_xlabel("time [s]")
    ax_delta.set_ylabel("G1 SDK joint index")
    fig.colorbar(im_delta, ax=ax_delta, label="rad")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_gif(
    q_current: torch.Tensor,
    q_desired: torch.Tensor,
    fps: float,
    output_path: Path,
    title: str,
    stride: int,
    gif_fps: int,
) -> None:
    frames = q_current.shape[0]
    time_s = np.arange(frames, dtype=np.float32) / fps
    root_pos = q_current[:, :3].numpy()
    joint_pos = q_current[:, 7:].numpy()
    joint_delta = (q_desired[:, 7:] - q_current[:, 7:]).numpy()
    frame_ids = list(range(0, frames, stride))
    if frame_ids[-1] != frames - 1:
        frame_ids.append(frames - 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle(title, fontsize=11)

    ax_xy = axes[0]
    ax_xy.plot(root_pos[:, 0], root_pos[:, 1], color="#d0d7de", linewidth=1.0)
    (path_line,) = ax_xy.plot([], [], color="#2f6f8f", linewidth=2.0)
    marker = ax_xy.scatter([], [], color="#d64545", s=35)
    ax_xy.set_title("Root XY Replay")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    pad = 0.01
    ax_xy.set_xlim(root_pos[:, 0].min() - pad, root_pos[:, 0].max() + pad)
    ax_xy.set_ylim(root_pos[:, 1].min() - pad, root_pos[:, 1].max() + pad)

    ax_joint = axes[1]
    joint_indexes = np.arange(joint_pos.shape[1])
    (joint_line,) = ax_joint.plot([], [], color="#2f6f8f", label="current")
    (delta_line,) = ax_joint.plot([], [], color="#d64545", label="desired-current")
    ax_joint.set_title("Joint Snapshot")
    ax_joint.set_xlabel("G1 SDK joint index")
    ax_joint.set_ylabel("rad")
    ax_joint.set_xlim(0, joint_pos.shape[1] - 1)
    y_min = min(joint_pos.min(), joint_delta.min()) - 0.1
    y_max = max(joint_pos.max(), joint_delta.max()) + 0.1
    ax_joint.set_ylim(y_min, y_max)
    ax_joint.grid(True, alpha=0.25)
    ax_joint.legend(loc="upper right")
    text = ax_joint.text(0.02, 0.95, "", transform=ax_joint.transAxes)

    def update(frame_index: int):
        path_line.set_data(
            root_pos[: frame_index + 1, 0], root_pos[: frame_index + 1, 1]
        )
        marker.set_offsets(root_pos[frame_index : frame_index + 1, :2])
        joint_line.set_data(joint_indexes, joint_pos[frame_index])
        delta_line.set_data(joint_indexes, joint_delta[frame_index])
        text.set_text(f"frame {frame_index} | t={time_s[frame_index]:.2f}s")
        return path_line, marker, joint_line, delta_line, text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=frame_ids, interval=1000 / gif_fps)
    animation.save(output_path, writer=PillowWriter(fps=gif_fps))
    plt.close(fig)


def main() -> None:
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.max_frames < 2:
        raise ValueError(f"--max_frames must be at least 2, got {args.max_frames}")
    if args.max_scan_rows < args.max_frames:
        raise ValueError(
            f"--max_scan_rows must be >= --max_frames, got {args.max_scan_rows}"
        )
    if args.gif_stride < 1:
        raise ValueError(f"--gif_stride must be at least 1, got {args.gif_stride}")
    if args.gif_fps < 1:
        raise ValueError(f"--gif_fps must be at least 1, got {args.gif_fps}")

    output = Path(args.output or _default_output(args.repo_id, args.episode_index))
    output = output.expanduser().resolve()
    npz_output = (
        Path(args.npz_output).expanduser().resolve()
        if args.npz_output is not None
        else output.with_suffix(".npz")
    )
    gif_output = (
        Path(args.gif_output).expanduser().resolve()
        if args.gif_output is not None
        else None
    )

    rows = _load_episode_rows(args)
    q_current = _stack_field(rows, args.state_field)
    q_desired = _stack_field(rows, args.action_field)

    title = (
        f"{args.repo_id} | {args.split} | episode {args.episode_index} | "
        f"{q_current.shape[0]} frames"
    )
    _plot_preview(q_current, q_desired, args.fps, output, title)
    if gif_output is not None:
        _plot_gif(
            q_current,
            q_desired,
            args.fps,
            gif_output,
            title,
            args.gif_stride,
            args.gif_fps,
        )

    npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_output,
        q_current=q_current.numpy(),
        q_desired=q_desired.numpy(),
        fps=np.asarray([args.fps], dtype=np.float32),
        episode_index=np.asarray([args.episode_index], dtype=np.int64),
    )

    print("[INFO]: Preview image saved to", output)
    if gif_output is not None:
        print("[INFO]: Preview GIF saved to", gif_output)
    print("[INFO]: Preview tensors saved to", npz_output)
    print("[INFO]: q_current shape:", tuple(q_current.shape))
    print("[INFO]: q_desired shape:", tuple(q_desired.shape))
    print("[INFO]: root xyz first:", q_current[0, :3].tolist())
    print("[INFO]: root xyz last:", q_current[-1, :3].tolist())


if __name__ == "__main__":
    main()
