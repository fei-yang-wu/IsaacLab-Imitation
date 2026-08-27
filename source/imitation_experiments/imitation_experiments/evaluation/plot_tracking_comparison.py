"""Plot a low-level tracking comparison curve from a clean CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV with method, frames_b, sr, and mpjpe_l columns.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument(
        "--title",
        default="",
        help="Figure title.",
    )
    return parser


def plot_tracking_comparison(input_path: Path, output_path: Path, title: str) -> None:
    df = pd.read_csv(input_path)
    required_columns = {"method", "frames_b", "sr", "mpjpe_l"}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {', '.join(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_names = {"SONIC scratch": "SONIC"}
    colors = {
        "SONIC scratch": "#3b6fb6",
        "SONIC": "#3b6fb6",
        "DiffSR latent": "#c4533b",
    }

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)

    for method, group in df.groupby("method"):
        group = group.sort_values("frames_b")
        axes[0].plot(
            group["frames_b"],
            group["sr"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=display_names.get(method, method),
            color=colors.get(method),
        )
        axes[1].plot(
            group["frames_b"],
            group["mpjpe_l"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=display_names.get(method, method),
            color=colors.get(method),
        )

    axes[0].set_ylabel("SR")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].set_ylabel("MPJPE-L (mm)")
    axes[1].set_xlabel("Environment Frames (B)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")

    if title:
        fig.suptitle(title, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    plot_tracking_comparison(args.input, args.output, args.title)


if __name__ == "__main__":
    main()
