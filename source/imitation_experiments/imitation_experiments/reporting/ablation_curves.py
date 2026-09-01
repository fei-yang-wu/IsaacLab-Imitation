"""Convergence figures for an ablation campaign, one figure per paper section.

Each figure has three panels -- success rate, MPJPE-L, MPJPE-G -- against
environment frames, with one line per arm of that section. The section mapping
comes from the campaign file, so a figure cannot disagree with the table beside
it.

Every point is a scored evaluation row. An arm with fewer than two points is
dropped from the plot and named in the returned report, because a single dot is
not a convergence curve and a silent omission reads as "this arm has no data"
when it may simply be early.

    python -m imitation_experiments.reporting.ablation_curves \
        --campaign experiments/campaigns/2026-08-30-latent-star-v2/campaign.yaml \
        --eval-dir logs/latent_star_v2_curves \
        --out-dir logs/report/star_v2_curves
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imitation_experiments.reporting.ablation_sections import (
    SECTION_TITLES,
    _metric_mean,
    load_campaign_arms,
)

_EVAL_RE = re.compile(
    r"^(?P<arm>.+)_seed(?P<seed>\d+)_(?P<row>[a-z0-9]+)_f(?P<frames>\d+)\.json$"
)

PANELS: tuple[tuple[str, str, bool], ...] = (
    ("success_rate", "Success rate", False),
    ("mpjpe_local_mm", "MPJPE-L (mm)", True),
    ("mpjpe_global_mm", "MPJPE-G (mm)", True),
)
MIN_POINTS = 2
# The screen budget the ablation tables report. Curves stop here so a figure
# and the table beside it describe the same runs at the same budget; arms train
# at different speeds, so an uncut figure would show some arms to 5B and others
# to 1B and invite a comparison the table does not make.
SCREEN_FRAMES = 2000486400


@dataclass
class Curve:
    arm: str
    frames: list[int]
    values: dict[str, list[float | None]]


def load_curves(
    eval_dir: Path,
    row: str = "clean",
    seed: int = 0,
    max_frames: int | None = SCREEN_FRAMES,
) -> dict[str, Curve]:
    """Every scored row for every arm, ordered by frame count and cut at
    `max_frames`."""
    per_arm: dict[str, dict[int, dict[str, float | None]]] = {}
    if not eval_dir.is_dir():
        return {}
    for path in sorted(eval_dir.glob(f"*_{row}_f*.json")):
        match = _EVAL_RE.match(path.name)
        if not match or int(match.group("seed")) != seed:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        frames = int(match.group("frames"))
        if max_frames is not None and frames > max_frames:
            continue
        aggregate = payload.get("aggregate") or {}
        successful = payload.get("successful_metrics") or {}
        sr = aggregate.get("tracking_success_rate")
        per_arm.setdefault(match.group("arm"), {})[frames] = {
            "success_rate": float(sr) if isinstance(sr, (int, float)) else None,
            "mpjpe_local_mm": _metric_mean(successful.get("tracking_mpjpe_mm")),
            "mpjpe_global_mm": _metric_mean(successful.get("tracking_mpjpe_g_mm")),
        }
    curves: dict[str, Curve] = {}
    for arm, by_frame in per_arm.items():
        frames = sorted(by_frame)
        curves[arm] = Curve(
            arm=arm,
            frames=frames,
            values={key: [by_frame[f][key] for f in frames] for key, _, _ in PANELS},
        )
    return curves


def section_arms(arms: Mapping[str, Mapping[str, Any]]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for arm, spec in arms.items():
        number = int(spec.get("section", 0))
        if number:
            grouped.setdefault(number, []).append(arm)
    for names in grouped.values():
        names.sort(key=lambda a: (a != "hub", a))
    return grouped


def plot_section(
    number: int,
    arms: list[str],
    curves: Mapping[str, Curve],
    out_path: Path,
    labels: Mapping[str, str] | None = None,
) -> tuple[bool, list[str]]:
    """Write one figure. Returns (wrote_anything, arms skipped for lack of data)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [
        a for a in arms if len(curves.get(a, Curve(a, [], {})).frames) >= MIN_POINTS
    ]
    skipped = [a for a in arms if a not in usable]
    if not usable:
        return False, skipped

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 3.8))
    for axis, (key, ylabel, lower_better) in zip(axes, PANELS, strict=True):
        for arm in usable:
            curve = curves[arm]
            xs = [
                f / 1e9
                for f, v in zip(curve.frames, curve.values[key])
                if v is not None
            ]
            ys = [v for v in curve.values[key] if v is not None]
            if len(xs) < MIN_POINTS:
                continue
            axis.plot(
                xs,
                ys,
                marker="o",
                markersize=2.5,
                linewidth=2.2 if arm == "hub" else 1.1,
                color="black" if arm == "hub" else None,
                zorder=3 if arm == "hub" else 2,
                label=(labels or {}).get(arm, arm),
            )
        axis.set_xlabel("environment frames (B)")
        axis.set_ylabel(ylabel + (" $\\downarrow$" if lower_better else " $\\uparrow$"))
        axis.grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle(f"Section {number}: {SECTION_TITLES.get(number, '')}", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--row", default="clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=SCREEN_FRAMES,
        help="cut every curve here so it matches the table budget; 0 disables",
    )
    args = parser.parse_args(argv)

    curves = load_curves(
        args.eval_dir,
        row=args.row,
        seed=args.seed,
        max_frames=args.max_frames or None,
    )
    if not curves:
        parser.error(f"no scored rows under {args.eval_dir}")
    arms_spec = load_campaign_arms(args.campaign)
    labels = {a: str(v.get("paper_label", a)) for a, v in arms_spec.items()}
    grouped = section_arms(arms_spec)
    wrote = 0
    for number in sorted(grouped):
        out = args.out_dir / f"section{number}_convergence.png"
        ok, skipped = plot_section(number, grouped[number], curves, out, labels)
        if ok:
            wrote += 1
            print(f"wrote {out}")
        else:
            print(f"[skip] section {number}: no arm has {MIN_POINTS}+ points")
        if skipped:
            print(f"  too few points to plot: {', '.join(sorted(skipped))}")
    if not wrote:
        parser.error("no figure had enough data to draw")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
