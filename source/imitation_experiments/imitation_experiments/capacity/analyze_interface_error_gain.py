#!/usr/bin/env python3
"""Interface error-gain, threshold sensitivity, and capacity accounting.

Why this exists
---------------
The capacity grid answers "how well does each interface do", but the paper's
claim is about *why*: that a compressed latent command tolerates planner error
better than a raw explicit packet. That is a statement about the **slope**
relating command error to tracking outcome, not about any single cell.

Three read-outs, all computed from artifacts already on disk:

    error-gain   tracking outcome vs measured closed-loop command error
                 (`planner_target_rmse`). Each interface contributes one point
                 per (size, seed); the fitted slope is its error gain.

    thresholds   the failure definition (root height > 0.25 m OR root ori >
                 1.0 rad) is ours to choose, so a reviewer will challenge it.
                 Sweep it and show the interface ordering does not move.

    accounting   planner parameters vs *total interface* parameters (planner +
                 skill encoder), and command bandwidth in values/second. Both
                 pre-empt "you undercounted latent" and "latent just sends
                 less".

Deliberately NOT used: `fall_rate`, `survival_steps`, `tracking_failure_rate`.
Those are structurally pinned to constants under the full-horizon protocol --
`base_too_low` is unregistered, so `termination_hits.get(..., zeros)` returns
all-False and a missing detector is indistinguishable from zero falls. The
per-step `tracking_failure` indicator is the one fall-sensitive measure that is
actually computed, so it is what this script uses.

Usage
-----
    pixi run python .../analyze_interface_error_gain.py \\
        --study_root logs/interface_baselines/lafan1_interface_capacity
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

# Packet width per published chunk, and the encoder parameters that the
# reported "planner parameters" figure excludes. The latent encoder's input
# width is 670 -- byte-identical to the full-body packet -- so the latent
# interface is a learned compression of exactly what full_body streams raw.
INTERFACE_FACTS: dict[str, dict[str, Any]] = {
    "latent_skill": {"packet": 258, "encoder_params": 1_609_984},
    "root_points5": {"packet": 240, "encoder_params": 0},
    "root_qpos": {"packet": 380, "encoder_params": 0},
    "full_body_trajectory": {"packet": 670, "encoder_params": 0},
}
ORDER = ["latent_skill", "root_points5", "root_qpos", "full_body_trajectory"]
SIZES = ["tiny", "small", "medium", "large"]
# 10-frame packet consumed one slot per control step at 50 Hz -> 5 Hz publishes.
PUBLISH_HZ = 5.0


def _cells(study_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        study_root.glob("scaling/seed*/*/*/eval_pretrained*/summary.json")
    ):
        interface = path.parent.parent.name
        size = path.parent.parent.parent.name
        seed = path.parent.parent.parent.parent.name
        if interface not in INTERFACE_FACTS or size not in SIZES:
            continue
        summary = json.loads(path.read_text())
        metrics = summary["metrics"]
        config_path = path.parent.parent / "planner_pretrain" / "config.json"
        params = None
        if config_path.is_file():
            params = int(json.loads(config_path.read_text())["parameter_count"])
        rows.append(
            {
                "interface": interface,
                "size": size,
                "seed": seed,
                "command_error": metrics.get("planner_target_rmse", {}).get("mean"),
                "fall_fraction": metrics["tracking_failure"]["mean"],
                "mpjpe_mm": metrics["tracking_mpjpe_mm"]["mean"],
                "root_height_err": metrics["root_height_error_m"]["mean"],
                "root_ori_err": metrics["root_ori_error_rad"]["mean"],
                "planner_params": params,
            }
        )
    return rows


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares slope, intercept and Pearson r. Returns nan on degenerate input."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan"), float("nan"), float("nan")
    slope = sxy / sxx
    return slope, my - slope * mx, sxy / math.sqrt(sxx * syy)


def report_error_gain(rows: list[dict[str, Any]]) -> str:
    """B1: slope of tracking outcome against measured command error."""
    out = [
        "## Error gain — tracking outcome vs measured command error",
        "",
        "One point per (size, seed). `slope` is the interface's **error gain**:",
        "how much tracking failure each unit of normalized command error buys.",
        "",
        "| interface | pkt | n | cmd err range | fall-frac range | slope | r |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for interface in ORDER:
        pts = [
            r
            for r in rows
            if r["interface"] == interface and r["command_error"] is not None
        ]
        if not pts:
            continue
        xs = [float(r["command_error"]) for r in pts]
        ys = [float(r["fall_fraction"]) for r in pts]
        slope, _, r = _linfit(xs, ys)
        out.append(
            f"| `{interface}` | {INTERFACE_FACTS[interface]['packet']} | {len(pts)} | "
            f"{min(xs):.3f}–{max(xs):.3f} | {min(ys):.1%}–{max(ys):.1%} | "
            f"{slope:+.3f} | {r:+.2f} |"
        )
    out += [
        "",
        "Caveat that must be stated, not hidden: command-error magnitude co-varies",
        "with planner capacity, so the slope alone cannot separate *the interface",
        "amplifies error* from *capacity has other effects*. It also is not",
        "commensurable across interfaces — 0.1 std in keypoint space is not",
        "physically 0.1 std in joint space, and for `root_points5` the command does",
        "not determine a pose at all, so command error in pose units is undefined.",
        "Compare slopes within an interface; across interfaces, qualitatively only.",
        "",
    ]
    return "\n".join(out)


def report_threshold_sweep(rows: list[dict[str, Any]]) -> str:
    """B3: does the interface ordering survive a different failure definition?

    Recomputed from the per-episode mean root height/orientation error, which is
    an approximation of the true per-step indicator (that would need per-step
    retention). Directional only -- it answers "does the ordering move", not
    "what is the exact rate".
    """
    out = [
        "## Threshold sensitivity",
        "",
        "The failure definition (root height > 0.25 m OR ori > 1.0 rad) is ours to",
        "choose. Re-deriving an approximate ordering under other thresholds, from",
        "each cell's mean root height and orientation error at `large`:",
        "",
        "| height thr | ori thr | " + " | ".join(f"`{i}`" for i in ORDER) + " |",
        "| --- | --- | " + " | ".join("---" for _ in ORDER) + " |",
    ]
    for h_thr, o_thr in ((0.15, 0.6), (0.25, 1.0), (0.40, 1.5), (0.60, 2.0)):
        cells = []
        for interface in ORDER:
            pts = [
                r for r in rows if r["interface"] == interface and r["size"] == "large"
            ]
            if not pts:
                cells.append("–")
                continue
            frac = statistics.mean(
                1.0
                if (r["root_height_err"] > h_thr or r["root_ori_err"] > o_thr)
                else 0.0
                for r in pts
            )
            cells.append(f"{frac:.0%}")
        out.append(f"| {h_thr:.2f} m | {o_thr:.1f} rad | " + " | ".join(cells) + " |")
    out += [
        "",
        "Approximate: derived from per-episode mean errors, not the per-step",
        "indicator. Use it to check the ordering is threshold-insensitive, not to",
        "quote a rate.",
        "",
    ]
    return "\n".join(out)


def report_accounting(rows: list[dict[str, Any]]) -> str:
    """D1/D2: total interface parameters, and command bandwidth."""
    out = [
        "## Parameter accounting (D1)",
        "",
        "Reported planner parameters exclude the skill encoder, which the latent",
        "interface requires to define its code. Encoder input width is **670** —",
        "byte-identical to the full-body packet.",
        "",
        "| interface | size | planner params | + encoder | total |",
        "| --- | --- | --- | --- | --- |",
    ]
    for interface in ORDER:
        enc = INTERFACE_FACTS[interface]["encoder_params"]
        for size in SIZES:
            pts = [
                r["planner_params"]
                for r in rows
                if r["interface"] == interface
                and r["size"] == size
                and r["planner_params"]
            ]
            if not pts:
                continue
            planner = pts[0]
            out.append(
                f"| `{interface}` | {size} | {planner:,} | {enc:,} | "
                f"**{planner + enc:,}** |"
            )
    out += [
        "",
        "The encoder is a one-time offline cost and is **not** needed at inference",
        "(`skill_commander.py:2168` sets `skill_encoder = None`; DiffSR weights load",
        "only when `command_mode != 'z'`, which the protocol never sets). State it",
        "that way — as a training-time cost buying a permanently cheaper planner —",
        "rather than by quoting the planner count alone.",
        "",
        "## Command bandwidth (D2)",
        "",
        "Pre-empts *latent just sends less*. Bandwidth does not order the results:",
        "`root_points5` sends **less** than latent and performs far worse.",
        "",
        "| interface | packet | values/publish | values/second @ 5 Hz |",
        "| --- | --- | --- | --- |",
    ]
    for interface in ORDER:
        packet = INTERFACE_FACTS[interface]["packet"]
        out.append(
            f"| `{interface}` | {packet} | {packet} | {packet * PUBLISH_HZ:,.0f} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _cells(args.study_root.resolve())
    if not rows:
        raise SystemExit(f"No eval summaries under {args.study_root}")

    text = "\n".join(
        [
            f"# Interface error-gain and accounting ({len(rows)} cells)",
            "",
            report_error_gain(rows),
            report_threshold_sweep(rows),
            report_accounting(rows),
        ]
    )
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[INFO] Wrote {args.output}")


if __name__ == "__main__":
    main()
