#!/usr/bin/env python3
"""Fail-fast audit for one 10M-frame latent-ablation qualification run."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


METRIC_RE = re.compile(
    r"iter=(?P<iteration>\d+)/\d+\s+\|\s+"
    r"frames=(?P<frames>\d+)/\d+\s+\|\s+"
    r"r_step=(?P<r_step>[-+0-9.eE]+)\s+\|\s+"
    r"ep_len=(?P<ep_len>[-+0-9.eE]+)\s+\|\s+"
    r"r_ep=(?P<r_ep>[-+0-9.eE]+)"
)
FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "RuntimeError:",
    "FloatingPointError:",
)


def _window_median(values: list[float], *, first: bool) -> float:
    width = max(5, len(values) // 5)
    window = values[:width] if first else values[-width:]
    return float(statistics.median(window))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.train_log.read_text(errors="replace")
    rows = [
        {
            "iteration": int(match["iteration"]),
            "frames": int(match["frames"]),
            "r_step": float(match["r_step"]),
            "ep_len": float(match["ep_len"]),
            "r_ep": float(match["r_ep"]),
        }
        for match in METRIC_RE.finditer(text)
    ]
    checkpoints = sorted(args.run_root.rglob("model_step_*.pt"))
    failures: list[str] = []
    if len(rows) < 20:
        failures.append(f"only {len(rows)} training metric rows (need at least 20)")
    if not checkpoints:
        failures.append("no model_step checkpoint found")
    for pattern in FATAL_PATTERNS:
        if pattern in text:
            failures.append(f"training log contains {pattern!r}")

    summary: dict[str, object] = {
        "arm": args.arm,
        "target_frames": args.target_frames,
        "metric_rows": len(rows),
        "checkpoint": str(checkpoints[-1].resolve()) if checkpoints else None,
    }
    if rows:
        final_frames = rows[-1]["frames"]
        finite = all(
            math.isfinite(row[key])
            for row in rows
            for key in ("r_step", "ep_len", "r_ep")
        )
        if final_frames < args.target_frames:
            failures.append(
                f"final logged frames {final_frames} below target {args.target_frames}"
            )
        if not finite:
            failures.append("non-finite return or episode-length metric")

        r_step = [row["r_step"] for row in rows]
        ep_len = [row["ep_len"] for row in rows]
        r_ep = [row["r_ep"] for row in rows]
        initial = {
            "r_step": _window_median(r_step, first=True),
            "ep_len": _window_median(ep_len, first=True),
            "r_ep": _window_median(r_ep, first=True),
        }
        final = {
            "r_step": _window_median(r_step, first=False),
            "ep_len": _window_median(ep_len, first=False),
            "r_ep": _window_median(r_ep, first=False),
        }
        relative_change = {
            key: (final[key] - initial[key]) / max(abs(initial[key]), 1.0e-8)
            for key in initial
        }
        # Strict-from-scratch learning is noisy at 10M. Require a meaningful
        # improvement in either survival or per-step reward, while rejecting
        # catastrophic regression in the other signal.
        improvement = (
            relative_change["ep_len"] >= 0.10
            or relative_change["r_step"] >= 0.02
        )
        stable = (
            relative_change["ep_len"] >= -0.20
            and relative_change["r_step"] >= -0.10
        )
        if not improvement:
            failures.append(
                "no early learning signal: need >=10% episode-length or >=2% "
                "per-step-reward improvement"
            )
        if not stable:
            failures.append(
                "episode length or per-step reward regressed catastrophically"
            )
        summary.update(
            {
                "final_frames": final_frames,
                "initial_window_median": initial,
                "final_window_median": final,
                "relative_change": relative_change,
                "finite_metrics": finite,
            }
        )

    summary["passed"] = not failures
    summary["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
