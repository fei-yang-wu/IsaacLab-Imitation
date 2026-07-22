#!/usr/bin/env python3
"""Promote an upstream eval summary into the ablation table format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--setting", required=True)
    return parser.parse_args()


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    payload = metrics.get(key)
    if isinstance(payload, dict) and "mean" in payload:
        value = payload.get("mean")
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and value != value:  # NaN
            continue
        return value
    return None


def normalize_primary_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Return flat primary metrics using upstream closed-loop summaries."""
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = {}
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    success = _first_present(
        aggregate.get("success_rate"),
        payload.get("success_rate"),
        aggregate.get("tracking_success_rate"),
        # Prefer strict tracking success; fall back to threshold-only if needed.
        aggregate.get("threshold_tracking_success_rate"),
    )
    mpjpe = _first_present(
        aggregate.get("mpjpe_l_mm"),
        payload.get("mpjpe_l_mm"),
        aggregate.get("tracking_mpjpe_mm"),
        _metric_mean(metrics, "tracking_mpjpe_mm"),
        # Convert meters mean if only SI key is present.
        (
            None
            if _metric_mean(metrics, "tracking_mpjpe_m") is None
            else float(_metric_mean(metrics, "tracking_mpjpe_m")) * 1000.0
        ),
    )
    e_vel = _first_present(
        aggregate.get("e_vel"),
        payload.get("e_vel"),
        _metric_mean(metrics, "tracking_velocity_distance_mps"),
        _metric_mean(metrics, "e_vel_mm_per_frame"),
        _metric_mean(metrics, "e_vel_mmps"),
    )
    e_acc = _first_present(
        aggregate.get("e_acc"),
        payload.get("e_acc"),
        _metric_mean(metrics, "tracking_acceleration_distance_mps2"),
        _metric_mean(metrics, "e_acc_mm_per_frame2"),
        _metric_mean(metrics, "e_acc_mmps2"),
    )
    return {
        "success_rate": success,
        "mpjpe_l_mm": mpjpe,
        "e_vel": e_vel,
        "e_acc": e_acc,
    }


def main() -> None:
    args = _parse_args()
    payload: dict[str, Any] = json.loads(args.src.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = {}
        payload["aggregate"] = aggregate

    # Ablation identity fields.
    payload.update(
        {
            "interface": args.interface,
            "table": args.table,
            "window": int(args.window),
            "rank": int(args.rank),
            "trajectory": args.trajectory,
            "setting": args.setting,
            "planner_unit": "trajectory",
            "status": "ok",
        }
    )
    primary = normalize_primary_metrics(payload)
    for key, value in primary.items():
        if value is not None:
            payload[key] = value
            aggregate.setdefault(key, value)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[promote_eval_summary] wrote {args.dst}")


if __name__ == "__main__":
    main()
