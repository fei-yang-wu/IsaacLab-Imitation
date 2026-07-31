#!/usr/bin/env python3
"""Aggregate a planner training-budget curve from milestone evaluations.

Reads the closed-loop summaries written by run_fb670_budget_curve.sh (one per
training-update milestone plus best.pt) and emits one table per model size:
updates vs survival, fall count, termination-truncated MPJPE, and the trainer's
held-out normalized target RMSE at the same update (joined from metrics.jsonl).

Every summary is verified against the rigorous protocol before it is admitted:
frame-0 start, the expected environment count, base_too_low-only termination at
the expected fall height, pushes enabled, and a pinned command joint order.
A summary that fails any check aborts the aggregation -- a silently different
protocol point would corrupt the curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Interfaces whose published packet carries a per-joint layout. Only these can
# suffer the joint-order permutation bug, so only these must prove the pin. The
# 258-value latent command has no joint layout and its evaluator never emits the
# field -- requiring it there would reject every valid latent summary.
_EXPLICIT_INTERFACES = frozenset(
    {"full_body_trajectory", "root_qpos", "root_points5", "ee_trajectory"}
)


def _verify_protocol(
    summary: dict[str, Any],
    *,
    path: Path,
    interface: str,
    expected_num_envs: int,
    expected_fall_height_m: float,
) -> None:
    metadata = summary["metadata"]
    push = metadata.get("push_perturbation") or {}
    starts = summary.get("start_trajectories", {}).get("local_steps", [])
    checks = {
        f"num_envs=={expected_num_envs}": int(metadata["num_envs"])
        == expected_num_envs,
        # Every environment must actually begin at reference frame 0. Reading
        # the recorded starts is stronger than trusting the reset-range config.
        "all_envs_start_at_frame_0": bool(starts) and set(starts) == {0},
        "base_only_termination": bool(metadata.get("base_only_termination")),
        f"fall_height_m=={expected_fall_height_m}": float(
            metadata.get("fall_height_m", -1.0)
        )
        == expected_fall_height_m,
        "push_perturbation_enabled": bool(push.get("enabled")),
        "no_oracle_substitution": not bool(
            summary.get("oracle_substitution", {}).get("enabled")
        ),
        "observation_corruption_disabled": not bool(
            metadata.get("policy_observation_corruption_enabled")
        ),
    }
    if interface in _EXPLICIT_INTERFACES:
        checks["command_joint_order_pinned"] = bool(
            summary.get("command_joint_order_pinned")
        )
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(
            f"[ERROR] {path} violates the budget-curve protocol: {failed}"
        )


def _val_rmse_by_update(metrics_path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    if not metrics_path.is_file():
        return values
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rmse = row.get("val/normalized_target_rmse_mean")
            if rmse is not None:
                values[int(row["update"])] = float(rmse)
    return values


def _collect_size(
    *,
    study_root: Path,
    train_root: Path | None,
    size: str,
    seed: int,
    interface: str,
    run_tag: str,
    eval_dir_name: str,
    expected_num_envs: int,
    expected_fall_height_m: float,
) -> dict[str, Any]:
    size_root = study_root / size / f"seed{seed}"
    eval_root = size_root / eval_dir_name
    # train_root lets a route diagnostic (e.g. an already-trained planner
    # evaluated through a different tracker) point config.json/metrics.jsonl
    # at the ORIGINAL training study root while writing its own eval outputs
    # under study_root -- the two need not coincide.
    train_size_root = (train_root or study_root) / size / f"seed{seed}"
    run_dir = train_size_root / run_tag
    if not eval_root.is_dir():
        raise SystemExit(f"[ERROR] Missing evaluation root: {eval_root}")

    train_config: dict[str, Any] = {}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        train_config = _read_json(config_path)
    val_rmse = _val_rmse_by_update(run_dir / "metrics.jsonl")

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(eval_root.glob("update_*/summary.json")):
        update = int(summary_path.parent.name.removeprefix("update_"))
        rows.append(
            _row_from_summary(
                summary_path,
                update=update,
                tag=summary_path.parent.name,
                interface=interface,
                val_rmse=val_rmse.get(update),
                expected_num_envs=expected_num_envs,
                expected_fall_height_m=expected_fall_height_m,
            )
        )
    if not rows:
        raise SystemExit(f"[ERROR] No milestone evaluations under {eval_root}")

    best_path = eval_root / "best" / "summary.json"
    best_row = None
    if best_path.is_file():
        best_update = int(train_config.get("best_validation_update", -1))
        best_row = _row_from_summary(
            best_path,
            update=best_update,
            tag="best",
            interface=interface,
            val_rmse=val_rmse.get(best_update),
            expected_num_envs=expected_num_envs,
            expected_fall_height_m=expected_fall_height_m,
        )

    rows.sort(key=lambda row: row["update"])
    return {
        "size": size,
        "seed": seed,
        "run_tag": run_tag,
        "best_validation_update": train_config.get("best_validation_update"),
        "best_validation_metric": train_config.get("best_validation_metric"),
        "milestones": rows,
        "best": best_row,
    }


def _row_from_summary(
    path: Path,
    *,
    update: int,
    tag: str,
    interface: str,
    val_rmse: float | None,
    expected_num_envs: int,
    expected_fall_height_m: float,
) -> dict[str, Any]:
    summary = _read_json(path)
    _verify_protocol(
        summary,
        path=path,
        interface=interface,
        expected_num_envs=expected_num_envs,
        expected_fall_height_m=expected_fall_height_m,
    )
    metadata = summary["metadata"]
    aggregate = summary["aggregate"]
    mpjpe = summary["metrics"]["tracking_mpjpe_mm"]
    latency = summary.get("planner_inference_latency_ms") or {}
    steps_run = int(summary.get("steps_run", summary.get("max_steps", 0)))
    causes = aggregate.get("termination_cause_env_counts", {}) or {}
    survival_steps_mean = float(aggregate["survival_steps_mean"])
    # `survival_rate` / `fallen_env_count` are computed over the evaluator's
    # active mask and badly under-report failure once environments reset and
    # fall again: a full-body point reading survival_rate 0.916 simultaneously
    # recorded done_rate 0.742, 3039 base_too_low events and a mean survival of
    # 283/500 steps. Carry the raw fields but lead with the honest ones.
    return {
        "tag": tag,
        "update": update,
        "episode_completion_rate": 1.0 - float(aggregate.get("done_rate", 0.0)),
        "survival_steps_mean": survival_steps_mean,
        "survival_steps_fraction": (
            survival_steps_mean / steps_run if steps_run else float("nan")
        ),
        "fall_event_count": int(causes.get("base_too_low", 0)),
        "reported_survival_rate_active_mask": float(aggregate["survival_rate"]),
        "reported_fallen_env_count_active_mask": int(aggregate["fallen_env_count"]),
        "steps_run": steps_run,
        "mpjpe_mm": float(mpjpe["mean"]),
        "mpjpe_std_mm": float(mpjpe["std"]),
        # MPJPE covers only pre-termination transitions. When this is well below
        # num_envs * steps_run the number is a truncated, favourable window --
        # the robot's error BEFORE it fell -- and understates full-horizon drift.
        "valid_transition_count": int(mpjpe["count"]),
        "metric_horizon_coverage": (
            int(mpjpe["count"]) / (int(metadata["num_envs"]) * steps_run)
            if steps_run
            else float("nan")
        ),
        "val_normalized_target_rmse": val_rmse,
        "planner_latency_ms_mean": latency.get("mean"),
        "summary_path": str(path),
    }


def _markdown(per_size: list[dict[str, Any]], interface: str) -> str:
    lines = [
        f"# {interface} training-budget curve",
        "",
        "Protocol: reference frame 0, training-time domain randomization and",
        "pushes active, base_too_low-only termination, MPJPE over valid",
        "pre-termination transitions.",
        "",
        "`completed` is the fraction of environments that finished the horizon",
        "without terminating (1 - done_rate). `fall events` counts every",
        "base_too_low termination including post-reset repeats. `horizon` is the",
        "fraction of the full num_envs x steps window the MPJPE mean covers --",
        "below 1.0 the MPJPE is a truncated, favourable window measured before",
        "the robot fell, so it UNDERSTATES full-horizon drift.",
        "",
        "The evaluator's own `survival_rate` is reported last for traceability",
        "only: it is computed over an active mask and under-reports failure.",
        "",
    ]
    header = (
        "| updates | completed | mean steps | fall events | MPJPE (mm) "
        "| horizon | held-out RMSE | (survival_rate) |"
    )
    for entry in per_size:
        lines += [f"## {entry['size']} (seed {entry['seed']})", "", header]
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        rows = list(entry["milestones"])
        if entry["best"] is not None:
            rows.append(entry["best"])
        for row in rows:
            rmse = row["val_normalized_target_rmse"]
            label = (
                f"{row['update']} (best)" if row["tag"] == "best" else row["update"]
            )
            lines.append(
                f"| {label} | {row['episode_completion_rate']:.3f} "
                f"| {row['survival_steps_mean']:.0f}/{row['steps_run']} "
                f"| {row['fall_event_count']} | {row['mpjpe_mm']:.2f} "
                f"| {row['metric_horizon_coverage']:.2f} "
                f"| {'-' if rmse is None else f'{rmse:.4f}'} "
                f"| {row['reported_survival_rate_active_mask']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument(
        "--train_root",
        type=Path,
        default=None,
        help=(
            "Study root holding config.json/metrics.jsonl per size/seed/run_tag, "
            "if different from --study_root (e.g. a route diagnostic reusing "
            "another study's already-trained planner checkpoints)."
        ),
    )
    parser.add_argument("--interface", required=True)
    parser.add_argument("--sizes", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--eval_dir_name", required=True)
    parser.add_argument("--expected_num_envs", type=int, required=True)
    parser.add_argument("--expected_fall_height_m", type=float, default=0.4)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(
            f"[ERROR] Refusing to overwrite existing output dir: {args.output_dir}"
        )

    per_size = [
        _collect_size(
            study_root=args.study_root,
            train_root=args.train_root,
            size=size,
            seed=int(args.seed),
            interface=args.interface,
            run_tag=args.run_tag,
            eval_dir_name=args.eval_dir_name,
            expected_num_envs=int(args.expected_num_envs),
            expected_fall_height_m=float(args.expected_fall_height_m),
        )
        for size in args.sizes
    ]

    args.output_dir.mkdir(parents=True)
    payload = {
        "interface": args.interface,
        "protocol": {
            "reference_start_frame": 0,
            "num_envs": int(args.expected_num_envs),
            "termination": "base_too_low_only",
            "fall_height_m": float(args.expected_fall_height_m),
            "domain_randomization": "training-time events and pushes active",
            "mpjpe_window": "valid_pre_termination_transitions",
        },
        "sizes": per_size,
    }
    curve_json = args.output_dir / "budget_curve.json"
    curve_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    curve_md = args.output_dir / "budget_curve.md"
    curve_md.write_text(_markdown(per_size, args.interface), encoding="utf-8")
    print(f"[PASS] Wrote {curve_json}")
    print(f"[PASS] Wrote {curve_md}")


if __name__ == "__main__":
    main()
