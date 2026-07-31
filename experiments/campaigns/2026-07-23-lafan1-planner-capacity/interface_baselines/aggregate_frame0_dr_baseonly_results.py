#!/usr/bin/env python3
"""Aggregate the fixed frame-0, domain-randomized, base-only diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SIZES = ("tiny", "small", "medium", "large")
ROUTES = ("root_qpos", "latent_skill")
UPDATES = {"tiny": 10_000, "small": 20_000, "medium": 30_000, "large": 50_000}
EVAL_DIR = "eval_frame0_dr_baseonly_100env_seed0"
H30_EVAL_DIR = "evaluation_frame0_dr_baseonly_100env_seed0"


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _validate(summary: dict[str, Any], path: Path) -> None:
    metadata = summary.get("metadata", {})
    starts = summary.get("start_trajectories", {}).get("local_steps", [])
    expected_disabled = {"anchor_pos", "anchor_ori", "ee_body_pos", "foot_pos_xyz"}
    checks = {
        "num_envs=100": int(metadata.get("num_envs", -1)) == 100,
        "eval seed=0": int(metadata.get("seed", -1)) == 0,
        "frame-0 starts": len(starts) == 100 and set(starts) == {0},
        "base-only termination": metadata.get("base_only_termination") is True,
        "fall height=0.4": float(metadata.get("fall_height_m", -1.0)) == 0.4,
        # Some streamed-vanilla summaries record ``time_out`` in the same
        # disabled-terms list even though it is not a tracking term. Validate
        # the tracking subset here and keep the explicit timeout check below.
        "tracking terms disabled": (
            set(metadata.get("disabled_tracking_termination_terms", []))
            - {"time_out"}
        )
        == expected_disabled,
        "timeout disabled": metadata.get("time_out_enabled") is False,
        "domain randomization enabled": bool(
            metadata.get("push_perturbation", {}).get("enabled")
        ),
        "500-step outer horizon": int(summary.get("max_steps", -1)) == 500,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Protocol mismatch in {path}: {failed}")


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    mpjpe = summary["metrics"]["tracking_mpjpe_mm"]
    latency = summary.get("planner_inference_latency_ms") or {}
    metadata = summary["metadata"]
    planner = metadata.get("planner_metadata") or {}
    return {
        "planner_parameters": int(planner["parameter_count"]),
        "survival_rate": float(aggregate["survival_rate"]),
        "fallen_env_count": int(aggregate["fallen_env_count"]),
        "mean_survival_steps": float(aggregate["survival_steps_mean"]),
        "termination_truncated_mpjpe_mm": float(mpjpe["mean"]),
        "termination_truncated_mpjpe_std_mm": float(mpjpe["std"]),
        "valid_transition_count": int(mpjpe["count"]),
        "planner_latency_ms": float(latency["mean"]),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Frame-0 DR/base-only evaluation",
        "",
        "100 environments, seed 0, 500 steps, torso height < 0.40 m as the "
        "only early termination. MPJPE uses valid pre-termination transitions.",
        "",
        "| Route | Size | Train updates | Params | Survival | Falls | "
        "Truncated MPJPE (mm) | Valid transitions | Planner ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "evaluated":
            values = ("", "", "", "", "", "")
        else:
            values = (
                f'{row["planner_parameters"]:,}',
                f'{row["survival_rate"]:.2f}',
                str(row["fallen_env_count"]),
                f'{row["termination_truncated_mpjpe_mm"]:.2f}',
                f'{row["valid_transition_count"]:,}',
                f'{row["planner_latency_ms"]:.2f}',
            )
        lines.append(
            f'| {row["route"]} | {row["size"]} | {row["train_updates"]:,} | '
            f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | "
            f"{values[4]} | {values[5]} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_root", type=Path, required=True)
    parser.add_argument(
        "--h30_root",
        type=Path,
        help=(
            "H30 study root. Defaults to the standard sibling of --study_root."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    study_root = args.study_root.expanduser().resolve()
    h30_root = (
        args.h30_root.expanduser().resolve()
        if args.h30_root is not None
        else study_root.parent / "lafan1_enc380_h30_temporal_medium_seed0_20260730"
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    capacity = study_root / "motions/walk1_subject1/capacity"
    for size in SIZES:
        for route in ROUTES:
            summary_path = (
                capacity / size / "seed0/matched" / route / EVAL_DIR / "summary.json"
            )
            row: dict[str, Any] = {
                "route": (
                    "root_qpos→encoder→latent tracker"
                    if route == "root_qpos"
                    else "latent→latent tracker"
                ),
                "route_key": route,
                "size": size,
                "train_updates": UPDATES[size],
                "status": "pending",
                "summary_path": str(summary_path),
            }
            if summary_path.is_file():
                summary = _read(summary_path)
                _validate(summary, summary_path)
                row.update(_row(summary))
                row["status"] = "evaluated"
            rows.append(row)

    for size in SIZES:
        pure_summary = (
            study_root
            / "pure_root_qpos_tracker"
            / size
            / "seed0"
            / EVAL_DIR
            / "summary.json"
        )
        pure_row: dict[str, Any] = {
            "route": "root_qpos→root_qpos tracker",
            "route_key": "pure_root_qpos",
            "size": size,
            "train_updates": UPDATES[size],
            "status": "pending",
            "summary_path": str(pure_summary),
        }
        if pure_summary.is_file():
            summary = _read(pure_summary)
            _validate(summary, pure_summary)
            pure_row.update(_row(summary))
            pure_row["status"] = "evaluated"
        rows.append(pure_row)

    for size in SIZES:
        for route_key, route_label in (
            (
                "h30_first10",
                "root_qpos H30 first-H10→encoder→latent tracker",
            ),
            (
                "h30_temporal",
                "root_qpos H30 temporal ensemble→encoder→latent tracker",
            ),
        ):
            summary_path = (
                h30_root
                / H30_EVAL_DIR
                / size
                / route_key
                / "summary.json"
            )
            row = {
                "route": route_label,
                "route_key": route_key,
                "size": size,
                "train_updates": UPDATES[size],
                "status": "pending",
                "summary_path": str(summary_path),
            }
            if summary_path.is_file():
                summary = _read(summary_path)
                _validate(summary, summary_path)
                row.update(_row(summary))
                row["status"] = "evaluated"
            rows.append(row)

    payload = {
        "protocol": {
            "motion_name": "walk1_subject1",
            "reference_start_frame": 0,
            "num_envs": 100,
            "eval_seed": 0,
            "max_steps": 500,
            "fall_body": "torso_link",
            "fall_height_m": 0.4,
            "domain_randomization": True,
            "mpjpe_window": "valid_pre_termination_transitions",
        },
        "h30_root": str(h30_root),
        "rows": rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "results.md").write_text(_markdown(rows), encoding="utf-8")
    print(f"[PASS] Wrote {output_dir / 'results.json'}")
    print(f"[PASS] Wrote {output_dir / 'results.md'}")


if __name__ == "__main__":
    main()
