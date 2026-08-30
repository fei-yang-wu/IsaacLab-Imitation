#!/usr/bin/env python3
"""Aggregate the suffix-k encoder-window arms into one dose-response table.

Each arm pretrains the same ``diffntp_chunk`` recipe and moves one variable,
``encoder_window_mode=suffix<N>``: the encoder sees only the last N slots of
the intermediate window. The question is whether the two losses the objective
optimizes improve as N grows.

Reported per arm, averaged over the tail of training and then over seeds:

- ``train/jepa_endpoint_loss_eval`` -- the endpoint DiffSR term
  ``p(s[t+H] | s_t, z)`` on the frozen trajectory eval split.
- ``train/jepa_ntp_loss_eval`` -- the next-chunk diffusion term
  ``p(s[t+H+1..t+2H] | s_t, z)`` on the same split.

Seed spread is reported as the min and max of the per-seed tail means, not as
a standard deviation: with two or three seeds a standard deviation implies
more precision than the sample supports. Two arms whose seed ranges overlap
are unresolved by this evidence.

Run from the repository root:

    pixi run python -m imitation_experiments.capacity.aggregate_window_suffix_arms \
        --runs_dir logs/endpoint_collapse_probe/ice_mirror \
        --output_dir <fresh dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

ENDPOINT_KEY = "train/jepa_endpoint_loss_eval"
NTP_KEY = "train/jepa_ntp_loss_eval"
RUN_NAME_PATTERN = re.compile(r"^(?P<arm>suffix\d+)(?:_seed(?P<seed>\d+))?$")


def parse_run_name(name: str) -> tuple[str, int]:
    """``suffix2_seed1`` -> ``("suffix2", 1)``; a bare arm name means seed 0."""
    match = RUN_NAME_PATTERN.match(str(name))
    if match is None:
        msg = f"Run directory {name!r} is not <suffixN> or <suffixN>_seed<S>."
        raise ValueError(msg)
    seed = match.group("seed")
    return match.group("arm"), 0 if seed is None else int(seed)


def suffix_length(arm: str) -> int:
    return int(arm.removeprefix("suffix"))


def read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} holds no metric records.")
    return records


def tail_means(
    records: Sequence[dict[str, Any]], *, keys: Iterable[str], tail_updates: int
) -> tuple[dict[str, float], int, int]:
    """Mean of each key over eval logs within ``tail_updates`` of the end.

    Averaging the tail instead of reading the final log matters here: a single
    diffusion eval batch is noisy enough to reorder adjacent arms.
    """
    final_update = max(int(record.get("update", 0)) for record in records)
    cutoff = final_update - int(tail_updates)
    tail = [
        record
        for record in records
        if int(record.get("update", 0)) >= cutoff and all(k in record for k in keys)
    ]
    if not tail:
        raise ValueError(
            f"No records carry every key {list(keys)} within the last "
            f"{tail_updates} updates (final update {final_update})."
        )
    means = {
        key: sum(float(record[key]) for record in tail) / len(tail) for key in keys
    }
    return means, len(tail), final_update


def collect_runs(
    runs_dir: Path, *, tail_updates: int, expected_updates: int | None
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        metrics_path = path / "metrics.jsonl"
        if not metrics_path.exists():
            metrics_path = path / "encoder" / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        arm, seed = parse_run_name(path.name)
        records = read_metrics(metrics_path)
        means, tail_count, final_update = tail_means(
            records, keys=(ENDPOINT_KEY, NTP_KEY), tail_updates=tail_updates
        )
        if expected_updates is not None and final_update != expected_updates:
            raise ValueError(
                f"{path.name} stopped at update {final_update}, expected "
                f"{expected_updates}. An incomplete arm must not enter the table."
            )
        runs.append(
            {
                "arm": arm,
                "suffix": suffix_length(arm),
                "seed": seed,
                "final_update": final_update,
                "tail_evals": tail_count,
                "endpoint_loss_eval": means[ENDPOINT_KEY],
                "ntp_loss_eval": means[NTP_KEY],
                "source": str(metrics_path),
            }
        )
    if not runs:
        raise ValueError(f"No suffix-arm runs with metrics found under {runs_dir}.")
    return runs


def _stat_block(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_arm.setdefault(run["arm"], []).append(run)
    rows: list[dict[str, Any]] = []
    for arm, arm_runs in sorted(
        by_arm.items(), key=lambda item: suffix_length(item[0])
    ):
        rows.append(
            {
                "arm": arm,
                "suffix": suffix_length(arm),
                "seeds": sorted(run["seed"] for run in arm_runs),
                "endpoint_loss_eval": _stat_block(
                    [run["endpoint_loss_eval"] for run in arm_runs]
                ),
                "ntp_loss_eval": _stat_block(
                    [run["ntp_loss_eval"] for run in arm_runs]
                ),
            }
        )
    return rows


def resolved_against_reference(
    rows: Sequence[dict[str, Any]], *, reference_arm: str, metric: str
) -> dict[str, dict[str, Any]]:
    """Per arm: relative change vs the reference and whether seed ranges overlap.

    ``overlaps=True`` means the two arms' per-seed tail means share a range,
    so this evidence does not separate them.
    """
    reference = next(row for row in rows if row["arm"] == reference_arm)
    ref = reference[metric]
    verdicts: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["arm"] == reference_arm:
            continue
        block = row[metric]
        overlaps = block["min"] <= ref["max"] and ref["min"] <= block["max"]
        verdicts[row["arm"]] = {
            "relative_change_vs_reference": (block["mean"] - ref["mean"]) / ref["mean"],
            "seed_ranges_overlap": overlaps,
            "resolved": not overlaps,
        }
    return verdicts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--tail_updates", type=int, default=5000)
    parser.add_argument(
        "--expected_updates",
        type=int,
        default=50000,
        help="Fail on any arm that stopped early. 0 disables the check.",
    )
    parser.add_argument("--reference_arm", type=str, default="suffix9")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs = collect_runs(
        args.runs_dir.expanduser().resolve(),
        tail_updates=args.tail_updates,
        expected_updates=args.expected_updates or None,
    )
    rows = aggregate(runs)
    report = {
        "schema": "window_suffix_dose_response_v1",
        "protocol": {
            "runs_dir": str(args.runs_dir.expanduser().resolve()),
            "tail_updates": int(args.tail_updates),
            "reference_arm": str(args.reference_arm),
            "metrics": [ENDPOINT_KEY, NTP_KEY],
            "spread": "min and max of per-seed tail means, not a std",
        },
        "runs": runs,
        "arms": rows,
        "endpoint_vs_reference": resolved_against_reference(
            rows, reference_arm=args.reference_arm, metric="endpoint_loss_eval"
        ),
        "ntp_vs_reference": resolved_against_reference(
            rows, reference_arm=args.reference_arm, metric="ntp_loss_eval"
        ),
    }
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "suffix_dose_response.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(f"{'arm':9s} {'seeds':10s} {'endpoint eval':>26s} {'next-chunk eval':>26s}")
    for row in rows:
        seeds = ",".join(str(seed) for seed in row["seeds"])
        ep, nt = row["endpoint_loss_eval"], row["ntp_loss_eval"]
        print(
            f"{row['arm']:9s} {seeds:10s} "
            f"{ep['mean']:.4f} [{ep['min']:.4f},{ep['max']:.4f}] "
            f"{nt['mean']:.4f} [{nt['min']:.4f},{nt['max']:.4f}]"
        )
    print(f"\nvs {args.reference_arm} (negative = better than the reference):")
    for metric, verdicts in (
        ("endpoint", report["endpoint_vs_reference"]),
        ("next-chunk", report["ntp_vs_reference"]),
    ):
        for arm, verdict in verdicts.items():
            state = "resolved" if verdict["resolved"] else "UNRESOLVED (ranges overlap)"
            print(
                f"  {metric:10s} {arm:9s} "
                f"{verdict['relative_change_vs_reference']:+7.1%}  {state}"
            )


if __name__ == "__main__":
    main()
