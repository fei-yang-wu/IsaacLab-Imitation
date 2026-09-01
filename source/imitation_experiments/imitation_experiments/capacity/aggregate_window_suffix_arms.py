#!/usr/bin/env python3
"""Aggregate the encoder-window ablation arms into dose-response tables.

Two families answer the same question from different sides, and the table
keeps them apart because their numbers are not interchangeable.

``suffixN`` holds the horizon at 10 and shrinks only the encoder-visible
slice to the last N slots. Every arm predicts the same targets, so the raw
losses compare directly. The catch is that a suffix slice is not re-anchored:
slots stay in slot 0's heading frame, so even the last visible frame carries
the net displacement across the whole window. A suffix arm therefore hides
path shape, not window-scale information.

``hN`` moves the horizon itself, so the encoder input, the endpoint target,
and the next-chunk target all shrink together and each arm is internally
coherent -- the comparison a deployed encoder would actually face. The catch
is the mirror image: the chunk target is ``horizon x state_dim`` wide, so raw
losses move with the horizon and cannot be compared across these arms.

Reported per arm, averaged over the tail of training and then over seeds:

- ``train/jepa_endpoint_loss_eval`` -- the endpoint DiffSR term
  ``p(s[t+H] | s_t, z)`` on the frozen trajectory eval split.
- ``train/jepa_ntp_loss_eval`` -- the next-chunk diffusion term
  ``p(s[t+H+1..t+2H] | s_t, z)`` on the same split.
- ``train/jepa_*_z_explained`` -- ``1 - real / shuffled-code loss``, the
  fraction of each head's loss that knowing this window's code removes. It is
  dimensionless, so it is the only column that means anything across
  horizons. Older runs predate this control and report it as ``-``.

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
ENDPOINT_EXPLAINED_KEY = "train/jepa_endpoint_z_explained"
NTP_EXPLAINED_KEY = "train/jepa_ntp_z_explained"
RUN_NAME_PATTERN = re.compile(
    r"^(?P<arm>(?:suffix|h|srccur|srcpast)\d+)(?:_seed(?P<seed>\d+))?$"
)
ARM_FAMILIES = (
    ("srccur", "source-current"),
    ("srcpast", "source-paststart"),
    ("suffix", "suffix"),
    ("h", "horizon"),
)


def parse_run_name(name: str) -> tuple[str, int]:
    """``suffix2_seed1`` -> ``("suffix2", 1)``; a bare arm name means seed 0.

    Two arm families share this table: ``suffixN`` (fixed horizon 10, only the
    encoder-visible slice shrinks) and ``hN`` (the horizon itself shrinks, so
    input and both targets move together).
    """
    match = RUN_NAME_PATTERN.match(str(name))
    if match is None:
        msg = (
            f"Run directory {name!r} is not <suffixN>/<hN> with an optional "
            "_seed<S> suffix."
        )
        raise ValueError(msg)
    seed = match.group("seed")
    return match.group("arm"), 0 if seed is None else int(seed)


def comparison_group(arm: str) -> str:
    """Arms whose RAW losses may be compared: those sharing a target.

    The diffusion loss is computed on the target, so two arms are comparable
    exactly when their targets have the same distribution. Widening phi's
    source does NOT change that, which is why `srccur10` compares directly
    against `h10` and the suffix arms: all of them predict s[t+10] and the
    next chunk in s_t's heading frame.

    `h1/h2/h5` change the horizon, so the chunk target changes width.
    `srcpast10` anchors the window on s[t-10], so its target carries twenty
    steps of drift instead of ten. Both get their own group.
    """
    if arm.startswith("srcpast"):
        return f"paststart-h{suffix_length(arm)}"
    if arm.startswith("h"):
        return f"horizon{suffix_length(arm)}"
    # suffix* and srccur* are horizon-10, s_t-anchored by construction.
    return "horizon10"


def arm_family(arm: str) -> str:
    for prefix, family in ARM_FAMILIES:
        if arm.startswith(prefix):
            return family
    msg = f"Cannot assign a family to arm {arm!r}."
    raise ValueError(msg)


def suffix_length(arm: str) -> int:
    """The arm's numeric parameter: visible slots, horizon, or history length."""
    for prefix, _ in ARM_FAMILIES:
        if arm.startswith(prefix):
            return int(arm[len(prefix) :])
    msg = f"Cannot read a numeric parameter from arm {arm!r}."
    raise ValueError(msg)


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
    runs_dir: Path,
    *,
    tail_updates: int,
    expected_updates: int | None,
    skip_incomplete: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every run under ``runs_dir``.

    Returns ``(runs, skipped)``. An arm that stopped short of
    ``expected_updates`` raises by default, because a half-trained arm in a
    comparison table is worse than a missing one. ``skip_incomplete`` drops it
    instead and names it in ``skipped``, so a progress check can score the
    finished arms without ever hiding what it left out.
    """
    runs: list[dict[str, Any]] = []
    skipped: list[str] = []
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
        # The scale-free pair only exists in runs trained after the shuffled
        # code control landed. Raw losses stay comparable WITHIN a family;
        # across horizons only the explained fraction is meaningful.
        try:
            explained, _, _ = tail_means(
                records,
                keys=(ENDPOINT_EXPLAINED_KEY, NTP_EXPLAINED_KEY),
                tail_updates=tail_updates,
            )
        except ValueError:
            explained = {}
        if expected_updates is not None and final_update != expected_updates:
            if skip_incomplete:
                skipped.append(f"{path.name} (update {final_update})")
                continue
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
                "endpoint_z_explained": explained.get(ENDPOINT_EXPLAINED_KEY),
                "ntp_z_explained": explained.get(NTP_EXPLAINED_KEY),
                "family": arm_family(arm),
                "comparison_group": comparison_group(arm),
                "source": str(metrics_path),
            }
        )
    if not runs:
        raise ValueError(f"No suffix-arm runs with metrics found under {runs_dir}.")
    return runs, skipped


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
                "family": arm_family(arm),
                "comparison_group": comparison_group(arm),
                "seeds": sorted(run["seed"] for run in arm_runs),
                "endpoint_loss_eval": _stat_block(
                    [run["endpoint_loss_eval"] for run in arm_runs]
                ),
                "ntp_loss_eval": _stat_block(
                    [run["ntp_loss_eval"] for run in arm_runs]
                ),
                **{
                    key: _stat_block(values)
                    for key in ("endpoint_z_explained", "ntp_z_explained")
                    if (
                        values := [
                            run[key] for run in arm_runs if run.get(key) is not None
                        ]
                    )
                },
            }
        )
    return rows


def resolved_against_reference(
    rows: Sequence[dict[str, Any]], *, reference_arm: str, metric: str
) -> dict[str, dict[str, Any]]:
    """Per arm: relative change vs the reference and whether seed ranges overlap.

    ``overlaps=True`` means the two arms' per-seed tail means share a range,
    so this evidence does not separate them.

    Only arms sharing the reference's comparison group are included: a ratio
    between raw losses of two arms with different targets would be arithmetic
    without meaning. See :func:`comparison_group`.
    """
    reference = next((row for row in rows if row["arm"] == reference_arm), None)
    if reference is None:
        return {}
    rows = [
        row
        for row in rows
        if row["comparison_group"] == reference["comparison_group"]
        and row["arm"] != reference_arm
    ]
    ref = reference[metric]
    verdicts: dict[str, dict[str, Any]] = {}
    for row in rows:
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
    parser.add_argument(
        "--skip_incomplete",
        action="store_true",
        help=(
            "Drop arms that stopped short of --expected_updates and list them "
            "instead of failing. For progress checks on a running campaign."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs, skipped = collect_runs(
        args.runs_dir.expanduser().resolve(),
        tail_updates=args.tail_updates,
        expected_updates=args.expected_updates or None,
        skip_incomplete=bool(args.skip_incomplete),
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
        "skipped_incomplete": skipped,
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

    if skipped:
        print(f"skipped {len(skipped)} incomplete run(s): " + ", ".join(skipped))
    for family, title, note in (
        (
            "suffix",
            "suffix arms (horizon 10 fixed; only the visible slice shrinks)",
            "raw losses comparable within this family",
        ),
        (
            "horizon",
            "horizon arms (input and both targets shrink together)",
            "raw losses NOT comparable across horizons; read z-explained",
        ),
        (
            "source-current",
            "phi sees a past chunk, window anchored on s_t",
            "encoder input matches h10; read z-explained against h10",
        ),
        (
            "source-paststart",
            "phi sees a past chunk, window anchored on the oldest past frame",
            "encoder input distribution differs; not tracker-bindable yet",
        ),
    ):
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        print(f"\n{title}\n  {note}")
        header = f"  {'arm':8s} {'N':>3s} {'seeds':8s} {'endpoint eval':>24s} {'next-chunk eval':>24s}"
        has_explained = any("endpoint_z_explained" in row for row in family_rows)
        if has_explained:
            header += f" {'ep z-expl':>10s} {'ntp z-expl':>11s}"
        print(header)
        for row in family_rows:
            seeds = ",".join(str(seed) for seed in row["seeds"])
            ep, nt = row["endpoint_loss_eval"], row["ntp_loss_eval"]
            line = (
                f"  {row['arm']:8s} {row['suffix']:3d} {seeds:8s} "
                f"{ep['mean']:.4f} [{ep['min']:.4f},{ep['max']:.4f}] "
                f"{nt['mean']:.4f} [{nt['min']:.4f},{nt['max']:.4f}]"
            )
            if has_explained:
                epx = row.get("endpoint_z_explained")
                ntx = row.get("ntp_z_explained")
                line += (f" {epx['mean']:10.4f}" if epx else f" {'-':>10s}") + (
                    f" {ntx['mean']:11.4f}" if ntx else f" {'-':>11s}"
                )
            print(line)

    reference_row = next(
        (row for row in rows if row["arm"] == args.reference_arm), None
    )
    if reference_row is None:
        print(f"\nreference arm {args.reference_arm} absent; skipping comparison")
        return
    print(
        f"\nvs {args.reference_arm} within family "
        f"{reference_row['family']!r} (negative = better than the reference):"
    )
    for metric, verdicts in (
        ("endpoint", report["endpoint_vs_reference"]),
        ("next-chunk", report["ntp_vs_reference"]),
    ):
        for arm, verdict in verdicts.items():
            state = "resolved" if verdict["resolved"] else "UNRESOLVED (ranges overlap)"
            print(
                f"  {metric:10s} {arm:8s} "
                f"{verdict['relative_change_vs_reference']:+7.1%}  {state}"
            )


if __name__ == "__main__":
    main()
