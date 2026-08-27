#!/usr/bin/env python3
"""Reduce one Isaac evaluation JSON to the canonical paper-facing row.

The canonical row is three numbers that must always travel together:

* success rate under SONIC's published definition,
* success-only, frame-weighted (micro) MPJPE-L in millimetres,
* success-only MPJPE-G in millimetres.

MPJPE-L alone flatters a policy that holds its pose while drifting, and a
success-only figure is meaningless beside a different success rate, so this
tool refuses to print one without the others.

The row also carries SONIC's other two published tracker metrics -- velocity
and acceleration distance -- and the per-termination counts that answer "why
does it fall". Those are reported alongside the three numbers, never instead of
them. Acceleration is absent from results produced before it was accumulated
per environment, so it is optional and stays ``None`` on an older file rather
than silently reporting the board-wide, all-transition mean in a success-only
column.

Pass ``--ranks_json`` to restrict the row to an explicit rank list, for a
board that is a subset of a larger scored run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["BoardRow", "summarize_board"]


@dataclass(frozen=True)
class BoardRow:
    """One reportable row. The first three numbers are required to publish it."""

    label: str
    subset: str
    episodes: int
    successes: int
    success_rate: float
    mpjpe_l_micro_mm: float
    mpjpe_l_macro_mm: float
    mpjpe_g_micro_mm: float
    successful_frames: int
    velocity_distance_mps: float | None = None
    acceleration_distance_mps2: float | None = None
    termination_counts: tuple[tuple[str, int], ...] = ()
    env_frames: int | None = None

    def render(self) -> str:
        head = (
            f"{self.label} [{self.subset}] "
            f"n={self.episodes} SR={self.success_rate:.4f} "
            f"MPJPE-L={self.mpjpe_l_micro_mm:.2f} mm (micro, success-only) "
            f"MPJPE-G={self.mpjpe_g_micro_mm:.2f} mm"
        )
        if self.env_frames is not None:
            head = f"{head} frames={self.env_frames / 1e9:.2f}B"
        if self.velocity_distance_mps is not None:
            head = f"{head} vel={self.velocity_distance_mps:.3f} m/s"
        if self.acceleration_distance_mps2 is not None:
            head = f"{head} acc={self.acceleration_distance_mps2:.2f} m/s2"
        failures = [
            f"{name}={count}"
            for name, count in self.termination_counts
            if name not in ("reference_finished", "time_out")
        ]
        if failures:
            head = f"{head} [{' '.join(failures)}]"
        return head


def _micro(values: Sequence[float], weights: Sequence[float]) -> float:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("no successful frames: the row has no defined MPJPE.")
    return float(sum(v * w for v, w in zip(values, weights)) / total)


def _micro_optional(
    values: Sequence[float | None], weights: Sequence[float]
) -> float | None:
    """Frame-weighted mean over the episodes that carry the metric.

    Returns None when no episode does, so an older result file reports an absent
    metric as absent rather than as zero.
    """
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    return _micro([v for v, _ in pairs], [w for _, w in pairs])


def _metric(episode: Mapping[str, Any], *names: str) -> float:
    """One per-episode metric, from either evaluator's schema.

    `evaluate_sonic_release` writes `metrics.mpjpe_l_mm`; `evaluate_checkpoint`
    writes `tracking_metrics.tracking_mpjpe_mm`. The two are the same quantity
    over the same 14 links, so a row must be computable from either without the
    caller knowing which tool produced the file.
    """
    for block in ("metrics", "tracking_metrics"):
        values = episode.get(block)
        if not isinstance(values, Mapping):
            continue
        for name in names:
            if name in values:
                return float(values[name])
    raise KeyError(
        f"episode carries none of {names} under 'metrics' or 'tracking_metrics'."
    )


def _optional_metric(episode: Mapping[str, Any], *names: str) -> float | None:
    """Like `_metric`, but returns None instead of raising.

    Used for metrics a result file may predate. Reporting the board-wide,
    all-transition value in a success-only column would be a different quantity,
    so an absent metric stays absent.
    """
    try:
        return _metric(episode, *names)
    except KeyError:
        return None


_FRAME_KEYS = (
    "cumulative_env_frames",
    "env_frames",
    "checkpoint_env_frames",
    "total_env_frames",
)
# Checkpoint trees name a segment by its TRUE cumulative frame count, e.g.
# `.../tracker/f10000269312/...`, because the per-segment step counter restarts
# on every chained resume.
_FRAME_IN_PATH = re.compile(r"(?:^|[/_])f(\d{6,})(?:[/_.]|$)")


def _env_frames(result: Mapping[str, Any]) -> int | None:
    """True cumulative environment frames behind this row, if recoverable."""
    metadata = result.get("metadata")
    if isinstance(metadata, Mapping):
        for key in _FRAME_KEYS:
            value = metadata.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        checkpoint = metadata.get("checkpoint")
        if isinstance(checkpoint, str):
            match = _FRAME_IN_PATH.search(checkpoint)
            if match is not None:
                return int(match.group(1))
    return None


def _termination_counts(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, int], ...]:
    """Per-term episode counts over the whole row, most frequent first.

    Counted over every episode, not only successes: the failure terms are the
    point, and they occur exactly on the episodes a success-only reduction
    drops.
    """
    counter: Counter[str] = Counter()
    for episode in episodes:
        terms = episode.get("termination_terms")
        if not terms:
            continue
        for term in terms:
            counter[str(term)] += 1
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def summarize_board(
    result: Mapping[str, Any],
    *,
    label: str | None = None,
    ranks: Iterable[int] | None = None,
    subset: str = "all",
) -> BoardRow:
    """One row from a per-environment evaluation result.

    Args:
        result: parsed ``evaluate_checkpoint`` / ``evaluate_sonic_release`` JSON.
        label: row label; defaults to the result's own label.
        ranks: restrict to these trajectory ranks. Every requested rank must be
            present, so a subset never silently scores fewer clips than asked.
        subset: name recorded on the row.
    """
    episodes = list(result["per_environment"])
    if ranks is not None:
        wanted = {int(rank) for rank in ranks}
        present = {int(item["trajectory_rank"]) for item in episodes}
        missing = sorted(wanted - present)
        if missing:
            raise ValueError(
                f"{len(missing)} requested ranks are absent from the result "
                f"(first: {missing[:5]}). Score the board that contains them."
            )
        episodes = [item for item in episodes if int(item["trajectory_rank"]) in wanted]
    if not episodes:
        raise ValueError("no episodes selected.")

    successes = [item for item in episodes if bool(item["completed_tracking_success"])]
    if not successes:
        raise ValueError("no successful episodes: MPJPE is undefined for this row.")
    weights = [float(item["survival_steps"]) for item in successes]
    local = [_metric(item, "mpjpe_l_mm", "tracking_mpjpe_mm") for item in successes]
    world = [_metric(item, "mpjpe_g_mm", "tracking_mpjpe_g_mm") for item in successes]
    velocity = [
        _optional_metric(item, "tracking_velocity_distance_mps") for item in successes
    ]
    acceleration = [
        _optional_metric(item, "tracking_acceleration_distance_mps2")
        for item in successes
    ]
    return BoardRow(
        label=str(label if label is not None else result.get("label", "unlabelled")),
        subset=subset,
        episodes=len(episodes),
        successes=len(successes),
        success_rate=len(successes) / len(episodes),
        mpjpe_l_micro_mm=_micro(local, weights),
        mpjpe_l_macro_mm=float(sum(local) / len(local)),
        mpjpe_g_micro_mm=_micro(world, weights),
        successful_frames=int(sum(weights)),
        velocity_distance_mps=_micro_optional(velocity, weights),
        acceleration_distance_mps2=_micro_optional(acceleration, weights),
        termination_counts=_termination_counts(episodes),
        env_frames=_env_frames(result),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--ranks_json",
        type=Path,
        default=None,
        help=(
            "JSON list of trajectory ranks to restrict the row to. Every rank "
            "must be present in the result, so a subset never silently scores "
            "fewer clips than asked."
        ),
    )
    parser.add_argument(
        "--subset_label",
        type=str,
        default=None,
        help="Name recorded on the row; defaults to the ranks file stem.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args(argv)

    ranks = None
    subset = "all"
    if args.ranks_json is not None:
        ranks = json.loads(args.ranks_json.read_text(encoding="utf-8"))
        subset = args.subset_label or args.ranks_json.stem
    rows = []
    for path in args.results:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        row = summarize_board(result, ranks=ranks, subset=subset)
        rows.append(row)
        print(row.render())
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([row.__dict__ for row in rows], indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
