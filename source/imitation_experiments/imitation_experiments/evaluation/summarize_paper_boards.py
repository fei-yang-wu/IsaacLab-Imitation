#!/usr/bin/env python3
"""Reduce one Isaac evaluation JSON to the canonical paper-facing row.

The canonical row is three numbers that must always travel together:

* success rate under SONIC's published definition,
* success-only, frame-weighted (micro) MPJPE-L in millimetres,
* success-only MPJPE-G in millimetres.

MPJPE-L alone flatters a policy that holds its pose while drifting, and a
success-only figure is meaningless beside a different success rate, so this
tool refuses to print one without the others.

Pass ``--subset deployable`` to restrict the row to the frozen
hardware-plausible clips (`evaluation.clip_features.DEPLOYABLE_CLIP_RULE_V1`).
That is the population comparable with SONIC's published 22.3 mm at 100%
success; the unrestricted block is comparable with its large held-out sets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from imitation_experiments.evaluation.protocol import DEPLOYABLE123_MOTIONS

__all__ = ["BoardRow", "summarize_board"]


@dataclass(frozen=True)
class BoardRow:
    """One reportable row. Every field is required to publish the row."""

    label: str
    subset: str
    episodes: int
    successes: int
    success_rate: float
    mpjpe_l_micro_mm: float
    mpjpe_l_macro_mm: float
    mpjpe_g_micro_mm: float
    successful_frames: int

    def render(self) -> str:
        return (
            f"{self.label} [{self.subset}] "
            f"n={self.episodes} SR={self.success_rate:.4f} "
            f"MPJPE-L={self.mpjpe_l_micro_mm:.2f} mm (micro, success-only) "
            f"MPJPE-G={self.mpjpe_g_micro_mm:.2f} mm"
        )


def _micro(values: Sequence[float], weights: Sequence[float]) -> float:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("no successful frames: the row has no defined MPJPE.")
    return float(sum(v * w for v, w in zip(values, weights)) / total)


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
    local = [float(item["metrics"]["mpjpe_l_mm"]) for item in successes]
    world = [float(item["metrics"]["mpjpe_g_mm"]) for item in successes]
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
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--subset",
        choices=("all", "deployable"),
        default="all",
        help=(
            "'deployable' restricts to the frozen 123-clip hardware-plausible "
            "board, which only applies to a result over the canonical block."
        ),
    )
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args(argv)

    ranks = (
        [rank for rank, _ in DEPLOYABLE123_MOTIONS]
        if args.subset == "deployable"
        else None
    )
    rows = []
    for path in args.results:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        row = summarize_board(result, ranks=ranks, subset=args.subset)
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
