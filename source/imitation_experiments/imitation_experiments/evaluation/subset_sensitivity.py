"""How easily could a chosen subset have produced a given number?

Selecting clips until a score matches a target is fitting, and a fitted board
looks exactly like an honest one from the outside. This module measures the
fit instead of hiding it, so a claim of the form "population P reproduces
X mm" can be checked before it is believed.

Two questions, both answered from a stored per-clip result file:

1. **Null.** Over many random subsets of the given size, where does the target
   sit? A target inside the bulk of that distribution is reached by chance and
   proves nothing about the population.
2. **Rule grid.** Over a bounded, pre-declared family of reference-only clip
   rules, what share land within a tolerance of the target? A high share means
   a "principled" rule matching the target is cheap to find, so finding one
   carries no evidence.

Built 2026-08-25 out of an attempt to match SONIC's 123-clip deployment figure
of 22.3 mm with `sonic_v1_1`. The answer was that 1.8% of random 123-clip
subsets and 13.1% of a 512-rule grid already reach it, and the closest rule
("short and slow") drops all but 12 of the 123 clips drawn from the motion
families SONIC deploys on hardware. See the campaign README.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

from .clip_features import ClipFeatures, difficulty_index

__all__ = [
    "RULE_GRID_V1",
    "ClipScore",
    "NullResult",
    "RuleGridResult",
    "load_clip_scores",
    "micro_mpjpe",
    "null_distribution",
    "rule_grid",
]


@dataclass(frozen=True)
class ClipScore:
    """One successful clip's contribution to a micro-averaged score."""

    rank: int
    frames: int
    value: float


@dataclass(frozen=True)
class NullResult:
    """Where a target sits among random subsets of a fixed size."""

    size: int
    draws: int
    mean: float
    stdev: float
    minimum: float
    maximum: float
    target: float
    share_at_or_below: float

    @property
    def z(self) -> float:
        return (self.target - self.mean) / self.stdev if self.stdev else float("nan")


@dataclass(frozen=True)
class RuleGridResult:
    """How many rules of a bounded grid land near a target."""

    rules_tried: int
    rules_valid: int
    hits: int
    tolerance: float
    target: float
    best_value: float
    best_rule: Mapping[str, Any]
    minimum: float
    median: float
    maximum: float

    @property
    def hit_share(self) -> float:
        return self.hits / self.rules_valid if self.rules_valid else 0.0


def load_clip_scores(
    result_json: str | Path, *, metric: str = "mpjpe_l_mm"
) -> list[ClipScore]:
    """Successful clips of one result file, with their frame weights.

    Accepts both per-clip schemas in this repo: the SONIC evaluator's
    `metrics.mpjpe_l_mm` and `evaluate_checkpoint`'s
    `tracking_metrics.tracking_mpjpe_mm`.
    """
    aliases = {
        "mpjpe_l_mm": ("tracking_metrics", "tracking_mpjpe_mm"),
        "mpjpe_g_mm": ("tracking_metrics", "tracking_mpjpe_g_mm"),
    }
    payload = json.loads(Path(result_json).read_text())
    scores: list[ClipScore] = []
    for entry in payload["per_environment"]:
        if not entry.get("completed_tracking_success"):
            continue
        value = (entry.get("metrics") or {}).get(metric)
        if value is None and metric in aliases:
            group, name = aliases[metric]
            value = (entry.get(group) or {}).get(name)
        if value is None:
            raise KeyError(f"clip {entry['trajectory_rank']} carries no {metric!r}.")
        scores.append(
            ClipScore(
                rank=int(entry["trajectory_rank"]),
                frames=int(entry["survival_steps"]),
                value=float(value),
            )
        )
    return scores


def micro_mpjpe(scores: Sequence[ClipScore]) -> float:
    """Frame-weighted mean, the repo's `micro` convention."""
    frames = sum(score.frames for score in scores)
    if frames == 0:
        raise ValueError("no frames to average.")
    return sum(score.value * score.frames for score in scores) / frames


def null_distribution(
    scores: Sequence[ClipScore],
    *,
    target: float,
    size: int,
    draws: int = 4000,
    seed: int = 20260825,
) -> NullResult:
    """Micro score of `draws` random subsets of `size` clips."""
    if len(scores) < size:
        raise ValueError(f"pool holds {len(scores)} clips, fewer than {size}.")
    rng = random.Random(int(seed))
    pool = list(scores)
    values = [micro_mpjpe(rng.sample(pool, size)) for _ in range(int(draws))]
    return NullResult(
        size=int(size),
        draws=int(draws),
        mean=statistics.mean(values),
        stdev=statistics.pstdev(values),
        minimum=min(values),
        maximum=max(values),
        target=float(target),
        share_at_or_below=sum(1 for value in values if value <= target) / len(values),
    )


# A bounded, pre-declared family of reference-only rules. It is deliberately
# small and boring: the point is to show that even a modest search reaches most
# targets, so a larger search proves even less. Every axis is a feature the
# canonical testbed rule already uses, so a rule here is the kind of thing a
# paper would plausibly present as principled.
RULE_GRID_V1: Mapping[str, Sequence[Any]] = {
    "difficulty_max": (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95),
    "frames_range": ((100, 300), (100, 500), (100, 800), (100, 1500)),
    "pel_z_min_floor": (0.0, 0.30, 0.50, 0.60),
    "root_speed_max_cap": (float("inf"), 3.5, 2.5, 1.5),
}


def rule_grid(
    scores: Sequence[ClipScore],
    features: Sequence[ClipFeatures],
    *,
    target: float,
    size: int,
    tolerance: float = 0.5,
    grid: Mapping[str, Sequence[Any]] = RULE_GRID_V1,
    seed: int = 20260825,
) -> RuleGridResult:
    """Share of a bounded rule grid whose `size`-clip draw lands near `target`.

    `features` must cover the whole board, not just the successful clips: the
    difficulty index is a percentile rank and shifts if the population does.
    """
    by_rank = {feature.rank: feature for feature in features}
    difficulty = dict(
        zip([feature.rank for feature in features], difficulty_index(features))
    )
    pool = [score for score in scores if score.rank in by_rank]

    axes = [grid[name] for name in grid]
    names = list(grid)
    values: list[float] = []
    best: tuple[float, float, dict[str, Any]] | None = None
    tried = 0
    for combination in itertools.product(*axes):
        tried += 1
        rule = dict(zip(names, combination))
        low, high = rule["frames_range"]
        keep = [
            score
            for score in pool
            if difficulty[score.rank] <= rule["difficulty_max"]
            and low <= by_rank[score.rank].frames <= high
            and by_rank[score.rank].pel_z_min >= rule["pel_z_min_floor"]
            and by_rank[score.rank].root_speed_max <= rule["root_speed_max_cap"]
        ]
        if len(keep) < size:
            continue
        keep.sort(key=lambda score: score.rank)
        value = micro_mpjpe(random.Random(int(seed)).sample(keep, size))
        values.append(value)
        gap = abs(value - target)
        if best is None or gap < best[0]:
            best = (gap, value, rule)

    if best is None:
        raise ValueError(f"no rule of the grid retains {size} clips.")
    ordered = sorted(values)
    return RuleGridResult(
        rules_tried=tried,
        rules_valid=len(values),
        hits=sum(1 for value in values if abs(value - target) <= tolerance),
        tolerance=float(tolerance),
        target=float(target),
        best_value=best[1],
        best_rule=best[2],
        minimum=ordered[0],
        median=ordered[len(ordered) // 2],
        maximum=ordered[-1],
    )
