"""Render an ablation campaign as its paper sections, with per-arm status.

The campaign file is the single source of truth. Each arm carries `section`,
`section_label` and optionally `section_group`; this module groups by those and
prints one table per section.

Status per arm is derived from three independent signals, in this order:

- a scored evaluation row (`<arm>_seed<N>_clean_f<frames>.json`) makes an arm
  SCORED and supplies success rate and both MPJPE columns;
- the deepest tracker checkpoint on disk gives the frames reached;
- a Slurm state file, when supplied, distinguishes RUNNING from PENDING and
  surfaces FAILED.

A missing signal is reported as unknown rather than guessed: an arm with no
checkpoint and no Slurm row prints `-`, never `0.00B`.

    python -m imitation_experiments.reporting.ablation_sections \
        --campaign experiments/campaigns/2026-08-30-latent-star-v2/campaign.yaml \
        --eval-dir logs/latent_star_v2_eval \
        --slurm-states /tmp/states.tsv
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CHECKPOINT_RE = re.compile(r"model_step_(\d+)\.pt$")
_EVAL_RE = re.compile(
    r"^(?P<arm>.+)_seed(?P<seed>\d+)_(?P<row>[a-z0-9]+)_f(?P<frames>\d+)\.json$"
)

# Slurm states that mean the arm still has work queued or in flight. Anything
# else that appears is reported verbatim so a new state is never silently
# folded into "pending".
_ACTIVE_STATES = ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")


@dataclass(frozen=True)
class ArmRow:
    """One arm of one section, with whatever is currently known about it."""

    arm: str
    section: int
    label: str
    group: str = ""
    is_default: bool = False
    status: str = "unknown"
    frames: int | None = None
    success_rate: float | None = None
    mpjpe_local_mm: float | None = None
    mpjpe_global_mm: float | None = None

    def frames_text(self) -> str:
        if self.frames is None:
            return "-"
        return f"{self.frames / 1e9:.2f}B"

    def metric_text(self, value: float | None, digits: int) -> str:
        return "-" if value is None else f"{value:.{digits}f}"


@dataclass
class Section:
    number: int
    title: str
    rows: list[ArmRow] = field(default_factory=list)


SECTION_TITLES: Mapping[int, str] = {
    1: "Factorization target -- what the encoder is trained to predict",
    2: "Predictive architecture -- how the prediction is estimated",
    3: "Remaining design choices -- latent prior, encoder input, cadence",
}


def load_campaign_arms(campaign_path: Path) -> dict[str, dict[str, Any]]:
    campaign = yaml.safe_load(campaign_path.read_text())
    arms = campaign.get("arms") or {}
    if not arms:
        raise ValueError(f"no arms in {campaign_path}")
    return {name: (spec.get("vars") or {}) for name, spec in arms.items()}


def deepest_checkpoint_frames(tree_root: Path, arm: str, seed: int) -> int | None:
    """Largest `model_step_N` under an arm's tracker tree, or None."""
    tracker = tree_root / f"{arm}_seed{seed}" / "tracker"
    if not tracker.is_dir():
        return None
    best: int | None = None
    for path in tracker.rglob("model_step_*.pt"):
        match = _CHECKPOINT_RE.search(path.name)
        if not match:
            continue
        frames = int(match.group(1))
        if best is None or frames > best:
            best = frames
    return best


def scored_rows(
    eval_dir: Path, row: str = "clean", at_frames: int | None = None
) -> dict[str, dict[str, Any]]:
    """Scored row per arm, keyed by arm name.

    With `at_frames` the table is pinned to one budget, which is what makes it
    a comparison: arms train at different speeds, so taking each arm's deepest
    row silently compares a 2.6B checkpoint against a 2.0B one. Without it the
    deepest row wins, which is right for a progress view and wrong for a table.
    """
    if not eval_dir.is_dir():
        return {}
    best: dict[str, dict[str, Any]] = {}
    for path in sorted(eval_dir.glob(f"*_{row}_f*.json")):
        match = _EVAL_RE.match(path.name)
        if not match:
            continue
        arm = match.group("arm")
        frames = int(match.group("frames"))
        if at_frames is not None and frames != at_frames:
            continue
        previous = best.get(arm)
        if previous is not None and previous["frames"] >= frames:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        aggregate = payload.get("aggregate") or payload
        # MPJPE is success-only, so it lives under `successful_metrics` rather
        # than `aggregate`, as `{mean, count, num_successful_envs}`. Reading it
        # from `aggregate` silently yields None and prints an empty column.
        successful = payload.get("successful_metrics") or {}
        best[arm] = {
            "frames": frames,
            "success_rate": _as_float(aggregate.get("tracking_success_rate")),
            "mpjpe_local_mm": _metric_mean(successful.get("tracking_mpjpe_mm")),
            "mpjpe_global_mm": _metric_mean(successful.get("tracking_mpjpe_g_mm")),
        }
    return best


def _metric_mean(entry: Any) -> float | None:
    """Success-only metrics are `{mean, count, ...}`; plain scalars also pass."""
    if isinstance(entry, Mapping):
        return _as_float(entry.get("mean"))
    return _as_float(entry)


def parse_frames_file(text: str) -> dict[str, int]:
    """Map arm -> deepest frame count, from `<arm> <frames>` lines.

    Used when the checkpoint tree lives on the cluster instead of locally. A
    line with no frame count means the arm has no checkpoint yet, which stays
    absent rather than becoming zero.
    """
    frames: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        frames[parts[0]] = int(parts[1])
    return frames


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_slurm_states(text: str) -> dict[str, set[str]]:
    """Map arm -> set of Slurm states, from `<jobname><whitespace><state>` lines.

    Job names follow `<campaign>-<arm>-s<seed>-<stage>`; the stage suffix is
    stripped so every stage of an arm folds into one entry.
    """
    states: dict[str, set[str]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, state = parts[0], parts[1]
        # Greedy prefix, so the arm is the LAST segment before `-s<seed>-`.
        # A non-greedy prefix would hand campaign name fragments to the arm,
        # because campaign names contain dashes and arm names do not.
        match = re.match(r"^.+-(?P<arm>[^-]+)-s\d+-(?:pretrain|lowlevel\d+)$", name)
        if not match:
            continue
        states.setdefault(match.group("arm"), set()).add(state.upper())
    return states


def arm_status(states: set[str] | None, has_score: bool, frames: int | None) -> str:
    """Collapse a set of per-stage Slurm states into one word for the arm."""
    if has_score:
        return "SCORED"
    if states:
        if "RUNNING" in states:
            # No tracker checkpoint yet means the encoder pretrain is the
            # stage that is running, which is a different kind of "not done".
            return "training" if frames else "pretraining"
        inactive = states - set(_ACTIVE_STATES)
        if "PENDING" in states:
            return "pending"
        if inactive == {"COMPLETED"}:
            return "trained"
        if inactive:
            return "/".join(sorted(state.lower() for state in inactive))
    if frames:
        return "trained"
    return "unknown"


def build_sections(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    tree_root: Path | None = None,
    eval_dir: Path | None = None,
    slurm_states: Mapping[str, set[str]] | None = None,
    frames_by_arm: Mapping[str, int] | None = None,
    seed: int = 0,
    row: str = "clean",
    at_frames: int | None = None,
) -> list[Section]:
    scores = (
        scored_rows(eval_dir, row=row, at_frames=at_frames)
        if eval_dir is not None
        else {}
    )
    sections: dict[int, Section] = {}
    for arm, spec in arms.items():
        number = int(spec.get("section", 0))
        if number == 0:
            continue
        frames = (
            deepest_checkpoint_frames(tree_root, arm, seed)
            if tree_root is not None
            else (frames_by_arm or {}).get(arm)
        )
        score = scores.get(arm)
        if score is not None and score.get("frames"):
            frames = max(frames or 0, int(score["frames"]))
        section = sections.setdefault(
            number, Section(number, SECTION_TITLES.get(number, f"Section {number}"))
        )
        section.rows.append(
            ArmRow(
                arm=arm,
                section=number,
                label=str(spec.get("section_label", "")),
                group=str(spec.get("section_group", "")),
                is_default=arm == "hub",
                status=arm_status(
                    (slurm_states or {}).get(arm), score is not None, frames
                ),
                frames=frames,
                success_rate=(score or {}).get("success_rate"),
                mpjpe_local_mm=(score or {}).get("mpjpe_local_mm"),
                mpjpe_global_mm=(score or {}).get("mpjpe_global_mm"),
            )
        )
    for section in sections.values():
        section.rows.sort(key=lambda r: (not r.is_default, r.group, r.arm))
    return [sections[key] for key in sorted(sections)]


def render(sections: Sequence[Section]) -> str:
    lines: list[str] = []
    for section in sections:
        lines.append(f"## Section {section.number} -- {section.title}")
        lines.append("")
        header = "| arm | what changes | status | frames | SR | MPJPE-L | MPJPE-G |"
        lines.append(header)
        lines.append("|---|---|---|---:|---:|---:|---:|")
        group = None
        for r in section.rows:
            if r.group and r.group != group:
                group = r.group
                lines.append(f"| *{group}* | | | | | | |")
            name = f"**{r.arm}**" if r.is_default else r.arm
            label = r.label + (" (default)" if r.is_default else "")
            lines.append(
                f"| {name} | {label} | {r.status} | {r.frames_text()} "
                f"| {r.metric_text(r.success_rate, 4)} "
                f"| {r.metric_text(r.mpjpe_local_mm, 2)} "
                f"| {r.metric_text(r.mpjpe_global_mm, 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def summarize(sections: Iterable[Section]) -> str:
    counts: dict[str, int] = {}
    total = 0
    for section in sections:
        for row in section.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
            total += 1
    parts = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    return f"{total} arms: {parts}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--tree-root", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument(
        "--slurm-states",
        type=Path,
        default=None,
        help="file of `<jobname> <state>` lines, e.g. from squeue/sacct",
    )
    parser.add_argument(
        "--frames-file",
        type=Path,
        default=None,
        help="file of `<arm> <frames>` lines, for a checkpoint tree on the cluster",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row", default="clean")
    parser.add_argument(
        "--at-frames",
        type=int,
        default=None,
        help="pin every scored column to this exact checkpoint, e.g. 2000486400",
    )
    args = parser.parse_args(argv)

    states = (
        parse_slurm_states(args.slurm_states.read_text())
        if args.slurm_states is not None and args.slurm_states.is_file()
        else {}
    )
    frames_by_arm = (
        parse_frames_file(args.frames_file.read_text())
        if args.frames_file is not None and args.frames_file.is_file()
        else {}
    )
    sections = build_sections(
        load_campaign_arms(args.campaign),
        tree_root=args.tree_root,
        eval_dir=args.eval_dir,
        slurm_states=states,
        frames_by_arm=frames_by_arm,
        seed=args.seed,
        row=args.row,
        at_frames=args.at_frames,
    )
    print(render(sections))
    print(summarize(sections))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
