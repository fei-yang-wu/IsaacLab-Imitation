"""Curation layer for the results page.

The evaluation artifacts say what a run measured. They do not say which runs
belong in the paper table, which arm is the control, or what the comparison was
supposed to isolate. That judgement is human, so it lives in a YAML spec next
to the launcher instead of being guessed from directory names.

Some run attributes are absent from older artifacts. Temporal ensembling was the
clearest case: before 2026-08-16 it survived only in the evaluation label, so the
spec carried it as an explicit field rather than parsing it back out of a string.
The evaluation now writes `temporal_ensemble`, its decay, the ODE-step count, the
samples per publication, and the consumed slot count into
`metadata.gr00t_planner`. Where a run has them the artifact wins over anything
declared here (`render.arm_attributes`); `attributes` stays for runs evaluated
before that date and for labels the evaluation genuinely does not know, such as
training row counts.

Every run reference must resolve. A missing summary file raises instead of
dropping a row, because a table that quietly renders four arms out of five
reads as a complete comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunRef:
    """A curated row: one evaluation run plus the labels a reader needs."""

    id: str
    run: str
    name: str
    attributes: dict[str, str] = field(default_factory=dict)
    note: str | None = None
    ceiling: str | None = None


@dataclass(frozen=True)
class HistoryPoint:
    """One entry in the moving best-so-far low-level record."""

    date: str
    run: str
    name: str
    note: str | None = None


@dataclass(frozen=True)
class Ablation:
    """A grouped comparison with a named control and an isolated variable."""

    id: str
    title: str
    variable: str
    question: str
    baseline: str
    rows: tuple[RunRef, ...]
    verdict: str | None = None
    status: str = "preliminary"


@dataclass(frozen=True)
class ReportSpec:
    """Everything the renderer needs, resolved against the repository."""

    title: str
    subtitle: str
    updated: str
    logs_root: Path
    noise_band_pct: float
    low_level_headline: RunRef
    low_level_history: tuple[HistoryPoint, ...]
    low_level_rows: tuple[RunRef, ...]
    planner_baseline: str
    planner_rows: tuple[RunRef, ...]
    ablations: tuple[Ablation, ...]
    method_cards: tuple[str, ...]
    caveats: tuple[str, ...]

    def summary_path(self, run: str) -> Path:
        """Return the ``summary.json`` for a spec run reference.

        Archived runs live on the HDD pool behind a symlink in ``logs/``; the
        resolve below follows it so a moved directory still reports.
        """
        candidate = (self.logs_root / run / "summary.json").resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Report spec references {run!r} but {candidate} does not exist. "
                "Fix the spec or re-run the evaluation; the page never drops a "
                "referenced row."
            )
        return candidate

    def referenced_runs(self) -> list[str]:
        """Every run reference in the spec, de-duplicated, in declaration order."""
        seen: dict[str, None] = {}
        for ref in (self.low_level_headline, *self.low_level_rows, *self.planner_rows):
            seen.setdefault(ref.run, None)
            if ref.ceiling:
                seen.setdefault(ref.ceiling, None)
        for point in self.low_level_history:
            seen.setdefault(point.run, None)
        for ablation in self.ablations:
            for ref in ablation.rows:
                seen.setdefault(ref.run, None)
                if ref.ceiling:
                    seen.setdefault(ref.ceiling, None)
        return list(seen)


def _run_ref(payload: dict[str, Any]) -> RunRef:
    missing = {"id", "run", "name"} - set(payload)
    if missing:
        raise ValueError(f"Run entry is missing {sorted(missing)}: {payload}")
    return RunRef(
        id=str(payload["id"]),
        run=str(payload["run"]),
        name=str(payload["name"]),
        attributes={
            str(k): str(v) for k, v in (payload.get("attributes") or {}).items()
        },
        note=payload.get("note"),
        ceiling=payload.get("ceiling"),
    )


def load_spec(spec_path: Path, repo_root: Path) -> ReportSpec:
    """Parse the report spec YAML and validate its structure."""
    with Path(spec_path).open() as handle:
        raw = yaml.safe_load(handle) or {}

    meta = raw.get("meta") or {}
    low_level = raw.get("low_level") or {}
    planner = raw.get("planner") or {}

    ablations: list[Ablation] = []
    for entry in raw.get("ablations") or []:
        rows = tuple(_run_ref(row) for row in entry.get("rows") or [])
        if not rows:
            raise ValueError(f"Ablation {entry.get('id')!r} declares no rows.")
        baseline = str(entry["baseline"])
        if baseline not in {row.id for row in rows}:
            raise ValueError(
                f"Ablation {entry.get('id')!r} names baseline {baseline!r}, "
                "which is not one of its rows."
            )
        ablations.append(
            Ablation(
                id=str(entry["id"]),
                title=str(entry["title"]),
                variable=str(entry.get("variable", "unspecified")),
                question=str(entry.get("question", "")),
                baseline=baseline,
                rows=rows,
                verdict=entry.get("verdict"),
                status=str(entry.get("status", "preliminary")),
            )
        )

    spec = ReportSpec(
        title=str(meta.get("title", "Results")),
        subtitle=str(meta.get("subtitle", "")),
        updated=str(meta.get("updated", "")),
        logs_root=Path(repo_root) / str(meta.get("logs_root", "logs")),
        noise_band_pct=float(meta.get("noise_band_pct", 15.0)),
        low_level_headline=_run_ref(low_level["headline"]),
        low_level_history=tuple(
            HistoryPoint(
                date=str(point["date"]),
                run=str(point["run"]),
                name=str(point["name"]),
                note=point.get("note"),
            )
            for point in low_level.get("history") or []
        ),
        low_level_rows=tuple(_run_ref(row) for row in low_level.get("rows") or []),
        planner_baseline=str(planner.get("baseline", "")),
        planner_rows=tuple(_run_ref(row) for row in planner.get("rows") or []),
        ablations=tuple(ablations),
        method_cards=tuple(str(name) for name in raw.get("method_cards") or []),
        caveats=tuple(str(text) for text in raw.get("caveats") or []),
    )

    row_ids = {row.id for row in spec.planner_rows}
    if spec.planner_baseline and spec.planner_baseline not in row_ids:
        raise ValueError(
            f"planner.baseline {spec.planner_baseline!r} is not one of the planner rows."
        )
    return spec
