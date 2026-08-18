"""Resolve a report spec against the evaluation artifacts and write the page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.reporting.records import EvalRecord, load_summary
from imitation_experiments.reporting.render import render_data, render_html
from imitation_experiments.reporting.spec import load_spec


@dataclass(frozen=True)
class BuildResult:
    """Where the build wrote, and what it read to get there."""

    html_path: Path
    data_path: Path
    record_count: int
    missing: tuple[str, ...]


def _git_describe(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def build_report(
    spec_path: Path,
    output_path: Path,
    *,
    repo_root: Path | None = None,
) -> BuildResult:
    """Build the results page from ``spec_path`` into ``output_path``.

    Every run the spec references must have a readable ``summary.json``. A
    missing one raises rather than rendering a table that silently lost a row.
    """
    root = Path(repo_root or REPO_ROOT)
    spec = load_spec(Path(spec_path), root)

    records: dict[str, EvalRecord] = {}
    for run in spec.referenced_runs():
        records[run] = load_summary(spec.summary_path(run), root)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(spec, records, git_describe=_git_describe(root)),
        encoding="utf-8",
    )

    data_path = output_path.with_suffix(".json")
    data_path.write_text(render_data(spec, records), encoding="utf-8")

    return BuildResult(
        html_path=output_path,
        data_path=data_path,
        record_count=len(records),
        missing=(),
    )
