"""Build the interactive results page from machine-readable evaluation output.

Nothing in this package invents a number. Every value on the rendered page is
reduced from an evaluation ``summary.json`` and carries the path, the tracker
checkpoint digest, and the protocol flags that produced it, so a reader can
check any cell against the artifact that backs it.
"""

from __future__ import annotations

from imitation_experiments.reporting.records import (
    EvalRecord,
    MotionScore,
    load_summary,
)
from imitation_experiments.reporting.spec import ReportSpec, load_spec
from imitation_experiments.reporting.build import build_report

__all__ = [
    "EvalRecord",
    "MotionScore",
    "ReportSpec",
    "build_report",
    "load_spec",
    "load_summary",
]
