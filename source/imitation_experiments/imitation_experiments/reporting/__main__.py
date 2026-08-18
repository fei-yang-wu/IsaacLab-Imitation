"""CLI: ``python -m imitation_experiments.reporting --spec ... --out ...``."""

from __future__ import annotations

import argparse
from pathlib import Path

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.reporting.build import build_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "experiments/paper/conf/report.yaml",
        help="Report spec YAML naming the curated runs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "logs/report/index.html",
        help="Output HTML path; the reduced records land beside it as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_report(args.spec, args.out)
    print(f"[report] {result.record_count} runs -> {result.html_path}")
    print(f"[report] reduced records -> {result.data_path}")


if __name__ == "__main__":
    main()
