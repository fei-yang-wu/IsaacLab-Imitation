#!/usr/bin/env bash
# Build the interactive results page from the evaluation artifacts on disk.
#
# Run from the repository root:
#
#   ./experiments/paper/build_report.sh
#   ./experiments/paper/build_report.sh --spec experiments/paper/conf/report.yaml \
#       --out logs/report/index.html
#
# The curated run list is `experiments/paper/conf/report.yaml`. The page is a
# generated artifact and is written under `logs/`, so it is never committed.
# Open the result with any browser; it needs no server and fetches nothing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SPEC="${REPO_ROOT}/experiments/paper/conf/report.yaml"
OUT="${REPO_ROOT}/logs/report/index.html"

while [ $# -gt 0 ]; do
  case "$1" in
    --spec) SPEC="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

pixi run python -m imitation_experiments.reporting --spec "${SPEC}" --out "${OUT}"
echo "[report] open file://${OUT}"
