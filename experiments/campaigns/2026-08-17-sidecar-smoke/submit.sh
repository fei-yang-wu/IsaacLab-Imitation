#!/usr/bin/env bash
# Plan the sidecar smoke run. Plans only; it never submits by itself.
#
#   ./submit.sh [seed] [plan args...]
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
SEED="${1:-0}"
[ $# -gt 0 ] && shift
cd "${REPO_ROOT}"
exec pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign "${CAMPAIGN_DIR}/campaign.yaml" --arm fsq64_hold10 --seed "${SEED}" "$@"
