#!/usr/bin/env bash
# Thin wrapper over the control plane. Plans (validates + preflights + freezes)
# and prints the submit command; it never submits by itself.
#
#   ./submit.sh <arm> <seed> [--only-stage lowlevel] [--set vars.frame_cap=2000000000] ...
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
ARM="${1:?usage: submit.sh <arm> <seed> [plan args...]}"
SEED="${2:?usage: submit.sh <arm> <seed> [plan args...]}"
shift 2
cd "${REPO_ROOT}"
exec pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign "${CAMPAIGN_DIR}/campaign.yaml" --arm "${ARM}" --seed "${SEED}" "$@"
