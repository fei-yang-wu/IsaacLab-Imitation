#!/usr/bin/env bash
# Plan one arm through the cluster control plane. This script never submits.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
ARM="${1:?usage: submit.sh <arm> <seed> [plan args...]}"
SEED="${2:?usage: submit.sh <arm> <seed> [plan args...]}"
shift 2
cd "${REPO_ROOT}"
exec pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign "${CAMPAIGN_DIR}/campaign.yaml" --arm "${ARM}" --seed "${SEED}" "$@"
