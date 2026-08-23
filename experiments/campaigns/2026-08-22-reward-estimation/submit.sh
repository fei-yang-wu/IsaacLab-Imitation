#!/usr/bin/env bash
# Thin wrapper over the control plane. Plans (validates + preflights +
# freezes) and prints the submit command; it never submits by itself.
#
#   First submission (segment 1 + insurance segment 2):
#     ./submit.sh 0
#   Relaunch after a walltime end (repeat until cumulative_env_frames = 10B):
#     ./submit.sh 0 --only-stage lowlevel_resume
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
SEED="${1:?usage: submit.sh <seed> [plan args...]}"
shift
exec pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign "${CAMPAIGN_DIR}/campaign.yaml" \
    --arm irl_explicit_root_qpos --seed "${SEED}" "$@"
