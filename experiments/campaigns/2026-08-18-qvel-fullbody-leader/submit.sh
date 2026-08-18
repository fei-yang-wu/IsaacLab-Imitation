#!/usr/bin/env bash
# Thin wrapper over the control plane. Plans (validates + preflights +
# freezes) and prints the submit command; it never submits by itself.
#
#   First submission:
#     ./submit.sh --only-stage lowlevel
#   Relaunch after a walltime end (repeat until cumulative_env_frames = 10B):
#     ./submit.sh --only-stage lowlevel_resume
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
exec pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign "${CAMPAIGN_DIR}/campaign.yaml" \
    --arm cont_det_ln_hold1_fullbody --seed 0 "$@"
