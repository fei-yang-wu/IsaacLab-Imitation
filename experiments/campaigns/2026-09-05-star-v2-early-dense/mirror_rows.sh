#!/usr/bin/env bash
# Mirror the scored early-dense rows from ICE to the workstation.
# Rows only (small JSONs), never the checkpoint trees.
#
#   ./mirror_rows.sh
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
REMOTE_HOST="${REMOTE_HOST:-ice}"
REMOTE_ROWS="${REMOTE_ROWS:-scratch/Research/IsaacLab/data/eval/star_v2_early_curves/}"
LOCAL_ROWS="${LOCAL_ROWS:-${REPO_ROOT}/logs/latent_star_v2_early_curves/}"
mkdir -p "${LOCAL_ROWS}"
rsync -av --include='*.json' --exclude='*' "${REMOTE_HOST}:${REMOTE_ROWS}" "${LOCAL_ROWS}"
printf '%s rows local\n' "$(ls "${LOCAL_ROWS}"/*.json 2>/dev/null | wc -l)"
