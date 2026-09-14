#!/usr/bin/env bash
# Score the newest checkpoint of the affine PoE combo-50B tracker mid-flight
# on the standard board (row latest_eval/affine_poe_seed0_clean_f<frames>.json).
# The combo-50b launcher with the tree swapped in; the tree lives on ice-shared.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REMOTE="${REMOTE:-ice}"
ROOT=/storage/ice-shared/vip-vwt/scratch-fwu91
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
newest="$(timeout 90 ssh "${REMOTE}" "ls ${ROOT}/affine_poe_combo50b/affine_poe_seed0/tracker/*/models/model_step_*.pt 2>/dev/null | sed 's|.*/||' | sed -E 's/model_step_([0-9]+)\.pt/\1/' | sort -n | tail -1")"
[ -n "${newest}" ] || { echo "[affine_poe] no checkpoint yet"; exit 0; }
if timeout 90 ssh "${REMOTE}" "test -s ${DATA}/eval/latest_eval/affine_poe_seed0_clean_f${newest}.json"; then
    echo "[affine_poe] f${newest} already scored; nothing to do"; exit 0
fi
if timeout 90 ssh "${REMOTE}" "squeue -u fwu91 -h -o '%j' | grep -q 'latest-eval-affine_poe_live'"; then
    echo "[affine_poe] an eval is already in flight; skipping"; exit 0
fi
TRAIN_TREE="${ROOT}/affine_poe_combo50b/affine_poe_seed0/tracker" \
LIVE_ROOT="${ROOT}/affine_poe_combo50b_live/affine_poe_seed0/tracker" \
LIVE_TREE_ROOT="${ROOT}/affine_poe_combo50b_live" \
ARM="affine_poe_live" EVAL_ARM="affine_poe" \
    "${REPO_ROOT}/experiments/campaigns/2026-09-03-combo-50b/submit_live_eval.sh"
