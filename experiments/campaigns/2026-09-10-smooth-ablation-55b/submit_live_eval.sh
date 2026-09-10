#!/usr/bin/env bash
# Score the newest checkpoint of each smoothness-ablation arm (a2/a3/c1) on
# the standard board, skipping checkpoints already scored or in flight.
#
# The trees are /data/smooth_ablation_55b/<arm>_seed0/tracker; rows land as
# latest_eval/smooth_<arm>_seed0_clean_f<frames>.json. Everything else is the
# combo-50b launcher with the tree swapped in.
#
#   ./submit_live_eval.sh              # all three arms
#   ARMS="c1" ./submit_live_eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REMOTE="${REMOTE:-ice}"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ARMS="${ARMS:-a2 a3 c1}"
export DATA REMOTE

for arm in ${ARMS}; do
    newest="$(timeout 90 ssh "${REMOTE}" "ls ${DATA}/smooth_ablation_55b/${arm}_seed0/tracker/*/models/model_step_*.pt 2>/dev/null | sed 's|.*/||' | sed -E 's/model_step_([0-9]+)\.pt/\1/' | sort -n | tail -1")"
    [ -n "${newest}" ] || { echo "[smooth_${arm}] no checkpoint yet"; continue; }
    if timeout 90 ssh "${REMOTE}" "test -s ${DATA}/eval/latest_eval/smooth_${arm}_seed0_clean_f${newest}.json"; then
        echo "[smooth_${arm}] f${newest} already scored; nothing to do"; continue
    fi
    if timeout 90 ssh "${REMOTE}" "squeue -u fwu91 -h -o '%j' | grep -q 'latest-eval-smooth_${arm}_live'"; then
        echo "[smooth_${arm}] an eval is already in flight; skipping"; continue
    fi
    echo "=== smooth_${arm}: newest f${newest} unscored, submitting"
    TRAIN_TREE="${DATA}/smooth_ablation_55b/${arm}_seed0/tracker" \
    LIVE_ROOT="${DATA}/smooth_ablation_55b_live/smooth_${arm}_seed0/tracker" \
    LIVE_TREE_ROOT="/data/smooth_ablation_55b_live" \
    ARM="smooth_${arm}_live" EVAL_ARM="smooth_${arm}" \
        "${REPO_ROOT}/experiments/campaigns/2026-09-03-combo-50b/submit_live_eval.sh"
done
