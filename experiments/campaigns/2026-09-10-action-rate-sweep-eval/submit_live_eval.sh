#!/usr/bin/env bash
# Score the newest checkpoint of each action-rate sweep arm (rate02/03/04),
# skipping any checkpoint another session already scored.
#
# The sweep's own session scores every 500M checkpoint into per-checkpoint
# pinned trees under
#   /data/eval/combo_<arm>_<frames>_<date>/testbed4096/<arm>_seed0_clean_f<frames>.json
# so this launcher gates on BOTH that layout and the latest_eval one before
# it submits. Everything else is the combo-50b launcher with the tree swapped.
#
#   ./submit_live_eval.sh              # all three arms
#   ARMS="rate04" ./submit_live_eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REMOTE="${REMOTE:-ice}"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ARMS="${ARMS:-rate02 rate03 rate04}"
export DATA REMOTE

for short in ${ARMS}; do
    arm="action01${short}"
    newest="$(timeout 90 ssh "${REMOTE}" "ls ${DATA}/combo_action01_rate_sweep/${arm}_seed0/tracker/*/models/model_step_*.pt 2>/dev/null | sed 's|.*/||' | sed -E 's/model_step_([0-9]+)\.pt/\1/' | sort -n | tail -1")"
    [ -n "${newest}" ] || { echo "[${arm}] no checkpoint"; continue; }
    if timeout 90 ssh "${REMOTE}" "ls ${DATA}/eval/combo_${arm}_${newest}_*/testbed4096/${arm}_seed0_clean_f${newest}.json ${DATA}/eval/latest_eval/${arm}_seed0_clean_f${newest}.json 2>/dev/null | grep -q ."; then
        echo "[${arm}] f${newest} already scored (by either session); nothing to do"
        continue
    fi
    echo "=== ${arm}: newest f${newest} unscored, submitting"
    TRAIN_TREE="${DATA}/combo_action01_rate_sweep/${arm}_seed0/tracker" \
    LIVE_ROOT="${DATA}/action01_rate_sweep_live/${arm}_seed0/tracker" \
    LIVE_TREE_ROOT="/data/action01_rate_sweep_live" \
    ARM="${short}_live" EVAL_ARM="${arm}" \
        "${REPO_ROOT}/experiments/campaigns/2026-09-03-combo-50b/submit_live_eval.sh"
done
