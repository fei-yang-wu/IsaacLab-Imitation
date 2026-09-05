#!/usr/bin/env bash
# Score combo-50b's newest checkpoint mid-flight, on the standard board.
#
#   ./submit_live_eval.sh            # submit if the newest checkpoint is unscored
#   DRY_RUN=1 ./submit_live_eval.sh  # print what it would do
#
# Two mechanics this handles so a cron caller does not have to:
#
#  * The training tree grows a new run directory per resume, which
#    `score_tree.milestone_checkpoints` refuses (`AmbiguousTree`). The newest
#    `model_step_*.pt` is therefore relinked into a milestone-layout tree,
#    `/data/combo_50b_live/combo_seed0/tracker/f<frames>/models/`, and the
#    eval reads that with `--final_only`.
#  * `cluster submit` refuses a working tree that changed since the plan was
#    sealed. Unrelated work in progress is stashed for the submit and popped
#    back on exit, including on failure.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
REMOTE="${REMOTE:-ice}"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
TRAIN_TREE="${TRAIN_TREE:-${DATA}/combo_50b/combo_seed0/tracker}"
LIVE_ROOT="${LIVE_ROOT:-${DATA}/combo_50b_live/combo_seed0/tracker}"
EVAL_DIR="${EVAL_DIR:-${DATA}/eval/latest_eval}"
EVAL_CAMPAIGN="${EVAL_CAMPAIGN:-experiments/campaigns/2026-09-02-latest-eval/campaign.yaml}"
ARM="${ARM:-combo50b_live}"
# The eval row is named by `vars.eval_arm` (combo50b), NOT by the campaign arm
# name, so a row of this chain can never collide with the flat-mix `combo`
# run's rows at the same frame count.
EVAL_ARM="${EVAL_ARM:-combo50b}"
DRY_RUN="${DRY_RUN:-0}"

newest="$(timeout 90 ssh "${REMOTE}" "ls ${TRAIN_TREE}/*/models/model_step_*.pt 2>/dev/null | sed 's|.*/||' | sed -E 's/model_step_([0-9]+)\.pt/\1/' | sort -n | tail -1")"
[ -n "${newest}" ] || { echo "[LIVE-EVAL] no checkpoint under ${TRAIN_TREE}"; exit 0; }
if timeout 90 ssh "${REMOTE}" "test -s ${EVAL_DIR}/${EVAL_ARM}_seed0_clean_f${newest}.json"; then
    echo "[LIVE-EVAL] f${newest} already scored; nothing to do"
    exit 0
fi
echo "[LIVE-EVAL] newest checkpoint: f${newest}"
[ "${DRY_RUN}" = "1" ] || timeout 90 ssh "${REMOTE}" "
    set -e
    src=\$(ls ${TRAIN_TREE}/*/models/model_step_${newest}.pt | head -1)
    dst=${LIVE_ROOT}/f${newest}/models
    mkdir -p \$dst
    ln -sfn \"\$(realpath --relative-to=\$dst \$src)\" \$dst/model_step_${newest}.pt"

# Keep unrelated working-tree changes out of the submitted archive.
STASHED=0
restore() { [ "${STASHED}" = "1" ] && git stash pop -q || true; }
trap restore EXIT
if [ -n "$(git status --porcelain)" ]; then
    git stash push -q -u -m "live-eval submit $(date -Iseconds)" && STASHED=1
fi
OUT="$(pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign "${EVAL_CAMPAIGN}" --arm "${ARM}" --seed 0 \
    --set vars.tree_root=/data/combo_50b_live 2>&1)" || { echo "${OUT}" | tail -20; exit 1; }
PLAN="$(echo "${OUT}" | grep 'local dir:' | awk '{print $NF}')"
SHA="$(echo "${OUT}" | grep '^PLAN_SHA=' | cut -d= -f2)"
[ -n "${PLAN}" ] || { echo "[LIVE-EVAL] plan failed"; echo "${OUT}" | tail -20; exit 1; }
if [ "${DRY_RUN}" = "1" ]; then
    echo "[LIVE-EVAL] would submit ${PLAN} (${SHA})"
    exit 0
fi
pixi run python -m imitation_experiments.pipeline.cluster submit \
    --plan "${PLAN}" --confirm "${SHA}" | grep -E 'job |refusing|ERROR'
echo "[LIVE-EVAL] scored row will land at ${EVAL_DIR}/${EVAL_ARM}_seed0_clean_f${newest}.json"
