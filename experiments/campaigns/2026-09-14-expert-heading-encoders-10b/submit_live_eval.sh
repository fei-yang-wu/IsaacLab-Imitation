#!/usr/bin/env bash
# Score the newest tracker checkpoint of each expert-heading encoder arm
# (eh, eh_ee) on the standard board, skipping checkpoints already scored or
# in flight.
#
# Trees: /data/expert_heading_encoders_10b/<arm>_seed0/tracker. The newest
# model_step_*.pt is relinked into a milestone-layout tree under
# /data/expert_heading_encoders_10b_live/ehenc_<arm>_seed0/tracker/f<frames>/
# and scored by the latest-eval arm `ehenc_<arm>_live`, whose row lands as
# latest_eval/ehenc_<arm>_seed0_clean_f<frames>.json.
#
# The plan and submit run from a clean detached worktree of HEAD (with the
# submodules at their recorded commits) instead of `git stash`: the scorer
# needs RLOpt a0add23 (the 50-wide window layout), and the main tree's RLOpt
# working copy may carry another session's uncommitted edits, which a
# top-level stash never touches.
#
#   ./submit_live_eval.sh              # both arms
#   ARMS="eh" ./submit_live_eval.sh
#   DRY_RUN=1 ./submit_live_eval.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REMOTE="${REMOTE:-ice}"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
EVAL_DIR="${DATA}/eval/latest_eval"
EVAL_CAMPAIGN="experiments/campaigns/2026-09-02-latest-eval/campaign.yaml"
ARMS="${ARMS:-eh eh_ee}"
DRY_RUN="${DRY_RUN:-0}"
PY="${REPO_ROOT}/.pixi/envs/default/bin/python"

WORKTREE=""
cleanup() {
    [ -n "${WORKTREE}" ] && git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" 2>/dev/null || true
}
trap cleanup EXIT
prepare_worktree() {
    [ -n "${WORKTREE}" ] && return 0
    WORKTREE="${REPO_ROOT}/.claude/worktrees/live-eval-ehenc-$$"
    git -C "${REPO_ROOT}" worktree add --detach -q "${WORKTREE}" HEAD
    git -C "${WORKTREE}" submodule update --init -q RLOpt ImitationLearningTools external/Embodied-Control external/Isaac-GR00T
}

for arm in ${ARMS}; do
    tree="${DATA}/expert_heading_encoders_10b/${arm}_seed0/tracker"
    live_root="${DATA}/expert_heading_encoders_10b_live/ehenc_${arm}_seed0/tracker"
    newest="$(timeout 90 ssh "${REMOTE}" "ls ${tree}/*/models/model_step_*.pt 2>/dev/null | sed 's|.*/||' | sed -E 's/model_step_([0-9]+)\.pt/\1/' | sort -n | tail -1")"
    [ -n "${newest}" ] || { echo "[ehenc_${arm}] no checkpoint yet"; continue; }
    if timeout 90 ssh "${REMOTE}" "test -s ${EVAL_DIR}/ehenc_${arm}_seed0_clean_f${newest}.json"; then
        echo "[ehenc_${arm}] f${newest} already scored; nothing to do"; continue
    fi
    if timeout 90 ssh "${REMOTE}" "squeue -u fwu91 -h -o '%j' | grep -q 'latest-eval-ehenc_${arm}_live'"; then
        echo "[ehenc_${arm}] an eval is already in flight; skipping"; continue
    fi
    echo "=== ehenc_${arm}: newest f${newest} unscored"
    if [ "${DRY_RUN}" = "1" ]; then echo "[ehenc_${arm}] would relink and submit"; continue; fi
    timeout 90 ssh "${REMOTE}" "
        set -e
        src=\$(ls ${tree}/*/models/model_step_${newest}.pt | head -1)
        dst=${live_root}/f${newest}/models
        mkdir -p \$dst
        ln -sfn \"\$(realpath --relative-to=\$dst \$src)\" \$dst/model_step_${newest}.pt"
    prepare_worktree
    out="$(cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}/source/imitation_experiments" timeout 900 "${PY}" -m imitation_experiments.pipeline.cluster plan \
        --campaign "${EVAL_CAMPAIGN}" --arm "ehenc_${arm}_live" --seed 0 \
        --out-root "${REPO_ROOT}/logs/cluster_control" 2>&1)" || { echo "${out}" | tail -20; continue; }
    plan="$(echo "${out}" | grep 'local dir:' | awk '{print $NF}')"
    sha="$(echo "${out}" | grep '^PLAN_SHA=' | cut -d= -f2)"
    [ -n "${plan}" ] || { echo "[ehenc_${arm}] plan failed"; echo "${out}" | tail -20; continue; }
    (cd "${WORKTREE}" && PYTHONPATH="${WORKTREE}/source/imitation_experiments" timeout 1500 "${PY}" -m imitation_experiments.pipeline.cluster submit \
        --plan "${plan}" --confirm "${sha}" | grep -E 'job |refusing|ERROR|drift')
    echo "[ehenc_${arm}] row will land at ${EVAL_DIR}/ehenc_${arm}_seed0_clean_f${newest}.json"
done
