#!/usr/bin/env bash
# Score every unscored 500M checkpoint of the 64-D merged-head family on the
# standard board, mid-flight: control `z64_merged`, `z64_merged_noreg`, `z64_poe`.
#
#   ./submit_live_eval.sh              # all three arms
#   ARMS="z64_poe" ./submit_live_eval.sh
#   DRY_RUN=1 ./submit_live_eval.sh
#
# Mechanics, per arm:
#  * every `model_step_*.pt` of the training tree is relinked into a
#    milestone-layout live tree (`<live_root>/<arm>_seed0/tracker/f<N>/models/`),
#    so a resume's second run directory cannot raise `AmbiguousTree`. All live
#    trees sit on ice-shared. A symlink target is the CANONICAL absolute path
#    when that path is under /storage/ice-shared (identical inside the
#    container), and a relative path otherwise. The first pass (2026-09-12)
#    used `realpath --relative-to` for everything: the control's scratch tree is
#    itself a symlink into the ice-shared archive, so the canonical relative
#    path climbed twelve levels to `/storage` and, inside the container where
#    /data is one level deep, landed on a non-existent `/ice-shared/...`; the
#    job saw an empty tree and exited "nothing to score".
#  * if every checkpoint already has a row in /data/eval/latest_eval, or the
#    arm's curve job is already queued, nothing is submitted.
#  * `cluster submit` refuses a working tree that changed since the plan was
#    sealed; unrelated work in progress is stashed for the submit and popped on
#    exit, as in 2026-09-03-combo-50b/submit_live_eval.sh.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
REMOTE="${REMOTE:-ice}"
DATA="${DATA:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
SHARED="${SHARED:-/storage/ice-shared/vip-vwt/scratch-fwu91}"
EVAL_DIR="${EVAL_DIR:-${DATA}/eval/latest_eval}"
EVAL_CAMPAIGN="${EVAL_CAMPAIGN:-experiments/campaigns/2026-09-02-latest-eval/campaign.yaml}"
ARMS="${ARMS:-z64_merged z64_merged_noreg z64_poe z64_poe_base poe_proj256 poe_proj256_base poe_z256 z64_poe_reg z64_poe_base_reg poe_proj256_reg poe_proj256_base_reg enc_hist p5_affine_ctrl}"
DRY_RUN="${DRY_RUN:-0}"

# arm -> training tree (login-node path) and live root (login-node path)
train_tree() {
    case "$1" in
        # On personal scratch, not ice-shared: trained before the move.
        z64_merged|enc_hist) echo "${DATA}/latent64_probe_10b/$1_seed0/tracker" ;;
        *) echo "${SHARED}/latent64_probe_10b/$1_seed0/tracker" ;;
    esac
}
# The live tree must sit on the same filesystem as the checkpoints it links
# when those checkpoints are NOT under /storage/ice-shared: the relink writes a
# relative target in that case, and a relative path from ice-shared to personal
# scratch climbs out of the container's binds. `z64_merged` is exempt because
# its scratch tree is a symlink INTO the ice-shared archive, so its targets are
# absolute; `enc_hist` still has its newest checkpoints on scratch proper.
live_root() {
    case "$1" in
        enc_hist) echo "${DATA}/latent64_probe_live/$1_seed0/tracker" ;;
        *) echo "${SHARED}/latent64_probe_live/$1_seed0/tracker" ;;
    esac
}

STASHED=0
restore() { [ "${STASHED}" = "1" ] && git stash pop -q || true; }
trap restore EXIT

for arm in ${ARMS}; do
    tree="$(train_tree "${arm}")"; live="$(live_root "${arm}")"
    mapfile -t frames < <(timeout 90 ssh "${REMOTE}" "ls ${tree}/*/models/model_step_*.pt 2>/dev/null | sed -E 's/.*model_step_([0-9]+)\.pt/\1/' | sort -n")
    [ "${#frames[@]}" -gt 0 ] || { echo "[${arm}] no checkpoint yet"; continue; }
    mapfile -t scored < <(timeout 90 ssh "${REMOTE}" "ls ${EVAL_DIR}/${arm}_seed0_clean_f*.json 2>/dev/null | sed -E 's/.*_f([0-9]+)\.json/\1/' | sort -n")
    unscored=()
    for f in "${frames[@]}"; do
        printf '%s\n' "${scored[@]:-}" | grep -qx "${f}" || unscored+=("${f}")
    done
    if [ "${#unscored[@]}" -eq 0 ]; then echo "[${arm}] ${#frames[@]} checkpoints, all scored"; continue; fi
    if timeout 90 ssh "${REMOTE}" 'squeue -u $USER -h -o "%j"' | grep -q "^latest-eval-${arm}_curve-"; then
        echo "[${arm}] curve job already queued; ${#unscored[@]} unscored"; continue
    fi
    echo "[${arm}] ${#frames[@]} checkpoints, ${#unscored[@]} unscored: $(printf '%s ' "${unscored[@]}" | sed -E 's/([0-9]+)0{8}[0-9]* ?/\1.xB /g' | cut -c1-80)..."
    [ "${DRY_RUN}" = "1" ] && continue
    # Relink every checkpoint (idempotent).
    timeout 120 ssh "${REMOTE}" "
        set -e
        for src in ${tree}/*/models/model_step_*.pt; do
            n=\$(basename \$src | sed -E 's/model_step_([0-9]+)\.pt/\1/')
            dst=${live}/f\$n/models; mkdir -p \$dst
            abs=\$(realpath \$src)
            case \$abs in
                /storage/ice-shared/*) target=\$abs ;;
                *) target=\$(realpath --relative-to=\$dst \$src) ;;
            esac
            ln -sfn \"\$target\" \$dst/model_step_\$n.pt
        done"
    if [ "${STASHED}" = "0" ] && [ -n "$(git status --porcelain)" ]; then
        git stash push -q -u -m "live-eval submit $(date -Iseconds)" && STASHED=1
    fi
    OUT="$(pixi run python -m imitation_experiments.pipeline.cluster plan \
        --campaign "${EVAL_CAMPAIGN}" --arm "${arm}_curve" --seed 0 2>&1)" || { echo "${OUT}" | tail -15; continue; }
    PLAN="$(echo "${OUT}" | grep 'local dir:' | awk '{print $NF}')"
    SHA="$(echo "${OUT}" | grep '^PLAN_SHA=' | cut -d= -f2)"
    [ -n "${PLAN}" ] || { echo "[${arm}] plan failed"; echo "${OUT}" | tail -15; continue; }
    pixi run python -m imitation_experiments.pipeline.cluster submit \
        --plan "${PLAN}" --confirm "${SHA}" --allow-resubmit 2>&1 | grep -E 'job |refusing|ERROR' || true
done
