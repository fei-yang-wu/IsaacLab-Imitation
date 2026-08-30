#!/usr/bin/env bash
# Plan and submit every optimizer-ablation-5b arm.
#
# All ten arms bind ONE pre-existing encoder, so there is nothing to wait for:
# the only precondition is that the encoder checkpoint is on the cluster. A
# missing encoder aborts before anything is submitted -- a tracker bound to a
# half-written checkpoint still trains and still produces plausible numbers.
#
# Commit first if the run must be reproducible from a SHA: `submit` packs the
# working tree, not git HEAD, and records `drift=true` when it is dirty. This
# campaign depends on two RLOpt knobs added alongside it
# (`ipmd.critic_lr_schedule`, the `actor_log_std` no-decay group), so the
# RLOpt submodule pointer must be committed too.
#
#   ./submit_all.sh                 # all ten arms, seed 0
#   ARMS="ctrl mb_half_e5" ./submit_all.sh
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CAMPAIGN="experiments/campaigns/2026-08-30-optimizer-ablation-5b/campaign.yaml"
LOGIN="${LOGIN:-ice}"
SEED="${SEED:-0}"
ENCODER="/home/hice1/fwu91/scratch/Research/IsaacLab/data/pareto_stack/diffntp_chunk_h1_ee_wide_seed0/encoder/checkpoints/latest.pt"
ARMS="${ARMS:-ctrl mb_half_e5 mb_half_e3 mb_full_e5 mb_full_e3 critic_lin wd_1e2 wd_1e4 wd_1e1 ent_only floor_late ent_sonic}"

echo "[verify] encoder checkpoint..."
if ssh "${LOGIN}" "test -s '${ENCODER}'"; then
    echo "  OK      ${ENCODER}"
else
    echo "[abort] encoder missing: ${ENCODER}" >&2
    exit 1
fi

for arm in ${ARMS}; do
    out=$(pixi run python -m imitation_experiments.pipeline.cluster plan \
        --campaign "${CAMPAIGN}" --arm "${arm}" --seed "${SEED}" 2>&1) || {
        echo "[abort] plan failed for ${arm}" >&2; echo "${out}" | tail -5 >&2; exit 1; }
    plan_dir=$(echo "${out}" | grep -oP '(?<=\[PLAN\] local dir:  ).*')
    sha=$(echo "${out}" | grep -oP '(?<=^PLAN_SHA=).*')
    sub=$(pixi run python -m imitation_experiments.pipeline.cluster submit \
        --plan "${plan_dir}" --confirm "${sha}" 2>&1)
    jobs=$(echo "${sub}" | grep -oP '(?<=job )\d+' | tr '\n' ' ' || true)
    if [ -z "${jobs}" ]; then
        echo "[abort] submit failed for ${arm}" >&2; echo "${sub}" | tail -5 >&2; exit 1
    fi
    echo "${arm} s${SEED} -> jobs ${jobs}"
done
echo "[done] all arms submitted."
