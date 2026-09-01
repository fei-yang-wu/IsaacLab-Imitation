#!/usr/bin/env bash
# Submit every tracker arm once its encoder exists on the cluster.
#
# Waits for the Tier B pretrain jobs to drain, then verifies each encoder
# checkpoint before planning anything. A missing encoder aborts the whole
# submission: a tracker bound to a half-written checkpoint is worse than no
# tracker, because it trains and produces plausible numbers.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CAMPAIGN="experiments/campaigns/2026-08-30-encoder-interface-500m/campaign.yaml"
LOGIN="${LOGIN:-ice}"
SEED="${SEED:-0}"
ENCODER_ROOT="/home/hice1/fwu91/scratch/Research/IsaacLab/data/endpoint_collapse_probe"
PARETO_ENCODER="/home/hice1/fwu91/scratch/Research/IsaacLab/data/pareto_stack/diffntp_chunk_h1_ee_wide_seed0/encoder/checkpoints/latest.pt"

ARMS=(prod suffix1 suffix2 suffix5 suffix9 h1 h2 h5 h10)
ENCODER_ARMS=(suffix1 suffix2 suffix5 suffix9 h1 h2 h5 h10)

echo "[wait] draining endpoint-collapse-probe pretrain jobs..."
while [ "$(ssh "${LOGIN}" 'squeue -u $USER -h -o "%.60j" | grep -c collapse' || echo 1)" != "0" ]; do
    sleep 180
done
echo "[wait] queue clear."

echo "[verify] encoder checkpoints..."
missing=0
for arm in "${ENCODER_ARMS[@]}"; do
    path="${ENCODER_ROOT}/${arm}_seed${SEED}/encoder/checkpoints/latest.pt"
    if ssh "${LOGIN}" "test -s '${path}'"; then
        echo "  OK      ${arm}"
    else
        echo "  MISSING ${arm}: ${path}"
        missing=1
    fi
done
if ssh "${LOGIN}" "test -s '${PARETO_ENCODER}'"; then
    echo "  OK      prod"
else
    echo "  MISSING prod: ${PARETO_ENCODER}"
    missing=1
fi
if [ "${missing}" != "0" ]; then
    echo "[abort] at least one encoder is missing; nothing submitted." >&2
    exit 1
fi

for arm in "${ARMS[@]}"; do
    out=$(pixi run python -m imitation_experiments.pipeline.cluster plan \
        --campaign "${CAMPAIGN}" --arm "${arm}" --seed "${SEED}" 2>&1) || {
        echo "[abort] plan failed for ${arm}" >&2; echo "${out}" | tail -5 >&2; exit 1; }
    plan_dir=$(echo "${out}" | grep -oP '(?<=\[PLAN\] local dir:  ).*')
    sha=$(echo "${out}" | grep -oP '(?<=^PLAN_SHA=).*')
    sub=$(pixi run python -m imitation_experiments.pipeline.cluster submit \
        --plan "${plan_dir}" --confirm "${sha}" 2>&1)
    job=$(echo "${sub}" | grep -oP '(?<=stage lowlevel: job )\d+' || true)
    if [ -z "${job}" ]; then
        echo "[abort] submit failed for ${arm}" >&2; echo "${sub}" | tail -5 >&2; exit 1
    fi
    echo "${arm} s${SEED} -> job ${job}"
done
echo "[done] all tracker arms submitted."
