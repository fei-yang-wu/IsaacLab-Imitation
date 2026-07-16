#!/usr/bin/env bash
set -euo pipefail

# Render or submit the fixed three-seed Phase-5 paper study. For a real
# submission, validate every seed before submitting the first dependency chain.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${VANILLA_TRACKER_CHECKPOINT:?Set the qualified vanilla checkpoint path}"
: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set the qualified latent checkpoint path}"
: "${LATENT_SKILL_CHECKPOINT:?Set the qualified skill checkpoint path}"

DRY_RUN="${DRY_RUN:-1}"
SEEDS="${SEEDS:-0 1 2}"
OUTPUT_ROOT_PREFIX="${OUTPUT_ROOT_PREFIX:-logs/interface_baselines/bones_seed_100_multigoal_language_seed}"

read -r -a seed_list <<< "${SEEDS}"
if [[ "${#seed_list[@]}" -ne 3 \
    || "${seed_list[0]}" != "0" \
    || "${seed_list[1]}" != "1" \
    || "${seed_list[2]}" != "2" ]]; then
    echo "[ERROR] The paper protocol fixes SEEDS='0 1 2'." >&2
    exit 2
fi

run_seed() {
    local seed="$1"
    local preflight_only="$2"
    env \
        SEED="${seed}" \
        OUTPUT_ROOT="${OUTPUT_ROOT_PREFIX}${seed}" \
        DRY_RUN="${DRY_RUN}" \
        PREFLIGHT_ONLY="${preflight_only}" \
        "${SCRIPT_DIR}/submit_bones_seed_multigoal_pipeline_skynet.sh"
}

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    echo "[INFO] Validating all three paper seeds before submitting any jobs."
    for seed in "${seed_list[@]}"; do
        run_seed "${seed}" 1
    done
fi

for seed in "${seed_list[@]}"; do
    echo "[INFO] Rendering/submitting BONES-SEED planner seed ${seed}."
    run_seed "${seed}" 0
done
