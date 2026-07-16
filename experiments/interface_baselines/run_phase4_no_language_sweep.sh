#!/usr/bin/env bash
set -euo pipefail

# One Slurm-array task: one motion and one planner seed, evaluated at every
# fixed sample budget. Demonstration rows and low-level protocol checks are
# shared across budgets within this task; planner-rollout rows remain separate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${MANIFEST:?Set MANIFEST to the qualified full motion manifest}"
: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set LATENT_LOW_LEVEL_CHECKPOINT}"
: "${LATENT_SKILL_CHECKPOINT:?Set LATENT_SKILL_CHECKPOINT}"
: "${VANILLA_TRACKER_CHECKPOINT:?Set VANILLA_TRACKER_CHECKPOINT}"
: "${LATENT_QUALIFICATION_AUDIT:?Set LATENT_QUALIFICATION_AUDIT}"
: "${VANILLA_QUALIFICATION_AUDIT:?Set VANILLA_QUALIFICATION_AUDIT}"
: "${STREAMED_EQUIVALENCE_CERTIFICATE:?Set STREAMED_EQUIVALENCE_CERTIFICATE}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${DATASET_PATH:?Set DATASET_PATH to the latent trajectory cache}"
: "${VANILLA_DATASET_PATH:?Set VANILLA_DATASET_PATH to the vanilla trajectory cache}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/phase4_no_language}"
SEEDS="${SEEDS:-0 1 2}"
SAMPLE_BUDGETS="${SAMPLE_BUDGETS:-1000 10000 50000}"
NUM_ENVS="${NUM_ENVS:-16}"
EXPECTED_MOTION_COUNT="${EXPECTED_MOTION_COUNT:-40}"
MIN_ORACLE_SUCCESS="${MIN_ORACLE_SUCCESS:-0.8}"
TASK_INDEX="${PHASE4_TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
FLOW_STEPS="${FLOW_STEPS:-16}"
TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS:-4}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"

if [[ -z "${TASK_INDEX}" || ! "${TASK_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] PHASE4_TASK_INDEX or SLURM_ARRAY_TASK_ID must be non-negative." >&2
    exit 2
fi
if [[ ! "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_ENVS must be positive." >&2
    exit 2
fi

read -r -a PYTHON_CMD <<< "${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a BUDGET_LIST <<< "${SAMPLE_BUDGETS}"

mapfile -t selection < <(
    "${PYTHON_CMD[@]}" experiments/interface_baselines/phase4_no_language_matrix.py \
        --manifest "${MANIFEST}" \
        --seeds "${SEED_LIST[@]}" \
        --sample_budgets "${BUDGET_LIST[@]}" \
        --task_index "${TASK_INDEX}" \
        --num_envs "${NUM_ENVS}" \
        --format lines
)
if [[ "${#selection[@]}" -ne 8 ]]; then
    echo "[ERROR] Phase-4 task resolver returned incomplete output." >&2
    exit 2
fi
motion_name="${selection[0]}"
motion_slug="${selection[1]}"
seed="${selection[2]}"
total_tasks="${selection[3]}"
motion_count="${selection[4]}"
max_budget="${selection[5]}"
collect_decisions="${selection[6]}"
available_rows="${selection[7]}"

if [[ "${motion_count}" != "${EXPECTED_MOTION_COUNT}" ]]; then
    echo "[ERROR] Resolved motion count differs from EXPECTED_MOTION_COUNT." >&2
    exit 2
fi
task_root="${OUTPUT_ROOT}/seed_${seed}/${motion_slug}"
single_manifest="${task_root}/input/manifest.json"
mkdir -p "${task_root}/input" "${task_root}/protocol_checks"

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    "${PYTHON_CMD[@]}" experiments/interface_baselines/validate_phase4_no_language_submission.py \
        "${MANIFEST}" \
        "${VANILLA_TRACKER_CHECKPOINT}" \
        "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        "${LATENT_SKILL_CHECKPOINT}" \
        "${VANILLA_QUALIFICATION_AUDIT}" \
        "${LATENT_QUALIFICATION_AUDIT}" \
        "${STREAMED_EQUIVALENCE_CERTIFICATE}" \
        "${DATASET_PATH}" \
        "${VANILLA_DATASET_PATH}" \
        "${EXPECTED_MANIFEST_SHA256}" \
        --expected_motion_count "${EXPECTED_MOTION_COUNT}" \
        --minimum_oracle_success "${MIN_ORACLE_SUCCESS}" \
        --output_json "${task_root}/input/submission_gate.json"
fi

"${PYTHON_CMD[@]}" experiments/interface_baselines/write_single_motion_manifest.py \
    --manifest "${MANIFEST}" \
    --motion_name "${motion_name}" \
    --output "${single_manifest}"
"${PYTHON_CMD[@]}" experiments/interface_baselines/phase4_no_language_matrix.py \
    --manifest "${MANIFEST}" \
    --seeds "${SEED_LIST[@]}" \
    --sample_budgets "${BUDGET_LIST[@]}" \
    --task_index "${TASK_INDEX}" \
    --num_envs "${NUM_ENVS}" \
    > "${task_root}/task_config.json"

echo "[INFO] Phase-4 task ${TASK_INDEX}/${total_tasks}: motion=${motion_name} seed=${seed}"
echo "[INFO] budgets=${SAMPLE_BUDGETS} max_budget=${max_budget} available_rows=${available_rows}"

audit_passed() {
    local path="$1"
    [[ -f "${path}" ]] || return 1
    "${PYTHON_CMD[@]}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("passed") is True else 1)
PY
}

for budget in "${BUDGET_LIST[@]}"; do
    budget_audit="${task_root}/protocol_checks/focused_protocol_audit_${budget}.json"
    if [[ "${RESUME}" == "1" ]] && audit_passed "${budget_audit}"; then
        echo "[INFO] Reusing completed Phase-4 budget ${budget}: ${budget_audit}"
        continue
    fi
    run_protocol_checks=0
    run_ceiling=0
    run_oracle=0
    [[ -f "${task_root}/protocol_checks/streamed_vanilla_equivalence.json" ]] || run_protocol_checks=1
    [[ -f "${task_root}/ceiling/direct_vanilla_50hz/summary.json" ]] || run_ceiling=1
    latent_oracle="${task_root}/planner_rows/latent_skill/latent_skill/oracle_low_level/summary.json"
    explicit_oracle="${task_root}/planner_rows/full_body_streamed_vanilla/full_body_trajectory_streamed_vanilla/oracle_low_level/summary.json"
    if [[ ! -f "${latent_oracle}" || ! -f "${explicit_oracle}" ]]; then
        run_oracle=1
    fi
    env \
        LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT}" \
        LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT}" \
        VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT}" \
        MANIFEST="${single_manifest}" \
        DATASET_PATH="${DATASET_PATH}" \
        VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH}" \
        OUTPUT_ROOT="${task_root}" \
        NUM_ENVS="${NUM_ENVS}" \
        SEED="${seed}" \
        COLLECT_SAMPLES="${collect_decisions}" \
        SAMPLE_BUDGET="${budget}" \
        EVAL_STEPS="${EVAL_STEPS}" \
        MODEL_SIZE="${MODEL_SIZE}" \
        PRETRAIN_UPDATES="${PRETRAIN_UPDATES}" \
        FINETUNE_UPDATES="${FINETUNE_UPDATES}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
        FLOW_STEPS="${FLOW_STEPS}" \
        TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS}" \
        FORCE_COLLECT=0 \
        RUN_ORACLE="${run_oracle}" \
        RUN_PROTOCOL_CHECKS="${run_protocol_checks}" \
        RUN_CEILING="${run_ceiling}" \
        DRY_RUN="${DRY_RUN}" \
        experiments/interface_baselines/run_focused_causal_interface_comparison.sh
done

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "[INFO] Phase-4 task dry run complete."
    exit 0
fi
for budget in "${BUDGET_LIST[@]}"; do
    audit="${task_root}/protocol_checks/focused_protocol_audit_${budget}.json"
    if [[ ! -f "${audit}" ]]; then
        echo "[ERROR] Missing completed budget audit: ${audit}" >&2
        exit 2
    fi
done
echo "[PASS] Phase-4 task complete: ${task_root}"
