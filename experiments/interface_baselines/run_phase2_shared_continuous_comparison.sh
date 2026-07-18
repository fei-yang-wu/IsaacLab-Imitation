#!/usr/bin/env bash
set -euo pipefail

# Run DiffSR latent, full-body chunk, and EE chunk with one matched Phase 2
# planner protocol. Use a one-motion manifest for the Phase 2 code gate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-}"
FULL_BODY_TRAJECTORY_CHECKPOINT="${FULL_BODY_TRAJECTORY_CHECKPOINT:-}"
EE_TRAJECTORY_CHECKPOINT="${EE_TRAJECTORY_CHECKPOINT:-}"
MANIFEST="${MANIFEST:-data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-data/lafan1/g1_hl_diffsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/phase2_shared_continuous}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
COLLECT_SAMPLES="${COLLECT_SAMPLES:-1200}"
SAMPLE_BUDGET="${SAMPLE_BUDGET:-1000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
FLOW_STEPS="${FLOW_STEPS:-16}"
TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS:-4}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
RUN_ORACLE="${RUN_ORACLE:-1}"
FORCE_COLLECT="${FORCE_COLLECT:-0}"
DRY_RUN="${DRY_RUN:-0}"

: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set LATENT_LOW_LEVEL_CHECKPOINT.}"
: "${LATENT_SKILL_CHECKPOINT:?Set LATENT_SKILL_CHECKPOINT.}"
: "${FULL_BODY_TRAJECTORY_CHECKPOINT:?Set FULL_BODY_TRAJECTORY_CHECKPOINT.}"
: "${EE_TRAJECTORY_CHECKPOINT:?Set EE_TRAJECTORY_CHECKPOINT.}"

if [[ "${SAMPLE_BUDGET}" == "all" || "${SAMPLE_BUDGET}" == "0" ]]; then
    BUDGET_LABEL=all
    EXPECTED_ROWS_PER_STAGE="$((COLLECT_SAMPLES * NUM_ENVS))"
else
    BUDGET_LABEL="${SAMPLE_BUDGET}"
    EXPECTED_ROWS_PER_STAGE="${SAMPLE_BUDGET}"
fi

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

COMMON_ENV=(
    "MANIFEST=${MANIFEST}"
    "TRAIN_MANIFEST=${MANIFEST}"
    "EVAL_MANIFEST=${MANIFEST}"
    "NUM_ENVS=${NUM_ENVS}"
    "SEED=${SEED}"
    "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}"
    "EVAL_STEPS=${EVAL_STEPS}"
    "MODEL_SIZE=${MODEL_SIZE}"
    "PRETRAIN_UPDATES=${PRETRAIN_UPDATES}"
    "FINETUNE_UPDATES=${FINETUNE_UPDATES}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    "LR=${LR}"
    "WEIGHT_DECAY=${WEIGHT_DECAY}"
    "FLOW_STEPS=${FLOW_STEPS}"
    "TRAIN_ENDPOINT_STEPS=${TRAIN_ENDPOINT_STEPS}"
    "FLOW_NOISE_STD=${FLOW_NOISE_STD}"
    "RUN_ORACLE=${RUN_ORACLE}"
    "FORCE_COLLECT=${FORCE_COLLECT}"
    "DRY_RUN=${DRY_RUN}"
)

run_cmd env \
    "${COMMON_ENV[@]}" \
    "LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}" \
    "SKILL_CHECKPOINT=${LATENT_SKILL_CHECKPOINT}" \
    "DATASET_PATH=${DATASET_PATH}" \
    "OUTPUT_ROOT=${OUTPUT_ROOT}/latent" \
    "HORIZON_STEPS=${HORIZON_STEPS}" \
    "COLLECT_SAMPLES=${COLLECT_SAMPLES}" \
    "SAMPLE_BUDGET=${SAMPLE_BUDGET}" \
    experiments/interface_baselines/run_shared_latent_interface_comparison.sh

run_cmd env \
    "${COMMON_ENV[@]}" \
    "TASK=Isaac-Imitation-G1-v0" \
    "ALGORITHM=IPMD" \
    "OUTPUT_ROOT=${OUTPUT_ROOT}/explicit" \
    "INTERFACES=full_body_trajectory ee_trajectory" \
    "FULL_BODY_TRAJECTORY_CHECKPOINT=${FULL_BODY_TRAJECTORY_CHECKPOINT}" \
    "EE_TRAJECTORY_CHECKPOINT=${EE_TRAJECTORY_CHECKPOINT}" \
    "COLLECT_STEPS=${COLLECT_SAMPLES}" \
    "SAMPLE_BUDGETS=${SAMPLE_BUDGET}" \
    "MODEL_SIZES=${MODEL_SIZE}" \
    "COMMAND_PAST_STEPS=0" \
    "COMMAND_FUTURE_STEPS=${HORIZON_STEPS}" \
    "PLANNER_UPDATE_INTERVAL=${HORIZON_STEPS}" \
    "COMMAND_HOLD_STEPS=${HORIZON_STEPS}" \
    experiments/interface_baselines/run_dance102_strong_interface_comparison.sh

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run complete; audit skipped because artifacts were not created."
    exit 0
fi

latent_root="${OUTPUT_ROOT}/latent/latent_skill/transformer_${MODEL_SIZE}_${BUDGET_LABEL}"
full_root="${OUTPUT_ROOT}/explicit/full_body_trajectory/chunked_transformer_${MODEL_SIZE}_${BUDGET_LABEL}"
ee_root="${OUTPUT_ROOT}/explicit/ee_trajectory/chunked_transformer_${MODEL_SIZE}_${BUDGET_LABEL}"

run_cmd pixi run python experiments/interface_baselines/audit_phase2_shared_continuous.py \
    --latent_checkpoint "${latent_root}/planner_finetune_planner_rollout/checkpoints/latest.pt" \
    --latent_merge_manifest "${latent_root}/demonstration_and_planner_rollout_samples/merge_manifest.json" \
    --latent_summary "${latent_root}/eval_finetuned_closed_loop/summary.json" \
    --full_body_checkpoint "${full_root}/planner_finetune_planner_rollout/checkpoints/latest.pt" \
    --full_body_merge_manifest "${full_root}/demonstration_and_planner_rollout_samples/merge_manifest.json" \
    --full_body_summary "${full_root}/eval_finetuned_closed_loop/summary.json" \
    --ee_checkpoint "${ee_root}/planner_finetune_planner_rollout/checkpoints/latest.pt" \
    --ee_merge_manifest "${ee_root}/demonstration_and_planner_rollout_samples/merge_manifest.json" \
    --ee_summary "${ee_root}/eval_finetuned_closed_loop/summary.json" \
    --expected_seed "${SEED}" \
    --expected_history_steps "${STATE_HISTORY_STEPS}" \
    --expected_planner_interval "${HORIZON_STEPS}" \
    --expected_pretrain_updates "${PRETRAIN_UPDATES}" \
    --expected_finetune_updates "${FINETUNE_UPDATES}" \
    --expected_rows_per_stage "${EXPECTED_ROWS_PER_STAGE}" \
    --expected_eval_steps "${EVAL_STEPS}" \
    --output_json "${OUTPUT_ROOT}/phase2_protocol_audit.json"

echo "[INFO] Phase 2 shared continuous comparison passed under ${OUTPUT_ROOT}."
