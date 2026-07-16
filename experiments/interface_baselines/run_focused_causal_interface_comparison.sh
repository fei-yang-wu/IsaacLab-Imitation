#!/usr/bin/env bash
set -euo pipefail

# Local-only focused comparison:
#   1. DiffSR latent skill predicted by the shared causal planner.
#   2. Full-body trajectory predicted by the same planner backbone and streamed
#      through the frozen single-frame vanilla tracker.
#   3. Direct 50 Hz vanilla reference tracking, reported only as a ceiling.
#
# Use a one-motion manifest when training one no-language planner per motion.
# This runner deliberately contains no cluster or Skynet submission path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

PYTHON_CMD_STR="${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
ISAACLAB_PYTHON_CMD_STR="${INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD:-pixi run -e isaaclab python}"
# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_CMD_STR})
# shellcheck disable=SC2206
ISAACLAB_PYTHON_CMD=(${ISAACLAB_PYTHON_CMD_STR})

LATENT_TASK="${LATENT_TASK:-Isaac-Imitation-G1-Latent-v0}"
VANILLA_TASK="${VANILLA_TASK:-Isaac-Imitation-G1-v0}"
ALGORITHM="${ALGORITHM:-IPMD}"
LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-}"
VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-}"
MANIFEST="${MANIFEST:-}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST}}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST}}"
DATASET_PATH="${DATASET_PATH:-}"
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/focused_causal_interface}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
CHUNK_STEPS="${CHUNK_STEPS:-${HORIZON_STEPS:-10}}"
HORIZON_STEPS="${HORIZON_STEPS:-${CHUNK_STEPS}}"
FULL_BODY_FUTURE_STEPS="$((CHUNK_STEPS - 1))"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
COLLECT_SAMPLES="${COLLECT_SAMPLES:-1200}"
SAMPLE_BUDGET="${SAMPLE_BUDGET:-1000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
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
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-}"
LANGUAGE_GOAL_NAME="${LANGUAGE_GOAL_NAME:-}"
RUN_ORACLE="${RUN_ORACLE:-1}"
RUN_PROTOCOL_CHECKS="${RUN_PROTOCOL_CHECKS:-1}"
RUN_CEILING="${RUN_CEILING:-1}"
FORCE_COLLECT="${FORCE_COLLECT:-1}"
USE_CHECKPOINT_NORMALIZATION="${USE_CHECKPOINT_NORMALIZATION:-0}"
EQUIVALENCE_STEPS="${EQUIVALENCE_STEPS:-20}"
EQUIVALENCE_NUM_ENVS="${EQUIVALENCE_NUM_ENVS:-2}"
DRY_RUN="${DRY_RUN:-0}"

: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set LATENT_LOW_LEVEL_CHECKPOINT.}"
: "${LATENT_SKILL_CHECKPOINT:?Set LATENT_SKILL_CHECKPOINT.}"
: "${VANILLA_TRACKER_CHECKPOINT:?Set VANILLA_TRACKER_CHECKPOINT.}"
: "${TRAIN_MANIFEST:?Set MANIFEST or TRAIN_MANIFEST to corrected motion data.}"
: "${EVAL_MANIFEST:?Set MANIFEST or EVAL_MANIFEST to corrected motion data.}"
: "${DATASET_PATH:?Set DATASET_PATH to the matching corrected latent dataset.}"
if [[ -n "${LANGUAGE_EMBEDDINGS}" && -z "${LANGUAGE_GOAL_NAME}" ]]; then
    echo "[ERROR] LANGUAGE_GOAL_NAME is required when LANGUAGE_EMBEDDINGS is set." >&2
    exit 2
fi

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] ${name} must be a positive integer, got ${value}." >&2
        exit 2
    fi
}

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} must be a non-negative integer, got ${value}." >&2
        exit 2
    fi
}

require_positive_integer NUM_ENVS "${NUM_ENVS}"
require_positive_integer CHUNK_STEPS "${CHUNK_STEPS}"
require_positive_integer HORIZON_STEPS "${HORIZON_STEPS}"
if [[ "${CHUNK_STEPS}" != "${HORIZON_STEPS}" ]]; then
    echo "[ERROR] CHUNK_STEPS and HORIZON_STEPS must match." >&2
    exit 2
fi
if [[ "${CHUNK_STEPS}" != "10" ]]; then
    echo "[ERROR] The focused paper protocol fixes CHUNK_STEPS=10." >&2
    exit 2
fi
require_positive_integer EQUIVALENCE_STEPS "${EQUIVALENCE_STEPS}"
require_positive_integer EQUIVALENCE_NUM_ENVS "${EQUIVALENCE_NUM_ENVS}"
if (( EQUIVALENCE_NUM_ENVS < 2 )); then
    echo "[ERROR] EQUIVALENCE_NUM_ENVS must be at least 2." >&2
    exit 2
fi
if (( EQUIVALENCE_STEPS <= CHUNK_STEPS + 1 )); then
    echo "[ERROR] EQUIVALENCE_STEPS must exceed CHUNK_STEPS + 1." >&2
    exit 2
fi
require_nonnegative_integer STATE_HISTORY_STEPS "${STATE_HISTORY_STEPS}"
require_positive_integer COLLECT_SAMPLES "${COLLECT_SAMPLES}"
require_positive_integer EVAL_STEPS "${EVAL_STEPS}"
require_nonnegative_integer PRETRAIN_UPDATES "${PRETRAIN_UPDATES}"
require_nonnegative_integer FINETUNE_UPDATES "${FINETUNE_UPDATES}"
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SEED must be a non-negative integer, got ${SEED}." >&2
    exit 2
fi
if [[ "${SAMPLE_BUDGET}" =~ ^[1-9][0-9]*$ ]]; then
    BUDGET_LABEL="${SAMPLE_BUDGET}"
    EXPECTED_ROWS_PER_STAGE="${SAMPLE_BUDGET}"
    available_rows="$((COLLECT_SAMPLES * NUM_ENVS))"
    if (( SAMPLE_BUDGET > available_rows )); then
        echo "[ERROR] SAMPLE_BUDGET=${SAMPLE_BUDGET} exceeds ${available_rows} collected rows." >&2
        exit 2
    fi
else
    echo "[ERROR] SAMPLE_BUDGET must be an exact positive row budget." >&2
    exit 2
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

PLANNER_ROWS_ROOT="${OUTPUT_ROOT}/planner_rows"
LATENT_OUTPUT_ROOT="${PLANNER_ROWS_ROOT}/latent_skill"
FULL_BODY_OUTPUT_ROOT="${PLANNER_ROWS_ROOT}/full_body_streamed_vanilla"
CEILING_ROOT="${OUTPUT_ROOT}/ceiling/direct_vanilla_50hz"
VANILLA_DATASET_ARGS=()
if [[ -n "${VANILLA_DATASET_PATH}" ]]; then
    VANILLA_DATASET_ARGS+=(--dataset_path "${VANILLA_DATASET_PATH}")
fi

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/write_interface_run_provenance.py \
    --label focused-causal-latent-vs-full-body-streamed-vanilla \
    --output_json "${OUTPUT_ROOT}/protocol_checks/run_provenance_${BUDGET_LABEL}.json" \
    --result_root "${OUTPUT_ROOT}"

# Certify the adapter before collecting or training a planner. This exercises
# all ten hold phases plus a deliberately desynchronized row.
if [[ "${RUN_PROTOCOL_CHECKS}" == "1" ]]; then
run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/command_space_ablation/evaluate_checkpoint.py \
    --headless \
    --task "${VANILLA_TASK}" \
    --algo "${ALGORITHM}" \
    --checkpoint "${VANILLA_TRACKER_CHECKPOINT}" \
    --low_level_command_mode streamed_vanilla \
    --command_space full_body_trajectory \
    --command_past_steps 0 \
    --command_future_steps "${FULL_BODY_FUTURE_STEPS}" \
    --planner_update_interval "${CHUNK_STEPS}" \
    --motion_manifest "${EVAL_MANIFEST}" \
    "${VANILLA_DATASET_ARGS[@]}" \
    --num_envs "${EQUIVALENCE_NUM_ENVS}" \
    --steps "${EQUIVALENCE_STEPS}" \
    --seed "${SEED}" \
    --certify_streamed_vanilla_equivalence \
    --equivalence_steps "${EQUIVALENCE_STEPS}" \
    --label streamed_vanilla_equivalence \
    --output_json "${OUTPUT_ROOT}/protocol_checks/streamed_vanilla_equivalence.json" \
    --kit_args=--/app/extensions/fsWatcherEnabled=false
fi

# This is intentionally outside PLANNER_ROWS_ROOT. It receives the fresh
# reference frame at every 50 Hz control step and is never trained as a planner.
if [[ "${RUN_CEILING}" == "1" ]]; then
run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/command_space_ablation/evaluate_checkpoint.py \
    --headless \
    --task "${VANILLA_TASK}" \
    --algo "${ALGORITHM}" \
    --checkpoint "${VANILLA_TRACKER_CHECKPOINT}" \
    --policy_only_checkpoint \
    --low_level_command_mode native \
    --command_space single_frame_full_body \
    --command_past_steps 0 \
    --command_future_steps 0 \
    --command_observation_source reference \
    --planner_update_interval 1 \
    --motion_manifest "${EVAL_MANIFEST}" \
    "${VANILLA_DATASET_ARGS[@]}" \
    --num_envs "${NUM_ENVS}" \
    --steps "${EVAL_STEPS}" \
    --seed "${SEED}" \
    --label direct_vanilla_50hz_ceiling \
    --output_json "${CEILING_ROOT}/summary.json" \
    --output_csv "${CEILING_ROOT}/summary.csv" \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    "env.command_hold_steps=0"
fi

COMMON_ENV=(
    "MANIFEST=${TRAIN_MANIFEST}"
    "TRAIN_MANIFEST=${TRAIN_MANIFEST}"
    "EVAL_MANIFEST=${EVAL_MANIFEST}"
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
    "LANGUAGE_EMBEDDINGS=${LANGUAGE_EMBEDDINGS}"
    "LANGUAGE_GOAL_NAME=${LANGUAGE_GOAL_NAME}"
    "RUN_ORACLE=${RUN_ORACLE}"
    "FORCE_COLLECT=${FORCE_COLLECT}"
    "USE_CHECKPOINT_NORMALIZATION=${USE_CHECKPOINT_NORMALIZATION}"
    "DRY_RUN=${DRY_RUN}"
)

run_cmd env \
    "${COMMON_ENV[@]}" \
    "TASK=${LATENT_TASK}" \
    "ALGORITHM=${ALGORITHM}" \
    "LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}" \
    "SKILL_CHECKPOINT=${LATENT_SKILL_CHECKPOINT}" \
    "DATASET_PATH=${DATASET_PATH}" \
    "OUTPUT_ROOT=${LATENT_OUTPUT_ROOT}" \
    "HORIZON_STEPS=${CHUNK_STEPS}" \
    "COLLECT_SAMPLES=${COLLECT_SAMPLES}" \
    "SAMPLE_BUDGET=${SAMPLE_BUDGET}" \
    experiments/interface_baselines/run_shared_latent_interface_comparison.sh

run_cmd env \
    "${COMMON_ENV[@]}" \
    "TASK=${VANILLA_TASK}" \
    "ALGORITHM=${ALGORITHM}" \
    "LOW_LEVEL_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}" \
    "VANILLA_TRACKER_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}" \
    "FULL_BODY_TRAJECTORY_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}" \
    "FULL_BODY_LOW_LEVEL_COMMAND_MODE=streamed_vanilla" \
    "DATASET_PATH=${VANILLA_DATASET_PATH}" \
    "OUTPUT_ROOT=${FULL_BODY_OUTPUT_ROOT}" \
    "INTERFACES=full_body_trajectory" \
    "COLLECT_STEPS=${COLLECT_SAMPLES}" \
    "SAMPLE_BUDGETS=${SAMPLE_BUDGET}" \
    "MODEL_SIZES=${MODEL_SIZE}" \
    "COMMAND_PAST_STEPS=0" \
    "COMMAND_FUTURE_STEPS=${FULL_BODY_FUTURE_STEPS}" \
    "PLANNER_UPDATE_INTERVAL=${CHUNK_STEPS}" \
    "COMMAND_HOLD_STEPS=${CHUNK_STEPS}" \
    experiments/interface_baselines/run_dance102_strong_interface_comparison.sh

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run complete; no local process or audit was executed."
    exit 0
fi

latent_root="${LATENT_OUTPUT_ROOT}/latent_skill/transformer_${MODEL_SIZE}_${BUDGET_LABEL}"
full_body_root="${FULL_BODY_OUTPUT_ROOT}/full_body_trajectory_streamed_vanilla/chunked_transformer_${MODEL_SIZE}_${BUDGET_LABEL}"

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/audit_focused_causal_interface_comparison.py \
    --latent_checkpoint "${latent_root}/planner_finetune_planner_rollout/checkpoints/latest.pt" \
    --latent_merge_manifest "${latent_root}/demonstration_and_planner_rollout_samples/merge_manifest.json" \
    --latent_pretrained_summary "${latent_root}/eval_pretrained_closed_loop/summary.json" \
    --latent_summary "${latent_root}/eval_finetuned_closed_loop/summary.json" \
    --full_body_checkpoint "${full_body_root}/planner_finetune_planner_rollout/checkpoints/latest.pt" \
    --full_body_merge_manifest "${full_body_root}/demonstration_and_planner_rollout_samples/merge_manifest.json" \
    --full_body_pretrained_summary "${full_body_root}/eval_pretrained_closed_loop/summary.json" \
    --full_body_summary "${full_body_root}/eval_finetuned_closed_loop/summary.json" \
    --latent_oracle_summary "${LATENT_OUTPUT_ROOT}/latent_skill/oracle_low_level/summary.json" \
    --full_body_oracle_summary "${FULL_BODY_OUTPUT_ROOT}/full_body_trajectory_streamed_vanilla/oracle_low_level/summary.json" \
    --direct_vanilla_summary "${CEILING_ROOT}/summary.json" \
    --streamed_equivalence "${OUTPUT_ROOT}/protocol_checks/streamed_vanilla_equivalence.json" \
    --expected_seed "${SEED}" \
    --expected_num_envs "${NUM_ENVS}" \
    --expected_history_steps "${STATE_HISTORY_STEPS}" \
    --expected_horizon_steps "${CHUNK_STEPS}" \
    --expected_full_body_future_steps "${FULL_BODY_FUTURE_STEPS}" \
    --expected_planner_interval "${CHUNK_STEPS}" \
    --expected_pretrain_updates "${PRETRAIN_UPDATES}" \
    --expected_finetune_updates "${FINETUNE_UPDATES}" \
    --expected_rows_per_stage "${EXPECTED_ROWS_PER_STAGE}" \
    --expected_eval_steps "${EVAL_STEPS}" \
    --output_json "${OUTPUT_ROOT}/protocol_checks/focused_protocol_audit_${BUDGET_LABEL}.json"

echo "[INFO] Focused causal-interface comparison passed under ${OUTPUT_ROOT}."
