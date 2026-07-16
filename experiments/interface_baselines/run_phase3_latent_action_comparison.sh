#!/usr/bin/env bash
set -euo pipefail

# Run the matched Future-CVAE and per-step-token planner protocols and audit
# their causal inputs, timing, sample counts, updates, and shared backbone.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${FUTURE_LOW_LEVEL_CHECKPOINT:?Set FUTURE_LOW_LEVEL_CHECKPOINT.}"
: "${TOKEN_LOW_LEVEL_CHECKPOINT:?Set TOKEN_LOW_LEVEL_CHECKPOINT.}"
: "${MOTION_MANIFEST:?Set MOTION_MANIFEST to the same one-motion manifest.}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/phase3_latent_actions}"
NUM_ENVS="${NUM_ENVS:-16}"
SEED="${SEED:-0}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
PLANNER_INTERVAL_STEPS="${PLANNER_INTERVAL_STEPS:-10}"
SAMPLE_BUDGET="${SAMPLE_BUDGET:-1000}"
COLLECT_PLANNER_STEPS="${COLLECT_PLANNER_STEPS:-1200}"
EVAL_CONTROL_STEPS="${EVAL_CONTROL_STEPS:-1000}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
FORCE="${FORCE:-0}"

for value in \
  "${NUM_ENVS}" \
  "${SAMPLE_BUDGET}" \
  "${COLLECT_PLANNER_STEPS}" \
  "${EVAL_CONTROL_STEPS}" \
  "${PRETRAIN_UPDATES}" \
  "${FINETUNE_UPDATES}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Phase 3 counts must be positive integers; got ${value}." >&2
    exit 2
  fi
done

COLLECTED_ROWS="$((COLLECT_PLANNER_STEPS * NUM_ENVS))"
CONTROL_STEPS="$((COLLECT_PLANNER_STEPS * PLANNER_INTERVAL_STEPS))"
if (( COLLECTED_ROWS < SAMPLE_BUDGET )); then
  echo "[ERROR] Collection produces ${COLLECTED_ROWS} rows, below SAMPLE_BUDGET=${SAMPLE_BUDGET}." >&2
  exit 2
fi

COMMON_ENV=(
  MOTION_MANIFEST="${MOTION_MANIFEST}"
  NUM_ENVS="${NUM_ENVS}"
  SEED="${SEED}"
  STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS}"
  PLANNER_INTERVAL_STEPS="${PLANNER_INTERVAL_STEPS}"
  SAMPLE_BUDGET="${SAMPLE_BUDGET}"
  COLLECT_PLANNER_STEPS="${COLLECT_PLANNER_STEPS}"
  EVAL_CONTROL_STEPS="${EVAL_CONTROL_STEPS}"
  PRETRAIN_UPDATES="${PRETRAIN_UPDATES}"
  FINETUNE_UPDATES="${FINETUNE_UPDATES}"
  BATCH_SIZE="${BATCH_SIZE}"
  MODEL_SIZE="${MODEL_SIZE}"
  FORCE="${FORCE}"
)

env \
  "${COMMON_ENV[@]}" \
  LOW_LEVEL_CHECKPOINT="${FUTURE_LOW_LEVEL_CHECKPOINT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/future_cvae" \
  experiments/interface_baselines/run_future_cvae_interface_comparison.sh

env \
  "${COMMON_ENV[@]}" \
  LOW_LEVEL_CHECKPOINT="${TOKEN_LOW_LEVEL_CHECKPOINT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/per_step_token_sequence" \
  experiments/interface_baselines/run_per_step_token_interface_comparison.sh

pixi run python experiments/interface_baselines/audit_phase3_latent_interfaces.py \
  --future_checkpoint \
    "${OUTPUT_ROOT}/future_cvae/planner_finetuned/checkpoints/latest.pt" \
  --future_merge_manifest "${OUTPUT_ROOT}/future_cvae/merged/merge_manifest.json" \
  --future_oracle_summary "${OUTPUT_ROOT}/future_cvae/demonstration/summary.json" \
  --future_rollout_summary "${OUTPUT_ROOT}/future_cvae/planner_rollout/summary.json" \
  --future_oracle_eval_summary "${OUTPUT_ROOT}/future_cvae/eval_oracle/summary.json" \
  --future_pretrained_eval_summary \
    "${OUTPUT_ROOT}/future_cvae/eval_planner_pretrained/summary.json" \
  --future_finetuned_eval_summary \
    "${OUTPUT_ROOT}/future_cvae/eval_planner_finetuned/summary.json" \
  --token_checkpoint \
    "${OUTPUT_ROOT}/per_step_token_sequence/planner_finetuned/checkpoints/latest.pt" \
  --token_merge_manifest \
    "${OUTPUT_ROOT}/per_step_token_sequence/merged/merge_manifest.json" \
  --token_oracle_summary \
    "${OUTPUT_ROOT}/per_step_token_sequence/demonstration/summary.json" \
  --token_rollout_summary \
    "${OUTPUT_ROOT}/per_step_token_sequence/planner_rollout/summary.json" \
  --token_oracle_eval_summary \
    "${OUTPUT_ROOT}/per_step_token_sequence/eval_oracle/summary.json" \
  --token_pretrained_eval_summary \
    "${OUTPUT_ROOT}/per_step_token_sequence/eval_planner_pretrained/summary.json" \
  --token_finetuned_eval_summary \
    "${OUTPUT_ROOT}/per_step_token_sequence/eval_planner_finetuned/summary.json" \
  --expected_seed "${SEED}" \
  --expected_history_steps "${STATE_HISTORY_STEPS}" \
  --expected_planner_interval "${PLANNER_INTERVAL_STEPS}" \
  --expected_pretrain_updates "${PRETRAIN_UPDATES}" \
  --expected_finetune_updates "${FINETUNE_UPDATES}" \
  --expected_rows_per_stage "${SAMPLE_BUDGET}" \
  --expected_collected_rows_per_stage "${COLLECTED_ROWS}" \
  --expected_collection_control_steps "${CONTROL_STEPS}" \
  --expected_eval_control_steps "${EVAL_CONTROL_STEPS}" \
  --expected_token_horizon 10 \
  --expected_codebook_size 512 \
  --output_json "${OUTPUT_ROOT}/phase3_protocol_audit.json"

echo "[INFO] Matched Phase 3 latent-action artifacts: ${OUTPUT_ROOT}"
