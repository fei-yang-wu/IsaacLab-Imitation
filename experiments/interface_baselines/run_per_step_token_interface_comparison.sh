#!/usr/bin/env bash
set -euo pipefail

# Two-stage causal planner protocol for the per-step categorical token interface.
# Required inputs:
#   LOW_LEVEL_CHECKPOINT=/abs/path/model.pt
#   MOTION_MANIFEST=/abs/path/one_motion_manifest.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${LOW_LEVEL_CHECKPOINT:?Set LOW_LEVEL_CHECKPOINT to a trained per-step VQ policy.}"
: "${MOTION_MANIFEST:?Set MOTION_MANIFEST to a one-motion manifest.}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/per_step_token_sequence}"
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

DEMO_DIR="${OUTPUT_ROOT}/demonstration"
PRETRAIN_DIR="${OUTPUT_ROOT}/planner_pretrained"
ROLLOUT_DIR="${OUTPUT_ROOT}/planner_rollout"
MERGED_DIR="${OUTPUT_ROOT}/merged"
FINAL_DIR="${OUTPUT_ROOT}/planner_finetuned"
ORACLE_EVAL_DIR="${OUTPUT_ROOT}/eval_oracle"
PRETRAIN_EVAL_DIR="${OUTPUT_ROOT}/eval_planner_pretrained"
FINAL_EVAL_DIR="${OUTPUT_ROOT}/eval_planner_finetuned"

if [[ "${FORCE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}"
elif [[ -d "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -print -quit)" ]]; then
  echo "[ERROR] OUTPUT_ROOT is not empty: ${OUTPUT_ROOT}. Set FORCE=1 or use a new path." >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

COMMON_COLLECT=(
  --task Isaac-Imitation-G1-Latent-PerStepVQ-v0
  --algo IPMD
  --checkpoint "${LOW_LEVEL_CHECKPOINT}"
  --interface per_step_token_sequence
  --motion_manifest "${MOTION_MANIFEST}"
  --num_envs "${NUM_ENVS}"
  --steps "${COLLECT_PLANNER_STEPS}"
  --seed "${SEED}"
  --state_history_steps "${STATE_HISTORY_STEPS}"
  --planner_interval_steps "${PLANNER_INTERVAL_STEPS}"
  --command_past_steps 0
  --command_future_steps 9
  --reset_schedule sequential
  --headless
  --kit_args=--/app/extensions/fsWatcherEnabled=false
)

pixi run -e isaaclab python \
  experiments/interface_baselines/collect_interface_rollout_samples.py \
  "${COMMON_COLLECT[@]}" \
  --output_dir "${DEMO_DIR}"

pixi run -e isaaclab python \
  experiments/interface_baselines/collect_interface_rollout_samples.py \
  "${COMMON_COLLECT[@]}" \
  --control_steps "${EVAL_CONTROL_STEPS}" \
  --evaluation_only \
  --stop_after_done \
  --output_dir "${ORACLE_EVAL_DIR}"

pixi run python \
  experiments/interface_baselines/train_categorical_token_planner.py \
  --samples_dir "${DEMO_DIR}/rollout_training_samples" \
  --output_dir "${PRETRAIN_DIR}" \
  --state_key expert_planner_state \
  --device auto \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_updates "${PRETRAIN_UPDATES}" \
  --max_samples "${SAMPLE_BUDGET}" \
  --model_size "${MODEL_SIZE}"

pixi run -e isaaclab python \
  experiments/interface_baselines/collect_interface_rollout_samples.py \
  "${COMMON_COLLECT[@]}" \
  --control_steps "${EVAL_CONTROL_STEPS}" \
  --evaluation_only \
  --stop_after_done \
  --planner_checkpoint "${PRETRAIN_DIR}/checkpoints/latest.pt" \
  --output_dir "${PRETRAIN_EVAL_DIR}"

pixi run -e isaaclab python \
  experiments/interface_baselines/collect_interface_rollout_samples.py \
  "${COMMON_COLLECT[@]}" \
  --planner_checkpoint "${PRETRAIN_DIR}/checkpoints/latest.pt" \
  --output_dir "${ROLLOUT_DIR}"

pixi run python experiments/interface_baselines/merge_planner_samples.py \
  --source "${DEMO_DIR}/rollout_training_samples" \
  --source_limit "${SAMPLE_BUDGET}" \
  --source "${ROLLOUT_DIR}/rollout_training_samples" \
  --source_limit "${SAMPLE_BUDGET}" \
  --seed "${SEED}" \
  --output_dir "${MERGED_DIR}"

pixi run python \
  experiments/interface_baselines/train_categorical_token_planner.py \
  --samples_dir "${MERGED_DIR}" \
  --output_dir "${FINAL_DIR}" \
  --state_key planner_state \
  --checkpoint "${PRETRAIN_DIR}/checkpoints/latest.pt" \
  --device auto \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_updates "${FINETUNE_UPDATES}" \
  --model_size "${MODEL_SIZE}"

pixi run -e isaaclab python \
  experiments/interface_baselines/collect_interface_rollout_samples.py \
  "${COMMON_COLLECT[@]}" \
  --control_steps "${EVAL_CONTROL_STEPS}" \
  --evaluation_only \
  --stop_after_done \
  --planner_checkpoint "${FINAL_DIR}/checkpoints/latest.pt" \
  --output_dir "${FINAL_EVAL_DIR}"

echo "[INFO] Per-step token comparison artifacts: ${OUTPUT_ROOT}"
