#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="${SESSION:-lafan1-rollout-ft-all}"
MAIN_REPO="${MAIN_REPO:-/mnt/hsstorage/fwu91/Projects/SL/IsaacLab-Imitation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/planner_robustness/${TIMESTAMP}_lafan1_rollout_ft_all}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
ALGORITHM="${ALGORITHM:-IPMD}"
CHECKPOINT="${CHECKPOINT:-logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-06-11_23-21-31/models/model_step_4600037376.pt}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-logs/language_skill_generator/2026-06-18_flow_matching_lafan1_w25_z256_full/checkpoints/latest.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-logs/hl_skill_diffsr/lafan1_w25_z256_seed0_intermediate_pipeline_20260611_161049/checkpoints/latest.pt}"
MANIFEST="${MANIFEST:-${MAIN_REPO}/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-${MAIN_REPO}/data/lafan1/g1_hl_diffsr}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-${MAIN_REPO}/data/lafan1/language/g1_lafan1_name_embeddings.pt}"
RANKS="${RANKS:-all}"
LIMIT="${LIMIT:-}"
MAX_STEPS="${MAX_STEPS:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-256}"
FLOW_NUM_INFERENCE_STEPS="${FLOW_NUM_INFERENCE_STEPS:-16}"
FLOW_INFERENCE_NOISE_STD="${FLOW_INFERENCE_NOISE_STD:-0.0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RESUME="${RESUME:-1}"
VIDEO="${VIDEO:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/tmux_rollout_ft_all.log}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session '${SESSION}' already exists. Attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

cmd=(
  env OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE="${WANDB_MODE}"
  pixi run python scripts/rlopt/run_lafan1_rollout_ft_all.py
  --task "${TASK}"
  --algorithm "${ALGORITHM}"
  --checkpoint "${CHECKPOINT}"
  --planner_checkpoint "${PLANNER_CHECKPOINT}"
  --skill_checkpoint "${SKILL_CHECKPOINT}"
  --manifest "${MANIFEST}"
  --dataset_path "${DATASET_PATH}"
  --language_embeddings "${LANGUAGE_EMBEDDINGS}"
  --output_root "${OUTPUT_ROOT}"
  --ranks "${RANKS}"
  --max_steps "${MAX_STEPS}"
  --video_length "${VIDEO_LENGTH}"
  --finetune_updates "${FINETUNE_UPDATES}"
  --finetune_batch_size "${FINETUNE_BATCH_SIZE}"
  --flow_num_inference_steps "${FLOW_NUM_INFERENCE_STEPS}"
  --flow_inference_noise_std "${FLOW_INFERENCE_NOISE_STD}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  cmd+=(--continue_on_error)
fi
if [[ "${VIDEO}" != "1" ]]; then
  cmd+=(--no_video)
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_repo '%q' "${REPO_ROOT}"
printf -v quoted_log '%q' "${LOG_FILE}"

{
  echo "# Started: $(date -Is)"
  echo "# Session: ${SESSION}"
  echo "# Output: ${OUTPUT_ROOT}"
  echo "${quoted_cmd}"
} >> "${LOG_FILE}"

tmux new-session -d -s "${SESSION}" "cd ${quoted_repo} && ${quoted_cmd} 2>&1 | tee -a ${quoted_log}"

echo "Started tmux session: ${SESSION}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Tail: tail -f ${LOG_FILE}"
