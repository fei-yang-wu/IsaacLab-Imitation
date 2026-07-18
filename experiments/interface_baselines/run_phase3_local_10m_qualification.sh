#!/usr/bin/env bash
set -euo pipefail

# Train and oracle-evaluate the two Phase 3 latent-action low-level policies
# locally. This is a qualification run, not a paper-scale experiment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MANIFEST="${MANIFEST:-${ROOT_DIR}/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/data/lafan1/g1_hl_diffsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/logs/interface_baselines/phase3_local_10m}"
NUM_ENVS="${NUM_ENVS:-4096}"
FRAMES_PER_ENV_BATCH="${FRAMES_PER_ENV_BATCH:-24}"
TARGET_FRAMES="${TARGET_FRAMES:-10000000}"
LOCAL_MAX_TARGET_FRAMES=50000000
SEED="${SEED:-0}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-40}"
EVAL_CONTROL_STEPS="${EVAL_CONTROL_STEPS:-1000}"
ORACLE_SUCCESS_THRESHOLD="${ORACLE_SUCCESS_THRESHOLD:-0.8}"
BC_COEF="${BC_COEF:-0.0}"
BC_PRETRAIN_UPDATES="${BC_PRETRAIN_UPDATES:-0}"
ROLLOUT_BC_COEF="${ROLLOUT_BC_COEF:-0.0}"
RECONSTRUCTED_ACTION_MODE="${RECONSTRUCTED_ACTION_MODE:-next_pose}"
INTERFACES="${INTERFACES:-future_cvae per_step_token_sequence}"
METRICS_BACKEND="${METRICS_BACKEND:-csv}"

for value in \
  "${NUM_ENVS}" \
  "${FRAMES_PER_ENV_BATCH}" \
  "${TARGET_FRAMES}" \
  "${EVAL_NUM_ENVS}" \
  "${EVAL_CONTROL_STEPS}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Expected a positive integer, got ${value}." >&2
    exit 2
  fi
done
if [[ ! "${BC_PRETRAIN_UPDATES}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] BC_PRETRAIN_UPDATES must be a non-negative integer." >&2
  exit 2
fi
if (( TARGET_FRAMES > LOCAL_MAX_TARGET_FRAMES )); then
  echo "[ERROR] TARGET_FRAMES=${TARGET_FRAMES} exceeds the 50M local qualification limit." >&2
  echo "[HINT] Use Skynet for longer convergence or paper runs." >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Manifest does not exist: ${MANIFEST}" >&2
  exit 2
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "[ERROR] Dataset does not exist: ${DATASET_PATH}" >&2
  exit 2
fi

FRAMES_PER_BATCH="$((NUM_ENVS * FRAMES_PER_ENV_BATCH))"
MAX_ITERATIONS="$(((TARGET_FRAMES + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))"
EFFECTIVE_FRAMES="$((MAX_ITERATIONS * FRAMES_PER_BATCH))"
mkdir -p "${OUTPUT_ROOT}"
pixi run python scripts/audit_g1_lafan1_body_frames.py \
  --manifest "${MANIFEST}" \
  --report "${OUTPUT_ROOT}/body_frame_audit.json"

COMMON_OVERRIDES=(
  "env.lafan1_manifest_path=${MANIFEST}"
  "env.dataset_path=${DATASET_PATH}"
  "env.refresh_zarr_dataset=false"
  "env.random_reset_step_min=0"
  "env.random_reset_step_max=200"
  "env.random_reset_full_trajectory=false"
  "agent.ipmd.reward_loss_coeff=0.0"
  "agent.ipmd.reward_l2_coeff=0.0"
  "agent.ipmd.reward_grad_penalty_coeff=0.0"
  "agent.ipmd.reward_logit_reg_coeff=0.0"
  "agent.ipmd.reward_param_weight_decay_coeff=0.0"
  "agent.ipmd.use_estimated_rewards_for_ppo=false"
  "agent.ipmd.env_reward_weight=1.0"
  "agent.ipmd.bc_coef=${BC_COEF}"
  "agent.ipmd.rollout_bc_coef=${ROLLOUT_BC_COEF}"
  "agent.ipmd.bc_pretrain_updates=${BC_PRETRAIN_UPDATES}"
  "env.reconstructed_reference_action=true"
  "env.reconstructed_reference_action_mode=${RECONSTRUCTED_ACTION_MODE}"
  "agent.save_interval=${TARGET_FRAMES}"
  "agent.logger.backend=${METRICS_BACKEND}"
)

train_and_evaluate() {
  local interface="$1"
  local task="$2"
  local run_root="${OUTPUT_ROOT}/${interface}"
  local task_log_root="${ROOT_DIR}/logs/rlopt/ipmd/${task}"
  mkdir -p "${run_root}"

  local started_at
  started_at="$(date +%s)"
  echo "[INFO] Training ${interface}: ${MAX_ITERATIONS} iterations, ${EFFECTIVE_FRAMES} frames."
  env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python scripts/rlopt/train.py \
      --task "${task}" \
      --algo IPMD \
      --num_envs "${NUM_ENVS}" \
      --max_iterations "${MAX_ITERATIONS}" \
      --seed "${SEED}" \
      --headless \
      --kit_args=--/app/extensions/fsWatcherEnabled=false \
      "${COMMON_OVERRIDES[@]}" 2>&1 | tee "${run_root}/train.log"

  local run_dir
  run_dir="$(
    find "${task_log_root}" -mindepth 1 -maxdepth 1 -type d \
      -newermt "@${started_at}" -printf '%T@ %p\n' \
      | sort -n | tail -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${run_dir}" ]]; then
    echo "[ERROR] Could not locate the new ${task} training directory." >&2
    exit 2
  fi

  local checkpoint
  checkpoint="$(
    find "${run_dir}/models" -maxdepth 1 -type f -name 'model_step_*.pt' \
      -printf '%f %p\n' \
      | sort -t_ -k3,3n | tail -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${checkpoint}" ]]; then
    echo "[ERROR] No checkpoint produced for ${interface}." >&2
    exit 2
  fi
  printf '%s\n' "${checkpoint}" > "${run_root}/checkpoint.txt"

  pixi run -e isaaclab python \
    experiments/interface_baselines/collect_interface_rollout_samples.py \
      --headless \
      --task "${task}" \
      --algo IPMD \
      --checkpoint "${checkpoint}" \
      --interface "${interface}" \
      --motion_manifest "${MANIFEST}" \
      --dataset_path "${DATASET_PATH}" \
      --num_envs "${EVAL_NUM_ENVS}" \
      --steps 1 \
      --control_steps "${EVAL_CONTROL_STEPS}" \
      --seed "${SEED}" \
      --state_history_steps 9 \
      --planner_interval_steps 10 \
      --command_past_steps 0 \
      --command_future_steps 9 \
      --reset_schedule sequential \
      --evaluation_only \
      --stop_after_done \
      --output_dir "${run_root}/eval_oracle" \
      --kit_args=--/app/extensions/fsWatcherEnabled=false
}

read -r -a INTERFACE_LIST <<< "${INTERFACES}"
AUDIT_INTERFACE_ARGS=()
for interface in "${INTERFACE_LIST[@]}"; do
  case "${interface}" in
    future_cvae)
      train_and_evaluate future_cvae Isaac-Imitation-G1-Latent-FutureCVAE-v0
      AUDIT_INTERFACE_ARGS+=(
        --future_checkpoint "$(<"${OUTPUT_ROOT}/future_cvae/checkpoint.txt")"
        --future_oracle_summary "${OUTPUT_ROOT}/future_cvae/eval_oracle/summary.json"
      )
      ;;
    per_step_token_sequence)
      train_and_evaluate per_step_token_sequence Isaac-Imitation-G1-Latent-PerStepVQ-v0
      AUDIT_INTERFACE_ARGS+=(
        --token_checkpoint "$(<"${OUTPUT_ROOT}/per_step_token_sequence/checkpoint.txt")"
        --token_oracle_summary \
          "${OUTPUT_ROOT}/per_step_token_sequence/eval_oracle/summary.json"
      )
      ;;
    *)
      echo "[ERROR] Unsupported interface: ${interface}" >&2
      exit 2
      ;;
  esac
done

pixi run python experiments/interface_baselines/audit_phase3_local_10m.py \
  --interfaces "${INTERFACE_LIST[@]}" \
  "${AUDIT_INTERFACE_ARGS[@]}" \
  --manifest "${MANIFEST}" \
  --dataset_path "${DATASET_PATH}" \
  --expected_seed "${SEED}" \
  --target_frames "${TARGET_FRAMES}" \
  --frames_per_batch "${FRAMES_PER_BATCH}" \
  --eval_control_steps "${EVAL_CONTROL_STEPS}" \
  --oracle_success_threshold "${ORACLE_SUCCESS_THRESHOLD}" \
  --bc_coef "${BC_COEF}" \
  --rollout_bc_coef "${ROLLOUT_BC_COEF}" \
  --bc_pretrain_updates "${BC_PRETRAIN_UPDATES}" \
  --reconstructed_action_mode "${RECONSTRUCTED_ACTION_MODE}" \
  --output_json "${OUTPUT_ROOT}/qualification_audit.json"

echo "[INFO] Phase 3 local qualification artifacts: ${OUTPUT_ROOT}"
