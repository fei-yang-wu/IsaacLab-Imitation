#!/usr/bin/env bash
set -euo pipefail

# Continue the corrected-data direct vanilla tracker for one bounded local
# qualification block, then run the strict 40-motion oracle evaluation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
BASE_CUMULATIVE_FRAMES="${BASE_CUMULATIVE_FRAMES:-0}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/logs/interface_baselines/phase3_vanilla_qualification_$(date +%Y%m%d_%H%M%S)}"
TARGET_FRAMES="${TARGET_FRAMES:-10000000}"
LOCAL_MAX_CUMULATIVE_FRAMES=50000000
NUM_ENVS="${NUM_ENVS:-4096}"
FRAMES_PER_ENV_BATCH="${FRAMES_PER_ENV_BATCH:-24}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-40}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SEED="${SEED:-0}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.8}"

for path in "${MANIFEST}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file does not exist: ${path}" >&2
        exit 2
    fi
done
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "[ERROR] RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 2
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset cache does not exist: ${DATASET_PATH}" >&2
    exit 2
fi
for value in \
    "${BASE_CUMULATIVE_FRAMES}" \
    "${TARGET_FRAMES}" \
    "${NUM_ENVS}" \
    "${FRAMES_PER_ENV_BATCH}" \
    "${EVAL_NUM_ENVS}" \
    "${EVAL_STEPS}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Expected a non-negative integer, got ${value}." >&2
        exit 2
    fi
done
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "[ERROR] OUTPUT_ROOT already exists: ${OUTPUT_ROOT}" >&2
    exit 2
fi

MANIFEST="$(realpath "${MANIFEST}")"
DATASET_PATH="$(realpath "${DATASET_PATH}")"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    RESUME_CHECKPOINT="$(realpath "${RESUME_CHECKPOINT}")"
fi
mkdir -p "${OUTPUT_ROOT}/direct_oracle"
OUTPUT_ROOT="$(realpath "${OUTPUT_ROOT}")"

FRAMES_PER_BATCH="$((NUM_ENVS * FRAMES_PER_ENV_BATCH))"
MAX_ITERATIONS="$(((TARGET_FRAMES + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))"
EFFECTIVE_FRAMES="$((MAX_ITERATIONS * FRAMES_PER_BATCH))"
CUMULATIVE_FRAMES="$((BASE_CUMULATIVE_FRAMES + EFFECTIVE_FRAMES))"
MAX_LOCAL_EFFECTIVE_FRAMES="$((((LOCAL_MAX_CUMULATIVE_FRAMES + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH) * FRAMES_PER_BATCH))"
if (( TARGET_FRAMES > LOCAL_MAX_CUMULATIVE_FRAMES )); then
    echo "[ERROR] TARGET_FRAMES=${TARGET_FRAMES} exceeds the 50M local qualification limit." >&2
    echo "[HINT] Use Skynet for longer convergence or paper runs." >&2
    exit 2
fi
if (( CUMULATIVE_FRAMES > MAX_LOCAL_EFFECTIVE_FRAMES )); then
    echo "[ERROR] This block would reach ${CUMULATIVE_FRAMES} cumulative frames, above the approximately 50M local limit." >&2
    echo "[HINT] Stop local continuation and use Skynet for convergence." >&2
    exit 2
fi

RESUME_ARGS=()
RESUME_NOTE="fresh training"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--checkpoint "${RESUME_CHECKPOINT}")
    RESUME_NOTE="resume checkpoint ${RESUME_CHECKPOINT}"
elif (( BASE_CUMULATIVE_FRAMES != 0 )); then
    echo "[ERROR] BASE_CUMULATIVE_FRAMES must be 0 without RESUME_CHECKPOINT." >&2
    exit 2
fi

pixi run python scripts/audit_g1_lafan1_body_frames.py \
    --manifest "${MANIFEST}" \
    --report "${OUTPUT_ROOT}/body_frame_audit.json"

env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python scripts/rlopt/train.py \
        --task Isaac-Imitation-G1-v0 \
        --algo IPMD \
        "${RESUME_ARGS[@]}" \
        --num_envs "${NUM_ENVS}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --seed "${SEED}" \
        --headless \
        --log_interval 10 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${DATASET_PATH}" \
        env.refresh_zarr_dataset=false \
        env.random_reset_step_min=0 \
        env.random_reset_step_max=200 \
        env.random_reset_full_trajectory=false \
        env.command_hold_steps=0 \
        env.latent_patch_past_steps=0 \
        env.latent_patch_future_steps=0 \
        env.reconstructed_reference_action=true \
        agent.command_space=single_frame_full_body \
        agent.ipmd.use_latent_command=false \
        agent.ipmd.reward_loss_coeff=0.0 \
        agent.ipmd.reward_l2_coeff=0.0 \
        agent.ipmd.reward_grad_penalty_coeff=0.0 \
        agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0 \
        agent.ipmd.use_estimated_rewards_for_ppo=false \
        agent.ipmd.env_reward_weight=1.0 \
        agent.ipmd.bc_coef=0.0 \
        agent.ipmd.rollout_bc_coef=0.0 \
        'agent.value_function.num_cells=[768,512,256]' \
        "agent.save_interval=${TARGET_FRAMES}" \
        agent.logger.backend=csv \
        >"${OUTPUT_ROOT}/train.log" 2>&1

RUN_NAME="$(sed -n 's/^Exact experiment name requested from command line: //p' "${OUTPUT_ROOT}/train.log" | tail -n 1)"
if [[ -z "${RUN_NAME}" ]]; then
    echo "[ERROR] Could not recover the training run name." >&2
    exit 2
fi
RUN_DIR="${ROOT_DIR}/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/${RUN_NAME}"
CHECKPOINT="${RUN_DIR}/models/model_step_${EFFECTIVE_FRAMES}.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Expected checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 2
fi
printf '%s\n' "${CHECKPOINT}" >"${OUTPUT_ROOT}/checkpoint.txt"

env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python \
        experiments/command_space_ablation/evaluate_checkpoint.py \
        --headless \
        --task Isaac-Imitation-G1-v0 \
        --algorithm IPMD \
        --checkpoint "${CHECKPOINT}" \
        --policy_only_checkpoint \
        --label "corrected_lafan1_vanilla_${CUMULATIVE_FRAMES}_direct" \
        --command_space single_frame_full_body \
        --low_level_command_mode native \
        --command_observation_source reference \
        --motion_manifest "${MANIFEST}" \
        --dataset_path "${DATASET_PATH}" \
        --num_envs "${EVAL_NUM_ENVS}" \
        --steps "${EVAL_STEPS}" \
        --seed "${SEED}" \
        --reset_schedule sequential \
        --reference_start_frame 0 \
        --output_json "${OUTPUT_ROOT}/direct_oracle/summary.json" \
        --output_csv "${OUTPUT_ROOT}/direct_oracle/summary.csv" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        >"${OUTPUT_ROOT}/direct_oracle/eval.log" 2>&1

pixi run python \
    experiments/interface_baselines/audit_vanilla_tracker_qualification.py \
    --summary "${OUTPUT_ROOT}/direct_oracle/summary.json" \
    --checkpoint "${CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --expected_dataset_path "${DATASET_PATH}" \
    --expected_num_envs "${EVAL_NUM_ENVS}" \
    --expected_steps "${EVAL_STEPS}" \
    --expected_seed "${SEED}" \
    --success_threshold "${SUCCESS_THRESHOLD}" \
    --output_json "${OUTPUT_ROOT}/qualification_audit.json"

CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)"
export DATASET_PATH TARGET_FRAMES NUM_ENVS MANIFEST OUTPUT_ROOT SEED
pixi run python experiments/interface_baselines/write_interface_run_provenance.py \
    --label phase3-corrected-lafan1-vanilla-qualification-block \
    --output_json "${OUTPUT_ROOT}/run_provenance.json" \
    --result_root "${OUTPUT_ROOT}" \
    --env_key DATASET_PATH \
    --env_key TARGET_FRAMES \
    --env_key NUM_ENVS \
    --note "${RESUME_NOTE}" \
    --note "fixed reset range 0-200; reward and termination config unchanged" \
    --note "${CUMULATIVE_FRAMES} cumulative frames after this block" \
    --note "checkpoint sha256 ${CHECKPOINT_SHA256}"

echo "[INFO] Qualification artifacts: ${OUTPUT_ROOT}"
