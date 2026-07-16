#!/usr/bin/env bash
set -euo pipefail

# Local corrected-data qualification for the paper's DiffSR latent low-level
# interface. This trains no high-level planner and is not a paper-scale run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MANIFEST="${MANIFEST:-${ROOT_DIR}/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/logs/interface_baselines/phase3_diffsr_latent_10m_$(date +%Y%m%d_%H%M%S)}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
FRAMES_PER_ENV_BATCH="${FRAMES_PER_ENV_BATCH:-24}"
TARGET_FRAMES="${TARGET_FRAMES:-10000000}"
LOCAL_MAX_TARGET_FRAMES=50000000
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-40}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SEED="${SEED:-0}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
SKILL_UPDATES="${SKILL_UPDATES:-500}"
SKILL_BATCH_SIZE="${SKILL_BATCH_SIZE:-8192}"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "[ERROR] Manifest does not exist: ${MANIFEST}" >&2
    exit 2
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset cache does not exist: ${DATASET_PATH}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "[ERROR] OUTPUT_ROOT already exists: ${OUTPUT_ROOT}" >&2
    exit 2
fi
for value in \
    "${NUM_ENVS}" \
    "${FRAMES_PER_ENV_BATCH}" \
    "${TARGET_FRAMES}" \
    "${EVAL_NUM_ENVS}" \
    "${EVAL_STEPS}" \
    "${HORIZON_STEPS}" \
    "${Z_DIM}" \
    "${SKILL_UPDATES}" \
    "${SKILL_BATCH_SIZE}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] Expected a positive integer, got ${value}." >&2
        exit 2
    fi
done
if [[ "${HORIZON_STEPS}" != "10" || "${Z_DIM}" != "256" ]]; then
    echo "[ERROR] The focused protocol requires HORIZON_STEPS=10 and Z_DIM=256." >&2
    exit 2
fi
if (( TARGET_FRAMES > LOCAL_MAX_TARGET_FRAMES )); then
    echo "[ERROR] TARGET_FRAMES=${TARGET_FRAMES} exceeds the 50M local qualification limit." >&2
    echo "[HINT] Use Skynet for longer convergence or paper runs." >&2
    exit 2
fi

MANIFEST="$(realpath "${MANIFEST}")"
DATASET_PATH="$(realpath "${DATASET_PATH}")"
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(realpath "${OUTPUT_ROOT}")"
SKILL_DIR="${OUTPUT_ROOT}/skill_encoder_h10_z256"
SKILL_CHECKPOINT="${SKILL_DIR}/checkpoints/latest.pt"

FRAMES_PER_BATCH="$((NUM_ENVS * FRAMES_PER_ENV_BATCH))"
MAX_ITERATIONS="$(((TARGET_FRAMES + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))"
EFFECTIVE_FRAMES="$((MAX_ITERATIONS * FRAMES_PER_BATCH))"

pixi run python scripts/audit_g1_lafan1_body_frames.py \
    --manifest "${MANIFEST}" \
    --report "${OUTPUT_ROOT}/body_frame_audit.json"

env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py \
        --headless \
        --device cuda:0 \
        --task "${TASK}" \
        --num_envs "${NUM_ENVS}" \
        --seed "${SEED}" \
        --output_dir "${SKILL_DIR}" \
        --horizon_steps "${HORIZON_STEPS}" \
        --encoder_window_mode intermediate \
        --z_dim "${Z_DIM}" \
        --diffsr_feature_dim 128 \
        --diffsr_embed_dim 512 \
        --batch_size "${SKILL_BATCH_SIZE}" \
        --num_updates "${SKILL_UPDATES}" \
        --log_interval 100 \
        --eval_batches 4 \
        --eval_batch_size "${SKILL_BATCH_SIZE}" \
        --train_split all \
        --eval_split all \
        --eval_trajectory_fraction 0.5 \
        --trajectory_split_seed "${SEED}" \
        --reconstruction_eval \
        --window_probe_eval \
        --window_probe_train_batches 8 \
        --window_probe_eval_batches 4 \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${DATASET_PATH}" \
        env.refresh_zarr_dataset=false \
        >"${OUTPUT_ROOT}/skill_train.log" 2>&1

if [[ ! -f "${SKILL_CHECKPOINT}" ]]; then
    echo "[ERROR] Skill checkpoint was not produced: ${SKILL_CHECKPOINT}" >&2
    exit 2
fi

env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python scripts/rlopt/train.py \
        --headless \
        --device cuda:0 \
        --num_envs "${NUM_ENVS}" \
        --task "${TASK}" \
        --algo IPMD \
        --seed "${SEED}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --log_interval 10 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.logger.backend=csv \
        "agent.save_interval=${TARGET_FRAMES}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${DATASET_PATH}" \
        env.refresh_zarr_dataset=false \
        env.random_reset_step_min=0 \
        env.random_reset_step_max=200 \
        env.random_reset_full_trajectory=false \
        env.reconstructed_reference_action=true \
        "env.latent_command_dim=$((Z_DIM + 2))" \
        "agent.ipmd.latent_dim=$((Z_DIM + 2))" \
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}" \
        agent.ipmd.hl_skill_command_mode=z \
        "agent.ipmd.latent_steps_min=${HORIZON_STEPS}" \
        "agent.ipmd.latent_steps_max=${HORIZON_STEPS}" \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}" \
        "agent.ipmd.latent_learning.code_period=${HORIZON_STEPS}" \
        agent.ipmd.reward_loss_coeff=0.0 \
        agent.ipmd.reward_l2_coeff=0.0 \
        agent.ipmd.reward_grad_penalty_coeff=0.0 \
        agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0 \
        agent.ipmd.use_estimated_rewards_for_ppo=false \
        agent.ipmd.env_reward_weight=1.0 \
        agent.ipmd.bc_coef=0.0 \
        agent.ipmd.rollout_bc_coef=0.0 \
        >"${OUTPUT_ROOT}/low_level_train.log" 2>&1

RUN_NAME="$(sed -n 's/^Exact experiment name requested from command line: //p' "${OUTPUT_ROOT}/low_level_train.log" | tail -n 1)"
RUN_DIR="${ROOT_DIR}/logs/rlopt/ipmd/${TASK}/${RUN_NAME}"
LOW_LEVEL_CHECKPOINT="${RUN_DIR}/models/model_step_${EFFECTIVE_FRAMES}.pt"
if [[ -z "${RUN_NAME}" || ! -f "${LOW_LEVEL_CHECKPOINT}" ]]; then
    echo "[ERROR] Could not locate the latent low-level checkpoint." >&2
    exit 2
fi
printf '%s\n' "${LOW_LEVEL_CHECKPOINT}" >"${OUTPUT_ROOT}/low_level_checkpoint.txt"

pixi run python \
    experiments/interface_baselines/validate_latent_skill_checkpoint_binding.py \
    --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${SKILL_CHECKPOINT}" \
    --output_json "${OUTPUT_ROOT}/latent_skill_binding.json"

mkdir -p "${OUTPUT_ROOT}/oracle_low_level"
env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless \
        --device cuda:0 \
        --task "${TASK}" \
        --algorithm IPMD \
        --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --state_history_steps 9 \
        --output_dir "${OUTPUT_ROOT}/oracle_low_level" \
        --label corrected_lafan1_diffsr_latent_10m_oracle \
        --num_envs "${EVAL_NUM_ENVS}" \
        --max_steps "${EVAL_STEPS}" \
        --seed "${SEED}" \
        --metric_interval 1 \
        --keep_time_out \
        --extend_episode_length_for_max_steps \
        --keep_early_terminations \
        --disable_reward_clipping \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${DATASET_PATH}" \
        env.refresh_zarr_dataset=false \
        env.reset_schedule=sequential \
        env.wrap_steps=false \
        env.observations.policy.enable_corruption=false \
        "env.latent_command_dim=$((Z_DIM + 2))" \
        "agent.ipmd.latent_dim=$((Z_DIM + 2))" \
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}" \
        agent.ipmd.hl_skill_command_mode=z \
        "agent.ipmd.latent_steps_min=${HORIZON_STEPS}" \
        "agent.ipmd.latent_steps_max=${HORIZON_STEPS}" \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}" \
        "agent.ipmd.latent_learning.code_period=${HORIZON_STEPS}" \
        agent.ipmd.reward_loss_coeff=0.0 \
        agent.ipmd.reward_l2_coeff=0.0 \
        agent.ipmd.reward_grad_penalty_coeff=0.0 \
        agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0 \
        >"${OUTPUT_ROOT}/oracle_low_level/eval.log" 2>&1

pixi run python \
    experiments/interface_baselines/audit_diffsr_latent_qualification.py \
    --summary "${OUTPUT_ROOT}/oracle_low_level/summary.json" \
    --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${SKILL_CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --expected_dataset_path "${DATASET_PATH}" \
    --expected_num_envs "${EVAL_NUM_ENVS}" \
    --expected_steps "${EVAL_STEPS}" \
    --expected_seed "${SEED}" \
    --output_json "${OUTPUT_ROOT}/qualification_audit.json"

export MANIFEST DATASET_PATH OUTPUT_ROOT NUM_ENVS SEED TARGET_FRAMES
pixi run python experiments/interface_baselines/write_interface_run_provenance.py \
    --label phase3-corrected-lafan1-diffsr-latent-local-qualification \
    --output_json "${OUTPUT_ROOT}/run_provenance.json" \
    --result_root "${OUTPUT_ROOT}" \
    --env_key DATASET_PATH \
    --env_key TARGET_FRAMES \
    --env_key NUM_ENVS \
    --note "plain IPMD; fixed reset range 0-200; unchanged rewards and terminations" \
    --note "local DiffSR skill updates ${SKILL_UPDATES}; final protocol uses 5000"

echo "[INFO] DiffSR latent qualification artifacts: ${OUTPUT_ROOT}"
