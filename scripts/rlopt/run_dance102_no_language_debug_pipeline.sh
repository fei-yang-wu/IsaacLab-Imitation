#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
LOW_LEVEL_ALGO="${LOW_LEVEL_ALGO:-IPMD_BILINEAR}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MANIFEST_PATH="${MANIFEST_PATH:-data/unitree/manifests/g1_unitree_dance102_manifest.json}"
DATASET_PATH="${DATASET_PATH:-data/unitree/g1_dance102_hl_diffsr}"
TRAJECTORY_NAME="${TRAJECTORY_NAME:-dance102}"
TRAJECTORY_SOURCE="${TRAJECTORY_SOURCE:-data/unitree/npz/g1/G1_Take_102.bvh_60hz.npz}"
LANGUAGE_CONDITION="${LANGUAGE_CONDITION:-none}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
# 9 past + current = 10 states in the planner condition window.
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
Z_DIM="${Z_DIM:-256}"
DIFFSR_FEATURE_DIM="${DIFFSR_FEATURE_DIM:-128}"
DIFFSR_EMBED_DIM="${DIFFSR_EMBED_DIM:-512}"
SKILL_UPDATES="${SKILL_UPDATES:-5000}"
SKILL_BATCH_SIZE="${SKILL_BATCH_SIZE:-8192}"
PLANNER_UPDATES="${PLANNER_UPDATES:-5000}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-8192}"
PLANNER_TYPE="${PLANNER_TYPE:-flow_matching}"
PLANNER_FLOW_STEPS="${PLANNER_FLOW_STEPS:-16}"
PLANNER_FLOW_TIME_DIM="${PLANNER_FLOW_TIME_DIM:-64}"
PLANNER_FLOW_TRAIN_NOISE_STD="${PLANNER_FLOW_TRAIN_NOISE_STD:-1.0}"
PLANNER_FLOW_INFERENCE_NOISE_STD="${PLANNER_FLOW_INFERENCE_NOISE_STD:-1.0}"
PLANNER_EVAL_FLOW_NOISE_STD="${PLANNER_EVAL_FLOW_NOISE_STD:-0.0}"
LOW_LEVEL_MAX_ITERATIONS="${LOW_LEVEL_MAX_ITERATIONS:-10000}"
LOW_LEVEL_VIDEO_LENGTH="${LOW_LEVEL_VIDEO_LENGTH:-500}"
LOW_LEVEL_VIDEO_INTERVAL="${LOW_LEVEL_VIDEO_INTERVAL:-2500}"
EVAL_VIDEO_LENGTH="${EVAL_VIDEO_LENGTH:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000000}"
LOGGER_BACKEND="${LOGGER_BACKEND:-}"
RUN_M1_EVAL="${RUN_M1_EVAL:-1}"
SKIP_SKILL="${SKIP_SKILL:-0}"
SKIP_PLANNER="${SKIP_PLANNER:-0}"
SKIP_LOW_LEVEL="${SKIP_LOW_LEVEL:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_dance102_h${HORIZON_STEPS}_hist$((STATE_HISTORY_STEPS + 1))_no_language_flow}"
RUN_ROOT="${RUN_ROOT:-logs/dance102_single_trajectory_debug/${RUN_ID}}"
MANIFEST_ABS="$(realpath "${MANIFEST_PATH}")"
DATASET_ABS="$(realpath -m "${DATASET_PATH}")"
RUN_ROOT_ABS="$(mkdir -p "${RUN_ROOT}" && realpath "${RUN_ROOT}")"
COMMAND_LOG="${RUN_ROOT_ABS}/commands.sh"

mkdir -p "${RUN_ROOT_ABS}"
: > "${COMMAND_LOG}"
STDOUT_LOG="${RUN_ROOT_ABS}/pipeline.stdout.log"
exec > >(tee -a "${STDOUT_LOG}") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_ROOT_ABS}/pipeline.log"
}

run_cmd() {
    log "RUN: $*"
    printf '%q ' "$@" >> "${COMMAND_LOG}"
    printf '\n' >> "${COMMAND_LOG}"
    "$@"
}

latest_checkpoint_from_log_dir() {
    local log_dir="$1"
    find "${log_dir}/models" -maxdepth 1 -type f -name 'model_step_*.pt' -printf '%f %p\n' \
        | sort -V \
        | tail -n 1 \
        | cut -d' ' -f2-
}

cat > "${RUN_ROOT_ABS}/metadata.txt" <<EOF
trajectory_name=${TRAJECTORY_NAME}
trajectory_source=${TRAJECTORY_SOURCE}
manifest_path=${MANIFEST_ABS}
dataset_path=${DATASET_ABS}
language_condition=${LANGUAGE_CONDITION}
system0=low_level_policy
system1=no_language_history_flow_planner
system2=absent
horizon_steps=${HORIZON_STEPS}
planner_state_history_steps=${STATE_HISTORY_STEPS}
planner_condition_window_states=$((STATE_HISTORY_STEPS + 1))
latent_skill_action_dim=${Z_DIM}
num_envs=${NUM_ENVS}
seed=${SEED}
EOF

log "Dance102 no-language debug pipeline"
log "trajectory=${TRAJECTORY_NAME} manifest=${MANIFEST_ABS} dataset=${DATASET_ABS} language=${LANGUAGE_CONDITION}"
log "planner condition: $((STATE_HISTORY_STEPS + 1)) state frames, latent action z_dim=${Z_DIM}, planner_type=${PLANNER_TYPE}"
log "run root: ${RUN_ROOT_ABS}"

SKILL_DIR="${SKILL_DIR:-${RUN_ROOT_ABS}/skill_encoder_h${HORIZON_STEPS}_z${Z_DIM}}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-${SKILL_DIR}/checkpoints/latest.pt}"
if [[ "${SKIP_SKILL}" == "1" ]]; then
    if [[ ! -f "${SKILL_CHECKPOINT}" ]]; then
        log "SKIP_SKILL=1 but SKILL_CHECKPOINT does not exist: ${SKILL_CHECKPOINT}"
        exit 1
    fi
    log "Skipping skill encoder training; using ${SKILL_CHECKPOINT}"
else
    run_cmd "${PYTHON_BIN}" scripts/rlopt/train_hl_skill_diffsr.py \
        --headless \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num_envs "${NUM_ENVS}" \
        --seed "${SEED}" \
        --output_dir "${SKILL_DIR}" \
        --horizon_steps "${HORIZON_STEPS}" \
        --encoder_window_mode intermediate \
        --z_dim "${Z_DIM}" \
        --diffsr_feature_dim "${DIFFSR_FEATURE_DIM}" \
        --diffsr_embed_dim "${DIFFSR_EMBED_DIM}" \
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
        "env.lafan1_manifest_path=${MANIFEST_ABS}" \
        "env.dataset_path=${DATASET_ABS}" \
        "env.refresh_zarr_dataset=true"
fi
if [[ ! -f "${SKILL_CHECKPOINT}" ]]; then
    log "Missing skill checkpoint: ${SKILL_CHECKPOINT}"
    exit 1
fi
log "Skill encoder checkpoint: ${SKILL_CHECKPOINT}"

PLANNER_DIR="${PLANNER_DIR:-${RUN_ROOT_ABS}/planner_${PLANNER_TYPE}_no_language_hist$((STATE_HISTORY_STEPS + 1))}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${PLANNER_DIR}/checkpoints/latest.pt}"
if [[ "${SKIP_PLANNER}" == "1" ]]; then
    if [[ ! -f "${PLANNER_CHECKPOINT}" ]]; then
        log "SKIP_PLANNER=1 but PLANNER_CHECKPOINT does not exist: ${PLANNER_CHECKPOINT}"
        exit 1
    fi
    log "Skipping planner training; using ${PLANNER_CHECKPOINT}"
else
    run_cmd "${PYTHON_BIN}" scripts/rlopt/train_skill_commander.py \
        --headless \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num_envs "${NUM_ENVS}" \
        --seed "${SEED}" \
        --output_dir "${PLANNER_DIR}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --no_language \
        --state_history_steps "${STATE_HISTORY_STEPS}" \
        --planner_type "${PLANNER_TYPE}" \
        --generator_hidden_dims 1024 512 512 \
        --flow_num_inference_steps "${PLANNER_FLOW_STEPS}" \
        --flow_time_embed_dim "${PLANNER_FLOW_TIME_DIM}" \
        --flow_train_noise_std "${PLANNER_FLOW_TRAIN_NOISE_STD}" \
        --flow_inference_noise_std "${PLANNER_FLOW_INFERENCE_NOISE_STD}" \
        --batch_size "${PLANNER_BATCH_SIZE}" \
        --num_updates "${PLANNER_UPDATES}" \
        --log_interval 100 \
        --eval_batches 4 \
        --eval_batch_size "${PLANNER_BATCH_SIZE}" \
        --train_split all \
        --eval_split all \
        --eval_trajectory_fraction 0.5 \
        --trajectory_split_seed "${SEED}" \
        "env.lafan1_manifest_path=${MANIFEST_ABS}" \
        "env.dataset_path=${DATASET_ABS}" \
        "env.refresh_zarr_dataset=false"
fi
if [[ ! -f "${PLANNER_CHECKPOINT}" ]]; then
    log "Missing planner checkpoint: ${PLANNER_CHECKPOINT}"
    exit 1
fi
log "Planner checkpoint: ${PLANNER_CHECKPOINT}"

if [[ "${RUN_M1_EVAL}" == "1" ]]; then
    run_cmd "${PYTHON_BIN}" scripts/rlopt/eval_skill_commander_m1.py \
        --headless \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num_envs "${NUM_ENVS}" \
        --seed "${SEED}" \
        --checkpoint "${PLANNER_CHECKPOINT}" \
        --output_dir "${RUN_ROOT_ABS}/m1_eval_planner_no_language" \
        --batch_size "${PLANNER_BATCH_SIZE}" \
        --eval_batches 4 \
        --splits all \
        --per_trajectory \
        --trajectory_ranks 0 \
        --per_trajectory_batch_size 1024 \
        --per_trajectory_batches 4 \
        --flow_inference_noise_std "${PLANNER_EVAL_FLOW_NOISE_STD}" \
        --flow_num_inference_steps "${PLANNER_FLOW_STEPS}" \
        "env.lafan1_manifest_path=${MANIFEST_ABS}" \
        "env.dataset_path=${DATASET_ABS}" \
        "env.refresh_zarr_dataset=false"
fi

COMMON_LATENT_OVERRIDES=(
    "env.lafan1_manifest_path=${MANIFEST_ABS}"
    "env.dataset_path=${DATASET_ABS}"
    "env.refresh_zarr_dataset=false"
    "env.latent_command_dim=$((Z_DIM + 2))"
    "agent.ipmd.latent_dim=$((Z_DIM + 2))"
    "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
    "agent.ipmd.hl_skill_command_mode=z"
    "agent.ipmd.latent_steps_min=${HORIZON_STEPS}"
    "agent.ipmd.latent_steps_max=${HORIZON_STEPS}"
    "agent.ipmd.latent_learning.command_phase_mode=sin_cos"
    "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
    "agent.ipmd.latent_learning.code_period=${HORIZON_STEPS}"
    "agent.ipmd.reward_loss_coeff=0.0"
    "agent.ipmd.reward_l2_coeff=0.0"
    "agent.ipmd.reward_grad_penalty_coeff=0.0"
    "agent.ipmd.reward_logit_reg_coeff=0.0"
    "agent.ipmd.reward_param_weight_decay_coeff=0.0"
)

LOW_LEVEL_LOG_DIR="${LOW_LEVEL_LOG_DIR:-}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-}"
if [[ "${SKIP_LOW_LEVEL}" == "1" ]]; then
    if [[ -z "${LOW_LEVEL_CHECKPOINT}" || ! -f "${LOW_LEVEL_CHECKPOINT}" ]]; then
        log "SKIP_LOW_LEVEL=1 requires LOW_LEVEL_CHECKPOINT to point at an existing model checkpoint."
        exit 1
    fi
    LOW_LEVEL_LOG_DIR="$(cd "$(dirname "${LOW_LEVEL_CHECKPOINT}")/.." && pwd)"
    log "Skipping low-level training; using ${LOW_LEVEL_CHECKPOINT}"
else
    LOW_LEVEL_MARKER="${RUN_ROOT_ABS}/low_level_train_started.marker"
    touch "${LOW_LEVEL_MARKER}"
    run_cmd "${PYTHON_BIN}" scripts/rlopt/train.py \
        --headless \
        --video \
        --video_length "${LOW_LEVEL_VIDEO_LENGTH}" \
        --video_interval "${LOW_LEVEL_VIDEO_INTERVAL}" \
        --device "${DEVICE}" \
        --num_envs "${NUM_ENVS}" \
        --task "${TASK}" \
        --algo "${LOW_LEVEL_ALGO}" \
        --seed "${SEED}" \
        --max_iterations "${LOW_LEVEL_MAX_ITERATIONS}" \
        "agent.logger.backend=${LOGGER_BACKEND}" \
        "agent.logger.project_name=G1-Imitation-Dance102-Debug" \
        "agent.logger.exp_name=${RUN_ID}_oracle_low_level" \
        "agent.logger.video=true" \
        "agent.save_interval=${SAVE_INTERVAL}" \
        "agent.ipmd.command_source=hl_skill" \
        "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
        "agent.ipmd.hl_skill_finetune_enabled=false" \
        "${COMMON_LATENT_OVERRIDES[@]}"
    LOG_ROOT="${REPO_ROOT}/logs/rlopt/${LOW_LEVEL_ALGO,,}/${TASK}"
    LOW_LEVEL_LOG_DIR="$(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -newer "${LOW_LEVEL_MARKER}" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "${LOW_LEVEL_LOG_DIR}" ]]; then
        log "Could not locate low-level log dir under ${LOG_ROOT}."
        exit 1
    fi
    LOW_LEVEL_CHECKPOINT="$(latest_checkpoint_from_log_dir "${LOW_LEVEL_LOG_DIR}")"
fi
if [[ -z "${LOW_LEVEL_CHECKPOINT}" || ! -f "${LOW_LEVEL_CHECKPOINT}" ]]; then
    log "Could not locate low-level checkpoint in ${LOW_LEVEL_LOG_DIR}."
    exit 1
fi
log "Low-level log dir: ${LOW_LEVEL_LOG_DIR}"
log "Low-level checkpoint: ${LOW_LEVEL_CHECKPOINT}"
printf '%s\n' "${LOW_LEVEL_LOG_DIR}" > "${RUN_ROOT_ABS}/low_level_log_dir.txt"
printf '%s\n' "${LOW_LEVEL_CHECKPOINT}" > "${RUN_ROOT_ABS}/low_level_checkpoint.txt"

if [[ "${SKIP_EVAL}" == "1" ]]; then
    log "Skipping eval videos."
    exit 0
fi

run_cmd "${PYTHON_BIN}" scripts/rlopt/play.py \
    --headless \
    --video \
    --video_length "${EVAL_VIDEO_LENGTH}" \
    --output_dir "${RUN_ROOT_ABS}/video_eval_oracle_hl_skill" \
    --device "${DEVICE}" \
    --num_envs 1 \
    --task "${TASK}" \
    --algo "${LOW_LEVEL_ALGO}" \
    --seed "${SEED}" \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    "agent.ipmd.command_source=hl_skill" \
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
    "agent.ipmd.hl_skill_finetune_enabled=false" \
    "${COMMON_LATENT_OVERRIDES[@]}"

run_cmd "${PYTHON_BIN}" scripts/rlopt/play.py \
    --headless \
    --video \
    --video_length "${EVAL_VIDEO_LENGTH}" \
    --output_dir "${RUN_ROOT_ABS}/video_eval_trained_planner_no_language" \
    --device "${DEVICE}" \
    --num_envs 1 \
    --task "${TASK}" \
    --algo "${LOW_LEVEL_ALGO}" \
    --seed "${SEED}" \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    "agent.ipmd.command_source=skill_commander" \
    "agent.ipmd.skill_commander_checkpoint_path=${PLANNER_CHECKPOINT}" \
    "agent.ipmd.skill_commander_embeddings_path=" \
    "agent.ipmd.skill_commander_flow_num_inference_steps=${PLANNER_FLOW_STEPS}" \
    "agent.ipmd.skill_commander_flow_inference_noise_std=${PLANNER_EVAL_FLOW_NOISE_STD}" \
    "agent.ipmd.skill_commander_use_achieved_state=true" \
    "agent.ipmd.hl_skill_finetune_enabled=false" \
    "${COMMON_LATENT_OVERRIDES[@]}"

log "Pipeline complete. Videos are under:"
log "  oracle:  ${RUN_ROOT_ABS}/video_eval_oracle_hl_skill/videos/play"
log "  planner: ${RUN_ROOT_ABS}/video_eval_trained_planner_no_language/videos/play"
