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
export TMPDIR="${TMPDIR:-${REPO_ROOT}/logs/tmp/isaaclab_pipeline}"
export ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT="${ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT:-${REPO_ROOT}/logs/tmp/isaaclab_unitree_usd}"
mkdir -p "${TMPDIR}" "${ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
LOW_LEVEL_ALGO="${LOW_LEVEL_ALGO:-IPMD}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
SKILL_NUM_ENVS="${SKILL_NUM_ENVS:-16}"
PLANNER_NUM_ENVS="${PLANNER_NUM_ENVS:-16}"
LOW_LEVEL_NUM_ENVS="${LOW_LEVEL_NUM_ENVS:-4096}"
MANIFEST_PATH="${MANIFEST_PATH:-data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-data/lafan1/g1_hl_diffsr}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-data/lafan1/language/g1_lafan1_minilm_attribute_codex_storyboard_v1.pt}"
LANGUAGE_CONDITION="${LANGUAGE_CONDITION:-caption_attribute_minilm_codex_storyboard_v1}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
Z_DIM="${Z_DIM:-256}"
DIFFSR_FEATURE_DIM="${DIFFSR_FEATURE_DIM:-256}"
DIFFSR_EMBED_DIM="${DIFFSR_EMBED_DIM:-512}"
DIFFSR_PHI_PARAMETERIZATION="${DIFFSR_PHI_PARAMETERIZATION:-concat}"
SKILL_ENCODER_WINDOW_MODE="${SKILL_ENCODER_WINDOW_MODE:-intermediate}"
HL_SKILL_COMMAND_MODE="${HL_SKILL_COMMAND_MODE:-z}"
SKILL_UPDATES="${SKILL_UPDATES:-50000}"
SKILL_BATCH_SIZE="${SKILL_BATCH_SIZE:-8192}"
SKILL_REG_COEFF="${SKILL_REG_COEFF:-0.001}"
PLANNER_UPDATES="${PLANNER_UPDATES:-10000}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-8192}"
PLANNER_TYPE="${PLANNER_TYPE:-flow_matching}"
PLANNER_FLOW_STEPS="${PLANNER_FLOW_STEPS:-16}"
PLANNER_FLOW_TIME_DIM="${PLANNER_FLOW_TIME_DIM:-64}"
PLANNER_FLOW_TRAIN_NOISE_STD="${PLANNER_FLOW_TRAIN_NOISE_STD:-1.0}"
PLANNER_FLOW_INFERENCE_NOISE_STD="${PLANNER_FLOW_INFERENCE_NOISE_STD:-1.0}"
PLANNER_EVAL_FLOW_NOISE_STD="${PLANNER_EVAL_FLOW_NOISE_STD:-0.0}"
PLANNER_LANGUAGE_CONTRASTIVE_COEFF="${PLANNER_LANGUAGE_CONTRASTIVE_COEFF:-0.1}"
PLANNER_LANGUAGE_CONTRASTIVE_MARGIN="${PLANNER_LANGUAGE_CONTRASTIVE_MARGIN:-0.05}"
PLANNER_STATE_NOISE_STD="${PLANNER_STATE_NOISE_STD:-0.0}"
PLANNER_STATE_FEATURE_DROPOUT_PROB="${PLANNER_STATE_FEATURE_DROPOUT_PROB:-0.1}"
PLANNER_STATE_FEATURE_DROPOUT_TERMS="${PLANNER_STATE_FEATURE_DROPOUT_TERMS:-expert_motion}"
PLANNER_STATE_FEATURE_DROPOUT_MODE="${PLANNER_STATE_FEATURE_DROPOUT_MODE:-shuffle}"
LOW_LEVEL_MAX_ITERATIONS="${LOW_LEVEL_MAX_ITERATIONS:-10000}"
LOW_LEVEL_VIDEO_LENGTH="${LOW_LEVEL_VIDEO_LENGTH:-500}"
LOW_LEVEL_VIDEO_INTERVAL="${LOW_LEVEL_VIDEO_INTERVAL:-2500}"
HL_SKILL_FINETUNE_ENABLED="${HL_SKILL_FINETUNE_ENABLED:-false}"
HL_SKILL_PG_COEFF="${HL_SKILL_PG_COEFF:-0.05}"
HL_SKILL_OFFLINE_DIFFSR_COEFF="${HL_SKILL_OFFLINE_DIFFSR_COEFF:-1.0}"
HL_SKILL_ANCHOR_COEFF="${HL_SKILL_ANCHOR_COEFF:-0.01}"
HL_SKILL_Z_NORM_COEFF="${HL_SKILL_Z_NORM_COEFF:-}"
HL_SKILL_LR="${HL_SKILL_LR:-3.0e-5}"
HL_SKILL_GRAD_CLIP_NORM="${HL_SKILL_GRAD_CLIP_NORM:-1.0}"
HL_SKILL_OFFLINE_BATCH_SIZE="${HL_SKILL_OFFLINE_BATCH_SIZE:-8192}"
HL_SKILL_UPDATE_INTERVAL="${HL_SKILL_UPDATE_INTERVAL:-1}"
HL_SKILL_TRAIN_DIFFSR="${HL_SKILL_TRAIN_DIFFSR:-false}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000000}"
LOGGER_BACKEND="${LOGGER_BACKEND:-}"
LOGGER_PROJECT_NAME="${LOGGER_PROJECT_NAME:-G1-Imitation-LAFAN1-Language}"
SKILL_LOGGER_BACKEND="${SKILL_LOGGER_BACKEND:-${LOGGER_BACKEND}}"
SKILL_WANDB_MODE="${SKILL_WANDB_MODE:-${WANDB_MODE:-offline}}"
RUN_M1_EVAL="${RUN_M1_EVAL:-1}"
SKIP_SKILL="${SKIP_SKILL:-0}"
SKIP_LOW_LEVEL="${SKIP_LOW_LEVEL:-0}"
STOP_AFTER_LOW_LEVEL="${STOP_AFTER_LOW_LEVEL:-0}"
SKIP_PLANNER="${SKIP_PLANNER:-0}"
SKIP_ROLLOUT_FINETUNE="${SKIP_ROLLOUT_FINETUNE:-0}"
ROLLOUT_RANKS="${ROLLOUT_RANKS:-all}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-0}"
ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-1}"
ROLLOUT_MAX_STEPS="${ROLLOUT_MAX_STEPS:-0}"
ROLLOUT_CHUNK_ROWS="${ROLLOUT_CHUNK_ROWS:-8192}"
ROLLOUT_FT_UPDATES="${ROLLOUT_FT_UPDATES:-20000}"
ROLLOUT_FT_BATCH_SIZE="${ROLLOUT_FT_BATCH_SIZE:-1024}"
ROLLOUT_FT_LR="${ROLLOUT_FT_LR:-1.0e-4}"
ROLLOUT_EVAL_RANKS="${ROLLOUT_EVAL_RANKS:-all}"
ROLLOUT_EVAL_VIDEO_RANKS="${ROLLOUT_EVAL_VIDEO_RANKS:-0,5,10,20,30,39}"
ROLLOUT_EVAL_VIDEO_LENGTH="${ROLLOUT_EVAL_VIDEO_LENGTH:-500}"
ROLLOUT_EVAL_MAX_STEPS="${ROLLOUT_EVAL_MAX_STEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_lafan1_language_w${HORIZON_STEPS}_z${Z_DIM}_ipmd_sincos}"
RUN_ROOT="${RUN_ROOT:-logs/lafan1_language_pipeline/${RUN_ID}}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "[ERROR] LAFAN1 manifest not found: ${MANIFEST_PATH}" >&2
    exit 1
fi
if [[ ! -e "${DATASET_PATH}" ]]; then
    echo "[ERROR] LAFAN1 dataset cache not found: ${DATASET_PATH}" >&2
    exit 1
fi
if [[ ! -f "${LANGUAGE_EMBEDDINGS}" ]]; then
    echo "[ERROR] Language embedding table not found: ${LANGUAGE_EMBEDDINGS}" >&2
    echo "[HINT] Set LANGUAGE_EMBEDDINGS=/path/to/table.pt to use a different encoder." >&2
    exit 1
fi
if [[ "${LOW_LEVEL_ALGO}" == "IPMD_BILINEAR" ]]; then
    echo "[ERROR] This language pipeline is configured for plain IPMD, not IPMD_BILINEAR." >&2
    exit 1
fi

MANIFEST_ABS="$(realpath "${MANIFEST_PATH}")"
DATASET_ABS="$(realpath "${DATASET_PATH}")"
LANGUAGE_EMBEDDINGS_ABS="$(realpath "${LANGUAGE_EMBEDDINGS}")"
RUN_ROOT_ABS="$(mkdir -p "${RUN_ROOT}" && realpath "${RUN_ROOT}")"
COMMAND_LOG="${RUN_ROOT_ABS}/commands.sh"
STDOUT_LOG="${RUN_ROOT_ABS}/pipeline.stdout.log"
mkdir -p "${RUN_ROOT_ABS}"

case "${HL_SKILL_COMMAND_MODE}" in
    z)
        COMMAND_CODE_DIM="${Z_DIM}"
        ;;
    phi|fz)
        HL_SKILL_COMMAND_MODE="phi"
        COMMAND_CODE_DIM="${DIFFSR_FEATURE_DIM}"
        ;;
    z_phi|z_fz)
        HL_SKILL_COMMAND_MODE="z_phi"
        COMMAND_CODE_DIM="$((Z_DIM + DIFFSR_FEATURE_DIM))"
        ;;
    *)
        echo "[ERROR] HL_SKILL_COMMAND_MODE must be z, phi, or z_phi; got ${HL_SKILL_COMMAND_MODE}" >&2
        exit 1
        ;;
esac
LOW_LEVEL_LATENT_DIM="$((COMMAND_CODE_DIM + 2))"
: > "${COMMAND_LOG}"
exec > >(tee -a "${STDOUT_LOG}") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_ROOT_ABS}/pipeline.log"
}

run_cmd() {
    log "RUN: $*"
    printf '%q ' "$@" >> "${COMMAND_LOG}"
    printf '\n' >> "${COMMAND_LOG}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
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
manifest_path=${MANIFEST_ABS}
dataset_path=${DATASET_ABS}
language_embeddings=${LANGUAGE_EMBEDDINGS_ABS}
language_condition=${LANGUAGE_CONDITION}
system0=plain_ipmd_low_level
system1=language_conditioned_flow_planner
system2=language_prompt_embedding
horizon_steps=${HORIZON_STEPS}
planner_state_history_steps=${STATE_HISTORY_STEPS}
planner_condition_window_states=$((STATE_HISTORY_STEPS + 1))
skill_z_dim=${Z_DIM}
diffsr_feature_dim=${DIFFSR_FEATURE_DIM}
diffsr_phi_parameterization=${DIFFSR_PHI_PARAMETERIZATION}
skill_encoder_window_mode=${SKILL_ENCODER_WINDOW_MODE}
skill_reg_coeff=${SKILL_REG_COEFF}
hl_skill_command_mode=${HL_SKILL_COMMAND_MODE}
low_level_code_dim=${COMMAND_CODE_DIM}
low_level_latent_dim=${LOW_LEVEL_LATENT_DIM}
low_level_command=${HL_SKILL_COMMAND_MODE}_plus_sin_cos_phase
hl_skill_finetune_enabled=${HL_SKILL_FINETUNE_ENABLED}
hl_skill_pg_coeff=${HL_SKILL_PG_COEFF}
hl_skill_offline_diffsr_coeff=${HL_SKILL_OFFLINE_DIFFSR_COEFF}
hl_skill_anchor_coeff=${HL_SKILL_ANCHOR_COEFF}
hl_skill_z_norm_coeff=${HL_SKILL_Z_NORM_COEFF}
hl_skill_lr=${HL_SKILL_LR}
hl_skill_grad_clip_norm=${HL_SKILL_GRAD_CLIP_NORM}
hl_skill_offline_batch_size=${HL_SKILL_OFFLINE_BATCH_SIZE}
hl_skill_update_interval=${HL_SKILL_UPDATE_INTERVAL}
hl_skill_train_diffsr=${HL_SKILL_TRAIN_DIFFSR}
low_level_algo=${LOW_LEVEL_ALGO}
seed=${SEED}
EOF

log "LAFAN1 language-conditioned pipeline"
log "run root: ${RUN_ROOT_ABS}"
log "manifest=${MANIFEST_ABS}"
log "dataset=${DATASET_ABS}"
log "language_embeddings=${LANGUAGE_EMBEDDINGS_ABS}"
log "skill: diffsr_phi_parameterization=${DIFFSR_PHI_PARAMETERIZATION} encoder_window_mode=${SKILL_ENCODER_WINDOW_MODE} reg_coeff=${SKILL_REG_COEFF}"
log "low-level: algo=${LOW_LEVEL_ALGO} command=${HL_SKILL_COMMAND_MODE}+sin_cos latent_dim=${LOW_LEVEL_LATENT_DIM} w=${HORIZON_STEPS} hl_finetune=${HL_SKILL_FINETUNE_ENABLED}"

SKILL_DIR="${SKILL_DIR:-${RUN_ROOT_ABS}/skill_encoder_h${HORIZON_STEPS}_z${Z_DIM}_${SKILL_ENCODER_WINDOW_MODE}_${DIFFSR_PHI_PARAMETERIZATION}}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-${SKILL_DIR}/checkpoints/latest.pt}"
if [[ "${SKIP_SKILL}" == "1" ]]; then
    if [[ ! -f "${SKILL_CHECKPOINT}" && "${DRY_RUN}" != "1" ]]; then
        log "SKIP_SKILL=1 but SKILL_CHECKPOINT does not exist: ${SKILL_CHECKPOINT}"
        exit 1
    fi
    log "Skipping skill encoder training; using ${SKILL_CHECKPOINT}"
else
    run_cmd "${PYTHON_BIN}" scripts/rlopt/train_hl_skill_diffsr.py \
        --headless \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num_envs "${SKILL_NUM_ENVS}" \
        --seed "${SEED}" \
        --output_dir "${SKILL_DIR}" \
        --horizon_steps "${HORIZON_STEPS}" \
        --encoder_window_mode "${SKILL_ENCODER_WINDOW_MODE}" \
        --z_dim "${Z_DIM}" \
        --latent_mode deterministic \
        --reg_coeff "${SKILL_REG_COEFF}" \
        --diffsr_feature_dim "${DIFFSR_FEATURE_DIM}" \
        --diffsr_embed_dim "${DIFFSR_EMBED_DIM}" \
        --diffsr_phi_parameterization "${DIFFSR_PHI_PARAMETERIZATION}" \
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
        --logger_backend "${SKILL_LOGGER_BACKEND}" \
        --wandb_project "${LOGGER_PROJECT_NAME}" \
        --wandb_group "${RUN_ID}" \
        --wandb_run_name "${RUN_ID}_skill_encoder" \
        --wandb_mode "${SKILL_WANDB_MODE}" \
        "env.lafan1_manifest_path=${MANIFEST_ABS}" \
        "env.dataset_path=${DATASET_ABS}" \
        "env.refresh_zarr_dataset=true"
fi
if [[ ! -f "${SKILL_CHECKPOINT}" && "${DRY_RUN}" != "1" ]]; then
    log "Missing skill checkpoint: ${SKILL_CHECKPOINT}"
    exit 1
fi
log "Skill encoder checkpoint: ${SKILL_CHECKPOINT}"

COMMON_LATENT_OVERRIDES=(
    "env.lafan1_manifest_path=${MANIFEST_ABS}"
    "env.dataset_path=${DATASET_ABS}"
    "env.refresh_zarr_dataset=false"
    "env.latent_command_dim=${LOW_LEVEL_LATENT_DIM}"
    "agent.ipmd.latent_dim=${LOW_LEVEL_LATENT_DIM}"
    "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
    "agent.ipmd.hl_skill_command_mode=${HL_SKILL_COMMAND_MODE}"
    "agent.ipmd.latent_steps_min=${HORIZON_STEPS}"
    "agent.ipmd.latent_steps_max=${HORIZON_STEPS}"
    "agent.ipmd.latent_learning.command_phase_mode=sin_cos"
    "agent.ipmd.latent_learning.code_latent_dim=${COMMAND_CODE_DIM}"
    "agent.ipmd.latent_learning.code_period=${HORIZON_STEPS}"
    "agent.ipmd.reward_loss_coeff=0.0"
    "agent.ipmd.reward_l2_coeff=0.0"
    "agent.ipmd.reward_grad_penalty_coeff=0.0"
    "agent.ipmd.reward_logit_reg_coeff=0.0"
    "agent.ipmd.reward_param_weight_decay_coeff=0.0"
)
if [[ -n "${HL_SKILL_Z_NORM_COEFF}" ]]; then
    COMMON_LATENT_OVERRIDES+=("agent.ipmd.hl_skill_z_norm_coeff=${HL_SKILL_Z_NORM_COEFF}")
fi

LOW_LEVEL_LOG_DIR="${LOW_LEVEL_LOG_DIR:-}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-}"
if [[ "${SKIP_LOW_LEVEL}" == "1" ]]; then
    if [[ -z "${LOW_LEVEL_CHECKPOINT}" || ( ! -f "${LOW_LEVEL_CHECKPOINT}" && "${DRY_RUN}" != "1" ) ]]; then
        log "SKIP_LOW_LEVEL=1 requires LOW_LEVEL_CHECKPOINT to point at an existing model checkpoint."
        exit 1
    fi
    LOW_LEVEL_LOG_DIR="$(cd "$(dirname "${LOW_LEVEL_CHECKPOINT}")/.." 2>/dev/null && pwd || true)"
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
        --num_envs "${LOW_LEVEL_NUM_ENVS}" \
        --task "${TASK}" \
        --algo "${LOW_LEVEL_ALGO}" \
        --seed "${SEED}" \
        --max_iterations "${LOW_LEVEL_MAX_ITERATIONS}" \
        "agent.logger.backend=${LOGGER_BACKEND}" \
        "agent.logger.project_name=${LOGGER_PROJECT_NAME}" \
        "agent.logger.exp_name=${RUN_ID}_oracle_low_level" \
        "agent.logger.video=true" \
        "agent.save_interval=${SAVE_INTERVAL}" \
        "agent.ipmd.command_source=hl_skill" \
        "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
        "agent.ipmd.hl_skill_finetune_enabled=${HL_SKILL_FINETUNE_ENABLED}" \
        "agent.ipmd.hl_skill_pg_coeff=${HL_SKILL_PG_COEFF}" \
        "agent.ipmd.hl_skill_offline_diffsr_coeff=${HL_SKILL_OFFLINE_DIFFSR_COEFF}" \
        "agent.ipmd.hl_skill_anchor_coeff=${HL_SKILL_ANCHOR_COEFF}" \
        "agent.ipmd.hl_skill_lr=${HL_SKILL_LR}" \
        "agent.ipmd.hl_skill_grad_clip_norm=${HL_SKILL_GRAD_CLIP_NORM}" \
        "agent.ipmd.hl_skill_offline_batch_size=${HL_SKILL_OFFLINE_BATCH_SIZE}" \
        "agent.ipmd.hl_skill_update_interval=${HL_SKILL_UPDATE_INTERVAL}" \
        "agent.ipmd.hl_skill_train_diffsr=${HL_SKILL_TRAIN_DIFFSR}" \
        "${COMMON_LATENT_OVERRIDES[@]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        LOW_LEVEL_LOG_DIR="${RUN_ROOT_ABS}/dry_run_low_level"
        LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_LOG_DIR}/models/model_step_placeholder.pt"
    else
        LOG_ROOT="${REPO_ROOT}/logs/rlopt/${LOW_LEVEL_ALGO,,}/${TASK}"
        LOW_LEVEL_LOG_DIR="$(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -newer "${LOW_LEVEL_MARKER}" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
        if [[ -z "${LOW_LEVEL_LOG_DIR}" ]]; then
            log "Could not locate low-level log dir under ${LOG_ROOT}."
            exit 1
        fi
        LOW_LEVEL_CHECKPOINT="$(latest_checkpoint_from_log_dir "${LOW_LEVEL_LOG_DIR}")"
    fi
fi
if [[ ( -z "${LOW_LEVEL_CHECKPOINT}" || ! -f "${LOW_LEVEL_CHECKPOINT}" ) && "${DRY_RUN}" != "1" ]]; then
    log "Could not locate low-level checkpoint in ${LOW_LEVEL_LOG_DIR}."
    exit 1
fi
log "Low-level log dir: ${LOW_LEVEL_LOG_DIR}"
log "Low-level checkpoint: ${LOW_LEVEL_CHECKPOINT}"
printf '%s\n' "${LOW_LEVEL_LOG_DIR}" > "${RUN_ROOT_ABS}/low_level_log_dir.txt"
printf '%s\n' "${LOW_LEVEL_CHECKPOINT}" > "${RUN_ROOT_ABS}/low_level_checkpoint.txt"

if [[ "${STOP_AFTER_LOW_LEVEL}" == "1" ]]; then
    log "Stopping after low-level training because STOP_AFTER_LOW_LEVEL=1."
    exit 0
fi

PLANNER_DIR="${PLANNER_DIR:-${RUN_ROOT_ABS}/planner_${PLANNER_TYPE}_language_hist$((STATE_HISTORY_STEPS + 1))}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${PLANNER_DIR}/checkpoints/latest.pt}"
if [[ "${SKIP_PLANNER}" == "1" ]]; then
    if [[ ! -f "${PLANNER_CHECKPOINT}" && "${DRY_RUN}" != "1" ]]; then
        log "SKIP_PLANNER=1 but PLANNER_CHECKPOINT does not exist: ${PLANNER_CHECKPOINT}"
        exit 1
    fi
    log "Skipping planner training; using ${PLANNER_CHECKPOINT}"
else
    run_cmd "${PYTHON_BIN}" scripts/rlopt/train_skill_commander.py \
        --headless \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num_envs "${PLANNER_NUM_ENVS}" \
        --seed "${SEED}" \
        --output_dir "${PLANNER_DIR}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --language_embeddings "${LANGUAGE_EMBEDDINGS_ABS}" \
        --state_history_steps "${STATE_HISTORY_STEPS}" \
        --planner_type "${PLANNER_TYPE}" \
        --generator_hidden_dims 1024 512 512 \
        --flow_num_inference_steps "${PLANNER_FLOW_STEPS}" \
        --flow_time_embed_dim "${PLANNER_FLOW_TIME_DIM}" \
        --flow_train_noise_std "${PLANNER_FLOW_TRAIN_NOISE_STD}" \
        --flow_inference_noise_std "${PLANNER_FLOW_INFERENCE_NOISE_STD}" \
        --state_noise_std "${PLANNER_STATE_NOISE_STD}" \
        --state_feature_dropout_prob "${PLANNER_STATE_FEATURE_DROPOUT_PROB}" \
        --state_feature_dropout_terms "${PLANNER_STATE_FEATURE_DROPOUT_TERMS}" \
        --state_feature_dropout_mode "${PLANNER_STATE_FEATURE_DROPOUT_MODE}" \
        --language_contrastive_coeff "${PLANNER_LANGUAGE_CONTRASTIVE_COEFF}" \
        --language_contrastive_margin "${PLANNER_LANGUAGE_CONTRASTIVE_MARGIN}" \
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
if [[ ! -f "${PLANNER_CHECKPOINT}" && "${DRY_RUN}" != "1" ]]; then
    log "Missing planner checkpoint: ${PLANNER_CHECKPOINT}"
    exit 1
fi
log "Planner checkpoint: ${PLANNER_CHECKPOINT}"

if [[ "${RUN_M1_EVAL}" == "1" ]]; then
    run_cmd "${PYTHON_BIN}" scripts/rlopt/eval_skill_commander_m1.py \
        --headless \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num_envs "${PLANNER_NUM_ENVS}" \
        --seed "${SEED}" \
        --checkpoint "${PLANNER_CHECKPOINT}" \
        --language_embeddings "${LANGUAGE_EMBEDDINGS_ABS}" \
        --output_dir "${RUN_ROOT_ABS}/m1_eval_planner_language" \
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

if [[ "${SKIP_ROLLOUT_FINETUNE}" == "1" ]]; then
    log "Skipping rollout finetune because SKIP_ROLLOUT_FINETUNE=1."
    exit 0
fi

ROLLOUT_FT_ROOT="${ROLLOUT_FT_ROOT:-${RUN_ROOT_ABS}/rollout_finetune_merged_language}"
ROLLOUT_CMD=(
    "${PYTHON_BIN}" scripts/rlopt/run_lafan1_no_language_rollout_ft_merged.py
    --python_bin "${PYTHON_BIN}"
    --task "${TASK}"
    --algorithm "${LOW_LEVEL_ALGO}"
    --manifest "${MANIFEST_ABS}"
    --dataset_path "${DATASET_ABS}"
    --language_embeddings "${LANGUAGE_EMBEDDINGS_ABS}"
    --checkpoint "${LOW_LEVEL_CHECKPOINT}"
    --planner_checkpoint "${PLANNER_CHECKPOINT}"
    --skill_checkpoint "${SKILL_CHECKPOINT}"
    --output_root "${ROLLOUT_FT_ROOT}"
    --ranks "${ROLLOUT_RANKS}"
    --seeds "${ROLLOUT_SEEDS}"
    --num_envs "${ROLLOUT_NUM_ENVS}"
    --device "${DEVICE}"
    --metric_interval 1
    --chunk_rows "${ROLLOUT_CHUNK_ROWS}"
    --finetune_updates "${ROLLOUT_FT_UPDATES}"
    --finetune_batch_size "${ROLLOUT_FT_BATCH_SIZE}"
    --finetune_lr "${ROLLOUT_FT_LR}"
    --flow_num_inference_steps "${PLANNER_FLOW_STEPS}"
    --flow_inference_noise_std "${PLANNER_EVAL_FLOW_NOISE_STD}"
    --eval_ranks "${ROLLOUT_EVAL_RANKS}"
    --eval_video_ranks "${ROLLOUT_EVAL_VIDEO_RANKS}"
    --eval_video_length "${ROLLOUT_EVAL_VIDEO_LENGTH}"
    --z_dim "${Z_DIM}"
    --command_mode "${HL_SKILL_COMMAND_MODE}"
    --command_code_dim "${COMMAND_CODE_DIM}"
    --horizon_steps "${HORIZON_STEPS}"
    --resume
)
if [[ -n "${ROLLOUT_LIMIT}" ]]; then
    ROLLOUT_CMD+=(--limit "${ROLLOUT_LIMIT}")
fi
if (( ROLLOUT_MAX_STEPS > 0 )); then
    ROLLOUT_CMD+=(--max_steps "${ROLLOUT_MAX_STEPS}")
fi
if (( ROLLOUT_EVAL_MAX_STEPS > 0 )); then
    ROLLOUT_CMD+=(--eval_max_steps "${ROLLOUT_EVAL_MAX_STEPS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    ROLLOUT_CMD+=(--dry_run)
fi

run_cmd "${ROLLOUT_CMD[@]}"

log "Language pipeline complete. Outputs are under:"
log "  run root: ${RUN_ROOT_ABS}"
log "  skill checkpoint: ${SKILL_CHECKPOINT}"
log "  low-level checkpoint: ${LOW_LEVEL_CHECKPOINT}"
log "  planner checkpoint: ${PLANNER_CHECKPOINT}"
log "  merged rollout finetune: ${ROLLOUT_FT_ROOT}"
