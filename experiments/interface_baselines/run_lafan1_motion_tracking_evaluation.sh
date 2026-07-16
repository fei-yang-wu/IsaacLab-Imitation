#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

PYTHON_CMD_STR="${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
ISAACLAB_PYTHON_CMD_STR="${INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD:-pixi run -e isaaclab python}"
if [[ -n "${LAFAN1_BASE_PIPELINE_CMD:-}" ]]; then
    PIPELINE_CMD_STR="${LAFAN1_BASE_PIPELINE_CMD}"
elif command -v pixi >/dev/null 2>&1; then
    PIPELINE_CMD_STR="pixi run -e isaaclab scripts/rlopt/run_lafan1_no_language_pipeline.sh"
else
    PIPELINE_CMD_STR="bash scripts/rlopt/run_lafan1_no_language_pipeline.sh"
    export PYTHON_BIN="${PYTHON_BIN:-${ISAACLAB_PYTHON_CMD_STR}}"
fi
# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_CMD_STR})
# shellcheck disable=SC2206
ISAACLAB_PYTHON_CMD=(${ISAACLAB_PYTHON_CMD_STR})
# shellcheck disable=SC2206
PIPELINE_CMD=(${PIPELINE_CMD_STR})

TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
LOW_LEVEL_ALGO="${LOW_LEVEL_ALGO:-IPMD}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MANIFEST_PATH="${MANIFEST_PATH:-data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-data/lafan1/g1_hl_diffsr}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
Z_DIM="${Z_DIM:-256}"
PLANNER_TYPE="${PLANNER_TYPE:-flow_matching}"
PLANNER_FLOW_STEPS="${PLANNER_FLOW_STEPS:-16}"
PLANNER_EVAL_FLOW_NOISE_STD="${PLANNER_EVAL_FLOW_NOISE_STD:-0.0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_lafan1_motion_tracking_h${HORIZON_STEPS}_ipmd}"
RUN_ROOT="${RUN_ROOT:-logs/lafan1_motion_tracking_evaluation/${RUN_ID}}"

RUN_BASE_PIPELINE="${RUN_BASE_PIPELINE:-1}"
RUN_ORACLE_RECON_EVAL="${RUN_ORACLE_RECON_EVAL:-1}"
RUN_BASE_PLANNER_PREDICT_EVAL="${RUN_BASE_PLANNER_PREDICT_EVAL:-1}"
RUN_ORACLE_LL_EVAL="${RUN_ORACLE_LL_EVAL:-1}"
RUN_BASE_PLANNER_LL_EVAL="${RUN_BASE_PLANNER_LL_EVAL:-0}"
RUN_PLANNER_ROLLOUT_FINETUNE="${RUN_PLANNER_ROLLOUT_FINETUNE:-1}"
RUN_PLANNER_FT_SAMPLE_COLLECTION="${RUN_PLANNER_FT_SAMPLE_COLLECTION:-${RUN_PLANNER_ROLLOUT_FINETUNE}}"
RUN_FINETUNED_PLANNER_PREDICT_EVAL="${RUN_FINETUNED_PLANNER_PREDICT_EVAL:-1}"
RUN_FINETUNED_PLANNER_LL_EVAL="${RUN_FINETUNED_PLANNER_LL_EVAL:-1}"
RUN_HAND_DESIGNED_BASELINES="${RUN_HAND_DESIGNED_BASELINES:-0}"
DRY_RUN="${DRY_RUN:-0}"

RANKS="${RANKS:-all}"
LIMIT="${LIMIT:-0}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-0}"
EVAL_VIDEO_LENGTH="${EVAL_VIDEO_LENGTH:-0}"
EVAL_METRIC_INTERVAL="${EVAL_METRIC_INTERVAL:-1}"
PLANNER_FT_COLLECT_MAX_STEPS="${PLANNER_FT_COLLECT_MAX_STEPS:-0}"
DANCE102_FINETUNE_UPDATES="${DANCE102_FINETUNE_UPDATES:-2000}"
PLANNER_FT_UPDATES="${PLANNER_FT_UPDATES:-${DANCE102_FINETUNE_UPDATES}}"
PLANNER_FT_BATCH_SIZE="${PLANNER_FT_BATCH_SIZE:-256}"
PLANNER_FT_LR="${PLANNER_FT_LR:-1.0e-4}"
PLANNER_FT_WEIGHT_DECAY="${PLANNER_FT_WEIGHT_DECAY:-0.0}"

BASELINE_INTERFACES="${BASELINE_INTERFACES:-}"
BASELINE_TASK="${BASELINE_TASK:-Isaac-Imitation-G1-v0}"
BASELINE_ALGO="${BASELINE_ALGO:-IPMD}"
BASELINE_MODEL_SIZE="${BASELINE_MODEL_SIZE:-medium}"
BASELINE_SAMPLE_BUDGETS="${BASELINE_SAMPLE_BUDGETS:-1000}"
BASELINE_PRETRAIN_UPDATES="${BASELINE_PRETRAIN_UPDATES:-2000}"
BASELINE_FINETUNE_UPDATES="${BASELINE_FINETUNE_UPDATES:-${DANCE102_FINETUNE_UPDATES}}"
BASELINE_COMMAND_PAST_STEPS="${BASELINE_COMMAND_PAST_STEPS:-0}"
BASELINE_COMMAND_FUTURE_STEPS="${BASELINE_COMMAND_FUTURE_STEPS:-${HORIZON_STEPS}}"
# Matched high-level cadence: chunk planners are queried once per horizon and
# their chunks are held/consumed VLA-style, mirroring the latent z hold.
# Set both to 1/0 to reproduce the legacy per-step receding-horizon protocol.
BASELINE_PLANNER_UPDATE_INTERVAL="${BASELINE_PLANNER_UPDATE_INTERVAL:-${HORIZON_STEPS}}"
BASELINE_COMMAND_HOLD_STEPS="${BASELINE_COMMAND_HOLD_STEPS:-${HORIZON_STEPS}}"
BASELINE_DEFAULT_STEPS="${EVAL_MAX_STEPS}"
if (( BASELINE_DEFAULT_STEPS <= 0 )); then
    BASELINE_DEFAULT_STEPS=1000
fi
BASELINE_COLLECT_STEPS="${BASELINE_COLLECT_STEPS:-${BASELINE_DEFAULT_STEPS}}"
BASELINE_EVAL_STEPS="${BASELINE_EVAL_STEPS:-${BASELINE_DEFAULT_STEPS}}"
BASELINE_NUM_ENVS="${BASELINE_NUM_ENVS:-${EVAL_NUM_ENVS}}"
BASELINE_USE_TRAJECTORY_STEPS="${BASELINE_USE_TRAJECTORY_STEPS:-1}"
FULL_BODY_TRAJECTORY_CHECKPOINT="${FULL_BODY_TRAJECTORY_CHECKPOINT:-}"
EE_TRAJECTORY_CHECKPOINT="${EE_TRAJECTORY_CHECKPOINT:-}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "[ERROR] LAFAN1 manifest not found: ${MANIFEST_PATH}" >&2
    exit 1
fi

MANIFEST_ABS="$(realpath "${MANIFEST_PATH}")"
DATASET_ABS="$(realpath -m "${DATASET_PATH}")"
RUN_ROOT_ABS="$(mkdir -p "${RUN_ROOT}" && realpath "${RUN_ROOT}")"
COMMAND_LOG="${RUN_ROOT_ABS}/commands.sh"
: > "${COMMAND_LOG}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_ROOT_ABS}/run.log"
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

log "LAFAN1 motion tracking evaluation"
log "run_root=${RUN_ROOT_ABS}"
log "manifest=${MANIFEST_ABS}"
log "low_level_algo=${LOW_LEVEL_ALGO} horizon=${HORIZON_STEPS} state_history_steps=${STATE_HISTORY_STEPS}"

if [[ "${RUN_BASE_PIPELINE}" == "1" ]]; then
    log "Running base latent pipeline with IPMD low-level and no merged rollout finetune."
    run_cmd env \
        "TASK=${TASK}" \
        "LOW_LEVEL_ALGO=${LOW_LEVEL_ALGO}" \
        "DEVICE=${DEVICE}" \
        "SEED=${SEED}" \
        "NUM_ENVS=${NUM_ENVS}" \
        "MANIFEST_PATH=${MANIFEST_ABS}" \
        "DATASET_PATH=${DATASET_ABS}" \
        "HORIZON_STEPS=${HORIZON_STEPS}" \
        "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}" \
        "Z_DIM=${Z_DIM}" \
        "PLANNER_TYPE=${PLANNER_TYPE}" \
        "PLANNER_FLOW_STEPS=${PLANNER_FLOW_STEPS}" \
        "PLANNER_EVAL_FLOW_NOISE_STD=${PLANNER_EVAL_FLOW_NOISE_STD}" \
        "RUN_ROOT=${RUN_ROOT_ABS}/base_pipeline" \
        "SKIP_ROLLOUT_FINETUNE=1" \
        "${PIPELINE_CMD[@]}"
fi

BASE_ROOT="${BASE_ROOT:-${RUN_ROOT_ABS}/base_pipeline}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-${BASE_ROOT}/skill_encoder_h${HORIZON_STEPS}_z${Z_DIM}/checkpoints/latest.pt}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${BASE_ROOT}/planner_${PLANNER_TYPE}_no_language_hist$((STATE_HISTORY_STEPS + 1))/checkpoints/latest.pt}"
if [[ -z "${LOW_LEVEL_CHECKPOINT:-}" && -f "${BASE_ROOT}/low_level_checkpoint.txt" ]]; then
    LOW_LEVEL_CHECKPOINT="$(<"${BASE_ROOT}/low_level_checkpoint.txt")"
fi
if [[ -z "${LOW_LEVEL_CHECKPOINT:-}" && -n "${LOW_LEVEL_LOG_DIR:-}" ]]; then
    LOW_LEVEL_CHECKPOINT="$(latest_checkpoint_from_log_dir "${LOW_LEVEL_LOG_DIR}")"
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-${BASE_ROOT}/models/model_step_DRY_RUN.pt}"
fi

NEEDS_LATENT_CHECKPOINTS=0
for flag_name in \
    RUN_ORACLE_RECON_EVAL \
    RUN_BASE_PLANNER_PREDICT_EVAL \
    RUN_ORACLE_LL_EVAL \
    RUN_BASE_PLANNER_LL_EVAL \
    RUN_PLANNER_FT_SAMPLE_COLLECTION \
    RUN_PLANNER_ROLLOUT_FINETUNE \
    RUN_FINETUNED_PLANNER_PREDICT_EVAL \
    RUN_FINETUNED_PLANNER_LL_EVAL; do
    if [[ "${!flag_name}" == "1" ]]; then
        NEEDS_LATENT_CHECKPOINTS=1
        break
    fi
done
if [[ "${NEEDS_LATENT_CHECKPOINTS}" == "1" ]]; then
    for path_label in SKILL_CHECKPOINT PLANNER_CHECKPOINT LOW_LEVEL_CHECKPOINT; do
        path_value="${!path_label:-}"
        if [[ "${DRY_RUN}" != "1" && ( -z "${path_value}" || ! -f "${path_value}" ) ]]; then
            echo "[ERROR] ${path_label} does not point to an existing file: ${path_value}" >&2
            echo "[HINT] Set ${path_label}=... or run with RUN_BASE_PIPELINE=1." >&2
            exit 1
        fi
        log "${path_label}=${path_value}"
    done
else
    log "Skipping latent checkpoint validation because all latent stages are disabled."
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

CLOSED_LOOP_COMMON_ARGS=()
if [[ "${NEEDS_LATENT_CHECKPOINTS}" == "1" ]]; then
    CLOSED_LOOP_COMMON_ARGS=(
        --headless
        --device "${DEVICE}"
        --num_envs "${EVAL_NUM_ENVS}"
        --task "${TASK}"
        --algorithm "${LOW_LEVEL_ALGO}"
        --seed "${SEED}"
        --checkpoint "${LOW_LEVEL_CHECKPOINT}"
        --skill_checkpoint "${SKILL_CHECKPOINT}"
        --metric_interval "${EVAL_METRIC_INTERVAL}"
        --flow_num_inference_steps "${PLANNER_FLOW_STEPS}"
        --flow_inference_noise_std "${PLANNER_EVAL_FLOW_NOISE_STD}"
    )
    if (( EVAL_VIDEO_LENGTH > 0 )); then
        CLOSED_LOOP_COMMON_ARGS+=(--video --video_length "${EVAL_VIDEO_LENGTH}")
    fi
fi
CLOSED_LOOP_EVAL_ARGS=("${CLOSED_LOOP_COMMON_ARGS[@]}")
if (( EVAL_MAX_STEPS > 0 )); then
    CLOSED_LOOP_EVAL_ARGS+=(--max_steps "${EVAL_MAX_STEPS}")
fi
CLOSED_LOOP_COLLECT_ARGS=("${CLOSED_LOOP_COMMON_ARGS[@]}")
if (( PLANNER_FT_COLLECT_MAX_STEPS > 0 )); then
    CLOSED_LOOP_COLLECT_ARGS+=(--max_steps "${PLANNER_FT_COLLECT_MAX_STEPS}")
elif (( EVAL_MAX_STEPS > 0 )); then
    CLOSED_LOOP_COLLECT_ARGS+=(--max_steps "${EVAL_MAX_STEPS}")
fi

mapfile -t TRAJECTORIES < <("${PYTHON_CMD[@]}" experiments/interface_baselines/select_lafan1_trajectories.py \
    --manifest "${MANIFEST_ABS}" \
    --ranks "${RANKS}" \
    --limit "${LIMIT}" \
    --fallback_steps "${BASELINE_DEFAULT_STEPS}" \
    --output_root "${RUN_ROOT_ABS}/per_trajectory")
if [[ "${#TRAJECTORIES[@]}" -eq 0 ]]; then
    echo "[ERROR] No trajectories selected by RANKS=${RANKS} LIMIT=${LIMIT}" >&2
    exit 1
fi

for row in "${TRAJECTORIES[@]}"; do
    IFS=$'\t' read -r rank motion_name _clean_name trajectory_root single_manifest motion_steps <<<"${row}"
    log "Trajectory rank=${rank} motion=${motion_name} root=${trajectory_root}"

    if [[ "${RUN_ORACLE_RECON_EVAL}" == "1" ]]; then
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/train_hl_skill_diffsr.py \
            --headless \
            --device "${DEVICE}" \
            --task "${TASK}" \
            --num_envs "${EVAL_NUM_ENVS}" \
            --seed "${SEED}" \
            --checkpoint "${SKILL_CHECKPOINT}" \
            --eval_only \
            --output_dir "${trajectory_root}/oracle_recon_eval" \
            --horizon_steps "${HORIZON_STEPS}" \
            --encoder_window_mode intermediate \
            --z_dim "${Z_DIM}" \
            --batch_size 1024 \
            --eval_batches 4 \
            --eval_batch_size 1024 \
            --train_split all \
            --eval_split all \
            --reconstruction_eval \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${DATASET_ABS}" \
            "env.refresh_zarr_dataset=false"
    fi

    if [[ "${RUN_BASE_PLANNER_PREDICT_EVAL}" == "1" ]]; then
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_m1.py \
            --headless \
            --device "${DEVICE}" \
            --task "${TASK}" \
            --num_envs "${EVAL_NUM_ENVS}" \
            --seed "${SEED}" \
            --checkpoint "${PLANNER_CHECKPOINT}" \
            --skill_checkpoint "${SKILL_CHECKPOINT}" \
            --output_dir "${trajectory_root}/planner_predict_base" \
            --batch_size 1024 \
            --eval_batches 4 \
            --splits all \
            --per_trajectory \
            --trajectory_ranks 0 \
            --per_trajectory_batch_size 1024 \
            --per_trajectory_batches 4 \
            --flow_inference_noise_std "${PLANNER_EVAL_FLOW_NOISE_STD}" \
            --flow_num_inference_steps "${PLANNER_FLOW_STEPS}" \
            "env.lafan1_manifest_path=${single_manifest}" \
            "env.dataset_path=${DATASET_ABS}" \
            "env.refresh_zarr_dataset=false"
    fi

    if [[ "${RUN_ORACLE_LL_EVAL}" == "1" ]]; then
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
            "${CLOSED_LOOP_EVAL_ARGS[@]}" \
            --planner_checkpoint "${PLANNER_CHECKPOINT}" \
            --output_dir "${trajectory_root}/oracle_ll_eval" \
            --motion_name "${motion_name}" \
            "agent.ipmd.command_source=hl_skill" \
            "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
            "agent.ipmd.hl_skill_finetune_enabled=false" \
            "${COMMON_LATENT_OVERRIDES[@]}"
    fi

    oracle_collect_dir="${trajectory_root}/oracle_ll_collect"
    if [[ "${RUN_PLANNER_FT_SAMPLE_COLLECTION}" == "1" ]]; then
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
            "${CLOSED_LOOP_COLLECT_ARGS[@]}" \
            --planner_checkpoint "${PLANNER_CHECKPOINT}" \
            --output_dir "${oracle_collect_dir}" \
            --save_rollout_training_samples \
            --motion_name "${motion_name}" \
            "agent.ipmd.command_source=hl_skill" \
            "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
            "agent.ipmd.hl_skill_finetune_enabled=false" \
            "${COMMON_LATENT_OVERRIDES[@]}"
    fi

    if [[ "${RUN_BASE_PLANNER_LL_EVAL}" == "1" ]]; then
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
            "${CLOSED_LOOP_EVAL_ARGS[@]}" \
            --planner_checkpoint "${PLANNER_CHECKPOINT}" \
            --output_dir "${trajectory_root}/planner_ll_base" \
            --motion_name "${motion_name}" \
            "agent.ipmd.command_source=skill_commander" \
            "agent.ipmd.skill_commander_checkpoint_path=${PLANNER_CHECKPOINT}" \
            "agent.ipmd.skill_commander_embeddings_path=" \
            "agent.ipmd.skill_commander_flow_num_inference_steps=${PLANNER_FLOW_STEPS}" \
            "agent.ipmd.skill_commander_flow_inference_noise_std=${PLANNER_EVAL_FLOW_NOISE_STD}" \
            "agent.ipmd.skill_commander_use_achieved_state=true" \
            "agent.ipmd.hl_skill_finetune_enabled=false" \
            "${COMMON_LATENT_OVERRIDES[@]}"
    fi

    ft_dir="${trajectory_root}/planner_rollout_ft_single"
    ft_checkpoint="${ft_dir}/checkpoints/latest.pt"
    samples_dir="${oracle_collect_dir}/rollout_training_samples"
    if [[ "${RUN_PLANNER_ROLLOUT_FINETUNE}" == "1" ]]; then
        if [[ ! -d "${samples_dir}" && "${DRY_RUN}" != "1" ]]; then
            echo "[ERROR] Missing oracle rollout samples: ${samples_dir}" >&2
            echo "[HINT] RUN_PLANNER_FT_SAMPLE_COLLECTION=1 is required before per-trajectory finetune." >&2
            exit 1
        fi
        run_cmd "${PYTHON_CMD[@]}" scripts/rlopt/finetune_skill_commander_rollout.py \
            --checkpoint "${PLANNER_CHECKPOINT}" \
            --samples_dir "${samples_dir}" \
            --output_dir "${ft_dir}" \
            --device "${DEVICE}" \
            --seed "${SEED}" \
            --num_updates "${PLANNER_FT_UPDATES}" \
            --batch_size "${PLANNER_FT_BATCH_SIZE}" \
            --lr "${PLANNER_FT_LR}" \
            --weight_decay "${PLANNER_FT_WEIGHT_DECAY}" \
            --flow_num_inference_steps "${PLANNER_FLOW_STEPS}" \
            --flow_inference_noise_std "${PLANNER_EVAL_FLOW_NOISE_STD}"
    fi

    if [[ "${RUN_FINETUNED_PLANNER_PREDICT_EVAL}" == "1" ]]; then
        if [[ ! -f "${ft_checkpoint}" && "${DRY_RUN}" != "1" ]]; then
            echo "[ERROR] Missing finetuned planner checkpoint: ${ft_checkpoint}" >&2
            exit 1
        fi
        run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_latent_skill_planner_offline.py \
            --samples_dir "${samples_dir}" \
            --planner_checkpoint "${ft_checkpoint}" \
            --output_json "${trajectory_root}/planner_predict_finetuned/summary.json" \
            --output_csv "${trajectory_root}/planner_predict_finetuned/summary.csv" \
            --state_key planner_state \
            --setting finetuned_single_trajectory_achieved_state \
            --label "latent_skill_rank_${rank}_finetuned_predict" \
            --device "${DEVICE}" \
            --batch_size 512 \
            --seed "${SEED}" \
            --flow_num_inference_steps "${PLANNER_FLOW_STEPS}" \
            --flow_inference_noise_std "${PLANNER_EVAL_FLOW_NOISE_STD}"
    fi

    if [[ "${RUN_FINETUNED_PLANNER_LL_EVAL}" == "1" ]]; then
        if [[ ! -f "${ft_checkpoint}" && "${DRY_RUN}" != "1" ]]; then
            echo "[ERROR] Missing finetuned planner checkpoint: ${ft_checkpoint}" >&2
            exit 1
        fi
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
            "${CLOSED_LOOP_EVAL_ARGS[@]}" \
            --planner_checkpoint "${ft_checkpoint}" \
            --output_dir "${trajectory_root}/planner_ll_finetuned" \
            --motion_name "${motion_name}" \
            "agent.ipmd.command_source=skill_commander" \
            "agent.ipmd.skill_commander_checkpoint_path=${ft_checkpoint}" \
            "agent.ipmd.skill_commander_embeddings_path=" \
            "agent.ipmd.skill_commander_flow_num_inference_steps=${PLANNER_FLOW_STEPS}" \
            "agent.ipmd.skill_commander_flow_inference_noise_std=${PLANNER_EVAL_FLOW_NOISE_STD}" \
            "agent.ipmd.skill_commander_use_achieved_state=true" \
            "agent.ipmd.hl_skill_finetune_enabled=false" \
            "${COMMON_LATENT_OVERRIDES[@]}"
    fi

    if [[ "${RUN_HAND_DESIGNED_BASELINES}" == "1" ]]; then
        if [[ -z "${BASELINE_INTERFACES}" ]]; then
            echo "[ERROR] RUN_HAND_DESIGNED_BASELINES=1 requires BASELINE_INTERFACES, e.g. ee_trajectory or full_body_trajectory." >&2
            exit 1
        fi
        for baseline_interface in ${BASELINE_INTERFACES}; do
            case "${baseline_interface}" in
                full_body_trajectory)
                    if [[ -z "${FULL_BODY_TRAJECTORY_CHECKPOINT}" ]]; then
                        echo "[ERROR] BASELINE_INTERFACES includes full_body_trajectory but FULL_BODY_TRAJECTORY_CHECKPOINT is empty." >&2
                        exit 1
                    fi
                    ;;
                ee_trajectory)
                    if [[ -z "${EE_TRAJECTORY_CHECKPOINT}" ]]; then
                        echo "[ERROR] BASELINE_INTERFACES includes ee_trajectory but EE_TRAJECTORY_CHECKPOINT is empty." >&2
                        exit 1
                    fi
                    ;;
                *)
                    echo "[ERROR] Unsupported baseline interface: ${baseline_interface}" >&2
                    exit 1
                    ;;
            esac
        done
        baseline_collect_steps="${BASELINE_COLLECT_STEPS}"
        baseline_eval_steps="${BASELINE_EVAL_STEPS}"
        if [[ "${BASELINE_USE_TRAJECTORY_STEPS}" == "1" ]]; then
            baseline_collect_steps="${motion_steps}"
            baseline_eval_steps="${motion_steps}"
        fi
        run_cmd env \
            "TASK=${BASELINE_TASK}" \
            "ALGORITHM=${BASELINE_ALGO}" \
            "TRAIN_MANIFEST=${single_manifest}" \
            "EVAL_MANIFEST=${single_manifest}" \
            "OUTPUT_ROOT=${trajectory_root}/hand_designed_chunk_baselines" \
            "INTERFACES=${BASELINE_INTERFACES}" \
            "FULL_BODY_TRAJECTORY_CHECKPOINT=${FULL_BODY_TRAJECTORY_CHECKPOINT}" \
            "EE_TRAJECTORY_CHECKPOINT=${EE_TRAJECTORY_CHECKPOINT}" \
            "NUM_ENVS=${BASELINE_NUM_ENVS}" \
            "EVAL_STEPS=${baseline_eval_steps}" \
            "COLLECT_STEPS=${baseline_collect_steps}" \
            "SEED=${SEED}" \
            "STATE_HISTORY_STEPS=${STATE_HISTORY_STEPS}" \
            "COMMAND_PAST_STEPS=${BASELINE_COMMAND_PAST_STEPS}" \
            "COMMAND_FUTURE_STEPS=${BASELINE_COMMAND_FUTURE_STEPS}" \
            "PLANNER_UPDATE_INTERVAL=${BASELINE_PLANNER_UPDATE_INTERVAL}" \
            "COMMAND_HOLD_STEPS=${BASELINE_COMMAND_HOLD_STEPS}" \
            "MODEL_SIZE=${BASELINE_MODEL_SIZE}" \
            "MODEL_SIZES=${BASELINE_MODEL_SIZE}" \
            "SAMPLE_BUDGETS=${BASELINE_SAMPLE_BUDGETS}" \
            "PRETRAIN_UPDATES=${BASELINE_PRETRAIN_UPDATES}" \
            "FINETUNE_UPDATES=${BASELINE_FINETUNE_UPDATES}" \
            "DRY_RUN=${DRY_RUN}" \
            bash experiments/interface_baselines/run_dance102_strong_interface_comparison.sh
    fi
done

log "Done. Results under ${RUN_ROOT_ABS}"
