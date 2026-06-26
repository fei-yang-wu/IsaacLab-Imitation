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

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("${PYTHON_BIN}")
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
elif command -v pixi >/dev/null 2>&1; then
    PYTHON_CMD=(pixi run -e isaaclab python)
else
    PYTHON_CMD=(python3)
fi
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
LOW_LEVEL_ALGO="${LOW_LEVEL_ALGO:-IPMD_BILINEAR}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
MANIFEST_PATH="${MANIFEST_PATH:-data/unitree/manifests/g1_unitree_dance102_manifest.json}"
DATASET_PATH="${DATASET_PATH:-data/unitree/g1_dance102_hl_diffsr}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
EVAL_VIDEO_LENGTH="${EVAL_VIDEO_LENGTH:-500}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-0}"
EVAL_METRIC_INTERVAL="${EVAL_METRIC_INTERVAL:-5}"
PLANNER_FLOW_STEPS="${PLANNER_FLOW_STEPS:-16}"
PLANNER_EVAL_FLOW_NOISE_STD="${PLANNER_EVAL_FLOW_NOISE_STD:-0.0}"
PIPELINE_SESSION="${PIPELINE_SESSION:-dance102-nolang}"
WAIT_FOR_LOW_LEVEL="${WAIT_FOR_LOW_LEVEL:-1}"

if [[ -z "${RUN_ROOT:-}" ]]; then
    echo "RUN_ROOT is required, for example:"
    echo "  RUN_ROOT=logs/dance102_single_trajectory_debug/<run-id> $0"
    exit 2
fi

RUN_ROOT_ABS="$(realpath -m "${RUN_ROOT}")"
MANIFEST_ABS="$(realpath "${MANIFEST_PATH}")"
DATASET_ABS="$(realpath -m "${DATASET_PATH}")"
mkdir -p "${RUN_ROOT_ABS}"

LOG_FILE="${RUN_ROOT_ABS}/posthoc_eval.log"
COMMAND_LOG="${RUN_ROOT_ABS}/posthoc_eval_commands.sh"
: > "${COMMAND_LOG}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_cmd() {
    log "RUN: $*"
    printf '%q ' "$@" >> "${COMMAND_LOG}"
    printf '\n' >> "${COMMAND_LOG}"
    "$@"
}

if [[ "${WAIT_FOR_LOW_LEVEL}" == "1" ]]; then
    log "Waiting for low-level checkpoint under ${RUN_ROOT_ABS}"
    while true; do
        if [[ -s "${RUN_ROOT_ABS}/low_level_checkpoint.txt" ]]; then
            if grep -q "Pipeline complete" "${RUN_ROOT_ABS}/pipeline.log" 2>/dev/null; then
                log "Main pipeline completed; running posthoc achieved-state eval."
                break
            fi
            if ! tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null; then
                log "Main pipeline session is gone; running posthoc eval with the available checkpoint."
                break
            fi
        fi
        sleep 60
    done
fi

SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-${RUN_ROOT_ABS}/skill_encoder_h${HORIZON_STEPS}_z${Z_DIM}/checkpoints/latest.pt}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${RUN_ROOT_ABS}/planner_flow_matching_no_language_hist${HORIZON_STEPS}/checkpoints/latest.pt}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-$(cat "${RUN_ROOT_ABS}/low_level_checkpoint.txt")}"

if [[ ! -f "${SKILL_CHECKPOINT}" ]]; then
    log "Missing skill checkpoint: ${SKILL_CHECKPOINT}"
    exit 1
fi
if [[ ! -f "${PLANNER_CHECKPOINT}" ]]; then
    log "Missing planner checkpoint: ${PLANNER_CHECKPOINT}"
    exit 1
fi
if [[ ! -f "${LOW_LEVEL_CHECKPOINT}" ]]; then
    log "Missing low-level checkpoint: ${LOW_LEVEL_CHECKPOINT}"
    exit 1
fi

log "trajectory=dance102 language=none"
log "skill_checkpoint=${SKILL_CHECKPOINT}"
log "planner_checkpoint=${PLANNER_CHECKPOINT}"
log "low_level_checkpoint=${LOW_LEVEL_CHECKPOINT}"
log "planner_condition=achieved_state_history_${HORIZON_STEPS}"
log "planner_eval=max_steps:${EVAL_MAX_STEPS} video_length:${EVAL_VIDEO_LENGTH} metric_interval:${EVAL_METRIC_INTERVAL}"

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

run_cmd "${PYTHON_CMD[@]}" scripts/rlopt/play.py \
    --headless \
    --video \
    --video_length "${EVAL_VIDEO_LENGTH}" \
    --output_dir "${RUN_ROOT_ABS}/video_eval_oracle_hl_skill_posthoc" \
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

PLANNER_EVAL_ARGS=(
    --headless \
    --video \
    --video_length "${EVAL_VIDEO_LENGTH}" \
    --device "${DEVICE}" \
    --num_envs 1 \
    --task "${TASK}" \
    --algorithm "${LOW_LEVEL_ALGO}" \
    --seed "${SEED}" \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --planner_checkpoint "${PLANNER_CHECKPOINT}" \
    --skill_checkpoint "${SKILL_CHECKPOINT}" \
    --output_dir "${RUN_ROOT_ABS}/closed_loop_eval_trained_planner_no_language_achieved_state" \
    --metric_interval "${EVAL_METRIC_INTERVAL}" \
    --flow_num_inference_steps "${PLANNER_FLOW_STEPS}" \
    --flow_inference_noise_std "${PLANNER_EVAL_FLOW_NOISE_STD}" \
    "agent.ipmd.command_source=skill_commander" \
    "agent.ipmd.skill_commander_checkpoint_path=${PLANNER_CHECKPOINT}" \
    "agent.ipmd.skill_commander_embeddings_path=" \
    "agent.ipmd.skill_commander_flow_num_inference_steps=${PLANNER_FLOW_STEPS}" \
    "agent.ipmd.skill_commander_flow_inference_noise_std=${PLANNER_EVAL_FLOW_NOISE_STD}" \
    "agent.ipmd.skill_commander_use_achieved_state=true" \
    "agent.ipmd.hl_skill_finetune_enabled=false" \
    "${COMMON_LATENT_OVERRIDES[@]}"
)
if (( EVAL_MAX_STEPS > 0 )); then
    PLANNER_EVAL_ARGS+=(--max_steps "${EVAL_MAX_STEPS}")
fi

run_cmd "${PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
    "${PLANNER_EVAL_ARGS[@]}"

log "Posthoc eval complete. Videos are under:"
log "  oracle:  ${RUN_ROOT_ABS}/video_eval_oracle_hl_skill_posthoc/videos/play"
log "  planner: ${RUN_ROOT_ABS}/closed_loop_eval_trained_planner_no_language_achieved_state/videos/play"
log "Planner metrics are under:"
log "  ${RUN_ROOT_ABS}/closed_loop_eval_trained_planner_no_language_achieved_state"
