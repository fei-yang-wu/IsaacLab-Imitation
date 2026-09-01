#!/usr/bin/env bash
set -euo pipefail

# Analysis 3: reference motion vs. FSQ-driven rollout, side by side.
#
# For each of N motions, two robots share one scene: env 0 replays the expert
# reference articulation and env 1 is the low-level tracker, commanded by the
# 64-value command the frozen encoder produces from that same reference,
# re-encoded every window. One MP4 per motion.
#
#   DRY_RUN=1 bash experiments/qualitative/qualitative_reference_rollout.sh
#   SMOKE=1   bash experiments/qualitative/qualitative_reference_rollout.sh
#   bash experiments/qualitative/qualitative_reference_rollout.sh
#
# The window-by-window command comes from the agent's own frozen sampler
# (agent.ipmd.command_source=hl_skill), so this playback cannot drift from the
# way the tracker was trained. Terminations are disabled by default, so the
# tracking error is measured over the intended horizon rather than a
# termination-truncated rollout, and the retained video is that same
# full-horizon pass -- the diagnostic AGENTS.md requires.
#
# Common knobs:
#
#   NUM_MOTIONS=4                  how many motions to render
#   RANKS=110659,15944             pin explicit trajectory ranks
#   MOTIONS=jog_ff_loop_180_R_002_A091_M   pin explicit motion names
#   MAX_STEPS=300                  cap each clip
#   VIDEO=0                        skip rendering (metrics only)
#   KEEP_TERMINATIONS=1            stop the default full-horizon diagnostic pass
#   TRACKER_ARM=sonic              bind the other capacity arm
#   CUDA_VISIBLE_DEVICES=3         pick the GPU (see qualitative_env.sh)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qualitative_env.sh"

# --- mode constants ---------------------------------------------------------
MODE_NAME="reference_rollout"
NUM_MOTIONS="${NUM_MOTIONS:-8}"
START_FRAME="${START_FRAME:-0}"
MAX_STEPS="${MAX_STEPS:-}"
RANKS="${RANKS:-}"
MOTIONS="${MOTIONS:-}"
VIDEO="${VIDEO:-1}"
KEEP_TERMINATIONS="${KEEP_TERMINATIONS:-0}"
FALL_HEIGHT="${FALL_HEIGHT:-0.4}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODE_NAME}}"

if [[ "${SMOKE}" == "1" ]]; then
    # 2, not 1: a single motion never exercises the second reset, where
    # cross-iteration state leaks show up.
    NUM_MOTIONS=2
    MAX_STEPS=40
    OUTPUT_DIR="${OUTPUT_ROOT}-smoke/${MODE_NAME}"
    OVERWRITE=1
fi

[[ "${VIDEO}" == "1" ]] && qualitative_require_render_gpus
qualitative_check_data
qualitative_check_encoder
qualitative_resolve_policy
ablate_base_overrides

# Drive the actor command from the frozen encoder, exactly as in training.
HL_SKILL_OVERRIDES=(
    agent.ipmd.command_source=hl_skill
    "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT}"
    "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
    agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.hl_skill_finetune_enabled=false
)

cmd=(
    pixi run -e isaaclab python experiments/qualitative/src/qualitative_reference_rollout.py
    --task "${TASK_NAME}" --headless --device "${DEVICE}"
    --encoder_checkpoint "${ENCODER_CKPT}"
    --policy_checkpoint "${POLICY_CKPT}"
    --reference_arrays_dir "${REFERENCE_ARRAYS_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --num_motions "${NUM_MOTIONS}"
    --seed "${SEED}"
    --start_frame "${START_FRAME}"
    --fall_height "${FALL_HEIGHT}"
)
[[ -n "${MAX_STEPS}" ]] && cmd+=(--max_steps "${MAX_STEPS}")
[[ -n "${RANKS}" ]] && cmd+=(--ranks "${RANKS}")
[[ -n "${MOTIONS}" ]] && cmd+=(--motions "${MOTIONS}")
[[ "${VIDEO}" == "1" ]] && cmd+=(--video)
[[ "${KEEP_TERMINATIONS}" == "1" ]] && cmd+=(--keep_terminations)
[[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
cmd+=("${BASE_OVERRIDES[@]}" "${HL_SKILL_OVERRIDES[@]}")

echo "[PLAN] mode        : ${MODE_NAME} (analysis 3)"
echo "[PLAN] code space  : ${ABLATE_CODE_SPACE_DESC}"
echo "[PLAN] encoder     : ${ENCODER_CKPT}"
echo "[PLAN] encoder sha : ${ENCODER_SHA256}"
echo "[PLAN] policy      : ${POLICY_CKPT}"
echo "[PLAN] policy sha  : ${POLICY_SHA256}"
echo "[PLAN] tracker     : ${TRACKER_ARM} ${ABLATE_TRACKER_CELLS} silu"
echo "[PLAN] command     : ${Z_DIM} code + 2 phase = ${LATENT_COMMAND_DIM}, re-encoded every ${LATENT_HOLD_STEPS} steps"
echo "[PLAN] motions     : ${NUM_MOTIONS} (seed ${SEED}), start frame ${START_FRAME}"
echo "[PLAN] steps cap   : ${MAX_STEPS:-<until the reference ends>}"
echo "[PLAN] video       : ${VIDEO} (full-horizon diagnostic: $([[ "${KEEP_TERMINATIONS}" == "1" ]] && echo no || echo yes))"
echo "[PLAN] GPU         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -> ${DEVICE}"
echo "[PLAN] output      : ${OUTPUT_DIR}"

qualitative_run "${cmd[@]}"

qualitative_require_output "${OUTPUT_DIR}"
