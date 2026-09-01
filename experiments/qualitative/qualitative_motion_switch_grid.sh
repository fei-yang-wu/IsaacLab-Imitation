#!/usr/bin/env bash
set -euo pipefail

# Analysis 6: eight robots, eight motions, one shared switch, ONE video.
#
# Each robot tracks a different reference motion for SWITCH_AT_STEP steps. Then
# every robot's reference is retargeted to the SAME motion -- a backward jump by
# default -- with no reset, so all eight carry their own physical state into the
# same new intent. The whole grid is rendered as a single MP4.
#
#   DRY_RUN=1 bash experiments/qualitative/qualitative_motion_switch_grid.sh
#   SMOKE=1   bash experiments/qualitative/qualitative_motion_switch_grid.sh
#   bash experiments/qualitative/qualitative_motion_switch_grid.sh
#
# This is the grid counterpart of qualitative_motion_switch.sh, which renders one
# clip per pair with a reference lane beside the robot. Here every environment is
# a controlled robot, which is what puts eight of them in one frame -- and it is
# why there is no reference lane here.
#
# The motion-frame command needs no such lane: the window arrives relative to
# each robot, and anchoring it on its own first frame cancels that transform
# exactly, leaving the motion alone. With SWITCH_COMMAND_FRAME=reference (the
# default) all eight robots therefore receive the IDENTICAL command after the
# switch and differ only in the state they carried into it -- which is the
# experiment for whether the code means anything on its own.
#
# Common knobs:
#
#   NUM_ROBOTS=8                   robots, and motions drawn
#   SWITCH_AT_STEP=200             steps on the first motion (200 = 4 s)
#   AFTER_STEPS=150                steps after the switch (150 = 3 s)
#   SWITCH_MOTION=<name>           what everyone switches to
#   SWITCH_RANK=4559               same, by rank
#   SWITCH_ALIGN=none              keep the dataset placement of the new motion
#   SWITCH_COMMAND_FRAME=robot     per-robot deployment command after the switch
#   MOTIONS=a,b,c                  pin the eight starting motions
#   SEED=1                         draw a different eight
#   ENV_SPACING=4.0                metres between robots on screen
#   VIDEO=0                        metrics only
#   CUDA_VISIBLE_DEVICES=1,3       two render-capable GPUs
#
# Motions are drawn only from those long enough to still be playing at the
# switch, so the "before" half is really the motion each robot is labelled with.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qualitative_env.sh"

# --- mode constants ---------------------------------------------------------
MODE_NAME="motion_switch_grid"
NUM_ROBOTS="${NUM_ROBOTS:-8}"
START_FRAME="${START_FRAME:-0}"
SWITCH_AT_STEP="${SWITCH_AT_STEP:-200}"
AFTER_STEPS="${AFTER_STEPS:-150}"
# jump_backward_004_A044 is rank 4559, 284 frames, and its name is unique in the
# 129,785-motion catalog, so it resolves without ambiguity.
SWITCH_MOTION="${SWITCH_MOTION:-jump_backward_004_A044}"
SWITCH_RANK="${SWITCH_RANK:-}"
SWITCH_START_FRAME="${SWITCH_START_FRAME:-0}"
SWITCH_ALIGN="${SWITCH_ALIGN:-xy}"
# reference: after the switch every robot receives the SAME command -- the
# shared motion encoded in its own frame, carrying no correction for where that
# robot is. robot: the ordinary deployment path, each robot's own view.
SWITCH_COMMAND_FRAME="${SWITCH_COMMAND_FRAME:-reference}"
MOTIONS="${MOTIONS:-}"
RANKS="${RANKS:-}"
ENV_SPACING="${ENV_SPACING:-}"
VIDEO="${VIDEO:-1}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODE_NAME}}"

if [[ "${SMOKE}" == "1" ]]; then
    NUM_ROBOTS=4
    SWITCH_AT_STEP=20
    AFTER_STEPS=20
    OUTPUT_DIR="${OUTPUT_ROOT}-smoke/${MODE_NAME}"
    OVERWRITE=1
fi

[[ "${VIDEO}" == "1" ]] && qualitative_require_render_gpus
qualitative_check_data
qualitative_check_encoder
qualitative_resolve_policy
ablate_base_overrides

# Drive the actor command from the frozen encoder, exactly as in training. Each
# environment gets its own command because the sampler reads its own window.
HL_SKILL_OVERRIDES=(
    agent.ipmd.command_source=hl_skill
    "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT}"
    "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
    agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.hl_skill_finetune_enabled=false
)

cmd=(
    pixi run -e isaaclab python experiments/qualitative/src/qualitative_motion_switch_grid.py
    --task "${TASK_NAME}" --headless --device "${DEVICE}"
    --encoder_checkpoint "${ENCODER_CKPT}"
    --policy_checkpoint "${POLICY_CKPT}"
    --reference_arrays_dir "${REFERENCE_ARRAYS_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --num_robots "${NUM_ROBOTS}"
    --seed "${SEED}"
    --start_frame "${START_FRAME}"
    --switch_at_step "${SWITCH_AT_STEP}"
    --after_steps "${AFTER_STEPS}"
    --switch_start_frame "${SWITCH_START_FRAME}"
    --switch_align "${SWITCH_ALIGN}"
    --switch_command_frame "${SWITCH_COMMAND_FRAME}"
    --njmax "${NJMAX}" --nconmax "${NCONMAX}"
)
if [[ -n "${SWITCH_RANK}" ]]; then
    cmd+=(--switch_rank "${SWITCH_RANK}")
else
    cmd+=(--switch_motion "${SWITCH_MOTION}")
fi
[[ -n "${MOTIONS}" ]] && cmd+=(--motions "${MOTIONS}")
[[ -n "${RANKS}" ]] && cmd+=(--ranks "${RANKS}")
[[ -n "${ENV_SPACING}" ]] && cmd+=(--env_spacing "${ENV_SPACING}")
[[ "${VIDEO}" == "1" ]] && cmd+=(--video)
[[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
cmd+=("${BASE_OVERRIDES[@]}" "${HL_SKILL_OVERRIDES[@]}")

echo "[PLAN] mode        : ${MODE_NAME} (analysis 6)"
echo "[PLAN] grid        : ${NUM_ROBOTS} robots, one motion each, ONE video"
echo "[PLAN] encoder     : ${ENCODER_CKPT}"
echo "[PLAN] encoder sha : ${ENCODER_SHA256}"
echo "[PLAN] policy      : ${POLICY_CKPT}"
echo "[PLAN] tracker     : ${TRACKER_ARM} ${ABLATE_TRACKER_CELLS} silu"
echo "[PLAN] switch      : step ${SWITCH_AT_STEP} -> ${SWITCH_RANK:-${SWITCH_MOTION}} frame ${SWITCH_START_FRAME}, align=${SWITCH_ALIGN}"
echo "[PLAN] command     : after the switch, encoded in the ${SWITCH_COMMAND_FRAME} frame"
echo "[PLAN] rollout     : ${SWITCH_AT_STEP} + ${AFTER_STEPS} = $((SWITCH_AT_STEP + AFTER_STEPS)) steps"
echo "[PLAN] motions     : ${MOTIONS:-<seeded draw, seed ${SEED}>}"
echo "[PLAN] GPU         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -> ${DEVICE}"
echo "[PLAN] output      : ${OUTPUT_DIR}"

qualitative_run "${cmd[@]}"

qualitative_require_output "${OUTPUT_DIR}"
