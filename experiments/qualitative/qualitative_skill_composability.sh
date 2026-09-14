#!/usr/bin/env bash
set -euo pipefail

# Analysis 5: skill composability -- eight robots chaining random codes.
#
# Every SEGMENT_STEPS steps each robot independently draws a fresh uniform code
# from the prior (for sonic_fsq: one of the 32 lattice levels for each of the 64
# coordinates) and holds it until the next boundary. NUM_SEGMENTS such segments
# make 20 s of continuously changing intent per robot, and the eight robots draw
# INDEPENDENT sequences, so one MP4 holds eight samples of the same question:
# can the tracker absorb an arbitrary new skill from whatever body state the
# previous one left it in?
#
#   DRY_RUN=1 bash experiments/qualitative/qualitative_skill_composability.sh
#   SMOKE=1   bash experiments/qualitative/qualitative_skill_composability.sh
#   bash experiments/qualitative/qualitative_skill_composability.sh
#
# That is the property a downstream planner depends on. A planner publishes a
# new z every window and never chooses the state it inherits, so a tracker that
# only works from a rest pose is not usable even when every individual code
# looks fine on its own.
#
# The clip opens with WARMUP_SECONDS of ordinary encoder-driven tracking on one
# real motion, rounded down to whole command windows, so the robots enter the
# first random code upright and moving and a later fall is attributable to the
# codes rather than to the spawn pose.
#
# A uniformly random code is almost certainly out of distribution, so robots do
# fall. With RESET_FALLEN=1 (the default) a robot below FALL_HEIGHT at a segment
# boundary is put back on its reference pose AT THAT BOUNDARY, which keeps all
# eight lanes alive for the whole clip. Every such reset is recorded per robot
# and per segment, so a large behavioural jump beside a reset is never read as
# the code alone. RESET_FALLEN=0 leaves a fallen robot down.
#
# fsq64 only: a deterministic latent has no alphabet to draw uniformly from, and
# no continuous analogue is invented here. The launcher refuses deter64 before
# Isaac starts and the entrypoint refuses it again at encoder load.
#
# Common knobs:
#
#   NUM_ROBOTS=8                   robots, each with its own code sequence
#   SEGMENT_STEPS=100              steps one code is held (100 = 2 s)
#   NUM_SEGMENTS=10                codes per robot (10 x 2 s = 20 s)
#   WARMUP_SECONDS=1.0             encoder-driven prefix before the first code
#   FALL_HEIGHT=0.4                base height counted as a fall
#   RESET_FALLEN=0                 leave a fallen robot down
#   MOTION=<name> / RANK=<n>       pin the warmup motion
#   SEED=1                         draw a different set of code sequences
#   ENV_SPACING=4.0                metres between robots on screen
#   VIDEO=0                        metrics only
#   CUDA_VISIBLE_DEVICES=1,3       two render-capable GPUs
#
# SEGMENT_STEPS must be a whole multiple of HORIZON_STEPS, so every switch lands
# on a command-window boundary and no window ever mixes two codes. The
# entrypoint asserts it.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qualitative_env.sh"

# --- mode constants ---------------------------------------------------------
MODE_NAME="skill_composability"
NUM_ROBOTS="${NUM_ROBOTS:-8}"
# 100 steps = 2 s at dt=0.02, and 10 whole command windows at HORIZON_STEPS=10.
SEGMENT_STEPS="${SEGMENT_STEPS:-100}"
# 10 segments x 2 s = 20 s of random codes, after the warmup.
NUM_SEGMENTS="${NUM_SEGMENTS:-10}"
WARMUP_SECONDS="${WARMUP_SECONDS:-1.0}"
START_FRAME="${START_FRAME:-0}"
FALL_HEIGHT="${FALL_HEIGHT:-0.4}"
RESET_FALLEN="${RESET_FALLEN:-1}"
MOTION="${MOTION:-}"
RANK="${RANK:-}"
ENV_SPACING="${ENV_SPACING:-}"
VIDEO="${VIDEO:-1}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODE_NAME}}"

if [[ "${SMOKE}" == "1" ]]; then
    NUM_ROBOTS=4
    NUM_SEGMENTS=3
    SEGMENT_STEPS=20
    WARMUP_SECONDS=0.2
    OUTPUT_DIR="${OUTPUT_ROOT}-smoke/${MODE_NAME}"
    OVERWRITE=1
fi

[[ "${VIDEO}" == "1" ]] && qualitative_require_render_gpus
qualitative_require_discrete_code \
    "draws a uniformly random code from the prior every ${SEGMENT_STEPS} steps,
  one category per group"
qualitative_check_data
qualitative_check_encoder
qualitative_resolve_policy
ablate_base_overrides

cmd=(
    pixi run -e isaaclab python experiments/qualitative/src/qualitative_skill_composability.py
    --task "${TASK_NAME}" --headless --device "${DEVICE}"
    --encoder_checkpoint "${ENCODER_CKPT}"
    --policy_checkpoint "${POLICY_CKPT}"
    --reference_arrays_dir "${REFERENCE_ARRAYS_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --num_robots "${NUM_ROBOTS}"
    --segment_steps "${SEGMENT_STEPS}"
    --num_segments "${NUM_SEGMENTS}"
    --warmup_seconds "${WARMUP_SECONDS}"
    --fall_height "${FALL_HEIGHT}"
    --start_frame "${START_FRAME}"
    --seed "${SEED}"
    --njmax "${NJMAX}" --nconmax "${NCONMAX}"
)
if [[ "${RESET_FALLEN}" == "1" ]]; then
    cmd+=(--reset_fallen)
else
    cmd+=(--no_reset_fallen)
fi
[[ -n "${MOTION}" ]] && cmd+=(--motion "${MOTION}")
[[ -n "${RANK}" ]] && cmd+=(--rank "${RANK}")
[[ -n "${ENV_SPACING}" ]] && cmd+=(--env_spacing "${ENV_SPACING}")
[[ "${VIDEO}" == "1" ]] && cmd+=(--video)
[[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
cmd+=("${BASE_OVERRIDES[@]}")

echo "[PLAN] mode        : ${MODE_NAME} (analysis 5)"
echo "[PLAN] code space  : ${ABLATE_CODE_SPACE_DESC}"
echo "[PLAN] encoder     : ${ENCODER_CKPT}"
echo "[PLAN] encoder sha : ${ENCODER_SHA256}"
echo "[PLAN] policy      : ${POLICY_CKPT}"
echo "[PLAN] policy sha  : ${POLICY_SHA256}"
echo "[PLAN] tracker     : ${TRACKER_ARM} ${ABLATE_TRACKER_CELLS} silu"
echo "[PLAN] grid        : ${NUM_ROBOTS} robots, independent code sequences, ONE video"
echo "[PLAN] schedule    : ${WARMUP_SECONDS}s warmup, then ${NUM_SEGMENTS} x ${SEGMENT_STEPS} steps"
echo "[PLAN] codes       : ${NUM_ROBOTS} x ${NUM_SEGMENTS} uniform draws over ${Z_DIM} coordinates"
echo "[PLAN] falls       : below ${FALL_HEIGHT} m -> $([[ "${RESET_FALLEN}" == "1" ]] && echo "reset at the next boundary" || echo "left down")"
echo "[PLAN] warmup ref  : motion=${MOTION:-<seeded draw>} rank=${RANK:-<derived>} frame=${START_FRAME}"
echo "[PLAN] GPU         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -> ${DEVICE}"
echo "[PLAN] output      : ${OUTPUT_DIR}"

qualitative_run "${cmd[@]}"

qualitative_require_output "${OUTPUT_DIR}"
