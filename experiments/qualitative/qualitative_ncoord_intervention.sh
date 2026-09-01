#!/usr/bin/env bash
set -euo pipefail

# Analysis 2: the n-group intervention, on FSQ coordinates.
#
# One knob: how many coordinates get resampled. Always 32 robots. One set of
# N_GROUPS coordinates is chosen and SHARED by every robot; robot 0 keeps the
# base code as the visual baseline, and robots 1..31 each resample the LEVEL of
# those N coordinates -- distinct levels per coordinate across robots, so no two
# robots duplicate each other.
#
# Sharing the coordinate set is what makes a sweep over N_GROUPS interpretable:
# between two runs the only thing that changes is how MANY coordinates moved,
# not also which ones.
#
#   DRY_RUN=1 bash experiments/qualitative/qualitative_ncoord_intervention.sh
#   SMOKE=1   bash experiments/qualitative/qualitative_ncoord_intervention.sh
#   N_GROUPS=4 bash experiments/qualitative/qualitative_ncoord_intervention.sh
#
# Sweep the axis (this is the point of the mode):
#
#   for n in 1 2 4 8 16 32; do
#     N_GROUPS=$n bash experiments/qualitative/qualitative_ncoord_intervention.sh
#   done
#
# Scale note against the multicat analysis: one FSQ coordinate owns 1 of 64
# latent values, where one multicat group owns 4 of 256. The FRACTION of z
# perturbed at a given N_GROUPS is therefore the same (N/64), but the absolute
# count differs, and the per-edit magnitude is bounded: a level edit moves that
# coordinate by (new - old)/16 within [-1, 0.9375]. 31 non-base levels exist, so
# 32 robots is the largest grid with no duplicate level on a coordinate.
#
# Protocol: execute the motion under normal encoder control for WARMUP_SECONDS,
# encode the next macro window as the base code, apply the edit, hold for
# ROLLOUT_STEPS. Only z is frozen; the sin/cos phase keeps cycling with period
# HORIZON_STEPS exactly as in training.
#
# Common knobs:
#
#   N_GROUPS=8                     coordinates resampled (1..64)
#   RANK=4844                      pin the motion by rank
#   MOTION=victory_dance_beyonce_moves_front_R_002_A309_M
#   ROLLOUT_STEPS=500              hold length (500 = 10 s)
#   WARMUP_SECONDS=1.0             encoder-driven prefix before the switch
#   TRACKER_ARM=sonic              bind the other capacity arm
#   SEED=1                         different base code and different coordinates
#   CUDA_VISIBLE_DEVICES=1,2       see qualitative_env.sh (>=2 needed for video)
#
# An edited lattice point may never have occurred in training, so a difference
# between robots is evidence about local controller response, not a semantic
# label.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qualitative_env.sh"

# --- mode constants ---------------------------------------------------------
MODE_NAME="ncoord_intervention"
# The single axis of the design. Named N_GROUPS because it is the shared
# entrypoint's --num_groups; for this encoder a "group" is one FSQ coordinate.
N_GROUPS="${N_GROUPS:-1}"
# 32 robots always: robot 0 is the baseline, 31 carry perturbed codes.
VARIANTS="${VARIANTS:-32}"
# 500 steps = 10 s at dt=0.02 -- the repo's standard episode length. The code is
# held for this whole span, so differences between variants accumulate.
ROLLOUT_STEPS="${ROLLOUT_STEPS:-500}"
START_FRAME="${START_FRAME:-0}"
WARMUP_SECONDS="${WARMUP_SECONDS:-1.0}"
BASE_CODE="${BASE_CODE:-encoded}"
# Parallel envs are not bit-deterministic and humanoid contact dynamics are
# chaotic, so the prefix always spreads the robots a little (~0.1 m after 1 s).
# This catches gross divergence only; the measured spread is always recorded.
MAX_WARMUP_DRIFT="${MAX_WARMUP_DRIFT:-0.5}"
MOTION="${MOTION:-}"
RANK="${RANK:-}"
VIDEO="${VIDEO:-1}"
# A 32-robot grid overruns the training contact limits. 320, not the training
# 289: the matched evaluation of an FSQ64 tracker overflowed the Newton
# constraint buffer at 289 and its valid results used 320.
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODE_NAME}_n${N_GROUPS}}"

if [[ "${SMOKE}" == "1" ]]; then
    VARIANTS=4
    ROLLOUT_STEPS=20
    WARMUP_SECONDS=0.2
    OUTPUT_DIR="${OUTPUT_ROOT}-smoke/${MODE_NAME}_n${N_GROUPS}"
    OVERWRITE=1
fi

[[ "${VIDEO}" == "1" ]] && qualitative_require_render_gpus
qualitative_require_discrete_code
qualitative_check_data
qualitative_check_encoder
qualitative_resolve_policy
ablate_base_overrides

cmd=(
    pixi run -e isaaclab python experiments/qualitative/src/qualitative_code_intervention.py
    --task "${TASK_NAME}" --headless --device "${DEVICE}"
    --mode n_groups
    --num_groups "${N_GROUPS}"
    --encoder_checkpoint "${ENCODER_CKPT}"
    --policy_checkpoint "${POLICY_CKPT}"
    --reference_arrays_dir "${REFERENCE_ARRAYS_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --variants "${VARIANTS}"
    --rollout_steps "${ROLLOUT_STEPS}"
    --start_frame "${START_FRAME}"
    --seed "${SEED}"
    --warmup_seconds "${WARMUP_SECONDS}"
    --base_code "${BASE_CODE}"
    --max_warmup_drift "${MAX_WARMUP_DRIFT}"
    --njmax "${NJMAX}" --nconmax "${NCONMAX}"
)
[[ -n "${MOTION}" ]] && cmd+=(--motion "${MOTION}")
[[ -n "${RANK}" ]] && cmd+=(--rank "${RANK}")
[[ "${VIDEO}" == "1" ]] && cmd+=(--video)
[[ "${OVERWRITE}" == "1" ]] && cmd+=(--overwrite)
cmd+=("${BASE_OVERRIDES[@]}")

echo "[PLAN] mode        : ${MODE_NAME} (analysis 2; N_GROUPS is the only axis)"
echo "[PLAN] code space  : ${ABLATE_CODE_SPACE_DESC}"
echo "[PLAN] encoder     : ${ENCODER_CKPT}"
echo "[PLAN] encoder sha : ${ENCODER_SHA256}"
echo "[PLAN] policy      : ${POLICY_CKPT}"
echo "[PLAN] policy sha  : ${POLICY_SHA256}"
echo "[PLAN] tracker     : ${TRACKER_ARM} ${ABLATE_TRACKER_CELLS} silu"
echo "[PLAN] design      : ${VARIANTS} robots, robot 0 = baseline, ${N_GROUPS} shared coordinate(s) resampled"
echo "[PLAN] edit size   : ${N_GROUPS} of ${Z_DIM} latent values move"
echo "[PLAN] warmup      : ${WARMUP_SECONDS}s encoder-driven prefix, base code = ${BASE_CODE}"
echo "[PLAN] hold        : ${ROLLOUT_STEPS} steps"
echo "[PLAN] start       : motion=${MOTION:-<seeded draw>} rank=${RANK:-<derived>} frame=${START_FRAME}"
echo "[PLAN] GPU         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -> ${DEVICE}"
echo "[PLAN] output      : ${OUTPUT_DIR}"

qualitative_run "${cmd[@]}"

qualitative_require_output "${OUTPUT_DIR}"
