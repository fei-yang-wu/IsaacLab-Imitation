#!/usr/bin/env bash
# Local two-stage G1 pipeline on the CURRENT default: pretrain a deterministic
# continuous (det-SR) skill encoder, then train the low-level tracker on the
# tuned recipe, conditioned on that encoder.
#
# This is the script to run for a local experiment. It is the same task, agent
# contract, encoder and data the ICE 5B runs use; only the scale differs.
#
#   bash scripts/rlopt/run_local_v2_pipeline.sh                      # both stages
#   TOTAL_FRAMES=10000000 bash scripts/rlopt/run_local_v2_pipeline.sh  # quick check
#   SKIP_PRETRAIN=1 SKILL_CKPT=<path> bash scripts/rlopt/run_local_v2_pipeline.sh
#   DRY_RUN=1 bash scripts/rlopt/run_local_v2_pipeline.sh            # print, run nothing
#
# `scripts/rlopt/run_local_pretrain_lowlevel.sh` is the FROZEN predecessor: it
# targets `Isaac-Imitation-G1-Latent-v0` with a 25-step horizon and the legacy
# flat `env.lafan1_manifest_path` surface. Keep it only to reproduce pre-v2 runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
TASK="${TASK:-Isaac-Imitation-G1-v2}"

# The tuned ("scaled") recipe, selected BY ENTRY POINT. It is a separate
# registered class, not a change to the base contract, so every earlier run and
# the in-flight cluster jobs keep resolving what they resolved. Do not
# reconstruct its fields from a copied override list -- a stale copy of the
# rollout is what shipped two 5B runs at the wrong geometry on 2026-08-03.
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"

# --- data -------------------------------------------------------------------
MANIFEST_PATH="${MANIFEST_PATH:-./data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-./data/lafan1/zarr/g1_hl_diffsr}"
# The FIRST run on a fresh dataset must build the zarr cache; after that leave
# this false, because a refresh rebuilds the cache underneath any other job
# reading it.
CACHE_REFRESH="${CACHE_REFRESH:-false}"

# --- stage 1: deterministic continuous (det-SR) encoder ----------------------
LATENT_MODE="${LATENT_MODE:-deterministic}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"

# `sin_cos` adds a 2-value phase to the published command. Dropping it is
# catastrophic (episode length 21 against 144 in the 2026-08-02 screen), so the
# tracker's command width is z_dim + 2, not z_dim.
PHASE_MODE="${PHASE_MODE:-sin_cos}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# --- stage 2: low-level tracker ---------------------------------------------
NUM_ENVS="${NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"   # what the tuned recipe resolves; sizes iterations only
TOTAL_FRAMES="${TOTAL_FRAMES:-50000000}"
LOGGER_BACKEND="${LOGGER_BACKEND:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-local-v2}"

# Physics: newton_mjwarp with the mjwarp-aligned solver settings the cluster
# runs use. `physics=physx` also works and is the faster path for a tiny debug
# run, but it is NOT what the cluster trains on.
PHYSICS="${PHYSICS:-newton_mjwarp}"
NJMAX="${NJMAX:-288}"
NCONMAX="${NCONMAX:-200}"

# Terminations are INSTANTANEOUS by default -- the registered protocol every
# recorded oracle-qualification number is stated against. Set TERMINATION_WINDOW
# to opt into the persistence window (a term must hold N consecutive steps).
# The probe measured that the transients are ~10 steps, so 3 is under-powered;
# and a window inflates episode length and return mechanically, so compare on
# MPJPE only.
TERMINATION_WINDOW="${TERMINATION_WINDOW:-}"

SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_v2_${LATENT_MODE}_h${HORIZON_STEPS}_z${Z_DIM}}"
RUN_ROOT="logs/local_v2/${RUN_ID}"
SKILL_DIR="${SKILL_DIR:-${RUN_ROOT}/encoder}"
SKILL_CKPT="${SKILL_CKPT:-${SKILL_DIR}/checkpoints/latest.pt}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
run() { if [[ "${DRY_RUN}" == "1" ]]; then printf '[DRY] '; printf '%q ' "$@"; printf '\n'; else "$@"; fi; }

[[ -f "${MANIFEST_PATH}" ]] || {
    log "ERROR: manifest not found: ${MANIFEST_PATH}"
    log "HINT: prepare the G1 LAFAN1 data first (README 'Data preparation')."
    exit 1
}

FRAMES_PER_ITER=$((NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((TOTAL_FRAMES + FRAMES_PER_ITER - 1) / FRAMES_PER_ITER))

mkdir -p "${RUN_ROOT}"
log "run root   : ${RUN_ROOT}"
log "task/agent : ${TASK} / ${AGENT_ENTRY_POINT}"
log "encoder    : ${LATENT_MODE}, horizon ${HORIZON_STEPS}, z ${Z_DIM} -> command width ${LATENT_COMMAND_DIM}"
log "tracker    : ${NUM_ENVS} envs x ${ROLLOUT_STEPS} = ${FRAMES_PER_ITER}/iter, ${MAX_ITERATIONS} iters (~${TOTAL_FRAMES} frames)"
log "physics    : ${PHYSICS} (njmax ${NJMAX}, nconmax ${NCONMAX})"
if [[ -n "${TERMINATION_WINDOW}" ]]; then
    log "terminations: window ${TERMINATION_WINDOW} consecutive steps (thresholds unchanged)"
else
    log "terminations: instantaneous (the registered protocol)"
fi

# --- stage 1 ----------------------------------------------------------------
if [[ "${SKIP_PRETRAIN}" == "1" ]]; then
    [[ -f "${SKILL_CKPT}" ]] || { log "ERROR: SKIP_PRETRAIN=1 but no encoder at ${SKILL_CKPT}"; exit 1; }
    log "STAGE 1 skipped; reusing ${SKILL_CKPT}"
else
    log "STAGE 1: det-SR encoder -> ${SKILL_CKPT}"
    run pixi run -e isaaclab python scripts/rlopt/train_hl_skill_pipeline.py \
        --pretrain-only --headless --assert-kitless \
        --task "${TASK}" --seed "${SEED}" --device "${DEVICE}" \
        --latent-mode "${LATENT_MODE}" \
        --horizon-steps "${HORIZON_STEPS}" \
        --z-dim "${Z_DIM}" \
        --encoder-window-mode intermediate \
        --phase-mode "${PHASE_MODE}" \
        --latent-hold-steps "${LATENT_HOLD_STEPS}" \
        --manifest-path "${MANIFEST_PATH}" \
        --dataset-path "${DATASET_PATH}" \
        --pretrain-output-dir "${SKILL_DIR}" \
        --pretrain-num-envs "${PRETRAIN_NUM_ENVS}" \
        --pretrain-updates "${PRETRAIN_UPDATES}" \
        --pretrain-batch-size "${PRETRAIN_BATCH_SIZE}" \
        --logger-backend "${LOGGER_BACKEND}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group "${WANDB_GROUP}" \
        --pretrain-override "physics=${PHYSICS}" \
        --pretrain-override "env.data.cache_refresh=${CACHE_REFRESH}"
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${SKILL_CKPT}" ]]; then
    log "ERROR: encoder checkpoint missing after stage 1: ${SKILL_CKPT}"
    exit 1
fi

# --- stage 2 ----------------------------------------------------------------
# The encoder is FROZEN here (hl_skill_finetune_enabled=false): the tracker
# learns against a fixed command space, which is what makes the published
# command comparable across runs.
window_args=()
[[ -n "${TERMINATION_WINDOW}" ]] && window_args=(--termination_window "${TERMINATION_WINDOW}")

log "STAGE 2: low-level tracker on the tuned recipe"
run pixi run -e isaaclab python scripts/rlopt/train.py \
    --task "${TASK}" --algo IPMD --agent "${AGENT_ENTRY_POINT}" \
    --headless --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" --seed "${SEED}" \
    --max_iterations "${MAX_ITERATIONS}" \
    "${window_args[@]}" \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    "physics=${PHYSICS}" \
    "env.sim.physics.solver_cfg.njmax=${NJMAX}" \
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX}" \
    "env.data.manifest=${MANIFEST_PATH}" \
    "env.data.cache_dir=${DATASET_PATH}" \
    "env.data.cache_refresh=false" \
    "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}" \
    "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}" \
    agent.ipmd.command_source=hl_skill \
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CKPT}" \
    "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}" \
    agent.ipmd.hl_skill_command_mode=z \
    "agent.ipmd.latent_steps_min=${LATENT_HOLD_STEPS}" \
    "agent.ipmd.latent_steps_max=${LATENT_HOLD_STEPS}" \
    "agent.ipmd.latent_learning.code_period=${LATENT_HOLD_STEPS}" \
    "agent.ipmd.latent_learning.command_phase_mode=${PHASE_MODE}" \
    "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    "agent.logger.backend=${LOGGER_BACKEND}" \
    "agent.logger.project_name=${WANDB_PROJECT}" \
    "agent.logger.group_name=${WANDB_GROUP}" \
    "agent.logger.log_dir=${RUN_ROOT}/rlopt_train" \
    env.rewards.action_rate_l2.weight=0.0 \
    env.rewards.tracking_reward_points.weight=4.0 \
    env.enable_termination_curriculum=true \
    env.termination_curriculum_start_frames=5000000 \
    env.termination_curriculum_end_frames=30000000

log "done. checkpoints under ${RUN_ROOT}/rlopt_train"
