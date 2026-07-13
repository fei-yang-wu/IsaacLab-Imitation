#!/usr/bin/env bash
# Reproducible local LAFAN1 pipeline: pretrain a DiffSR skill encoder, then train
# the low-level oracle IPMD policy conditioned on that encoder (command_source=hl_skill).
#
# Stage 1 builds the zarr cache from scratch (refresh_zarr_dataset=true, joint order
# canonicalized to the robot articulation order); stage 2 reuses it.
#
# Every parameter below is the validated default and can be overridden via env vars, e.g.
#   TOTAL_FRAMES=500000000 LOGGER_BACKEND=none bash scripts/rlopt/run_local_pretrain_lowlevel.sh
# The low-level hl_skill/latent hyperparameters are baked config defaults
# (G1ImitationLatentRLOptIPMDConfig); the only per-run override is the skill checkpoint path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

# --- shared -----------------------------------------------------------------
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
MANIFEST_PATH="${MANIFEST_PATH:-data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-data/lafan1/g1_hl_diffsr}"

# --- stage 1: skill-encoder pretrain (default DiffSR architecture) ----------
HORIZON_STEPS="${HORIZON_STEPS:-25}"
Z_DIM="${Z_DIM:-256}"
ENCODER_WINDOW_MODE="${ENCODER_WINDOW_MODE:-intermediate}"
DIFFSR_FEATURE_DIM="${DIFFSR_FEATURE_DIM:-128}"
DIFFSR_EMBED_DIM="${DIFFSR_EMBED_DIM:-512}"
SKILL_UPDATES="${SKILL_UPDATES:-5000}"
SKILL_BATCH_SIZE="${SKILL_BATCH_SIZE:-8192}"
EVAL_TRAJECTORY_FRACTION="${EVAL_TRAJECTORY_FRACTION:-0.5}"

# --- stage 2: low-level oracle IPMD -----------------------------------------
LOW_LEVEL_ALGO="${LOW_LEVEL_ALGO:-IPMD}"
TOTAL_FRAMES="${TOTAL_FRAMES:-2000000000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2500}"
LOGGER_BACKEND="${LOGGER_BACKEND:-wandb}"
LOGGER_PROJECT_NAME="${LOGGER_PROJECT_NAME:-g1-lafan1-hl-skill-2b}"

# Skip toggles for re-runs (SKIP_PRETRAIN reuses an existing SKILL_CKPT).
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"

MANIFEST_ABS="$(realpath "${MANIFEST_PATH}")"
DATASET_ABS="$(realpath -m "${DATASET_PATH}")"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_lafan1_h${HORIZON_STEPS}_z${Z_DIM}_pretrain_lowlevel}"
RUN_ROOT="$(mkdir -p "logs/local_pretrain_lowlevel/${RUN_ID}" && realpath "logs/local_pretrain_lowlevel/${RUN_ID}")"
SKILL_DIR="${SKILL_DIR:-${RUN_ROOT}/skill_encoder_h${HORIZON_STEPS}_z${Z_DIM}}"
SKILL_CKPT="${SKILL_CKPT:-${SKILL_DIR}/checkpoints/best.pt}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

if [[ ! -f "${MANIFEST_ABS}" ]]; then
    log "ERROR: LAFAN1 manifest not found: ${MANIFEST_ABS}"
    log "HINT: prepare the full G1 LAFAN1 data first (see README 'Data preparation')."
    exit 1
fi

log "Run root: ${RUN_ROOT}"
log "Manifest: ${MANIFEST_ABS}"
log "Dataset (zarr): ${DATASET_ABS}"
log "Stage 1: W=${HORIZON_STEPS} z=${Z_DIM} window=${ENCODER_WINDOW_MODE} updates=${SKILL_UPDATES}"
log "Stage 2: algo=${LOW_LEVEL_ALGO} frames=${TOTAL_FRAMES} logger=${LOGGER_BACKEND}"

# ---------------------------------------------------------------------------
# Stage 1: pretrain skill encoder. refresh_zarr_dataset=true rebuilds the cache
# from scratch with the canonical (articulation-order) joint layout.
# ---------------------------------------------------------------------------
if [[ "${SKIP_PRETRAIN}" == "1" ]]; then
    if [[ ! -f "${SKILL_CKPT}" ]]; then
        log "SKIP_PRETRAIN=1 but SKILL_CKPT missing: ${SKILL_CKPT}"
        exit 1
    fi
    log "STAGE 1 skipped; reusing ${SKILL_CKPT}"
else
    log "STAGE 1: skill-encoder pretrain -> ${SKILL_CKPT}"
    pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py \
        --headless \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num_envs "${NUM_ENVS}" \
        --seed "${SEED}" \
        --output_dir "${SKILL_DIR}" \
        --horizon_steps "${HORIZON_STEPS}" \
        --encoder_window_mode "${ENCODER_WINDOW_MODE}" \
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
        --eval_trajectory_fraction "${EVAL_TRAJECTORY_FRACTION}" \
        --trajectory_split_seed "${SEED}" \
        --reconstruction_eval \
        --window_probe_eval \
        --window_probe_train_batches 8 \
        --window_probe_eval_batches 4 \
        "env.lafan1_manifest_path=${MANIFEST_ABS}" \
        "env.dataset_path=${DATASET_ABS}" \
        "env.refresh_zarr_dataset=true"
fi

if [[ ! -f "${SKILL_CKPT}" ]]; then
    log "ERROR: skill checkpoint not found: ${SKILL_CKPT}"
    exit 1
fi
log "Skill checkpoint ready: ${SKILL_CKPT}"

# ---------------------------------------------------------------------------
# Stage 2: low-level oracle IPMD (command_source=hl_skill). All hl_skill/latent
# params are baked config defaults; per-run override is only the checkpoint path.
# Reuses the stage-1 zarr (refresh_zarr_dataset=false).
# ---------------------------------------------------------------------------
log "STAGE 2: low-level oracle ${LOW_LEVEL_ALGO} (frames=${TOTAL_FRAMES})"
pixi run -e isaaclab python scripts/rlopt/train.py \
    --headless \
    --video \
    --video_length "${VIDEO_LENGTH}" \
    --video_interval "${VIDEO_INTERVAL}" \
    --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" \
    --task "${TASK}" \
    --algo "${LOW_LEVEL_ALGO}" \
    --seed "${SEED}" \
    "agent.collector.total_frames=${TOTAL_FRAMES}" \
    "agent.logger.backend=${LOGGER_BACKEND}" \
    "agent.logger.project_name=${LOGGER_PROJECT_NAME}" \
    "agent.logger.exp_name=${RUN_ID}_oracle_low_level" \
    "agent.logger.video=true" \
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CKPT}" \
    "env.lafan1_manifest_path=${MANIFEST_ABS}" \
    "env.dataset_path=${DATASET_ABS}" \
    "env.refresh_zarr_dataset=false"

log "DONE: pretrain + low-level complete. Run root: ${RUN_ROOT}"
