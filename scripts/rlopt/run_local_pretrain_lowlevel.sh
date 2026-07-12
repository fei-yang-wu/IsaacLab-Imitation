#!/usr/bin/env bash
# Local chained pretrain (DiffSR skill encoder) -> low-level oracle IPMD job.
# From-scratch: builds the zarr cache in stage 1, reuses it in stage 2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

DEVICE="cuda:0"
SEED=0
NUM_ENVS=4096
TASK="Isaac-Imitation-G1-Latent-v0"
MANIFEST_ABS="$(realpath data/lafan1/manifests/g1_lafan1_manifest.json)"
DATASET_ABS="$(realpath -m data/lafan1/g1_hl_diffsr)"

RUN_ID="$(date +%Y%m%d_%H%M%S)_lafan1_h25_z256_pretrain_lowlevel"
RUN_ROOT="$(mkdir -p "logs/local_pretrain_lowlevel/${RUN_ID}" && realpath "logs/local_pretrain_lowlevel/${RUN_ID}")"
SKILL_DIR="${RUN_ROOT}/skill_encoder_h25_z256"
SKILL_CKPT="${SKILL_DIR}/checkpoints/best.pt"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

log "Run root: ${RUN_ROOT}"
log "Manifest: ${MANIFEST_ABS}"
log "Dataset (zarr): ${DATASET_ABS}"

# ---------------------------------------------------------------------------
# Stage 1: pretrain skill encoder (default DiffSR architecture, W=25, z=256).
# refresh_zarr_dataset=true builds the cache from scratch with the fixed loader.
# ---------------------------------------------------------------------------
log "STAGE 1: skill-encoder pretrain -> ${SKILL_CKPT}"
pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py \
    --headless \
    --device "${DEVICE}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --output_dir "${SKILL_DIR}" \
    --horizon_steps 25 \
    --encoder_window_mode intermediate \
    --z_dim 256 \
    --diffsr_feature_dim 128 \
    --diffsr_embed_dim 512 \
    --batch_size 8192 \
    --num_updates 5000 \
    --log_interval 100 \
    --eval_batches 4 \
    --eval_batch_size 8192 \
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

if [[ ! -f "${SKILL_CKPT}" ]]; then
    log "ERROR: skill checkpoint not found: ${SKILL_CKPT}"
    exit 1
fi
log "Skill checkpoint ready: ${SKILL_CKPT}"

# ---------------------------------------------------------------------------
# Stage 2: low-level oracle IPMD (hl_skill). hl_skill/latent params are baked
# config defaults; per-run override is only the skill checkpoint path.
# 2B frame cap, wandb logging, reuse the stage-1 zarr.
# ---------------------------------------------------------------------------
log "STAGE 2: low-level oracle IPMD (2B frames, wandb)"
pixi run -e isaaclab python scripts/rlopt/train.py \
    --headless \
    --video \
    --video_length 500 \
    --video_interval 2500 \
    --device "${DEVICE}" \
    --num_envs "${NUM_ENVS}" \
    --task "${TASK}" \
    --algo IPMD \
    --seed "${SEED}" \
    "agent.collector.total_frames=2000000000" \
    "agent.logger.backend=wandb" \
    "agent.logger.project_name=g1-lafan1-hl-skill-2b" \
    "agent.logger.exp_name=${RUN_ID}_oracle_low_level" \
    "agent.logger.video=true" \
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CKPT}" \
    "env.lafan1_manifest_path=${MANIFEST_ABS}" \
    "env.dataset_path=${DATASET_ABS}" \
    "env.refresh_zarr_dataset=false"

log "DONE: pretrain + low-level complete. Run root: ${RUN_ROOT}"
