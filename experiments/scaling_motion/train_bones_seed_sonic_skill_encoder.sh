#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"

# DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/bones_seed_100}"
# MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/manifests/g1_bones_seed_100_sonic_filtered_manifest.json}"
# DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/g1_hl_diffsr_sonic_filtered}"
# LATENT_MODE="deterministic"
# RUN_NAME=${RUN_NAME:-bs91-deter}

# # bs5000 deter
# DATA_ROOT="data/bones_seed_sonic_129k_50hz"
# MANIFEST_PATH="${DATA_ROOT}/manifests/bones-seed-sonic-5000.json"
# DATASET_PATH="${DATA_ROOT}/g1_hl_diffsr_5000"
# LATENT_MODE="deterministic"
# RUN_NAME="bs5000-deter-f512-z512-alldim512x2"

# bs5000 multicat
DATA_ROOT="data/bones_seed_sonic_129k_50hz"
MANIFEST_PATH="${DATA_ROOT}/manifests/bones-seed-sonic-5000.json"
DATASET_PATH="${DATA_ROOT}/g1_hl_diffsr_5000"
LATENT_MODE="gumbel_multicat"
RUN_NAME="bs5000-multicat-fdim512-zdim512-g128-c1024"

OUTPUT_DIR="${OUTPUT_DIR:-${LOG_ROOT}/skill_encoder/${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-skill_encoder}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"

# --task Isaac-Imitation-G1-Latent-v0
# --z_dim 256
# --latent_mode deterministic
# --diffsr_embed_dim 512
# --batch_size "${PRETRAIN_BATCH_SIZE:-8192}"
# --log_interval 100
# --eval_batches 4

exec pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py \
    --headless \
    --assert-kitless \
    --num_envs "${PRETRAIN_NUM_ENVS:-16}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --latent_mode ${LATENT_MODE} \
    --horizon_steps 10 \
    --encoder_window_mode intermediate \
    --diffsr_feature_dim 512 \
    --num_updates "${PRETRAIN_UPDATES:-50000}" \
    --reconstruction_eval \
    --window_probe_eval \
    --window_probe_train_batches 8 \
    --window_probe_eval_batches 4 \
    --logger_backend wandb \
    --wandb_project g1-bones-seed-scaling \
    --wandb_entity gaochenxiao \
    --wandb_group "${WANDB_GROUP}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    "env.lafan1_manifest_path=${MANIFEST_PATH}" \
    "env.dataset_path=${DATASET_PATH}" \
    physics=newton_mjwarp \
    env.refresh_zarr_dataset=false \
    --z_dim 512 \
    --categorical_groups 128 \
    --categorical_categories 1024
