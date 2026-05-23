#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MANIFEST="${MANIFEST:-./data/unitree/manifests/g1_unitree_dance102_rlopt_ipmd_500m_actions_manifest.json}"
VARIANTS="${VARIANTS:-scratch pretrained_finetune pretrained_labeled_bc_finetune labeled_bc_finetune}"
GROUP_NAME="${GROUP_NAME:-g1_dance102_action_labels_ipmd500m_bilinear_vqvae_latent_4096_1b}"
RUN_PREFIX="${RUN_PREFIX:-g1_dance102_action_labels_ipmd500m_bilinear_vqvae_latent_4096_1b}"

if [[ ! -f "$REPO_ROOT/${MANIFEST#./}" && ! -f "$MANIFEST" ]]; then
    echo "[ERROR] Dance102 action-label manifest not found: $MANIFEST" >&2
    echo "[HINT] Expected the manifest produced for G1_Take_102.bvh_60hz.rlopt_ipmd_500m_actions.npz." >&2
    exit 1
fi

TASK="${TASK:-Isaac-Imitation-G1-Latent-VQVAE-v0}" \
NUM_ENVS="${NUM_ENVS:-4096}" \
MAX_ITERATIONS="${MAX_ITERATIONS:-10173}" \
SEEDS="${SEEDS:-42}" \
VARIANTS="$VARIANTS" \
MANIFEST="$MANIFEST" \
REFRESH_ZARR_DATASET="${REFRESH_ZARR_DATASET:-false}" \
PROJECT_NAME="${PROJECT_NAME:-G1-Imitation-RLOpt-Pretrain}" \
GROUP_NAME="$GROUP_NAME" \
RUN_PREFIX="$RUN_PREFIX" \
OFFLINE_NUM_UPDATES="${OFFLINE_NUM_UPDATES:-2000}" \
OFFLINE_BATCH_SIZE="${OFFLINE_BATCH_SIZE:-8192}" \
OFFLINE_LOG_INTERVAL="${OFFLINE_LOG_INTERVAL:-100}" \
OFFLINE_POLICY_BC_UPDATES="${OFFLINE_POLICY_BC_UPDATES:-2000}" \
OFFLINE_POLICY_BC_BATCH_SIZE="${OFFLINE_POLICY_BC_BATCH_SIZE:-8192}" \
OFFLINE_POLICY_BC_TRAIN_LATENT="${OFFLINE_POLICY_BC_TRAIN_LATENT:-true}" \
ONLINE_SR_UPDATE_STEPS="${ONLINE_SR_UPDATE_STEPS:-8}" \
SR_BATCH_SIZE="${SR_BATCH_SIZE:-4096}" \
SAMPLE_EVAL_INTERVAL="${SAMPLE_EVAL_INTERVAL:-50}" \
DRY_RUN="${DRY_RUN:-0}" \
VIDEO="${VIDEO:-1}" \
VIDEO_LENGTH="${VIDEO_LENGTH:-200}" \
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}" \
CLUSTER_PROFILE="${CLUSTER_PROFILE:-}" \
"$SCRIPT_DIR/submit_cluster_ablation.sh"
