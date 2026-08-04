#!/usr/bin/env bash
set -euo pipefail

# Build the shared corrected-LAFAN1 DiffSR dataset cache under /data exactly
# once, in a single short job.
#
# Why this exists: the cache is shared by every arm of the capacity grid. If
# several arms run with `env.refresh_zarr_dataset=true` at the same time they
# rebuild the same directory underneath each other. On 2026-07-26 that killed
# four of seven arms within 50 seconds and truncated the cache to 56 KB.
#
# Run this first, then submit the arms with
# `CACHE_DEPENDENCY=afterok:<this job id>` so they start only after the cache
# is complete. The arms themselves always pass refresh=false.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    if [ "${REPO_ROOT}" = "/" ]; then
        echo "[ERROR] Could not locate the repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
SEED="${SEED:-0}"
PARTITION="${PARTITION:-coe-gpu}"
GPU_GRES="${GPU_GRES:-gpu:h100:1}"
# Only enough updates to force the dataset build and exit.
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50}"

case "${MODE}" in
    print|submit) ;;
    *) echo "[ERROR] MODE must be print or submit; got ${MODE}." >&2; exit 2 ;;
esac

extra=(
    --assert-kitless
    --pretrain-only
    --pretrain-output-dir "logs/groupvq_ablation/_cache_build/skill_encoder"
    --z-dim 256
    --encoder-hidden-dims 1024 512 512
    --latent-mode gumbel_multicat
    --categorical-groups 64
    --categorical-categories 128
    --gumbel-hard
    --phase-mode sin_cos
    --latent-hold-steps 10
    --pretrain-override physics=newton_mjwarp
    --pretrain-override env.refresh_zarr_dataset=true
)
printf -v extra_string '%q ' "${extra[@]}"

cmd=(env
    TASK=Isaac-Imitation-G1-Latent-v0
    SEED="${SEED}"
    FRAME_CAP=1000000
    MAX_ITERATIONS=1
    TRAIN_NUM_ENVS=16
    ROLLOUT_STEPS=12
    PRETRAIN_NUM_ENVS=16
    PRETRAIN_UPDATES="${PRETRAIN_UPDATES}"
    PRETRAIN_BATCH_SIZE=8192
    HORIZON_STEPS=10
    TRAIN_VIDEO=0
    MANIFEST_PATH="${MANIFEST_PATH}"
    DATASET_PATH="${DATASET_PATH}"
    WANDB_PROJECT=g1-lafan1-groupvq-capacity-ablation-ice
    WANDB_GROUP=cache-build
    EXP_NAME=lafan1_groupvq_cache_build
    CLUSTER_CONFIG=ice_runtime
    CLUSTER_SLURM_TIME_LIMIT=01:00:00
    CLUSTER_SLURM_PARTITION="${PARTITION}"
    CLUSTER_SLURM_QOS=coe-ice
    CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
    CLUSTER_SLURM_CPUS_PER_TASK=16
    CLUSTER_SLURM_MEM=128G
    CLUSTER_SLURM_JOB_NAME_PREFIX=lafan-groupvq-cachebuild
    # See the arm launcher: this node's GPU is dead but Slurm still schedules
    # onto it.
    CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES:-atl1-1-03-010-15-0}"
    CLUSTER_G1_USD_PATH=repo
    EXTRA_PIPELINE_ARGS="${extra_string}"
    DRY_RUN=0
    "${REPO_ROOT}/experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_hl_skill_pipeline_pace_2b.sh"
)

printf '[PLAN] cache build: '
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${MODE}" == "submit" ]]; then
    "${cmd[@]}"
else
    echo "[INFO] MODE=print: nothing was submitted."
fi
