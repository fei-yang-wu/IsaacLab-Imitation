#!/usr/bin/env bash
set -euo pipefail

# Submit the strict corrected-LAFAN1 qualification after both final 5B
# low-level jobs finish.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${VANILLA_TRACKER_CHECKPOINT:?Set the final container-visible vanilla checkpoint path}"
: "${LATENT_SKILL_CHECKPOINT:?Set the final container-visible skill checkpoint path}"
: "${LATENT_LOW_LEVEL_RUN_ID:?Set the unique latent training RUN_ID}"

DRY_RUN="${DRY_RUN:-1}"
MANIFEST="${MANIFEST:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_corrected_8e95d557_low_level_qualification_seed0}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/data}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-218d5d41b5e6a47e272c07babb84b8c51c9af54e5576ecb8322fb66528d366d8}"
EXPECTED_LATENT_CHECKPOINT_BASENAME="${EXPECTED_LATENT_CHECKPOINT_BASENAME:-model_step_5000232960.pt}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"

if [[ -z "${SLURM_DEPENDENCY}" && "${DRY_RUN}" != "1" ]]; then
    echo "[ERROR] Qualification must depend on both final low-level jobs." >&2
    exit 2
fi

remote_manifest="${REMOTE_DATA_ROOT}/${MANIFEST#/data/}"
if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    actual_manifest_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 skynet \
        "sha256sum '${remote_manifest}'" | awk '{print $1}')"
    if [[ "${actual_manifest_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
        echo "[ERROR] Corrected-LAFAN1 manifest hash mismatch." >&2
        exit 2
    fi
fi

cmd=(
    env
    CLUSTER_AUTO_SETUP_G1_DATA=0
    CLUSTER_ARCHIVE_SYNC=1
    CLUSTER_GIT_SYNC_FIRST=0
    CLUSTER_INCREMENTAL_SYNC=0
    CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=0
    "CLUSTER_EXTRA_RSYNC_EXCLUDES=data/ .tmp/ RLOpt/ ImitationLearningTools/"
    CLUSTER_SKIP_CACHE_COPY=1
    CLUSTER_USE_SHARED_SIF=1
    CLUSTER_OVERLAY_SIZE_MB=8192
    CLUSTER_SLURM_TIME_LIMIT=0-06:00:00
    CLUSTER_SLURM_QOS=long
    CLUSTER_SLURM_JOB_NAME=lafan1-low-level-gate
    AUTO_SYNC_LOCAL_CHECKPOINTS=0
    "DRY_RUN=${DRY_RUN}"
    MODE=lafan1-low-level-qualification
    "VANILLA_TRACKER_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}"
    "LATENT_SKILL_CHECKPOINT=${LATENT_SKILL_CHECKPOINT}"
    "LATENT_LOW_LEVEL_RUN_ID=${LATENT_LOW_LEVEL_RUN_ID}"
    "EXPECTED_LATENT_CHECKPOINT_BASENAME=${EXPECTED_LATENT_CHECKPOINT_BASENAME}"
    "MANIFEST=${MANIFEST}"
    "DATASET_PATH=${DATASET_PATH}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    NUM_ENVS=40
    EVAL_STEPS=1000
    EQUIVALENCE_NUM_ENVS=2
    EQUIVALENCE_STEPS=20
    SEED=0
    MIN_ORACLE_SUCCESS=0.8
    experiments/interface_baselines/submit_cluster_interface_baselines.sh
)

if [[ -n "${SLURM_DEPENDENCY}" ]]; then
    cmd=(env "CLUSTER_SLURM_DEPENDENCY=${SLURM_DEPENDENCY}" "${cmd[@]:1}")
fi

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
