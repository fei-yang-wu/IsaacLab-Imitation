#!/usr/bin/env bash
set -euo pipefail

# Submit the strict 100-motion BONES-SEED oracle and equivalence gate after the
# two final low-level training jobs finish.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${VANILLA_TRACKER_CHECKPOINT:?Set the container-visible vanilla checkpoint path}"
: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set the container-visible latent checkpoint path}"
: "${LATENT_SKILL_CHECKPOINT:?Set the container-visible skill checkpoint path}"

DRY_RUN="${DRY_RUN:-1}"
MANIFEST="${MANIFEST:-/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json}"
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed0}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_100_low_level_qualification_seed0}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/data}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-fd285a287d98a8478574da211b7dbf1cf8fbfca974ecf9ba62c200e4a3b87b97}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DEFER_CHECKPOINT_VALIDATION="${DEFER_CHECKPOINT_VALIDATION:-0}"

if [[ "${DEFER_CHECKPOINT_VALIDATION}" == "1" && -z "${SLURM_DEPENDENCY}" ]]; then
    echo "[ERROR] Deferred checkpoint validation requires SLURM_DEPENDENCY=afterok:<jobs>." >&2
    exit 2
fi

remote_path_for_container_path() {
    local path="$1"
    case "${path}" in
        logs/*)
            printf '%s/logs/%s' "${REMOTE_PROJECT_ROOT}" "${path#logs/}"
            ;;
        /workspace/isaaclab/project/logs/*)
            printf '%s/logs/%s' "${REMOTE_PROJECT_ROOT}" "${path#/workspace/isaaclab/project/logs/}"
            ;;
        /data/*)
            printf '%s/%s' "${REMOTE_DATA_ROOT}" "${path#/data/}"
            ;;
        /*)
            printf '%s' "${path}"
            ;;
        *)
            echo "[ERROR] Use a logs/... or absolute container path: ${path}" >&2
            return 2
            ;;
    esac
}

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    remote_manifest="$(remote_path_for_container_path "${MANIFEST}")"
    actual_manifest_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 skynet "sha256sum '${remote_manifest}'" | awk '{print $1}')"
    if [[ "${actual_manifest_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
        echo "[ERROR] Fresh BONES manifest hash mismatch." >&2
        exit 2
    fi
    if [[ "${DEFER_CHECKPOINT_VALIDATION}" != "1" ]]; then
        for path in \
            "${VANILLA_TRACKER_CHECKPOINT}" \
            "${LATENT_LOW_LEVEL_CHECKPOINT}" \
            "${LATENT_SKILL_CHECKPOINT}"; do
            remote_path="$(remote_path_for_container_path "${path}")"
            if ! ssh -o BatchMode=yes -o ConnectTimeout=10 skynet "test -f '${remote_path}'"; then
                echo "[ERROR] Remote checkpoint does not exist: ${remote_path}" >&2
                exit 2
            fi
        done
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
    AUTO_SYNC_LOCAL_CHECKPOINTS=0
    "DRY_RUN=${DRY_RUN}"
    MODE=bones-seed-low-level-qualification
    "VANILLA_TRACKER_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}"
    "LATENT_LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}"
    "LATENT_SKILL_CHECKPOINT=${LATENT_SKILL_CHECKPOINT}"
    "MANIFEST=${MANIFEST}"
    "VANILLA_DATASET_PATH=${VANILLA_DATASET_PATH}"
    "LATENT_DATASET_PATH=${LATENT_DATASET_PATH}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    NUM_ENVS=100
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
