#!/usr/bin/env bash
set -euo pipefail

# Submit the fixed three-seed, three-budget, per-motion no-language study as
# one Slurm array from one verified workspace archive.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${VANILLA_TRACKER_CHECKPOINT:?Set the qualified vanilla checkpoint path}"
: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set the qualified latent checkpoint path}"
: "${LATENT_SKILL_CHECKPOINT:?Set the qualified skill checkpoint path}"
: "${VANILLA_QUALIFICATION_AUDIT:?Set the vanilla qualification audit path}"
: "${LATENT_QUALIFICATION_AUDIT:?Set the latent qualification audit path}"
: "${STREAMED_EQUIVALENCE_CERTIFICATE:?Set the equivalence certificate path}"

DRY_RUN="${DRY_RUN:-1}"
MANIFEST="${MANIFEST:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-${DATASET_PATH}}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-218d5d41b5e6a47e272c07babb84b8c51c9af54e5576ecb8322fb66528d366d8}"
EXPECTED_MOTION_COUNT="${EXPECTED_MOTION_COUNT:-40}"
MIN_ORACLE_SUCCESS="${MIN_ORACLE_SUCCESS:-0.8}"
SEEDS="${SEEDS:-0 1 2}"
SAMPLE_BUDGETS="${SAMPLE_BUDGETS:-1000 10000 50000}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/phase4_no_language_lafan1}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/data}"

if [[ ! "${EXPECTED_MOTION_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] EXPECTED_MOTION_COUNT must be positive." >&2
    exit 2
fi
if [[ ! "${MAX_PARALLEL_TASKS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_PARALLEL_TASKS must be positive." >&2
    exit 2
fi
case "${OUTPUT_ROOT}" in
    logs/*|/workspace/isaaclab/project/logs/*) ;;
    *)
        echo "[ERROR] Paper OUTPUT_ROOT must be under the persistent project logs directory: ${OUTPUT_ROOT}" >&2
        exit 2
        ;;
esac
read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a BUDGET_LIST <<< "${SAMPLE_BUDGETS}"
if [[ "${SEED_LIST[*]}" != "0 1 2" ]]; then
    echo "[ERROR] Paper protocol fixes SEEDS='0 1 2'." >&2
    exit 2
fi
if [[ "${BUDGET_LIST[*]}" != "1000 10000 50000" ]]; then
    echo "[ERROR] Paper protocol fixes SAMPLE_BUDGETS='1000 10000 50000'." >&2
    exit 2
fi
for value in "${SEED_LIST[@]}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "[ERROR] Invalid seed: ${value}" >&2; exit 2; }
done
for value in "${BUDGET_LIST[@]}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] Invalid sample budget: ${value}" >&2; exit 2; }
done

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

REMOTE_OUTPUT_ROOT="$(remote_path_for_container_path "${OUTPUT_ROOT}")"

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    remote_manifest="$(remote_path_for_container_path "${MANIFEST}")"
    remote_vanilla_checkpoint="$(remote_path_for_container_path "${VANILLA_TRACKER_CHECKPOINT}")"
    remote_latent_checkpoint="$(remote_path_for_container_path "${LATENT_LOW_LEVEL_CHECKPOINT}")"
    remote_skill_checkpoint="$(remote_path_for_container_path "${LATENT_SKILL_CHECKPOINT}")"
    remote_vanilla_audit="$(remote_path_for_container_path "${VANILLA_QUALIFICATION_AUDIT}")"
    remote_latent_audit="$(remote_path_for_container_path "${LATENT_QUALIFICATION_AUDIT}")"
    remote_equivalence="$(remote_path_for_container_path "${STREAMED_EQUIVALENCE_CERTIFICATE}")"
    remote_latent_dataset="$(remote_path_for_container_path "${DATASET_PATH}")"
    remote_vanilla_dataset="$(remote_path_for_container_path "${VANILLA_DATASET_PATH}")"
    if ssh -o BatchMode=yes -o ConnectTimeout=10 skynet "test -e '${REMOTE_OUTPUT_ROOT}'"; then
        echo "[ERROR] Refusing to reuse existing Phase-4 output root: ${REMOTE_OUTPUT_ROOT}" >&2
        exit 2
    fi
    for dataset_path in "${remote_latent_dataset}" "${remote_vanilla_dataset}"; do
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 skynet "test -d '${dataset_path}'"; then
            echo "[ERROR] Required Phase-4 dataset cache is missing: ${dataset_path}" >&2
            exit 2
        fi
    done
    ssh -o BatchMode=yes -o ConnectTimeout=10 skynet python3 - \
        "${remote_manifest}" \
        "${remote_vanilla_checkpoint}" \
        "${remote_latent_checkpoint}" \
        "${remote_skill_checkpoint}" \
        "${remote_vanilla_audit}" \
        "${remote_latent_audit}" \
        "${remote_equivalence}" \
        "${DATASET_PATH}" \
        "${VANILLA_DATASET_PATH}" \
        "${EXPECTED_MANIFEST_SHA256}" \
        --expected_motion_count "${EXPECTED_MOTION_COUNT}" \
        --minimum_oracle_success "${MIN_ORACLE_SUCCESS}" \
        < "${SCRIPT_DIR}/validate_phase4_no_language_submission.py"
fi

task_count="$((EXPECTED_MOTION_COUNT * ${#SEED_LIST[@]}))"
array_spec="0-$((task_count - 1))%${MAX_PARALLEL_TASKS}"
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
    CLUSTER_SLURM_TIME_LIMIT=2-00:00:00
    CLUSTER_SLURM_QOS=long
    CLUSTER_SLURM_JOB_NAME=phase4-no-language
    "CLUSTER_SLURM_ARRAY=${array_spec}"
    CLUSTER_SLURM_SUBMIT_SCRIPT=phase4
    "CLUSTER_SLURM_SUBMISSION_RECORD_ROOT=${REMOTE_OUTPUT_ROOT}"
    AUTO_SYNC_LOCAL_CHECKPOINTS=0
    AUTO_SYNC_EXTRA_AGGREGATE_ROOTS=0
    "DRY_RUN=${DRY_RUN}"
    MODE=phase4-no-language
    "VANILLA_TRACKER_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}"
    "LATENT_LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}"
    "LATENT_SKILL_CHECKPOINT=${LATENT_SKILL_CHECKPOINT}"
    "VANILLA_QUALIFICATION_AUDIT=${VANILLA_QUALIFICATION_AUDIT}"
    "LATENT_QUALIFICATION_AUDIT=${LATENT_QUALIFICATION_AUDIT}"
    "STREAMED_EQUIVALENCE_CERTIFICATE=${STREAMED_EQUIVALENCE_CERTIFICATE}"
    "MANIFEST=${MANIFEST}"
    "DATASET_PATH=${DATASET_PATH}"
    "VANILLA_DATASET_PATH=${VANILLA_DATASET_PATH}"
    "EXPECTED_MANIFEST_SHA256=${EXPECTED_MANIFEST_SHA256}"
    "EXPECTED_MOTION_COUNT=${EXPECTED_MOTION_COUNT}"
    "MIN_ORACLE_SUCCESS=${MIN_ORACLE_SUCCESS}"
    "SEEDS=${SEEDS}"
    "SAMPLE_BUDGETS=${SAMPLE_BUDGETS}"
    "NUM_ENVS=${NUM_ENVS}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "MODEL_SIZE=${MODEL_SIZE}"
    "PRETRAIN_UPDATES=${PRETRAIN_UPDATES}"
    "FINETUNE_UPDATES=${FINETUNE_UPDATES}"
    experiments/interface_baselines/submit_cluster_interface_baselines.sh
)

echo "[INFO] Phase-4 array covers ${task_count} seed/motion tasks: ${array_spec}"
printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
