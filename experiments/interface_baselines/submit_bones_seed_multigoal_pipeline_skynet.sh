#!/usr/bin/env bash
set -euo pipefail

# Submit the paper-facing BONES-SEED planner comparison as a guarded Slurm
# dependency chain. This launcher is intentionally blocked until the strict
# low-level qualification artifacts exist and pass.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${VANILLA_TRACKER_CHECKPOINT:?Set the qualified vanilla checkpoint path}"
: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set the qualified latent checkpoint path}"
: "${LATENT_SKILL_CHECKPOINT:?Set the qualified skill checkpoint path}"

DRY_RUN="${DRY_RUN:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
ALLOW_EXISTING_OUTPUT_ROOT="${ALLOW_EXISTING_OUTPUT_ROOT:-0}"
SEED="${SEED:-0}"
GOAL_LIMIT="${GOAL_LIMIT:-100}"
MAX_PARALLEL_GOALS="${MAX_PARALLEL_GOALS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_100_multigoal_language_seed${SEED}}"

MANIFEST="${MANIFEST:-/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-/data/bones_seed_phase5/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"
PREPARATION_RECORD="${PREPARATION_RECORD:-/data/bones_seed_phase5/bones_seed_100/preparation/preparation.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed0}"
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed0}"
QUALIFICATION_ROOT="${QUALIFICATION_ROOT:-logs/interface_baselines/bones_seed_100_low_level_qualification_seed0}"
VANILLA_QUALIFICATION_AUDIT="${VANILLA_QUALIFICATION_AUDIT:-${QUALIFICATION_ROOT}/vanilla_qualification_audit.json}"
LATENT_QUALIFICATION_AUDIT="${LATENT_QUALIFICATION_AUDIT:-${QUALIFICATION_ROOT}/latent_qualification_audit.json}"
STREAMED_EQUIVALENCE_CERTIFICATE="${STREAMED_EQUIVALENCE_CERTIFICATE:-${QUALIFICATION_ROOT}/streamed_vanilla_equivalence.json}"

EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-fd285a287d98a8478574da211b7dbf1cf8fbfca974ecf9ba62c200e4a3b87b97}"
EXPECTED_PREPARATION_SHA256="${EXPECTED_PREPARATION_SHA256:-53dfcb3718f758edbf81b817066f4573548aa2a214ed17642162c29b6169bd37}"
EXPECTED_LANGUAGE_SHA256="${EXPECTED_LANGUAGE_SHA256:-3a50746d575d3c8d36c2c4e460acf4834a22a74e663a27d9f04ac8a6137c7975}"

DEMO_ROWS_PER_GOAL="${DEMO_ROWS_PER_GOAL:-1000}"
ROLLOUT_ROWS_PER_GOAL="${ROLLOUT_ROWS_PER_GOAL:-1000}"
ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-10}"
EVAL_STEPS="${EVAL_STEPS:-500}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/data}"

if [[ "${GOAL_LIMIT}" != "100" ]]; then
    echo "[ERROR] The paper-facing BONES-SEED pipeline is fixed to all 100 goals." >&2
    exit 2
fi
if [[ ! "${MAX_PARALLEL_GOALS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_PARALLEL_GOALS must be a positive integer." >&2
    exit 2
fi
if [[ "${ROLLOUT_NUM_ENVS}" != "10" ]]; then
    echo "[ERROR] The paper-facing planner-rollout collector is fixed to 10 same-goal environments." >&2
    exit 2
fi
if [[ "${EVAL_STEPS}" != "500" ]]; then
    echo "[ERROR] The paper-facing M3 evaluation is fixed to the normal 500-step episode." >&2
    exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SEED must be a non-negative integer." >&2
    exit 2
fi
case "${OUTPUT_ROOT}" in
    logs/*|/workspace/isaaclab/project/logs/*) ;;
    *)
        echo "[ERROR] Paper OUTPUT_ROOT must be under the persistent project logs directory: ${OUTPUT_ROOT}" >&2
        exit 2
        ;;
esac
if [[ "${PREFLIGHT_ONLY}" != "0" && "${PREFLIGHT_ONLY}" != "1" ]]; then
    echo "[ERROR] PREFLIGHT_ONLY must be 0 or 1." >&2
    exit 2
fi
if [[ "${ALLOW_EXISTING_OUTPUT_ROOT}" != "0" && "${ALLOW_EXISTING_OUTPUT_ROOT}" != "1" ]]; then
    echo "[ERROR] ALLOW_EXISTING_OUTPUT_ROOT must be 0 or 1." >&2
    exit 2
fi
if [[ "${PREFLIGHT_ONLY}" == "1" && ("${DRY_RUN}" == "1" || "${DRY_RUN}" == "true") ]]; then
    echo "[ERROR] PREFLIGHT_ONLY=1 requires DRY_RUN=0 so remote gates are checked." >&2
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

REMOTE_OUTPUT_ROOT="$(remote_path_for_container_path "${OUTPUT_ROOT}")"

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    remote_manifest="$(remote_path_for_container_path "${MANIFEST}")"
    remote_language="$(remote_path_for_container_path "${LANGUAGE_EMBEDDINGS}")"
    remote_preparation="$(remote_path_for_container_path "${PREPARATION_RECORD}")"
    remote_vanilla_checkpoint="$(remote_path_for_container_path "${VANILLA_TRACKER_CHECKPOINT}")"
    remote_latent_checkpoint="$(remote_path_for_container_path "${LATENT_LOW_LEVEL_CHECKPOINT}")"
    remote_skill_checkpoint="$(remote_path_for_container_path "${LATENT_SKILL_CHECKPOINT}")"
    remote_vanilla_audit="$(remote_path_for_container_path "${VANILLA_QUALIFICATION_AUDIT}")"
    remote_latent_audit="$(remote_path_for_container_path "${LATENT_QUALIFICATION_AUDIT}")"
    remote_equivalence="$(remote_path_for_container_path "${STREAMED_EQUIVALENCE_CERTIFICATE}")"
    remote_latent_dataset="$(remote_path_for_container_path "${LATENT_DATASET_PATH}")"
    remote_vanilla_dataset="$(remote_path_for_container_path "${VANILLA_DATASET_PATH}")"

    if [[ "${ALLOW_EXISTING_OUTPUT_ROOT}" != "1" ]] \
        && ssh -o BatchMode=yes -o ConnectTimeout=10 skynet "test -e '${REMOTE_OUTPUT_ROOT}'"; then
        echo "[ERROR] Refusing to reuse existing paper output root: ${REMOTE_OUTPUT_ROOT}" >&2
        echo "[HINT] Choose a new OUTPUT_ROOT. Set ALLOW_EXISTING_OUTPUT_ROOT=1 only for an audited resume." >&2
        exit 2
    fi

    for dataset_path in "${remote_latent_dataset}" "${remote_vanilla_dataset}"; do
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 skynet "test -d '${dataset_path}'"; then
            echo "[ERROR] Required BONES dataset cache is missing: ${dataset_path}" >&2
            exit 2
        fi
    done

    ssh -o BatchMode=yes -o ConnectTimeout=10 skynet python3 - \
        "${remote_manifest}" \
        "${remote_language}" \
        "${remote_preparation}" \
        "${remote_vanilla_checkpoint}" \
        "${remote_latent_checkpoint}" \
        "${remote_skill_checkpoint}" \
        "${remote_vanilla_audit}" \
        "${remote_latent_audit}" \
        "${remote_equivalence}" \
        "${LATENT_DATASET_PATH}" \
        "${VANILLA_DATASET_PATH}" \
        "${EXPECTED_MANIFEST_SHA256}" \
        "${EXPECTED_LANGUAGE_SHA256}" \
        "${EXPECTED_PREPARATION_SHA256}" \
        < "${SCRIPT_DIR}/validate_bones_seed_planner_submission.py"
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PASS] BONES-SEED seed ${SEED} submission preflight passed."
    exit 0
fi

pipeline_array="0-$((GOAL_LIMIT - 1))%${MAX_PARALLEL_GOALS}"
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
    CLUSTER_SLURM_QOS=long
    CLUSTER_SLURM_SUBMIT_SCRIPT=bones_pipeline
    "CLUSTER_SLURM_SUBMISSION_RECORD_ROOT=${REMOTE_OUTPUT_ROOT}"
    "CLUSTER_SLURM_PIPELINE_ARRAY=${pipeline_array}"
    CLUSTER_SLURM_PREPARE_TIME_LIMIT=1-00:00:00
    CLUSTER_SLURM_ROLLOUT_TIME_LIMIT=12:00:00
    CLUSTER_SLURM_FINETUNE_TIME_LIMIT=12:00:00
    CLUSTER_SLURM_FINAL_EVAL_TIME_LIMIT=4:00:00
    CLUSTER_SLURM_SUMMARIZE_TIME_LIMIT=1:00:00
    AUTO_SYNC_LOCAL_CHECKPOINTS=0
    AUTO_SYNC_EXTRA_AGGREGATE_ROOTS=0
    "DRY_RUN=${DRY_RUN}"
    MODE=bones-seed-multigoal-language
    "VANILLA_TRACKER_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}"
    "LATENT_LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}"
    "LATENT_SKILL_CHECKPOINT=${LATENT_SKILL_CHECKPOINT}"
    "MANIFEST=${MANIFEST}"
    "LANGUAGE_EMBEDDINGS=${LANGUAGE_EMBEDDINGS}"
    "LATENT_DATASET_PATH=${LATENT_DATASET_PATH}"
    "VANILLA_DATASET_PATH=${VANILLA_DATASET_PATH}"
    "PREPARATION_RECORD=${PREPARATION_RECORD}"
    "VANILLA_QUALIFICATION_AUDIT=${VANILLA_QUALIFICATION_AUDIT}"
    "LATENT_QUALIFICATION_AUDIT=${LATENT_QUALIFICATION_AUDIT}"
    "STREAMED_EQUIVALENCE_CERTIFICATE=${STREAMED_EQUIVALENCE_CERTIFICATE}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "GOAL_LIMIT=${GOAL_LIMIT}"
    "DEMO_ROWS_PER_GOAL=${DEMO_ROWS_PER_GOAL}"
    "ROLLOUT_ROWS_PER_GOAL=${ROLLOUT_ROWS_PER_GOAL}"
    "ROLLOUT_NUM_ENVS=${ROLLOUT_NUM_ENVS}"
    "EVAL_STEPS=${EVAL_STEPS}"
    "MODEL_SIZE=${MODEL_SIZE}"
    "PRETRAIN_UPDATES=${PRETRAIN_UPDATES}"
    "FINETUNE_UPDATES=${FINETUNE_UPDATES}"
    "SEED=${SEED}"
    REFRESH_DATASETS=0
    SKIP_PRETRAINED_CLOSED_LOOP=0
    CONTINUE_ON_ERROR=0
    experiments/interface_baselines/submit_cluster_interface_baselines.sh
)

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
