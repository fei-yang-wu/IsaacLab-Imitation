#!/usr/bin/env bash
set -euo pipefail

# Submit a clearly labeled, unqualified DiffSR-only BONES-SEED planner run.
# This is a preliminary diagnostic and cannot satisfy the paired paper protocol.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SEED="${SEED:-0}"
GOAL_LIMIT="${GOAL_LIMIT:-10}"
MAX_PARALLEL_GOALS="${MAX_PARALLEL_GOALS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_100_diffsr_preliminary_${GOAL_LIMIT}goals_seed${SEED}_20260715}"

LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-07-15_00-31-18/models/model_step_1000046592.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/interface_baselines/bones_seed_100_phase5_1b_seed0/latent/base_pipeline/skill_encoder_h10_z256/checkpoints/latest.pt}"
# Required by the shared runner's input schema but never loaded in latent-only mode.
VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-07-15_00-32-35/models/model_step_1000046592.pt}"

MANIFEST="${MANIFEST:-/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-/data/bones_seed_phase5/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"
PREPARATION_RECORD="${PREPARATION_RECORD:-/data/bones_seed_phase5/bones_seed_100/preparation/preparation.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed0}"
# Required by the shared runner's input schema but unused in latent-only mode.
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed0}"
LATENT_SKILL_BINDING="${LATENT_SKILL_BINDING:-logs/interface_baselines/bones_seed_100_low_level_qualification_seed0/protocol_checks/latent_skill_binding.json}"

EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-fd285a287d98a8478574da211b7dbf1cf8fbfca974ecf9ba62c200e4a3b87b97}"
EXPECTED_PREPARATION_SHA256="${EXPECTED_PREPARATION_SHA256:-53dfcb3718f758edbf81b817066f4573548aa2a214ed17642162c29b6169bd37}"
EXPECTED_LANGUAGE_SHA256="${EXPECTED_LANGUAGE_SHA256:-3a50746d575d3c8d36c2c4e460acf4834a22a74e663a27d9f04ac8a6137c7975}"
EXPECTED_LATENT_SHA256="${EXPECTED_LATENT_SHA256:-904229f5737256843e8046ac77b96a39d8e5dd441274e91813b7efc63d00b202}"
EXPECTED_SKILL_SHA256="${EXPECTED_SKILL_SHA256:-64ec4eebfdff0bf1150c47ee56997c0643a6549655ba73c52820cfde19362a74}"

DEMO_ROWS_PER_GOAL="${DEMO_ROWS_PER_GOAL:-1000}"
ROLLOUT_ROWS_PER_GOAL="${ROLLOUT_ROWS_PER_GOAL:-1000}"
ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-10}"
EVAL_STEPS="${EVAL_STEPS:-500}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/data}"

if [[ ! "${GOAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || ((GOAL_LIMIT > 100)); then
    echo "[ERROR] GOAL_LIMIT must be between 1 and 100." >&2
    exit 2
fi
if [[ ! "${MAX_PARALLEL_GOALS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] MAX_PARALLEL_GOALS must be positive." >&2
    exit 2
fi
if [[ "${ROLLOUT_NUM_ENVS}" != "10" ]]; then
    echo "[ERROR] Preliminary rollout collection keeps the paper value of 10 environments." >&2
    exit 2
fi
if [[ "${EVAL_STEPS}" != "500" ]]; then
    echo "[ERROR] Preliminary M3 evaluation is fixed to the normal 500-step episode." >&2
    exit 2
fi

remote_path_for_container_path() {
    local path="$1"
    case "${path}" in
        logs/*)
            printf '%s/logs/%s' "${REMOTE_PROJECT_ROOT}" "${path#logs/}"
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
    remote_latent="$(remote_path_for_container_path "${LATENT_LOW_LEVEL_CHECKPOINT}")"
    remote_skill="$(remote_path_for_container_path "${LATENT_SKILL_CHECKPOINT}")"
    remote_vanilla="$(remote_path_for_container_path "${VANILLA_TRACKER_CHECKPOINT}")"
    remote_binding="$(remote_path_for_container_path "${LATENT_SKILL_BINDING}")"
    remote_latent_dataset="$(remote_path_for_container_path "${LATENT_DATASET_PATH}")"
    remote_vanilla_dataset="$(remote_path_for_container_path "${VANILLA_DATASET_PATH}")"

    if ssh -o BatchMode=yes -o ConnectTimeout=10 skynet "test -e '${REMOTE_OUTPUT_ROOT}'"; then
        echo "[ERROR] Refusing to reuse preliminary output root: ${REMOTE_OUTPUT_ROOT}" >&2
        exit 2
    fi
    ssh -o BatchMode=yes -o ConnectTimeout=10 skynet python3 - \
        "${remote_manifest}" "${EXPECTED_MANIFEST_SHA256}" \
        "${remote_language}" "${EXPECTED_LANGUAGE_SHA256}" \
        "${remote_preparation}" "${EXPECTED_PREPARATION_SHA256}" \
        "${remote_latent}" "${EXPECTED_LATENT_SHA256}" \
        "${remote_skill}" "${EXPECTED_SKILL_SHA256}" \
        "${remote_vanilla}" "${remote_binding}" \
        "${remote_latent_dataset}" "${remote_vanilla_dataset}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

items = sys.argv[1:]
for index in range(0, 10, 2):
    path = Path(items[index])
    expected = items[index + 1]
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"hash mismatch for {path}: {actual} != {expected}")
vanilla = Path(items[10])
binding_path = Path(items[11])
latent_dataset = Path(items[12])
vanilla_dataset = Path(items[13])
for path in (vanilla, binding_path):
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
for path in (latent_dataset, vanilla_dataset):
    if not path.is_dir():
        raise SystemExit(f"missing required dataset: {path}")
binding = json.loads(binding_path.read_text())
if binding.get("passed") is not True:
    raise SystemExit("latent skill binding did not pass")
if binding.get("low_level_checkpoint_sha256") != items[7]:
    raise SystemExit("binding low-level checkpoint hash mismatch")
if binding.get("skill_checkpoint_sha256") != items[9]:
    raise SystemExit("binding skill checkpoint hash mismatch")
print("[PASS] Preliminary DiffSR inputs and skill binding verified.")
PY
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
    CLUSTER_SLURM_PREPARE_TIME_LIMIT=12:00:00
    CLUSTER_SLURM_ROLLOUT_TIME_LIMIT=12:00:00
    CLUSTER_SLURM_FINETUNE_TIME_LIMIT=12:00:00
    CLUSTER_SLURM_FINAL_EVAL_TIME_LIMIT=4:00:00
    CLUSTER_SLURM_SUMMARIZE_TIME_LIMIT=1:00:00
    AUTO_SYNC_LOCAL_CHECKPOINTS=0
    AUTO_SYNC_EXTRA_AGGREGATE_ROOTS=0
    "DRY_RUN=${DRY_RUN}"
    MODE=bones-seed-multigoal-language
    INTERFACES=latent_skill
    ALLOW_UNQUALIFIED_PRELIMINARY=1
    "VANILLA_TRACKER_CHECKPOINT=${VANILLA_TRACKER_CHECKPOINT}"
    "LATENT_LOW_LEVEL_CHECKPOINT=${LATENT_LOW_LEVEL_CHECKPOINT}"
    "LATENT_SKILL_CHECKPOINT=${LATENT_SKILL_CHECKPOINT}"
    "MANIFEST=${MANIFEST}"
    "LANGUAGE_EMBEDDINGS=${LANGUAGE_EMBEDDINGS}"
    "LATENT_DATASET_PATH=${LATENT_DATASET_PATH}"
    "VANILLA_DATASET_PATH=${VANILLA_DATASET_PATH}"
    "PREPARATION_RECORD=${PREPARATION_RECORD}"
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
