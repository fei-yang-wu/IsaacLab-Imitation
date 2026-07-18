#!/usr/bin/env bash
set -euo pipefail

# Submit only the final corrected-LAFAN1 DiffSR low-level controller. This
# deliberately does not submit EE/full-body low-level variants or any
# paper-facing high-level planner comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50865}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250000000}"
SKILL_UPDATES="${SKILL_UPDATES:-5000}"
PLANNER_UPDATES="${PLANNER_UPDATES:-5000}"
# The measured latent-controller throughput puts 5B frames above two days.
# Four days leaves room for the encoder/planner setup and final log sync.
WALLTIME="${WALLTIME:-4-00:00:00}"
QOS="${QOS:-long}"
VERIFY_REMOTE_DATA="${VERIFY_REMOTE_DATA:-1}"

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/data/lafan1_corrected_8e95d557}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-218d5d41b5e6a47e272c07babb84b8c51c9af54e5576ecb8322fb66528d366d8}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
RUN_TAG="${RUN_TAG:-lafan1_corrected_8e95d557_diffsr_5b_seed${SEED}}"
RUN_ROOT="${RUN_ROOT:-logs/interface_baselines/${RUN_TAG}/latent}"

if [[ "${SEED}" != "0" ]]; then
    echo "[ERROR] The final corrected-LAFAN1 low-level qualification uses seed 0." >&2
    exit 2
fi
if [[ "${NUM_ENVS}" != "4096" || "${MAX_ITERATIONS}" != "50865" ]]; then
    echo "[ERROR] The matched final run is fixed at 4096 envs x 50865 iterations (~5B frames)." >&2
    echo "[HINT] Change both only for an explicitly recorded protocol revision." >&2
    exit 2
fi

if [[ "${VERIFY_REMOTE_DATA}" == "1" ]]; then
    remote_manifest="${REMOTE_DATA_ROOT}/manifests/g1_lafan1_manifest.json"
    remote_dataset="${REMOTE_DATA_ROOT}/g1_hl_diffsr"
    actual_manifest_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 skynet \
        "sha256sum '${remote_manifest}'" | awk '{print $1}')"
    if [[ "${actual_manifest_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
        echo "[ERROR] Corrected-LAFAN1 manifest hash does not match the frozen protocol." >&2
        echo "[INFO] expected: ${EXPECTED_MANIFEST_SHA256}" >&2
        echo "[INFO] actual:   ${actual_manifest_sha}" >&2
        exit 2
    fi
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 skynet \
        "test -f '${remote_dataset}/zarr.json' -o -f '${remote_dataset}/lafan1/zarr.json'"; then
        echo "[ERROR] Corrected-LAFAN1 DiffSR cache is missing: ${remote_dataset}" >&2
        exit 2
    fi
    echo "[PASS] Corrected-LAFAN1 manifest and DiffSR cache match the frozen data root."
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
    "CLUSTER_SLURM_TIME_LIMIT=${WALLTIME}"
    "CLUSTER_SLURM_QOS=${QOS}"
    CLUSTER_SLURM_JOB_NAME=lafan1-diffsr-5b
    "DRY_RUN=${DRY_RUN}"
    MODE=lafan1-motion-tracking
    TASK=Isaac-Imitation-G1-Latent-v0
    RUN_BASE_PIPELINE=1
    RUN_ORACLE_RECON_EVAL=0
    RUN_BASE_PLANNER_PREDICT_EVAL=0
    RUN_ORACLE_LL_EVAL=0
    RUN_BASE_PLANNER_LL_EVAL=0
    RUN_PLANNER_FT_SAMPLE_COLLECTION=0
    RUN_PLANNER_ROLLOUT_FINETUNE=0
    RUN_FINETUNED_PLANNER_PREDICT_EVAL=0
    RUN_FINETUNED_PLANNER_LL_EVAL=0
    RUN_HAND_DESIGNED_BASELINES=0
    SKIP_EVAL=1
    RUN_M1_EVAL=0
    LOW_LEVEL_ALGO=IPMD
    "LOW_LEVEL_MAX_ITERATIONS=${MAX_ITERATIONS}"
    "SAVE_INTERVAL=${SAVE_INTERVAL}"
    "NUM_ENVS=${NUM_ENVS}"
    HORIZON_STEPS=10
    STATE_HISTORY_STEPS=9
    Z_DIM=256
    "SKILL_UPDATES=${SKILL_UPDATES}"
    "PLANNER_UPDATES=${PLANNER_UPDATES}"
    LOGGER_BACKEND=wandb
    LOGGER_PROJECT_NAME=G1-Imitation-LAFAN1-Causal-Interface
    "SEED=${SEED}"
    "MANIFEST_PATH=${MANIFEST_PATH}"
    "DATASET_PATH=${DATASET_PATH}"
    "RUN_ID=${RUN_TAG}_latent_train"
    "RUN_ROOT=${RUN_ROOT}"
    RANKS=0
    LIMIT=1
    experiments/interface_baselines/submit_cluster_interface_baselines.sh
)

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

cat <<EOF
[INFO] Submitted interface: DiffSR latent only
[INFO] Frozen manifest: ${MANIFEST_PATH}
[INFO] Frozen cache: ${DATASET_PATH}
[INFO] Budget: $((NUM_ENVS * 24 * MAX_ITERATIONS)) environment frames
[INFO] This job trains no paper-facing planner. Phase 4 remains gated on the
[INFO] final latent oracle audit, vanilla oracle audit, and streamed equivalence.
EOF
