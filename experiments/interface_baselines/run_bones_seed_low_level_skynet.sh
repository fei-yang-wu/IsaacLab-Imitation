#!/usr/bin/env bash
set -euo pipefail

# Submit the two final-candidate low-level controllers on the fresh BONES-SEED
# 100-motion manifest. This does not submit high-level planner comparisons.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SUBMIT_LATENT="${SUBMIT_LATENT:-1}"
SUBMIT_VANILLA="${SUBMIT_VANILLA:-1}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10173}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"
SKILL_UPDATES="${SKILL_UPDATES:-5000}"
PLANNER_UPDATES="${PLANNER_UPDATES:-5000}"
WALLTIME="${WALLTIME:-2-00:00:00}"
QOS="${QOS:-long}"
VERIFY_REMOTE_DATA="${VERIFY_REMOTE_DATA:-1}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/nethome/fwu91/scratch/Research/IsaacLab/data/bones_seed_phase5/bones_seed_100}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-fd285a287d98a8478574da211b7dbf1cf8fbfca974ecf9ba62c200e4a3b87b97}"
EXPECTED_PREPARATION_SHA256="${EXPECTED_PREPARATION_SHA256:-53dfcb3718f758edbf81b817066f4573548aa2a214ed17642162c29b6169bd37}"
EXPECTED_LANGUAGE_SHA256="${EXPECTED_LANGUAGE_SHA256:-3a50746d575d3c8d36c2c4e460acf4834a22a74e663a27d9f04ac8a6137c7975}"

MANIFEST="${MANIFEST:-/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json}"
PREPARATION_RECORD="${PREPARATION_RECORD:-/data/bones_seed_phase5/bones_seed_100/preparation/preparation.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed${SEED}}"
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed${SEED}}"
RUN_TAG="${RUN_TAG:-bones_seed_100_phase5_1b_seed${SEED}}"
LATENT_RUN_ROOT="${LATENT_RUN_ROOT:-logs/interface_baselines/${RUN_TAG}/latent}"

if [[ "${NUM_ENVS}" != "4096" || "${MAX_ITERATIONS}" != "10173" ]]; then
    echo "[ERROR] The default final-candidate block is fixed at 4096 envs x 10173 iterations (~1B frames)." >&2
    echo "[HINT] Change both only for an explicitly recorded protocol revision." >&2
    exit 2
fi
if [[ "${SEED}" != "0" ]]; then
    echo "[ERROR] Low-level qualification is fixed to seed 0 before planner multi-seed runs." >&2
    exit 2
fi

if [[ "${VERIFY_REMOTE_DATA}" == "1" ]]; then
    remote_hashes="$(ssh -o BatchMode=yes -o ConnectTimeout=10 skynet \
        "sha256sum '${REMOTE_DATA_ROOT}/manifests/g1_bones_seed_100_phase5_manifest.json' '${REMOTE_DATA_ROOT}/preparation/preparation.json' '${REMOTE_DATA_ROOT}/language/g1_bones_seed_100_minilm_goal_embeddings.pt'")"
    mapfile -t actual_hashes < <(printf '%s\n' "${remote_hashes}" | awk '{print $1}')
    expected_hashes=(
        "${EXPECTED_MANIFEST_SHA256}"
        "${EXPECTED_PREPARATION_SHA256}"
        "${EXPECTED_LANGUAGE_SHA256}"
    )
    if [[ "${actual_hashes[*]}" != "${expected_hashes[*]}" ]]; then
        echo "[ERROR] Persistent BONES-SEED hashes do not match the frozen data gate." >&2
        printf '[INFO] expected: %s\n' "${expected_hashes[*]}" >&2
        printf '[INFO] actual:   %s\n' "${actual_hashes[*]}" >&2
        exit 2
    fi
    echo "[PASS] Persistent BONES-SEED manifest, preparation, and language hashes match."
fi

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
        "$@"
        return
    fi
    "$@"
}

common_cluster_env=(
    "CLUSTER_AUTO_SETUP_G1_DATA=0"
    # Store one verified archive remotely and extract it on compute-local disk.
    # Skynet's NFS mount can stall on full small-file workspace copies.
    "CLUSTER_ARCHIVE_SYNC=1"
    "CLUSTER_GIT_SYNC_FIRST=0"
    "CLUSTER_INCREMENTAL_SYNC=0"
    "CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=0"
    "CLUSTER_EXTRA_RSYNC_EXCLUDES=data/ .tmp/ RLOpt/ ImitationLearningTools/"
    "CLUSTER_SKIP_CACHE_COPY=1"
    "CLUSTER_USE_SHARED_SIF=1"
    "CLUSTER_OVERLAY_SIZE_MB=8192"
    "CLUSTER_SLURM_TIME_LIMIT=${WALLTIME}"
    "CLUSTER_SLURM_QOS=${QOS}"
    "DRY_RUN=${DRY_RUN}"
)

if [[ "${SUBMIT_LATENT}" == "1" ]]; then
    run_cmd env \
        "${common_cluster_env[@]}" \
        MODE=lafan1-motion-tracking \
        RUN_BASE_PIPELINE=1 \
        RUN_ORACLE_RECON_EVAL=0 \
        RUN_BASE_PLANNER_PREDICT_EVAL=0 \
        RUN_ORACLE_LL_EVAL=0 \
        RUN_BASE_PLANNER_LL_EVAL=0 \
        RUN_PLANNER_FT_SAMPLE_COLLECTION=0 \
        RUN_PLANNER_ROLLOUT_FINETUNE=0 \
        RUN_FINETUNED_PLANNER_PREDICT_EVAL=0 \
        RUN_FINETUNED_PLANNER_LL_EVAL=0 \
        RUN_HAND_DESIGNED_BASELINES=0 \
        SKIP_EVAL=1 \
        RUN_M1_EVAL=0 \
        LOW_LEVEL_ALGO=IPMD \
        LOW_LEVEL_MAX_ITERATIONS="${MAX_ITERATIONS}" \
        SAVE_INTERVAL="${SAVE_INTERVAL}" \
        NUM_ENVS="${NUM_ENVS}" \
        HORIZON_STEPS=10 \
        STATE_HISTORY_STEPS=9 \
        Z_DIM=256 \
        SKILL_UPDATES="${SKILL_UPDATES}" \
        PLANNER_UPDATES="${PLANNER_UPDATES}" \
        LOGGER_BACKEND=wandb \
        LOGGER_PROJECT_NAME=G1-Imitation-BONES-SEED-Phase5 \
        SEED="${SEED}" \
        MANIFEST_PATH="${MANIFEST}" \
        DATASET_PATH="${LATENT_DATASET_PATH}" \
        RUN_ID="${RUN_TAG}_latent_train" \
        RUN_ROOT="${LATENT_RUN_ROOT}" \
        RANKS=0 \
        LIMIT=1 \
        experiments/interface_baselines/submit_cluster_interface_baselines.sh
fi

if [[ "${SUBMIT_VANILLA}" == "1" ]]; then
    vanilla_overrides="env.dataset_path=${VANILLA_DATASET_PATH} env.random_reset_step_min=0 env.random_reset_step_max=200 env.random_reset_full_trajectory=false env.command_hold_steps=0 env.reconstructed_reference_action=true agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0 agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0 agent.ipmd.reward_param_weight_decay_coeff=0.0 agent.ipmd.use_estimated_rewards_for_ppo=false agent.ipmd.env_reward_weight=1.0 agent.ipmd.bc_coef=0.0 agent.ipmd.rollout_bc_coef=0.0 agent.value_function.num_cells=[768,512,256] agent.logger.backend=wandb"
    run_cmd env \
        "${common_cluster_env[@]}" \
        COMMAND_SPACES=single_frame_full_body \
        SEEDS="${SEED}" \
        NUM_ENVS="${NUM_ENVS}" \
        MAX_ITERATIONS="${MAX_ITERATIONS}" \
        COMMAND_FUTURE_STEPS=0 \
        MANIFEST="${MANIFEST}" \
        REFRESH_ZARR_DATASET=true \
        SAVE_INTERVAL="${SAVE_INTERVAL}" \
        VIDEO=0 \
        PROJECT_NAME=G1-Imitation-BONES-SEED-Phase5 \
        GROUP_NAME="${RUN_TAG}" \
        RUN_PREFIX="${RUN_TAG}" \
        EXTRA_OVERRIDES="${vanilla_overrides}" \
        experiments/command_space_ablation/submit_cluster_oracle_ablation.sh
fi

cat <<EOF
[INFO] Fresh manifest: ${MANIFEST}
[INFO] Preparation record: ${PREPARATION_RECORD}
[INFO] Budget per controller: $((NUM_ENVS * 24 * MAX_ITERATIONS)) environment frames
[INFO] This stage trains low-level candidates only. Run strict 100-motion oracle
[INFO] evaluation and streamed-vanilla equivalence before planner submission.
EOF
