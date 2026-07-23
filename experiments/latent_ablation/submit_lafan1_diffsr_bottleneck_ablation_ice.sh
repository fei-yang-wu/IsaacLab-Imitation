#!/usr/bin/env bash
set -euo pipefail

# DiffSR bottleneck study. Each job performs a fresh, matched 50k-update h10
# pretrain and then trains a frozen-encoder low-level controller. Default mode
# prints commands and cannot submit with the unapproved example profile.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
PROFILE_FILE="${TRAINING_PROFILE:-${SCRIPT_DIR}/training_profile.example.env}"
# shellcheck disable=SC1090
source "${PROFILE_FILE}"

MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-diffsr-bottleneck-ablation-ice}"
WANDB_GROUP="${WANDB_GROUP:-diffsr-bottlenecks-h10-seed0}"
SEED="${SEED:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25000000}"
ARMS="${ARMS:-deterministic gaussian categorical gumbel_multicat gumbel fsq vq}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"

# Match the established multi-group capacity: 64 independent symbols with
# 128 choices each = 64 * log2(128) = 448 bits per latent command.
GROUPED_SYMBOLS=64
GROUPED_LEVELS=128
FSQ_448_LEVELS=()
for ((group_idx = 0; group_idx < GROUPED_SYMBOLS; group_idx++)); do
    FSQ_448_LEVELS+=("${GROUPED_LEVELS}")
done

if [[ -n "${TRAIN_CHECKPOINT}" ]]; then
    if [[ "${ARMS}" == *" "* || -z "${PRETRAINED_CHECKPOINT}" ]]; then
        echo "[ERROR] DiffSR resume requires one ARMS value plus TRAIN_CHECKPOINT and PRETRAINED_CHECKPOINT." >&2
        exit 2
    fi
fi
remaining_frames=$((FRAME_CAP - COMPLETED_FRAMES))
if (( remaining_frames <= 0 )); then
    echo "[INFO] FRAME_CAP=${FRAME_CAP} already credited by COMPLETED_FRAMES=${COMPLETED_FRAMES}."
    exit 0
fi

FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
cap_iterations=$((remaining_frames / FRAMES_PER_BATCH))
wall_iterations=$((SEGMENT_TRAIN_SECONDS * ASSUMED_FPS / FRAMES_PER_BATCH))
MAX_ITERATIONS=$((cap_iterations < wall_iterations ? cap_iterations : wall_iterations))

case "${MODE}" in
    print) ;;
    submit)
        if [[ "${PROFILE_APPROVED}" != "1" || "${CONFIRM_SUBMIT:-}" != "lafan1-latent-ablation" ]]; then
            echo "[ERROR] Submission requires PROFILE_APPROVED=1 and CONFIRM_SUBMIT=lafan1-latent-ablation." >&2
            exit 2
        fi
        ;;
    *) echo "[ERROR] MODE must be print or submit; got ${MODE}." >&2; exit 2 ;;
esac

mode_args() {
    local arm="$1"
    MODE_ARGS=(--latent-mode "${arm}")
    CAPACITY_TAG="continuous"
    case "${arm}" in
        deterministic|gaussian) ;;
        categorical|gumbel_multicat)
            MODE_ARGS+=(
                --categorical-groups "${GROUPED_SYMBOLS}"
                --categorical-categories "${GROUPED_LEVELS}"
            )
            CAPACITY_TAG="b448"
            if [[ "${arm}" == gumbel_multicat ]]; then
                MODE_ARGS+=(--gumbel-hard)
            fi
            ;;
        gumbel)
            MODE_ARGS+=(--gumbel-codebook-size 512 --gumbel-hard)
            CAPACITY_TAG="b9_unmatched"
            ;;
        fsq)
            MODE_ARGS+=(--fsq-levels "${FSQ_448_LEVELS[@]}")
            CAPACITY_TAG="b448"
            ;;
        vq)
            MODE_ARGS+=(--vq-codebook-size 512 --vq-ema-decay 0.99 --vq-dead-code-reset-iters 1000)
            CAPACITY_TAG="b9_unmatched"
            ;;
        *) echo "[ERROR] Unknown DiffSR arm: ${arm}." >&2; exit 2 ;;
    esac
}

for arm in ${ARMS}; do
    mode_args "${arm}"
    run_tag="lafan1_diffsr_${arm}_${CAPACITY_TAG}_h10_z256_seed${SEED}"
    pretrain_dir="logs/latent_ablation/${run_tag}/skill_encoder"
    extra=(
        --assert-kitless
        --pretrain-output-dir "${pretrain_dir}"
        --encoder-hidden-dims 1024 512 512
        "${MODE_ARGS[@]}"
        --phase-mode sin_cos
        --latent-hold-steps 10
        --pretrain-override physics=newton_mjwarp
        --pretrain-override env.refresh_zarr_dataset=true
        --train-override physics=newton_mjwarp
        --train-override agent.ipmd.actor_learning_rate="${ACTOR_LR}"
        --train-override agent.ipmd.critic_learning_rate="${CRITIC_LR}"
        --train-override agent.optim.max_lr="${ACTOR_LR_CAP}"
        --train-override env.sim.physics.solver_cfg.njmax=320
        --train-override env.sim.physics.solver_cfg.nconmax=40
        --train-override env.refresh_zarr_dataset=false
    )
    if [[ -n "${TRAIN_CHECKPOINT}" ]]; then
        extra=(
            --assert-kitless
            --skip-pretrain
            --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
            --train-checkpoint "${TRAIN_CHECKPOINT}"
            --phase-mode sin_cos
            --latent-hold-steps 10
            --train-override physics=newton_mjwarp
            --train-override agent.ipmd.actor_learning_rate="${ACTOR_LR}"
            --train-override agent.ipmd.critic_learning_rate="${CRITIC_LR}"
            --train-override agent.optim.max_lr="${ACTOR_LR_CAP}"
            --train-override env.sim.physics.solver_cfg.njmax=320
            --train-override env.sim.physics.solver_cfg.nconmax=40
            --train-override env.refresh_zarr_dataset=false
        )
    fi
    printf -v extra_string '%q ' "${extra[@]}"
    cmd=(env
        TASK=Isaac-Imitation-G1-Latent-v0
        SEED="${SEED}"
        FRAME_CAP="${FRAME_CAP}"
        MAX_ITERATIONS="${MAX_ITERATIONS}"
        TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS}"
        ROLLOUT_STEPS="${ROLLOUT_STEPS}"
        MINIBATCH_SIZE="${MINIBATCH_SIZE}"
        PRETRAIN_NUM_ENVS=16
        PRETRAIN_UPDATES=50000
        PRETRAIN_BATCH_SIZE=8192
        HORIZON_STEPS=10
        TRAIN_VIDEO=0
        SAVE_INTERVAL="${SAVE_INTERVAL}"
        MANIFEST_PATH="${MANIFEST_PATH}"
        DATASET_PATH="${DATASET_PATH}"
        WANDB_PROJECT="${WANDB_PROJECT}"
        WANDB_GROUP="${WANDB_GROUP}"
        EXP_NAME="${run_tag}"
        CLUSTER_CONFIG=ice_runtime
        CLUSTER_SLURM_TIME_LIMIT=15:59:00
        CLUSTER_SLURM_PARTITION=ice-gpu
        CLUSTER_SLURM_QOS=coe-ice
        CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
        CLUSTER_SLURM_CPUS_PER_TASK=16
        CLUSTER_SLURM_MEM=128G
        CLUSTER_SLURM_JOB_NAME_PREFIX="lafan-diffsr-${arm}"
        CLUSTER_G1_USD_PATH=repo
        EXTRA_PIPELINE_ARGS="${extra_string}"
        DRY_RUN=0
        "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
    )
    printf '[PLAN] %s: ' "${arm}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${MODE}" == "submit" ]]; then
        "${cmd[@]}"
    fi
done
