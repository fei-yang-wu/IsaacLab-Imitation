#!/usr/bin/env bash
set -euo pipefail

# Plan/launcher for reconstruction-trained latent families on corrected LAFAN1.
# Default mode only prints commands. No job can be submitted until the separate
# GPU/LR study produces an explicitly approved training profile.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
PROFILE_FILE="${TRAINING_PROFILE:-${SCRIPT_DIR}/training_profile.example.env}"
# shellcheck disable=SC1090
source "${PROFILE_FILE}"

TASK="Isaac-Imitation-G1-Latent-Ablation-v0"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-latent-learning-ablation-ice}"
WANDB_GROUP="${WANDB_GROUP:-reconstruction-families-h10-seed0}"
SEED="${SEED:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25000000}"
ARMS="${ARMS:-continuous_ae vqvae fsq_recon sonic_fsq_pg cvae}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"

if [[ -n "${TRAIN_CHECKPOINT}" && "${ARMS}" == *" "* ]]; then
    echo "[ERROR] Resume one reconstruction arm at a time by setting ARMS to one value." >&2
    exit 2
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
if (( MAX_ITERATIONS < 1 )); then
    echo "[ERROR] Training profile resolves to zero iterations." >&2
    exit 2
fi

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

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-128G}"
export CLUSTER_G1_USD_PATH=repo

arm_overrides() {
    local arm="$1"
    ARM_OVERRIDES=()
    case "${arm}" in
        continuous_ae)
            ARM_OVERRIDES=(
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=identity
                agent.ipmd.latent_learning.train_posterior_through_policy=false
            )
            ;;
        vqvae)
            ARM_OVERRIDES=(
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=vq_ema
                agent.ipmd.latent_learning.codebook_size=512
                agent.ipmd.latent_learning.codebook_embed_dim=64
                agent.ipmd.latent_learning.commitment_coeff=0.25
                agent.ipmd.latent_learning.dead_code_reset_iters=1000
                agent.ipmd.latent_learning.train_posterior_through_policy=false
            )
            ;;
        fsq_recon)
            ARM_OVERRIDES=(
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=fsq
                'agent.ipmd.latent_learning.fsq_levels=[4,4,4,4,4]'
                agent.ipmd.latent_learning.train_posterior_through_policy=false
            )
            ;;
        sonic_fsq_pg)
            ARM_OVERRIDES=(
                agent.ipmd.latent_learning.method=patch_vqvae
                agent.ipmd.latent_learning.quantizer=fsq
                'agent.ipmd.latent_learning.fsq_levels=[4,4,4,4,4]'
                agent.ipmd.latent_learning.train_posterior_through_policy=true
            )
            ;;
        cvae)
            ARM_OVERRIDES=(
                env.latent_command_dim=64
                agent.ipmd.latent_dim=64
                agent.ipmd.latent_learning.method=future_cvae
                agent.ipmd.latent_learning.code_latent_dim=64
                agent.ipmd.latent_learning.command_phase_mode=none
                agent.ipmd.latent_learning.posterior_command_period=10
                agent.ipmd.latent_learning.recon_coeff=1.0
                agent.ipmd.latent_learning.kl_coeff=0.01
                agent.ipmd.latent_learning.train_posterior_through_policy=false
            )
            ;;
        *) echo "[ERROR] Unknown reconstruction arm: ${arm}." >&2; exit 2 ;;
    esac
}

for arm in ${ARMS}; do
    arm_overrides "${arm}"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="lafan-latent-${arm}"
    exp_name="lafan1_latent_recon_${arm}_h10_seed${SEED}"
    checkpoint_args=()
    if [[ -n "${TRAIN_CHECKPOINT}" ]]; then
        checkpoint_args=(--checkpoint "${TRAIN_CHECKPOINT}")
    fi
    cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}"
        --num_envs "${TRAIN_NUM_ENVS}"
        --headless
        --algo IPMD
        --max_iterations "${MAX_ITERATIONS}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "${checkpoint_args[@]}"
        physics=newton_mjwarp
        env.sim.physics.solver_cfg.njmax=320
        env.sim.physics.solver_cfg.nconmax=40
        "env.lafan1_manifest_path=${MANIFEST_PATH}"
        "env.dataset_path=${DATASET_PATH}"
        env.refresh_zarr_dataset=false
        "agent.seed=${SEED}"
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        "agent.ipmd.expert_batch_size=${MINIBATCH_SIZE}"
        "agent.ipmd.actor_learning_rate=${ACTOR_LR}"
        "agent.ipmd.critic_learning_rate=${CRITIC_LR}"
        "agent.optim.max_lr=${ACTOR_LR_CAP}"
        "agent.ipmd.latent_learning.lr=${LATENT_LR}"
        agent.ipmd.latent_learning.recon_coeff=1.0
        agent.ipmd.latent_learning.action_recon_coeff=0.0
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        agent.ipmd.latent_learning.code_latent_dim=64
        agent.ipmd.latent_dim=66
        env.latent_command_dim=66
        agent.ipmd.reward_loss_coeff=0.0
        agent.ipmd.reward_l2_coeff=0.0
        agent.ipmd.reward_grad_penalty_coeff=0.0
        agent.ipmd.use_estimated_rewards_for_ppo=false
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${exp_name}"
        "${ARM_OVERRIDES[@]}"
    )
    printf '[PLAN] %s [%s]: ' "${arm}" "${GPU_GRES}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${MODE}" == "submit" ]]; then
        "${cmd[@]}"
    fi
done
