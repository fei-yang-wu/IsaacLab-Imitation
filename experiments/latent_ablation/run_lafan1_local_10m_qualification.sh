#!/usr/bin/env bash
set -euo pipefail

# Sequential local qualification for every latent-learning arm. This validates
# wiring and an early learning signal; it is not a converged comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
TARGET_FRAMES="${TARGET_FRAMES:-10000000}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((NUM_ENVS * ROLLOUT_STEPS / 8))}"
# This is a wiring/early-learning qualification rather than the production
# encoder fit. H200 runs retain the full 50k-update pretraining budget.
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-5000}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/latent_ablation/local_10m_${RUN_ID}}"
ARMS="${ARMS:-continuous_ae vqvae fsq_recon sonic_fsq_pg cvae deterministic gaussian categorical gumbel_multicat gumbel fsq vq}"

if [[ "${MODE}" != "print" && "${MODE}" != "run" ]]; then
    echo "[ERROR] MODE must be print or run; got ${MODE}." >&2
    exit 2
fi
if [[ ! -f "${MANIFEST_PATH}" || ! -d "${DATASET_PATH}" ]]; then
    echo "[ERROR] Missing corrected LAFAN1 manifest or cache." >&2
    echo "[ERROR] manifest=${MANIFEST_PATH} dataset=${DATASET_PATH}" >&2
    exit 2
fi
if (( TARGET_FRAMES > 10000000 )); then
    echo "[ERROR] This qualification launcher is capped at 10M frames per arm." >&2
    exit 2
fi

FRAMES_PER_BATCH=$((NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((TARGET_FRAMES + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
EFFECTIVE_FRAMES=$((MAX_ITERATIONS * FRAMES_PER_BATCH))

is_reconstruction_arm() {
    case "$1" in
        continuous_ae|vqvae|fsq_recon|sonic_fsq_pg|cvae) return 0 ;;
        *) return 1 ;;
    esac
}

reconstruction_overrides() {
    local arm="$1"
    ARM_OVERRIDES=()
    case "${arm}" in
        continuous_ae)
            ARM_OVERRIDES+=(agent.ipmd.latent_learning.quantizer=identity)
            ;;
        vqvae)
            ARM_OVERRIDES+=(
                agent.ipmd.latent_learning.quantizer=vq_ema
                agent.ipmd.latent_learning.codebook_size=512
                agent.ipmd.latent_learning.codebook_embed_dim=64
                agent.ipmd.latent_learning.commitment_coeff=0.25
                agent.ipmd.latent_learning.dead_code_reset_iters=1000
            )
            ;;
        fsq_recon|sonic_fsq_pg)
            ARM_OVERRIDES+=(
                agent.ipmd.latent_learning.quantizer=fsq
                'agent.ipmd.latent_learning.fsq_levels=[4,4,4,4,4]'
            )
            ;;
        cvae)
            ARM_OVERRIDES+=(
                env.latent_command_dim=64
                agent.ipmd.latent_dim=64
                agent.ipmd.latent_learning.method=future_cvae
                agent.ipmd.latent_learning.code_latent_dim=64
                agent.ipmd.latent_learning.command_phase_mode=none
                agent.ipmd.latent_learning.recon_coeff=1.0
                agent.ipmd.latent_learning.kl_coeff=0.01
            )
            ;;
    esac
    if [[ "${arm}" == "sonic_fsq_pg" ]]; then
        ARM_OVERRIDES+=(agent.ipmd.latent_learning.train_posterior_through_policy=true)
    else
        ARM_OVERRIDES+=(agent.ipmd.latent_learning.train_posterior_through_policy=false)
    fi
}

diffsr_args() {
    local arm="$1"
    DIFFSR_ARGS=(--latent-mode "${arm}")
    case "${arm}" in
        deterministic|gaussian) ;;
        categorical|gumbel_multicat)
            DIFFSR_ARGS+=(--categorical-groups 64 --categorical-categories 128)
            if [[ "${arm}" == "gumbel_multicat" ]]; then
                DIFFSR_ARGS+=(--gumbel-hard)
            fi
            ;;
        gumbel) DIFFSR_ARGS+=(--gumbel-codebook-size 512 --gumbel-hard) ;;
        fsq)
            local levels=()
            for ((idx = 0; idx < 64; idx++)); do levels+=(128); done
            DIFFSR_ARGS+=(--fsq-levels "${levels[@]}")
            ;;
        vq) DIFFSR_ARGS+=(--vq-codebook-size 512 --vq-ema-decay 0.99 --vq-dead-code-reset-iters 1000) ;;
        *) echo "[ERROR] Unknown arm: ${arm}." >&2; exit 2 ;;
    esac
}

mkdir -p "${OUTPUT_ROOT}"
overall_status=0
for arm in ${ARMS}; do
    arm_root="${OUTPUT_ROOT}/${arm}"
    train_log="${arm_root}/train.log"
    qualification="${arm_root}/qualification.json"
    mkdir -p "${arm_root}"
    if is_reconstruction_arm "${arm}"; then
        reconstruction_overrides "${arm}"
        cmd=(env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1
            pixi run -e isaaclab python scripts/rlopt/train.py
            --task Isaac-Imitation-G1-Latent-Ablation-v0
            --algo IPMD
            --num_envs "${NUM_ENVS}"
            --max_iterations "${MAX_ITERATIONS}"
            --log_interval 1
            --seed "${SEED}"
            --headless
            --kit_args=--/app/extensions/fsWatcherEnabled=false
            physics=newton_mjwarp
            env.sim.physics.solver_cfg.njmax=320
            env.sim.physics.solver_cfg.nconmax=40
            "env.lafan1_manifest_path=${MANIFEST_PATH}"
            "env.dataset_path=${DATASET_PATH}"
            env.refresh_zarr_dataset=false
            "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
            "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
            "agent.ipmd.expert_batch_size=${MINIBATCH_SIZE}"
            agent.ipmd.actor_learning_rate=1.0e-3
            agent.ipmd.critic_learning_rate=1.0e-3
            agent.optim.max_lr=1.0e-3
            agent.ipmd.latent_learning.lr=3.0e-4
            agent.ipmd.latent_learning.recon_coeff=1.0
            agent.ipmd.latent_learning.action_recon_coeff=0.0
            agent.ipmd.latent_learning.code_period=10
            agent.ipmd.latent_learning.posterior_command_period=10
            agent.ipmd.latent_learning.command_phase_mode=sin_cos
            agent.ipmd.latent_learning.code_latent_dim=64
            agent.ipmd.latent_dim=66
            env.latent_command_dim=66
            agent.ipmd.reward_loss_coeff=0.0
            agent.ipmd.reward_l2_coeff=0.0
            agent.ipmd.reward_grad_penalty_coeff=0.0
            agent.ipmd.use_estimated_rewards_for_ppo=false
            "agent.save_interval=${TARGET_FRAMES}"
            agent.logger.backend=csv
            "agent.logger.log_dir=${arm_root}/rlopt_logs"
            "agent.logger.exp_name=local10m_${arm}_seed${SEED}"
            "${ARM_OVERRIDES[@]}")
    else
        diffsr_args "${arm}"
        cmd=(env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1
            pixi run -e isaaclab python scripts/rlopt/train_hl_skill_pipeline.py
            --headless
            --assert-kitless
            --app-arg=--kit_args=--/app/extensions/fsWatcherEnabled=false
            --task Isaac-Imitation-G1-Latent-v0
            --seed "${SEED}"
            --manifest-path "${MANIFEST_PATH}"
            --dataset-path "${DATASET_PATH}"
            --pretrain-output-dir "${arm_root}/skill_encoder"
            --pretrain-num-envs 16
            --pretrain-updates "${PRETRAIN_UPDATES}"
            --pretrain-batch-size 8192
            --horizon-steps 10
            --z-dim 256
            --encoder-hidden-dims 1024 512 512
            "${DIFFSR_ARGS[@]}"
            --train-num-envs "${NUM_ENVS}"
            --train-max-iterations "${MAX_ITERATIONS}"
            --train-log-interval 1
            --no-train-video
            --phase-mode sin_cos
            --latent-hold-steps 10
            --save-interval "${TARGET_FRAMES}"
            --logger-backend csv
            --exp-name "local10m_diffsr_${arm}_seed${SEED}"
            --pretrain-override physics=newton_mjwarp
            --pretrain-override env.refresh_zarr_dataset=false
            --train-override physics=newton_mjwarp
            --train-override env.sim.physics.solver_cfg.njmax=320
            --train-override env.sim.physics.solver_cfg.nconmax=40
            --train-override env.refresh_zarr_dataset=false
            --train-override "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
            --train-override "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
            --train-override agent.ipmd.actor_learning_rate=1.0e-3
            --train-override agent.ipmd.critic_learning_rate=1.0e-3
            --train-override agent.optim.max_lr=1.0e-3
            --train-override "agent.logger.log_dir=${arm_root}/rlopt_logs")
    fi

    printf '[PLAN] %s (%s frames): ' "${arm}" "${EFFECTIVE_FRAMES}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${MODE}" == "run" ]]; then
        if ! "${cmd[@]}" 2>&1 | tee "${train_log}"; then
            echo "[ERROR] Training command failed for ${arm}; recording the failed audit." >&2
            overall_status=1
        fi
        if ! pixi run python "${SCRIPT_DIR}/analyze_local_qualification.py" \
                --arm "${arm}" \
                --train-log "${train_log}" \
                --run-root "${arm_root}" \
                --target-frames "${TARGET_FRAMES}" \
                --output "${qualification}"; then
            overall_status=1
        fi
    fi
done

echo "[INFO] Local qualification root: ${OUTPUT_ROOT}"
exit "${overall_status}"
