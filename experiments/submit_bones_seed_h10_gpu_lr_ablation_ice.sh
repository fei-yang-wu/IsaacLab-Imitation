#!/usr/bin/env bash
set -euo pipefail

# Short BONES-SEED-91 screen for wall-clock convergence of the h10 latent
# controller.  Pretrain one shared encoder first, then reuse that exact
# checkpoint in every low-level arm so GPU and optimizer comparisons are not
# confounded by different latent encoders.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${STAGE:-pretrain}"
DRY_RUN="${DRY_RUN:-1}"
SEED="${SEED:-0}"
FRAME_CAP="${FRAME_CAP:-500000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25000000}"
ARM_FILTER="${ARM_FILTER:-}"

TASK="Isaac-Imitation-G1-Latent-v0"
MANIFEST_PATH="/data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json"
DATASET_PATH="/data/bones_seed_100/g1_hl_diffsr_sonic_filtered"
EXPECTED_MANIFEST_SHA256="8d48750177efb3e9118c5d0ca14b69d62abedff16eb8c00585920a34bd87ee8d"
EXPECTED_NPZ_COUNT=100
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"

ABLATION_TAG="${ABLATION_TAG:-bones_seed_91_h10_gpu_lr_ablation_20260722}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-logs/${ABLATION_TAG}/skill_encoder_h10_z256}"
PRETRAIN_CHECKPOINT="${PRETRAIN_OUTPUT_DIR}/checkpoints/latest.pt"
REMOTE_PRETRAIN_CHECKPOINT="${REMOTE_PROJECT_ROOT}/${PRETRAIN_CHECKPOINT}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed-h10-gpu-lr-ablation-ice}"
WANDB_GROUP="${WANDB_GROUP:-${ABLATION_TAG}}"

local_manifest="${REPO_ROOT}/data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json"
actual_local_sha="$(sha256sum "${local_manifest}" | awk '{print $1}')"
if [[ "${actual_local_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Local BONES-SEED manifest hash mismatch: ${actual_local_sha}" >&2
    exit 2
fi

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *) echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2; exit 2 ;;
esac

if (( is_dry_run == 0 )); then
    read -r remote_sha remote_npz_count < <(
        ssh -o BatchMode=yes -o ConnectTimeout=10 ice bash -s -- \
            "${REMOTE_DATA_ROOT}" <<'REMOTE_EOF'
set -euo pipefail
root="$1"
sha256sum "${root}/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json" \
    | awk '{printf "%s ", $1}'
find "${root}/bones_seed_100/npz/g1" -type f -name '*.npz' | wc -l
REMOTE_EOF
    )
    if [[ "${remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE BONES-SEED data gate failed: sha=${remote_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
fi

common_extra=(
    --assert-kitless
    --phase-mode sin_cos
    --latent-hold-steps 10
    --train-override physics=newton_mjwarp
    --train-override env.sim.physics.solver_cfg.njmax=320
    --train-override env.sim.physics.solver_cfg.nconmax=40
    --train-override env.refresh_zarr_dataset=false
)

submit_pipeline() {
    local gpu_type="$1"
    local job_prefix="$2"
    local exp_name="$3"
    local train_envs="$4"
    local rollout_steps="$5"
    local minibatch_size="$6"
    shift 6
    local -a extra=("${common_extra[@]}" "$@")
    local extra_string
    printf -v extra_string '%q ' "${extra[@]}"

    TASK="${TASK}" \
    SEED="${SEED}" \
    FRAME_CAP="${FRAME_CAP}" \
    TRAIN_NUM_ENVS="${train_envs}" \
    ROLLOUT_STEPS="${rollout_steps}" \
    MINIBATCH_SIZE="${minibatch_size}" \
    PRETRAIN_NUM_ENVS=16 \
    PRETRAIN_UPDATES=50000 \
    PRETRAIN_BATCH_SIZE=8192 \
    HORIZON_STEPS=10 \
    TRAIN_VIDEO=0 \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    DATASET_PATH="${DATASET_PATH}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_GROUP="${WANDB_GROUP}" \
    EXP_NAME="${exp_name}" \
    CLUSTER_CONFIG=ice_runtime \
    CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-05:00:00}" \
    CLUSTER_SLURM_PARTITION=ice-gpu \
    CLUSTER_SLURM_QOS=coe-ice \
    CLUSTER_SLURM_GPU_GRES="gpu:${gpu_type}:1" \
    CLUSTER_SLURM_CPUS_PER_TASK=16 \
    CLUSTER_SLURM_MEM=128G \
    CLUSTER_SLURM_JOB_NAME_PREFIX="${job_prefix}" \
    CLUSTER_G1_USD_PATH=repo \
    EXTRA_PIPELINE_ARGS="${extra_string}" \
    DRY_RUN="${DRY_RUN}" \
        "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
}

case "${STAGE}" in
    pretrain)
        if (( is_dry_run == 0 )) && ssh ice "test -e '${REMOTE_PROJECT_ROOT}/${PRETRAIN_OUTPUT_DIR}'"; then
            echo "[ERROR] Refusing to reuse existing shared encoder output: ${REMOTE_PROJECT_ROOT}/${PRETRAIN_OUTPUT_DIR}" >&2
            exit 2
        fi
        common_extra=(
            --assert-kitless
            --pretrain-only
            --pretrain-output-dir "${PRETRAIN_OUTPUT_DIR}"
            --categorical-groups 64
            --categorical-categories 128
            --gumbel-hard
            --pretrain-override physics=newton_mjwarp
            --pretrain-override env.refresh_zarr_dataset=true
        )
        submit_pipeline h100 bones-h10-encoder "${ABLATION_TAG}_encoder" 64 12 96
        ;;
    train)
        if (( is_dry_run == 0 )) && ! ssh ice "test -s '${REMOTE_PRETRAIN_CHECKPOINT}'"; then
            if [[ ! "${CLUSTER_SLURM_DEPENDENCY:-}" =~ ^afterok:[0-9]+(:[0-9]+)*$ ]]; then
                echo "[ERROR] Shared h10 encoder is not ready and no valid afterok dependency was supplied: ${REMOTE_PRETRAIN_CHECKPOINT}" >&2
                exit 2
            fi
            echo "[INFO] Encoder checkpoint is pending; deferring to ${CLUSTER_SLURM_DEPENDENCY}."
        fi

        # name:gpu:envs:rollout:minibatch:actor_lr:actor_lr_cap
        # The two base_lr1e3 arms isolate GPU hardware.  h200_e16384 isolates
        # added H200 capacity.  The remaining H100 arms isolate actor LR while
        # keeping critic LR and optimizer work per sample fixed.
        arms=(
            "h100_e12288_lr1e3:h100:12288:12:18432:1.0e-3:1.0e-3"
            "h200_e12288_lr1e3:h200:12288:12:18432:1.0e-3:1.0e-3"
            "h200_e16384_lr1e3:h200:16384:12:24576:1.0e-3:1.0e-3"
            "h100_e12288_lr6e4:h100:12288:12:18432:6.0e-4:6.0e-4"
            "h100_e12288_lr3e4:h100:12288:12:18432:3.0e-4:3.0e-4"
        )
        for arm in "${arms[@]}"; do
            IFS=: read -r name gpu envs rollout minibatch actor_lr actor_lr_cap <<< "${arm}"
            if [[ -n "${ARM_FILTER}" ]]; then
                case " ${ARM_FILTER} " in *" ${name} "*) ;; *) continue ;; esac
            fi
            echo "[INFO] Arm ${name}: gpu=${gpu}, envs=${envs}, rollout=${rollout}, actor_lr=${actor_lr}, cap=${actor_lr_cap}"
            submit_pipeline \
                "${gpu}" "bones-h10-${name}" "${ABLATION_TAG}_${name}" \
                "${envs}" "${rollout}" "${minibatch}" \
                --skip-pretrain \
                --pretrained-checkpoint "${PRETRAIN_CHECKPOINT}" \
                --train-override "agent.ipmd.actor_learning_rate=${actor_lr}" \
                --train-override agent.ipmd.critic_learning_rate=1.0e-3 \
                --train-override "agent.optim.max_lr=${actor_lr_cap}"
        done
        ;;
    *)
        echo "[ERROR] STAGE must be pretrain or train; got '${STAGE}'." >&2
        exit 2
        ;;
esac
