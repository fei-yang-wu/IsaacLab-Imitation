#!/usr/bin/env bash
set -euo pipefail

# Sanity check for the 2026-07-21 default reversal: submit corrected-LAFAN1
# runs on Isaac-Imitation-G1-Latent-v0 (now the Strict/legacy-optimizer
# surface again) to confirm it reproduces good training behavior after the
# gym-registration flip.
#
# Two arms:
#   - scaled:            8192 envs x 12 rollout steps, minibatch 12288 --
#                         exactly reproduces W&B run bn931wny
#                         (g1-lafan1-strict/ice3-l1-novideo), which reached
#                         episode/length=244.18 / episode/return=13.11. This
#                         is the actual correctness check.
#   - hardcoded_default:  4096 envs x 24 rollout steps, minibatch 24576 (the
#                         code's own literal default shape,
#                         rlopt_ipmd_cfg.py: 4096 envs, frames_per_batch=24,
#                         mini_batch_size=4096*24//4) -- a second, cheaper
#                         data point on whether scale matters here, not
#                         validated against any known-good run.
#
# njmax=320/nconmax=40 (validated headroom; see the SONIC VRAM ablation
# findings in wiki/current-status.md) is used for both arms since bn931wny's
# own solver settings were not recorded in its W&B config.
#
# Frame cap: ~1B per AGENTS.md's default cluster-job budget (bn931wny itself
# targeted collector.total_frames=999948288, i.e. ~1B).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
FRAME_CAP=1000000000
MANIFEST_PATH="/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json"
DATASET_PATH="/data/lafan1_corrected_8e95d557/g1_hl_diffsr"
EXPECTED_MANIFEST_SHA256="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"
EXPECTED_NPZ_COUNT=40
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"

# ARM_NAME:NUM_ENVS:ROLLOUT_STEPS:MINIBATCH_SIZE
ARMS=(
    "scaled_e8192_r12:8192:12:12288"
    "hardcoded_default_e4096_r24:4096:24:24576"
)
ARM_FILTER="${ARM_FILTER:-}"

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    0|false|FALSE|no|NO|off|OFF)
        actual_remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
        remote_npz_count="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
        if [[ "${actual_remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
            echo "[ERROR] ICE corrected-LAFAN1 data gate failed: sha=${actual_remote_sha}, npz=${remote_npz_count}." >&2
            exit 2
        fi
        ;;
    *)
        echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2
        exit 2
        ;;
esac

for arm in "${ARMS[@]}"; do
    IFS=':' read -r arm_name num_envs rollout_steps minibatch_size <<< "${arm}"
    if [[ -n "${ARM_FILTER}" ]]; then
        case " ${ARM_FILTER} " in
            *" ${arm_name} "*) ;;
            *) continue ;;
        esac
    fi
    run_tag="lafan1_strict_default_sanity_${arm_name}_1b_seed${SEED}_20260721"
    pretrain_output_dir="logs/lafan1_strict_default_sanity/${run_tag}/skill_encoder_h25_z256"

    if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && "${DRY_RUN}" != "TRUE" && "${DRY_RUN}" != "yes" && "${DRY_RUN}" != "YES" && "${DRY_RUN}" != "on" && "${DRY_RUN}" != "ON" ]]; then
        remote_pretrain_output="${REMOTE_PROJECT_ROOT}/${pretrain_output_dir}"
        if ssh -o BatchMode=yes -o ConnectTimeout=10 ice "test -e '${remote_pretrain_output}'"; then
            echo "[ERROR] Refusing to reuse existing ICE output: ${remote_pretrain_output}" >&2
            exit 2
        fi
    fi

    echo "[INFO] Submitting arm ${arm_name}: envs=${num_envs} rollout=${rollout_steps} minibatch=${minibatch_size}"

    extra_args=(
        --assert-kitless
        --pretrain-output-dir "${pretrain_output_dir}"
        --pretrain-override physics=newton_mjwarp
        --pretrain-override env.refresh_zarr_dataset=true
        --train-override physics=newton_mjwarp
        --train-override env.sim.physics.solver_cfg.njmax=320
        --train-override env.sim.physics.solver_cfg.nconmax=40
        --train-override env.refresh_zarr_dataset=false
    )
    printf -v extra_args_string '%q ' "${extra_args[@]}"

    TASK=Isaac-Imitation-G1-Latent-v0 \
    SEED="${SEED}" \
    FRAME_CAP="${FRAME_CAP}" \
    TRAIN_NUM_ENVS="${num_envs}" \
    ROLLOUT_STEPS="${rollout_steps}" \
    MINIBATCH_SIZE="${minibatch_size}" \
    PRETRAIN_NUM_ENVS=16 \
    PRETRAIN_UPDATES=5000 \
    PRETRAIN_BATCH_SIZE=8192 \
    HORIZON_STEPS=25 \
    TRAIN_VIDEO=0 \
    SAVE_INTERVAL=100000000 \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    DATASET_PATH="${DATASET_PATH}" \
    WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-strict}" \
    WANDB_GROUP="${WANDB_GROUP:-default-reversal-sanity-1b}" \
    EXP_NAME="${run_tag}" \
    CLUSTER_CONFIG=ice_runtime \
    CLUSTER_SLURM_TIME_LIMIT=15:59:00 \
    CLUSTER_SLURM_PARTITION=ice-gpu \
    CLUSTER_SLURM_QOS=coe-ice \
    CLUSTER_SLURM_GPU_GRES=gpu:h100:1 \
    CLUSTER_SLURM_CPUS_PER_TASK=16 \
    CLUSTER_SLURM_MEM=96G \
    CLUSTER_SLURM_JOB_NAME_PREFIX="lafan1-strict-sanity-${arm_name}" \
    CLUSTER_G1_USD_PATH=repo \
    EXTRA_PIPELINE_ARGS="${extra_args_string}" \
    DRY_RUN="${DRY_RUN}" \
        "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
done

echo "[INFO] Submitted all requested LAFAN1 strict-default sanity arms."
