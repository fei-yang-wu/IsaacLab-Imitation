#!/usr/bin/env bash
set -euo pipefail

# Train a fresh BONES-SEED-100 DiffSR skill encoder, then the evidence-backed
# pelvis/strict latent oracle used by the healthy ICE SONIC L1 submission.
# This is not a Phase-5 planner run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
RUN_TAG="${RUN_TAG:-bones_seed_100_strict_h25_z256_1b_seed${SEED}_20260720_nj288_nc32}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/bones_seed_100/manifests/g1_bones_seed_100_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/bones_seed_100/g1_hl_diffsr}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-logs/bones_seed_sonic/${RUN_TAG}/skill_encoder_h25_z256}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-891c883ce192565d5f02eec175d5f2cc40dd39245565b6f1c3c8004dc29af2b5}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

local_manifest="${REPO_ROOT}/data/bones_seed_100/manifests/g1_bones_seed_100_manifest.json"
actual_local_sha="$(sha256sum "${local_manifest}" | awk '{print $1}')"
if [[ "${actual_local_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Local BONES-SEED-100 manifest hash mismatch." >&2
    exit 2
fi

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    0|false|FALSE|no|NO|off|OFF)
        remote_manifest="${REMOTE_DATA_ROOT}/bones_seed_100/manifests/g1_bones_seed_100_manifest.json"
        remote_pretrain_output="${REMOTE_PROJECT_ROOT}/${PRETRAIN_OUTPUT_DIR}"
        actual_remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "sha256sum '${remote_manifest}'" | awk '{print $1}')"
        remote_npz_count="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "find '${REMOTE_DATA_ROOT}/bones_seed_100/npz/g1' -type f -name '*.npz' | wc -l")"
        if [[ "${actual_remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "100" ]]; then
            echo "[ERROR] ICE BONES-SEED-100 data gate failed: sha=${actual_remote_sha}, npz=${remote_npz_count}." >&2
            exit 2
        fi
        if ssh -o BatchMode=yes -o ConnectTimeout=10 ice "test -e '${remote_pretrain_output}'"; then
            echo "[ERROR] Refusing to reuse existing ICE output: ${remote_pretrain_output}" >&2
            exit 2
        fi
        ;;
    *)
        echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2
        exit 2
        ;;
esac

extra_args=(
    --assert-kitless
    --pretrain-output-dir "${PRETRAIN_OUTPUT_DIR}"
    --pretrain-override physics=newton_mjwarp
    --pretrain-override env.refresh_zarr_dataset=true
    --train-override physics=newton_mjwarp
    --train-override env.sim.physics.solver_cfg.njmax=288
    --train-override env.sim.physics.solver_cfg.nconmax=32
    --train-override env.refresh_zarr_dataset=false
    --train-override env.curriculum.anchor_pos_threshold.params.start_value=0.15
    --train-override env.curriculum.anchor_ori_threshold.params.start_value=0.2
    --train-override env.curriculum.ee_body_pos_threshold.params.start_value=0.15
    --train-override env.curriculum.foot_pos_xyz_threshold.params.start_value=0.2
)
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    extra_args+=(--skip-pretrain --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}")
fi
printf -v extra_args_string '%q ' "${extra_args[@]}"

export TASK=Isaac-Imitation-G1-Latent-Strict-v0
export FRAME_CAP=1000000000
export TRAIN_NUM_ENVS=8192
export ROLLOUT_STEPS=12
export MINIBATCH_SIZE=12288
export PRETRAIN_NUM_ENVS=16
export PRETRAIN_UPDATES=5000
export PRETRAIN_BATCH_SIZE=8192
export HORIZON_STEPS=25
export TRAIN_VIDEO=0
export SAVE_INTERVAL=100000000
export MANIFEST_PATH
export DATASET_PATH
export WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed-100-sonic-latent-ice}"
export WANDB_GROUP="${WANDB_GROUP:-sonic-l1-strict-1b}"
export EXP_NAME="${EXP_NAME:-${RUN_TAG}_oracle_low_level}"
export CLUSTER_CONFIG=ice_runtime
export CLUSTER_SLURM_TIME_LIMIT=15:59:00
export CLUSTER_SLURM_PARTITION=ice-gpu
export CLUSTER_SLURM_QOS=coe-ice
export CLUSTER_SLURM_GPU_GRES=gpu:h100:1
export CLUSTER_SLURM_CPUS_PER_TASK=16
export CLUSTER_SLURM_MEM=96G
export CLUSTER_SLURM_JOB_NAME_PREFIX=bones-sonic-latent
export CLUSTER_G1_USD_PATH=repo
export EXTRA_PIPELINE_ARGS="${extra_args_string}"
export DRY_RUN

exec "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
