#!/usr/bin/env bash
set -euo pipefail

# Train a fresh BONES-SEED-100 DiffSR skill encoder, then the confirmed-
# default SONIC latent oracle (Isaac-Imitation-G1-Latent-v0, release policy
# contract) at the L1 scale config (8192 envs x 12 steps, minibatch 12288),
# capped at 3B frames. Uses the SONIC-release-exclusion-filtered manifest
# (91/100 motions; 9 dropped by scripts/filter_bones_seed_sonic_exclusions.py
# using the exact keyword list from the public SONIC release's
# filter_and_copy_bones_data.py). This is not a Phase-5 planner run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
RUN_TAG="${RUN_TAG:-bones_seed_91_sonic_h25_z256_3b_seed${SEED}_20260720_nj288_nc32}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/bones_seed_100/g1_hl_diffsr_sonic_filtered}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-logs/bones_seed_sonic/${RUN_TAG}/skill_encoder_h25_z256}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-8d48750177efb3e9118c5d0ca14b69d62abedff16eb8c00585920a34bd87ee8d}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-100}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

local_manifest="${REPO_ROOT}/data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json"
actual_local_sha="$(sha256sum "${local_manifest}" | awk '{print $1}')"
if [[ "${actual_local_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Local SONIC-filtered BONES-SEED-100 manifest hash mismatch." >&2
    exit 2
fi

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) ;;
    0|false|FALSE|no|NO|off|OFF)
        remote_manifest="${REMOTE_DATA_ROOT}/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json"
        remote_pretrain_output="${REMOTE_PROJECT_ROOT}/${PRETRAIN_OUTPUT_DIR}"
        actual_remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "sha256sum '${remote_manifest}'" | awk '{print $1}')"
        # The NPZ tree itself is unfiltered (91 of the 100 kept motions are a
        # subset of names within it); only the manifest's trajectory list is
        # SONIC-filtered, so the NPZ count gate stays at 100.
        remote_npz_count="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "find '${REMOTE_DATA_ROOT}/bones_seed_100/npz/g1' -type f -name '*.npz' | wc -l")"
        if [[ "${actual_remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
            echo "[ERROR] ICE BONES-SEED-100 SONIC-filtered data gate failed: sha=${actual_remote_sha}, npz=${remote_npz_count}." >&2
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

# No curriculum overrides: SONIC's strict adaptive terminations are active
# from iteration zero (the release's train-strict convention), matching the
# L1 strict-from-scratch scale config this job mirrors.
extra_args=(
    --assert-kitless
    --pretrain-output-dir "${PRETRAIN_OUTPUT_DIR}"
    --pretrain-override physics=newton_mjwarp
    --pretrain-override env.refresh_zarr_dataset=true
    --train-override physics=newton_mjwarp
    --train-override env.sim.physics.solver_cfg.njmax=288
    --train-override env.sim.physics.solver_cfg.nconmax=32
    --train-override env.refresh_zarr_dataset=false
)
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
    extra_args+=(--skip-pretrain --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}")
fi
printf -v extra_args_string '%q ' "${extra_args[@]}"

export TASK=Isaac-Imitation-G1-Latent-v0
export FRAME_CAP=3000000000
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
export WANDB_GROUP="${WANDB_GROUP:-sonic-default-l1-scale-3b}"
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
