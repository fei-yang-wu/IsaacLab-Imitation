#!/usr/bin/env bash
set -euo pipefail

# H100 VRAM/throughput ablation for the SONIC surface, explicitly targeted
# at Isaac-Imitation-G1-Latent-Sonic-v0 (NOT the Isaac-Imitation-G1-Latent-v0
# default, which reverted to the Strict/legacy-optimizer surface on
# 2026-07-21 after the SONIC release-optimizer contract underperformed it at
# matched scale -- see config/g1/__init__.py and rlopt_ipmd_cfg.py). Sweeps
# TRAIN_NUM_ENVS/ROLLOUT_STEPS to find the setting that best uses one H100's
# 80 GB VRAM without contact-solver overflow, each capped at 2B frames. Not a
# paper run: corrected-LAFAN1 fresh h25/z256 skill encoder + SONIC oracle
# low-level policy, one arm per Slurm job.
#
# njmax/nconmax REVISED (2026-07-21) after the first round: v1 (njmax=95 @
# 8192 envs) hit 7.4M contact-solver overflow events over ~9.5h and v2
# (njmax=143 @ 12288 envs) hit 59k; peak requested njmax was ~230-245 in
# BOTH arms despite the different env counts, and the one arm run at
# njmax=288 (the BONES-SEED job) saw zero overflow. njmax/nconmax are a
# per-step contact-complexity budget driven by the SONIC env's domain
# randomization/push events and early strict-from-scratch falling, NOT
# something that scales with num_envs — the original proportional-scaling
# assumption was wrong. All arms now use a fixed, inflated-for-headroom
# njmax=320/nconmax=40 (vs. the 288/32 that already measured zero overflow)
# regardless of env count. v3 (16384 envs) failed with a genuine CUDA OOM
# last round, unrelated to njmax; it is expected to OOM again since a higher
# njmax slightly increases memory further, but is retained here to confirm.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
FRAME_CAP=2000000000
MANIFEST_PATH="/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json"
DATASET_PATH="/data/lafan1_corrected_8e95d557/g1_hl_diffsr"
EXPECTED_MANIFEST_SHA256="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"
EXPECTED_NPZ_COUNT=40
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"

# ARM_NAME:NUM_ENVS:ROLLOUT_STEPS:NJMAX:NCONMAX
# v3/v4 both hit genuine CUDA OOM on the 2026-07-21 rerun even at fixed
# njmax=320/nconmax=40 (v3: 16384 envs too many; v4: 12288 envs x 24 rollout
# doubles the collector buffer vs. v2's 12288x12, which also overflows 80GB)
# -- both are real VRAM-ceiling results, not a solver misconfiguration.
# v5 is the code's own hardcoded baked-in default shape (scene.num_envs=4096,
# agent.collector.frames_per_batch=24 in rlopt_ipmd_cfg.py) run explicitly
# under the new SONIC-default + release-optimizer contract, per user
# hypothesis that this exact shape is a "magic number" worth re-confirming
# now that njmax is fixed at a validated-safe value instead of the
# under-provisioned defaults used the first time this shape was tried.
ARMS=(
    "v1_e8192_r12:8192:12:320:40"
    "v2_e12288_r12:12288:12:320:40"
    "v3_e16384_r12:16384:12:320:40"
    "v4_e12288_r24:12288:24:320:40"
    "v5_e4096_r24:4096:24:320:40"
)
# Restrict to a subset by setting ARM_FILTER to a space-separated list of arm
# names, e.g. ARM_FILTER="v5_e4096_r24" to submit only the new arm.
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
    IFS=':' read -r arm_name num_envs rollout_steps njmax nconmax <<< "${arm}"
    if [[ -n "${ARM_FILTER}" ]]; then
        case " ${ARM_FILTER} " in
            *" ${arm_name} "*) ;;
            *) continue ;;
        esac
    fi
    run_tag="sonic_latent_vram_ablation_${arm_name}_2b_seed${SEED}_20260720"
    pretrain_output_dir="logs/sonic_vram_ablation/${run_tag}/skill_encoder_h25_z256"

    if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && "${DRY_RUN}" != "TRUE" && "${DRY_RUN}" != "yes" && "${DRY_RUN}" != "YES" && "${DRY_RUN}" != "on" && "${DRY_RUN}" != "ON" ]]; then
        remote_pretrain_output="${REMOTE_PROJECT_ROOT}/${pretrain_output_dir}"
        if ssh -o BatchMode=yes -o ConnectTimeout=10 ice "test -e '${remote_pretrain_output}'"; then
            echo "[ERROR] Refusing to reuse existing ICE output: ${remote_pretrain_output}" >&2
            exit 2
        fi
    fi

    frames_per_batch=$((num_envs * rollout_steps))
    if [[ "${arm_name}" == "v5_e4096_r24" ]]; then
        # Match the code's own hardcoded default exactly (rlopt_ipmd_cfg.py:
        # self.loss.mini_batch_size = 4096 * 24 // 4 = 24576), not the other
        # arms' uniform frames_per_batch/8 heuristic, since this arm exists
        # specifically to re-test that exact baked-in default shape.
        minibatch_size=24576
    else
        minibatch_size=$((frames_per_batch / 8))
    fi

    extra_args=(
        --assert-kitless
        --pretrain-output-dir "${pretrain_output_dir}"
        --pretrain-override physics=newton_mjwarp
        --pretrain-override env.refresh_zarr_dataset=true
        --train-override physics=newton_mjwarp
        --train-override "env.sim.physics.solver_cfg.njmax=${njmax}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${nconmax}"
        --train-override env.refresh_zarr_dataset=false
    )
    printf -v extra_args_string '%q ' "${extra_args[@]}"

    echo "[INFO] Submitting arm ${arm_name}: envs=${num_envs} rollout=${rollout_steps} minibatch=${minibatch_size} njmax=${njmax} nconmax=${nconmax}"

    TASK=Isaac-Imitation-G1-Latent-Sonic-v0 \
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
    WANDB_PROJECT="${WANDB_PROJECT:-g1-sonic-latent-vram-ablation-ice}" \
    WANDB_GROUP="${WANDB_GROUP:-sonic-vram-h100-2b}" \
    EXP_NAME="${run_tag}" \
    CLUSTER_CONFIG=ice_runtime \
    CLUSTER_SLURM_TIME_LIMIT=15:59:00 \
    CLUSTER_SLURM_PARTITION=ice-gpu \
    CLUSTER_SLURM_QOS=coe-ice \
    CLUSTER_SLURM_GPU_GRES=gpu:h100:1 \
    CLUSTER_SLURM_CPUS_PER_TASK=16 \
    CLUSTER_SLURM_MEM=96G \
    CLUSTER_SLURM_JOB_NAME_PREFIX="sonic-vram-${arm_name}" \
    CLUSTER_G1_USD_PATH=repo \
    EXTRA_PIPELINE_ARGS="${extra_args_string}" \
    DRY_RUN="${DRY_RUN}" \
        "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
done

echo "[INFO] Submitted all requested SONIC latent VRAM ablation arms."
