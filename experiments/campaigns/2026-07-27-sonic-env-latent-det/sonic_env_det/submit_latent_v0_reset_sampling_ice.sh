#!/usr/bin/env bash
set -euo pipefail

# Reset-sampling screen on Isaac-Imitation-G1-Latent-v0.
#
# The SONIC-environment arm (ICE 5541139) moved ~6 axes at once against the
# 2026-07-22 Study B `deterministic` row, so its 5x episode-length deficit is
# unattributed. This screen isolates the axis that most directly explains
# dying on `foot_pos_xyz` at 60% while essentially never reaching `time_out`:
# the reset distribution.
#
# Two arms, both on the *unmodified* Latent-v0 surface
# (`ImitationG1LatentStrictEnvCfg`: pelvis anchor, strict-from-scratch
# adaptive terminations, no curriculum, mimic actuators, no level0_4
# randomization, legacy rewards):
#
#   legacy   -- reset starts in [0, 200], `failure_rate_max_over_mean=50`
#               (the Study B `deterministic` protocol, unchanged)
#   fulltraj -- SONIC's sampler: full-trajectory starts,
#               `failure_rate_max_over_mean=200`
#
# `legacy` is not redundant with the completed Study B run: that row trained at
# 16,384 x 12, and this screen runs at SONIC's 4,096 x 24. It is the
# matched-geometry Latent-v0 companion the campaign README already flags as
# missing, so it also makes the SONIC arms interpretable for the first time.
#
# The encoder is NOT re-pretrained; both arms reuse the exact frozen
# `deterministic` skill encoder from the 2026-07-22 ablation, as the SONIC
# arms do.
#
# Default MODE=print. Nothing is submitted without CONFIRM_SUBMIT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    if [ "${REPO_ROOT}" = "/" ]; then
        echo "[ERROR] Could not locate the repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
# Space-separated subset of: legacy fulltraj
ARMS="${ARMS:-legacy fulltraj}"
TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
SEED="${SEED:-0}"

# SONIC release training geometry, matched to the SONIC-environment arms so
# this screen doubles as their missing geometry control.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((TRAIN_NUM_ENVS * ROLLOUT_STEPS / 8))}"
ACTOR_LR="${ACTOR_LR:-1.0e-3}"
CRITIC_LR="${CRITIC_LR:-1.0e-3}"

# A screen, not a convergence run. The base/SONIC divergence is already
# decisive by 500M frames (episode length 296 vs 51); 1B is one clean segment
# and leaves headroom. 4096 x 24 = 98,304 frames per batch.
FRAME_CAP="${FRAME_CAP:-1000000000}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS="${MAX_ITERATIONS:-$((FRAME_CAP / FRAMES_PER_BATCH))}"

# Corrected LAFAN1, identical to the Study B `deterministic` row and the
# SONIC-environment arms.
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-logs/latent_ablation/lafan1_diffsr_deterministic_continuous_h10_z256_seed0/skill_encoder/checkpoints/latest.pt}"

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-sonic-env-latent-det-ice}"
WANDB_GROUP="${WANDB_GROUP:-reset-sampling-screen-seed0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"

GPU_TYPE="${GPU_TYPE:-h100}"
PARTITION="${PARTITION:-coe-gpu}"
# atl1-1-03-010-15-0 reported "No devices were found" on 2026-07-26 while Slurm
# still advertised it as healthy, so it keeps accepting and killing jobs.
EXCLUDE_NODES="${EXCLUDE_NODES:-atl1-1-03-010-15-0}"

case "${MODE}" in
    print) ;;
    validate|submit)
        if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "latent-v0-reset-sampling" ]]; then
            echo "[ERROR] Submission requires CONFIRM_SUBMIT=latent-v0-reset-sampling." >&2
            exit 2
        fi
        ;;
    *) echo "[ERROR] MODE must be print, validate, or submit; got ${MODE}." >&2; exit 2 ;;
esac

for arm in ${ARMS}; do
    case "${arm}" in
        legacy|fulltraj) ;;
        *) echo "[ERROR] Unknown arm '${arm}'; expected legacy or fulltraj." >&2; exit 2 ;;
    esac
done

local_manifest="${REPO_ROOT}/data/lafan1/manifests/g1_lafan1_manifest.json"
actual_local_sha="$(sha256sum "${local_manifest}" | awk '{print $1}')"
if [[ "${actual_local_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Local corrected LAFAN1 manifest hash mismatch: ${actual_local_sha}" >&2
    exit 2
fi

if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    # Same gate as the SONIC arm: the cluster-side manifest, the shared dataset
    # cache, and the frozen encoder must already exist. This screen never
    # rebuilds the cache; concurrent arms share it and a refresh truncated it
    # on 2026-07-26.
    remote_manifest="${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json"
    remote_dataset="${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/g1_hl_diffsr"
    read -r remote_sha remote_cache_ok remote_ckpt_bytes < <(
        ssh -o BatchMode=yes -o ConnectTimeout=15 ice bash -s -- \
            "${remote_manifest}" "${remote_dataset}" \
            "${REMOTE_PROJECT_ROOT}/${PRETRAINED_CHECKPOINT}" <<'REMOTE_EOF'
set -euo pipefail
sha256sum "$1" | awk '{printf "%s ", $1}'
if [ -d "$2" ]; then printf "yes "; else printf "no "; fi
if [ -s "$3" ]; then stat -c %s "$3"; else echo 0; fi
REMOTE_EOF
    )
    if [[ "${remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
        echo "[ERROR] ICE manifest hash mismatch: ${remote_sha}" >&2
        exit 2
    fi
    if [[ "${remote_cache_ok}" != "yes" ]]; then
        echo "[ERROR] ICE dataset cache missing: ${remote_dataset}" >&2
        exit 2
    fi
    if (( remote_ckpt_bytes < 1000000 )); then
        echo "[ERROR] Frozen deterministic encoder missing or truncated (${remote_ckpt_bytes} bytes):" >&2
        echo "[ERROR]   ${REMOTE_PROJECT_ROOT}/${PRETRAINED_CHECKPOINT}" >&2
        exit 2
    fi
    echo "[INFO] ICE gate passed: manifest sha=${remote_sha}, cache present, encoder ${remote_ckpt_bytes} bytes."
fi

submit_arm() {
    local arm="$1"
    local run_tag="${RUN_TAG_PREFIX:-latent_v0}_e4096_s24_reset_${arm}_seed${SEED}"

    local extra=(
        --assert-kitless
        --skip-pretrain
        --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
        --phase-mode sin_cos
        --latent-hold-steps 10
        --train-override physics=newton_mjwarp
        --train-override "agent.ipmd.actor_learning_rate=${ACTOR_LR}"
        --train-override "agent.ipmd.critic_learning_rate=${CRITIC_LR}"
        --train-override "agent.optim.max_lr=${ACTOR_LR}"
        --train-override env.sim.physics.solver_cfg.njmax=320
        --train-override env.sim.physics.solver_cfg.nconmax=40
        # MUST stay false: this cache is shared with the running groupvq arms.
        --train-override env.refresh_zarr_dataset=false
    )
    if [[ "${arm}" == "fulltraj" ]]; then
        # The only axis that moves. Everything else is the Study B protocol.
        extra+=(
            --train-override env.random_reset_full_trajectory=true
            --train-override env.random_reset_step_min=0
            --train-override env.random_reset_step_max=0
            --train-override env.adaptive_failure_reset_failure_rate_max_over_mean=200.0
        )
    fi

    local extra_string
    printf -v extra_string '%q ' "${extra[@]}"
    local cmd=(env
        TASK="${TASK}"
        SEED="${SEED}"
        FRAME_CAP="${FRAME_CAP}"
        MAX_ITERATIONS="${MAX_ITERATIONS}"
        TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS}"
        ROLLOUT_STEPS="${ROLLOUT_STEPS}"
        MINIBATCH_SIZE="${MINIBATCH_SIZE}"
        HORIZON_STEPS=10
        TRAIN_VIDEO=0
        SAVE_INTERVAL="${SAVE_INTERVAL}"
        MANIFEST_PATH="${MANIFEST_PATH}"
        DATASET_PATH="${DATASET_PATH}"
        WANDB_PROJECT="${WANDB_PROJECT}"
        WANDB_GROUP="${WANDB_GROUP}"
        EXP_NAME="${run_tag}"
        CLUSTER_CONFIG=ice_runtime
        CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
        CLUSTER_SLURM_PARTITION="${PARTITION}"
        CLUSTER_SLURM_QOS=coe-ice
        CLUSTER_SLURM_GPU_GRES="gpu:${GPU_TYPE}:1"
        CLUSTER_SLURM_CPUS_PER_TASK=16
        CLUSTER_SLURM_MEM=128G
        CLUSTER_SLURM_JOB_NAME_PREFIX="reset-${arm}"
        CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES}"
        CLUSTER_G1_USD_PATH=repo
        EXTRA_PIPELINE_ARGS="${extra_string}"
        DRY_RUN=0
        "${REPO_ROOT}/experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_hl_skill_pipeline_pace_2b.sh"
    )

    echo "[PLAN] arm=${arm} run_tag=${run_tag}"
    if [[ "${arm}" == "fulltraj" ]]; then
        echo "[PLAN]   resets: full trajectory, failure_rate_max_over_mean=200"
    else
        echo "[PLAN]   resets: [0, 200] starts, failure_rate_max_over_mean=50 (default)"
    fi
    printf '[CMD] '
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [[ "${MODE}" == "submit" ]]; then
        "${cmd[@]}"
    fi
}

echo "[PLAN] task=${TASK} envs=${TRAIN_NUM_ENVS} steps=${ROLLOUT_STEPS} minibatch=${MINIBATCH_SIZE}"
echo "[PLAN] frames_per_batch=${FRAMES_PER_BATCH} max_iterations=${MAX_ITERATIONS} frame_cap=${FRAME_CAP}"
echo "[PLAN] gpu=${GPU_TYPE} partition=${PARTITION} encoder=${PRETRAINED_CHECKPOINT}"
echo "[PLAN] arms=${ARMS}"

for arm in ${ARMS}; do
    submit_arm "${arm}"
done

if [[ "${MODE}" != "submit" ]]; then
    echo "[INFO] MODE=${MODE}: nothing was submitted."
fi
