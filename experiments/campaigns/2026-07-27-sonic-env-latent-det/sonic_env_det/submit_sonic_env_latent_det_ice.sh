#!/usr/bin/env bash
set -euo pipefail

# SONIC-environment screen for the continuous deterministic DiffSR latent.
#
# One arm. It changes exactly one axis against the 2026-07-22 Study B
# `deterministic` row: the environment. Task
# `Isaac-Imitation-G1-Latent-Sonic-NoHist-v0` is the SONIC release recipe --
# rewards (`base_5point_local_feet_acc`), adaptive strict terminations
# (`base_adaptive_strict_ori_foot_xyz`, no `base_too_low`), the termination
# threshold curriculum, `level0_4` domain randomization, SONIC actuators and
# robot preset, pelvis anchor, and full-trajectory adaptive-failure reset
# sampling with `failure_rate_max_over_mean=200` -- with this repo's
# single-frame observations instead of SONIC's 10-step proprioceptive
# histories, because the 2026-07-21 isolated history ablation found those
# histories buy little at our scale.
#
# Geometry is SONIC's own release training setup: 4096 environments x 24
# steps per environment per PPO batch (`gear_sonic/config/exp/manager/
# universal_token/all_modes/sonic_release.yaml`: `num_envs: 4096`,
# `algo.config.num_steps_per_env: 24`), with SONIC's `decimation: 4`,
# `sim_dt: 0.005`, and `episode_length_s: 10.0` already baked into the task.
#
# The latent encoder is NOT re-pretrained. This job reuses the exact
# `deterministic` skill-encoder checkpoint from the 2026-07-22 latent-learning
# ablation so the encoder is held fixed and the environment is the only thing
# that moves. Optimizer contract stays the local one (512/256/128 ELU, actor
# lr 1e-3), matching the Study B rows.
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
TASK="${TASK:-Isaac-Imitation-G1-Latent-Sonic-NoHist-v0}"
SEED="${SEED:-0}"

# SONIC release training geometry.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((TRAIN_NUM_ENVS * ROLLOUT_STEPS / 8))}"
ACTOR_LR="${ACTOR_LR:-1.0e-3}"
ACTOR_LR_CAP="${ACTOR_LR_CAP:-1.0e-3}"
CRITIC_LR="${CRITIC_LR:-1.0e-3}"

# Same 5B cap as the 2026-07-22 / 2026-07-26 studies so checkpoints line up at
# matched frame counts. ICE hard-caps GPU walltime at 16-18h, so a run needs
# continuation segments; MAX_ITERATIONS is the smaller of the remaining cap and
# what one segment can cover.
FRAME_CAP="${FRAME_CAP:-5000000000}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"
# Deliberately conservative. The 2026-07-26 H100 runs sustained ~77-83k FPS at
# 12288 envs x 12; 4096 envs is less parallel-efficient per step, so assume
# 45k FPS until this arm reports its own throughput.
ASSUMED_FPS="${ASSUMED_FPS:-45000}"
SEGMENT_TRAIN_SECONDS="${SEGMENT_TRAIN_SECONDS:-50400}"

# Corrected LAFAN1, identical to the Study B `deterministic` row.
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"

# The frozen encoder from the 2026-07-22 Study B `deterministic` arm.
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-logs/latent_ablation/lafan1_diffsr_deterministic_continuous_h10_z256_seed0/skill_encoder/checkpoints/latest.pt}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"

REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

RUN_TAG="${RUN_TAG:-sonic_env_lafan1_diffsr_deterministic_e4096_s24_seed${SEED}}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-sonic-env-latent-det-ice}"
WANDB_GROUP="${WANDB_GROUP:-sonic-env-det-h10-seed0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"

GPU_TYPE="${GPU_TYPE:-h100}"
PARTITION="${PARTITION:-coe-gpu}"
# atl1-1-03-010-15-0 reported "No devices were found" on 2026-07-26 while Slurm
# still advertised it as healthy, so it keeps accepting and killing jobs.
EXCLUDE_NODES="${EXCLUDE_NODES:-atl1-1-03-010-15-0}"

case "${MODE}" in
    print) ;;
    validate|submit)
        if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "sonic-env-latent-det" ]]; then
            echo "[ERROR] Submission requires CONFIRM_SUBMIT=sonic-env-latent-det." >&2
            exit 2
        fi
        ;;
    *) echo "[ERROR] MODE must be print, validate, or submit; got ${MODE}." >&2; exit 2 ;;
esac

# Local manifest hash gate: the corrected-LAFAN1 tree is the only acceptable
# source for a row that will be compared against the Study B table.
local_manifest="${REPO_ROOT}/data/lafan1/manifests/g1_lafan1_manifest.json"
actual_local_sha="$(sha256sum "${local_manifest}" | awk '{print $1}')"
if [[ "${actual_local_sha}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "[ERROR] Local corrected LAFAN1 manifest hash mismatch: ${actual_local_sha}" >&2
    exit 2
fi

if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    # The cluster-side manifest, the shared dataset cache, and the frozen
    # encoder must all already exist. This arm never rebuilds the cache: seven
    # groupvq arms share it and a concurrent refresh truncated it on
    # 2026-07-26.
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

remaining_frames=$((FRAME_CAP - COMPLETED_FRAMES))
if (( remaining_frames <= 0 )); then
    echo "[INFO] FRAME_CAP=${FRAME_CAP} already credited by COMPLETED_FRAMES=${COMPLETED_FRAMES}."
    exit 0
fi

FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
cap_iterations=$((remaining_frames / FRAMES_PER_BATCH))
wall_iterations=$((SEGMENT_TRAIN_SECONDS * ASSUMED_FPS / FRAMES_PER_BATCH))
MAX_ITERATIONS=$((cap_iterations < wall_iterations ? cap_iterations : wall_iterations))

extra=(
    --assert-kitless
    --skip-pretrain
    --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}"
    --phase-mode sin_cos
    --latent-hold-steps 10
    --train-override physics=newton_mjwarp
    --train-override "agent.ipmd.actor_learning_rate=${ACTOR_LR}"
    --train-override "agent.ipmd.critic_learning_rate=${CRITIC_LR}"
    --train-override "agent.optim.max_lr=${ACTOR_LR_CAP}"
    --train-override env.sim.physics.solver_cfg.njmax=320
    --train-override env.sim.physics.solver_cfg.nconmax=40
    # MUST stay false: this cache is shared with the running groupvq arms.
    --train-override env.refresh_zarr_dataset=false
)
if [[ -n "${TRAIN_CHECKPOINT}" ]]; then
    extra+=(--train-checkpoint "${TRAIN_CHECKPOINT}")
fi

printf -v extra_string '%q ' "${extra[@]}"
cmd=(env
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
    EXP_NAME="${RUN_TAG}"
    CLUSTER_CONFIG=ice_runtime
    CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
    CLUSTER_SLURM_PARTITION="${PARTITION}"
    CLUSTER_SLURM_QOS=coe-ice
    CLUSTER_SLURM_GPU_GRES="gpu:${GPU_TYPE}:1"
    CLUSTER_SLURM_CPUS_PER_TASK=16
    CLUSTER_SLURM_MEM=128G
    CLUSTER_SLURM_JOB_NAME_PREFIX="sonic-env-det"
    CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES}"
    CLUSTER_G1_USD_PATH=repo
    EXTRA_PIPELINE_ARGS="${extra_string}"
    DRY_RUN=0
    "${REPO_ROOT}/experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_hl_skill_pipeline_pace_2b.sh"
)

echo "[PLAN] task=${TASK} envs=${TRAIN_NUM_ENVS} steps=${ROLLOUT_STEPS} minibatch=${MINIBATCH_SIZE}"
echo "[PLAN] frames_per_batch=${FRAMES_PER_BATCH} max_iterations=${MAX_ITERATIONS} segment_frames=$((MAX_ITERATIONS * FRAMES_PER_BATCH)) frame_cap=${FRAME_CAP}"
echo "[PLAN] gpu=${GPU_TYPE} partition=${PARTITION} encoder=${PRETRAINED_CHECKPOINT}"
printf '[CMD] '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${MODE}" == "submit" ]]; then
    "${cmd[@]}"
else
    echo "[INFO] MODE=${MODE}: nothing was submitted."
fi
