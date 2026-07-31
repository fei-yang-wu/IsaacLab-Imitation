#!/usr/bin/env bash
set -euo pipefail

# Train the official-window SONIC FSQ32 controller on corrected LAFAN1.
# Default MODE=print; scheduler mutation requires an explicit token.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[ERROR] Repository root not found." >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
TASK="Isaac-Imitation-G1-Latent-SonicOfficialFSQ-v0"
SEED="${SEED:-0}"
FSQ_LEVEL=32

fsq_levels="["
for ((idx = 0; idx < 64; idx++)); do
    (( idx == 0 )) || fsq_levels+=","
    fsq_levels+="${FSQ_LEVEL}"
done
fsq_levels+="]"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH / 4))}"
FRAME_CAP="${FRAME_CAP:-5000000000}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"
ASSUMED_FPS="${ASSUMED_FPS:-25000}"
SEGMENT_TRAIN_SECONDS="${SEGMENT_TRAIN_SECONDS:-50400}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"

MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

RUN_TAG="${RUN_TAG:-sonic_official_window_fsq64_l32_h10_hold1_reset_legacy_e4096_s24_seed${SEED}}"
LOG_DIR="${LOG_DIR:-/data/sonic_official_fsq_store/${RUN_TAG}/rlopt_train}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-sonic-official-fsq-ice}"
WANDB_GROUP="${WANDB_GROUP:-official32-h10-hold1-legacy-reset-seed${SEED}}"

remaining_frames=$((FRAME_CAP - COMPLETED_FRAMES))
(( remaining_frames > 0 )) || { echo "[INFO] Frame cap already credited."; exit 0; }
cap_iterations=$((remaining_frames / FRAMES_PER_BATCH))
wall_iterations=$((SEGMENT_TRAIN_SECONDS * ASSUMED_FPS / FRAMES_PER_BATCH))
MAX_ITERATIONS="${MAX_ITERATIONS:-$((cap_iterations < wall_iterations ? cap_iterations : wall_iterations))}"
(( MAX_ITERATIONS > 0 )) || { echo "[ERROR] Zero training iterations." >&2; exit 2; }

case "${MODE}" in
    print) ;;
    validate|submit)
        if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "sonic-official-fsq32" ]]; then
            echo "[ERROR] Submission requires CONFIRM_SUBMIT=sonic-official-fsq32." >&2
            exit 2
        fi
        ;;
    *) echo "[ERROR] MODE must be print, validate, or submit; got ${MODE}." >&2; exit 2 ;;
esac

local_manifest="${REPO_ROOT}/data/lafan1/manifests/g1_lafan1_manifest.json"
actual_local_sha="$(sha256sum "${local_manifest}" | awk '{print $1}')"
[[ "${actual_local_sha}" == "${EXPECTED_MANIFEST_SHA256}" ]] || {
    echo "[ERROR] Corrected LAFAN1 manifest hash mismatch: ${actual_local_sha}" >&2
    exit 2
}

if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    remote_manifest="${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json"
    remote_dataset="${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/g1_hl_diffsr"
    remote_log_dir="${REMOTE_DATA_ROOT}${LOG_DIR#/data}"
    read -r remote_sha remote_cache_ok remote_log_exists < <(
        ssh -o BatchMode=yes -o ConnectTimeout=15 ice bash -s -- \
            "${remote_manifest}" "${remote_dataset}" "${remote_log_dir}" <<'REMOTE_EOF'
set -euo pipefail
sha256sum "$1" | awk '{printf "%s ", $1}'
if [[ -d "$2" ]]; then printf "yes "; else printf "no "; fi
if [[ -e "$3" ]]; then echo yes; else echo no; fi
REMOTE_EOF
    )
    [[ "${remote_sha}" == "${EXPECTED_MANIFEST_SHA256}" ]] || {
        echo "[ERROR] ICE manifest hash mismatch: ${remote_sha}" >&2; exit 2;
    }
    [[ "${remote_cache_ok}" == "yes" ]] || {
        echo "[ERROR] ICE dataset cache is missing: ${remote_dataset}" >&2; exit 2;
    }
    if [[ -z "${TRAIN_CHECKPOINT}" && "${remote_log_exists}" == "yes" ]]; then
        echo "[ERROR] Refusing to overwrite existing initial-run log dir: ${LOG_DIR}" >&2
        exit 2
    fi
    echo "[INFO] ICE data gate passed; existing_log_dir=${remote_log_exists}."
fi

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h200:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-128G}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-sonic-fsq-l${FSQ_LEVEL}}"
export CLUSTER_G1_USD_PATH=repo

checkpoint_args=()
[[ -z "${TRAIN_CHECKPOINT}" ]] || checkpoint_args=(--checkpoint "${TRAIN_CHECKPOINT}")
cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
    --task "${TASK}"
    --num_envs "${TRAIN_NUM_ENVS}"
    --headless
    --algo IPMD
    --max_iterations "${MAX_ITERATIONS}"
    --match-sonic-release-overrides
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    "${checkpoint_args[@]}"
    physics=newton_mjwarp
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
    "env.lafan1_manifest_path=${MANIFEST_PATH}"
    "env.dataset_path=${DATASET_PATH}"
    env.refresh_zarr_dataset=false
    env.random_reset_step_min=0
    env.random_reset_step_max=200
    env.random_reset_full_trajectory=false
    env.adaptive_failure_reset_failure_rate_max_over_mean=50.0
    "agent.seed=${SEED}"
    "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
    agent.loss.epochs=5
    "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
    "agent.ipmd.expert_batch_size=${MINIBATCH_SIZE}"
    agent.ipmd.latent_steps_min=1
    agent.ipmd.latent_steps_max=1
    agent.ipmd.latent_learning.method=patch_vqvae
    agent.ipmd.latent_learning.quantizer=fsq
    "agent.ipmd.latent_learning.fsq_levels=${fsq_levels}"
    agent.ipmd.latent_learning.patch_past_steps=0
    agent.ipmd.latent_learning.patch_future_steps=9
    agent.ipmd.latent_learning.code_period=1
    agent.ipmd.latent_learning.posterior_command_period=1
    agent.ipmd.latent_learning.command_phase_mode=none
    agent.ipmd.latent_learning.train_posterior_through_policy=true
    agent.ipmd.latent_learning.recon_coeff=0.01
    agent.ipmd.latent_learning.action_recon_coeff=0.0
    "agent.save_interval=${SAVE_INTERVAL}"
    agent.logger.backend=wandb
    "agent.logger.project_name=${WANDB_PROJECT}"
    "agent.logger.group_name=${WANDB_GROUP}"
    "agent.logger.exp_name=${RUN_TAG}"
    "agent.logger.log_dir=${LOG_DIR}"
)

printf '[PLAN] level=%s task=%s frames=%s iterations=%s effective_segment_frames=%s\n' \
    "${FSQ_LEVEL}" "${TASK}" "${FRAME_CAP}" "${MAX_ITERATIONS}" \
    "$((MAX_ITERATIONS * FRAMES_PER_BATCH))"
printf '[PLAN] 10-frame window -> one 64-D FSQ command; code_period=1; envs=%s steps=%s minibatch=%s\n' \
    "${TRAIN_NUM_ENVS}" "${ROLLOUT_STEPS}" "${MINIBATCH_SIZE}"
printf '[PLAN] resets=[0,200], full_trajectory=false, failure_rate_max_over_mean=50\n'
printf '[CMD] '; printf '%q ' "${cmd[@]}"; printf '\n'

if [[ "${MODE}" == "submit" ]]; then
    "${cmd[@]}"
else
    echo "[INFO] MODE=${MODE}: nothing was submitted."
fi
