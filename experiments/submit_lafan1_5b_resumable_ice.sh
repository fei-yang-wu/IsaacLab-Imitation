#!/usr/bin/env bash
set -euo pipefail

# Resumable corrected-LAFAN1 low-level job at a 5B-frame cap, chained across
# multiple Slurm segments (ICE QoS caps every 1-GPU job at 16h via
# MaxTRESMins gres/gpu=960). Sibling of
# experiments/submit_bones_seed_sonic_5b_resumable_ice.sh; see that script's
# header for the full cross-segment plumbing rationale (per-submission
# isaaclab_<timestamp>/ workspace dirs, exp_name-filtered resume scan, and
# /data-bind staging of pretrain + resume checkpoints).
#
# TASK is Isaac-Imitation-G1-Latent-v0 (the strict/legacy-optimizer surface,
# the default since the 2026-07-21 revert; the SONIC release surface is
# opt-in only via Isaac-Imitation-G1-Latent-Sonic-v0).
#
# Scale is the VRAM-ablation-validated scaled config (2026-07-21): 12288 envs
# x 12 rollout steps, minibatch = frames_per_batch / 8 = 18432, and
# njmax=320/nconmax=40. Runs on ice-gpu H100 80 GB: the 48 GB ice-bw-gpu
# Blackwell cards OOM at this scale (job 5525245; the stack itself worked).
#
# Submitted only with post-joint-fix code (fix/migration, merged as PR #24 /
# 900c66c). The pre-fix LAFAN1 sanity runs (jobs 5524387/5524390) left
# invalidated Newton checkpoints in the same central log trees; the exp_name
# filter in resume detection excludes them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-1}"
FRAME_CAP=5000000000
TRAIN_NUM_ENVS=12288
ROLLOUT_STEPS=12
MINIBATCH_SIZE=18432
NJMAX=320
NCONMAX=40
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
HORIZON_STEPS="${HORIZON_STEPS:-10}"
RUN_TAG="${RUN_TAG:-lafan1_strict_h${HORIZON_STEPS}_z256_5b_seed${SEED}_20260721_jointfix_nocur_e12288_r12_nj320_nc40}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}_oracle_low_level}"
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-Latent-v0}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

# DURABILITY (2026-07-22): every output this run must keep -- the skill encoder
# and all low-level checkpoints -- is written DIRECTLY to the /data bind, which
# is the only host path a job's container can write that outlives the job.
# Slurm TIMEOUT is a hard SIGKILL: it kills the job step before
# run_singularity.sh's sync_project_logs_back can copy node-local $TMPDIR out,
# and the epilog then wipes /tmp. That destroyed all three 16h segments'
# output on 2026-07-22 (jobs 5525663/5525664/5525687, ~48 GPU-hours, zero
# retained checkpoints, encoders included). A save is only ~12 MB, so direct
# /data writes are cheap and make durability independent of how the job dies.
PRETRAIN_OUTPUT_DIR="/data/pretrain_store/${RUN_TAG}"
PRETRAIN_CKPT_CONTAINER="${PRETRAIN_OUTPUT_DIR}/checkpoints/latest.pt"
CKPT_DIR_CONTAINER="/data/ckpt_store/${RUN_TAG}/rlopt_train"
CKPT_DIR_HOST="${REMOTE_DATA_ROOT}/ckpt_store/${RUN_TAG}/rlopt_train"
PRETRAIN_STORE_HOST="${REMOTE_DATA_ROOT}/pretrain_store/${RUN_TAG}"

# Cap each segment so training ENDS before the wall instead of being SIGKILLed:
# a clean exit gets a final save and a normal shutdown. ~78-83k fps observed at
# this scale; 70k is deliberately conservative, and ~1.5h of the 16h is left
# for container start, Zarr load, env construction, and encoder pretrain.
SEGMENT_TRAIN_SECONDS="${SEGMENT_TRAIN_SECONDS:-52200}"
ASSUMED_FPS="${ASSUMED_FPS:-70000}"
WALLTIME_ITERATIONS=$(( SEGMENT_TRAIN_SECONDS * ASSUMED_FPS / FRAMES_PER_BATCH ))

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *)
        echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2
        exit 2
        ;;
esac

if [[ "${is_dry_run}" == "0" ]]; then
    actual_remote_sha="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_remote_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE corrected-LAFAN1 data gate failed: sha=${actual_remote_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
fi

# --- Resume detection (durable /data checkpoints; see DURABILITY above) ---
# Because CKPT_DIR is RUN_TAG-scoped and lives on /data, resume detection is a
# plain scan of this run's own directory -- no cross-submission exp_name
# filtering, and no staging step, because the checkpoints are already on a path
# the next segment's container can read.
STATE_FILE="${REMOTE_DATA_ROOT}/ckpt_store/${RUN_TAG}/resume_state.tsv"
cumulative_frames=0
latest_checkpoint=""
pretrain_ready=0
if [[ "${is_dry_run}" == "0" ]]; then
    resume_state="$(ssh -o BatchMode=yes -o ConnectTimeout=10 ice bash -s -- \
        "${CKPT_DIR_HOST}" "${STATE_FILE}" "${PRETRAIN_STORE_HOST}" <<'REMOTE_EOF'
set -uo pipefail
ckpt_dir="$1"
state_file="$2"
pretrain_store="$3"

pretrain_ready=0
[[ -f "${pretrain_store}/checkpoints/latest.pt" ]] && pretrain_ready=1

cumulative=0
last_counted=""
if [[ -f "${state_file}" ]]; then
    IFS=$'\t' read -r cumulative last_counted < "${state_file}"
fi

# RLOpt restarts model_step_<N> numbering per segment (load_model restores
# weights + optimizer state but NOT the frame counter), and each segment writes
# its own timestamped subdirectory, so credit each segment's own max once.
latest="$(find "${ckpt_dir}" -name 'model_step_*.pt' -printf '%T@\t%p\n' 2>/dev/null \
    | sort -n -k1,1 | tail -1 | cut -f2-)"

if [[ -n "${latest}" && "${latest}" != "${last_counted}" ]]; then
    segment_dir="$(dirname "${latest}")"
    segment_frames="$(find "${segment_dir}" -name 'model_step_*.pt' \
        | sed -E 's#.*model_step_([0-9]+)\.pt#\1#' | sort -n | tail -1)"
    cumulative=$((cumulative + segment_frames))
    mkdir -p "$(dirname "${state_file}")"
    printf '%s\t%s\n' "${cumulative}" "${latest}" > "${state_file}"
fi

printf '%s\t%s\t%s\n' "${cumulative}" "${pretrain_ready}" "${latest}"
REMOTE_EOF
    )"
    cumulative_frames="$(printf '%s' "${resume_state}" | cut -f1)"
    pretrain_ready="$(printf '%s' "${resume_state}" | cut -f2)"
    latest_checkpoint="$(printf '%s' "${resume_state}" | cut -f3-)"
    cumulative_frames="${cumulative_frames:-0}"
    pretrain_ready="${pretrain_ready:-0}"
fi

if [[ -n "${latest_checkpoint}" ]]; then
    if [[ "${pretrain_ready}" != "1" ]]; then
        echo "[ERROR] Found a training checkpoint but no staged pretrain for ${RUN_TAG}; refusing an inconsistent resume." >&2
        exit 2
    fi
    remaining_frames=$((FRAME_CAP - cumulative_frames))
    if (( remaining_frames <= 0 )); then
        echo "[INFO] ${RUN_TAG} already reached FRAME_CAP=${FRAME_CAP} (cumulative ${cumulative_frames} frames). Not submitting."
        exit 0
    fi
    remaining_iterations=$(( (remaining_frames + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    max_iterations=$(( remaining_iterations < WALLTIME_ITERATIONS ? remaining_iterations : WALLTIME_ITERATIONS ))
    # The checkpoint already lives on /data; translate the host path the resume
    # scan returned into its container-visible counterpart.
    resume_ckpt_container="/data${latest_checkpoint#${REMOTE_DATA_ROOT}}"
    echo "[INFO] Resuming ${RUN_TAG} from ${latest_checkpoint} (${cumulative_frames}/${FRAME_CAP} cumulative frames done; ${remaining_iterations} remaining, running ${max_iterations} this segment)."
    extra_args=(
        --assert-kitless
        --skip-pretrain
        --pretrained-checkpoint "${PRETRAIN_CKPT_CONTAINER}"
        --train-checkpoint "${resume_ckpt_container}"
        --train-override "physics=newton_mjwarp"
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override "env.refresh_zarr_dataset=false"
        --train-override "agent.logger.log_dir=${CKPT_DIR_CONTAINER}"
    )
elif [[ "${pretrain_ready}" == "1" ]]; then
    remaining_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    max_iterations=$(( remaining_iterations < WALLTIME_ITERATIONS ? remaining_iterations : WALLTIME_ITERATIONS ))
    echo "[INFO] No training checkpoint for ${RUN_TAG}, but a durable pretrain exists; starting low-level training from scratch with that skill encoder (${max_iterations} iterations this segment)."
    extra_args=(
        --assert-kitless
        --skip-pretrain
        --pretrained-checkpoint "${PRETRAIN_CKPT_CONTAINER}"
        --train-override physics=newton_mjwarp
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override env.refresh_zarr_dataset=false
        --train-override "agent.logger.log_dir=${CKPT_DIR_CONTAINER}"
    )
else
    remaining_iterations=$(( (FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    max_iterations=$(( remaining_iterations < WALLTIME_ITERATIONS ? remaining_iterations : WALLTIME_ITERATIONS ))
    echo "[INFO] No checkpoint or durable pretrain found for ${RUN_TAG}; submitting a fresh first segment with pretrain (${max_iterations} iterations this segment)."
    extra_args=(
        --assert-kitless
        --pretrain-output-dir "${PRETRAIN_OUTPUT_DIR}"
        --train-override "agent.logger.log_dir=${CKPT_DIR_CONTAINER}"
        # Previous-setting pretrain contract: 0.9/0.1 trajectory split is the
        # train_hl_skill_diffsr.py default (eval fraction 0.1, not overridden);
        # groups/categories restate the pipeline defaults explicitly; and
        # --gumbel-hard restores the diffsr-side default (true) that the
        # pipeline's own default=False would otherwise override.
        --categorical-groups 64
        --categorical-categories 128
        --gumbel-hard
        --pretrain-override physics=newton_mjwarp
        --pretrain-override env.refresh_zarr_dataset=true
        --train-override physics=newton_mjwarp
        --train-override "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        --train-override "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        --train-override env.refresh_zarr_dataset=false
    )
fi
printf -v extra_args_string '%q ' "${extra_args[@]}"

export TASK="${TASK_NAME}"
export FRAME_CAP
export TRAIN_NUM_ENVS
export ROLLOUT_STEPS
export MINIBATCH_SIZE
export MAX_ITERATIONS="${max_iterations}"
export PRETRAIN_NUM_ENVS=16
# 50k updates: the pipeline's own default and the previous full pretrain
# setting.
export PRETRAIN_UPDATES=50000
export PRETRAIN_BATCH_SIZE=8192
export HORIZON_STEPS
export TRAIN_VIDEO=0
export SAVE_INTERVAL=100000000
export MANIFEST_PATH
export DATASET_PATH
export WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1-strict}"
export WANDB_GROUP="${WANDB_GROUP:-scaled-e12288-5b-resumable-jointfix}"
export EXP_NAME
export CLUSTER_CONFIG=ice_runtime
# H100 80 GB is the default: the scaled 12288-env config OOMs on the 48 GB
# ice-bw-gpu Blackwell cards (see header). QoS caps 1-GPU jobs at 16h.
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS=coe-ice
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK=16
export CLUSTER_SLURM_MEM=96G
export CLUSTER_SLURM_JOB_NAME_PREFIX=lafan1-5b-resume
export CLUSTER_G1_USD_PATH=repo
export EXTRA_PIPELINE_ARGS="${extra_args_string}"
export DRY_RUN

exec "${REPO_ROOT}/experiments/submit_hl_skill_pipeline_pace_2b.sh"
