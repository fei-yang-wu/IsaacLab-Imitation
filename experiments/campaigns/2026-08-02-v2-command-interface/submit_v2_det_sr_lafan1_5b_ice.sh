#!/usr/bin/env bash
set -euo pipefail

# 5B-frame LAFAN1 run of the deterministic-SR continuous latent recipe on the
# v2 command interface (`Isaac-Imitation-G1-v2`).
#
# Two stages, submitted separately because the ICE wall is ~16 h and a combined
# job that dies mid-pretrain loses both:
#
#   STAGE=pretrain   train a fresh deterministic SR skill encoder (z=256, h=10)
#                    on corrected LAFAN1, into /data/pretrain_store/<tag>.
#   STAGE=lowlevel   IPMD low-level training conditioned on that frozen encoder,
#                    segmented to fit under the wall and resumable to 5B.
#
# This is the first campaign on the declared command interface (2026-08-02): the
# published latent width is `env.command_interface.actor.dim`, not the removed
# `env.latent_command_dim`, and the actor kind is derived rather than declared
# twice -- the training entry point binds the agent to the environment's
# interface, so an env/agent command mismatch can no longer be expressed.
#
# Geometry is the established scaled-up config (12288 x 12), NOT the 4096 x 24
# SONIC-release geometry: 147,456 frames per iteration.
#
# DRY_RUN=1 by default. Nothing reaches the cluster without DRY_RUN=0.
#
# Usage:
#   DRY_RUN=1 STAGE=pretrain ./submit_v2_det_sr_lafan1_5b_ice.sh
#   DRY_RUN=0 STAGE=pretrain ./submit_v2_det_sr_lafan1_5b_ice.sh
#   # once the encoder exists:
#   DRY_RUN=0 STAGE=lowlevel ./submit_v2_det_sr_lafan1_5b_ice.sh
#   # next segment, after the first one hits the wall:
#   DRY_RUN=0 STAGE=lowlevel COMPLETED_FRAMES=4260000000 \
#       TRAIN_CHECKPOINT=/data/v2_command_interface/<tag>/rlopt_train/.../latest.pt \
#       ./submit_v2_det_sr_lafan1_5b_ice.sh

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

DRY_RUN="${DRY_RUN:-1}"
STAGE="${STAGE:-pretrain}"
SEED="${SEED:-0}"

case "${STAGE}" in
    pretrain|lowlevel) ;;
    *) echo "[ERROR] STAGE must be pretrain or lowlevel; got '${STAGE}'." >&2; exit 2 ;;
esac

case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON) is_dry_run=1 ;;
    0|false|FALSE|no|NO|off|OFF) is_dry_run=0 ;;
    *) echo "[ERROR] DRY_RUN must be a boolean; got '${DRY_RUN}'." >&2; exit 2 ;;
esac

# --- Task + latent recipe ----------------------------------------------------
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"       # encoder window; binds the checkpoint
Z_DIM="${Z_DIM:-256}"
LATENT_MODE="${LATENT_MODE:-deterministic}"  # det SR, continuous latent
LATENT_COMMAND_DIM=$((Z_DIM + 2))            # + sin/cos phase
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}" # one latent per encoder window

# --- Geometry: the scaled-up config ------------------------------------------
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-12288}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-18432}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"
FRAME_CAP="${FRAME_CAP:-5000000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"

# ICE TIMEOUT is a hard SIGKILL: the final save never runs and everything since
# the last save_interval boundary is lost. Size the segment to exit cleanly
# under the wall. 76k fps is the measured latent-arm rate at this geometry
# (job 5546958); the local 4096-env number does NOT transfer, and the explicit
# arms' 80k+ overran the wall when it was used here.
SEGMENT_FPS="${SEGMENT_FPS:-76000}"
SEGMENT_WALL_S="${SEGMENT_WALL_S:-57540}"      # 15:59:00
SEGMENT_STARTUP_S="${SEGMENT_STARTUP_S:-900}"  # Isaac boot + data load
SEGMENT_TAIL_S="${SEGMENT_TAIL_S:-600}"        # final save + log sync
SEGMENT_MAX_ITERATIONS=$((
    (SEGMENT_WALL_S - SEGMENT_STARTUP_S - SEGMENT_TAIL_S) * SEGMENT_FPS / FRAMES_PER_BATCH
))

# --- Corrected LAFAN1 (paper-facing data) ------------------------------------
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

RUN_TAG="${RUN_TAG:-lafan1_v2_det_sr_h${HORIZON_STEPS}_z${Z_DIM}_5b_seed${SEED}_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}}"
ENCODER_DIR_CONTAINER="/data/pretrain_store/${RUN_TAG}"
ENCODER_CKPT_CONTAINER="${ENCODER_CKPT_CONTAINER:-${ENCODER_DIR_CONTAINER}/checkpoints/latest.pt}"
ENCODER_CKPT_REMOTE="${REMOTE_DATA_ROOT}/pretrain_store/${RUN_TAG}/checkpoints/latest.pt"
TRAIN_LOG_DIR="/data/v2_command_interface/${RUN_TAG}/rlopt_train"

# --- Pretrain sizing ---------------------------------------------------------
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"

# --- W&B ---------------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-v2}"
# Tags cross the container boundary through CLUSTER_WANDB_TAGS (wandb reads
# WANDB_TAGS itself); see docker/cluster/run_singularity.sh.
WANDB_TAGS="${WANDB_TAGS:-sr,det,v2,lafan1,${STAGE}}"

EXCLUDE_NODES="${EXCLUDE_NODES:-atl1-1-03-010-15-0}"

ssh_ice() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 ice "$@"
}

check_data_gate() {
    local actual_sha remote_npz_count
    actual_sha="$(ssh_ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    remote_npz_count="$(ssh_ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    if [[ "${actual_sha}" != "${EXPECTED_MANIFEST_SHA256}" || "${remote_npz_count}" != "${EXPECTED_NPZ_COUNT}" ]]; then
        echo "[ERROR] ICE corrected-LAFAN1 data gate failed: sha=${actual_sha}, npz=${remote_npz_count}." >&2
        exit 2
    fi
    echo "[PASS] corrected-LAFAN1 manifest sha and NPZ count match the frozen protocol."
}

check_encoder_gate() {
    local bytes
    bytes="$(ssh_ice "if [ -s '${ENCODER_CKPT_REMOTE}' ]; then stat -c %s '${ENCODER_CKPT_REMOTE}'; else echo 0; fi")"
    if (( bytes < 1000000 )); then
        echo "[ERROR] Skill encoder missing or truncated (${bytes} bytes):" >&2
        echo "[ERROR]   ${ENCODER_CKPT_REMOTE}" >&2
        echo "[ERROR] Run STAGE=pretrain first and wait for it to finish." >&2
        exit 2
    fi
    echo "[PASS] skill encoder present (${bytes} bytes): ${ENCODER_CKPT_REMOTE}"
}

export_cluster_env() {
    export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
    export CLUSTER_SLURM_SUBMIT_SCRIPT="${CLUSTER_SLURM_SUBMIT_SCRIPT:-pace}"
    export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    export CLUSTER_G1_MANIFEST_REFRESH_POLICY="${CLUSTER_G1_MANIFEST_REFRESH_POLICY:-auto}"
    export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
    export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
    export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
    export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
    export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
    export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
    export CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES}"
    export CLUSTER_GIT_SYNC_FIRST="${CLUSTER_GIT_SYNC_FIRST:-0}"
    export CLUSTER_G1_USD_PATH=repo
    export CLUSTER_WANDB_TAGS="${WANDB_TAGS}"
}

submit_pretrain() {
    export_cluster_env
    export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train_hl_skill_diffsr.py"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="v2-detsr-pretrain"

    local cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}"
        --num_envs "${PRETRAIN_NUM_ENVS}"
        --headless
        --assert-kitless
        --seed "${SEED}"
        --output_dir "${ENCODER_DIR_CONTAINER}"
        --horizon_steps "${HORIZON_STEPS}"
        --encoder_window_mode intermediate
        --z_dim "${Z_DIM}"
        --latent_mode "${LATENT_MODE}"
        --batch_size "${PRETRAIN_BATCH_SIZE}"
        --num_updates "${PRETRAIN_UPDATES}"
        --reconstruction_eval
        --logger_backend wandb
        --wandb_project "${WANDB_PROJECT}"
        --wandb_group "${WANDB_GROUP}"
        --wandb_run_name "${RUN_TAG}_pretrain"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        physics=newton_mjwarp
        "env.lafan1_manifest_path=${MANIFEST_PATH}"
        "env.dataset_path=${DATASET_PATH}"
        # MUST stay false: the /data cache is shared with every other LAFAN1
        # arm and a refresh=true job rebuilds it underneath them.
        env.refresh_zarr_dataset=false
    )

    echo
    echo "[INFO] STAGE=pretrain  task='${TASK_NAME}'  run_tag='${RUN_TAG}'"
    echo "[INFO]   latent: ${LATENT_MODE} SR, z=${Z_DIM}, horizon=${HORIZON_STEPS} (published width ${LATENT_COMMAND_DIM})"
    echo "[INFO]   encoder -> ${ENCODER_CKPT_CONTAINER}"
    echo "[INFO]   updates=${PRETRAIN_UPDATES} batch=${PRETRAIN_BATCH_SIZE} envs=${PRETRAIN_NUM_ENVS}"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    if [[ "${is_dry_run}" == "1" ]]; then
        echo "[INFO] DRY_RUN=${DRY_RUN}; not contacting the cluster."
        return 0
    fi
    "${cmd[@]}"
}

submit_lowlevel() {
    local remaining_frames=$((FRAME_CAP - COMPLETED_FRAMES))
    if (( remaining_frames <= 0 )); then
        echo "[INFO] ${RUN_TAG} already reached FRAME_CAP=${FRAME_CAP} (COMPLETED_FRAMES=${COMPLETED_FRAMES}). Not submitting."
        return 0
    fi
    local max_iterations=$(( (remaining_frames + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
    if (( max_iterations > SEGMENT_MAX_ITERATIONS )); then
        echo "[INFO] Capping this segment at ${SEGMENT_MAX_ITERATIONS} iterations so it exits before the ${CLUSTER_SLURM_TIME_LIMIT:-15:59:00} wall; re-run with an updated COMPLETED_FRAMES/TRAIN_CHECKPOINT for the next segment."
        max_iterations="${SEGMENT_MAX_ITERATIONS}"
    fi

    local checkpoint_args=()
    if [[ -n "${TRAIN_CHECKPOINT}" ]]; then
        checkpoint_args=(--checkpoint "${TRAIN_CHECKPOINT}")
        echo "[INFO] Resuming from ${TRAIN_CHECKPOINT} (${COMPLETED_FRAMES}/${FRAME_CAP} frames done)."
    fi

    export_cluster_env
    export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train.py"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="v2-detsr-lowlevel"

    local cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}"
        --num_envs "${TRAIN_NUM_ENVS}"
        --headless
        --assert-kitless
        --algo IPMD
        --seed "${SEED}"
        --max_iterations "${max_iterations}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "${checkpoint_args[@]}"
        physics=newton_mjwarp
        "env.sim.physics.solver_cfg.njmax=${NJMAX}"
        "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
        "env.lafan1_manifest_path=${MANIFEST_PATH}"
        "env.dataset_path=${DATASET_PATH}"
        env.refresh_zarr_dataset=false
        # The published latent width is declared on the actor channel of the
        # command interface; the agent side must match it. The training entry
        # point binds the agent to this interface, so the actor kind
        # (latent) and every command input key are derived from here.
        "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
        "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}"
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT_CONTAINER}"
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
        agent.ipmd.hl_skill_command_mode=z
        "agent.ipmd.latent_steps_min=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_steps_max=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_learning.code_period=${LATENT_HOLD_STEPS}"
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        # Frozen encoder: this run tests the v2 interface, not representation
        # learning.
        agent.ipmd.hl_skill_finetune_enabled=false
        agent.ipmd.hl_skill_pg_coeff=0.05
        agent.ipmd.hl_skill_anchor_coeff=0.01
        agent.ipmd.hl_skill_offline_diffsr_coeff=1.0
        agent.ipmd.hl_skill_lr=3e-05
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb
        agent.logger.video=false
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${RUN_TAG}"
        # Checkpoints go to the /data bind, never the per-submission workspace:
        # a TIMEOUT wipes node-local output before any log sync runs.
        "agent.logger.log_dir=${TRAIN_LOG_DIR}"
    )

    echo
    echo "[INFO] STAGE=lowlevel  task='${TASK_NAME}'  run_tag='${RUN_TAG}'"
    echo "[INFO]   max_iterations=${max_iterations} (~$((max_iterations * FRAMES_PER_BATCH)) frames of the ${FRAME_CAP} cap)"
    echo "[INFO]   encoder=${ENCODER_CKPT_CONTAINER}"
    echo "[INFO]   checkpoints -> ${TRAIN_LOG_DIR}"
    printf "[CMD] "
    printf "%q " "${cmd[@]}"
    printf "\n"

    if [[ "${is_dry_run}" == "1" ]]; then
        echo "[INFO] DRY_RUN=${DRY_RUN}; not contacting the cluster."
        return 0
    fi
    "${cmd[@]}"
}

echo "[INFO] stage=${STAGE} seed=${SEED} task=${TASK_NAME}"
echo "[INFO] geometry: ${TRAIN_NUM_ENVS} envs x ${ROLLOUT_STEPS} steps = ${FRAMES_PER_BATCH} frames/iter"
echo "[INFO] budget: ${FRAME_CAP} frames; segment cap ${SEGMENT_MAX_ITERATIONS} iters (~$((SEGMENT_MAX_ITERATIONS * FRAMES_PER_BATCH)) frames per 16h segment)"
echo "[INFO] wandb: project=${WANDB_PROJECT} group=${WANDB_GROUP} tags=${WANDB_TAGS}"

if [[ "${is_dry_run}" == "0" ]]; then
    check_data_gate
    if [[ "${STAGE}" == "lowlevel" ]]; then
        check_encoder_gate
    fi
else
    echo "[INFO] DRY_RUN=${DRY_RUN}; skipping remote data/encoder gates."
fi

if [[ "${STAGE}" == "pretrain" ]]; then
    submit_pretrain
else
    submit_lowlevel
fi

if [[ "${is_dry_run}" == "1" ]]; then
    echo
    echo "[INFO] Nothing was submitted. Re-run with DRY_RUN=0 to submit."
fi
