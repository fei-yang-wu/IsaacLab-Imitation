#!/usr/bin/env bash
set -euo pipefail

# 5B-frame LAFAN1 run of the v2 det-SR latent recipe on the TUNED optimizer
# configuration found by the 2026-08-02 hyperparameter screen (69 arms, W&B
# group `rlopt-hparam-search`). Same task, encoder, data and geometry as
# `2026-08-02-v2-command-interface`; what differs is the settings below.
#
# The configuration is arm `p4_tracking_points_2x` (W&B gptgtrqg), which tied
# best on the two weight-independent metrics at 100M: MPJPE 60.75 mm and 11.36
# episode-length-per-minute.
#
# ONE CAVEAT ON THAT ARM, because it will otherwise be misread. p4 doubles
# `tracking_reward_points` from 2.0 to 4.0, and that term is the largest positive
# component of the reward -- its Episode_Reward went 1.117 -> 1.994, close to the
# 2x the weight change implies. So p4's return, and any return-derived rate, is
# RESCALED relative to every other arm and to every historical run. Its genuine
# advantage is in MPJPE and episode length, which no reward weight can inflate.
# Do not compare this run's `episode/return` against the v2 baseline runs.
#
# At 100M the screen measured 62,406 fps at this geometry, so a 15:59 wall fits
# ~3.5B frames and 5B needs two segments. Segment sizing is computed below and
# capped; `save_interval` bounds what a TIMEOUT can destroy, because an ICE
# TIMEOUT is a hard SIGKILL that runs no final save.
#
# DRY_RUN=1 by default.
#
#   DRY_RUN=1 ./submit_tuned_5b_ice.sh                 # plan only
#   DRY_RUN=0 ./submit_tuned_5b_ice.sh                 # segment 1
#   DRY_RUN=0 COMPLETED_FRAMES=<n> TRAIN_CHECKPOINT=<path> ./submit_tuned_5b_ice.sh   # segment 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]; do
    [ "${REPO_ROOT}" = "/" ] && { echo "[ERROR] repo root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) is_dry_run=1 ;;
    0|false|FALSE|no|NO) is_dry_run=0 ;;
    *) echo "[ERROR] DRY_RUN must be boolean; got '${DRY_RUN}'." >&2; exit 2 ;;
esac
fail() { echo "[FATAL] $*" >&2; exit 1; }

SEED="${SEED:-0}"
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"

# --- Geometry: unchanged from the v2 campaign and from the screen -------------
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-12288}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-18432}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-5000000000}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"

# 62,406 fps measured on this exact configuration at 100M (screen arm p4). Kept
# a margin below it: finishing a segment early costs one extra submission,
# overrunning costs everything since the last save.
SEGMENT_FPS="${SEGMENT_FPS:-58000}"
SEGMENT_WALL_S="${SEGMENT_WALL_S:-57540}"      # 15:59:00
SEGMENT_STARTUP_S="${SEGMENT_STARTUP_S:-900}"
SEGMENT_TAIL_S="${SEGMENT_TAIL_S:-600}"
SEGMENT_MAX_ITERATIONS=$((
    (SEGMENT_WALL_S - SEGMENT_STARTUP_S - SEGMENT_TAIL_S) * SEGMENT_FPS / FRAMES_PER_BATCH
))
# A TIMEOUT is a hard SIGKILL with no final save, so this bounds the loss.
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"

# --- Data + encoder: identical to the screen and the v2 campaign --------------
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ENCODER_TAG="${ENCODER_TAG:-lafan1_v2_det_sr_h${HORIZON_STEPS}_z${Z_DIM}_seed0}"
ENCODER_CKPT_CONTAINER="/data/pretrain_store/${ENCODER_TAG}/checkpoints/latest.pt"
ENCODER_CKPT_REMOTE="${REMOTE_DATA_ROOT}/pretrain_store/${ENCODER_TAG}/checkpoints/latest.pt"

RUN_TAG="${RUN_TAG:-lafan1_v2_tuned_5b_seed${SEED}_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}}"
TRAIN_LOG_DIR="/data/tuned_5b/${RUN_TAG}/rlopt_train"

WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-tuned-5b}"
WANDB_TAGS="${WANDB_TAGS:-sr,det,v2,lafan1,tuned,5b}"
EXCLUDE_NODES="${EXCLUDE_NODES:-atl1-1-03-010-15-0,atl1-1-03-013-13-0}"

# --- The tuned configuration --------------------------------------------------
# Every entry below is a screen result, not a guess. The two marked (code)
# require changes carried on this branch.
TUNED_OVERRIDES=(
    agent.optim.kl_adapt_step=iteration        # (code) KL rule per iteration, not per minibatch
    agent.optim.desired_kl=0.02                # peaked; 0.04 measured worse
    agent.ppo.entropy_coeff=0.0                # re-confirmed on the final base
    agent.policy.normalize_input=true          # latent command already excluded by config
    agent.value_function.normalize_input=true
    agent.policy.activation_fn=silu
    agent.value_function.activation_fn=silu
    "agent.policy.num_cells=[1024,1024,512]"
    "agent.value_function.num_cells=[1024,1024,512]"
    agent.loss.gamma=0.97
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0   # rescales return; see header
    env.enable_termination_curriculum=true          # (code)
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
)

ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=10 ice "$@"; }

check_gates() {
    local sha n bytes
    sha="$(ssh_ice "sha256sum '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json'" | awk '{print $1}')"
    n="$(ssh_ice "find '${REMOTE_DATA_ROOT}/lafan1_corrected_8e95d557' -type f -name '*.npz' | wc -l")"
    [[ "${sha}" == "${EXPECTED_MANIFEST_SHA256}" && "${n}" == "${EXPECTED_NPZ_COUNT}" ]] \
        || fail "corrected-LAFAN1 gate failed: sha=${sha} npz=${n}"
    echo "[PASS] corrected-LAFAN1 manifest sha + NPZ count"
    bytes="$(ssh_ice "if [ -s '${ENCODER_CKPT_REMOTE}' ]; then stat -c %s '${ENCODER_CKPT_REMOTE}'; else echo 0; fi")"
    (( bytes > 1000000 )) || fail "encoder missing/truncated (${bytes} B): ${ENCODER_CKPT_REMOTE}"
    echo "[PASS] skill encoder present (${bytes} bytes)"
    # The tuned recipe is not runnable without these two local code changes.
    grep -q "kl_adapt_step" "${REPO_ROOT}/RLOpt/rlopt/config_base.py" \
        || fail "RLOpt lacks kl_adapt_step; the tuned recipe would fail on an unknown key."
    grep -q "enable_termination_curriculum" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/imitation_g1_env_v2.py" \
        || fail "v2 env config lacks enable_termination_curriculum."
    echo "[PASS] tuned-recipe code changes present in the working tree"
}

remaining=$((FRAME_CAP - COMPLETED_FRAMES))
(( remaining > 0 )) || { echo "[INFO] ${RUN_TAG} already at FRAME_CAP."; exit 0; }
max_iterations=$(( (remaining + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH ))
if (( max_iterations > SEGMENT_MAX_ITERATIONS )); then
    echo "[INFO] capping this segment at ${SEGMENT_MAX_ITERATIONS} iters to exit under the wall;"
    echo "[INFO] re-run with COMPLETED_FRAMES/TRAIN_CHECKPOINT for the next segment."
    max_iterations="${SEGMENT_MAX_ITERATIONS}"
fi

checkpoint_args=()
[[ -n "${TRAIN_CHECKPOINT}" ]] && checkpoint_args=(--checkpoint "${TRAIN_CHECKPOINT}")

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT=pace
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
export CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES}"
export CLUSTER_G1_USD_PATH=repo
export CLUSTER_WANDB_TAGS="${WANDB_TAGS}"
export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train.py"
export CLUSTER_SLURM_JOB_NAME_PREFIX="tuned5b"

cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
    --task "${TASK_NAME}" --num_envs "${TRAIN_NUM_ENVS}" --headless --assert-kitless
    --algo IPMD --seed "${SEED}" --max_iterations "${max_iterations}"
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    "${checkpoint_args[@]}"
    physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
    "env.lafan1_manifest_path=${MANIFEST_PATH}"
    "env.dataset_path=${DATASET_PATH}"
    env.refresh_zarr_dataset=false
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
    agent.ipmd.hl_skill_finetune_enabled=false
    agent.ipmd.hl_skill_pg_coeff=0.05
    agent.ipmd.hl_skill_anchor_coeff=0.01
    agent.ipmd.hl_skill_offline_diffsr_coeff=1.0
    agent.ipmd.hl_skill_lr=3e-05
    "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
    "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
    "agent.save_interval=${SAVE_INTERVAL}"
    agent.logger.backend=wandb agent.logger.video=false
    "agent.logger.project_name=${WANDB_PROJECT}"
    "agent.logger.group_name=${WANDB_GROUP}"
    "agent.logger.exp_name=${RUN_TAG}"
    "agent.logger.log_dir=${TRAIN_LOG_DIR}"
    "${TUNED_OVERRIDES[@]}"
)

echo "[INFO] run_tag     : ${RUN_TAG}"
echo "[INFO] geometry    : ${TRAIN_NUM_ENVS} x ${ROLLOUT_STEPS} = ${FRAMES_PER_BATCH}/iter"
echo "[INFO] budget      : ${FRAME_CAP} cap; ${COMPLETED_FRAMES} done; this segment ${max_iterations} iters (~$((max_iterations * FRAMES_PER_BATCH)) frames)"
echo "[INFO] segment cap : ${SEGMENT_MAX_ITERATIONS} iters at ${SEGMENT_FPS} fps under a ${CLUSTER_SLURM_TIME_LIMIT} wall"
echo "[INFO] save every  : ${SAVE_INTERVAL} frames (bounds TIMEOUT loss)"
echo "[INFO] encoder     : ${ENCODER_CKPT_CONTAINER}"
echo "[INFO] checkpoints : ${TRAIN_LOG_DIR}"
echo "[INFO] wandb       : ${WANDB_PROJECT} / ${WANDB_GROUP}"
echo

if [[ "${is_dry_run}" == "1" ]]; then
    echo "[INFO] DRY_RUN=1; skipping remote gates."
    printf "[CMD] "; printf "%q " "${cmd[@]}"; printf "\n\n"
    echo "[INFO] Nothing submitted. Re-run with DRY_RUN=0."
    exit 0
fi

check_gates
echo
printf "[CMD] "; printf "%q " "${cmd[@]}"; printf "\n\n"
"${cmd[@]}"
