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
# Select the tuned recipe by ENTRY POINT. `Isaac-Imitation-G1-v2` registers the
# base Sonic contract under `rlopt_ipmd_cfg_entry_point`, deliberately, so that
# earlier runs keep resolving what they resolved; the tuned recipe is a separate
# registered alternative. This launcher used to reconstruct it from a copied
# override list instead, which is how it shipped two 5B runs at the wrong
# rollout on 2026-08-03 -- the copy said 12, the class said 6, and nothing
# compared them. Copying a config is the bug; selecting it is not.
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"
NJMAX="${NJMAX:-320}"
NCONMAX="${NCONMAX:-40}"

# --- Geometry -----------------------------------------------------------------
# ROLLOUT_STEPS is NOT passed to the agent. It is declared here only to size the
# wall-clock segment (FRAMES_PER_BATCH below) and to name the run, and it must
# equal what the recipe actually resolves -- `check_gates` asserts that rather
# than trusting this comment.
#
# 24 comes from the base contract, which the tuned class inherits. The screen
# ranked 6 first, at 23 training minutes; that ranking is real but it is an
# early-progress ranking, and a short rollout recollects more often so it adapts
# faster out of the gate regardless of where it ends up. With gamma 0.97 and
# gae_lambda 0.95 the GAE horizon is 12.7 steps, and a length-n rollout captures
# only 1 - 0.9215^n of the advantage weight: 39% at 6, 63% at 12, 86% at 24. At
# a 5B-frame budget the unbiased advantage estimate is worth more than the early
# rate.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-12288}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-18432}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-5000000000}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"

# 62,406 fps measured at 100M (screen arm p4), at rollout 12. Kept a margin
# below it: finishing a segment early costs one extra submission, overrunning
# costs everything since the last save.
#
# At rollout 24 the optimizer work per frame is unchanged -- update density is
# `epochs / mini_batch_size` and does not involve the batch size -- and there are
# HALF as many iterations, so the fixed per-iteration cost (logging, LR
# adaptation, sync) halves. The real rate should therefore be at or above the
# rollout-12 figure; read it off segment 1 and raise this for segment 2 rather
# than trusting an estimate twice.
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
LOG_ROOT="${LOG_ROOT:-/data/tuned_5b}"
TRAIN_LOG_DIR="${LOG_ROOT}/${RUN_TAG}/rlopt_train"

# --- Termination protocol -----------------------------------------------------
# Unset is the instantaneous protocol the registered task id defines, which every
# recorded oracle-qualification and M3 number is stated against. A window changes
# only where the episode ends: the strict thresholds are inherited unchanged, so
# a run that sets this is comparable on MPJPE but NOT on episode length, return,
# or any per-minute rate -- a window inflates those exactly the way loosening a
# threshold does.
TERMINATION_WINDOW="${TERMINATION_WINDOW:-}"
TERMINATION_WINDOW_PROBE="${TERMINATION_WINDOW_PROBE:-0}"
# Comma-separated terms the window applies to; empty means every strict tracking
# term. Scoping to `foot_pos_xyz` targets the dominant failure without loosening
# the height and orientation terms.
TERMINATION_WINDOW_TERMS="${TERMINATION_WINDOW_TERMS:-}"

# Extra env-side Hydra overrides appended verbatim, for a campaign that needs a
# knob this launcher does not own (e.g. a new reward term's weight). Agent-side
# settings do NOT belong here -- they arrive via AGENT_ENTRY_POINT.
EXTRA_TUNED_OVERRIDES="${EXTRA_TUNED_OVERRIDES:-}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-tuned-5b}"
WANDB_TAGS="${WANDB_TAGS:-sr,det,v2,lafan1,tuned,5b}"
EXCLUDE_NODES="${EXCLUDE_NODES:-atl1-1-03-010-15-0,atl1-1-03-013-13-0}"

# --- The environment half of the recipe ----------------------------------------
# ONLY env-side settings belong here. Everything agent-side (KL rule, desired_kl,
# entropy, input normalization, activations, widths, gamma, rollout) now arrives
# via AGENT_ENTRY_POINT and must NOT be restated -- restating it is what let the
# launcher and the class disagree.
TUNED_OVERRIDES=(
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0   # rescales return; see header
    env.enable_termination_curriculum=true          # (code)
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
)

check_rollout_matches_recipe() {
    # The recipe owns the rollout; ROLLOUT_STEPS here only sizes the wall-clock
    # segment and names the run. If the two disagree the segment arithmetic is
    # wrong and the run is mislabelled, so this checks rather than assumes --
    # a stale copy of this number is exactly what shipped two 5B runs at the
    # wrong geometry on 2026-08-03.
    local cfg owners resolved
    cfg="${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_cfg.py"
    [ -r "${cfg}" ] || fail "cannot read ${cfg}"

    # Every class that assigns the rollout, with its value. The tuned recipe must
    # not be among them: it inherits, so the rollout has one definition.
    owners="$(awk '/^class /{cls=$2} /self\.collector\.frames_per_batch *=/{sub(/\(.*/,"",cls); print cls, $NF}' "${cfg}")"
    grep -q '^G1ImitationTunedRLOptIPMDConfig ' <<<"${owners}" \
        && fail "G1ImitationTunedRLOptIPMDConfig sets collector.frames_per_batch; it must inherit."
    [ "$(wc -l <<<"${owners}")" -eq 1 ] \
        || fail "expected exactly one rollout definition in ${cfg}, found:"$'\n'"${owners}"
    resolved="$(awk '{print $2}' <<<"${owners}")"
    [ "${resolved}" = "${ROLLOUT_STEPS}" ] \
        || fail "ROLLOUT_STEPS=${ROLLOUT_STEPS} but the recipe resolves ${resolved}."
    echo "[PASS] rollout ${ROLLOUT_STEPS} matches the recipe ($(awk '{print $1}' <<<"${owners}"))"
}

ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=10 ice "$@"; }

# The container sees the data under /data; the login node sees the same tree under
# REMOTE_DATA_ROOT. Deriving the remote paths from the container ones keeps the
# gate honest for any dataset instead of only the LAFAN1 tree it was written for.
remote_of() { printf '%s' "${REMOTE_DATA_ROOT}${1#/data}"; }

check_gates() {
    local sha n bytes manifest_remote data_remote
    manifest_remote="$(remote_of "${MANIFEST_PATH}")"
    data_remote="$(dirname "$(dirname "${manifest_remote}")")"
    sha="$(ssh_ice "sha256sum '${manifest_remote}'" | awk '{print $1}')"
    n="$(ssh_ice "find '${data_remote}' -type f -name '*.npz' | wc -l")"
    [[ "${sha}" == "${EXPECTED_MANIFEST_SHA256}" && "${n}" == "${EXPECTED_NPZ_COUNT}" ]] \
        || fail "dataset gate failed for ${manifest_remote}: sha=${sha} npz=${n}"
    echo "[PASS] manifest sha + NPZ count (${n}) for $(basename "${MANIFEST_PATH}")"
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
    if [[ -n "${TERMINATION_WINDOW}" || "${TERMINATION_WINDOW_PROBE}" == "1" ]]; then
        grep -q "termination_window" "${REPO_ROOT}/scripts/rlopt/train.py" \
            || fail "scripts/rlopt/train.py lacks --termination_window; the flag would be an unknown argument."
        echo "[PASS] termination-window flag present in the working tree"
    fi
}

check_termination_window() {
    # A window and the probe are mutually exclusive: the probe measures how long
    # violations last, which requires terminating on none of them.
    if [[ "${TERMINATION_WINDOW_PROBE}" == "1" && -n "${TERMINATION_WINDOW}" ]]; then
        fail "TERMINATION_WINDOW_PROBE=1 cannot be combined with TERMINATION_WINDOW=${TERMINATION_WINDOW}."
    fi
    if [[ -n "${TERMINATION_WINDOW}" ]]; then
        [[ "${TERMINATION_WINDOW}" =~ ^[0-9]+$ ]] && (( TERMINATION_WINDOW >= 1 )) \
            || fail "TERMINATION_WINDOW must be a positive integer; got '${TERMINATION_WINDOW}'."
    fi
}

check_rollout_matches_recipe
check_termination_window

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

window_args=()
if [[ -n "${TERMINATION_WINDOW}" ]]; then
    window_args=(--termination_window "${TERMINATION_WINDOW}")
    [[ -n "${TERMINATION_WINDOW_TERMS}" ]] \
        && window_args+=(--termination_window_terms "${TERMINATION_WINDOW_TERMS}")
fi
[[ "${TERMINATION_WINDOW_PROBE}" == "1" ]] && window_args=(--termination_window_probe)

extra_overrides=()
[[ -n "${EXTRA_TUNED_OVERRIDES}" ]] && read -r -a extra_overrides <<<"${EXTRA_TUNED_OVERRIDES}"

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
    --algo IPMD --agent "${AGENT_ENTRY_POINT}"
    --seed "${SEED}" --max_iterations "${max_iterations}"
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    "${checkpoint_args[@]}"
    "${window_args[@]}"
    physics=newton_mjwarp
    "env.sim.physics.solver_cfg.njmax=${NJMAX}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
    "env.data.manifest=${MANIFEST_PATH}"
    "env.data.cache_dir=${DATASET_PATH}"
    env.data.cache_refresh=false
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
    "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
    "agent.save_interval=${SAVE_INTERVAL}"
    agent.logger.backend=wandb agent.logger.video=false
    "agent.logger.project_name=${WANDB_PROJECT}"
    "agent.logger.group_name=${WANDB_GROUP}"
    "agent.logger.exp_name=${RUN_TAG}"
    "agent.logger.log_dir=${TRAIN_LOG_DIR}"
    "${TUNED_OVERRIDES[@]}"
    "${extra_overrides[@]}"
)

echo "[INFO] run_tag     : ${RUN_TAG}"
echo "[INFO] geometry    : ${TRAIN_NUM_ENVS} x ${ROLLOUT_STEPS} = ${FRAMES_PER_BATCH}/iter"
echo "[INFO] budget      : ${FRAME_CAP} cap; ${COMPLETED_FRAMES} done; this segment ${max_iterations} iters (~$((max_iterations * FRAMES_PER_BATCH)) frames)"
echo "[INFO] segment cap : ${SEGMENT_MAX_ITERATIONS} iters at ${SEGMENT_FPS} fps under a ${CLUSTER_SLURM_TIME_LIMIT} wall"
echo "[INFO] save every  : ${SAVE_INTERVAL} frames (bounds TIMEOUT loss)"
if [[ "${TERMINATION_WINDOW_PROBE}" == "1" ]]; then
    echo "[INFO] terminations: PROBE -- tracking terminations off, run lengths logged (diagnostic only)"
elif [[ -n "${TERMINATION_WINDOW}" ]]; then
    echo "[INFO] terminations: window ${TERMINATION_WINDOW} consecutive steps on ${TERMINATION_WINDOW_TERMS:-all strict terms} (thresholds unchanged)"
else
    echo "[INFO] terminations: instantaneous (the registered protocol)"
fi
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
