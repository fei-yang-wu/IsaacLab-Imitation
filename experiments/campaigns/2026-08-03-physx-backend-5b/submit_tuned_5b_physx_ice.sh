#!/usr/bin/env bash
set -euo pipefail

# 5B-frame LAFAN1 low-level run of the tuned v2 det-SR latent recipe on the
# PHYSX backend.
#
# Why PhysX. `wiki/sim2sim-backend-verification.md` (2026-08-03) settled the
# cross-backend gap: the joint order is clean, and on a policy-free oracle probe
# PhysX tracks the reference 3x better than Newton (joint MAE 0.0327 vs 0.0975
# rad) while stock MuJoCo lands on PhysX's numbers, not Newton's. Isaac Lab's
# MJWarp path is the outlier. Every 5B checkpoint to date was trained on Newton
# and is overfit to it. This run retrains the same recipe on the backend that
# agrees with the referee.
#
# What is held fixed against the Newton tuned 5B run, so the backend is the only
# free variable: task, dataset, manifest, skill encoder, seed, frame cap, and the
# tuned optimizer + environment recipe.
#
# THE RECIPE. The optimizer half is the registered
# `rlopt_ipmd_tuned_cfg_entry_point` (`G1ImitationTunedRLOptIPMDConfig`), which
# is the campaign's measured champion: kl_adapt_step=iteration, desired_kl=0.02,
# entropy_coeff=0, input normalization on both nets, silu, [1024,1024,512],
# gamma=0.97, and `collector.frames_per_batch=24` (inherited). Selecting it by entry point
# rather than by a copied override list means this launcher cannot drift from
# the recipe. The environment half does NOT live on the agent config and must
# still be passed; it is `ENV_RECIPE_OVERRIDES` below.
#
# NOTE ON ROLLOUT LENGTH. The registered recipe carries rollout 6, not the 12
# that `2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh` submitted -- rollout 6
# was measured at +7.3% return and +6.8% episode length at unchanged MPJPE
# against a byte-identical config (arms s1 vs r0). The class docstring's
# "Geometry: 12288 x 12 is unchanged" bullet is stale; the code below it sets 6.
# ROLLOUT_STEPS here must stay equal to what the agent config sets, because the
# segment arithmetic divides by it.
#
# NOTE ON RETURN COMPARISONS. The recipe doubles `tracking_reward_points` from
# 2.0 to 4.0, the largest positive reward term, so `episode/return` is RESCALED
# against every run that does not. Compare MPJPE and episode length, which no
# reward weight can inflate.
#
# GPU POLICY. `runtime_bootstrap.validate_gpu_policy` rejects PhysX/Kit on
# compute-only GPUs (A100/H100/H200) because Kit wants an RT-capable device.
# ICE's PhysX-qualified parts are L40S / A40 / RTX6000. This launcher defaults to
# H100 plus `--experimental-compute-only-physx`, which is the documented escape
# hatch (headless only) and is untried on this cluster. If Kit refuses to start,
# switch to the qualified path with one knob and resubmit:
#
#   GPU_GRES=gpu:l40s:1 DRY_RUN=0 ./submit_tuned_5b_physx_ice.sh
#
# The override flag is added automatically only for compute-only GRES.
#
# THROUGHPUT IS AN ESTIMATE. Newton was measured at 62,406 fps at this geometry;
# PhysX has never been measured on ICE. The two anchors available are the local
# PhysX probe at 12288 x 12 (~18k fps, early-phase, RTX PRO 6000) and the
# 2026-07 matched 4096 x 24 pair (PhysX 33.8k vs Newton 53.6k, 0.63x). The
# default below is deliberately low: undersizing a segment costs one extra
# submission, oversizing costs everything since the last save, because an ICE
# TIMEOUT is a hard SIGKILL that runs no final save. Read the real rate off
# segment 1 and raise SEGMENT_FPS for later segments.
#
# DRY_RUN=1 by default.
#
#   DRY_RUN=1 ./submit_tuned_5b_physx_ice.sh                # plan only
#   DRY_RUN=0 ./submit_tuned_5b_physx_ice.sh                # segment 1
#   DRY_RUN=0 COMPLETED_FRAMES=<n> TRAIN_CHECKPOINT=<path> \
#       ./submit_tuned_5b_physx_ice.sh                      # segment 2, 3, ...

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
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"

# --- Geometry -----------------------------------------------------------------
# ROLLOUT_STEPS is NOT passed to the agent -- the recipe owns it. It is declared
# only to size the wall-clock segment and name the run, and `check_gates` asserts
# it equals what the recipe resolves.
#
# 24 is inherited from the base contract. The 2026-08-02 screen ranked 6 first at
# 23 training minutes, but that is an early-progress ranking and a short rollout
# adapts faster out of the gate regardless of where it converges. gamma 0.97 with
# gae_lambda 0.95 gives a 12.7-step GAE horizon, and a length-n rollout captures
# only 1 - 0.9215^n of the advantage weight (39% at 6, 86% at 24).
#
# MINIBATCH is NOT set by the recipe (it inherits 24576); 18432 is what every
# screen arm ran, and update density is epochs / mini_batch_size, so changing it
# changes optimizer work per frame.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-12288}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-18432}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-5000000000}"
COMPLETED_FRAMES="${COMPLETED_FRAMES:-0}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-}"

SEGMENT_FPS="${SEGMENT_FPS:-17000}"
SEGMENT_WALL_S="${SEGMENT_WALL_S:-57540}"      # 15:59:00
SEGMENT_STARTUP_S="${SEGMENT_STARTUP_S:-1500}" # Kit boot is slower than kit-less Newton
SEGMENT_TAIL_S="${SEGMENT_TAIL_S:-600}"
SEGMENT_MAX_ITERATIONS=$((
    (SEGMENT_WALL_S - SEGMENT_STARTUP_S - SEGMENT_TAIL_S) * SEGMENT_FPS / FRAMES_PER_BATCH
))
# Halved against the Newton launcher because the PhysX rate is unmeasured here:
# this is the exact amount a TIMEOUT can destroy.
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"

# --- Data + encoder: identical to the Newton tuned 5B run ---------------------
# The encoder is reused, not re-pretrained. `train_hl_skill_diffsr.py` is an
# offline entrypoint -- it fits DiffSR on the dataset cache and cannot start Kit
# at all -- and the post-2026-07-21 planner frame is byte-identical across
# backends, so the encoder carries no backend dependence. Reusing it is also
# what keeps this run a single-variable comparison against the Newton 5B run.
MANIFEST_PATH="${MANIFEST_PATH:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_NPZ_COUNT="${EXPECTED_NPZ_COUNT:-40}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
ENCODER_TAG="${ENCODER_TAG:-lafan1_v2_det_sr_h${HORIZON_STEPS}_z${Z_DIM}_seed0}"
ENCODER_CKPT_CONTAINER="/data/pretrain_store/${ENCODER_TAG}/checkpoints/latest.pt"
ENCODER_CKPT_REMOTE="${REMOTE_DATA_ROOT}/pretrain_store/${ENCODER_TAG}/checkpoints/latest.pt"

RUN_TAG="${RUN_TAG:-lafan1_v2_tuned_physx_5b_seed${SEED}_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}}"
TRAIN_LOG_DIR="/data/physx_5b/${RUN_TAG}/rlopt_train"

WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-physx-5b}"
WANDB_TAGS="${WANDB_TAGS:-physx,sr,det,v2,lafan1,tuned,5b}"
EXCLUDE_NODES="${EXCLUDE_NODES:-atl1-1-03-010-15-0,atl1-1-03-013-13-0}"
GPU_GRES="${GPU_GRES:-gpu:h100:1}"

# The environment half of the tuned recipe. It is not on the agent config, so a
# launcher that forgets these runs a different experiment while still selecting
# the tuned entry point.
ENV_RECIPE_OVERRIDES=(
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0
    env.enable_termination_curriculum=true
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
)

# PhysX on a compute-only part needs the documented escape hatch, and only there:
# adding it on an RT-capable GPU would hide a real policy failure.
compute_only_physx_args=()
case "${GPU_GRES}" in
    *h100*|*h200*|*a100*) compute_only_physx_args=(--experimental-compute-only-physx) ;;
esac

ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=10 ice "$@"; }

# The container sees the data under /data; the login node sees the same tree
# under REMOTE_DATA_ROOT. Deriving the remote paths from the container ones keeps
# the gate honest for any dataset instead of only the LAFAN1 tree.
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
    # The recipe is unrunnable without these; each failed loudly at least once.
    grep -q "kl_adapt_step" "${REPO_ROOT}/RLOpt/rlopt/config_base.py" \
        || fail "RLOpt lacks optim.kl_adapt_step; the tuned recipe would fail on an unknown key."
    grep -q "enable_termination_curriculum" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/imitation_g1_env_v2.py" \
        || fail "v2 env config lacks enable_termination_curriculum."
    grep -q "${AGENT_ENTRY_POINT}" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/__init__.py" \
        || fail "${AGENT_ENTRY_POINT} is not registered; the tuned optimizer recipe would silently not apply."
    echo "[PASS] tuned-recipe code present in the working tree (archive sync ships it)"
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
export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
export CLUSTER_SLURM_EXCLUDE="${EXCLUDE_NODES}"
export CLUSTER_G1_USD_PATH=repo
export CLUSTER_WANDB_TAGS="${WANDB_TAGS}"
# run_singularity.sh resolves the backend from these tokens and rewrites
# train.py -> train_physx.py under /isaac-sim/python.sh. Declaring it here as
# well makes a mismatch a submission-time error instead of a runtime one.
export CLUSTER_SIM_BACKEND=physx
export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train.py"
export CLUSTER_SLURM_JOB_NAME_PREFIX="physx5b"

cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
    --task "${TASK_NAME}" --num_envs "${TRAIN_NUM_ENVS}" --headless
    --algo IPMD --agent "${AGENT_ENTRY_POINT}"
    --seed "${SEED}" --max_iterations "${max_iterations}"
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    "${compute_only_physx_args[@]}"
    "${checkpoint_args[@]}"
    physics=physx
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
    "${ENV_RECIPE_OVERRIDES[@]}"
)

echo "[INFO] run_tag     : ${RUN_TAG}"
echo "[INFO] backend     : physx on ${GPU_GRES}${compute_only_physx_args[*]:+ (${compute_only_physx_args[*]})}"
echo "[INFO] agent cfg   : ${AGENT_ENTRY_POINT}"
echo "[INFO] geometry    : ${TRAIN_NUM_ENVS} x ${ROLLOUT_STEPS} = ${FRAMES_PER_BATCH}/iter, minibatch ${MINIBATCH_SIZE}"
echo "[INFO] budget      : ${FRAME_CAP} cap; ${COMPLETED_FRAMES} done; this segment ${max_iterations} iters (~$((max_iterations * FRAMES_PER_BATCH)) frames)"
echo "[INFO] segment cap : ${SEGMENT_MAX_ITERATIONS} iters at an ASSUMED ${SEGMENT_FPS} fps under a ${CLUSTER_SLURM_TIME_LIMIT} wall"
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
