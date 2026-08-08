#!/usr/bin/env bash
set -euo pipefail

# BONES-SEED 129k ablation screen on ICE H200.
#
# WHY THIS EXISTS. The local 10B run looked plateaued: training MPJPE-L sat at
# 41-44 mm from 0.35B to 4.0B and `reference_finished` fell monotonically. It
# was not plateaued. Scored on the fixed protocol (frame 0, no randomization,
# MODE actions, 1024 envs, 500 steps) the same checkpoints improve on every
# axis -- 25.8 -> 23.9 -> 23.4 mm MPJPE, 0.606 -> 0.668 -> 0.692 success,
# 222.6 -> 241.7 survival steps. The flat training curves were an artifact of
# the `sonic` adaptive sampler hardening the task at about the rate the policy
# improved.
#
# So this screen is not repairing a broken recipe. It is trying to beat a
# working one whose returns are decelerating: 0.35B->2.00B bought 1.9 mm over
# 1.65B frames, 2.00B->4.03B bought 0.5 mm over 2.03B.
#
# HOW TO READ IT. Every arm is one delta from `control` (see arms.sh), which
# reproduces the local run exactly. Arms are scored at the same frame count on
# the same fixed protocol, against the control AT THAT COUNT -- never against
# the 4B number, and never against a training-time curve, which this campaign
# has already shown to be uninterpretable under adaptive resets.
#
# METRICS. Report mpjpe_l AND mpjpe_g plus anchor_pos_err. The 2026-08-02
# rollout screen concluded "unchanged MPJPE" from a column that resolves to
# mpjpe_l only; global tracking was never measured, and it is the axis the
# rollout24 arm exists to move.
#
# DATA. The reference arrays, not the Zarr and not a replay buffer. 49.4 GB
# holding exactly what the two derived caches consume. The job READS them into
# host RAM (`env.data.reference_arrays_resident=true`) rather than mapping them:
# CLUSTER_DATA_DIR is Lustre, and mapping defers reads to per-step page faults,
# which are random small reads. Measured 2026-08-05: ~48 fps mapped with three
# jobs cold-starting on a node, ~71,000 fps resident. The Zarr (196 GiB, ~5.3M
# files) and the 95 GiB replay are not on ICE and are not needed.
#
# GPU. H200, no flag, either backend. Headless PhysX on Hopper was settled
# 2026-08-03; if you find text claiming Kit needs an RT-capable device, it is
# stale. The ice-gpu default is a QUEUEING choice, not a capability one: when
# ice-gpu has no idle nodes, CLUSTER_SLURM_PARTITION=coe-gpu is the same
# gpu:h200:8 hardware and often has capacity.
#
#   DRY_RUN=1 ./submit_bones129k_ablation_ice.sh
#   DRY_RUN=0 ./submit_bones129k_ablation_ice.sh
#   DRY_RUN=0 ARMS="physx rollout24" ./submit_bones129k_ablation_ice.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -f "${REPO_ROOT}/scripts/rlopt/train.py" ]]; do
    [[ "${REPO_ROOT}" == "/" ]] && { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }
# shellcheck source=arms.sh
source "${SCRIPT_DIR}/arms.sh"

DRY_RUN="${DRY_RUN:-1}"
ARMS="${ARMS:-control ${ABLATION_ALL_ARM_NAMES[*]}}"
SEED="${SEED:-0}"
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"

# Container-visible paths. CLUSTER_DATA_DIR binds the ICE data root at /data.
REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
ENCODER_CKPT="${ENCODER_CKPT:-/data/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/checkpoints/latest.pt}"
EXPECTED_ENCODER_SHA256="${EXPECTED_ENCODER_SHA256:-d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7}"
EXPECTED_MOTIONS="${EXPECTED_MOTIONS:-129785}"
EXPECTED_TRANSITIONS="${EXPECTED_TRANSITIONS:-47491234}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
remote_of() { printf '%s' "${REMOTE_DATA_ROOT}${1#/data}"; }

# Geometry. Env count is held fixed across arms: the 2026-08-02 screen showed
# environment COUNT is not the lever (20480 and 24576 do not beat 12288) and
# that rollout was what mattered, so rollout is the only arm that moves it.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS_DEFAULT="${ROLLOUT_STEPS_DEFAULT:-6}"

# RUN TO THE WALL, ON PURPOSE. Both ice-gpu and coe-gpu cap at 16:00:00, so the
# wall -- not this cap -- is what ends an arm. FRAME_CAP is set past anything
# reachable (Newton at ~100k fps for 15:59 is ~5.7B; PhysX at ~0.6x is ~3.4B) so
# no arm stops early with time left on the clock.
#
# TIMEOUT is the intended terminator and is safe HERE specifically because
# `agent.logger.log_dir` is under /data, which binds to persistent scratch. A
# Slurm TIMEOUT is a hard SIGKILL that runs no final save, so anything written
# to node-local storage would be lost; on /data every checkpoint survives and
# the loss is bounded by SAVE_INTERVAL (~8 min of frames at 50M / 100k fps).
# Do not "fix" this by pointing log_dir at $TMPDIR.
FRAME_CAP="${FRAME_CAP:-8000000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

HORIZON_STEPS=10
Z_DIM=256
LATENT_HOLD_STEPS=10
LATENT_COMMAND_DIM=$((Z_DIM + 2))

RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

# The environment half of the recipe, which does NOT live on the agent config.
# Identical to the local 10B run so `control` reproduces it.
ENV_RECIPE_OVERRIDES=(
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0
    env.enable_termination_curriculum=true
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
    env.command_interface.reference.selection=sonic
    env.sim.physics.solver_cfg.njmax=289
    env.sim.physics.solver_cfg.nconmax=200
)

GPU_GRES="${GPU_GRES:-gpu:h200:1}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-bones129k-ablation}"

# -- gates ----------------------------------------------------------------- #

ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=20 ice "$@"; }

check_gates() {
    local arrays_remote encoder_remote sha n rows
    arrays_remote="$(remote_of "${REF_ARRAYS}")"
    encoder_remote="$(remote_of "${ENCODER_CKPT}")"

    # The sidecar is the authority on the arrays' identity, and it carries the
    # trajectory table without which the arrays are unloadable.
    rows="$(ssh_ice "python3 -c \"
import json
d=json.load(open('${arrays_remote}/reference_arrays_manifest.json'))
t=d['traj_info']; k=d['key']
print(len(t['ordered_traj_list']), t['written'], k['source']['persist_id'], k['anchor_body'], len(k['body_names']))
\"" 2>/dev/null)" || fail "reference arrays or sidecar missing at ${arrays_remote}"
    read -r n_motions n_trans pid anchor n_bodies <<<"${rows}"
    [[ "${n_motions}" == "${EXPECTED_MOTIONS}" ]] || fail "motions ${n_motions} != ${EXPECTED_MOTIONS}"
    [[ "${n_trans}" == "${EXPECTED_TRANSITIONS}" ]] || fail "transitions ${n_trans} != ${EXPECTED_TRANSITIONS}"
    [[ "${pid}" == "${PERSIST_ID}" ]] || fail "persist_id ${pid} != ${PERSIST_ID}"
    [[ "${n_bodies}" == "${#RUNTIME_BODY_NAMES[@]}" ]] \
        || fail "arrays hold ${n_bodies} bodies, this recipe tracks ${#RUNTIME_BODY_NAMES[@]}"
    echo "[PASS] reference arrays: ${n_motions} motions, ${n_trans} transitions, anchor=${anchor}, ${n_bodies} bodies"

    # No .incomplete: hf download leaves those behind on a reaped transfer, and
    # a short array would surface as a shape error hours into a job.
    n="$(ssh_ice "find '${arrays_remote}' -name '*.incomplete' | wc -l")"
    [[ "${n}" == "0" ]] || fail "${n} unfinished downloads under ${arrays_remote}"

    sha="$(ssh_ice "sha256sum '${encoder_remote}' 2>/dev/null | cut -d' ' -f1")"
    [[ "${sha}" == "${EXPECTED_ENCODER_SHA256}" ]] \
        || fail "encoder SHA mismatch at ${encoder_remote}: got '${sha}'"
    echo "[PASS] encoder pinned by hash: ${EXPECTED_ENCODER_SHA256:0:16}…"

    grep -q "adaptive_pre_failure_window" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/commands/reference.py" \
        || fail "reference.py lacks adaptive_pre_failure_window; the reset arms would silently not apply."
    grep -q "reference_arrays_dir" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/motion_data.py" \
        || fail "motion_data.py lacks reference_arrays_dir; the archive sync would ship a tree that cannot read the arrays."
    echo "[PASS] arm-bearing code present in the working tree"
}

# -- submit ---------------------------------------------------------------- #

submit_arm() {
    local arm="$1" rollout extra run_tag frames_per_batch max_iterations minibatch backend
    rollout="${ABLATION_ARM_ROLLOUT[${arm}]:-${ROLLOUT_STEPS_DEFAULT}}"
    extra="${ABLATION_ARM_OVERRIDES[${arm}]:-}"
    [[ "${arm}" == "control" || -n "${extra}" || -n "${ABLATION_ARM_ROLLOUT[${arm}]:-}" ]] \
        || fail "unknown arm '${arm}'"

    frames_per_batch=$((TRAIN_NUM_ENVS * rollout))
    max_iterations=$(( (FRAME_CAP + frames_per_batch - 1) / frames_per_batch ))
    # Keep optimizer work per frame constant across rollout arms: update density
    # is epochs/mini_batch_size, so a 4x larger batch needs a 4x larger
    # minibatch or rollout24 silently also becomes an update-density arm.
    minibatch=$((frames_per_batch * 3 / 4))
    backend=physx; [[ "${extra}" == *"physics=physx"* ]] || backend=newton_mjwarp
    # No frame count in the tag: the wall ends these, so the number would be a
    # guess that the directory then asserts as fact.
    run_tag="bones129k_ablation_${arm}_seed${SEED}_e${TRAIN_NUM_ENVS}_r${rollout}"

    export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
    export CLUSTER_SLURM_SUBMIT_SCRIPT=pace
    export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
    export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
    export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
    export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
    export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
    # The 44.8 GiB runtime cache is a private row-packed resident allocation;
    # 160G leaves it and the simulator explicit headroom.
    export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-160G}"
    export CLUSTER_G1_USD_PATH=repo
    # STAGE THE ARRAYS TO NODE-LOCAL DISK. CLUSTER_DATA_DIR is Lustre, which is
    # built for large sequential I/O; a memory-mapped reference set does random
    # per-step row gathers, its worst case. Measured 2026-08-05 without staging:
    # ~48 fps with three jobs on a node, ~1,914 fps alone, against ~100,000 fps
    # off local NVMe. The container still sees /data/<subdir>, so no override
    # changes -- only the bind moves.
    export CLUSTER_SIM_BACKEND="${backend%%_*}"
    export CLUSTER_PYTHON_EXECUTABLE="scripts/rlopt/train.py"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="abl129k"
    export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,ablation,${arm},${backend}"

    local cmd=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK_NAME}" --num_envs "${TRAIN_NUM_ENVS}" --headless
        --algo IPMD --agent "${AGENT_ENTRY_POINT}"
        --seed "${SEED}" --max_iterations "${max_iterations}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        "physics=${backend}"
        env.data.manifest=null
        "env.data.reference_arrays_dir=${REF_ARRAYS}"
        "env.data.persist_id=${PERSIST_ID}"
        env.data.reference_arrays_warm_workers=16
        # READ THE ARRAYS INTO RAM, DO NOT MAP THEM. CLUSTER_DATA_DIR is Lustre.
        # Mapping defers reads to per-step page faults, which on Lustre are
        # random small reads -- its worst case. Measured 2026-08-05: ~48 fps with
        # three jobs cold-starting on one node, against ~100,000 fps locally off
        # NVMe. Reading up front is one sequential pass, which Lustre is good at,
        # and afterwards no filesystem is in the loop. Needs ~50 GB of the job's
        # memory; CLUSTER_SLURM_MEM below covers it and the load refuses rather
        # than inviting the OOM killer.
        env.data.reference_arrays_resident=true
        env.data.runtime_cache_device=cpu
        # Exact protocol: stage the single next-frame gather while physics runs.
        env.data.reference_prefetch_mode=next
        env.data.macro_cache_device=cuda:0
        "${BODY_NAMES_OVERRIDE}"
        "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
        "agent.collector.frames_per_batch=${rollout}"
        "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}"
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT}"
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
        agent.ipmd.hl_skill_command_mode=z
        "agent.ipmd.latent_steps_min=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_steps_max=${LATENT_HOLD_STEPS}"
        "agent.ipmd.latent_learning.code_period=${LATENT_HOLD_STEPS}"
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        agent.ipmd.hl_skill_finetune_enabled=false
        "agent.loss.mini_batch_size=${minibatch}"
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb agent.logger.video=false
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${run_tag}"
        "agent.logger.log_dir=/data/bones129k_ablation/${run_tag}/rlopt_train"
        "${ENV_RECIPE_OVERRIDES[@]}"
    )
    # Arm deltas go LAST so an arm always wins over the shared recipe. Hydra
    # takes the rightmost assignment, which is what makes each arm one line.
    [[ -n "${extra}" ]] && read -r -a extra_arr <<<"${extra}" && cmd+=("${extra_arr[@]}")

    echo "[ARM ] ${arm}"
    echo "       backend  : ${backend} on ${GPU_GRES}"
    echo "       geometry : ${TRAIN_NUM_ENVS} x ${rollout} = ${frames_per_batch}/iter, minibatch ${minibatch}"
    echo "       budget   : wall-limited (${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}); cap ${max_iterations} iters is unreachable by design"
    local delta_desc="${extra}"
    [[ -n "${ABLATION_ARM_ROLLOUT[${arm}]:-}" ]] \
        && delta_desc="rollout ${ROLLOUT_STEPS_DEFAULT} -> ${rollout}${extra:+ ; ${extra}}"
    echo "       delta    : ${delta_desc:-<none: control>}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf "       [CMD] "; printf "%q " "${cmd[@]}"; printf "\n\n"
    else
        "${cmd[@]}"
        echo
    fi
}

echo "[INFO] arms      : ${ARMS}"
echo "[INFO] data      : ${REF_ARRAYS}"
echo "[INFO] encoder   : ${ENCODER_CKPT}"
echo "[INFO] budget    : ${FRAME_CAP} frames/arm, save every ${SAVE_INTERVAL}"
echo "[INFO] wandb     : ${WANDB_PROJECT} / ${WANDB_GROUP}"
echo

if [[ "${DRY_RUN}" != "1" ]]; then
    check_gates
    echo
fi

for arm in ${ARMS}; do
    submit_arm "${arm}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] DRY_RUN=1; nothing submitted and no remote gates run."
    echo "[INFO] Re-run with DRY_RUN=0."
fi
