#!/usr/bin/env bash
set -euo pipefail

# Continue the completed ICE FSQ tracker from its 5.750B-frame checkpoint.
# This is a new protocol record: the only training change is restoring the
# SONIC action-rate reward from 0.0 to -0.1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }
print_cmd() { printf '[CMD] '; printf '%q ' "$@"; printf '\n'; }
ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=20 ice "$@"; }

MODE="${MODE:-print}"
CONFIRM_TOKEN="fsq-action-rate-continuation"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
SEED="${SEED:-0}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"
WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
# Keep the continuation beside the original arm in W&B; tags identify the
# reward change and the resumed checkpoint.
WANDB_GROUP="${WANDB_GROUP:-fsq-anchor-critic}"
RUN_NAME="${RUN_NAME:-bones129k_fsq64_action_rate_m01_continuation_seed${SEED}}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
ENCODER_CKPT="${ENCODER_CKPT:-/data/bones129k_fsq_anchor_critic/fsq64_heading_critic_no_latent_encoder/checkpoints/latest.pt}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-/data/bones129k_fsq_anchor_critic/fsq64_heading_critic_no_latent_tracker/rlopt_train/2026-08-09_00-34-44_wandb-o5pqhvgr/models/model_step_5750390784.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_fsq_action_rate_continuation}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/fsq64_action_rate_tracker/rlopt_train}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-10000000000}"
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-294912}"
EXPERT_BATCH_SIZE="${EXPERT_BATCH_SIZE:-24576}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

source_contract_hash() {
    sha256sum \
        "${SCRIPT_DIR}/run.sh" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_encoder.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_diffsr.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/ipmd/module.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/RLOpt/rlopt/env_interface.py" \
        "${REPO_ROOT}/scripts/rlopt/train_hl_skill_diffsr.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_cfg.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/tracking_env.py" \
        | sha256sum | awk '{print $1}'
}

build_command() {
    CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}"
        --num_envs "${TRAIN_NUM_ENVS}"
        --headless
        --algo IPMD
        --agent rlopt_ipmd_tuned_cfg_entry_point
        --seed "${SEED}"
        --max_iterations "${MAX_ITERATIONS}"
        --log_interval "${LOG_INTERVAL}"
        --checkpoint "${TRAIN_CHECKPOINT}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        physics=newton_mjwarp
        env.data.manifest=null
        "env.data.reference_arrays_dir=${REF_ARRAYS}"
        "env.data.persist_id=${PERSIST_ID}"
        env.data.reference_arrays_resident=true
        env.data.reference_arrays_warm_workers=16
        env.data.runtime_cache_device=cpu
        env.data.reference_prefetch_mode=next
        env.data.macro_cache_device=cuda:0
        "${BODY_NAMES_OVERRIDE}"
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        "agent.ipmd.expert_batch_size=${EXPERT_BATCH_SIZE}"
        agent.loss.gamma=0.97
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb
        agent.logger.video=false
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${RUN_NAME}"
        "agent.logger.log_dir=${LOG_DIR}"
        env.command_interface.actor=latent
        env.command_interface.actor.dim=66
        env.command_interface.encoder=single
        agent.ipmd.latent_dim=66
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT}"
        agent.ipmd.hl_skill_horizon_steps=10
        agent.ipmd.hl_skill_command_mode=z
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        agent.ipmd.latent_learning.code_latent_dim=64
        agent.ipmd.hl_skill_finetune_enabled=false
        agent.policy.num_cells=[2048,2048,1024,1024,512,512]
        agent.policy.activation_fn=silu
        agent.value_function.num_cells=[2048,2048,1024,1024,512,512]
        agent.value_function.activation_fn=silu
        env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
        env.expert_macro_frame_stride=1
        env.expert_macro_anchor_mode=expert_heading
        env.rewards.action_rate_l2.weight=-0.1
        env.rewards.tracking_reward_points.weight=4.0
        env.enable_termination_curriculum=true
        env.termination_curriculum_start_frames=5000000
        env.termination_curriculum_end_frames=30000000
        env.command_interface.reference.selection=random80_adaptive20
        env.sim.physics.solver_cfg.njmax=289
        env.sim.physics.solver_cfg.nconmax=200
        env.command_interface.critic_channels=[reference]
    )
}

check_remote_inputs() {
    local remote_root="/home/hice1/fwu91/scratch/Research/IsaacLab/data"
    local arrays_remote="${remote_root}${REF_ARRAYS#/data}"
    local encoder_remote="${remote_root}${ENCODER_CKPT#/data}"
    local checkpoint_remote="${remote_root}${TRAIN_CHECKPOINT#/data}"
    local output_remote="${remote_root}${OUTPUT_ROOT#/data}"
    local identity
    identity="$(ssh_ice "python3 -c \"import json; d=json.load(open('${arrays_remote}/reference_arrays_manifest.json')); print(len(d['traj_info']['ordered_traj_list']), d['traj_info']['written'], d['key']['source']['persist_id'])\"")" \
        || fail "remote reference arrays unavailable"
    [[ "${identity}" == "129785 47491234 ${PERSIST_ID}" ]] \
        || fail "remote reference identity mismatch: ${identity}"
    [[ "$(ssh_ice "test -f '${encoder_remote}' && echo yes || echo no")" == yes ]] \
        || fail "encoder checkpoint missing: ${ENCODER_CKPT}"
    [[ "$(ssh_ice "test -f '${checkpoint_remote}' && echo yes || echo no")" == yes ]] \
        || fail "training checkpoint missing: ${TRAIN_CHECKPOINT}"
    [[ "$(ssh_ice "test ! -e '${output_remote}' && echo yes || echo no")" == yes ]] \
        || fail "refusing existing output root: ${OUTPUT_ROOT}"
    echo "[PASS] remote reference, encoder, resume checkpoint, and fresh output root"
}

record_submission() {
    local job_id="$1" record_path="${SCRIPT_DIR}/cluster_submission.json"
    [[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
    python3 - "${record_path}" "${job_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(source_contract_hash)" <<'PY'
import json
import sys

path, job_id, submitted, source_hash = sys.argv[1:]
payload = {
    "campaign": "2026-08-10-bones129k-fsq-action-rate-continuation",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": "g1-bones-seed", "group": "fsq-anchor-critic"},
    "resume": {
        "checkpoint": "/data/bones129k_fsq_anchor_critic/fsq64_heading_critic_no_latent_tracker/rlopt_train/2026-08-09_00-34-44_wandb-o5pqhvgr/models/model_step_5750390784.pt",
        "frames_at_resume": 5750390784,
        "source_job_id": 5573503,
        "source_state": "TIMEOUT",
    },
    "change": {"env.rewards.action_rate_l2.weight": -0.1},
    "protocol": {
        "task": "Isaac-Imitation-G1-v2",
        "latent_mode": "sonic_fsq",
        "z_dim": 64,
        "fsq_levels": 32,
        "command_dim": 66,
        "macro_frame_stride": 1,
        "macro_anchor_mode": "expert_heading",
        "critic_channels": ["reference"],
        "num_envs": 16384,
        "rollout_steps": 24,
        "frames_per_batch": 393216,
        "mini_batch_size": 294912,
        "gamma": 0.97,
        "seed": 0,
        "frame_cap": 10000000000,
        "max_iterations": 25432,
        "checkpoint_interval_frames": 250000000,
        "expected_state": "ICE walltime may stop this segment before the 10B cap; checkpoints persist under /data.",
    },
    "job_id": int(job_id),
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
    echo "[RECORDED] ${record_path}"
}

case "${MODE}" in
    print|validate|submit) ;;
    *) fail "MODE must be print, validate, or submit; got ${MODE}" ;;
esac
if [[ "${MODE}" == submit && "${CONFIRM_SUBMIT:-}" != "${CONFIRM_TOKEN}" ]]; then
    fail "submission requires CONFIRM_SUBMIT=${CONFIRM_TOKEN}"
fi

echo "[INFO] ICE FSQ continuation from 5,750,390,784 frames"
echo "[INFO] action_rate_l2=-0.1; FSQ64 x 32; stride=1; expert_heading; critic=[reference]"
echo "[INFO] ${TRAIN_NUM_ENVS} envs x ${ROLLOUT_STEPS} steps = ${FRAMES_PER_BATCH} frames/iteration"
echo "[INFO] cap=${FRAME_CAP} frames (${MAX_ITERATIONS} iterations); output=${OUTPUT_ROOT}"
build_command
print_cmd "${CMD[@]}"

if [[ "${MODE}" == validate ]]; then
    check_remote_inputs
    echo "[PASS] validation complete; no scheduler mutation"
    exit 0
fi
if [[ "${MODE}" != submit ]]; then
    echo "[INFO] MODE=${MODE}; no scheduler mutation"
    exit 0
fi

check_remote_inputs
export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT=pace
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-160G}"
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_G1_USD_PATH=repo
export CLUSTER_SIM_BACKEND=newton
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-b129k-fsq-ar}"
export CLUSTER_WANDB_TAGS="bones-seed,129785,fsq64,sonic-fsq64,action-rate-minus0.1,resume-5750m,v2,root-qpos,stride1,expert-heading,critic-no-latent,newton,rollout24,gamma097,reset80-adaptive20"

submission_output="$("${CMD[@]}" 2>&1 | tee /dev/stderr)"
job_id="$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"${submission_output}" | tail -1)"
[[ "${job_id}" =~ ^[0-9]+$ ]] || fail "could not parse Slurm job ID"
echo "[SUBMITTED] tracker=${job_id}"
record_submission "${job_id}"
