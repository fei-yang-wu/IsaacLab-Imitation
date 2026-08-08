#!/usr/bin/env bash
set -euo pipefail

# BONES-129k H200 comparison on the gamma=.97 / rollout=24 tuned controller.
# MODE=print and MODE=smoke are local and non-mutating to the scheduler.
# MODE=validate runs remote gates. MODE=submit additionally requires the
# explicit confirmation token and a passing local smoke root from this source.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

# shellcheck source=arms.sh
source "${SCRIPT_DIR}/arms.sh"

fail() { echo "[FATAL] $*" >&2; exit 1; }

MODE="${MODE:-print}"
ARMS="${ARMS:-${LATENT_SCHEME_ARMS[*]}}"
SEED="${SEED:-0}"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
FRAME_CAP="${FRAME_CAP:-10000000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
ENCODER_CKPT="${ENCODER_CKPT:-/data/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/checkpoints/latest.pt}"
LOCAL_ENCODER_CKPT="${LOCAL_ENCODER_CKPT:-/mnt/storage/fwu91/bones_seed_full/runs/bones129k_root_qpos_v2_splitcache_e24576_r6_1b_seed0/encoder/checkpoints/latest.pt}"
EXPECTED_ENCODER_SHA256="${EXPECTED_ENCODER_SHA256:-d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH * 3 / 4))}"
ONLINE_EXPERT_BATCH_SIZE="${ONLINE_EXPERT_BATCH_SIZE:-24576}"

SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-4}"
SMOKE_ROLLOUT_STEPS="${SMOKE_ROLLOUT_STEPS:-4}"
SMOKE_MINIBATCH_SIZE="${SMOKE_MINIBATCH_SIZE:-8}"
SMOKE_EXPERT_BATCH_SIZE="${SMOKE_EXPERT_BATCH_SIZE:-8}"
LOCAL_SMOKE_ROOT="${LOCAL_SMOKE_ROOT:-}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-bones129k-ablation}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"

FSQ_LEVELS="["
for ((idx = 0; idx < 64; idx++)); do
    (( idx == 0 )) || FSQ_LEVELS+=","
    FSQ_LEVELS+="32"
done
FSQ_LEVELS+="]"

RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

COMMON_ENV_OVERRIDES=(
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0
    env.enable_termination_curriculum=true
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
    env.command_interface.reference.selection=random80_adaptive20
    env.sim.physics.solver_cfg.njmax=289
    env.sim.physics.solver_cfg.nconmax=200
)

source_contract_hash() {
    sha256sum \
        "${SCRIPT_DIR}/run.sh" \
        "${SCRIPT_DIR}/arms.sh" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/commands/reference.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/commands/reset_sampling.py" \
        | sha256sum | awk '{print $1}'
}

append_arm_overrides() {
    local -n command_ref="$1"
    local arm="$2"
    configure_arm "${arm}"
    command_ref+=("${COMMON_ENV_OVERRIDES[@]}")
    command_ref+=("${ARM_ENV_OVERRIDES[@]}")
    command_ref+=("${ARM_AGENT_OVERRIDES[@]}")
    if [[ "${arm}" == sonic_fsq32* ]]; then
        command_ref+=("agent.ipmd.latent_learning.fsq_levels=${FSQ_LEVELS}")
    fi
}

build_local_command() {
    local arm="$1"
    local output_dir="$2"
    LOCAL_CMD=(env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled
        pixi run -e isaaclab python -u scripts/rlopt/train.py
        --task "${TASK}" --num_envs "${SMOKE_NUM_ENVS}" --headless
        --algo IPMD --agent "${AGENT_ENTRY_POINT}" --seed "${SEED}"
        --max_iterations 1 --log_interval 1
        --kit_args=--/app/extensions/fsWatcherEnabled=false
        physics=newton_mjwarp
        env.data.manifest=null
        "env.data.reference_arrays_dir=${LOCAL_REF_ARRAYS}"
        "env.data.persist_id=${PERSIST_ID}"
        env.data.reference_arrays_resident=false
        env.data.reference_arrays_warm_workers=2
        env.data.runtime_cache_device=cpu
        env.data.reference_prefetch_mode=off
        env.data.macro_cache_device=cuda:0
        "${BODY_NAMES_OVERRIDE}"
        "agent.collector.frames_per_batch=${SMOKE_ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${SMOKE_MINIBATCH_SIZE}"
        "agent.ipmd.expert_batch_size=${SMOKE_EXPERT_BATCH_SIZE}"
        agent.loss.gamma=0.97
        agent.logger.backend=csv agent.logger.video=false
        "agent.logger.log_dir=${output_dir}/rlopt_train"
        "agent.logger.exp_name=smoke_${arm}")
    configure_arm "${arm}"
    if (( ARM_NEEDS_ENCODER )); then
        LOCAL_CMD+=("agent.ipmd.hl_skill_checkpoint_path=${LOCAL_ENCODER_CKPT}")
    fi
    append_arm_overrides LOCAL_CMD "${arm}"
}

build_cluster_command() {
    local arm="$1"
    local run_tag="$2"
    CLUSTER_CMD=(./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}" --num_envs "${TRAIN_NUM_ENVS}" --headless
        --algo IPMD --agent "${AGENT_ENTRY_POINT}" --seed "${SEED}"
        --max_iterations "${MAX_ITERATIONS}"
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
        "agent.ipmd.expert_batch_size=${ONLINE_EXPERT_BATCH_SIZE}"
        agent.loss.gamma=0.97
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb agent.logger.video=false
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${run_tag}"
        "agent.logger.log_dir=/data/bones129k_latent_sampler/${run_tag}/rlopt_train")
    configure_arm "${arm}"
    if (( ARM_NEEDS_ENCODER )); then
        CLUSTER_CMD+=("agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT}")
    fi
    append_arm_overrides CLUSTER_CMD "${arm}"
}

check_local_smokes() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}."
    local expected_hash arm got status
    expected_hash="$(source_contract_hash)"
    for arm in ${ARMS}; do
        [[ -f "${LOCAL_SMOKE_ROOT}/${arm}/status.json" ]] \
            || fail "missing local smoke marker for ${arm}: ${LOCAL_SMOKE_ROOT}/${arm}/status.json"
        read -r status got < <(python3 - "${LOCAL_SMOKE_ROOT}/${arm}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("status", ""), d.get("source_contract_sha256", ""))
PY
        )
        [[ "${status}" == "pass" ]] || fail "local smoke did not pass for ${arm}."
        [[ "${got}" == "${expected_hash}" ]] \
            || fail "local smoke for ${arm} is stale (${got} != ${expected_hash})."
    done
    echo "[PASS] local smokes match source contract ${expected_hash:0:16}…"
}

remote_of() { printf '%s' "${REMOTE_DATA_ROOT}${1#/data}"; }
ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=20 ice "$@"; }

check_remote_gates() {
    local arrays_remote encoder_remote rows sha arm run_tag remote_log
    arrays_remote="$(remote_of "${REF_ARRAYS}")"
    encoder_remote="$(remote_of "${ENCODER_CKPT}")"
    rows="$(ssh_ice "python3 -c \"
import json
d=json.load(open('${arrays_remote}/reference_arrays_manifest.json'))
t=d['traj_info']; k=d['key']
print(len(t['ordered_traj_list']), t['written'], k['source']['persist_id'])
\"")" || fail "remote reference arrays unavailable"
    [[ "${rows}" == "129785 47491234 ${PERSIST_ID}" ]] \
        || fail "remote reference identity mismatch: ${rows}"
    sha="$(ssh_ice "sha256sum '${encoder_remote}' | cut -d' ' -f1")"
    [[ "${sha}" == "${EXPECTED_ENCODER_SHA256}" ]] || fail "remote encoder hash mismatch: ${sha}"
    for arm in ${ARMS}; do
        run_tag="bones129k_${arm}_reset80_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}_10b_seed${SEED}"
        remote_log="$(remote_of "/data/bones129k_latent_sampler/${run_tag}/rlopt_train")"
        [[ "$(ssh_ice "if [[ -e '${remote_log}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
            || fail "refusing to overwrite ${remote_log}"
    done
    grep -q "random80_adaptive20" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
        || fail "new reset preset is absent from the submitted source"
    echo "[PASS] remote arrays, encoder, output paths, and reset sampler"
}

case "${MODE}" in
    print|smoke|validate|submit) ;;
    *) fail "MODE must be print, smoke, validate, or submit; got ${MODE}." ;;
esac
if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "bones129k-latent-sampler" ]]; then
    fail "submission requires CONFIRM_SUBMIT=bones129k-latent-sampler"
fi

echo "[INFO] mode=${MODE} arms=${ARMS}"
echo "[INFO] common: BONES-129k, v2 tuned, gamma=.97, rollout=24, reset80/adaptive20"
echo "[INFO] cluster: ${TRAIN_NUM_ENVS} envs, ${FRAME_CAP} frames, ${GPU_GRES}"
echo "[INFO] wandb: ${WANDB_PROJECT} / ${WANDB_GROUP}"
echo "[WARN] VQ K=32 matches SONIC's per-coordinate cardinality 32, not its 32^64 product capacity."

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_latent_sampler_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}"
    contract_hash="$(source_contract_hash)"
    for arm in ${ARMS}; do
        arm_root="${LOCAL_SMOKE_ROOT}/${arm}"
        mkdir -p "${arm_root}"
        build_local_command "${arm}" "${arm_root}"
        configure_arm "${arm}"
        echo "[SMOKE] ${arm}: ${ARM_DESCRIPTION}"
        printf '[CMD] '; printf '%q ' "${LOCAL_CMD[@]}"; printf '\n'
        "${LOCAL_CMD[@]}" 2>&1 | tee "${arm_root}/train.log"
        python3 - "${arm_root}/status.json" "${arm}" "${contract_hash}" <<'PY'
import json, sys
json.dump({"status": "pass", "arm": sys.argv[2], "source_contract_sha256": sys.argv[3]}, open(sys.argv[1], "w"), indent=2)
PY
    done
    echo "[PASS] all local smokes: ${LOCAL_SMOKE_ROOT}"
    exit 0
fi

if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    check_local_smokes
    check_remote_gates
fi

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT=pace
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_TIME_LIMIT="${CLUSTER_SLURM_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-160G}"
export CLUSTER_G1_USD_PATH=repo
export CLUSTER_SIM_BACKEND=newton
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py

for arm in ${ARMS}; do
    configure_arm "${arm}"
    run_tag="bones129k_${arm}_reset80_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}_10b_seed${SEED}"
    build_cluster_command "${arm}" "${run_tag}"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-${arm}"
    export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,reset80-adaptive20,rollout24,gamma097,${arm},${ARM_WANDB_TAGS},newton"
    echo "[ARM] ${arm}: ${ARM_DESCRIPTION}"
    printf '[CMD] '; printf '%q ' "${CLUSTER_CMD[@]}"; printf '\n'
    if [[ "${MODE}" == "submit" ]]; then
        "${CLUSTER_CMD[@]}"
    fi
done

[[ "${MODE}" == "submit" ]] || echo "[INFO] MODE=${MODE}; no scheduler mutation."
