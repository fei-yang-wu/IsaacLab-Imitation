#!/usr/bin/env bash
set -euo pipefail

# One shared scaled SONIC-FSQ64 pretrain followed by two afterok-dependent 5B
# controllers. Both controllers load the same frozen encoder checkpoint; only
# actor/critic capacity changes.

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
remote_of() { printf '%s' "${REMOTE_DATA_ROOT}${1#/data}"; }

MODE="${MODE:-print}"
SEED="${SEED:-0}"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
TRACKER_ARMS="${TRACKER_ARMS:-tuned sonic}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_sonic_fsq_scale}"

HORIZON_STEPS=10
Z_DIM=64
LATENT_COMMAND_DIM=$((Z_DIM + 2))
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-1000}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-5000000000}"
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH * 3 / 4))}"
ONLINE_EXPERT_BATCH_SIZE="${ONLINE_EXPERT_BATCH_SIZE:-24576}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-4}"
SMOKE_ROLLOUT_STEPS="${SMOKE_ROLLOUT_STEPS:-4}"
SMOKE_MINIBATCH_SIZE="${SMOKE_MINIBATCH_SIZE:-8}"
SMOKE_EXPERT_BATCH_SIZE="${SMOKE_EXPERT_BATCH_SIZE:-8}"
LOCAL_SMOKE_ROOT="${LOCAL_SMOKE_ROOT:-}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-skill-encoding-ablation}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"

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
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_encoder.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_diffsr.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/ipmd/module.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/scripts/rlopt/train_hl_skill_diffsr.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_cfg.py" \
        | sha256sum | awk '{print $1}'
}

encoder_dir() {
    printf '%s/shared_scaled_fsq64_encoder' "${OUTPUT_ROOT}"
}

lowlevel_dir() {
    printf '%s/%s_tracker/rlopt_train' "${OUTPUT_ROOT}" "$1"
}

append_pretrain_contract() {
    local -n command_ref="$1"
    command_ref+=(
        --horizon_steps "${HORIZON_STEPS}"
        --encoder_window_mode intermediate
        --transition_objective endpoint
        --z_dim "${Z_DIM}" --latent_mode sonic_fsq
        --encoder_hidden_dims 2048 1024 512 512
        --encoder_activation silu --no_encoder_layer_norm
        --diffsr_feature_dim 256 --diffsr_embed_dim 1024
        --diffsr_g_hidden_dims 1024 1024 512
        --diffsr_mu_hidden_dims 1024 1024 512
        --batch_size "${PRETRAIN_BATCH_SIZE}"
        --num_updates "${PRETRAIN_UPDATES}"
        --log_interval "${PRETRAIN_LOG_INTERVAL}"
        --eval_batches 4
        --reconstruction_eval --window_probe_eval
        --window_probe_train_batches 8 --window_probe_eval_batches 4
    )
}

tracker_widths() {
    case "$1" in
        tuned) TRACKER_CELLS="[1024,1024,512]" ;;
        sonic) TRACKER_CELLS="[2048,2048,1024,1024,512,512]" ;;
        *) fail "unknown tracker arm $1" ;;
    esac
}

append_lowlevel_contract() {
    local -n command_ref="$1"
    local arm="$2"
    local checkpoint_path="$3"
    tracker_widths "${arm}"
    command_ref+=(
        "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
        "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}"
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${checkpoint_path}"
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
        agent.ipmd.hl_skill_command_mode=z
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        agent.ipmd.hl_skill_finetune_enabled=false
        "agent.policy.num_cells=${TRACKER_CELLS}"
        agent.policy.activation_fn=silu
        "agent.value_function.num_cells=${TRACKER_CELLS}"
        agent.value_function.activation_fn=silu
        "${COMMON_ENV_OVERRIDES[@]}"
    )
}

build_local_pretrain_command() {
    local output_dir="$1"
    LOCAL_PRETRAIN_CMD=(
        env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
        TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled
        pixi run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py
        --task "${TASK}" --num_envs 4 --seed "${SEED}" --device cuda:0
        --headless --assert-kitless --output_dir "${output_dir}"
        --logger_backend none
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
        env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    )
    append_pretrain_contract LOCAL_PRETRAIN_CMD
    LOCAL_PRETRAIN_CMD+=(
        --batch_size 8 --num_updates 1 --log_interval 1 --eval_batches 1
        --eval_batch_size 8 --window_probe_train_batches 1
        --window_probe_eval_batches 1
    )
}

build_local_lowlevel_command() {
    local arm="$1"
    local checkpoint_path="$2"
    local output_dir="$3"
    LOCAL_LOWLEVEL_CMD=(
        env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
        TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled
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
        "agent.logger.log_dir=${output_dir}"
        "agent.logger.exp_name=smoke_fsq64_${arm}"
    )
    append_lowlevel_contract LOCAL_LOWLEVEL_CMD "${arm}" "${checkpoint_path}"
}

build_cluster_pretrain_command() {
    PRETRAIN_CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}" --num_envs "${PRETRAIN_NUM_ENVS}" --seed "${SEED}"
        --device cuda:0 --headless --assert-kitless
        --output_dir "$(encoder_dir)"
        --logger_backend wandb --wandb_project "${WANDB_PROJECT}"
        --wandb_group "${WANDB_GROUP}"
        --wandb_run_name bones129k_scaled_sonic_fsq64_pretrain_seed0
        physics=newton_mjwarp
        env.data.manifest=null
        "env.data.reference_arrays_dir=${REF_ARRAYS}"
        "env.data.persist_id=${PERSIST_ID}"
        env.data.reference_arrays_resident=true
        env.data.reference_arrays_warm_workers=16
        env.data.runtime_cache_device=cpu
        env.data.reference_prefetch_mode=off
        env.data.macro_cache_device=cuda:0
        "${BODY_NAMES_OVERRIDE}"
        env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    )
    append_pretrain_contract PRETRAIN_CMD
}

build_cluster_lowlevel_command() {
    local arm="$1"
    local checkpoint_path="$(encoder_dir)/checkpoints/latest.pt"
    local run_tag="bones129k_scaled_fsq64_${arm}_tracker_5b_seed${SEED}"
    LOWLEVEL_CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}" --num_envs "${TRAIN_NUM_ENVS}" --headless
        --algo IPMD --agent "${AGENT_ENTRY_POINT}" --seed "${SEED}"
        --max_iterations "${MAX_ITERATIONS}" --log_interval "${LOG_INTERVAL}"
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
        "agent.logger.log_dir=$(lowlevel_dir "${arm}")"
    )
    append_lowlevel_contract LOWLEVEL_CMD "${arm}" "${checkpoint_path}"
}

check_local_smoke() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}."
    [[ -f "${LOCAL_SMOKE_ROOT}/status.json" ]] || fail "missing local smoke marker"
    local expected status got
    expected="$(source_contract_hash)"
    read -r status got < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    echo "[PASS] local smoke matches source contract ${expected:0:16}…"
}

check_remote_gates() {
    local arrays_remote rows path
    arrays_remote="$(remote_of "${REF_ARRAYS}")"
    rows="$(ssh_ice "python3 -c \"
import json
d=json.load(open('${arrays_remote}/reference_arrays_manifest.json'))
t=d['traj_info']; k=d['key']
print(len(t['ordered_traj_list']), t['written'], k['source']['persist_id'])
\"")" || fail "remote reference arrays unavailable"
    [[ "${rows}" == "129785 47491234 ${PERSIST_ID}" ]] \
        || fail "remote reference identity mismatch: ${rows}"
    for path in "$(encoder_dir)" "$(lowlevel_dir tuned)" "$(lowlevel_dir sonic)"; do
        path="$(remote_of "${path}")"
        [[ "$(ssh_ice "if [[ -e '${path}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
            || fail "refusing to overwrite ${path}"
    done
    echo "[PASS] remote arrays and fresh output paths"
}

submit_and_capture_job_id() {
    local output job_id
    output="$("$@" 2>&1 | tee /dev/stderr)"
    job_id="$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"${output}" | tail -1)"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || fail "could not parse Slurm job ID"
    printf '%s' "${job_id}"
}

case "${MODE}" in
    print|smoke|validate|submit) ;;
    *) fail "MODE must be print, smoke, validate, or submit; got ${MODE}" ;;
esac
if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "${WANDB_GROUP}" ]]; then
    fail "submission requires CONFIRM_SUBMIT=${WANDB_GROUP}"
fi

echo "[INFO] mode=${MODE}; tracker arms=${TRACKER_ARMS}"
echo "[INFO] shared pretrain: root-qpos 380, FSQ64, scaled DiffSR, endpoint objective"
echo "[INFO] controllers: ${FRAME_CAP} frames each on ${GPU_GRES}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}"
    build_local_pretrain_command "${LOCAL_SMOKE_ROOT}/encoder"
    print_cmd "${LOCAL_PRETRAIN_CMD[@]}"
    "${LOCAL_PRETRAIN_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/pretrain.log"
    local_checkpoint="${LOCAL_SMOKE_ROOT}/encoder/checkpoints/latest.pt"
    [[ -f "${local_checkpoint}" ]] || fail "smoke pretrain did not write ${local_checkpoint}"
    for arm in ${TRACKER_ARMS}; do
        mkdir -p "${LOCAL_SMOKE_ROOT}/${arm}_tracker"
        build_local_lowlevel_command "${arm}" "${local_checkpoint}" "${LOCAL_SMOKE_ROOT}/${arm}_tracker"
        print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
        "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/${arm}_tracker/train.log"
    done
    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" <<'PY'
import json, sys
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump({"status": "pass", "source_contract_sha256": sys.argv[2]}, stream, indent=2)
    stream.write("\n")
PY
    echo "[PASS] local smoke: ${LOCAL_SMOKE_ROOT}"
    exit 0
fi

if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    check_local_smoke
    check_remote_gates
fi

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT=pace
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-160G}"
export CLUSTER_G1_USD_PATH=repo
export CLUSTER_SIM_BACKEND=newton

build_cluster_pretrain_command
echo "[PRETRAIN]"
print_cmd "${PRETRAIN_CMD[@]}"
for arm in ${TRACKER_ARMS}; do
    build_cluster_lowlevel_command "${arm}"
    echo "[LOWLEVEL ${arm} afterok:<pretrain-job>]"
    print_cmd "${LOWLEVEL_CMD[@]}"
done

if [[ "${MODE}" != "submit" ]]; then
    echo "[INFO] MODE=${MODE}; no scheduler mutation."
    exit 0
fi

export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train_hl_skill_diffsr.py
export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX=b129k-fsq64-pre
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,sonic-fsq64,scaled-pretrain,endpoint"
unset CLUSTER_SLURM_DEPENDENCY
pretrain_job="$(submit_and_capture_job_id "${PRETRAIN_CMD[@]}")"
echo "[SUBMITTED] shared pretrain=${pretrain_job}"

submission_records=()
for arm in ${TRACKER_ARMS}; do
    build_cluster_lowlevel_command "${arm}"
    export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
    export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-fsq64-${arm}"
    export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,sonic-fsq64,lowlevel,${arm}-tracker,newton,rollout24,gamma097,reset80-adaptive20"
    export CLUSTER_SLURM_DEPENDENCY="afterok:${pretrain_job}"
    lowlevel_job="$(submit_and_capture_job_id "${LOWLEVEL_CMD[@]}")"
    echo "[SUBMITTED] ${arm} tracker=${lowlevel_job} afterok:${pretrain_job}"
    submission_records+=("${arm}:${lowlevel_job}")
done

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${pretrain_job}" "${WANDB_PROJECT}" \
    "${WANDB_GROUP}" "${FRAME_CAP}" "${submission_records[@]}" <<'PY'
import json, sys

path, submitted, source_hash, pretrain, project, group, frame_cap, *records = sys.argv[1:]
controllers = []
for record in records:
    arm, job = record.split(":")
    controllers.append(
        {"arm": arm, "job_id": int(job), "dependency": f"afterok:{pretrain}"}
    )
payload = {
    "campaign": "2026-08-06-bones129k-sonic-fsq-scale",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "pretrain": {
        "job_id": int(pretrain),
        "macro_interface": "root_qpos",
        "encoder_input_dim": 380,
        "latent": "sonic_fsq64_levels32",
        "encoder_hidden_dims": [2048, 1024, 512, 512],
        "encoder_activation": "silu",
        "encoder_layer_norm": False,
        "diffsr_feature_dim": 256,
        "diffsr_embed_dim": 1024,
        "diffsr_g_hidden_dims": [1024, 1024, 512],
        "diffsr_mu_hidden_dims": [1024, 1024, 512],
        "updates": 50000,
        "batch_size": 8192,
    },
    "controllers": controllers,
    "common": {
        "task": "Isaac-Imitation-G1-v2",
        "motions": 129785,
        "command_dim": 66,
        "hold_steps": 10,
        "num_envs": 16384,
        "rollout_steps": 24,
        "gamma": 0.97,
        "frame_cap": int(frame_cap),
        "checkpoint_interval_frames": 250000000,
    },
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
echo "[RECORDED] ${record_path}"
