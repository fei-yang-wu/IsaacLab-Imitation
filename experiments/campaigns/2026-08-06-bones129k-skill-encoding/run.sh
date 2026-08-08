#!/usr/bin/env bash
set -euo pipefail

# Three DiffSR transition-factorization arms on BONES-129k. Each encoder job is
# followed by one afterok-dependent low-level controller job using that encoder.

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
print_cmd() { printf '[CMD] '; printf '%q ' "$@"; printf '\n'; }
ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=20 ice "$@"; }
remote_of() { printf '%s' "${REMOTE_DATA_ROOT}${1#/data}"; }

MODE="${MODE:-print}"
ARMS="${ARMS:-${SKILL_ENCODING_ARMS[*]}}"
SEED="${SEED:-0}"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_skill_encoding}"

HORIZON_STEPS=10
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-1000}"
PRETRAIN_EVAL_BATCHES="${PRETRAIN_EVAL_BATCHES:-4}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-5000000000}"
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH * 3 / 4))}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"
ONLINE_EXPERT_BATCH_SIZE="${ONLINE_EXPERT_BATCH_SIZE:-24576}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-skill-encoding-ablation}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"
LOCAL_SMOKE_ROOT="${LOCAL_SMOKE_ROOT:-}"

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
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_diffsr.py" \
        "${REPO_ROOT}/scripts/rlopt/train_hl_skill_diffsr.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        | sha256sum | awk '{print $1}'
}

arm_run_tag() {
    printf 'bones129k_skill_%s_h10_z256_seed%s' "$1" "${SEED}"
}

encoder_dir_for() {
    printf '%s/%s/encoder' "${OUTPUT_ROOT}" "$(arm_run_tag "$1")"
}

lowlevel_dir_for() {
    printf '%s/%s/rlopt_train' "${OUTPUT_ROOT}" "$(arm_run_tag "$1")"
}

append_pretrain_contract() {
    local -n command_ref="$1"
    local arm="$2"
    configure_skill_encoding_arm "${arm}"
    command_ref+=(
        --horizon_steps "${HORIZON_STEPS}"
        --encoder_window_mode intermediate
        --z_dim "${Z_DIM}" --latent_mode deterministic
        --batch_size "${PRETRAIN_BATCH_SIZE}"
        --num_updates "${PRETRAIN_UPDATES}"
        --log_interval "${PRETRAIN_LOG_INTERVAL}"
        --eval_batches "${PRETRAIN_EVAL_BATCHES}"
        --reconstruction_eval --window_probe_eval
        --window_probe_train_batches 8 --window_probe_eval_batches 4
        "${ARM_PRETRAIN_ARGS[@]}"
    )
}

build_local_smoke_command() {
    local arm="$1"
    local arm_root="$2"
    LOCAL_CMD=(
        env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
        TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled
        pixi run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py
        --task "${TASK}" --num_envs 4 --seed "${SEED}" --device cuda:0
        --headless --assert-kitless --output_dir "${arm_root}/encoder"
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
    append_pretrain_contract LOCAL_CMD "${arm}"
    LOCAL_CMD+=(
        --batch_size 8 --num_updates 1 --log_interval 1 --eval_batches 1
        --eval_batch_size 8 --window_probe_train_batches 1
        --window_probe_eval_batches 1
    )
}

build_cluster_pretrain_command() {
    local arm="$1"
    local encoder_dir
    encoder_dir="$(encoder_dir_for "${arm}")"
    PRETRAIN_CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}" --num_envs "${PRETRAIN_NUM_ENVS}" --seed "${SEED}"
        --device cuda:0 --headless --assert-kitless
        --output_dir "${encoder_dir}"
        --logger_backend wandb --wandb_project "${WANDB_PROJECT}"
        --wandb_group "${WANDB_GROUP}"
        --wandb_run_name "$(arm_run_tag "${arm}")_pretrain"
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
    append_pretrain_contract PRETRAIN_CMD "${arm}"
}

build_cluster_lowlevel_command() {
    local arm="$1"
    local encoder_ckpt run_tag lowlevel_dir
    encoder_ckpt="$(encoder_dir_for "${arm}")/checkpoints/latest.pt"
    run_tag="$(arm_run_tag "${arm}")_lowlevel"
    lowlevel_dir="$(lowlevel_dir_for "${arm}")"
    LOWLEVEL_CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
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
        "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
        agent.loss.gamma=0.97
        "agent.ipmd.expert_batch_size=${ONLINE_EXPERT_BATCH_SIZE}"
        "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}"
        agent.ipmd.command_source=hl_skill
        "agent.ipmd.hl_skill_checkpoint_path=${encoder_ckpt}"
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
        agent.ipmd.hl_skill_command_mode=z
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        agent.ipmd.latent_learning.code_latent_dim=256
        agent.ipmd.hl_skill_finetune_enabled=false
        "agent.save_interval=${SAVE_INTERVAL}"
        agent.logger.backend=wandb agent.logger.video=false
        "agent.logger.project_name=${WANDB_PROJECT}"
        "agent.logger.group_name=${WANDB_GROUP}"
        "agent.logger.exp_name=${run_tag}"
        "agent.logger.log_dir=${lowlevel_dir}"
        "${COMMON_ENV_OVERRIDES[@]}"
    )
}

check_local_smokes() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}."
    local arm expected_hash status got
    expected_hash="$(source_contract_hash)"
    for arm in ${ARMS}; do
        [[ -f "${LOCAL_SMOKE_ROOT}/${arm}/status.json" ]] \
            || fail "missing local smoke marker for ${arm}"
        read -r status got < <(python3 - "${LOCAL_SMOKE_ROOT}/${arm}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""))
PY
        )
        [[ "${status}" == "pass" ]] || fail "local smoke did not pass for ${arm}"
        [[ "${got}" == "${expected_hash}" ]] \
            || fail "local smoke for ${arm} is stale (${got} != ${expected_hash})"
    done
    echo "[PASS] local smokes match source contract ${expected_hash:0:16}…"
}

check_remote_gates() {
    local arrays_remote rows arm encoder_remote lowlevel_remote
    arrays_remote="$(remote_of "${REF_ARRAYS}")"
    rows="$(ssh_ice "python3 -c \"
import json
d=json.load(open('${arrays_remote}/reference_arrays_manifest.json'))
t=d['traj_info']; k=d['key']
print(len(t['ordered_traj_list']), t['written'], k['source']['persist_id'])
\"")" || fail "remote reference arrays unavailable"
    [[ "${rows}" == "129785 47491234 ${PERSIST_ID}" ]] \
        || fail "remote reference identity mismatch: ${rows}"
    for arm in ${ARMS}; do
        configure_skill_encoding_arm "${arm}"
        encoder_remote="$(remote_of "$(encoder_dir_for "${arm}")")"
        lowlevel_remote="$(remote_of "$(lowlevel_dir_for "${arm}")")"
        [[ "$(ssh_ice "if [[ -e '${encoder_remote}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
            || fail "refusing to overwrite ${encoder_remote}"
        [[ "$(ssh_ice "if [[ -e '${lowlevel_remote}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
            || fail "refusing to overwrite ${lowlevel_remote}"
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

echo "[INFO] mode=${MODE} arms=${ARMS}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"
echo "[INFO] low level=${TRAIN_NUM_ENVS}x${ROLLOUT_STEPS}, gamma=.97, ${FRAME_CAP} frame cap"
echo "[WARN] ICE scratch was 90.3% used before this campaign; checkpoint interval=${SAVE_INTERVAL}."

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_skill_encoding_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}"
    contract_hash="$(source_contract_hash)"
    for arm in ${ARMS}; do
        configure_skill_encoding_arm "${arm}"
        arm_root="${LOCAL_SMOKE_ROOT}/${arm}"
        mkdir -p "${arm_root}"
        build_local_smoke_command "${arm}" "${arm_root}"
        echo "[SMOKE] ${arm}: ${ARM_DESCRIPTION}"
        print_cmd "${LOCAL_CMD[@]}"
        "${LOCAL_CMD[@]}" 2>&1 | tee "${arm_root}/train.log"
        python3 - "${arm_root}/status.json" "${arm}" "${contract_hash}" <<'PY'
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(
        {"status": "pass", "arm": sys.argv[2], "source_contract_sha256": sys.argv[3]},
        stream,
        indent=2,
    )
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
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-160G}"
export CLUSTER_G1_USD_PATH=repo
export CLUSTER_SIM_BACKEND=newton

submission_records=()
if [[ "${MODE}" != "submit" ]]; then
    for arm in ${ARMS}; do
        configure_skill_encoding_arm "${arm}"
        build_cluster_pretrain_command "${arm}"
        build_cluster_lowlevel_command "${arm}"
        echo "[ARM] ${arm}: ${ARM_DESCRIPTION}"
        echo "[PRETRAIN]"
        print_cmd "${PRETRAIN_CMD[@]}"
        echo "[LOWLEVEL afterok:<pretrain-job>]"
        print_cmd "${LOWLEVEL_CMD[@]}"
    done
else
    declare -A pretrain_jobs=()

    # Submit every encoder first. Controller jobs are added only after all
    # encoder job IDs exist, and each keeps an arm-specific afterok dependency.
    for arm in ${ARMS}; do
        configure_skill_encoding_arm "${arm}"
        build_cluster_pretrain_command "${arm}"
        echo "[ARM] ${arm}: ${ARM_DESCRIPTION}"
        export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train_hl_skill_diffsr.py
        export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT:-03:00:00}"
        export CLUSTER_SLURM_JOB_NAME_PREFIX="skenc-${arm}-pre"
        export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,skill-encoding,pretrain,${arm},${ARM_WANDB_TAGS}"
        unset CLUSTER_SLURM_DEPENDENCY
        pretrain_jobs["${arm}"]="$(submit_and_capture_job_id "${PRETRAIN_CMD[@]}")"
        echo "[SUBMITTED] ${arm} pretrain=${pretrain_jobs[${arm}]}"
    done

    for arm in ${ARMS}; do
        configure_skill_encoding_arm "${arm}"
        build_cluster_lowlevel_command "${arm}"
        pretrain_job="${pretrain_jobs[${arm}]}"
        export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
        export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
        export CLUSTER_SLURM_JOB_NAME_PREFIX="skenc-${arm}-ll"
        export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,skill-encoding,lowlevel,${arm},${ARM_WANDB_TAGS},newton,rollout24,gamma097,reset80-adaptive20"
        export CLUSTER_SLURM_DEPENDENCY="afterok:${pretrain_job}"
        lowlevel_job="$(submit_and_capture_job_id "${LOWLEVEL_CMD[@]}")"
        echo "[SUBMITTED] ${arm} lowlevel=${lowlevel_job} afterok:${pretrain_job}"
        submission_records+=("${arm}:${pretrain_job}:${lowlevel_job}")
    done
fi

if [[ "${MODE}" == "submit" ]]; then
    record_path="${SCRIPT_DIR}/cluster_submission.json"
    [[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
    python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" \
        "${FRAME_CAP}" "${submission_records[@]}" <<'PY'
import json, sys

path, submitted, source_hash, project, group, frame_cap, *records = sys.argv[1:]
jobs = []
for record in records:
    arm, pretrain, lowlevel = record.split(":")
    jobs.append(
        {
            "arm": arm,
            "pretrain_job_id": int(pretrain),
            "lowlevel_job_id": int(lowlevel),
            "dependency": f"afterok:{pretrain}",
        }
    )
payload = {
    "campaign": "2026-08-06-bones129k-skill-encoding",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "task": "Isaac-Imitation-G1-v2",
    "wandb": {"project": project, "group": group},
    "common": {
        "motions": 129785,
        "macro_interface": "root_qpos",
        "horizon_steps": 10,
        "z_dim": 256,
        "hold_steps": 10,
        "num_envs": 16384,
        "rollout_steps": 24,
        "gamma": 0.97,
        "frame_cap": int(frame_cap),
        "checkpoint_interval_frames": 250000000,
    },
    "jobs": jobs,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
    echo "[RECORDED] ${record_path}"
else
    echo "[INFO] MODE=${MODE}; no scheduler mutation."
fi
