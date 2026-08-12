#!/usr/bin/env bash
set -euo pipefail

# BONES-SEED 129k learning-to-track (L2T): a privileged explicit-command
# teacher controls training rollouts. A deployable latent-command student
# learns the executed teacher action on the same rollouts.
#
#   MODE=print ./run.sh
#   MODE=smoke ./run.sh
#   MODE=validate LOCAL_SMOKE_ROOT=<dir> ./run.sh
#   MODE=submit LOCAL_SMOKE_ROOT=<dir> CONFIRM_SUBMIT=l2t ./run.sh

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
ALGO="IPMD_L2T"
AGENT_ENTRY_POINT="rlopt_ipmd_l2t_cfg_entry_point"
ARM="l2t_teacher_explicit_student_latent"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-${REPO_ROOT}/data/bones_seed/reference_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="bones_seed_sonic_full_129785@e714bbff"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_l2t_10b}"

ENCODER_REMOTE="/data/bones129k_anchor_frame/expert_heading_encoder/checkpoints/latest.pt"
ENCODER_SHA256="be6d533f1d1ca4aa6b1e819af1d3ef63eb033125018c8309c7448384b6a9583e"
LOCAL_ENCODER="${LOCAL_ENCODER:-${REPO_ROOT}/logs/downloaded_checkpoints/bones129k_anchor_frame/expert_heading_encoder_latest.pt}"

ANCHOR_MODE="expert_heading"
MACRO_FRAME_STRIDE=1
HORIZON_STEPS=10
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))
EXPECTED_TEACHER_INPUT_DIM=286
EXPECTED_STUDENT_INPUT_DIM=351

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-10000000000}"
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH * 3 / 4))}"
ONLINE_EXPERT_BATCH_SIZE="${ONLINE_EXPERT_BATCH_SIZE:-24576}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-4}"
SMOKE_ROLLOUT_STEPS="${SMOKE_ROLLOUT_STEPS:-4}"
SMOKE_MINIBATCH_SIZE="${SMOKE_MINIBATCH_SIZE:-8}"
SMOKE_EXPERT_BATCH_SIZE="${SMOKE_EXPERT_BATCH_SIZE:-8}"
LOCAL_SMOKE_ROOT="${LOCAL_SMOKE_ROOT:-}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-l2t}"
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
    "env.expert_macro_frame_stride=${MACRO_FRAME_STRIDE}"
    "env.expert_macro_anchor_mode=${ANCHOR_MODE}"
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0
    env.enable_termination_curriculum=true
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
    env.command_interface.reference.selection=random80_adaptive20
    env.sim.physics.solver_cfg.njmax=289
    env.sim.physics.solver_cfg.nconmax=200
    env.command_interface.critic_channels=[reference]
)

source_contract_hash() {
    sha256sum \
        "${SCRIPT_DIR}/run.sh" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/ipmd/ipmd.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/ipmd/ipmd_l2t.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/scripts/rlopt/train_impl.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_l2t_cfg.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        | sha256sum | awk '{print $1}'
}

lowlevel_dir() { printf '%s/%s/rlopt_train' "${OUTPUT_ROOT}" "${ARM}"; }

append_l2t_contract() {
    local -n command_ref="$1"
    local checkpoint_path="$2"
    command_ref+=(
        env.command_interface.actor=latent
        "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
        env.command_interface.encoder=single
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
        agent.ipmd_l2t.imitation_coeff=1.0
        "${COMMON_ENV_OVERRIDES[@]}"
    )
}

build_local_command() {
    local output_dir="$1"
    LOCAL_CMD=(
        env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
        HYDRA_FULL_ERROR=1 TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled
        pixi run -e isaaclab python -u scripts/rlopt/train.py
        --task "${TASK}" --num_envs "${SMOKE_NUM_ENVS}" --headless
        --algo "${ALGO}" --agent "${AGENT_ENTRY_POINT}" --seed "${SEED}"
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
        "agent.logger.exp_name=smoke_${ARM}"
    )
    append_l2t_contract LOCAL_CMD "${LOCAL_ENCODER}"
}

build_cluster_command() {
    LOWLEVEL_CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}" --num_envs "${TRAIN_NUM_ENVS}" --headless
        --algo "${ALGO}" --agent "${AGENT_ENTRY_POINT}" --seed "${SEED}"
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
        "agent.logger.exp_name=bones129k_${ARM}_seed${SEED}"
        "agent.logger.log_dir=$(lowlevel_dir)"
    )
    append_l2t_contract LOWLEVEL_CMD "${ENCODER_REMOTE}"
}

check_encoder() {
    [[ -f "${LOCAL_ENCODER}" ]] || fail "local encoder is missing: ${LOCAL_ENCODER}"
    local got
    got="$(sha256sum "${LOCAL_ENCODER}" | awk '{print $1}')"
    [[ "${got}" == "${ENCODER_SHA256}" ]] || fail "local encoder SHA-256 mismatch: ${got}"
    pixi run -q python - "${LOCAL_ENCODER}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = checkpoint["config"]
checks = {
    "macro_anchor_mode": (str(config.get("macro_anchor_mode", "robot")), "expert_heading"),
    "macro_frame_stride": (int(config.get("macro_frame_stride", 1)), 1),
    "horizon_steps": (int(config["horizon_steps"]), 10),
    "z_dim": (int(config["z_dim"]), 256),
}
for field, (got, want) in checks.items():
    if got != want:
        raise SystemExit(f"[FATAL] encoder {field}={got!r}, expected {want!r}")
width = int(checkpoint["skill_encoder_state_dict"]["net.0.weight"].shape[1])
if width != 380:
    raise SystemExit(f"[FATAL] encoder input width {width} != 380 (root_qpos)")
print("[PASS] encoder: expert_heading, stride 1, h10, z256, root_qpos width 380")
PY
}

check_remote_gates() {
    local arrays_remote rows remote_sha output_remote
    arrays_remote="$(remote_of "${REF_ARRAYS}")"
    rows="$(ssh_ice "python3 -c \"
import json
d=json.load(open('${arrays_remote}/reference_arrays_manifest.json'))
t=d['traj_info']; print(len(t['ordered_traj_list']), t['written'], d['key']['source']['persist_id'])
\"")" || fail "remote reference arrays are unavailable"
    [[ "${rows}" == "129785 47491234 ${PERSIST_ID}" ]] || fail "remote reference identity mismatch: ${rows}"

    remote_sha="$(ssh_ice "sha256sum '$(remote_of "${ENCODER_REMOTE}")' | awk '{print \$1}'")" \
        || fail "remote encoder is unavailable"
    [[ "${remote_sha}" == "${ENCODER_SHA256}" ]] || fail "remote encoder SHA-256 mismatch: ${remote_sha}"

    output_remote="$(remote_of "$(lowlevel_dir)")"
    [[ "$(ssh_ice "if [[ -e '${output_remote}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
        || fail "refusing to overwrite ${output_remote}"
    echo "[PASS] ICE arrays, encoder, and fresh persistent output path"
}

check_smoke_marker() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}"
    [[ -f "${LOCAL_SMOKE_ROOT}/status.json" ]] || fail "missing smoke marker: ${LOCAL_SMOKE_ROOT}/status.json"
    local status got expected
    expected="$(source_contract_hash)"
    read -r status got < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    echo "[PASS] local L2T smoke at source contract ${expected:0:16}…"
}

case "${MODE}" in
    print|smoke|validate|submit) ;;
    *) fail "MODE must be print, smoke, validate, or submit; got ${MODE}" ;;
esac
if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "${WANDB_GROUP}" ]]; then
    fail "submission requires CONFIRM_SUBMIT=${WANDB_GROUP}"
fi

echo "[INFO] mode=${MODE}; arm=${ARM}; algorithm=${ALGO}"
echo "[INFO] teacher explicit/privileged input=${EXPECTED_TEACHER_INPUT_DIM}; student latent/deployable input=${EXPECTED_STUDENT_INPUT_DIM}"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations (${FRAMES_PER_BATCH} frames/batch)"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}; output=$(lowlevel_dir)"

if [[ "${MODE}" == "smoke" ]]; then
    check_encoder
    [[ -d "${LOCAL_REF_ARRAYS}" ]] || fail "local reference arrays are missing: ${LOCAL_REF_ARRAYS}"
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_l2t_campaign_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"
    build_local_command "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_CMD[@]}"
    "${LOCAL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/train.log"
    python3 - "${LOCAL_SMOKE_ROOT}/train.log" "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" <<'PY'
import json
import re
import sys

log_path, marker_path, source_hash = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()
plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
required = (
    "IPMDL2T",
    "latent_command           |  (258,)",
    "in_features=286",
    "student_loss=",
    "student_rmse=",
    "Training time:",
)
missing = [value for value in required if value not in plain]
if missing:
    raise SystemExit(f"[FATAL] incomplete L2T smoke, missing {missing!r}")
if "Traceback (most recent call last)" in plain or "nan" in plain.lower():
    raise SystemExit("[FATAL] L2T smoke contains an exception or NaN")
with open(marker_path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "teacher_input_dim": 286,
            "student_input_dim": 351,
            "motions": 129785,
        },
        stream,
        indent=2,
    )
    stream.write("\n")
print("[PASS] one real BONES-SEED L2T iteration completed with finite student metrics")
PY
    echo "[PASS] local smoke: ${LOCAL_SMOKE_ROOT}"
    exit 0
fi

if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    check_encoder
    check_smoke_marker
    check_remote_gates
fi

export CLUSTER_LOGIN="${CLUSTER_LOGIN:-login-ice.pace.gatech.edu}"
export CLUSTER_SLURM_SUBMIT_SCRIPT=pace
export CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
export CLUSTER_ARCHIVE_SYNC=1
# The archive already contains this repo and its current submodule worktrees.
# Empty exported values override stale machine-specific paths in the shared
# ICE profile when cluster_interface.sh restores the caller's environment.
export CLUSTER_ISAACLAB_LOCAL_PATH=""
export CLUSTER_RLOPT_LOCAL_PATH=""
export CLUSTER_IMITATION_TOOLS_LOCAL_PATH=""
export CLUSTER_EXTRA_SYNC_SPECS=""
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${GPU_GRES}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-160G}"
export CLUSTER_G1_USD_PATH=repo
export CLUSTER_SIM_BACKEND=newton
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-l2t"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,l2t,teacher-explicit,student-latent,root-qpos,stride1,z256,hold10,expert-heading-anchor,newton,rollout24,gamma097,reset80-adaptive20"
unset CLUSTER_SLURM_DEPENDENCY

build_cluster_command
echo "[LOWLEVEL ${ARM}]"
print_cmd "${LOWLEVEL_CMD[@]}"

if [[ "${MODE}" != "submit" ]]; then
    echo "[INFO] MODE=${MODE}; no scheduler mutation."
    exit 0
fi

output="$("${LOWLEVEL_CMD[@]}" 2>&1 | tee /dev/stderr)"
job_id="$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"${output}" | tail -1)"
archive_sha="$(sed -n 's/.*Uploading workspace archive: .*sha256=\([0-9a-f]\{64\}\).*/\1/p' <<<"${output}" | tail -1)"
[[ "${job_id}" =~ ^[0-9]+$ ]] || fail "could not parse Slurm job ID"
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "could not parse workspace archive SHA-256"
echo "[SUBMITTED] ${ARM} job=${job_id} archive=${archive_sha}"

record_path="${SUBMISSION_RECORD_PATH:-${SCRIPT_DIR}/cluster_submission_10b.json}"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${archive_sha}" "${job_id}" "${FRAME_CAP}" \
    "${MAX_ITERATIONS}" "${WANDB_PROJECT}" "${WANDB_GROUP}" <<'PY'
import json, sys

path, submitted, source_hash, archive_hash, job, frame_cap, iterations, project, group = sys.argv[1:]
payload = {
    "campaign": "2026-08-09-bones129k-l2t",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "job_id": int(job),
    "source_contract_sha256": source_hash,
    "workspace_archive_sha256": archive_hash,
    "wandb": {"project": project, "group": group},
    "data": {
        "reference_arrays": "/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1",
        "motions": 129785,
        "frames": 47491234,
        "persist_id": "bones_seed_sonic_full_129785@e714bbff",
    },
    "encoder": {
        "path": "/data/bones129k_anchor_frame/expert_heading_encoder/checkpoints/latest.pt",
        "sha256": "be6d533f1d1ca4aa6b1e819af1d3ef63eb033125018c8309c7448384b6a9583e",
        "input_interface": "root_qpos",
        "input_dim": 380,
        "macro_anchor_mode": "expert_heading",
        "macro_frame_stride": 1,
        "horizon_steps": 10,
        "z_dim": 256,
    },
    "l2t": {
        "teacher_command": "explicit reference",
        "teacher_input_dim": 286,
        "student_command": "258-D DiffSR latent plus phase",
        "student_input_dim": 351,
        "imitation_coeff": 1.0,
    },
    "training": {
        "task": "Isaac-Imitation-G1-v2",
        "algorithm": "IPMD_L2T",
        "num_envs": 16384,
        "rollout_steps": 24,
        "mini_batch_size": 294912,
        "gamma": 0.97,
        "seed": 0,
        "frame_cap": int(frame_cap),
        "max_iterations": int(iterations),
        "checkpoint_interval_frames": 50000000,
        "reset_sampler": "random80_adaptive20",
        "walltime": "15:59:00",
        "persistent_output": "/data/bones129k_l2t_10b/l2t_teacher_explicit_student_latent/rlopt_train",
    },
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
echo "[RECORDED] ${record_path}"
