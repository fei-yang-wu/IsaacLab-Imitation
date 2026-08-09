#!/usr/bin/env bash
set -euo pipefail

# 2026-08-08 -- expert-heading macro frame, critic WITHOUT the actor latent.
#
# One low-level job. One variable against the running arm ICE `5573234`
# (W&B `9lraqu2e`, group `latent-anchor-frame`):
#
#   env.command_interface.critic_channels=[reference]
#
# The critic keeps the noise-free reference channel plus privileged state and
# loses only the 258-D actor latent. The actor is untouched.
#
# There is NO pretrain job. This arm loads the SAME frozen encoder the running
# arm loads, by path and by SHA-256, so the encoder cannot drift between the
# two rows. That is what makes the comparison single-variable.
#
#   MODE=print   ./run.sh
#   MODE=smoke   ./run.sh                     # local 1-iteration gate
#   MODE=validate LOCAL_SMOKE_ROOT=<dir> ./run.sh
#   MODE=submit  LOCAL_SMOKE_ROOT=<dir> CONFIRM_SUBMIT=latent-anchor-frame ./run.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }
print_cmd() { printf '  %q' "$@"; printf '\n'; }
ssh_ice() { ssh -o BatchMode=yes -o ConnectTimeout=20 ice "$@"; }
remote_of() { printf '%s' "${REMOTE_DATA_ROOT}${1#/data}"; }

MODE="${MODE:-print}"
SEED="${SEED:-0}"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
ARM="${ARM:-expert_heading_critic_no_latent}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_anchor_frame}"

# -- the axis under test --
CRITIC_CHANNELS_OVERRIDE="env.command_interface.critic_channels=[reference]"

# -- the shared encoder, frozen, identical to the control arm's --
CONTROL_LOWLEVEL_JOB=5573234
CONTROL_WANDB_RUN="9lraqu2e"
ENCODER_REMOTE="${OUTPUT_ROOT}/expert_heading_encoder/checkpoints/latest.pt"
ENCODER_SHA256="be6d533f1d1ca4aa6b1e819af1d3ef63eb033125018c8309c7448384b6a9583e"
LOCAL_ENCODER="${LOCAL_ENCODER:-${REPO_ROOT}/logs/downloaded_checkpoints/bones129k_anchor_frame/expert_heading_encoder_latest.pt}"

# -- everything below is the control arm's contract, reproduced exactly --
ANCHOR_MODE="expert_heading"
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=1
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# Actor input width is unchanged; the critic loses exactly the latent command.
EXPECTED_ACTOR_INPUT_DIM=351
EXPECTED_CRITIC_INPUT_DIM=286

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
# The same group as the arm this is compared against, on purpose.
WANDB_GROUP="${WANDB_GROUP:-latent-anchor-frame}"
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

MACRO_INTERFACE_OVERRIDES=(
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    "env.expert_macro_frame_stride=${MACRO_FRAME_STRIDE}"
    "env.expert_macro_anchor_mode=${ANCHOR_MODE}"
)

COMMON_ENV_OVERRIDES=(
    "${MACRO_INTERFACE_OVERRIDES[@]}"
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
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/RLOpt/rlopt/env_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/_compiled.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/tracking_env.py" \
        | sha256sum | awk '{print $1}'
}

lowlevel_dir() { printf '%s/%s_tracker/rlopt_train' "${OUTPUT_ROOT}" "${ARM}"; }

append_lowlevel_contract() {
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
        "${COMMON_ENV_OVERRIDES[@]}"
        "${CRITIC_CHANNELS_OVERRIDE}"
    )
    # No tracker capacity override, exactly like the control arm.
}

build_local_lowlevel_command() {
    local checkpoint_path="$1"
    local output_dir="$2"
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
        "agent.logger.exp_name=smoke_${ARM}"
    )
    append_lowlevel_contract LOCAL_LOWLEVEL_CMD "${checkpoint_path}"
}

build_cluster_lowlevel_command() {
    local run_tag="bones129k_${ARM}_seed${SEED}"
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
        "agent.logger.log_dir=$(lowlevel_dir)"
    )
    append_lowlevel_contract LOWLEVEL_CMD "${ENCODER_REMOTE}"
}

check_local_encoder() {
    [[ -f "${LOCAL_ENCODER}" ]] || fail "local encoder copy missing: ${LOCAL_ENCODER}"
    local got
    got="$(sha256sum "${LOCAL_ENCODER}" | awk '{print $1}')"
    [[ "${got}" == "${ENCODER_SHA256}" ]] \
        || fail "local encoder sha ${got} != control encoder ${ENCODER_SHA256}"
    pixi run -q python - "${LOCAL_ENCODER}" "${ANCHOR_MODE}" "${MACRO_FRAME_STRIDE}" \
        "${HORIZON_STEPS}" "${Z_DIM}" <<'PY'
import sys
import torch

path, mode, stride, horizon, z_dim = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint["config"]
checks = {
    "macro_anchor_mode": (str(config.get("macro_anchor_mode", "robot")), mode),
    "macro_frame_stride": (int(config.get("macro_frame_stride", 1)), int(stride)),
    "horizon_steps": (int(config["horizon_steps"]), int(horizon)),
    "z_dim": (int(config["z_dim"]), int(z_dim)),
}
for field, (got, want) in checks.items():
    if got != want:
        raise SystemExit(f"[FATAL] encoder {field}={got!r}, expected {want!r}")
width = int(checkpoint["skill_encoder_state_dict"]["net.0.weight"].shape[1])
if width != 380:
    raise SystemExit(f"[FATAL] encoder input width {width} != 380 (root_qpos)")
print(f"[PASS] shared encoder: {mode}, stride {stride}, h{horizon}, z{z_dim}, width 380")
PY
}

check_remote_gates() {
    local arrays_remote rows path remote_sha
    arrays_remote="$(remote_of "${REF_ARRAYS}")"
    rows="$(ssh_ice "python3 -c \"
import json
d=json.load(open('${arrays_remote}/reference_arrays_manifest.json'))
t=d['traj_info']; k=d['key']
print(len(t['ordered_traj_list']), t['written'], k['source']['persist_id'])
\"")" || fail "remote reference arrays unavailable"
    [[ "${rows}" == "129785 47491234 ${PERSIST_ID}" ]] \
        || fail "remote reference identity mismatch: ${rows}"

    remote_sha="$(ssh_ice "sha256sum '$(remote_of "${ENCODER_REMOTE}")' | awk '{print \$1}'")" \
        || fail "shared encoder unreadable on ICE"
    [[ "${remote_sha}" == "${ENCODER_SHA256}" ]] \
        || fail "remote encoder sha ${remote_sha} != ${ENCODER_SHA256}"

    path="$(remote_of "$(lowlevel_dir)")"
    [[ "$(ssh_ice "if [[ -e '${path}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
        || fail "refusing to overwrite ${path}"
    echo "[PASS] remote arrays, shared encoder identity, fresh output path"
}

check_local_smoke() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}."
    [[ -f "${LOCAL_SMOKE_ROOT}/status.json" ]] || fail "missing local smoke marker"
    local expected status got channels
    expected="$(source_contract_hash)"
    read -r status got channels < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""), d.get("critic_channels", ""))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${channels}" == "reference" ]] || fail "local smoke covered critic channels ${channels}"
    echo "[PASS] local smoke at source contract ${expected:0:16}…"
}

case "${MODE}" in
    print|smoke|validate|submit) ;;
    *) fail "MODE must be print, smoke, validate, or submit; got ${MODE}" ;;
esac
if [[ "${MODE}" == "submit" && "${CONFIRM_SUBMIT:-}" != "${WANDB_GROUP}" ]]; then
    fail "submission requires CONFIRM_SUBMIT=${WANDB_GROUP}"
fi

echo "[INFO] mode=${MODE}; arm=${ARM}"
echo "[INFO] axis: ${CRITIC_CHANNELS_OVERRIDE}"
echo "[INFO] control = ICE ${CONTROL_LOWLEVEL_JOB} (W&B ${CONTROL_WANDB_RUN}), critic channels [actor, reference]"
echo "[INFO] shared frozen encoder ${ENCODER_SHA256:0:16}… (${ANCHOR_MODE}, stride ${MACRO_FRAME_STRIDE}); no pretrain job"
echo "[INFO] actor input ${EXPECTED_ACTOR_INPUT_DIM} unchanged; critic input ${EXPECTED_CRITIC_INPUT_DIM} (was 544)"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] ICE allocations end at 16 h, so this job is EXPECTED to TIMEOUT before the cap;"
echo "[INFO] checkpoints land every ${SAVE_INTERVAL} frames under persistent /data, not node-local storage."
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    check_local_encoder
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_anchor_critic_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"
    build_local_lowlevel_command "${LOCAL_ENCODER}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # The gate: the critic observation group must have LOST latent_command and
    # nothing else, and the policy group must still have it. A width check
    # alone would pass if some other term silently changed too.
    python3 - "${LOCAL_SMOKE_ROOT}/tracker/train.log" \
        "${EXPECTED_ACTOR_INPUT_DIM}" "${EXPECTED_CRITIC_INPUT_DIM}" <<'PY'
import re
import sys

log_path, actor_dim, critic_dim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
text = open(log_path, encoding="utf-8", errors="replace").read()
groups = {}
current = None
for line in text.splitlines():
    header = re.search(r"Active Observation Terms in Group: '(\w+)'", line)
    if header:
        current = header.group(1)
        groups[current] = []
        continue
    if current is None:
        continue
    row = re.match(r"\|\s*\d+\s*\|\s*(\w+)\s*\|\s*\((\d+),\)\s*\|", line)
    if row:
        groups[current].append((row.group(1), int(row.group(2))))
    elif line.startswith("[INFO]") and "Observation" not in line:
        current = None

for name in ("policy", "critic"):
    if not groups.get(name):
        raise SystemExit(f"[FATAL] no {name} observation group parsed from {log_path}")

policy_terms = [t for t, _ in groups["policy"]]
critic_terms = [t for t, _ in groups["critic"]]
if "latent_command" not in policy_terms:
    raise SystemExit("[FATAL] actor lost latent_command; only the critic may change")
if "latent_command" in critic_terms:
    raise SystemExit("[FATAL] critic still reads latent_command; the override did nothing")

expected_critic = [
    "expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b",
    "body_pos", "body_ori", "base_lin_vel", "base_ang_vel",
    "joint_pos_rel", "joint_vel_rel", "last_action",
]
if critic_terms != expected_critic:
    raise SystemExit(f"[FATAL] critic terms changed beyond the latent: {critic_terms}")

actor_keys = {
    "latent_command", "projected_gravity", "base_ang_vel",
    "joint_pos_rel", "joint_vel_rel", "last_action",
}
actor_width = sum(w for t, w in groups["policy"] if t in actor_keys)
critic_width = sum(w for _, w in groups["critic"])
if actor_width != actor_dim:
    raise SystemExit(f"[FATAL] actor input {actor_width} != {actor_dim}")
if critic_width != critic_dim:
    raise SystemExit(f"[FATAL] critic input {critic_width} != {critic_dim}")
print(f"[PASS] actor {actor_width} unchanged; critic {critic_width}, no latent_command")
PY

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" <<'PY'
import json, sys

path, source_hash = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "critic_channels": "reference",
        },
        stream,
        indent=2,
    )
    stream.write("\n")
PY
    echo "[PASS] local smoke: ${LOCAL_SMOKE_ROOT}"
    exit 0
fi

if [[ "${MODE}" == "validate" || "${MODE}" == "submit" ]]; then
    check_local_encoder
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

build_cluster_lowlevel_command
echo "[LOWLEVEL ${ARM}]"
print_cmd "${LOWLEVEL_CMD[@]}"

if [[ "${MODE}" != "submit" ]]; then
    echo "[INFO] MODE=${MODE}; no scheduler mutation."
    exit 0
fi

export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-anch-cnl"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,hold10,expert-heading-anchor,critic-no-latent,critic-reference-only,lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
unset CLUSTER_SLURM_DEPENDENCY

output="$("${LOWLEVEL_CMD[@]}" 2>&1 | tee /dev/stderr)"
lowlevel_job="$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"${output}" | tail -1)"
[[ "${lowlevel_job}" =~ ^[0-9]+$ ]] || fail "could not parse Slurm job ID"
echo "[SUBMITTED] ${ARM} tracker=${lowlevel_job}"

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" "${ARM}" \
    "${lowlevel_job}" "${CONTROL_LOWLEVEL_JOB}" "${CONTROL_WANDB_RUN}" \
    "${ENCODER_SHA256}" "${ENCODER_REMOTE}" "${FRAME_CAP}" "${MAX_ITERATIONS}" \
    "${EXPECTED_ACTOR_INPUT_DIM}" "${EXPECTED_CRITIC_INPUT_DIM}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm, lowlevel, control_job,
    control_run, encoder_sha, encoder_path, frame_cap, iterations,
    actor_dim, critic_dim,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-08-bones129k-anchor-critic-no-latent",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "axis": {
        "field": "env.command_interface.critic_channels",
        "arm_value": ["reference"],
        "control_value": ["actor", "reference"],
    },
    "control": {
        "job_id": int(control_job),
        "wandb_run": control_run,
        "arm": "expert_heading",
        "note": "not resubmitted; this campaign adds one arm against it",
    },
    "arm": {"arm": arm, "lowlevel_job_id": int(lowlevel), "pretrain_job_id": None},
    "shared_encoder": {
        "path": encoder_path,
        "sha256": encoder_sha,
        "note": "identical file the control arm loads; no pretrain job for this arm",
        "macro_anchor_mode": "expert_heading",
        "macro_frame_stride": 1,
        "horizon_steps": 10,
        "z_dim": 256,
        "encoder_input_dim": 380,
    },
    "lowlevel": {
        "task": "Isaac-Imitation-G1-v2",
        "motions": 129785,
        "command_dim": 258,
        "hold_steps": 10,
        "actor_input_dim": int(actor_dim),
        "critic_input_dim": int(critic_dim),
        "tracker_num_cells": "tuned entry point default (not overridden)",
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
        "expected_termination": (
            "TIMEOUT before the 10B cap; ICE allocations end at 16 h. "
            "Checkpoints persist under /data, so a TIMEOUT loses no training."
        ),
    },
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
echo "[RECORDED] ${record_path}"
