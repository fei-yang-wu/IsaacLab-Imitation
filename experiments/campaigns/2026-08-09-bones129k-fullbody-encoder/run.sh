#!/usr/bin/env bash
set -euo pipefail

# 2026-08-09 -- what the skill encoder READS, one variable.
#
# Control is the running ICE job `5573413` (`expert_heading_critic_no_latent`):
# expert-heading macro frame, stride 1, z256, critic channels `[reference]`,
# frozen DiffSR encoder over the `root_qpos` macro state (joint positions plus
# root pose, 380 values per window). Nothing is resubmitted for it.
#
# The axis:
#
#   env.expert_macro_state_terms
#     control: [expert_motion_qpos, expert_anchor_pos_b, expert_anchor_ori_b]   380
#     arm:     [expert_motion,      expert_anchor_pos_b, expert_anchor_ori_b]   670
#
# `expert_motion` is the `joint_qpos_qvel` component: the same joint positions
# the control reads PLUS joint velocities. So the arm adds reference joint
# VELOCITY to the encoder input and changes nothing else.
#
# Why velocity specifically. The released SONIC checkpoint we benchmark against
# (SR 0.9937 against our 0.9062) feeds its tokenizer reference joint_pos(29) +
# joint_vel(29) + a 6D root-orientation difference per window frame. Our v2
# default dropped to positions only on 2026-08-04. Velocity is the concrete
# encoder-input difference between the two recipes -- NOT 14-body keypoints,
# which belong to the paper's `sonic_bones_seed.yaml` experiment config rather
# than to the released model.
#
# The macro window keeps stride 1 by explicit user choice on 2026-08-09, so
# this arm does not confound the input-content question with window span.
#
# The encoder input width DOES change here (380 -> 670), so unlike the anchor
# and stride axes a mispairing is caught by shape. The pretrain still records
# the anchor mode and stride, and the low level still checks them.
#
#   MODE=print   ./run.sh
#   MODE=smoke   ./run.sh                     # local pretrain + tracker gate
#   MODE=validate LOCAL_SMOKE_ROOT=<dir> ./run.sh
#   MODE=submit  LOCAL_SMOKE_ROOT=<dir> CONFIRM_SUBMIT=encoder-input ./run.sh

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
ARM="${ARM:-expert_heading_fullbody_encoder}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_fullbody_encoder}"

CONTROL_LOWLEVEL_JOB=5573413
CONTROL_ARM="expert_heading_critic_no_latent"

# -- the axis --
MACRO_STATE_TERMS="[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]"
EXPECTED_ENCODER_INPUT_DIM=670

# -- the control's window and bottleneck, reproduced exactly --
ANCHOR_MODE="expert_heading"
HORIZON_STEPS=10
MACRO_FRAME_STRIDE="${MACRO_FRAME_STRIDE:-1}"
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))
EXPECTED_ACTOR_INPUT_DIM=351
EXPECTED_CRITIC_INPUT_DIM=286

# -- the control's pretrain geometry, passed explicitly so a default change
# -- cannot move this arm away from it --
ENCODER_HIDDEN_DIMS=(1024 512 512)
DIFFSR_FEATURE_DIM=128
DIFFSR_EMBED_DIM=512
DIFFSR_HEAD_DIMS=(512)

PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-1000}"

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
WANDB_GROUP="${WANDB_GROUP:-encoder-input}"
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
    "env.expert_macro_state_terms=${MACRO_STATE_TERMS}"
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
    env.command_interface.critic_channels=[reference]
)

source_contract_hash() {
    sha256sum \
        "${SCRIPT_DIR}/run.sh" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_encoder.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_diffsr.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/RLOpt/rlopt/env_interface.py" \
        "${REPO_ROOT}/scripts/rlopt/train_hl_skill_diffsr.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/tracking_env.py" \
        | sha256sum | awk '{print $1}'
}

encoder_dir() { printf '%s/%s_encoder' "${OUTPUT_ROOT}" "${ARM}"; }
lowlevel_dir() { printf '%s/%s_tracker/rlopt_train' "${OUTPUT_ROOT}" "${ARM}"; }

append_pretrain_contract() {
    local -n command_ref="$1"
    command_ref+=(
        --horizon_steps "${HORIZON_STEPS}"
        --encoder_window_mode intermediate
        --transition_objective endpoint
        --z_dim "${Z_DIM}"
        --latent_mode deterministic
        --encoder_hidden_dims "${ENCODER_HIDDEN_DIMS[@]}"
        --encoder_activation mish
        --diffsr_feature_dim "${DIFFSR_FEATURE_DIM}"
        --diffsr_embed_dim "${DIFFSR_EMBED_DIM}"
        --diffsr_g_hidden_dims "${DIFFSR_HEAD_DIMS[@]}"
        --diffsr_mu_hidden_dims "${DIFFSR_HEAD_DIMS[@]}"
        --batch_size "${PRETRAIN_BATCH_SIZE}"
        --num_updates "${PRETRAIN_UPDATES}"
        --log_interval "${PRETRAIN_LOG_INTERVAL}"
        --eval_batches 4
        --reconstruction_eval --window_probe_eval
        --window_probe_train_batches 8 --window_probe_eval_batches 4
    )
}

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
    )
    # No tracker capacity override, exactly like the control arm.
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
        "${MACRO_INTERFACE_OVERRIDES[@]}"
    )
    append_pretrain_contract LOCAL_PRETRAIN_CMD
    LOCAL_PRETRAIN_CMD+=(
        --batch_size 8 --num_updates 1 --log_interval 1 --eval_batches 1
        --eval_batch_size 8 --window_probe_train_batches 1
        --window_probe_eval_batches 1
    )
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

build_cluster_pretrain_command() {
    PRETRAIN_CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}" --num_envs "${PRETRAIN_NUM_ENVS}" --seed "${SEED}"
        --device cuda:0 --headless --assert-kitless
        --output_dir "$(encoder_dir)"
        --logger_backend wandb --wandb_project "${WANDB_PROJECT}"
        --wandb_group "${WANDB_GROUP}"
        --wandb_run_name "bones129k_${ARM}_pretrain_seed${SEED}"
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
        "${MACRO_INTERFACE_OVERRIDES[@]}"
    )
    append_pretrain_contract PRETRAIN_CMD
}

build_cluster_lowlevel_command() {
    local checkpoint_path="$(encoder_dir)/checkpoints/latest.pt"
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
    append_lowlevel_contract LOWLEVEL_CMD "${checkpoint_path}"
}

check_local_smoke() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}."
    [[ -f "${LOCAL_SMOKE_ROOT}/status.json" ]] || fail "missing local smoke marker"
    local expected status got width mode
    expected="$(source_contract_hash)"
    read -r status got width mode < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    d.get("status", ""),
    d.get("source_contract_sha256", ""),
    d.get("encoder_input_dim", ""),
    d.get("macro_anchor_mode", ""),
)
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${width}" == "${EXPECTED_ENCODER_INPUT_DIM}" ]] || fail "smoke encoder width ${width}"
    [[ "${mode}" == "${ANCHOR_MODE}" ]] || fail "smoke anchor mode ${mode}"
    echo "[PASS] local smoke at source contract ${expected:0:16}…"
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
    for path in "$(encoder_dir)" "$(lowlevel_dir)"; do
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

echo "[INFO] mode=${MODE}; arm=${ARM}"
echo "[INFO] axis: env.expert_macro_state_terms=${MACRO_STATE_TERMS} (${EXPECTED_ENCODER_INPUT_DIM}), control root_qpos (380)"
echo "[INFO] delta = reference joint VELOCITY added to the encoder input"
echo "[INFO] control = ICE ${CONTROL_LOWLEVEL_JOB} (${CONTROL_ARM})"
echo "[INFO] window: ${HORIZON_STEPS} slots at stride ${MACRO_FRAME_STRIDE}, ${ANCHOR_MODE} frame, z${Z_DIM}"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_fullbody_encoder_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"

    build_local_pretrain_command "${LOCAL_SMOKE_ROOT}/encoder"
    print_cmd "${LOCAL_PRETRAIN_CMD[@]}"
    "${LOCAL_PRETRAIN_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/pretrain.log"
    SMOKE_CHECKPOINT="${LOCAL_SMOKE_ROOT}/encoder/checkpoints/latest.pt"
    [[ -f "${SMOKE_CHECKPOINT}" ]] || fail "smoke pretrain did not write ${SMOKE_CHECKPOINT}"

    pixi run -q python - "${SMOKE_CHECKPOINT}" "${ANCHOR_MODE}" "${MACRO_FRAME_STRIDE}" \
        "${HORIZON_STEPS}" "${Z_DIM}" "${EXPECTED_ENCODER_INPUT_DIM}" <<'PY'
import sys
import torch

path, mode, stride, horizon, z_dim, width = sys.argv[1:]
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
got_width = int(checkpoint["skill_encoder_state_dict"]["net.0.weight"].shape[1])
if got_width != int(width):
    raise SystemExit(f"[FATAL] encoder input width {got_width} != {width}")
print(f"[PASS] encoder: {mode}, stride {stride}, h{horizon}, z{z_dim}, width {got_width}")
PY

    build_local_lowlevel_command "${SMOKE_CHECKPOINT}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # The actor and critic must be UNCHANGED by this axis: the encoder input
    # grew, the published command did not.
    python3 - "${LOCAL_SMOKE_ROOT}/tracker/train.log" \
        "${EXPECTED_ACTOR_INPUT_DIM}" "${EXPECTED_CRITIC_INPUT_DIM}" \
        "${LATENT_COMMAND_DIM}" <<'PY'
import re
import sys

log_path, actor_dim, critic_dim, command_dim = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
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

policy_terms = dict(groups["policy"])
critic_terms = [t for t, _ in groups["critic"]]
if int(policy_terms.get("latent_command", -1)) != command_dim:
    raise SystemExit(
        f"[FATAL] latent_command is {policy_terms.get('latent_command')}, expected {command_dim}"
    )
if "latent_command" in critic_terms:
    raise SystemExit("[FATAL] critic reads latent_command; expected critic_channels=[reference]")

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
print(f"[PASS] actor {actor_width} and critic {critic_width} unchanged by the encoder-input axis")
PY

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${EXPECTED_ENCODER_INPUT_DIM}" "${ANCHOR_MODE}" "${MACRO_FRAME_STRIDE}" <<'PY'
import json, sys

path, source_hash, width, mode, stride = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "encoder_input_dim": int(width),
            "macro_anchor_mode": mode,
            "macro_frame_stride": int(stride),
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
echo "[PRETRAIN ${ARM}]"
print_cmd "${PRETRAIN_CMD[@]}"
build_cluster_lowlevel_command
echo "[LOWLEVEL ${ARM} afterok:<pretrain-job>]"
print_cmd "${LOWLEVEL_CMD[@]}"

if [[ "${MODE}" != "submit" ]]; then
    echo "[INFO] MODE=${MODE}; no scheduler mutation."
    exit 0
fi

COMMON_TAGS="bones-seed,129785,v2,full-body-670,joint-vel,stride${MACRO_FRAME_STRIDE},z${Z_DIM},hold10,expert-heading-anchor,critic-no-latent"

build_cluster_pretrain_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train_hl_skill_diffsr.py
export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-fb-enc-pre"
export CLUSTER_WANDB_TAGS="${COMMON_TAGS},pretrain,endpoint"
unset CLUSTER_SLURM_DEPENDENCY
pretrain_job="$(submit_and_capture_job_id "${PRETRAIN_CMD[@]}")"
echo "[SUBMITTED] ${ARM} pretrain=${pretrain_job}"

build_cluster_lowlevel_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-fb-enc"
export CLUSTER_WANDB_TAGS="${COMMON_TAGS},lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
export CLUSTER_SLURM_DEPENDENCY="afterok:${pretrain_job}"
lowlevel_job="$(submit_and_capture_job_id "${LOWLEVEL_CMD[@]}")"
echo "[SUBMITTED] ${ARM} tracker=${lowlevel_job} afterok:${pretrain_job}"

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" "${ARM}" \
    "${pretrain_job}" "${lowlevel_job}" "${CONTROL_LOWLEVEL_JOB}" "${CONTROL_ARM}" \
    "${MACRO_FRAME_STRIDE}" "${HORIZON_STEPS}" "${Z_DIM}" "${LATENT_COMMAND_DIM}" \
    "${EXPECTED_ENCODER_INPUT_DIM}" "${FRAME_CAP}" "${MAX_ITERATIONS}" "${SEED}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm, pretrain, lowlevel,
    control_job, control_arm, stride, horizon, z_dim, command_dim,
    encoder_dim, frame_cap, iterations, seed,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-09-bones129k-fullbody-encoder",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "axis": {
        "field": "env.expert_macro_state_terms",
        "arm_value": ["expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b"],
        "control_value": [
            "expert_motion_qpos", "expert_anchor_pos_b", "expert_anchor_ori_b",
        ],
        "delta": "reference joint velocity added to the encoder input",
        "encoder_input_dim": {"arm": int(encoder_dim), "control": 380},
    },
    "control": {
        "job_id": int(control_job),
        "arm": control_arm,
        "wandb_group": "latent-anchor-frame",
        "note": "not resubmitted; this campaign adds one arm against it",
    },
    "arm": {
        "arm": arm,
        "pretrain_job_id": int(pretrain),
        "lowlevel_job_id": int(lowlevel),
        "dependency": f"afterok:{pretrain}",
    },
    "pretrain": {
        "macro_interface": "full_body",
        "encoder_input_dim": int(encoder_dim),
        "macro_frame_stride": int(stride),
        "macro_anchor_mode": "expert_heading",
        "horizon_steps": int(horizon),
        "z_dim": int(z_dim),
        "latent_mode": "deterministic",
        "encoder_window_mode": "intermediate",
        "transition_objective": "endpoint",
        "encoder_hidden_dims": [1024, 512, 512],
        "encoder_activation": "mish",
        "diffsr_feature_dim": 128,
        "diffsr_embed_dim": 512,
        "diffsr_g_hidden_dims": [512],
        "diffsr_mu_hidden_dims": [512],
        "updates": 50000,
        "batch_size": 8192,
    },
    "lowlevel": {
        "task": "Isaac-Imitation-G1-v2",
        "motions": 129785,
        "command_dim": int(command_dim),
        "hold_steps": 10,
        "actor_input_dim": 351,
        "critic_input_dim": 286,
        "critic_channels": ["reference"],
        "tracker_num_cells": "tuned entry point default (not overridden)",
        "num_envs": 16384,
        "rollout_steps": 24,
        "mini_batch_size": 294912,
        "gamma": 0.97,
        "seed": int(seed),
        "frame_cap": int(frame_cap),
        "max_iterations": int(iterations),
        "checkpoint_interval_frames": 50000000,
        "reset_sampler": "random80_adaptive20",
        "walltime": "15:59:00",
    },
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
echo "[RECORDED] ${record_path}"
