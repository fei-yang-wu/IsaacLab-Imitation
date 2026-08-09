#!/usr/bin/env bash
set -euo pipefail

# One variable: the frame the DiffSR macro window is expressed in.
#
# Control is the existing ICE job 5567801 (`reset80_diffsr`, W&B group
# `bones129k-ablation`): frozen root-qpos DiffSR, z256 + sin/cos phase, h10,
# hold 10, macro stride 1, pretrained encoder sha256 d191d865…f8c5e7. Nothing
# is resubmitted for it.
#
# This campaign submits ONE new arm, `expert_heading`: a fresh encoder pretrain
# whose command line is byte-identical to the control's except for
# `env.expert_macro_anchor_mode=expert_heading`, plus an afterok-dependent
# controller whose overrides are byte-identical to the control arm's except for
# the same flag.
#
# What the flag changes
# --------------------
# The historical "robot" convention splits the macro window across two frames:
#
#   pretrain  anchor = the EXPERT's full pose at window slot 0
#             -> slot 0 is always (0,0,0) + identity, in every sample
#   rollout   anchor = the ROBOT's full live anchor pose
#             -> slot 0 is the live tracking error, nonzero and growing
#
# So a frozen encoder is queried off its pretraining manifold exactly when
# tracking is worst, and being frozen it never adapts. "expert_heading" uses
# ONE convention for both: the expert's slot-0 heading (yaw-only, swing-twist)
# frame with an xy-only origin. Pretrain and rollout inputs then match by
# construction, and the latent becomes a pure function of (trajectory, cursor).
#
# The frame cancels global yaw and xy and nothing else, so absolute height and
# roll/pitch relative to gravity survive -- exactly the invariance group of the
# re-rooted tracking reward (`reroot_body_positions`: heading-only delta, xy
# from the robot, z from the reference).
#
# Declared risk, accepted by the user: the actor input keys are UNCHANGED, so
# no observation term replaces the drift signal the robot-anchored latent used
# to carry implicitly. The policy therefore cannot observe xy/yaw drift.
# Expect MPJPE-L to hold and MPJPE-G / push recovery to degrade; that is the
# measurement, not a defect. Keeping the actor contract fixed is what makes
# this arm single-variable against 5567801.
#
# The macro state is 380 wide in BOTH modes, so a mispaired encoder cannot be
# caught by a shape check: `env.expert_macro_anchor_mode` is recorded into the
# skill checkpoint at pretrain time and the low level refuses an encoder
# trained under a different mode. The local smoke asserts the recording and
# exercises the refusal.

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
ARM="${ARM:-expert_heading}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_anchor_frame}"

# The control's encoder, for the provenance line in the record and for the
# smoke's negative pairing check. Never loaded by this campaign's arm.
CONTROL_ENCODER_SHA256="d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7"
LOCAL_CONTROL_ENCODER="${LOCAL_CONTROL_ENCODER:-/mnt/storage/fwu91/bones_seed_full/runs/bones129k_root_qpos_v2_splitcache_e24576_r6_1b_seed0/encoder/checkpoints/latest.pt}"

# -- the axis under test --
ANCHOR_MODE="expert_heading"

# -- everything below is the control's contract, reproduced exactly --
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=1
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# The control encoder used the trainer's defaults; they are passed explicitly
# here so a future default change cannot silently move this arm off the
# control's recipe. Verified against the control checkpoint's recorded config
# (encoder_hidden_dims [1024,512,512], mish, LayerNorm on, feature 128,
# embed 512, g/mu [512], 50k updates at batch 8192, endpoint objective).
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-100}"

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

# Both jobs must declare the same macro interface, the same cadence AND the
# same frame. Two of the three fail loudly at low-level startup; this array is
# what keeps them consistent in the first place.
MACRO_INTERFACE_OVERRIDES=(
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    "env.expert_macro_frame_stride=${MACRO_FRAME_STRIDE}"
    "env.expert_macro_anchor_mode=${ANCHOR_MODE}"
)

# Byte-identical to the control arm's common overrides.
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
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_diffsr.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/RLOpt/rlopt/env_interface.py" \
        "${REPO_ROOT}/scripts/rlopt/train_hl_skill_diffsr.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/_compiled.py" \
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
        --encoder_hidden_dims 1024 512 512
        --encoder_activation mish --encoder_layer_norm
        --diffsr_feature_dim 128 --diffsr_embed_dim 512
        --diffsr_g_hidden_dims 512
        --diffsr_mu_hidden_dims 512
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
    # Deliberately NO agent.policy.num_cells / agent.value_function.num_cells:
    # the control took the tuned entry point's tracker capacity, so this arm
    # must too.
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
    local anchor_override="${3:-${ANCHOR_MODE}}"
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
    # The negative pairing check overrides the mode AFTER the contract block,
    # so Hydra's last-wins ordering makes it the effective value.
    if [[ "${anchor_override}" != "${ANCHOR_MODE}" ]]; then
        LOCAL_LOWLEVEL_CMD+=("env.expert_macro_anchor_mode=${anchor_override}")
    fi
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
    local run_tag="bones129k_${ARM}_tracker_seed${SEED}"
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
    local expected status got mode
    expected="$(source_contract_hash)"
    read -r status got mode < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""), d.get("anchor_mode", ""))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${mode}" == "${ANCHOR_MODE}" ]] || fail "local smoke covered mode ${mode}"
    echo "[PASS] local smoke covers ${mode} at source contract ${expected:0:16}…"
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

echo "[INFO] mode=${MODE}; arm=${ARM}; anchor mode=${ANCHOR_MODE}"
echo "[INFO] control = ICE 5567801 (bones129k-ablation), robot anchor, encoder ${CONTROL_ENCODER_SHA256:0:16}…"
echo "[INFO] macro window: root-qpos 380, ${HORIZON_STEPS} slots at stride ${MACRO_FRAME_STRIDE}, endpoint objective"
echo "[INFO] z${Z_DIM} + sin/cos phase = ${LATENT_COMMAND_DIM}; hold 10; tuned tracker capacity (no override)"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_anchor_frame_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}"

    build_local_pretrain_command "${LOCAL_SMOKE_ROOT}/encoder"
    print_cmd "${LOCAL_PRETRAIN_CMD[@]}"
    "${LOCAL_PRETRAIN_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/pretrain.log"
    local_checkpoint="${LOCAL_SMOKE_ROOT}/encoder/checkpoints/latest.pt"
    [[ -f "${local_checkpoint}" ]] || fail "smoke pretrain did not write ${local_checkpoint}"

    # Gate 1: the checkpoint records the mode. Without this the pairing guard
    # has nothing to compare and the whole detection story is vacuous.
    pixi run -q python - "${local_checkpoint}" "${ANCHOR_MODE}" "${MACRO_FRAME_STRIDE}" <<'PY'
import sys, torch

path, expected_mode, expected_stride = sys.argv[1], sys.argv[2], int(sys.argv[3])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint["config"]
mode = str(config.get("macro_anchor_mode", "robot"))
stride = int(config.get("macro_frame_stride", 1))
if mode != expected_mode:
    raise SystemExit(f"[FATAL] {path} recorded macro_anchor_mode={mode!r}, expected {expected_mode!r}")
if stride != expected_stride:
    raise SystemExit(f"[FATAL] {path} recorded macro_frame_stride={stride}, expected {expected_stride}")
weight = checkpoint["skill_encoder_state_dict"]["net.0.weight"]
if int(weight.shape[1]) != 380:
    raise SystemExit(f"[FATAL] encoder input width {int(weight.shape[1])} != 380 (root_qpos)")
print(f"[PASS] {path} records macro_anchor_mode={mode!r}, stride={stride}, width=380")
PY

    build_local_lowlevel_command "${local_checkpoint}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # Gate 2: the negative check. An expert_heading encoder driven by a robot
    # environment must REFUSE, because 380 is 380 in both modes and nothing
    # downstream can catch it.
    build_local_lowlevel_command "${local_checkpoint}" \
        "${LOCAL_SMOKE_ROOT}/tracker_negative" robot
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker_negative"
    if "${LOCAL_LOWLEVEL_CMD[@]}" > "${LOCAL_SMOKE_ROOT}/tracker_negative/train.log" 2>&1; then
        fail "mismatched anchor mode was ACCEPTED; the pairing guard is not wired"
    fi
    grep -q "macro-window anchor mode does not match" \
        "${LOCAL_SMOKE_ROOT}/tracker_negative/train.log" \
        || fail "mismatch failed for the wrong reason; see ${LOCAL_SMOKE_ROOT}/tracker_negative/train.log"
    echo "[PASS] mismatched anchor mode refused at pairing"

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${ANCHOR_MODE}" "${MACRO_FRAME_STRIDE}" <<'PY'
import json, sys

path, source_hash, anchor_mode, stride = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "anchor_mode": anchor_mode,
            "macro_frame_stride": int(stride),
            "negative_pairing_check": "refused",
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

build_cluster_pretrain_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train_hl_skill_diffsr.py
export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-anch-pre"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,expert-heading-anchor,pretrain,endpoint"
unset CLUSTER_SLURM_DEPENDENCY
pretrain_job="$(submit_and_capture_job_id "${PRETRAIN_CMD[@]}")"
echo "[SUBMITTED] ${ARM} pretrain=${pretrain_job}"

build_cluster_lowlevel_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-anch"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,hold10,expert-heading-anchor,lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
export CLUSTER_SLURM_DEPENDENCY="afterok:${pretrain_job}"
lowlevel_job="$(submit_and_capture_job_id "${LOWLEVEL_CMD[@]}")"
echo "[SUBMITTED] ${ARM} tracker=${lowlevel_job} afterok:${pretrain_job}"

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" \
    "${ARM}" "${ANCHOR_MODE}" "${pretrain_job}" "${lowlevel_job}" \
    "${FRAME_CAP}" "${MACRO_FRAME_STRIDE}" "${HORIZON_STEPS}" "${Z_DIM}" \
    "${CONTROL_ENCODER_SHA256}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm, anchor_mode,
    pretrain, lowlevel, frame_cap, stride, horizon, z_dim, control_encoder,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-08-bones129k-anchor-frame",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "axis": {
        "field": "env.expert_macro_anchor_mode",
        "arm_value": anchor_mode,
        "control_value": "robot",
    },
    "control": {
        "job_id": 5567801,
        "arm": "reset80_diffsr",
        "wandb_group": "bones129k-ablation",
        "encoder_sha256": control_encoder,
        "note": "not resubmitted; this campaign adds one arm against it",
    },
    "arm": {
        "arm": arm,
        "pretrain_job_id": int(pretrain),
        "lowlevel_job_id": int(lowlevel),
        "dependency": f"afterok:{pretrain}",
    },
    "pretrain": {
        "macro_interface": "root_qpos",
        "encoder_input_dim": 380,
        "macro_frame_stride": int(stride),
        "macro_anchor_mode": anchor_mode,
        "horizon_steps": int(horizon),
        "z_dim": int(z_dim),
        "latent_mode": "deterministic",
        "encoder_window_mode": "intermediate",
        "transition_objective": "endpoint",
        "encoder_hidden_dims": [1024, 512, 512],
        "encoder_activation": "mish",
        "encoder_layer_norm": True,
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
        "command_dim": int(z_dim) + 2,
        "hold_steps": 10,
        "tracker_num_cells": "tuned entry point default (not overridden)",
        "num_envs": 16384,
        "rollout_steps": 24,
        "mini_batch_size": 294912,
        "gamma": 0.97,
        "seed": 0,
        "frame_cap": int(frame_cap),
        "checkpoint_interval_frames": 50000000,
        "reset_sampler": "random80_adaptive20",
    },
    "actor_contract_note": (
        "Actor input keys unchanged from the control: no anchor-delta "
        "observation replaces the drift signal the robot-anchored latent "
        "carried implicitly. Deliberate, so the arm stays single-variable."
    ),
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
echo "[RECORDED] ${record_path}"
