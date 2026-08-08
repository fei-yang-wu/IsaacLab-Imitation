#!/usr/bin/env bash
set -euo pipefail

# Three latent bottlenecks at SONIC's stride-5 macro window, one scaled recipe.
#
# Each arm is a DiffSR pretrain followed by an afterok-dependent 5B controller.
# The ONLY difference between arms is --latent_mode:
#
#   det64      deterministic continuous 64-D code
#   fsq64      sonic_fsq, 64 coordinates x 32 levels (the quantizer output IS
#              the command; no learned projection at the boundary)
#   gumbel64   gumbel_multicat, 8 groups x 32 categories, annealed temperature
#
# All three publish a 66-wide actor command (64 code + 2 sin/cos phase), hold it
# 10 control steps, and read the same 380-wide root_qpos macro state.
#
# Stride 5 is the delta against 2026-08-06-bones129k-sonic-fsq-scale. The macro
# window keeps 10 slots but spaces them 5 reference frames apart, so at 50 Hz it
# spans 0.9 s instead of 0.18 s -- SONIC's released `dt_future_ref_frames=0.1`
# cadence -- and the DiffSR endpoint target moves from s[t+10] to s[t+50]. The
# macro state's WIDTH is identical at both strides, so a mispaired encoder
# cannot be caught by a shape check: `env.expert_macro_frame_stride` is recorded
# into the skill checkpoint at pretrain and the low level refuses an encoder
# trained at a different stride.

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
LATENT_ARMS="${LATENT_ARMS:-det64 fsq64 gumbel64}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_latent_mode_stride5}"

# -- the axis under test and the window it is tested on --
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=5
Z_DIM=64
LATENT_COMMAND_DIM=$((Z_DIM + 2))
GUMBEL_GROUPS=8
GUMBEL_CATEGORIES=32
GUMBEL_TAU_START=2.0
GUMBEL_TAU_END=0.5
# 20% of the pretrain budget; the default 2000 would finish annealing in the
# first 4% and leave the codebook effectively hard for the rest of the run.
GUMBEL_TAU_ANNEAL_ITERS=10000

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
WANDB_GROUP="${WANDB_GROUP:-latent-mode-stride5}"
GPU_GRES="${GPU_GRES:-gpu:h200:1}"

# The 14 bodies SONIC's released tokenizer tracks; also this campaign's cache.
RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

# Every job -- pretrain and low level -- must declare the same macro interface
# AND the same macro cadence, or the encoder is paired with a window it never
# saw. The stride check fails loudly at low-level startup; this array is what
# keeps both sides honest in the first place.
MACRO_INTERFACE_OVERRIDES=(
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    "env.expert_macro_frame_stride=${MACRO_FRAME_STRIDE}"
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

# Scaled actor/critic, identical for all three arms: the latent mode is the
# only axis, so capacity must not move with it.
TRACKER_CELLS="[2048,2048,1024,1024,512,512]"

source_contract_hash() {
    sha256sum \
        "${SCRIPT_DIR}/run.sh" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_encoder.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_diffsr.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/ipmd/module.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/scripts/rlopt/train_hl_skill_diffsr.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/agents/rlopt_ipmd_cfg.py" \
        | sha256sum | awk '{print $1}'
}

encoder_dir() {
    printf '%s/%s_encoder' "${OUTPUT_ROOT}" "$1"
}

lowlevel_dir() {
    printf '%s/%s_tracker/rlopt_train' "${OUTPUT_ROOT}" "$1"
}

# The latent-mode deltas. Everything else is shared by construction.
append_latent_mode() {
    # Distinct nameref name: this is called from a function that already holds a
    # nameref, and bash refuses a nameref that resolves to itself.
    local -n latent_cmd_ref="$1"
    case "$2" in
        det64)
            latent_cmd_ref+=(--latent_mode deterministic)
            ;;
        fsq64)
            latent_cmd_ref+=(--latent_mode sonic_fsq)
            ;;
        gumbel64)
            latent_cmd_ref+=(
                --latent_mode gumbel_multicat
                --categorical_groups "${GUMBEL_GROUPS}"
                --categorical_categories "${GUMBEL_CATEGORIES}"
                --gumbel_tau_start "${GUMBEL_TAU_START}"
                --gumbel_tau_end "${GUMBEL_TAU_END}"
                --gumbel_tau_anneal_iters "${GUMBEL_TAU_ANNEAL_ITERS}"
                --gumbel_hard
            )
            ;;
        *) fail "unknown latent arm $2" ;;
    esac
}

append_pretrain_contract() {
    local -n command_ref="$1"
    local arm="$2"
    command_ref+=(
        --horizon_steps "${HORIZON_STEPS}"
        --encoder_window_mode intermediate
        --transition_objective endpoint
        --z_dim "${Z_DIM}"
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
    append_latent_mode "$1" "$2"
}

append_lowlevel_contract() {
    local -n command_ref="$1"
    local checkpoint_path="$2"
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
    local arm="$1"
    local output_dir="$2"
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
    append_pretrain_contract LOCAL_PRETRAIN_CMD "${arm}"
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
        "agent.logger.exp_name=smoke_${arm}"
    )
    append_lowlevel_contract LOCAL_LOWLEVEL_CMD "${checkpoint_path}"
}

build_cluster_pretrain_command() {
    local arm="$1"
    PRETRAIN_CMD=(
        ./docker/cluster/cluster_interface.sh -c ice_runtime job
        --task "${TASK}" --num_envs "${PRETRAIN_NUM_ENVS}" --seed "${SEED}"
        --device cuda:0 --headless --assert-kitless
        --output_dir "$(encoder_dir "${arm}")"
        --logger_backend wandb --wandb_project "${WANDB_PROJECT}"
        --wandb_group "${WANDB_GROUP}"
        --wandb_run_name "bones129k_stride5_${arm}_pretrain_seed${SEED}"
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
    append_pretrain_contract PRETRAIN_CMD "${arm}"
}

build_cluster_lowlevel_command() {
    local arm="$1"
    local checkpoint_path="$(encoder_dir "${arm}")/checkpoints/latest.pt"
    local run_tag="bones129k_stride5_${arm}_tracker_5b_seed${SEED}"
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
    append_lowlevel_contract LOWLEVEL_CMD "${checkpoint_path}"
}

# The smoke is the real gate on the stride plumbing: it pretrains at stride 5
# and then loads that encoder into a low level at stride 5. A regression that
# drops the stride on either side surfaces here as the pairing error, not as a
# quietly wrong 5B run.
check_local_smoke() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}."
    [[ -f "${LOCAL_SMOKE_ROOT}/status.json" ]] || fail "missing local smoke marker"
    local expected status got arms
    expected="$(source_contract_hash)"
    read -r status got arms < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""), ",".join(d.get("arms", [])))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    local arm
    for arm in ${LATENT_ARMS}; do
        [[ ",${arms}," == *",${arm},"* ]] || fail "local smoke did not cover arm ${arm}"
    done
    echo "[PASS] local smoke covers ${arms} at source contract ${expected:0:16}…"
}

check_remote_gates() {
    local arrays_remote rows path arm
    arrays_remote="$(remote_of "${REF_ARRAYS}")"
    rows="$(ssh_ice "python3 -c \"
import json
d=json.load(open('${arrays_remote}/reference_arrays_manifest.json'))
t=d['traj_info']; k=d['key']
print(len(t['ordered_traj_list']), t['written'], k['source']['persist_id'])
\"")" || fail "remote reference arrays unavailable"
    [[ "${rows}" == "129785 47491234 ${PERSIST_ID}" ]] \
        || fail "remote reference identity mismatch: ${rows}"
    for arm in ${LATENT_ARMS}; do
        for path in "$(encoder_dir "${arm}")" "$(lowlevel_dir "${arm}")"; do
            path="$(remote_of "${path}")"
            [[ "$(ssh_ice "if [[ -e '${path}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
                || fail "refusing to overwrite ${path}"
        done
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

echo "[INFO] mode=${MODE}; latent arms=${LATENT_ARMS}"
echo "[INFO] macro window: root-qpos 380, ${HORIZON_STEPS} slots at stride ${MACRO_FRAME_STRIDE} (0.9 s), endpoint objective"
echo "[INFO] scaled DiffSR + tracker ${TRACKER_CELLS}; ${FRAME_CAP} frames per controller on ${GPU_GRES}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_latent_mode_stride5_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}"
    smoked_arms=()
    for arm in ${LATENT_ARMS}; do
        build_local_pretrain_command "${arm}" "${LOCAL_SMOKE_ROOT}/${arm}/encoder"
        print_cmd "${LOCAL_PRETRAIN_CMD[@]}"
        mkdir -p "${LOCAL_SMOKE_ROOT}/${arm}"
        "${LOCAL_PRETRAIN_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/${arm}/pretrain.log"
        local_checkpoint="${LOCAL_SMOKE_ROOT}/${arm}/encoder/checkpoints/latest.pt"
        [[ -f "${local_checkpoint}" ]] || fail "smoke pretrain did not write ${local_checkpoint}"
        # Through Pixi: the system python3 used elsewhere in this script has no
        # torch, and reading the checkpoint is the whole point of this gate.
        pixi run -q python - "${local_checkpoint}" "${MACRO_FRAME_STRIDE}" <<'PY'
import sys, torch

path, expected = sys.argv[1], int(sys.argv[2])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
recorded = int(checkpoint["config"].get("macro_frame_stride", 1))
if recorded != expected:
    raise SystemExit(
        f"[FATAL] {path} recorded macro_frame_stride={recorded}, expected {expected}"
    )
print(f"[PASS] {path} records macro_frame_stride={recorded}")
PY
        build_local_lowlevel_command "${arm}" "${local_checkpoint}" \
            "${LOCAL_SMOKE_ROOT}/${arm}/tracker"
        print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
        mkdir -p "${LOCAL_SMOKE_ROOT}/${arm}/tracker"
        "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/${arm}/tracker/train.log"
        smoked_arms+=("${arm}")
    done
    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${MACRO_FRAME_STRIDE}" "${smoked_arms[@]}" <<'PY'
import json, sys

path, source_hash, stride, *arms = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "macro_frame_stride": int(stride),
            "arms": arms,
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

for arm in ${LATENT_ARMS}; do
    build_cluster_pretrain_command "${arm}"
    echo "[PRETRAIN ${arm}]"
    print_cmd "${PRETRAIN_CMD[@]}"
    build_cluster_lowlevel_command "${arm}"
    echo "[LOWLEVEL ${arm} afterok:<${arm}-pretrain-job>]"
    print_cmd "${LOWLEVEL_CMD[@]}"
done

if [[ "${MODE}" != "submit" ]]; then
    echo "[INFO] MODE=${MODE}; no scheduler mutation."
    exit 0
fi

submission_records=()
for arm in ${LATENT_ARMS}; do
    build_cluster_pretrain_command "${arm}"
    export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train_hl_skill_diffsr.py
    export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT:-15:59:00}"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-s5-${arm}-pre"
    export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride5,${arm},scaled-pretrain,endpoint"
    unset CLUSTER_SLURM_DEPENDENCY
    pretrain_job="$(submit_and_capture_job_id "${PRETRAIN_CMD[@]}")"
    echo "[SUBMITTED] ${arm} pretrain=${pretrain_job}"

    build_cluster_lowlevel_command "${arm}"
    export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
    export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
    export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-s5-${arm}"
    export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride5,${arm},lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
    export CLUSTER_SLURM_DEPENDENCY="afterok:${pretrain_job}"
    lowlevel_job="$(submit_and_capture_job_id "${LOWLEVEL_CMD[@]}")"
    echo "[SUBMITTED] ${arm} tracker=${lowlevel_job} afterok:${pretrain_job}"
    submission_records+=("${arm}:${pretrain_job}:${lowlevel_job}")
done

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" \
    "${FRAME_CAP}" "${MACRO_FRAME_STRIDE}" "${HORIZON_STEPS}" \
    "${submission_records[@]}" <<'PY'
import json, sys

(
    path,
    submitted,
    source_hash,
    project,
    group,
    frame_cap,
    stride,
    horizon,
    *records,
) = sys.argv[1:]
latent = {
    "det64": "deterministic_continuous_z64",
    "fsq64": "sonic_fsq64_levels32",
    "gumbel64": "gumbel_multicat_g8_c32",
}
arms = []
for record in records:
    arm, pretrain, lowlevel = record.split(":")
    arms.append(
        {
            "arm": arm,
            "latent": latent[arm],
            "pretrain_job_id": int(pretrain),
            "lowlevel_job_id": int(lowlevel),
            "dependency": f"afterok:{pretrain}",
        }
    )
payload = {
    "campaign": "2026-08-07-bones129k-latent-mode-stride5",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "arms": arms,
    "shared_pretrain": {
        "macro_interface": "root_qpos",
        "encoder_input_dim": 380,
        "macro_frame_stride": int(stride),
        "horizon_steps": int(horizon),
        "macro_window_seconds": round(int(stride) * (int(horizon) - 1) / 50.0, 3),
        "encoder_window_mode": "intermediate",
        "transition_objective": "endpoint",
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
    "common": {
        "task": "Isaac-Imitation-G1-v2",
        "motions": 129785,
        "z_dim": 64,
        "command_dim": 66,
        "hold_steps": 10,
        "tracker_num_cells": [2048, 2048, 1024, 1024, 512, 512],
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
