#!/usr/bin/env bash
set -euo pipefail

# 2026-08-08 -- SONIC-FSQ latent, scaled nets, expert-heading macro frame, and a
# critic WITHOUT the actor latent, all in one arm.
#
# This is a COMBINED arm, not a single-variable ablation. Each ingredient came
# from a separate 2026-08 campaign and each was measured alone; this run asks
# whether they compose. State that plainly in any comparison.
#
#   1. SONIC-FSQ bottleneck   `--latent_mode sonic_fsq`, 64 coordinates x 32
#                             levels. The quantizer output IS the command, so
#                             `--z_dim` must equal 64 and the published command
#                             is 64 code + 2 sin/cos phase = 66 wide.
#                             From 2026-08-07-bones129k-latent-mode-stride5.
#   2. Scaled capacity        encoder [2048, 1024, 512, 512] SiLU without layer
#                             norm, DiffSR feature 256 / embed 1024 with
#                             [1024, 1024, 512] heads, and tracker actor+critic
#                             [2048, 2048, 1024, 1024, 512, 512] SiLU.
#                             From the same stride-5 campaign.
#   3. Expert-heading frame   `env.expert_macro_anchor_mode=expert_heading`:
#                             pretrain and rollout macro windows share ONE
#                             frame (the expert slot-0 yaw-only heading frame,
#                             xy-only origin), so a frozen encoder is never
#                             queried off its pretraining manifold.
#                             From 2026-08-08-bones129k-anchor-frame (ICE
#                             5573233 -> 5573234).
#   4. Critic without latent  `env.command_interface.critic_channels=[reference]`:
#                             the critic keeps the noise-free reference channel
#                             plus privileged state and drops the actor latent.
#                             The actor is untouched.
#                             From 2026-08-08-bones129k-anchor-critic-no-latent.
#
# MACRO STRIDE IS 1, by explicit user choice on 2026-08-08. The scaled fsq64 row
# in the stride-5 campaign used stride 5 (0.9 s window); this arm keeps the
# stride-1 (0.18 s) window of the running expert-heading arm instead, so the
# expert-heading and critic ingredients stay comparable to the arms that
# measured them. The macro state width is 380 at EITHER stride, so a mispaired
# encoder cannot be caught by a shape check -- the stride and the anchor mode
# are both recorded into the skill checkpoint at pretrain time and the low level
# refuses a mismatch.
#
#   MODE=print   ./run.sh
#   MODE=smoke   ./run.sh                     # local pretrain + tracker gate
#   MODE=validate LOCAL_SMOKE_ROOT=<dir> ./run.sh
#   MODE=submit  LOCAL_SMOKE_ROOT=<dir> CONFIRM_SUBMIT=fsq-anchor-critic ./run.sh

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
ARM="${ARM:-fsq64_heading_critic_no_latent}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_fsq_anchor_critic}"

# -- the macro window --
ANCHOR_MODE="expert_heading"
HORIZON_STEPS=10
MACRO_FRAME_STRIDE="${MACRO_FRAME_STRIDE:-1}"

# -- the bottleneck: sonic_fsq publishes the quantizer output, so Z_DIM must
# -- equal the number of FSQ coordinates (64 x 32 levels is the trainer default)
Z_DIM=64
FSQ_LEVELS=32
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# Actor: 93 non-command inputs + the 66-wide latent command.
# Critic: reference channel + privileged state, no latent command.
EXPECTED_ACTOR_INPUT_DIM=159
EXPECTED_CRITIC_INPUT_DIM=286

# -- scaled capacity, shared by the encoder/DiffSR and the tracker --
ENCODER_HIDDEN_DIMS=(2048 1024 512 512)
DIFFSR_FEATURE_DIM=256
DIFFSR_EMBED_DIM=1024
DIFFSR_HEAD_DIMS=(1024 1024 512)
TRACKER_CELLS="[2048,2048,1024,1024,512,512]"

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
# 250M, not the 50M of the anchor campaigns: this arm must produce a checkpoint
# exactly at the 5B frame mark that the 4096-motion scoreboard scores, and 250M
# divides 5B evenly.
SAVE_INTERVAL="${SAVE_INTERVAL:-250000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-4}"
SMOKE_ROLLOUT_STEPS="${SMOKE_ROLLOUT_STEPS:-4}"
SMOKE_MINIBATCH_SIZE="${SMOKE_MINIBATCH_SIZE:-8}"
SMOKE_EXPERT_BATCH_SIZE="${SMOKE_EXPERT_BATCH_SIZE:-8}"
LOCAL_SMOKE_ROOT="${LOCAL_SMOKE_ROOT:-}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-fsq-anchor-critic}"
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

# Both jobs must declare the same macro terms, cadence, AND anchor frame, or the
# encoder is paired with a window it never saw. All three are recorded into the
# skill checkpoint and re-checked at low-level startup.
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

CRITIC_CHANNELS_OVERRIDE="env.command_interface.critic_channels=[reference]"

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

encoder_dir() { printf '%s/%s_encoder' "${OUTPUT_ROOT}" "${ARM}"; }
lowlevel_dir() { printf '%s/%s_tracker/rlopt_train' "${OUTPUT_ROOT}" "${ARM}"; }

append_pretrain_contract() {
    local -n command_ref="$1"
    command_ref+=(
        --horizon_steps "${HORIZON_STEPS}"
        --encoder_window_mode intermediate
        --transition_objective endpoint
        --z_dim "${Z_DIM}"
        --latent_mode sonic_fsq
        --sonic_fsq_levels $(for _ in $(seq "${Z_DIM}"); do printf '%s ' "${FSQ_LEVELS}"; done)
        --encoder_hidden_dims "${ENCODER_HIDDEN_DIMS[@]}"
        --encoder_activation silu --no_encoder_layer_norm
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
        "agent.policy.num_cells=${TRACKER_CELLS}"
        agent.policy.activation_fn=silu
        "agent.value_function.num_cells=${TRACKER_CELLS}"
        agent.value_function.activation_fn=silu
        "${COMMON_ENV_OVERRIDES[@]}"
        "${CRITIC_CHANNELS_OVERRIDE}"
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
        "${MACRO_INTERFACE_OVERRIDES[@]}"
    )
    append_pretrain_contract LOCAL_PRETRAIN_CMD
    LOCAL_PRETRAIN_CMD+=(
        --batch_size 8 --num_updates 1 --log_interval 1 --eval_batches 1
        --eval_batch_size 8 --window_probe_train_batches 1
        --window_probe_eval_batches 1
    )
}

# `extra_overrides` exists for exactly one caller: the negative gate, which
# reruns this command with the WRONG anchor mode and requires a refusal.
build_local_lowlevel_command() {
    local checkpoint_path="$1"
    local output_dir="$2"
    shift 2
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
    LOCAL_LOWLEVEL_CMD+=("$@")
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
    local expected status got stride mode channels
    expected="$(source_contract_hash)"
    read -r status got stride mode channels < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    d.get("status", ""),
    d.get("source_contract_sha256", ""),
    d.get("macro_frame_stride", ""),
    d.get("macro_anchor_mode", ""),
    d.get("critic_channels", ""),
)
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${stride}" == "${MACRO_FRAME_STRIDE}" ]] || fail "smoke stride ${stride}"
    [[ "${mode}" == "${ANCHOR_MODE}" ]] || fail "smoke anchor mode ${mode}"
    [[ "${channels}" == "reference" ]] || fail "smoke critic channels ${channels}"
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
echo "[INFO] COMBINED arm: sonic_fsq ${Z_DIM}x${FSQ_LEVELS} + scaled nets + ${ANCHOR_MODE} frame + critic [reference]"
echo "[INFO] macro window: root-qpos 380, ${HORIZON_STEPS} slots at stride ${MACRO_FRAME_STRIDE}, endpoint objective"
echo "[INFO] command ${LATENT_COMMAND_DIM} (=${Z_DIM} code + 2 phase), held 10 control steps"
echo "[INFO] actor input ${EXPECTED_ACTOR_INPUT_DIM}; critic input ${EXPECTED_CRITIC_INPUT_DIM} (no latent_command)"
echo "[INFO] tracker ${TRACKER_CELLS}; encoder ${ENCODER_HIDDEN_DIMS[*]} silu, no layer norm"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] ICE allocations end at 16 h, so the tracker is EXPECTED to TIMEOUT before the cap;"
echo "[INFO] checkpoints land every ${SAVE_INTERVAL} frames under persistent /data, not node-local storage."
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_fsq_anchor_critic_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"

    build_local_pretrain_command "${LOCAL_SMOKE_ROOT}/encoder"
    print_cmd "${LOCAL_PRETRAIN_CMD[@]}"
    "${LOCAL_PRETRAIN_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/pretrain.log"
    SMOKE_CHECKPOINT="${LOCAL_SMOKE_ROOT}/encoder/checkpoints/latest.pt"
    [[ -f "${SMOKE_CHECKPOINT}" ]] || fail "smoke pretrain did not write ${SMOKE_CHECKPOINT}"

    # Gate 1: the checkpoint must record every pairing-critical field. The macro
    # state is 380 wide under both anchor modes and both strides, so nothing
    # downstream can catch a mismatch by shape.
    pixi run -q python - "${SMOKE_CHECKPOINT}" "${ANCHOR_MODE}" "${MACRO_FRAME_STRIDE}" \
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
    "latent_mode": (str(config["latent_mode"]), "sonic_fsq"),
}
for field, (got, want) in checks.items():
    if got != want:
        raise SystemExit(f"[FATAL] encoder {field}={got!r}, expected {want!r}")
width = int(checkpoint["skill_encoder_state_dict"]["net.0.weight"].shape[1])
if width != 380:
    raise SystemExit(f"[FATAL] encoder input width {width} != 380 (root_qpos)")
print(f"[PASS] encoder: sonic_fsq, {mode}, stride {stride}, h{horizon}, z{z_dim}, width 380")
PY

    build_local_lowlevel_command "${SMOKE_CHECKPOINT}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # Gate 2: the critic must have lost latent_command and NOTHING else, and the
    # actor must still have it. A width check alone would pass if some other term
    # silently changed too.
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
if "latent_command" not in policy_terms:
    raise SystemExit("[FATAL] actor lost latent_command; only the critic may change")
if int(policy_terms["latent_command"]) != command_dim:
    raise SystemExit(
        f"[FATAL] latent_command is {policy_terms['latent_command']}, expected {command_dim}"
    )
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
print(f"[PASS] actor {actor_width} with a {command_dim}-wide command; critic {critic_width}, no latent_command")
PY

    # Gate 3, negative: the SAME encoder against a `robot`-frame environment must
    # be refused. Without this, gate 1 only proves the field was written, not
    # that anything reads it.
    build_local_lowlevel_command "${SMOKE_CHECKPOINT}" \
        "${LOCAL_SMOKE_ROOT}/tracker_mismatch" env.expert_macro_anchor_mode=robot
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker_mismatch"
    if "${LOCAL_LOWLEVEL_CMD[@]}" > "${LOCAL_SMOKE_ROOT}/tracker_mismatch/train.log" 2>&1; then
        fail "mismatched anchor mode was ACCEPTED; the pairing check is dead"
    fi
    grep -q "anchor mode does not match" "${LOCAL_SMOKE_ROOT}/tracker_mismatch/train.log" \
        || fail "mismatched run failed for the wrong reason; see ${LOCAL_SMOKE_ROOT}/tracker_mismatch/train.log"
    echo "[PASS] a robot-frame environment is refused for this encoder"

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${MACRO_FRAME_STRIDE}" "${ANCHOR_MODE}" <<'PY'
import json, sys

path, source_hash, stride, mode = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "macro_frame_stride": int(stride),
            "macro_anchor_mode": mode,
            "latent_mode": "sonic_fsq",
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

COMMON_TAGS="bones-seed,129785,v2,root-qpos,stride${MACRO_FRAME_STRIDE},z${Z_DIM},sonic-fsq${Z_DIM},hold10,expert-heading-anchor,critic-no-latent,scaled-nets"

build_cluster_pretrain_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train_hl_skill_diffsr.py
export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-fsq-ac-pre"
export CLUSTER_WANDB_TAGS="${COMMON_TAGS},pretrain,endpoint"
unset CLUSTER_SLURM_DEPENDENCY
pretrain_job="$(submit_and_capture_job_id "${PRETRAIN_CMD[@]}")"
echo "[SUBMITTED] ${ARM} pretrain=${pretrain_job}"

build_cluster_lowlevel_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-fsq-ac"
export CLUSTER_WANDB_TAGS="${COMMON_TAGS},lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
export CLUSTER_SLURM_DEPENDENCY="afterok:${pretrain_job}"
lowlevel_job="$(submit_and_capture_job_id "${LOWLEVEL_CMD[@]}")"
echo "[SUBMITTED] ${ARM} tracker=${lowlevel_job} afterok:${pretrain_job}"

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" "${ARM}" \
    "${pretrain_job}" "${lowlevel_job}" "${MACRO_FRAME_STRIDE}" "${HORIZON_STEPS}" \
    "${Z_DIM}" "${FSQ_LEVELS}" "${LATENT_COMMAND_DIM}" "${FRAME_CAP}" \
    "${MAX_ITERATIONS}" "${MINIBATCH_SIZE}" "${SAVE_INTERVAL}" \
    "${EXPECTED_ACTOR_INPUT_DIM}" "${EXPECTED_CRITIC_INPUT_DIM}" "${SEED}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm, pretrain, lowlevel,
    stride, horizon, z_dim, fsq_levels, command_dim, frame_cap, iterations,
    minibatch, save_interval, actor_dim, critic_dim, seed,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-08-bones129k-fsq-anchor-critic",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "design": {
        "kind": "combined arm, not a single-variable ablation",
        "ingredients": {
            "latent_mode": "sonic_fsq (from 2026-08-07-bones129k-latent-mode-stride5)",
            "capacity": "scaled encoder/DiffSR/tracker (same campaign)",
            "macro_anchor_mode": "expert_heading (from 2026-08-08-bones129k-anchor-frame)",
            "critic_channels": "[reference] (from 2026-08-08-bones129k-anchor-critic-no-latent)",
        },
        "nearest_references": {
            "scaled_fsq64_stride5": "ICE latent-mode-stride5 fsq64 arm (stride 5, full critic)",
            "expert_heading_z256": 5573234,
            "critic_no_latent_z256": "2026-08-06-bones129k-critic-no-latent / 2026-08-08 anchor-critic arm",
        },
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
        "macro_anchor_mode": "expert_heading",
        "horizon_steps": int(horizon),
        "z_dim": int(z_dim),
        "latent_mode": "sonic_fsq",
        "sonic_fsq_levels": [int(fsq_levels)] * int(z_dim),
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
    "lowlevel": {
        "task": "Isaac-Imitation-G1-v2",
        "motions": 129785,
        "command_dim": int(command_dim),
        "hold_steps": 10,
        "actor_input_dim": int(actor_dim),
        "critic_input_dim": int(critic_dim),
        "critic_channels": ["reference"],
        "tracker_num_cells": [2048, 2048, 1024, 1024, 512, 512],
        "tracker_activation": "silu",
        "num_envs": 16384,
        "rollout_steps": 24,
        "mini_batch_size": int(minibatch),
        "gamma": 0.97,
        "seed": int(seed),
        "frame_cap": int(frame_cap),
        "max_iterations": int(iterations),
        "checkpoint_interval_frames": int(save_interval),
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
