#!/usr/bin/env bash
set -euo pipefail

# 2026-08-09 -- a dedicated wrist tracking reward, one variable.
#
# Control is the running ICE job `5573413` (`expert_heading_critic_no_latent`):
# expert-heading macro frame, stride 1, z256, critic channels `[reference]`,
# frozen DiffSR encoder. Nothing is resubmitted for it.
#
# The axis:
#
#   env.rewards.motion_ee_pos.weight=2.0     (control: 0.0, an inert term)
#
# `motion_ee_pos` is rerooted wrist position error against the reference,
# std 0.1, over `G1_WRIST_BODY_NAMES` (the two wrist yaw links). It exists in
# the config today and is deliberately switched off, so this arm turns a term
# ON rather than introducing new reward code.
#
# Why the wrists
# --------------
# Measured 2026-08-08/09 on the frozen 4,096-motion scoreboard (ranks
# 12288-16383, 5B frames, released-SONIC thresholds, `foot_pos_xyz` and
# `base_too_low` disabled), the best arm's failures are almost entirely one
# termination:
#
#   ee_body_pos  7.76%   (318 of 4096 environments)
#   anchor_ori   1.39%
#   anchor_pos   0.42%
#
# `ee_body_pos` is a Z-only height error over `G1_EE_BODY_NAMES`, which is both
# ankles AND both wrists. Rerunning the same evaluation with that termination
# narrowed to one pair at a time:
#
#   wrists only   231 environments   (SR 0.9197)
#   ankles only   208 environments   (SR 0.9182)
#
# Both pairs fail on height, and 231 + 208 > 318, so many environments fail on
# both. This arm is therefore NOT claiming the wrists are the whole story. The
# released SONIC checkpoint fails the same way on the same ranks, 26 times.
#
# The wrists are picked because they are the least-rewarded bodies in the
# contract, so they are where a cheap reward change has room: no termination
# bounds them horizontally, their only positional reward is 2 of 5 points in
# `tracking_reward_points`, and the dedicated term is inert. The feet, by
# contrast, were given `motion_foot_pos` at weight 2.0 for exactly this
# reasoning once `foot_pos_xyz` was found to dominate terminations. This arm
# does the same thing on the hands, at the same weight and the same std.
#
# Declared counter-evidence, so it is not rediscovered as a surprise: an
# earlier screen (s13) added a local wrist term and improved root-relative EE
# error while root drift ROSE and MPJPE-G got 28% worse. That was measured
# before the v2 tuned rewards raised both global anchor terms from 0.5 to 2.0,
# so the drift counterweight is now 4x stronger than it was then. If root drift
# rises again here, that is the result, not a bug.
#
#   MODE=print   ./run.sh
#   MODE=smoke   ./run.sh                     # local 1-iteration gate
#   MODE=validate LOCAL_SMOKE_ROOT=<dir> ./run.sh
#   MODE=submit  LOCAL_SMOKE_ROOT=<dir> CONFIRM_SUBMIT=ee-reward ./run.sh

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
ARM="${ARM:-expert_heading_critic_no_latent_ee_reward}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_ee_reward}"

# -- the axis under test --
EE_REWARD_WEIGHT="${EE_REWARD_WEIGHT:-2.0}"
EE_REWARD_OVERRIDE="env.rewards.motion_ee_pos.weight=${EE_REWARD_WEIGHT}"

CONTROL_LOWLEVEL_JOB=5573413
CONTROL_ARM="expert_heading_critic_no_latent"

# -- the shared encoder, identical to the control's, by path and by SHA-256 --
ENCODER_REMOTE="/data/bones129k_anchor_frame/expert_heading_encoder/checkpoints/latest.pt"
ENCODER_SHA256="be6d533f1d1ca4aa6b1e819af1d3ef63eb033125018c8309c7448384b6a9583e"
LOCAL_ENCODER="${LOCAL_ENCODER:-${REPO_ROOT}/logs/downloaded_checkpoints/bones129k_anchor_frame/expert_heading_encoder_latest.pt}"

# -- everything below reproduces the control arm's contract exactly --
ANCHOR_MODE="expert_heading"
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=1
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))
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
WANDB_GROUP="${WANDB_GROUP:-ee-reward}"
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
    env.command_interface.critic_channels=[reference]
)

source_contract_hash() {
    sha256sum \
        "${SCRIPT_DIR}/run.sh" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_encoder.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/RLOpt/rlopt/env_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/rewards.py" \
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
        "${EE_REWARD_OVERRIDE}"
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
    local expected status got weight
    expected="$(source_contract_hash)"
    read -r status got weight < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""), d.get("motion_ee_pos_weight", ""))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${weight}" == "${EE_REWARD_WEIGHT}" ]] || fail "smoke covered weight ${weight}"
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
echo "[INFO] axis: ${EE_REWARD_OVERRIDE} (control 0.0, inert)"
echo "[INFO] control = ICE ${CONTROL_LOWLEVEL_JOB} (${CONTROL_ARM})"
echo "[INFO] shared frozen encoder ${ENCODER_SHA256:0:16}… (${ANCHOR_MODE}, stride ${MACRO_FRAME_STRIDE}); no pretrain job"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    check_local_encoder
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_ee_reward_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"
    build_local_lowlevel_command "${LOCAL_ENCODER}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # The gate: `motion_ee_pos` must carry the requested weight in the reward
    # table, and no other reward term may have moved. The table lists
    # zero-weight terms too (RewardManager skips computing them but still
    # prints them), so the weight VALUE is the thing to assert, not presence.
    python3 - "${LOCAL_SMOKE_ROOT}/tracker/train.log" "${EE_REWARD_WEIGHT}" \
        "${EXPECTED_ACTOR_INPUT_DIM}" "${EXPECTED_CRITIC_INPUT_DIM}" <<'PY'
import re
import sys

log_path, want_weight, actor_dim, critic_dim = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()

rewards = {}
in_rewards = False
for line in text.splitlines():
    if "Active Reward Terms" in line:
        in_rewards = True
        continue
    if in_rewards:
        row = re.match(r"\|\s*\d+\s*\|\s*([\w]+)\s*\|\s*(-?[\d.eE+]+)\s*\|", line)
        if row:
            rewards[row.group(1)] = float(row.group(2))
        elif line.startswith("[INFO]") and "Reward" not in line:
            in_rewards = False
if not rewards:
    raise SystemExit(f"[FATAL] no active reward terms parsed from {log_path}")
if "motion_ee_pos" not in rewards:
    raise SystemExit("[FATAL] motion_ee_pos is not active; the override did nothing")
if abs(rewards["motion_ee_pos"] - float(want_weight)) > 1e-9:
    raise SystemExit(
        f"[FATAL] motion_ee_pos weight {rewards['motion_ee_pos']} != {want_weight}"
    )

expected_others = {
    "motion_body_pos": 2.0,
    "motion_foot_pos": 2.0,
    "motion_global_anchor_pos": 2.0,
    "motion_global_anchor_ori": 2.0,
    "tracking_reward_points": 4.0,
    "action_rate_l2": 0.0,
}
for name, weight in expected_others.items():
    if name not in rewards:
        raise SystemExit(f"[FATAL] {name} is missing from the reward table")
    if abs(rewards[name] - weight) > 1e-9:
        raise SystemExit(f"[FATAL] {name} weight {rewards[name]} != {weight}")

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

actor_keys = {
    "latent_command", "projected_gravity", "base_ang_vel",
    "joint_pos_rel", "joint_vel_rel", "last_action",
}
actor_width = sum(w for t, w in groups.get("policy", []) if t in actor_keys)
critic_width = sum(w for _, w in groups.get("critic", []))
if actor_width != int(actor_dim):
    raise SystemExit(f"[FATAL] actor input {actor_width} != {actor_dim}")
if critic_width != int(critic_dim):
    raise SystemExit(f"[FATAL] critic input {critic_width} != {critic_dim}")
if any(t == "latent_command" for t, _ in groups.get("critic", [])):
    raise SystemExit("[FATAL] critic reads latent_command; expected critic_channels=[reference]")
print(
    f"[PASS] motion_ee_pos active at {rewards['motion_ee_pos']}, "
    f"every other reward term unchanged; actor {actor_width}, critic {critic_width}"
)
PY

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${EE_REWARD_WEIGHT}" <<'PY'
import json, sys

path, source_hash, weight = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "motion_ee_pos_weight": weight,
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
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-ee-rew"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,hold10,expert-heading-anchor,critic-no-latent,ee-reward,wrist-tracking,lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
unset CLUSTER_SLURM_DEPENDENCY

output="$("${LOWLEVEL_CMD[@]}" 2>&1 | tee /dev/stderr)"
lowlevel_job="$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"${output}" | tail -1)"
[[ "${lowlevel_job}" =~ ^[0-9]+$ ]] || fail "could not parse Slurm job ID"
echo "[SUBMITTED] ${ARM} tracker=${lowlevel_job}"

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" "${ARM}" \
    "${lowlevel_job}" "${CONTROL_LOWLEVEL_JOB}" "${CONTROL_ARM}" \
    "${ENCODER_SHA256}" "${ENCODER_REMOTE}" "${EE_REWARD_WEIGHT}" \
    "${FRAME_CAP}" "${MAX_ITERATIONS}" "${SEED}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm, lowlevel, control_job,
    control_arm, encoder_sha, encoder_path, weight, frame_cap, iterations, seed,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-09-bones129k-ee-reward",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "axis": {
        "field": "env.rewards.motion_ee_pos.weight",
        "arm_value": float(weight),
        "control_value": 0.0,
        "term": {
            "func": "reference_relative_body_position_error_exp",
            "bodies": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "anchor_body_name": "pelvis",
            "std": 0.1,
        },
    },
    "motivation": {
        "scoreboard": "2026-08-08-bones129k-4096-scoreboard, ranks 12288-16383",
        "control_failures": {
            "ee_body_pos": 318,
            "anchor_ori": 57,
            "anchor_pos": 17,
            "total_envs": 4096,
        },
        "body_attribution": {
            "wrists_only": {"ee_body_pos": 231, "success_rate": 0.9197},
            "ankles_only": {"ee_body_pos": 208, "success_rate": 0.9182},
            "note": (
                "231 + 208 > 318, so many environments fail on both pairs; the "
                "wrists are targeted because they have no dedicated reward, not "
                "because they are the only failing bodies"
            ),
        },
        "released_sonic_same_ranks": {"ee_body_pos": 26, "success_rate": 0.9937},
        "counter_evidence": (
            "the earlier s13 screen found a local wrist term improved "
            "root-relative EE while root drift rose and MPJPE-G worsened 28%, "
            "measured before the v2 tuned rewards raised both anchor terms "
            "from 0.5 to 2.0"
        ),
    },
    "control": {
        "job_id": int(control_job),
        "arm": control_arm,
        "wandb_group": "latent-anchor-frame",
        "note": "not resubmitted; this campaign adds one arm against it",
    },
    "arm": {"arm": arm, "lowlevel_job_id": int(lowlevel), "pretrain_job_id": None},
    "shared_encoder": {
        "path": encoder_path,
        "sha256": encoder_sha,
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
