#!/usr/bin/env bash
set -euo pipefail

# 2026-08-09 -- command hold period, one variable, on the old z256 recipe.
#
# Control is the scoreboard row `old_z256` = ICE job `5567801`
# (`reset80_diffsr`, W&B group `bones129k-ablation`): frozen root-qpos DiffSR
# encoder, z256 + sin/cos phase, h10, macro stride 1, ROBOT anchor frame, the
# tuned entry point's default tracker capacity, and the default critic channels
# `[actor, reference]`. Nothing is resubmitted for it. Its 4,096-motion row is
# SR 0.9058, success-only MPJPE-L 24.52 mm at 5B frames.
#
# The axis is how long one published latent command is held:
#
#   agent.ipmd.latent_steps_min=1
#   agent.ipmd.latent_steps_max=1
#   agent.ipmd.latent_learning.code_period=1
#
# Control holds 10 control steps (200 ms at 50 Hz). This arm re-encodes every
# control step, which is what the released SONIC tokenizer does (hold 1, no
# phase). Holding for 200 ms means the tracker acts on a command computed from
# a macro window up to 200 ms stale; hold 1 removes that staleness entirely.
#
# TWO CONSEQUENCES, both deliberate and both worth reading before comparing:
#
#   1. Bandwidth. Hold 1 publishes 50 commands/second instead of 5. As a
#      LOW-LEVEL ceiling that is fine and is exactly the question. It is NOT a
#      planner-interface row: the paper's planner comparison publishes at 5 Hz,
#      and a 50 Hz latent stream is not something the high level can produce.
#   2. Cost. The skill encoder now runs on every control step rather than every
#      tenth, so expect lower frames per second than the control at equal
#      settings. Compare at equal FRAMES; if you compare at equal wall-clock,
#      say so.
#
# The sin/cos phase channel is KEPT even though `code_period=1` makes it
# constant. Dropping it would shrink the published command from 258 to 256 and
# move the actor input width, which would make the hold period no longer the
# only variable. Two constant inputs are the cheaper price.
#
#   MODE=print   ./run.sh
#   MODE=smoke   ./run.sh                     # local 1-iteration gate
#   MODE=validate LOCAL_SMOKE_ROOT=<dir> ./run.sh
#   MODE=submit  LOCAL_SMOKE_ROOT=<dir> CONFIRM_SUBMIT=hold-period ./run.sh

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
ARM="${ARM:-old_z256_hold1}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_hold_period}"

# -- the axis under test --
HOLD_STEPS="${HOLD_STEPS:-1}"
CONTROL_HOLD_STEPS=10

CONTROL_LOWLEVEL_JOB=5567801
CONTROL_ARM="reset80_diffsr (scoreboard row old_z256)"
CONTROL_SR="0.9058"
CONTROL_MPJPE_MM="24.52"

# -- the old z256 encoder, frozen, the exact file the control loaded --
ENCODER_REMOTE="/data/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/checkpoints/latest.pt"
ENCODER_SHA256="d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7"
LOCAL_ENCODER="${LOCAL_ENCODER:-${REPO_ROOT}/logs/downloaded_checkpoints/bones129k_old_z256/root_qpos_det_sr_h10_z256_latest.pt}"

# -- the old z256 contract, reproduced exactly --
ANCHOR_MODE="robot"
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=1
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))
# Default critic channels [actor, reference], like the control: the critic
# keeps the actor latent here. Only the hold moves.
EXPECTED_ACTOR_INPUT_DIM=351
EXPECTED_CRITIC_INPUT_DIM=544

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
WANDB_GROUP="${WANDB_GROUP:-hold-period}"
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
    # Explicit, though it is the default: the control's encoder was pretrained
    # before `expert_heading` existed and reads back as `robot`, and pairing it
    # with an `expert_heading` environment is refused at load.
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
        "${REPO_ROOT}/RLOpt/rlopt/agent/ipmd/ipmd.py" \
        "${REPO_ROOT}/RLOpt/rlopt/agent/skill_commander.py" \
        "${REPO_ROOT}/RLOpt/rlopt/env_interface.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py" \
        "${REPO_ROOT}/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/command_interface.py" \
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
        "agent.ipmd.latent_steps_min=${HOLD_STEPS}"
        "agent.ipmd.latent_steps_max=${HOLD_STEPS}"
        "agent.ipmd.latent_learning.code_period=${HOLD_STEPS}"
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        agent.ipmd.hl_skill_finetune_enabled=false
        "${COMMON_ENV_OVERRIDES[@]}"
    )
    # No tracker capacity override and no critic-channel override: both are the
    # control's defaults.
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
    # Written before the field existed: a missing value IS "robot".
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
print(f"[PASS] old z256 encoder: {mode}, stride {stride}, h{horizon}, z{z_dim}, width 380")
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
        || fail "old z256 encoder unreadable on ICE"
    [[ "${remote_sha}" == "${ENCODER_SHA256}" ]] \
        || fail "remote encoder sha ${remote_sha} != ${ENCODER_SHA256}"

    path="$(remote_of "$(lowlevel_dir)")"
    [[ "$(ssh_ice "if [[ -e '${path}' ]]; then echo yes; else echo no; fi")" == "no" ]] \
        || fail "refusing to overwrite ${path}"
    echo "[PASS] remote arrays, old z256 encoder identity, fresh output path"
}

check_local_smoke() {
    [[ -n "${LOCAL_SMOKE_ROOT}" ]] || fail "LOCAL_SMOKE_ROOT is required for ${MODE}."
    [[ -f "${LOCAL_SMOKE_ROOT}/status.json" ]] || fail "missing local smoke marker"
    local expected status got hold
    expected="$(source_contract_hash)"
    read -r status got hold < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""), d.get("hold_steps", ""))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${hold}" == "${HOLD_STEPS}" ]] || fail "local smoke covered hold ${hold}"
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
echo "[INFO] axis: command hold ${CONTROL_HOLD_STEPS} -> ${HOLD_STEPS} control steps"
echo "[INFO] control = ICE ${CONTROL_LOWLEVEL_JOB} ${CONTROL_ARM}: SR ${CONTROL_SR}, MPJPE-L ${CONTROL_MPJPE_MM} mm at 5B"
echo "[INFO] old z256 recipe: robot anchor frame, stride ${MACRO_FRAME_STRIDE}, z${Z_DIM} + sin/cos phase, tuned default capacity, critic [actor, reference]"
echo "[INFO] frozen encoder ${ENCODER_SHA256:0:16}…; no pretrain job"
echo "[INFO] actor input ${EXPECTED_ACTOR_INPUT_DIM}; critic input ${EXPECTED_CRITIC_INPUT_DIM}"
echo "[INFO] hold ${HOLD_STEPS} publishes 50 commands/s, not 5: a low-level ceiling, NOT a planner-interface row"
echo "[INFO] the encoder now runs every control step, so expect lower fps than the control; compare at equal FRAMES"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    check_local_encoder
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_hold1_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"
    build_local_lowlevel_command "${LOCAL_ENCODER}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # Gate 1: the EFFECTIVE agent config, as resolved and written by the run --
    # not the command line. All three hold knobs must agree, because holding is
    # enforced in two places (the latent command controller's step budget and
    # the code period) and setting only one of them silently keeps the other.
    python3 - "${LOCAL_SMOKE_ROOT}/tracker" "${HOLD_STEPS}" "${Z_DIM}" <<'PY'
import pathlib
import re
import sys

run_root, hold, z_dim = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
configs = sorted(run_root.rglob("params/agent.yaml"))
if not configs:
    raise SystemExit(f"[FATAL] no resolved agent config under {run_root}")
text = configs[-1].read_text(encoding="utf-8")


def scalar(name):
    match = re.search(rf"^\s*{name}:\s*(\S+)\s*$", text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"[FATAL] {name} missing from {configs[-1]}")
    return match.group(1)


for field in ("latent_steps_min", "latent_steps_max", "code_period"):
    got = int(scalar(field))
    if got != hold:
        raise SystemExit(f"[FATAL] resolved {field}={got}, expected {hold}")
if scalar("command_phase_mode").strip("'\"") != "sin_cos":
    raise SystemExit("[FATAL] the phase channel was dropped; command width would move")
if int(scalar("code_latent_dim")) != z_dim:
    raise SystemExit(f"[FATAL] resolved code_latent_dim != {z_dim}")
print(f"[PASS] resolved config holds {hold} step(s) with the phase channel kept")
PY

    # Gate 2: the actor and critic must be the CONTROL's, untouched. In
    # particular the critic still reads the latent -- this arm is not the
    # critic ablation.
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
critic_terms = dict(groups["critic"])
if int(policy_terms.get("latent_command", -1)) != command_dim:
    raise SystemExit(
        f"[FATAL] latent_command is {policy_terms.get('latent_command')}, "
        f"expected {command_dim}"
    )
if int(critic_terms.get("latent_command", -1)) != command_dim:
    raise SystemExit(
        "[FATAL] the critic lost latent_command; this arm keeps the control's "
        "default critic channels [actor, reference]"
    )

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
print(f"[PASS] actor {actor_width} and critic {critic_width} match the control")
PY

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${HOLD_STEPS}" <<'PY'
import json, sys

path, source_hash, hold = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "hold_steps": hold,
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
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-hold1"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,hold1,robot-anchor,old-z256-recipe,lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
unset CLUSTER_SLURM_DEPENDENCY

output="$("${LOWLEVEL_CMD[@]}" 2>&1 | tee /dev/stderr)"
lowlevel_job="$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"${output}" | tail -1)"
[[ "${lowlevel_job}" =~ ^[0-9]+$ ]] || fail "could not parse Slurm job ID"
echo "[SUBMITTED] ${ARM} tracker=${lowlevel_job}"

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" "${ARM}" \
    "${lowlevel_job}" "${CONTROL_LOWLEVEL_JOB}" "${HOLD_STEPS}" \
    "${CONTROL_HOLD_STEPS}" "${ENCODER_SHA256}" "${ENCODER_REMOTE}" \
    "${FRAME_CAP}" "${MAX_ITERATIONS}" "${SEED}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm, lowlevel, control_job,
    hold, control_hold, encoder_sha, encoder_path, frame_cap, iterations, seed,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-09-bones129k-hold1",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "axis": {
        "field": "command hold period",
        "knobs": [
            "agent.ipmd.latent_steps_min",
            "agent.ipmd.latent_steps_max",
            "agent.ipmd.latent_learning.code_period",
        ],
        "arm_value": int(hold),
        "control_value": int(control_hold),
        "publication_rate_hz": {"arm": 50, "control": 5},
    },
    "control": {
        "job_id": int(control_job),
        "arm": "reset80_diffsr",
        "scoreboard_row": "old_z256",
        "wandb_group": "bones129k-ablation",
        "scoreboard_5b": {"success_rate": 0.9058, "succ_mpjpe_l_mm": 24.52},
        "note": "not resubmitted; this campaign adds one arm against it",
    },
    "arm": {"arm": arm, "lowlevel_job_id": int(lowlevel), "pretrain_job_id": None},
    "frozen_encoder": {
        "path": encoder_path,
        "sha256": encoder_sha,
        "note": "the exact file the control loaded",
        "macro_anchor_mode": "robot",
        "macro_frame_stride": 1,
        "horizon_steps": 10,
        "z_dim": 256,
        "encoder_input_dim": 380,
    },
    "lowlevel": {
        "task": "Isaac-Imitation-G1-v2",
        "motions": 129785,
        "command_dim": 258,
        "hold_steps": int(hold),
        "phase": (
            "sin_cos retained though code_period=1 makes it constant, so the "
            "published command stays 258 wide and the hold stays the only axis"
        ),
        "actor_input_dim": 351,
        "critic_input_dim": 544,
        "critic_channels": ["actor", "reference"],
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
    "caveats": [
        "hold 1 publishes 50 commands/s; this is a low-level ceiling, not a "
        "planner-interface row (the paper's planner publishes at 5 Hz)",
        "the skill encoder runs every control step instead of every tenth, so "
        "throughput is lower than the control; compare at equal frames",
    ],
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
echo "[RECORDED] ${record_path}"
