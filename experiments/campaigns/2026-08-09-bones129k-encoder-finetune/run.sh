#!/usr/bin/env bash
set -euo pipefail

# 2026-08-09 -- online skill-encoder finetuning, one variable.
#
# Control is the running ICE job `5573413` (`expert_heading_critic_no_latent`,
# W&B group `latent-anchor-frame`): expert-heading macro frame, critic channels
# `[reference]`, frozen z256 DiffSR encoder. Nothing is resubmitted for it.
#
# The axis:
#
#   agent.ipmd.hl_skill_finetune_enabled=true
#
# Every latent arm this project has run freezes the encoder after pretraining.
# The released SONIC checkpoint does not: its tokenizer sits inside the actor
# backbone and takes the PPO gradient natively, alongside a reconstruction
# loss. That is the largest remaining structural difference between our recipe
# and the checkpoint that scores 0.9937 against our 0.9062.
#
# Unfreezing was previously unattractive because the frozen encoder was already
# queried off-distribution at rollout time (the `robot` anchor convention), so
# an online gradient would have been chasing a moving target. The expert-heading
# frame removes that, which is why this arm sits on top of `5573413` rather than
# on the older robot-frame recipe.
#
# The finetune loss is not raw policy gradient. It is
#
#   pg_coeff * (second-pass PPO actor objective, gradient through the encoder)
#   + offline_diffsr_coeff * (the original offline DiffSR loss)
#   + anchor_coeff * (distance to the FROZEN checkpoint encoder's output)
#
# so the offline objective and the anchor term act as a trust region around the
# pretrained encoder. Every coefficient is passed explicitly below, at its
# current library default, so a later default change cannot move this arm.
#
#   MODE=print   ./run.sh
#   MODE=smoke   ./run.sh                     # local 1-iteration gate
#   MODE=validate LOCAL_SMOKE_ROOT=<dir> ./run.sh
#   MODE=submit  LOCAL_SMOKE_ROOT=<dir> CONFIRM_SUBMIT=encoder-finetune ./run.sh

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
ARM="${ARM:-expert_heading_critic_no_latent_finetune}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_encoder_finetune}"

# -- the shared encoder, identical to the control's, by path and by SHA-256 --
CONTROL_LOWLEVEL_JOB=5573413
CONTROL_ARM="expert_heading_critic_no_latent"
ENCODER_REMOTE="/data/bones129k_anchor_frame/expert_heading_encoder/checkpoints/latest.pt"
ENCODER_SHA256="be6d533f1d1ca4aa6b1e819af1d3ef63eb033125018c8309c7448384b6a9583e"
LOCAL_ENCODER="${LOCAL_ENCODER:-${REPO_ROOT}/logs/downloaded_checkpoints/bones129k_anchor_frame/expert_heading_encoder_latest.pt}"

# -- the axis under test, every coefficient explicit at its library default --
FINETUNE_LR="${FINETUNE_LR:-3.0e-5}"
FINETUNE_PG_COEFF="${FINETUNE_PG_COEFF:-0.05}"
FINETUNE_DIFFSR_COEFF="${FINETUNE_DIFFSR_COEFF:-1.0}"
FINETUNE_ANCHOR_COEFF="${FINETUNE_ANCHOR_COEFF:-0.01}"
FINETUNE_GRAD_CLIP="${FINETUNE_GRAD_CLIP:-1.0}"
FINETUNE_OFFLINE_BATCH="${FINETUNE_OFFLINE_BATCH:-8192}"
FINETUNE_UPDATE_INTERVAL="${FINETUNE_UPDATE_INTERVAL:-1}"
# The loaded DiffSR transition model stays frozen: only the encoder moves, so
# the axis is "the command function adapts", not "the whole pretrain restarts".
FINETUNE_TRAIN_DIFFSR="${FINETUNE_TRAIN_DIFFSR:-false}"

FINETUNE_OVERRIDES=(
    agent.ipmd.hl_skill_finetune_enabled=true
    "agent.ipmd.hl_skill_lr=${FINETUNE_LR}"
    "agent.ipmd.hl_skill_pg_coeff=${FINETUNE_PG_COEFF}"
    "agent.ipmd.hl_skill_offline_diffsr_coeff=${FINETUNE_DIFFSR_COEFF}"
    "agent.ipmd.hl_skill_anchor_coeff=${FINETUNE_ANCHOR_COEFF}"
    "agent.ipmd.hl_skill_grad_clip_norm=${FINETUNE_GRAD_CLIP}"
    "agent.ipmd.hl_skill_offline_batch_size=${FINETUNE_OFFLINE_BATCH}"
    "agent.ipmd.hl_skill_update_interval=${FINETUNE_UPDATE_INTERVAL}"
    "agent.ipmd.hl_skill_train_diffsr=${FINETUNE_TRAIN_DIFFSR}"
)

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
SMOKE_OFFLINE_BATCH="${SMOKE_OFFLINE_BATCH:-8}"
LOCAL_SMOKE_ROOT="${LOCAL_SMOKE_ROOT:-}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-encoder-finetune}"
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
        "${REPO_ROOT}/RLOpt/rlopt/agent/hl_skill_diffsr.py" \
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
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
        "${COMMON_ENV_OVERRIDES[@]}"
        "${FINETUNE_OVERRIDES[@]}"
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
    LOCAL_LOWLEVEL_CMD+=("agent.ipmd.hl_skill_offline_batch_size=${SMOKE_OFFLINE_BATCH}")
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
    local expected status got finetune
    expected="$(source_contract_hash)"
    read -r status got finetune < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("status", ""), d.get("source_contract_sha256", ""), d.get("finetune_updates", ""))
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${finetune}" -ge 1 ]] || fail "local smoke recorded ${finetune} finetune updates"
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
echo "[INFO] axis: agent.ipmd.hl_skill_finetune_enabled=true"
echo "[INFO] lr ${FINETUNE_LR}, pg ${FINETUNE_PG_COEFF}, diffsr ${FINETUNE_DIFFSR_COEFF}, anchor ${FINETUNE_ANCHOR_COEFF}, clip ${FINETUNE_GRAD_CLIP}"
echo "[INFO] control = ICE ${CONTROL_LOWLEVEL_JOB} (${CONTROL_ARM}), frozen encoder"
echo "[INFO] same starting encoder ${ENCODER_SHA256:0:16}… (${ANCHOR_MODE}, stride ${MACRO_FRAME_STRIDE}); no pretrain job"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] ICE allocations end at 16 h, so this job is EXPECTED to TIMEOUT before the cap."
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    check_local_encoder
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_encoder_finetune_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"
    build_local_lowlevel_command "${LOCAL_ENCODER}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # The gate: enabling the flag must actually MOVE the encoder. A run that
    # logs zero finetune updates, or logs updates without changing a weight,
    # is indistinguishable from the frozen control in W&B.
    finetune_updates="$(pixi run -q python - "${LOCAL_SMOKE_ROOT}/tracker" \
        "${LOCAL_ENCODER}" <<'PY'
import csv
import pathlib
import sys

import torch

run_root, initial_path = pathlib.Path(sys.argv[1]), sys.argv[2]

# The CSV logger writes one file per scalar, named for the metric, with
# `step,value` rows -- the metric name is the PATH, not a column header.
def scalar_max(stem):
    best = None
    for csv_path in run_root.rglob(f"{stem}.csv"):
        with csv_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 2:
                    continue
                value = float(row[1])
                best = value if best is None else max(best, value)
    return best


updates = scalar_max("hl_skill_updates")
if updates is None or updates < 1:
    raise SystemExit("[FATAL] no hl_skill finetune updates logged")
grad_norm = scalar_max("hl_skill_grad_norm")
if grad_norm is None or grad_norm <= 0.0:
    raise SystemExit(f"[FATAL] hl_skill gradient norm is {grad_norm}; nothing moved")
print(f"[PASS] hl_skill grad norm {grad_norm:.4f}")
updates = int(updates)

checkpoints = sorted(run_root.rglob("model_step_*.pt")) + sorted(run_root.rglob("*.pt"))
if not checkpoints:
    print("[WARN] the one-iteration smoke wrote no checkpoint; update count only")
    print(updates)
    raise SystemExit(0)
trained = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
initial = torch.load(initial_path, map_location="cpu", weights_only=False)
initial_state = initial["skill_encoder_state_dict"]
trained_state = None
for key in ("skill_encoder_state_dict", "hl_skill_encoder_state_dict"):
    if key in trained:
        trained_state = trained[key]
        break
if trained_state is None:
    print(f"[WARN] {checkpoints[-1].name} embeds no encoder state; update count only")
else:
    moved = any(
        not torch.equal(trained_state[name].float(), initial_state[name].float())
        for name in initial_state
        if name in trained_state
    )
    if not moved:
        raise SystemExit("[FATAL] encoder weights are unchanged after finetuning")
    print("[PASS] encoder weights moved")
print(updates)
PY
    )" || fail "finetune gate failed"
    finetune_updates="$(tail -1 <<<"${finetune_updates}")"
    echo "[PASS] finetune ran: ${finetune_updates} high-level updates"

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${finetune_updates}" <<'PY'
import json, sys

path, source_hash, updates = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "finetune_updates": int(updates),
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
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-enc-ft"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,hold10,expert-heading-anchor,critic-no-latent,encoder-finetune,online-pg,lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
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
    "${ENCODER_SHA256}" "${ENCODER_REMOTE}" "${FRAME_CAP}" "${MAX_ITERATIONS}" \
    "${FINETUNE_LR}" "${FINETUNE_PG_COEFF}" "${FINETUNE_DIFFSR_COEFF}" \
    "${FINETUNE_ANCHOR_COEFF}" "${FINETUNE_GRAD_CLIP}" "${FINETUNE_OFFLINE_BATCH}" \
    "${FINETUNE_UPDATE_INTERVAL}" "${FINETUNE_TRAIN_DIFFSR}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm, lowlevel, control_job,
    control_arm, encoder_sha, encoder_path, frame_cap, iterations,
    lr, pg, diffsr, anchor, clip, offline_batch, interval, train_diffsr,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-09-bones129k-encoder-finetune",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "axis": {
        "field": "agent.ipmd.hl_skill_finetune_enabled",
        "arm_value": True,
        "control_value": False,
    },
    "control": {
        "job_id": int(control_job),
        "arm": control_arm,
        "wandb_group": "latent-anchor-frame",
        "note": "not resubmitted; this campaign adds one arm against it",
    },
    "arm": {"arm": arm, "lowlevel_job_id": int(lowlevel), "pretrain_job_id": None},
    "starting_encoder": {
        "path": encoder_path,
        "sha256": encoder_sha,
        "note": (
            "identical file the control loads and keeps frozen; this arm starts "
            "from it and lets it move"
        ),
        "macro_anchor_mode": "expert_heading",
        "macro_frame_stride": 1,
        "horizon_steps": 10,
        "z_dim": 256,
        "encoder_input_dim": 380,
    },
    "finetune": {
        "lr": float(lr),
        "pg_coeff": float(pg),
        "offline_diffsr_coeff": float(diffsr),
        "anchor_coeff": float(anchor),
        "grad_clip_norm": float(clip),
        "offline_batch_size": int(offline_batch),
        "update_interval": int(interval),
        "train_diffsr": train_diffsr == "true",
        "note": (
            "offline DiffSR loss plus the anchor term act as a trust region "
            "around the pretrained encoder; the DiffSR transition model itself "
            "stays frozen"
        ),
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
        "seed": 0,
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
