#!/usr/bin/env bash
set -euo pipefail

# One variable: capacity.
#
# Control is the existing ICE job 5567801 (`reset80_diffsr`, W&B group
# `bones129k-ablation`), the row the 4,096-motion scoreboard calls `old_z256`
# (SONIC SR 0.9058, success-only MPJPE-L 24.52 mm at 5.0B frames). Nothing is
# resubmitted for it.
#
# This campaign submits ONE new arm, `z256_scaled`: a fresh encoder pretrain
# plus an afterok-dependent controller whose command lines are byte-identical
# to the control's EXCEPT for the width of three networks.
#
# What changes
# ------------
#                        control (old_z256)          this arm (z256_scaled)
#   encoder trunk        [1024, 512, 512] mish       [2048, 1024, 512, 512] silu
#                        LayerNorm on                LayerNorm off
#   DiffSR feature/embed 128 / 512                   256 / 1024
#   DiffSR g, mu heads   [512]                       [1024, 1024, 512]
#   policy num_cells     [1024, 1024, 512]           [2048, 2048, 1024, 1024, 512, 512]
#   value  num_cells     [1024, 1024, 512]           [2048, 2048, 1024, 1024, 512, 512]
#   activation           entry-point default         silu
#
# The scaled widths are not invented here: they are the exact geometry of
# campaign `2026-08-06-bones129k-sonic-fsq-scale`, which measured them under an
# FSQ bottleneck. This arm moves that capacity onto the continuous z256
# bottleneck so the capacity axis can be read against a control that differs in
# nothing else.
#
# What does NOT change: the macro interface (root-qpos 380, ten slots at stride
# 1, endpoint objective), the anchor frame (`robot`, the historical
# convention), the command (z256 + sin/cos phase = 258, held 10 control steps),
# the critic channels ([actor, reference], i.e. the critic still sees the
# latent), the reset sampler, the rewards, the termination curriculum, the PPO
# geometry, the seed, and the frame cap.
#
# Deliberately NOT combined with the two 2026-08-08 ingredients that are still
# in flight (`expert_heading` anchor frame, ICE 5573234; critic without the
# actor latent, ICE 5573413). Folding those in would make this a combined arm
# and destroy the capacity attribution. Combine later, once each is measured.

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
ARM="${ARM:-z256_scaled}"

REF_ARRAYS="${REF_ARRAYS:-/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/bones129k_z256_scale}"

# The control's encoder, for the provenance line in the record. Never loaded.
CONTROL_ENCODER_SHA256="d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7"

# -- the control's contract, reproduced exactly --
ANCHOR_MODE="robot"
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=1
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# -- the axis under test --
ENCODER_HIDDEN_DIMS=(2048 1024 512 512)
DIFFSR_FEATURE_DIM=256
DIFFSR_EMBED_DIM=1024
DIFFSR_HEAD_DIMS=(1024 1024 512)
TRACKER_CELLS="[2048,2048,1024,1024,512,512]"
TRACKER_ACTIVATION=silu
# Control values, asserted against in the smoke so a silent revert to the
# tuned geometry cannot pass as this arm.
CONTROL_ENCODER_HIDDEN_0=1024
CONTROL_TRACKER_WIDTH_0=1024

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
WANDB_GROUP="${WANDB_GROUP:-latent-capacity-scale}"
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
        --encoder_hidden_dims "${ENCODER_HIDDEN_DIMS[@]}"
        --encoder_activation "${TRACKER_ACTIVATION}" --no_encoder_layer_norm
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
        "agent.policy.activation_fn=${TRACKER_ACTIVATION}"
        "agent.value_function.num_cells=${TRACKER_CELLS}"
        "agent.value_function.activation_fn=${TRACKER_ACTIVATION}"
        "${COMMON_ENV_OVERRIDES[@]}"
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
    local expected status got width cells
    expected="$(source_contract_hash)"
    read -r status got width cells < <(python3 - "${LOCAL_SMOKE_ROOT}/status.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    d.get("status", ""),
    d.get("source_contract_sha256", ""),
    d.get("encoder_hidden_0", ""),
    d.get("tracker_width_0", ""),
)
PY
    )
    [[ "${status}" == "pass" ]] || fail "local smoke did not pass"
    [[ "${got}" == "${expected}" ]] || fail "local smoke is stale (${got} != ${expected})"
    [[ "${width}" == "${ENCODER_HIDDEN_DIMS[0]}" ]] \
        || fail "local smoke covered encoder width ${width}, not ${ENCODER_HIDDEN_DIMS[0]}"
    [[ "${cells}" == "2048" ]] \
        || fail "local smoke covered tracker width ${cells}, not 2048"
    echo "[PASS] local smoke: encoder ${width}, tracker ${cells}, contract ${expected:0:16}…"
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

echo "[INFO] mode=${MODE}; arm=${ARM}; axis=capacity only"
echo "[INFO] control = ICE 5567801 (bones129k-ablation / scoreboard old_z256), encoder ${CONTROL_ENCODER_SHA256:0:16}…"
echo "[INFO] encoder ${CONTROL_ENCODER_HIDDEN_0}-wide mish+LN -> ${ENCODER_HIDDEN_DIMS[*]} ${TRACKER_ACTIVATION}, no LN"
echo "[INFO] DiffSR feature/embed 128/512 -> ${DIFFSR_FEATURE_DIM}/${DIFFSR_EMBED_DIM}, heads [512] -> [${DIFFSR_HEAD_DIMS[*]}]"
echo "[INFO] tracker [1024,1024,512] -> ${TRACKER_CELLS} ${TRACKER_ACTIVATION} (actor AND critic)"
echo "[INFO] unchanged: root-qpos 380, ${HORIZON_STEPS} slots at stride ${MACRO_FRAME_STRIDE}, anchor ${ANCHOR_MODE}, z${Z_DIM}+phase=${LATENT_COMMAND_DIM}, hold 10"
echo "[INFO] ${FRAME_CAP} frame cap = ${MAX_ITERATIONS} iterations on ${GPU_GRES}"
echo "[INFO] W&B=${WANDB_PROJECT}/${WANDB_GROUP}"

if [[ "${MODE}" == "smoke" ]]; then
    if [[ -z "${LOCAL_SMOKE_ROOT}" ]]; then
        LOCAL_SMOKE_ROOT="${REPO_ROOT}/logs/bones129k_z256_scale_smoke/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "${LOCAL_SMOKE_ROOT}"

    build_local_pretrain_command "${LOCAL_SMOKE_ROOT}/encoder"
    print_cmd "${LOCAL_PRETRAIN_CMD[@]}"
    "${LOCAL_PRETRAIN_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/pretrain.log"
    local_checkpoint="${LOCAL_SMOKE_ROOT}/encoder/checkpoints/latest.pt"
    [[ -f "${local_checkpoint}" ]] || fail "smoke pretrain did not write ${local_checkpoint}"

    # Gate 1: the encoder really is the scaled one, and the rest of the macro
    # contract really is the control's. A silent revert to the tuned geometry
    # would otherwise run 10B frames as a duplicate of the control.
    pixi run -q python - "${local_checkpoint}" "${ANCHOR_MODE}" \
        "${MACRO_FRAME_STRIDE}" "${ENCODER_HIDDEN_DIMS[0]}" \
        "${DIFFSR_FEATURE_DIM}" "${DIFFSR_EMBED_DIM}" \
        "${CONTROL_ENCODER_HIDDEN_0}" <<'PY'
import sys, torch

(
    path, expected_mode, expected_stride, expected_hidden0,
    expected_feature, expected_embed, control_hidden0,
) = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint["config"]

mode = str(config.get("macro_anchor_mode", "robot"))
if mode != expected_mode:
    raise SystemExit(f"[FATAL] macro_anchor_mode={mode!r}, expected {expected_mode!r}")
stride = int(config.get("macro_frame_stride", 1))
if stride != int(expected_stride):
    raise SystemExit(f"[FATAL] macro_frame_stride={stride}, expected {expected_stride}")

weight = checkpoint["skill_encoder_state_dict"]["net.0.weight"]
if int(weight.shape[1]) != 380:
    raise SystemExit(f"[FATAL] encoder input width {int(weight.shape[1])} != 380 (root_qpos)")
hidden0 = int(weight.shape[0])
if hidden0 == int(control_hidden0):
    raise SystemExit(
        f"[FATAL] encoder trunk is {hidden0} wide -- this is the CONTROL geometry, "
        "not the scaled arm"
    )
if hidden0 != int(expected_hidden0):
    raise SystemExit(f"[FATAL] encoder trunk {hidden0} != {expected_hidden0}")

feature = int(config.get("diffsr_feature_dim", 0))
embed = int(config.get("diffsr_embed_dim", 0))
if feature != int(expected_feature) or embed != int(expected_embed):
    raise SystemExit(
        f"[FATAL] DiffSR feature/embed {feature}/{embed} != "
        f"{expected_feature}/{expected_embed}"
    )
if config.get("encoder_layer_norm", False):
    raise SystemExit("[FATAL] encoder LayerNorm is on; the scaled recipe has it off")
print(f"[PASS] scaled encoder: trunk {hidden0}x380, DiffSR {feature}/{embed}, no LayerNorm")
PY

    build_local_lowlevel_command "${local_checkpoint}" "${LOCAL_SMOKE_ROOT}/tracker"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    mkdir -p "${LOCAL_SMOKE_ROOT}/tracker"
    "${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${LOCAL_SMOKE_ROOT}/tracker/train.log"

    # Gate 2: the tracker really is the scaled one. The printed module graph is
    # the only capacity evidence a one-iteration run leaves behind, so it is
    # read with whitespace stripped -- the console wraps it.
    tracker_width="$(python3 - "${LOCAL_SMOKE_ROOT}/tracker/train.log" \
        "${LATENT_COMMAND_DIM}" "${CONTROL_TRACKER_WIDTH_0}" <<'PY'
import re, sys

path, command_dim, control_width = sys.argv[1:]
text = re.sub(r"\s+", "", open(path, encoding="utf-8", errors="replace").read())
# The actor's first layer consumes 93 non-command inputs plus the command.
actor_in = 93 + int(command_dim)
widths = {
    int(m)
    for m in re.findall(rf"in_features={actor_in},out_features=(\d+)", text)
}
if not widths:
    raise SystemExit(
        f"[FATAL] no first layer with in_features={actor_in} in {path}; "
        "the actor input contract changed"
    )
if len(widths) != 1:
    raise SystemExit(f"[FATAL] inconsistent first-layer widths {sorted(widths)}")
width = widths.pop()
if width == int(control_width):
    raise SystemExit(
        f"[FATAL] tracker first layer is {width} wide -- CONTROL geometry, not scaled"
    )
print(width)
PY
    )" || fail "tracker capacity gate failed"
    echo "[PASS] scaled tracker: first layer ${tracker_width} wide"

    python3 - "${LOCAL_SMOKE_ROOT}/status.json" "$(source_contract_hash)" \
        "${ENCODER_HIDDEN_DIMS[0]}" "${tracker_width}" "${ANCHOR_MODE}" \
        "${MACRO_FRAME_STRIDE}" <<'PY'
import json, sys

path, source_hash, encoder_hidden0, tracker_width, anchor_mode, stride = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "source_contract_sha256": source_hash,
            "encoder_hidden_0": int(encoder_hidden0),
            "tracker_width_0": int(tracker_width),
            "anchor_mode": anchor_mode,
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

build_cluster_pretrain_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train_hl_skill_diffsr.py
export CLUSTER_SLURM_TIME_LIMIT="${PRETRAIN_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-z256s-pre"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,scaled-capacity,scaled-encoder,pretrain,endpoint"
unset CLUSTER_SLURM_DEPENDENCY
pretrain_job="$(submit_and_capture_job_id "${PRETRAIN_CMD[@]}")"
echo "[SUBMITTED] ${ARM} pretrain=${pretrain_job}"

build_cluster_lowlevel_command
export CLUSTER_PYTHON_EXECUTABLE=scripts/rlopt/train.py
export CLUSTER_SLURM_TIME_LIMIT="${LOWLEVEL_TIME_LIMIT:-15:59:00}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="b129k-z256s"
export CLUSTER_WANDB_TAGS="bones-seed,129785,v2,root-qpos,stride1,z256,hold10,scaled-capacity,scaled-tracker,lowlevel,newton,rollout24,gamma097,reset80-adaptive20"
export CLUSTER_SLURM_DEPENDENCY="afterok:${pretrain_job}"
lowlevel_job="$(submit_and_capture_job_id "${LOWLEVEL_CMD[@]}")"
echo "[SUBMITTED] ${ARM} tracker=${lowlevel_job} afterok:${pretrain_job}"

record_path="${SCRIPT_DIR}/cluster_submission.json"
[[ ! -e "${record_path}" ]] || fail "refusing to overwrite ${record_path}"
python3 - "${record_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(source_contract_hash)" "${WANDB_PROJECT}" "${WANDB_GROUP}" \
    "${ARM}" "${pretrain_job}" "${lowlevel_job}" \
    "${FRAME_CAP}" "${MACRO_FRAME_STRIDE}" "${HORIZON_STEPS}" "${Z_DIM}" \
    "${CONTROL_ENCODER_SHA256}" "${TRACKER_CELLS}" <<'PY'
import json, sys

(
    path, submitted, source_hash, project, group, arm,
    pretrain, lowlevel, frame_cap, stride, horizon, z_dim, control_encoder,
    tracker_cells,
) = sys.argv[1:]
payload = {
    "campaign": "2026-08-09-bones129k-z256-scale",
    "submitted_utc": submitted,
    "cluster": "ICE",
    "source_contract_sha256": source_hash,
    "wandb": {"project": project, "group": group},
    "axis": {
        "field": "network capacity (encoder trunk, DiffSR heads, actor, critic)",
        "arm_value": "scaled",
        "control_value": "tuned",
    },
    "control": {
        "job_id": 5567801,
        "arm": "reset80_diffsr",
        "scoreboard_row": "old_z256",
        "wandb_group": "bones129k-ablation",
        "encoder_sha256": control_encoder,
        "scoreboard_sonic_sr": 0.9058,
        "scoreboard_successful_mpjpe_l_mm": 24.52,
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
        "macro_anchor_mode": "robot",
        "horizon_steps": int(horizon),
        "z_dim": int(z_dim),
        "latent_mode": "deterministic",
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
        "control_geometry": {
            "encoder_hidden_dims": [1024, 512, 512],
            "encoder_activation": "mish",
            "encoder_layer_norm": True,
            "diffsr_feature_dim": 128,
            "diffsr_embed_dim": 512,
            "diffsr_g_hidden_dims": [512],
            "diffsr_mu_hidden_dims": [512],
        },
    },
    "lowlevel": {
        "task": "Isaac-Imitation-G1-v2",
        "motions": 129785,
        "command_dim": int(z_dim) + 2,
        "hold_steps": 10,
        "tracker_num_cells": tracker_cells,
        "tracker_activation": "silu",
        "control_tracker_num_cells": "[1024,1024,512] (tuned entry point default)",
        "critic_channels": "[actor, reference] (unchanged from control)",
        "num_envs": 16384,
        "rollout_steps": 24,
        "mini_batch_size": 294912,
        "gamma": 0.97,
        "seed": 0,
        "frame_cap": int(frame_cap),
        "checkpoint_interval_frames": 50000000,
        "reset_sampler": "random80_adaptive20",
    },
    "capacity_note": (
        "The scaled widths are copied from 2026-08-06-bones129k-sonic-fsq-scale, "
        "which measured them under an FSQ bottleneck. Nothing else differs from "
        "the control, so a difference here is attributable to capacity."
    ),
    "throughput_note": (
        "The scaled tracker is slower per frame. The matched-budget comparison "
        "is the 5,000,134,656-frame checkpoint, which the control also has; a "
        "15:59:00 allocation may TIMEOUT before the 10B cap."
    ),
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY
echo "[RECORDED] ${record_path}"
