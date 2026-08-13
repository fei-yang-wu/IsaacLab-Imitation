#!/usr/bin/env bash
set -euo pipefail

# Local qualification of `env.expert_macro_anchor_mode=robot_heading`.
#
# This script does NOT produce a paper number. It qualifies one new code path
# in the real environment (Isaac Lab + Newton), and proves the pairing guard
# that protects it. The cluster arm that measures the mode is deferred: ICE was
# down on 2026-08-13.
#
# What the mode is
# ----------------
# `expert_macro_anchor_mode` selects the frame the DiffSR macro window -- the
# reference window the skill encoder reads -- is expressed in. Upstream
# `gear_sonic` has three canonicalizations of that window's anchor orientation
# and ships two of them:
#
#   upstream `motion_anchor_ori_b_mf`        inv(LIVE robot FULL quat) * ref
#                                            = our "robot" (rollout half)
#   upstream `motion_anchor_ori_heading_mf`  inv(LIVE robot HEADING quat) * ref
#                                            = our "robot_heading"  <-- NEW
#   upstream `motion_anchor_ori_refheading_mf`
#                                            inv(REFERENCE first-frame heading)
#                                            * ref = our "expert_heading"
#
# The released SONIC checkpoint reads `b_mf`; SONIC v1.1 -- the tracker behind
# the 25.41 mm oracle row of 2026-08-12-gr00t-language30-compositionality --
# reads `heading_mf`. No gear_sonic config selects `refheading`, so our
# `expert_heading` reproduced the one upstream variant nobody ships, and until
# now we had NO mode for the live-robot heading frame.
#
# `robot_heading` anchors the live window at
# `heading_anchor_frame(live robot anchor)`: yaw-only twist, xy-only origin, so
# the reference keeps absolute height and its tilt relative to gravity while
# the encoder input carries the live tracking error. Pretraining has no robot,
# so it keeps the expert slot-0 heading frame -- the same frame in the
# perfect-tracking limit, and the closest offline analogue.
#
# Why the guard matters here
# --------------------------
# The macro state is 380 wide in EVERY mode. A frozen encoder paired with the
# wrong frame produces no shape error, only a silently off-distribution
# command. The mode is therefore recorded into the skill checkpoint at pretrain
# time and compared at low-level startup. Gates 3 and 4 below are the only
# things that prove that comparison is wired.
#
# Contract note: the pretrain/tracker contract reproduces the
# 2026-08-08-bones129k-anchor-frame arm exactly except for the anchor mode, so
# the eventual cluster arm is single-variable against the `robot` control
# (ICE 5567801) that campaign was built around.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    [[ "${REPO_ROOT}" != "/" ]] || { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }
print_cmd() { printf '[CMD] '; printf '%q ' "$@"; printf '\n'; }

SEED="${SEED:-0}"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"

# -- the axis under test --
ANCHOR_MODE="robot_heading"
# Modes the guard must REFUSE for an encoder pretrained under ANCHOR_MODE.
REFUSED_MODES=(robot expert_heading)

LOCAL_REF_ARRAYS="${LOCAL_REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"

# -- the 2026-08-08 anchor-frame contract, reproduced exactly --
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=1
Z_DIM=256
LATENT_COMMAND_DIM=$((Z_DIM + 2))
MACRO_STATE_WIDTH=380

# W&B group for the deferred cluster arm. The local qualification logs nothing
# to W&B (WANDB_MODE=disabled below); the name is recorded so the submit that
# follows cannot drift from what was agreed.
WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-sonic-frame-parity}"

QUAL_NUM_ENVS="${QUAL_NUM_ENVS:-4}"
QUAL_ROLLOUT_STEPS="${QUAL_ROLLOUT_STEPS:-4}"
QUAL_MINIBATCH_SIZE="${QUAL_MINIBATCH_SIZE:-8}"
QUAL_EXPERT_BATCH_SIZE="${QUAL_EXPERT_BATCH_SIZE:-8}"

PIXI="${PIXI:-pixi}"
command -v "${PIXI}" >/dev/null 2>&1 || fail "pixi not found; set PIXI=/path/to/pixi"

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
        "${SCRIPT_DIR}/run_local_qualification.sh" \
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

build_local_pretrain_command() {
    local output_dir="$1"
    LOCAL_PRETRAIN_CMD=(
        env TERM=xterm PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
        TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled
        "${PIXI}" run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py
        --task "${TASK}" --num_envs "${QUAL_NUM_ENVS}" --seed "${SEED}"
        --device cuda:0 --headless --assert-kitless
        --output_dir "${output_dir}" --logger_backend none
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
        --reconstruction_eval --window_probe_eval
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
        "${PIXI}" run -e isaaclab python -u scripts/rlopt/train.py
        --task "${TASK}" --num_envs "${QUAL_NUM_ENVS}" --headless
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
        "agent.collector.frames_per_batch=${QUAL_ROLLOUT_STEPS}"
        "agent.loss.mini_batch_size=${QUAL_MINIBATCH_SIZE}"
        "agent.ipmd.expert_batch_size=${QUAL_EXPERT_BATCH_SIZE}"
        agent.loss.gamma=0.97
        agent.logger.backend=csv agent.logger.video=false
        "agent.logger.log_dir=${output_dir}"
        agent.logger.exp_name=robot_heading_qualification
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
    # Hydra is last-wins, so the negative check's override must come after the
    # contract block above to be the effective value.
    if [[ "${anchor_override}" != "${ANCHOR_MODE}" ]]; then
        LOCAL_LOWLEVEL_CMD+=("env.expert_macro_anchor_mode=${anchor_override}")
    fi
}

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/sonic_frame_parity_qualification/$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "${OUTPUT_ROOT}/status.json" ]] || fail "refusing to overwrite ${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] local qualification only; no cluster submission, no paper number"
echo "[INFO] anchor mode under test: ${ANCHOR_MODE}"
echo "[INFO] macro window: root-qpos ${MACRO_STATE_WIDTH}, ${HORIZON_STEPS} slots at stride ${MACRO_FRAME_STRIDE}"
echo "[INFO] z${Z_DIM} + sin/cos phase = ${LATENT_COMMAND_DIM}; hold 10; tuned tracker capacity"
echo "[INFO] deferred cluster arm would log to ${WANDB_PROJECT}/${WANDB_GROUP}"
echo "[INFO] output: ${OUTPUT_ROOT}"

# Gate 1 -- the pretrain path accepts the mode and records it. Without the
# recording the pairing guard has nothing to compare and the whole detection
# story is vacuous.
build_local_pretrain_command "${OUTPUT_ROOT}/encoder"
print_cmd "${LOCAL_PRETRAIN_CMD[@]}"
"${LOCAL_PRETRAIN_CMD[@]}" 2>&1 | tee "${OUTPUT_ROOT}/pretrain.log"
ENCODER="${OUTPUT_ROOT}/encoder/checkpoints/latest.pt"
[[ -f "${ENCODER}" ]] || fail "pretrain did not write ${ENCODER}"

"${PIXI}" run -q python - "${ENCODER}" "${ANCHOR_MODE}" "${MACRO_FRAME_STRIDE}" \
    "${MACRO_STATE_WIDTH}" <<'PY'
import sys

import torch

path, expected_mode, expected_stride, expected_width = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint["config"]
mode = str(config.get("macro_anchor_mode", "robot"))
stride = int(config.get("macro_frame_stride", 1))
if mode != expected_mode:
    raise SystemExit(
        f"[FATAL] {path} recorded macro_anchor_mode={mode!r}, expected {expected_mode!r}"
    )
if stride != int(expected_stride):
    raise SystemExit(
        f"[FATAL] {path} recorded macro_frame_stride={stride}, expected {expected_stride}"
    )
width = int(checkpoint["skill_encoder_state_dict"]["net.0.weight"].shape[1])
if width != int(expected_width):
    raise SystemExit(f"[FATAL] encoder input width {width} != {expected_width}")
print(f"[PASS] {path} records macro_anchor_mode={mode!r}, stride={stride}, width={width}")
PY

# Gate 2 -- the LIVE path. This is the only gate that runs the new
# `robot_heading` rollout context inside Isaac: the macro window is built from
# the live robot anchor's heading frame every control step.
build_local_lowlevel_command "${ENCODER}" "${OUTPUT_ROOT}/tracker"
print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
mkdir -p "${OUTPUT_ROOT}/tracker"
"${LOCAL_LOWLEVEL_CMD[@]}" 2>&1 | tee "${OUTPUT_ROOT}/tracker/train.log"
echo "[PASS] ${ANCHOR_MODE} encoder drives the low level in a ${ANCHOR_MODE} env"

# Gates 3, 4 -- the negative checks. The macro state is the same width in every
# mode, so a mispaired frame is invisible downstream; only this refusal proves
# the guard is wired.
for refused in "${REFUSED_MODES[@]}"; do
    negative_dir="${OUTPUT_ROOT}/tracker_negative_${refused}"
    build_local_lowlevel_command "${ENCODER}" "${negative_dir}" "${refused}"
    print_cmd "${LOCAL_LOWLEVEL_CMD[@]}"
    mkdir -p "${negative_dir}"
    if "${LOCAL_LOWLEVEL_CMD[@]}" > "${negative_dir}/train.log" 2>&1; then
        fail "env.expert_macro_anchor_mode=${refused} ACCEPTED a ${ANCHOR_MODE} encoder; guard not wired"
    fi
    grep -q "macro-window anchor mode does not match" "${negative_dir}/train.log" \
        || fail "mode ${refused} failed for the wrong reason; see ${negative_dir}/train.log"
    echo "[PASS] env mode ${refused} refuses a ${ANCHOR_MODE} encoder"
done

python3 - "${OUTPUT_ROOT}/status.json" "$(source_contract_hash)" "${ANCHOR_MODE}" \
    "${MACRO_FRAME_STRIDE}" "${WANDB_PROJECT}" "${WANDB_GROUP}" \
    "${REFUSED_MODES[@]}" <<'PY'
import json
import sys

path, source_hash, anchor_mode, stride, project, group, *refused = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(
        {
            "status": "pass",
            "scope": "local qualification only; not a measurement",
            "source_contract_sha256": source_hash,
            "anchor_mode": anchor_mode,
            "macro_frame_stride": int(stride),
            "negative_pairing_checks": {mode: "refused" for mode in refused},
            "deferred_cluster_arm": {"wandb_project": project, "wandb_group": group},
        },
        stream,
        indent=2,
    )
    stream.write("\n")
PY
echo "[PASS] local qualification: ${OUTPUT_ROOT}"
