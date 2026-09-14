#!/usr/bin/env bash
set -euo pipefail

# Stage 2 of 2: train the IPMD low-level tracker against the frozen scaled
# SONIC-FSQ64 encoder from stage 1, on the full 129,785-motion BONES-SEED
# reference arrays.
#
#   DRY_RUN=1 bash experiments/qualitative/train_lowlevel_fsq64.sh
#   DRY_RUN=1 TRACKER_ARM=sonic bash experiments/qualitative/train_lowlevel_fsq64.sh
#   bash experiments/qualitative/train_lowlevel_fsq64.sh
#
# Every training argument below is copied verbatim from the accepted campaign
# experiments/campaigns/2026-08-06-bones129k-sonic-fsq-scale/run.sh
# (`build_cluster_lowlevel_command` + `append_lowlevel_contract` +
# `COMMON_ENV_OVERRIDES`; ICE jobs 5570680 `tuned` and 5570936 `sonic`). The
# only deltas are local execution: pixi instead of
# docker/cluster/cluster_interface.sh, a repo-local reference-arrays path, an
# explicit --device, and the data/encoder gates below. Nothing that reaches the
# trainer changed.
#
# The two campaign arms differ ONLY in actor/critic capacity:
#   TRACKER_ARM=tuned  [1024,1024,512]                     tracker-capacity control
#   TRACKER_ARM=sonic  [2048,2048,1024,1024,512,512]       SONIC-sized tracker
# The `sonic` arm ran out of memory on an H100 in the campaign and needed an
# H200; size TRAIN_NUM_ENVS to the card before launching it.
#
# Self-contained by design: every constant is declared here. Stage 1 is
# experiments/qualitative/pretrain_skill_encoder_fsq64.sh and declares its own
# copy. HORIZON_STEPS, Z_DIM, LATENT_MODE, and the macro state terms must match
# the encoder being loaded -- the width assert below catches a mismatch before
# training, so a pre-v2 670-wide encoder is refused rather than silently paired.

# -f, not -x: train.py is mode 644 and is invoked as `python scripts/rlopt/train.py`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -f "${REPO_ROOT}/scripts/rlopt/train.py" ]]; do
    [[ "${REPO_ROOT}" == "/" ]] && { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }

DRY_RUN="${DRY_RUN:-0}"

# --- protocol (campaign values) ---------------------------------------------
TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
PHYSICS="${PHYSICS:-newton_mjwarp}"
NJMAX="${NJMAX:-289}"
NCONMAX="${NCONMAX:-200}"

# Must match the encoder being loaded: root+qpos is 29 joint positions + root
# position + 6D root orientation = 38 per frame, so the encoder input is 380.
MACRO_STATE_TERMS_OVERRIDE=env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
MACRO_STATE_VALUES_PER_FRAME=38

HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-64}"
LATENT_MODE="${LATENT_MODE:-sonic_fsq}"
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"
ENCODER_INPUT_WIDTH=$((HORIZON_STEPS * MACRO_STATE_VALUES_PER_FRAME))
# The published command is the 64-value FSQ code + the 2-value sin_cos phase.
# Dropping the phase is catastrophic: episode length 21 against 144 on the
# 2026-08-02 screen.
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# --- tracker capacity (the only difference between campaign arms) -----------
TRACKER_ARM="${TRACKER_ARM:-tuned}"
case "${TRACKER_ARM}" in
    tuned) TRACKER_CELLS="[1024,1024,512]" ;;
    sonic) TRACKER_CELLS="[2048,2048,1024,1024,512,512]" ;;
    *) fail "TRACKER_ARM must be tuned or sonic; got ${TRACKER_ARM}" ;;
esac
TRACKER_ACTIVATION="${TRACKER_ACTIVATION:-silu}"

# --- rollout geometry (campaign values) -------------------------------------
# 16,384 x 24 is the campaign's single-H200 geometry. Lower TRAIN_NUM_ENVS on a
# smaller card; the minibatch follows it automatically.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-5000000000}"
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH * 3 / 4))}"
ONLINE_EXPERT_BATCH_SIZE="${ONLINE_EXPERT_BATCH_SIZE:-24576}"
GAMMA="${GAMMA:-0.97}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

# --- reference data ---------------------------------------------------------
# The campaign read /data/bones_seed_ref_arrays/... inside the ICE container.
# Same content, same persist id, local path.
REFERENCE_ARRAYS_DIR="${REFERENCE_ARRAYS_DIR:-${REPO_ROOT}/data/g1_bones_seed_sonic_129k_50hz_refarrays}"
REFERENCE_ARRAYS_RESIDENT="${REFERENCE_ARRAYS_RESIDENT:-true}"
REFERENCE_ARRAYS_WARM_WORKERS="${REFERENCE_ARRAYS_WARM_WORKERS:-16}"
REFERENCE_PREFETCH_MODE="${REFERENCE_PREFETCH_MODE:-next}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
EXPECTED_MOTIONS="${EXPECTED_MOTIONS:-129785}"
EXPECTED_TRANSITIONS="${EXPECTED_TRANSITIONS:-47491234}"
ANCHOR_BODY="${ANCHOR_BODY:-pelvis}"

MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/data/bones_seed_sonic_129k_50hz/g1_bones_seed_sonic_full_manifest.json}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-eb0ad052afe72fb6228f4be9d52132c9cb9a52ac9c561751e49e6ca31346e688}"

# Order is column position in the arrays, not a set. Do not sort or edit.
RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
RUNTIME_BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

# --- environment contract (COMMON_ENV_OVERRIDES, verbatim) ------------------
# `random80_adaptive20` chooses a trajectory uniformly and a frame uniformly
# within its first half on 80% of resets; the other 20% use the learned SONIC
# failure distribution.
REFERENCE_SELECTION="${REFERENCE_SELECTION:-random80_adaptive20}"
COMMON_ENV_OVERRIDES=(
    "${MACRO_STATE_TERMS_OVERRIDE}"
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0
    env.enable_termination_curriculum=true
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
    "env.command_interface.reference.selection=${REFERENCE_SELECTION}"
    "env.sim.physics.solver_cfg.njmax=${NJMAX}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
)

# --- encoder ----------------------------------------------------------------
# Mirrors stage 1's default output path, including its latent mode, so an
# ablated encoder is found without passing ENCODER_CKPT by hand. The campaign
# loaded checkpoints/latest.pt of the shared pretrain.
ABLATE_LOG_ROOT="${ABLATE_LOG_ROOT:-${REPO_ROOT}/logs/ablate_latent}"
ENCODER_RUN_TAG="${ENCODER_RUN_TAG:-bones129k_encoder_${LATENT_MODE}_h${HORIZON_STEPS}_z${Z_DIM}_seed${SEED}}"
ENCODER_CKPT="${ENCODER_CKPT:-${ABLATE_LOG_ROOT}/encoder/${ENCODER_RUN_TAG}/checkpoints/latest.pt}"

# --- output and logging -----------------------------------------------------
# The campaign's W&B destination. Confirm or override the group before a real
# launch; do not silently mix an ablation into the campaign's group.
WANDB_PROJECT="${WANDB_PROJECT:-g1-bones-seed}"
WANDB_GROUP="${WANDB_GROUP:-skill-encoding-ablation}"
BUDGET_LABEL="$((FRAME_CAP / 1000000))m"
LOWLEVEL_RUN_TAG="${LOWLEVEL_RUN_TAG:-bones129k_scaled_fsq64_${TRACKER_ARM}_tracker_${BUDGET_LABEL}_seed${SEED}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${ABLATE_LOG_ROOT}/lowlevel/${LOWLEVEL_RUN_TAG}}"
WANDB_TAGS="${WANDB_TAGS:-bones-seed,129785,v2,root-qpos,sonic-fsq64,lowlevel,${TRACKER_ARM}-tracker,newton,rollout${ROLLOUT_STEPS},gamma097,reset80-adaptive20}"

# --- data and encoder gates -------------------------------------------------
if [[ -f "${MANIFEST_PATH}" ]]; then
    actual_manifest_sha="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
    [[ "${actual_manifest_sha}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
        || fail "manifest SHA mismatch: expected=${EXPECTED_MANIFEST_SHA256} actual=${actual_manifest_sha}"
    echo "[PASS] manifest provenance: ${MANIFEST_PATH}"
else
    echo "[NOTE] manifest absent (${MANIFEST_PATH}); relying on the arrays' own identity gate."
fi

[[ -d "${REFERENCE_ARRAYS_DIR}" ]] || fail "reference arrays not found: ${REFERENCE_ARRAYS_DIR}
  Fetch them with
    pixi run python -m imitation_experiments.data.publish_reference_arrays fetch \\
      --repo_id GeorgiaTech/g1_bones_seed_sonic_129k_50hz_refarrays \\
      --dest_dir ${REFERENCE_ARRAYS_DIR} --persist_id ${PERSIST_ID} \\
      --expected_motions ${EXPECTED_MOTIONS} --expected_transitions ${EXPECTED_TRANSITIONS}
  or build them from the NPZ tree with
    pixi run python -m imitation_experiments.data.build_reference_arrays --help"

pixi run python -m imitation_experiments.data.build_reference_arrays \
    --manifest "${MANIFEST_PATH}" \
    --output_dir "${REFERENCE_ARRAYS_DIR}" \
    --persist_id "${PERSIST_ID}" \
    --anchor_body "${ANCHOR_BODY}" \
    --body_names "${RUNTIME_BODY_NAMES[@]}" \
    --expected_motions "${EXPECTED_MOTIONS}" \
    --expected_transitions "${EXPECTED_TRANSITIONS}" \
    --validate_only

[[ -s "${ENCODER_CKPT}" ]] || fail "encoder checkpoint not found: ${ENCODER_CKPT}
  Run experiments/qualitative/pretrain_skill_encoder_fsq64.sh first, or point
  ENCODER_CKPT at an accepted h${HORIZON_STEPS} root_qpos ${LATENT_MODE} encoder."

# A z_dim, window-mode, or macro-state mismatch is silent until the command
# space is already wrong, so assert the encoder's input width instead.
#
# Read the trunk's first Linear by name. The quantized latent modes
# (sonic_fsq, fsq, gumbel_multicat, categorical, vq) prepend a codebook or
# quantizer parameter to the state dict, so taking the first entry reads that
# instead and reports a bogus mismatch.
pixi run python -c '
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = int(sys.argv[2])
state_dict = checkpoint["skill_encoder_state_dict"]
trunk = [
    (k, v) for k, v in state_dict.items()
    if k.startswith("net.") and k.endswith(".weight") and v.ndim == 2
]
if not trunk:
    raise SystemExit(
        "no trunk Linear weight (net.<i>.weight) in skill_encoder_state_dict; "
        f"keys: {list(state_dict)[:8]}"
    )
key, value = trunk[0]
if int(value.shape[1]) != expected:
    raise SystemExit(
        f"root_qpos encoder width mismatch: {key} {tuple(value.shape)}, expected {expected}"
    )
print(f"[PASS] root_qpos encoder: {key} {tuple(value.shape)}")
' "${ENCODER_CKPT}" "${ENCODER_INPUT_WIDTH}"
ENCODER_SHA256="$(sha256sum "${ENCODER_CKPT}" | awk '{print $1}')"
echo "[PASS] encoder sha256: ${ENCODER_SHA256}"

export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_TAGS

# --- command ----------------------------------------------------------------
# The encoder is FROZEN here (hl_skill_finetune_enabled=false): the tracker
# learns against a fixed command space, which is what makes the published
# command comparable across arms.
lowlevel_cmd=(
    pixi run -e isaaclab python scripts/rlopt/train.py
    --task "${TASK_NAME}" --num_envs "${TRAIN_NUM_ENVS}" --headless
    --algo IPMD --agent "${AGENT_ENTRY_POINT}" --seed "${SEED}"
    --device "${DEVICE}"
    --max_iterations "${MAX_ITERATIONS}" --log_interval "${LOG_INTERVAL}"
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    "physics=${PHYSICS}"
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS_DIR}"
    "env.data.persist_id=${PERSIST_ID}"
    "env.data.reference_arrays_resident=${REFERENCE_ARRAYS_RESIDENT}"
    "env.data.reference_arrays_warm_workers=${REFERENCE_ARRAYS_WARM_WORKERS}"
    env.data.runtime_cache_device=cpu
    "env.data.reference_prefetch_mode=${REFERENCE_PREFETCH_MODE}"
    "env.data.macro_cache_device=${DEVICE}"
    "${RUNTIME_BODY_NAMES_OVERRIDE}"
    "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
    "agent.loss.mini_batch_size=${MINIBATCH_SIZE}"
    "agent.ipmd.expert_batch_size=${ONLINE_EXPERT_BATCH_SIZE}"
    "agent.loss.gamma=${GAMMA}"
    "agent.save_interval=${SAVE_INTERVAL}"
    agent.logger.backend=wandb agent.logger.video=false
    "agent.logger.project_name=${WANDB_PROJECT}"
    "agent.logger.group_name=${WANDB_GROUP}"
    "agent.logger.exp_name=${LOWLEVEL_RUN_TAG}"
    "agent.logger.log_dir=${TRAIN_LOG_DIR}"
    # --- append_lowlevel_contract, verbatim ---
    "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
    "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}"
    agent.ipmd.command_source=hl_skill
    "agent.ipmd.hl_skill_checkpoint_path=${ENCODER_CKPT}"
    "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
    agent.ipmd.hl_skill_command_mode=z
    "agent.ipmd.latent_steps_min=${LATENT_HOLD_STEPS}"
    "agent.ipmd.latent_steps_max=${LATENT_HOLD_STEPS}"
    "agent.ipmd.latent_learning.code_period=${LATENT_HOLD_STEPS}"
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
    agent.ipmd.hl_skill_finetune_enabled=false
    "agent.policy.num_cells=${TRACKER_CELLS}"
    "agent.policy.activation_fn=${TRACKER_ACTIVATION}"
    "agent.value_function.num_cells=${TRACKER_CELLS}"
    "agent.value_function.activation_fn=${TRACKER_ACTIVATION}"
    "${COMMON_ENV_OVERRIDES[@]}"
)

echo "[PLAN] stage       : 2 of 2 -- low-level tracker"
echo "[PLAN] contract    : 2026-08-06-bones129k-sonic-fsq-scale ${TRACKER_ARM} arm"
echo "[PLAN] data        : ${EXPECTED_MOTIONS} motions / ${EXPECTED_TRANSITIONS} transitions"
echo "[PLAN] source      : reference arrays (mapped) ${REFERENCE_ARRAYS_DIR}"
echo "[PLAN] encoder     : ${ENCODER_CKPT}"
echo "[PLAN] encoder sha : ${ENCODER_SHA256}"
echo "[PLAN] command     : ${Z_DIM} FSQ code + 2 phase = ${LATENT_COMMAND_DIM}, held ${LATENT_HOLD_STEPS} steps"
echo "[PLAN] tracker     : ${TRACKER_ARM} ${TRACKER_CELLS} ${TRACKER_ACTIVATION}, actor and critic"
echo "[PLAN] reset       : ${REFERENCE_SELECTION}"
echo "[PLAN] live cache  : qpos/qvel + ${#RUNTIME_BODY_NAMES[@]} tracked bodies in host RAM"
echo "[PLAN] geometry    : ${TRAIN_NUM_ENVS} envs x ${ROLLOUT_STEPS} steps = ${FRAMES_PER_BATCH} frames/iter, minibatch ${MINIBATCH_SIZE}"
echo "[PLAN] frame budget: ${MAX_ITERATIONS} iterations -> $((MAX_ITERATIONS * FRAMES_PER_BATCH)) frames"
echo "[PLAN] output      : ${TRAIN_LOG_DIR}"
echo "[PLAN] W&B         : ${WANDB_PROJECT} / ${WANDB_GROUP} / ${WANDB_TAGS}"
printf '  '; printf '%q ' "${lowlevel_cmd[@]}"; printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

mkdir -p "${TRAIN_LOG_DIR}"
exec "${lowlevel_cmd[@]}"
