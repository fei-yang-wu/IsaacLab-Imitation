#!/usr/bin/env bash
set -euo pipefail

# Low-level-only BONES-SEED reset ablation. Reuses the accepted root_qpos
# encoder and the guarded full replay cache; it never reruns pretraining.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
# -f, not -x: train.py is mode 644 and is invoked as `python scripts/rlopt/train.py`,
# so an executable-bit test walks past the root every time.
while [[ ! -f "${REPO_ROOT}/scripts/rlopt/train.py" ]]; do
    [[ "${REPO_ROOT}" == "/" ]] && { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }
print_cmd() { printf '  '; printf '%q ' "$@"; printf '\n'; }

STAGE="${STAGE:-plan}"
case "${STAGE}" in
    plan|lowlevel) ;;
    *) fail "STAGE must be plan or lowlevel; got '${STAGE}'." ;;
esac

TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"

DATA_ROOT="${DATA_ROOT:-/mnt/storage/fwu91/bones_seed_full}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/manifests/g1_bones_seed_sonic_full_manifest.json}"
ZARR_PATH="${ZARR_PATH:-${DATA_ROOT}/zarr/g1_bones_seed_sonic_full}"
BUFFER_PATH="${BUFFER_PATH:-${DATA_ROOT}/rb/g1_bones_seed_sonic_full_129785_e714bbff_v2}"

# Where the reference data comes from.
#   arrays : prebuilt training-shaped arrays, memory-mapped from NVMe. The Zarr
#            and the replay are never opened; process start reads about 50 GB
#            sequentially instead of gathering about 133 GB from the 94.5 GiB
#            replay, which on the spinning DATA_ROOT disk is 12-20 minutes.
#   replay : the original Zarr -> persisted replay -> derived caches path,
#            retained so the two can be compared directly.
DATA_SOURCE="${DATA_SOURCE:-arrays}"
case "${DATA_SOURCE}" in
    arrays|replay) ;;
    *) fail "DATA_SOURCE must be arrays or replay; got '${DATA_SOURCE}'." ;;
esac
REFERENCE_ARRAYS_DIR="${REFERENCE_ARRAYS_DIR:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
REFERENCE_ARRAYS_WARM_WORKERS="${REFERENCE_ARRAYS_WARM_WORKERS:-8}"
# Matches the v2 reference channel's anchor_body_name; the arrays bake it in and
# loading refuses a directory built for a different one.
ANCHOR_BODY="${ANCHOR_BODY:-pelvis}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-eb0ad052afe72fb6228f4be9d52132c9cb9a52ac9c561751e49e6ca31346e688}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
EXPECTED_MOTIONS="${EXPECTED_MOTIONS:-129785}"
EXPECTED_TRANSITIONS="${EXPECTED_TRANSITIONS:-47491234}"

DATA_KEYS=(
    qpos qvel root_pos root_quat root_lin_vel root_ang_vel
    joint_pos joint_vel body_pos_w body_quat_w body_lin_vel_w body_ang_vel_w
)
DATA_KEYS_OVERRIDE="env.data.keys=[$(IFS=,; echo "${DATA_KEYS[*]}")]"
RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
RUNTIME_BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"
RUNTIME_CACHE_CHUNK_SIZE="${RUNTIME_CACHE_CHUNK_SIZE:-262144}"

ENCODER_CKPT="${ENCODER_CKPT:-${DATA_ROOT}/runs/bones129k_root_qpos_v2_splitcache_e24576_r6_1b_seed0/encoder/checkpoints/latest.pt}"
EXPECTED_ENCODER_SHA256="${EXPECTED_ENCODER_SHA256:-d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))

# The prior 1B run established 32,768 x 6 as the largest local geometry with
# a reasonable reserve on the 96 GiB RTX PRO 6000. The replay cache is disk
# mapped, the live 14-body cache consumes about 44.8 GiB host RAM, and the
# compact root_qpos source consumes about 6.4 GiB VRAM.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-32768}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-6}"
TOTAL_FRAMES="${TOTAL_FRAMES:-10000000000}"
FRAMES_PER_ITER=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((TOTAL_FRAMES + FRAMES_PER_ITER - 1) / FRAMES_PER_ITER))
SAVE_INTERVAL="${SAVE_INTERVAL:-25000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000000}"

PHYSICS="${PHYSICS:-newton_mjwarp}"
NJMAX="${NJMAX:-289}"
NCONMAX="${NCONMAX:-200}"
REFERENCE_SELECTION="${REFERENCE_SELECTION:-sonic}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-bones129k-v2-adaptive-reset}"
WANDB_TAGS="${WANDB_TAGS:-bones-seed,129785,Isaac-Imitation-G1-v2,v2,root-qpos,split-cache,runtime-cache,det-sr,h10,z256,tuned,adaptive-full-trajectory,e${TRAIN_NUM_ENVS},r${ROLLOUT_STEPS},10b}"
RUN_TAG="${RUN_TAG:-bones129k_root_qpos_v2_adaptive_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}_10b_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/runs/${RUN_TAG}}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${OUTPUT_ROOT}/rlopt_train}"

[[ -f "${MANIFEST_PATH}" ]] || fail "manifest not found: ${MANIFEST_PATH}"
[[ -s "${ENCODER_CKPT}" ]] || fail "accepted encoder not found: ${ENCODER_CKPT}"

actual_manifest_sha="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
[[ "${actual_manifest_sha}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
    || fail "manifest SHA mismatch: expected=${EXPECTED_MANIFEST_SHA256} actual=${actual_manifest_sha}"
actual_encoder_sha="$(sha256sum "${ENCODER_CKPT}" | awk '{print $1}')"
[[ "${actual_encoder_sha}" == "${EXPECTED_ENCODER_SHA256}" ]] \
    || fail "encoder SHA mismatch: expected=${EXPECTED_ENCODER_SHA256} actual=${actual_encoder_sha}"

# Both sources are gated on the same content identity: 129,785 motions,
# 47,491,234 transitions, and the persist id derived from the selection hash.
if [[ "${DATA_SOURCE}" == "arrays" ]]; then
    [[ -d "${REFERENCE_ARRAYS_DIR}" ]] \
        || fail "reference arrays not found: ${REFERENCE_ARRAYS_DIR}. Build them with
  pixi run python -m imitation_experiments.data.build_reference_arrays \\
    --manifest ${MANIFEST_PATH} \\
    --traj_info ${DATA_ROOT}/rb_packed/g1_bones_seed_sonic_full/iltools_rb_manifest.json \\
    --output_dir ${REFERENCE_ARRAYS_DIR} --persist_id ${PERSIST_ID} \\
    --anchor_body ${ANCHOR_BODY} --body_names ${RUNTIME_BODY_NAMES[*]} \\
    --workers 5 --expected_motions ${EXPECTED_MOTIONS} \\
    --expected_transitions ${EXPECTED_TRANSITIONS} --verify_load"
    pixi run python -m imitation_experiments.data.build_reference_arrays \
        --manifest "${MANIFEST_PATH}" \
        --output_dir "${REFERENCE_ARRAYS_DIR}" \
        --persist_id "${PERSIST_ID}" \
        --anchor_body "${ANCHOR_BODY}" \
        --body_names "${RUNTIME_BODY_NAMES[@]}" \
        --expected_motions "${EXPECTED_MOTIONS}" \
        --expected_transitions "${EXPECTED_TRANSITIONS}" \
        --validate_only \
        --verify_load
else
    [[ -d "${ZARR_PATH}" ]] || fail "Zarr not found: ${ZARR_PATH}"
    pixi run python \
        .agents/skills/bones-seed-dataset/scripts/build_replay_cache.py \
        --zarr "${ZARR_PATH}" \
        --persist-dir "${BUFFER_PATH}" \
        --persist-id "${PERSIST_ID}" \
        --expected-motions "${EXPECTED_MOTIONS}" \
        --expected-transitions "${EXPECTED_TRANSITIONS}" \
        --validate-only \
        --verify-load
fi

pixi run python -c '
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
key, value = next(iter(checkpoint["skill_encoder_state_dict"].items()))
if int(value.shape[1]) != 380:
    raise SystemExit(f"root_qpos encoder width mismatch: {key} {tuple(value.shape)}")
print(f"[PASS] root_qpos encoder: {key} {tuple(value.shape)}")
' "${ENCODER_CKPT}"

[[ "${TRAIN_NUM_ENVS}" =~ ^[0-9]+$ ]] && (( TRAIN_NUM_ENVS > 0 )) \
    || fail "TRAIN_NUM_ENVS must be positive."
[[ "${ROLLOUT_STEPS}" =~ ^[0-9]+$ ]] && (( ROLLOUT_STEPS > 0 )) \
    || fail "ROLLOUT_STEPS must be positive."
[[ "${TOTAL_FRAMES}" =~ ^[0-9]+$ ]] && (( TOTAL_FRAMES > 0 )) \
    || fail "TOTAL_FRAMES must be positive."
[[ "${REFERENCE_SELECTION}" == "sonic" ]] \
    || fail "this campaign is pinned to REFERENCE_SELECTION=sonic"

export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_TAGS

if [[ "${DATA_SOURCE}" == "arrays" ]]; then
    # No cache_dir, no persist_dir, no key list: the arrays are the source, and
    # naming a Zarr here would only invite a future run to open one.
    DATA_SOURCE_OVERRIDES=(
        "env.data.reference_arrays_dir=${REFERENCE_ARRAYS_DIR}"
        "env.data.reference_arrays_warm_workers=${REFERENCE_ARRAYS_WARM_WORKERS}"
        env.data.runtime_cache_device=cpu
        "env.data.macro_cache_device=${DEVICE}"
        "${RUNTIME_BODY_NAMES_OVERRIDE}"
        "env.data.persist_id=${PERSIST_ID}"
    )
else
    DATA_SOURCE_OVERRIDES=(
        "env.data.cache_dir=${ZARR_PATH}"
        env.data.cache_refresh=false
        env.data.storage_device=cpu
        "env.data.macro_cache_device=${DEVICE}"
        env.data.runtime_cache_device=cpu
        "env.data.runtime_cache_chunk_size=${RUNTIME_CACHE_CHUNK_SIZE}"
        "${RUNTIME_BODY_NAMES_OVERRIDE}"
        "env.data.persist_dir=${BUFFER_PATH}"
        "env.data.persist_id=${PERSIST_ID}"
        "${DATA_KEYS_OVERRIDE}"
    )
fi

lowlevel_cmd=(
    pixi run -e isaaclab python scripts/rlopt/train.py
    --task "${TASK_NAME}" --algo IPMD --agent "${AGENT_ENTRY_POINT}"
    --headless --assert-kitless --device "${DEVICE}"
    --num_envs "${TRAIN_NUM_ENVS}" --seed "${SEED}"
    --max_iterations "${MAX_ITERATIONS}" --log_interval "${LOG_INTERVAL}"
    --kit_args=--/app/extensions/fsWatcherEnabled=false
    "physics=${PHYSICS}"
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    "env.sim.physics.solver_cfg.njmax=${NJMAX}"
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX}"
    env.data.manifest=null
    "${DATA_SOURCE_OVERRIDES[@]}"
    "env.command_interface.reference.selection=${REFERENCE_SELECTION}"
    "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}"
    "agent.collector.frames_per_batch=${ROLLOUT_STEPS}"
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
    "agent.save_interval=${SAVE_INTERVAL}"
    agent.logger.backend=wandb agent.logger.video=false
    "agent.logger.project_name=${WANDB_PROJECT}"
    "agent.logger.group_name=${WANDB_GROUP}"
    "agent.logger.exp_name=${RUN_TAG}_lowlevel"
    "agent.logger.log_dir=${TRAIN_LOG_DIR}"
    env.rewards.action_rate_l2.weight=0.0
    env.rewards.tracking_reward_points.weight=4.0
    env.enable_termination_curriculum=true
    env.termination_curriculum_start_frames=5000000
    env.termination_curriculum_end_frames=30000000
)

echo "[PLAN] data        : ${EXPECTED_MOTIONS} motions / ${EXPECTED_TRANSITIONS} transitions"
echo "[PLAN] encoder     : root_qpos / ${EXPECTED_ENCODER_SHA256}"
echo "[PLAN] reset       : sonic full-trajectory adaptive failure sampling"
if [[ "${DATA_SOURCE}" == "arrays" ]]; then
    echo "[PLAN] source      : reference arrays (mapped) ${REFERENCE_ARRAYS_DIR}"
else
    echo "[PLAN] source      : Zarr + persisted replay ${BUFFER_PATH}"
fi
echo "[PLAN] live cache  : qpos/qvel + ${#RUNTIME_BODY_NAMES[@]} tracked bodies in host RAM"
echo "[PLAN] low level   : ${TRAIN_NUM_ENVS} envs x ${ROLLOUT_STEPS} steps = ${FRAMES_PER_ITER} frames/iter"
echo "[PLAN] frame budget: ${MAX_ITERATIONS} iterations -> $((MAX_ITERATIONS * FRAMES_PER_ITER)) frames"
echo "[PLAN] output      : ${OUTPUT_ROOT}"
echo "[PLAN] W&B         : ${WANDB_PROJECT} / ${WANDB_GROUP} / ${WANDB_TAGS}"
print_cmd "${lowlevel_cmd[@]}"

if [[ "${STAGE}" == "plan" ]]; then
    exit 0
fi

[[ "${CONFIRM_RUN:-}" == "bones129k-v2-adaptive-10b" ]] \
    || fail "launch requires CONFIRM_RUN=bones129k-v2-adaptive-10b"
if [[ -d "${TRAIN_LOG_DIR}" && -n "$(find "${TRAIN_LOG_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    fail "low-level output already exists under ${TRAIN_LOG_DIR}; choose a new OUTPUT_ROOT"
fi
mkdir -p "${TRAIN_LOG_DIR}"
exec "${lowlevel_cmd[@]}"
