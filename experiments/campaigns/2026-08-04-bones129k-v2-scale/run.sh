#!/usr/bin/env bash
set -euo pipefail

# Local two-stage scale run for the complete SONIC-filtered BONES-SEED export.
# Pretraining and low-level training are deliberately separate stages: inspect
# the held-out encoder curve before allowing the expensive controller run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/scripts/rlopt/run_local_v2_pipeline.sh" ]]; do
    [[ "${REPO_ROOT}" == "/" ]] && { echo "[FATAL] repository root not found" >&2; exit 2; }
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

fail() { echo "[FATAL] $*" >&2; exit 1; }
print_cmd() { printf '  '; printf '%q ' "$@"; printf '\n'; }

STAGE="${STAGE:-plan}"
case "${STAGE}" in
    plan|pretrain|lowlevel) ;;
    *) fail "STAGE must be plan, pretrain, or lowlevel; got '${STAGE}'." ;;
esac

TASK_NAME="${TASK_NAME:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"

DATA_ROOT="${DATA_ROOT:-/mnt/storage/fwu91/bones_seed_full}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/manifests/g1_bones_seed_sonic_full_manifest.json}"
ZARR_PATH="${ZARR_PATH:-${DATA_ROOT}/zarr/g1_bones_seed_sonic_full}"
BUFFER_PATH="${BUFFER_PATH:-${DATA_ROOT}/rb/g1_bones_seed_sonic_full}"
BUFFER_MANIFEST="${BUFFER_PATH}/iltools_rb_manifest.json"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-eb0ad052afe72fb6228f4be9d52132c9cb9a52ac9c561751e49e6ca31346e688}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
EXPECTED_MOTIONS="${EXPECTED_MOTIONS:-129785}"

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

HORIZON_STEPS="${HORIZON_STEPS:-10}"
Z_DIM="${Z_DIM:-256}"
LATENT_HOLD_STEPS="${LATENT_HOLD_STEPS:-10}"
LATENT_COMMAND_DIM=$((Z_DIM + 2))
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-100}"
PRETRAIN_EVAL_BATCHES="${PRETRAIN_EVAL_BATCHES:-4}"

# 24,576 x 24 already OOMed in the v2 screen. The 24,576 x 6 geometry fit at
# 63.2 GiB and is the deliberate wall-clock-convergence probe requested here.
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-24576}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-6}"
TOTAL_FRAMES="${TOTAL_FRAMES:-1000000000}"
FRAMES_PER_ITER=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
MAX_ITERATIONS=$(((TOTAL_FRAMES + FRAMES_PER_ITER - 1) / FRAMES_PER_ITER))
SAVE_INTERVAL="${SAVE_INTERVAL:-25000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000000}"

PHYSICS="${PHYSICS:-newton_mjwarp}"
NJMAX="${NJMAX:-288}"
NCONMAX="${NCONMAX:-200}"

WANDB_PROJECT="${WANDB_PROJECT:-g1-lafan1}"
WANDB_GROUP="${WANDB_GROUP:-bones129k-v2-scale}"
WANDB_TAGS="${WANDB_TAGS:-bones-seed,129785,v2,root-qpos,split-cache,runtime-cache,det-sr,h10,z256,tuned,e24576,r6,1b}"
RUN_TAG="${RUN_TAG:-bones129k_root_qpos_v2_splitcache_e${TRAIN_NUM_ENVS}_r${ROLLOUT_STEPS}_1b_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/runs/${RUN_TAG}}"
ENCODER_DIR="${ENCODER_DIR:-${OUTPUT_ROOT}/encoder}"
ENCODER_CKPT="${ENCODER_CKPT:-${ENCODER_DIR}/checkpoints/latest.pt}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-${OUTPUT_ROOT}/rlopt_train}"

[[ -f "${MANIFEST_PATH}" ]] || fail "manifest not found: ${MANIFEST_PATH}"
[[ -d "${ZARR_PATH}" ]] || fail "Zarr not found: ${ZARR_PATH}"
[[ -f "${BUFFER_MANIFEST}" ]] || fail "persisted buffer manifest not found: ${BUFFER_MANIFEST}"

actual_manifest_sha="$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')"
[[ "${actual_manifest_sha}" == "${EXPECTED_MANIFEST_SHA256}" ]] \
    || fail "manifest SHA mismatch: expected=${EXPECTED_MANIFEST_SHA256} actual=${actual_manifest_sha}"

python3 -c '
import json, sys
manifest_path, expected_id, expected_count, expected_keys = sys.argv[1:]
record = json.load(open(manifest_path, encoding="utf-8"))
source = record.get("key", {}).get("source", {})
traj = record.get("traj_info", {})
actual_keys = record.get("key", {}).get("keys")
expected_keys = expected_keys.split(",")
if source.get("persist_id") != expected_id:
    raise SystemExit("persist_id mismatch: {!r}".format(source.get("persist_id")))
if len(traj.get("ordered_traj_list", [])) != int(expected_count):
    raise SystemExit("persisted trajectory count mismatch")
if actual_keys != expected_keys:
    raise SystemExit(f"persisted key mismatch: {actual_keys!r}")
print("[PASS] persisted buffer: {} transitions, {} motions, {}".format(
    traj.get("written"), expected_count, expected_id
))
' "${BUFFER_MANIFEST}" "${PERSIST_ID}" "${EXPECTED_MOTIONS}" "$(IFS=,; echo "${DATA_KEYS[*]}")"

[[ "${TRAIN_NUM_ENVS}" =~ ^[0-9]+$ ]] && (( TRAIN_NUM_ENVS > 0 )) \
    || fail "TRAIN_NUM_ENVS must be positive."
[[ "${ROLLOUT_STEPS}" =~ ^[0-9]+$ ]] && (( ROLLOUT_STEPS > 0 )) \
    || fail "ROLLOUT_STEPS must be positive."
[[ "${TOTAL_FRAMES}" =~ ^[0-9]+$ ]] && (( TOTAL_FRAMES > 0 )) \
    || fail "TOTAL_FRAMES must be positive."

export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export WANDB_TAGS

pretrain_cmd=(
    pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py
    --task "${TASK_NAME}" --num_envs "${PRETRAIN_NUM_ENVS}"
    --seed "${SEED}" --device "${DEVICE}" --headless --assert-kitless
    --output_dir "${ENCODER_DIR}"
    --horizon_steps "${HORIZON_STEPS}"
    --encoder_window_mode intermediate
    --z_dim "${Z_DIM}" --latent_mode deterministic
    --batch_size "${PRETRAIN_BATCH_SIZE}"
    --num_updates "${PRETRAIN_UPDATES}"
    --log_interval "${PRETRAIN_LOG_INTERVAL}"
    --eval_batches "${PRETRAIN_EVAL_BATCHES}"
    --reconstruction_eval --window_probe_eval
    --window_probe_train_batches 8 --window_probe_eval_batches 4
    --logger_backend wandb --wandb_project "${WANDB_PROJECT}"
    --wandb_group "${WANDB_GROUP}"
    --wandb_run_name "${RUN_TAG}_pretrain"
    "physics=${PHYSICS}"
    env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
    env.data.manifest=null
    "env.data.cache_dir=${ZARR_PATH}"
    env.data.cache_refresh=false
    env.data.storage_device=cpu
    "env.data.macro_cache_device=${DEVICE}"
    "env.data.persist_dir=${BUFFER_PATH}"
    "env.data.persist_id=${PERSIST_ID}"
    "${DATA_KEYS_OVERRIDE}"
)

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

echo "[PLAN] data        : ${EXPECTED_MOTIONS} motions / ${PERSIST_ID}"
echo "[PLAN] pretrain    : ${PRETRAIN_UPDATES} updates x ${PRETRAIN_BATCH_SIZE} samples"
echo "[PLAN] macro state : root+qpos (38/frame, $((HORIZON_STEPS * 38))D h${HORIZON_STEPS} window)"
echo "[PLAN] live cache  : qpos/qvel + ${#RUNTIME_BODY_NAMES[@]} tracked bodies in host RAM"
echo "[PLAN] low level   : ${TRAIN_NUM_ENVS} envs x ${ROLLOUT_STEPS} steps = ${FRAMES_PER_ITER} frames/iter"
echo "[PLAN] frame budget: ${MAX_ITERATIONS} iterations -> $((MAX_ITERATIONS * FRAMES_PER_ITER)) frames"
echo "[PLAN] output      : ${OUTPUT_ROOT}"
echo "[PLAN] W&B         : ${WANDB_PROJECT} / ${WANDB_GROUP} / ${WANDB_TAGS}"

if [[ "${STAGE}" == "plan" ]]; then
    echo "[PLAN] pretrain command:"
    print_cmd "${pretrain_cmd[@]}"
    echo "[PLAN] low-level command (run only after the pretrain curve is accepted):"
    print_cmd "${lowlevel_cmd[@]}"
    exit 0
fi

[[ "${CONFIRM_RUN:-}" == "bones129k-v2-scale" ]] \
    || fail "launch requires CONFIRM_RUN=bones129k-v2-scale"

case "${STAGE}" in
    pretrain)
        [[ ! -e "${ENCODER_DIR}/metrics.jsonl" && ! -e "${ENCODER_CKPT}" ]] \
            || fail "pretrain output already exists under ${ENCODER_DIR}; choose a new OUTPUT_ROOT"
        mkdir -p "${ENCODER_DIR}"
        print_cmd "${pretrain_cmd[@]}"
        exec "${pretrain_cmd[@]}"
        ;;
    lowlevel)
        [[ -s "${ENCODER_CKPT}" ]] || fail "accepted encoder checkpoint not found: ${ENCODER_CKPT}"
        if [[ -d "${TRAIN_LOG_DIR}" && -n "$(find "${TRAIN_LOG_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            fail "low-level output already exists under ${TRAIN_LOG_DIR}; choose a new OUTPUT_ROOT"
        fi
        mkdir -p "${TRAIN_LOG_DIR}"
        print_cmd "${lowlevel_cmd[@]}"
        exec "${lowlevel_cmd[@]}"
        ;;
esac
