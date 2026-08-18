#!/usr/bin/env bash
# 2026-08-13 — fsq64 low level under the ROBOT-CENTRIC HEADING anchor frame,
# full BONES-SEED (129,785 motions), local workstation.
#
# One variable against the deployed fsq64 tracker: the frame the DiffSR macro
# window is expressed in.
#
#   robot          (deployed)  full live robot pose; upstream `motion_anchor_ori_b_mf`
#   robot_heading  (this arm)  live robot HEADING (yaw-only) frame, xy-only origin;
#                              upstream `motion_anchor_ori_heading_mf_nonflat`,
#                              i.e. SONIC v1.1's own convention
#
# `robot_heading` keeps the reference's tilt relative to gravity and cancels
# only global yaw and xy — the invariance group of the re-rooted tracking
# reward. Unlike `expert_heading` it anchors at the ROBOT, so the encoder input
# still carries live tracking error.
#
# TWO STAGES, and stage 1 is not optional. The macro state has the SAME WIDTH
# in every anchor mode, so a mismatched encoder cannot be caught by a shape
# check. The agent rebuilds its SkillConfig FROM the checkpoint
# (`HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])`), so the
# encoder's recorded layer-norm setting must match its actual weights --
# `--no_encoder_layer_norm` keeps both False, as the deployed encoder is.
# `_require_matching_macro_anchor_mode` compares the mode recorded in
# the skill checkpoint against the environment and refuses to pair them; there
# is no override. The deployed fsq64 encoder was pretrained under `robot`, so
# this arm needs a FRESH encoder pretrained under `robot_heading`. Only the
# tracker weights can be warm-started.
#
# Usage:
#   run.sh pretrain     # stage 1: fresh fsq64 encoder in the new frame
#   run.sh lowlevel     # stage 2: RL, warm-started from the deployed tracker
#   run.sh smoke        # tiny end-to-end check of both stages
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STAGE="${1:?usage: run.sh <pretrain|lowlevel|smoke>}"

ANCHOR_MODE="robot_heading"
TASK="${TASK:-Isaac-Imitation-G1-v2}"
AGENT_ENTRY_POINT="${AGENT_ENTRY_POINT:-rlopt_ipmd_tuned_cfg_entry_point}"
SEED="${SEED:-0}"

REF_ARRAYS="${REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/bones129k_fsq_robot_heading}"

# Warm start for stage 2. The tracker's proprioception and actuation knowledge
# transfer; its COMMAND interpretation does not, because the latent it reads is
# now expressed in a different frame. Expect early progress to look like
# partial retraining, not a smooth continuation.
# NO encoder warm start, deliberately. `train_hl_skill_diffsr.py --checkpoint`
# restores the checkpoint's ENTIRE recorded config and overrides the CLI, so
# warm-starting from the deployed fsq64 encoder silently reset
# `macro_anchor_mode` back to 'robot' -- the exact variable under test.
# Verified 2026-08-13: a smoke asking for 20 updates at batch 64 under
# 'robot_heading' recorded anchor='robot', num_updates=50000, batch=8192 and
# resumed from update 50000. Stage 1 pretrains from scratch in the new frame;
# only the TRACKER is warm-started, in stage 2, via a separate mechanism.
# FROM SCRATCH, both stages (user decision 2026-08-13). No tracker warm start:
# `train.py --checkpoint` is a full resume (optimizer + step counter), which
# would leave the 2B frame budget ambiguous; from scratch the cap is exact and
# both frames' trackers see only their own convention from step 0.

# -- fsq64 contract, reproduced from the deployed encoder's recorded config --
HORIZON_STEPS=10
MACRO_FRAME_STRIDE=1
Z_DIM=64
LATENT_COMMAND_DIM=$((Z_DIM + 2))   # + 2-wide sin_cos phase channel
PRETRAIN_NUM_ENVS="${PRETRAIN_NUM_ENVS:-16}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-50000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8192}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-16384}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-24}"
FRAMES_PER_BATCH=$((TRAIN_NUM_ENVS * ROLLOUT_STEPS))
FRAME_CAP="${FRAME_CAP:-2000000000}"
MAX_ITERATIONS=$(((FRAME_CAP + FRAMES_PER_BATCH - 1) / FRAMES_PER_BATCH))
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((FRAMES_PER_BATCH * 3 / 4))}"
# `--checkpoint` is a full resume: it restores the optimizer and the step
# counter too, so the smoke below is what proves the 4.5B-step counter does
# not immediately satisfy MAX_ITERATIONS.
ONLINE_EXPERT_BATCH_SIZE="${ONLINE_EXPERT_BATCH_SIZE:-24576}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000000}"
LOG_INTERVAL="${LOG_INTERVAL:-2000000}"

RUNTIME_BODY_NAMES=(
    pelvis
    left_hip_roll_link left_knee_link left_ankle_roll_link
    right_hip_roll_link right_knee_link right_ankle_roll_link
    torso_link
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
)
BODY_NAMES_OVERRIDE="env.data.runtime_cache_body_names=[$(IFS=,; echo "${RUNTIME_BODY_NAMES[*]}")]"

# Cadence AND frame must agree between the two stages. Two of these fail loudly
# at low-level startup; keeping them in one array is what stops them drifting.
MACRO_INTERFACE_OVERRIDES=(
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]'
    "env.expert_macro_frame_stride=${MACRO_FRAME_STRIDE}"
    "env.expert_macro_anchor_mode=${ANCHOR_MODE}"
)
DATA_OVERRIDES=(
    physics=newton_mjwarp
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REF_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=false
    env.data.reference_arrays_warm_workers=2
    env.data.runtime_cache_device=cpu
    env.data.macro_cache_device=cuda:0
    "${BODY_NAMES_OVERRIDE}"
)
# SONIC-scale tracker capacity, matching the deployed fsq64 arm.
TRACKER_CAPACITY=(
    'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
    'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
    agent.policy.activation_fn=silu
    agent.value_function.activation_fn=silu
)

ENCODER_DIR="${OUTPUT_ROOT}/encoder"
TRACKER_DIR="${OUTPUT_ROOT}/tracker"

if [ "${STAGE}" = "smoke" ]; then
    PRETRAIN_UPDATES=20; PRETRAIN_BATCH_SIZE=64; PRETRAIN_NUM_ENVS=4
    TRAIN_NUM_ENVS=4; ROLLOUT_STEPS=4; MINIBATCH_SIZE=8
    ONLINE_EXPERT_BATCH_SIZE=8; MAX_ITERATIONS=2
    ENCODER_DIR="${OUTPUT_ROOT}/smoke_encoder"; TRACKER_DIR="${OUTPUT_ROOT}/smoke_tracker"
    # The smoke reruns from clean, so it deletes its own directories. Refuse to
    # delete anything not named `smoke_*`: on 2026-08-13 a stage-2 smoke was
    # pointed at the real `encoder/` to reuse its weights and this `rm` then
    # destroyed a finished 50k-update pretrain. Use SMOKE_ENCODER_DIR to borrow
    # a trained encoder instead of redirecting ENCODER_DIR.
    for victim in "${ENCODER_DIR}" "${TRACKER_DIR}"; do
        case "$(basename "${victim}")" in
            smoke_*) rm -rf "${victim}" ;;
            *) echo "refusing to delete non-smoke directory ${victim}" >&2; exit 1 ;;
        esac
    done
    # Borrow a trained encoder for a stage-2-only smoke, read-only.
    if [ -n "${SMOKE_ENCODER_DIR:-}" ]; then ENCODER_DIR="${SMOKE_ENCODER_DIR}"; fi
fi

run_pretrain() {
    [ -e "${ENCODER_DIR}" ] && [ "${STAGE}" != "smoke" ] && {
        echo "Refusing to overwrite existing ${ENCODER_DIR}" >&2; exit 1; }
    pixi run -e isaaclab python -u scripts/rlopt/train_hl_skill_diffsr.py \
        --task "${TASK}" --num_envs "${PRETRAIN_NUM_ENVS}" --seed "${SEED}" \
        --device cuda:0 --headless --assert-kitless \
        --output_dir "${ENCODER_DIR}" --logger_backend none \
        --horizon_steps "${HORIZON_STEPS}" \
        --encoder_window_mode intermediate --transition_objective endpoint \
        --z_dim "${Z_DIM}" --latent_mode sonic_fsq \
        --encoder_hidden_dims 2048 1024 512 512 --encoder_activation silu \
        --no_encoder_layer_norm \
        --diffsr_feature_dim 256 --diffsr_embed_dim 1024 \
        --diffsr_g_hidden_dims 1024 1024 512 \
        --diffsr_mu_hidden_dims 1024 1024 512 \
        --batch_size "${PRETRAIN_BATCH_SIZE}" --num_updates "${PRETRAIN_UPDATES}" \
        --log_interval 1000 --eval_batches 4 \
        "${DATA_OVERRIDES[@]}" "${MACRO_INTERFACE_OVERRIDES[@]}"
}

run_lowlevel() {
    local encoder="${ENCODER_DIR}/checkpoints/latest.pt"
    [ -f "${encoder}" ] || { echo "missing stage-1 encoder: ${encoder}" >&2; exit 1; }
    [ -e "${TRACKER_DIR}" ] && [ "${STAGE}" != "smoke" ] && {
        echo "Refusing to overwrite existing ${TRACKER_DIR}" >&2; exit 1; }
    # `agent.collector.frames_per_batch` is PER ENVIRONMENT, so one iteration
    # is TRAIN_NUM_ENVS * ROLLOUT_STEPS frames -- that is what MAX_ITERATIONS
    # is derived from above.
    pixi run -e isaaclab python -u scripts/rlopt/train.py \
        --task "${TASK}" --algo IPMD --agent "${AGENT_ENTRY_POINT}" \
        --num_envs "${TRAIN_NUM_ENVS}" --seed "${SEED}" --headless --assert-kitless \
        --max_iterations "${MAX_ITERATIONS}" \
        "agent.logger.log_dir=${TRACKER_DIR}" \
        agent.logger.backend=csv agent.logger.video=false \
        "agent.logger.exp_name=fsq64_robot_heading" \
        env.command_interface.actor=latent \
        "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}" \
        env.command_interface.encoder=single \
        "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${encoder}" \
        "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}" \
        agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.code_period=10 \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}" \
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}" \
        "agent.ipmd.expert_batch_size=${ONLINE_EXPERT_BATCH_SIZE}" \
        agent.loss.gamma=0.97 \
        "agent.save_interval=${SAVE_INTERVAL}" \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=200 \
        "${DATA_OVERRIDES[@]}" "${MACRO_INTERFACE_OVERRIDES[@]}" \
        "${TRACKER_CAPACITY[@]}"
}

case "${STAGE}" in
    pretrain) run_pretrain ;;
    lowlevel) run_lowlevel ;;
    smoke)    run_pretrain && run_lowlevel ;;
    *) echo "stage must be pretrain|lowlevel|smoke" >&2; exit 1 ;;
esac
echo "retained: ${OUTPUT_ROOT}"
