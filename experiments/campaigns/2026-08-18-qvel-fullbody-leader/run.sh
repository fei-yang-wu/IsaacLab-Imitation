#!/usr/bin/env bash
set -euo pipefail

# Local 10B retrain of the leading arm with joint velocities added back to the
# skill-encoder input.
#
# Arm: cont_det_ln_hold1_fullbody. One variable against the frozen
# `cont_det_ln_hold1` row of 2026-08-15-latent-bottleneck-10b:
# `env.expert_macro_state_terms` moves from `root_qpos` (38/frame, 380-wide
# encoder input) to `full_body` (67/frame, 670-wide encoder input). The added
# channels are the 29 joint velocities. Every other value is byte-identical to
# the base campaign's contract.
#
# Usage, from the repository root:
#   ./experiments/campaigns/2026-08-18-qvel-fullbody-leader/run.sh pretrain
#   ./experiments/campaigns/2026-08-18-qvel-fullbody-leader/run.sh lowlevel
#   ./experiments/campaigns/2026-08-18-qvel-fullbody-leader/run.sh all
#
# The lowlevel stage resumes from its own tracker checkpoints automatically,
# so re-running `lowlevel` after an interruption continues the same 10B
# budget through `cumulative_env_frames`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STAGE="${1:?usage: run.sh pretrain|lowlevel|all}"

ARM="cont_det_ln_hold1_fullbody"
SEED="${SEED:-0}"
TASK="Isaac-Imitation-G1-v2"
WANDB_PROJECT="g1-bones-seed"
WANDB_GROUP="${WANDB_GROUP:-latent-bottleneck-10b}"
WANDB_ARM="cont-det-ln-hold1-fullbody"

Z_DIM=256
COMMAND_DIM=258            # z_dim + sin_cos phase
HOLD=1
STRIDE=1
ANCHOR_MODE="robot_heading"

# The one variable: full_body macro state (qpos + qvel + root), 670-wide.
MACRO_TERMS="[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]"

REF_ARRAYS="${REF_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="bones_seed_sonic_full_129785@e714bbff"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/qvel_fullbody_10b/${ARM}_seed${SEED}}"

# 10B frames: 16384 envs x 24 rollout steps = 393,216 frames/iteration;
# ceil(1e10 / 393216) = 25,432 iterations = 10,000,269,312 frames.
TRAIN_NUM_ENVS=16384
ROLLOUT_STEPS=24
MAX_ITERATIONS=25432
MINIBATCH_SIZE=294912      # 3/4 of frames_per_batch

RUNTIME_BODY_NAMES="[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"

DATA_OVERRIDES=(
    physics=newton_mjwarp
    env.data.manifest=null
    "env.data.reference_arrays_dir=${REF_ARRAYS}"
    "env.data.persist_id=${PERSIST_ID}"
    env.data.reference_arrays_resident=true
    env.data.reference_arrays_warm_workers=16
    env.data.runtime_cache_device=cpu
    env.data.macro_cache_device=cuda:0
    "env.data.runtime_cache_body_names=${RUNTIME_BODY_NAMES}"
    "env.expert_macro_state_terms=${MACRO_TERMS}"
    "env.expert_macro_frame_stride=${STRIDE}"
    "env.expert_macro_anchor_mode=${ANCHOR_MODE}"
)

[[ -d "${REF_ARRAYS}" ]] || { echo "missing reference arrays: ${REF_ARRAYS}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

run_pretrain() {
    pixi run -e isaaclab python scripts/rlopt/train_hl_skill_diffsr.py \
        --task "${TASK}" \
        --num_envs 16 \
        --seed "${SEED}" \
        --device cuda:0 \
        --headless \
        --assert-kitless \
        --output_dir "${OUTPUT_ROOT}/encoder" \
        --logger_backend wandb \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group "${WANDB_GROUP}" \
        --wandb_run_name "${WANDB_ARM}-pretrain-s${SEED}" \
        --horizon_steps 10 \
        --encoder_window_mode intermediate \
        --z_dim "${Z_DIM}" \
        --latent_mode deterministic \
        --encoder_hidden_dims 2048 1024 512 512 \
        --encoder_activation silu \
        --encoder_layer_norm \
        --diffsr_feature_dim 256 \
        --diffsr_embed_dim 1024 \
        --diffsr_g_hidden_dims 1024 1024 512 \
        --diffsr_mu_hidden_dims 1024 1024 512 \
        --batch_size 8192 \
        --num_updates 50000 \
        --log_interval 1000 \
        --eval_batches 4 \
        "${DATA_OVERRIDES[@]}"
}

run_lowlevel() {
    local encoder="${OUTPUT_ROOT}/encoder/checkpoints/latest.pt"
    [[ -f "${encoder}" ]] || { echo "missing encoder: ${encoder} (run pretrain first)" >&2; exit 1; }

    # Refuse a stale root_qpos encoder: the first layer must read 670 inputs.
    pixi run python - "${encoder}" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
k, v = next(iter(d["skill_encoder_state_dict"].items()))
n = int(v.shape[1])
assert n == 670, f"encoder input width {n}, expected 670 (full_body); {k}"
print(f"encoder ok: {k} {tuple(v.shape)} full_body")
PY

    local resume_args=()
    if compgen -G "${OUTPUT_ROOT}/tracker/*/models/*" > /dev/null; then
        resume_args=(--checkpoint "${OUTPUT_ROOT}/tracker")
        echo "resuming from ${OUTPUT_ROOT}/tracker"
    fi

    pixi run -e isaaclab python scripts/rlopt/train.py \
        --task "${TASK}" \
        --algo IPMD \
        --agent rlopt_ipmd_tuned_cfg_entry_point \
        --num_envs "${TRAIN_NUM_ENVS}" \
        --seed "${SEED}" \
        --headless \
        --assert-kitless \
        --max_iterations "${MAX_ITERATIONS}" \
        "${resume_args[@]}" \
        "agent.logger.log_dir=${OUTPUT_ROOT}/tracker" \
        agent.logger.backend=wandb \
        agent.logger.video=false \
        "agent.logger.project_name=${WANDB_PROJECT}" \
        "agent.logger.group_name=${WANDB_GROUP}" \
        "agent.logger.exp_name=${WANDB_ARM}-s${SEED}" \
        env.command_interface.actor=latent \
        "env.command_interface.actor.dim=${COMMAND_DIM}" \
        env.command_interface.encoder=single \
        "agent.ipmd.latent_dim=${COMMAND_DIM}" \
        agent.ipmd.command_source=hl_skill \
        "agent.ipmd.hl_skill_checkpoint_path=${encoder}" \
        agent.ipmd.hl_skill_horizon_steps=10 \
        agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "agent.ipmd.latent_steps_min=${HOLD}" \
        "agent.ipmd.latent_steps_max=${HOLD}" \
        "agent.ipmd.latent_learning.code_period=${HOLD}" \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}" \
        env.rewards.action_rate_l2.weight=0.0 \
        env.rewards.tracking_reward_points.weight=4.0 \
        env.enable_termination_curriculum=true \
        env.termination_curriculum_start_frames=5000000 \
        env.termination_curriculum_end_frames=30000000 \
        env.command_interface.reference.selection=random80_adaptive20 \
        env.data.reference_prefetch_mode=next_and_reset \
        "agent.collector.frames_per_batch=${ROLLOUT_STEPS}" \
        "agent.loss.mini_batch_size=${MINIBATCH_SIZE}" \
        agent.ipmd.expert_batch_size=24576 \
        agent.loss.gamma=0.97 \
        agent.save_interval=500000000 \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=200 \
        "agent.policy.num_cells=[2048,2048,1024,1024,512,512]" \
        "agent.value_function.num_cells=[2048,2048,1024,1024,512,512]" \
        agent.policy.activation_fn=silu \
        agent.value_function.activation_fn=silu \
        "${DATA_OVERRIDES[@]}"
}

case "${STAGE}" in
    pretrain) run_pretrain ;;
    lowlevel) run_lowlevel ;;
    all) run_pretrain && run_lowlevel ;;
    *) echo "unknown stage: ${STAGE}" >&2; exit 1 ;;
esac
