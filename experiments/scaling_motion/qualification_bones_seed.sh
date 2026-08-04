#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# # Select one checkpoint/data configuration.
# NAME="${NAME:-bs91-deter}"
# SKILL_DIR="${SKILL_DIR:-logs/skill_encoder/bs91-deter}"
# POLICY_DIR="${POLICY_DIR:-logs/policy/policy-bs91-deter-E12288-R12}"
# MANIFEST="${MANIFEST:-data/bones_seed_100/manifests/g1_bones_seed_100_sonic_filtered_manifest.json}"
# DATASET="${DATASET:-data/bones_seed_100/g1_hl_diffsr_sonic_filtered}"
# EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-91}"

# NAME="${NAME:-bs5000-deter}"
# SKILL_DIR="${SKILL_DIR:-logs/skill_encoder/bs5000-deter}"
# POLICY_DIR="${POLICY_DIR:-logs/policy/policy-bs5000-deter-E12288-R12}"
# MANIFEST="${MANIFEST:-data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000.json}"
# DATASET="${DATASET:-data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000}"
# EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-5000}"

# NAME="${NAME:-bs5000-deter-bc0.1}"
# SKILL_DIR="${SKILL_DIR:-logs/skill_encoder/bs5000-deter}"
# POLICY_DIR="${POLICY_DIR:-logs/policy/policy-bs5000-deter-E12288-R12-bc0.1}"
# MANIFEST="${MANIFEST:-data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000.json}"
# DATASET="${DATASET:-data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000}"
# EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-5000}"

NAME="${NAME:-bs5000-multicat-bc0.5}"
SKILL_DIR="${SKILL_DIR:-logs/skill_encoder/bs5000-multicat}"
POLICY_DIR="${POLICY_DIR:-logs/policy/policy-bs5000-multicat-E12288-R12-bc0.5}"
MANIFEST="${MANIFEST:-data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000.json}"
DATASET="${DATASET:-data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-5000}"

GPU_INDEX="${GPU_INDEX:-0}"
# Examples: MOTION_RANKS="0 1 2 10" or MOTION_RANKS="0,1,2,10".
MOTION_RANKS="${MOTION_RANKS:-${MOTION_RANK:-0}}"
read -r -a MOTION_RANK_LIST <<< "${MOTION_RANKS//,/ }"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/qualification/${NAME}-$(date +%Y%m%d_%H%M%S)}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-${SKILL_DIR}/checkpoints/best.pt}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-$(find "${POLICY_DIR}" -type f -name 'model_step_*.pt' | sort -V | tail -n 1)}"

SKILL_CHECKPOINT="$(realpath -m "${SKILL_CHECKPOINT}")"
LOW_LEVEL_CHECKPOINT="$(realpath -m "${LOW_LEVEL_CHECKPOINT}")"
MANIFEST="$(realpath -m "${MANIFEST}")"
DATASET="$(realpath -m "${DATASET}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"
ORACLE_DIR="${OUTPUT_ROOT}/oracle"
mkdir -p "${ORACLE_DIR}"

COMMON_OVERRIDES=(
    physics=newton_mjwarp
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}"
    agent.ipmd.hl_skill_horizon_steps=10
    agent.ipmd.latent_steps_min=10
    agent.ipmd.latent_steps_max=10
    agent.ipmd.latent_learning.code_period=10
    "env.lafan1_manifest_path=${MANIFEST}"
    "env.dataset_path=${DATASET}"
    "env.keys=[qpos,qvel,next_qpos,next_qvel,body_pos_w,body_quat_w,body_lin_vel_w,body_ang_vel_w]"
    env.sim.physics.solver_cfg.njmax=320
    env.sim.physics.solver_cfg.nconmax=40
)

pixi run python experiments/interface_baselines/validate_latent_skill_checkpoint_binding.py \
    --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${SKILL_CHECKPOINT}" \
    --output_json "${OUTPUT_ROOT}/skill_binding.json"

env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" TERM=xterm PYTHONUNBUFFERED=1 \
pixi run -e isaaclab python experiments/scaling_motion/eval_skill_commander_closed_loop.py \
    --algorithm IPMD \
    --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${SKILL_CHECKPOINT}" \
    --output_dir "${ORACLE_DIR}" \
    --label "${NAME}_diffsr_oracle" \
    --num_envs "${EVAL_NUM_ENVS}" \
    --max_steps 1000 \
    --keep_time_out \
    --extend_episode_length_for_max_steps \
    --keep_early_terminations \
    --disable_reward_clipping \
    agent.logger.backend= \
    env.reset_schedule=sequential \
    env.wrap_steps=false \
    env.observations.policy.enable_corruption=false \
    "${COMMON_OVERRIDES[@]}"

for MOTION_RANK in "${MOTION_RANK_LIST[@]}"; do
    VIDEO_DIR="${OUTPUT_ROOT}/side_by_side_rank${MOTION_RANK}"
    mkdir -p "${VIDEO_DIR}"

    env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" TERM=xterm PYTHONUNBUFFERED=1 \
    pixi run -e isaaclab python experiments/scaling_motion/compare_policy_reference.py \
        --video \
        --task Isaac-Imitation-G1-Latent-v0 \
        --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --output_dir "${VIDEO_DIR}" \
        --seed 0 \
        --policy_trajectory_rank "${MOTION_RANK}" \
        --restrict_dataset_to_policy_trajectory \
        --reference_visualization robot \
        --metrics_json "${VIDEO_DIR}/metrics.json" \
        env.observations.policy.enable_corruption=false \
        "${COMMON_OVERRIDES[@]}"

    VIDEO_PATH="$(find "${VIDEO_DIR}" -type f -name '*.mp4' | sort | tail -n 1)"
    printf 'Rank %s side-by-side video: %s\n' "${MOTION_RANK}" "${VIDEO_PATH}"
done

pixi run python experiments/interface_baselines/audit_diffsr_latent_qualification.py \
    --summary "${ORACLE_DIR}/summary.json" \
    --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${SKILL_CHECKPOINT}" \
    --manifest "${MANIFEST}" \
    --expected_dataset_path "${DATASET}" \
    --expected_num_envs "${EVAL_NUM_ENVS}" \
    --require_pass \
    --output_json "${OUTPUT_ROOT}/qualification.json"
