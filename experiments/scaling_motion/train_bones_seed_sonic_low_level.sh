#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"

TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-12288}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-12}"
FRAME_CAP="${FRAME_CAP:-5000000000}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-$((TRAIN_NUM_ENVS * ROLLOUT_STEPS / 8))}"
MAX_ITERATIONS="${MAX_ITERATIONS:-$(( (FRAME_CAP + TRAIN_NUM_ENVS * ROLLOUT_STEPS - 1) / (TRAIN_NUM_ENVS * ROLLOUT_STEPS) ))}"

# # bs91 deter
# DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/bones_seed_100}"
# MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/manifests/g1_bones_seed_100_sonic_filtered_manifest.json}"
# DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/g1_hl_diffsr_sonic_filtered}"
# LATENT_MODE="deterministic"
# SKILL_CHECKPOINT="logs/bs91-deter/checkpoints/best.pt"
# RUN_NAME=${RUN_NAME:-policy-bs91-deter-E12288-R12}

# # bs5000 deter
# DATA_ROOT="data/bones_seed_sonic_129k_50hz"
# MANIFEST_PATH="${DATA_ROOT}/manifests/bones-seed-sonic-5000.json"
# DATASET_PATH="${DATA_ROOT}/g1_hl_diffsr_5000"
# LATENT_MODE="deterministic"
# SKILL_CHECKPOINT="logs/skill_encoder/bs5000-deter/checkpoints/best.pt"
# RUN_NAME="policy-bs5000-deter-E12288-R12"

# # bs5000 deter larger
# DATA_ROOT="data/bones_seed_sonic_129k_50hz"
# MANIFEST_PATH="${DATA_ROOT}/manifests/bones-seed-sonic-5000.json"
# DATASET_PATH="${DATA_ROOT}/g1_hl_diffsr_5000"
# LATENT_MODE="deterministic"
# SKILL_CHECKPOINT="logs/skill_encoder/bs5000-deter-f512-z512-alldim512x2/checkpoints/best.pt"
# RUN_NAME="policy-bs5000-deter-f512-z512-alldim512x2-E12288-R12"

# bs5000 multicat
DATA_ROOT="data/bones_seed_sonic_129k_50hz"
MANIFEST_PATH="${DATA_ROOT}/manifests/bones-seed-sonic-5000.json"
DATASET_PATH="${DATA_ROOT}/g1_hl_diffsr_5000"
LATENT_MODE="gumbel_multicat"
SKILL_CHECKPOINT="logs/skill_encoder/bs5000-multicat/checkpoints/best.pt"
RUN_NAME="policy-bs5000-multicat-E12288-R12-bc0.5"

OUTPUT_DIR="${OUTPUT_DIR:-${LOG_ROOT}/policy/${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-policy}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"

# agent.logger.backend=wandb
# agent.logger.video=false
# env.latent_command_dim=258
# agent.ipmd.latent_dim=258
# agent.ipmd.command_source=hl_skill
# agent.ipmd.hl_skill_command_mode=z
# agent.ipmd.hl_skill_finetune_enabled=false
# agent.ipmd.latent_learning.command_phase_mode=sin_cos
# agent.ipmd.latent_learning.code_latent_dim=256
# agent.ipmd.reward_loss_coeff=0.0
# agent.ipmd.reward_l2_coeff=0.0
# agent.ipmd.reward_grad_penalty_coeff=0.0
# agent.ipmd.reward_logit_reg_coeff=0.0
# agent.ipmd.reward_param_weight_decay_coeff=0.0
# env.refresh_zarr_dataset=false


# env.random_reset_full_trajectory=true
# env.random_reset_step_min=0
# env.random_reset_step_max=0
# env.adaptive_failure_reset_uniform_ratio=1.0
# env.adaptive_failure_reset_sequence_length_agnostic=true
# env.adaptive_failure_reset_pre_failure_window=200

    # agent.policy.num_cells=[1024,768,768,768] \
    # agent.policy.output_dim=768 \
    # agent.value_function.num_cells=[1024,768,768]

exec pixi run -e isaaclab python scripts/rlopt/train.py \
    --headless \
    --assert-kitless \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algo IPMD \
    --num_envs "${TRAIN_NUM_ENVS}" \
    --seed "${SEED}" \
    --max_iterations "${MAX_ITERATIONS}" \
    agent.logger.project_name=g1-bones-seed-scaling \
    agent.logger.entity=gaochenxiao \
    "agent.logger.group_name=${WANDB_GROUP}" \
    "agent.logger.exp_name=${WANDB_RUN_NAME}" \
    "agent.logger.log_dir=${OUTPUT_DIR}" \
    "agent.save_interval=${SAVE_INTERVAL:-100000000}" \
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}" \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.latent_steps_min=10 \
    agent.ipmd.latent_steps_max=10 \
    agent.ipmd.latent_learning.code_period=10 \
    "agent.collector.frames_per_batch=${ROLLOUT_STEPS}" \
    "agent.loss.mini_batch_size=${MINIBATCH_SIZE}" \
    "env.lafan1_manifest_path=${MANIFEST_PATH}" \
    "env.dataset_path=${DATASET_PATH}" \
    physics=newton_mjwarp \
    "env.sim.physics.solver_cfg.njmax=${NJMAX:-320}" \
    "env.sim.physics.solver_cfg.nconmax=${NCONMAX:-40}" \
    agent.ipmd.rollout_bc_coef=0.5 \
    agent.ipmd.rollout_bc_loss_type=mse
