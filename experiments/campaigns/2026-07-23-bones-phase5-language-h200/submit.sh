#!/usr/bin/env bash
set -euo pipefail

# ICE binds the centralized project logs tree at this container-visible path.
# The latent checkpoint is the verified H200 run under the RLOpt log layout;
# the H10 skill encoder is kept under the campaign's BONES-SEED artifact tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ICE_LATENT_CHECKPOINT="${ICE_LATENT_CHECKPOINT:-/workspace/isaaclab/project/logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-07-22_16-13-52_wandb-jlhhpdvm/models/model_step_4975165440.pt}"
export ICE_SKILL_CHECKPOINT="${ICE_SKILL_CHECKPOINT:-/workspace/isaaclab/project/logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt}"
export VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-/workspace/isaaclab/project/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-07-15_00-32-35/models/model_step_1000046592.pt}"

exec "${SCRIPT_DIR}/submit_impl.sh" "$@"
