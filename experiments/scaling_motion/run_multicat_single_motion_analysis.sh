#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cmd=(
    pixi run -e isaaclab python experiments/multicat_qualitative_analysis.py single
    --headless --device cuda:0
    --skill-checkpoint "${SKILL_CHECKPOINT:-logs/skill_encoder/bs5000-multicat/checkpoints/best.pt}"
    --manifest "${MANIFEST:-data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000.json}"
    --dataset "${DATASET:-data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000}"
    --output-root "${OUTPUT_ROOT:-outputs/qualitative_analysis/bs5000-multicat}"
    --motion "${MOTION:-jog_ff_loop_180_R_002_A091_M}"
    --stride "${STRIDE:-10}" --seed "${SEED:-0}"
    physics=newton_mjwarp
)
[[ "${OVERWRITE:-0}" == 1 ]] && cmd+=(--overwrite)
[[ "${SMOKE:-0}" == 1 ]] && cmd+=(--smoke)
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}" "${cmd[@]}"; printf '\n'; exit 0; fi
env CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}" TERM=xterm PYTHONUNBUFFERED=1 "${cmd[@]}"
