#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

category_count="${CATEGORY_COUNT:-32}"
output_root="${OUTPUT_ROOT:-outputs/qualitative_analysis/bs5000-multicat-random${category_count}-group-sweep}"
group_spec="${GROUP_IDS:-${GROUP:-}}"
if [[ -n "${group_spec}" ]]; then
    read -r -a groups <<< "${group_spec}"
elif [[ "${SMOKE:-0}" == 1 ]]; then
    groups=(0)
else
    groups=({0..63})
fi

for group in "${groups[@]}"; do
    [[ "${group}" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ ]] || { echo "Invalid group: ${group}" >&2; exit 2; }
    printf -v group_name 'group_%02d' "${group}"
    cmd=(
        pixi run -e isaaclab python experiments/multicat_qualitative_analysis.py intervene
        --headless --device cuda:0 --video --random-base-code --category-count "${category_count}"
        --skill-checkpoint "${SKILL_CHECKPOINT:-logs/skill_encoder/bs5000-multicat/checkpoints/best.pt}"
        --policy-checkpoint "${POLICY_CHECKPOINT:-logs/policy/policy-bs5000-multicat-E12288-R12/2026-07-26_02-36-06_wandb-vv8iswll/models/model_step_5000085504.pt}"
        --manifest "${MANIFEST:-data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000.json}"
        --dataset "${DATASET:-data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000}"
        --output-root "${output_root}/${group_name}"
        --motion "${MOTION:-jog_ff_loop_180_R_002_A091_M}"
        --group "${group}" --rollout-steps "${ROLLOUT_STEPS:-100}" --seed "${SEED:-0}"
        physics=newton_mjwarp
        env.sim.physics.solver_cfg.njmax=320
        env.sim.physics.solver_cfg.nconmax=40
    )
    [[ "${OVERWRITE:-0}" == 1 ]] && cmd+=(--overwrite)
    [[ "${SMOKE:-0}" == 1 ]] && cmd+=(--smoke)
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' env CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}" "${cmd[@]}"
        printf '\n'
    else
        env CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}" TERM=xterm PYTHONUNBUFFERED=1 "${cmd[@]}"
    fi
done
