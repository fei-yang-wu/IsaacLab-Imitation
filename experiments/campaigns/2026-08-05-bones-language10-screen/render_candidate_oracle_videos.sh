#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

CANDIDATES="${CANDIDATES:-${SCRIPT_DIR}/candidates.json}"
CHECKPOINT="${CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt}"
ENCODER="${ENCODER:-logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/bones_language30_oracle_videos/rollout24_gamma097_3p5b_seed0_randomized_no_push}"

for artifact in "${CANDIDATES}" "${CHECKPOINT}" "${ENCODER}" \
    "${REFERENCE_ARRAYS}/reference_arrays_manifest.json"; do
    [[ -f "${artifact}" ]] || { echo "[ERROR] Missing ${artifact}" >&2; exit 2; }
done

mapfile -t TRAJECTORY_RANKS < <(
    pixi run python -c \
        'import json,sys; print(*[x["trajectory_rank"] for x in json.load(open(sys.argv[1]))["candidates"]], sep="\n")' \
        "${CANDIDATES}"
)

pixi run python .agents/skills/policy-eval-video/scripts/render_policy_videos.py \
    --checkpoint "${CHECKPOINT}" \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --output_root "${OUTPUT_ROOT}" \
    --reference_arrays "${REFERENCE_ARRAYS}" --persist_id "${PERSIST_ID}" \
    --ranks "${TRAJECTORY_RANKS[@]}" --randomized_no_push --skip_existing -- \
    physics=newton_mjwarp \
    env.sim.physics.solver_cfg.njmax=289 env.sim.physics.solver_cfg.nconmax=200 \
    env.data.reference_arrays_warm_workers=8 \
    env.data.reference_prefetch_mode=next \
    env.data.macro_cache_device=cuda:0 \
    env.data.runtime_cache_device=cpu \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    env.command_interface.actor.dim=258 \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    env.terminations.foot_pos_xyz=null \
    agent.ipmd.latent_dim=258 \
    agent.ipmd.command_source=hl_skill \
    "agent.ipmd.hl_skill_checkpoint_path=${ENCODER}" \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_steps_min=10 \
    agent.ipmd.latent_steps_max=10 \
    agent.ipmd.latent_learning.code_period=10 \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim=256 \
    agent.ipmd.hl_skill_finetune_enabled=false
