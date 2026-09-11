#!/usr/bin/env bash
# Render the same clips with the parent tracker and its action-rate finetune,
# for a before/after side-by-side (Isaac Sim, PhysX, RTX camera, studio look).
#
#   before  combo-50b at 50.0B      action_rate_l2 -0.03 (the hub recipe)
#   after   action01rate05 at 60.0B action_rate_l2 -0.5, finetuned from
#           action01 (-0.1, itself a continuation of combo-50b)
#
# Same encoder (p5_affine), same interface, same ranks, same seed and camera,
# so the only difference between the two columns is the tracker weights.
# All terminations are disabled by the renderer, so both columns run the whole
# clip and the frame counts match.
#
#   ./render.sh                # both policies, default ranks
#   POLICIES="after" RANKS="23968" ./render.sh
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

BEFORE_CKPT="${BEFORE_CKPT:-${REPO_ROOT}/logs/latent64_probe_mirror/ckpt/combo50b_latest_f50000166912.pt}"
AFTER_CKPT="${AFTER_CKPT:-${REPO_ROOT}/logs/action_rate_bundles/ckpt/action01rate05_f60000043008.pt}"
ENCODER="${ENCODER:-${REPO_ROOT}/logs/latent64_probe_mirror/p5_affine_encoder/latest.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/logs/action_rate_side_by_side}"
# Chosen from the two arms' 4096-board rows (both succeed on every one):
#   23968  reaching_down_R_001_A047_M      body jerk 120.5 -> 8.9   (slow reach)
#   121068 look_over_fence_180_R_001_A462  176.6 -> 33.8            (turn + lean)
#   68390  legs_relax_003_A037             136.6 -> 27.5            (near-idle)
#   102061 mohak_backward_loop_001_A030_M  519.9 -> 161.3           (locomotion)
#   22475  dance_hiphop_hip_hop_ii_R_003   499.5 -> 279.4           (dance)
RANKS="${RANKS:-23968 121068 68390 102061 22475}"
POLICIES="${POLICIES:-before after}"
STYLE="${STYLE:-studio_light}"
SHOT="${SHOT:-hero_low}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"

for required in "${BEFORE_CKPT}" "${AFTER_CKPT}" "${ENCODER}" "${REFERENCE_ARRAYS}"; do
    [ -e "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
done

for policy in ${POLICIES}; do
    case "${policy}" in
        before) TRACKER="${BEFORE_CKPT}" ;;
        after)  TRACKER="${AFTER_CKPT}" ;;
        *) echo "unknown policy ${policy}" >&2; exit 1 ;;
    esac
    OUTPUT_DIR="${OUTPUT_ROOT}/${policy}"
    echo "=== ${policy}: ${TRACKER} -> ${OUTPUT_DIR}"
    pixi run -e isaaclab python scripts/viz/render_paper_policy_video.py \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --agent_entry_point rlopt_ipmd_tuned_fullbatch_cfg_entry_point \
        --checkpoint "${TRACKER}" --ranks ${RANKS} --seed 0 \
        --style "${STYLE}" --shot "${SHOT}" \
        --video_width "${VIDEO_WIDTH}" --video_height "${VIDEO_HEIGHT}" \
        --output_dir "${OUTPUT_DIR}" --headless \
        physics=physx env.data.manifest=null env.data.cache_dir=null \
        env.data.reference_arrays_dir="${REFERENCE_ARRAYS}" \
        env.data.persist_id="${PERSIST_ID}" \
        env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
        env.data.wrap_steps=false \
        'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
        env.observations.policy.projected_gravity.history_length=10 \
        env.observations.policy.base_ang_vel.history_length=10 \
        env.observations.policy.joint_pos_rel.history_length=10 \
        env.observations.policy.joint_vel_rel.history_length=10 \
        env.observations.policy.last_action.history_length=10 \
        env.command_interface.actor=latent \
        env.command_interface.actor.dim=66 \
        env.command_interface.encoder=single \
        'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
        env.expert_macro_frame_stride=1 \
        env.expert_macro_anchor_mode=robot_heading \
        agent.logger.backend= agent.ipmd.command_source=hl_skill \
        agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        agent.ipmd.latent_dim=66 \
        agent.ipmd.latent_steps_min=1 agent.ipmd.latent_steps_max=1 \
        agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=64 \
        agent.ipmd.latent_learning.code_period=1 \
        'agent.policy.num_cells=[2048,2048,1024,1024,512,512]' \
        'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]' \
        agent.policy.activation_fn=silu agent.value_function.activation_fn=silu \
        "$@" > "${OUTPUT_ROOT}/render_${policy}.log" 2>&1
    echo "wrote $(ls "${OUTPUT_DIR}"/videos/*.mp4 2>/dev/null | wc -l) videos to ${OUTPUT_DIR}/videos"
done
