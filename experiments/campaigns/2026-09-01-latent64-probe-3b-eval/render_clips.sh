#!/usr/bin/env bash
# Studio-lit inspection clips of the latent64-probe / lstm-hub64 arms, driven
# by the frozen encoder's own latents (the tracker's ceiling, not a planner).
#
#   ./render_clips.sh                      # the three 3B arms, the five ranks below
#   CKPT=latest ARMS="z64_merged enc_hist obs_hist z64_wd_clin lstm lstm_affine" ./render_clips.sh
#   ARMS="obs_hist" RANKS="8550 13134" ./render_clips.sh
#
# CKPT=3B (default) renders `<arm>_3B.pt`; CKPT=latest renders
# `<arm>_latest_f<frames>.pt` (2026-09-02: 9.5B / 9.5B / 8.5B / 7.0B / 4.5B /
# 4.5B). `lstm*` arms build the recurrent actor (`agent.ppo.rnn_hidden_size`);
# `lstm_affine` binds the past-5 affine encoder.
#
# Same interface overrides as the training campaign; PhysX because the Kit RTX
# camera exists only there (every reported number stays on Newton).
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/latent64_probe_mirror}"
HUB_ENCODER="${HUB_ENCODER:-${REPO_ROOT}/logs/latent_star_v2_mirror/hub_seed0/encoder/checkpoints/latest.pt}"
P5_ENCODER="${P5_ENCODER:-${MIRROR}/p5_concat_encoder/latest.pt}"
P5_AFFINE_ENCODER="${P5_AFFINE_ENCODER:-${MIRROR}/p5_affine_encoder/latest.pt}"
CKPT="${CKPT:-3B}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
# Ranks in the 129,785-clip corpus, chosen from the 3B eval JSONs (2026-09-02):
#   7629  step_in_shit           all three succeed; obs_hist fails ee_body_pos at 175/292
#   8550  dance latino kick      ctrl + enc_hist succeed; obs_hist fails at 249/464
#   13134 vogue cat walk 180     ctrl + enc_hist succeed; obs_hist fails at 92/328
#   831   mohak forward stop     ctrl fails anchor_ori at 288/321; the other two succeed
#   55373 crawl start            all three succeed, MPJPE 86 / 68 / 132 mm
RANKS="${RANKS:-7629 8550 13134 831 55373}"
ARMS="${ARMS:-z64_merged enc_hist obs_hist}"
STYLE="${STYLE:-studio_light}"
SHOT="${SHOT:-hero_low}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
for arm in ${ARMS}; do
    if [ "${CKPT}" = "3B" ]; then
        TRACKER="${MIRROR}/ckpt/${arm}_3B.pt"
    else
        # Numeric sort on the frame count: a plain sort puts f10000269312
        # (10B) before f4500357120 (4.5B).
        TRACKER="$(ls "${MIRROR}"/ckpt/"${arm}"_latest_f*.pt 2>/dev/null | sed -E 's/.*_f([0-9]+)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)"
    fi
    ENCODER="${HUB_ENCODER}"
    ARM_ARGS=()
    case "${arm}" in
        z64_merged|z64_wd_clin) ;;
        enc_hist) ENCODER="${P5_ENCODER}" ;;
        lstm) ENCODER="${P5_ENCODER}"; ARM_ARGS=(agent.ppo.rnn_hidden_size=256) ;;
        lstm_affine) ENCODER="${P5_AFFINE_ENCODER}"; ARM_ARGS=(agent.ppo.rnn_hidden_size=256) ;;
        combo)
            ENCODER="${P5_AFFINE_ENCODER}"
            ARM_ARGS=(
                env.observations.policy.projected_gravity.history_length=10
                env.observations.policy.base_ang_vel.history_length=10
                env.observations.policy.joint_pos_rel.history_length=10
                env.observations.policy.joint_vel_rel.history_length=10
                env.observations.policy.last_action.history_length=10
            ) ;;
        obs_hist)
            ARM_ARGS=(
                env.observations.policy.projected_gravity.history_length=10
                env.observations.policy.base_ang_vel.history_length=10
                env.observations.policy.joint_pos_rel.history_length=10
                env.observations.policy.joint_vel_rel.history_length=10
                env.observations.policy.last_action.history_length=10
            ) ;;
        *) echo "unknown arm ${arm}" >&2; exit 1 ;;
    esac
    for required in "${TRACKER}" "${ENCODER}" "${REFERENCE_ARRAYS}"; do
        [ -e "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
    done
    OUTPUT_DIR="${OUTPUT_ROOT:-${MIRROR}/clips}/${arm}_${CKPT}"
    echo "=== ${arm}: ${TRACKER} -> ${OUTPUT_DIR}"
    pixi run -e isaaclab python scripts/viz/render_paper_policy_video.py \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --checkpoint "${TRACKER}" --ranks ${RANKS} \
        --style "${STYLE}" --shot "${SHOT}" \
        --video_width "${VIDEO_WIDTH}" --video_height "${VIDEO_HEIGHT}" \
        --stills_every 50 \
        --output_dir "${OUTPUT_DIR}" --headless \
        physics=physx env.data.manifest=null env.data.cache_dir=null \
        env.data.reference_arrays_dir="${REFERENCE_ARRAYS}" \
        env.data.persist_id="${PERSIST_ID}" \
        env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
        env.data.wrap_steps=false \
        'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
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
        ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} \
        "$@"
    echo "retained: ${OUTPUT_DIR}"
done
