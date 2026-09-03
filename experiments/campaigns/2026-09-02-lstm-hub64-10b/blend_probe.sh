#!/usr/bin/env bash
# Composability probe: drive a tracker on clip A, then blend clip B's skill
# code in with a linear ramp, and record the target environment.
#
#   ./blend_probe.sh                                  # lstm_affine then lstm
#   ARMS="lstm_affine" RANKS="31865 108687" START=200 RAMP=100 STEPS=450 ./blend_probe.sh
#
# Environment 0 tracks RANK_A (the walk) and is the one recorded; environment
# 1 tracks RANK_B (the jog) and only supplies its code. From step START the
# published code of env 0 is (1 - a) z_A + a z_B with a ramping to 1 over
# RAMP control steps (50 Hz), then held. `evaluate_checkpoint.py
# --latent_blend_*` does the mixing (`imitation_experiments.evaluation.
# latent_blend`); the summary JSON carries alpha and the A-B code distance
# per step under metadata.latent_blend. PhysX, because the recording camera
# needs the Kit render pipeline.
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
MIRROR="${MIRROR:-${REPO_ROOT}/logs/latent64_probe_mirror}"
HUB_ENCODER="${HUB_ENCODER:-${REPO_ROOT}/logs/latent_star_v2_mirror/hub_seed0/encoder/checkpoints/latest.pt}"
P5_ENCODER="${P5_ENCODER:-${MIRROR}/p5_concat_encoder/latest.pt}"
P5_AFFINE_ENCODER="${P5_AFFINE_ENCODER:-${MIRROR}/p5_affine_encoder/latest.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
PERSIST_ID="${PERSIST_ID:-bones_seed_sonic_full_129785@e714bbff}"
# 31865 walk_forward_professional_003_A001_M (1388 frames), 108687
# jog_forward_loop_001_A029_M (467 frames); both survive on the 4,096 board.
RANKS="${RANKS:-31865 108687}"
START="${START:-200}"
RAMP="${RAMP:-100}"
STEPS="${STEPS:-450}"
ARMS="${ARMS:-lstm_affine lstm}"
OUT_ROOT="${OUT_ROOT:-${MIRROR}/blend}"
# TERMINATIONS=board judges env 0 against ITS OWN reference (the walk) with the
# board's anchor / ee_body_pos terminations, which fire as soon as the blended
# gait leaves the walk (v4, 2026-09-02: every run ended on ee_body_pos ~10-20
# steps after the ramp). TERMINATIONS=fall_only (default) drops the
# reference-relative terminations and keeps the env's fall detector
# (base_too_low), so survival means "did not fall".
TERMINATIONS="${TERMINATIONS:-fall_only}"
if [ "${TERMINATIONS}" = "fall_only" ]; then
    TERM_ARGS=(
        env.terminations.anchor_pos=null
        env.terminations.anchor_ori=null
        env.terminations.ee_body_pos=null
        env.terminations.foot_pos_xyz=null
    )
else
    TERM_ARGS=(
        env.terminations.anchor_pos.params.threshold=0.25
        env.terminations.anchor_pos.params.down_threshold=0.25
        env.terminations.anchor_ori.params.threshold=1.0
        env.terminations.ee_body_pos.params.threshold=0.25
        env.terminations.ee_body_pos.params.down_threshold=0.25
        env.terminations.foot_pos_xyz=null env.terminations.base_too_low=null
    )
fi
for arm in ${ARMS}; do
    # Numeric sort on the frame count (10B sorts before 4.5B as text).
    TRACKER="$(ls "${MIRROR}"/ckpt/"${arm}"_latest_f*.pt 2>/dev/null | sed -E 's/.*_f([0-9]+)\.pt/\1 &/' | sort -k1,1n | tail -1 | cut -d' ' -f2-)"
    ENCODER="${HUB_ENCODER}"; ARM_ARGS=()
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
        *) echo "unknown arm ${arm}" >&2; exit 1 ;;
    esac
    for required in "${TRACKER}" "${ENCODER}" "${REFERENCE_ARRAYS}"; do
        [ -e "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
    done
    OUT="${OUT_ROOT}/${arm}_r$(echo ${RANKS} | tr ' ' '-')_s${START}_r${RAMP}_${TERMINATIONS}"
    mkdir -p "${OUT}"
    echo "=== ${arm}: ${TRACKER} -> ${OUT}"
    pixi run -e isaaclab python -m imitation_experiments.lowlevel.evaluate_checkpoint \
        --task Isaac-Imitation-G1-v2 --algo IPMD \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --checkpoint "${TRACKER}" --output_json "${OUT}/summary.json" --label "${arm}_blend" \
        --num_envs 2 --trajectory_ranks ${RANKS} --steps "${STEPS}" \
        --randomization none --action_sampling mode --seed 0 \
        --reference_start_frame 0 --reset_schedule sequential \
        --skill_encoder_source pretrained \
        --video --video_dir "${OUT}/video" --video_length "${STEPS}" --video_follow_env 0 \
        --latent_blend_source_env 1 --latent_blend_target_env 0 \
        --latent_blend_start_step "${START}" --latent_blend_ramp_steps "${RAMP}" \
        --headless \
        physics=physx env.data.manifest=null env.data.cache_dir=null \
        env.data.reference_arrays_dir="${REFERENCE_ARRAYS}" \
        env.data.persist_id="${PERSIST_ID}" \
        env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
        env.data.wrap_steps=false \
        'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
        env.events.push_robot=null \
        env.command_interface.actor=latent env.command_interface.actor.dim=66 \
        env.command_interface.encoder=single \
        'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
        env.expert_macro_frame_stride=1 env.expert_macro_anchor_mode=robot_heading \
        "${TERM_ARGS[@]}" \
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
    echo "retained: ${OUT}"
done
