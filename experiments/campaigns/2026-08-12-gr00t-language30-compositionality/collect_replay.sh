#!/usr/bin/env bash
# Replay (kinematic-reference) collection over the 30 compositionality motions.
#
# "Replay data" = `env.replay_only=true`: the robot state is written from the
# reference every step, so the causal sensor pipeline reports expert
# kinematics and no policy dynamics are involved. Rows land at every
# publication boundary (control step 0, 10, 20, ...), each carrying a
# 30-frame expert `root_qpos` lookahead anchored at that publication.
#
# Usage: collect_replay.sh <z256|fsq64>
#
# Two encoder variants are collected because each latent arm must carry its
# OWN stored `z_target` — that is the parity anchor the dataset preparation
# gate checks its offline recompute against. The 30-frame windows and the
# state histories are identical between the two (deterministic replay of the
# same references); `prepare_gr00t_dataset` re-verifies this.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

VARIANT="${1:?usage: collect_replay.sh <z256|fsq64>}"
case "${VARIANT}" in
    z256)
        CHECKPOINT="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt"
        ENCODER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt"
        ACTOR_DIM=258
        CODE_DIM=256
        NJMAX=289
        EXTRA=()
        ;;
    fsq64)
        CHECKPOINT="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/4500357120/fsq64_sonic/model_step_4500357120.pt"
        ENCODER="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/encoders/fsq64_scaled.pt"
        ACTOR_DIM=66
        CODE_DIM=64
        NJMAX=320
        EXTRA=(
            'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
            'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
            agent.policy.activation_fn=silu
            agent.value_function.activation_fn=silu
        )
        ;;
    *)
        echo "Unknown variant ${VARIANT} (expected z256 or fsq64)" >&2
        exit 1
        ;;
esac

OUTPUT_DIR="${REPO_ROOT}/logs/gr00t_language30_replay_${VARIANT}/collection"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_compositionality_v1"
MOTIONS=(
    Neutral_stoop_down_001_A057
    big_heavy_one_hand_front_high_to_front_low_R_001_A524
    big_heavy_one_hand_front_low_to_front_high_R_001_A524
    big_light_two_hands_put_down_right_side_high_R_001_A520
    casual_greeting_R_001_A428
    cellphone_typing_sequence_one_hand_idle_R_001_A423
    cough_tuberculosis_R_001_A500
    crossed_arms_idle_R_001_A456
    drinking_standing_mug_R_001_A282
    exercise_3_A029
    feeding_birds_start_R_001_A456
    fishing_standing_loop_R_001_A500
    hurry_idle_001_A277
    inside_door_handle_right_side_open_walk_turn_close_R_001_A514
    injured_R_leg_turn_walk_360_001_A069
    injured_torso_walk_ff_start_315_R_001_A214
    jump_around_001_A492
    lift_crate_walk_ff_start_180_R_001_A140
    looking_in_the_mirror_amateur_001_A001
    medium_big_light_one_hand_walk_ff_start_360_R_001_A504
    mosquito_drive_away_R_001_A500
    painting_R_001_A097
    panic_run_away_180_R_001_A423
    rock_out_002_A484
    surrender_stop_R_001_A468
    talking_with_adult_turn_walk_360_R_walk_ff_loop_270_R_004_A476
    triumph_001_A033
    walk_arc_cw_start_R_slow_001_A443
    walk_big_dog_ff_225_stop_R_001_A492
    walk_ff_loop_180_R_slow_001_A443
)
RANKS=($(seq 0 $(( ${#MOTIONS[@]} - 1 ))))

# One parallel environment per motion: replay is deterministic, so one
# trajectory per motion carries all the information repeats would.
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${CHECKPOINT}" \
    --skill_checkpoint "${ENCODER}" \
    --language_embeddings "${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 \
    --output_dir "${OUTPUT_DIR}" \
    --label "gr00t_language30_replay_${VARIANT}" \
    --num_envs 30 --max_steps 1200 --seed 0 --metric_interval 1200 \
    --motion_names "${MOTIONS[@]}" \
    --trajectory_ranks "${RANKS[@]}" \
    --balanced_motion_names "${MOTIONS[@]}" \
    --balanced_trajectories_per_motion 1 \
    --save_rollout_training_samples --sample_rows_per_file 8192 \
    --sample_future_window_frames 30 --require_root_qpos_samples \
    --sonic_success_terminations --disable_push_event \
    --disable_reward_clipping --assert-kitless \
    physics=newton_mjwarp \
    env.replay_only=true \
    env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.reference_arrays_warm_workers=2 \
    env.data.macro_cache_device=cuda:0 \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    env.data.wrap_steps=false \
    env.command_interface.actor.dim="${ACTOR_DIM}" \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    agent.logger.backend= \
    agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim="${ACTOR_DIM}" \
    agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim="${CODE_DIM}" \
    agent.ipmd.latent_learning.code_period=10 \
    env.sim.physics.solver_cfg.njmax="${NJMAX}" \
    env.sim.physics.solver_cfg.nconmax=200 \
    ${EXTRA[@]+"${EXTRA[@]}"}
