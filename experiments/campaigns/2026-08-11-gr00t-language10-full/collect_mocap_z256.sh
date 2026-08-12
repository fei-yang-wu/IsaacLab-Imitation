#!/usr/bin/env bash
# Mocap (kinematic reference replay) collection for the z256/chunk arms.
# Identical protocol to the frozen 08-05 oracle collection except:
# env.replay_only=true, 10 envs, 1 trajectory per motion (expert rows are
# deterministic), and a fresh output directory.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/logs/gr00t_language10_mocap_z256/collection"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

MOTIONS=(
    Neutral_stoop_down_001_A057
    lift_crate_walk_ff_start_180_R_001_A140
    drinking_standing_mug_R_001_A282
    fishing_standing_loop_R_001_A500
    cellphone_typing_sequence_one_hand_idle_R_001_A423
    feeding_birds_start_R_001_A456
    walk_arc_cw_start_R_slow_001_A443
    mosquito_drive_away_R_001_A500
    casual_greeting_R_001_A428
    surrender_stop_R_001_A468
)

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt" \
    --skill_checkpoint "${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt" \
    --language_embeddings "${REPO_ROOT}/data/bones_seed_language10_v1/language/g1_bones_seed_language10_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 \
    --output_dir "${OUTPUT_DIR}" \
    --label gr00t_language10_mocap_replay_collection \
    --num_envs 10 --max_steps 1200 --seed 0 --metric_interval 1200 \
    --motion_names "${MOTIONS[@]}" \
    --trajectory_ranks 0 1 2 3 4 5 6 7 8 9 \
    --balanced_motion_names "${MOTIONS[@]}" \
    --balanced_trajectories_per_motion 1 \
    --save_rollout_training_samples --sample_rows_per_file 8192 \
    --sample_future_window_frames 30 --require_root_qpos_samples \
    --sonic_success_terminations --disable_push_event \
    --disable_reward_clipping --assert-kitless \
    physics=newton_mjwarp \
    env.replay_only=true \
    env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${REPO_ROOT}/data/bones_seed_language10_v1/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language10_v1@60a5b7a5 \
    env.data.persist_dir=null env.data.reference_arrays_warm_workers=2 \
    env.data.macro_cache_device=cuda:0 \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    env.data.wrap_steps=false \
    env.command_interface.actor.dim=258 \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    agent.logger.backend= \
    agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim=258 agent.ipmd.latent_steps_min=10 \
    agent.ipmd.latent_steps_max=10 agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim=256 \
    agent.ipmd.latent_learning.code_period=10 \
    env.sim.physics.solver_cfg.njmax=289 env.sim.physics.solver_cfg.nconmax=200
