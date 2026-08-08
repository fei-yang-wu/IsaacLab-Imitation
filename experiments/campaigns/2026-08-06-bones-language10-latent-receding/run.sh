#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STAGES="${STAGES:-materialize,train,eval,aggregate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/bones_language10_latent_receding_seed0}"
SOURCE_ROOT="${SOURCE_ROOT:-logs/bones_language10_oracle_pretrain_seed0}"
SOURCE_SAMPLES="${SOURCE_SAMPLES:-${SOURCE_ROOT}/collection/rollout_training_samples}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-data/bones_seed_language10_v1/language/g1_bones_seed_language10_v1_minilm_goal_embeddings.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-data/bones_seed_language10_v1/reference_arrays/root_qpos_v1}"
REFERENCE_ARRAYS_PERSIST_ID="${REFERENCE_ARRAYS_PERSIST_ID:-bones_seed_language10_v1@60a5b7a5}"
H1_CHECKPOINT="${H1_CHECKPOINT:-${SOURCE_ROOT}/planner_oracle_pretrain/checkpoints/update_0010000.pt}"
UPDATES="${UPDATES:-10000}"
EVAL_TRAJECTORIES_PER_GOAL="${EVAL_TRAJECTORIES_PER_GOAL:-100}"

FUTURE_SAMPLES="${OUTPUT_ROOT}/samples/h3_future_publication"
CURRENT_SAMPLES="${OUTPUT_ROOT}/samples/h3_current_publication"
FUTURE_PLANNER="${OUTPUT_ROOT}/planner/h3_future_publication"
CURRENT_PLANNER="${OUTPUT_ROOT}/planner/h3_current_publication"

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }
require_file() { [[ -f "$1" ]] || { echo "[ERROR] Missing $1" >&2; exit 2; }; }
run_if_missing() {
    local expected="$1"
    shift
    if [[ -f "${expected}" ]]; then
        echo "[SKIP] ${expected}"
    else
        echo "[CMD] $*"
        "$@"
        require_file "${expected}"
    fi
}

for artifact in "${LOW_LEVEL_CHECKPOINT}" "${SKILL_CHECKPOINT}" \
    "${LANGUAGE_EMBEDDINGS}" "${H1_CHECKPOINT}" \
    "${REFERENCE_ARRAYS}/reference_arrays_manifest.json"; do
    require_file "${artifact}"
done

if has_stage materialize; then
    run_if_missing "${FUTURE_SAMPLES}/materialization_manifest.json" \
        pixi run python -m imitation_experiments.planner.materialize_latent_receding_horizon \
        --samples_dir "${SOURCE_SAMPLES}" --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --output_dir "${FUTURE_SAMPLES}" --target_frame future_publication \
        --batch_size 4096 --device auto
    run_if_missing "${CURRENT_SAMPLES}/materialization_manifest.json" \
        pixi run python -m imitation_experiments.planner.materialize_latent_receding_horizon \
        --samples_dir "${SOURCE_SAMPLES}" --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --output_dir "${CURRENT_SAMPLES}" --target_frame current_publication \
        --batch_size 4096 --device auto
fi

train_h3() {
    local samples="$1" output="$2"
    run_if_missing "${output}/checkpoints/update_0010000.pt" \
        pixi run python -m imitation_experiments.planner.train_chunked_transformer_planner \
        --samples_dir "${samples}" --output_dir "${output}" \
        --interface latent_skill --state_key planner_state --training_stage oracle \
        --planner_family flow --model_size medium --seed 0 \
        --num_updates "${UPDATES}" --milestone_interval 2000 --log_interval 100 \
        --batch_size 256 --micro_batch_size 32 --lr 0.0001 --weight_decay 0.0001 \
        --flow_num_inference_steps 16 --endpoint_num_inference_steps 4 \
        --val_trajectory_fraction 0.2 --val_split_seed 0
}

if has_stage train; then
    require_file "${FUTURE_SAMPLES}/materialization_manifest.json"
    require_file "${CURRENT_SAMPLES}/materialization_manifest.json"
    train_h3 "${FUTURE_SAMPLES}" "${FUTURE_PLANNER}"
    train_h3 "${CURRENT_SAMPLES}" "${CURRENT_PLANNER}"
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

COMMON_OVERRIDES=(
    physics=newton_mjwarp
    env.data.manifest=null
    env.data.cache_dir=null
    "env.data.reference_arrays_dir=${REFERENCE_ARRAYS}"
    "env.data.persist_id=${REFERENCE_ARRAYS_PERSIST_ID}"
    env.data.persist_dir=null
    env.data.reference_arrays_warm_workers=2
    env.data.macro_cache_device=cuda:0
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]'
    env.data.wrap_steps=false
    env.command_interface.actor.dim=258
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]'
    'agent.logger.backend='
    agent.ipmd.command_source=skill_commander
    "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}"
    agent.ipmd.hl_skill_finetune_enabled=false
    agent.ipmd.latent_dim=258
    agent.ipmd.latent_steps_min=10
    agent.ipmd.latent_steps_max=10
    agent.ipmd.hl_skill_horizon_steps=10
    agent.ipmd.hl_skill_command_mode=z
    agent.ipmd.latent_learning.command_phase_mode=sin_cos
    agent.ipmd.latent_learning.code_latent_dim=256
    agent.ipmd.latent_learning.code_period=10
    env.sim.physics.solver_cfg.njmax=289
    env.sim.physics.solver_cfg.nconmax=200
)

eval_variant() {
    local name="$1" checkpoint="$2" mode="$3"
    require_file "${checkpoint}"
    for rank in "${!MOTIONS[@]}"; do
        local motion="${MOTIONS[$rank]}"
        local output="${OUTPUT_ROOT}/evaluation/${name}/${motion}"
        run_if_missing "${output}/summary.json" \
            pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
            --headless --assert-kitless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
            --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
            --checkpoint "${LOW_LEVEL_CHECKPOINT}" --planner_checkpoint "${checkpoint}" \
            --skill_checkpoint "${SKILL_CHECKPOINT}" \
            --language_embeddings "${LANGUAGE_EMBEDDINGS}" --state_history_steps 9 \
            --output_dir "${output}" --label "${name}_${motion}" \
            --num_envs "${EVAL_TRAJECTORIES_PER_GOAL}" --max_steps 0 --seed 0 \
            --metric_interval 100 --motion_name "${motion}" --trajectory_ranks "${rank}" \
            --require_goal_motion_match --sonic_success_terminations \
            --disable_push_event --disable_reward_clipping \
            --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
            --latent_temporal_ensemble "${mode}" \
            --latent_temporal_ensemble_decay 0.5 \
            --latent_temporal_clip_std 1.0 \
            --latent_temporal_gate_distance 2.0 \
            --latent_temporal_gate_cosine 0.5 \
            "${COMMON_OVERRIDES[@]}" \
            "agent.ipmd.skill_commander_checkpoint_path=${checkpoint}" \
            "agent.ipmd.skill_commander_embeddings_path=${LANGUAGE_EMBEDDINGS}" \
            "agent.ipmd.skill_commander_goal_name=${motion}" \
            agent.ipmd.skill_commander_use_achieved_state=true \
            agent.ipmd.skill_commander_flow_num_inference_steps=16 \
            agent.ipmd.skill_commander_flow_inference_noise_std=0.0
    done
}

if has_stage eval; then
    eval_variant h1_baseline "${H1_CHECKPOINT}" first
    for mode in first exponential clipped_gated; do
        eval_variant "h3_future_${mode}" \
            "${FUTURE_PLANNER}/checkpoints/update_0010000.pt" "${mode}"
        eval_variant "h3_current_${mode}" \
            "${CURRENT_PLANNER}/checkpoints/update_0010000.pt" "${mode}"
    done
fi

if has_stage aggregate; then
    run_if_missing "${OUTPUT_ROOT}/aggregate/results.json" \
        pixi run python -m imitation_experiments.evaluation.aggregate_language_latent_receding \
        --output_root "${OUTPUT_ROOT}" --expected_goals 10 \
        --expected_trajectories_per_goal "${EVAL_TRAJECTORIES_PER_GOAL}"
fi

echo "[PASS] BONES language10 latent receding-horizon stages: ${STAGES}"
