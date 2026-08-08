#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STAGES="${STAGES:-materialize,train,eval,diagnostic,aggregate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/bones_language10_loco_manip_v3_h30_future_seed0}"
SOURCE_ROOT="${SOURCE_ROOT:-logs/bones_language10_loco_manip_v3_oracle_seed0}"
SOURCE_SAMPLES="${SOURCE_SAMPLES:-${SOURCE_ROOT}/collection/rollout_training_samples}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt}"
SELECTION="${SELECTION:-experiments/campaigns/2026-08-05-bones-language10-screen/selected10_loco_manip_v3.json}"
SELECTION_SHA256="${SELECTION_SHA256:-c9f7e7d2ac76d1ded6c33ce1128f8681806719bf422827202ca0ef458e687abb}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-data/bones_seed_language10_loco_manip_v3/language/g1_bones_seed_language10_loco_manip_v3_minilm_goal_embeddings.pt}"
PHASE_ANNOTATIONS="${PHASE_ANNOTATIONS:-${SCRIPT_DIR}/semantic_phase_annotations.json}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-data/bones_seed_language10_loco_manip_v3/reference_arrays/root_qpos_v1}"
REFERENCE_ARRAYS_PERSIST_ID="${REFERENCE_ARRAYS_PERSIST_ID:-bones_seed_language10_loco_manip_v3@c9f7e7d2}"
EVAL_TRAJECTORIES_PER_GOAL="${EVAL_TRAJECTORIES_PER_GOAL:-100}"
UPDATES=(2000 4000 6000 8000 10000)

SAMPLES="${OUTPUT_ROOT}/samples/h3_future_publication"
PLANNER="${OUTPUT_ROOT}/planner/h3_future_publication"

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
    "${SELECTION}" "${LANGUAGE_EMBEDDINGS}" \
    "${REFERENCE_ARRAYS}/reference_arrays_manifest.json"; do
    require_file "${artifact}"
done
actual_selection_sha="$(sha256sum "${SELECTION}" | cut -d' ' -f1)"
if [[ "${actual_selection_sha}" != "${SELECTION_SHA256}" ]]; then
    echo "[ERROR] Selection SHA mismatch: ${actual_selection_sha}" >&2
    exit 2
fi

if has_stage materialize; then
    run_if_missing "${SAMPLES}/materialization_manifest.json" \
        pixi run python -m imitation_experiments.planner.materialize_latent_receding_horizon \
        --samples_dir "${SOURCE_SAMPLES}" --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --output_dir "${SAMPLES}" --target_frame future_publication \
        --batch_size 4096 --device auto
fi

if has_stage train; then
    require_file "${SAMPLES}/materialization_manifest.json"
    train_command=(
        pixi run python -m imitation_experiments.planner.train_chunked_transformer_planner
        --samples_dir "${SAMPLES}" --output_dir "${PLANNER}"
        --interface latent_skill --state_key planner_state --training_stage oracle
        --planner_family flow --model_size medium --seed 0
        --num_updates 10000 --milestone_interval 2000 --log_interval 100
        --batch_size 256 --micro_batch_size 32 --lr 0.0001 --weight_decay 0.0001
        --flow_num_inference_steps 16 --endpoint_num_inference_steps 4
        --val_trajectory_fraction 0.2 --val_split_seed 0
    )
    if [[ -f "${PLANNER}/checkpoints/latest.pt" ]]; then
        train_command+=(--resume_checkpoint "${PLANNER}/checkpoints/latest.pt")
    fi
    run_if_missing "${PLANNER}/checkpoints/update_0010000.pt" "${train_command[@]}"
fi

MOTIONS=(
    lift_crate_walk_ff_start_180_R_001_A140
    walk_arc_cw_start_R_slow_001_A443
    Neutral_stoop_down_001_A057
    walk_ff_loop_180_R_slow_001_A443
    medium_big_light_one_hand_walk_ff_start_360_R_001_A504
    inside_door_handle_right_side_open_walk_turn_close_R_001_A514
    injured_torso_walk_ff_start_315_R_001_A214
    big_heavy_one_hand_front_high_to_front_low_R_001_A524
    street_waiting_at_crossroad_button_press_sequence_002_A423
    street_passing_out_flyers_move_R_002_A428
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

eval_one() {
    local name="$1" checkpoint="$2" mode="$3" rank="$4" motion="$5"
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
}

if has_stage eval; then
    for update in "${UPDATES[@]}"; do
        checkpoint="${PLANNER}/checkpoints/update_$(printf '%07d' "${update}").pt"
        require_file "${checkpoint}"
        for mode in exponential clipped_gated; do
            variant="h30_future_${mode}"
            name="update_$(printf '%07d' "${update}")/${variant}"
            for rank in "${!MOTIONS[@]}"; do
                eval_one "${name}" "${checkpoint}" "${mode}" "${rank}" "${MOTIONS[$rank]}"
            done
        done
    done
fi

diagnostic_mode() {
    local mode="$1"
    local checkpoint="${PLANNER}/checkpoints/update_0010000.pt"
    local name="h30_future_${mode}"
    local output="${OUTPUT_ROOT}/full_horizon_diagnostic/${name}"
    pixi run python .agents/skills/policy-eval-video/scripts/render_policy_videos.py \
        --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --planner_checkpoint "${checkpoint}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
        --output_root "${output}" --reference_arrays "${REFERENCE_ARRAYS}" \
        --persist_id "${REFERENCE_ARRAYS_PERSIST_ID}" --randomized_no_push \
        --latent_temporal_ensemble "${mode}" \
        --latent_temporal_ensemble_decay 0.5 \
        --latent_temporal_clip_std 1.0 \
        --latent_temporal_gate_distance 2.0 \
        --latent_temporal_gate_cosine 0.5 --skip_existing \
        -- "${COMMON_OVERRIDES[@]}"
}

if has_stage diagnostic; then
    require_file "${PLANNER}/checkpoints/update_0010000.pt"
    for mode in exponential clipped_gated; do
        diagnostic_mode "${mode}"
    done
fi

if has_stage aggregate; then
    run_if_missing "${OUTPUT_ROOT}/aggregate/results.json" \
        pixi run python -m imitation_experiments.evaluation.aggregate_language_h30_fusion \
        --output_root "${OUTPUT_ROOT}" --expected_goals 10 \
        --expected_trajectories_per_goal "${EVAL_TRAJECTORIES_PER_GOAL}" \
        --updates "${UPDATES[@]}"
fi

if has_stage semantic; then
    require_file "${PHASE_ANNOTATIONS}"
    run_if_missing "${OUTPUT_ROOT}/latent_space_analysis/analysis.json" \
        pixi run python \
        -m imitation_experiments.evaluation.analyze_collected_latent_space \
        --samples_dir "${SAMPLES}" \
        --output_dir "${OUTPUT_ROOT}/latent_space_analysis" \
        --selection "${SELECTION}" \
        --phase_annotations "${PHASE_ANNOTATIONS}" \
        --max_points_per_motion 600 --seed 0 \
        --tsne_perplexity 40 --tsne_iterations 1500
    run_if_missing \
        "${OUTPUT_ROOT}/latent_space_analysis/semantic_phase_clips/phase_clip_manifest.json" \
        pixi run python \
        -m imitation_experiments.evaluation.segment_semantic_phase_videos \
        --annotations "${PHASE_ANNOTATIONS}" \
        --output_dir "${OUTPUT_ROOT}/latent_space_analysis/semantic_phase_clips"
fi

echo "[PASS] BONES language10 loco-manip H30 stages: ${STAGES}"
