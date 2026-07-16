#!/usr/bin/env bash
set -euo pipefail

# Local code gate for the Phase-5 language-conditioned comparison. This does
# not train a paper model: it collects two rows, takes one planner update, and
# runs twenty closed-loop control steps for each interface.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

PYTHON_CMD_STR="${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
ISAACLAB_PYTHON_CMD_STR="${INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD:-pixi run -e isaaclab python}"
# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_CMD_STR})
# shellcheck disable=SC2206
ISAACLAB_PYTHON_CMD=(${ISAACLAB_PYTHON_CMD_STR})

LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-}"
VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-}"
MANIFEST="${MANIFEST:-data/bones_seed/manifests/g1_bones_seed_10_manifest.json}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-data/bones_seed/language/g1_bones_seed_10_minilm_goal_embeddings.pt}"
GOAL_NAME="${GOAL_NAME:-Neutral_kick_trash_001_A057}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_language_smoke_$(date +%Y%m%d_%H%M%S)}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/tmp/iltools_g1_bones_seed_10_phase5_language_smoke}"
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-/tmp/iltools_g1_bones_seed_10_phase5_language_smoke_vanilla}"
CONTROL_STEPS="${CONTROL_STEPS:-20}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set LATENT_LOW_LEVEL_CHECKPOINT.}"
: "${LATENT_SKILL_CHECKPOINT:?Set LATENT_SKILL_CHECKPOINT.}"
: "${VANILLA_TRACKER_CHECKPOINT:?Set VANILLA_TRACKER_CHECKPOINT.}"

if [[ ! "${CONTROL_STEPS}" =~ ^[1-9][0-9]*$ ]] || ((CONTROL_STEPS % 10 != 0)); then
    echo "[ERROR] CONTROL_STEPS must be a positive multiple of 10." >&2
    exit 2
fi
EXPECTED_ROWS="$((CONTROL_STEPS / 10))"

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

preflight="${OUTPUT_ROOT}/protocol_checks/bones_seed_preflight.json"
single_manifest="${OUTPUT_ROOT}/protocol_checks/${GOAL_NAME}_manifest.json"
full_samples="${OUTPUT_ROOT}/full_body/demonstration/rollout_training_samples"
latent_samples="${OUTPUT_ROOT}/latent_skill/demonstration/rollout_training_samples"
full_planner="${OUTPUT_ROOT}/full_body/planner"
latent_planner="${OUTPUT_ROOT}/latent_skill/planner"
full_closed_loop="${OUTPUT_ROOT}/full_body/closed_loop"
latent_closed_loop="${OUTPUT_ROOT}/latent_skill/closed_loop"

run_cmd "${PYTHON_CMD[@]}" scripts/audit_bones_seed_phase5.py \
    --manifest "${MANIFEST}" \
    --report "${preflight}" \
    --require-body-names \
    --require-temporal-events

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/write_single_motion_manifest.py \
    --manifest "${MANIFEST}" \
    --motion_name "${GOAL_NAME}" \
    --output "${single_manifest}"

run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/collect_interface_rollout_samples.py \
    --headless \
    --task Isaac-Imitation-G1-v0 \
    --algo IPMD \
    --checkpoint "${VANILLA_TRACKER_CHECKPOINT}" \
    --interface full_body_trajectory \
    --output_dir "${OUTPUT_ROOT}/full_body/demonstration" \
    --motion_manifest "${MANIFEST}" \
    --dataset_path "${VANILLA_DATASET_PATH}" \
    --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
    --num_envs 1 \
    --control_steps "${CONTROL_STEPS}" \
    --seed "${SEED}" \
    --state_history_steps 9 \
    --planner_interval_steps 10 \
    --command_past_steps 0 \
    --command_future_steps 9 \
    --reset_schedule sequential \
    --refresh_zarr_dataset \
    --low_level_command_mode streamed_vanilla \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.backend= \
    env.observations.policy.enable_corruption=false

run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algorithm IPMD \
    --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
    --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
    --state_history_steps 9 \
    --output_dir "${OUTPUT_ROOT}/latent_skill/demonstration" \
    --label bones_seed_language_latent_demonstration_smoke \
    --num_envs 1 \
    --max_steps "${CONTROL_STEPS}" \
    --seed "${SEED}" \
    --metric_interval 1 \
    --keep_time_out \
    --extend_episode_length_for_max_steps \
    --keep_early_terminations \
    --disable_reward_clipping \
    --flow_num_inference_steps 2 \
    --flow_inference_noise_std 0.0 \
    --save_rollout_training_samples \
    --continue_after_reset \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.ipmd.command_source=hl_skill \
    "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
    agent.logger.backend= \
    agent.ipmd.hl_skill_finetune_enabled=false \
    "env.lafan1_manifest_path=${MANIFEST}" \
    "env.dataset_path=${LATENT_DATASET_PATH}" \
    env.reset_schedule=sequential \
    env.wrap_steps=false \
    env.observations.policy.enable_corruption=false \
    env.refresh_zarr_dataset=true \
    env.latent_command_dim=258 \
    agent.ipmd.latent_dim=258 \
    agent.ipmd.latent_steps_min=10 \
    agent.ipmd.latent_steps_max=10 \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim=256 \
    agent.ipmd.latent_learning.code_period=10 \
    agent.ipmd.reward_loss_coeff=0.0 \
    agent.ipmd.reward_l2_coeff=0.0 \
    agent.ipmd.reward_grad_penalty_coeff=0.0 \
    agent.ipmd.reward_logit_reg_coeff=0.0 \
    agent.ipmd.reward_param_weight_decay_coeff=0.0

for row in "full_body:${full_samples}:${full_planner}" "latent_skill:${latent_samples}:${latent_planner}"; do
    IFS=: read -r interface samples planner <<<"${row}"
    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_chunked_transformer_planner.py \
        --samples_dir "${samples}" \
        --output_dir "${planner}" \
        --interface "${interface/full_body/full_body_trajectory}" \
        --state_key expert_planner_state \
        --model_size tiny \
        --seed "${SEED}" \
        --batch_size "${EXPECTED_ROWS}" \
        --micro_batch_size "${EXPECTED_ROWS}" \
        --num_updates 1 \
        --log_interval 1 \
        --eval_max_samples "${EXPECTED_ROWS}" \
        --flow_num_inference_steps 2 \
        --endpoint_num_inference_steps 1 \
        --device cpu
    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_offline.py \
        --samples_dir "${samples}" \
        --planner_checkpoint "${planner}/checkpoints/latest.pt" \
        --output_json "${planner}/offline_eval.json" \
        --state_key expert_planner_state \
        --batch_size "${EXPECTED_ROWS}" \
        --flow_num_inference_steps 2 \
        --device cpu
done

run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_closed_loop.py \
    --headless \
    --task Isaac-Imitation-G1-v0 \
    --algo IPMD \
    --checkpoint "${VANILLA_TRACKER_CHECKPOINT}" \
    --low_level_command_mode streamed_vanilla \
    --planner_checkpoint "${full_planner}/checkpoints/latest.pt" \
    --output_json "${full_closed_loop}/summary.json" \
    --label bones_seed_language_full_body_smoke \
    --motion_manifest "${single_manifest}" \
    --dataset_path "${VANILLA_DATASET_PATH}" \
    --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
    --language_goal_name "${GOAL_NAME}" \
    --motion_name "${GOAL_NAME}" \
    --num_envs 1 \
    --steps "${CONTROL_STEPS}" \
    --seed "${SEED}" \
    --state_history_steps 9 \
    --command_past_steps 0 \
    --command_future_steps 9 \
    --planner_update_interval 10 \
    --flow_num_inference_steps 2 \
    --flow_inference_noise_std 0.0 \
    --reset_schedule sequential \
    --refresh_zarr_dataset \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.logger.backend= \
    env.observations.policy.enable_corruption=false

run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless \
    --task Isaac-Imitation-G1-Latent-v0 \
    --algorithm IPMD \
    --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    --planner_checkpoint "${latent_planner}/checkpoints/latest.pt" \
    --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
    --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
    --state_history_steps 9 \
    --output_dir "${latent_closed_loop}" \
    --label bones_seed_language_latent_smoke \
    --num_envs 1 \
    --max_steps "${CONTROL_STEPS}" \
    --seed "${SEED}" \
    --motion_name "${GOAL_NAME}" \
    --metric_interval 1 \
    --keep_time_out \
    --extend_episode_length_for_max_steps \
    --keep_early_terminations \
    --disable_reward_clipping \
    --flow_num_inference_steps 2 \
    --flow_inference_noise_std 0.0 \
    --kit_args=--/app/extensions/fsWatcherEnabled=false \
    agent.ipmd.command_source=skill_commander \
    "agent.ipmd.skill_commander_checkpoint_path=${latent_planner}/checkpoints/latest.pt" \
    "agent.ipmd.skill_commander_embeddings_path=${LANGUAGE_EMBEDDINGS}" \
    "agent.ipmd.skill_commander_goal_name=${GOAL_NAME}" \
    agent.ipmd.skill_commander_use_achieved_state=true \
    agent.ipmd.skill_commander_flow_num_inference_steps=2 \
    agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
    agent.logger.backend= \
    agent.ipmd.hl_skill_finetune_enabled=false \
    "env.lafan1_manifest_path=${MANIFEST}" \
    "env.dataset_path=${LATENT_DATASET_PATH}" \
    env.reset_schedule=sequential \
    env.wrap_steps=false \
    env.observations.policy.enable_corruption=false \
    env.refresh_zarr_dataset=false \
    env.latent_command_dim=258 \
    agent.ipmd.latent_dim=258 \
    agent.ipmd.latent_steps_min=10 \
    agent.ipmd.latent_steps_max=10 \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim=256 \
    agent.ipmd.latent_learning.code_period=10 \
    agent.ipmd.reward_loss_coeff=0.0 \
    agent.ipmd.reward_l2_coeff=0.0 \
    agent.ipmd.reward_grad_penalty_coeff=0.0 \
    agent.ipmd.reward_logit_reg_coeff=0.0 \
    agent.ipmd.reward_param_weight_decay_coeff=0.0

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/audit_bones_seed_language_interface.py \
    --preflight "${preflight}" \
    --latent_samples "${latent_samples}" \
    --full_body_samples "${full_samples}" \
    --latent_checkpoint "${latent_planner}/checkpoints/latest.pt" \
    --full_body_checkpoint "${full_planner}/checkpoints/latest.pt" \
    --latent_summary "${latent_closed_loop}/summary.json" \
    --full_body_summary "${full_closed_loop}/summary.json" \
    --single_motion_manifest "${single_manifest}" \
    --expected_goal_name "${GOAL_NAME}" \
    --expected_rows "${EXPECTED_ROWS}" \
    --output_json "${OUTPUT_ROOT}/language_interface_audit.json"

echo "[INFO] Phase-5 language smoke passed: ${OUTPUT_ROOT}"
