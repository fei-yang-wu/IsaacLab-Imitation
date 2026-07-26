#!/usr/bin/env bash
set -euo pipefail

# Resumable local launcher for one matched planner-capacity point. This is a
# one-motion diagnostic, not a paper-facing experiment. It reuses the frozen
# seed-0 demonstration rows and varies only planner initialization/minibatches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODEL_SIZE="${MODEL_SIZE:-tiny}"
PLANNER_FAMILY="${PLANNER_FAMILY:-flow}"
PLANNER_SEED="${PLANNER_SEED:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
if [[ "${PLANNER_FAMILY}" == "flow" ]]; then
    default_output_root="logs/interface_baselines/lafan1_one_motion_capacity_scaling_multiseed_20260716/seed${PLANNER_SEED}"
else
    default_output_root="logs/interface_baselines/lafan1_one_motion_planner_families_20260716/${PLANNER_FAMILY}/seed${PLANNER_SEED}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output_root}}"
POINT_ROOT="${OUTPUT_ROOT}/${MODEL_SIZE}"
DRY_RUN="${DRY_RUN:-0}"

LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-07-15_06-42-18/models/model_step_1250033664.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/interface_baselines/lafan1_corrected_8e95d557_diffsr_5b_seed0/latent/base_pipeline/skill_encoder_h10_z256/checkpoints/latest.pt}"
VANILLA_LOW_LEVEL_CHECKPOINT="${VANILLA_LOW_LEVEL_CHECKPOINT:-logs/downloaded_checkpoints/lafan1_corrected_vanilla_job3500993/model_step_1050083328.pt}"
MANIFEST="${MANIFEST:-data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json}"
MOTION_NAME="${MOTION_NAME:-walk1_subject1}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-data/lafan1/zarr/latent_walk1_subject1_corrected_8e95d557}"
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-logs/interface_baselines/local_vanilla_m3_job3500993_step1050083328_walk1/dataset_cache}"
LATENT_DEMONSTRATIONS="${LATENT_DEMONSTRATIONS:-logs/interface_baselines/lafan1_one_motion_latent_gate_20260715/oracle_demonstration_collection/rollout_training_samples}"
VANILLA_DEMONSTRATIONS="${VANILLA_DEMONSTRATIONS:-logs/interface_baselines/lafan1_one_motion_explicit_gate_20260716/oracle_demonstration_collection/rollout_training_samples}"
LATENT_ORACLE_SUMMARY="${LATENT_ORACLE_SUMMARY:-logs/interface_baselines/lafan1_one_motion_latent_gate_20260715/oracle_walk1_subject1_10starts/summary.json}"
VANILLA_ORACLE_SUMMARY="${VANILLA_ORACLE_SUMMARY:-logs/interface_baselines/lafan1_one_motion_explicit_gate_20260716/oracle_walk1_subject1_10starts/summary.json}"

case "${MODEL_SIZE}" in
    tiny|small|medium|large) ;;
    *)
        echo "[ERROR] MODEL_SIZE must be tiny, small, medium, or large." >&2
        exit 2
        ;;
esac
case "${PLANNER_FAMILY}" in
    flow|diffusion|deterministic) ;;
    *)
        echo "[ERROR] PLANNER_FAMILY must be flow, diffusion, or deterministic." >&2
        exit 2
        ;;
esac
for value in "${PLANNER_SEED}" "${EVAL_SEED}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Seeds must be non-negative integers, got ${value}." >&2
        exit 2
    fi
done
for path in \
    "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    "${LATENT_SKILL_CHECKPOINT}" \
    "${VANILLA_LOW_LEVEL_CHECKPOINT}" \
    "${MANIFEST}" \
    "${LATENT_ORACLE_SUMMARY}" \
    "${VANILLA_ORACLE_SUMMARY}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Required file is missing: ${path}" >&2
        exit 2
    fi
done
for path in \
    "${LATENT_DATASET_PATH}" \
    "${VANILLA_DATASET_PATH}" \
    "${LATENT_DEMONSTRATIONS}" \
    "${VANILLA_DEMONSTRATIONS}"; do
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] Required directory is missing: ${path}" >&2
        exit 2
    fi
done

run_if_missing() {
    local marker="$1"
    shift
    if [[ -e "${marker}" ]]; then
        echo "[SKIP] ${marker}"
        return 0
    fi
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    TERM=xterm PYTHONUNBUFFERED=1 "$@"
    if [[ ! -e "${marker}" ]]; then
        echo "[ERROR] Command completed without expected artifact: ${marker}" >&2
        exit 2
    fi
}

latent_root="${POINT_ROOT}/latent_skill"
explicit_root="${POINT_ROOT}/full_body_trajectory"
latent_pretrain="${latent_root}/planner_pretrain"
explicit_pretrain="${explicit_root}/planner_pretrain"
latent_finetune="${latent_root}/planner_finetune"
explicit_finetune="${explicit_root}/planner_finetune"

mkdir -p "${latent_root}" "${explicit_root}"

train_planner() {
    local interface="$1"
    local samples="$2"
    local output="$3"
    local checkpoint="${4:-}"
    local command=(
        pixi run python experiments/interface_baselines/train_chunked_transformer_planner.py
        --samples_dir "${samples}"
        --output_dir "${output}"
        --interface "${interface}"
        --planner_family "${PLANNER_FAMILY}"
        --state_key planner_state
        --device "${DEVICE}"
        --seed "${PLANNER_SEED}"
        --batch_size 256
        --micro_batch_size 32
        --num_updates 2000
        --log_interval 100
        --eval_batch_size 512
        --eval_max_samples 4096
        --lr 0.0001
        --weight_decay 0.0001
        --model_size "${MODEL_SIZE}"
        --flow_num_inference_steps 16
        --endpoint_num_inference_steps 4
        --flow_inference_noise_std 0.0
    )
    if [[ -n "${checkpoint}" ]]; then
        command+=(--checkpoint "${checkpoint}")
    else
        command+=(--max_samples 1000)
    fi
    run_if_missing "${output}/checkpoints/latest.pt" "${command[@]}"
}

eval_latent() {
    local planner="$1"
    local output="$2"
    local label="$3"
    run_if_missing "${output}/summary.json" \
        pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --device "${DEVICE}" \
        --task Isaac-Imitation-G1-Latent-v0 --algorithm IPMD \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        --planner_checkpoint "${planner}" \
        --state_history_steps 9 --output_dir "${output}" --label "${label}" \
        --num_envs 10 --max_steps 500 --seed "${EVAL_SEED}" --metric_interval 10 \
        --keep_time_out --allow_random_reset --keep_early_terminations \
        --disable_tracking_terminations --disable_reward_clipping \
        --motion_name "${MOTION_NAME}" --flow_num_inference_steps 16 \
        --flow_inference_noise_std 0.0 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.logger.backend= \
        agent.ipmd.command_source=skill_commander \
        "agent.ipmd.skill_commander_checkpoint_path=${planner}" \
        agent.ipmd.skill_commander_use_achieved_state=true \
        agent.ipmd.skill_commander_flow_num_inference_steps=16 \
        agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        env.refresh_zarr_dataset=false env.random_reset_step_min=0 \
        env.random_reset_step_max=200 env.random_reset_full_trajectory=false \
        env.reset_schedule=sequential env.wrap_steps=false \
        env.observations.policy.enable_corruption=false env.latent_command_dim=258 \
        agent.ipmd.latent_dim=258 agent.ipmd.hl_skill_horizon_steps=10 \
        agent.ipmd.hl_skill_command_mode=z agent.ipmd.latent_steps_min=10 \
        agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=256 \
        agent.ipmd.latent_learning.code_period=10 agent.ipmd.reward_loss_coeff=0.0 \
        agent.ipmd.reward_l2_coeff=0.0 agent.ipmd.reward_grad_penalty_coeff=0.0 \
        agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0
}

eval_explicit() {
    local planner="$1"
    local output="$2"
    local label="$3"
    run_if_missing "${output}/summary.json" \
        pixi run -e isaaclab python experiments/interface_baselines/eval_interface_planner_closed_loop.py \
        --headless --device "${DEVICE}" --task Isaac-Imitation-G1-v0 \
        --algorithm IPMD --checkpoint "${VANILLA_LOW_LEVEL_CHECKPOINT}" \
        --low_level_command_mode streamed_vanilla --planner_checkpoint "${planner}" \
        --output_json "${output}/summary.json" --label "${label}" \
        --motion_manifest "${MANIFEST}" --dataset_path "${VANILLA_DATASET_PATH}" \
        --motion_name "${MOTION_NAME}" --num_envs 10 --steps 500 \
        --seed "${EVAL_SEED}" --state_history_steps 9 --command_past_steps 0 \
        --command_future_steps 9 --planner_update_interval 10 \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
        --reset_schedule sequential --keep_configured_episode_length \
        --disable_tracking_terminations \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.logger.backend= \
        env.random_reset_step_min=0 env.random_reset_step_max=200 \
        env.random_reset_full_trajectory=false env.reset_schedule=sequential \
        env.wrap_steps=false env.observations.policy.enable_corruption=false
}

collect_latent() {
    local output="${latent_root}/planner_rollout_collection"
    run_if_missing "${output}/rollout_training_samples/sample_step_000000.pt" \
        pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless --device "${DEVICE}" --task Isaac-Imitation-G1-Latent-v0 \
        --algorithm IPMD --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        --planner_checkpoint "${latent_pretrain}/checkpoints/latest.pt" \
        --state_history_steps 9 --output_dir "${output}" \
        --label "capacity_${PLANNER_FAMILY}_${MODEL_SIZE}_seed${PLANNER_SEED}_latent_rollout_collection" \
        --num_envs 10 --max_steps 1000 --seed "${EVAL_SEED}" --metric_interval 10 \
        --keep_time_out --allow_random_reset --keep_early_terminations \
        --disable_tracking_terminations --disable_reward_clipping \
        --motion_name "${MOTION_NAME}" --balanced_motion_names "${MOTION_NAME}" \
        --balanced_rows_per_motion 1000 --save_rollout_training_samples \
        --continue_after_reset --sample_rows_per_file 1000 \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.logger.backend= \
        agent.ipmd.command_source=skill_commander \
        "agent.ipmd.skill_commander_checkpoint_path=${latent_pretrain}/checkpoints/latest.pt" \
        agent.ipmd.skill_commander_use_achieved_state=true \
        agent.ipmd.skill_commander_flow_num_inference_steps=16 \
        agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        "env.lafan1_manifest_path=${MANIFEST}" "env.dataset_path=${LATENT_DATASET_PATH}" \
        env.refresh_zarr_dataset=false env.random_reset_step_min=0 \
        env.random_reset_step_max=200 env.random_reset_full_trajectory=false \
        env.reset_schedule=sequential env.wrap_steps=false \
        env.observations.policy.enable_corruption=false env.latent_command_dim=258 \
        agent.ipmd.latent_dim=258 agent.ipmd.hl_skill_horizon_steps=10 \
        agent.ipmd.hl_skill_command_mode=z agent.ipmd.latent_steps_min=10 \
        agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=256 \
        agent.ipmd.latent_learning.code_period=10 agent.ipmd.reward_loss_coeff=0.0 \
        agent.ipmd.reward_l2_coeff=0.0 agent.ipmd.reward_grad_penalty_coeff=0.0 \
        agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0
}

collect_explicit() {
    local output="${explicit_root}/planner_rollout_collection"
    run_if_missing "${output}/rollout_training_samples/sample_step_000000.pt" \
        pixi run -e isaaclab python experiments/interface_baselines/eval_interface_planner_closed_loop.py \
        --headless --device "${DEVICE}" --task Isaac-Imitation-G1-v0 \
        --algorithm IPMD --checkpoint "${VANILLA_LOW_LEVEL_CHECKPOINT}" \
        --low_level_command_mode streamed_vanilla \
        --planner_checkpoint "${explicit_pretrain}/checkpoints/latest.pt" \
        --output_json "${output}/summary.json" \
        --label "capacity_${PLANNER_FAMILY}_${MODEL_SIZE}_seed${PLANNER_SEED}_explicit_rollout_collection" \
        --motion_manifest "${MANIFEST}" --dataset_path "${VANILLA_DATASET_PATH}" \
        --motion_name "${MOTION_NAME}" --num_envs 10 --steps 1000 \
        --seed "${EVAL_SEED}" --state_history_steps 9 --command_past_steps 0 \
        --command_future_steps 9 --planner_update_interval 10 \
        --flow_num_inference_steps 16 --flow_inference_noise_std 0.0 \
        --reset_schedule sequential --keep_configured_episode_length \
        --disable_tracking_terminations --keep_after_done \
        --save_rollout_training_samples \
        --samples_output_dir "${output}/rollout_training_samples" \
        --sample_rows_per_file 1000 --balanced_rows_per_motion 1000 \
        --kit_args=--/app/extensions/fsWatcherEnabled=false \
        agent.logger.backend= \
        env.random_reset_step_min=0 env.random_reset_step_max=200 \
        env.random_reset_full_trajectory=false env.reset_schedule=sequential \
        env.wrap_steps=false env.observations.policy.enable_corruption=false
}

merge_samples() {
    local demonstrations="$1"
    local rollout="$2"
    local output="$3"
    run_if_missing "${output}/merge_manifest.json" \
        pixi run python experiments/interface_baselines/merge_planner_samples.py \
        --source "${demonstrations}" --source_limit 1000 \
        --source "${rollout}" --source_limit 1000 \
        --seed "${EVAL_SEED}" --output_dir "${output}"
}

train_planner latent_skill "${LATENT_DEMONSTRATIONS}" "${latent_pretrain}"
train_planner full_body_trajectory "${VANILLA_DEMONSTRATIONS}" "${explicit_pretrain}"
eval_latent "${latent_pretrain}/checkpoints/latest.pt" \
    "${latent_root}/eval_pretrained_10starts" \
    "capacity_${PLANNER_FAMILY}_${MODEL_SIZE}_seed${PLANNER_SEED}_latent_pretrained"
eval_explicit "${explicit_pretrain}/checkpoints/latest.pt" \
    "${explicit_root}/eval_pretrained_10starts" \
    "capacity_${PLANNER_FAMILY}_${MODEL_SIZE}_seed${PLANNER_SEED}_explicit_pretrained"
collect_latent
collect_explicit
merge_samples "${LATENT_DEMONSTRATIONS}" \
    "${latent_root}/planner_rollout_collection/rollout_training_samples" \
    "${latent_root}/demonstration_and_rollout_samples"
merge_samples "${VANILLA_DEMONSTRATIONS}" \
    "${explicit_root}/planner_rollout_collection/rollout_training_samples" \
    "${explicit_root}/demonstration_and_rollout_samples"
train_planner latent_skill "${latent_root}/demonstration_and_rollout_samples" \
    "${latent_finetune}" "${latent_pretrain}/checkpoints/latest.pt"
train_planner full_body_trajectory \
    "${explicit_root}/demonstration_and_rollout_samples" \
    "${explicit_finetune}" "${explicit_pretrain}/checkpoints/latest.pt"
eval_latent "${latent_finetune}/checkpoints/latest.pt" \
    "${latent_root}/eval_finetuned_10starts" \
    "capacity_${PLANNER_FAMILY}_${MODEL_SIZE}_seed${PLANNER_SEED}_latent_finetuned"
eval_explicit "${explicit_finetune}/checkpoints/latest.pt" \
    "${explicit_root}/eval_finetuned_10starts" \
    "capacity_${PLANNER_FAMILY}_${MODEL_SIZE}_seed${PLANNER_SEED}_explicit_finetuned"

run_if_missing "${latent_root}/causal_no_reference_leak_finetuned.json" \
    pixi run python experiments/interface_baselines/audit_one_motion_causal_latent_gate.py \
    --samples_dir "${latent_root}/demonstration_and_rollout_samples" \
    --planner_checkpoint "${latent_finetune}/checkpoints/latest.pt" \
    --low_level_checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    --interface latent_skill --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
    --motion_name "${MOTION_NAME}" --expected_rows 2000 \
    --output_json "${latent_root}/causal_no_reference_leak_finetuned.json"
run_if_missing "${explicit_root}/causal_no_reference_leak_finetuned.json" \
    pixi run python experiments/interface_baselines/audit_one_motion_causal_latent_gate.py \
    --samples_dir "${explicit_root}/demonstration_and_rollout_samples" \
    --planner_checkpoint "${explicit_finetune}/checkpoints/latest.pt" \
    --low_level_checkpoint "${VANILLA_LOW_LEVEL_CHECKPOINT}" \
    --interface full_body_trajectory --motion_name "${MOTION_NAME}" \
    --expected_rows 2000 \
    --output_json "${explicit_root}/causal_no_reference_leak_finetuned.json"

if [[ "${DRY_RUN}" != "1" ]]; then
    pixi run python experiments/interface_baselines/aggregate_one_motion_capacity_scaling.py \
        --scaling_root "${OUTPUT_ROOT}" --sizes "${MODEL_SIZE}" \
        --latent_oracle "${LATENT_ORACLE_SUMMARY}" \
        --explicit_oracle "${VANILLA_ORACLE_SUMMARY}" \
        --output_dir "${POINT_ROOT}/point_summary" --overwrite
fi

echo "[PASS] Completed ${PLANNER_FAMILY} ${MODEL_SIZE} planner seed ${PLANNER_SEED} with evaluation seed ${EVAL_SEED}."
