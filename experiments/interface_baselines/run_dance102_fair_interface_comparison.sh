#!/usr/bin/env bash
set -euo pipefail

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

TASK="${TASK:-Isaac-Imitation-G1-v0}"
ALGORITHM="${ALGORITHM:-IPMD}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-}"
FULL_BODY_TRAJECTORY_CHECKPOINT="${FULL_BODY_TRAJECTORY_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT}}"
EE_TRAJECTORY_CHECKPOINT="${EE_TRAJECTORY_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT}}"
MANIFEST="${MANIFEST:-data/unitree/manifests/g1_unitree_dance102_manifest.json}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST}}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/dance102_fair_interface_comparison}"
INTERFACES="${INTERFACES-full_body_trajectory ee_trajectory}"
RUN_LATENT="${RUN_LATENT:-1}"
LATENT_TASK="${LATENT_TASK:-Isaac-Imitation-G1-Latent-v0}"
LATENT_ALGORITHM="${LATENT_ALGORITHM:-IPMD_BILINEAR}"
LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-}"
LATENT_PLANNER_CHECKPOINT="${LATENT_PLANNER_CHECKPOINT:-}"
LATENT_LANGUAGE_EMBEDDINGS="${LATENT_LANGUAGE_EMBEDDINGS:-}"
LATENT_MOTION_NAME="${LATENT_MOTION_NAME-dance102}"
LATENT_TRAJECTORY_NAME="${LATENT_TRAJECTORY_NAME-}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-data/unitree/g1_dance102_hl_diffsr}"
LATENT_COMMAND_MODE="${LATENT_COMMAND_MODE:-z}"
LATENT_DIM="${LATENT_DIM:-258}"
LATENT_CODE_DIM="${LATENT_CODE_DIM:-256}"
LATENT_STEPS="${LATENT_STEPS:-10}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-${STEPS}}"
COLLECT_STEPS="${COLLECT_STEPS:-${STEPS}}"
LATENT_EVAL_STEPS="${LATENT_EVAL_STEPS:-${EVAL_STEPS}}"
LATENT_COLLECT_STEPS="${LATENT_COLLECT_STEPS:-${COLLECT_STEPS}}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-0}"
COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS:-0}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-${FINETUNE_UPDATES}}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-256}"
FINETUNE_LR="${FINETUNE_LR:-1.0e-4}"
FINETUNE_WEIGHT_DECAY="${FINETUNE_WEIGHT_DECAY:-${WEIGHT_DECAY:-1.0e-4}}"
FLOW_STEPS="${FLOW_STEPS:-16}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
DRY_RUN="${DRY_RUN:-0}"

export TASK
export ALGORITHM
export LOW_LEVEL_CHECKPOINT
export FULL_BODY_TRAJECTORY_CHECKPOINT
export EE_TRAJECTORY_CHECKPOINT
export MANIFEST
export TRAIN_MANIFEST
export EVAL_MANIFEST
export OUTPUT_ROOT
export INTERFACES
export RUN_LATENT
export LATENT_TASK
export LATENT_ALGORITHM
export LATENT_LOW_LEVEL_CHECKPOINT
export LATENT_SKILL_CHECKPOINT
export LATENT_PLANNER_CHECKPOINT
export LATENT_LANGUAGE_EMBEDDINGS
export LATENT_MOTION_NAME
export LATENT_TRAJECTORY_NAME
export LATENT_DATASET_PATH
export LATENT_COMMAND_MODE
export LATENT_DIM
export LATENT_CODE_DIM
export LATENT_STEPS
export SEED
export NUM_ENVS
export STEPS
export EVAL_STEPS
export COLLECT_STEPS
export LATENT_EVAL_STEPS
export LATENT_COLLECT_STEPS
export STATE_HISTORY_STEPS
export COMMAND_PAST_STEPS
export COMMAND_FUTURE_STEPS
export PRETRAIN_UPDATES
export FINETUNE_UPDATES
export FINETUNE_BATCH_SIZE
export FINETUNE_LR
export FINETUNE_WEIGHT_DECAY
export FLOW_STEPS
export FLOW_NOISE_STD

mkdir -p "${OUTPUT_ROOT}"
echo "[INFO] Train manifest: ${TRAIN_MANIFEST}"
echo "[INFO] Eval manifest: ${EVAL_MANIFEST}"

if [[ "${RUN_LATENT}" == "1" ]]; then
    : "${LATENT_LOW_LEVEL_CHECKPOINT:?Set LATENT_LOW_LEVEL_CHECKPOINT when RUN_LATENT=1.}"
    : "${LATENT_SKILL_CHECKPOINT:?Set LATENT_SKILL_CHECKPOINT when RUN_LATENT=1.}"
    : "${LATENT_PLANNER_CHECKPOINT:?Set LATENT_PLANNER_CHECKPOINT when RUN_LATENT=1.}"
fi

LATENT_LANGUAGE_ARGS=()
LATENT_COMMANDER_EMBEDDING_OVERRIDES=()
if [[ -n "${LATENT_LANGUAGE_EMBEDDINGS}" ]]; then
    LATENT_LANGUAGE_ARGS+=(--language_embeddings "${LATENT_LANGUAGE_EMBEDDINGS}")
    LATENT_COMMANDER_EMBEDDING_OVERRIDES+=(
        "agent.ipmd.skill_commander_embeddings_path=${LATENT_LANGUAGE_EMBEDDINGS}"
    )
fi

LATENT_FILTER_ARGS=()
if [[ -n "${LATENT_MOTION_NAME}" ]]; then
    LATENT_FILTER_ARGS+=(--motion_name "${LATENT_MOTION_NAME}")
fi
if [[ -n "${LATENT_TRAJECTORY_NAME}" ]]; then
    LATENT_FILTER_ARGS+=(--trajectory_name "${LATENT_TRAJECTORY_NAME}")
fi

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/write_interface_run_provenance.py \
    --label dance102-fair-single-seed \
    --output_json "${OUTPUT_ROOT}/interface_comparison_run_provenance.json" \
    --result_root "${OUTPUT_ROOT}"

if [[ "${RUN_LATENT}" == "1" ]]; then
    latent_root="${OUTPUT_ROOT}/latent_skill"
    latent_oracle_dir="${latent_root}/oracle_low_level"
    latent_samples_dir="${latent_root}/oracle_drive_samples"
    latent_pretrained_offline_eval_dir="${latent_root}/eval_pretrained_expert_state"
    latent_pretrained_eval_dir="${latent_root}/eval_pretrained_closed_loop"
    latent_finetune_dir="${latent_root}/planner_finetune_achieved_state"
    latent_finetuned_offline_eval_dir="${latent_root}/eval_finetuned_achieved_state"
    latent_finetuned_eval_dir="${latent_root}/eval_finetuned_closed_loop"
    mkdir -p "${latent_root}"

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless \
        --task "${LATENT_TASK}" \
        --algorithm "${LATENT_ALGORITHM}" \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --planner_checkpoint "${LATENT_PLANNER_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        "${LATENT_LANGUAGE_ARGS[@]}" \
        --output_dir "${latent_oracle_dir}" \
        --label latent_skill_oracle \
        "${LATENT_FILTER_ARGS[@]}" \
        --num_envs "${NUM_ENVS}" \
        --max_steps "${LATENT_EVAL_STEPS}" \
        --seed "${SEED}" \
        --metric_interval 1 \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}" \
        "agent.ipmd.command_source=hl_skill" \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        "env.lafan1_manifest_path=${EVAL_MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        "env.refresh_zarr_dataset=false" \
        "env.latent_command_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_steps_min=${LATENT_STEPS}" \
        "agent.ipmd.latent_steps_max=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_horizon_steps=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_command_mode=${LATENT_COMMAND_MODE}" \
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos" \
        "agent.ipmd.latent_learning.code_latent_dim=${LATENT_CODE_DIM}" \
        "agent.ipmd.latent_learning.code_period=${LATENT_STEPS}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless \
        --task "${LATENT_TASK}" \
        --algorithm "${LATENT_ALGORITHM}" \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --planner_checkpoint "${LATENT_PLANNER_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        "${LATENT_LANGUAGE_ARGS[@]}" \
        --output_dir "${latent_samples_dir}" \
        --label latent_skill_oracle_drive_samples \
        "${LATENT_FILTER_ARGS[@]}" \
        --num_envs "${NUM_ENVS}" \
        --max_steps "${LATENT_COLLECT_STEPS}" \
        --seed "${SEED}" \
        --metric_interval 1 \
        --continue_after_reset \
        --save_rollout_training_samples \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}" \
        "agent.ipmd.command_source=hl_skill" \
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}" \
        "env.lafan1_manifest_path=${TRAIN_MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        "env.refresh_zarr_dataset=false" \
        "env.latent_command_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_steps_min=${LATENT_STEPS}" \
        "agent.ipmd.latent_steps_max=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_horizon_steps=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_command_mode=${LATENT_COMMAND_MODE}" \
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos" \
        "agent.ipmd.latent_learning.code_latent_dim=${LATENT_CODE_DIM}" \
        "agent.ipmd.latent_learning.code_period=${LATENT_STEPS}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false

    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_latent_skill_planner_offline.py \
        --samples_dir "${latent_samples_dir}/rollout_training_samples" \
        --planner_checkpoint "${LATENT_PLANNER_CHECKPOINT}" \
        --output_json "${latent_pretrained_offline_eval_dir}/summary.json" \
        --output_csv "${latent_pretrained_offline_eval_dir}/summary.csv" \
        --state_key expert_planner_state \
        --setting eval_pretrained_expert_state \
        --label latent_skill_pretrained_expert_state \
        --seed "${SEED}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}"

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless \
        --task "${LATENT_TASK}" \
        --algorithm "${LATENT_ALGORITHM}" \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --planner_checkpoint "${LATENT_PLANNER_CHECKPOINT}" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        "${LATENT_LANGUAGE_ARGS[@]}" \
        --output_dir "${latent_pretrained_eval_dir}" \
        --label latent_skill_pretrained_closed_loop \
        "${LATENT_FILTER_ARGS[@]}" \
        --num_envs "${NUM_ENVS}" \
        --max_steps "${LATENT_EVAL_STEPS}" \
        --seed "${SEED}" \
        --metric_interval 1 \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}" \
        "agent.ipmd.command_source=skill_commander" \
        "agent.ipmd.skill_commander_checkpoint_path=${LATENT_PLANNER_CHECKPOINT}" \
        "${LATENT_COMMANDER_EMBEDDING_OVERRIDES[@]}" \
        "agent.ipmd.skill_commander_use_achieved_state=true" \
        "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}" \
        "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}" \
        "env.lafan1_manifest_path=${EVAL_MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        "env.refresh_zarr_dataset=false" \
        "env.latent_command_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_steps_min=${LATENT_STEPS}" \
        "agent.ipmd.latent_steps_max=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_horizon_steps=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_command_mode=${LATENT_COMMAND_MODE}" \
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos" \
        "agent.ipmd.latent_learning.code_latent_dim=${LATENT_CODE_DIM}" \
        "agent.ipmd.latent_learning.code_period=${LATENT_STEPS}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false

    run_cmd "${PYTHON_CMD[@]}" scripts/rlopt/finetune_skill_commander_rollout.py \
        --checkpoint "${LATENT_PLANNER_CHECKPOINT}" \
        --samples_dir "${latent_samples_dir}/rollout_training_samples" \
        --output_dir "${latent_finetune_dir}" \
        --seed "${SEED}" \
        --num_updates "${FINETUNE_UPDATES}" \
        --batch_size "${FINETUNE_BATCH_SIZE}" \
        --lr "${FINETUNE_LR}" \
        --weight_decay "${FINETUNE_WEIGHT_DECAY}" \
        --flow_loss_coeff 1.0 \
        --endpoint_loss_coeff 1.0 \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}"

    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_latent_skill_planner_offline.py \
        --samples_dir "${latent_samples_dir}/rollout_training_samples" \
        --planner_checkpoint "${latent_finetune_dir}/checkpoints/latest.pt" \
        --output_json "${latent_finetuned_offline_eval_dir}/summary.json" \
        --output_csv "${latent_finetuned_offline_eval_dir}/summary.csv" \
        --state_key planner_state \
        --setting eval_finetuned_achieved_state \
        --label latent_skill_finetuned_achieved_state \
        --seed "${SEED}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}"

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" scripts/rlopt/eval_skill_commander_closed_loop.py \
        --headless \
        --task "${LATENT_TASK}" \
        --algorithm "${LATENT_ALGORITHM}" \
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}" \
        --planner_checkpoint "${latent_finetune_dir}/checkpoints/latest.pt" \
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}" \
        "${LATENT_LANGUAGE_ARGS[@]}" \
        --output_dir "${latent_finetuned_eval_dir}" \
        --label latent_skill_finetuned_closed_loop \
        "${LATENT_FILTER_ARGS[@]}" \
        --num_envs "${NUM_ENVS}" \
        --max_steps "${LATENT_EVAL_STEPS}" \
        --seed "${SEED}" \
        --metric_interval 1 \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}" \
        "agent.ipmd.command_source=skill_commander" \
        "agent.ipmd.skill_commander_checkpoint_path=${latent_finetune_dir}/checkpoints/latest.pt" \
        "${LATENT_COMMANDER_EMBEDDING_OVERRIDES[@]}" \
        "agent.ipmd.skill_commander_use_achieved_state=true" \
        "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}" \
        "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}" \
        "env.lafan1_manifest_path=${EVAL_MANIFEST}" \
        "env.dataset_path=${LATENT_DATASET_PATH}" \
        "env.refresh_zarr_dataset=false" \
        "env.latent_command_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_dim=${LATENT_DIM}" \
        "agent.ipmd.latent_steps_min=${LATENT_STEPS}" \
        "agent.ipmd.latent_steps_max=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_horizon_steps=${LATENT_STEPS}" \
        "agent.ipmd.hl_skill_command_mode=${LATENT_COMMAND_MODE}" \
        "agent.ipmd.latent_learning.command_phase_mode=sin_cos" \
        "agent.ipmd.latent_learning.code_latent_dim=${LATENT_CODE_DIM}" \
        "agent.ipmd.latent_learning.code_period=${LATENT_STEPS}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false
fi

for interface in ${INTERFACES}; do
    case "${interface}" in
        full_body_trajectory)
            interface_checkpoint="${FULL_BODY_TRAJECTORY_CHECKPOINT}"
            ;;
        ee_trajectory)
            interface_checkpoint="${EE_TRAJECTORY_CHECKPOINT}"
            ;;
        *)
            interface_checkpoint="${LOW_LEVEL_CHECKPOINT}"
            ;;
    esac
    if [[ -z "${interface_checkpoint}" ]]; then
        echo "[ERROR] Set a checkpoint for ${interface}: LOW_LEVEL_CHECKPOINT or the interface-specific override." >&2
        exit 1
    fi

    interface_root="${OUTPUT_ROOT}/${interface}"
    oracle_dir="${interface_root}/oracle_low_level"
    samples_dir="${interface_root}/oracle_drive_samples"
    pretrain_dir="${interface_root}/planner_pretrain_expert_state"
    pretrained_offline_eval_dir="${interface_root}/eval_pretrained_expert_state"
    pretrained_eval_dir="${interface_root}/eval_pretrained_closed_loop"
    finetune_dir="${interface_root}/planner_finetune_achieved_state"
    finetuned_offline_eval_dir="${interface_root}/eval_finetuned_achieved_state"
    finetuned_eval_dir="${interface_root}/eval_finetuned_closed_loop"

    mkdir -p "${interface_root}"

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/command_space_ablation/evaluate_checkpoint.py \
        --headless \
        --task "${TASK}" \
        --algo "${ALGORITHM}" \
        --checkpoint "${interface_checkpoint}" \
        --command_space "${interface}" \
        --command_past_steps "${COMMAND_PAST_STEPS}" \
        --command_future_steps "${COMMAND_FUTURE_STEPS}" \
        --command_observation_source reference \
        --motion_manifest "${EVAL_MANIFEST}" \
        --num_envs "${NUM_ENVS}" \
        --steps "${EVAL_STEPS}" \
        --seed "${SEED}" \
        --label "${interface}_oracle" \
        --output_json "${oracle_dir}/summary.json" \
        --output_csv "${oracle_dir}/summary.csv" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/collect_interface_rollout_samples.py \
        --headless \
        --task "${TASK}" \
        --algo "${ALGORITHM}" \
        --checkpoint "${interface_checkpoint}" \
        --interface "${interface}" \
        --output_dir "${samples_dir}" \
        --motion_manifest "${TRAIN_MANIFEST}" \
        --num_envs "${NUM_ENVS}" \
        --steps "${COLLECT_STEPS}" \
        --seed "${SEED}" \
        --state_history_steps "${STATE_HISTORY_STEPS}" \
        --command_past_steps "${COMMAND_PAST_STEPS}" \
        --command_future_steps "${COMMAND_FUTURE_STEPS}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false

    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_interface_planner.py \
        --samples_dir "${samples_dir}/rollout_training_samples" \
        --output_dir "${pretrain_dir}" \
        --interface "${interface}" \
        --state_key expert_planner_state \
        --seed "${SEED}" \
        --num_updates "${PRETRAIN_UPDATES}" \
        --batch_size "${FINETUNE_BATCH_SIZE}" \
        --lr "${FINETUNE_LR}" \
        --weight_decay "${FINETUNE_WEIGHT_DECAY}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}"

    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_offline.py \
        --samples_dir "${samples_dir}/rollout_training_samples" \
        --planner_checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
        --output_json "${pretrained_offline_eval_dir}/summary.json" \
        --output_csv "${pretrained_offline_eval_dir}/summary.csv" \
        --interface "${interface}" \
        --state_key expert_planner_state \
        --setting eval_pretrained_expert_state \
        --label "${interface}_pretrained_expert_state" \
        --seed "${SEED}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}"

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_closed_loop.py \
        --headless \
        --task "${TASK}" \
        --algo "${ALGORITHM}" \
        --checkpoint "${interface_checkpoint}" \
        --planner_checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
        --output_json "${pretrained_eval_dir}/summary.json" \
        --output_csv "${pretrained_eval_dir}/summary.csv" \
        --label "${interface}_pretrained_closed_loop" \
        --motion_manifest "${EVAL_MANIFEST}" \
        --num_envs "${NUM_ENVS}" \
        --steps "${EVAL_STEPS}" \
        --seed "${SEED}" \
        --state_history_steps "${STATE_HISTORY_STEPS}" \
        --command_past_steps "${COMMAND_PAST_STEPS}" \
        --command_future_steps "${COMMAND_FUTURE_STEPS}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false

    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_interface_planner.py \
        --samples_dir "${samples_dir}/rollout_training_samples" \
        --output_dir "${finetune_dir}" \
        --interface "${interface}" \
        --state_key planner_state \
        --checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
        --seed "${SEED}" \
        --num_updates "${FINETUNE_UPDATES}" \
        --batch_size "${FINETUNE_BATCH_SIZE}" \
        --lr "${FINETUNE_LR}" \
        --weight_decay "${FINETUNE_WEIGHT_DECAY}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}"

    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_offline.py \
        --samples_dir "${samples_dir}/rollout_training_samples" \
        --planner_checkpoint "${finetune_dir}/checkpoints/latest.pt" \
        --output_json "${finetuned_offline_eval_dir}/summary.json" \
        --output_csv "${finetuned_offline_eval_dir}/summary.csv" \
        --interface "${interface}" \
        --state_key planner_state \
        --setting eval_finetuned_achieved_state \
        --label "${interface}_finetuned_achieved_state" \
        --seed "${SEED}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}"

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_closed_loop.py \
        --headless \
        --task "${TASK}" \
        --algo "${ALGORITHM}" \
        --checkpoint "${interface_checkpoint}" \
        --planner_checkpoint "${finetune_dir}/checkpoints/latest.pt" \
        --output_json "${finetuned_eval_dir}/summary.json" \
        --output_csv "${finetuned_eval_dir}/summary.csv" \
        --label "${interface}_finetuned_closed_loop" \
        --motion_manifest "${EVAL_MANIFEST}" \
        --num_envs "${NUM_ENVS}" \
        --steps "${EVAL_STEPS}" \
        --seed "${SEED}" \
        --state_history_steps "${STATE_HISTORY_STEPS}" \
        --command_past_steps "${COMMAND_PAST_STEPS}" \
        --command_future_steps "${COMMAND_FUTURE_STEPS}" \
        --flow_num_inference_steps "${FLOW_STEPS}" \
        --flow_inference_noise_std "${FLOW_NOISE_STD}" \
        --kit_args=--/app/extensions/fsWatcherEnabled=false
done

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/summarize_interface_comparison.py \
    --result_root "${OUTPUT_ROOT}"

echo "[INFO] Done. Results under ${OUTPUT_ROOT}"
