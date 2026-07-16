#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate the DiffSR latent interface with the same continuous
# planner and two-stage data contract used by the explicit chunk baselines.

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

TASK="${TASK:-Isaac-Imitation-G1-Latent-v0}"
ALGORITHM="${ALGORITHM:-IPMD}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-}"
MANIFEST="${MANIFEST:-data/lafan1/manifests/g1_lafan1_manifest.json}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST}}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST}}"
DATASET_PATH="${DATASET_PATH:-data/lafan1/g1_hl_diffsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/shared_latent_interface}"
MOTION_NAME="${MOTION_NAME:-}"
TRAJECTORY_NAME="${TRAJECTORY_NAME:-}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
HORIZON_STEPS="${HORIZON_STEPS:-10}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
Z_DIM="${Z_DIM:-256}"
LATENT_DIM="${LATENT_DIM:-$((Z_DIM + 2))}"
COLLECT_SAMPLES="${COLLECT_SAMPLES:-1200}"
SAMPLE_BUDGET="${SAMPLE_BUDGET:-1000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
FLOW_STEPS="${FLOW_STEPS:-16}"
TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS:-4}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-}"
LANGUAGE_GOAL_NAME="${LANGUAGE_GOAL_NAME:-}"
RUN_ORACLE="${RUN_ORACLE:-1}"
FORCE_COLLECT="${FORCE_COLLECT:-0}"
USE_CHECKPOINT_NORMALIZATION="${USE_CHECKPOINT_NORMALIZATION:-0}"
DRY_RUN="${DRY_RUN:-0}"

: "${LOW_LEVEL_CHECKPOINT:?Set LOW_LEVEL_CHECKPOINT to the latent low-level policy.}"
: "${SKILL_CHECKPOINT:?Set SKILL_CHECKPOINT to the frozen DiffSR skill encoder.}"

if [[ "${SAMPLE_BUDGET}" == "all" || "${SAMPLE_BUDGET}" == "0" ]]; then
    MAX_SAMPLES=0
    DAGGER_SAMPLE_COUNT="${COLLECT_SAMPLES}"
    BUDGET_LABEL=all
else
    MAX_SAMPLES="${SAMPLE_BUDGET}"
    DAGGER_SAMPLE_COUNT="${SAMPLE_BUDGET}"
    BUDGET_LABEL="${SAMPLE_BUDGET}"
fi
COLLECT_CONTROL_STEPS="$((COLLECT_SAMPLES * HORIZON_STEPS))"
DAGGER_CONTROL_STEPS="$((DAGGER_SAMPLE_COUNT * HORIZON_STEPS))"

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

sample_row_count() {
    local samples_path="$1"
    if [[ ! -d "${samples_path}" ]]; then
        echo 0
        return
    fi
    "${PYTHON_CMD[@]}" - "${samples_path}" <<'PY'
import sys
from pathlib import Path

sys.path.append(str(Path("experiments/interface_baselines").resolve()))
from interface_planner_common import load_rollout_samples

data, _ = load_rollout_samples(Path(sys.argv[1]))
print(int(data["causal_target"].shape[0]))
PY
}

FILTER_ARGS=()
if [[ -n "${MOTION_NAME}" ]]; then
    FILTER_ARGS+=(--motion_name "${MOTION_NAME}")
fi
if [[ -n "${TRAJECTORY_NAME}" ]]; then
    FILTER_ARGS+=(--trajectory_name "${TRAJECTORY_NAME}")
fi

COMMON_OVERRIDES=(
    "agent.logger.backend="
    "agent.ipmd.hl_skill_finetune_enabled=false"
    "env.dataset_path=${DATASET_PATH}"
    "env.reset_schedule=sequential"
    "env.wrap_steps=false"
    "env.observations.policy.enable_corruption=false"
    "env.refresh_zarr_dataset=false"
    "env.latent_command_dim=${LATENT_DIM}"
    "agent.ipmd.latent_dim=${LATENT_DIM}"
    "agent.ipmd.latent_steps_min=${HORIZON_STEPS}"
    "agent.ipmd.latent_steps_max=${HORIZON_STEPS}"
    "agent.ipmd.hl_skill_horizon_steps=${HORIZON_STEPS}"
    "agent.ipmd.hl_skill_command_mode=z"
    "agent.ipmd.latent_learning.command_phase_mode=sin_cos"
    "agent.ipmd.latent_learning.code_latent_dim=${Z_DIM}"
    "agent.ipmd.latent_learning.code_period=${HORIZON_STEPS}"
    "agent.ipmd.reward_loss_coeff=0.0"
    "agent.ipmd.reward_l2_coeff=0.0"
    "agent.ipmd.reward_grad_penalty_coeff=0.0"
    "agent.ipmd.reward_logit_reg_coeff=0.0"
    "agent.ipmd.reward_param_weight_decay_coeff=0.0"
)

run_latent_eval() {
    local command_source="$1"
    local planner_checkpoint="$2"
    local manifest="$3"
    local output_dir="$4"
    local max_steps="$5"
    local save_samples="$6"
    local continue_after_reset="$7"
    local label="$8"

    local args=(
        --headless
        --task "${TASK}"
        --algorithm "${ALGORITHM}"
        --checkpoint "${LOW_LEVEL_CHECKPOINT}"
        --skill_checkpoint "${SKILL_CHECKPOINT}"
        --state_history_steps "${STATE_HISTORY_STEPS}"
        --output_dir "${output_dir}"
        --label "${label}"
        "${FILTER_ARGS[@]}"
        --num_envs "${NUM_ENVS}"
        --max_steps "${max_steps}"
        --seed "${SEED}"
        --metric_interval 1
        --keep_time_out
        --keep_early_terminations
        --disable_reward_clipping
        --flow_num_inference_steps "${FLOW_STEPS}"
        --flow_inference_noise_std "${FLOW_NOISE_STD}"
        --kit_args=--/app/extensions/fsWatcherEnabled=false
    )
    if [[ -n "${planner_checkpoint}" ]]; then
        args+=(
            --planner_checkpoint "${planner_checkpoint}"
        )
    fi
    if [[ -n "${planner_checkpoint}" || "${save_samples}" == "1" ]]; then
        args+=(
            --allow_random_reset
            --disable_tracking_terminations
        )
    else
        args+=(--extend_episode_length_for_max_steps)
    fi
    if [[ -n "${LANGUAGE_EMBEDDINGS}" ]]; then
        args+=(--language_embeddings "${LANGUAGE_EMBEDDINGS}")
    fi
    if [[ "${save_samples}" == "1" ]]; then
        args+=(--save_rollout_training_samples)
    fi
    if [[ "${continue_after_reset}" == "1" ]]; then
        args+=(--continue_after_reset)
    fi

    local source_overrides=("agent.ipmd.command_source=${command_source}")
    if [[ "${command_source}" == "hl_skill" ]]; then
        source_overrides+=(
            "agent.ipmd.hl_skill_checkpoint_path=${SKILL_CHECKPOINT}"
        )
    else
        if [[ -n "${LANGUAGE_EMBEDDINGS}" && -z "${LANGUAGE_GOAL_NAME}" ]]; then
            echo "[ERROR] LANGUAGE_GOAL_NAME is required for language-conditioned closed-loop evaluation." >&2
            exit 2
        fi
        source_overrides+=(
            "agent.ipmd.skill_commander_checkpoint_path=${planner_checkpoint}"
            "agent.ipmd.skill_commander_use_achieved_state=true"
            "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}"
            "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}"
        )
        if [[ -n "${LANGUAGE_EMBEDDINGS}" ]]; then
            source_overrides+=(
                "agent.ipmd.skill_commander_embeddings_path=${LANGUAGE_EMBEDDINGS}"
                "agent.ipmd.skill_commander_goal_name=${LANGUAGE_GOAL_NAME}"
            )
        fi
    fi

    run_cmd "${ISAACLAB_PYTHON_CMD[@]}" \
        scripts/rlopt/eval_skill_commander_closed_loop.py \
        "${args[@]}" \
        "${source_overrides[@]}" \
        "env.lafan1_manifest_path=${manifest}" \
        "${COMMON_OVERRIDES[@]}"
}

run_root="${OUTPUT_ROOT}/latent_skill/transformer_${MODEL_SIZE}_${BUDGET_LABEL}"
oracle_dir="${OUTPUT_ROOT}/latent_skill/oracle_low_level"
demonstration_dir="${OUTPUT_ROOT}/latent_skill/demonstration_samples"
demonstration_samples_dir="${demonstration_dir}/rollout_training_samples"
pretrain_dir="${run_root}/planner_pretrain_demonstration"
pretrained_offline_eval_dir="${run_root}/eval_pretrained_offline"
pretrained_eval_dir="${run_root}/eval_pretrained_closed_loop"
planner_rollout_dir="${run_root}/planner_rollout_collection"
planner_rollout_samples_dir="${planner_rollout_dir}/rollout_training_samples"
merged_samples_dir="${run_root}/demonstration_and_planner_rollout_samples"
finetune_dir="${run_root}/planner_finetune_planner_rollout"
finetuned_offline_eval_dir="${run_root}/eval_finetuned_offline"
finetuned_eval_dir="${run_root}/eval_finetuned_closed_loop"
mkdir -p "${run_root}"

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/write_interface_run_provenance.py \
    --label shared-latent-single-seed \
    --output_json "${OUTPUT_ROOT}/interface_comparison_run_provenance.json" \
    --result_root "${OUTPUT_ROOT}"

if [[ "${RUN_ORACLE}" == "1" ]]; then
    run_latent_eval hl_skill "" "${EVAL_MANIFEST}" "${oracle_dir}" \
        "${EVAL_STEPS}" 0 0 latent_skill_oracle
fi

existing_sample_count="$(sample_row_count "${demonstration_samples_dir}")"
required_sample_count="${COLLECT_SAMPLES}"
if [[ "${NUM_ENVS}" -gt 1 ]]; then
    required_sample_count="$((COLLECT_SAMPLES * NUM_ENVS))"
fi
if [[ "${FORCE_COLLECT}" == "1" || "${existing_sample_count}" -lt "${required_sample_count}" ]]; then
    if [[ "${DRY_RUN}" != "1" ]]; then
        rm -rf "${demonstration_dir}"
    fi
    run_latent_eval hl_skill "" "${TRAIN_MANIFEST}" "${demonstration_dir}" \
        "${COLLECT_CONTROL_STEPS}" 1 1 latent_skill_demonstration_collection
else
    echo "[INFO] Reusing ${existing_sample_count} demonstration rows from ${demonstration_samples_dir}."
fi
if [[ "${DRY_RUN}" != "1" ]]; then
    existing_sample_count="$(sample_row_count "${demonstration_samples_dir}")"
fi

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_chunked_transformer_planner.py \
    --samples_dir "${demonstration_samples_dir}" \
    --output_dir "${pretrain_dir}" \
    --interface latent_skill \
    --state_key expert_planner_state \
    --model_size "${MODEL_SIZE}" \
    --seed "${SEED}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_updates "${PRETRAIN_UPDATES}" \
    --batch_size "${BATCH_SIZE}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --flow_num_inference_steps "${FLOW_STEPS}" \
    --endpoint_num_inference_steps "${TRAIN_ENDPOINT_STEPS}" \
    --flow_inference_noise_std "${FLOW_NOISE_STD}"

offline_filter_args=()
if [[ "${MAX_SAMPLES}" -gt 0 && "${existing_sample_count}" -gt "${MAX_SAMPLES}" ]]; then
    offline_filter_args+=(--exclude_checkpoint_selected_indices)
fi
run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_offline.py \
    --samples_dir "${demonstration_samples_dir}" \
    --planner_checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
    --output_json "${pretrained_offline_eval_dir}/summary.json" \
    --output_csv "${pretrained_offline_eval_dir}/summary.csv" \
    --interface latent_skill \
    --state_key expert_planner_state \
    --setting eval_pretrained_demonstration \
    --label latent_skill_pretrained_demonstration \
    --seed "${SEED}" \
    --flow_num_inference_steps "${FLOW_STEPS}" \
    --flow_inference_noise_std "${FLOW_NOISE_STD}" \
    "${offline_filter_args[@]}"

run_latent_eval skill_commander "${pretrain_dir}/checkpoints/latest.pt" \
    "${EVAL_MANIFEST}" "${pretrained_eval_dir}" "${EVAL_STEPS}" 0 0 \
    latent_skill_pretrained_closed_loop

if [[ "${DRY_RUN}" != "1" ]]; then
    rm -rf "${planner_rollout_dir}" "${merged_samples_dir}"
fi
run_latent_eval skill_commander "${pretrain_dir}/checkpoints/latest.pt" \
    "${TRAIN_MANIFEST}" "${planner_rollout_dir}" "${DAGGER_CONTROL_STEPS}" 1 1 \
    latent_skill_planner_rollout_collection

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/merge_planner_samples.py \
    --source "${demonstration_samples_dir}" \
    --source_limit "${MAX_SAMPLES}" \
    --source "${planner_rollout_samples_dir}" \
    --source_limit "${MAX_SAMPLES}" \
    --seed "${SEED}" \
    --output_dir "${merged_samples_dir}"

normalization_args=()
if [[ "${USE_CHECKPOINT_NORMALIZATION}" == "1" ]]; then
    normalization_args+=(--use_checkpoint_normalization)
fi
run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_chunked_transformer_planner.py \
    --samples_dir "${merged_samples_dir}" \
    --output_dir "${finetune_dir}" \
    --interface latent_skill \
    --state_key planner_state \
    --checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
    --model_size "${MODEL_SIZE}" \
    --seed "${SEED}" \
    --max_samples 0 \
    --num_updates "${FINETUNE_UPDATES}" \
    --batch_size "${BATCH_SIZE}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --flow_num_inference_steps "${FLOW_STEPS}" \
    --endpoint_num_inference_steps "${TRAIN_ENDPOINT_STEPS}" \
    --flow_inference_noise_std "${FLOW_NOISE_STD}" \
    "${normalization_args[@]}"

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_offline.py \
    --samples_dir "${merged_samples_dir}" \
    --planner_checkpoint "${finetune_dir}/checkpoints/latest.pt" \
    --output_json "${finetuned_offline_eval_dir}/summary.json" \
    --output_csv "${finetuned_offline_eval_dir}/summary.csv" \
    --interface latent_skill \
    --state_key planner_state \
    --setting eval_finetuned_planner_rollout \
    --label latent_skill_finetuned_planner_rollout \
    --seed "${SEED}" \
    --flow_num_inference_steps "${FLOW_STEPS}" \
    --flow_inference_noise_std "${FLOW_NOISE_STD}"

run_latent_eval skill_commander "${finetune_dir}/checkpoints/latest.pt" \
    "${EVAL_MANIFEST}" "${finetuned_eval_dir}" "${EVAL_STEPS}" 0 0 \
    latent_skill_finetuned_closed_loop

echo "[INFO] Done. Results under ${OUTPUT_ROOT}."
