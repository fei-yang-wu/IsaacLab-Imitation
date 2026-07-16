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
VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT}}"
FULL_BODY_TRAJECTORY_CHECKPOINT="${FULL_BODY_TRAJECTORY_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT}}"
EE_TRAJECTORY_CHECKPOINT="${EE_TRAJECTORY_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT}}"
FULL_BODY_LOW_LEVEL_COMMAND_MODE="${FULL_BODY_LOW_LEVEL_COMMAND_MODE:-native}"
MANIFEST="${MANIFEST:-data/unitree/manifests/g1_unitree_dance102_manifest.json}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${MANIFEST}}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${MANIFEST}}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/dance102_strong_interface_comparison}"
INTERFACES="${INTERFACES-full_body_trajectory ee_trajectory}"
NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-500}"
COLLECT_STEPS_WAS_SET=0
if [[ -n "${COLLECT_STEPS:-}" ]]; then
    COLLECT_STEPS_WAS_SET=1
fi
EVAL_STEPS="${EVAL_STEPS:-${STEPS}}"
COLLECT_STEPS="${COLLECT_STEPS:-${STEPS}}"
SEED="${SEED:-0}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-9}"
COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS:-0}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"
# Held-chunk contract: planner queried every N steps at eval
# (PLANNER_UPDATE_INTERVAL) and the low-level consumes held command windows
# (COMMAND_HOLD_STEPS, applied to oracle eval and sample collection).
# Main paper protocol: one planner decision and held command every ten 50 Hz
# control steps, giving a 5 Hz high-level interface.
PLANNER_UPDATE_INTERVAL="${PLANNER_UPDATE_INTERVAL:-10}"
COMMAND_HOLD_STEPS="${COMMAND_HOLD_STEPS:-10}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
MODEL_SIZES="${MODEL_SIZES:-${MODEL_SIZE}}"
SAMPLE_BUDGETS="${SAMPLE_BUDGETS:-1000}"
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
FORCE_COLLECT="${FORCE_COLLECT:-0}"
RUN_ORACLE="${RUN_ORACLE:-1}"
USE_CHECKPOINT_NORMALIZATION="${USE_CHECKPOINT_NORMALIZATION:-0}"
DRY_RUN="${DRY_RUN:-0}"

max_numeric_sample_requirement() {
    local budgets="$1"
    local fallback="$2"
    local extra="${3:-}"
    local max_value="${fallback}"
    local token
    for token in ${budgets} ${extra}; do
        if [[ "${token}" == "all" || "${token}" == "0" || -z "${token}" ]]; then
            continue
        fi
        if [[ "${token}" =~ ^[0-9]+$ && "${token}" -gt "${max_value}" ]]; then
            max_value="${token}"
        fi
    done
    printf '%s' "${max_value}"
}

if [[ "${COLLECT_STEPS_WAS_SET}" == "0" ]]; then
    COLLECT_STEPS="$(max_numeric_sample_requirement \
        "${SAMPLE_BUDGETS}" \
        "${COLLECT_STEPS}" \
        "${SELECTED_SAMPLE_COUNT:-}")"
fi

export TASK
export ALGORITHM
export LOW_LEVEL_CHECKPOINT
export VANILLA_TRACKER_CHECKPOINT
export FULL_BODY_TRAJECTORY_CHECKPOINT
export EE_TRAJECTORY_CHECKPOINT
export FULL_BODY_LOW_LEVEL_COMMAND_MODE
export MANIFEST
export TRAIN_MANIFEST
export EVAL_MANIFEST
export DATASET_PATH
export OUTPUT_ROOT
export INTERFACES
export NUM_ENVS
export STEPS
export EVAL_STEPS
export COLLECT_STEPS
export SEED
export STATE_HISTORY_STEPS
export COMMAND_PAST_STEPS
export COMMAND_FUTURE_STEPS
export MODEL_SIZE
export MODEL_SIZES
export SAMPLE_BUDGETS
export PRETRAIN_UPDATES
export FINETUNE_UPDATES
export BATCH_SIZE
export MICRO_BATCH_SIZE
export LR
export WEIGHT_DECAY
export FLOW_STEPS
export TRAIN_ENDPOINT_STEPS
export FLOW_NOISE_STD
export LANGUAGE_EMBEDDINGS
export LANGUAGE_GOAL_NAME
export PLANNER_UPDATE_INTERVAL
export COMMAND_HOLD_STEPS
export FORCE_COLLECT
export RUN_ORACLE
export USE_CHECKPOINT_NORMALIZATION

mkdir -p "${OUTPUT_ROOT}"
echo "[INFO] Train manifest: ${TRAIN_MANIFEST}"
echo "[INFO] Eval manifest: ${EVAL_MANIFEST}"

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
    --label dance102-strong-single-seed \
    --output_json "${OUTPUT_ROOT}/interface_comparison_run_provenance.json" \
    --result_root "${OUTPUT_ROOT}"

max_samples_arg() {
    local budget="$1"
    if [[ "${budget}" == "all" || "${budget}" == "0" ]]; then
        printf '0'
    else
        printf '%s' "${budget}"
    fi
}

sample_row_count() {
    local samples_path="$1"
    if [[ ! -d "${samples_path}" ]]; then
        echo 0
        return
    fi
    local summary_path
    summary_path="$(dirname "${samples_path}")/summary.json"
    if [[ -f "${summary_path}" ]]; then
        local row_count
        row_count="$("${PYTHON_CMD[@]}" - "${summary_path}" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
saved_rows = payload.get("saved_rows")
if saved_rows not in (None, ""):
    print(int(saved_rows))
    raise SystemExit(0)
metadata = payload.get("metadata")
saved_steps = payload.get("saved_steps")
if saved_steps not in (None, "") and isinstance(metadata, dict):
    num_envs = metadata.get("num_envs")
    if num_envs not in (None, ""):
        print(int(saved_steps) * int(num_envs))
        raise SystemExit(0)
if saved_steps not in (None, ""):
    print(int(saved_steps))
PY
)"
        if [[ "${row_count}" =~ ^[0-9]+$ ]]; then
            echo "${row_count}"
            return
        fi
    fi
    find "${samples_path}" -maxdepth 1 -type f -name "sample_step_*.pt" 2>/dev/null | wc -l | tr -d ' '
}

finetune_normalization_args=()
if [[ "${USE_CHECKPOINT_NORMALIZATION}" == "1" ]]; then
    finetune_normalization_args+=(--use_checkpoint_normalization)
fi

language_collection_args=()
language_eval_args=()
dataset_args=()
if [[ -n "${DATASET_PATH}" ]]; then
    dataset_args+=(--dataset_path "${DATASET_PATH}")
fi
if [[ -n "${LANGUAGE_EMBEDDINGS}" ]]; then
    language_collection_args+=(--language_embeddings "${LANGUAGE_EMBEDDINGS}")
    if [[ -z "${LANGUAGE_GOAL_NAME}" ]]; then
        echo "[ERROR] LANGUAGE_GOAL_NAME is required for language-conditioned closed-loop evaluation." >&2
        exit 2
    fi
    language_eval_args+=(
        --language_embeddings "${LANGUAGE_EMBEDDINGS}"
        --language_goal_name "${LANGUAGE_GOAL_NAME}"
        --motion_name "${LANGUAGE_GOAL_NAME}"
    )
fi

for interface in ${INTERFACES}; do
    interface_command_mode=native
    case "${interface}" in
        full_body_trajectory)
            interface_command_mode="${FULL_BODY_LOW_LEVEL_COMMAND_MODE}"
            if [[ "${interface_command_mode}" == "streamed_vanilla" ]]; then
                interface_checkpoint="${VANILLA_TRACKER_CHECKPOINT}"
            else
                interface_checkpoint="${FULL_BODY_TRAJECTORY_CHECKPOINT}"
            fi
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

    interface_result_name="${interface}"
    if [[ "${interface_command_mode}" != "native" ]]; then
        interface_result_name="${interface}_${interface_command_mode}"
    fi
    interface_root="${OUTPUT_ROOT}/${interface_result_name}"
    oracle_dir="${interface_root}/oracle_low_level"
    samples_dir="${interface_root}/oracle_drive_samples"
    samples_tensor_dir="${samples_dir}/rollout_training_samples"
    min_reuse_sample_count="${MIN_REUSE_SAMPLE_COUNT:-${SELECTED_SAMPLE_COUNT:-${COLLECT_STEPS}}}"
    mkdir -p "${interface_root}"

    if [[ "${RUN_ORACLE}" == "1" ]]; then
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/command_space_ablation/evaluate_checkpoint.py \
            --headless \
            --task "${TASK}" \
            --algo "${ALGORITHM}" \
            --checkpoint "${interface_checkpoint}" \
            --low_level_command_mode "${interface_command_mode}" \
            --command_space "${interface}" \
            --command_past_steps "${COMMAND_PAST_STEPS}" \
            --command_future_steps "${COMMAND_FUTURE_STEPS}" \
            --planner_update_interval "${PLANNER_UPDATE_INTERVAL}" \
            --command_observation_source reference \
            --motion_manifest "${EVAL_MANIFEST}" \
            "${dataset_args[@]}" \
            --num_envs "${NUM_ENVS}" \
            --steps "${EVAL_STEPS}" \
            --seed "${SEED}" \
            --label "${interface}_oracle" \
            --output_json "${oracle_dir}/summary.json" \
            --output_csv "${oracle_dir}/summary.csv" \
            --kit_args=--/app/extensions/fsWatcherEnabled=false \
            "env.command_hold_steps=${COMMAND_HOLD_STEPS}"
    fi

    existing_sample_count="$(sample_row_count "${samples_tensor_dir}")"
    if [[ "${FORCE_COLLECT}" == "1" || "${existing_sample_count}" -lt "${min_reuse_sample_count}" ]]; then
        if [[ -d "${samples_tensor_dir}" ]]; then
            echo "[INFO] Recollecting samples: found ${existing_sample_count} rows, need ${min_reuse_sample_count} in ${samples_tensor_dir}"
            rm -rf "${samples_tensor_dir}"
        fi
        run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/collect_interface_rollout_samples.py \
            --headless \
            --task "${TASK}" \
            --algo "${ALGORITHM}" \
            --checkpoint "${interface_checkpoint}" \
            --low_level_command_mode "${interface_command_mode}" \
            --interface "${interface}" \
            --output_dir "${samples_dir}" \
            --motion_manifest "${TRAIN_MANIFEST}" \
            "${dataset_args[@]}" \
            "${language_collection_args[@]}" \
            --num_envs "${NUM_ENVS}" \
            --steps "${COLLECT_STEPS}" \
            --seed "${SEED}" \
            --state_history_steps "${STATE_HISTORY_STEPS}" \
            --planner_interval_steps "${PLANNER_UPDATE_INTERVAL}" \
            --command_past_steps "${COMMAND_PAST_STEPS}" \
            --command_future_steps "${COMMAND_FUTURE_STEPS}" \
            --keep_configured_episode_length \
            --disable_tracking_terminations \
            --kit_args=--/app/extensions/fsWatcherEnabled=false \
            "env.command_hold_steps=${COMMAND_HOLD_STEPS}"
    else
        echo "[INFO] Reusing existing samples: ${samples_tensor_dir} (${existing_sample_count} rows)"
    fi

    for model_size in ${MODEL_SIZES}; do
        for budget in ${SAMPLE_BUDGETS}; do
            max_samples="$(max_samples_arg "${budget}")"
            budget_label="${budget}"
            if [[ "${budget_label}" == "0" ]]; then
                budget_label="all"
            fi
            run_root="${interface_root}/chunked_transformer_${model_size}_${budget_label}"
            pretrain_dir="${run_root}/planner_pretrain_expert_state"
            pretrained_offline_eval_dir="${run_root}/eval_pretrained_expert_state"
            pretrained_eval_dir="${run_root}/eval_pretrained_closed_loop"
            planner_rollout_dir="${run_root}/planner_rollout_collection"
            planner_rollout_samples_dir="${planner_rollout_dir}/rollout_training_samples"
            merged_samples_dir="${run_root}/demonstration_and_planner_rollout_samples"
            finetune_dir="${run_root}/planner_finetune_planner_rollout"
            finetuned_offline_eval_dir="${run_root}/eval_finetuned_planner_rollout"
            finetuned_eval_dir="${run_root}/eval_finetuned_closed_loop"

            if [[ "${max_samples}" == "0" ]]; then
                dagger_planner_samples="${COLLECT_STEPS}"
            else
                dagger_planner_samples="${max_samples}"
            fi
            dagger_control_steps="$((dagger_planner_samples * PLANNER_UPDATE_INTERVAL))"

            run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_chunked_transformer_planner.py \
                --samples_dir "${samples_tensor_dir}" \
                --output_dir "${pretrain_dir}" \
                --interface "${interface}" \
                --state_key expert_planner_state \
                --model_size "${model_size}" \
                --seed "${SEED}" \
                --max_samples "${max_samples}" \
                --num_updates "${PRETRAIN_UPDATES}" \
                --batch_size "${BATCH_SIZE}" \
                --micro_batch_size "${MICRO_BATCH_SIZE}" \
                --lr "${LR}" \
                --weight_decay "${WEIGHT_DECAY}" \
                --flow_num_inference_steps "${FLOW_STEPS}" \
                --endpoint_num_inference_steps "${TRAIN_ENDPOINT_STEPS}" \
                --flow_inference_noise_std "${FLOW_NOISE_STD}"

            offline_eval_filter_args=()
            if [[ "${max_samples}" != "0" ]]; then
                offline_eval_filter_args+=(--exclude_checkpoint_selected_indices)
            fi

            run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_offline.py \
                --samples_dir "${samples_tensor_dir}" \
                --planner_checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
                --output_json "${pretrained_offline_eval_dir}/summary.json" \
                --output_csv "${pretrained_offline_eval_dir}/summary.csv" \
                --interface "${interface}" \
                --state_key expert_planner_state \
                --setting eval_pretrained_expert_state \
                --label "${interface}_chunked_${model_size}_${budget_label}_pretrained_expert_state" \
                --seed "${SEED}" \
                --flow_num_inference_steps "${FLOW_STEPS}" \
                --flow_inference_noise_std "${FLOW_NOISE_STD}" \
                "${offline_eval_filter_args[@]}"

            run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_closed_loop.py \
                --headless \
                --task "${TASK}" \
                --algo "${ALGORITHM}" \
                --checkpoint "${interface_checkpoint}" \
                --low_level_command_mode "${interface_command_mode}" \
                --planner_checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
                --output_json "${pretrained_eval_dir}/summary.json" \
                --output_csv "${pretrained_eval_dir}/summary.csv" \
                --label "${interface}_chunked_${model_size}_${budget_label}_pretrained_closed_loop" \
                --motion_manifest "${EVAL_MANIFEST}" \
                "${dataset_args[@]}" \
                "${language_eval_args[@]}" \
                --num_envs "${NUM_ENVS}" \
                --steps "${EVAL_STEPS}" \
                --seed "${SEED}" \
                --state_history_steps "${STATE_HISTORY_STEPS}" \
                --command_past_steps "${COMMAND_PAST_STEPS}" \
                --command_future_steps "${COMMAND_FUTURE_STEPS}" \
                --planner_update_interval "${PLANNER_UPDATE_INTERVAL}" \
                --keep_configured_episode_length \
                --disable_tracking_terminations \
                --flow_num_inference_steps "${FLOW_STEPS}" \
                --flow_inference_noise_std "${FLOW_NOISE_STD}" \
                --kit_args=--/app/extensions/fsWatcherEnabled=false

            if [[ "${DRY_RUN}" != "1" ]]; then
                rm -rf "${planner_rollout_dir}" "${merged_samples_dir}"
            fi
            run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_closed_loop.py \
                --headless \
                --task "${TASK}" \
                --algo "${ALGORITHM}" \
                --checkpoint "${interface_checkpoint}" \
                --low_level_command_mode "${interface_command_mode}" \
                --planner_checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
                --output_json "${planner_rollout_dir}/summary.json" \
                --label "${interface}_chunked_${model_size}_${budget_label}_planner_rollout_collection" \
                --motion_manifest "${TRAIN_MANIFEST}" \
                "${dataset_args[@]}" \
                "${language_eval_args[@]}" \
                --num_envs "${NUM_ENVS}" \
                --steps "${dagger_control_steps}" \
                --seed "${SEED}" \
                --state_history_steps "${STATE_HISTORY_STEPS}" \
                --command_past_steps "${COMMAND_PAST_STEPS}" \
                --command_future_steps "${COMMAND_FUTURE_STEPS}" \
                --planner_update_interval "${PLANNER_UPDATE_INTERVAL}" \
                --keep_configured_episode_length \
                --disable_tracking_terminations \
                --flow_num_inference_steps "${FLOW_STEPS}" \
                --flow_inference_noise_std "${FLOW_NOISE_STD}" \
                --keep_after_done \
                --save_rollout_training_samples \
                --samples_output_dir "${planner_rollout_samples_dir}" \
                --kit_args=--/app/extensions/fsWatcherEnabled=false

            run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/merge_planner_samples.py \
                --source "${samples_tensor_dir}" \
                --source_limit "${max_samples}" \
                --source "${planner_rollout_samples_dir}" \
                --source_limit "${max_samples}" \
                --seed "${SEED}" \
                --output_dir "${merged_samples_dir}"

            run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/train_chunked_transformer_planner.py \
                --samples_dir "${merged_samples_dir}" \
                --output_dir "${finetune_dir}" \
                --interface "${interface}" \
                --state_key planner_state \
                --checkpoint "${pretrain_dir}/checkpoints/latest.pt" \
                --model_size "${model_size}" \
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
                "${finetune_normalization_args[@]}"

            run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_offline.py \
                --samples_dir "${merged_samples_dir}" \
                --planner_checkpoint "${finetune_dir}/checkpoints/latest.pt" \
                --output_json "${finetuned_offline_eval_dir}/summary.json" \
                --output_csv "${finetuned_offline_eval_dir}/summary.csv" \
                --interface "${interface}" \
                --state_key planner_state \
                --setting eval_finetuned_planner_rollout \
                --label "${interface}_chunked_${model_size}_${budget_label}_finetuned_planner_rollout" \
                --seed "${SEED}" \
                --flow_num_inference_steps "${FLOW_STEPS}" \
                --flow_inference_noise_std "${FLOW_NOISE_STD}"

            run_cmd "${ISAACLAB_PYTHON_CMD[@]}" experiments/interface_baselines/eval_interface_planner_closed_loop.py \
                --headless \
                --task "${TASK}" \
                --algo "${ALGORITHM}" \
                --checkpoint "${interface_checkpoint}" \
                --low_level_command_mode "${interface_command_mode}" \
                --planner_checkpoint "${finetune_dir}/checkpoints/latest.pt" \
                --output_json "${finetuned_eval_dir}/summary.json" \
                --output_csv "${finetuned_eval_dir}/summary.csv" \
                --label "${interface}_chunked_${model_size}_${budget_label}_finetuned_closed_loop" \
                --motion_manifest "${EVAL_MANIFEST}" \
                "${dataset_args[@]}" \
                "${language_eval_args[@]}" \
                --num_envs "${NUM_ENVS}" \
                --steps "${EVAL_STEPS}" \
                --seed "${SEED}" \
                --state_history_steps "${STATE_HISTORY_STEPS}" \
                --command_past_steps "${COMMAND_PAST_STEPS}" \
                --command_future_steps "${COMMAND_FUTURE_STEPS}" \
                --planner_update_interval "${PLANNER_UPDATE_INTERVAL}" \
                --keep_configured_episode_length \
                --disable_tracking_terminations \
                --flow_num_inference_steps "${FLOW_STEPS}" \
                --flow_inference_noise_std "${FLOW_NOISE_STD}" \
                --kit_args=--/app/extensions/fsWatcherEnabled=false
        done
    done
done

run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/summarize_interface_comparison.py \
    --result_root "${OUTPUT_ROOT}"

echo "[INFO] Done. Results under ${OUTPUT_ROOT}"
