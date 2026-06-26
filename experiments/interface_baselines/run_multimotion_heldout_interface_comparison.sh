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

: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to the multi-motion training manifest.}"
: "${EVAL_MANIFEST:?Set EVAL_MANIFEST to the held-out evaluation manifest.}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/multimotion_heldout_interface_comparison}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
MODEL_SIZES="${MODEL_SIZES:-${MODEL_SIZE}}"
SAMPLE_BUDGETS="${SAMPLE_BUDGETS:-10000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS:-4}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-0}"
COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS:-0}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"

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

COLLECT_STEPS="${COLLECT_STEPS:-$(max_numeric_sample_requirement "${SAMPLE_BUDGETS}" 10000 "${SELECTED_SAMPLE_COUNT:-}")}"

export OUTPUT_ROOT
export MODEL_SIZE
export MODEL_SIZES
export SAMPLE_BUDGETS
export MICRO_BATCH_SIZE
export TRAIN_ENDPOINT_STEPS
export PRETRAIN_UPDATES
export FINETUNE_UPDATES
export EVAL_STEPS
export COLLECT_STEPS
export STATE_HISTORY_STEPS
export COMMAND_PAST_STEPS
export COMMAND_FUTURE_STEPS
export TRAIN_MANIFEST
export EVAL_MANIFEST

experiments/interface_baselines/run_dance102_strong_interface_comparison.sh
