#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

PYTHON_CMD_STR="${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_CMD_STR})

FULL_MANIFEST="${FULL_MANIFEST:-${CLUSTER_DATA_DIR:-data}/lafan1/manifests/g1_lafan1_manifest.json}"
FULL_MANIFEST_DIR="$(dirname "${FULL_MANIFEST}")"
SPLIT_OUTPUT_DIR="${SPLIT_OUTPUT_DIR:-${FULL_MANIFEST_DIR}/interface_baselines}"
SPLIT_PREFIX="${SPLIT_PREFIX:-g1_lafan1_heldout_seed${SEED:-0}}"
HELDOUT_NAMES="${HELDOUT_NAMES:-}"
HELDOUT_PATTERNS="${HELDOUT_PATTERNS:-}"
HELDOUT_COUNT="${HELDOUT_COUNT:-0}"
HELDOUT_FRACTION="${HELDOUT_FRACTION:-0.2}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/lafan1_heldout_interface_comparison_seed${SEED}}"
INTERFACES="${INTERFACES-full_body_trajectory ee_trajectory}"
RUN_LATENT_BASELINE="${RUN_LATENT_BASELINE:-0}"
LATENT_MOTION_NAME="${LATENT_MOTION_NAME-}"
LATENT_TRAJECTORY_NAME="${LATENT_TRAJECTORY_NAME-}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH-}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
ALLOW_MISSING_MOTION_FILES="${ALLOW_MISSING_MOTION_FILES:-0}"

needs_split=0
if [[ -z "${TRAIN_MANIFEST:-}" || -z "${EVAL_MANIFEST:-}" ]]; then
    needs_split=1
fi

if [[ "${needs_split}" == "1" && ! -f "${FULL_MANIFEST}" ]]; then
    cat >&2 <<EOF
[ERROR] Full multi-motion manifest not found: ${FULL_MANIFEST}

Prepare the local G1 LAFAN1 data first, for example:

  ./scripts/download_g1_lafan1_data.sh

or, if NPZ files already exist:

  "${PYTHON_CMD[@]}" scripts/write_lafan1_npz_manifest.py \\
      --npz_dir data/lafan1/npz/g1 \\
      --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
EOF
    exit 2
fi

if [[ "${RUN_LATENT_BASELINE}" == "1" && -z "${LATENT_DATASET_PATH}" ]]; then
    cat >&2 <<EOF
[ERROR] Set LATENT_DATASET_PATH when RUN_LATENT_BASELINE=1 for held-out runs.

Do not rely on the Dance102 latent dataset default for a multi-motion held-out
comparison. Use the dataset path that matches LATENT_SKILL_CHECKPOINT and
LATENT_PLANNER_CHECKPOINT.
EOF
    exit 2
fi

mkdir -p "${SPLIT_OUTPUT_DIR}"

if [[ "${needs_split}" == "1" ]]; then
    "${PYTHON_CMD[@]}" experiments/interface_baselines/split_lafan1_manifest.py \
        --manifest "${FULL_MANIFEST}" \
        --heldout_names "${HELDOUT_NAMES}" \
        --heldout_patterns "${HELDOUT_PATTERNS}" \
        --heldout_count "${HELDOUT_COUNT}" \
        --heldout_fraction "${HELDOUT_FRACTION}" \
        --seed "${SEED}" \
        --output_dir "${SPLIT_OUTPUT_DIR}" \
        --prefix "${SPLIT_PREFIX}"
    TRAIN_MANIFEST="${SPLIT_OUTPUT_DIR}/${SPLIT_PREFIX}_train.json"
    EVAL_MANIFEST="${SPLIT_OUTPUT_DIR}/${SPLIT_PREFIX}_heldout.json"
fi

export TRAIN_MANIFEST
export EVAL_MANIFEST
export OUTPUT_ROOT
export SEED
export INTERFACES

if [[ "${RUN_PREFLIGHT}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
    preflight_cmd=(
        "${PYTHON_CMD[@]}" experiments/interface_baselines/preflight_interface_comparison.py
        --train_manifest "${TRAIN_MANIFEST}"
        --eval_manifest "${EVAL_MANIFEST}"
        --interfaces
    )
    for interface in ${INTERFACES}; do
        preflight_cmd+=("${interface}")
    done
    preflight_cmd+=(
        --full_body_checkpoint "${FULL_BODY_TRAJECTORY_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT:-}}"
        --ee_checkpoint "${EE_TRAJECTORY_CHECKPOINT:-${LOW_LEVEL_CHECKPOINT:-}}"
        --model_sizes
    )
    for model_size in ${MODEL_SIZES:-${MODEL_SIZE:-medium}}; do
        preflight_cmd+=("${model_size}")
    done
    preflight_cmd+=(--sample_budgets)
    for budget in ${SAMPLE_BUDGETS:-10000}; do
        preflight_cmd+=("${budget}")
    done
    if [[ "${RUN_LATENT_BASELINE}" == "1" ]]; then
        preflight_cmd+=(
            --run_latent
            --latent_low_level_checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT:-}"
            --latent_skill_checkpoint "${LATENT_SKILL_CHECKPOINT:-}"
            --latent_planner_checkpoint "${LATENT_PLANNER_CHECKPOINT:-}"
            --latent_dataset_path "${LATENT_DATASET_PATH}"
            --require_latent_dataset_path
        )
    else
        preflight_cmd+=(--no-run_latent)
    fi
    if [[ "${ALLOW_MISSING_MOTION_FILES}" == "1" ]]; then
        preflight_cmd+=(--allow_missing_motion_files)
    fi
    "${preflight_cmd[@]}"
fi

experiments/interface_baselines/run_multimotion_heldout_interface_comparison.sh

if [[ "${RUN_LATENT_BASELINE}" == "1" ]]; then
    RUN_LATENT=1 \
    INTERFACES= \
    LATENT_MOTION_NAME="${LATENT_MOTION_NAME}" \
    LATENT_TRAJECTORY_NAME="${LATENT_TRAJECTORY_NAME}" \
    LATENT_DATASET_PATH="${LATENT_DATASET_PATH}" \
    experiments/interface_baselines/run_dance102_fair_interface_comparison.sh
fi
