#!/usr/bin/env bash
set -euo pipefail

# Full LAFAN1 one-motion planner-capacity sweep:
#   prepare shared oracle baselines -> {sizes} x {seeds} capacity points ->
#   per-seed 3-interface aggregation -> across-seed aggregation (2 figures data).
#
# Usage:
#   DRY_RUN=1 experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_sweep.sh
#   experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_sweep.sh
#
# Knobs (env): SIZES, SEEDS, DEVICE, SKIP_ORACLE_PREP, STUDY_ROOT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}/../../.." && pwd))"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/paths.env"

DRY_RUN="${DRY_RUN:-0}"
DEVICE="${DEVICE:-cuda:0}"
: "${PLAIN_PY:=pixi run python}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"
read -r -a SIZES <<<"${SIZES:-tiny small medium large}"
read -r -a SEEDS <<<"${SEEDS:-0 1 2}"
# EE waits on the ee-chunk env adapter; default to latent + FB for now.
read -r -a INTERFACE_LIST <<<"${INTERFACES:-latent_skill full_body_trajectory}"
export INTERFACES="${INTERFACE_LIST[*]}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_planner_capacity_20260723}"
ORACLE_ROOT="${STUDY_ROOT}/oracle_baselines"
SKIP_ORACLE_PREP="${SKIP_ORACLE_PREP:-0}"

export DEVICE STUDY_ROOT ORACLE_ROOT DRY_RUN

oracle_arg() { echo "--oracle $1=${ORACLE_ROOT}/$1/oracle_frame0_700/summary.json"; }

# 1) shared oracle baselines (once)
if [[ "${SKIP_ORACLE_PREP}" != "1" ]]; then
    echo "== oracle baselines =="
    "${SCRIPT_DIR}/prepare_oracle_baselines.sh"
fi

# 2) capacity points
for seed in "${SEEDS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "== capacity point: size=${size} seed=${seed} =="
        MODEL_SIZE="${size}" PLANNER_SEED="${seed}" \
            "${SCRIPT_DIR}/run_capacity_point.sh"
    done
done

[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] skipping aggregation"; exit 0; }

# 3) per-seed aggregation over the active interfaces
oracle_args=()
for i in "${INTERFACE_LIST[@]}"; do
    oracle_args+=(--oracle "${i}=${ORACLE_ROOT}/${i}/oracle_frame0_700/summary.json")
done
seed_inputs=()
for seed in "${SEEDS[@]}"; do
    seed_root="${STUDY_ROOT}/scaling/seed${seed}"
    out="${seed_root}/capacity_summary"
    "${PLAIN_PY_ARR[@]}" experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/aggregate_one_motion_capacity_scaling.py \
        --scaling_root "${seed_root}" --sizes "${SIZES[@]}" \
        "${oracle_args[@]}" \
        --output_dir "${out}" --overwrite
    seed_inputs+=("${out}/capacity_results.json")
done

# 4) across-seed aggregation (final tables for the two figures)
final="${STUDY_ROOT}/capacity_seeds_summary"
input_args=()
for path in "${seed_inputs[@]}"; do input_args+=(--input "${path}"); done
"${PLAIN_PY_ARR[@]}" experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/aggregate_one_motion_capacity_seeds.py \
    "${input_args[@]}" --min_seeds "${#SEEDS[@]}" \
    --survival_target 1.0 --normalized_mpjpe_target 1.5 \
    --output_dir "${final}"

echo "[PASS] Sweep complete. Final aggregate: ${final}"
