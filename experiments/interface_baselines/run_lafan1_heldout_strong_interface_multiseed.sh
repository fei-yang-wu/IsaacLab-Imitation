#!/usr/bin/env bash
set -euo pipefail
set -f

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

PYTHON_CMD_STR="${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
# shellcheck disable=SC2206
PYTHON_CMD=(${PYTHON_CMD_STR})

FULL_MANIFEST="${FULL_MANIFEST:-${CLUSTER_DATA_DIR:-data}/lafan1/manifests/g1_lafan1_manifest.json}"
SEEDS="${SEEDS:-0 1 2}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-logs/interface_baselines/lafan1_heldout_interface_comparison}"
AGGREGATE_OUTPUT_DIR="${AGGREGATE_OUTPUT_DIR:-${OUTPUT_PREFIX}_multiseed}"
RUN_CAPACITY_BACKFILL="${RUN_CAPACITY_BACKFILL:-1}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_SWEEP_ANALYSIS="${RUN_SWEEP_ANALYSIS:-1}"
SELECTED_SAMPLE_COUNT="${SELECTED_SAMPLE_COUNT:-}"
AUDIT_PLANNER_VARIANTS="${AUDIT_PLANNER_VARIANTS:-}"
AUDIT_EXPECTED_SEEDS="${AUDIT_EXPECTED_SEEDS:-${SEEDS}}"
AUDIT_EXPECTED_PRETRAIN_UPDATES="${AUDIT_EXPECTED_PRETRAIN_UPDATES:-}"
MIN_ORACLE_SURVIVAL="${MIN_ORACLE_SURVIVAL:-}"
MIN_ORACLE_SUCCESS_RATE="${MIN_ORACLE_SUCCESS_RATE:-}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_SIZE="${MODEL_SIZE:-medium}"
MODEL_SIZES="${MODEL_SIZES:-${MODEL_SIZE}}"
SAMPLE_BUDGETS="${SAMPLE_BUDGETS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
FLOW_STEPS="${FLOW_STEPS:-16}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS:-4}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-${BATCH_SIZE}}"
FINETUNE_LR="${FINETUNE_LR:-${LR}}"
FINETUNE_WEIGHT_DECAY="${FINETUNE_WEIGHT_DECAY:-${WEIGHT_DECAY}}"
STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS:-0}"
COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS:-0}"
COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS:-25}"
INTERFACES="${INTERFACES-ee_trajectory full_body_trajectory}"
RUN_LATENT_BASELINE="${RUN_LATENT_BASELINE:-0}"
LATENT_MOTION_NAME="${LATENT_MOTION_NAME-}"
LATENT_TRAJECTORY_NAME="${LATENT_TRAJECTORY_NAME-}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH-}"

if [[ -z "${EXPECTED_INTERFACES:-}" ]]; then
    EXPECTED_INTERFACES=""
    if [[ "${RUN_LATENT_BASELINE}" == "1" ]]; then
        EXPECTED_INTERFACES="latent_skill"
    fi
    for interface in ${INTERFACES}; do
        EXPECTED_INTERFACES="${EXPECTED_INTERFACES} ${interface}"
    done
fi

export MODEL_SIZE
export MODEL_SIZES
export SAMPLE_BUDGETS
export BATCH_SIZE
export MICRO_BATCH_SIZE
export PRETRAIN_UPDATES
export FINETUNE_UPDATES
export LR
export WEIGHT_DECAY
export FLOW_STEPS
export FLOW_NOISE_STD
export TRAIN_ENDPOINT_STEPS
export FINETUNE_BATCH_SIZE
export FINETUNE_LR
export FINETUNE_WEIGHT_DECAY
export STATE_HISTORY_STEPS
export COMMAND_PAST_STEPS
export COMMAND_FUTURE_STEPS
export INTERFACES
export DRY_RUN
export FULL_MANIFEST

run_cmd() {
    printf '[CMD]'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "$@"
}

word_count() {
    local count=0
    local token
    for token in $1; do
        count=$((count + 1))
    done
    printf '%s' "${count}"
}

budget_label() {
    local budget="$1"
    if [[ "${budget}" == "0" ]]; then
        printf 'all'
    else
        printf '%s' "${budget}"
    fi
}

needs_split=0
if [[ -z "${TRAIN_MANIFEST:-}" || -z "${EVAL_MANIFEST:-}" ]]; then
    needs_split=1
fi

if [[ "${DRY_RUN}" != "1" && "${needs_split}" == "1" && ! -f "${FULL_MANIFEST}" ]]; then
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

seed_roots=()
for seed in ${SEEDS}; do
    seed_roots+=("${OUTPUT_PREFIX}_seed${seed}")
done

provenance_cmd=(
    "${PYTHON_CMD[@]}" experiments/interface_baselines/write_interface_run_provenance.py
    --label lafan1-heldout-strong-multiseed
    --output_json "${AGGREGATE_OUTPUT_DIR}/interface_comparison_run_provenance.json"
)
for root in "${seed_roots[@]}"; do
    provenance_cmd+=(--result_root "${root}")
done
run_cmd "${provenance_cmd[@]}"

for seed in ${SEEDS}; do
    seed_root="${OUTPUT_PREFIX}_seed${seed}"
    run_cmd env \
        SEED="${seed}" \
        OUTPUT_ROOT="${seed_root}" \
        MODEL_SIZE="${MODEL_SIZE}" \
        MODEL_SIZES="${MODEL_SIZES}" \
        SAMPLE_BUDGETS="${SAMPLE_BUDGETS}" \
        PRETRAIN_UPDATES="${PRETRAIN_UPDATES}" \
        WEIGHT_DECAY="${WEIGHT_DECAY}" \
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
        TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS}" \
        STATE_HISTORY_STEPS="${STATE_HISTORY_STEPS}" \
        COMMAND_PAST_STEPS="${COMMAND_PAST_STEPS}" \
        COMMAND_FUTURE_STEPS="${COMMAND_FUTURE_STEPS}" \
        RUN_LATENT_BASELINE="${RUN_LATENT_BASELINE}" \
        LATENT_MOTION_NAME="${LATENT_MOTION_NAME}" \
        LATENT_TRAJECTORY_NAME="${LATENT_TRAJECTORY_NAME}" \
        LATENT_DATASET_PATH="${LATENT_DATASET_PATH}" \
        experiments/interface_baselines/run_lafan1_heldout_strong_interface_comparison.sh
done

if [[ "${RUN_CAPACITY_BACKFILL}" == "1" ]]; then
    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/backfill_planner_capacity_metadata.py \
        --result_roots "${seed_roots[@]}"
fi

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
    run_cmd "${PYTHON_CMD[@]}" experiments/interface_baselines/aggregate_interface_comparison_seeds.py \
        --output_dir "${AGGREGATE_OUTPUT_DIR}" \
        --refresh \
        --result_roots "${seed_roots[@]}"
fi

if [[ "${RUN_SWEEP_ANALYSIS}" == "1" ]]; then
    sweep_cmd=(
        "${PYTHON_CMD[@]}" experiments/interface_baselines/analyze_interface_sweep.py
        --aggregate_dir "${AGGREGATE_OUTPUT_DIR}"
    )
    if [[ -n "${EXPECTED_INTERFACES}" ]]; then
        sweep_cmd+=(--expected_selected_interfaces)
        for interface in ${EXPECTED_INTERFACES}; do
            sweep_cmd+=("${interface}")
        done
    fi
    if [[ -n "${SELECTED_SAMPLE_COUNT}" ]]; then
        sweep_cmd+=(--selected_sample_count "${SELECTED_SAMPLE_COUNT}")
    fi
    run_cmd "${sweep_cmd[@]}"
fi

if [[ "${RUN_AUDIT}" == "1" ]]; then
    audit_cmd=(
        "${PYTHON_CMD[@]}" experiments/interface_baselines/audit_interface_comparison.py
        --aggregate_dir "${AGGREGATE_OUTPUT_DIR}"
        --output_json "${AGGREGATE_OUTPUT_DIR}/interface_comparison_audit.json"
        --output_md "${AGGREGATE_OUTPUT_DIR}/interface_comparison_audit.md"
        --require_provenance
        --expected_planner_num_updates "${FINETUNE_UPDATES}"
        --expected_planner_finetune_num_updates "${FINETUNE_UPDATES}"
        --expected_planner_batch_size "${BATCH_SIZE}"
        --expected_planner_lr "${LR}"
        --expected_planner_weight_decay "${WEIGHT_DECAY}"
        --expected_planner_flow_num_inference_steps "${FLOW_STEPS}"
        --expected_planner_flow_inference_noise_std "${FLOW_NOISE_STD}"
        --expected_hand_designed_planner_state_history_steps "${STATE_HISTORY_STEPS}"
        --expected_hand_designed_planner_command_past_steps "${COMMAND_PAST_STEPS}"
        --expected_hand_designed_planner_command_future_steps "${COMMAND_FUTURE_STEPS}"
    )
    if [[ -n "${AUDIT_EXPECTED_PRETRAIN_UPDATES}" ]]; then
        audit_cmd+=(
            --expected_planner_pretrain_num_updates "${AUDIT_EXPECTED_PRETRAIN_UPDATES}"
        )
    fi
    audit_cmd+=(--expected_interfaces)
    for interface in ${EXPECTED_INTERFACES}; do
        audit_cmd+=("${interface}")
    done
    audit_cmd+=(--expected_seeds)
    for seed in ${AUDIT_EXPECTED_SEEDS}; do
        audit_cmd+=("${seed}")
    done
    if [[ "${RUN_SWEEP_ANALYSIS}" == "1" ]]; then
        audit_cmd+=(--require_selected --use_selected_variants)
    fi
    if [[ -n "${MIN_ORACLE_SURVIVAL}" ]]; then
        audit_cmd+=(--min_oracle_survival "${MIN_ORACLE_SURVIVAL}")
    fi
    if [[ -n "${MIN_ORACLE_SUCCESS_RATE}" ]]; then
        audit_cmd+=(--min_oracle_success_rate "${MIN_ORACLE_SUCCESS_RATE}")
    fi

    model_count="$(word_count "${MODEL_SIZES}")"
    budget_count="$(word_count "${SAMPLE_BUDGETS}")"
    if [[ "${model_count}" == "1" && "${budget_count}" == "1" ]]; then
        model_size="${MODEL_SIZES}"
        sample_budget="${SAMPLE_BUDGETS}"
        sample_budget_label="$(budget_label "${sample_budget}")"
        if [[ "${sample_budget}" != "all" && "${sample_budget}" != "0" ]]; then
            audit_cmd+=(--expected_sample_count "${sample_budget}")
        fi
        if [[ " ${EXPECTED_INTERFACES} " == *" ee_trajectory "* ]]; then
            audit_cmd+=(
                --planner_variant "ee_trajectory=chunked_transformer_${model_size}_${sample_budget_label}"
            )
        fi
        if [[ " ${EXPECTED_INTERFACES} " == *" full_body_trajectory "* ]]; then
            audit_cmd+=(
                --planner_variant "full_body_trajectory=chunked_transformer_${model_size}_${sample_budget_label}"
            )
        fi
    elif [[ -n "${AUDIT_PLANNER_VARIANTS}" ]]; then
        for variant_spec in ${AUDIT_PLANNER_VARIANTS}; do
            audit_cmd+=(--planner_variant "${variant_spec}")
        done
    elif [[ "${RUN_SWEEP_ANALYSIS}" == "1" ]]; then
        echo "[INFO] Auditing variants selected by interface_sweep_selected.csv." >&2
    else
        echo "[INFO] Skipping audit because MODEL_SIZES/SAMPLE_BUDGETS define multiple variants." >&2
        echo "[INFO] Set RUN_SWEEP_ANALYSIS=1 or AUDIT_PLANNER_VARIANTS='interface=variant ...' to audit a selected variant." >&2
        RUN_AUDIT=0
    fi

    if [[ "${RUN_AUDIT}" == "1" ]]; then
        run_cmd "${audit_cmd[@]}"
    fi
fi

echo "[INFO] LAFAN1 held-out multiseed comparison complete."
echo "[INFO] Per-seed roots: ${seed_roots[*]}"
echo "[INFO] Aggregate root: ${AGGREGATE_OUTPUT_DIR}"
