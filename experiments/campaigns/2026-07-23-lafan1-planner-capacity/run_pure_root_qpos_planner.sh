#!/usr/bin/env bash
set -euo pipefail

# Train the direct root_qpos-tracker planner from the already-collected,
# provenance-bound 100-trajectory oracle dataset. This is the content-matched
# third route:
#   root_qpos planner -> frozen root_qpos tracker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ ! -x "${REPO_ROOT}/docker/cluster/cluster_interface.sh" ]]; do
    if [[ "${REPO_ROOT}" == / ]]; then
        echo "[ERROR] Could not locate repository root above ${SCRIPT_DIR}." >&2
        exit 2
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
cd "${REPO_ROOT}"

SAMPLES_DIR="${SAMPLES_DIR:-logs/interface_baselines/lafan1_interface_capacity_controlled/oracle_baselines/root_qpos/oracle_demonstrations/rollout_training_samples}"
EXPECTED_SAMPLE_SHA256="${EXPECTED_SAMPLE_SHA256:-1374eb44647098e1e0a1da21e89ad1edab02fd58195a434a2dc1a6b3b809a3b0}"
STUDY_ROOT="${STUDY_ROOT:-logs/interface_baselines/lafan1_enc380_route_capacity_5b_oracle100_progressive_b1024_20260730}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
DEVICE="${DEVICE:-cuda:0}"
: "${PLAIN_PY:=pixi run python}"
read -r -a PLAIN_PY_ARR <<<"${PLAIN_PY}"

case "${MODEL_SIZE}" in
    tiny)
        default_updates=10000
        default_micro_batch=1024
        ;;
    small)
        default_updates=20000
        default_micro_batch=512
        ;;
    medium)
        default_updates=30000
        default_micro_batch=256
        ;;
    large)
        default_updates=50000
        default_micro_batch=128
        ;;
    *)
        echo "[ERROR] Unknown model size: ${MODEL_SIZE}" >&2
        exit 2
        ;;
esac
NUM_UPDATES="${NUM_UPDATES:-${default_updates}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${default_micro_batch}}"
OUTPUT_DIR="${OUTPUT_DIR:-${STUDY_ROOT}/pure_root_qpos_tracker/${MODEL_SIZE}/seed${SEED}/planner_oracle_u${NUM_UPDATES}_b${BATCH_SIZE}}"

sample="${SAMPLES_DIR}/sample_step_000000.pt"
[[ -f "${sample}" ]] || {
    echo "[ERROR] Missing direct root_qpos oracle samples: ${sample}" >&2
    exit 2
}
actual_sha="$(sha256sum "${sample}" | awk '{print $1}')"
[[ "${actual_sha}" == "${EXPECTED_SAMPLE_SHA256}" ]] || {
    echo "[ERROR] Direct root_qpos sample SHA mismatch: ${actual_sha}" >&2
    exit 2
}

marker="${OUTPUT_DIR}/checkpoints/best.pt"
if [[ -f "${marker}" ]]; then
    echo "[SKIP] Existing direct root_qpos planner: ${marker}"
    exit 0
fi

"${PLAIN_PY_ARR[@]}" \
    experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/train_chunked_transformer_planner.py \
    --samples_dir "${SAMPLES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --interface root_qpos \
    --planner_family flow \
    --state_key planner_state \
    --training_stage oracle \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --model_size "${MODEL_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --num_updates "${NUM_UPDATES}" \
    --max_samples 0 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --flow_num_inference_steps 16 \
    --endpoint_num_inference_steps 4 \
    --flow_inference_noise_std 0.0

[[ -f "${marker}" ]] || {
    echo "[ERROR] Planner training completed without ${marker}." >&2
    exit 2
}
echo "[PASS] Direct root_qpos planner: $(realpath "${marker}")"
