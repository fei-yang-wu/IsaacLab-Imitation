#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
PRESET="${PRESET:-ridge}"
TARGET_FRAMES_PER_RUN="${TARGET_FRAMES_PER_RUN:-10000000}"
AGGREGATE_FRAME_CAP="${AGGREGATE_FRAME_CAP:-51000000}"
SEED="${SEED:-0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/training_scale/${TIMESTAMP}_lafan1_latent_local_${PRESET}_seed${SEED}}"

MANIFEST="${MANIFEST:-${REPO_ROOT}/data/lafan1/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/tmp/iltools_g1_lafan1_tracking_corrected_8029acbce33a}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-${REPO_ROOT}/logs/interface_baselines/lafan1_corrected_8e95d557_diffsr_5b_seed0/latent/base_pipeline/skill_encoder_h10_z256/checkpoints/latest.pt}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-d972c37c41dadbb68c30fc456a9dc9c1bd6d30ed0b7aa9d34b1797472c945db8}"
EXPECTED_SKILL_SHA256="${EXPECTED_SKILL_SHA256:-c9cf9691823043d63cafababfb3b3ce2182215f905b50540801ffc210e040728}"
EXPECTED_DATASET_FINGERPRINT="${EXPECTED_DATASET_FINGERPRINT:-ff301211535cc51a6a560d07d9b19f7b09ed14fdf60c6f7025ef78279da52826}"

EXTRA_RUNNER_ARGS_STR="${EXTRA_RUNNER_ARGS:-}"
read -r -a EXTRA_RUNNER_ARGS_LIST <<< "${EXTRA_RUNNER_ARGS_STR}"

cmd=(
    pixi run -e isaaclab python
    experiments/training_scale/run_latent_scale_benchmark.py
    --preset "${PRESET}"
    --manifest "${MANIFEST}"
    --dataset-path "${DATASET_PATH}"
    --skill-checkpoint "${SKILL_CHECKPOINT}"
    --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}"
    --expected-skill-sha256 "${EXPECTED_SKILL_SHA256}"
    --expected-dataset-fingerprint "${EXPECTED_DATASET_FINGERPRINT}"
    --output-root "${OUTPUT_ROOT}"
    --run-label "lafan1_latent_local_${PRESET}"
    --target-frames-per-run "${TARGET_FRAMES_PER_RUN}"
    --aggregate-frame-cap "${AGGREGATE_FRAME_CAP}"
    --target-return 7.5
    --sustain-points 3
    --log-interval-frames 5000000
    --seed "${SEED}"
)
if [[ -n "${EXTRA_RUNNER_ARGS_STR}" ]]; then
    cmd+=("${EXTRA_RUNNER_ARGS_LIST[@]}")
fi
case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON)
        cmd+=(--dry-run)
        ;;
esac

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
