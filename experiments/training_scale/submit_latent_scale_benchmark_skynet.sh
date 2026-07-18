#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
PRESET="${PRESET:-a40-screen}"
TARGET_FRAMES_PER_RUN="${TARGET_FRAMES_PER_RUN:-75000000}"
AGGREGATE_FRAME_CAP="${AGGREGATE_FRAME_CAP:-2000000000}"
SEED="${SEED:-0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/training_scale/${TIMESTAMP}_lafan1_latent_a40_${PRESET}_seed${SEED}}"

MANIFEST="${MANIFEST:-/data/lafan1_corrected_8e95d557/manifests/g1_lafan1_manifest.json}"
DATASET_PATH="${DATASET_PATH:-/data/lafan1_corrected_8e95d557/g1_hl_diffsr}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-/workspace/isaaclab/project/logs/interface_baselines/lafan1_corrected_8e95d557_diffsr_5b_seed0/latent/base_pipeline/skill_encoder_h10_z256/checkpoints/latest.pt}"
EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256:-218d5d41b5e6a47e272c07babb84b8c51c9af54e5576ecb8322fb66528d366d8}"
EXPECTED_SKILL_SHA256="${EXPECTED_SKILL_SHA256:-c9cf9691823043d63cafababfb3b3ce2182215f905b50540801ffc210e040728}"
EXPECTED_DATASET_FINGERPRINT="${EXPECTED_DATASET_FINGERPRINT:-317bacaa8d3541df9a6f450e41f9fccb2ffa9e37f383a54f9ff6b1beebe0b3b4}"

EXTRA_RUNNER_ARGS_STR="${EXTRA_RUNNER_ARGS:-}"
read -r -a EXTRA_RUNNER_ARGS_LIST <<< "${EXTRA_RUNNER_ARGS_STR}"

runner_args=(
    --preset "${PRESET}"
    --manifest "${MANIFEST}"
    --dataset-path "${DATASET_PATH}"
    --skill-checkpoint "${SKILL_CHECKPOINT}"
    --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}"
    --expected-skill-sha256 "${EXPECTED_SKILL_SHA256}"
    --expected-dataset-fingerprint "${EXPECTED_DATASET_FINGERPRINT}"
    --output-root "${OUTPUT_ROOT}"
    --run-label "lafan1_latent_a40_${PRESET}"
    --target-frames-per-run "${TARGET_FRAMES_PER_RUN}"
    --aggregate-frame-cap "${AGGREGATE_FRAME_CAP}"
    --target-return 7.5
    --sustain-points 3
    --log-interval-frames 5000000
    --seed "${SEED}"
    --vram-limit-mib 42000
)
if [[ -n "${EXTRA_RUNNER_ARGS_STR}" ]]; then
    runner_args+=("${EXTRA_RUNNER_ARGS_LIST[@]}")
fi

cmd=(
    env
    CLUSTER_AUTO_SETUP_G1_DATA=0
    CLUSTER_APPEND_DEFAULT_G1_MANIFEST=0
    CLUSTER_ARCHIVE_SYNC=1
    CLUSTER_GIT_SYNC_FIRST=0
    CLUSTER_INCREMENTAL_SYNC=0
    CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=0
    "CLUSTER_EXTRA_RSYNC_EXCLUDES=data/ .tmp/ RLOpt/ ImitationLearningTools/"
    CLUSTER_SKIP_CACHE_COPY=1
    CLUSTER_USE_SHARED_SIF=1
    CLUSTER_OVERLAY_SIZE_MB=16384
    CLUSTER_SLURM_TIME_LIMIT=2-00:00:00
    CLUSTER_SLURM_QOS=long
    CLUSTER_SLURM_GPU_SPEC=a40:1
    CLUSTER_SLURM_JOB_NAME=latent-a40-scale
    CLUSTER_PYTHON_EXECUTABLE=experiments/training_scale/run_latent_scale_benchmark.py
    ./docker/cluster/cluster_interface.sh job
    "${runner_args[@]}"
)

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
case "${DRY_RUN}" in
    1|true|TRUE|yes|YES|on|ON)
        echo "[INFO] DRY_RUN=${DRY_RUN}; not contacting Skynet."
        exit 0
        ;;
esac
"${cmd[@]}"
