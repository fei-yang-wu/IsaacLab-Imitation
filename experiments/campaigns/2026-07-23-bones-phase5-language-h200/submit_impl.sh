#!/usr/bin/env bash
set -euo pipefail

# Thin campaign front door. Reusable collection/training/evaluation code stays
# in experiments/interface_baselines/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" \
    && "${CONFIRM_SUBMIT:-}" != "I_UNDERSTAND_UNQUALIFIED_PHASE5" ]]; then
    echo "[ERROR] This is an unqualified latent-only pilot." >&2
    echo "[ERROR] Re-render with DRY_RUN=1, then set CONFIRM_SUBMIT=I_UNDERSTAND_UNQUALIFIED_PHASE5 for a real submission." >&2
    exit 2
fi

GOAL_NAMES="${GOAL_NAMES:-Neutral_stoop_down_001_A057 avoid_bump_let_go_R_003_A460 axe_cutting_tree_horizontal_R_004_A355 big_heavy_two_hands_front_high_to_front_high_R_001_A524 big_light_two_hands_pick_up_front_medium_R_001_A509 body_check_001_A180 burning_loop_R_001_A528 casual_greeting_R_001_A428 cellphone_typing_sequence_one_hand_idle_R_001_A423 cough_tuberculosis_R_001_A500}"
GOAL_LIMIT="${GOAL_LIMIT:-10}"
goal_count="$(wc -w <<< "${GOAL_NAMES}")"
if [[ "${goal_count}" != "${GOAL_LIMIT}" ]]; then
    echo "[ERROR] GOAL_NAMES contains ${goal_count} names but GOAL_LIMIT=${GOAL_LIMIT}." >&2
    exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_h200_language_preliminary_seed0_20260723}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"

# Keep local copies for immutable hash/binding checks, while the container uses
# the persistent ICE artifacts and archive sync does not transfer logs/models.
LOCAL_LATENT_CHECKPOINT="${LOCAL_LATENT_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt}"
LOCAL_SKILL_CHECKPOINT="${LOCAL_SKILL_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt}"
ICE_LATENT_CHECKPOINT="${ICE_LATENT_CHECKPOINT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab/${LOCAL_LATENT_CHECKPOINT}}"
ICE_SKILL_CHECKPOINT="${ICE_SKILL_CHECKPOINT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab/${LOCAL_SKILL_CHECKPOINT}}"

export MODE=bones-seed-multigoal-language
export INTERFACES="latent_skill"
export ALLOW_UNQUALIFIED_PRELIMINARY=1
export LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-${ICE_LATENT_CHECKPOINT}}"
export LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-${ICE_SKILL_CHECKPOINT}}"
# The shared runner requires this argument even for latent-only mode. It is not
# loaded or used as a comparison row in this campaign.
export VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-07-15_00-32-35/models/model_step_1000046592.pt}"
export MANIFEST="${MANIFEST:-/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json}"
export LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-/data/bones_seed_phase5/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"
export PREPARATION_RECORD="${PREPARATION_RECORD:-/data/bones_seed_phase5/bones_seed_100/preparation/preparation.json}"
export LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed0}"
export VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed0}"

EXPECTED_LATENT_SHA256="6765a324a840b33a84f9a0b5a817c60303979bbec7a36ebc31242086d61d1572"
EXPECTED_SKILL_SHA256="562e4f9d0cebcdeb0bdddf6fb77ea8d0b488a8e576442b7106b54a13d6eceadc"
latent_file="${LOCAL_LATENT_CHECKPOINT}"
skill_file="${LOCAL_SKILL_CHECKPOINT}"
if [[ "${latent_file}" != /* ]]; then latent_file="${REPO_ROOT}/${latent_file}"; fi
if [[ "${skill_file}" != /* ]]; then skill_file="${REPO_ROOT}/${skill_file}"; fi
for artifact in "${latent_file}" "${skill_file}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "[ERROR] Required local checkpoint is missing: ${artifact}" >&2
        exit 2
    fi
done
actual_latent_sha256="$(sha256sum "${latent_file}" | awk '{print $1}')"
actual_skill_sha256="$(sha256sum "${skill_file}" | awk '{print $1}')"
if [[ "${actual_latent_sha256}" != "${EXPECTED_LATENT_SHA256}" ]]; then
    echo "[ERROR] H200 latent checkpoint hash changed: ${actual_latent_sha256}" >&2
    exit 2
fi
if [[ "${actual_skill_sha256}" != "${EXPECTED_SKILL_SHA256}" ]]; then
    echo "[ERROR] H10 skill checkpoint hash changed: ${actual_skill_sha256}" >&2
    exit 2
fi
BINDING_RECORD="${REPO_ROOT}/logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/latent_skill_binding.json"
if [[ ! -f "${BINDING_RECORD}" ]] || ! python3 -c 'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("passed") is True else 1)' "${BINDING_RECORD}"; then
    echo "[ERROR] The H200 latent/skill encoder binding record is missing or failed: ${BINDING_RECORD}" >&2
    exit 2
fi
export OUTPUT_ROOT
export GOAL_NAMES
export GOAL_LIMIT
export SEED="${SEED:-0}"
export DEMO_ROWS_PER_GOAL="${DEMO_ROWS_PER_GOAL:-150}"
export ROLLOUT_ROWS_PER_GOAL="${ROLLOUT_ROWS_PER_GOAL:-150}"
export ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-10}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export MODEL_SIZE="${MODEL_SIZE:-medium}"
export PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
export FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
export LR="${LR:-1.0e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
export FLOW_STEPS="${FLOW_STEPS:-16}"
export TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS:-4}"
export FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
export DRY_RUN

export CLUSTER_PROFILE="${CLUSTER_PROFILE:-base}"
export CLUSTER_CONFIG="${CLUSTER_CONFIG:-ice_runtime}"
export CLUSTER_LOGIN="${CLUSTER_LOGIN:-ice}"
export CLUSTER_SLURM_ACCOUNT="${CLUSTER_SLURM_ACCOUNT:-cse}"
export CLUSTER_SLURM_PARTITION="${CLUSTER_SLURM_PARTITION:-ice-gpu}"
export CLUSTER_SLURM_QOS="${CLUSTER_SLURM_QOS:-coe-ice}"
export CLUSTER_SLURM_GPU_GRES="${CLUSTER_SLURM_GPU_GRES:-gpu:h100:1}"
export CLUSTER_SLURM_CPUS_PER_TASK="${CLUSTER_SLURM_CPUS_PER_TASK:-16}"
export CLUSTER_SLURM_MEM="${CLUSTER_SLURM_MEM:-96G}"
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-bones-lang-h200}"
export CLUSTER_SLURM_STAGE_SUBMIT_SCRIPT="${CLUSTER_SLURM_STAGE_SUBMIT_SCRIPT:-pace}"
export CLUSTER_AUTO_SETUP_G1_DATA=0
export CLUSTER_ARCHIVE_SYNC=1
export CLUSTER_GIT_SYNC_FIRST=0
export CLUSTER_INCREMENTAL_SYNC=0
export CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=0
export CLUSTER_EXTRA_RSYNC_EXCLUDES="${CLUSTER_EXTRA_RSYNC_EXCLUDES:-data/ .tmp/ logs/ RLOpt/ ImitationLearningTools/}"
export CLUSTER_SKIP_CACHE_COPY=1
export CLUSTER_USE_SHARED_SIF=1
export CLUSTER_OVERLAY_SIZE_MB=8192
export CLUSTER_SLURM_SUBMIT_SCRIPT=bones_pipeline
export CLUSTER_SLURM_PIPELINE_ARRAY="0-$((GOAL_LIMIT - 1))%${MAX_PARALLEL_GOALS:-4}"
export CLUSTER_SLURM_SUBMISSION_RECORD_ROOT="${REMOTE_PROJECT_ROOT}/${OUTPUT_ROOT}"
export CLUSTER_SLURM_PREPARE_TIME_LIMIT="${CLUSTER_SLURM_PREPARE_TIME_LIMIT:-15:00:00}"
export CLUSTER_SLURM_ROLLOUT_TIME_LIMIT="${CLUSTER_SLURM_ROLLOUT_TIME_LIMIT:-12:00:00}"
export CLUSTER_SLURM_FINETUNE_TIME_LIMIT="${CLUSTER_SLURM_FINETUNE_TIME_LIMIT:-12:00:00}"
export CLUSTER_SLURM_FINAL_EVAL_TIME_LIMIT="${CLUSTER_SLURM_FINAL_EVAL_TIME_LIMIT:-4:00:00}"
export CLUSTER_SLURM_SUMMARIZE_TIME_LIMIT="${CLUSTER_SLURM_SUMMARIZE_TIME_LIMIT:-1:00:00}"
export AUTO_SYNC_LOCAL_CHECKPOINTS=0
export AUTO_SYNC_EXTRA_AGGREGATE_ROOTS=0
export REFRESH_DATASETS=0
export SKIP_PRETRAINED_CLOSED_LOOP=0
export CONTINUE_ON_ERROR=0

exec experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_cluster_interface_baselines.sh
