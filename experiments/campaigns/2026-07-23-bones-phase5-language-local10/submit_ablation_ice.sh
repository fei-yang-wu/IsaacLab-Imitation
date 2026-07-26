#!/usr/bin/env bash
set -euo pipefail

# Parameterized ICE submission for the data-composition ablation. ARM selects
# the finetune-data composition (all at the LARGE planner, 1000 rows/motion,
# ten goals, against the ICE-staged full 100-motion Phase-5 tree restricted via
# GOAL_NAMES; latent-only, preliminary/unqualified):
#
#   ARM=A  DAgger + demo             : demo(oracle) + planner-driven rollout   [default pipeline]
#   ARM=B  oracle + demo             : demo + oracle-driven rollout            [ROLLOUT_COMMAND_SOURCE=oracle]
#   ARM=C  DAgger + oracle + demo    : demo + planner-rollout + oracle-rollout [EXTRA_FINETUNE_SAMPLES=arm B oracle pool]
#
# Arm C reuses arm B's oracle-rollout pool via EXTRA_FINETUNE_SAMPLES, so submit
# C only after B's rollout+finetune merge exists (or pass CLUSTER_SLURM_DEPENDENCY).
#
# Dry-run (default):
#   ARM=B experiments/campaigns/2026-07-23-bones-phase5-language-local10/submit_ablation_ice.sh
# Real submit:
#   ARM=B DRY_RUN=0 CONFIRM_SUBMIT=I_UNDERSTAND_UNQUALIFIED_PHASE5 \
#     experiments/campaigns/2026-07-23-bones-phase5-language-local10/submit_ablation_ice.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${ARM:?Set ARM to A, B, or C}"
case "${ARM}" in
    A) ARM_SLUG="A_dagger_demo";        ROLLOUT_COMMAND_SOURCE="planner" ;;
    B) ARM_SLUG="B_oracle_demo";        ROLLOUT_COMMAND_SOURCE="oracle"  ;;
    C) ARM_SLUG="C_dagger_oracle_demo"; ROLLOUT_COMMAND_SOURCE="planner" ;;
    *) echo "[ERROR] ARM must be A, B, or C (got ${ARM})." >&2; exit 2 ;;
esac
export ROLLOUT_COMMAND_SOURCE
# Container-visible default location of arm B's oracle-rollout pool, used only
# by arm C. Override EXTRA_FINETUNE_SAMPLES to point elsewhere.
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/isaaclab/project}"
if [[ "${ARM}" == "C" ]]; then
    export EXTRA_FINETUNE_SAMPLES="${EXTRA_FINETUNE_SAMPLES:-${CONTAINER_PROJECT_ROOT}/logs/interface_baselines/bones_seed_phase5_ice_ablation_B_oracle_demo_seed0/latent_skill/planner_rollout_samples}"
fi

DRY_RUN="${DRY_RUN:-1}"
if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" \
    && "${CONFIRM_SUBMIT:-}" != "I_UNDERSTAND_UNQUALIFIED_PHASE5" ]]; then
    echo "[ERROR] Unqualified preliminary ablation. Re-render with DRY_RUN=1, then" >&2
    echo "[ERROR] set CONFIRM_SUBMIT=I_UNDERSTAND_UNQUALIFIED_PHASE5 for a real submission." >&2
    exit 2
fi

GOAL_NAMES="${GOAL_NAMES:-Neutral_stoop_down_001_A057 avoid_bump_let_go_R_003_A460 axe_cutting_tree_horizontal_R_004_A355 big_heavy_two_hands_front_high_to_front_high_R_001_A524 big_light_two_hands_pick_up_front_medium_R_001_A509 body_check_001_A180 burning_loop_R_001_A528 casual_greeting_R_001_A428 cellphone_typing_sequence_one_hand_idle_R_001_A423 cough_tuberculosis_R_001_A500}"
GOAL_LIMIT="${GOAL_LIMIT:-10}"
if [[ "$(wc -w <<< "${GOAL_NAMES}")" != "${GOAL_LIMIT}" ]]; then
    echo "[ERROR] GOAL_NAMES count != GOAL_LIMIT (${GOAL_LIMIT})." >&2; exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_phase5_ice_ablation_${ARM_SLUG}_seed0}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/hice1/fwu91/scratch/Research/IsaacLab/isaaclab}"

# Local copies for immutable hash/binding checks (archive sync excludes logs/).
LOCAL_LATENT_CHECKPOINT="${LOCAL_LATENT_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt}"
LOCAL_SKILL_CHECKPOINT="${LOCAL_SKILL_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt}"
# Container-visible ICE paths. The host logs tree (CLUSTER_PROJECT_LOGS_DIR,
# i.e. ${CLUSTER_ISAACLAB_DIR}/logs where I staged the checkpoints) is bind
# mounted at /workspace/isaaclab/project/logs, so the container sees the
# repo-relative logs/ path under that project root — NOT the host home path.
CONTAINER_PROJECT_ROOT="${CONTAINER_PROJECT_ROOT:-/workspace/isaaclab/project}"
ICE_LATENT_CHECKPOINT="${ICE_LATENT_CHECKPOINT:-${CONTAINER_PROJECT_ROOT}/${LOCAL_LATENT_CHECKPOINT}}"
ICE_SKILL_CHECKPOINT="${ICE_SKILL_CHECKPOINT:-${CONTAINER_PROJECT_ROOT}/${LOCAL_SKILL_CHECKPOINT}}"

export MODE=bones-seed-multigoal-language
export INTERFACES="latent_skill"
export ALLOW_UNQUALIFIED_PRELIMINARY=1
export LATENT_LOW_LEVEL_CHECKPOINT="${ICE_LATENT_CHECKPOINT}"
export LATENT_SKILL_CHECKPOINT="${ICE_SKILL_CHECKPOINT}"
# Required by the runner even latent-only; never loaded here.
export VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-/workspace/isaaclab/project/logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-07-15_00-32-35/models/model_step_1000046592.pt}"
# ICE-staged full 100-motion tree (restricted to GOAL_NAMES).
export MANIFEST="${MANIFEST:-/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json}"
export LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-/data/bones_seed_phase5/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"
export PREPARATION_RECORD="${PREPARATION_RECORD:-/data/bones_seed_phase5/bones_seed_100/preparation/preparation.json}"
export LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed0}"
export VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed0}"

EXPECTED_LATENT_SHA256="6765a324a840b33a84f9a0b5a817c60303979bbec7a36ebc31242086d61d1572"
EXPECTED_SKILL_SHA256="562e4f9d0cebcdeb0bdddf6fb77ea8d0b488a8e576442b7106b54a13d6eceadc"
lf="${LOCAL_LATENT_CHECKPOINT}"; sf="${LOCAL_SKILL_CHECKPOINT}"
[[ "${lf}" = /* ]] || lf="${REPO_ROOT}/${lf}"
[[ "${sf}" = /* ]] || sf="${REPO_ROOT}/${sf}"
for a in "${lf}" "${sf}"; do
    [[ -f "${a}" ]] || { echo "[ERROR] Missing local checkpoint: ${a}" >&2; exit 2; }
done
[[ "$(sha256sum "${lf}" | awk '{print $1}')" == "${EXPECTED_LATENT_SHA256}" ]] || { echo "[ERROR] latent ckpt hash changed" >&2; exit 2; }
[[ "$(sha256sum "${sf}" | awk '{print $1}')" == "${EXPECTED_SKILL_SHA256}" ]] || { echo "[ERROR] skill ckpt hash changed" >&2; exit 2; }
BINDING="${REPO_ROOT}/logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/latent_skill_binding.json"
python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") is True else 1)' "${BINDING}" \
    || { echo "[ERROR] binding record missing/failed: ${BINDING}" >&2; exit 2; }

export OUTPUT_ROOT GOAL_NAMES GOAL_LIMIT
export SEED="${SEED:-0}"
export MODEL_SIZE="${MODEL_SIZE:-large}"
export DEMO_ROWS_PER_GOAL="${DEMO_ROWS_PER_GOAL:-1000}"
export ROLLOUT_ROWS_PER_GOAL="${ROLLOUT_ROWS_PER_GOAL:-1000}"
export ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-10}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-4000}"
export FINETUNE_UPDATES="${FINETUNE_UPDATES:-4000}"
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
export CLUSTER_SLURM_JOB_NAME_PREFIX="${CLUSTER_SLURM_JOB_NAME_PREFIX:-bones-abl-${ARM}}"
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
export CLUSTER_SLURM_PIPELINE_ARRAY="0-$((GOAL_LIMIT - 1))%${MAX_PARALLEL_GOALS:-5}"
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
