#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${LATENT_LOW_LEVEL_CHECKPOINT:?Set LATENT_LOW_LEVEL_CHECKPOINT}"
: "${LATENT_SKILL_CHECKPOINT:?Set LATENT_SKILL_CHECKPOINT}"
: "${VANILLA_TRACKER_CHECKPOINT:?Set VANILLA_TRACKER_CHECKPOINT}"
: "${MANIFEST:?Set MANIFEST to the fresh BONES-SEED manifest}"
: "${LANGUAGE_EMBEDDINGS:?Set LANGUAGE_EMBEDDINGS}"
: "${LATENT_DATASET_PATH:?Set LATENT_DATASET_PATH to a cache for this manifest}"
: "${VANILLA_DATASET_PATH:?Set VANILLA_DATASET_PATH to a cache for this manifest}"
: "${PREPARATION_RECORD:?Set PREPARATION_RECORD from the fresh BONES-SEED export}"

INTERFACES="${INTERFACES:-latent_skill full_body_trajectory}"
ALLOW_UNQUALIFIED_PRELIMINARY="${ALLOW_UNQUALIFIED_PRELIMINARY:-0}"
if [[ "${ALLOW_UNQUALIFIED_PRELIMINARY}" == "1" ]]; then
    if [[ "${INTERFACES}" != "latent_skill" ]]; then
        echo "[ERROR] Unqualified preliminary mode is restricted to INTERFACES=latent_skill." >&2
        exit 2
    fi
else
    : "${VANILLA_QUALIFICATION_AUDIT:?Set VANILLA_QUALIFICATION_AUDIT}"
    : "${LATENT_QUALIFICATION_AUDIT:?Set LATENT_QUALIFICATION_AUDIT}"
    : "${STREAMED_EQUIVALENCE_CERTIFICATE:?Set STREAMED_EQUIVALENCE_CERTIFICATE}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_multigoal_language_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_STAGE="${PIPELINE_STAGE:-all}"
GOAL_INDEX="${GOAL_INDEX:-}"
SEED="${SEED:-0}"
GOAL_LIMIT="${GOAL_LIMIT:-0}"
DEMO_ROWS_PER_GOAL="${DEMO_ROWS_PER_GOAL:-1000}"
ROLLOUT_ROWS_PER_GOAL="${ROLLOUT_ROWS_PER_GOAL:-1000}"
ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-10}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
PRETRAIN_UPDATES="${PRETRAIN_UPDATES:-2000}"
FINETUNE_UPDATES="${FINETUNE_UPDATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-4}"
FLOW_STEPS="${FLOW_STEPS:-16}"
TRAIN_ENDPOINT_STEPS="${TRAIN_ENDPOINT_STEPS:-4}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
MIN_ORACLE_SUCCESS="${MIN_ORACLE_SUCCESS:-0.8}"

read -r -a PYTHON_CMD <<< "${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
read -r -a ISAACLAB_PYTHON_CMD <<< "${INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD:-pixi run -e isaaclab python}"

cmd=(
    "${PYTHON_CMD[@]}"
    experiments/interface_baselines/run_bones_seed_multigoal_language_comparison.py
    --latent_low_level_checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}"
    --latent_skill_checkpoint "${LATENT_SKILL_CHECKPOINT}"
    --vanilla_tracker_checkpoint "${VANILLA_TRACKER_CHECKPOINT}"
    --manifest "${MANIFEST}"
    --language_embeddings "${LANGUAGE_EMBEDDINGS}"
    --latent_dataset_path "${LATENT_DATASET_PATH}"
    --vanilla_dataset_path "${VANILLA_DATASET_PATH}"
    --preparation_record "${PREPARATION_RECORD}"
    --min_oracle_success "${MIN_ORACLE_SUCCESS}"
    --output_root "${OUTPUT_ROOT}"
    --stage "${PIPELINE_STAGE}"
    --goal_limit "${GOAL_LIMIT}"
    --seed "${SEED}"
    --demo_rows_per_goal "${DEMO_ROWS_PER_GOAL}"
    --rollout_rows_per_goal "${ROLLOUT_ROWS_PER_GOAL}"
    --rollout_num_envs "${ROLLOUT_NUM_ENVS}"
    --eval_steps "${EVAL_STEPS}"
    --model_size "${MODEL_SIZE}"
    --pretrain_updates "${PRETRAIN_UPDATES}"
    --finetune_updates "${FINETUNE_UPDATES}"
    --batch_size "${BATCH_SIZE}"
    --micro_batch_size "${MICRO_BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --flow_steps "${FLOW_STEPS}"
    --train_endpoint_steps "${TRAIN_ENDPOINT_STEPS}"
    --flow_noise_std "${FLOW_NOISE_STD}"
    --python_cmd "${INTERFACE_BASELINE_PYTHON_CMD:-pixi run python}"
    --isaaclab_python_cmd "${INTERFACE_BASELINE_ISAACLAB_PYTHON_CMD:-pixi run -e isaaclab python}"
)

read -r -a interfaces <<< "${INTERFACES}"
cmd+=(--interfaces "${interfaces[@]}")
if [[ "${ALLOW_UNQUALIFIED_PRELIMINARY}" != "1" ]]; then
    cmd+=(
        --vanilla_qualification_audit "${VANILLA_QUALIFICATION_AUDIT}"
        --latent_qualification_audit "${LATENT_QUALIFICATION_AUDIT}"
        --streamed_equivalence_certificate "${STREAMED_EQUIVALENCE_CERTIFICATE}"
    )
fi

if [[ -n "${GOAL_INDEX}" ]]; then
    cmd+=(--goal_index "${GOAL_INDEX}")
fi

if [[ -n "${GOAL_NAMES:-}" ]]; then
    read -r -a goal_names <<< "${GOAL_NAMES}"
    cmd+=(--goal_names "${goal_names[@]}")
fi
if [[ "${SKIP_PRETRAINED_CLOSED_LOOP:-0}" == "1" ]]; then
    cmd+=(--skip_pretrained_closed_loop)
fi
if [[ "${REFRESH_DATASETS:-0}" == "1" ]]; then
    cmd+=(--refresh_datasets)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    cmd+=(--resume)
fi
if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
    cmd+=(--continue_on_error)
fi
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "true" ]]; then
    cmd+=(--dry_run)
fi

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
