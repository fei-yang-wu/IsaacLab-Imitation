#!/usr/bin/env bash
set -euo pipefail

# Thin campaign front door for the LOCAL ten-goal BONES-SEED Phase-5 language
# planner run (latent-only, preliminary). Reusable implementation stays in
# experiments/paper/interface_baselines/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

IMPL_DIR="experiments/paper/interface_baselines"

# ---------------------------------------------------------------------------
# Frozen inputs: H200 91-motion latent controller + its bound h10 encoder,
# and the provenance-complete corrected 100-motion Phase-5 tree.
# ---------------------------------------------------------------------------
LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt}"
BINDING_RECORD="${BINDING_RECORD:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/latent_skill_binding.json}"
EXPECTED_LATENT_SHA256="6765a324a840b33a84f9a0b5a817c60303979bbec7a36ebc31242086d61d1572"
EXPECTED_SKILL_SHA256="562e4f9d0cebcdeb0bdddf6fb77ea8d0b488a8e576442b7106b54a13d6eceadc"

# Required by the shared runner CLI but never loaded in latent-only mode; any
# existing vanilla-task checkpoint satisfies the file-existence check.
VANILLA_TRACKER_CHECKPOINT="${VANILLA_TRACKER_CHECKPOINT:-logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-06-23_09-37-39/models/model_step_600047616.pt}"

SOURCE_MANIFEST="${SOURCE_MANIFEST:-data/bones_seed_phase5_corrected/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-data/bones_seed_phase5_corrected/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"

# Same ten goals as the 2026-07-23 H200 Skynet pilot: present in both the
# 91-motion SONIC-filtered training manifest and the corrected Phase-5 tree.
GOAL_NAMES="${GOAL_NAMES:-Neutral_stoop_down_001_A057 avoid_bump_let_go_R_003_A460 axe_cutting_tree_horizontal_R_004_A355 big_heavy_two_hands_front_high_to_front_high_R_001_A524 big_light_two_hands_pick_up_front_medium_R_001_A509 body_check_001_A180 burning_loop_R_001_A528 casual_greeting_R_001_A428 cellphone_typing_sequence_one_hand_idle_R_001_A423 cough_tuberculosis_R_001_A500}"

SEED="${SEED:-0}"
# Derived ten-motion tree: subset manifest + fresh latent cache. Kept outside
# the frozen corrected tree so the source data is never modified in place.
DERIVED_ROOT="${DERIVED_ROOT:-data/bones_seed_phase5_local10}"
SUBSET_MANIFEST="${SUBSET_MANIFEST:-${DERIVED_ROOT}/manifests/g1_bones_seed_phase5_local10_manifest.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-${DERIVED_ROOT}/zarr/latent_seed${SEED}}"
# Compatibility argument only; no vanilla samples are collected in this campaign.
VANILLA_DATASET_PATH="${VANILLA_DATASET_PATH:-${DERIVED_ROOT}/zarr/vanilla_seed${SEED}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_phase5_local10_seed${SEED}}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-1}"
RESUME="${RESUME:-0}"
REFRESH_DATASETS="${REFRESH_DATASETS:-0}"
SKIP_PRETRAINED_CLOSED_LOOP="${SKIP_PRETRAINED_CLOSED_LOOP:-0}"

DEMO_ROWS_PER_GOAL="${DEMO_ROWS_PER_GOAL:-150}"
ROLLOUT_ROWS_PER_GOAL="${ROLLOUT_ROWS_PER_GOAL:-150}"
ROLLOUT_NUM_ENVS="${ROLLOUT_NUM_ENVS:-10}"
EVAL_STEPS="${EVAL_STEPS:-500}"
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

for artifact in "${LATENT_LOW_LEVEL_CHECKPOINT}" "${LATENT_SKILL_CHECKPOINT}" \
    "${VANILLA_TRACKER_CHECKPOINT}" "${SOURCE_MANIFEST}" "${LANGUAGE_EMBEDDINGS}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "[ERROR] Required input is missing: ${artifact}" >&2
        exit 2
    fi
done
actual_latent_sha256="$(sha256sum "${LATENT_LOW_LEVEL_CHECKPOINT}" | awk '{print $1}')"
actual_skill_sha256="$(sha256sum "${LATENT_SKILL_CHECKPOINT}" | awk '{print $1}')"
if [[ "${actual_latent_sha256}" != "${EXPECTED_LATENT_SHA256}" ]]; then
    echo "[ERROR] H200 latent checkpoint hash changed: ${actual_latent_sha256}" >&2
    exit 2
fi
if [[ "${actual_skill_sha256}" != "${EXPECTED_SKILL_SHA256}" ]]; then
    echo "[ERROR] H10 skill checkpoint hash changed: ${actual_skill_sha256}" >&2
    exit 2
fi
if [[ ! -f "${BINDING_RECORD}" ]] || ! python3 -c 'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("passed") is True else 1)' "${BINDING_RECORD}"; then
    echo "[ERROR] The latent/skill encoder binding record is missing or failed: ${BINDING_RECORD}" >&2
    exit 2
fi

read -r -a goal_names <<< "${GOAL_NAMES}"

if [[ ! -f "${SUBSET_MANIFEST}" ]]; then
    echo "[INFO] Building the ${#goal_names[@]}-motion subset manifest: ${SUBSET_MANIFEST}"
    pixi run python -m imitation_experiments.data.write_motion_subset_manifest \
        --manifest "${SOURCE_MANIFEST}" \
        --motion_names "${goal_names[@]}" \
        --output "${SUBSET_MANIFEST}"
fi

cmd=(
    pixi run python -m imitation_experiments.pipeline.run_bones_seed_multigoal_language_comparison
    --latent_low_level_checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}"
    --latent_skill_checkpoint "${LATENT_SKILL_CHECKPOINT}"
    --vanilla_tracker_checkpoint "${VANILLA_TRACKER_CHECKPOINT}"
    --manifest "${SUBSET_MANIFEST}"
    --language_embeddings "${LANGUAGE_EMBEDDINGS}"
    --latent_dataset_path "${LATENT_DATASET_PATH}"
    --vanilla_dataset_path "${VANILLA_DATASET_PATH}"
    --interfaces latent_skill
    --goal_names "${goal_names[@]}"
    --output_root "${OUTPUT_ROOT}"
    --stage "${STAGE}"
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
)

if [[ -n "${GOAL_INDEX:-}" ]]; then
    cmd+=(--goal_index "${GOAL_INDEX}")
fi
if [[ "${SKIP_PRETRAINED_CLOSED_LOOP}" == "1" ]]; then
    cmd+=(--skip_pretrained_closed_loop)
fi
if [[ "${REFRESH_DATASETS}" == "1" ]]; then
    cmd+=(--refresh_datasets)
fi
if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
fi
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    cmd+=(--dry_run)
fi

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
