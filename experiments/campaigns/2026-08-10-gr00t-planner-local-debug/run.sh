#!/usr/bin/env bash
set -euo pipefail

# GR00T planner local debug loop: collect (z256_scaled oracle) -> goals
# (Cosmos features) -> train (GR00T head, warm-start stage A). See README.md.
#
#   ./run.sh print|collect|goals|train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODE="${1:-print}"

CHECKPOINT_ROOT="logs/downloaded_checkpoints/bones129k_recent_ice"
LOW_LEVEL_CHECKPOINT="${CHECKPOINT_ROOT}/z256_scaled/model_step_5750390784.pt"
SKILL_CHECKPOINT="${CHECKPOINT_ROOT}/encoders/z256_scaled.pt"
DERIVED_ROOT="data/bones_seed_language10_v1"
MANIFEST="${DERIVED_ROOT}/manifests/g1_bones_seed_language10_v1_manifest.json"
LANGUAGE_SIDECAR="${DERIVED_ROOT}/manifests/g1_bones_seed_language10_v1_manifest_language.json"
LANGUAGE_EMBEDDINGS="${DERIVED_ROOT}/language/g1_bones_seed_language10_v1_minilm_goal_embeddings.pt"
REFERENCE_ARRAYS="${DERIVED_ROOT}/reference_arrays/root_qpos_v1"
REFERENCE_ARRAYS_PERSIST_ID="bones_seed_language10_v1@60a5b7a5"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/gr00t_planner_local_debug}"
TRAJECTORIES_PER_MOTION="${TRAJECTORIES_PER_MOTION:-5}"
NUM_UPDATES="${NUM_UPDATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# z256_scaled tracker geometry (recent-local-eval SCALED_CELLS).
POLICY_CELLS=(2048 2048 1024 1024 512 512)

EXPECTED_LOW_LEVEL_SHA256="bc4569e60f7a92309d5832683c1f1e66188f534eb445bf96adff60b17a10204c"
EXPECTED_SKILL_SHA256="862eadd77aa7564cd3b5743be7bcbab65e78bd215fa886945d11620882533ec6"
EXPECTED_MANIFEST_SHA256="60a5b7a5cf0056261d295f6ad02f70bbaf866409f69790932ad33d8ae736e7d1"
EXPECTED_LANGUAGE_SHA256="04624a22adba42f8db9acdc8c74f85ff985305c98ee9857f43b352c54048e0cd"
EXPECTED_REFERENCE_ARRAYS_SHA256="e8996c26ef32b91a2fabf5ae503d896825b81f59bb5eabbbe66036a4576e90ee"

require() {
    local path="$1" expected="$2" actual
    [[ -f "${path}" ]] || { echo "[FATAL] missing ${path}" >&2; exit 2; }
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] \
        || { echo "[FATAL] hash mismatch ${path}: ${actual}" >&2; exit 2; }
}

require "${LOW_LEVEL_CHECKPOINT}" "${EXPECTED_LOW_LEVEL_SHA256}"
require "${SKILL_CHECKPOINT}" "${EXPECTED_SKILL_SHA256}"
require "${MANIFEST}" "${EXPECTED_MANIFEST_SHA256}"
require "${LANGUAGE_EMBEDDINGS}" "${EXPECTED_LANGUAGE_SHA256}"
require "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" "${EXPECTED_REFERENCE_ARRAYS_SHA256}"

collect_cmd=(
    pixi run python -m imitation_experiments.pipeline.run_language_planner_oracle_pretrain
    --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}"
    --skill_checkpoint "${SKILL_CHECKPOINT}"
    --manifest "${MANIFEST}"
    --language_embeddings "${LANGUAGE_EMBEDDINGS}"
    --reference_arrays_dir "${REFERENCE_ARRAYS}"
    --reference_arrays_persist_id "${REFERENCE_ARRAYS_PERSIST_ID}"
    --output_root "${OUTPUT_ROOT}"
    --stage collect
    --trajectories_per_motion "${TRAJECTORIES_PER_MOTION}"
    --eval_trajectories_per_motion 1
    --policy_num_cells "${POLICY_CELLS[@]}"
)

GOAL_FEATURES_DIR="${OUTPUT_ROOT}/goal_features"
SAMPLES_DIR="${OUTPUT_ROOT}/collection/rollout_training_samples"

case "${MODE}" in
    print)
        collect_cmd+=(--dry_run)
        printf '[CMD]'; printf ' %q' "${collect_cmd[@]}"; printf '\n'
        "${collect_cmd[@]}"
        ;;
    collect)
        "${collect_cmd[@]}"
        ;;
    goals)
        pixi run -e gr00t python -m imitation_experiments.planner.cache_gr00t_goal_features \
            --language_sidecar "${LANGUAGE_SIDECAR}" \
            --output_dir "${GOAL_FEATURES_DIR}"
        ;;
    train)
        shopt -s nullglob
        samples=("${SAMPLES_DIR}"/*.pt)
        (( ${#samples[@]} > 0 )) \
            || { echo "[FATAL] no samples in ${SAMPLES_DIR}; run collect first." >&2; exit 2; }
        [[ -f "${GOAL_FEATURES_DIR}/goal_features.pt" ]] \
            || { echo "[FATAL] missing goal features; run goals first." >&2; exit 2; }
        pixi run -e gr00t python -m imitation_experiments.planner.train_gr00t_head \
            --output_dir "${OUTPUT_ROOT}/gr00t_head_stage_a" \
            --preset n17 \
            --samples "${samples[@]}" \
            --goal_features "${GOAL_FEATURES_DIR}/goal_features.pt" \
            --pretrained_bundle "${GOAL_FEATURES_DIR}/action_head_trunk.pt" \
            --num_updates "${NUM_UPDATES}" \
            --batch_size "${BATCH_SIZE}" \
            --stage_a_updates "${NUM_UPDATES}"
        ;;
    *)
        echo "[ERROR] MODE must be print, collect, goals, or train." >&2
        exit 2
        ;;
esac
