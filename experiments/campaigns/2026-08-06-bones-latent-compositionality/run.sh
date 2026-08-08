#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STAGES="${STAGES:-analyze,trajectory,scale,gallery}"
SELECTION="${SELECTION:-${SCRIPT_DIR}/selection30.json}"
TRAITS="${TRAITS:-${SCRIPT_DIR}/semantic_traits30.json}"
ANNOTATIONS="${ANNOTATIONS:-${SCRIPT_DIR}/semantic_phase_annotations30.json}"
TAXONOMY="${TAXONOMY:-${SCRIPT_DIR}/semantic_region_taxonomy.json}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/storage/fwu91/bones_seed_full/manifests/g1_bones_seed_sonic_full_manifest.json}"
DATA_ROOT="${DATA_ROOT:-data/bones_seed_language30_compositionality_v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/manifests/g1_bones_seed_language30_compositionality_v1_manifest.json}"
LANGUAGE_SIDECAR="${LANGUAGE_SIDECAR:-${DATA_ROOT}/manifests/g1_bones_seed_language30_compositionality_v1_manifest_language.json}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-${DATA_ROOT}/reference_arrays/root_qpos_v1}"
REFERENCE_ARRAYS_PERSIST_ID="${REFERENCE_ARRAYS_PERSIST_ID:-bones_seed_language30_compositionality_v1@f31fd755}"
FULL_REFERENCE_ARRAYS="${FULL_REFERENCE_ARRAYS:-/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt}"
VIDEO_ROOT="${VIDEO_ROOT:-logs/bones_language30_oracle_videos/rollout24_gamma097_3p5b_seed0_randomized_no_push}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/bones_language30_compositionality_oracle_seed0}"
SAMPLES="${SAMPLES:-${OUTPUT_ROOT}/collection/rollout_training_samples}"
PHASE_CLIPS="${PHASE_CLIPS:-${OUTPUT_ROOT}/semantic_phase_clips}"

EXPECTED_SELECTION_SHA256="f31fd7551ea8d1d22d74e7616108a2507da21bec39830553210de3a440e66a6e"
EXPECTED_LOW_LEVEL_SHA256="23fdd62a784fd3c57f30a466e8b5a1fb94d31176a211254c0443e126c8ea283e"
EXPECTED_SKILL_SHA256="d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7"

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }
require_file() { [[ -f "$1" ]] || { echo "[ERROR] Missing $1" >&2; exit 2; }; }
run_if_missing() {
    local expected="$1"
    shift
    if [[ -f "${expected}" ]]; then
        echo "[SKIP] ${expected}"
    else
        echo "[CMD] $*"
        "$@"
        require_file "${expected}"
    fi
}

require_file "${SELECTION}"
require_file "${LOW_LEVEL_CHECKPOINT}"
require_file "${SKILL_CHECKPOINT}"
[[ "$(sha256sum "${SELECTION}" | cut -d' ' -f1)" == "${EXPECTED_SELECTION_SHA256}" ]] \
    || { echo "[ERROR] Selection hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${LOW_LEVEL_CHECKPOINT}" | cut -d' ' -f1)" == "${EXPECTED_LOW_LEVEL_SHA256}" ]] \
    || { echo "[ERROR] Low-level checkpoint hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${SKILL_CHECKPOINT}" | cut -d' ' -f1)" == "${EXPECTED_SKILL_SHA256}" ]] \
    || { echo "[ERROR] Skill checkpoint hash mismatch." >&2; exit 2; }

if has_stage prepare; then
    require_file "${SOURCE_MANIFEST}"
    run_if_missing "${MANIFEST}" \
        pixi run python -m imitation_experiments.data.prepare_language_motion_selection \
        --source_manifest "${SOURCE_MANIFEST}" --selection "${SELECTION}" \
        --output_manifest "${MANIFEST}"
    run_if_missing "${LANGUAGE_EMBEDDINGS}" \
        pixi run -e isaaclab python scripts/rlopt/build_language_goal_embeddings.py \
        --manifest "${MANIFEST}" --language_sidecar "${LANGUAGE_SIDECAR}" \
        --require_language_sidecar_matches --backend sentence-transformer \
        --model all-MiniLM-L6-v2 --output "${LANGUAGE_EMBEDDINGS}"
    run_if_missing "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" \
        pixi run python -m imitation_experiments.data.build_reference_arrays \
        --manifest "${MANIFEST}" --output_dir "${REFERENCE_ARRAYS}" \
        --persist_id "${REFERENCE_ARRAYS_PERSIST_ID}" \
        --body_names pelvis left_hip_roll_link left_knee_link left_ankle_roll_link \
            right_hip_roll_link right_knee_link right_ankle_roll_link torso_link \
            left_shoulder_roll_link left_elbow_link left_wrist_yaw_link \
            right_shoulder_roll_link right_elbow_link right_wrist_yaw_link \
        --anchor_body pelvis --dataset_name bones_seed --workers 4 \
        --expected_motions 30 --verify_load --verify_samples 30
fi

for artifact in "${MANIFEST}" "${LANGUAGE_SIDECAR}" "${LANGUAGE_EMBEDDINGS}" \
    "${REFERENCE_ARRAYS}/reference_arrays_manifest.json"; do
    require_file "${artifact}"
done

if has_stage collect; then
    run_if_missing "${OUTPUT_ROOT}/collection/summary.json" \
        pixi run python -m imitation_experiments.pipeline.run_language_planner_oracle_pretrain \
        --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" --manifest "${MANIFEST}" \
        --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
        --reference_arrays_dir "${REFERENCE_ARRAYS}" \
        --reference_arrays_persist_id "${REFERENCE_ARRAYS_PERSIST_ID}" \
        --output_root "${OUTPUT_ROOT}" --stage collect --seed 0 \
        --trajectories_per_motion 100 --collection_num_envs 3000 \
        --collection_max_steps 1200 --future_window_frames 30 \
        --sample_rows_per_file 8192
fi

if has_stage annotate; then
    require_file "${TRAITS}"
    run_if_missing "${ANNOTATIONS}" \
        pixi run python -m imitation_experiments.evaluation.build_semantic_phase_annotations \
        --selection "${SELECTION}" --language_sidecar "${LANGUAGE_SIDECAR}" \
        --trait_overrides "${TRAITS}" --output "${ANNOTATIONS}" \
        --video_root "${VIDEO_ROOT}" --output_fps 50
    run_if_missing "${PHASE_CLIPS}/phase_clip_manifest.json" \
        pixi run python -m imitation_experiments.evaluation.segment_semantic_phase_videos \
        --annotations "${ANNOTATIONS}" --output_dir "${PHASE_CLIPS}"
fi

if has_stage analyze; then
    require_file "${ANNOTATIONS}"
    run_if_missing "${OUTPUT_ROOT}/cross_motion_analysis_all30/analysis.json" \
        pixi run python -m imitation_experiments.evaluation.analyze_cross_motion_latent_structure \
        --samples_dir "${SAMPLES}" --reference_arrays_dir "${REFERENCE_ARRAYS}" \
        --output_dir "${OUTPUT_ROOT}/cross_motion_analysis_all30" \
        --phase_annotations "${ANNOTATIONS}" --selection "${SELECTION}" \
        --seed 0 --bootstrap_samples 2000 --tsne_iterations 1500
    run_if_missing "${OUTPUT_ROOT}/cross_motion_analysis_robust27/analysis.json" \
        pixi run python -m imitation_experiments.evaluation.analyze_cross_motion_latent_structure \
        --samples_dir "${SAMPLES}" --reference_arrays_dir "${REFERENCE_ARRAYS}" \
        --output_dir "${OUTPUT_ROOT}/cross_motion_analysis_robust27" \
        --phase_annotations "${ANNOTATIONS}" --selection "${SELECTION}" \
        --exclude_motion_names panic_run_away_180_R_001_A423 \
            walk_big_dog_ff_225_stop_R_001_A492 rock_out_002_A484 \
        --seed 0 --bootstrap_samples 2000 --tsne_iterations 1500
fi

if has_stage trajectory; then
    require_file "${ANNOTATIONS}"
    require_file "${TAXONOMY}"
    run_if_missing "${OUTPUT_ROOT}/semantic_trajectory_analysis_all30/semantic_trajectory_map.html" \
        pixi run python -m imitation_experiments.evaluation.analyze_semantic_latent_trajectories \
        --samples_dir "${SAMPLES}" --phase_annotations "${ANNOTATIONS}" \
        --taxonomy "${TAXONOMY}" --selection "${SELECTION}" \
        --output_dir "${OUTPUT_ROOT}/semantic_trajectory_analysis_all30" \
        --trajectory_names Neutral_stoop_down_001_A057 \
            drinking_standing_mug_R_001_A282 \
            inside_door_handle_right_side_open_walk_turn_close_R_001_A514 \
        --seed 0 --bootstrap_samples 2000 --tsne_iterations 1500
    run_if_missing "${OUTPUT_ROOT}/semantic_trajectory_analysis_robust27/semantic_trajectory_map.html" \
        pixi run python -m imitation_experiments.evaluation.analyze_semantic_latent_trajectories \
        --samples_dir "${SAMPLES}" --phase_annotations "${ANNOTATIONS}" \
        --taxonomy "${TAXONOMY}" --selection "${SELECTION}" \
        --output_dir "${OUTPUT_ROOT}/semantic_trajectory_analysis_robust27" \
        --exclude_motion_names panic_run_away_180_R_001_A423 \
            walk_big_dog_ff_225_stop_R_001_A492 rock_out_002_A484 \
        --trajectory_names Neutral_stoop_down_001_A057 \
            drinking_standing_mug_R_001_A282 \
            inside_door_handle_right_side_open_walk_turn_close_R_001_A514 \
        --seed 0 --bootstrap_samples 2000 --tsne_iterations 1500
fi

if has_stage scale; then
    require_file "${FULL_REFERENCE_ARRAYS}/reference_arrays_manifest.json"
    run_if_missing "${OUTPUT_ROOT}/reference_scale_500families_seed0/analysis.json" \
        pixi run python -m imitation_experiments.evaluation.analyze_reference_latent_scale \
        --reference_arrays_dir "${FULL_REFERENCE_ARRAYS}" \
        --skill_checkpoint "${SKILL_CHECKPOINT}" \
        --output_dir "${OUTPUT_ROOT}/reference_scale_500families_seed0" \
        --motion_count 500 --windows_per_motion 5 --seed 0 \
        --batch_size 4096 --bootstrap_samples 2000 --tsne_iterations 1500
fi

if has_stage gallery; then
    require_file "${PHASE_CLIPS}/phase_clip_manifest.json"
    run_if_missing "${OUTPUT_ROOT}/cross_motion_neighbor_gallery_robust27_distinct/gallery.html" \
        pixi run python -m imitation_experiments.evaluation.build_latent_neighbor_gallery \
        --analysis_dir "${OUTPUT_ROOT}/cross_motion_analysis_robust27" \
        --phase_clip_manifest "${PHASE_CLIPS}/phase_clip_manifest.json" \
        --output_dir "${OUTPUT_ROOT}/cross_motion_neighbor_gallery_robust27_distinct" \
        --neighbors 5
fi

echo "[PASS] BONES latent-compositionality stages: ${STAGES}"
