#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
SELECTION="${SELECTION:-experiments/campaigns/2026-08-05-bones-language10-screen/selected10.json}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/storage/fwu91/bones_seed_full/manifests/g1_bones_seed_sonic_full_manifest.json}"
DERIVED_ROOT="${DERIVED_ROOT:-data/bones_seed_language10_v1}"
MANIFEST="${MANIFEST:-${DERIVED_ROOT}/manifests/g1_bones_seed_language10_v1_manifest.json}"
LANGUAGE_SIDECAR="${LANGUAGE_SIDECAR:-${DERIVED_ROOT}/manifests/g1_bones_seed_language10_v1_manifest_language.json}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-${DERIVED_ROOT}/language/g1_bones_seed_language10_v1_minilm_goal_embeddings.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-${DERIVED_ROOT}/reference_arrays/root_qpos_v1}"
REFERENCE_ARRAYS_PERSIST_ID="${REFERENCE_ARRAYS_PERSIST_ID:-bones_seed_language10_v1@60a5b7a5}"

LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt}"
EXPECTED_LOW_LEVEL_SHA256="23fdd62a784fd3c57f30a466e8b5a1fb94d31176a211254c0443e126c8ea283e"
EXPECTED_SKILL_SHA256="d191d8656620059a569edbad82ca182cb2d2f85839300153cb618d1e29f8c5e7"
EXPECTED_MANIFEST_SHA256="60a5b7a5cf0056261d295f6ad02f70bbaf866409f69790932ad33d8ae736e7d1"
EXPECTED_LANGUAGE_SHA256="04624a22adba42f8db9acdc8c74f85ff985305c98ee9857f43b352c54048e0cd"
EXPECTED_REFERENCE_ARRAYS_SHA256="e8996c26ef32b91a2fabf5ae503d896825b81f59bb5eabbbe66036a4576e90ee"

prepare_data() {
    if [[ ! -f "${MANIFEST}" ]]; then
        pixi run python -m imitation_experiments.data.prepare_language_motion_selection \
            --source_manifest "${SOURCE_MANIFEST}" \
            --selection "${SELECTION}" \
            --output_manifest "${MANIFEST}"
    fi
    if [[ ! -f "${LANGUAGE_EMBEDDINGS}" ]]; then
        pixi run -e isaaclab python scripts/rlopt/build_language_goal_embeddings.py \
            --manifest "${MANIFEST}" \
            --language_sidecar "${LANGUAGE_SIDECAR}" \
            --require_language_sidecar_matches \
            --backend sentence-transformer \
            --model all-MiniLM-L6-v2 \
            --output "${LANGUAGE_EMBEDDINGS}"
    fi
    if [[ ! -f "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]]; then
        pixi run python -m imitation_experiments.data.build_reference_arrays \
            --manifest "${MANIFEST}" \
            --output_dir "${REFERENCE_ARRAYS}" \
            --persist_id "${REFERENCE_ARRAYS_PERSIST_ID}" \
            --body_names pelvis left_hip_roll_link left_knee_link left_ankle_roll_link \
                right_hip_roll_link right_knee_link right_ankle_roll_link torso_link \
                left_shoulder_roll_link left_elbow_link left_wrist_yaw_link \
                right_shoulder_roll_link right_elbow_link right_wrist_yaw_link \
            --anchor_body pelvis --dataset_name bones_seed --workers 4 \
            --expected_motions 10 --verify_load --verify_samples 10
    fi
}

prepare_data
for artifact in "${SELECTION}" "${SOURCE_MANIFEST}" "${MANIFEST}" \
    "${LANGUAGE_EMBEDDINGS}" "${LOW_LEVEL_CHECKPOINT}" "${SKILL_CHECKPOINT}"; do
    [[ -f "${artifact}" ]] || { echo "[ERROR] Missing ${artifact}" >&2; exit 2; }
done
[[ -f "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" ]] \
    || { echo "[ERROR] Missing selected-ten reference arrays." >&2; exit 2; }
[[ "$(sha256sum "${LOW_LEVEL_CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_LOW_LEVEL_SHA256}" ]] \
    || { echo "[ERROR] Low-level checkpoint hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${SKILL_CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_SKILL_SHA256}" ]] \
    || { echo "[ERROR] Skill checkpoint hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]] \
    || { echo "[ERROR] Selected-ten manifest hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${LANGUAGE_EMBEDDINGS}" | awk '{print $1}')" == "${EXPECTED_LANGUAGE_SHA256}" ]] \
    || { echo "[ERROR] MiniLM table hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" | awk '{print $1}')" == "${EXPECTED_REFERENCE_ARRAYS_SHA256}" ]] \
    || { echo "[ERROR] Selected-ten reference-array hash mismatch." >&2; exit 2; }

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/bones_language10_oracle_pretrain_seed0}"
TRAJECTORIES_PER_MOTION="${TRAJECTORIES_PER_MOTION:-100}"
EVAL_TRAJECTORIES_PER_MOTION="${EVAL_TRAJECTORIES_PER_MOTION:-100}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
NUM_UPDATES="${NUM_UPDATES:-10000}"
MILESTONE_INTERVAL="${MILESTONE_INTERVAL:-2000}"
STAGE="${STAGE:-all}"
RESUME="${RESUME:-0}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${OUTPUT_ROOT}/planner_oracle_pretrain/checkpoints/update_$(printf '%07d' "${NUM_UPDATES}").pt}"
VIDEO_OUTPUT_ROOT="${VIDEO_OUTPUT_ROOT:-${OUTPUT_ROOT}/nonterminating_video/update_$(printf '%07d' "${NUM_UPDATES}")_randomized_no_push}"

if [[ "${MODE}" == "smoke" ]]; then
    OUTPUT_ROOT="logs/bones_language10_oracle_pretrain_smoke"
    TRAJECTORIES_PER_MOTION=1
    EVAL_TRAJECTORIES_PER_MOTION=1
    NUM_UPDATES=20
    MILESTONE_INTERVAL=10
    STAGE=all
    RESUME=0
fi

cmd=(
    pixi run python -m imitation_experiments.pipeline.run_language_planner_oracle_pretrain
    --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}"
    --skill_checkpoint "${SKILL_CHECKPOINT}"
    --manifest "${MANIFEST}"
    --language_embeddings "${LANGUAGE_EMBEDDINGS}"
    --reference_arrays_dir "${REFERENCE_ARRAYS}"
    --reference_arrays_persist_id "${REFERENCE_ARRAYS_PERSIST_ID}"
    --output_root "${OUTPUT_ROOT}"
    --stage "${STAGE}"
    --trajectories_per_motion "${TRAJECTORIES_PER_MOTION}"
    --eval_trajectories_per_motion "${EVAL_TRAJECTORIES_PER_MOTION}"
    --model_size "${MODEL_SIZE}"
    --num_updates "${NUM_UPDATES}"
    --milestone_interval "${MILESTONE_INTERVAL}"
)
if [[ "${RESUME}" == "1" ]]; then cmd+=(--resume); fi

case "${MODE}" in
    print)
        cmd+=(--dry_run)
        printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
        "${cmd[@]}"
        ;;
    prepare)
        echo "[PASS] Prepared manifest and MiniLM table."
        ;;
    smoke|run)
        "${cmd[@]}"
        ;;
    video)
        [[ -f "${PLANNER_CHECKPOINT}" ]] \
            || { echo "[ERROR] Missing planner checkpoint ${PLANNER_CHECKPOINT}" >&2; exit 2; }
        pixi run python \
            .agents/skills/policy-eval-video/scripts/render_policy_videos.py \
            --checkpoint "${LOW_LEVEL_CHECKPOINT}" \
            --planner_checkpoint "${PLANNER_CHECKPOINT}" \
            --skill_checkpoint "${SKILL_CHECKPOINT}" \
            --language_embeddings "${LANGUAGE_EMBEDDINGS}" \
            --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
            --output_root "${VIDEO_OUTPUT_ROOT}" \
            --reference_arrays "${REFERENCE_ARRAYS}" \
            --persist_id "${REFERENCE_ARRAYS_PERSIST_ID}" \
            --randomized_no_push \
            -- physics=newton_mjwarp \
            env.data.reference_arrays_warm_workers=2 \
            env.data.macro_cache_device=cuda:0 \
            'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
            env.data.wrap_steps=false \
            env.command_interface.actor.dim=258 \
            'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
            'agent.logger.backend=' \
            agent.ipmd.latent_dim=258 \
            agent.ipmd.hl_skill_finetune_enabled=false \
            agent.ipmd.latent_steps_min=10 \
            agent.ipmd.latent_steps_max=10 \
            agent.ipmd.hl_skill_horizon_steps=10 \
            agent.ipmd.hl_skill_command_mode=z \
            agent.ipmd.latent_learning.command_phase_mode=sin_cos \
            agent.ipmd.latent_learning.code_latent_dim=256 \
            agent.ipmd.latent_learning.code_period=10 \
            env.sim.physics.solver_cfg.njmax=289 \
            env.sim.physics.solver_cfg.nconmax=200
        ;;
    *)
        echo "[ERROR] MODE must be print, prepare, smoke, run, or video." >&2
        exit 2
        ;;
esac
