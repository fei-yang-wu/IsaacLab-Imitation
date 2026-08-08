#!/usr/bin/env bash
set -euo pipefail

# Oracle-trajectory language planner for the discrete FSQ64 command interface.
#
# This is the 2026-08-05 selected-ten oracle pretraining protocol, unchanged,
# driven by the scaled SONIC-sized tracker of campaign
# 2026-08-06-bones129k-sonic-fsq-scale (ICE job 5570936, 4.5B frames) and its
# frozen scaled FSQ64 encoder. The command is 64 FSQ coordinates at 32 levels
# plus two sin/cos phase channels, so the actor command is 66 values wide
# instead of the 258 used by the continuous z256 baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

MODE="${MODE:-print}"
# SET selects the goal set. v3 (selected10_loco_manip_v3) is the current
# development selection; v1 (selected10) is the superseded set kept only to
# reproduce the completed 2026-08-07 discrete run and its matched 2026-08-05
# continuous pairing.
SET="${SET:-v3}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/storage/fwu91/bones_seed_full/manifests/g1_bones_seed_sonic_full_manifest.json}"
SCREEN_DIR="experiments/campaigns/2026-08-05-bones-language10-screen"

case "${SET}" in
    v1)
        SET_SELECTION="${SCREEN_DIR}/selected10.json"
        SET_ROOT="data/bones_seed_language10_v1"
        SET_STEM="g1_bones_seed_language10_v1"
        SET_PERSIST_ID="bones_seed_language10_v1@60a5b7a5"
        SET_MANIFEST_SHA256="60a5b7a5cf0056261d295f6ad02f70bbaf866409f69790932ad33d8ae736e7d1"
        SET_LANGUAGE_SHA256="04624a22adba42f8db9acdc8c74f85ff985305c98ee9857f43b352c54048e0cd"
        SET_REFERENCE_ARRAYS_SHA256="e8996c26ef32b91a2fabf5ae503d896825b81f59bb5eabbbe66036a4576e90ee"
        SET_OUTPUT_ROOT="logs/bones_language10_fsq64_planner_seed0"
        ;;
    v3)
        SET_SELECTION="${SCREEN_DIR}/selected10_loco_manip_v3.json"
        SET_ROOT="data/bones_seed_language10_loco_manip_v3"
        SET_STEM="g1_bones_seed_language10_loco_manip_v3"
        SET_PERSIST_ID="bones_seed_language10_loco_manip_v3@c9f7e7d2"
        SET_MANIFEST_SHA256="1b0a2597d4e7e32e5315cdd90129922e9f0022e30f8e129a55031bd6aea2e95f"
        SET_LANGUAGE_SHA256="3c5ff2e8f8e9780d8c9e6a4656bcab6391f80c25c3590cb68f3bd450416b0f93"
        SET_REFERENCE_ARRAYS_SHA256="e913eb9e848906f802015a7efc8dfc801008102a70cf3432de9dc27fbaab8191"
        SET_OUTPUT_ROOT="logs/bones_language10_fsq64_planner_v3_seed0"
        ;;
    *)
        echo "[ERROR] SET must be v1 or v3; got ${SET}" >&2
        exit 2
        ;;
esac

SELECTION="${SELECTION:-${SET_SELECTION}}"
DERIVED_ROOT="${DERIVED_ROOT:-${SET_ROOT}}"
MANIFEST="${MANIFEST:-${DERIVED_ROOT}/manifests/${SET_STEM}_manifest.json}"
LANGUAGE_SIDECAR="${LANGUAGE_SIDECAR:-${DERIVED_ROOT}/manifests/${SET_STEM}_manifest_language.json}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-${DERIVED_ROOT}/language/${SET_STEM}_minilm_goal_embeddings.pt}"
REFERENCE_ARRAYS="${REFERENCE_ARRAYS:-${DERIVED_ROOT}/reference_arrays/root_qpos_v1}"
REFERENCE_ARRAYS_PERSIST_ID="${REFERENCE_ARRAYS_PERSIST_ID:-${SET_PERSIST_ID}}"

FSQ_EVAL_ROOT="${FSQ_EVAL_ROOT:-logs/bones129k_sonic_fsq_scale_eval}"
LOW_LEVEL_CHECKPOINT="${LOW_LEVEL_CHECKPOINT:-${FSQ_EVAL_ROOT}/4500357120/fsq64_sonic/model_step_4500357120.pt}"
SKILL_CHECKPOINT="${SKILL_CHECKPOINT:-${FSQ_EVAL_ROOT}/encoders/fsq64_scaled.pt}"
EXPECTED_LOW_LEVEL_SHA256="1e8555a5b92437899a1c8bfbf6f7c7e3e978e1dc732ca2bb9e4095f84fce9653"
EXPECTED_SKILL_SHA256="6a4a724872273a6a2850e433881e7746dbce2b7ccb92e4ee18153cffad77da14"
EXPECTED_MANIFEST_SHA256="${SET_MANIFEST_SHA256}"
EXPECTED_LANGUAGE_SHA256="${SET_LANGUAGE_SHA256}"
EXPECTED_REFERENCE_ARRAYS_SHA256="${SET_REFERENCE_ARRAYS_SHA256}"

# Discrete FSQ64 command contract, identical to the low-level training run.
SKILL_Z_DIM="${SKILL_Z_DIM:-64}"
POLICY_NUM_CELLS="${POLICY_NUM_CELLS:-2048 2048 1024 1024 512 512}"
POLICY_ACTIVATION="${POLICY_ACTIVATION:-silu}"
# 289 overflows the Newton constraint buffer on this contract; the matched
# low-level evaluation of the same checkpoint used 320.
SOLVER_NJMAX="${SOLVER_NJMAX:-320}"
SOLVER_NCONMAX="${SOLVER_NCONMAX:-200}"

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
    || { echo "[ERROR] FSQ64 tracker checkpoint hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${SKILL_CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_SKILL_SHA256}" ]] \
    || { echo "[ERROR] Scaled FSQ64 encoder hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]] \
    || { echo "[ERROR] Selected-ten manifest hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${LANGUAGE_EMBEDDINGS}" | awk '{print $1}')" == "${EXPECTED_LANGUAGE_SHA256}" ]] \
    || { echo "[ERROR] MiniLM table hash mismatch." >&2; exit 2; }
[[ "$(sha256sum "${REFERENCE_ARRAYS}/reference_arrays_manifest.json" | awk '{print $1}')" == "${EXPECTED_REFERENCE_ARRAYS_SHA256}" ]] \
    || { echo "[ERROR] Selected-ten reference-array hash mismatch." >&2; exit 2; }

OUTPUT_ROOT="${OUTPUT_ROOT:-${SET_OUTPUT_ROOT}}"
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
    OUTPUT_ROOT="logs/bones_language10_fsq64_planner_smoke"
    TRAJECTORIES_PER_MOTION=1
    EVAL_TRAJECTORIES_PER_MOTION=1
    NUM_UPDATES=20
    MILESTONE_INTERVAL=10
    STAGE=all
    RESUME=0
fi

LATENT_COMMAND_DIM=$((SKILL_Z_DIM + 2))
POLICY_CELLS_HYDRA="[$(echo "${POLICY_NUM_CELLS}" | tr ' ' ',')]"

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
    --skill_z_dim "${SKILL_Z_DIM}"
    --policy_num_cells ${POLICY_NUM_CELLS}
    --policy_activation "${POLICY_ACTIVATION}"
    --solver_njmax "${SOLVER_NJMAX}"
    --solver_nconmax "${SOLVER_NCONMAX}"
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
            "env.command_interface.actor.dim=${LATENT_COMMAND_DIM}" \
            'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
            'agent.logger.backend=' \
            "agent.ipmd.latent_dim=${LATENT_COMMAND_DIM}" \
            agent.ipmd.hl_skill_finetune_enabled=false \
            agent.ipmd.latent_steps_min=10 \
            agent.ipmd.latent_steps_max=10 \
            agent.ipmd.hl_skill_horizon_steps=10 \
            agent.ipmd.hl_skill_command_mode=z \
            agent.ipmd.latent_learning.command_phase_mode=sin_cos \
            "agent.ipmd.latent_learning.code_latent_dim=${SKILL_Z_DIM}" \
            agent.ipmd.latent_learning.code_period=10 \
            "agent.policy.num_cells=${POLICY_CELLS_HYDRA}" \
            "agent.policy.activation_fn=${POLICY_ACTIVATION}" \
            "agent.value_function.num_cells=${POLICY_CELLS_HYDRA}" \
            "agent.value_function.activation_fn=${POLICY_ACTIVATION}" \
            "env.sim.physics.solver_cfg.njmax=${SOLVER_NJMAX}" \
            "env.sim.physics.solver_cfg.nconmax=${SOLVER_NCONMAX}"
        ;;
    *)
        echo "[ERROR] MODE must be print, prepare, smoke, run, or video." >&2
        exit 2
        ;;
esac
