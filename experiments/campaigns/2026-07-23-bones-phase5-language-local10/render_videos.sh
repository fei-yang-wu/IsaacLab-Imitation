#!/usr/bin/env bash
set -euo pipefail

# Render per-goal closed-loop videos for the LOCAL ten-goal Phase-5 run using
# the single shared planner + single frozen low-level policy.
#
# Protocol (AGENTS.md): the retained video comes from the non-terminating
# full-horizon diagnostic pass with ALL early terminations disabled, including
# base_too_low, so the clip is not truncated by a fall. That mode is selected by
# passing NEITHER --disable_tracking_terminations NOR --keep_early_terminations,
# which triggers the evaluator's _disable_non_reference_terminations path.
#
# One boot per goal, one environment, one explicit language goal each, matching
# the single-goal-per-deployment contract. Absolute video paths are printed to
# stdout for direct access on this workstation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_phase5_local10_seed${SEED}}"
# Which planner to visualize: the rollout-finetuned one by default; set
# PLANNER=pretrained to visualize the demonstration-only planner instead.
PLANNER="${PLANNER:-finetuned}"
case "${PLANNER}" in
    finetuned)
        PLANNER_CHECKPOINT="${OUTPUT_ROOT}/latent_skill/planner_finetune_planner_rollout/checkpoints/latest.pt" ;;
    pretrained)
        PLANNER_CHECKPOINT="${OUTPUT_ROOT}/latent_skill/planner_pretrain_demonstration/checkpoints/latest.pt" ;;
    *)
        echo "[ERROR] PLANNER must be 'finetuned' or 'pretrained', got: ${PLANNER}" >&2; exit 2 ;;
esac

LATENT_LOW_LEVEL_CHECKPOINT="${LATENT_LOW_LEVEL_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt}"
LATENT_SKILL_CHECKPOINT="${LATENT_SKILL_CHECKPOINT:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt}"
LANGUAGE_EMBEDDINGS="${LANGUAGE_EMBEDDINGS:-data/bones_seed_phase5_corrected/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"
MANIFEST="${MANIFEST:-data/bones_seed_phase5_local10/manifests/g1_bones_seed_phase5_local10_manifest.json}"
LATENT_DATASET_PATH="${LATENT_DATASET_PATH:-data/bones_seed_phase5_local10/zarr/latent_seed${SEED}}"

GOAL_NAMES="${GOAL_NAMES:-Neutral_stoop_down_001_A057 avoid_bump_let_go_R_003_A460 axe_cutting_tree_horizontal_R_004_A355 big_heavy_two_hands_front_high_to_front_high_R_001_A524 big_light_two_hands_pick_up_front_medium_R_001_A509 body_check_001_A180 burning_loop_R_001_A528 casual_greeting_R_001_A428 cellphone_typing_sequence_one_hand_idle_R_001_A423 cough_tuberculosis_R_001_A500}"
MAX_STEPS="${MAX_STEPS:-500}"
FLOW_STEPS="${FLOW_STEPS:-16}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"
VIDEO_ROOT="${VIDEO_ROOT:-${OUTPUT_ROOT}/videos_${PLANNER}}"

for artifact in "${PLANNER_CHECKPOINT}" "${LATENT_LOW_LEVEL_CHECKPOINT}" \
    "${LATENT_SKILL_CHECKPOINT}" "${LANGUAGE_EMBEDDINGS}" "${MANIFEST}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "[ERROR] Required input is missing: ${artifact}" >&2
        exit 2
    fi
done

read -r -a goals <<< "${GOAL_NAMES}"
declare -a rendered_paths=()
index=0
for goal in "${goals[@]}"; do
    slug="$(printf '%04d_%s' "${index}" "${goal}")"
    out_dir="${VIDEO_ROOT}/${slug}"
    echo "[INFO] Rendering goal ${index}: ${goal} -> ${out_dir}"
    cmd=(
        pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py
        --headless --video --video_length "${MAX_STEPS}"
        --task Isaac-Imitation-G1-Latent-v0
        --algorithm IPMD
        --checkpoint "${LATENT_LOW_LEVEL_CHECKPOINT}"
        --skill_checkpoint "${LATENT_SKILL_CHECKPOINT}"
        --language_embeddings "${LANGUAGE_EMBEDDINGS}"
        --state_history_steps 9
        --output_dir "${out_dir}"
        --label "bones_seed_local10_${PLANNER}_video_${slug}"
        --num_envs 1
        --max_steps "${MAX_STEPS}"
        --seed "${SEED}"
        --metric_interval "$((MAX_STEPS + 1))"
        --allow_random_reset
        # NON-TERMINATING full-horizon pass: no --disable_tracking_terminations
        # and no --keep_early_terminations, so base_too_low is also disabled.
        --disable_reward_clipping
        --flow_num_inference_steps "${FLOW_STEPS}"
        --flow_inference_noise_std "${FLOW_NOISE_STD}"
        --assert-kitless
        --motion_name "${goal}" --require_goal_motion_match
        --planner_checkpoint "${PLANNER_CHECKPOINT}"
        agent.ipmd.command_source=skill_commander
        "agent.ipmd.skill_commander_checkpoint_path=${PLANNER_CHECKPOINT}"
        "agent.ipmd.skill_commander_embeddings_path=${LANGUAGE_EMBEDDINGS}"
        "agent.ipmd.skill_commander_goal_name=${goal}"
        agent.ipmd.skill_commander_use_achieved_state=true
        "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}"
        "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}"
        "agent.ipmd.hl_skill_checkpoint_path=${LATENT_SKILL_CHECKPOINT}"
        agent.logger.backend=
        agent.ipmd.hl_skill_finetune_enabled=false
        "env.lafan1_manifest_path=${MANIFEST}"
        "env.dataset_path=${LATENT_DATASET_PATH}"
        env.reset_schedule=sequential
        env.wrap_steps=false
        env.observations.policy.enable_corruption=false
        env.refresh_zarr_dataset=false
        env.latent_command_dim=258
        agent.ipmd.latent_dim=258
        agent.ipmd.latent_steps_min=10
        agent.ipmd.latent_steps_max=10
        agent.ipmd.hl_skill_horizon_steps=10
        agent.ipmd.hl_skill_command_mode=z
        agent.ipmd.latent_learning.command_phase_mode=sin_cos
        agent.ipmd.latent_learning.code_latent_dim=256
        agent.ipmd.latent_learning.code_period=10
        agent.ipmd.reward_loss_coeff=0.0
        agent.ipmd.reward_l2_coeff=0.0
        agent.ipmd.reward_grad_penalty_coeff=0.0
        agent.ipmd.reward_logit_reg_coeff=0.0
        agent.ipmd.reward_param_weight_decay_coeff=0.0
        physics=newton_mjwarp
        env.sim.physics.solver_cfg.njmax=320
        env.sim.physics.solver_cfg.nconmax=40
    )
    OMNI_KIT_ACCEPT_EULA=YES "${cmd[@]}"
    for mp4 in "${out_dir}"/videos/play/*.mp4; do
        [[ -e "${mp4}" ]] || continue
        abs="$(cd "$(dirname "${mp4}")" && pwd)/$(basename "${mp4}")"
        echo "[VIDEO] ${goal}: ${abs}"
        rendered_paths+=("${abs}")
    done
    index=$((index + 1))
done

echo
echo "[INFO] Rendered ${#rendered_paths[@]} video(s) under: ${VIDEO_ROOT}"
for path in "${rendered_paths[@]}"; do
    echo "[VIDEO] ${path}"
done
