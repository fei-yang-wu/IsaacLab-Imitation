#!/usr/bin/env bash
set -euo pipefail

# Side-by-side "reference vs language-planner+policy" videos for the LOCAL
# ten-goal Phase-5 run. In one scene: env 0 replays the expert reference
# (cyan), env 1 runs the frozen H200 low-level policy driven by the shared
# language planner for the same goal (red). Terminations are disabled by the
# comparison script, and --video_length forces the full 500-step horizon
# regardless of how short the reference clip is.
#
# Usage:
#   experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_reference_comparison.sh
#   SUBSET="0 2 8" .../render_reference_comparison.sh   # only those goal indices
#   PLANNER=pretrained .../render_reference_comparison.sh
#   REF_VIS=both .../render_reference_comparison.sh      # robot + body markers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/interface_baselines/bones_seed_phase5_local10_seed${SEED}}"
PLANNER="${PLANNER:-finetuned}"
# PLANNER_CHECKPOINT may be pre-set (e.g. a planner checkpoint pulled back from
# an ICE ablation arm); only derive it from OUTPUT_ROOT when unset.
if [[ -z "${PLANNER_CHECKPOINT:-}" ]]; then
    case "${PLANNER}" in
        finetuned) PLANNER_CHECKPOINT="${OUTPUT_ROOT}/latent_skill/planner_finetune_planner_rollout/checkpoints/latest.pt" ;;
        pretrained) PLANNER_CHECKPOINT="${OUTPUT_ROOT}/latent_skill/planner_pretrain_demonstration/checkpoints/latest.pt" ;;
        *) echo "[ERROR] PLANNER must be 'finetuned' or 'pretrained', got: ${PLANNER}" >&2; exit 2 ;;
    esac
fi

LOW="${LOW:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt}"
SKILL_ENCODER="${SKILL_ENCODER:-logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt}"
LANG_EMB="${LANG_EMB:-data/bones_seed_phase5_corrected/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt}"
MANIFEST="${MANIFEST:-data/bones_seed_phase5_local10/manifests/g1_bones_seed_phase5_local10_manifest.json}"
DATASET="${DATASET:-data/bones_seed_phase5_local10/zarr/latent_seed${SEED}}"
OUT_ROOT="${OUT_ROOT:-${OUTPUT_ROOT}/compare_reference_vs_planner_${PLANNER}}"
REF_VIS="${REF_VIS:-robot}"     # robot | body_markers | both
DEVICE="${DEVICE:-cuda:0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
FLOW_STEPS="${FLOW_STEPS:-16}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.0}"

for f in "${PLANNER_CHECKPOINT}" "${LOW}" "${SKILL_ENCODER}" "${LANG_EMB}" "${MANIFEST}"; do
    [[ -f "${f}" ]] || { echo "[ERROR] Missing required input: ${f}" >&2; exit 2; }
done

# Goal order == subset-manifest trajectory order == policy_trajectory_rank.
mapfile -t GOALS < <(python3 -c "
import json,sys
man=json.load(open('${MANIFEST}'))
for e in man['dataset']['trajectories']['lafan1_csv']:
    print(e['name'])
")
[[ "${#GOALS[@]}" -gt 0 ]] || { echo '[ERROR] No motions in subset manifest.' >&2; exit 2; }

mkdir -p "${OUT_ROOT}"
INDEX="${OUT_ROOT}/video_index.md"
{
    echo "# Reference vs Language-Planner Videos (side by side) — local10 ${PLANNER}"
    echo
    echo "One scene, two robots: **cyan = reference (ground truth)**, **red = planner+policy**. Full ${VIDEO_LENGTH}-step horizon, terminations disabled."
    echo
    echo "| rank | goal | status | video |"
    echo "|---:|---|---|---|"
} > "${INDEX}"

declare -a rendered=()
for RANK in "${!GOALS[@]}"; do
    MOTION="${GOALS[$RANK]}"
    if [[ -n "${SUBSET:-}" ]] && ! echo " ${SUBSET} " | grep -q " ${RANK} "; then
        continue
    fi
    OUT="${OUT_ROOT}/rank_$(printf '%04d' "${RANK}")_${MOTION}"
    mkdir -p "${OUT}"
    echo "===== rank ${RANK}  ${MOTION} ====="
    set +e
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TORCHDYNAMO_DISABLE=1 \
    pixi run -e isaaclab python scripts/viz/compare_policy_reference.py \
        --headless --device "${DEVICE}" --seed "${SEED}" \
        --task Isaac-Imitation-G1-Latent-v0 --algo IPMD \
        --checkpoint "${LOW}" \
        --policy_trajectory_rank "${RANK}" \
        --policy_start_step 0 \
        --reference_visualization "${REF_VIS}" \
        --video --video_length "${VIDEO_LENGTH}" \
        --output_dir "${OUT}" \
        env.lafan1_manifest_path="${MANIFEST}" \
        env.dataset_path="${DATASET}" \
        env.refresh_zarr_dataset=false \
        env.observations.policy.enable_corruption=false \
        env.latent_command_dim=258 \
        agent.ipmd.latent_dim=258 \
        agent.ipmd.command_source=skill_commander \
        "agent.ipmd.skill_commander_checkpoint_path=${PLANNER_CHECKPOINT}" \
        "agent.ipmd.skill_commander_goal_name=${MOTION}" \
        "agent.ipmd.skill_commander_embeddings_path=${LANG_EMB}" \
        agent.ipmd.skill_commander_use_achieved_state=true \
        "agent.ipmd.skill_commander_flow_num_inference_steps=${FLOW_STEPS}" \
        "agent.ipmd.skill_commander_flow_inference_noise_std=${FLOW_NOISE_STD}" \
        "agent.ipmd.hl_skill_checkpoint_path=${SKILL_ENCODER}" \
        agent.ipmd.hl_skill_finetune_enabled=false \
        agent.ipmd.hl_skill_horizon_steps=10 \
        agent.ipmd.hl_skill_command_mode=z \
        agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
        agent.ipmd.latent_learning.command_phase_mode=sin_cos \
        agent.ipmd.latent_learning.code_latent_dim=256 \
        agent.ipmd.latent_learning.code_period=10 \
        agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0 \
        agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0 \
        agent.ipmd.reward_param_weight_decay_coeff=0.0 \
        physics=newton_mjwarp \
        env.sim.physics.solver_cfg.njmax=320 \
        env.sim.physics.solver_cfg.nconmax=40 \
        > "${OUT}/compare.log" 2>&1
    RC=$?
    set -e
    MP4="$(find "${OUT}" -name '*.mp4' | head -1 || true)"
    if [[ "${RC}" -eq 0 && -n "${MP4}" ]]; then
        ABS="$(cd "$(dirname "${MP4}")" && pwd)/$(basename "${MP4}")"
        echo "  OK -> ${ABS}"
        echo "[VIDEO] ${MOTION}: ${ABS}"
        rendered+=("${ABS}")
        echo "| ${RANK} | ${MOTION} | ok | ${ABS} |" >> "${INDEX}"
    else
        echo "  FAIL rc=${RC} (see ${OUT}/compare.log)"
        echo "| ${RANK} | ${MOTION} | FAIL rc=${RC} | - |" >> "${INDEX}"
    fi
done

echo
echo "[INFO] Rendered ${#rendered[@]} comparison video(s). Index: ${OUT_ROOT}/video_index.md"
for path in "${rendered[@]}"; do echo "[VIDEO] ${path}"; done
