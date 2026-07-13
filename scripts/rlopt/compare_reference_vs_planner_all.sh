#!/usr/bin/env bash
# Side-by-side "reference vs language-planner+policy" videos for the BONES Seed
# demo8 set. Env 0 replays the expert reference; env 1 runs the frozen low-level
# IPMD-bilinear policy driven by the merged SkillCommander language planner.
# Role markers: cyan = reference (ground truth), red = policy (planner+policy).
#
# Usage:
#   scripts/rlopt/compare_reference_vs_planner_all.sh                # all 8
#   SUBSET="4 7" scripts/rlopt/compare_reference_vs_planner_all.sh   # ranks 4 and 7 only
#   REF_VIS=both scripts/rlopt/compare_reference_vs_planner_all.sh   # robot + body markers
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-logs/bones_seed_language/from_scratch_1b_demo8_20260706_220758}"
PLANNER="${PLANNER:-$RUN_ROOT/m3_rollout_ft_merged_20260707_080013/planner_rollout_ft_merged/checkpoints/latest.pt}"
LOW="${LOW:-logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/2026-07-06_22-30-42/models/model_step_1000046592.pt}"
# NOTE: do not name this LANG — that collides with the shell locale env var.
LANG_EMB="${LANG_EMB:-data/bones_seed/language/g1_bones_seed_language_demo_8_minilm_goal_embeddings.pt}"
MANIFEST="${MANIFEST:-data/bones_seed/manifests/g1_bones_seed_language_demo_8_manifest.json}"
DATASET="${DATASET:-data/bones_seed/g1_hl_diffsr}"
OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/compare_reference_vs_planner}"
REF_VIS="${REF_VIS:-robot}"   # robot | body_markers | both
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"

# rank:motion pairs (demo8 order)
MOTIONS=(
  "0:Neutral_stoop_down_001_A057"
  "1:big_heavy_one_hand_front_high_to_front_low_R_001_A524"
  "2:big_heavy_one_hand_front_low_to_front_high_R_001_A524"
  "3:big_light_two_hands_pick_up_front_medium_R_001_A509"
  "4:drinking_standing_mug_R_001_A282"
  "5:inside_door_handle_left_side_open_walk_close_behind_R_001_A513"
  "6:inside_door_handle_right_side_open_walk_turn_close_R_001_A514"
  "7:read_book_both_hands_sitting_R_001_A456"
)

mkdir -p "$OUT_ROOT"
INDEX="$OUT_ROOT/video_index.md"
echo "# Reference vs Language-Planner Videos (side by side)" > "$INDEX"
echo "" >> "$INDEX"
echo "Left/right robots in one scene. Marker above each robot: **cyan = reference (ground truth)**, **red = policy (language planner + low-level)**." >> "$INDEX"
echo "" >> "$INDEX"
echo "| rank | motion | status | video |" >> "$INDEX"
echo "|---:|---|---|---|" >> "$INDEX"

for pair in "${MOTIONS[@]}"; do
  RANK="${pair%%:*}"
  MOTION="${pair#*:}"
  if [ -n "${SUBSET:-}" ] && ! echo " $SUBSET " | grep -q " $RANK "; then
    continue
  fi
  OUT="$OUT_ROOT/rank_$(printf '%04d' "$RANK")_$MOTION"
  mkdir -p "$OUT"
  echo "===== rank $RANK  $MOTION ====="
  set +e
  OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TORCHDYNAMO_DISABLE=1 \
  pixi run -e isaaclab python scripts/compare_policy_reference.py \
    --headless --device "$DEVICE" --seed "$SEED" \
    --task Isaac-Imitation-G1-Latent-v0 --algo IPMD_BILINEAR \
    --checkpoint "$LOW" \
    --policy_trajectory_rank "$RANK" \
    --policy_start_step 0 \
    --reference_visualization "$REF_VIS" \
    --video \
    --output_dir "$OUT" \
    env.lafan1_manifest_path="$MANIFEST" \
    env.dataset_path="$DATASET" \
    env.refresh_zarr_dataset=false \
    env.latent_command_dim=258 \
    agent.ipmd.latent_dim=258 \
    agent.ipmd.command_source=skill_commander \
    "agent.ipmd.skill_commander_checkpoint_path=$PLANNER" \
    "agent.ipmd.skill_commander_goal_name=$MOTION" \
    "agent.ipmd.skill_commander_embeddings_path=$LANG_EMB" \
    agent.ipmd.skill_commander_use_achieved_state=true \
    agent.ipmd.skill_commander_flow_num_inference_steps=16 \
    agent.ipmd.skill_commander_flow_inference_noise_std=0.0 \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.hl_skill_horizon_steps=25 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_steps_min=25 agent.ipmd.latent_steps_max=25 \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim=256 \
    agent.ipmd.latent_learning.code_period=25 \
    agent.ipmd.reward_loss_coeff=0.0 agent.ipmd.reward_l2_coeff=0.0 \
    agent.ipmd.reward_grad_penalty_coeff=0.0 agent.ipmd.reward_logit_reg_coeff=0.0 \
    agent.ipmd.reward_param_weight_decay_coeff=0.0 \
    > "$OUT/compare.log" 2>&1
  RC=$?
  set -e
  MP4="$(find "$OUT" -name '*.mp4' | head -1)"
  if [ "$RC" -eq 0 ] && [ -n "$MP4" ]; then
    echo "  OK -> $MP4"
    echo "| $RANK | $MOTION | ok | $MP4 |" >> "$INDEX"
  else
    echo "  FAIL rc=$RC (see $OUT/compare.log)"
    echo "| $RANK | $MOTION | FAIL rc=$RC | - |" >> "$INDEX"
  fi
done

echo ""
echo "Index written to: $INDEX"
