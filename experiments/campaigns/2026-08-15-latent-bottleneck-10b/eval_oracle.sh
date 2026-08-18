#!/usr/bin/env bash
# Oracle eval of one 10B-campaign arm on the 30 compositionality motions.
#
# Protocol of record, identical for every arm: M3 fall-only (tracking
# terminations off, base_too_low only), Newton MJWarp, 30 motions x 5 episodes
# = 150 environments, 2000-step cap, metric_interval 10, no push events, no
# reward clipping. Commands come from the arm's OWN encoder at the arm's OWN
# hold, which is the interface it trained on -- a mismatched hold or command
# width is not a fairer comparison, it is a broken one.
#
# The arm table below must stay in step with campaign.yaml. Each row is
#   <arm> <z_dim> <command_dim> <hold>
# and phase is sin_cos everywhere in this campaign, so the command width is
# always z_dim + 2.
#
# The second argument is the checkpoint TAG, i.e. the part of the staged file
# name after "tracker_". Segment-1 checkpoints are named by their step
# (2500067328); later segments carry a "segN_" prefix because pre-2026-08-16
# checkpoints number steps segment-locally, so the raw step is ambiguous
# across segments.
#
# Usage: eval_oracle.sh <arm> [checkpoint_tag]
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: eval_oracle.sh <arm> [checkpoint_step]}"
STEP="${2:-2500067328}"
BINDING_JSON="binding.json"
case "${STEP}" in seg*) BINDING_JSON="binding_${STEP%%_*}.json" ;; esac

# Staged off the repo disk: /mnt/hsstorage is full, so checkpoints and eval
# outputs live on the root filesystem.
STAGE="${BOTTLENECK_STAGE:-/home/fwu91/Documents/SL/IsaacLab-Imitation/logs/bottleneck_10b_ckpts}"
OUT_ROOT="${BOTTLENECK_EVAL_ROOT:-/home/fwu91/Documents/SL/IsaacLab-Imitation/logs/bottleneck_10b_eval}"

case "${ARM}" in
    fsq64_hold10)                Z=64;  DIM=66;  HOLD=10 ;;
    gumbel_multicat64_hold10)    Z=64;  DIM=66;  HOLD=10 ;;
    group_vq64_hold10)           Z=64;  DIM=66;  HOLD=10 ;;
    cont_det_hold1)              Z=256; DIM=258; HOLD=1  ;;
    cont_det_ln_hold1)           Z=256; DIM=258; HOLD=1  ;;
    cont_det_hold1_resetramp)    Z=256; DIM=258; HOLD=1  ;;
    jepa_pure_256d_hold1)        Z=256; DIM=258; HOLD=1  ;;
    jepa_ntp_hold10_256d)        Z=256; DIM=258; HOLD=10 ;;
    jepa_sigreg_ebm_hold10_256d) Z=256; DIM=258; HOLD=10 ;;
    jepa_sigreg_ebm_hold10_fsq64) Z=64;  DIM=66;  HOLD=10 ;;
    gumbel_multicat64g64_hold10)  Z=64;  DIM=66;  HOLD=10 ;;
    *) echo "unknown arm: ${ARM}" >&2; exit 2 ;;
esac

TRACKER="${STAGE}/${ARM}/tracker_${STEP}.pt"
ENCODER="${STAGE}/${ARM}/encoder_latest.pt"
[ -f "${TRACKER}" ] || { echo "missing tracker: ${TRACKER}" >&2; exit 1; }
[ -f "${ENCODER}" ] || { echo "missing encoder: ${ENCODER}" >&2; exit 1; }
# The binding record is a gate, not a formality: pairing a tracker with an
# encoder it did not train against silently changes the interface under test.
[ -f "${STAGE}/${ARM}/${BINDING_JSON}" ] || {
    echo "missing ${BINDING_JSON} for ${ARM}; run validate_latent_skill_checkpoint_binding first" >&2
    exit 1
}

DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_compositionality_v1"
MANIFEST_LANG="${DATA_ROOT}/manifests/g1_bones_seed_language30_compositionality_v1_manifest_language.json"
readarray -t MOTIONS < <(
    "${REPO_ROOT}/.pixi/envs/default/bin/python" - "${MANIFEST_LANG}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
motions = manifest["motions"] if "motions" in manifest else manifest
print("\n".join(m["name"] for m in motions))
PY
)
RANKS=($(seq 0 $((${#MOTIONS[@]} - 1))))
EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-5}"
NUM_ENVS=$(( ${#MOTIONS[@]} * EPISODES_PER_GOAL ))

OUTPUT_DIR="${OUT_ROOT}/${ARM}_step${STEP}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${TRACKER}" --skill_checkpoint "${ENCODER}" \
    --language_embeddings "${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 --output_dir "${OUTPUT_DIR}" --label "bneck_${ARM}_oracle" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS:-2000}" --seed "${SEED:-0}" \
    --metric_interval 10 \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    --disable_tracking_terminations --fall_only_success \
    physics=newton_mjwarp env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    env.command_interface.actor=latent \
    "env.command_interface.actor.dim=${DIM}" \
    env.command_interface.encoder=single \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    "env.expert_macro_frame_stride=${STRIDE:-1}" \
    env.expert_macro_anchor_mode=robot_heading \
    agent.logger.backend= agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    "agent.ipmd.latent_dim=${DIM}" \
    "agent.ipmd.latent_steps_min=${HOLD}" "agent.ipmd.latent_steps_max=${HOLD}" \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    "agent.ipmd.latent_learning.code_latent_dim=${Z}" \
    "agent.ipmd.latent_learning.code_period=${HOLD}" \
    'agent.policy.num_cells=[2048,2048,1024,1024,512,512]' \
    'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]' \
    agent.policy.activation_fn=silu agent.value_function.activation_fn=silu \
    env.sim.physics.solver_cfg.njmax=320 \
    env.sim.physics.solver_cfg.nconmax=200

echo "retained: ${OUTPUT_DIR}"
