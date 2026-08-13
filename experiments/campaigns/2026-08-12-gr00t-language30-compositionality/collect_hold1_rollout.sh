#!/usr/bin/env bash
# Phase 2B: hold-1 rollout collection on OUR tracker.
#
# One row per environment per CONTROL STEP, so `_join_slots(hold_steps=1)`
# can build a 30-latent consecutive target. The tracker still publishes on its
# own hold-10 schedule during collection — only the SAMPLING rate changes, so
# the state distribution is the deployment one and the per-step oracle latent
# is what a hold-1 planner would have to emit.
#
# No `root_qpos` lookahead is stored: the target is the stored per-step latent
# (`latent.source: stored`), and a 30x38 window on every row would cost more
# disk than the states it explains. Prepare with `chunk.enabled: false`.
#
# Usage: collect_hold1_rollout.sh <z256|fsq64>
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

VARIANT="${1:?usage: collect_hold1_rollout.sh <z256|fsq64>}"
case "${VARIANT}" in
    fsq64)
        CHECKPOINT="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/4500357120/fsq64_sonic/model_step_4500357120.pt"
        ENCODER="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/encoders/fsq64_scaled.pt"
        ACTOR_DIM=66; CODE_DIM=64; NJMAX=320
        EXTRA=(
            'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
            'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
            agent.policy.activation_fn=silu agent.value_function.activation_fn=silu
        )
        ;;
    z256)
        CHECKPOINT="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt"
        ENCODER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt"
        ACTOR_DIM=258; CODE_DIM=256; NJMAX=289
        EXTRA=()
        ;;
    *)
        echo "Unknown variant ${VARIANT} (expected z256 or fsq64)" >&2
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/gr00t_language30_hold1_${VARIANT}/collection}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

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
NUM_GOALS="${#MOTIONS[@]}"
RANKS=($(seq 0 $((NUM_GOALS - 1))))
# EXCLUDE_RANKS drops motions from the collection by manifest rank. Used to
# leave out the two motions the TRACKER itself cannot do (its oracle falls 4/5
# on them): training a planner on states it can never recover from teaches it
# nothing, and those motions are scored separately.
if [ -n "${EXCLUDE_RANKS:-}" ]; then
    KEEP_M=(); KEEP_R=()
    for i in "${RANKS[@]}"; do
        skip=0
        for x in ${EXCLUDE_RANKS}; do [ "${i}" = "${x}" ] && skip=1; done
        [ "${skip}" = "0" ] && { KEEP_M+=("${MOTIONS[$i]}"); KEEP_R+=("${i}"); }
    done
    MOTIONS=("${KEEP_M[@]}"); RANKS=("${KEEP_R[@]}"); NUM_GOALS="${#MOTIONS[@]}"
fi
# 30 trajectories per motion. Every-step rows are 10x denser than the
# publication rows the hold-10 arms trained on, so a smaller environment count
# still yields more rows than those collections had.
PER_GOAL="${PER_GOAL:-30}"
NUM_ENVS=$(( NUM_GOALS * PER_GOAL ))

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${CHECKPOINT}" --skill_checkpoint "${ENCODER}" \
    --language_embeddings "${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 \
    --output_dir "${OUTPUT_DIR}" \
    --label "gr00t_language30_hold1_${VARIANT}" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS:-2000}" --seed "${SEED:-0}" \
    --metric_interval 10 \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --save_rollout_training_samples --sample_every_control_step \
    --sample_rows_per_file 8192 \
    ${WITH_ROOT_QPOS:+--require_root_qpos_samples --sample_future_window_frames 30} \
    --disable_tracking_terminations --fall_only_success \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    physics=newton_mjwarp \
    env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.reference_arrays_warm_workers=2 \
    env.data.macro_cache_device=cuda:0 \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    env.data.wrap_steps=false \
    env.command_interface.actor.dim="${ACTOR_DIM}" \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    agent.logger.backend= \
    agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim="${ACTOR_DIM}" \
    agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim="${CODE_DIM}" \
    agent.ipmd.latent_learning.code_period=10 \
    env.sim.physics.solver_cfg.njmax="${NJMAX}" \
    env.sim.physics.solver_cfg.nconmax=200 \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "retained: ${OUTPUT_DIR}"
