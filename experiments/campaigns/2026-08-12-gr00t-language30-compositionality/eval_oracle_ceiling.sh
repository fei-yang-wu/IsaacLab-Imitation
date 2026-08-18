#!/usr/bin/env bash
# Oracle ceiling for OUR trackers under the M3 planner protocol.
#
# Same binary, same protocol, same 30 goals as the GR00T arm grid — the only
# difference is that the latent comes from the frozen encoder reading the
# reference instead of from a planner. That makes it the ceiling each arm is
# normalized against, and it is the cell missing from the matched-protocol
# table (our previously quoted 18.13 mm was measured with tracking
# terminations ACTIVE, so it is a success-only mean over an easier subset).
#
# HOLD is the PUBLICATION period only (`latent_steps_min/max`). The encoder's
# macro horizon stays 10: `hl_skill_horizon_steps` says what a latent
# describes — a 10-step-ahead reference window — and the checkpoint pins it, so
# changing it is a different encoder, not a different cadence. SONIC works the
# same way: its window spans 45 frames ahead yet is re-encoded every step.
#
# HOLD=1 re-encodes every control step,
# which is the SONIC-style interface; it is OFF-DISTRIBUTION for our trackers,
# which were trained at hold 10, and for fsq64 it also pins the `sin_cos`
# phase channel to slot 0 forever. That is the point: this run measures how
# much the tracker loses to hold 1 before any planner is trained for it.
#
# Usage: eval_oracle_ceiling.sh <z256|fsq64> [extra args...]
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: eval_oracle_ceiling.sh <z256|fsq64>}"
shift || true
HOLD="${HOLD:-10}"

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
EPISODES_PER_GOAL="${EPISODES_PER_GOAL:-5}"
NUM_ENVS=$(( NUM_GOALS * EPISODES_PER_GOAL ))

case "${ARM}" in
    fsq64)
        TRACKER="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/4500357120/fsq64_sonic/model_step_4500357120.pt"
        ENCODER="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/encoders/fsq64_scaled.pt"
        ACTOR_DIM=66; CODE_DIM=64; NJMAX=320
        EXTRA_CFG=(
            'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
            'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
            agent.policy.activation_fn=silu agent.value_function.activation_fn=silu
        )
        ;;
    z256)
        TRACKER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt"
        ENCODER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt"
        ACTOR_DIM=258; CODE_DIM=256; NJMAX=289
        EXTRA_CFG=()
        ;;
    # The 2026-08-15 latent-bottleneck arms at 10B. Both were trained with the
    # SONIC v1.1 `robot_heading` macro frame, which is NOT the environment
    # default, so the override below is mandatory: a wrong anchor frame is
    # width-invisible and would silently score a different interface.
    fsq64_10b)
        MIRROR="${REPO_ROOT}/logs/bottleneck_10b_mirror/fsq64_hold10_seed0"
        TRACKER="${MIRROR}/tracker/f10000269312/models/model_step_10000269312.pt"
        ENCODER="${MIRROR}/encoder/checkpoints/latest.pt"
        ACTOR_DIM=66; CODE_DIM=64; NJMAX=320
        EXTRA_CFG=(
            'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
            'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
            agent.policy.activation_fn=silu agent.value_function.activation_fn=silu
            env.expert_macro_anchor_mode=robot_heading
        )
        ;;
    # Trained at hold 1, so run it with HOLD=1 CODE_PERIOD=1; at that period the
    # `sin_cos` phase channel is the constant (0, 1) both in training and here.
    ln_hold1_10b)
        MIRROR="${REPO_ROOT}/logs/bottleneck_10b_mirror/cont_det_ln_hold1_seed0"
        TRACKER="${MIRROR}/tracker/f10000269312/models/model_step_10000269312.pt"
        ENCODER="${MIRROR}/encoder/checkpoints/latest.pt"
        ACTOR_DIM=258; CODE_DIM=256; NJMAX=320
        EXTRA_CFG=(
            'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
            'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
            agent.policy.activation_fn=silu agent.value_function.activation_fn=silu
            env.expert_macro_anchor_mode=robot_heading
        )
        ;;
    *)
        echo "arm must be z256|fsq64|fsq64_10b|ln_hold1_10b, got ${ARM}" >&2
        exit 1
        ;;
esac

# TERMINATION_MODE picks what ends an episode, and therefore what "success"
# means. They are mutually exclusive in the evaluator:
#   fall_only (default) — tracking terminations off, success = did not fall.
#                         This is the planner protocol, so it is the ceiling a
#                         planner arm is normalized against.
#   sonic               — the released SONIC thresholds terminate (anchor
#                         height 0.25 m, squared anchor orientation 1.0 rad^2,
#                         end-effector height 0.25 m), success = finished the
#                         reference without tripping one. This is the SR the
#                         4,096-motion scoreboard and the SONIC release report.
TERMINATION_MODE="${TERMINATION_MODE:-fall_only}"
case "${TERMINATION_MODE}" in
    fall_only) TERMINATION_ARGS=(--disable_tracking_terminations --fall_only_success) ;;
    sonic)     TERMINATION_ARGS=(--sonic_success_terminations) ;;
    *) echo "TERMINATION_MODE must be fall_only|sonic, got ${TERMINATION_MODE}" >&2; exit 1 ;;
esac

# LABEL_SUFFIX separates protocol variants (step cap, seed) of the same
# (arm, hold), because the launcher refuses to overwrite a retained run.
LABEL="oracle_${ARM}_hold${HOLD}${LABEL_SUFFIX:+_${LABEL_SUFFIX}}"
OUTPUT_DIR="${REPO_ROOT}/logs/gr00t_language30_oracle_ceiling/${LABEL}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${TRACKER}" --skill_checkpoint "${ENCODER}" \
    --language_embeddings "${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 --output_dir "${OUTPUT_DIR}" --label "${LABEL}" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS:-500}" --seed "${SEED:-0}" \
    --metric_interval "${METRIC_INTERVAL:-10}" \
    --motion_names "${MOTIONS[@]}" --trajectory_ranks "${RANKS[@]}" \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    "${TERMINATION_ARGS[@]}" \
    physics=newton_mjwarp env.data.manifest=null env.data.cache_dir=null \
    env.data.reference_arrays_dir="${DATA_ROOT}/reference_arrays/root_qpos_v1" \
    env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755 \
    env.data.persist_dir=null env.data.macro_cache_device=cuda:0 \
    env.data.wrap_steps=false \
    'env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]' \
    env.command_interface.actor.dim="${ACTOR_DIM}" \
    'env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]' \
    agent.logger.backend= agent.ipmd.command_source=hl_skill \
    agent.ipmd.hl_skill_checkpoint_path="${ENCODER}" \
    agent.ipmd.hl_skill_finetune_enabled=false \
    agent.ipmd.latent_dim="${ACTOR_DIM}" \
    agent.ipmd.latent_steps_min="${HOLD}" agent.ipmd.latent_steps_max="${HOLD}" \
    agent.ipmd.hl_skill_horizon_steps=10 \
    agent.ipmd.hl_skill_command_mode=z \
    agent.ipmd.latent_learning.command_phase_mode=sin_cos \
    agent.ipmd.latent_learning.code_latent_dim="${CODE_DIM}" \
    agent.ipmd.latent_learning.code_period="${CODE_PERIOD:-10}" \
    env.sim.physics.solver_cfg.njmax="${NJMAX}" \
    env.sim.physics.solver_cfg.nconmax=200 \
    ${EXTRA_CFG[@]+"${EXTRA_CFG[@]}"} \
    "$@"

echo "retained: ${OUTPUT_DIR}"
