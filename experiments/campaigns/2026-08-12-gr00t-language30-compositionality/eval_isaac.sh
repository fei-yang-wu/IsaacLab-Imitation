#!/usr/bin/env bash
# Isaac closed-loop evaluation of a trained GR00T head — the number of record.
#
# Usage: eval_isaac.sh <z256|fsq64> <goal-name> [extra hydra/CLI args...]
#
# Protocol (M3): tracking-error terminations disabled, `base_too_low` active,
# so survival means "completed the episode without falling"; tracking errors
# stay continuous metrics. The mandated full-horizon diagnostic pass (all
# early terminations off, video retained) is a separate invocation with
# DIAGNOSTIC=1 — it prints the retained video's absolute path.
#
# The language goal is passed explicitly and the reference is restricted to
# the matching named motion; the goal is never inferred from the reference
# cursor or trajectory rank.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

ARM="${1:?usage: eval_isaac.sh <z256|fsq64> <goal-name>}"
GOAL="${2:?usage: eval_isaac.sh <z256|fsq64> <goal-name>}"
shift 2

DIAGNOSTIC="${DIAGNOSTIC:-0}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_STEPS="${MAX_STEPS:-500}"
SEED="${SEED:-0}"
UPDATE="${UPDATE:-0012000}"

case "${ARM}" in
    z256)
        CHECKPOINT="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/checkpoints/model_step_3500015616.pt"
        ENCODER="${REPO_ROOT}/logs/rollout24_gamma097_foot_disabled_eval/encoder/latest.pt"
        ACTOR_DIM=258
        CODE_DIM=256
        NJMAX=289
        EXTRA_CFG=()
        ;;
    fsq64)
        CHECKPOINT="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/4500357120/fsq64_sonic/model_step_4500357120.pt"
        ENCODER="${REPO_ROOT}/logs/bones129k_sonic_fsq_scale_eval/encoders/fsq64_scaled.pt"
        ACTOR_DIM=66
        CODE_DIM=64
        NJMAX=320
        EXTRA_CFG=(
            'agent.policy.num_cells=[2048,2048,1024,1024,512,512]'
            'agent.value_function.num_cells=[2048,2048,1024,1024,512,512]'
            agent.policy.activation_fn=silu
            agent.value_function.activation_fn=silu
        )
        ;;
    *)
        echo "Unknown arm ${ARM} (expected z256 or fsq64)." >&2
        echo "The explicit arm publishes through the explicit command interface," >&2
        echo "not this latent sampler — see the campaign README." >&2
        exit 1
        ;;
esac

HEAD="${REPO_ROOT}/outputs/gr00t_language30/arms/${ARM}/checkpoints/update_${UPDATE}.pt"
GOAL_FEATURES="${REPO_ROOT}/outputs/gr00t_language30/goal_features/goal_features.pt"
DATA_ROOT="${REPO_ROOT}/data/bones_seed_language30_compositionality_v1"
for required in "${HEAD}" "${GOAL_FEATURES}" "${CHECKPOINT}" "${ENCODER}"; do
    [ -f "${required}" ] || { echo "missing input: ${required}" >&2; exit 1; }
done

LABEL="isaac_${ARM}_${GOAL}"
MODE_ARGS=(--disable_tracking_terminations)
if [ "${DIAGNOSTIC}" = "1" ]; then
    # Full-horizon diagnostic: tracking terminations off and the fall detector
    # dropped to floor level, so no early termination truncates the rollout and
    # MPJPE covers the intended horizon. Video is retained and its absolute
    # path printed.
    MODE_ARGS=(--disable_tracking_terminations --fall_height_m 0.01 --video)
    LABEL="${LABEL}_diagnostic"
fi
OUTPUT_DIR="${REPO_ROOT}/logs/gr00t_language30_isaac_eval/${LABEL}"
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR}" >&2
    exit 1
fi

pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
    --checkpoint "${CHECKPOINT}" \
    --skill_checkpoint "${ENCODER}" \
    --gr00t_checkpoint "${HEAD}" \
    --gr00t_goal_features "${GOAL_FEATURES}" \
    --gr00t_goal "${GOAL}" \
    --gr00t_consumption "${GR00T_CONSUMPTION:-open_loop}" \
    --language_embeddings "${DATA_ROOT}/language/g1_bones_seed_language30_compositionality_v1_minilm_goal_embeddings.pt" \
    --state_history_steps 9 \
    --output_dir "${OUTPUT_DIR}" \
    --label "${LABEL}" \
    --num_envs "${NUM_ENVS}" --max_steps "${MAX_STEPS}" --seed "${SEED}" \
    --metric_interval "${MAX_STEPS}" \
    --motion_names "${GOAL}" \
    --disable_push_event --disable_reward_clipping --assert-kitless \
    "${MODE_ARGS[@]}" \
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
    ${EXTRA_CFG[@]+"${EXTRA_CFG[@]}"} \
    "$@"
